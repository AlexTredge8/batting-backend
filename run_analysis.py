"""
BattingIQ Phase 2 — Main Entry Point
=======================================
Usage:
    python run_analysis.py path/to/video.mp4 [--output-dir output/]

The reference baseline must exist at reference/reference_baseline.json.
Run  python reference_builder.py test_batting.mov  to generate it first.
"""

import sys
import json
import argparse
import base64
from pathlib import Path
from typing import Any

import cv2

from pose_extractor     import extract_poses
from metrics_calculator import calculate_metrics
from phase_detector     import detect_phases, print_phase_summary, apply_anchor_overrides
from coaching_rules     import run_all_rules, collect_rule_measurements
from scorer             import build_scores
from report_generator   import save_json_report, print_report
from video_annotator    import annotate_video, generate_storyboard
from reference_builder  import load_reference_baseline, build_reference_baseline
from anchor_accuracy    import (
    ANCHOR_DETECTOR_VERSION,
    build_anchor_confidence,
    build_anchor_frames,
    build_anchor_quality_summary,
)
from config             import (
    REFERENCE_BASELINE_PATH,
    DEFAULT_HANDEDNESS,
    CONTACT_DETECTOR_VERSION,
    SETUP_DETECTOR_VERSION,
    PROCESSING_MODE,
)


def _handedness_to_front_side(handedness: str) -> str:
    """Convert handedness ('right'/'left') to front_side ('left'/'right')."""
    return "left" if handedness == "right" else "right"


def _parse_anchor_frames_json(anchor_frames_json: str | None) -> dict[str, int | None] | None:
    """Parse a JSON string of anchor overrides into the dict shape used internally."""
    if anchor_frames_json in (None, ""):
        return None

    parsed = json.loads(anchor_frames_json)
    if not isinstance(parsed, dict):
        raise ValueError("anchor_frames_json must decode to a JSON object")

    normalized: dict[str, int | None] = {}
    for key, value in parsed.items():
        if value in (None, ""):
            normalized[str(key)] = None
            continue
        normalized[str(key)] = int(value)
    return normalized


def _build_analysis_quality(result, video_meta: dict) -> tuple[dict, list[str]]:
    """
    Summarise how trustworthy this analysis is, in one place, with plain-English
    warnings the frontend can show. Every field already exists deeper in the
    report; this block makes degraded results impossible to miss.
    """
    phases = result.phases
    diag = phases.contact_diagnostics or {}
    contact_method = str(diag.get("method") or "unknown")
    audio_available = contact_method.startswith("audio")
    audio_status = None
    audio_diag = diag.get("audio_diagnostics") or {}
    if isinstance(audio_diag, dict):
        audio_status = audio_diag.get("status") or (audio_diag.get("extract") or {}).get("status")
    anchor_quality = video_meta.get("anchor_quality_summary") or {}
    low_anchors = list(anchor_quality.get("low_confidence_anchors") or [])
    detection_rate = float(video_meta.get("detection_rate") or 0.0)
    processing_mode = video_meta.get("processing_mode") or PROCESSING_MODE

    warnings: list[str] = []
    if not audio_available and phases.resolved_contact_source != "manual":
        warnings.append(
            "No usable audio was found in this clip, so bat-on-ball contact was estimated "
            "from body movement alone. Contact-based scores are less reliable — record with "
            "sound on and avoid compression that strips the audio track."
        )
    if phases.contact_confidence == "low" and phases.resolved_contact_source != "manual":
        warnings.append(
            "Contact confidence is low for this video, so contact-derived deductions have been softened."
        )
    if low_anchors:
        pretty = ", ".join(a.replace("_frame", "").replace("_", " ") for a in low_anchors)
        warnings.append(
            f"Some key moments could not be pinned confidently ({pretty}); rules that depend on "
            "them were suppressed or softened."
        )
    if detection_rate and detection_rate < 80.0:
        warnings.append(
            f"The batter was only detected in {detection_rate:.0f}% of frames. Film the full body "
            "from behind the bowler's arm with good light for a more reliable analysis."
        )
    baseline_status = video_meta.get("baseline_status") or "reference"
    if baseline_status != "reference":
        warnings.append(
            "The reference baseline was unavailable on the server, so this clip was compared "
            "against itself. Scores are not comparable to other analyses."
        )
    if processing_mode != "full_rate_calibrated":
        warnings.append(
            "This analysis ran in fast (subsampled) mode, which was not used for calibration; "
            "anchors and scores may differ from the calibrated pipeline."
        )

    quality = {
        "processing_mode": processing_mode,
        "frame_step": video_meta.get("frame_step"),
        "frames_processed": video_meta.get("frames_processed"),
        "total_frames": video_meta.get("total_frames"),
        "detection_rate": detection_rate,
        "audio_available": audio_available,
        "audio_status": audio_status,
        "contact_method": contact_method,
        "contact_confidence": phases.contact_confidence,
        "contact_source": phases.resolved_contact_source,
        "low_confidence_anchors": low_anchors,
        "all_anchors_high_confidence": bool(anchor_quality.get("all_high_confidence", False)),
        "rules_suppressed": (video_meta.get("rule_evaluation") or {}).get("rules_suppressed", 0),
        "baseline_status": baseline_status,
        "reliable": bool(
            audio_available or phases.resolved_contact_source == "manual"
        ) and not low_anchors and detection_rate >= 80.0 and baseline_status == "reference"
        and processing_mode == "full_rate_calibrated",
    }
    return quality, warnings


STORYBOARD_FRAME_KEYS = (
    ("setup", "setup_end"),
    ("hands_start_up", "backlift_start"),
    ("front_foot_down", "front_foot_down"),
    ("hands_peak", "hands_peak"),
    ("contact", "contact"),
    ("follow_through", "follow_through_start"),
)


def _metric_index_to_original_frame(metrics: list[Any], metric_index: int | None) -> int | None:
    if metric_index is None or not metrics:
        return None
    idx = int(metric_index)
    if idx < 0 or idx >= len(metrics):
        return None
    return int(getattr(metrics[idx], "frame_idx", idx))


# Storyboard phase key (frontend contract) -> storyboard still phase value
_KEYFRAME_TO_STILL_PHASE = {
    "setup": "setup",
    "hands_start_up": "backlift_starts",
    "front_foot_down": "front_foot_down",
    "hands_peak": "hands_peak",
    "contact": "contact",
    "follow_through": "follow_through",
}
_KEYFRAME_LABELS = {
    "setup": "Setup",
    "hands_start_up": "Hands Up",
    "front_foot_down": "Front Foot Down",
    "hands_peak": "Hands Peak",
    "contact": "Contact",
    "follow_through": "Follow Through",
}
_KEYFRAME_WIDTH = 400
_KEYFRAME_JPEG_QUALITY = 78


def _jpeg_b64(image) -> str | None:
    """Resize to _KEYFRAME_WIDTH and encode as raw base64 JPEG (small payload)."""
    if image is None or getattr(image, "size", 0) == 0:
        return None
    height, width = image.shape[:2]
    if width > _KEYFRAME_WIDTH:
        scale = _KEYFRAME_WIDTH / float(width)
        image = cv2.resize(image, (_KEYFRAME_WIDTH, max(1, int(round(height * scale)))),
                           interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, _KEYFRAME_JPEG_QUALITY])
    return base64.b64encode(buffer).decode("ascii") if ok else None


def _read_video_frame(video_path: str, frame_index: int):
    """Read one frame, honouring phone rotation metadata (portrait clips)."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = max(0, min(int(frame_index), max(0, total - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _still_image_without_label_bar(path: str | None):
    """Load an annotated storyboard still and crop off its label bar."""
    if not path or not Path(path).exists():
        return None
    panel = cv2.imread(str(path))
    if panel is None:
        return None
    from video_annotator import _LABEL_H
    if panel.shape[0] > _LABEL_H + 10:
        panel = panel[: panel.shape[0] - _LABEL_H]
    return panel


def build_storyboard_keyframes(
    video_path: str,
    phases: Any,
    metrics: list[Any],
    storyboard_items: list[dict],
    fps: float,
) -> dict[str, dict]:
    """
    The frontend storyboard contract: an object keyed by phase
    (setup, hands_start_up, front_foot_down, hands_peak, contact, follow_through),
    each value an object carrying the image AND the frame numbers.

    The image is the annotated still (pose skeleton drawn) when available, else the
    raw rotated video frame. It is exposed under several common field names
    (image, image_url, data_url, all the same data URI) so any
    reasonable frontend lookup finds it. The API layer adds ``url``.
    """
    fps = fps or 30.0
    items_by_phase = {item.get("phase"): item for item in (storyboard_items or []) if isinstance(item, dict)}
    keyframes: dict[str, dict] = {}

    for key, phase_attr in STORYBOARD_FRAME_KEYS:
        item = items_by_phase.get(_KEYFRAME_TO_STILL_PHASE[key])
        if item is not None:
            original_frame = int(item.get("original_frame_idx", 0))
            metric_idx = int(item.get("metric_idx", original_frame))
            image = _still_image_without_label_bar(item.get("path"))
            source = "annotated_still"
        else:
            metric_idx = getattr(phases, phase_attr, None)
            original_frame = _metric_index_to_original_frame(metrics, metric_idx)
            image = None
            source = "raw_frame"
        if image is None and original_frame is not None:
            image = _read_video_frame(video_path, original_frame)
            source = "raw_frame"

        b64 = _jpeg_b64(image)
        data_uri = f"data:image/jpeg;base64,{b64}" if b64 else None
        frame_no = int(original_frame) if original_frame is not None else None
        timestamp_ms = round(frame_no / fps * 1000, 1) if frame_no is not None else None
        keyframes[key] = {
            "phase": key,
            "label": _KEYFRAME_LABELS[key],
            "available": data_uri is not None,
            # image as a data URI, under the three most common field names
            "image": data_uri,
            "image_url": data_uri,
            "data_url": data_uri,
            "mime_type": "image/jpeg",
            # frame numbers (original video frames) — aliases on purpose
            "frame": frame_no,
            "frame_index": frame_no,
            "frame_idx": frame_no,
            "original_frame": frame_no,
            "original_frame_idx": frame_no,
            "metric_idx": int(metric_idx) if metric_idx is not None else None,
            "timestamp_ms": timestamp_ms,
            "timestamp_s": round(timestamp_ms / 1000, 3) if timestamp_ms is not None else None,
            "image_source": source,
            "_path": item.get("path") if item else None,
        }
    return keyframes


def analyse(video_path: str, output_dir: str = None, verbose: bool = True,
            handedness: str = None, handedness_source: str = "default",
            contact_frame: int | None = None,
            anchor_frames: dict[str, int | None] | None = None) -> dict:
    """
    Full BattingIQ Phase 2 analysis pipeline.

    Returns the JSON-serialisable report dict.
    """
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # --- Resolve handedness ---
    if handedness is None:
        handedness = DEFAULT_HANDEDNESS
        handedness_source = "default"
    handedness = handedness.lower().strip()
    if handedness not in ("right", "left"):
        if verbose:
            print(f"  Warning: unknown handedness '{handedness}', defaulting to 'right'")
        handedness = "right"
        handedness_source = "default"
    front_side = _handedness_to_front_side(handedness)
    if verbose:
        print(f"  Handedness: {handedness} (source: {handedness_source}, front_side: {front_side})")

    out_dir = Path(output_dir) if output_dir else vpath.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = vpath.stem

    # --- Load reference baseline ---
    ref_path = Path(REFERENCE_BASELINE_PATH)
    baseline_status = "reference"  # will be included in report
    # A self-calibrated baseline is written next to this job's outputs, NEVER to the
    # reference path: persisting a user's upload as the gold standard would silently
    # re-baseline every later analysis on the server.
    self_cal_path = str(out_dir / f"{stem}_self_calibrated_baseline.json")
    if not ref_path.exists():
        if verbose:
            print(f"  WARNING: Reference baseline not found at {ref_path}")
            print(f"  Self-calibrating from input video — scores may be less accurate")
        baseline = build_reference_baseline(video_path, output_path=self_cal_path)
        baseline_status = "self_calibrated"
    else:
        baseline = load_reference_baseline()
        if verbose:
            print(f"Reference baseline loaded from {ref_path}")
        # Validate baseline has expected keys
        if "setup" not in baseline or "contact" not in baseline:
            if verbose:
                print(f"  WARNING: Reference baseline is incomplete — self-calibrating")
            baseline = build_reference_baseline(video_path, output_path=self_cal_path)
            baseline_status = "self_calibrated"

    # --- Step 1: Extract poses ---
    if verbose:
        print(f"\nAnalysing: {vpath.name}")
    frame_poses, video_meta = extract_poses(video_path, verbose=verbose)
    video_meta["baseline_status"] = baseline_status
    fps = video_meta["fps"]

    # --- Step 2: Calculate metrics ---
    if verbose:
        print("  Calculating metrics...")
    metrics = calculate_metrics(frame_poses, fps, front_side=front_side)

    # --- Step 3: Detect phases ---
    if verbose:
        print("  Detecting phases...")
    anchor_override_frames = dict(anchor_frames or {})
    raw_phases = detect_phases(metrics, fps, video_path=video_path)
    if contact_frame is not None and anchor_override_frames.get("contact_frame") is None:
        anchor_override_frames["contact_frame"] = int(contact_frame)
    phases = apply_anchor_overrides(raw_phases, metrics, anchor_override_frames or None)
    anchor_frame_map = build_anchor_frames(phases, metrics)
    anchor_confidence = build_anchor_confidence(phases, metrics)
    if anchor_override_frames:
        for anchor_key, anchor_value in anchor_override_frames.items():
            if anchor_value is not None:
                anchor_confidence[anchor_key] = "validated"
    anchor_quality_summary = build_anchor_quality_summary(anchor_confidence)
    if verbose:
        print_phase_summary(phases, fps)
    video_meta["phase_diagnostics"] = {
        "contact_method": phases.contact_diagnostics.get("method"),
        "contact_method_reason": phases.contact_diagnostics.get("reason"),
        "audio_contact_confidence": phases.contact_diagnostics.get("audio_confidence"),
        "contact_confidence": phases.contact_confidence,
        "estimated_contact_confidence": phases.estimated_contact_confidence,
        "contact_candidates": phases.contact_candidates,
        "contact_window": phases.contact_window,
        "contact_diagnostics": phases.contact_diagnostics,
        "ordering_guard_log": phases.ordering_guard_log,
    }
    video_meta["contact_resolution"] = {
        "estimated_frame": phases.estimated_contact_frame,
        "estimated_original_frame": phases.estimated_contact_original_frame,
        "resolved_frame": phases.resolved_contact_frame or phases.contact,
        "resolved_original_frame": phases.resolved_contact_original_frame,
        "source": phases.resolved_contact_source,
        "status": phases.resolved_contact_status,
    }
    video_meta["detector_version"] = CONTACT_DETECTOR_VERSION
    video_meta["contact_detector_version"] = CONTACT_DETECTOR_VERSION
    video_meta["setup_detector_version"] = SETUP_DETECTOR_VERSION
    video_meta["anchor_detector_version"] = ANCHOR_DETECTOR_VERSION
    video_meta["anchor_frames"] = anchor_frame_map
    video_meta["anchor_confidence"] = anchor_confidence
    video_meta["anchor_quality_summary"] = anchor_quality_summary
    video_meta["ordering_guard_log"] = phases.ordering_guard_log
    if anchor_override_frames:
        video_meta["anchor_overrides"] = {key: value for key, value in anchor_override_frames.items() if value is not None}
    if phases.contact_confidence == "low":
        video_meta["contact_notice"] = (
            "Contact confidence is low for this video, so contact-derived deductions "
            "have been softened."
        )
    if phases.resolved_contact_source == "manual":
        video_meta["contact_notice"] = (
            "Contact frame was manually validated and pinned for storyboard and scoring."
        )

    # --- Step 4: Run coaching rules ---
    if verbose:
        print("\n  Running coaching rules...")
    fault_map = run_all_rules(metrics, phases, baseline, front_side=front_side)
    video_meta["rule_measurements"] = collect_rule_measurements(
        metrics,
        phases,
        baseline,
        front_side=front_side,
    )

    # --- Step 5: Score ---
    result = build_scores(fault_map, phases, baseline, video_meta,
                          handedness=handedness, handedness_source=handedness_source)

    # --- Step 6: Print & save JSON report ---
    if verbose:
        print_report(result)

    json_path = out_dir / f"{stem}_battingiq.json"
    save_json_report(result, str(json_path))
    if verbose:
        print(f"\n  JSON report → {json_path}")

    # --- Step 7: Annotated video (best-effort — codec may be unavailable) ---
    if verbose:
        print("  Generating annotated video...")
    video_out = out_dir / f"{stem}_battingiq_annotated.mp4"
    media_generation = {
        "annotated_video": {"status": "pending", "path": str(video_out), "error": None},
        "storyboard": {"status": "pending", "path": str(out_dir / f"{stem}_battingiq_storyboard.png"), "error": None},
    }
    try:
        annotate_video(video_path, result, metrics, str(video_out), frame_poses=frame_poses)
        media_generation["annotated_video"]["status"] = "ok"
    except Exception as ann_exc:
        if verbose:
            print(f"  Warning: annotated video generation failed ({ann_exc})")
        media_generation["annotated_video"]["status"] = "failed"
        media_generation["annotated_video"]["error"] = str(ann_exc)
        video_out = None

    # --- Step 8: Storyboard (6 key phase frames as a single horizontal strip) ---
    if verbose:
        print("  Generating storyboard...")
    storyboard_out = out_dir / f"{stem}_battingiq_storyboard.png"
    storyboard_frames = []
    try:
        storyboard_result = generate_storyboard(video_path, result, metrics, str(storyboard_out), frame_poses=frame_poses)
        storyboard_frames = storyboard_result.get("frames", []) if isinstance(storyboard_result, dict) else []
        if isinstance(storyboard_result, dict) and storyboard_result.get("strip_path"):
            storyboard_out = Path(storyboard_result["strip_path"])
        if not storyboard_frames:
            raise RuntimeError("Storyboard generation returned no frames")
        media_generation["storyboard"]["status"] = "ok"
        media_generation["storyboard"]["frame_count"] = len(storyboard_frames)
    except Exception as sb_exc:
        if verbose:
            print(f"  Warning: storyboard generation failed ({sb_exc})")
        media_generation["storyboard"]["status"] = "failed"
        media_generation["storyboard"]["error"] = str(sb_exc)
        media_generation["storyboard"]["frame_count"] = 0
        storyboard_frames = []
        storyboard_out = None

    storyboard_keyframes = build_storyboard_keyframes(
        video_path, phases, metrics, storyboard_frames, video_meta.get("fps") or fps
    )

    if verbose:
        print(f"\nDone. Output in: {out_dir}/")

    result.metadata = dict(result.metadata or {})
    result.metadata["media_generation"] = media_generation
    result.metadata["storyboard_generation"] = {
        "strip_path": str(storyboard_out) if storyboard_out and Path(storyboard_out).exists() else None,
        "frame_count": len(storyboard_frames),
        "frames": storyboard_frames,
        "selection_mode": "stage_aware_local_refinement",
        "selection_note": (
            "Storyboard stills are chosen from narrow windows around the detected "
            "phase anchors so setup/backlift/hands peak/front foot/contact/follow-through "
            "frames can be nudged toward the clearest nearby original frame."
        ),
    }

    from report_generator import build_json_report
    quality, warnings = _build_analysis_quality(result, result.metadata)
    result.metadata["analysis_quality"] = quality
    result.metadata["warnings"] = warnings
    report = build_json_report(result)
    report["analysis_quality"] = quality
    report["warnings"] = warnings

    # Embed file paths so the API can build public URLs (stripped before sending to client)
    report["_annotated_video"] = str(video_out) if video_out and Path(video_out).exists() else None
    report["_storyboard"]      = str(storyboard_out) if storyboard_out and Path(storyboard_out).exists() else None
    report["_storyboard_frames"] = storyboard_frames
    report["_storyboard_keyframes"] = storyboard_keyframes
    report["storyboard_frames"] = {k: {kk: vv for kk, vv in v.items() if kk != "_path"}
                                   for k, v in storyboard_keyframes.items()}

    return report


def main():
    parser = argparse.ArgumentParser(description="BattingIQ Phase 2 Analysis")
    parser.add_argument("video", help="Path to batting video file")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: video_directory/output/)")
    parser.add_argument("--rebuild-baseline", action="store_true",
                        help="Force rebuild of reference baseline from this video")
    parser.add_argument(
        "--anchor-frames-json",
        default=None,
        help="JSON object of original-frame anchor overrides to use instead of auto-detected anchors",
    )
    args = parser.parse_args()

    if args.rebuild_baseline:
        print(f"Rebuilding reference baseline from {args.video} ...")
        build_reference_baseline(args.video)

    anchor_frames = _parse_anchor_frames_json(args.anchor_frames_json)
    analyse(args.video, args.output_dir, anchor_frames=anchor_frames)


if __name__ == "__main__":
    main()


def run_full_analysis(video_path: str, output_dir: str = None,
                      handedness: str = None, handedness_source: str = "default",
                      contact_frame: int | None = None,
                      anchor_frames: dict[str, int | None] | None = None) -> dict:
    """Programmatic entry point for the FastAPI wrapper. Returns the full report as a dict."""
    return analyse(video_path, output_dir=output_dir, verbose=False,
                   handedness=handedness, handedness_source=handedness_source,
                   contact_frame=contact_frame, anchor_frames=anchor_frames)
