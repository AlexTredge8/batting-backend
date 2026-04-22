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
from pathlib import Path
from typing import Any

from pose_extractor     import extract_poses
from metrics_calculator import calculate_metrics
from phase_detector     import detect_phases, print_phase_summary, apply_anchor_overrides
from coaching_rules     import run_all_rules
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
    if not ref_path.exists():
        if verbose:
            print(f"  WARNING: Reference baseline not found at {ref_path}")
            print(f"  Self-calibrating from input video — scores may be less accurate")
        baseline = build_reference_baseline(video_path)
        baseline_status = "self_calibrated"
    else:
        baseline = load_reference_baseline()
        if verbose:
            print(f"Reference baseline loaded from {ref_path}")
        # Validate baseline has expected keys
        if "setup" not in baseline or "contact" not in baseline:
            if verbose:
                print(f"  WARNING: Reference baseline is incomplete — self-calibrating")
            baseline = build_reference_baseline(video_path)
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
    raw_phases = detect_phases(metrics, fps)
    if contact_frame is not None and anchor_override_frames.get("contact_frame") is None:
        anchor_override_frames["contact_frame"] = int(contact_frame)
    anchor_frame_map = build_anchor_frames(raw_phases, metrics)
    anchor_confidence = build_anchor_confidence(raw_phases, metrics)
    anchor_quality_summary = build_anchor_quality_summary(anchor_confidence)
    phases = apply_anchor_overrides(raw_phases, metrics, anchor_override_frames or None)
    if verbose:
        print_phase_summary(phases, fps)
    video_meta["phase_diagnostics"] = {
        "contact_confidence": phases.contact_confidence,
        "estimated_contact_confidence": phases.estimated_contact_confidence,
        "contact_candidates": phases.contact_candidates,
        "contact_window": phases.contact_window,
        "contact_diagnostics": phases.contact_diagnostics,
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
    video_meta["anchor_detector_version"] = ANCHOR_DETECTOR_VERSION
    video_meta["anchor_frames"] = anchor_frame_map
    video_meta["anchor_confidence"] = anchor_confidence
    video_meta["anchor_quality_summary"] = anchor_quality_summary
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
        annotate_video(video_path, result, metrics, str(video_out))
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
        storyboard_result = generate_storyboard(video_path, result, metrics, str(storyboard_out))
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
    report = build_json_report(result)

    # Embed file paths so the API can build public URLs (stripped before sending to client)
    report["_annotated_video"] = str(video_out) if video_out and Path(video_out).exists() else None
    report["_storyboard"]      = str(storyboard_out) if storyboard_out and Path(storyboard_out).exists() else None
    report["_storyboard_frames"] = storyboard_frames

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
