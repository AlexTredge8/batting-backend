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

from pose_extractor     import extract_poses
from metrics_calculator import calculate_metrics
from phase_detector     import detect_phases, print_phase_summary
from coaching_rules     import run_all_rules
from scorer             import build_scores
from report_generator   import save_json_report, print_report
from video_annotator    import annotate_video, generate_storyboard
from reference_builder  import load_reference_baseline, build_reference_baseline
from config             import REFERENCE_BASELINE_PATH, DEFAULT_HANDEDNESS


def _handedness_to_front_side(handedness: str) -> str:
    """Convert handedness ('right'/'left') to front_side ('left'/'right')."""
    return "left" if handedness == "right" else "right"


def analyse(video_path: str, output_dir: str = None, verbose: bool = True,
            handedness: str = None, handedness_source: str = "default") -> dict:
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
    phases = detect_phases(metrics, fps)
    if verbose:
        print_phase_summary(phases, fps)

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
    args = parser.parse_args()

    if args.rebuild_baseline:
        print(f"Rebuilding reference baseline from {args.video} ...")
        build_reference_baseline(args.video)

    analyse(args.video, args.output_dir)


if __name__ == "__main__":
    main()


def run_full_analysis(video_path: str, output_dir: str = None,
                      handedness: str = None, handedness_source: str = "default") -> dict:
    """Programmatic entry point for the FastAPI wrapper. Returns the full report as a dict."""
    return analyse(video_path, output_dir=output_dir, verbose=False,
                   handedness=handedness, handedness_source=handedness_source)
