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
from video_annotator    import annotate_video
from reference_builder  import load_reference_baseline, build_reference_baseline
from config             import REFERENCE_BASELINE_PATH


def analyse(video_path: str, output_dir: str = None, verbose: bool = True) -> dict:
    """
    Full BattingIQ Phase 2 analysis pipeline.

    Returns the JSON-serialisable report dict.
    """
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = Path(output_dir) if output_dir else vpath.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = vpath.stem

    # --- Load reference baseline ---
    ref_path = Path(REFERENCE_BASELINE_PATH)
    if not ref_path.exists():
        if verbose:
            print(f"Reference baseline not found — building from {video_path} ...")
        baseline = build_reference_baseline(video_path)
    else:
        baseline = load_reference_baseline()
        if verbose:
            print(f"Reference baseline loaded from {ref_path}")

    # --- Step 1: Extract poses ---
    if verbose:
        print(f"\nAnalysing: {vpath.name}")
    frame_poses, video_meta = extract_poses(video_path, verbose=verbose)
    fps = video_meta["fps"]

    # --- Step 2: Calculate metrics ---
    if verbose:
        print("  Calculating metrics...")
    metrics = calculate_metrics(frame_poses, fps)

    # --- Step 3: Detect phases ---
    if verbose:
        print("  Detecting phases...")
    phases = detect_phases(metrics, fps)
    if verbose:
        print_phase_summary(phases, fps)

    # --- Step 4: Run coaching rules ---
    if verbose:
        print("\n  Running coaching rules...")
    fault_map = run_all_rules(metrics, phases, baseline)

    # --- Step 5: Score ---
    result = build_scores(fault_map, phases, baseline, video_meta)

    # --- Step 6: Print & save JSON report ---
    if verbose:
        print_report(result)

    json_path = out_dir / f"{stem}_battingiq.json"
    save_json_report(result, str(json_path))
    if verbose:
        print(f"\n  JSON report → {json_path}")

    # --- Step 7: Annotated video ---
    if verbose:
        print("  Generating annotated video...")
    video_out = out_dir / f"{stem}_battingiq_annotated.mp4"
    annotate_video(video_path, result, metrics, str(video_out))

    if verbose:
        print(f"\nDone. Output in: {out_dir}/")

    from report_generator import build_json_report
    return build_json_report(result)


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


def run_full_analysis(video_path: str, output_dir: str = None) -> dict:
    """Programmatic entry point for the FastAPI wrapper. Returns the full report as a dict."""
    return analyse(video_path, output_dir=output_dir, verbose=False)
