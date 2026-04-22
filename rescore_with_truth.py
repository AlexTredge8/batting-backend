#!/usr/bin/env python3
"""
Re-run BattingIQ scores with manual anchor corrections applied from anchor_truth.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "calibration_output"
DEFAULT_ANCHOR_TRUTH = SCRIPT_DIR / "anchor_truth.csv"
DEFAULT_GROUND_TRUTH = SCRIPT_DIR / "ground_truth_scores.csv"
FALLBACK_GROUND_TRUTH = SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"
DEFAULT_VIDEO_DIR = SCRIPT_DIR / "ground_truth_videos"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
PILLARS = ("access", "tracking", "stability", "flow")
ANCHOR_KEYS = (
    "setup_frame",
    "hands_start_up_frame",
    "front_foot_down_frame",
    "hands_peak_frame",
    "contact_frame",
    "follow_through_frame",
)
TIER_ORDER = {"Elite": 0, "Good Club": 1, "Average": 2, "Beginner": 3}

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "battingiq-mplconfig"))
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("LOCAL_MODE", "1")

import pose_extractor  # noqa: E402

pose_extractor.LOCAL_MODEL_COMPLEXITY = 2

from run_analysis import run_full_analysis  # noqa: E402


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_int(value: Any) -> int | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def _resolve_path(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    if path.name == DEFAULT_GROUND_TRUTH.name and FALLBACK_GROUND_TRUTH.exists():
        return FALLBACK_GROUND_TRUTH.resolve()
    raise FileNotFoundError(f"Path not found: {path}")


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: _clean_text(value) for key, value in row.items()} for row in reader]


def _discover_videos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES
    )


def _build_video_lookup(videos: list[Path]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    exact_lookup: dict[str, list[Path]] = {}
    stem_lookup: dict[str, list[Path]] = {}
    for video in videos:
        exact_lookup.setdefault(video.name.casefold(), []).append(video)
        stem_lookup.setdefault(video.stem.casefold(), []).append(video)
    return exact_lookup, stem_lookup


def _match_video(
    filename: str,
    exact_lookup: dict[str, list[Path]],
    stem_lookup: dict[str, list[Path]],
) -> Path | None:
    raw_name = _clean_text(filename)
    if not raw_name:
        return None
    name_key = Path(raw_name).name.casefold()
    stem_key = Path(raw_name).stem.casefold()
    candidates = exact_lookup.get(name_key) or stem_lookup.get(stem_key) or []
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    desired_name = Path(raw_name).name.casefold()
    for candidate in candidates:
        if candidate.name.casefold() == desired_name:
            return candidate
    return sorted(candidates, key=lambda path: (path.name.casefold(), str(path.parent).casefold()))[0]


def _prepare_ground_truth_lookup(path: Path) -> dict[str, dict[str, str]]:
    rows = _load_csv_rows(path)
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        filename = _clean_text(row.get("filename"))
        if filename:
            lookup[filename.casefold()] = row
            lookup[Path(filename).stem.casefold()] = row
    return lookup


def _read_report_scores(report: dict[str, Any]) -> dict[str, Any]:
    pillars = report.get("pillars", {}) or {}
    phases = report.get("phases", {}) or {}
    return {
        "overall": _coerce_int(report.get("battingiq_score")),
        "pillars": {
            pillar: _coerce_int((pillars.get(pillar) or {}).get("score"))
            for pillar in PILLARS
        },
        "contact_frame": _coerce_int((phases.get("contact") or {}).get("frame")),
    }


def _anchor_map_to_original_frames(anchor_frames: dict[str, dict[str, Any]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for anchor_key in ANCHOR_KEYS:
        anchor = anchor_frames.get(anchor_key) or {}
        value = anchor.get("original_frame")
        result[anchor_key] = _coerce_int(value)
    return result


def _merge_anchor_overrides(
    auto_anchor_frames: dict[str, int | None],
    corrections: dict[str, int | None],
) -> dict[str, int | None]:
    merged = dict(auto_anchor_frames)
    for key, value in corrections.items():
        if value is not None:
            merged[key] = value
    return merged


def _frames_corrected(corrections: dict[str, int | None]) -> list[str]:
    return [key for key in ANCHOR_KEYS if corrections.get(key) is not None]


def _tier_gap(rows: list[dict[str, Any]], tier_a: str, tier_b: str, key: str) -> str:
    tier_values = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None:
            tier_values[row["tier"]].append(int(value))

    if tier_a not in tier_values or tier_b not in tier_values:
        return "n/a"
    return f"{min(tier_values[tier_b]) - min(tier_values[tier_a]):.1f}"


def _summary_for(rows: list[dict[str, Any]], label: str, key: str) -> list[str]:
    lines = [label]
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None:
            grouped[row["tier"]].append(int(value))

    for tier in ("Elite", "Good Club", "Average", "Beginner"):
        if tier in grouped:
            lines.append(f"  {tier}: min {min(grouped[tier])} / mean {sum(grouped[tier]) / len(grouped[tier]):.1f}")

    if "Beginner" in grouped and "Good Club" in grouped:
        lines.append(
            f"  Beginner -> Good Club gap: {min(grouped['Good Club']) - min(grouped['Beginner']):.1f}"
        )
    if "Average" in grouped and "Elite" in grouped:
        lines.append(
            f"  Average -> Elite gap: {min(grouped['Elite']) - min(grouped['Average']):.1f}"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-score videos with manual anchor corrections.")
    parser.add_argument(
        "--anchor-truth",
        default=str(DEFAULT_ANCHOR_TRUTH),
        help="CSV containing manual anchor corrections.",
    )
    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH.name),
        help="CSV containing manual scores and tiers. Defaults to ground_truth_scores.csv in this directory.",
    )
    parser.add_argument(
        "--video-dir",
        default=str(DEFAULT_VIDEO_DIR),
        help="Directory containing the source batting videos.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for the comparison CSV.",
    )
    args = parser.parse_args()

    anchor_truth_path = Path(args.anchor_truth).expanduser()
    if not anchor_truth_path.is_absolute():
        anchor_truth_path = (SCRIPT_DIR / anchor_truth_path).resolve()
    if not anchor_truth_path.exists():
        raise SystemExit(f"Anchor truth file not found: {anchor_truth_path}")

    ground_truth_path = _resolve_path((SCRIPT_DIR / args.ground_truth).expanduser())
    video_dir = Path(args.video_dir).expanduser()
    if not video_dir.is_absolute():
        cwd_candidate = Path.cwd() / video_dir
        video_dir = cwd_candidate if cwd_candidate.exists() else (SCRIPT_DIR / video_dir)
    video_dir = video_dir.resolve()
    if not video_dir.exists() or not video_dir.is_dir():
        raise SystemExit(f"Video directory not found: {video_dir}")

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = _discover_videos(video_dir)
    if not videos:
        raise SystemExit(f"No supported video files found in {video_dir}")

    exact_lookup, stem_lookup = _build_video_lookup(videos)
    manual_lookup = _prepare_ground_truth_lookup(ground_truth_path)
    corrections_rows = _load_csv_rows(anchor_truth_path)

    comparison_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for row in corrections_rows:
        filename = _clean_text(row.get("filename"))
        if not filename:
            continue

        corrections = {
            anchor_key: _coerce_int(row.get(anchor_key))
            for anchor_key in ANCHOR_KEYS
        }
        if not any(value is not None for value in corrections.values()):
            continue

        video_path = _match_video(filename, exact_lookup, stem_lookup)
        if video_path is None:
            failures.append(f"{filename}: video not found in {video_dir}")
            continue

        manual_row = manual_lookup.get(filename.casefold()) or manual_lookup.get(Path(filename).stem.casefold())
        if not manual_row:
            failures.append(f"{filename}: no matching row in {ground_truth_path}")
            continue

        tier = _clean_text(manual_row.get("tier"))
        manual_overall = _coerce_int(manual_row.get("overall_score"))

        auto_report = run_full_analysis(str(video_path), output_dir=str(output_dir / "auto_runs" / video_path.stem))
        auto_scores = _read_report_scores(auto_report)
        auto_anchor_frames = _anchor_map_to_original_frames((auto_report.get("metadata", {}) or {}).get("anchor_frames", {}) or {})

        corrected_anchor_frames = _merge_anchor_overrides(auto_anchor_frames, corrections)
        corrected_report = run_full_analysis(
            str(video_path),
            output_dir=str(output_dir / "corrected_runs" / video_path.stem),
            anchor_frames=corrected_anchor_frames,
        )
        corrected_scores = _read_report_scores(corrected_report)

        comparison_rows.append(
            {
                "filename": video_path.name,
                "tier": tier,
                "manual_overall": manual_overall,
                "auto_overall": auto_scores["overall"],
                "auto_access": auto_scores["pillars"]["access"],
                "auto_tracking": auto_scores["pillars"]["tracking"],
                "auto_stability": auto_scores["pillars"]["stability"],
                "auto_flow": auto_scores["pillars"]["flow"],
                "corrected_overall": corrected_scores["overall"],
                "corrected_access": corrected_scores["pillars"]["access"],
                "corrected_tracking": corrected_scores["pillars"]["tracking"],
                "corrected_stability": corrected_scores["pillars"]["stability"],
                "corrected_flow": corrected_scores["pillars"]["flow"],
                "overall_delta": None if auto_scores["overall"] is None or corrected_scores["overall"] is None else corrected_scores["overall"] - auto_scores["overall"],
                "access_delta": None if auto_scores["pillars"]["access"] is None or corrected_scores["pillars"]["access"] is None else corrected_scores["pillars"]["access"] - auto_scores["pillars"]["access"],
                "tracking_delta": None if auto_scores["pillars"]["tracking"] is None or corrected_scores["pillars"]["tracking"] is None else corrected_scores["pillars"]["tracking"] - auto_scores["pillars"]["tracking"],
                "stability_delta": None if auto_scores["pillars"]["stability"] is None or corrected_scores["pillars"]["stability"] is None else corrected_scores["pillars"]["stability"] - auto_scores["pillars"]["stability"],
                "flow_delta": None if auto_scores["pillars"]["flow"] is None or corrected_scores["pillars"]["flow"] is None else corrected_scores["pillars"]["flow"] - auto_scores["pillars"]["flow"],
                "frames_corrected": json.dumps(_frames_corrected(corrections)),
            }
        )
        print(f"Processed {video_path.name}")

    if not comparison_rows:
        raise SystemExit("No videos were rescored. Check the anchor truth file and video directory.")

    comparison_path = output_dir / "rescore_comparison.csv"
    fieldnames = [
        "filename",
        "tier",
        "manual_overall",
        "auto_overall",
        "auto_access",
        "auto_tracking",
        "auto_stability",
        "auto_flow",
        "corrected_overall",
        "corrected_access",
        "corrected_tracking",
        "corrected_stability",
        "corrected_flow",
        "overall_delta",
        "access_delta",
        "tracking_delta",
        "stability_delta",
        "flow_delta",
        "frames_corrected",
    ]
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print()
    for line in _summary_for(comparison_rows, "Tier separation before correction", "auto_overall"):
        print(line)
    print()
    for line in _summary_for(comparison_rows, "Tier separation after correction", "corrected_overall"):
        print(line)
    print()
    print(f"Comparison CSV: {comparison_path}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
