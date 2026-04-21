"""
Run BattingIQ analysis across a directory of videos and write a summary CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_analysis import run_full_analysis

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv"}


def _iter_videos(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )


def _rules_fired(report: dict) -> list[str]:
    rule_ids: list[str] = []
    for pillar in ("access", "tracking", "stability", "flow"):
        faults = report.get("pillars", {}).get(pillar, {}).get("faults", [])
        rule_ids.extend(fault.get("rule_id") for fault in faults if fault.get("rule_id"))
    return rule_ids


def _total_deduction(report: dict) -> int:
    total = 0
    for pillar in ("access", "tracking", "stability", "flow"):
        faults = report.get("pillars", {}).get(pillar, {}).get("faults", [])
        total += sum(int(fault.get("deduction", 0)) for fault in faults)
    return total


def _row_from_report(video: Path, report: dict) -> dict:
    contact = report.get("phases", {}).get("contact", {}) or {}
    metadata = report.get("metadata", {}) or {}
    return {
        "filename": video.name,
        "battingiq_score": report.get("battingiq_score"),
        "access_score": report.get("pillars", {}).get("access", {}).get("score"),
        "tracking_score": report.get("pillars", {}).get("tracking", {}).get("score"),
        "stability_score": report.get("pillars", {}).get("stability", {}).get("score"),
        "flow_score": report.get("pillars", {}).get("flow", {}).get("score"),
        "contact_frame": contact.get("frame"),
        "estimated_contact_frame": contact.get("estimated_frame"),
        "estimated_contact_original_frame": contact.get("estimated_original_frame"),
        "resolved_contact_original_frame": contact.get("resolved_original_frame"),
        "resolved_contact_source": contact.get("source"),
        "resolved_contact_status": contact.get("status"),
        "contact_confidence": contact.get("confidence"),
        "contact_detector_version": contact.get("detector_version") or metadata.get("contact_detector_version") or metadata.get("detector_version"),
        "anchor_detector_version": metadata.get("anchor_detector_version") or metadata.get("detector_version"),
        "detector_version": contact.get("detector_version") or metadata.get("contact_detector_version") or metadata.get("detector_version"),
        "rules_fired": ",".join(_rules_fired(report)),
        "total_deduction": _total_deduction(report),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BattingIQ analysis across a directory of videos.")
    parser.add_argument("directory", help="Directory containing batting videos")
    args = parser.parse_args()

    directory = Path(args.directory).expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")

    videos = _iter_videos(directory)
    if not videos:
        raise SystemExit(f"No supported video files found in {directory}")

    batch_output_dir = directory / "batch_output"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "batch_results.csv"
    fieldnames = [
        "filename",
        "battingiq_score",
        "access_score",
        "tracking_score",
        "stability_score",
        "flow_score",
        "contact_frame",
        "estimated_contact_frame",
        "estimated_contact_original_frame",
        "resolved_contact_original_frame",
        "resolved_contact_source",
        "resolved_contact_status",
        "contact_confidence",
        "contact_detector_version",
        "anchor_detector_version",
        "detector_version",
        "rules_fired",
        "total_deduction",
    ]

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for video in videos:
        print(f"Analysing {video.name}...")
        try:
            output_dir = batch_output_dir / video.stem
            report = run_full_analysis(str(video), output_dir=str(output_dir))
            rows.append(_row_from_report(video, report))
        except Exception as exc:  # pragma: no cover - batch path should continue on failures
            failures.append((video.name, str(exc)))
            print(f"  FAILED: {exc}")

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nBatch analysis complete.")
    print(f"  Videos processed: {len(rows)}")
    print(f"  Failures: {len(failures)}")
    print(f"  CSV: {csv_path}")
    if failures:
        print("  Failed files:")
        for filename, error in failures:
            print(f"    - {filename}: {error}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
