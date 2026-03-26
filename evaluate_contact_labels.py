"""
Compare automatic contact estimates against manually validated contact frames.

Usage:
    python evaluate_contact_labels.py path/to/contact_labels.csv [--output contact_eval.csv]

Expected CSV columns:
    filename
    estimated_contact_frame
    manual_contact_frame

Optional grouping columns:
    fps
    filming_angle
    shot_type
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _safe_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _group_key(row: dict, field: str) -> str:
    return row.get(field) or "unknown"


def evaluate_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    detailed: list[dict] = []
    valid_rows: list[dict] = []

    for row in rows:
        estimated = _safe_int(row.get("estimated_contact_frame") or row.get("contact_frame"))
        manual = _safe_int(row.get("manual_contact_frame"))
        if estimated is None or manual is None:
            continue
        signed_error = estimated - manual
        valid = {
            **row,
            "estimated_contact_frame": estimated,
            "manual_contact_frame": manual,
            "signed_frame_error": signed_error,
            "absolute_frame_error": abs(signed_error),
        }
        valid_rows.append(valid)
        detailed.append(valid)

    summary: dict = {
        "rows_compared": len(valid_rows),
        "mean_signed_error": 0.0,
        "mean_absolute_error": 0.0,
        "by_fps": {},
        "by_filming_angle": {},
        "by_shot_type": {},
    }
    if not valid_rows:
        return detailed, summary

    summary["mean_signed_error"] = round(
        sum(r["signed_frame_error"] for r in valid_rows) / len(valid_rows), 3
    )
    summary["mean_absolute_error"] = round(
        sum(r["absolute_frame_error"] for r in valid_rows) / len(valid_rows), 3
    )

    for field, out_key in (
        ("fps", "by_fps"),
        ("filming_angle", "by_filming_angle"),
        ("shot_type", "by_shot_type"),
    ):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in valid_rows:
            grouped[_group_key(row, field)].append(row)
        summary[out_key] = {
            key: {
                "count": len(group),
                "mean_signed_error": round(sum(r["signed_frame_error"] for r in group) / len(group), 3),
                "mean_absolute_error": round(sum(r["absolute_frame_error"] for r in group) / len(group), 3),
            }
            for key, group in grouped.items()
        }

    return detailed, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate automatic vs manual contact labels")
    parser.add_argument("csv_path", help="CSV with estimated_contact_frame and manual_contact_frame columns")
    parser.add_argument("--output", "-o", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    detailed, summary = evaluate_rows(rows)
    print(f"Compared rows: {summary['rows_compared']}")
    print(f"Mean signed frame error: {summary['mean_signed_error']}")
    print(f"Mean absolute frame error: {summary['mean_absolute_error']}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = csv_path.with_name("contact_evaluation.csv")

    if detailed:
        fieldnames = list(detailed[0].keys())
        with open(output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed)
        print(f"Detailed evaluation written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
