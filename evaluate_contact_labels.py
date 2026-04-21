"""
Evaluate automatic contact estimates against manually validated contact truth.

Usage:
    python evaluate_contact_labels.py path/to/contact_truth.csv [--output contact_eval.csv]

Expected CSV columns:
    upload_id or filename
    estimated_contact_frame (or estimated_contact_original_frame)
    manual_contact_frame (or validated_contact_frame)

Optional grouping columns:
    fps
    filming_angle
    handedness
    shot_type
    shot_family
    shot_variant
    detector_version
    estimated_confidence
"""

from __future__ import annotations

import argparse
import csv
import json
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


def _first_present(row: dict, *fields: str) -> str | None:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return None


def _derive_shot_type(row: dict) -> str:
    explicit = _group_key(row, "shot_type")
    if explicit != "unknown":
        return explicit

    family = _group_key(row, "shot_family")
    variant = _group_key(row, "shot_variant")
    if family == "unknown" and variant == "unknown":
        return "unknown"
    if variant == "unknown":
        return family
    if family == "unknown":
        return variant
    return f"{family}:{variant}"


def _group_summary(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "mean_signed_error": round(sum(r["signed_frame_error"] for r in rows) / len(rows), 3),
        "mean_absolute_error": round(sum(r["absolute_frame_error"] for r in rows) / len(rows), 3),
    }


def evaluate_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    detailed: list[dict] = []
    valid_rows: list[dict] = []

    for row in rows:
        estimated = _safe_int(
            _first_present(
                row,
                "estimated_contact_original_frame",
                "estimated_contact_frame",
                "contact_frame",
            )
        )
        manual = _safe_int(_first_present(row, "manual_contact_frame", "validated_contact_frame"))
        if estimated is None or manual is None:
            continue
        signed_error = estimated - manual
        valid = {
            **row,
            "row_id": _first_present(row, "upload_id", "filename") or "unknown",
            "estimated_contact_frame": estimated,
            "manual_contact_frame": manual,
            "signed_frame_error": signed_error,
            "absolute_frame_error": abs(signed_error),
            "shot_type": _derive_shot_type(row),
            "detector_version": _group_key(row, "detector_version"),
            "estimated_confidence": _group_key(row, "estimated_confidence"),
        }
        valid_rows.append(valid)
        detailed.append(valid)

    summary: dict = {
        "rows_compared": len(valid_rows),
        "mean_signed_error": 0.0,
        "mean_absolute_error": 0.0,
        "by_fps": {},
        "by_filming_angle": {},
        "by_handedness": {},
        "by_shot_type": {},
        "by_detector_version": {},
        "by_estimated_confidence": {},
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
        ("handedness", "by_handedness"),
        ("shot_type", "by_shot_type"),
        ("detector_version", "by_detector_version"),
        ("estimated_confidence", "by_estimated_confidence"),
    ):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in valid_rows:
            grouped[_group_key(row, field)].append(row)
        summary[out_key] = {key: _group_summary(group) for key, group in grouped.items()}

    return detailed, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate automatic vs manual contact labels")
    parser.add_argument(
        "csv_path",
        help="CSV exported from contact truth labels with estimated and manual contact columns",
    )
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

    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary written to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
