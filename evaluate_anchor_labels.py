"""
Evaluate automatic six-anchor predictions against canonical truth.

Usage:
    python evaluate_anchor_labels.py path/to/anchor_truth.csv [--output anchor_eval.csv]
    python evaluate_anchor_labels.py current.csv --candidate-csv candidate.csv

Expected CSV columns:
    upload_id
    ai_<anchor_key>
    truth_<anchor_key>

Optional grouping columns:
    handedness
    shot_family
    shot_variant
    filming_angle
    fps
    detector_version
    ai_<anchor_key>_confidence
    truth_status
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ANCHOR_KEYS = (
    "setup_frame",
    "hands_start_up_frame",
    "front_foot_down_frame",
    "hands_peak_frame",
    "contact_frame",
    "follow_through_frame",
)


def _safe_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _group_key(row: dict, field: str) -> str:
    value = row.get(field)
    if value in (None, ""):
        return "unknown"
    return str(value)


def _fps_bucket(row: dict) -> str:
    fps = _safe_int(row.get("fps"))
    if fps is None:
        return "unknown"
    if fps <= 25:
        return "<=25"
    if fps <= 30:
        return "26-30"
    if fps <= 50:
        return "31-50"
    return "51+"


def _shot_type(row: dict) -> str:
    family = _group_key(row, "shot_family")
    variant = _group_key(row, "shot_variant")
    if family == "unknown" and variant == "unknown":
        return "unknown"
    if variant == "unknown":
        return family
    if family == "unknown":
        return variant
    return f"{family}:{variant}"


def _metric_summary(rows: list[dict]) -> dict:
    count = len(rows)
    if count == 0:
        return {
            "count": 0,
            "mean_signed_error": 0.0,
            "mean_absolute_error": 0.0,
            "within_1": 0.0,
            "within_2": 0.0,
            "within_3": 0.0,
        }

    within_1 = sum(1 for row in rows if row["absolute_frame_error"] <= 1)
    within_2 = sum(1 for row in rows if row["absolute_frame_error"] <= 2)
    within_3 = sum(1 for row in rows if row["absolute_frame_error"] <= 3)
    return {
        "count": count,
        "mean_signed_error": round(sum(row["signed_frame_error"] for row in rows) / count, 3),
        "mean_absolute_error": round(sum(row["absolute_frame_error"] for row in rows) / count, 3),
        "within_1": round(within_1 / count, 3),
        "within_2": round(within_2 / count, 3),
        "within_3": round(within_3 / count, 3),
    }


def evaluate_rows(rows: list[dict], prediction_prefix: str = "ai_") -> tuple[list[dict], dict]:
    detailed: list[dict] = []
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    unresolved_count = 0
    adjudication_count = 0

    for row in rows:
        truth_status = _group_key(row, "truth_status")
        if truth_status == "needs_adjudication":
            adjudication_count += 1
        if truth_status not in ("truth_ready", "replayed_on_candidate", "promoted_version_reference", "unknown"):
            unresolved_count += 1

        for anchor_key in ANCHOR_KEYS:
            predicted = _safe_int(row.get(f"{prediction_prefix}{anchor_key}"))
            truth = _safe_int(row.get(f"truth_{anchor_key}"))
            if predicted is None or truth is None:
                continue

            signed_error = predicted - truth
            detailed_row = {
                **row,
                "anchor_key": anchor_key,
                "predicted_frame": predicted,
                "truth_frame": truth,
                "signed_frame_error": signed_error,
                "absolute_frame_error": abs(signed_error),
                "detector_version": _group_key(row, "detector_version"),
                "confidence_bucket": _group_key(row, f"{prediction_prefix}{anchor_key}_confidence"),
                "fps_bucket": _fps_bucket(row),
                "shot_type": _shot_type(row),
            }
            detailed.append(detailed_row)
            grouped_rows[anchor_key].append(detailed_row)

    summary: dict = {
        "rows_compared": len({row.get("upload_id") for row in detailed}),
        "anchor_observations": len(detailed),
        "unresolved_case_count": unresolved_count,
        "needs_adjudication_count": adjudication_count,
        "overall": _metric_summary(detailed),
        "by_anchor": {},
        "by_handedness": {},
        "by_shot_type": {},
        "by_filming_angle": {},
        "by_fps_bucket": {},
        "by_detector_version": {},
        "by_confidence_bucket": {},
    }

    if not detailed:
        return detailed, summary

    summary["by_anchor"] = {
        anchor_key: _metric_summary(anchor_rows)
        for anchor_key, anchor_rows in grouped_rows.items()
    }

    for field, out_key in (
        ("handedness", "by_handedness"),
        ("shot_type", "by_shot_type"),
        ("filming_angle", "by_filming_angle"),
        ("fps_bucket", "by_fps_bucket"),
        ("detector_version", "by_detector_version"),
        ("confidence_bucket", "by_confidence_bucket"),
    ):
        bucketed: dict[str, list[dict]] = defaultdict(list)
        for detailed_row in detailed:
            bucketed[_group_key(detailed_row, field)].append(detailed_row)
        summary[out_key] = {
            bucket: _metric_summary(bucket_rows)
            for bucket, bucket_rows in bucketed.items()
        }

    return detailed, summary


def compare_candidate_rows(current_rows: list[dict], candidate_rows: list[dict]) -> dict:
    current_detail, current_summary = evaluate_rows(current_rows)
    candidate_detail, candidate_summary = evaluate_rows(candidate_rows)

    current_by_key = {(row.get("upload_id"), row["anchor_key"]): row for row in current_detail}
    candidate_by_key = {(row.get("upload_id"), row["anchor_key"]): row for row in candidate_detail}
    shared_keys = sorted(set(current_by_key) & set(candidate_by_key))

    improvement_rows = []
    for shared_key in shared_keys:
        current = current_by_key[shared_key]
        candidate = candidate_by_key[shared_key]
        improvement_rows.append({
            "upload_id": shared_key[0],
            "anchor_key": shared_key[1],
            "current_absolute_error": current["absolute_frame_error"],
            "candidate_absolute_error": candidate["absolute_frame_error"],
            "delta_absolute_error": round(
                candidate["absolute_frame_error"] - current["absolute_frame_error"],
                3,
            ),
        })

    by_anchor = {}
    for anchor_key in ANCHOR_KEYS:
        current_anchor = current_summary["by_anchor"].get(anchor_key, {})
        candidate_anchor = candidate_summary["by_anchor"].get(anchor_key, {})
        by_anchor[anchor_key] = {
            "current_mean_absolute_error": current_anchor.get("mean_absolute_error", 0.0),
            "candidate_mean_absolute_error": candidate_anchor.get("mean_absolute_error", 0.0),
            "delta_mean_absolute_error": round(
                candidate_anchor.get("mean_absolute_error", 0.0)
                - current_anchor.get("mean_absolute_error", 0.0),
                3,
            ),
        }

    return {
        "shared_anchor_observations": len(improvement_rows),
        "overall_delta_mean_absolute_error": round(
            candidate_summary["overall"]["mean_absolute_error"]
            - current_summary["overall"]["mean_absolute_error"],
            3,
        ),
        "by_anchor": by_anchor,
        "current_summary": current_summary,
        "candidate_summary": candidate_summary,
        "detail": improvement_rows,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate automatic vs canonical six-anchor labels")
    parser.add_argument("csv_path", help="CSV exported from the anchor calibration workflow")
    parser.add_argument("--output", "-o", default=None, help="Optional output CSV path")
    parser.add_argument("--candidate-csv", default=None, help="Optional candidate detector CSV to compare against current")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    output_path = Path(args.output) if args.output else csv_path.with_name("anchor_evaluation.csv")

    if args.candidate_csv:
        candidate_path = Path(args.candidate_csv)
        if not candidate_path.exists():
            print(f"Candidate CSV not found: {candidate_path}")
            return 1
        with open(candidate_path, newline="") as fh:
            candidate_rows = list(csv.DictReader(fh))
        comparison = compare_candidate_rows(rows, candidate_rows)
        _write_csv(output_path, comparison["detail"])
        summary_path = output_path.with_suffix(".comparison.json")
        with open(summary_path, "w") as fh:
            json.dump(comparison, fh, indent=2)
        print(f"Candidate comparison written to {summary_path}")
        return 0

    detailed, summary = evaluate_rows(rows)
    _write_csv(output_path, detailed)
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Compared cases: {summary['rows_compared']}")
    print(f"Anchor observations: {summary['anchor_observations']}")
    print(f"Mean absolute frame error: {summary['overall']['mean_absolute_error']}")
    print(f"Detailed evaluation written to {output_path}")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
