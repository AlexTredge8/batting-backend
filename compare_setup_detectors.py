#!/usr/bin/env python3
"""Compare setup detector v1 vs v2 against anchor truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ANCHOR_KEYS = (
    "setup_frame",
    "front_foot_down_frame",
    "hands_peak_frame",
    "contact_frame",
)
DOWNSTREAM_KEYS = (
    "front_foot_down_frame",
    "hands_peak_frame",
    "contact_frame",
)
TIER_ORDER = ("Beginner", "Average", "Good Club", "Elite")


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


def _load_score_tiers(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        tiers: dict[str, str] = {}
        for row in reader:
            if _clean_text(row.get("filmed_correctly")).upper() != "Y":
                continue
            filename = _clean_text(row.get("filename"))
            if filename:
                tiers[filename] = _clean_text(row.get("tier"))
        return tiers


def _load_anchor_truth(path: Path, tiers: dict[str, str]) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        truth: dict[str, dict[str, Any]] = {}
        for row in reader:
            filename = _clean_text(row.get("filename"))
            if not filename or filename not in tiers:
                continue
            truth[filename] = {
                "tier": tiers[filename],
                **{anchor_key: _coerce_int(row.get(anchor_key)) for anchor_key in ANCHOR_KEYS},
            }
        return truth


def _extract_anchor_frame(report: dict[str, Any], anchor_key: str) -> int | None:
    metadata = report.get("metadata") or {}
    anchor_frames = metadata.get("anchor_frames") or {}
    anchor_info = anchor_frames.get(anchor_key) or {}
    return _coerce_int(anchor_info.get("original_frame"))


def _load_reports(run_root: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for report_path in sorted((run_root / "local_runs").glob("*/*_battingiq.json")):
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        metadata = report.get("metadata") or {}
        filename = _clean_text(metadata.get("video_name")) or report_path.parent.name
        reports[filename] = {
            "setup_detector_version": _clean_text(metadata.get("setup_detector_version")),
            "anchor_detector_version": _clean_text(metadata.get("anchor_detector_version")),
            **{anchor_key: _extract_anchor_frame(report, anchor_key) for anchor_key in ANCHOR_KEYS},
        }
    return reports


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _build_rows(
    truth: dict[str, dict[str, Any]],
    v1_reports: dict[str, dict[str, Any]],
    v2_reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename in sorted(truth):
        row = {
            "filename": filename,
            "tier": truth[filename]["tier"],
        }
        for anchor_key in ANCHOR_KEYS:
            truth_frame = truth[filename].get(anchor_key)
            v1_frame = (v1_reports.get(filename) or {}).get(anchor_key)
            v2_frame = (v2_reports.get(filename) or {}).get(anchor_key)
            row[f"truth_{anchor_key}"] = truth_frame
            row[f"v1_{anchor_key}"] = v1_frame
            row[f"v2_{anchor_key}"] = v2_frame
            row[f"v1_{anchor_key}_abs_error"] = None if truth_frame is None or v1_frame is None else abs(v1_frame - truth_frame)
            row[f"v2_{anchor_key}_abs_error"] = None if truth_frame is None or v2_frame is None else abs(v2_frame - truth_frame)
        rows.append(row)
    return rows


def _anchor_stats(rows: list[dict[str, Any]], anchor_key: str) -> dict[str, Any]:
    usable = [
        row for row in rows
        if row.get(f"v1_{anchor_key}_abs_error") is not None and row.get(f"v2_{anchor_key}_abs_error") is not None
    ]
    if not usable:
        return {"count": 0, "v1_mae": None, "v2_mae": None, "v1_within_2": 0.0, "v2_within_2": 0.0, "v1_within_5": 0.0, "v2_within_5": 0.0, "improved": 0, "regressed": 0, "tied": 0}

    v1_errors = [row[f"v1_{anchor_key}_abs_error"] for row in usable]
    v2_errors = [row[f"v2_{anchor_key}_abs_error"] for row in usable]
    improved = sum(1 for v1, v2 in zip(v1_errors, v2_errors) if v2 < v1)
    regressed = sum(1 for v1, v2 in zip(v1_errors, v2_errors) if v2 > v1)
    tied = len(usable) - improved - regressed
    return {
        "count": len(usable),
        "v1_mae": sum(v1_errors) / len(v1_errors),
        "v2_mae": sum(v2_errors) / len(v2_errors),
        "v1_within_2": _rate(sum(1 for value in v1_errors if value <= 2), len(usable)),
        "v2_within_2": _rate(sum(1 for value in v2_errors if value <= 2), len(usable)),
        "v1_within_5": _rate(sum(1 for value in v1_errors if value <= 5), len(usable)),
        "v2_within_5": _rate(sum(1 for value in v2_errors if value <= 5), len(usable)),
        "improved": improved,
        "regressed": regressed,
        "tied": tied,
    }


def _rows_for_tier(rows: list[dict[str, Any]], tier: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("tier") == tier]


def _setup_regressions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regressions = []
    for row in rows:
        v1_error = row.get("v1_setup_frame_abs_error")
        v2_error = row.get("v2_setup_frame_abs_error")
        if v1_error is None or v2_error is None or v2_error <= v1_error:
            continue
        regressions.append({
            "filename": row["filename"],
            "tier": row["tier"],
            "truth": row.get("truth_setup_frame"),
            "v1": row.get("v1_setup_frame"),
            "v2": row.get("v2_setup_frame"),
            "v1_error": v1_error,
            "v2_error": v2_error,
            "delta": v2_error - v1_error,
        })
    return sorted(regressions, key=lambda row: (-row["delta"], row["filename"]))


def _write_detailed_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "filename",
        "tier",
    ]
    for anchor_key in ANCHOR_KEYS:
        fieldnames.extend([
            f"truth_{anchor_key}",
            f"v1_{anchor_key}",
            f"v2_{anchor_key}",
            f"v1_{anchor_key}_abs_error",
            f"v2_{anchor_key}_abs_error",
        ])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _summary_block(rows: list[dict[str, Any]], anchor_key: str, title: str) -> list[str]:
    lines = [title]
    overall = _anchor_stats(rows, anchor_key)
    lines.append(
        f"  Overall ({overall['count']} videos): v1 MAE {_format_float(overall['v1_mae'])} | "
        f"v2 MAE {_format_float(overall['v2_mae'])} | "
        f"within 2: {_format_pct(overall['v1_within_2'])} -> {_format_pct(overall['v2_within_2'])} | "
        f"within 5: {_format_pct(overall['v1_within_5'])} -> {_format_pct(overall['v2_within_5'])} | "
        f"improved/regressed/tied: {overall['improved']}/{overall['regressed']}/{overall['tied']}"
    )
    for tier in TIER_ORDER:
        tier_rows = _rows_for_tier(rows, tier)
        if not tier_rows:
            continue
        stats = _anchor_stats(tier_rows, anchor_key)
        lines.append(
            f"  {tier}: v1 MAE {_format_float(stats['v1_mae'])} | "
            f"v2 MAE {_format_float(stats['v2_mae'])} | "
            f"within 2: {_format_pct(stats['v1_within_2'])} -> {_format_pct(stats['v2_within_2'])} | "
            f"within 5: {_format_pct(stats['v1_within_5'])} -> {_format_pct(stats['v2_within_5'])}"
        )
    return lines


def build_report(rows: list[dict[str, Any]], v1_dir: Path, v2_dir: Path) -> str:
    lines: list[str] = []
    lines.append("Setup detector comparison (v1 vs v2)")
    lines.append(f"v1 run: {v1_dir}")
    lines.append(f"v2 run: {v2_dir}")
    lines.append("")
    lines.extend(_summary_block(rows, "setup_frame", "Setup accuracy"))
    lines.append("")
    regressions = _setup_regressions(rows)
    lines.append("Setup regressions (v2 worse than v1)")
    if not regressions:
        lines.append("  None")
    else:
        for row in regressions:
            lines.append(
                f"  {row['filename']} [{row['tier']}]: truth={row['truth']}, "
                f"v1={row['v1']} (|e|={row['v1_error']}), "
                f"v2={row['v2']} (|e|={row['v2_error']}), delta=+{row['delta']}"
            )
    lines.append("")
    lines.append("Cascade effect on downstream anchors")
    for anchor_key in DOWNSTREAM_KEYS:
        anchor_title = anchor_key.replace("_frame", "").replace("_", " ").title()
        lines.extend(_summary_block(rows, anchor_key, f"{anchor_title}"))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare setup detector v1 vs v2 run outputs.")
    parser.add_argument("--score-truth", required=True, help="Ground-truth score CSV with filename/tier/filmed_correctly columns.")
    parser.add_argument("--anchor-truth", required=True, help="Anchor truth CSV.")
    parser.add_argument("--v1-dir", required=True, help="Batch output root for setup v1 auto run.")
    parser.add_argument("--v2-dir", required=True, help="Batch output root for setup v2 auto run.")
    parser.add_argument("--output-txt", required=True, help="Text report path.")
    parser.add_argument("--output-csv", required=True, help="Detailed CSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    score_truth = Path(args.score_truth).expanduser().resolve()
    anchor_truth = Path(args.anchor_truth).expanduser().resolve()
    v1_dir = Path(args.v1_dir).expanduser().resolve()
    v2_dir = Path(args.v2_dir).expanduser().resolve()
    output_txt = Path(args.output_txt).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    tiers = _load_score_tiers(score_truth)
    truth = _load_anchor_truth(anchor_truth, tiers)
    rows = _build_rows(truth, _load_reports(v1_dir), _load_reports(v2_dir))

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    _write_detailed_csv(rows, output_csv)
    output_txt.write_text(build_report(rows, v1_dir, v2_dir), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
