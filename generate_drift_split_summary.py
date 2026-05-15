#!/usr/bin/env python3
"""
Join drift_report.csv with heldout_split.csv and emit per-split drift summaries.

Produces:
  drift_summary_tuning.txt   — same metrics as drift_summary.txt, tuning videos only
  drift_summary_heldout.txt  — same metrics, held-out videos only
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ANCHOR_KEYS = (
    "setup_frame",
    "hands_start_up_frame",
    "front_foot_down_frame",
    "hands_peak_frame",
    "contact_frame",
    "follow_through_frame",
)
PILLARS = ("access", "tracking", "stability", "flow")
TIER_ORDER = ("Beginner", "Average", "Good Club", "Elite")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_float(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"


def _load_heldout_split(path: Path) -> dict[str, str]:
    """Return {filename.casefold(): split_label} from heldout_split.csv."""
    lookup: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            fn = _clean_text(row.get("filename", ""))
            split_val = _clean_text(row.get("split", "tuning")) or "tuning"
            if fn:
                lookup[fn.casefold()] = split_val
                lookup[Path(fn).stem.casefold()] = split_val
    return lookup


def _resolve_split(filename: str, lookup: dict[str, str]) -> str:
    key_full = Path(filename).name.casefold()
    key_stem = Path(filename).stem.casefold()
    return lookup.get(key_full) or lookup.get(key_stem) or "tuning"


def _build_split_summary_text(df: pd.DataFrame, split_label: str) -> str:
    lines: list[str] = []
    n = len(df)
    lines.append(f"BattingIQ drift summary — {split_label} set ({n} videos)")
    lines.append("Sign convention: positive delta = auto detected LATER than validated; negative = EARLIER.")
    lines.append("Score deltas reported as auto minus validated.")
    lines.append("")

    # Overall anchor MAEs
    lines.append("Mean absolute frame delta per anchor (all videos in set):")
    for anchor_key in ANCHOR_KEYS:
        col = anchor_key.replace("_frame", "_frame_delta")
        mae = df[col].abs().mean() if col in df.columns else None
        lines.append(f"  {anchor_key:<30}  {_format_float(mae)}")
    overall_frame_mae = df["mean_abs_frame_delta"].mean() if "mean_abs_frame_delta" in df.columns else None
    lines.append(f"  {'mean across all 6 anchors':<30}  {_format_float(overall_frame_mae)}")
    lines.append("")

    # Score delta overall and per pillar
    lines.append("Mean abs overall score delta (auto vs validated):")
    lines.append(f"  all {split_label}  {_format_float(df['overall_score_delta'].abs().mean())}")
    lines.append("")
    lines.append("Mean abs pillar score delta (auto vs validated):")
    for pillar in PILLARS:
        col = f"{pillar}_score_delta"
        mae = df[col].abs().mean() if col in df.columns else None
        lines.append(f"  {pillar:<14}  {_format_float(mae)}")
    lines.append("")

    # Per-tier breakdown
    lines.append("Per-tier breakdown:")
    tier_groups = []
    for tier in TIER_ORDER:
        t = df[df["tier"] == tier]
        if not t.empty:
            tier_groups.append((tier, t))
    # catch any tiers not in TIER_ORDER
    for tier in sorted(df["tier"].dropna().astype(str).unique()):
        if tier not in [g[0] for g in tier_groups]:
            tier_groups.append((tier, df[df["tier"] == tier]))

    for tier, t in tier_groups:
        frame_mae = t["mean_abs_frame_delta"].mean() if "mean_abs_frame_delta" in t.columns else None
        score_mae = t["overall_score_delta"].abs().mean() if "overall_score_delta" in t.columns else None
        lines.append(f"  {tier:<12}  n={len(t)}  frame_mae={_format_float(frame_mae)}  score_mae={_format_float(score_mae)}")
        for anchor_key in ANCHOR_KEYS:
            col = anchor_key.replace("_frame", "_frame_delta")
            mae = t[col].abs().mean() if col in t.columns else None
            lines.append(f"    {anchor_key:<30}  {_format_float(mae)}")
    lines.append("")

    # Extreme frame deltas
    extreme: list[str] = []
    for _, row in df.iterrows():
        exceeded = [
            anchor_key
            for anchor_key in ANCHOR_KEYS
            if pd.notna(row.get(anchor_key.replace("_frame", "_frame_delta")))
            and abs(float(row[anchor_key.replace("_frame", "_frame_delta")])) > 8
        ]
        if exceeded:
            extreme.append(f"  {row['filename']} [{row['tier']}]: {', '.join(exceeded)}")
    lines.append("Videos with any anchor delta > 8 frames:")
    if extreme:
        lines.extend(extreme)
    else:
        lines.append("  none")
    lines.append("")

    # Score-sensitive
    score_sensitive = df[df["overall_score_delta"].abs() > 10] if "overall_score_delta" in df.columns else pd.DataFrame()
    lines.append("Videos with |overall score delta| > 10:")
    if score_sensitive.empty:
        lines.append("  none")
    else:
        for _, row in score_sensitive.sort_values(["tier", "filename"]).iterrows():
            lines.append(
                f"  {row['filename']} [{row['tier']}]: delta={int(row['overall_score_delta'])}"
            )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit per-split drift summaries from drift_report.csv + heldout_split.csv.")
    parser.add_argument(
        "--drift-report",
        default=None,
        help="Path to drift_report.csv. Defaults to calibration_output/d1_new_baseline/drift_report.csv.",
    )
    parser.add_argument(
        "--heldout-split",
        default=None,
        help="Path to heldout_split.csv. Defaults to heldout_split.csv next to this script.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write drift_summary_tuning.txt and drift_summary_heldout.txt. Defaults to drift_report parent dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    drift_path = (
        Path(args.drift_report).expanduser()
        if args.drift_report
        else SCRIPT_DIR / "calibration_output" / "d1_new_baseline" / "drift_report.csv"
    )
    if not drift_path.is_absolute():
        drift_path = (SCRIPT_DIR / drift_path).resolve()
    if not drift_path.exists():
        raise SystemExit(f"drift_report.csv not found: {drift_path}")

    split_path = (
        Path(args.heldout_split).expanduser()
        if args.heldout_split
        else SCRIPT_DIR / "heldout_split.csv"
    )
    if not split_path.is_absolute():
        split_path = (SCRIPT_DIR / split_path).resolve()
    if not split_path.exists():
        raise SystemExit(f"heldout_split.csv not found: {split_path}")

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else drift_path.parent
    )
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    drift_df = pd.read_csv(drift_path, dtype=str, keep_default_na=False)
    numeric_cols = [c for c in drift_df.columns if c not in {"filename", "tier"}]
    for col in numeric_cols:
        drift_df[col] = pd.to_numeric(drift_df[col], errors="coerce")

    split_lookup = _load_heldout_split(split_path)
    drift_df["split"] = drift_df["filename"].apply(lambda fn: _resolve_split(fn, split_lookup))

    for split_label in ("tuning", "heldout"):
        subset = drift_df[drift_df["split"] == split_label].copy()
        if subset.empty:
            print(f"Warning: no {split_label} rows found in drift_report.csv")
            continue
        text = _build_split_summary_text(subset, split_label)
        out_path = output_dir / f"drift_summary_{split_label}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"{split_label} drift summary: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
