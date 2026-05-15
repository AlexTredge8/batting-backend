#!/usr/bin/env python3
"""
D3-new: Rule health audit measuring per-rule firing rate by tier.
Reads fault lists from JSON reports in a local_runs/ directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / "calibration_output" / "d1_new_baseline"

ACTIVE_RULES: list[tuple[str, str]] = [
    ("A1", "Bat path / wrist spread"),
    ("A3", "Contact window compression"),
    ("A5", "Shoulder-hip lag gap"),
    ("A6", "Early opening fraction"),
    ("T2", "Early head change"),
    ("S1", "Hip shift"),
    ("S2", "Post-contact instability"),
    ("S4", "Post-contact rotation"),
    ("F1", "Sync frames (peak/FFD)"),
    ("F3", "Timing ratio"),
    ("F4", "Pause at peak"),
    ("F5", "Mid-swing hitch"),
    ("F6", "Follow-through"),
]
ACTIVE_RULE_IDS = {r[0] for r in ACTIVE_RULES}

TIER_ORDER = ("Beginner", "Average", "Good Club", "Elite")

PILLAR_RULES: dict[str, list[str]] = {
    "access":    ["A1", "A3", "A5", "A6"],
    "tracking":  ["T2"],
    "stability": ["S1", "S2", "S4"],
    "flow":      ["F1", "F3", "F4", "F5", "F6"],
}

# Max possible deduction per rule (from config.py as of 2026-04-25)
MAX_DEDUCTION: dict[str, int] = {
    "A1": 10, "A3": 3, "A5": 6, "A6": 6,
    "T2": 6,
    "S1": 10, "S2": 6, "S4": 6,
    "F1": 4, "F3": 9, "F4": 6, "F5": 6, "F6": 8,
}
PILLAR_MAX = 25


def _clean(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _fmt(v: Any, p: int = 2) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{float(v):.{p}f}"


def _load_json_reports(local_runs_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for json_path in sorted(local_runs_dir.glob("*/*_battingiq.json")):
        with json_path.open(encoding="utf-8") as fh:
            reports.append(json.load(fh))
    return reports


def _extract_video_name(report: dict[str, Any], fallback_path: str) -> str:
    meta = report.get("metadata") or {}
    name = _clean(meta.get("video_name") or meta.get("filename") or "")
    if not name:
        name = Path(fallback_path).parent.name
    return name


def _extract_tier_from_gt(filename: str, gt_lookup: dict[str, str]) -> str:
    key_full = Path(filename).name.casefold()
    key_stem = Path(filename).stem.casefold()
    return gt_lookup.get(key_full) or gt_lookup.get(key_stem) or ""


def _extract_rule_deductions(report: dict[str, Any]) -> dict[str, int]:
    deductions: dict[str, int] = {}
    for pillar_info in (report.get("pillars") or {}).values():
        for fault in (pillar_info.get("faults") or []):
            rule_id = _clean(fault.get("rule_id", ""))
            ded = fault.get("deduction", 0)
            if rule_id:
                # sum in case a rule fires multiple faults (shouldn't happen but safe)
                deductions[rule_id] = deductions.get(rule_id, 0) + int(ded)
    return deductions


def _build_video_table(reports: list[dict[str, Any]],
                       gt_lookup: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in reports:
        fn = _extract_video_name(r, "")
        tier = _extract_tier_from_gt(fn, gt_lookup)
        deds = _extract_rule_deductions(r)
        row: dict[str, Any] = {"filename": fn, "tier": tier}
        for rule_id, _ in ACTIVE_RULES:
            row[rule_id] = deds.get(rule_id, 0)
        # also capture any suspended-rule firings for completeness
        for rule_id, ded in deds.items():
            if rule_id not in ACTIVE_RULE_IDS:
                row[f"suspended_{rule_id}"] = ded
        rows.append(row)
    return pd.DataFrame(rows)


def _status_flag(beg_rate: float, eli_rate: float,
                 disc: float) -> str:
    if eli_rate > beg_rate + 0.25 or disc < -1.0:
        return "inverted"
    if abs(beg_rate - eli_rate) < 0.25 and abs(disc) < 1.0:
        return "flat"
    if beg_rate < 0.4 and eli_rate < 0.4:
        return "sparse"
    return "healthy"


def _build_rule_health(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rule_id, rule_name in ACTIVE_RULES:
        if rule_id not in df.columns:
            continue
        tier_stats: dict[str, dict[str, float]] = {}
        for tier in TIER_ORDER:
            t = df[df["tier"] == tier][rule_id]
            n = len(t)
            if n == 0:
                tier_stats[tier] = {"fire_rate": float("nan"), "mean_ded": float("nan")}
            else:
                tier_stats[tier] = {
                    "fire_rate": float((t > 0).sum()) / n,
                    "mean_ded": float(t.mean()),
                }

        beg_rate = tier_stats["Beginner"]["fire_rate"]
        eli_rate = tier_stats["Elite"]["fire_rate"]
        beg_ded  = tier_stats["Beginner"]["mean_ded"]
        eli_ded  = tier_stats["Elite"]["mean_ded"]
        disc = (
            beg_ded - eli_ded
            if pd.notna(beg_ded) and pd.notna(eli_ded)
            else float("nan")
        )
        flag = (
            _status_flag(beg_rate, eli_rate, disc)
            if pd.notna(beg_rate) and pd.notna(eli_rate) and pd.notna(disc)
            else "unknown"
        )

        row: dict[str, Any] = {
            "rule_id":   rule_id,
            "rule_name": rule_name,
            "beginner_fire_rate":   beg_rate,
            "average_fire_rate":    tier_stats["Average"]["fire_rate"],
            "good_club_fire_rate":  tier_stats["Good Club"]["fire_rate"],
            "elite_fire_rate":      eli_rate,
            "beginner_mean_deduction": beg_ded,
            "elite_mean_deduction":    eli_ded,
            "discrimination_score":    disc,
            "status_flag":             flag,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _build_summary_text(
    health_df: pd.DataFrame,
    video_df: pd.DataFrame,
    mode_label: str,
    health_auto: pd.DataFrame | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"BattingIQ Rule Health Audit — D3-new")
    lines.append(f"Mode: {mode_label}")
    lines.append(f"Videos: {len(video_df)}")
    lines.append("")

    # Counts per status
    counts = health_df["status_flag"].value_counts().to_dict()
    lines.append("── STATUS COUNTS ──")
    for s in ("healthy", "flat", "inverted", "sparse"):
        n = counts.get(s, 0)
        ids = sorted(health_df[health_df["status_flag"] == s]["rule_id"].tolist())
        lines.append(f"  {s:<10}: {n}  [{', '.join(ids) if ids else 'none'}]")
    lines.append("")

    # CRITICAL: validated-mode inversions
    inv_rows = health_df[health_df["status_flag"] == "inverted"]
    lines.append("── CRITICAL: VALIDATED-MODE INVERSIONS ──")
    lines.append("  These rules fire MORE on Elite/Good Club than Beginner even with correct anchor frames.")
    lines.append("  Detection fixing will NOT resolve these — the measurement concept is wrong.")
    if inv_rows.empty:
        lines.append("  None.")
    else:
        for _, r in inv_rows.iterrows():
            lines.append(
                f"  {r['rule_id']}  {r['rule_name']}"
                f"  beg_rate={_fmt(r['beginner_fire_rate'])}  eli_rate={_fmt(r['elite_fire_rate'])}"
                f"  disc={_fmt(r['discrimination_score'])}  beg_ded={_fmt(r['beginner_mean_deduction'])}"
                f"  eli_ded={_fmt(r['elite_mean_deduction'])}"
            )
    lines.append("")

    # CRITICAL: Beginner rule gaps
    lines.append("── CRITICAL: BEGINNER RULE GAPS PER PILLAR ──")
    lines.append("  Max possible deduction = sum of MAX_DEDUCTION for active rules in pillar.")
    lines.append("  Mean Beginner deduction = average total deduction on Beginner videos.")
    beg_df = video_df[video_df["tier"] == "Beginner"]
    for pillar, rule_ids in PILLAR_RULES.items():
        max_ded = sum(MAX_DEDUCTION.get(r, 0) for r in rule_ids if r in health_df["rule_id"].values)
        mean_ded = beg_df[[r for r in rule_ids if r in beg_df.columns]].sum(axis=1).mean() if not beg_df.empty else float("nan")
        gap = max_ded - mean_ded if pd.notna(mean_ded) else float("nan")
        lines.append(
            f"  {pillar:<12}  max_possible={max_ded:>3}  mean_beg_deducted={_fmt(mean_ded, 1)}"
            f"  gap={_fmt(gap, 1)} pts"
        )
    lines.append("")

    # Per-pillar discrimination score totals
    lines.append("── PER-PILLAR DISCRIMINATION SCORE TOTALS ──")
    lines.append("  (sum of beginner_mean_ded - elite_mean_ded across all active rules in pillar)")
    lines.append("  Positive = pillar rules are collectively penalising Beginners more than Elite (good).")
    lines.append("  Negative = pillar rules are on net favouring Beginners over Elite (bad).")
    for pillar, rule_ids in PILLAR_RULES.items():
        pil_df = health_df[health_df["rule_id"].isin(rule_ids)]
        total_disc = pil_df["discrimination_score"].sum()
        lines.append(f"  {pillar:<12}  total_disc={_fmt(total_disc)}")
    lines.append("")

    # Per-rule firing table
    lines.append("── PER-RULE FIRING TABLE ──")
    lines.append(f"  {'rule':<5}  {'status':<10}  {'beg%':>5}  {'avg%':>5}  {'gc%':>5}  {'eli%':>5}  {'disc':>6}  rule_name")
    for _, r in health_df.iterrows():
        lines.append(
            f"  {r['rule_id']:<5}  {r['status_flag']:<10}"
            f"  {_fmt(r['beginner_fire_rate']*100 if pd.notna(r['beginner_fire_rate']) else float('nan'), 0):>5}"
            f"  {_fmt(r['average_fire_rate']*100 if pd.notna(r['average_fire_rate']) else float('nan'), 0):>5}"
            f"  {_fmt(r['good_club_fire_rate']*100 if pd.notna(r['good_club_fire_rate']) else float('nan'), 0):>5}"
            f"  {_fmt(r['elite_fire_rate']*100 if pd.notna(r['elite_fire_rate']) else float('nan'), 0):>5}"
            f"  {_fmt(r['discrimination_score']):>6}"
            f"  {r['rule_name']}"
        )
    lines.append("")

    # Comparison note (validated vs auto)
    if health_auto is not None:
        lines.append("── DETECTION SENSITIVITY COMPARISON (validated vs auto) ──")
        lines.append("  'detector-sensitive': rule changes status between auto and validated")
        lines.append("  → status improves in validated: detection is masking the rule's true signal")
        lines.append("  → status same flat/inverted in both: concept/measurement problem, not detection")
        lines.append("")
        for _, rv in health_df.iterrows():
            ra_rows = health_auto[health_auto["rule_id"] == rv["rule_id"]]
            if ra_rows.empty:
                continue
            ra = ra_rows.iloc[0]
            val_status = rv["status_flag"]
            auto_status = ra["status_flag"]
            if val_status != auto_status:
                lines.append(
                    f"  {rv['rule_id']}  auto={auto_status:<10}  validated={val_status:<10}"
                    f"  → detector-sensitive"
                )
            else:
                lines.append(
                    f"  {rv['rule_id']}  auto={auto_status:<10}  validated={val_status:<10}"
                    f"  → same in both modes"
                )

    return "\n".join(lines) + "\n"


def _load_gt_lookup(gt_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    df = pd.read_csv(gt_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for _, row in df.iterrows():
        fn = _clean(row.get("filename", ""))
        tier = _clean(row.get("tier", ""))
        if fn and tier:
            lookup[fn.casefold()] = tier
            lookup[Path(fn).stem.casefold()] = tier
    return lookup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default=str(BASE))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--gt-csv", default=None,
                   help="Path to coach_ground_truth_from_screenshot.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base_dir).expanduser()
    if not base.is_absolute():
        base = (SCRIPT_DIR / base).resolve()
    out = Path(args.output_dir).expanduser() if args.output_dir else base
    if not out.is_absolute():
        out = (SCRIPT_DIR / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    gt_path = (
        Path(args.gt_csv).expanduser()
        if args.gt_csv
        else SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"
    )
    gt_lookup = _load_gt_lookup(gt_path)

    # ── Validated mode ──
    val_runs = base / "validated" / "local_runs"
    val_reports = _load_json_reports(val_runs)
    val_video_df = _build_video_table(val_reports, gt_lookup)
    val_health_df = _build_rule_health(val_video_df)

    # ── Auto mode ──
    auto_runs = base / "auto" / "local_runs"
    auto_reports = _load_json_reports(auto_runs)
    auto_video_df = _build_video_table(auto_reports, gt_lookup)
    auto_health_df = _build_rule_health(auto_video_df)

    # Output columns
    health_cols = [
        "rule_id", "rule_name",
        "beginner_fire_rate", "average_fire_rate",
        "good_club_fire_rate", "elite_fire_rate",
        "beginner_mean_deduction", "elite_mean_deduction",
        "discrimination_score", "status_flag",
    ]

    val_health_df.reindex(columns=health_cols).to_csv(
        out / "rule_health.csv", index=False)
    auto_health_df.reindex(columns=health_cols).to_csv(
        out / "rule_health_auto.csv", index=False)

    summary = _build_summary_text(
        val_health_df, val_video_df,
        "validated (correct anchor frames)",
        health_auto=auto_health_df,
    )
    (out / "rule_health_summary.txt").write_text(summary, encoding="utf-8")

    print(f"rule_health.csv:      {out / 'rule_health.csv'}")
    print(f"rule_health_auto.csv: {out / 'rule_health_auto.csv'}")
    print(f"rule_health_summary.txt: {out / 'rule_health_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
