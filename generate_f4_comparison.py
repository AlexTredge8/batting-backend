#!/usr/bin/env python3
"""
Generate F4 hands_peak v1 vs v2 comparison report.

Reads:
  - calibration_output/f4_hands_peak_v2/auto/local_runs/   (v2 auto results)
  - calibration_output/f4_hands_peak_v1_baseline/local_runs/  (v1 auto results)
  - calibration_output/f4_hands_peak_v2/validated/local_runs/ (validated reference)
  - anchor_truth.csv  (ground truth anchor frames)
  - ground_truth_scores.csv  (tier labels + manual overall scores)

Emits: calibration_output/f4_hands_peak_v2/f4_comparison.txt
"""

from __future__ import annotations

import json
import pathlib
from statistics import mean

import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "calibration_output" / "f4_hands_peak_v2"

V2_AUTO_DIR = OUT_DIR / "auto" / "local_runs"
V1_AUTO_DIR = SCRIPT_DIR / "calibration_output" / "f4_hands_peak_v1_baseline" / "local_runs"
VAL_DIR = OUT_DIR / "validated" / "local_runs"

ANCHOR_TRUTH = SCRIPT_DIR / "anchor_truth.csv"
GT_SCORES = SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_json(run_dir: pathlib.Path, stem: str) -> dict | None:
    folder = run_dir / stem
    if not folder.exists():
        return None
    for f in folder.iterdir():
        if f.suffix == ".json":
            try:
                return json.loads(f.read_text())
            except Exception:
                return None
    return None


def _get_frame(phases: dict, key: str) -> int | None:
    section = phases.get(key, {})
    if isinstance(section, dict):
        # contact stores resolved_original_frame; others store .frame
        for field in ("resolved_original_frame", "frame"):
            v = section.get(field)
            if v is not None:
                return int(v)
    return None


def _get_score(data: dict) -> float | None:
    v = data.get("battingiq_score")
    if v is not None:
        return float(v)
    return None


def _get_hp_confidence(data: dict) -> str:
    phases = data.get("phases", {})
    hp = phases.get("hands_peak", {})
    if isinstance(hp, dict):
        return hp.get("confidence", "unknown")
    return "unknown"


# ── load anchor truth ─────────────────────────────────────────────────────────

truth_df = pd.read_csv(ANCHOR_TRUTH)
truth_map: dict[str, dict] = {}
for _, row in truth_df.iterrows():
    fname = str(row["filename"])
    stem = pathlib.Path(fname).stem
    truth_map[stem] = {
        "hands_peak": int(row["hands_peak_frame"]) if pd.notna(row["hands_peak_frame"]) else None,
        "contact":    int(row["contact_frame"])    if pd.notna(row["contact_frame"])    else None,
        "front_foot_down": int(row["front_foot_down_frame"]) if pd.notna(row["front_foot_down_frame"]) else None,
    }

# ── load ground truth scores + tier labels ────────────────────────────────────

gt_df = pd.read_csv(GT_SCORES)
# Normalise filename to stem
tier_map: dict[str, str] = {}
manual_score_map: dict[str, float] = {}
for _, row in gt_df.iterrows():
    fname = str(row["filename"])
    stem = pathlib.Path(fname).stem
    tier_map[stem] = str(row["tier"])
    score_col = "overall_score" if "overall_score" in gt_df.columns else ("overall" if "overall" in gt_df.columns else gt_df.columns[2])
    manual_score_map[stem] = float(row[score_col]) if pd.notna(row[score_col]) else None

# ── collect per-video results ─────────────────────────────────────────────────

TIERS = ["Beginner", "Average", "Good Club", "Elite"]

rows = []  # list of dicts

all_stems = sorted(set(truth_map.keys()))

for stem in all_stems:
    truth = truth_map.get(stem, {})
    tier = tier_map.get(stem, "Unknown")

    v2_data = _load_json(V2_AUTO_DIR, stem)
    v1_data = _load_json(V1_AUTO_DIR, stem)
    val_data = _load_json(VAL_DIR, stem)

    # Anchor frames
    def _anchors(data):
        if data is None:
            return {}
        phases = data.get("phases", {})
        return {
            "hp": _get_frame(phases, "hands_peak"),
            "contact": _get_frame(phases, "contact"),
            "ffd": _get_frame(phases, "front_foot_down"),
        }

    v2 = _anchors(v2_data)
    v1 = _anchors(v1_data)
    val_anchors = _anchors(val_data)

    def _abs_err(detected, truth_val):
        if detected is None or truth_val is None:
            return None
        return abs(detected - truth_val)

    row = {
        "stem": stem,
        "tier": tier,
        # hands_peak
        "true_hp": truth.get("hands_peak"),
        "v1_hp": v1.get("hp"),
        "v2_hp": v2.get("hp"),
        "v1_hp_err": _abs_err(v1.get("hp"), truth.get("hands_peak")),
        "v2_hp_err": _abs_err(v2.get("hp"), truth.get("hands_peak")),
        # contact
        "true_contact": truth.get("contact"),
        "v1_contact": v1.get("contact"),
        "v2_contact": v2.get("contact"),
        "v1_contact_err": _abs_err(v1.get("contact"), truth.get("contact")),
        "v2_contact_err": _abs_err(v2.get("contact"), truth.get("contact")),
        # front_foot_down
        "true_ffd": truth.get("front_foot_down"),
        "v1_ffd": v1.get("ffd"),
        "v2_ffd": v2.get("ffd"),
        "v1_ffd_err": _abs_err(v1.get("ffd"), truth.get("front_foot_down")),
        "v2_ffd_err": _abs_err(v2.get("ffd"), truth.get("front_foot_down")),
        # scores
        "manual_score": manual_score_map.get(stem),
        "v2_auto_score": _get_score(v2_data) if v2_data else None,
        "v1_auto_score": _get_score(v1_data) if v1_data else None,
        "val_score": _get_score(val_data) if val_data else None,
        # confidence
        "v2_hp_conf": _get_hp_confidence(v2_data) if v2_data else "unknown",
    }
    rows.append(row)

# ── aggregate ─────────────────────────────────────────────────────────────────

def _mean_errs(rows, key, tier=None):
    vals = [r[key] for r in rows if (tier is None or r["tier"] == tier) and r[key] is not None]
    return round(mean(vals), 1) if vals else None

def _within_n(rows, v1_key, v2_key, n):
    v1_w = sum(1 for r in rows if r[v1_key] is not None and r[v1_key] <= n)
    v2_w = sum(1 for r in rows if r[v2_key] is not None and r[v2_key] <= n)
    total = len([r for r in rows if r[v1_key] is not None])
    return v1_w, v2_w, total

def _score_delta_mae(rows, auto_key, val_key, tier=None):
    vals = []
    for r in rows:
        if tier is not None and r["tier"] != tier:
            continue
        a = r.get(auto_key)
        v = r.get(val_key)
        if a is not None and v is not None:
            vals.append(abs(a - v))
    return round(mean(vals), 1) if vals else None

# ── build report ───────────────────────────────────────────────────────────────

lines = []

lines += [
    "=" * 70,
    "F4 HANDS PEAK DETECTOR v1 vs v2 COMPARISON (re-run, post R3/R4/F3c)",
    "  v1: velocity-reversal primary, fixed window (backlift + 45% remaining)",
    "  v2: position-minimum primary, adaptive window (setup+5% → setup+45%)",
    f"  Videos evaluated: {len(rows)}",
    "=" * 70,
    "",
]

# — Hands peak MAE —
lines.append("── MEAN ABSOLUTE FRAME ERROR (hands_peak) ──────────────────────────")
ov1 = _mean_errs(rows, "v1_hp_err")
ov2 = _mean_errs(rows, "v2_hp_err")
delta = round(ov2 - ov1, 1) if (ov1 is not None and ov2 is not None) else None
sign = "+" if (delta is not None and delta >= 0) else ""
lines.append(f"  Overall        v1={ov1}  v2={ov2}  delta={sign}{delta}")
for tier in TIERS:
    n = sum(1 for r in rows if r["tier"] == tier)
    tv1 = _mean_errs(rows, "v1_hp_err", tier)
    tv2 = _mean_errs(rows, "v2_hp_err", tier)
    td = round(tv2 - tv1, 1) if (tv1 is not None and tv2 is not None) else None
    sign = "+" if (td is not None and td >= 0) else ""
    lines.append(f"  {tier:<12}   v1={tv1}  v2={tv2}  delta={sign}{td}  (n={n})")
lines.append("")

# — % within N frames —
lines.append("── % WITHIN N FRAMES OF TRUTH (hands_peak) ─────────────────────────")
for n in (2, 5):
    v1w, v2w, total = _within_n(rows, "v1_hp_err", "v2_hp_err", n)
    lines.append(
        f"  ±{n} frames  v1={v1w}/{total} ({round(v1w/total*100) if total else 0}%)"
        f"  v2={v2w}/{total} ({round(v2w/total*100) if total else 0}%)"
    )
lines.append("")

# — Contact cascade —
lines.append("── CASCADE: MEAN ABSOLUTE FRAME ERROR (contact) ────────────────────")
cv1 = _mean_errs(rows, "v1_contact_err")
cv2 = _mean_errs(rows, "v2_contact_err")
cd = round(cv2 - cv1, 1) if (cv1 is not None and cv2 is not None) else None
sign = "+" if (cd is not None and cd >= 0) else ""
lines.append(f"  v1={cv1}  v2={cv2}  delta={sign}{cd}")
for tier in TIERS:
    n = sum(1 for r in rows if r["tier"] == tier)
    tv1 = _mean_errs(rows, "v1_contact_err", tier)
    tv2 = _mean_errs(rows, "v2_contact_err", tier)
    td = round(tv2 - tv1, 1) if (tv1 is not None and tv2 is not None) else None
    sign = "+" if (td is not None and td >= 0) else ""
    lines.append(f"  {tier:<12}   v1={tv1}  v2={tv2}  delta={sign}{td}")
lines.append("")

# — FFD cascade —
lines.append("── CASCADE: MEAN ABSOLUTE FRAME ERROR (front_foot_down) ────────────")
fv1 = _mean_errs(rows, "v1_ffd_err")
fv2 = _mean_errs(rows, "v2_ffd_err")
fd = round(fv2 - fv1, 1) if (fv1 is not None and fv2 is not None) else None
sign = "+" if (fd is not None and fd >= 0) else ""
lines.append(f"  v1={fv1}  v2={fv2}  delta={sign}{fd}")
for tier in TIERS:
    n = sum(1 for r in rows if r["tier"] == tier)
    tv1 = _mean_errs(rows, "v1_ffd_err", tier)
    tv2 = _mean_errs(rows, "v2_ffd_err", tier)
    td = round(tv2 - tv1, 1) if (tv1 is not None and tv2 is not None) else None
    sign = "+" if (td is not None and td >= 0) else ""
    lines.append(f"  {tier:<12}   v1={tv1}  v2={tv2}  delta={sign}{td}")
lines.append("")

# — Score delta (auto vs validated) —
lines.append("── MEAN ABS OVERALL SCORE DELTA (auto vs validated) ────────────────")
v1_sd = _score_delta_mae(rows, "v1_auto_score", "val_score")
v2_sd = _score_delta_mae(rows, "v2_auto_score", "val_score")
lines.append(f"  v1 (f4_v1_baseline)  mean_abs_overall_delta={v1_sd}  (n={len(rows)})")
for tier in TIERS:
    t_sd = _score_delta_mae(rows, "v1_auto_score", "val_score", tier)
    n = sum(1 for r in rows if r["tier"] == tier)
    lines.append(f"    {tier:<14}{t_sd}  (n={n})")
lines.append(f"  v2 (f4_hands_peak_v2)  mean_abs_overall_delta={v2_sd}  (n={len(rows)})")
for tier in TIERS:
    t_sd = _score_delta_mae(rows, "v2_auto_score", "val_score", tier)
    n = sum(1 for r in rows if r["tier"] == tier)
    lines.append(f"    {tier:<14}{t_sd}  (n={n})")
lines.append("")

# — Per-video table —
lines.append("── PER-VIDEO HANDS PEAK: v1 vs v2 ─────────────────────────────────")
header = f"  {'Filename':<42} {'Tier':<12} {'TrueHP':>6} {'v1':>5} {'err':>4} {'v2':>5} {'err':>4} {'Δ':>5} {'conf'}"
lines.append(header)
for r in sorted(rows, key=lambda x: x["tier"]):
    stem = r["stem"]
    tier = r["tier"]
    true_hp = r["true_hp"] if r["true_hp"] is not None else "?"
    v1_hp = r["v1_hp"] if r["v1_hp"] is not None else "?"
    v2_hp = r["v2_hp"] if r["v2_hp"] is not None else "?"
    v1e = r["v1_hp_err"] if r["v1_hp_err"] is not None else "?"
    v2e = r["v2_hp_err"] if r["v2_hp_err"] is not None else "?"
    if isinstance(v2e, int) and isinstance(v1e, int):
        delta_str = f"{v2e - v1e:+d}"
    else:
        delta_str = "?"
    conf = r["v2_hp_conf"]
    lines.append(
        f"  {stem:<42} {tier:<12} {str(true_hp):>6} {str(v1_hp):>5} {str(v1e):>4}"
        f" {str(v2_hp):>5} {str(v2e):>4}  {delta_str:>4} {conf}"
    )
lines.append("")

# — Regressions —
regressions = [
    r for r in rows
    if r["v1_hp_err"] is not None and r["v2_hp_err"] is not None
    and r["v2_hp_err"] > r["v1_hp_err"]
]
regressions.sort(key=lambda r: r["v2_hp_err"] - r["v1_hp_err"], reverse=True)
lines.append("── REGRESSION LIST (v2 worse than v1 on hands_peak) ────────────────")
if regressions:
    for r in regressions:
        worsened = r["v2_hp_err"] - r["v1_hp_err"]
        lines.append(
            f"  {r['stem']:<44} [{r['tier']}]  v1_err={r['v1_hp_err']}  v2_err={r['v2_hp_err']}  +{worsened} worse"
        )
else:
    lines.append("  None — v2 did not regress on any video.")
lines.append("")

# — Confidence distribution —
lines.append("── hands_peak_confidence DISTRIBUTION ──────────────────────────────")
from collections import defaultdict
conf_buckets = defaultdict(list)
for r in rows:
    conf_buckets[r["v2_hp_conf"]].append(r["stem"])

for label in ["high", "low", "unknown"]:
    names = conf_buckets[label]
    lines.append(f"  {label:<7}  {len(names)}/{len(rows)}  — {names}")

low_names = conf_buckets["low"]
if low_names:
    lines.append("")
    lines.append("  Low-confidence detail (wrist did not rise ≥0.03 units above setup):")
    for stem in low_names:
        r = next(x for x in rows if x["stem"] == stem)
        lines.append(f"    {stem}  [{r['tier']}]  true_hp={r['true_hp']}")
lines.append("")

# — Verdict —
improved = sum(1 for r in rows if r["v1_hp_err"] is not None and r["v2_hp_err"] is not None and r["v2_hp_err"] < r["v1_hp_err"])
same = sum(1 for r in rows if r["v1_hp_err"] is not None and r["v2_hp_err"] is not None and r["v2_hp_err"] == r["v1_hp_err"])
regressed_n = sum(1 for r in rows if r["v1_hp_err"] is not None and r["v2_hp_err"] is not None and r["v2_hp_err"] > r["v1_hp_err"])

c_improved = sum(1 for r in rows if r["v1_contact_err"] is not None and r["v2_contact_err"] is not None and r["v2_contact_err"] < r["v1_contact_err"])
c_same = sum(1 for r in rows if r["v1_contact_err"] is not None and r["v2_contact_err"] is not None and r["v2_contact_err"] == r["v1_contact_err"])
c_regressed = sum(1 for r in rows if r["v1_contact_err"] is not None and r["v2_contact_err"] is not None and r["v2_contact_err"] > r["v1_contact_err"])

f_improved = sum(1 for r in rows if r["v1_ffd_err"] is not None and r["v2_ffd_err"] is not None and r["v2_ffd_err"] < r["v1_ffd_err"])
f_same = sum(1 for r in rows if r["v1_ffd_err"] is not None and r["v2_ffd_err"] is not None and r["v2_ffd_err"] == r["v1_ffd_err"])
f_regressed = sum(1 for r in rows if r["v1_ffd_err"] is not None and r["v2_ffd_err"] is not None and r["v2_ffd_err"] > r["v1_ffd_err"])

lines.append("── VERDICT ──────────────────────────────────────────────────────────")
lines.append(f"  hands_peak: {improved} improved, {same} same, {regressed_n} regressed")
lines.append(f"  contact cascade: {c_improved} improved, {c_same} same, {c_regressed} regressed")
lines.append(f"  front_foot_down cascade: {f_improved} improved, {f_same} same, {f_regressed} regressed")

# overall recommendation
lines.append("")
if ov2 is not None and ov1 is not None:
    if ov2 < ov1:
        lines.append(f"  ADOPT v2: overall hands_peak MAE {ov1} → {ov2} (improvement of {round(ov1-ov2,1)} frames)")
    elif ov2 == ov1:
        lines.append(f"  NEUTRAL: overall hands_peak MAE unchanged at {ov1}")
    else:
        lines.append(f"  REVERT: v2 is {round(ov2-ov1,1)} frames WORSE overall than v1 ({ov1} → {ov2})")

out_path = OUT_DIR / "f4_comparison.txt"
out_path.write_text("\n".join(lines) + "\n")
print(f"Written: {out_path}")
print("\n".join(lines))
