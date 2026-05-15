"""
D2 — Concordance report: system scores vs Alex's manual scores.
Reads  : calibration_output/diagnostics_d1_drift/drift_report.csv
         coach_ground_truth_from_screenshot.csv
Writes : concordance_report.csv
         concordance_summary.csv
         concordance_ranking.txt
         concordance_bias.csv
"""

import csv
import os
from collections import defaultdict

BASE = os.path.dirname(__file__)
DRIFT_REPORT = os.path.join(BASE, "calibration_output", "diagnostics_d1_drift", "drift_report.csv")
GROUND_TRUTH = os.path.join(BASE, "coach_ground_truth_from_screenshot.csv")
OUT_REPORT   = os.path.join(BASE, "concordance_report.csv")
OUT_SUMMARY  = os.path.join(BASE, "concordance_summary.csv")
OUT_RANKING  = os.path.join(BASE, "concordance_ranking.txt")
OUT_BIAS     = os.path.join(BASE, "concordance_bias.csv")

PILLARS = ["overall", "access", "tracking", "stability", "flow"]

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def float_or_none(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

# ── Load inputs ─────────────────────────────────────────────────────────────

drift_rows = load_csv(DRIFT_REPORT)
gt_rows    = load_csv(GROUND_TRUTH)

# Index ground truth by filename (case-insensitive for safety)
gt_index = {}
for row in gt_rows:
    gt_index[row["filename"].strip().lower()] = row

# ── Build concordance_report.csv ─────────────────────────────────────────────

report_rows = []
unmatched   = []

for d in drift_rows:
    fname = d["filename"].strip()
    key   = fname.lower()
    if key not in gt_index:
        unmatched.append(fname)
        continue

    gt = gt_index[key]
    tier = d["tier"].strip()

    manual = {
        "overall":   float_or_none(gt["overall_score"]),
        "access":    float_or_none(gt["access_score"]),
        "tracking":  float_or_none(gt["tracking_score"]),
        "stability": float_or_none(gt["stability_score"]),
        "flow":      float_or_none(gt["flow_score"]),
    }

    sys_auto = {
        "overall":   float_or_none(d["auto_overall"]),
        "access":    float_or_none(d["auto_access"]),
        "tracking":  float_or_none(d["auto_tracking"]),
        "stability": float_or_none(d["auto_stability"]),
        "flow":      float_or_none(d["auto_flow"]),
    }

    sys_val = {
        "overall":   float_or_none(d["validated_overall"]),
        "access":    float_or_none(d["validated_access"]),
        "tracking":  float_or_none(d["validated_tracking"]),
        "stability": float_or_none(d["validated_stability"]),
        "flow":      float_or_none(d["validated_flow"]),
    }

    def abs_err(sys_v, man_v):
        if sys_v is None or man_v is None:
            return None
        return abs(sys_v - man_v)

    row = {
        "filename": fname,
        "tier": tier,
        "manual_overall":   manual["overall"],
        "manual_access":    manual["access"],
        "manual_tracking":  manual["tracking"],
        "manual_stability": manual["stability"],
        "manual_flow":      manual["flow"],
    }

    for p in PILLARS:
        row[f"auto_{p}_abs_err"]      = abs_err(sys_auto[p], manual[p])
        row[f"validated_{p}_abs_err"] = abs_err(sys_val[p],  manual[p])

    report_rows.append(row)

# ── Write concordance_report.csv ─────────────────────────────────────────────

report_cols = (
    ["filename", "tier",
     "manual_overall", "manual_access", "manual_tracking", "manual_stability", "manual_flow"]
    + [f"auto_{p}_abs_err"      for p in PILLARS]
    + [f"validated_{p}_abs_err" for p in PILLARS]
)

with open(OUT_REPORT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=report_cols)
    w.writeheader()
    w.writerows(report_rows)

# ── Build concordance_summary.csv (per-tier means) ──────────────────────────

tier_buckets = defaultdict(list)
for r in report_rows:
    tier_buckets[r["tier"]].append(r)

def tier_order(t):
    return {"Beginner": 0, "Average": 1, "Good Club": 2, "Elite": 3}.get(t, 99)

summary_rows = []
for tier in sorted(tier_buckets, key=tier_order):
    bucket = tier_buckets[tier]
    n = len(bucket)
    s_row = {"tier": tier, "n": n}
    for p in PILLARS:
        for mode in ("auto", "validated"):
            col = f"{mode}_{p}_abs_err"
            vals = [r[col] for r in bucket if r[col] is not None]
            s_row[f"mean_{col}"] = round(sum(vals) / len(vals), 2) if vals else None
    summary_rows.append(s_row)

# Add overall-all-tiers row
all_row = {"tier": "ALL", "n": len(report_rows)}
for p in PILLARS:
    for mode in ("auto", "validated"):
        col = f"{mode}_{p}_abs_err"
        vals = [r[col] for r in report_rows if r[col] is not None]
        all_row[f"mean_{col}"] = round(sum(vals) / len(vals), 2) if vals else None
summary_rows.append(all_row)

summary_cols = (
    ["tier", "n"]
    + [f"mean_auto_{p}_abs_err"      for p in PILLARS]
    + [f"mean_validated_{p}_abs_err" for p in PILLARS]
)

with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=summary_cols)
    w.writeheader()
    w.writerows(summary_rows)

# ── Build concordance_bias.csv (signed error: system - manual) ───────────────

def signed_err(d, gt_manual, mode, pillar):
    col_map = {
        ("auto", "overall"):   "auto_overall",
        ("auto", "access"):    "auto_access",
        ("auto", "tracking"):  "auto_tracking",
        ("auto", "stability"): "auto_stability",
        ("auto", "flow"):      "auto_flow",
        ("validated", "overall"):   "validated_overall",
        ("validated", "access"):    "validated_access",
        ("validated", "tracking"):  "validated_tracking",
        ("validated", "stability"): "validated_stability",
        ("validated", "flow"):      "validated_flow",
    }
    sys_v = float_or_none(d.get(col_map[(mode, pillar)]))
    man_v = gt_manual[pillar]
    if sys_v is None or man_v is None:
        return None
    return sys_v - man_v

# Rebuild with drift rows (need raw drift data for signed error)
drift_index = {row["filename"].strip(): row for row in drift_rows}

bias_rows = []
for r in report_rows:
    fname = r["filename"]
    d     = drift_index[fname]
    tier  = r["tier"]
    gt    = gt_index[fname.lower()]

    manual = {
        "overall":   float_or_none(gt["overall_score"]),
        "access":    float_or_none(gt["access_score"]),
        "tracking":  float_or_none(gt["tracking_score"]),
        "stability": float_or_none(gt["stability_score"]),
        "flow":      float_or_none(gt["flow_score"]),
    }

    b_row = {"filename": fname, "tier": tier}
    for p in PILLARS:
        b_row[f"auto_{p}_bias"]      = signed_err(d, manual, "auto", p)
        b_row[f"validated_{p}_bias"] = signed_err(d, manual, "validated", p)
    bias_rows.append(b_row)

bias_cols = (
    ["filename", "tier"]
    + [f"auto_{p}_bias"      for p in PILLARS]
    + [f"validated_{p}_bias" for p in PILLARS]
)

with open(OUT_BIAS, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=bias_cols)
    w.writeheader()
    w.writerows(bias_rows)

# ── Build concordance_ranking.txt ────────────────────────────────────────────

# Worst video×pillar combos (excluding overall from pillar ranking to keep it
# separate; we rank across all 5 including overall per the task spec)
def worst_combos(report_rows, mode, top_n=5):
    combos = []
    for r in report_rows:
        for p in PILLARS:
            err = r[f"{mode}_{p}_abs_err"]
            if err is not None:
                combos.append((err, r["filename"], r["tier"], p))
    combos.sort(key=lambda x: -x[0])
    return combos[:top_n]

auto_worst      = worst_combos(report_rows, "auto")
validated_worst = worst_combos(report_rows, "validated")

def meets_overall_5(r, mode):
    err = r[f"{mode}_overall_abs_err"]
    return err is not None and err <= 5

def meets_all_pillars_5(r, mode):
    for p in ["access", "tracking", "stability", "flow"]:
        err = r[f"{mode}_{p}_abs_err"]
        if err is None or err > 5:
            return False
    return True

n = len(report_rows)
auto_overall_5      = sum(1 for r in report_rows if meets_overall_5(r, "auto"))
validated_overall_5 = sum(1 for r in report_rows if meets_overall_5(r, "validated"))
auto_all_pillars_5      = sum(1 for r in report_rows if meets_all_pillars_5(r, "auto"))
validated_all_pillars_5 = sum(1 for r in report_rows if meets_all_pillars_5(r, "validated"))

# Tier+pillar with largest mean abs error (auto mode)
all_row_for_ranking = next(r for r in summary_rows if r["tier"] == "ALL")
worst_tier_pillar_err = -1
worst_tier_pillar_label = ""
for s_row in summary_rows:
    if s_row["tier"] == "ALL":
        continue
    for p in PILLARS:
        col = f"mean_auto_{p}_abs_err"
        val = s_row.get(col)
        if val is not None and val > worst_tier_pillar_err:
            worst_tier_pillar_err = val
            worst_tier_pillar_label = f"tier={s_row['tier']}, pillar={p}"

lines = []
lines.append("=" * 60)
lines.append("CONCORDANCE RANKING REPORT")
lines.append(f"Videos in report: {n}  (matched from drift_report vs ground truth)")
lines.append("=" * 60)
lines.append("")
lines.append("── TOP 5 WORST VIDEO×PILLAR IN AUTO MODE ──────────────────")
for rank, (err, fname, tier, pillar) in enumerate(auto_worst, 1):
    lines.append(f"  {rank}. |err|={err:.1f}  {fname}  [{tier}]  pillar={pillar}")

lines.append("")
lines.append("── TOP 5 WORST VIDEO×PILLAR IN VALIDATED MODE ─────────────")
for rank, (err, fname, tier, pillar) in enumerate(validated_worst, 1):
    lines.append(f"  {rank}. |err|={err:.1f}  {fname}  [{tier}]  pillar={pillar}")

lines.append("")
lines.append("── THRESHOLD COUNTS (±5) ───────────────────────────────────")
lines.append(f"  Videos within ±5 on OVERALL  — auto mode:      {auto_overall_5}/{n}")
lines.append(f"  Videos within ±5 on OVERALL  — validated mode: {validated_overall_5}/{n}")
lines.append(f"  Videos within ±5 on ALL 4 pillars — auto mode:      {auto_all_pillars_5}/{n}")
lines.append(f"  Videos within ±5 on ALL 4 pillars — validated mode: {validated_all_pillars_5}/{n}")

lines.append("")
lines.append("── TIER×PILLAR WITH LARGEST MEAN ABS ERROR (AUTO MODE) ────")
lines.append(f"  {worst_tier_pillar_label}  mean_abs_err={worst_tier_pillar_err:.2f}")

lines.append("")
lines.append("── PER-TIER MEAN ABS ERROR TABLE (AUTO / VALIDATED) ────────")
header = f"  {'Tier':<12} {'n':>3}  " + "  ".join(
    f"{'auto_'+p[:3]:>10}" for p in PILLARS
) + "  " + "  ".join(
    f"{'val_'+p[:3]:>10}" for p in PILLARS
)
lines.append(header)
for s_row in summary_rows:
    auto_vals = "  ".join(
        f"{s_row.get(f'mean_auto_{p}_abs_err', '—'):>10}" for p in PILLARS
    )
    val_vals  = "  ".join(
        f"{s_row.get(f'mean_validated_{p}_abs_err', '—'):>10}" for p in PILLARS
    )
    lines.append(f"  {s_row['tier']:<12} {s_row['n']:>3}  {auto_vals}  {val_vals}")

lines.append("")
if unmatched:
    lines.append(f"NOTE: {len(unmatched)} drift row(s) had no ground-truth match: {unmatched}")

with open(OUT_RANKING, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

# ── Console summary ──────────────────────────────────────────────────────────
print(f"Wrote {OUT_REPORT}")
print(f"Wrote {OUT_SUMMARY}")
print(f"Wrote {OUT_RANKING}")
print(f"Wrote {OUT_BIAS}")
print()
print("\n".join(lines))
