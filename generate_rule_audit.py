"""
D5/R-prep — Validated rule firing audit.
Reads validated JSON reports + ground truth, emits:
  validated_rule_audit.csv
  validated_rule_audit_summary.txt

Optional args:
  --input-dir PATH    local_runs dir to audit (default: D1 validated)
  --output-csv PATH   output CSV path
  --output-summary PATH output summary txt path
"""

import argparse, csv, json, os
from collections import defaultdict
from pathlib import Path

BASE      = Path(__file__).resolve().parent

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  default=None,
        help="Path to local_runs dir containing *_battingiq.json subdirs")
    p.add_argument("--output-csv", default=None,
        help="Output CSV path (default: <BASE>/validated_rule_audit.csv)")
    p.add_argument("--output-summary", default=None,
        help="Output summary txt path (default: <BASE>/validated_rule_audit_summary.txt)")
    return p.parse_args()

_args   = _parse_args()
D1_VAL  = Path(_args.input_dir) if _args.input_dir else (
              BASE / "calibration_output" / "diagnostics_d1_drift" / "validated" / "local_runs")
GT_PATH = BASE / "coach_ground_truth_from_screenshot.csv"
OUT_CSV  = Path(_args.output_csv)      if _args.output_csv      else BASE / "validated_rule_audit.csv"
OUT_SUMM = Path(_args.output_summary)  if _args.output_summary  else BASE / "validated_rule_audit_summary.txt"

PILLARS = ["access", "tracking", "stability", "flow"]
TIER_ORDER = ["Beginner", "Average", "Good Club", "Elite"]

# Complete rule registry: rule_id → (pillar, short_name, suspended)
RULE_REGISTRY = {
    "A1": ("access",    "Bat path around body",          False),
    "A2": ("access",    "Elbow angle at contact",         True),
    "A3": ("access",    "Compression frames",             False),
    "A4": ("access",    "Torso lean",                     True),
    "A5": ("access",    "Shoulder-hip gap",               False),
    "A6": ("access",    "Early opening fraction",         False),
    "T1": ("tracking",  "Head offset",                    True),
    "T2": ("tracking",  "Early head change",              False),
    "T3": ("tracking",  "Head position variance",         True),
    "T4": ("tracking",  "Eye tilt",                       True),
    "T5": ("tracking",  "Setup head variance",            True),
    "S1": ("stability", "Hip shift",                      False),
    "S2": ("stability", "Post-contact instability",       False),
    "S3": ("stability", "Hip drift frames",               True),
    "S4": ("stability", "Post-contact body rotation",     False),
    "F1": ("flow",      "Sync frames",                    False),
    "F2": ("flow",      "Velocity direction changes",     True),
    "F3": ("flow",      "Timing ratio",                   False),
    "F4": ("flow",      "Pause frames",                   False),
    "F5": ("flow",      "Mid-downswing hitch",            False),
    "F6": ("flow",      "Follow-through frames",          False),
}
ACTIVE_RULES = [rid for rid, (_, _, susp) in RULE_REGISTRY.items() if not susp]


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Load ground truth ────────────────────────────────────────────────────────
gt_rows = load_csv(GT_PATH)
gt_index = {row["filename"].strip().lower(): row for row in gt_rows}

# ── Load validated JSON reports ──────────────────────────────────────────────
video_reports = {}
for video_dir in sorted(D1_VAL.iterdir()):
    if not video_dir.is_dir():
        continue
    for jf in video_dir.glob("*_battingiq.json"):
        report = load_json(jf)
        vname = report["metadata"].get("video_name", video_dir.name)
        video_reports[vname.lower()] = (vname, report)

# ── Build per-video, per-rule rows ───────────────────────────────────────────
audit_rows = []
unmatched  = []

for key, (vname, report) in sorted(video_reports.items()):
    gt = gt_index.get(key)
    if gt is None:
        # try stem match (strip extension)
        stem = key.rsplit(".", 1)[0]
        gt = gt_index.get(stem)
    if gt is None:
        unmatched.append(vname)
        continue

    tier            = gt["tier"].strip()
    manual_overall  = int(gt["overall_score"])
    manual_pillar   = {p: int(gt[f"{p}_score"]) for p in PILLARS}
    system_overall  = int(report["battingiq_score"])
    system_pillar   = {p: int(report["pillars"][p]["score"]) for p in PILLARS}

    # Build fault index: rule_id → deduction
    fault_index: dict[str, int] = {}
    for pillar in PILLARS:
        for fault in report["pillars"][pillar].get("faults", []):
            fault_index[fault["rule_id"]] = int(fault["deduction"])

    for rule_id, (pillar, rule_name, suspended) in RULE_REGISTRY.items():
        fired      = rule_id in fault_index
        deduction  = fault_index.get(rule_id, 0)
        audit_rows.append({
            "filename":          vname,
            "tier":              tier,
            "rule_id":           rule_id,
            "rule_name":         rule_name,
            "pillar":            pillar,
            "suspended":         suspended,
            "fired":             fired,
            "points_deducted":   deduction,
            "system_pillar_score":  system_pillar[pillar],
            "manual_pillar_score":  manual_pillar[pillar],
            "pillar_abs_error":     abs(system_pillar[pillar] - manual_pillar[pillar]),
            "system_overall":       system_overall,
            "manual_overall":       manual_overall,
            "overall_abs_error":    abs(system_overall - manual_overall),
        })

# ── Write CSV ────────────────────────────────────────────────────────────────
csv_cols = [
    "filename", "tier", "rule_id", "rule_name", "pillar", "suspended",
    "fired", "points_deducted",
    "system_pillar_score", "manual_pillar_score", "pillar_abs_error",
    "system_overall", "manual_overall", "overall_abs_error",
]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=csv_cols)
    w.writeheader()
    w.writerows(audit_rows)

# ── Helper aggregates ────────────────────────────────────────────────────────
def rows_for(tier=None, rule=None, pillar=None, active_only=True):
    out = audit_rows
    if tier:    out = [r for r in out if r["tier"] == tier]
    if rule:    out = [r for r in out if r["rule_id"] == rule]
    if pillar:  out = [r for r in out if r["pillar"] == pillar]
    if active_only: out = [r for r in out if not r["suspended"]]
    return out

def mean(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else 0.0

def video_summary(tier=None):
    """One row per video (de-dup from rule rows)."""
    seen = {}
    for r in audit_rows:
        if tier and r["tier"] != tier:
            continue
        k = r["filename"]
        if k not in seen:
            seen[k] = {
                "filename": r["filename"], "tier": r["tier"],
                "system_overall": r["system_overall"],
                "manual_overall": r["manual_overall"],
            }
            for p in PILLARS:
                seen[k][f"sys_{p}"]  = r["system_pillar_score"] if r["pillar"] == p else None
                seen[k][f"man_{p}"]  = r["manual_pillar_score"]  if r["pillar"] == p else None
        else:
            for p in PILLARS:
                if r["pillar"] == p:
                    seen[k][f"sys_{p}"] = r["system_pillar_score"]
                    seen[k][f"man_{p}"] = r["manual_pillar_score"]
    return list(seen.values())

def rule_deductions_by_tier(rule_id):
    out = {}
    for t in TIER_ORDER:
        ded = [r["points_deducted"] for r in rows_for(tier=t, rule=rule_id, active_only=False)]
        out[t] = mean(ded)
    return out

# ── Build summary text ───────────────────────────────────────────────────────
lines = []

lines += [
    "=" * 70,
    "VALIDATED RULE AUDIT SUMMARY",
    f"Source: {D1_VAL} ({len(video_reports)} videos)",
    f"Anchors: anchor_truth.csv (validated frames)",
    "=" * 70,
    "",
]

# ── SECTION A — Beginner tier breakdown ─────────────────────────────────────
lines += ["── SECTION A — BEGINNER TIER BREAKDOWN ─────────────────────────────", ""]

beginner_videos = sorted(
    set(r["filename"] for r in audit_rows if r["tier"] == "Beginner")
)

for vname in beginner_videos:
    v_rows = [r for r in audit_rows if r["filename"] == vname]
    sys_ov  = v_rows[0]["system_overall"]
    man_ov  = v_rows[0]["manual_overall"]
    gap_ov  = sys_ov - man_ov
    lines.append(f"  {vname}")
    lines.append(f"    Overall:  system={sys_ov}  manual={man_ov}  gap={gap_ov:+d}")

    for p in PILLARS:
        p_rows = [r for r in v_rows if r["pillar"] == p]
        if not p_rows:
            continue
        sys_p  = p_rows[0]["system_pillar_score"]
        man_p  = p_rows[0]["manual_pillar_score"]
        gap_p  = sys_p - man_p
        total_ded = sum(r["points_deducted"] for r in p_rows)
        fired_rules = [r["rule_id"] for r in p_rows if r["fired"] and not r["suspended"]]
        quiet_rules = [r["rule_id"] for r in p_rows if not r["fired"] and not r["suspended"]]
        lines.append(
            f"    {p.capitalize():<12} sys={sys_p:>2}  man={man_p:>2}  gap={gap_p:+d}  "
            f"total_deducted={total_ded}  fired={fired_rules}  not_fired={quiet_rules}"
        )

    # Diagnosis
    if gap_ov > 10:
        cause = "OVER-SCORING: system deducts too little vs manual"
    elif gap_ov < -10:
        cause = "UNDER-SCORING: system deducts too much vs manual"
    else:
        cause = "Moderate gap — check per-pillar attribution"
    lines.append(f"    >> {cause}")
    lines.append("")

# ── SECTION B — Inverted rules in validated mode ─────────────────────────────
lines += ["── SECTION B — RULES WITH INVERTED FIRING (validated frames) ────────", ""]
lines.append(
    f"  {'Rule':<5} {'Pillar':<12} {'Beginner':>9} {'Average':>9} {'Good Club':>10} {'Elite':>8}  Verdict"
)

for rule_id in sorted(RULE_REGISTRY.keys()):
    pillar, rule_name, suspended = RULE_REGISTRY[rule_id]
    if suspended:
        continue
    ded = rule_deductions_by_tier(rule_id)
    b, a, gc, e = ded["Beginner"], ded["Average"], ded["Good Club"], ded["Elite"]
    # Inverted: elite or good club mean deduction ≥ beginner
    if e >= b or gc >= b:
        verdict = "INVERTED"
    elif b < 0.5:
        verdict = "no-fire"
    else:
        verdict = "ok"
    marker = " ◄" if verdict == "INVERTED" else ""
    lines.append(
        f"  {rule_id:<5} {pillar:<12} {b:>9.2f} {a:>9.2f} {gc:>10.2f} {e:>8.2f}  {verdict}{marker}"
    )
lines.append("")

# ── SECTION C — Zero-discrimination rules in validated mode ───────────────────
lines += ["── SECTION C — ZERO-DISCRIMINATION RULES (validated frames) ─────────", ""]
lines.append(
    "  Rules where mean deduction varies ≤ 1pt across all four tiers:"
)
lines.append("")

for rule_id in sorted(RULE_REGISTRY.keys()):
    pillar, rule_name, suspended = RULE_REGISTRY[rule_id]
    if suspended:
        continue
    ded = rule_deductions_by_tier(rule_id)
    vals = list(ded.values())
    span = max(vals) - min(vals)
    if span <= 1.0:
        lines.append(
            f"  {rule_id:<5} {pillar:<12} {rule_name:<36} "
            f"span={span:.2f}  values={[round(v,2) for v in vals]}"
        )
lines.append("")

# ── SECTION D — Pillar-level gap table ───────────────────────────────────────
lines += ["── SECTION D — PILLAR-LEVEL GAP TABLE ───────────────────────────────", ""]
header = f"  {'Tier':<12} {'Pillar':<12} {'Mean sys':>9} {'Mean man':>9} {'Mean gap':>9} {'Mean|err|':>10}"
lines.append(header)

for tier in TIER_ORDER:
    tier_vids = video_summary(tier=tier)
    for p in PILLARS:
        sys_scores = [v[f"sys_{p}"] for v in tier_vids if v[f"sys_{p}"] is not None]
        man_scores = [v[f"man_{p}"] for v in tier_vids if v[f"man_{p}"] is not None]
        if not sys_scores:
            continue
        m_sys  = mean(sys_scores)
        m_man  = mean(man_scores)
        m_gap  = m_sys - m_man
        m_err  = mean([abs(s - m) for s, m in zip(sys_scores, man_scores)])
        lines.append(
            f"  {tier:<12} {p:<12} {m_sys:>9.1f} {m_man:>9.1f} {m_gap:>+9.1f} {m_err:>10.1f}"
        )
    lines.append("")

# ── SECTION E — Top 3 rule changes for Beginner improvement ──────────────────
lines += ["── SECTION E — TOP 3 RULE CHANGES TO REDUCE BEGINNER ERROR ─────────", ""]

# Calculate per-rule over/under contribution for Beginner videos
# For each rule: (mean deduction on Beginner) vs (what deduction would close the pillar gap)
beginner_rows = [r for r in audit_rows if r["tier"] == "Beginner" and not r["suspended"]]
beginner_videos_list = list(set(r["filename"] for r in beginner_rows))

rule_analysis = {}
for rule_id in ACTIVE_RULES:
    p = RULE_REGISTRY[rule_id][0]
    b_rows = [r for r in beginner_rows if r["rule_id"] == rule_id]
    fire_count  = sum(1 for r in b_rows if r["fired"])
    mean_ded    = mean([r["points_deducted"] for r in b_rows])
    # Per video: system pillar gap (system_pillar - manual_pillar) for this rule's pillar
    pillar_gaps = [r["system_pillar_score"] - r["manual_pillar_score"] for r in b_rows]
    mean_pg     = mean(pillar_gaps)  # positive = system too high (under-deducting)
    rule_analysis[rule_id] = {
        "pillar": p,
        "fire_count": fire_count,
        "n": len(b_rows),
        "mean_ded": mean_ded,
        "mean_pillar_gap": mean_pg,   # + = system over-scores pillar = needs more deduction
    }

# Score each rule by how much its change could reduce Beginner error
# Rules where system too high on a pillar AND rule not firing → prime candidate for activation
# Rules where system too low on a pillar AND rule firing heavily → candidate for reduction
candidates = []
for rule_id, ra in rule_analysis.items():
    pillar_gap = ra["mean_pillar_gap"]
    if pillar_gap > 2 and ra["fire_count"] < ra["n"]:
        # System over-scores, rule not always firing → increase deduction / lower threshold
        impact = pillar_gap * (ra["n"] - ra["fire_count"]) / max(ra["n"], 1)
        candidates.append((impact, rule_id, "INCREASE deduction / lower threshold",
            f"Mean pillar gap={pillar_gap:+.1f} (system too high), fires on {ra['fire_count']}/{ra['n']} Beginner videos, mean_ded={ra['mean_ded']:.1f}"))
    elif pillar_gap < -2 and ra["mean_ded"] > 2:
        # System under-scores, rule deducting heavily → reduce deduction
        impact = abs(pillar_gap)
        candidates.append((impact, rule_id, "REDUCE deduction",
            f"Mean pillar gap={pillar_gap:+.1f} (system too low), mean_ded={ra['mean_ded']:.1f}"))

candidates.sort(key=lambda x: -x[0])

for rank, (impact, rule_id, action, detail) in enumerate(candidates[:3], 1):
    p, rname, _ = RULE_REGISTRY[rule_id]
    lines.append(f"  {rank}. {rule_id} ({p}) — {rname}")
    lines.append(f"     Action:  {action}")
    lines.append(f"     Evidence: {detail}")
    lines.append("")

if not candidates:
    lines.append("  No clear single-rule changes identified — gaps are distributed.")
    lines.append("")

# ── Unmatched notice ─────────────────────────────────────────────────────────
if unmatched:
    lines += ["", f"NOTE: {len(unmatched)} report(s) had no ground-truth match: {unmatched}"]

with open(OUT_SUMM, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote {OUT_CSV}")
print(f"Wrote {OUT_SUMM}")
print(f"Videos audited: {len(video_reports)}, unmatched: {len(unmatched)}")
if unmatched:
    print(f"  Unmatched: {unmatched}")
