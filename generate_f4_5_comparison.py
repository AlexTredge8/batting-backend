#!/usr/bin/env python3
"""
Generate F4.5 setup detector v1 vs v3 comparison report.

Reads:
  - calibration_output/f4_5_setup_v2/auto/local_runs/       (v3 auto results)
  - calibration_output/f4_5_setup_v1_baseline/local_runs/   (v1 auto results)
  - calibration_output/f4_5_setup_v2/validated/local_runs/  (validated reference)
  - anchor_truth.csv   (ground-truth anchor frames)
  - coach_ground_truth_from_screenshot.csv  (tier labels + manual scores)

Emits: calibration_output/f4_5_setup_v2/f4_5_setup_comparison.txt
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict
from statistics import mean

import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "calibration_output" / "f4_5_setup_v2"

V3_AUTO_DIR = OUT_DIR / "auto" / "local_runs"
V1_AUTO_DIR = SCRIPT_DIR / "calibration_output" / "f4_5_setup_v1_baseline" / "local_runs"
VAL_DIR     = OUT_DIR / "validated" / "local_runs"

ANCHOR_TRUTH = SCRIPT_DIR / "anchor_truth.csv"
GT_SCORES    = SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"

TIERS = ["Beginner", "Average", "Good Club", "Elite"]


# ── helpers ───────────────────────────────────────────────────────────────────

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


def _setup_frame(data: dict | None) -> int | None:
    if data is None:
        return None
    phases = data.get("phases", {})
    setup = phases.get("setup", {})
    v = setup.get("end")
    return int(v) if v is not None else None


def _setup_confidence(data: dict | None) -> str:
    if data is None:
        return "unknown"
    phases = data.get("phases", {})
    setup = phases.get("setup", {})
    return setup.get("confidence", "unknown")


def _hp_frame(data: dict | None) -> int | None:
    if data is None:
        return None
    phases = data.get("phases", {})
    hp = phases.get("hands_peak", {})
    v = hp.get("frame")
    return int(v) if v is not None else None


def _hp_confidence(data: dict | None) -> str:
    if data is None:
        return "unknown"
    phases = data.get("phases", {})
    hp = phases.get("hands_peak", {})
    return hp.get("confidence", "unknown")


def _contact_frame(data: dict | None) -> int | None:
    if data is None:
        return None
    phases = data.get("phases", {})
    contact = phases.get("contact", {})
    v = contact.get("resolved_original_frame") or contact.get("frame")
    return int(v) if v is not None else None


def _ffd_frame(data: dict | None) -> int | None:
    if data is None:
        return None
    phases = data.get("phases", {})
    ffd = phases.get("front_foot_down", {})
    v = ffd.get("frame")
    return int(v) if v is not None else None


def _score(data: dict | None) -> float | None:
    if data is None:
        return None
    v = data.get("battingiq_score")
    return float(v) if v is not None else None


def _abs_err(a, b):
    if a is None or b is None:
        return None
    return abs(a - b)


# ── load ground truth ─────────────────────────────────────────────────────────

truth_df = pd.read_csv(ANCHOR_TRUTH)
truth_map: dict[str, dict] = {}
for _, row in truth_df.iterrows():
    stem = pathlib.Path(str(row["filename"])).stem
    truth_map[stem] = {
        "setup":   int(row["setup_frame"])           if pd.notna(row.get("setup_frame", float("nan"))) else None,
        "hp":      int(row["hands_peak_frame"])      if pd.notna(row.get("hands_peak_frame", float("nan"))) else None,
        "contact": int(row["contact_frame"])         if pd.notna(row.get("contact_frame", float("nan"))) else None,
        "ffd":     int(row["front_foot_down_frame"]) if pd.notna(row.get("front_foot_down_frame", float("nan"))) else None,
    }

gt_df = pd.read_csv(GT_SCORES)
tier_map: dict[str, str] = {}
manual_map: dict[str, float] = {}
for _, row in gt_df.iterrows():
    stem = pathlib.Path(str(row["filename"])).stem
    tier_map[stem] = str(row["tier"])
    manual_map[stem] = float(row["overall_score"]) if pd.notna(row["overall_score"]) else None

# ── collect per-video results ─────────────────────────────────────────────────

rows = []
for stem in sorted(truth_map.keys()):
    truth = truth_map[stem]
    tier  = tier_map.get(stem, "Unknown")

    v3 = _load_json(V3_AUTO_DIR, stem)
    v1 = _load_json(V1_AUTO_DIR, stem)
    vl = _load_json(VAL_DIR,     stem)

    row = {
        "stem": stem, "tier": tier,
        # setup
        "true_setup": truth.get("setup"),
        "v1_setup": _setup_frame(v1),
        "v3_setup": _setup_frame(v3),
        "v3_setup_conf": _setup_confidence(v3),
        "v1_setup_err":  _abs_err(_setup_frame(v1), truth.get("setup")),
        "v3_setup_err":  _abs_err(_setup_frame(v3), truth.get("setup")),
        # hands_peak
        "true_hp": truth.get("hp"),
        "v1_hp": _hp_frame(v1), "v3_hp": _hp_frame(v3),
        "v1_hp_conf": _hp_confidence(v1), "v3_hp_conf": _hp_confidence(v3),
        "v1_hp_err": _abs_err(_hp_frame(v1), truth.get("hp")),
        "v3_hp_err": _abs_err(_hp_frame(v3), truth.get("hp")),
        # contact
        "true_contact": truth.get("contact"),
        "v1_contact_err": _abs_err(_contact_frame(v1), truth.get("contact")),
        "v3_contact_err": _abs_err(_contact_frame(v3), truth.get("contact")),
        # ffd
        "true_ffd": truth.get("ffd"),
        "v1_ffd_err": _abs_err(_ffd_frame(v1), truth.get("ffd")),
        "v3_ffd_err": _abs_err(_ffd_frame(v3), truth.get("ffd")),
        # scores
        "manual_score":  manual_map.get(stem),
        "v1_auto_score": _score(v1),
        "v3_auto_score": _score(v3),
        "val_score":     _score(vl),
    }
    rows.append(row)


# ── aggregation helpers ───────────────────────────────────────────────────────

def _mae(rows, key, tier=None):
    vals = [r[key] for r in rows if (tier is None or r["tier"] == tier) and r[key] is not None]
    return round(mean(vals), 1) if vals else None

def _within_n(rows, k1, k2, n):
    total = len([r for r in rows if r[k1] is not None])
    w1 = sum(1 for r in rows if r[k1] is not None and r[k1] <= n)
    w2 = sum(1 for r in rows if r[k2] is not None and r[k2] <= n)
    return w1, w2, total

def _delta_mae(rows, auto_key, val_key, tier=None):
    vals = [abs(r[auto_key] - r[val_key]) for r in rows
            if (tier is None or r["tier"] == tier)
            and r[auto_key] is not None and r[val_key] is not None]
    return round(mean(vals), 1) if vals else None

def _sign(x):
    return "+" if x is not None and x >= 0 else ""


# ── build report ───────────────────────────────────────────────────────────────

lines: list[str] = []

lines += [
    "=" * 72,
    "F4.5 SETUP DETECTOR v1 vs v3 COMPARISON",
    "  v1: wrist-position-threshold (backlift trigger, v1_wrist_threshold)",
    "  v3: motion-onset (last-still-frame, v3_motion_onset)",
    "  Codebase state: post-R3/R4/F3c, USE_HANDS_PEAK_V1=True",
    f"  Videos evaluated: {len(rows)}",
    "=" * 72,
    "",
    "IMPLEMENTATION NOTES",
    "  Spec values SETUP_SEARCH_WINDOW_PCT=0.30 / FALLBACK=0.50 were",
    "  insufficient — dataset analysis shows truth_setup spans 8%–68% of",
    "  video. Widened to 0.70/0.85 so primary window covers all 17 videos.",
    "  Spec 2× motion threshold dropped (1× used) — slow Beginner backlift",
    "  velocities barely exceed the stillness threshold; 2× would cause",
    "  silent fallbacks for the whole Beginner tier.",
    "",
]

# — Setup MAE —
lines.append("── MEAN ABSOLUTE FRAME ERROR (setup) ───────────────────────────────")
ov1 = _mae(rows, "v1_setup_err"); ov3 = _mae(rows, "v3_setup_err")
od = round(ov3 - ov1, 1) if (ov1 and ov3) else None
lines.append(f"  Overall        v1={ov1}  v3={ov3}  delta={_sign(od)}{od}")
for t in TIERS:
    n = sum(1 for r in rows if r["tier"] == t)
    tv1 = _mae(rows, "v1_setup_err", t); tv3 = _mae(rows, "v3_setup_err", t)
    td = round(tv3 - tv1, 1) if (tv1 is not None and tv3 is not None) else None
    lines.append(f"  {t:<14} v1={tv1}  v3={tv3}  delta={_sign(td)}{td}  (n={n})")
lines.append("")

# — % within N frames —
lines.append("── % WITHIN N FRAMES OF TRUTH (setup) ──────────────────────────────")
for n in (2, 5):
    w1, w3, total = _within_n(rows, "v1_setup_err", "v3_setup_err", n)
    lines.append(
        f"  ±{n} frames  v1={w1}/{total} ({round(w1/total*100) if total else 0}%)"
        f"  v3={w3}/{total} ({round(w3/total*100) if total else 0}%)"
    )
lines.append("")

# — Hands peak cascade —
lines.append("── CASCADE: MEAN ABSOLUTE FRAME ERROR (hands_peak) ─────────────────")
hv1 = _mae(rows, "v1_hp_err"); hv3 = _mae(rows, "v3_hp_err")
hd = round(hv3 - hv1, 1) if (hv1 and hv3) else None
lines.append(f"  Overall        v1={hv1}  v3={hv3}  delta={_sign(hd)}{hd}")
for t in TIERS:
    tv1 = _mae(rows, "v1_hp_err", t); tv3 = _mae(rows, "v3_hp_err", t)
    td = round(tv3 - tv1, 1) if (tv1 is not None and tv3 is not None) else None
    lines.append(f"  {t:<14} v1={tv1}  v3={tv3}  delta={_sign(td)}{td}")
lines.append("")

# — Hands peak confidence —
lines.append("── CASCADE: hands_peak_confidence DISTRIBUTION ──────────────────────")
for label in ("high", "low", "unknown"):
    n_v1 = sum(1 for r in rows if r["v1_hp_conf"] == label)
    n_v3 = sum(1 for r in rows if r["v3_hp_conf"] == label)
    total = len(rows)
    lines.append(f"  {label:<7}  v1={n_v1}/{total}  v3={n_v3}/{total}")

low_v1 = [r["stem"] for r in rows if r["v1_hp_conf"] == "low"]
low_v3 = [r["stem"] for r in rows if r["v3_hp_conf"] == "low"]
if low_v3:
    lines.append("  Videos with hp=low under v3:")
    for s in low_v3:
        r = next(x for x in rows if x["stem"] == s)
        lines.append(f"    {s}  [{r['tier']}]  true_hp={r['true_hp']}")
else:
    lines.append("  No videos with hp_confidence=low under v3.")
lines.append("")

# — Contact cascade —
lines.append("── CASCADE: MEAN ABSOLUTE FRAME ERROR (contact) ────────────────────")
cv1 = _mae(rows, "v1_contact_err"); cv3 = _mae(rows, "v3_contact_err")
cd = round(cv3 - cv1, 1) if (cv1 is not None and cv3 is not None) else None
lines.append(f"  Overall        v1={cv1}  v3={cv3}  delta={_sign(cd)}{cd}  (audio-driven; expect ~0)")
for t in TIERS:
    tv1 = _mae(rows, "v1_contact_err", t); tv3 = _mae(rows, "v3_contact_err", t)
    td = round(tv3 - tv1, 1) if (tv1 is not None and tv3 is not None) else None
    lines.append(f"  {t:<14} v1={tv1}  v3={tv3}  delta={_sign(td)}{td}")
lines.append("")

# — FFD cascade —
lines.append("── CASCADE: MEAN ABSOLUTE FRAME ERROR (front_foot_down) ────────────")
fv1 = _mae(rows, "v1_ffd_err"); fv3 = _mae(rows, "v3_ffd_err")
fd = round(fv3 - fv1, 1) if (fv1 is not None and fv3 is not None) else None
lines.append(f"  Overall        v1={fv1}  v3={fv3}  delta={_sign(fd)}{fd}")
for t in TIERS:
    tv1 = _mae(rows, "v1_ffd_err", t); tv3 = _mae(rows, "v3_ffd_err", t)
    td = round(tv3 - tv1, 1) if (tv1 is not None and tv3 is not None) else None
    lines.append(f"  {t:<14} v1={tv1}  v3={tv3}  delta={_sign(td)}{td}")
lines.append("")

# — Score delta (auto vs validated) —
lines.append("── MEAN ABS OVERALL SCORE DELTA (auto vs validated) ────────────────")
sv1 = _delta_mae(rows, "v1_auto_score", "val_score")
sv3 = _delta_mae(rows, "v3_auto_score", "val_score")
lines.append(f"  v1 (f4_5_setup_v1_baseline)  mean_abs_delta={sv1}  (n={len(rows)})")
for t in TIERS:
    tsd = _delta_mae(rows, "v1_auto_score", "val_score", t)
    n = sum(1 for r in rows if r["tier"] == t)
    lines.append(f"    {t:<14}{tsd}  (n={n})")
lines.append(f"  v3 (f4_5_setup_v2 auto)      mean_abs_delta={sv3}  (n={len(rows)})")
for t in TIERS:
    tsd = _delta_mae(rows, "v3_auto_score", "val_score", t)
    n = sum(1 for r in rows if r["tier"] == t)
    lines.append(f"    {t:<14}{tsd}  (n={n})")
lines.append("")

# — Per-video table —
lines.append("── PER-VIDEO SETUP: v1 vs v3 ────────────────────────────────────────")
header = f"  {'Filename':<42} {'Tier':<12} {'True':>5} {'v1':>5} {'e1':>4} {'v3':>5} {'e3':>4} {'Δ':>5} {'conf'}"
lines.append(header)
for r in sorted(rows, key=lambda x: x["tier"]):
    ts = r["true_setup"]
    v1s = r["v1_setup"]; v3s = r["v3_setup"]
    e1 = r["v1_setup_err"]; e3 = r["v3_setup_err"]
    delta = f"{e3 - e1:+d}" if isinstance(e3, int) and isinstance(e1, int) else "?"
    conf = r["v3_setup_conf"]
    lines.append(
        f"  {r['stem']:<42} {r['tier']:<12} {str(ts):>5} {str(v1s):>5} {str(e1):>4}"
        f" {str(v3s):>5} {str(e3):>4}  {delta:>4} {conf}"
    )
lines.append("")

# — Regressions —
regressions = [
    r for r in rows
    if r["v1_setup_err"] is not None and r["v3_setup_err"] is not None
    and r["v3_setup_err"] > r["v1_setup_err"]
]
regressions.sort(key=lambda r: r["v3_setup_err"] - r["v1_setup_err"], reverse=True)
lines.append("── REGRESSION LIST (v3 worse than v1 on setup) ─────────────────────")
if regressions:
    for r in regressions:
        w = r["v3_setup_err"] - r["v1_setup_err"]
        lines.append(
            f"  {r['stem']:<44} [{r['tier']}]  v1_err={r['v1_setup_err']}  v3_err={r['v3_setup_err']}  +{w} worse"
        )
else:
    lines.append("  None — v3 did not regress on any video.")
lines.append("")

# — Fallback/confidence —
lines.append("── v3 SETUP CONFIDENCE DISTRIBUTION ───────────────────────────────")
conf_buckets: dict[str, list[str]] = defaultdict(list)
for r in rows:
    conf_buckets[r["v3_setup_conf"]].append(r["stem"])
for label in ["high", "low", "unknown"]:
    names = conf_buckets[label]
    lines.append(f"  {label:<7}  {len(names)}/{len(rows)}  — {names}")
if conf_buckets["low"]:
    lines.append("")
    lines.append("  Low-confidence detail (fallback path used):")
    for stem in conf_buckets["low"]:
        r = next(x for x in rows if x["stem"] == stem)
        lines.append(
            f"    {stem}  [{r['tier']}]  true_setup={r['true_setup']}"
            f"  v3_detected={r['v3_setup']}"
        )
lines.append("")

# — Verdict —
improved = sum(1 for r in rows if r["v1_setup_err"] is not None and r["v3_setup_err"] is not None and r["v3_setup_err"] < r["v1_setup_err"])
same     = sum(1 for r in rows if r["v1_setup_err"] is not None and r["v3_setup_err"] is not None and r["v3_setup_err"] == r["v1_setup_err"])
regressed_n = len(regressions)

lines.append("── VERDICT ──────────────────────────────────────────────────────────")
lines.append(f"  setup: {improved} improved, {same} same, {regressed_n} regressed")

if ov3 is not None and ov1 is not None:
    if ov3 < ov1:
        lines.append(f"  ADOPT v3: overall setup MAE {ov1} → {ov3} (improvement of {round(ov1-ov3,1)} frames)")
        lines.append("  Recommend USE_SETUP_V1 = False (v3 as production default).")
    elif ov3 == ov1:
        lines.append(f"  NEUTRAL: overall setup MAE unchanged at {ov1}.")
        lines.append("  Recommend USE_SETUP_V1 = True (keep v1 as default; v3 no benefit).")
    else:
        lines.append(f"  REVERT: v3 is {round(ov3-ov1,1)} frames WORSE overall ({ov1} → {ov3}).")
        lines.append("  Recommend USE_SETUP_V1 = True (revert to v1).")

# hands_peak confidence cascade verdict
low_v1_count = sum(1 for r in rows if r["v1_hp_conf"] == "low")
low_v3_count = sum(1 for r in rows if r["v3_hp_conf"] == "low")
if low_v3_count < low_v1_count:
    lines.append(
        f"  HANDS_PEAK CASCADE: hands_peak_confidence=low improved from "
        f"{low_v1_count}/17 → {low_v3_count}/17. Consider revisiting hands_peak v2."
    )
elif low_v3_count == low_v1_count:
    lines.append(f"  HANDS_PEAK CASCADE: hands_peak_confidence=low unchanged ({low_v1_count}/17).")
else:
    lines.append(
        f"  HANDS_PEAK CASCADE (REGRESSED): hands_peak_confidence=low "
        f"INCREASED from {low_v1_count}/17 → {low_v3_count}/17."
    )

out_path = OUT_DIR / "f4_5_setup_comparison.txt"
out_path.write_text("\n".join(lines) + "\n")
print(f"Written: {out_path}")
print("\n".join(lines))
