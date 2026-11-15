#!/usr/bin/env python3
"""
Phase 5 entry tests for the checkpoint measurements (phase5_checkpoints.py).

A measurement is used only if it passes all of these. Thresholds fixed 14 Nov 2026,
before any clip other than the reference clip was measured. Do not change them after
seeing results; log a failed measurement and drop it instead.

  0. Coverage    the measurement has a value (validated anchors) on >= 80% of clips.
  1. Tracks      Spearman rho between the measurement, turned so that higher = better,
                 and the rating of its checkpoint (2 / 1 / 0; X left out) is >= 0.30,
                 on >= 10 rated clips that use at least two rating levels. With several
                 raters, the clip's rating is the mean. Ratings from the model only are
                 PROVISIONAL: the test is final only on human coach ratings.
                 Added 15 Nov 2026 after the first run: the rater must also use at least two
                 levels on 3+ clips each; otherwise the test is "not decided" (a rater who gives
                 almost every clip the same rating cannot confirm or reject a measurement). This
                 makes the test stricter, not easier; the first run's table is kept as it was.
  2. Worse       (replaces the two-phone test, 15 Nov 2026: no net session.) The same clip
     filming     made worse three ways (low resolution, half frame rate, tilted and cropped;
                 phase5_degrade_test.py): for every variant, median |change| <= 0.25 x the
                 between-clip IQR. --degrade <degrade_test.csv>. The two-phone check still
                 runs if --two-phone pairs are given.
  3. Auto        with automatic anchors in place of validated ones: Spearman across clips
     anchors     >= 0.80 AND median |auto - validated| <= 0.25 x the between-clip IQR.

A pillar is "assessed" in the product only when two or more of its measurements pass
every test (plan, Phase 5 "done when"). Also reported, not a test: Spearman with Alex's
pillar score for the measurement's pillar.

  python phase5_entry_tests.py --measurements <dir>/checkpoint_measurements.csv --out <dir>
      [--ratings baselines/phase2_preliminary_2026-11-14/model_vs_alex.csv:model ...]
      [--two-phone pairs.csv]   (columns: filename_a, filename_b)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from statistics import median

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase0_baseline_table import key_of, load_csv  # noqa: E402
from phase0_wrong_plane import spearman_ci  # noqa: E402

MIN_COVERAGE, MIN_RHO_TRACK, MIN_RATED, MIN_PER_LEVEL = 0.80, 0.30, 10, 3
MIN_RHO_AUTO, MAX_IQR_SHARE = 0.80, 0.25
PILLARS = ["access", "tracking", "stability", "flow"]


def _num(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _iqr(v: list[float]) -> float:
    v = [x for x in v if np.isfinite(x)]
    return float(np.subtract(*np.percentile(v, [75, 25]))) if len(v) >= 4 else np.nan


def load_ratings(specs: list[str]) -> tuple[dict[str, dict[str, float]], list[str]]:
    """{clip_key: {checkpoint: mean rating}}, rater kinds. Spec = path[:label]; wide checkpoint columns
    (C01..C12 for rubric v0; any Cnn column is read, so v0.2's C13..C16 ratings are used when given)."""
    acc: dict[str, dict[str, list[float]]] = {}
    kinds = []
    for spec in specs:
        path, _, label = spec.partition(":")
        kinds.append(label or Path(path).stem)
        for r in load_csv(Path(path)):
            k = key_of(r["filename"])
            for cp in (col for col in r if re.fullmatch(r"C\d{2}", col or "")):
                v = _num(r.get(cp, ""))  # "X" -> NaN, left out
                if np.isfinite(v):
                    acc.setdefault(k, {}).setdefault(cp, []).append(v)
    return {k: {cp: float(np.mean(v)) for cp, v in d.items()} for k, d in acc.items()}, kinds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--ratings", nargs="*", default=[str(SCRIPT_DIR / "baselines" / "phase2_preliminary_2026-11-14" / "model_vs_alex.csv") + ":model"])
    ap.add_argument("--ground-truth", default=str(SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"))
    ap.add_argument("--two-phone", default=None)
    ap.add_argument("--degrade", default=None, help="degrade_test.csv from phase5_degrade_test.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows = load_csv(Path(args.measurements))
    val: dict[str, dict[str, float]] = {}
    auto: dict[str, dict[str, float]] = {}
    meta: dict[str, dict[str, str]] = {}
    for r in rows:
        tgt = val if r["anchor_set"] == "validated" else auto if r["anchor_set"] == "auto" else None
        if tgt is not None:
            tgt.setdefault(r["measurement"], {})[key_of(r["filename"])] = _num(r["value"])
        meta.setdefault(r["measurement"], {k: r[k] for k in ("checkpoint", "fault", "pillar", "unit", "direction")})
    ratings, kinds = load_ratings(args.ratings) if args.ratings else ({}, [])
    human = any(k != "model" for k in kinds)
    gt = {key_of(r["filename"]): r for r in load_csv(Path(args.ground_truth))}
    pairs = [(key_of(r["filename_a"]), key_of(r["filename_b"])) for r in load_csv(Path(args.two_phone))] if args.two_phone else []
    degrade: dict[str, list[str]] = {}
    if args.degrade:
        for r in load_csv(Path(args.degrade)):
            degrade.setdefault(r["measurement"], []).append(f"{r['variant']}:{r['pass']}")

    out_rows = []
    for m in sorted(meta):
        info, v = meta[m], val.get(m, {})
        sign = -1.0 if info["direction"] == "higher_worse" else 1.0
        clips = sorted(v)
        values = [v[k] for k in clips]
        iqr = _iqr(values)
        coverage = float(np.mean([np.isfinite(x) for x in values])) if values else 0.0
        t0 = coverage >= MIN_COVERAGE

        cp = info["checkpoint"]
        xs, ys = [], []
        for k in clips:
            if np.isfinite(v[k]) and cp in ratings.get(k, {}):
                xs.append(sign * v[k])
                ys.append(ratings[k][cp])
        levels = len(set(ys))
        spread = sum(1 for lv in set(ys) if ys.count(lv) >= MIN_PER_LEVEL)
        rho, lo, hi = spearman_ci(xs, ys) if len(xs) >= 4 else (np.nan, np.nan, np.nan)
        if len(xs) < MIN_RATED or levels < 2:
            t1 = "not enough ratings"
        elif spread < 2:
            t1 = "not decided: rater gives nearly every clip the same rating"
        else:
            t1 = ("pass" if rho >= MIN_RHO_TRACK else "fail") + ("" if human else " (provisional: model ratings)")

        if pairs:
            d = [abs(v[a] - v[b]) for a, b in pairs if np.isfinite(v.get(a, np.nan)) and np.isfinite(v.get(b, np.nan))]
            t2 = ("pass" if d and median(d) <= MAX_IQR_SHARE * iqr else "fail") if d else "no usable pairs"
        else:
            res = degrade.get(m)
            if res:
                bad = [x.split(":")[0] for x in res if not x.endswith("pass")]
                t2 = "pass" if not bad else "fail (" + ", ".join(bad) + ")"
            else:
                t2 = "pending (worse-filming test)"

        a = auto.get(m, {})
        both = [k for k in clips if np.isfinite(v[k]) and np.isfinite(a.get(k, np.nan))]
        rho_a = spearman_ci([v[k] for k in both], [a[k] for k in both])[0] if len(both) >= 4 else np.nan
        mad = float(median([abs(a[k] - v[k]) for k in both])) if both else np.nan
        t3 = "pass" if (np.isfinite(rho_a) and rho_a >= MIN_RHO_AUTO and mad <= MAX_IQR_SHARE * iqr) else "fail"
        if len(both) < 4:
            t3 = "not enough clips"

        pil = info["pillar"]
        px, py = [], []
        for k in clips:
            s = _num(gt.get(k, {}).get(f"{pil}_score", ""))
            if np.isfinite(v[k]) and np.isfinite(s):
                px.append(sign * v[k])
                py.append(s)
        rho_p = spearman_ci(px, py)[0] if len(px) >= 4 else np.nan

        passed_known = t0 and t1.startswith("pass") and t3 == "pass"
        t1_final_fail = t1.startswith("fail") and human  # a model-only disagreement is logged, not a drop
        verdict = ("fail" if (not t0 or t1_final_fail or t3 == "fail" or t2.startswith("fail"))
                   else "robust; model disagrees (provisional)" if t1.startswith("fail") and t0 and t3 == "pass"
                   else "pass so far (test 2 pending)" if passed_known and t2.startswith("pending")
                   else "pass" if passed_known and t2 == "pass"
                   else "robust; awaiting coach ratings" if t0 and t3 == "pass" and not t1.startswith("pass")
                   else "undecided")
        out_rows.append({
            "measurement": m, **info, "clips": len(clips), "coverage": round(coverage, 2), "test0_coverage": "pass" if t0 else "fail",
            "rated_clips": len(xs), "rho_checkpoint": round(rho, 2) if np.isfinite(rho) else "",
            "rho_ci": f"{lo:.2f} to {hi:.2f}" if np.isfinite(lo) else "", "test1_tracks": t1, "test2_worse_filming": t2,
            "auto_clips": len(both), "rho_auto": round(rho_a, 2) if np.isfinite(rho_a) else "",
            "median_abs_auto_diff": round(mad, 3) if np.isfinite(mad) else "", "iqr": round(iqr, 3) if np.isfinite(iqr) else "",
            "test3_auto_anchors": t3, "rho_alex_pillar": round(rho_p, 2) if np.isfinite(rho_p) else "", "verdict": verdict,
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "entry_tests.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    raters = ", ".join(kinds) or "none"
    lines = ["# Phase 5 entry tests", "",
             f"Checkpoint ratings from: {raters}." + ("" if human else " Model ratings only, so test 1 is provisional."),
             "Thresholds were fixed on 14 Nov 2026 before the clips were measured (see phase5_entry_tests.py).", "",
             "| Meas. | Checkpoint | Pillar | Coverage | Tracks rating (rho) | Auto anchors (rho, gap/IQR) | Worse filming | Verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for r in out_rows:
        share = (f"{float(r['median_abs_auto_diff']) / float(r['iqr']):.2f}" if r["iqr"] not in ("", 0, 0.0) and r["median_abs_auto_diff"] != "" else "n/a")
        lines.append(f"| {r['measurement']} | {r['checkpoint']} ({r['fault']}) | {r['pillar']} | {r['coverage']:.0%} | "
                     f"{r['rho_checkpoint'] or 'n/a'}: {r['test1_tracks']} | {r['rho_auto'] or 'n/a'}, {share}: {r['test3_auto_anchors']} | "
                     f"{r['test2_worse_filming']} | {r['verdict']} |")
    lines += ["", "## Pillars", ""]
    for p in PILLARS:
        ok = [r["measurement"] for r in out_rows if r["pillar"] == p and r["verdict"].startswith("pass")]
        wait = [r["measurement"] for r in out_rows if r["pillar"] == p and r["verdict"].startswith("robust")]
        if not human:  # model-only "passes" are provisional too
            wait, ok = wait + ok, []
        lines.append(f"- {p}: {len(ok)} passing ({', '.join(ok) or 'none'}); {len(wait)} robust and awaiting coach ratings "
                     f"({', '.join(wait) or 'none'}) -> " + ("can be assessed" if len(ok) >= 2 else
                     "could reach 2 once coaches rate checkpoints" if len(ok) + len(wait) >= 2 else "NOT ASSESSED (needs 2)"))
    (out / "entry_tests.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
