#!/usr/bin/env python3
"""
Before/after for ANCHOR_MODE=contact_prior on the 17 clips (automatic mode, full production path).

Compares the contact-prior batch with the Phase 0 automatic batch (current detectors):
key-frame error against the validated frames, overall score error and rank agreement against
Alex's scores, and the four live rules' measurements and pillar statuses.

  python phase5_anchor_prior_compare.py --run-dir <batch output with local_runs/> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase0_baseline_table import ANCHORS, extract, find_report, key_of, load_csv, pillar_rule_map  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default=str(SCRIPT_DIR / "baselines" / "phase0_2026-11-14" / "baseline_table.csv"))
    ap.add_argument("--anchor-truth", default=str(SCRIPT_DIR / "anchor_truth.csv"))
    args = ap.parse_args(argv)

    base = {key_of(r["filename"]): r for r in load_csv(Path(args.baseline)) if r["mode"] == "auto"}
    truth = {key_of(r["filename"]): r for r in load_csv(Path(args.anchor_truth))}
    rules = pillar_rule_map()
    rows = []
    for k, b in sorted(base.items()):
        rep = find_report(Path(args.run_dir), b["filename"])
        if rep is None:
            print("  no report:", b["filename"])
            continue
        new = extract(json.loads(rep.read_text()), rules)
        row = {"filename": b["filename"], "coach_overall": b["coach_overall"],
               "overall_before": b["sys_overall"], "overall_after": new.get("sys_overall")}
        for a in ANCHORS:
            t = truth.get(k, {}).get(a, "")
            row[f"{a}_truth"] = t
            row[f"{a}_before"] = b.get(f"{a}_sys", "")
            row[f"{a}_after"] = new.get(f"{a}_sys", "")
        for p in ("access", "tracking", "stability", "flow"):
            row[f"{p}_before"] = b.get(f"sys_{p}")
            row[f"{p}_after"] = new.get(f"sys_{p}")
            row[f"{p}_status_before"] = b.get(f"{p}_status")
            row[f"{p}_status_after"] = new.get(f"{p}_status")
        rows.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "anchor_prior_compare.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    lines = ["# Contact-anchored key frames: before and after (17 clips, automatic, production path)", "",
             "| Key frame | Before: mean error (within 3) | After: mean error (within 3) |", "|---|---|---|"]
    for a in ANCHORS:
        e = {s: [abs(num(r[f"{a}_{s}"]) - num(r[f"{a}_truth"])) for r in rows] for s in ("before", "after")}
        e = {s: [x for x in v if np.isfinite(x)] for s, v in e.items()}
        lines.append(f"| {a} | {np.mean(e['before']):.1f} ({np.mean([x <= 3 for x in e['before']]):.0%}) | "
                     f"{np.mean(e['after']):.1f} ({np.mean([x <= 3 for x in e['after']]):.0%}) |")
    c = np.array([num(r["coach_overall"]) for r in rows])
    for s in ("before", "after"):
        o = np.array([num(r[f"overall_{s}"]) for r in rows])
        ok = np.isfinite(o) & np.isfinite(c)
        import pandas as pd
        rho = pd.Series(o[ok]).rank().corr(pd.Series(c[ok]).rank())
        lines.append(f"\nOverall score {s}: mean error {np.mean(np.abs(o[ok] - c[ok])):.1f} against Alex, Spearman {rho:.2f}.")
    changed = sum(r["overall_before"] != str(r["overall_after"]) for r in rows)
    lines.append(f"\nClips whose overall score changed: {changed} of {len(rows)}.")
    lines.append("Ship only if key frames improve and the score error does not get worse.")
    (out / "anchor_prior_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
