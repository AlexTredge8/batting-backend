#!/usr/bin/env python3
"""
Phase 2 agreement report: the coach-to-coach ceiling.

Reads the returned blind score sheets (<coach>_score_sheet.csv), the private
blind key and Alex's original scores, and writes a one-page report:

  - Alex's repeat error (blind re-score vs his original scores)
  - coach-to-coach error, overall and per pillar: each coach against the mean
    of the other coaches on the same clips (the ceiling in section 4 of the plan)
  - bias by ability band (coach minus the other coaches)
  - priority-fault agreement between pairs of coaches
  - ICC(2,1), two-way random, absolute agreement (Shrout and Fleiss), with a
    bootstrap 95% interval, on clips that every human coach scored
  - pillar correlation between coaches (below 0.5 sends the pillar back to Phase 1)
  - halo check: how far each coach's four pillars move together
  - overall against the pillar sum, checkpoint agreement, minutes per clip
  - the provisional targets, replaced by numbers

A sheet whose coach name starts with "Model" is the vision-language model. It
is measured exactly like a coach but is left out of the human ceiling.

  python phase2_agreement.py --sheets-dir <pack>/sheets --key <pack>/blind_key_PRIVATE.csv --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PILLARS = ("access", "tracking", "stability", "flow")
MEASURES = ("overall", *PILLARS)
BANDS = ((0, 39, "<40"), (40, 59, "40-59"), (60, 79, "60-79"), (80, 100, ">=80"))
CHECKPOINTS = [f"C{i:02d}" for i in range(1, 13)]


def load_sheets(sheets_dir: Path, key: dict[str, str]) -> pd.DataFrame:
    frames = []
    for sheet in sorted(sheets_dir.glob("*_score_sheet.csv")):
        coach = sheet.name.replace("_score_sheet.csv", "")
        df = pd.read_csv(sheet, dtype=str, keep_default_na=False)
        df["coach"] = coach
        frames.append(df)
    if not frames:
        raise SystemExit(f"No *_score_sheet.csv in {sheets_dir}")
    df = pd.concat(frames, ignore_index=True)
    for m in MEASURES:
        df[m] = pd.to_numeric(df.get(m), errors="coerce")
    df = df[df[list(MEASURES)].notna().any(axis=1)].copy()  # rows the coach actually scored
    df["filename"] = df["clip_id"].map(key)
    df["is_model"] = df["coach"].str.lower().str.startswith("model")
    return df


def band(v: float) -> str:
    for lo, hi, label in BANDS:
        if lo <= v <= hi:
            return label
    return ""


def icc_2_1(matrix: np.ndarray) -> float:
    """Shrout and Fleiss ICC(2,1): two-way random, absolute agreement, single rater. matrix = clips x raters."""
    n, k = matrix.shape
    if n < 3 or k < 2:
        return float("nan")
    grand = matrix.mean()
    msr = k * ((matrix.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((matrix.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    sse = ((matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom else float("nan")


def icc_with_ci(matrix: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    point = icc_2_1(matrix)
    if np.isnan(point):
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [icc_2_1(matrix[rng.integers(0, len(matrix), len(matrix))]) for _ in range(reps)]
    boots = [b for b in boots if not np.isnan(b)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def icc_reading(v: float) -> str:
    if np.isnan(v):
        return "n/a"
    return "poor" if v < 0.5 else "moderate" if v < 0.75 else "good" if v < 0.9 else "excellent"


def vs_others(df: pd.DataFrame, rater: str, humans: list[str]) -> pd.DataFrame:
    """Rows of rater's scores with the mean of the other human coaches on the same clip."""
    mine = df[df["coach"] == rater].set_index("clip_id")
    others = df[df["coach"].isin([h for h in humans if h != rater])]
    mean_others = others.groupby("clip_id")[list(MEASURES)].mean()
    n_others = others.groupby("clip_id")["coach"].nunique()
    joined = mine[list(MEASURES)].join(mean_others, rsuffix="_others", how="inner")
    joined["n_others"] = n_others.reindex(joined.index)
    return joined


def f(v: Any, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)


def table(header: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
        ["| " + " | ".join(f(c) for c in r) + " |" for r in rows]


def build(args: argparse.Namespace) -> int:
    with Path(args.key).open(encoding="utf-8-sig", newline="") as h:
        key = {r["clip_id"]: r["filename"] for r in csv.DictReader(h)}
    df = load_sheets(Path(args.sheets_dir), key)
    humans = sorted(df.loc[~df["is_model"], "coach"].unique())
    models = sorted(df.loc[df["is_model"], "coach"].unique())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 2 agreement report", "",
             f"Coaches: {', '.join(humans)}" + (f"; model: {', '.join(models)}" if models else "") +
             f". Clips: {df['clip_id'].nunique()}. Scored rows: {len(df)}.", ""]
    results: dict[str, Any] = {"humans": humans, "models": models}

    # 1. Alex repeat error
    if args.original and Path(args.original).exists() and args.repeat_coach in humans:
        orig = pd.read_csv(args.original, dtype=str, keep_default_na=False)
        orig = orig.rename(columns={"overall_score": "overall", **{f"{p}_score": p for p in PILLARS}})
        for m in MEASURES:
            orig[m] = pd.to_numeric(orig[m], errors="coerce")
        again = df[df["coach"] == args.repeat_coach].merge(orig[["filename", *MEASURES]], on="filename", suffixes=("", "_orig"))
        rows = [[m.title(), len(again), (again[m] - again[f"{m}_orig"]).abs().mean(), (again[m] - again[f"{m}_orig"]).mean(),
                 int(((again[m] - again[f"{m}_orig"]).abs() <= (5 if m == "overall" else 2)).sum())] for m in MEASURES]
        lines += [f"## 1. {args.repeat_coach}'s repeat error (blind re-score vs original)", ""]
        lines += table(["Measure", "n", "MAE", "Bias (new − old)", "Within ±5 overall / ±2 pillar"], rows)
        lines += ["", "If this is above 5 overall, no system can be held to ±5.", ""]
        results["repeat_mae"] = {r[0].lower(): r[2] for r in rows}

    # 2. Coach-to-coach error (the ceiling)
    per_coach_rows, ceiling = [], {}
    pairs_all = []
    for c in humans:
        j = vs_others(df, c, humans)
        pairs_all.append(j.assign(coach=c))
        per_coach_rows.append([c, len(j), *[(j[m] - j[f"{m}_others"]).abs().mean() for m in MEASURES]])
    for i, m in enumerate(MEASURES):
        vals = [r[2 + i] for r in per_coach_rows if not np.isnan(r[2 + i])]
        ceiling[m] = float(np.mean(vals)) if vals else float("nan")
    lines += ["## 2. Coach-to-coach error (each coach against the mean of the other coaches)", ""]
    lines += table(["Coach", "Clips", *[m.title() for m in MEASURES]], per_coach_rows + [["**Ceiling (mean)**", "", *[ceiling[m] for m in MEASURES]]])
    lines.append("")
    results["ceiling_mae"] = ceiling

    # 3. Bias by band
    allp = pd.concat(pairs_all) if pairs_all else pd.DataFrame()
    bias_rows = []
    if not allp.empty:
        allp["band"] = allp["overall_others"].map(band)
        for c in humans:
            sub = allp[allp["coach"] == c]
            for _, _, lab in BANDS:
                s = sub[sub["band"] == lab]
                if len(s):
                    b = (s["overall"] - s["overall_others"]).mean()
                    bias_rows.append([c, lab, len(s), b, "ABOVE 5" if abs(b) > 5 else ""])
    lines += ["## 3. Bias by ability band (coach minus the other coaches, overall)", ""]
    lines += table(["Coach", "Band (others' mean)", "n", "Bias", "Flag"], bias_rows) + [""]

    # 4. Priority-fault agreement
    def fault_rate(raters: list[str]) -> tuple[float, int]:
        hits = total = 0
        for clip, g in df[df["coach"].isin(raters)].groupby("clip_id"):
            picks = {r["coach"]: r.get("priority_fault", "").strip().upper() for _, r in g.iterrows()}
            for a, b in itertools.combinations(sorted(picks), 2):
                if picks[a] and picks[b]:
                    total += 1
                    hits += picks[a] == picks[b]
        return (hits / total if total else float("nan")), total
    human_rate, human_n = fault_rate(humans)
    lines += ["## 4. Priority-fault agreement", "",
              f"- Two human coaches pick the same priority fault: {f(100 * human_rate, 0)}% of {human_n} coach pairs.", ""]
    results["fault_agreement_human"] = human_rate

    # 5. ICC
    icc_rows = []
    complete = df[df["coach"].isin(humans)].pivot_table(index="clip_id", columns="coach", values=list(MEASURES))
    for m in MEASURES:
        mat = complete[m].dropna().to_numpy() if m in complete else np.empty((0, 0))
        point, lo, hi = icc_with_ci(mat) if mat.size else (float("nan"),) * 3
        icc_rows.append([m.title(), len(mat), point, lo, hi, icc_reading(lo)])
    lines += ["## 5. ICC(2,1), absolute agreement, clips scored by every human coach", ""]
    lines += table(["Measure", "Clips", "ICC", "95% low", "95% high", "Reading (on the low end)"], icc_rows)
    lines += ["", "Reading: below 0.5 poor, 0.5–0.75 moderate, 0.75–0.9 good, above 0.9 excellent (Koo and Li 2016), "
              "judged on the interval. Clips that show a coach are left out of this table, because that coach did not score them.", ""]

    # 6. Pillar correlation between coaches
    corr_rows = []
    for p in PILLARS:
        rs = []
        for a, b in itertools.combinations(humans, 2):
            sub = complete[p][[a, b]].dropna() if p in complete and {a, b} <= set(complete[p].columns) else pd.DataFrame()
            if len(sub) >= 4:
                rs.append(sub[a].corr(sub[b]))
        mean_r = float(np.mean(rs)) if rs else float("nan")
        corr_rows.append([p.title(), len(rs), mean_r, "BACK TO PHASE 1" if not np.isnan(mean_r) and mean_r < 0.5 else ""])
    lines += ["## 6. Pillar correlation between coaches (mean Pearson r over coach pairs)", ""]
    lines += table(["Pillar", "Coach pairs", "Mean r", "Flag"], corr_rows) + [""]

    # 7. Halo, sum check, checkpoints, minutes
    halo_rows = []
    for c in [*humans, *models]:
        sub = df[df["coach"] == c][list(PILLARS)].dropna()
        if len(sub) >= 4:
            cm = sub.corr().to_numpy()
            eig = np.linalg.eigvalsh(cm)
            off = cm[np.triu_indices(4, 1)]
            share = float(eig[-1] / eig.sum())
        else:
            off, share = np.array([np.nan]), float("nan")
        s = df[df["coach"] == c]
        sum_gap = (s["overall"] - s[list(PILLARS)].sum(axis=1, min_count=4)).abs().mean()
        minutes = pd.to_numeric(s.get("minutes_taken"), errors="coerce").median()
        x_rate = float((s[[c_ for c_ in CHECKPOINTS if c_ in s]].apply(lambda col: col.str.upper() == "X")).to_numpy().mean()) \
            if any(c_ in s for c_ in CHECKPOINTS) else float("nan")
        halo_rows.append([c, float(np.nanmin(off)), float(np.nanmax(off)), share, sum_gap, minutes, 100 * x_rate])
    lines += ["## 7. Halo, overall against the pillar sum, cannot-judge rate, time", ""]
    lines += table(["Rater", "Min pillar r", "Max pillar r", "One-factor share", "Mean gap, overall vs pillar sum",
                    "Median minutes per clip", "% checkpoints X"], halo_rows)
    lines += ["", "A one-factor share near 0.95 (as in the original scores) means the four pillars are one judgement shown four ways.", ""]

    cp_rows = []
    for cp in CHECKPOINTS:
        if cp not in df:
            continue
        g = df[df["coach"].isin(humans)].pivot_table(index="clip_id", columns="coach", values=cp, aggfunc="first")
        same = total = 0
        for a, b in itertools.combinations([h for h in humans if h in g], 2):
            pair = g[[a, b]].dropna()
            pair = pair[(pair[a].isin(["0", "1", "2"])) & (pair[b].isin(["0", "1", "2"]))]
            total += len(pair)
            same += int((pair[a] == pair[b]).sum())
        cp_rows.append([cp, total, (100 * same / total) if total else float("nan")])
    if cp_rows:
        lines += ["Checkpoint agreement between coach pairs (exact match on 0/1/2):", ""]
        lines += table(["Checkpoint", "Pairs", "% same"], cp_rows) + [""]

    # 8. Model as a fourth coach
    for mname in models:
        j = vs_others(df, mname, humans + [mname])
        rows = [[m.title(), len(j), (j[m] - j[f"{m}_others"]).abs().mean(), ceiling.get(m, float("nan"))] for m in MEASURES]
        mr, mn = fault_rate(humans + [mname])
        lines += [f"## 8. {mname} as a fourth coach (against the mean of the human coaches)", ""]
        lines += table(["Measure", "Clips", f"{mname} MAE", "Human ceiling"], rows)
        lines += ["", f"- Fault agreement over all coach pairs including {mname}: {f(100 * mr, 0)}% ({mn} pairs); humans only {f(100 * human_rate, 0)}%.",
                  "- Decision rule: use it for a checkpoint only where it sits inside the coach-to-coach band.", ""]

    # 9. Pairwise
    if args.pairwise_scale and Path(args.pairwise_scale).exists():
        scale = pd.read_csv(args.pairwise_scale)
        means = df[df["coach"].isin(humans)].groupby("filename")["overall"].mean()
        merged = scale.set_index("filename").join(means, how="inner")
        rho = merged["theta"].rank().corr(merged["overall"].rank())
        lines += ["## 9. Pairwise scale against mean coach scores", "",
                  f"- Spearman rho between the pairwise strength and the mean overall score: {f(rho)} (n = {len(merged)}).", ""]

    # 10. Targets
    target_rows = [[m.title(), ceiling[m], 1.25 * ceiling[m] if not np.isnan(ceiling[m]) else float("nan")] for m in MEASURES]
    lines += ["## 10. Targets, with the provisional figures replaced", ""]
    lines += table(["Measure", "Coach-to-coach MAE", "System target MAE (≤ 1.25×)"], target_rows)
    lines += ["", f"- Priority-fix match: at least {f(100 * human_rate - 10, 0)}% (coach rate {f(100 * human_rate, 0)}% minus 10 points).",
              "- No ability band with a mean bias above 5 points; at least 80% of gate-passing clips scored; same clip, same score.",
              "- With 18 clips this ceiling is rough; firm it up at 40 clips.", ""]
    results["targets_mae"] = {m: r[2] for m, r in zip(MEASURES, target_rows)}

    (out / "agreement_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "agreement_results.json").write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
    df.to_csv(out / "all_scores_long.csv", index=False)
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheets-dir", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--original", default=str(SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"))
    parser.add_argument("--repeat-coach", default="Alex")
    parser.add_argument("--pairwise-scale", default=None)
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
