#!/usr/bin/env python3
"""
Phase 2 pairwise comparison: schedule and fit.

A coach picks the better of two drives instead of scoring out of 100. With
18 clips there are 153 pairs; each coach does about 60 (about 25 minutes).

  schedule_pairs()  balanced random schedule: every clip appears about equally
                    often, each pair at most once per coach, and left or right at random.
  fit (CLI)         reads the returned <coach>_pairs.csv sheets and fits a
                    Bradley-Terry model (MM algorithm, Hunter 2004, with a weak
                    prior so that an unbeaten clip stays finite). Writes the strength of
                    each clip with its standard error, and the scale separation
                    reliability (SSR), the usual reliability figure in comparative
                    judgement.

  python phase2_pairs.py fit --pairs-dir <pack>/pairs --key <pack>/blind_key_PRIVATE.csv --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
from pathlib import Path
from typing import Sequence

import numpy as np


def schedule_pairs(ids: Sequence[str], n_pairs: int, rng: random.Random | None = None) -> list[tuple[str, str]]:
    rng = rng or random.Random()
    all_pairs = list(itertools.combinations(ids, 2))
    n_pairs = min(n_pairs, len(all_pairs))
    rng.shuffle(all_pairs)
    count = {i: 0 for i in ids}
    chosen: list[tuple[str, str]] = []
    pool = all_pairs[:]
    while len(chosen) < n_pairs and pool:
        # pick the pair whose clips have been seen least so far (ties broken by the shuffle)
        best = min(pool, key=lambda p: (count[p[0]] + count[p[1]], max(count[p[0]], count[p[1]])))
        pool.remove(best)
        a, b = best if rng.random() < 0.5 else (best[1], best[0])
        chosen.append((a, b))
        count[a] += 1
        count[b] += 1
    rng.shuffle(chosen)
    return chosen


def fit_bradley_terry(ids: Sequence[str], judgements: Sequence[tuple[str, str]], prior: float = 0.5,
                      iters: int = 2000, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, float]:
    """judgements = [(winner, loser), ...]. Returns (theta, se, ssr), theta on the log scale, centred."""
    idx = {c: i for i, c in enumerate(ids)}
    n = len(ids)
    wins = np.zeros((n, n))
    for w, l in judgements:
        wins[idx[w], idx[l]] += 1
    # weak prior: each clip gets `prior` virtual wins and losses against a virtual average clip
    games = wins + wins.T
    p = np.ones(n)
    for _ in range(iters):
        w_i = wins.sum(axis=1) + prior
        denom = (games / (p[:, None] + p[None, :])).sum(axis=1) + 2 * prior / (p + 1.0)
        new = w_i / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(new - p)) < tol:
            p = new
            break
        p = new
    theta = np.log(p)
    theta -= theta.mean()
    prob = p[:, None] / (p[:, None] + p[None, :])
    info = (games * prob * (1 - prob)).sum(axis=1)
    seen = games.sum(axis=1) > 0  # a clip in no pair has no information: no se, not in the SSR
    se = np.full(n, np.nan)
    se[seen] = 1 / np.sqrt(np.maximum(info[seen], 1e-9))
    th = theta[seen]
    var_theta = th.var(ddof=1) if len(th) > 1 else 0.0
    ssr = float((var_theta - np.mean(se[seen] ** 2)) / var_theta) if var_theta > 0 else float("nan")
    return theta, se, ssr


def _read_judgements(pairs_dir: Path) -> tuple[list[tuple[str, str]], dict[str, int]]:
    judgements, per_coach = [], {}
    for sheet in sorted(pairs_dir.glob("*_pairs.csv")):
        coach = sheet.name.replace("_pairs.csv", "")
        with sheet.open(encoding="utf-8-sig", newline="") as h:
            for r in csv.DictReader(h):
                pick = (r.get("better") or "").strip().upper()
                if pick in ("L", "R"):
                    win, lose = (r["left_id"], r["right_id"]) if pick == "L" else (r["right_id"], r["left_id"])
                    judgements.append((win, lose))
                    per_coach[coach] = per_coach.get(coach, 0) + 1
    return judgements, per_coach


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("from-artifact", help="turn the 'Which drive is better?' link's saved answers into <coach>_pairs.csv sheets")
    j.add_argument("--json-dir", required=True, help="folder of answers/*.json documents saved with ArtifactData list out_dir")
    j.add_argument("--out", required=True, help="pairs folder to write <coach>_pairs.csv into")
    f = sub.add_parser("fit")
    f.add_argument("--pairs-dir", required=True)
    f.add_argument("--key", required=True, help="blind_key_PRIVATE.csv")
    f.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "from-artifact":
        import json
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for doc in sorted(Path(args.json_dir).rglob("*.json")):
            d = json.loads(doc.read_text())
            d = d.get("data", d)
            coach = str(d.get("coach") or doc.stem).replace("/", "_")
            picks = d.get("picks") or {}
            with (out / f"{coach}_pairs.csv").open("w", newline="", encoding="utf-8") as h:
                w = csv.writer(h)
                w.writerow(["pair_no", "left_id", "right_id", "better", "confidence", "notes"])
                for p in d.get("pairs") or []:
                    a = picks.get(str(p["pair_no"]), {})
                    w.writerow([p["pair_no"], p["left_id"], p["right_id"], a.get("better", ""), a.get("confidence", ""), ""])
            print(f"{coach}: {sum(1 for a in picks.values() if a.get('better'))} answered -> {out / (coach + '_pairs.csv')}")
        return 0

    with Path(args.key).open(encoding="utf-8-sig", newline="") as h:
        key = {r["clip_id"]: r["filename"] for r in csv.DictReader(h)}
    judgements, per_coach = _read_judgements(Path(args.pairs_dir))
    ids = sorted(key)
    theta, se, ssr = fit_bradley_terry(ids, judgements)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-theta)
    with (out / "pairwise_scale.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["rank", "clip_id", "filename", "theta", "se"])
        for rank, i in enumerate(order, 1):
            w.writerow([rank, ids[i], key[ids[i]], round(float(theta[i]), 3), "" if np.isnan(se[i]) else round(float(se[i]), 3)])
    print(f"Judgements: {len(judgements)} ({', '.join(f'{c} {n}' for c, n in per_coach.items())}); "
          f"scale separation reliability {ssr:.2f}. Wrote {out / 'pairwise_scale.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
