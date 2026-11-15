#!/usr/bin/env python3
"""
Decode answer codes from the checkpoint coach check (each rated checkpoint 2/1/0/X per clip)
into a ratings sheet that phase5_entry_tests.py reads with --ratings.

A code lists ratings by position. The key file says which checkpoint each position is (its
"checkpoints" column, written by phase2_offline_pairs.py from rubric v0.3 on, in chain order,
e.g. "C01 C02 C17 C13 ..."). Keys from older builds have no such column; their positions are
C01..Cn (rubric v0.1: 12, v0.2: 16).

  python phase5_checkpoint_codes.py --codes codes.txt --key <offline files>_key_PRIVATE.csv \
      --out baselines/phase5_checkpoint_ratings/checkpoint_ratings.csv

Paste whole messages into codes.txt: lines without "CIQ1." or "CIQ2." are ignored, duplicates dropped.
The output has filenames and ratings only (no clips). With two or more raters it also prints
how often raters agree on each checkpoint (exact and within one point).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase2_pairs import LEVELS, ROLES  # noqa: E402

CPS = [f"C{i:02d}" for i in range(1, 13)]


def parse_checkpoint(code: str, ids: list[str], build: str, cp_ids: list[str] | None = None) -> dict:
    """CIQ1 (12 checkpoints, rubric v0.1) or CIQ2 (n checkpoints given in the code). cp_ids names the
    checkpoint at each position (from the key); without it, positions are C01..Cn."""
    m = re.search(r"CIQ([12])\.[0-9A-F]{6}\.[A-Z][a-z]\.[^.\s]*(?:\.[0-9a-z]+)?\.[0-9a-z]*\.[0-9a-z]", code)
    if not m:
        raise ValueError("no checkpoint code found")
    body, check = m.group(0).rsplit(".", 1)
    if np.base_repr(sum(map(ord, body)) % 36, 36).lower() != check:
        raise ValueError("checksum does not match (code copied incompletely?)")
    parts = body.split(".")
    if parts[0] == "CIQ1":
        _, b, rl, name, items = parts
        n = 12
    else:
        _, b, rl, name, n36, items = parts
        n = int(n36, 36)
    if b != build:
        raise ValueError(f"code is for build {b}, this key is build {build}")
    if len(items) % (n + 1):
        raise ValueError("ratings block has the wrong length")
    if cp_ids:
        if len(cp_ids) != n:
            raise ValueError(f"code has {n} checkpoints, the key lists {len(cp_ids)}")
        cps = list(cp_ids)
    else:
        cps = [f"C{i:02d}" for i in range(1, n + 1)]
    order = sorted(ids)
    answers = {}
    for k in range(0, len(items), n + 1):
        it = items[k:k + n + 1]
        answers[order[int(it[0], 36)]] = {cp: ("X" if v == "x" else v) for cp, v in zip(cps, it[1:])}
    return {"role": ROLES.get(rl[0], rl[0]), "level": LEVELS.get(rl[1], rl[1]), "name": unquote(name), "answers": answers, "n": n}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    with open(args.key, newline="", encoding="utf-8-sig") as h:
        key = list(csv.DictReader(h))
    ids, build = [r["clip_id"] for r in key], key[0]["build_id"]
    fname = {r["clip_id"]: r["filename"] for r in key}
    cp_ids = (key[0].get("checkpoints") or "").split() or None
    raters, seen = [], set()
    for n, line in enumerate(Path(args.codes).read_text(encoding="utf-8").splitlines(), 1):
        if "CIQ1." not in line and "CIQ2." not in line:
            continue
        try:
            d = parse_checkpoint(line, ids, build, cp_ids)
        except ValueError as exc:
            print(f"line {n}: skipped ({exc})")
            continue
        sig = tuple(sorted((c, tuple(a.values())) for c, a in d["answers"].items()))
        if sig in seen:
            continue
        seen.add(sig)
        d["rater"] = f"K{len(raters) + 1:02d}_{d['role']}"
        raters.append(d)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    global CPS
    CPS = cp_ids or [f"C{i:02d}" for i in range(1, max([r["n"] for r in raters] or [12]) + 1)]
    with out.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["rater", "role", "level", "filename", *CPS], restval="")
        w.writeheader()
        for r in raters:
            for clip, a in sorted(r["answers"].items()):
                w.writerow({"rater": r["rater"], "role": r["role"], "level": r["level"], "filename": fname[clip], **a})
    print(f"{len(raters)} rater(s), {sum(len(r['answers']) for r in raters)} clip ratings -> {out}")

    for r in raters:  # how much each rater spreads ratings: a rater who gives every clip the same rating cannot test a measurement
        spread = []
        for cp in CPS:
            vals = [a.get(cp, "X") for a in r["answers"].values() if a.get(cp, "X") != "X"]
            levels = sum(1 for v in set(vals) if vals.count(v) >= 3)
            spread.append(f"{cp}:{levels}")
        print(f"{r['rater']} levels used on 3+ clips per checkpoint: " + " ".join(spread))
    if len(raters) >= 2:
        print("Agreement between raters (exact / within one point):")
        for cp in CPS:
            ex = near = tot = 0
            for a, b in itertools.combinations(raters, 2):
                for clip in set(a["answers"]) & set(b["answers"]):
                    x, y = a["answers"][clip].get(cp, "X"), b["answers"][clip].get(cp, "X")
                    if "X" in (x, y):
                        continue
                    tot += 1
                    ex += x == y
                    near += abs(int(x) - int(y)) <= 1
            print(f"  {cp}: {ex / tot:.0%} / {near / tot:.0%} of {tot}" if tot else f"  {cp}: no shared ratings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
