#!/usr/bin/env python3
"""
Decode answer codes from the quick coach check (overall score + main fault per clip) and
compare raters with each other, with Alex's original scores and with the model.

  python phase2_quick.py --codes codes.txt --key BattingIQ_which_drive_key_PRIVATE.csv --out <dir>
      [--model-sheet Model_score_sheet.csv --blind-key blind_key_PRIVATE.csv]

Paste whole messages into codes.txt: lines without "QIQ1." are ignored, duplicates dropped.
The report leads with what matters most for the app's tips: do raters pick the same main fault?
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from pathlib import Path
from statistics import mean
from urllib.parse import unquote

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase2_pairs import LEVELS, ROLES  # noqa: E402

FAULT_TEXT = {
    "F00": "No real fault", "F01": "Head moves early", "F02": "Unbalanced stance", "F03": "Backlift wide/across",
    "F04": "Head/shoulder not leading to line", "F05": "Front foot not to line", "F06": "Front leg stiff, weight back",
    "F07": "Opens up early", "F08": "Bat across the line", "F09": "Pause or hitch", "F10": "Head falls away",
    "F11": "Off balance at finish", "F12": "Checked/wild follow-through",
}


def parse_quick(code: str, ids: list[str], build: str) -> dict:
    m = re.search(r"QIQ1\.[0-9A-F]{6}\.[A-Z][a-z]\.[^.\s]*\.[0-9a-z]*\.[0-9a-z]", code)
    if not m:
        raise ValueError("no quick-check code found")
    body, check = m.group(0).rsplit(".", 1)
    if np.base_repr(sum(map(ord, body)) % 36, 36).lower() != check:
        raise ValueError("checksum does not match (code copied incompletely?)")
    _, b, rl, name, items = body.split(".")
    if b != build:
        raise ValueError(f"code is for build {b}, this key is build {build}")
    order = sorted(ids)
    fault = lambda ch: "" if ch == "z" else f"F{int(ch, 36):02d}"
    answers = {}
    for k in range(0, len(items), 6):
        it = items[k:k + 6]
        clip = order[int(it[0], 36)]
        answers[clip] = {"overall": int(it[1:3], 36), "main_fault": fault(it[3]), "second_fault": fault(it[4]),
                         "confidence": {"l": "low", "m": "medium", "h": "high"}.get(it[5], "")}
    return {"role": ROLES.get(rl[0], rl[0]), "level": LEVELS.get(rl[1], rl[1]), "name": unquote(name), "answers": answers}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--key", required=True, help="the offline files' *_key_PRIVATE.csv (D-codes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ground-truth", default=str(SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"))
    ap.add_argument("--model-sheet", default=None, help="Model_score_sheet.csv (BQ codes)")
    ap.add_argument("--blind-key", default=None, help="blind_key_PRIVATE.csv mapping BQ codes to filenames")
    args = ap.parse_args(argv)

    with open(args.key, newline="", encoding="utf-8-sig") as h:
        key = list(csv.DictReader(h))
    ids, build = [r["clip_id"] for r in key], key[0]["build_id"]
    fname = {r["clip_id"]: r["filename"] for r in key}
    raters, seen = [], set()
    for n, line in enumerate(Path(args.codes).read_text(encoding="utf-8").splitlines(), 1):
        if "QIQ1." not in line:
            continue
        try:
            d = parse_quick(line, ids, build)
        except ValueError as exc:
            print(f"line {n}: skipped ({exc})")
            continue
        sig = tuple(sorted((c, a["overall"], a["main_fault"]) for c, a in d["answers"].items()))
        if sig in seen:
            continue
        seen.add(sig)
        d["rater"] = f"Q{len(raters) + 1:02d}_{d['role']}"
        raters.append(d)

    # per-clip table: one row per rater per clip, keyed by filename
    rows = []
    for r in raters:
        for clip, a in r["answers"].items():
            rows.append({"rater": r["rater"], "role": r["role"], "level": r["level"], "filename": fname[clip], **a})
    # Alex's original scores and the model, as two more raters (overall; model also has a main fault)
    with open(args.ground_truth, newline="", encoding="utf-8-sig") as h:
        gt = {r["filename"]: int(r["overall_score"]) for r in csv.DictReader(h)}
    model = {}
    if args.model_sheet and args.blind_key:
        with open(args.blind_key, newline="", encoding="utf-8-sig") as h:
            bq = {r["clip_id"]: r["filename"] for r in csv.DictReader(h)}
        with open(args.model_sheet, newline="", encoding="utf-8-sig") as h:
            for r in csv.DictReader(h):
                if r["clip_id"] in bq and r.get("overall"):
                    model[bq[r["clip_id"]]] = {"overall": int(float(r["overall"])), "main_fault": r.get("priority_fault", "")}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "quick_answers.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["rater", "role", "level", "filename", "overall", "main_fault", "second_fault", "confidence"])
        w.writeheader()
        w.writerows(rows)

    lines = ["# Quick coach check: results", "", f"Raters: {len(raters)} (" + ", ".join(r["rater"] for r in raters) + ")", ""]
    # 1. main-fault agreement between raters (the measure that matters most for the app's tips)
    by_clip: dict[str, dict[str, str]] = {}
    for r in rows:
        by_clip.setdefault(r["filename"], {})[r["rater"]] = r["main_fault"]
    for f, m in model.items():
        by_clip.setdefault(f, {})["Model"] = m["main_fault"]
    def agree(group: list[str]) -> tuple[float, int]:
        hits = tot = 0
        for picks in by_clip.values():
            for a, b in itertools.combinations([g for g in group if picks.get(g)], 2):
                tot += 1
                hits += picks[a] == picks[b]
        return (hits / tot if tot else float("nan")), tot
    humans = [r["rater"] for r in raters]
    h_rate, h_n = agree(humans)
    lines += ["## Main fault: how often two raters pick the same one", "",
              f"- Between people: {h_rate:.0%} of {h_n} rater pairs." if h_n else "- Between people: needs two or more raters."]
    if model:
        hits = [by_clip[f]["Model"] == by_clip[f][r] for f in by_clip for r in humans if by_clip[f].get(r) and by_clip[f].get("Model")]
        if hits:
            lines.append(f"- Model against people: {mean(hits):.0%} of {len(hits)} comparisons.")
    lines.append("")
    # 2. overall score against Alex's original scores
    lines += ["## Overall score against Alex's original scores", "", "| Rater | Clips | Mean abs gap | Mean bias |", "|---|---|---|---|"]
    for r in raters:
        gaps = [a["overall"] - gt[fname[c]] for c, a in r["answers"].items() if fname[c] in gt]
        if gaps:
            lines.append(f"| {r['rater']} | {len(gaps)} | {mean(map(abs, gaps)):.1f} | {mean(gaps):+.1f} |")
    if model:
        gaps = [m["overall"] - gt[f] for f, m in model.items() if f in gt]
        lines.append(f"| Model | {len(gaps)} | {mean(map(abs, gaps)):.1f} | {mean(gaps):+.1f} |")
    lines += ["", "## Main fault per clip", "", "| Clip | " + " | ".join(humans + (["Model"] if model else [])) + " |",
              "|---|" + "---|" * (len(humans) + (1 if model else 0))]
    for f in sorted(by_clip):
        cells = [by_clip[f].get(r, "") for r in humans + (["Model"] if model else [])]
        lines.append(f"| {f} | " + " | ".join(f"{c} {FAULT_TEXT.get(c, '')}".strip() for c in cells) + " |")
    (out / "quick_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
