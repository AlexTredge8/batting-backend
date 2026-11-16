#!/usr/bin/env python3
"""
Decode answer codes from the "Check the AI's tips" file (phase2_offline_pairs.py --validate-out)
into a sheet, and print how often the AI's main fault was right.

  python phase6_validation_codes.py --codes codes.txt --key ~/Desktop/BattingIQ_check_the_AI_key_PRIVATE.csv \
      --out baselines/phase6_validation/alex_validation.csv

Code: VIQ1.<build>.<role><level>.<name>.<items>.<check>; each item is 4 characters: clip index (base 36,
into the sorted clip codes) + verdict (y yes / p partly / n no / x cannot judge) + the coach's main fault
(its number in base 36, z when the verdict is y or x) + confidence (l not very / m fairly / h very / x not given).
Paste whole messages into codes.txt: lines without "VIQ1." are ignored, duplicates dropped.

The key says which AI source made each suggestion (phase6_suggestions.py picks it at random per clip);
the page never shows it. Printed per rater:
  - agreement with the AI overall and per AI source (yes; yes or partly);
  - the coach's main fault against Cam's main fault (cam_quick_check.csv, mapped to rubric v0.3): exact agreement;
  - how often the measurement-based (Phase 6) suggestion was right: on the clips where it was shown, and
    against the coach's main fault on every clip it has an answer for (also for the other sources).
The output has filenames and answers only (no clips).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase0_baseline_table import key_of, load_csv  # noqa: E402
from phase2_pairs import LEVELS, ROLES  # noqa: E402
from phase6_suggestions import RUBRIC, SOURCES, to_v03, v03_faults  # noqa: E402

VERDICTS = {"y": "yes", "p": "partly", "n": "no", "x": "cannot_judge"}
CONFIDENCE = {"l": "not_very", "m": "fairly", "h": "very", "x": ""}
CAM = SCRIPT_DIR / "baselines" / "phase2_preliminary_2026-11-14" / "cam_quick_check.csv"
FIELDS = ["rater", "role", "level", "filename", "suggested_fault", "source", "verdict", "alex_fault", "confidence"]
MEASUREMENT_SOURCE = "phase6_measurements"


def parse_validation(code: str, ids: list[str], build: str, suggested: dict[str, str]) -> dict:
    """One VIQ1 code -> {role, level, name, answers: {clip_id: {verdict, alex_fault, confidence}}}.
    suggested maps clip id -> the AI's main fault (from the key): a 'yes' takes it as the coach's main fault."""
    m = re.search(r"VIQ1\.[0-9A-F]{6}\.[A-Z][a-z]\.[^.\s]*\.[0-9a-z]*\.[0-9a-z]", code)
    if not m:
        raise ValueError("no validation code found")
    body, check = m.group(0).rsplit(".", 1)
    if np.base_repr(sum(map(ord, body)) % 36, 36).lower() != check:
        raise ValueError("checksum does not match (code copied incompletely?)")
    _, b, rl, name, items = body.split(".")
    if b != build:
        raise ValueError(f"code is for build {b}, this key is build {build}")
    if len(items) % 4:
        raise ValueError("answers block has the wrong length")
    order = sorted(ids)
    answers = {}
    for k in range(0, len(items), 4):
        idx, v, f, c = items[k:k + 4]
        clip = order[int(idx, 36)]
        if v not in VERDICTS or c not in CONFIDENCE:
            raise ValueError(f"unknown answer {items[k:k + 4]}")
        if v == "y":
            fault = suggested.get(clip, "")
        elif v in "pn":
            if f == "z":
                raise ValueError(f"{items[k:k + 4]}: partly / no without a main fault")
            fault = f"F{int(f, 36):02d}"
        else:
            fault = ""
        answers[clip] = {"verdict": VERDICTS[v], "alex_fault": fault, "confidence": CONFIDENCE[c]}
    return {"role": ROLES.get(rl[0], rl[0]), "level": LEVELS.get(rl[1], rl[1]), "name": unquote(name), "answers": answers}


def _rate(hit: int, n: int) -> str:
    return f"{hit}/{n} ({hit / n:.0%})" if n else "0/0 (n/a)"


def summarise(rows: list[dict[str, str]], key_rows: list[dict[str, str]], cam: dict[str, str]) -> list[str]:
    """Report lines for one rater's decoded rows."""
    lines = []
    judged = [r for r in rows if r["verdict"] != "cannot_judge"]
    yes = sum(r["verdict"] == "yes" for r in judged)
    yp = sum(r["verdict"] in ("yes", "partly") for r in judged)
    lines.append(f"  AI's main fault right: {_rate(yes, len(judged))}; right or partly right: {_rate(yp, len(judged))}; "
                 f"cannot judge: {len(rows) - len(judged)}")
    lines.append("  Per AI source (clips where it was shown): yes / yes or partly")
    for s in SOURCES + sorted({r["source"] for r in rows} - set(SOURCES)):
        sr = [r for r in judged if r["source"] == s]
        if sr:
            lines.append(f"    {s:20s} {_rate(sum(r['verdict'] == 'yes' for r in sr), len(sr))}  "
                         f"{_rate(sum(r['verdict'] in ('yes', 'partly') for r in sr), len(sr))}")
    mine = {key_of(r["filename"]): r["alex_fault"] for r in judged if r["alex_fault"]}
    both = [k for k in mine if k in cam]
    same = sum(mine[k] == cam[k] for k in both)
    lines.append(f"  Main fault, this rater v Cam (exact): {_rate(same, len(both))}")
    by_key = {key_of(k["filename"]): k for k in key_rows}
    m_shown = [r for r in judged if r["source"] == MEASUREMENT_SOURCE]
    lines.append(f"  Measurement-based (Phase 6) suggestion right when shown: {_rate(sum(r['verdict'] == 'yes' for r in m_shown), len(m_shown))}")
    if any(s in key_rows[0] for s in SOURCES):
        lines.append("  Each source's main fault v this rater's main fault, every clip it has an answer for (shown clips can anchor):")
        for s in SOURCES:
            pairs = [(by_key[k].get(s, ""), f) for k, f in mine.items() if k in by_key and by_key[k].get(s, "")]
            if pairs:
                lines.append(f"    {s:20s} {_rate(sum(a == b for a, b in pairs), len(pairs))}")
    return lines


def load_cam(path: Path, rubric: Path = RUBRIC) -> dict[str, str]:
    """clip key -> Cam's main fault in rubric v0.3 IDs (F14 -> F07)."""
    valid, retired = v03_faults(rubric)
    out = {}
    if Path(path).exists():
        for r in load_csv(Path(path)):
            f = to_v03(r.get("main_fault"), valid, retired)
            if f:
                out[key_of(r["filename"])] = f
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--key", required=True, help="the *_key_PRIVATE.csv written next to the validation file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default=str(CAM), help="Cam's quick check (main fault per clip)")
    args = ap.parse_args(argv)

    key_rows = load_csv(Path(args.key))
    if not key_rows or "suggested_fault" not in key_rows[0]:
        raise SystemExit("This key has no suggested_fault column: use the key written with --validate-out.")
    ids, build = [r["clip_id"] for r in key_rows], key_rows[0]["build_id"]
    by_id = {r["clip_id"]: r for r in key_rows}
    suggested = {r["clip_id"]: r["suggested_fault"] for r in key_rows}
    raters, seen = [], set()
    for n, line in enumerate(Path(args.codes).read_text(encoding="utf-8").splitlines(), 1):
        if "VIQ1." not in line:
            continue
        try:
            d = parse_validation(line, ids, build, suggested)
        except ValueError as exc:
            print(f"line {n}: skipped ({exc})")
            continue
        sig = tuple(sorted((c, tuple(a.values())) for c, a in d["answers"].items()))
        if sig in seen:
            continue
        seen.add(sig)
        d["rater"] = f"V{len(raters) + 1:02d}_{d['role']}"
        raters.append(d)

    rows_by_rater = []
    for r in raters:
        rows = [{"rater": r["rater"], "role": r["role"], "level": r["level"], "filename": by_id[c]["filename"],
                 "suggested_fault": by_id[c]["suggested_fault"], "source": by_id[c].get("source", ""), **a}
                for c, a in sorted(r["answers"].items(), key=lambda t: by_id[t[0]]["filename"])]
        rows_by_rater.append(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=FIELDS)
        w.writeheader()
        for rows in rows_by_rater:
            w.writerows(rows)
    print(f"{len(raters)} rater(s), {sum(len(r) for r in rows_by_rater)} clip answers -> {out}")
    cam = load_cam(Path(args.cam))
    for r, rows in zip(raters, rows_by_rater):
        print(f"{r['rater']} ({r['name']}, {len(rows)} clips):")
        print("\n".join(summarise(rows, key_rows, cam)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
