#!/usr/bin/env python3
"""
Phase 3 gate, version 0 (PROVISIONAL thresholds): scored, part-scored or not scored.

Runs on the facts in clip_sheet.csv (measured by phase0_clip_sheet.py, judgement columns
from Alex or the hub's guesses). Each failed check gives one plain retake instruction;
a clip that is not scored shows only the first one. Thresholds here are first guesses
and will be replaced by thresholds taken from the Phase 3 net-session footage.

It also runs the plan's own test of a gate: clips the gate passes must show lower error
against the coach than clips it rejects (using the Phase 0 baseline table).

  python phase3_gate.py --baseline baselines/phase0_2026-11-14/baseline_table.csv --out baselines/phase3_gate_v0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

SCRIPT_DIR = Path(__file__).resolve().parent

# (id, outcome if failed, retake message, test). Order = priority of the message shown.
CHECKS = [
    ("tracked", "not scored", "Only the batter should be in the frame, start to finish.",
     lambda r: _f(r, "batter_height_px", 1) > 0 and _f(r, "pose_detection_rate", 0) >= 90),
    ("others", "not scored", "Only the batter should be in the frame: film from behind the bowler with the net lane clear.",
     lambda r: (r.get("others_in_frame") or "N").upper() != "Y"),
    ("angle", "not scored", "Stand directly behind the bowler, in line with the stumps.",
     lambda r: not (r.get("camera_position") or "").lower().startswith("angle")),
    ("length", "not scored", "Start filming before the bowler runs in and stop after the follow-through.",
     lambda r: _f(r, "duration_s", 99) >= 2.0),
    ("size", "not scored", "Move closer: the batter should fill at least a fifth of the picture's height.",
     lambda r: _f(r, "batter_height_pct", 100) >= 20),
    ("sound", "part-scored", "Film with the sound on: the sound of bat on ball is how we find contact.",
     lambda r: str(r.get("audio_streams", "1")) != "0"),
    ("timing", "part-scored", "Send the original video from your camera roll, not an edited or forwarded copy.",
     lambda r: not ("frame-rate mismatch" in (r.get("flags_hub") or ""))),
    ("resolution", "part-scored", "Film in 1080p or higher, and send the original file.",
     lambda r: min(_f(r, "width", 9999), _f(r, "height", 9999)) >= 720),
]


def _f(r: dict, k: str, default: float) -> float:
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return default


def gate(row: dict) -> tuple[str, list[str], str]:
    failed = [(cid, outcome, msg) for cid, outcome, msg, test in CHECKS if not test(row)]
    if any(o == "not scored" for _, o, _ in failed):
        outcome = "not scored"
    elif failed:
        outcome = "part-scored"
    else:
        outcome = "scored"
    first = next((m for _, o, m in failed if o == outcome), "")
    return outcome, [c for c, _, _ in failed], first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip-sheet", default=str(SCRIPT_DIR / "clip_sheet.csv"))
    parser.add_argument("--baseline", default=None, help="Phase 0 baseline_table.csv for the pass/fail error test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    with open(args.clip_sheet, newline="", encoding="utf-8-sig") as h:
        sheet = list(csv.DictReader(h))
    err: dict[tuple[str, str], float] = {}
    if args.baseline:
        with open(args.baseline, newline="", encoding="utf-8-sig") as h:
            for r in csv.DictReader(h):
                if r.get("err_overall") not in ("", None):
                    err[(r["filename"], r["mode"])] = abs(float(r["err_overall"]))

    rows = []
    for r in sheet:
        outcome, failed, msg = gate(r)
        rows.append({"filename": r["filename"], "tier": r.get("tier_csv", ""), "outcome": outcome,
                     "failed_checks": ";".join(failed), "retake_message": msg,
                     "auto_abs_error": err.get((r["filename"], "auto"), ""),
                     "judgement_columns": "unconfirmed" if "not confirmed" in (r.get("flags_confirmed") or "") else "confirmed"})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "gate_v0.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["# Gate v0 on the 18 clips (provisional thresholds)", "",
             "| Clip | Tier | Outcome | Failed checks | Retake message | Phase 0 auto error |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['filename']} | {r['tier']} | {r['outcome']} | {r['failed_checks'] or '-'} | {r['retake_message'] or '-'} | {r['auto_abs_error']} |")
    counts = {o: sum(r["outcome"] == o for r in rows) for o in ("scored", "part-scored", "not scored")}
    lines += ["", f"- Outcomes: scored {counts['scored']}, part-scored {counts['part-scored']}, not scored {counts['not scored']}."]
    if err:
        passed = [float(r["auto_abs_error"]) for r in rows if r["outcome"] != "not scored" and r["auto_abs_error"] != ""]
        failed = [float(r["auto_abs_error"]) for r in rows if r["outcome"] == "not scored" and r["auto_abs_error"] != ""]
        if passed and failed:
            lines.append(f"- Plan test: clips the gate lets through have a mean Phase 0 error of {mean(passed):.1f} points (n = {len(passed)}); "
                         f"clips it refuses, {mean(failed):.1f} (n = {len(failed)}). "
                         + ("The gate separates them, as it should." if mean(passed) < mean(failed) else "The gate does not separate them: it is checking the wrong things."))
            lines.append("- Caution: refused clips are mostly lower tiers, where the four-rule scorer is worst anyway, so part of the gap is tier, not capture.")
    if any(r["judgement_columns"] == "unconfirmed" for r in rows):
        lines.append("- The angle and others-in-frame checks rest on the hub's guesses; Alex has not confirmed them.")
    (out / "gate_v0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
