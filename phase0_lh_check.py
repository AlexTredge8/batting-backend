#!/usr/bin/env python3
"""
Phase 0 left-hander check.

The batch never passes handedness, so every calibration run scored the
left-handed clip as a right-hander. This re-runs that clip with
handedness="left" (auto and validated anchors) and compares overall, pillars,
anchors and the four active-rule measurements with the right-handed reports
from the batch. Writes handedness_check.csv. No scoring code is changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

from phase0_baseline_table import (  # noqa: E402
    ANCHORS, determinism_signature, extract, find_report, find_video, key_of, load_csv, pillar_rule_map,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--clip", default="Cam T_LH Offdrive Elite.mov")
    parser.add_argument("--anchor-truth", default=str(SCRIPT_DIR / "anchor_truth.csv"))
    args = parser.parse_args(argv)

    from run_analysis import run_full_analysis  # heavy import; only when run

    run_dir, out_dir = Path(args.run_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = find_video(Path(args.video_dir).expanduser(), args.clip)
    if video is None:
        print(f"Clip not found: {args.clip}", file=sys.stderr)
        return 1
    truth_row = {key_of(r["filename"]): r for r in load_csv(Path(args.anchor_truth))}.get(key_of(args.clip), {})
    truth = {a: int(truth_row[a]) for a in ANCHORS if truth_row.get(a, "").strip()}
    rules = pillar_rule_map()

    rows = []
    for mode, anchors in (("auto", None), ("validated", truth or None)):
        rh_path = find_report(run_dir / mode, args.clip)
        if rh_path is None:
            print(f"No right-handed {mode} report for {args.clip} under {run_dir}", file=sys.stderr)
            return 1
        rh = determinism_signature(extract(json.loads(rh_path.read_text()), rules))
        lh_report = run_full_analysis(
            str(video), output_dir=str(run_dir / "lh_check" / mode / video.stem),
            handedness="left", handedness_source="manual", anchor_frames=anchors,
        )
        lh = determinism_signature(extract(lh_report, rules))
        diffs = [k for k in rh if rh[k] != lh[k]]
        row = {"mode": mode, "clip": args.clip, "any_difference": "Y" if diffs else "N",
               "fields_differing": ";".join(f"{k}:{rh[k]}->{lh[k]}" for k in diffs)}
        row.update({f"right_{k}": v for k, v in rh.items()})
        row.update({f"left_{k}": v for k, v in lh.items()})
        rows.append(row)

    with (out_dir / "handedness_check.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['mode']}: difference={r['any_difference']} {r['fields_differing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
