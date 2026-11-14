#!/usr/bin/env python3
"""
Phase 0 determinism check: pose running mode "video" (tracking) against "image" (per frame).

The Phase 0 baseline on Apple silicon gave different scores for 6 of 17 clips across two
processes (up to 25 points). The 2026-05-13 determinism fix (static-image mode) never reached
the mp.tasks path that production uses; POSE_RUNNING_MODE now selects it. This script runs
each clip N times, each in a FRESH process (the failure was across processes), in each mode,
and reports:
  - landmark drift: the largest |difference| in any landmark coordinate against run 1
  - whether the score, pillars, six anchors and four active measurements are identical
  - for "image" against "video": how far the score and anchors move (a mode change is a
    re-calibration, so it must be measured before it is adopted)

  python phase0_determinism.py --video-dir "<clips>" --out baselines/phase0_<date>/determinism_modes
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
UNSTABLE = ["Drive_Average 2.MOV", "Drive_Average.MOV", "Drive_Beginner 2.MOV", "Drive_Beginner 3.MOV",
            "Drive_Beginner 4.MOV", "Drive_beginner.MOV"]
CONTROLS = ["Viv P_Offdrive Elite.mov", "Cam T_LH Offdrive Elite.mov"]
ANCHORS = ("setup_frame", "hands_start_up_frame", "front_foot_down_frame", "hands_peak_frame", "contact_frame",
           "follow_through_frame")
MEAS = {"A5": "A5_shoulder_hip_gap_deg", "T2": "T2_early_head_change", "S4": "S4_post_contact_rotation_deg",
        "F4": "F4_pause_frames"}


def worker(video: str, out_json: str, out_npz: str) -> None:
    """One fresh process: landmarks + full auto analysis."""
    sys.path.insert(0, str(SCRIPT_DIR))
    os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
    import pose_extractor
    from run_analysis import run_full_analysis
    frames, meta = pose_extractor.extract_poses(video, verbose=False)
    arr = np.full((len(frames), 33, 4), np.nan, dtype=np.float64)
    for i, fp in enumerate(frames):
        if fp.landmarks:
            arr[i] = [[lm.x, lm.y, lm.z, lm.visibility] for lm in fp.landmarks]
    np.save(out_npz, arr)
    with tempfile.TemporaryDirectory() as tmp:
        rep = run_full_analysis(video, output_dir=tmp)
    md = rep["metadata"]
    sig = {"overall": rep["battingiq_score"], **{p: rep["pillars"][p]["score"] for p in rep["pillars"]},
           **{a: md["anchor_frames"][a]["original_frame"] for a in ANCHORS},
           **{k: md["rule_measurements"].get(v) for k, v in MEAS.items()},
           "pose_running_mode": meta.get("pose_running_mode"), "detection_rate": meta.get("detection_rate")}
    Path(out_json).write_text(json.dumps(sig))


def main(argv: list[str] | None = None) -> int:
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:5])
        return 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--clips", nargs="*", default=UNSTABLE + CONTROLS)
    parser.add_argument("--modes", nargs="*", default=["video", "image"])
    args = parser.parse_args(argv)

    sys.path.insert(0, str(SCRIPT_DIR))
    from phase0_baseline_table import find_video
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, sigs = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.clips:
            video = find_video(Path(args.video_dir).expanduser(), name)
            if video is None:
                print(f"skip {name}: not found")
                continue
            for mode in args.modes:
                runs = []
                for r in range(args.runs):
                    js, npz = Path(tmp) / f"{video.stem}_{mode}_{r}.json", Path(tmp) / f"{video.stem}_{mode}_{r}.npy"
                    env = dict(os.environ, POSE_RUNNING_MODE=mode)
                    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(video), str(js), str(npz)],
                                          env=env, capture_output=True, text=True)
                    if proc.returncode != 0:
                        print(f"{name} {mode} run {r + 1} failed:\n{proc.stderr[-800:]}")
                        break
                    runs.append((json.loads(js.read_text()), np.load(npz)))
                if not runs:
                    continue
                base_sig, base_arr = runs[0]
                drift = max((float(np.nanmax(np.abs(a[:, :, :3] - base_arr[:, :, :3]))) if a.shape == base_arr.shape else float("inf"))
                            for _, a in runs[1:]) if len(runs) > 1 else 0.0
                varied = sorted({k for s, _ in runs[1:] for k in base_sig if s[k] != base_sig[k]})
                sigs[(name, mode)] = base_sig
                rows.append({"clip": name, "mode": mode, "runs": len(runs), "identical": "Y" if not varied else "N",
                             "max_landmark_drift": f"{drift:.2e}", "fields_varied": ";".join(varied),
                             "overall_per_run": "/".join(str(s["overall"]) for s, _ in runs),
                             **{f"run1_{k}": v for k, v in base_sig.items()}})
                print(f"{name:30} {mode:5}  identical={rows[-1]['identical']}  drift={rows[-1]['max_landmark_drift']}  overall {rows[-1]['overall_per_run']}")

    if not rows:
        print("No clips run.")
        return 1
    with (out / "determinism_modes.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["# Determinism by pose running mode", "",
             f"Each clip run {args.runs} times, each run in a fresh process, automatic anchors.", "",
             "| Clip | Mode | Identical | Max landmark drift | Overall per run |", "|---|---|---|---|---|"]
    lines += [f"| {r['clip']} | {r['mode']} | {r['identical']} | {r['max_landmark_drift']} | {r['overall_per_run']} |" for r in rows]
    for mode in args.modes:
        sub = [r for r in rows if r["mode"] == mode]
        lines.append(f"\n- {mode}: {sum(r['identical'] == 'Y' for r in sub)}/{len(sub)} clips identical across runs.")
    if {"video", "image"} <= set(args.modes):
        lines += ["", "Image mode against video mode (run 1 of each):", "",
                  "| Clip | Overall video → image | Largest anchor move (frames) |", "|---|---|---|"]
        for name in args.clips:
            v, i = sigs.get((name, "video")), sigs.get((name, "image"))
            if v and i:
                move = max(abs((i[a] or 0) - (v[a] or 0)) for a in ANCHORS)
                lines.append(f"| {name} | {v['overall']} → {i['overall']} | {move} |")
    (out / "determinism_modes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None if len(sys.argv) > 1 and sys.argv[1] == "--worker" else sys.argv[1:]))
