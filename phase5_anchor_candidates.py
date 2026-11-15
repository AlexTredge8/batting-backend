#!/usr/bin/env python3
"""
Contact-anchored key-frame candidates, evaluated against the validated key frames.

Finding (15 Nov 2026, from baselines/phase0_2026-11-14, leave-one-clip-out): automatic contact
plus the typical gap to each key frame is closer to the validated frames than the current
detectors for hands-start-up, front-foot-down, hands-peak and follow-through (mean error in
frames 2.8 / 1.9 / 2.1 / 7.8 against 3.8 / 3.3 / 4.6 / 10.8), and level for set-up (7.4 vs 7.9).

This script tests three detectors per key frame, all starting from AUTOMATIC contact:
  current  the production detector (from the Phase 0 auto run);
  prior    automatic contact + the median gap (ms) on the OTHER clips (leave one clip out);
  refined  the prior, then a local search in the pose around it:
             hands_peak      highest top-hand point within prior ±150 ms
             front_foot_down front ankle's first near-stop (vertical speed under 15% of its
                             peak) within prior ±150 ms
             follow_through  highest top-hand point from 100 ms after contact to prior +400 ms
             hands_start_up  last frame before the hands' rise where top-hand speed is under
                             20% of its backlift peak, within prior ±200 ms
             setup           middle of the stillest 300 ms (nose + hips) within prior ±400 ms
No production code changes. Output: anchor_candidates.csv and anchor_candidates.md.

  python phase5_anchor_candidates.py --cache-dir calibration_output/pose_cache --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase0_baseline_table import key_of, load_csv  # noqa: E402
from phase5_checkpoints import L, NOSE, R, load_clip  # noqa: E402

KEYS = ["setup_frame", "hands_start_up_frame", "front_foot_down_frame", "hands_peak_frame", "follow_through_frame"]


def _smooth(v: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.array([np.nanmedian(v[max(0, i - 1): i + 2]) for i in range(len(v))])


def refine(c, k: int, prior: dict[str, int]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    wr, ank = c.front["wr"], c.front["ank"]
    n = len(c.xy)

    def window(center: int, ms_lo: int, ms_hi: int) -> list[int]:
        return [f for f in range(center - c.ms(ms_lo), center + c.ms(ms_hi) + 1) if 0 <= f < n]

    fr = window(prior["hands_peak_frame"], 150, 150)
    y = c.series(fr, wr)[:, 1]
    out["hands_peak_frame"] = fr[int(np.nanargmin(y))] if fr and np.isfinite(y).any() else None

    fr = window(prior["front_foot_down_frame"], 150, 150)
    ya = _smooth(c.series(fr, ank)[:, 1])
    vy = np.abs(np.diff(ya))
    if np.isfinite(vy).sum() >= 3:
        peak = np.nanmax(vy)
        i_peak = int(np.nanargmax(vy))
        stops = [i for i in range(i_peak, len(vy)) if np.isfinite(vy[i]) and vy[i] < 0.15 * peak]
        out["front_foot_down_frame"] = fr[stops[0] + 1] if stops else None
    else:
        out["front_foot_down_frame"] = None

    lo = k + c.ms(100)
    fr = [f for f in range(lo, max(lo, prior["follow_through_frame"]) + c.ms(400) + 1) if 0 <= f < n]
    y = c.series(fr, wr)[:, 1]
    out["follow_through_frame"] = fr[int(np.nanargmin(y))] if fr and np.isfinite(y).any() else None

    fr = window(prior["hands_start_up_frame"], 200, 200)
    sp = _smooth(np.linalg.norm(np.diff(c.series(fr, wr), axis=0), axis=1))
    if np.isfinite(sp).sum() >= 3:
        peak = np.nanmax(sp)
        i_peak = int(np.nanargmax(sp))
        quiet = [i for i in range(0, i_peak) if np.isfinite(sp[i]) and sp[i] < 0.20 * peak]
        out["hands_start_up_frame"] = fr[quiet[-1]] if quiet else None
    else:
        out["hands_start_up_frame"] = None

    fr = window(prior["setup_frame"], 400, 400)
    pts = (NOSE, (L["hip"], R["hip"]))
    mv = sum(np.linalg.norm(np.diff(c.series(fr, p), axis=0), axis=1) for p in pts)
    half = max(1, c.ms(150))
    best, best_i = np.inf, None
    for i in range(half, len(mv) - half):
        seg = mv[i - half: i + half]
        if np.isfinite(seg).sum() >= half:
            m = float(np.nanmean(seg))
            if m < best:
                best, best_i = m, i
    out["setup_frame"] = fr[best_i] if best_i is not None else None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(SCRIPT_DIR / "calibration_output" / "pose_cache"))
    ap.add_argument("--auto-table", default=str(SCRIPT_DIR / "baselines" / "phase0_2026-11-14" / "baseline_table.csv"))
    ap.add_argument("--clip-sheet", default=str(SCRIPT_DIR / "clip_sheet.csv"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    hand = {key_of(r["filename"]): r.get("handedness") or "right" for r in load_csv(Path(args.clip_sheet))}
    rows = [r for r in load_csv(Path(args.auto_table)) if r["mode"] == "auto" and r.get("contact_frame_truth", "").strip()]
    caches = {key_of(p.stem): p for p in Path(args.cache_dir).glob("*.npz")}
    clips = []
    for r in rows:
        k = key_of(r["filename"])
        if k not in caches:
            continue
        c = load_clip(caches[k], hand.get(k, "right"))
        clips.append((r, c))

    # gaps in ms from validated contact, per clip
    gaps = {a: [1000 * (int(r[f"{a}_truth"]) - int(r["contact_frame_truth"])) / c.fps for r, c in clips] for a in KEYS}
    out_rows = []
    for i, (r, c) in enumerate(clips):
        k = int(r["contact_frame_sys"])
        prior = {a: k + int(round(np.median([g for j, g in enumerate(gaps[a]) if j != i] or gaps[a]) * c.fps / 1000)) for a in KEYS}
        ref = refine(c, k, prior)
        for a in KEYS:
            truth = int(r[f"{a}_truth"])
            row = {"filename": r["filename"], "key_frame": a, "truth": truth,
                   "current": r[f"{a}_sys"], "prior": prior[a], "refined": "" if ref[a] is None else ref[a]}
            for m in ("current", "prior", "refined"):
                row[f"err_{m}"] = "" if row[m] == "" else abs(int(row[m]) - truth)
            out_rows.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "anchor_candidates.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    lines = ["# Contact-anchored key frames against the current detectors", "",
             f"{len(clips)} clips; automatic contact throughout; prior gaps left one clip out. Error = |frame − validated frame|.", "",
             "| Key frame | Current: mean (median) | Contact + gap: mean (median) | Refined: mean (median), found | Within 3 frames: current / prior / refined |",
             "|---|---|---|---|---|"]
    for a in KEYS:
        rr = [x for x in out_rows if x["key_frame"] == a]
        def st(m):
            e = [x[f"err_{m}"] for x in rr if x[f"err_{m}"] != ""]
            return (np.mean(e), np.median(e), len(e), np.mean([v <= 3 for v in e])) if e else (np.nan, np.nan, 0, np.nan)
        cu, pr, rf = st("current"), st("prior"), st("refined")
        lines.append(f"| {a} | {cu[0]:.1f} ({cu[1]:.0f}) | {pr[0]:.1f} ({pr[1]:.0f}) | {rf[0]:.1f} ({rf[1]:.0f}), {rf[2]}/{len(rr)} | "
                     f"{cu[3]:.0%} / {pr[3]:.0%} / {rf[3]:.0%} |")
    lines += ["", "A detector is worth shipping only if it beats the current one on mean AND on the share within 3 frames.",
              "Shipping it changes the live T2 / F4 measurements, so a Phase 0 batch runs before and after."]
    (out / "anchor_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
