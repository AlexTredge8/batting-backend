#!/usr/bin/env python3
"""
Phase 0 wrong-plane test.

Hypothesis: the suspended measurements A2, A3, A4, S1 and S3 fail because
they measure movement toward the camera. A bowler's-end camera sees width
and height, not depth along the pitch. The four active measurements (A5,
T2, S4, F4) are run as comparators.

Indices (pre-registered 2026-11-14; do not change after seeing data):
  1. Foreshortening ratio per segment: 2D length at the measured frames
     divided by that segment's 95th-percentile 2D length from set-up to
     follow-through. Median over measured frames. A measurement takes the
     minimum over its segments (its most foreshortened part).
  2. Depth share of motion over the measured frames for the tracked points:
     range(z) / (range(x) + range(y) + range(z)) of 5-frame moving-median
     positions, in pixels (z scaled by the frame width), summed over points.
     The hip midpoint is skipped: MediaPipe image z is relative to it. Rough:
     MediaPipe image z is an estimate.
  3. Spearman rho of the measurement (validated mode) with the coach score of
     its pillar and with the coach overall, with a 95% bootstrap interval
     (2,000 resamples, seed 0).
Verdict per measurement (fixed 2026-11-14, before the 17 clips were run):
  supported      median foreshortening ratio < 0.70 AND |rho pillar| < 0.30
  not supported  median foreshortening ratio >= 0.85
  inconclusive   otherwise
The depth share is descriptive only. On the reference-clip dry run, z ranges
exceeded the x and y ranges even for points that barely move (the nose in T2),
so image z is too noisy to drive a verdict.
Validated anchors only, so detection error does not mix with geometry.
Inputs: the pose cache from phase0_clip_sheet.py, anchor_truth.csv, the P0-A
validated rule_measurements.csv, the coach CSV and clip_sheet.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import CONTACT_WINDOW_FRAMES, S4_POST_CONTACT_ROTATION_FRAMES  # noqa: E402
from phase0_baseline_table import key_of, load_csv  # noqa: E402

NOSE = 0
SH = {"left": 11, "right": 12}
EL = {"left": 13, "right": 14}
WR = {"left": 15, "right": 16}
HIP = {"left": 23, "right": 24}
KNEE = {"left": 25, "right": 26}
ANK = {"left": 27, "right": 28}

MEASUREMENTS = {
    # id: (measurement column, pillar, suspended?)
    "A2": ("A2_elbow_angle_deg", "access", True),
    "A3": ("A3_compression_frames", "access", True),
    "A4": ("A4_torso_lean", "access", True),
    "S1": ("S1_hip_shift", "stability", True),
    "S3": ("S3_hip_drift_frames", "stability", True),
    "A5": ("A5_shoulder_hip_gap_deg", "access", False),
    "S4": ("S4_post_contact_rotation_deg", "stability", False),
    "T2": ("T2_early_head_change", "tracking", False),
    "F4": ("F4_pause_frames", "flow", False),
}


def _pt(arr: np.ndarray, i: int, spec: Any, w: float, h: float) -> np.ndarray:
    """Pixel point (x, y, z) for a landmark index or a tuple of indices (midpoint)."""
    idx = spec if isinstance(spec, tuple) else (spec,)
    p = arr[i, list(idx), :3].mean(axis=0)
    return np.array([p[0] * w, p[1] * h, p[2] * w])


def _moving_median(pts: np.ndarray, k: int) -> np.ndarray:
    half = k // 2
    return np.array([np.median(pts[max(0, i - half): i + half + 1], axis=0) for i in range(len(pts))])


def geometry(rule: str, a: dict[str, int], front: str, n: int) -> tuple[list[int], list[tuple[Any, Any]], list[Any]]:
    """Measured frames, segments (point pairs) and tracked points for a rule."""
    back = "right" if front == "left" else "left"
    c, hp, setup, bs, ft = a["contact_frame"], a["hands_peak_frame"], a["setup_frame"], a["hands_start_up_frame"], a["follow_through_frame"]
    cw = list(range(max(0, c - CONTACT_WINDOW_FRAMES), min(n, c + CONTACT_WINDOW_FRAMES + 1)))
    sh_mid, hip_mid = (SH["left"], SH["right"]), (HIP["left"], HIP["right"])
    shoulders, hips = (SH["left"], SH["right"]), (HIP["left"], HIP["right"])
    forearms = [(EL[front], WR[front]), (EL[back], WR[back])]
    if rule == "A2":
        return cw, [(SH[front], EL[front]), (EL[front], WR[front])], [EL[front], WR[front]]
    if rule == "A3":
        start = max(0, c - CONTACT_WINDOW_FRAMES)
        return list(range(start, min(n, start + 15))), forearms, [WR["left"], WR["right"]]
    if rule == "A4":
        return cw, [(sh_mid, hip_mid)], [sh_mid, hip_mid]
    if rule == "S1":
        return list(range(max(0, setup), min(n, c + CONTACT_WINDOW_FRAMES + 1))), [hips], [hip_mid]
    if rule == "S3":
        return list(range(max(0, bs), min(n, ft + 5))), [hips, (KNEE[front], ANK[front])], [hip_mid, ANK[front]]
    if rule == "A5":
        return cw, [shoulders, hips], [SH["left"], SH["right"], HIP["left"], HIP["right"]]
    if rule == "S4":
        return list(range(c, min(n, c + S4_POST_CONTACT_ROTATION_FRAMES + 1))), [shoulders, hips], \
            [SH["left"], SH["right"], HIP["left"], HIP["right"]]
    if rule == "T2":
        frames = list(range(max(0, setup - 10), setup + 1)) + list(range(bs, min(n, bs + 8)))
        return [f for f in frames if f < n], [(NOSE, hip_mid)], [NOSE, hip_mid]
    if rule == "F4":
        return list(range(max(0, hp - 2), min(n, hp + 4))), forearms, [WR["left"], WR["right"]]
    raise KeyError(rule)


def clip_indices(npz: Path, anchors: dict[str, int], front: str) -> dict[str, dict[str, float]]:
    data = np.load(npz)
    arr, frame_idx = data["landmarks"], data["frame_idx"]
    w, h = float(data["width"]), float(data["height"])
    pos = {int(f): i for i, f in enumerate(frame_idx)}
    a = {k: pos.get(v, v) for k, v in anchors.items()}
    n = len(arr)
    span = [i for i in range(max(0, a["setup_frame"]), min(n, a["follow_through_frame"] + 1))]
    out: dict[str, dict[str, float]] = {}
    for rule in MEASUREMENTS:
        frames, segments, tracked = geometry(rule, a, front, n)
        frames = [f for f in frames if 0 <= f < n and not np.isnan(arr[f]).any()]
        ratios = []
        for s0, s1 in segments:
            lengths_span = [np.linalg.norm((_pt(arr, i, s0, w, h) - _pt(arr, i, s1, w, h))[:2])
                            for i in span if not np.isnan(arr[i]).any()]
            ref = np.percentile(lengths_span, 95) if lengths_span else np.nan
            at = [np.linalg.norm((_pt(arr, i, s0, w, h) - _pt(arr, i, s1, w, h))[:2]) for i in frames]
            ratios.append(float(np.median(at) / ref) if at and ref else np.nan)
        dx = dy = dz = 0.0
        for spec in tracked:
            if spec == (HIP["left"], HIP["right"]):
                continue  # MediaPipe image z is relative to the hip midpoint: always ~0 there
            pts = np.array([_pt(arr, i, spec, w, h) for i in frames])
            if len(pts) > 2:
                smooth = _moving_median(pts, 5)
                d = smooth.max(axis=0) - smooth.min(axis=0)  # range, not path length: jitter does not accumulate
                dx, dy, dz = dx + d[0], dy + d[1], dz + d[2]
        total = dx + dy + dz
        sh_len = np.nan
        if rule in ("A5", "S4") and frames:
            sh_len = float(np.median([np.linalg.norm((_pt(arr, i, SH["left"], w, h) - _pt(arr, i, SH["right"], w, h))[:2])
                                      for i in frames]))
        out[rule] = {
            "foreshortening_ratio": float(np.nanmin(ratios)) if not all(np.isnan(ratios)) else np.nan,
            "depth_share": float(dz / total) if total else np.nan,
            "shoulder_line_px": sh_len,
            "frames_used": len(frames),
        }
    return out


def spearman_ci(x: list[float], y: list[float], seed: int = 0, reps: int = 2000) -> tuple[float, float, float]:
    xa, ya = np.array(x, float), np.array(y, float)
    keep = ~(np.isnan(xa) | np.isnan(ya))
    xa, ya = xa[keep], ya[keep]
    if len(xa) < 4:
        return np.nan, np.nan, np.nan

    def rho(u: np.ndarray, v: np.ndarray) -> float:
        import pandas as pd  # average ranks for ties (argsort ranks break ties arbitrarily)
        ru, rv = pd.Series(u).rank().to_numpy(), pd.Series(v).rank().to_numpy()
        if ru.std() == 0 or rv.std() == 0:
            return np.nan
        return float(np.corrcoef(ru, rv)[0, 1])

    rng = np.random.default_rng(seed)
    boots = [rho(xa[s], ya[s]) for s in (rng.integers(0, len(xa), len(xa)) for _ in range(reps))]
    boots = [b for b in boots if not np.isnan(b)]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return rho(xa, ya), float(lo), float(hi)


def verdict(ratio: float, share: float, rho_p: float) -> str:
    """share is accepted for the record but does not decide the verdict (see docstring)."""
    if np.isnan(ratio):
        return "inconclusive"
    weak = np.isnan(rho_p) or abs(rho_p) < 0.30
    if ratio < 0.70 and weak:
        return "supported"
    if ratio >= 0.85:
        return "not supported"
    return "inconclusive"


def f(v: Any, nd: int = 2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default=str(SCRIPT_DIR / "calibration_output" / "pose_cache"))
    parser.add_argument("--measurements", required=True, help="validated rule_measurements.csv from the P0-A run")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--anchor-truth", default=str(SCRIPT_DIR / "anchor_truth.csv"))
    parser.add_argument("--ground-truth", default=str(SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"))
    parser.add_argument("--clip-sheet", default=str(SCRIPT_DIR / "clip_sheet.csv"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchors = {key_of(r["filename"]): r for r in load_csv(Path(args.anchor_truth))}
    coach = {key_of(r["filename"]): r for r in load_csv(Path(args.ground_truth))}
    meas = {key_of(r["filename"]): r for r in load_csv(Path(args.measurements))}
    sheet = {key_of(r["filename"]): r for r in load_csv(Path(args.clip_sheet))} if Path(args.clip_sheet).exists() else {}
    cache = {p.stem.casefold(): p for p in Path(args.cache_dir).glob("*.npz")}

    by_clip: list[dict[str, Any]] = []
    missing = []
    for k, arow in anchors.items():
        if k not in cache:
            missing.append(arow["filename"])
            continue
        a = {key: int(arow[key]) for key in ("setup_frame", "hands_start_up_frame", "front_foot_down_frame",
                                             "hands_peak_frame", "contact_frame", "follow_through_frame")}
        hand = (sheet.get(k, {}).get("handedness") or "right").lower()
        front = "right" if hand == "left" else "left"
        idx = clip_indices(cache[k], a, front)
        cam = sheet.get(k, {}).get("camera_position") or "pending"
        for rule, vals in idx.items():
            col, pillar, suspended = MEASUREMENTS[rule]
            mv = meas.get(k, {}).get(col, "")
            by_clip.append({
                "filename": arow["filename"], "rule": rule, "suspended": suspended, "handedness": hand,
                "camera_position": cam, "measurement": mv,
                "coach_pillar": coach.get(k, {}).get(f"{pillar}_score", ""),
                "coach_overall": coach.get(k, {}).get("overall_score", ""), **vals,
            })

    with (out_dir / "wrong_plane_by_clip.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(by_clip[0].keys()) if by_clip else ["filename"])
        w.writeheader()
        w.writerows(by_clip)

    def num(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    summary = []
    for rule, (col, pillar, suspended) in MEASUREMENTS.items():
        rows = [r for r in by_clip if r["rule"] == rule]
        ratio = float(np.nanmedian([r["foreshortening_ratio"] for r in rows])) if rows else np.nan
        share = float(np.nanmedian([r["depth_share"] for r in rows])) if rows else np.nan
        rp, rp_lo, rp_hi = spearman_ci([num(r["measurement"]) for r in rows], [num(r["coach_pillar"]) for r in rows])
        ro, _, _ = spearman_ci([num(r["measurement"]) for r in rows], [num(r["coach_overall"]) for r in rows])
        cams = {}
        for r in rows:
            cams.setdefault(r["camera_position"], []).append(r["foreshortening_ratio"])
        summary.append({
            "rule": rule, "measurement": col, "pillar": pillar, "status": "suspended" if suspended else "active",
            "n": len(rows), "median_foreshortening_ratio": ratio, "median_depth_share": share,
            "rho_pillar": rp, "rho_pillar_ci_low": rp_lo, "rho_pillar_ci_high": rp_hi, "rho_overall": ro,
            "ratio_by_camera_position": "; ".join(f"{c}: {np.nanmedian(v):.2f} (n={len(v)})" for c, v in sorted(cams.items())),
            "median_shoulder_line_px": float(np.nanmedian([r["shoulder_line_px"] for r in rows])) if rule in ("A5", "S4") and rows else np.nan,
            "verdict": verdict(ratio, share, rp),
        })
    with (out_dir / "wrong_plane_summary.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    lines = ["# Wrong-plane test (Phase 0)", "",
             f"Clips: {len({r['filename'] for r in by_clip})} with validated anchors. n is small, so all of this is exploratory.",
             "Verdict rules were fixed before the data was seen (see the script docstring).", "",
             "| Rule | Status | Pillar | Foreshortening ratio | Depth share (descriptive) | ρ vs coach pillar [95% CI] | ρ vs overall | Verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['rule']} | {s['status']} | {s['pillar']} | {f(s['median_foreshortening_ratio'])} | {f(s['median_depth_share'])} | "
                     f"{f(s['rho_pillar'])} [{f(s['rho_pillar_ci_low'])}, {f(s['rho_pillar_ci_high'])}] | {f(s['rho_overall'])} | {s['verdict']} |")
    lines += ["", "Foreshortening ratio by camera position (descriptive; tier is confounded with venue):", ""]
    lines += [f"- {s['rule']}: {s['ratio_by_camera_position']}" for s in summary]
    sh = [s for s in summary if s["rule"] in ("A5", "S4")]
    lines += ["", "Shoulder line at the measured frames (A5, S4 use atan2(|dx|, |dy|), which is unstable when the "
              f"shoulders point at the camera): median {f(sh[0]['median_shoulder_line_px'], 0)} px.", ""]
    lines += ["Limits: MediaPipe image z is a rough depth estimate; the ratio uses each clip as its own reference, "
              "so a segment that never faces the camera has no unforeshortened length to compare with."]
    if missing:
        lines += ["", "No pose cache for: " + ", ".join(missing)]
    (out_dir / "wrong_plane_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
