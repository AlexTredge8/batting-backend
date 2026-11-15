#!/usr/bin/env python3
"""
Phase 5: one measurement per rubric checkpoint (docs/phase1/rubric_v0.json).

Each measurement M01..M12 belongs to one checkpoint C01..C12 and names the fault
F01..F12 that a tip would give. Rules for every measurement (plan, Section 6):
  - time windows in milliseconds, converted with the clip's fps (never fixed frame counts);
  - windows anchored on contact wherever possible (the most accurate key frame: automatic
    contact is out by a median 1 frame, automatic set-up by 6 and follow-through by 8);
  - distances in body lengths (BL): median nose-to-ankle height from 1000 to 700 ms before
    contact (the set-up), in pixels; values are reported as % BL;
  - left-handers mirrored (x -> width - x) and front/back sides swapped, so every
    clip reads as a right-hander. After the mirror, the off side is camera-left
    (smaller x) for a camera behind the bowler. Checked on the reference clip from the
    feet (toe minus heel points camera-left at set-up); toes_point_off flags clips where not;
  - the hands are the top (front) wrist: the bottom wrist is mostly hidden behind the body;
  - landmarks with visibility under 0.2 are dropped, then smoothed with a 5-frame moving median before measuring;
  - a measurement is blank (NaN) with a reason when its frames or landmarks are missing.

Nothing here scores or changes the product. Phase 5 entry tests
(phase5_entry_tests.py) decide which measurements survive.

  python phase5_checkpoints.py --cache-dir calibration_output/pose_cache --out <dir>
      [--anchors anchor_truth.csv] [--auto-table baselines/phase0_2026-11-14/baseline_table.csv]

Writes checkpoint_measurements.csv: one row per clip x anchor set x measurement.
No landmarks or frames are written.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase0_baseline_table import key_of, load_csv  # noqa: E402

NOSE = 0
L = {"sh": 11, "el": 13, "wr": 15, "hip": 23, "knee": 25, "ank": 27}
R = {"sh": 12, "el": 14, "wr": 16, "hip": 24, "knee": 26, "ank": 28}
ANCHORS = ["setup_frame", "hands_start_up_frame", "front_foot_down_frame", "hands_peak_frame",
           "contact_frame", "follow_through_frame"]
MEASUREMENT_SET = "v2"  # v1 (14 Nov) used set-up / hands / follow-through anchors; v2 (15 Nov) anchors M01, M03, M06, M08, M09, M11, M12 on contact
MIN_VIS = 0.2  # MediaPipe still places hidden points; below 0.2 they are guesses


@dataclass
class Clip:
    """Smoothed pixel landmarks for one clip, mirrored to a right-hander."""
    xy: np.ndarray        # [n, 33, 2] pixels, NaN where not detected or low visibility
    fps: float
    width: float
    height: float
    pos: dict[int, int]   # video frame index -> row in xy
    front: dict[str, int]
    back: dict[str, int]
    drive: str = "off"    # off / straight / on, from clip_sheet.csv

    def row(self, frame: int) -> int | None:
        return self.pos.get(int(frame))

    def ms(self, ms: float) -> int:
        return max(1, int(round(ms * self.fps / 1000.0)))

    def pt(self, frame: int, idx: int | tuple[int, ...]) -> np.ndarray:
        r = self.row(frame)
        if r is None:
            return np.array([np.nan, np.nan])
        ids = idx if isinstance(idx, tuple) else (idx,)
        return self.xy[r, list(ids)].mean(axis=0)

    def series(self, frames: range | list[int], idx: int | tuple[int, ...]) -> np.ndarray:
        return np.array([self.pt(f, idx) for f in frames]).reshape(-1, 2)


def _moving_median(a: np.ndarray, k: int = 5) -> np.ndarray:
    half, out = k // 2, np.empty_like(a)
    for i in range(len(a)):
        with np.errstate(all="ignore"):
            out[i] = np.nanmedian(a[max(0, i - half): i + half + 1], axis=0)
    return out


def load_clip(npz: Path, handedness: str, drive: str = "off") -> Clip:
    d = np.load(npz)
    arr, frame_idx = d["landmarks"].astype(float), d["frame_idx"]
    w, h, fps = float(d["width"]), float(d["height"]), float(d["fps"])
    xy = np.stack([arr[..., 0] * w, arr[..., 1] * h], axis=-1)
    xy[arr[..., 3] < MIN_VIS] = np.nan
    left_handed = handedness.strip().lower() == "left"
    if left_handed:
        xy[..., 0] = w - xy[..., 0]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        xy = _moving_median(xy, 5)
    front, back = (R, L) if left_handed else (L, R)  # a right-hander's front side is the left
    return Clip(xy, fps, w, h, {int(f): i for i, f in enumerate(frame_idx)}, front, back, drive or "off")


def _setup_window(c: Clip, a: dict[str, int]) -> range:
    """The set-up, anchored on contact: 1000 to 700 ms before it (validated set-up is a median
    964 ms before contact; automatic contact is out by 1 frame, automatic set-up by 6)."""
    k = a["contact_frame"]
    return range(k - c.ms(1000), k - c.ms(700) + 1)


def body_length(c: Clip, a: dict[str, int]) -> float:
    frames = _setup_window(c, a)
    nose = c.series(frames, NOSE)
    ank = c.series(frames, (L["ank"], R["ank"]))
    with np.errstate(all="ignore"):
        bl = float(np.nanmedian(ank[:, 1] - nose[:, 1]))
    if not np.isfinite(bl) or bl <= 0:  # set-up cut off: fall back to the clip's tallest 10%
        allh = c.xy[:, (L["ank"], R["ank"]), 1].mean(axis=1) - c.xy[:, NOSE, 1]
        bl = float(np.nanpercentile(allh, 90)) if np.isfinite(allh).any() else np.nan
    return bl


def _pct(v: float, bl: float) -> float:
    return 100.0 * v / bl


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1, 1)))) if n else np.nan


def _speed(c: Clip, frames: list[int], idx: int | tuple[int, ...]) -> np.ndarray:
    p = c.series(frames, idx)
    return np.linalg.norm(np.diff(p, axis=0), axis=1) * c.fps  # px per second


# --- the twelve measurements -------------------------------------------------------
# Each returns (value, reason-if-blank). Direction: "higher_worse" or "higher_better",
# registered before the entry tests are run (14 Nov 2026).

def _top_hand_peak(c, k: int) -> int | None:
    """Frame where the top hand is highest between 450 and 100 ms before contact (contact-anchored)."""
    fr = list(range(k - c.ms(450), k - c.ms(100) + 1))
    y = c.series(fr, c.front["wr"])[:, 1]
    return fr[int(np.nanargmin(y))] if np.isfinite(y).any() else None


def m01_head_still_setup(c, a, bl):
    """SD of head position from 1000 to 700 ms before contact (the set-up, before the hands start), % BL."""
    k = a["contact_frame"]
    p = c.series(range(k - c.ms(1000), k - c.ms(700) + 1), NOSE)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 4:
        return np.nan, "set-up not in the clip (needs 1 s before contact)"
    return _pct(float(np.sqrt(p.var(axis=0).sum())), bl), ""


def m02_balanced_base(c, a, bl):
    """Sideways offset of the hips from the middle of the feet at set-up, % BL (absolute)."""
    s = a["setup_frame"]
    fr = range(s - c.ms(200), s + 1)
    hip = c.series(fr, (L["hip"], R["hip"]))[:, 0]
    ank = c.series(fr, (L["ank"], R["ank"]))[:, 0]
    with np.errstate(all="ignore"):
        v = float(np.nanmedian(np.abs(hip - ank)))
    return (_pct(v, bl), "") if np.isfinite(v) else (np.nan, "hips or feet not seen at set-up")


def m03_backlift_straight(c, a, bl):
    """Sideways travel of the top hand towards the off side (gully), relative to the shoulders,
    from the set-up (850-700 ms before contact) to the top of the backlift, % BL. The whole body
    moves across in the stride, so the hand is measured against the shoulder midpoint."""
    k = a["contact_frame"]
    w, sh = c.front["wr"], (L["sh"], R["sh"])
    top = _top_hand_peak(c, k)
    if top is None:
        return np.nan, "top hand not seen in the backlift"
    def rel(fr):
        with np.errstate(all="ignore"):
            return float(np.nanmedian(c.series(fr, w)[:, 0] - c.series(fr, sh)[:, 0]))
    v = rel(range(k - c.ms(850), k - c.ms(700) + 1)) - rel(range(top - 1, top + 2))  # off side is smaller x
    return (_pct(v, bl), "") if np.isfinite(v) else (np.nan, "top hand or shoulders not seen at set-up")


def m04_head_leads(c, a, bl):
    """Head position towards the off side relative to the front foot when the front foot lands,
    % BL (signed: positive = head over or beyond the foot, negative = head left behind on the leg side)."""
    f = a["front_foot_down_frame"]
    fr = range(f - 1, f + 2)
    v = c.series(fr, c.front["ank"])[:, 0] - c.series(fr, NOSE)[:, 0]  # off side is smaller x after the mirror
    with np.errstate(all="ignore"):
        v = float(np.nanmedian(v))
    return (_pct(v, bl), "") if np.isfinite(v) else (np.nan, "head or front foot not seen at foot-down")


def m05_front_foot_line(c, a, bl):
    """Sideways step of the front foot towards the drive's side, set-up to foot-down, % BL (signed).
    Off and straight drives count a step to the off side as positive; on drives a step to the leg side."""
    s, f = a["setup_frame"], a["front_foot_down_frame"]
    p0 = c.series(range(s - c.ms(100), s + 1), c.front["ank"])[:, 0]
    p1 = c.series(range(f, f + c.ms(70) + 1), c.front["ank"])[:, 0]
    with np.errstate(all="ignore"):
        v = float(np.nanmedian(p0) - np.nanmedian(p1))  # off side is smaller x after the mirror
    if c.drive == "on":
        v = -v
    return (_pct(v, bl), "") if np.isfinite(v) else (np.nan, "front foot not seen")


def m06_front_knee_bends(c, a, bl):
    """Smallest front-knee angle from 150 ms before to 100 ms after contact, degrees (2D)."""
    k = a["contact_frame"]
    fr = range(k - c.ms(150), k + c.ms(100) + 1)
    angs = [_angle(c.pt(f, c.front["hip"]), c.pt(f, c.front["knee"]), c.pt(f, c.front["ank"])) for f in fr]
    angs = [x for x in angs if np.isfinite(x)]
    return (float(min(angs)), "") if angs else (np.nan, "front leg not seen")


def _shoulder_width(c, frames) -> float:
    d = c.series(frames, c.front["sh"]) - c.series(frames, c.back["sh"])
    with np.errstate(all="ignore"):
        return float(np.nanmedian(np.abs(d[:, 0])))


def m07_stays_side_on(c, a, bl):
    """Shoulder-line width 100 ms before contact over its width at set-up (1.0 = unchanged)."""
    s, k = a["setup_frame"], a["contact_frame"]
    w0 = _shoulder_width(c, range(s - c.ms(200), s + 1))
    t = k - c.ms(100)
    w1 = _shoulder_width(c, range(t - 1, t + 2))
    if not (np.isfinite(w0) and np.isfinite(w1)):
        return np.nan, "shoulders not seen"
    w0 = max(w0, 0.02 * bl)  # a perfectly side-on set-up must not divide by ~0
    return w1 / w0, ""


def m08_downswing_straight(c, a, bl):
    """Largest sideways bow of the top hand's path from the top of the backlift to contact, % BL."""
    k = a["contact_frame"]
    hp = _top_hand_peak(c, k)
    if hp is None:
        return np.nan, "top hand not seen in the backlift"
    p = c.series(range(hp, k + 1), c.front["wr"])
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 3:
        return np.nan, "hands not seen in the downswing"
    chord = p[-1] - p[0]
    n = np.linalg.norm(chord)
    if n < 1e-6:
        return np.nan, "hands do not move"
    rel = p - p[0]
    perp = np.abs(chord[0] * rel[:, 1] - chord[1] * rel[:, 0]) / n
    return _pct(float(perp.max()), bl), ""


def m09_one_rhythm(c, a, bl):
    """Longest pause of the top hand (speed under 15% of its peak) from the start of the swing
    to contact, ms. The swing starts at the first frame after 700 ms before contact where the
    hand moves faster than 30% of its peak (contact-anchored, no set-up detector)."""
    k = a["contact_frame"]
    fr = list(range(k - c.ms(700), k + 1))
    sp = _speed(c, fr, c.front["wr"])
    if not np.isfinite(sp).any():
        return np.nan, "hands not seen"
    peak = np.nanmax(sp)
    moving = np.where(sp > 0.30 * peak)[0]
    if len(moving) == 0:
        return np.nan, "hands do not move"
    slow = sp[moving[0]:] < 0.15 * peak
    run = best = 0
    for s in slow:
        run = run + 1 if s else 0
        best = max(best, run)
    return 1000.0 * best / c.fps, ""


def m10_head_over_ball(c, a, bl):
    """Head position towards the off side relative to the middle of the hips at contact, % BL
    (signed: negative = head falling away to the leg side)."""
    k = a["contact_frame"]
    fr = range(k - 1, k + 2)
    v = c.series(fr, (L["hip"], R["hip"]))[:, 0] - c.series(fr, NOSE)[:, 0]
    with np.errstate(all="ignore"):
        v = float(np.nanmedian(v))
    return (_pct(v, bl), "") if np.isfinite(v) else (np.nan, "head or hips not seen at contact")


def m11_balance_finish(c, a, bl):
    """Sideways gap between the hips and the front foot, 400 to 700 ms after contact, % BL."""
    k = a["contact_frame"]
    fr = range(k + c.ms(400), k + c.ms(700) + 1)
    v = np.abs(c.series(fr, (L["hip"], R["hip"]))[:, 0] - c.series(fr, c.front["ank"])[:, 0])
    with np.errstate(all="ignore"):
        v = float(np.nanmedian(v))
    return (_pct(v, bl), "") if np.isfinite(v) else (np.nan, "finish not in the clip (needs 0.7 s after contact)")


def m12_hands_through(c, a, bl):
    """Highest the top hand gets above the shoulders from 150 to 700 ms after contact, % BL."""
    k = a["contact_frame"]
    fr = range(k + c.ms(150), k + c.ms(700) + 1)
    hands = c.series(fr, c.front["wr"])[:, 1]
    sh = c.series(fr, (L["sh"], R["sh"]))[:, 1]
    with np.errstate(all="ignore"):
        v = float(np.nanmax(sh - hands))  # image y grows downwards
    return (_pct(v, bl), "") if np.isfinite(v) else (np.nan, "hands or shoulders not seen after contact")


MEASURES: list[tuple[str, str, str, str, str, str, Callable]] = [
    # id, checkpoint, fault, pillar, unit, direction, function
    ("M01", "C01", "F01", "tracking", "% BL", "higher_worse", m01_head_still_setup),
    ("M02", "C02", "F02", "stability", "% BL", "higher_worse", m02_balanced_base),
    ("M03", "C03", "F03", "access", "% BL, off side +", "higher_worse", m03_backlift_straight),
    ("M04", "C04", "F04", "tracking", "% BL, off side +", "higher_better", m04_head_leads),
    ("M05", "C05", "F05", "access", "% BL, drive side +", "higher_better", m05_front_foot_line),
    ("M06", "C06", "F06", "stability", "degrees", "higher_worse", m06_front_knee_bends),
    ("M07", "C07", "F07", "access", "ratio", "higher_worse", m07_stays_side_on),
    ("M08", "C08", "F08", "access", "% BL", "higher_worse", m08_downswing_straight),
    ("M09", "C09", "F09", "flow", "ms", "higher_worse", m09_one_rhythm),
    ("M10", "C10", "F10", "tracking", "% BL, off side +", "higher_better", m10_head_over_ball),
    ("M11", "C11", "F11", "stability", "% BL", "higher_worse", m11_balance_finish),
    ("M12", "C12", "F12", "flow", "% BL", "higher_better", m12_hands_through),
]


def toes_point_off(c: Clip, a: dict[str, int]) -> str:
    """'yes' if the toes point camera-left (the off side after the mirror) over the set-up window.
    'no' means the handedness or the camera position is probably wrong for this clip."""
    fr = _setup_window(c, a)
    d = np.concatenate([c.series(fr, 31)[:, 0] - c.series(fr, 29)[:, 0], c.series(fr, 32)[:, 0] - c.series(fr, 30)[:, 0]])
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return "unknown"
    return "yes" if np.median(d) < 0 else "no"


def measure_clip(c: Clip, anchors: dict[str, int]) -> list[dict]:
    a = {k: int(v) for k, v in anchors.items()}
    bl = body_length(c, a)
    toes = toes_point_off(c, a)
    rows = []
    for mid, cp, fault, pillar, unit, direction, fn in MEASURES:
        if not np.isfinite(bl):
            v, why = np.nan, "body length unknown"
        else:
            try:
                v, why = fn(c, a, bl)
            except (KeyError, IndexError, ValueError) as exc:
                v, why = np.nan, f"error: {type(exc).__name__}"
        rows.append({"measurement": mid, "checkpoint": cp, "fault": fault, "pillar": pillar, "unit": unit,
                     "direction": direction, "value": "" if not np.isfinite(v) else round(float(v), 3),
                     "blank_reason": why, "body_length_px": round(bl, 1) if np.isfinite(bl) else "",
                     "toes_point_off": toes})
    return rows


def anchor_sets(anchor_csv: Path, auto_table: Path | None) -> dict[str, dict[str, dict[str, int]]]:
    """{'validated': {clip_key: anchors}, 'auto': {...}}"""
    sets: dict[str, dict[str, dict[str, int]]] = {"validated": {}}
    for r in load_csv(anchor_csv):
        if all(r.get(k, "").strip() for k in ANCHORS):
            sets["validated"][key_of(r["filename"])] = {k: int(float(r[k])) for k in ANCHORS}
    if auto_table and auto_table.exists():
        sets["auto"] = {}
        for r in load_csv(auto_table):
            if r.get("mode") == "auto" and all(r.get(f"{k}_sys", "").strip() for k in ANCHORS):
                sets["auto"][key_of(r["filename"])] = {k: int(float(r[f"{k}_sys"])) for k in ANCHORS}
    return sets


def main(argv: list[str] | None = None) -> int:
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows are reported as blanks with a reason
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(SCRIPT_DIR / "calibration_output" / "pose_cache"))
    ap.add_argument("--anchors", default=str(SCRIPT_DIR / "anchor_truth.csv"))
    ap.add_argument("--auto-table", default=str(SCRIPT_DIR / "baselines" / "phase0_2026-11-14" / "baseline_table.csv"))
    ap.add_argument("--clip-sheet", default=str(SCRIPT_DIR / "clip_sheet.csv"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    sheet = load_csv(Path(args.clip_sheet))
    hand = {key_of(r["filename"]): (r.get("handedness") or "right") for r in sheet}
    drive = {key_of(r["filename"]): (r.get("drive_type") or "off") for r in sheet}
    sets = anchor_sets(Path(args.anchors), Path(args.auto_table))
    caches = {key_of(p.stem): p for p in Path(args.cache_dir).glob("*.npz")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for k, npz in sorted(caches.items()):
        clip = load_clip(npz, hand.get(k, "right"), drive.get(k, "off"))
        for set_name, by_clip in sets.items():
            if k not in by_clip:
                continue
            for r in measure_clip(clip, by_clip[k]):
                rows.append({"filename": npz.stem, "measurement_set": MEASUREMENT_SET, "anchor_set": set_name, "handedness": hand.get(k, "right"), "drive_type": clip.drive,
                             "fps": round(clip.fps, 3), **r})
    for k in sets["validated"]:
        if k not in caches:
            missing.append(k)
    with (out / "checkpoint_measurements.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    spec = [{"measurement": m, "checkpoint": c, "fault": f, "pillar": p, "unit": u, "direction": d,
             "definition": (fn.__doc__ or "").strip()} for m, c, f, p, u, d, fn in MEASURES]
    (out / "measurement_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"{len(caches)} clips, {len(rows)} rows -> {out / 'checkpoint_measurements.csv'}")
    if missing:
        print(f"No pose cache for {len(missing)} clips with anchors: {', '.join(sorted(missing))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
