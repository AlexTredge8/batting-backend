#!/usr/bin/env python3
"""
Phase 5: one measurement per rubric checkpoint (docs/phase1/rubric_v0.json).

Each measurement belongs to one checkpoint and names the fault that a tip would give:
M01..M13 to C01..C12 / F01..F12 (rubric v0); M14..M17 (v5) to C09, C01, C16 and C13 (C13 and C16
are Cam's checkpoints in rubric v0.2, docs/phase1/rubric_v0_2.json). Rules for every measurement (plan, Section 6):
  - time windows in milliseconds, converted with the clip's fps (never fixed frame counts);
  - windows anchored on contact wherever possible (the most accurate key frame: automatic
    contact is out by a median 1 frame, automatic set-up by 6 and follow-through by 8);
  - distances in body lengths (BL): median nose-to-ankle height from 1000 to 700 ms before
    contact (the set-up), in pixels; values are reported as % BL;
  - left-handers mirrored (x -> width - x) and front/back sides swapped, so every
    clip reads as a right-hander. From v3 the feet decide the geometry when they disagree
    with the clip sheet (a mirrored video or a camera behind the batter flips the picture
    just as a left-hander does); geometry_source says which was used. After the mirror, the off side is camera-left
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
MEASUREMENT_SET = "v5"  # v5 (15 Nov): M14-M17 added (foot-landing/downswing coupling, head over the base at set-up, weight over the front foot at contact, trigger), M01-M13 unchanged; v4 (15 Nov): hands = the more visible wrist; M09 total slow time; v3 (15 Nov): M06, M08, M09 reworked for robustness, M13 added, geometry handedness from the feet; v1 (14 Nov) used set-up / hands / follow-through anchors; v2 (15 Nov) anchors M01, M03, M06, M08, M09, M11, M12 on contact
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
    hand: int = 15        # the wrist tracked as "the hands": the more visible one over the clip (v4)

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
    # v4: the hands are whichever wrist is more visible over the clip. Both hands are on the bat, so
    # either tracks the hands; the hidden one is a guess. This does not depend on left/right labels,
    # which a mirrored or behind-the-batter clip can swap (v3: blank M13 on the 4 flipped clips).
    with np.errstate(all="ignore"):
        vis = {i: float(np.nanmedian(arr[:, i, 3])) for i in (L["wr"], R["wr"])}
    other = back["wr"]
    hand = other if (np.isfinite(vis[other]) and vis[other] > (vis[front["wr"]] if np.isfinite(vis[front["wr"]]) else -1) + 0.1) \
        else front["wr"]  # the top (front) hand unless the other wrist is clearly more visible
    return Clip(xy, fps, w, h, {int(f): i for i, f in enumerate(frame_idx)}, front, back, drive or "off", hand)


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
    y = c.series(fr, c.hand)[:, 1]
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
    w, sh = c.hand, (L["sh"], R["sh"])
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
    """Front-knee bend around contact: 10th percentile of the 3-frame-median front-knee angle
    from 200 ms before to 200 ms after contact, degrees (2D). v3: a percentile of a smoothed
    series, not the single smallest frame, so one bad pose frame cannot set the value."""
    k = a["contact_frame"]
    fr = range(k - c.ms(200), k + c.ms(200) + 1)
    angs = np.array([_angle(c.pt(f, c.front["hip"]), c.pt(f, c.front["knee"]), c.pt(f, c.front["ank"])) for f in fr])
    if np.isfinite(angs).sum() < 4:
        return np.nan, "front leg not seen"
    sm = np.array([np.nanmedian(angs[max(0, i - 1): i + 2]) for i in range(len(angs))])
    return float(np.nanpercentile(sm, 10)), ""


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
    """Hands across the line: sideways travel of the top hand, relative to the shoulders, from
    200 ms before to 100 ms after contact, AWAY from the drive's side, % BL (signed). v3: a fixed
    contact window with 3-frame medians at each end, in place of v2's bow from a detected top of
    backlift, which moved by several frames when contact moved by one."""
    k = a["contact_frame"]
    w, sh = c.hand, (L["sh"], R["sh"])
    def rel(f):
        fr = range(f - 1, f + 2)
        with np.errstate(all="ignore"):
            return float(np.nanmedian(c.series(fr, w)[:, 0] - c.series(fr, sh)[:, 0]))
    v = rel(k + c.ms(100)) - rel(k - c.ms(200))  # + = towards the leg side (larger x after the mirror)
    if c.drive == "on":
        v = -v
    return (_pct(v, bl), "") if np.isfinite(v) else (np.nan, "top hand or shoulders not seen around contact")


def _longest_below(sp: np.ndarray, thr: float, fps: float) -> float:
    """Longest time the speed stays under thr, in ms, with crossings placed by linear
    interpolation between frames (so the value is not stuck on whole frames). A slow stretch
    that is already under way when the window opens is the set-up stillness, not a pause, and
    is not counted."""
    best, start = 0.0, None
    for i in range(len(sp)):
        below = np.isfinite(sp[i]) and sp[i] < thr
        if below and start is None:
            if i == 0 or not np.isfinite(sp[i - 1]):
                start = -1.0  # still before the window: stillness before the swing, not a pause in it
            else:
                start = i - 1 + (sp[i - 1] - thr) / (sp[i - 1] - sp[i])
        if start is not None and (not below or i == len(sp) - 1):
            if start < 0:
                start = None
                continue
            if below:
                end = float(i)
            else:
                end = i - 1 + (thr - sp[i - 1]) / (sp[i] - sp[i - 1])
            best = max(best, end - start)
            start = None
    return 1000.0 * best / fps


def _time_below(sp: np.ndarray, thr: float, fps: float) -> float:
    """Total time the speed is under thr, ms, with crossings placed by linear interpolation.
    Moving the window by one frame changes this by at most one frame's worth (v4)."""
    tot = 0.0
    for i in range(1, len(sp)):
        a, b = sp[i - 1], sp[i]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a < thr and b < thr:
            tot += 1.0
        elif (a < thr) != (b < thr):
            tot += ((thr - a) / (b - a)) if a < thr else ((thr - b) / (a - b))
    return 1000.0 * tot / fps


def m09_one_rhythm(c, a, bl):
    """Hands pause in the swing: total time the hands' 3-frame-median speed is under 15% of its
    90th percentile, from 400 ms before contact to contact, ms (timed between frames). Every
    swing has a short turn at the top; a pause or hitch adds to it. v4: total time, not the
    longest run, because v3's longest run jumped 0 <-> 140 ms when contact moved one frame."""
    k = a["contact_frame"]
    fr = list(range(k - c.ms(400), k + 1))
    raw = _speed(c, fr, c.hand)
    if np.isfinite(raw).sum() < 4:
        return np.nan, "hands not seen in the swing"
    with np.errstate(all="ignore"):
        sp = np.array([np.nanmedian(raw[max(0, i - 1): i + 2]) for i in range(len(raw))])
    peak = np.nanpercentile(sp, 90)
    if peak <= 0:
        return np.nan, "hands do not move"
    return _time_below(sp, 0.15 * peak, c.fps), ""

def m13_hands_accelerate(c, a, bl):
    """Swing carries through contact: the top hand's fastest upward speed in the 400 ms after
    contact over its fastest downward speed in the 300 ms before it (vertical only: from behind
    the bowler the hands travel towards the camera at contact, so total 2D speed there is
    mostly hidden; the down-and-up is visible). A checked swing scores low. Added 15 Nov 2026 (v3)."""
    k = a["contact_frame"]
    fr = list(range(k - c.ms(300), k + c.ms(400) + 1))
    y = c.series(fr, c.hand)[:, 1]
    raw = np.diff(y) * c.fps  # image y grows downwards: + = moving down
    with np.errstate(all="ignore"):  # 3-frame median against single bad pose frames
        vy = np.array([np.nanmedian(raw[max(0, j - 1): j + 2]) for j in range(len(raw))])
    i = fr.index(k)
    down, up = vy[:i], -vy[i:]
    if np.isfinite(down).sum() < 2 or np.isfinite(up).sum() < 2:
        return np.nan, "top hand not seen around contact"
    d, u = float(np.nanmax(down)), float(np.nanmax(up))
    return (u / d, "") if d > 0 else (np.nan, "hands do not come down before contact")

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
    hands = c.series(fr, c.hand)[:, 1]
    sh = c.series(fr, (L["sh"], R["sh"]))[:, 1]
    with np.errstate(all="ignore"):
        v = float(np.nanmax(sh - hands))  # image y grows downwards
    return (_pct(v, bl), "") if np.isfinite(v) else (np.nan, "hands or shoulders not seen after contact")


# --- added 15 Nov 2026 (v5), registered in docs/phase5/README.md before any clip was run ----------

def _front_foot_lands(c, k: int) -> tuple[float | None, str]:
    """Front-foot landing near contact, without the legacy foot-down detector: the first moment in
    the 450 ms before contact, after the front ankle's peak vertical speed, when its 3-frame-median
    vertical speed falls below 15% of that peak. Returned as a (fractional) video frame: the crossing
    is placed by linear interpolation between frames, so a one-frame shift of contact cannot move it
    by a whole frame on its own."""
    fr = list(range(k - c.ms(450), k + 1))
    y = c.series(fr, c.front["ank"])[:, 1]
    raw = np.abs(np.diff(y)) * c.fps  # px per second; speed j lies between fr[j] and fr[j + 1]
    if np.isfinite(raw).sum() < 4:
        return None, "front foot not seen before contact"
    with np.errstate(all="ignore"):
        sp = np.array([np.nanmedian(raw[max(0, j - 1): j + 2]) for j in range(len(raw))])
    p = int(np.nanargmax(sp))
    thr = 0.15 * sp[p]
    if not sp[p] > 0:
        return None, "front foot does not move before contact"
    for j in range(p + 1, len(sp)):
        if np.isfinite(sp[j]) and sp[j] < thr:
            prev = sp[j - 1]
            t = (j - 1 + (prev - thr) / (prev - sp[j])) if np.isfinite(prev) else float(j)
            return fr[0] + t + 0.5, ""
    return None, "front foot still moving at contact"


def m14_foot_downswing_coupled(c, a, bl):
    """Front-foot landing coupled to the start of the downswing (Weissensteiner et al. 2011): the gap
    in ms between the front foot landing (_front_foot_lands: front ankle's 3-frame-median vertical
    speed first under 15% of its peak, after the peak, in the 450 ms before contact) and the start of
    the downswing (the top hand's highest point 450 to 100 ms before contact, _top_hand_peak).
    The signed gap is downswing start minus landing: negative = the hands start down before the
    stride is complete (the less-skilled pattern), positive = the foot lands first and the hands wait.
    Skilled batters land as the downswing starts, so closer to zero is better: the value stored is
    the ABSOLUTE gap, ms, and higher is worse. Both events are detected near contact; the legacy
    foot-down and hands-peak anchors are not used."""
    k = a["contact_frame"]
    land, why = _front_foot_lands(c, k)
    if land is None:
        return np.nan, why
    top = _top_hand_peak(c, k)
    if top is None:
        return np.nan, "top hand not seen in the backlift"
    return abs(1000.0 * (top - land) / c.fps), ""


def _head_offset_setup(c, a) -> float:
    """Signed sideways offset of the nose from the midpoint of the two ankles over the set-up window,
    pixels, off side positive (the off side is smaller x after the mirror)."""
    fr = _setup_window(c, a)
    v = c.series(fr, (L["ank"], R["ank"]))[:, 0] - c.series(fr, NOSE)[:, 0]
    with np.errstate(all="ignore"):
        return float(np.nanmedian(v))


def m15_head_over_base_setup(c, a, bl):
    """Head position at set-up (Radford: head over off stump or middle-and-off, eyes inside off stump):
    sideways offset of the nose from the midpoint of the two ankles, median over the set-up window
    (1000 to 700 ms before contact), % BL. Measured signed with the off side positive; the entry tests
    do not allow an unknown sign, so the value stored is the ABSOLUTE offset and higher is worse:
    the head should sit over the base, and a head well outside the feet to either side is the fault."""
    v = _head_offset_setup(c, a)
    return (_pct(abs(v), bl), "") if np.isfinite(v) else (np.nan, "head or feet not seen at set-up (needs 1 s before contact)")


def m16_weight_over_front_foot(c, a, bl):
    """Centre of mass over the front foot at contact: sideways gap between the hip-shoulder midpoint
    (mean of both hips and both shoulders, a trunk proxy for the centre of mass) and the front ankle,
    median over 70 ms either side of contact, % BL (absolute). Sideways only: the forward part is depth
    along the pitch, which this camera cannot see."""
    k = a["contact_frame"]
    fr = range(k - c.ms(70), k + c.ms(70) + 1)
    com = c.series(fr, (L["hip"], R["hip"], L["sh"], R["sh"]))[:, 0]
    v = np.abs(com - c.series(fr, c.front["ank"])[:, 0])
    with np.errstate(all="ignore"):
        v = float(np.nanmedian(v))
    return (_pct(v, bl), "") if np.isfinite(v) else (np.nan, "trunk or front foot not seen at contact")


def m17_trigger_present(c, a, bl):
    """Trigger movement (Taliep et al. 2007: 70% of skilled against 20% of less-skilled batters): the
    largest sideways movement (range of x) of either ankle from 1000 to 450 ms before contact, before
    the stride, % BL. A trigger is a small foot movement before committing forward, so higher is
    better. Mirror-invariant (a range, maximised over both feet)."""
    k = a["contact_frame"]
    fr = range(k - c.ms(1000), k - c.ms(450) + 1)
    best = np.nan
    for ank in (L["ank"], R["ank"]):
        x = c.series(fr, ank)[:, 0]
        x = x[np.isfinite(x)]
        if len(x) >= 4:
            best = np.nanmax([best, float(x.max() - x.min())])
    return (_pct(best, bl), "") if np.isfinite(best) else (np.nan, "feet not seen before the stride (needs 1 s before contact)")


MEASURES: list[tuple[str, str, str, str, str, str, Callable]] = [
    # id, checkpoint, fault, pillar, unit, direction, function
    ("M01", "C01", "F01", "tracking", "% BL", "higher_worse", m01_head_still_setup),
    ("M02", "C02", "F02", "stability", "% BL", "higher_worse", m02_balanced_base),
    ("M03", "C03", "F03", "access", "% BL, off side +", "higher_worse", m03_backlift_straight),
    ("M04", "C04", "F04", "tracking", "% BL, off side +", "higher_better", m04_head_leads),
    ("M05", "C05", "F05", "access", "% BL, drive side +", "higher_better", m05_front_foot_line),
    ("M06", "C06", "F06", "stability", "degrees", "higher_worse", m06_front_knee_bends),
    ("M07", "C07", "F07", "access", "ratio", "higher_worse", m07_stays_side_on),
    ("M08", "C08", "F08", "access", "% BL, away from drive side +", "higher_worse", m08_downswing_straight),
    ("M09", "C09", "F09", "flow", "ms", "higher_worse", m09_one_rhythm),
    ("M10", "C10", "F10", "tracking", "% BL, off side +", "higher_better", m10_head_over_ball),
    ("M11", "C11", "F11", "stability", "% BL", "higher_worse", m11_balance_finish),
    ("M12", "C12", "F12", "flow", "% BL", "higher_better", m12_hands_through),
    ("M13", "C12", "F12", "flow", "ratio", "higher_better", m13_hands_accelerate),
    # v5 (15 Nov 2026): registered before any clip was run; directions fixed then (docs/phase5/README.md)
    ("M14", "C09", "F09", "flow", "ms (absolute)", "higher_worse", m14_foot_downswing_coupled),
    ("M15", "C01", "F01", "tracking", "% BL (absolute)", "higher_worse", m15_head_over_base_setup),
    ("M16", "C16", "F16", "stability", "% BL", "higher_worse", m16_weight_over_front_foot),
    ("M17", "C13", "F13", "flow", "% BL", "higher_better", m17_trigger_present),
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
        given = hand.get(k, "right")
        clip = load_clip(npz, given, drive.get(k, "off"))
        geo, source = given, "clip sheet"
        decide = sets.get("auto", {}).get(k) or sets["validated"].get(k)  # the product only has automatic key frames
        if decide and toes_point_off(clip, decide) == "no":
            other = "left" if given == "right" else "right"
            flipped = load_clip(npz, other, drive.get(k, "off"))
            if toes_point_off(flipped, decide) == "yes":
                clip, geo, source = flipped, other, "feet (toes point the other way: left-handed, mirrored video or camera behind the batter)"
        for set_name, by_clip in sets.items():
            if k not in by_clip:
                continue
            for r in measure_clip(clip, by_clip[k]):
                rows.append({"filename": npz.stem, "measurement_set": MEASUREMENT_SET, "anchor_set": set_name, "handedness": hand.get(k, "right"), "geometry_handedness": geo, "geometry_source": source, "drive_type": clip.drive,
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
