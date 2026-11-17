#!/usr/bin/env python3
"""
"You vs best practice": the player's pose at the key moment of their priority fault, next to the
averaged best-practice figure (reference/pose_reference_v1.json, built by pose_reference.py).

For a fault (rubric v0.3, F01-F19, or F00 "nothing major to fix") this:
  1. picks the key moment and the joints that matter (FAULT_TABLE);
  2. normalises the player's pose exactly as the reference was normalised (left-handers mirrored,
     camera roll removed, origin at the set-up ankle midpoint, body length as the unit);
  3. aligns the reference to the player: the player's own set-up ankle midpoint and body length,
     shifted so the fault's alignment point (e.g. the hips for "head over the ball") coincides, then
     taken back into the picture as filmed (un-levelled, un-mirrored), so the ghost sits on the
     player's real frame;
  4. computes the fault's Phase 5 measurement (phase5_checkpoints.py) and compares it with the
     range of Elite and Good Club drives stored in the reference;
  5. renders (a) the player's own frame, face blurred, with their skeleton in a strong colour, the
     reference as a translucent ghost, and a ring and an arrow on each joint that differs, and
     (b) a side-by-side panel of the two figures on a plain background;
  6. returns structured data for the frontend (schema in docs/phase6/pose_compare.md).

The live app calls pose_comparison_for_report only when config.POSE_COMPARE is "on" (default off).
It never changes a score or an existing report field.
"""

from __future__ import annotations

import base64
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import capture_geometry
import phase5_checkpoints as p5
import pose_reference as pr

COMPARE_VERSION = "pose_compare_v1"
SCRIPT_DIR = Path(__file__).resolve().parent
TIPS_PATH = SCRIPT_DIR / "docs" / "phase6" / "fault_tips_v0_3.json"
RUBRIC_PATH = SCRIPT_DIR / "docs" / "phase1" / "rubric_v0_3.json"

# A joint is marked as different when it is further than this from the ghost (body lengths).
DIFF_THRESHOLD_BL = 0.05
MAX_HIGHLIGHTS = 3

# Live rules (scorer.py priority fix) -> rubric v0.3 fault. STOPGAP until the live app names rubric
# faults itself: each live rule is mapped to the nearest fault by what it measures. The four active
# rules (T2, A5, S4, F4) are the ones agreed for the stopgap; the dormant ones follow the frontend's
# table (docs/lovable/results_tips_and_sharing_prompt.md) in case they are switched back on.
RULE_TO_FAULT = {"T2": "F01", "A5": "F07", "S4": "F12", "F4": "F09"}
DORMANT_RULE_TO_FAULT = {"T1": "F10", "A1": "F08", "A2": "F10", "A3": "F08", "A4": "F10", "S1": "F16", "S2": "F11",
                         "S3": "F11", "F1": "F13", "F2": "F09", "F3": "F13", "F5": "F09", "F6": "F12"}
PILLAR_ORDER = ["stability", "tracking", "access", "flow"]  # the backend's own tie-break order

MOMENT_LEADS = {
    "setup": "At set-up", "top_of_backlift": "At the top of the backlift", "front_foot_down": "At front foot down",
    "before_contact": "Just before contact", "contact": "At contact", "follow_through": "At the finish",
}

# The full fault table. moment: pose_reference.MOMENTS key. joints: what matters (first = main).
# align: what the ghost is lined up on at that moment ("setup_ankles" = the normalisation origin only;
# "shoulders" / "hips" = the midpoint; or one joint). measurement: Phase 5 id or None (pose only).
# subject / pos / neg: the words around the value in change_text (neg = when the value is negative).
FAULT_TABLE: dict[str, dict[str, Any]] = {
    "F01": {"moment": "setup", "joints": ["nose"], "align": "setup_ankles", "measurement": "M01",
            "lead": "During the set-up", "subject": "your head moves", "pos": "", "neg": "",
            "action": "Keep your head still and your eyes level as the bowler runs in."},
    "F02": {"moment": "setup", "joints": ["front_hip", "back_hip", "nose"], "align": "setup_ankles", "measurement": "M02",
            "subject": "your hips are", "pos": "off the middle of your feet", "neg": "off the middle of your feet",
            "action": "Stand with your weight even between your feet and your knees soft."},
    "F17": {"moment": "setup", "joints": ["nose", "hands", "front_ankle", "back_ankle"], "align": "setup_ankles",
            "measurement": "M18", "subject": "your head, hands and feet are spread", "pos": "apart sideways",
            "neg": "apart sideways", "action": "As the bowler lets go, line up your head, hands and feet over off stump."},
    "F13": {"moment": "setup", "joints": ["back_ankle", "front_ankle", "hands"], "align": "setup_ankles",
            "measurement": "M17", "lead": "Before your stride", "subject": "your feet move", "pos": "", "neg": "",
            "action": "Add a small trigger movement and start lifting the bat a little earlier."},
    "F03": {"moment": "top_of_backlift", "joints": ["hands", "front_elbow", "back_elbow"], "align": "shoulders",
            "measurement": "M03", "subject": "your hands have moved", "pos": "out towards gully",
            "neg": "in towards your body", "action": "Lift your hands up and back, close to your body."},
    "F19": {"moment": None, "joints": [], "align": None, "measurement": None,
            "unavailable": "This is about choosing which ball to drive, not about body position, so there is no pose to compare."},
    "F15": {"moment": "front_foot_down", "joints": ["front_ankle", "nose", "front_knee"], "align": "back_ankle",
            "measurement": None, "action": "Move forward towards the ball rather than across the crease."},
    "F04": {"moment": "front_foot_down", "joints": ["nose", "front_shoulder"], "align": "front_ankle", "measurement": "M04",
            "subject": "your head is", "pos": "ahead of your front foot, towards the off side",
            "neg": "behind your front foot, on the leg side",
            "range_pos": "ahead of your front foot", "range_neg": "behind it",
            "action": "Let your head and front shoulder lead you towards the ball."},
    "F05": {"moment": "front_foot_down", "joints": ["front_ankle", "nose", "front_hip", "back_hip"], "align": "back_ankle",
            "measurement": "M05", "subject": "your front foot has stepped", "pos": "towards the off side",
            "neg": "towards the leg side", "action": "Step towards the ball and land your front foot on its line."},
    "F06": {"moment": "contact", "joints": ["front_knee", "front_hip", "front_ankle"], "align": "front_ankle",
            "measurement": "M06", "lead": "Around contact", "subject": "your front knee is at", "pos": "", "neg": "",
            "action": "Bend your front knee as you land so that it takes your weight."},
    "F18": {"moment": "contact", "joints": ["back_knee", "back_ankle", "back_hip"], "align": "front_ankle",
            "measurement": None, "action": "Push off your back foot as you step forward, so your weight drives into the shot."},
    "F16": {"moment": "contact", "joints": ["front_hip", "back_hip", "front_shoulder", "back_shoulder"],
            "align": "front_ankle", "measurement": "M16", "subject": "the middle of your body is",
            "pos": "to the side of your front foot", "neg": "to the side of your front foot",
            "action": "Get your head and chest over your front foot as you hit the ball."},
    "F07": {"moment": "before_contact", "joints": ["front_shoulder", "back_shoulder", "front_hip", "back_hip"],
            "align": "hips", "measurement": "M07", "subject": "your shoulder line is",
            "pos": "as wide as at set-up (wider means more open)", "neg": "",
            "action": "Keep your front shoulder pointing down the line until you have hit the ball."},
    "F08": {"moment": "contact", "joints": ["hands", "front_elbow", "back_elbow"], "align": "shoulders",
            "measurement": "M08", "lead": "Through contact", "subject": "your hands travel",
            "pos": "across the line towards the leg side", "neg": "out along the line of the drive",
            "range_pos": "across the line", "range_neg": "out along the line",
            "action": "Bring the bat down straight along the line of the ball."},
    "F09": {"moment": "top_of_backlift", "joints": ["hands", "front_elbow"], "align": "shoulders", "measurement": "M09",
            "lead": "In the swing", "subject": "your hands are almost still for", "pos": "", "neg": "",
            "action": "Make it one smooth movement from backlift to contact."},
    "F10": {"moment": "contact", "joints": ["nose", "front_shoulder"], "align": "hips", "measurement": "M10",
            "subject": "your head is", "pos": "to the off side of your hips",
            "neg": "on the leg side of your hips, falling away",
            "range_pos": "to the off side of your hips", "range_neg": "on the leg side of your hips",
            "action": "Keep your head over the ball and meet it close to your front pad."},
    "F11": {"moment": "follow_through", "joints": ["front_hip", "back_hip", "nose"], "align": "front_ankle",
            "measurement": "M11", "subject": "your hips are", "pos": "to the side of your front foot",
            "neg": "to the side of your front foot", "action": "Hold your finish with your head over your front knee."},
    "F12": {"moment": "follow_through", "joints": ["hands", "front_elbow", "back_elbow"], "align": "shoulders",
            "measurement": "M12", "lead": "After contact", "subject": "your hands reach", "pos": "above your shoulders",
            "neg": "below your shoulders at their highest",
            "action": "Let your hands keep going towards your target after you hit the ball."},
    "F00": {"moment": "contact", "joints": [], "align": "hips", "measurement": None, "nothing_to_fix": True},
}
# F14 was retired in rubric v0.3 and is now part of F07: compared the same way.
FAULT_TABLE["F14"] = dict(FAULT_TABLE["F07"], retired_into="F07")


# --- small helpers --------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def fault_texts() -> dict[str, str]:
    r = _load_json(RUBRIC_PATH)
    out = {f["id"]: f["text"] for f in r.get("faults_in_chain_order", [])}
    out.update({f["id"]: f["text"] for f in r.get("faults_retired", [])})
    out.setdefault("F00", "Nothing major to fix")
    return out


def tips() -> dict[str, str]:
    return dict(_load_json(TIPS_PATH).get("tips", {}))


def _unit_kind(unit: str) -> str:
    u = (unit or "").lower()
    if u.startswith("% bl"):
        return "pct"
    if u.startswith("degrees"):
        return "deg"
    if u.startswith("ratio"):
        return "ratio"
    if u.startswith("ms"):
        return "ms"
    return "other"


def _signed_unit(unit: str) -> bool:
    return "+" in (unit or "")


def _fmt_number(v: float, kind: str) -> str:
    if kind == "pct":
        return f"{abs(v):.0f}%"
    if kind == "deg":
        return f"{v:.0f} degrees"
    if kind == "ratio":
        return f"{v:.1f} times"
    if kind == "ms":
        return f"{v:.0f} ms"
    return f"{v:.1f}"


def value_phrase(v: float, unit: str, pos: str = "", neg: str = "") -> str:
    """'15% of your height ahead of your front foot', '1.4 times as wide...', '160 degrees'."""
    kind = _unit_kind(unit)
    words = (pos if v >= 0 else neg) if _signed_unit(unit) else pos
    base = _fmt_number(v, kind) + (" of your height" if kind == "pct" else "")
    return f"{base} {words}".strip()


def range_phrase(lo: float, hi: float, unit: str, direction: str, pos: str = "", neg: str = "") -> str:
    """How good drives read on this measurement, e.g. 'within 4%', 'at least 16% ...', or a band."""
    kind = _unit_kind(unit)
    if _signed_unit(unit):
        if (lo >= 0) == (hi >= 0):  # both ends on the same side: one set of words
            words = pos if hi >= 0 else neg
            a, b = sorted([abs(lo), abs(hi)])
            nums = f"{a:.0f}% and {b:.0f}% of your height" if kind == "pct" else f"{_fmt_number(a, kind)} and {_fmt_number(b, kind)}"
            return f"between {nums} {words}".strip()
        if kind == "pct":
            return f"between {abs(lo):.0f}% {neg} and {abs(hi):.0f}% of your height {pos}"
        return f"between {value_phrase(lo, unit, pos, neg)} and {value_phrase(hi, unit, pos, neg)}"
    if direction == "higher_worse":
        if kind == "pct":
            return f"within {abs(hi):.0f}%"
        if kind == "deg":
            return f"at {hi:.0f} degrees or less (a smaller angle is a deeper bend)"
        return f"at {_fmt_number(hi, kind)} or less"
    if kind == "pct":
        return f"at least {abs(lo):.0f}%"
    return f"at least {_fmt_number(lo, kind)}"


def measurement_status(v: float, lo: float, hi: float, direction: str) -> tuple[str, str]:
    """('outside' | 'inside', 'below' | 'within' | 'above'). Only the worse side is outside: a value
    better than the whole good range is still inside."""
    position = "below" if v < lo else ("above" if v > hi else "within")
    worse = (direction == "higher_worse" and v > hi) or (direction == "higher_better" and v < lo)
    return ("outside" if worse else "inside"), position


def change_text(fault: str, measurement: dict | None, tip: str = "") -> str:
    """One plain sentence (two at most) from the fault's template and the measurement."""
    spec = FAULT_TABLE[fault]
    if spec.get("nothing_to_fix"):
        return tip or "Nothing major to fix: this is a good, balanced drive, so keep repeating it."
    lead = spec.get("lead") or MOMENT_LEADS.get(spec.get("moment"), "")
    action = spec.get("action") or tip
    if not measurement or measurement.get("value") is None or not measurement.get("reference_range"):
        joints = [pr.JOINT_LABELS.get(j, j) for j in spec.get("joints", [])][:2]
        what = " and ".join(joints) if joints else "position"
        return f"{lead}, compare your {what} with the best-practice figure. {action}".strip()
    v = float(measurement["value"])
    lo, hi = measurement["reference_range"]
    unit, direction = measurement.get("unit", ""), measurement.get("direction", "")
    pos, neg = spec.get("pos", ""), spec.get("neg", "")
    said = f"{lead} {spec['subject']} {value_phrase(v, unit, pos, neg)}"
    good = range_phrase(float(lo), float(hi), unit, direction, spec.get("range_pos", pos), spec.get("range_neg", neg))
    if measurement.get("status") == "outside":
        return f"{said}; good drives are {good}. {action}"
    return f"{said}; good drives are {good}, so on this measure you are already in the good range."


# --- geometry -------------------------------------------------------------------------------

def to_measure(xy: np.ndarray, w: float, h: float, mirrored: bool, roll: float | None) -> np.ndarray:
    """Pixels as filmed -> the Phase 5 measurement picture (mirrored for a left-hander, then levelled)."""
    p = np.array(xy, dtype=float)
    if mirrored:
        p[..., 0] = w - p[..., 0]
    if roll is not None:
        p = capture_geometry.rotate_points(p, -roll, (w / 2.0, h / 2.0))
    return p


def to_image(xy: np.ndarray, w: float, h: float, mirrored: bool, roll: float | None) -> np.ndarray:
    """The inverse of to_measure: measurement picture -> pixels as filmed."""
    p = np.array(xy, dtype=float)
    if roll is not None:
        p = capture_geometry.rotate_points(p, roll, (w / 2.0, h / 2.0))
    if mirrored:
        p[..., 0] = w - p[..., 0]
    return p


def _mid(pose: dict, a: str, b: str) -> np.ndarray | None:
    if a in pose and b in pose:
        return (np.asarray(pose[a], float) + np.asarray(pose[b], float)) / 2.0
    return None


def align_point(pose: dict, align: str | None) -> np.ndarray | None:
    if align in (None, "setup_ankles"):
        return None
    if align == "shoulders":
        return _mid(pose, "front_shoulder", "back_shoulder")
    if align == "hips":
        return _mid(pose, "front_hip", "back_hip")
    return np.asarray(pose[align], float) if align in pose else None


def align_reference(ref_pose: dict, player_pose: dict, align: str | None) -> tuple[dict, list[float]]:
    """Shift the reference figure (body lengths) so its alignment point lands on the player's.
    Both poses are in normalised units (the player's own origin and body length). Returns the
    shifted reference and the shift applied."""
    pa, ra = align_point(player_pose, align), align_point(ref_pose, align)
    shift = (pa - ra) if (pa is not None and ra is not None) else np.zeros(2)
    return {k: [float(v[0] + shift[0]), float(v[1] + shift[1])] for k, v in ref_pose.items()}, [float(shift[0]), float(shift[1])]


def denormalise(pose: dict, origin: np.ndarray, bl: float) -> dict:
    return {k: np.asarray(v, float) * bl + origin for k, v in pose.items()}


def normalise(points: dict, origin: np.ndarray, bl: float) -> dict:
    return {k: [float((v[0] - origin[0]) / bl), float((v[1] - origin[1]) / bl)] for k, v in points.items()}


# --- the player's side ----------------------------------------------------------------------

def arrays_from_frame_poses(frame_poses) -> tuple[np.ndarray, np.ndarray]:
    """FramePose list -> (landmarks [n, 33, 4], frame_idx [n]) in the pose-cache layout."""
    rows, idx = [], []
    for fp in frame_poses or []:
        arr = np.full((33, 4), np.nan)
        arr[:, 3] = 0.0
        lm = getattr(fp, "landmarks", None)
        if lm:
            for i, p in enumerate(lm[:33]):
                arr[i] = [p.x, p.y, getattr(p, "z", 0.0), getattr(p, "visibility", 1.0)]
        rows.append(arr)
        idx.append(int(fp.frame_idx))
    if not rows:
        return np.zeros((0, 33, 4)), np.zeros(0, dtype=int)
    return np.stack(rows), np.asarray(idx)


def _raw_points(landmarks: np.ndarray, frame_idx: np.ndarray, frame: int, w: float, h: float) -> np.ndarray | None:
    """[33, 2] pixels as filmed for one frame (NaN under the Phase 5 visibility floor), or None."""
    hit = np.where(frame_idx == frame)[0]
    if not len(hit):
        return None
    a = np.asarray(landmarks[hit[0]], float)
    xy = np.stack([a[:, 0] * w, a[:, 1] * h], axis=-1)
    xy[~(a[:, 3] >= p5.MIN_VIS)] = np.nan
    return xy


def _nearest_posed(frame_idx: np.ndarray, landmarks: np.ndarray, frame: int, max_gap: int = 2) -> int | None:
    seen = {int(f) for f, a in zip(frame_idx, landmarks) if np.isfinite(a[p5.NOSE, 0]) and a[p5.NOSE, 3] >= p5.MIN_VIS}
    for d in range(max_gap + 1):
        for f in (frame - d, frame + d):
            if f in seen:
                return f
    return None


def player_pose_at(c: p5.Clip, raw_meas: np.ndarray | None, frame: int) -> dict:
    """Player joints at a frame in the measurement picture (pixels): the raw landmarks of that frame
    (what is drawn on the picture), each falling back to the smoothed clip where the raw one is missing."""
    out = {}
    for name in pr.JOINTS:
        idx = pr.joint_index(c, name)
        p = raw_meas[idx] if raw_meas is not None else np.array([np.nan, np.nan])
        if not np.isfinite(p).all():
            p = c.pt(frame, idx)
        if np.isfinite(p).all():
            out[name] = np.asarray(p, float)
    return out


def compute_measurement(c: p5.Clip, anchors: dict, mid: str, bl: float) -> tuple[float | None, str]:
    fn = {m[0]: m[6] for m in p5.MEASURES}.get(mid)
    if fn is None:
        return None, "unknown measurement"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            v, why = fn(c, {k: int(x) for k, x in anchors.items()}, bl)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            return None, f"error: {type(exc).__name__}"
    return (float(v), "") if np.isfinite(v) else (None, why)


# --- rendering ------------------------------------------------------------------------------

C_PLAYER = (0, 140, 255)      # strong orange (BGR)
C_GHOST = (255, 255, 255)     # white, drawn translucent
C_RING = (0, 230, 255)        # yellow
C_TARGET = (90, 220, 90)      # green
C_REF_PANEL = (150, 130, 40)  # teal-blue for the reference figure on the plain panel
C_PANEL_RING = (40, 40, 210)  # red rings on the plain panel (yellow does not show on a light background)
OUT_WIDTH = 480
JPEG_QUALITY = 80


def _bones_px(pose: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    pts = dict(pose)
    neck = _mid(pts, "front_shoulder", "back_shoulder")
    if neck is not None:
        pts["neck"] = neck
    out = []
    for a, b in pr.BONES:
        if a in pts and b in pts:
            pa, pb = np.asarray(pts[a], float), np.asarray(pts[b], float)
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                out.append((pa, pb))
    return out


def _draw_skeleton(img, pose: dict, colour, thickness: int, joint_r: int, head_r: int = 0) -> None:
    import cv2
    for pa, pb in _bones_px(pose):
        cv2.line(img, tuple(int(round(v)) for v in pa), tuple(int(round(v)) for v in pb), colour, thickness, cv2.LINE_AA)
    for k, p in pose.items():
        if k == "hands" or not np.isfinite(np.asarray(p, float)).all():
            continue
        cv2.circle(img, tuple(int(round(v)) for v in p), joint_r, colour, -1, cv2.LINE_AA)
    if "nose" in pose and head_r > 0:  # the head as a circle round the nose (a stick-figure head, not a marker)
        cv2.circle(img, tuple(int(round(v)) for v in pose["nose"]), head_r, colour, thickness, cv2.LINE_AA)


def _draw_highlights(img, player: dict, ghost: dict, names: list[str], thickness: int, ring_r: int,
                     arrows: bool = True) -> None:
    import cv2
    for n in names:
        if n not in player:
            continue
        p = tuple(int(round(v)) for v in player[n])
        cv2.circle(img, p, ring_r, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.circle(img, p, ring_r, C_RING, thickness, cv2.LINE_AA)
        if arrows and n in ghost:
            g = tuple(int(round(v)) for v in ghost[n])
            cv2.circle(img, g, max(3, ring_r // 2), C_TARGET, -1, cv2.LINE_AA)
            cv2.arrowedLine(img, p, g, (0, 0, 0), thickness + 2, cv2.LINE_AA, tipLength=0.25)
            cv2.arrowedLine(img, p, g, C_RING, thickness, cv2.LINE_AA, tipLength=0.25)


def blur_face(img, raw_px: np.ndarray | None, all_px: np.ndarray | None = None) -> bool:
    """Pixelate and blur an oval over the head, sized from the shoulders and ears (as the offline
    pack does, phase2_offline_pairs.py). raw_px: [33, 2] pixels as filmed (NaN = not seen);
    all_px: the same without the visibility floor, used when the head points are not seen, so a
    face is blurred rather than shown. Returns True if a region was blurred."""
    import cv2
    h, w = img.shape[:2]
    pts = raw_px if raw_px is not None else all_px
    if pts is None:
        return False
    head = [i for i in range(11) if np.isfinite(pts[i]).all()]
    if not head and all_px is not None:
        pts = all_px
        head = [i for i in range(11) if np.isfinite(pts[i]).all()]
    if not head:
        return False
    cx, cy = (pts[0] if np.isfinite(pts[0]).all() else np.mean(pts[head], axis=0))
    sh = np.hypot(*(pts[11] - pts[12])) if np.isfinite(pts[[11, 12]]).all() else 0.0
    ear = np.hypot(*(pts[7] - pts[8])) if np.isfinite(pts[[7, 8]]).all() else 0.0
    r = max(10.0, 0.45 * sh, 0.9 * ear, 0.035 * max(w, h))
    x0, y0 = max(0, int(cx - r)), max(0, int(cy - 1.2 * r))
    x1, y1 = min(w, int(cx + r)), min(h, int(cy + 1.0 * r))
    if x1 <= x0 or y1 <= y0:
        return False
    roi = img[y0:y1, x0:x1]
    small = cv2.resize(roi, (max(1, (x1 - x0) // 10), max(1, (y1 - y0) // 10)), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.ellipse(mask, ((x1 - x0) // 2, (y1 - y0) // 2), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 255, -1)
    roi[mask > 0] = cv2.GaussianBlur(pix, (0, 0), 3)[mask > 0]
    return True


def _header(width: int, left: list[tuple[str, tuple]], right: str):
    import cv2
    bar = np.full((34, width, 3), 22, np.uint8)
    x = 10
    for text, colour in left:
        cv2.rectangle(bar, (x, 12), (x + 14, 24), colour, -1)
        x += 20
        cv2.putText(bar, text, (x, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        x += int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]) + 16
    tw = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.putText(bar, right, (max(x, width - tw - 10), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return bar


def _footer(img, text: str) -> None:
    import cv2
    if text:
        cv2.putText(img, text, (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)


def render_overlay(frame, player_img: dict, ghost_img: dict, highlights: list[str], raw_px, all_px,
                   moment_label: str, footnote: str = "") -> tuple[np.ndarray, bool]:
    """The player's frame (face blurred) with the ghost, the player's skeleton and the highlights,
    cropped around the two figures and scaled to OUT_WIDTH. Returns (image, face_blurred)."""
    import cv2
    img = frame.copy()
    blurred = blur_face(img, raw_px, all_px)
    pts = np.array([p for p in list(player_img.values()) + list(ghost_img.values()) if np.isfinite(p).all()])
    h, w = img.shape[:2]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bh = max(y1 - y0, 1.0)
    pad = 0.18 * bh
    x0, x1 = x0 - pad, x1 + pad
    y0, y1 = y0 - 1.6 * pad, y1 + pad
    want_w = (y1 - y0) * 0.75  # portrait crop, 3:4
    if x1 - x0 < want_w:
        cx = (x0 + x1) / 2
        x0, x1 = cx - want_w / 2, cx + want_w / 2
    x0, y0 = int(max(0, x0)), int(max(0, y0))
    x1, y1 = int(min(w, x1)), int(min(h, y1))
    crop_h = max(1, y1 - y0)
    t = max(2, int(round(crop_h / 160)))
    jr = max(2, t)
    ghost_layer = img.copy()
    _draw_skeleton(ghost_layer, ghost_img, C_GHOST, t + 2, jr + 1)
    img = cv2.addWeighted(ghost_layer, 0.55, img, 0.45, 0)
    _draw_skeleton(img, player_img, C_PLAYER, t, jr)
    _draw_highlights(img, player_img, ghost_img, highlights, t, max(8, int(round(crop_h / 28))))
    crop = img[y0:y1, x0:x1]
    scale = OUT_WIDTH / float(max(1, crop.shape[1]))
    crop = cv2.resize(crop, (OUT_WIDTH, max(1, int(round(crop.shape[0] * scale)))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    _footer(crop, footnote)
    out = np.vstack([_header(OUT_WIDTH, [("You", C_PLAYER), ("Best practice", (200, 200, 200))], moment_label), crop])
    return out, blurred


def render_side_by_side(player_n: dict, ghost_n: dict, highlights: list[str], mirrored: bool,
                        moment_label: str, footnote: str = "") -> np.ndarray:
    """Two stick figures on a plain background, the same scale, as filmed (a left-hander is shown
    as a left-hander). player_n / ghost_n: normalised poses (body lengths, ghost already aligned)."""
    import cv2
    pw, ph = OUT_WIDTH // 2, 360

    def oriented(pose):
        return {k: np.array([-v[0] if mirrored else v[0], v[1]], float) for k, v in pose.items()}
    a, b = oriented(player_n), oriented(ghost_n)
    pts = np.array(list(a.values()) + list(b.values()))
    head = 0.06  # stick-figure head radius, body lengths
    lo, hi = pts.min(axis=0) - head, pts.max(axis=0) + head
    span = np.maximum(hi - lo, 1e-3)
    top, bottom, side = 40, 26, 20  # room for the labels above and the footnote below
    s = min((pw - 2 * side) / span[0], (ph - top - bottom) / span[1])
    centre = (lo + hi) / 2

    def to_px(pose, ox):
        return {k: np.array([ox + pw / 2 + (v[0] - centre[0]) * s, top + (ph - top - bottom) / 2 + (v[1] - centre[1]) * s])
                for k, v in pose.items()}
    pa, pb = to_px(a, 0), to_px(b, pw)
    canvas = np.full((ph, OUT_WIDTH, 3), 244, np.uint8)
    cv2.line(canvas, (pw, 10), (pw, ph - 10), (210, 210, 210), 1)
    head_r = max(6, int(round(head * s)))
    _draw_skeleton(canvas, pa, C_PLAYER, 3, 3, head_r)
    _draw_skeleton(canvas, pb, C_REF_PANEL, 3, 3, head_r)
    for n in highlights:
        for pose in (pa, pb):
            if n in pose:
                p = tuple(int(round(v)) for v in pose[n])
                cv2.circle(canvas, p, 12, C_PANEL_RING, 2, cv2.LINE_AA)
    for text, x in (("You", 10), ("Best practice", pw + 10)):
        cv2.putText(canvas, text, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    if footnote:
        cv2.putText(canvas, footnote, (8, ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
    return np.vstack([_header(OUT_WIDTH, [("You", C_PLAYER), ("Best practice", C_REF_PANEL)], moment_label), canvas])


def jpeg_data_url(img) -> str | None:
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}" if ok else None


def read_frame(video_path, frame_index: int):
    """One frame, with the phone's rotation applied (as pose_extractor reads it)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


# --- the comparison -------------------------------------------------------------------------

def unavailable(fault: str | None, reason: str, **extra) -> dict:
    return {"version": COMPARE_VERSION, "available": False, "fault": fault, "reason": reason, **extra}


def compute_pose_comparison(landmarks: np.ndarray, frame_idx: np.ndarray, anchors: dict, handedness: str,
                            fps: float, width: float, height: float, fault: str, video_path=None,
                            roll: dict | None = None, drive: str = "off", reference: dict | None = None,
                            frame_image=None, render: bool = True) -> dict:
    """The comparison for one fault. landmarks [n, 33, 4] (normalised x, y, z, visibility) with
    frame_idx [n] original video frames, as the pose cache stores them; anchors: the six key frames
    (original frames); roll: {"roll_deg", "confidence"} as filmed (only "ok" is used). frame_image
    (BGR, as filmed) is used instead of reading video_path when given."""
    fault = (fault or "").upper()
    spec = FAULT_TABLE.get(fault)
    if spec is None:
        return unavailable(fault, f"no comparison is defined for fault {fault!r}")
    texts, tip_texts = fault_texts(), tips()
    base = {"version": COMPARE_VERSION, "fault": fault, "fault_text": texts.get(fault, ""), "tip": tip_texts.get(fault, "")}
    if spec.get("unavailable"):
        return {**unavailable(fault, spec["unavailable"]), **base, "available": False}
    ref = reference if reference is not None else pr.load_reference()
    if not ref:
        return {**unavailable(fault, "the best-practice reference has not been built (reference/pose_reference_v1.json)"), **base,
                "available": False}
    moment = spec["moment"]
    ref_pose = (ref.get("moments", {}).get(moment) or {}).get("joints") or {}
    if not ref_pose:
        return {**unavailable(fault, f"the reference has no figure for {moment}"), **base, "available": False}
    if not width or not height or not fps:
        return {**unavailable(fault, "the video size or frame rate is unknown"), **base, "available": False}
    need = {"contact_frame", "setup_frame", "front_foot_down_frame"}
    if not anchors or any(anchors.get(k) is None for k in need):
        return {**unavailable(fault, "key frames missing"), **base, "available": False}
    a = {k: int(v) for k, v in anchors.items() if v is not None}

    mirrored_hand = (handedness or "right").strip().lower()
    c = p5.clip_from_arrays(landmarks, frame_idx, width, height, fps, mirrored_hand, drive,
                            roll if roll else None)
    lc, roll_removed, origin, bl = pr.prepare_clip(c, a)
    if origin is None or not np.isfinite(bl) or bl <= 0:
        return {**unavailable(fault, "the set-up is not in the clip, so the body size is unknown (needs 1 s before contact)"),
                **base, "available": False}
    want = pr.moment_frame(lc, a, moment)
    frame = _nearest_posed(np.asarray(frame_idx), np.asarray(landmarks), want) if want is not None else None
    if frame is None:
        return {**unavailable(fault, f"no pose at the {pr.MOMENTS[moment]['label'].lower()} frame"), **base, "available": False}

    # player, in the measurement picture
    raw_img = _raw_points(np.asarray(landmarks), np.asarray(frame_idx), frame, width, height)
    raw_meas = to_measure(raw_img, width, height, lc.mirrored, roll_removed) if raw_img is not None else None
    player_meas = player_pose_at(lc, raw_meas, frame)
    player_n = normalise(player_meas, origin, bl)
    ghost_n, shift = align_reference(ref_pose, player_n, spec.get("align"))
    ghost_meas = denormalise(ghost_n, origin, bl)

    # measurement against the good range
    measurement = None
    mid = spec.get("measurement")
    if mid:
        rng = (ref.get("measurement_ranges") or {}).get(mid)
        val, why = compute_measurement(lc, a, mid, bl)
        measurement = {"id": mid, "name": pr.MEASUREMENT_NAMES.get(mid, mid),
                       "value": None if val is None else round(val, 2),
                       "unit": (rng or {}).get("unit") or next((m[4] for m in p5.MEASURES if m[0] == mid), ""),
                       "direction": (rng or {}).get("direction") or next((m[5] for m in p5.MEASURES if m[0] == mid), ""),
                       "reference_range": [rng["low"], rng["high"]] if rng else None,
                       "reference_median": rng.get("median") if rng else None,
                       "reference_n_clips": rng.get("n") if rng else 0}
        if val is None:
            measurement.update(status="unknown", blank_reason=why)
        elif not rng:
            measurement.update(status="unknown", blank_reason="no reference range for this measurement")
        else:
            st, position = measurement_status(val, rng["low"], rng["high"], measurement["direction"])
            measurement.update(status=st, position=position)
        measurement["unit_text"] = "% of your height" if _unit_kind(measurement["unit"]) == "pct" else measurement["unit"].split(",")[0]

    # joints that matter, and which differ
    align = spec.get("align")
    align_names = {"shoulders": {"front_shoulder", "back_shoulder"}, "hips": {"front_hip", "back_hip"}}.get(align, {align})
    joints, highlights = [], []
    for name in spec.get("joints", []):
        if name not in player_n or name not in ghost_n:
            continue
        d = float(np.hypot(*(np.asarray(player_n[name]) - np.asarray(ghost_n[name]))))
        hl = name not in align_names and d > DIFF_THRESHOLD_BL and len(highlights) < MAX_HIGHLIGHTS
        if hl:
            highlights.append(name)
        pi = to_image(player_meas[name], width, height, lc.mirrored, roll_removed)
        gi = to_image(ghost_meas[name], width, height, lc.mirrored, roll_removed)
        joints.append({"name": name, "label": pr.JOINT_LABELS.get(name, name),
                       "player_xy": [round(float(pi[0] / width), 4), round(float(pi[1] / height), 4)],
                       "reference_xy": [round(float(gi[0] / width), 4), round(float(gi[1] / height), 4)],
                       "difference_pct_height": round(100 * d, 1), "highlighted": bool(hl)})

    nothing = bool(spec.get("nothing_to_fix"))
    out = {**base, "available": True, "nothing_to_fix": nothing,
           "moment": moment, "moment_label": pr.MOMENTS[moment]["label"],
           "frame": int(frame), "timestamp_ms": round(1000.0 * frame / float(fps or 30.0), 1),
           "measurement": measurement, "joints": joints,
           "change_text": change_text(fault, measurement, base["tip"]),
           "alignment": {"align_on": align or "setup_ankles", "shift_body_lengths": [round(s, 4) for s in shift],
                         "body_length_px": round(float(bl), 1), "mirrored": bool(lc.mirrored),
                         "camera_roll_removed_deg": None if roll_removed is None else round(float(roll_removed), 2)},
           "reference": {"version": ref.get("version"), "provisional": bool(ref.get("provisional", True)),
                         "anonymised": bool(ref.get("anonymised", False)),
                         "n_clips": (ref.get("moments", {}).get(moment) or {}).get("n_clips"),
                         "status": ref.get("status", "")},
           "images": None}
    if spec.get("retired_into"):
        out["retired_into"] = spec["retired_into"]
    if not render:
        return out

    img = frame_image if frame_image is not None else (read_frame(video_path, frame) if video_path else None)
    footnote = "Reference: provisional" if out["reference"]["provisional"] else ""
    to_px = lambda pose: {k: to_image(v, width, height, lc.mirrored, roll_removed) for k, v in pose.items()}  # noqa: E731
    player_img, ghost_img = to_px(player_meas), to_px(ghost_meas)
    images: dict[str, Any] = {"mime_type": "image/jpeg", "overlay_data_url": None, "side_by_side_data_url": None}
    if img is not None:
        ih, iw = img.shape[:2]
        sx, sy = iw / float(width), ih / float(height)  # the video's own size can differ from the pose size
        scale_pose = lambda pose: {k: np.array([v[0] * sx, v[1] * sy]) for k, v in pose.items()}  # noqa: E731
        all_a = np.asarray(landmarks)[np.where(np.asarray(frame_idx) == frame)[0][0]]
        all_px = np.stack([all_a[:, 0] * iw, all_a[:, 1] * ih], axis=-1)
        raw_px = None if raw_img is None else np.stack([raw_img[:, 0] * sx, raw_img[:, 1] * sy], axis=-1)
        overlay, blurred = render_overlay(img, scale_pose(player_img), scale_pose(ghost_img), highlights,
                                          raw_px, all_px, out["moment_label"], footnote)
        images["overlay_data_url"] = jpeg_data_url(overlay)
        images["overlay_size"] = [int(overlay.shape[1]), int(overlay.shape[0])]
        out["privacy"] = {"face_blurred": bool(blurred),
                          "note": "The player's own frame, face blurred; the reference is an averaged stick figure."}
    else:
        out["privacy"] = {"face_blurred": None, "note": "No video frame was available, so only the stick figures are shown."}
    sbs = render_side_by_side(player_n, ghost_n, highlights, lc.mirrored, out["moment_label"], footnote)
    images["side_by_side_data_url"] = jpeg_data_url(sbs)
    images["side_by_side_size"] = [int(sbs.shape[1]), int(sbs.shape[0])]
    out["images"] = images
    return out


def _map_rule(rule: str) -> str | None:
    return RULE_TO_FAULT.get(rule) or DORMANT_RULE_TO_FAULT.get(rule)


def fault_for_report(report: dict) -> tuple[str | None, str | None, str]:
    """(fault, live rule, how it was chosen), the same steps the results page uses to pick its one
    priority fix (docs/lovable/results_tips_and_sharing_prompt.md, section 3), so the comparison and
    the priority card always name the same fault:
      A. a rubric fault the backend names itself (report["priority_fault"]["id"], when it exists);
      B. the live priority fix's rule, mapped (RULE_TO_FAULT);
      C. no priority fix: the pillar that lost the most points, its largest fault, mapped;
      D. no fault anywhere: F00 "nothing major to fix" (None if no pillar was measured)."""
    pf = report.get("priority_fault") or {}
    fid = str(pf.get("id") or "").upper()
    if fid in FAULT_TABLE:
        return fid, None, "rubric fault named by the backend"
    fix = report.get("priority_fix") or {}
    rule = str(fix.get("rule_id") or "")
    if rule:
        fault = _map_rule(rule)
        if fault is None:
            return None, rule, f"live rule {rule} has no rubric fault mapped"
        return fault, rule, f"stopgap: live rule {rule} mapped to rubric fault {fault}"
    pillars = report.get("pillars") or {}
    best = None
    for name in PILLAR_ORDER:
        p = pillars.get(name) or {}
        faults = p.get("faults") or []
        if p.get("assessed") is False or not faults:
            continue
        lost = sum(float(f.get("deduction") or 0) for f in faults)
        if best is None or lost > best[0]:
            best = (lost, max(faults, key=lambda f: float(f.get("deduction") or 0)))
    if best is not None:
        rule = str(best[1].get("rule_id") or "")
        fault = _map_rule(rule)
        if fault is None:
            return None, rule, f"live rule {rule} has no rubric fault mapped"
        return fault, rule, f"stopgap: no priority fix in the report; largest fault (live rule {rule}) mapped to {fault}"
    measured = [n for n in PILLAR_ORDER if n in pillars and (pillars[n] or {}).get("assessed") is not False]
    if pillars and not measured:
        return None, None, "no part of the shot could be measured"
    return "F00", None, "no fault found: nothing major to fix"


def fault_for_priority_fix(priority_fix: dict | None) -> tuple[str | None, str | None, str]:
    """(fault, live rule, how) from the priority fix alone (see fault_for_report for the full steps)."""
    return fault_for_report({"priority_fix": priority_fix})


def pose_comparison_for_report(report: dict, frame_poses, video_meta: dict, video_path) -> dict:
    """The comparison for the live report's priority fix (run_analysis.analyse, POSE_COMPARE on)."""
    fault, rule, how = fault_for_report(report)
    if fault is None:
        return unavailable(None, how, source_rule=rule)
    md = report.get("metadata") or {}
    anchors = {k: (v or {}).get("original_frame") for k, v in (md.get("anchor_frames") or video_meta.get("anchor_frames") or {}).items()}
    capture = (report.get("analysis_quality") or {}).get("capture") or {}
    roll = None
    if capture.get("roll_confidence") == "ok" and capture.get("camera_roll_deg") is not None:
        roll = {"roll_deg": float(capture["camera_roll_deg"]), "confidence": "ok"}
    lm, idx = arrays_from_frame_poses(frame_poses)
    if not len(idx):
        return unavailable(fault, "no pose landmarks", source_rule=rule, fault_source=how)
    out = compute_pose_comparison(lm, idx, anchors, report.get("handedness") or "right",
                                  float(video_meta.get("fps") or 30.0), float(video_meta.get("width") or 0),
                                  float(video_meta.get("height") or 0), fault, video_path=video_path, roll=roll)
    out["source_rule"] = rule
    out["fault_source"] = how
    return out
