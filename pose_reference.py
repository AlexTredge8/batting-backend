#!/usr/bin/env python3
"""
"Best practice" reference pose for the You-vs-best-practice comparison (pose_compare.py).

For each key moment of the drive (set-up, top of the backlift, front foot down, just before contact,
contact, finish) this builds ONE averaged stick figure from the Elite clips, and stores, for each
Phase 5 measurement (phase5_checkpoints.py, M01-M18), the range that good drives fall in.

How each clip is normalised (the same rules as the Phase 5 measurements):
  - left-handers are mirrored, so every clip reads as a right-hander filmed from the bowler's end
    (the off side is camera-left); front and back sides are swapped with the mirror;
  - every landmark is rotated by minus the camera roll from the background lines
    (capture_geometry.camera_roll_deg, confidence "ok" only), so a phone held off level does not
    lean the figure; the batter's own lean is kept;
  - the origin is the middle of the two ankles at set-up (median over 1000 to 700 ms before
    contact) and the unit is the body length (BL: nose-to-ankle height over the same window), so
    a tall batter close to the camera and a small batter far away give the same figure;
  - x is camera-right and y is down, in body lengths.
Then the per-joint MEDIAN across clips is taken, moment by moment.

The measurement ranges are the 10th to 90th percentile of the Elite and Good Club values in the
Phase 5 measurement table (validated key frames), with the median.

PRIVACY: the JSON written holds only averaged, normalised joint coordinates and value ranges: no
names, no clip names, no frames or images, no per-clip values. With fewer than MIN_CLIPS_ANONYMISED
clips it is NOT an anonymous average (one clip's figure is that batter's figure) and the file says
so ("anonymised": false, "provisional": true). The provisional v1 file in the repo is built from the
bundled reference clip only (test_batting.mov); Alex rebuilds it on his Mac from all Elite clips:

  python pose_reference.py --cache-dir calibration_output/pose_cache --anchors anchor_truth.csv \
      --clip-sheet clip_sheet.csv --measurements baselines/phase5_2026-11-16_v8/checkpoint_measurements.csv \
      [--include-good-club] [--video-dir <clips>] [--out reference/pose_reference_v1.json]

(pose_reference_build_mac.sh runs this and pushes only the JSON.)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import warnings
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import phase5_checkpoints as p5  # noqa: E402
from phase0_baseline_table import key_of, load_csv  # noqa: E402

REFERENCE_VERSION = "pose_reference_v1"
DEFAULT_REFERENCE_PATH = SCRIPT_DIR / "reference" / "pose_reference_v1.json"
DEFAULT_MEASUREMENTS = SCRIPT_DIR / "baselines" / "phase5_2026-11-16_v8" / "checkpoint_measurements.csv"
MIN_CLIPS_ANONYMISED = 3   # below this the "average" is too close to one batter's own figure
RANGE_PERCENTILES = (10, 90)

# Joints of the stick figure, by side of the body (front = the side nearer the bowler: a
# right-hander's left). Index pairs are MediaPipe Pose landmark numbers (left side, right side).
_SIDE_INDEX = {"shoulder": (11, 12), "elbow": (13, 14), "wrist": (15, 16), "hip": (23, 24),
               "knee": (25, 26), "ankle": (27, 28), "heel": (29, 30), "toe": (31, 32)}
JOINTS = ["nose"] + [f"{s}_{p}" for p in _SIDE_INDEX for s in ("front", "back")] + ["hands"]
JOINT_LABELS = {
    "nose": "head", "hands": "hands",
    **{f"{s}_{p}": f"{s} {p}" for p in _SIDE_INDEX for s in ("front", "back")},
}
BONES = [
    ("front_shoulder", "back_shoulder"), ("front_hip", "back_hip"),
    ("front_shoulder", "front_hip"), ("back_shoulder", "back_hip"),
    ("front_shoulder", "front_elbow"), ("front_elbow", "front_wrist"),
    ("back_shoulder", "back_elbow"), ("back_elbow", "back_wrist"),
    ("front_hip", "front_knee"), ("front_knee", "front_ankle"),
    ("back_hip", "back_knee"), ("back_knee", "back_ankle"),
    ("front_ankle", "front_heel"), ("front_heel", "front_toe"), ("front_ankle", "front_toe"),
    ("back_ankle", "back_heel"), ("back_heel", "back_toe"), ("back_ankle", "back_toe"),
    ("nose", "neck"),  # neck = middle of the shoulders (drawn, not stored)
]

# Key moments, all anchored on contact where possible (automatic contact is the most accurate key
# frame: out by a median 1 frame; automatic set-up by 6 and follow-through by 8).
MOMENTS = {
    "setup": {"label": "Set-up",
              "definition": "median over 1000 to 700 ms before contact (the Phase 5 set-up window); "
                            "the frame shown is 850 ms before contact"},
    "top_of_backlift": {"label": "Top of the backlift",
                        "definition": "the top hand's highest point 450 to 100 ms before contact "
                                      "(as Phase 5 M03); the hands-peak key frame if the hand is not seen"},
    "front_foot_down": {"label": "Front foot down", "definition": "the front-foot-down key frame"},
    "before_contact": {"label": "Just before contact", "definition": "100 ms before contact (as Phase 5 M07)"},
    "contact": {"label": "Contact", "definition": "the contact key frame"},
    "follow_through": {"label": "Finish",
                       "definition": "550 ms after contact (inside the Phase 5 M11 window, 400 to 700 ms); "
                                     "the last frame with a pose if the clip ends sooner"},
}

MEASUREMENT_NAMES = {
    "M01": "Head movement at set-up",
    "M02": "Hips over the middle of the feet at set-up",
    "M03": "Hands out towards gully in the backlift",
    "M04": "Head ahead of the front foot at foot-down",
    "M05": "Front foot step towards the line",
    "M06": "Front-knee angle around contact",
    "M07": "Shoulder line opening before contact",
    "M08": "Hands across the line through contact",
    "M09": "Hands pause before contact",
    "M10": "Head over the ball at contact",
    "M11": "Hips over the front foot at the finish",
    "M12": "Hands above the shoulders after contact",
    "M13": "Swing carries through contact",
    "M14": "Front foot landing against the downswing",
    "M15": "Head over the base at set-up",
    "M16": "Weight over the front foot at contact",
    "M17": "Trigger movement before the stride",
    "M18": "Head, hands and feet in line at set-up",
}


# --- normalisation ----------------------------------------------------------------------------

def joint_index(c: p5.Clip, name: str) -> int | None:
    """MediaPipe index of a named joint in a (mirrored) Phase 5 clip; None for 'hands' (c.hand)."""
    if name == "nose":
        return p5.NOSE
    if name == "hands":
        return int(c.hand)
    side, part = name.split("_", 1)
    left_front = c.front["ank"] == p5.L["ank"]
    li, ri = _SIDE_INDEX[part]
    if side == "front":
        return li if left_front else ri
    return ri if left_front else li


def prepare_clip(c: p5.Clip, anchors: dict) -> tuple[p5.Clip, float | None, np.ndarray | None, float]:
    """Level the clip by its camera roll (Phase 5 v8) and find the normalisation: the set-up ankle
    midpoint (origin, pixels in the levelled, mirrored picture) and the body length (pixels).
    Returns (levelled clip, roll removed or None, origin or None, body length or NaN)."""
    a = {k: int(v) for k, v in anchors.items()}
    c, roll = p5.level_by_camera_roll(c)
    bl = p5.body_length(c, a)
    fr = p5._setup_window(c, a)
    ank = c.series(fr, (p5.L["ank"], p5.R["ank"]))
    ok = ank[np.isfinite(ank).all(axis=1)]
    origin = np.median(ok, axis=0) if len(ok) else None
    return c, roll, origin, bl


def moment_frame(c: p5.Clip, anchors: dict, moment: str) -> int | None:
    """The video frame (original index) for a key moment, from contact where possible."""
    a = {k: int(v) for k, v in anchors.items()}
    k = a["contact_frame"]
    if moment == "setup":
        f = k - c.ms(850)
    elif moment == "top_of_backlift":
        f = p5._top_hand_peak(c, k)
        if f is None:
            f = a.get("hands_peak_frame")
    elif moment == "front_foot_down":
        f = a.get("front_foot_down_frame")
    elif moment == "before_contact":
        f = k - c.ms(100)
    elif moment == "contact":
        f = k
    elif moment == "follow_through":
        f = k + c.ms(550)
        last = max(c.pos) if c.pos else f
        posed = [fr for fr in c.pos if np.isfinite(c.xy[c.pos[fr], p5.NOSE]).all()]
        if f > last and posed:
            f = max(posed)
    else:
        raise KeyError(f"unknown moment {moment!r}")
    return None if f is None else int(f)


def _point(c: p5.Clip, frame: int, idx: int) -> np.ndarray:
    p = c.pt(frame, idx)
    if np.isfinite(p).all():
        return p
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(c.series(range(frame - 1, frame + 2), idx), axis=0)


def normalised_pose(c: p5.Clip, anchors: dict, moment: str, origin: np.ndarray, bl: float) -> tuple[int | None, dict]:
    """(frame, {joint: [x, y] in body lengths from the set-up ankle midpoint}) for one moment of a
    levelled clip. Joints not seen are left out. The set-up is a median over the set-up window."""
    f = moment_frame(c, anchors, moment)
    if f is None or origin is None or not np.isfinite(bl) or bl <= 0:
        return f, {}
    out = {}
    for name in JOINTS:
        idx = joint_index(c, name)
        if moment == "setup":
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                p = np.nanmedian(c.series(p5._setup_window(c, {k: int(v) for k, v in anchors.items()}), idx), axis=0)
        else:
            p = _point(c, f, idx)
        if np.isfinite(p).all():
            out[name] = [float((p[0] - origin[0]) / bl), float((p[1] - origin[1]) / bl)]
    return f, out


def clip_poses(c: p5.Clip, anchors: dict) -> dict[str, dict]:
    """Normalised pose of one clip at every moment: {moment: {joint: [x, y]}}."""
    lc, _roll, origin, bl = prepare_clip(c, anchors)
    return {m: normalised_pose(lc, anchors, m, origin, bl)[1] for m in MOMENTS}


def median_pose(poses: list[dict]) -> tuple[dict, int]:
    """Per-joint median of several normalised poses; joints seen in no clip are left out."""
    out = {}
    for name in JOINTS:
        pts = np.array([p[name] for p in poses if name in p])
        if len(pts):
            m = np.median(pts, axis=0)
            out[name] = [round(float(m[0]), 4), round(float(m[1]), 4)]
    return out, len([p for p in poses if p])


# --- measurement ranges ---------------------------------------------------------------------

def measurement_ranges(table: Path, clip_sheet: Path, tiers=("Elite", "Good Club"), anchor_set: str = "validated",
                       percentiles=RANGE_PERCENTILES) -> tuple[dict, dict]:
    """{measurement: {name, unit, direction, low, high, median, n}} from the Phase 5 measurement
    table, Elite and Good Club clips (by clip_sheet tier_csv), one anchor set. Only the summary is
    kept: no per-clip value leaves this function. Also returns facts about the table."""
    tier = {key_of(r["filename"]): r.get("tier_csv", "") for r in load_csv(clip_sheet)}
    vals: dict[str, list[float]] = {}
    meta: dict[str, dict] = {}
    sets = set()
    clips = set()
    for r in load_csv(table):
        if r.get("anchor_set") != anchor_set or tier.get(key_of(r["filename"])) not in tiers:
            continue
        sets.add(r.get("measurement_set", ""))
        m = r["measurement"]
        meta.setdefault(m, {"unit": r.get("unit", ""), "direction": r.get("direction", "")})
        if r.get("value", "") not in ("", None):
            vals.setdefault(m, []).append(float(r["value"]))
            clips.add(key_of(r["filename"]))
    out = {}
    for m in sorted(meta):
        v = np.array(vals.get(m, []))
        if not len(v):
            continue
        lo, hi = np.percentile(v, percentiles)
        out[m] = {"name": MEASUREMENT_NAMES.get(m, m), "unit": meta[m]["unit"], "direction": meta[m]["direction"],
                  "low": round(float(lo), 3), "high": round(float(hi), 3), "median": round(float(np.median(v)), 3),
                  "n": int(len(v))}
    return out, {"measurement_set": ",".join(sorted(s for s in sets if s)), "n_clips": len(clips)}


# --- build / load ---------------------------------------------------------------------------

def geometry_clip(npz: Path, handedness: str, drive: str, roll: dict | None, anchors: dict) -> p5.Clip:
    """Load a cached clip as Phase 5 does, including the feet check: if the toes point the wrong
    way for the clip-sheet handedness and the right way when flipped, the flipped geometry is used."""
    clip = p5.load_clip(npz, handedness, drive, roll)
    if p5.toes_point_off(clip, anchors) == "no":
        other = "left" if handedness == "right" else "right"
        flipped = p5.load_clip(npz, other, drive, roll)
        if p5.toes_point_off(flipped, anchors) == "yes":
            return flipped
    return clip


def build_reference(per_clip_poses: list[dict], ranges: dict, sources: dict) -> dict:
    """The reference JSON from each clip's normalised poses (clip_poses) and the ranges. Nothing
    that identifies a clip is written."""
    n = len(per_clip_poses)
    anonymised = n >= MIN_CLIPS_ANONYMISED
    moments = {}
    for m, info in MOMENTS.items():
        joints, used = median_pose([p.get(m, {}) for p in per_clip_poses])
        moments[m] = {"label": info["label"], "definition": info["definition"], "n_clips": used, "joints": joints}
    status = (f"Averaged from {n} clips." if anonymised else
              f"PROVISIONAL: built from {n} clip{'s' if n != 1 else ''} only, so the figure is not an anonymous "
              f"average (it needs at least {MIN_CLIPS_ANONYMISED}). Rebuild on all Elite clips with "
              "pose_reference_build_mac.sh before this is used with players.")
    return {
        "version": REFERENCE_VERSION,
        "built": _dt.date.today().isoformat(),
        "provisional": not anonymised,
        "anonymised": anonymised,
        "status": status,
        "privacy": ("Only averaged, normalised joint positions (body lengths from the set-up ankle midpoint) and "
                    "measurement ranges. No names, clip names, frames, images or per-clip values."),
        "coordinates": ("x, y in body lengths (nose-to-ankle height at set-up) from the middle of the two ankles at "
                        "set-up; x camera-right, y down; a right-hander filmed from the bowler's end (left-handers "
                        "mirrored, front/back swapped); camera roll removed"),
        "sources": sources,
        "joints": JOINTS,
        "joint_labels": JOINT_LABELS,
        "bones": [list(b) for b in BONES],
        "moments": moments,
        "measurement_ranges": ranges,
        "range_rule": f"{RANGE_PERCENTILES[0]}th to {RANGE_PERCENTILES[1]}th percentile of the clips' values, with the median",
    }


def load_reference(path: Path | str | None = None) -> dict | None:
    """The reference JSON, or None if it is missing or unreadable."""
    p = Path(path) if path else DEFAULT_REFERENCE_PATH
    try:
        ref = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return ref if isinstance(ref, dict) and ref.get("moments") else None


def main(argv: list[str] | None = None) -> int:
    warnings.simplefilter("ignore", RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(SCRIPT_DIR / "calibration_output" / "pose_cache"))
    ap.add_argument("--anchors", default=str(SCRIPT_DIR / "anchor_truth.csv"))
    ap.add_argument("--clip-sheet", default=str(SCRIPT_DIR / "clip_sheet.csv"))
    ap.add_argument("--measurements", default=str(DEFAULT_MEASUREMENTS),
                    help="Phase 5 checkpoint_measurements.csv for the ranges (validated key frames)")
    ap.add_argument("--include-good-club", action="store_true",
                    help="also average the Good Club clips into the figure (default: Elite only)")
    ap.add_argument("--range-tiers", default="Elite,Good Club", help="tiers for the measurement ranges")
    ap.add_argument("--video-dir", default=None, help="the clips, to compute a camera roll not yet cached")
    ap.add_argument("--out", default=str(DEFAULT_REFERENCE_PATH))
    args = ap.parse_args(argv)

    sheet = load_csv(Path(args.clip_sheet))
    tiers = ("Elite", "Good Club") if args.include_good_club else ("Elite",)
    rows = {key_of(r["filename"]): r for r in sheet}
    anchors = p5.anchor_sets(Path(args.anchors), None)["validated"]
    caches = {key_of(p.stem): p for p in Path(args.cache_dir).glob("*.npz")}
    video_dir = Path(args.video_dir).expanduser() if args.video_dir else None
    poses, used, skipped = [], 0, []
    for k, npz in sorted(caches.items()):
        r = rows.get(k)
        if not r or r.get("tier_csv") not in tiers:
            continue
        if k not in anchors:
            skipped.append(f"{npz.stem}: no validated key frames")
            continue
        roll = p5.roll_for_npz(npz, video_dir)
        clip = geometry_clip(npz, r.get("handedness") or "right", r.get("drive_type") or "off", roll, anchors[k])
        p = clip_poses(clip, anchors[k])
        if not p.get("setup"):
            skipped.append(f"{npz.stem}: set-up not in the clip")
            continue
        poses.append(p)
        used += 1
    if not poses:
        print("No clips to build from (check --cache-dir, --anchors and the tiers).")
        return 1
    ranges, table_facts = measurement_ranges(Path(args.measurements), Path(args.clip_sheet),
                                             tuple(t.strip() for t in args.range_tiers.split(",") if t.strip()))
    sources = {"pose_tiers": list(tiers), "pose_n_clips": used, "anchor_set": "validated",
               "range_tiers": [t.strip() for t in args.range_tiers.split(",") if t.strip()],
               "range_n_clips": table_facts["n_clips"], "range_measurement_set": table_facts["measurement_set"],
               "pose_measurement_rules": f"Phase 5 {p5.MEASUREMENT_SET} (mirror, camera-roll levelling, body length)"}
    ref = build_reference(poses, ranges, sources)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref, indent=1) + "\n", encoding="utf-8")
    print(f"{used} clip(s) averaged ({', '.join(tiers)}); ranges from {table_facts['n_clips']} clips -> {out}")
    if ref["provisional"]:
        print(f"  PROVISIONAL: fewer than {MIN_CLIPS_ANONYMISED} clips, so this is not an anonymous average.")
    for s in skipped:
        print("  skipped", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
