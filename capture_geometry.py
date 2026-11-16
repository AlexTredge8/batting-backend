"""
Camera roll from the background (added 16 Nov 2026).

Why: the batter's standing axis (metrics_calculator.standing_axis_tilt_deg) mixes two things, a phone
held off level and a real sideways lean of the batter. On the reference clip (Viv P_Offdrive Elite)
the camera is level but the batter leans -7 degrees at set-up (his head over off stump, which is
coached), so levelling to his axis (Phase 5 v7) removed real technique and the live tilt warning was a
false alarm. The background does not lean: the long straight edges behind the batter (net poles,
door frames, wall corners, the edges of the lane) are vertical in the world, so their tilt in the
picture is the phone's roll.

How (camera_roll_deg):
  - about 5 frames, evenly spaced over the first 40% of the clip (the bowler's run-up and the
    batter's set-up; the shot itself, and any follow-through camera movement, comes later);
  - the batter is masked out (the pose bounding box padded by 20% on each side) where the pose is
    known, so the batter's own lean cannot leak in; a thin border (2%) is masked too;
  - Canny edges, then cv2.HoughLinesP, each segment refined by a least-squares fit to its edge
    pixels (sub-pixel angle); long lines only (at least 15% of the frame height), within
    20 degrees of vertical;
  - the roll is the length-weighted median angle from vertical. Sign as standing_axis_tilt_deg:
    positive = the top of a vertical line is camera-right of its bottom (the phone rolled so the
    picture turns clockwise, as ffmpeg rotate=+angle does);
  - confidence "ok" when there are at least 6 distinct lines and their spread (length-weighted IQR)
    is at most 3 degrees; otherwise "unknown" and roll_deg is None (raw_roll_deg keeps the value for
    information).
  - Fallback, used only when fewer than 6 distinct vertical lines are found: near-horizontal lines
    (within 20 degrees of horizontal, at least 15% of the frame WIDTH long; crease, floor edges), with
    the same confidence rule. Their angle from horizontal is the roll (same sign). The result says so
    (source = "horizontal_fallback"): perspective tilts horizontal lines that are not square to the
    camera, so this is the weaker estimate.

Nothing here changes a score. Used by phase5_checkpoints (measurement set v8: landmarks are rotated by
the camera roll, only when its confidence is "ok") and by run_analysis._capture_checks (the live
"phone looks tilted" warning).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROLL_VERSION = "roll_v1"
ROLL_N_FRAMES = 5
ROLL_CLIP_SHARE = 0.40       # frames come from the first 40% of the clip
ROLL_MIN_LEN_FRAC = 0.15     # a line must be at least 15% of the frame height (horizontals: width)
ROLL_MAX_DEV_DEG = 20.0      # within 20 degrees of vertical (horizontals: of horizontal)
ROLL_MIN_LINES = 6           # distinct lines needed for confidence "ok"
ROLL_MAX_IQR_DEG = 3.0       # and their spread (IQR) at most this
ROLL_BOX_PAD = 0.20          # the batter's pose box is padded by 20% of its size on each side
ROLL_BORDER_FRAC = 0.02      # edges this close to the frame border are ignored (letterboxing, rotation fill)
MIN_VIS = 0.2                # pose points under this visibility are not used for the batter's box


def roll_frame_indices(n_frames: int, share: float = ROLL_CLIP_SHARE, k: int = ROLL_N_FRAMES) -> list[int]:
    """k frame indices evenly spaced over the first `share` of a clip of n_frames frames."""
    if n_frames <= 0:
        return []
    last = max(0, int(math.floor((n_frames - 1) * share)))
    return sorted({int(round(v)) for v in np.linspace(0, last, k)})


def batter_box(landmarks, width: float, height: float, pad: float = ROLL_BOX_PAD) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) in pixels: the bounding box of the visible pose points, padded by `pad` of its
    width and height on each side and clipped to the frame. landmarks: [33, >=2] normalised (x, y[, z,
    visibility]) or a list of objects with .x, .y (and .visibility). None when no point is usable."""
    if landmarks is None:
        return None
    if hasattr(landmarks, "__len__") and len(landmarks) and hasattr(landmarks[0], "x"):
        arr = np.array([[lm.x, lm.y, getattr(lm, "visibility", 1.0)] for lm in landmarks], dtype=float)
    else:
        arr = np.asarray(landmarks, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        arr = np.column_stack([arr[:, 0], arr[:, 1], arr[:, 3] if arr.shape[1] >= 4 else np.ones(len(arr))])
    ok = np.isfinite(arr[:, :2]).all(axis=1) & ~(arr[:, 2] < MIN_VIS)
    if ok.sum() < 3:
        return None
    x, y = arr[ok, 0] * width, arr[ok, 1] * height
    x0, x1, y0, y1 = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    pw, ph = pad * (x1 - x0), pad * (y1 - y0)
    return (int(max(0, math.floor(x0 - pw))), int(max(0, math.floor(y0 - ph))),
            int(min(width, math.ceil(x1 + pw))), int(min(height, math.ceil(y1 + ph))))


def line_segments(frame: np.ndarray, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Long straight segments [m, 4] (x1, y1, x2, y2) in one frame, the batter's box masked out."""
    import cv2

    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    b = max(1, int(round(ROLL_BORDER_FRAC * min(w, h))))
    edges[:b, :] = 0
    edges[-b:, :] = 0
    edges[:, :b] = 0
    edges[:, -b:] = 0
    if box is not None:
        x0, y0, x1, y1 = box
        edges[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
    min_len = ROLL_MIN_LEN_FRAC * min(h, w)  # the per-direction length rule is applied in roll_from_segments
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=max(20, int(0.5 * min_len)),
                           minLineLength=min_len, maxLineGap=max(3, int(0.01 * h)))
    return np.zeros((0, 4)) if segs is None else _refine(edges, segs.reshape(-1, 4).astype(float))


def _refine(edges: np.ndarray, segs: np.ndarray, tol: float = 1.5) -> np.ndarray:
    """Least-squares fit of each segment to the edge pixels within tol px of it. Hough endpoints sit on
    whole pixels, so a long edge a fraction of a degree off vertical snaps to exactly 0; the fit
    recovers the fraction (on the reference clip it moved the estimate from 0.00 to 0.28 degrees, and
    the +3/-3 degree copies from +3.46/-2.49 to +3.45/-2.61, i.e. +3.17/-2.89 against the original)."""
    import cv2

    ys, xs = np.nonzero(edges)
    pts = np.column_stack([xs, ys]).astype(float)
    out = []
    for x1, y1, x2, y2 in segs:
        d = np.array([x2 - x1, y2 - y1])
        length = float(np.hypot(*d))
        if length == 0:
            continue
        u = d / length
        rel = pts - (x1, y1)
        t, s = rel @ u, rel @ np.array([-u[1], u[0]])
        k = (np.abs(s) <= tol) & (t >= 0) & (t <= length)
        if k.sum() < 10:
            out.append((x1, y1, x2, y2))
            continue
        vx, vy, x0, y0 = cv2.fitLine(pts[k].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        tt = (pts[k] - (x0, y0)) @ (vx, vy)
        out.append((x0 + vx * tt.min(), y0 + vy * tt.min(), x0 + vx * tt.max(), y0 + vy * tt.max()))
    return np.array(out, dtype=float).reshape(-1, 4)


def _weighted_percentile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q / 100.0 * w.sum(), c, v))


def _angles(segs: np.ndarray, vertical: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(angle in degrees, length, position) per segment. Vertical: angle from vertical, oriented upwards
    (positive = top camera-right); position = x at the segment's middle. Horizontal: angle from
    horizontal, oriented rightwards (positive = right end lower, the same roll); position = y."""
    if len(segs) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    if vertical:
        flip = dy > 0
        dx, dy = np.where(flip, -dx, dx), np.where(flip, -dy, dy)
        ang = np.degrees(np.arctan2(dx, -dy))
        pos = (segs[:, 0] + segs[:, 2]) / 2
    else:
        flip = dx < 0
        dx, dy = np.where(flip, -dx, dx), np.where(flip, -dy, dy)
        ang = np.degrees(np.arctan2(dy, dx))
        pos = (segs[:, 1] + segs[:, 3]) / 2
    return ang, length, pos


def _distinct(ang: np.ndarray, pos: np.ndarray, pos_tol: float, ang_tol: float = 1.5) -> int:
    """Number of distinct lines: segments of the same edge (in one frame or across frames, the camera
    being fixed) are counted once when their angle and middle position are close."""
    reps: list[tuple[float, float]] = []
    for a, p in sorted(zip(ang.tolist(), pos.tolist()), key=lambda t: t[1]):
        if not any(abs(a - ra) <= ang_tol and abs(p - rp) <= pos_tol for ra, rp in reps):
            reps.append((a, p))
    return len(reps)


def roll_from_segments(segments: Iterable[np.ndarray], width: float, height: float) -> dict:
    """The camera roll from line segments pooled over frames (see the module notes)."""
    segs = [np.asarray(s, dtype=float).reshape(-1, 4) for s in segments]
    segs = np.concatenate(segs) if segs else np.zeros((0, 4))
    out = {"roll_deg": None, "raw_roll_deg": None, "confidence": "unknown", "n_lines": 0, "n_segments": 0,
           "iqr_deg": None, "source": "none", "version": ROLL_VERSION}
    for vertical in (True, False):
        ang, length, pos = _angles(segs, vertical)
        keep = (np.abs(ang) <= ROLL_MAX_DEV_DEG) & (length >= ROLL_MIN_LEN_FRAC * (height if vertical else width))
        ang, length, pos = ang[keep], length[keep], pos[keep]
        n = _distinct(ang, pos, 0.01 * (width if vertical else height))
        if vertical and n < ROLL_MIN_LINES:
            if n:  # keep the vertical evidence for the record, then try the fallback
                out.update(n_lines=n, n_segments=int(len(ang)), raw_roll_deg=round(_weighted_percentile(ang, length, 50), 2),
                           iqr_deg=round(_weighted_percentile(ang, length, 75) - _weighted_percentile(ang, length, 25), 2),
                           source="vertical")
            continue
        if n == 0:
            break
        med = _weighted_percentile(ang, length, 50)
        iqr = _weighted_percentile(ang, length, 75) - _weighted_percentile(ang, length, 25)
        ok = n >= ROLL_MIN_LINES and iqr <= ROLL_MAX_IQR_DEG
        out.update(raw_roll_deg=round(med, 2), roll_deg=round(med, 2) if ok else None, confidence="ok" if ok else "unknown",
                   n_lines=n, n_segments=int(len(ang)), iqr_deg=round(iqr, 2),
                   source="vertical" if vertical else "horizontal_fallback")
        break
    return out


def _read_frames(video_path: str | Path, indices: Sequence[int] | None) -> tuple[list[int], list[np.ndarray]]:
    """Decode the wanted frames (sequentially, so the index is exact). indices None: the default sample
    over the first 40% of the clip, from the container's frame count."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # the same orientation handling as the pose extractor
        if not cap.isOpened():
            return [], []
        if indices is None:
            indices = roll_frame_indices(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        want = set(int(i) for i in indices)
        got_idx, frames = [], []
        i, last = 0, max(want) if want else -1
        while i <= last:
            if not cap.grab():
                break
            if i in want:
                ok, f = cap.retrieve()
                if ok:
                    got_idx.append(i)
                    frames.append(f)
            i += 1
        return got_idx, frames
    finally:
        cap.release()


def camera_roll_deg(source, frame_indices: Sequence[int] | None = None, boxes: dict | Sequence | None = None,
                    box_frame_size: tuple[float, float] | None = None) -> dict:
    """The phone's roll, in degrees, from the background's long vertical lines (module notes).

    source: a video path (frames are decoded at frame_indices; default: 5 evenly spaced over the first
    40% of the clip), or a sequence of frames (BGR or grey arrays), used as given.
    boxes: the batter's box per frame, to mask out: {frame index: (x0, y0, x1, y1) or None} for a video,
    or a list parallel to the frames. Missing: no mask for that frame.
    box_frame_size: (width, height) the boxes were made for; if the decoded frames differ (e.g. the
    orientation was handled differently), the boxes are not used and "mask_skipped" says so.

    Returns {"roll_deg": float or None (None unless confidence is "ok"), "raw_roll_deg", "confidence":
    "ok" / "unknown", "n_lines" (distinct lines), "n_segments", "iqr_deg", "source": "vertical" /
    "horizontal_fallback" / "none", "frames" (indices used), "masked_frames", "version"}.
    """
    if isinstance(source, (str, Path)):
        idx, frames = _read_frames(source, frame_indices)
    else:
        frames = list(source)
        idx = list(frame_indices) if frame_indices is not None else list(range(len(frames)))

    def box_of(j: int, i: int):
        if boxes is None:
            return None
        if isinstance(boxes, dict):
            return boxes.get(i)
        return boxes[j] if j < len(boxes) else None
    if not frames:
        out = roll_from_segments([], 1, 1)
        out.update(frames=[], masked_frames=0)
        return out
    h, w = frames[0].shape[:2]
    mask_skipped = box_frame_size is not None and (abs(box_frame_size[0] - w) > 1 or abs(box_frame_size[1] - h) > 1)
    if mask_skipped:
        box_of = lambda j, i: None  # noqa: E731
    segs, masked = [], 0
    for j, (i, f) in enumerate(zip(idx, frames)):
        b = box_of(j, i)
        masked += b is not None
        segs.append(line_segments(f, b))
    out = roll_from_segments(segs, w, h)
    out.update(frames=[int(i) for i in idx], masked_frames=int(masked))
    if mask_skipped:
        out["mask_skipped"] = f"frame size {w}x{h} is not the pose's {box_frame_size[0]:.0f}x{box_frame_size[1]:.0f}"
    return out


def rotate_points(xy: np.ndarray, deg: float, pivot) -> np.ndarray:
    """Rotate pixel points [..., 2] by deg degrees about pivot, in image coordinates (y down): a positive
    angle turns the top of a vertical line camera-right, i.e. ADDS a roll of deg. To remove a camera
    roll r, rotate by -r."""
    t = math.radians(deg)
    cs, sn = math.cos(t), math.sin(t)
    p = np.asarray(pivot, dtype=float)
    d = np.asarray(xy, dtype=float) - p
    # a vertical vector (0, -1) (pointing up) must go to (sin t, -cos t): top camera-right for t > 0
    return np.stack([p[0] + d[..., 0] * cs - d[..., 1] * sn, p[1] + d[..., 0] * sn + d[..., 1] * cs], axis=-1)
