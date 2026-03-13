"""
BattingIQ Phase 2 — Metrics Calculator
=========================================
Converts raw per-frame landmark data into coaching-relevant metrics.

Coordinate system (MediaPipe normalised):
  X: 0=left edge → 1=right edge of frame
  Y: 0=top → 1=bottom  (so higher hands = SMALLER Y)
  Z: depth, negative = closer to camera

For a right-handed batter from the bowler's end:
  FRONT side = left  (MediaPipe indices 11,13,15,23,25,27)
  BACK  side = right (MediaPipe indices 12,14,16,24,26,28)
Change FRONT_SIDE in config.py for left-handed batters.
"""

import math
import numpy as np
from models import FramePose, FrameMetrics
from config import FRONT_SIDE, METRICS_SMOOTH_WINDOW

# ---------------------------------------------------------------------------
# MediaPipe BlazePose landmark indices
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Side mapping
if FRONT_SIDE == "left":
    FRONT_SHOULDER = LEFT_SHOULDER
    BACK_SHOULDER  = RIGHT_SHOULDER
    FRONT_ELBOW    = LEFT_ELBOW
    BACK_ELBOW     = RIGHT_ELBOW
    FRONT_WRIST    = LEFT_WRIST
    BACK_WRIST     = RIGHT_WRIST
    FRONT_HIP      = LEFT_HIP
    BACK_HIP       = RIGHT_HIP
    FRONT_KNEE     = LEFT_KNEE
    BACK_KNEE      = RIGHT_KNEE
    FRONT_ANKLE    = LEFT_ANKLE
    BACK_ANKLE     = RIGHT_ANKLE
else:
    FRONT_SHOULDER = RIGHT_SHOULDER
    BACK_SHOULDER  = LEFT_SHOULDER
    FRONT_ELBOW    = RIGHT_ELBOW
    BACK_ELBOW     = LEFT_ELBOW
    FRONT_WRIST    = RIGHT_WRIST
    BACK_WRIST     = LEFT_WRIST
    FRONT_HIP      = RIGHT_HIP
    BACK_HIP       = LEFT_HIP
    FRONT_KNEE     = RIGHT_KNEE
    BACK_KNEE      = LEFT_KNEE
    FRONT_ANKLE    = RIGHT_ANKLE
    BACK_ANKLE     = LEFT_ANKLE


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angle_at_b(a, b, c) -> float:
    """Angle at joint b formed by segments a-b and b-c (degrees)."""
    ax, ay = a.x - b.x, a.y - b.y
    cx, cy = c.x - b.x, c.y - b.y
    dot = ax * cx + ay * cy
    mag = math.sqrt(ax*ax + ay*ay) * math.sqrt(cx*cx + cy*cy)
    if mag == 0:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos_a))


def _shoulder_openness(ls, rs) -> float:
    """
    Shoulder openness angle (degrees).
    0° = perfectly side-on (shoulders stacked vertically in frame).
    90° = fully square-on / open (shoulders level horizontally).
    Computed as atan2(Δx, Δy) from vertical.
    """
    dx = abs(rs.x - ls.x)
    dy = abs(rs.y - ls.y)
    return math.degrees(math.atan2(dx, dy + 1e-9))


def _hip_openness(lh, rh) -> float:
    """Same calculation as shoulder openness but for hips."""
    dx = abs(rh.x - lh.x)
    dy = abs(rh.y - lh.y)
    return math.degrees(math.atan2(dx, dy + 1e-9))


# ---------------------------------------------------------------------------
# Per-frame calculation
# ---------------------------------------------------------------------------

def _calc_frame(fp: FramePose) -> FrameMetrics:
    """Calculate all metrics for a single detected frame."""
    lm = fp.landmarks

    # Convenience accessor
    def L(idx):
        return lm[idx]

    ls = L(LEFT_SHOULDER);  rs = L(RIGHT_SHOULDER)
    le = L(LEFT_ELBOW);     re = L(RIGHT_ELBOW)
    lw = L(LEFT_WRIST);     rw = L(RIGHT_WRIST)
    lh = L(LEFT_HIP);       rh = L(RIGHT_HIP)
    lk = L(LEFT_KNEE);      rk = L(RIGHT_KNEE)
    la = L(LEFT_ANKLE);     ra = L(RIGHT_ANKLE)
    nose = L(NOSE)
    ley = L(LEFT_EYE);      rey = L(RIGHT_EYE)

    front_shoulder = L(FRONT_SHOULDER); back_shoulder = L(BACK_SHOULDER)
    front_elbow    = L(FRONT_ELBOW);    back_elbow    = L(BACK_ELBOW)
    front_wrist    = L(FRONT_WRIST);    back_wrist    = L(BACK_WRIST)
    front_hip      = L(FRONT_HIP);      back_hip      = L(BACK_HIP)
    front_knee     = L(FRONT_KNEE);     back_knee     = L(BACK_KNEE)
    front_ankle    = L(FRONT_ANKLE);    back_ankle    = L(BACK_ANKLE)

    # Wrist positions
    wrist_y   = (lw.y + rw.y) / 2
    wrist_x_l = lw.x
    wrist_x_r = rw.x
    wrist_x_m = (lw.x + rw.x) / 2

    # Midpoints
    shoulder_mid_x = (ls.x + rs.x) / 2
    shoulder_mid_y = (ls.y + rs.y) / 2
    hip_mid_x      = (lh.x + rh.x) / 2
    hip_mid_y      = (lh.y + rh.y) / 2

    # Head
    head_offset = nose.x - hip_mid_x
    eye_tilt    = abs(ley.y - rey.y)

    # Shoulder / hip openness
    sh_open  = _shoulder_openness(ls, rs)
    hip_open = _hip_openness(lh, rh)
    sh_hip_gap = sh_open - hip_open

    # Torso lean (horizontal distance between shoulder mid and hip mid)
    torso_lean = abs(shoulder_mid_x - hip_mid_x)

    # Stance width
    stance_width = abs(la.x - ra.x)

    # Knee angles
    front_knee_ang = _angle_at_b(front_hip, front_knee, front_ankle)
    back_knee_ang  = _angle_at_b(back_hip,  back_knee,  back_ankle)

    # Elbow angles
    front_elbow_ang = _angle_at_b(front_shoulder, front_elbow, front_wrist)
    back_elbow_ang  = _angle_at_b(back_shoulder,  back_elbow,  back_wrist)

    m = FrameMetrics(
        frame_idx=fp.frame_idx,
        timestamp_s=fp.timestamp_s,
        detected=True,
        wrist_height=wrist_y,
        wrist_x_left=wrist_x_l,
        wrist_x_right=wrist_x_r,
        wrist_x_mid=wrist_x_m,
        stance_width=stance_width,
        front_ankle_x=front_ankle.x,
        front_ankle_y=front_ankle.y,
        front_ankle_z=front_ankle.z,
        back_ankle_x=back_ankle.x,
        back_ankle_y=back_ankle.y,
        head_offset=head_offset,
        head_x=nose.x,
        head_y=nose.y,
        eye_tilt=eye_tilt,
        shoulder_openness=sh_open,
        hip_openness=hip_open,
        shoulder_hip_gap=sh_hip_gap,
        shoulder_mid_x=shoulder_mid_x,
        shoulder_mid_y=shoulder_mid_y,
        hip_mid_x=hip_mid_x,
        hip_mid_y=hip_mid_y,
        torso_lean=torso_lean,
        front_knee_angle=front_knee_ang,
        back_knee_angle=back_knee_ang,
        front_elbow_angle=front_elbow_ang,
        back_elbow_angle=back_elbow_ang,
        hip_centre_x=hip_mid_x,
    )
    return m


def _fill_gaps(metrics: list[FrameMetrics]) -> None:
    """
    Forward-fill metrics for frames with no detection using the
    last known values so velocity calculations don't get corrupted.
    """
    last_valid = None
    for m in metrics:
        if m.detected:
            last_valid = m
        elif last_valid is not None:
            # Copy spatial values from last valid frame; keep frame/time
            fi, ts = m.frame_idx, m.timestamp_s
            metrics[m.frame_idx] = FrameMetrics(
                frame_idx=fi,
                timestamp_s=ts,
                detected=False,
                wrist_height=last_valid.wrist_height,
                wrist_x_left=last_valid.wrist_x_left,
                wrist_x_right=last_valid.wrist_x_right,
                wrist_x_mid=last_valid.wrist_x_mid,
                stance_width=last_valid.stance_width,
                front_ankle_x=last_valid.front_ankle_x,
                front_ankle_y=last_valid.front_ankle_y,
                front_ankle_z=last_valid.front_ankle_z,
                back_ankle_x=last_valid.back_ankle_x,
                back_ankle_y=last_valid.back_ankle_y,
                head_offset=last_valid.head_offset,
                head_x=last_valid.head_x,
                head_y=last_valid.head_y,
                eye_tilt=last_valid.eye_tilt,
                shoulder_openness=last_valid.shoulder_openness,
                hip_openness=last_valid.hip_openness,
                shoulder_hip_gap=last_valid.shoulder_hip_gap,
                shoulder_mid_x=last_valid.shoulder_mid_x,
                shoulder_mid_y=last_valid.shoulder_mid_y,
                hip_mid_x=last_valid.hip_mid_x,
                hip_mid_y=last_valid.hip_mid_y,
                torso_lean=last_valid.torso_lean,
                front_knee_angle=last_valid.front_knee_angle,
                back_knee_angle=last_valid.back_knee_angle,
                front_elbow_angle=last_valid.front_elbow_angle,
                back_elbow_angle=last_valid.back_elbow_angle,
                hip_centre_x=last_valid.hip_centre_x,
            )


def _compute_velocities(metrics: list[FrameMetrics], fps: float) -> None:
    """Add frame-to-frame velocity fields (second pass)."""
    for i in range(1, len(metrics)):
        curr = metrics[i]
        prev = metrics[i - 1]

        curr.wrist_velocity_y = curr.wrist_height - prev.wrist_height
        curr.wrist_velocity_x = curr.wrist_x_mid - prev.wrist_x_mid
        curr.wrist_speed = math.sqrt(
            curr.wrist_velocity_y**2 + curr.wrist_velocity_x**2
        )

        curr.front_ankle_vy = curr.front_ankle_y - prev.front_ankle_y
        curr.front_ankle_vz = curr.front_ankle_z - prev.front_ankle_z
        curr.wrist_forward_velocity = curr.front_ankle_vz  # proxy

        curr.head_vx = curr.head_x - prev.head_x
        curr.head_vy = curr.head_y - prev.head_y

        curr.hip_centre_vx = curr.hip_centre_x - prev.hip_centre_x


def _smooth_metric(metrics: list[FrameMetrics], attr: str, window: int) -> None:
    """Apply a simple moving-average smoothing to a named attribute."""
    values = [getattr(m, attr) for m in metrics]
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        smoothed.append(float(np.mean(values[lo:hi])))
    for i, m in enumerate(metrics):
        setattr(m, attr, smoothed[i])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_metrics(frame_poses: list[FramePose], fps: float) -> list[FrameMetrics]:
    """
    Full metrics pipeline:
      1. Calculate per-frame metrics for detected frames
      2. Fill gaps with last-known values
      3. Compute velocities
      4. Smooth key signals
    Returns one FrameMetrics per input frame.
    """
    metrics: list[FrameMetrics] = []
    for fp in frame_poses:
        if fp.detected:
            metrics.append(_calc_frame(fp))
        else:
            metrics.append(FrameMetrics(
                frame_idx=fp.frame_idx,
                timestamp_s=fp.timestamp_s,
                detected=False,
            ))

    _fill_gaps(metrics)
    _compute_velocities(metrics, fps)

    # Smooth key noisy signals
    for attr in (
        "wrist_height",
        "wrist_velocity_y",
        "wrist_speed",
        "shoulder_openness",
        "hip_openness",
        "front_ankle_y",
        "front_ankle_z",
        "stance_width",
        "head_x",
        "head_y",
        "head_offset",
    ):
        _smooth_metric(metrics, attr, METRICS_SMOOTH_WINDOW)

    # Recompute velocities after smoothing for cleaner signals
    _compute_velocities(metrics, fps)

    return metrics


def metrics_to_dict(m: FrameMetrics) -> dict:
    """Serialise a FrameMetrics to a plain dict for JSON output."""
    return {k: round(v, 5) if isinstance(v, float) else v
            for k, v in m.__dict__.items()}
