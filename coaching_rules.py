"""
BattingIQ Phase 2 — Coaching Rules Engine
============================================
21 rules across four pillars: Access, Tracking, Stability, Flow.

Each rule function receives:
  - metrics:   list[FrameMetrics]   full per-frame metrics
  - phases:    PhaseResult          detected phase events
  - baseline:  dict                 reference_baseline.json

Returns a list of Fault objects (empty = no fault detected for that rule).
"""

from __future__ import annotations
import numpy as np
from models import Fault, FrameMetrics, PhaseResult
from config import (
    CONTACT_WINDOW_FRAMES,
    FOLLOW_THROUGH_ANALYSIS_FRAMES,
    # Access
    A1_WRIST_SPREAD_THRESHOLD,
    A2_ELBOW_STRAIGHT_THRESHOLD,
    A3_COMPRESSION_MIN_FRAMES,
    A3_COMPRESSION_GOOD_FRAMES,
    A4_TORSO_LEAN_THRESHOLD,
    A5_SHOULDER_HIP_GAP_THRESHOLD,
    A6_EARLY_OPEN_SHOULDER_FRACTION,
    A6_EARLY_OPEN_HIP_FRACTION,
    # Tracking
    T1_HEAD_OFFSET_THRESHOLD,
    T2_EARLY_HEAD_CHANGE_THRESHOLD,
    T3_HEAD_STILLNESS_VARIANCE,
    T4_EYE_TILT_THRESHOLD,
    T5_SETUP_HEAD_VARIANCE,
    # Stability
    S1_FRONT_KNEE_STRAIGHT_THRESHOLD,
    S1_BACK_KNEE_BENT_THRESHOLD,
    S1_HIP_SHIFT_THRESHOLD,
    S2_POST_CONTACT_HIP_STD,
    S2_POST_CONTACT_HEAD_STD,
    S3_HIP_DRIFT_TOLERANCE,
    S4_POST_CONTACT_ROTATION_FRAMES,
    S4_ROTATION_THRESHOLD,
    # Flow
    F1_SMALL_DESYNC,
    F1_MEDIUM_DESYNC,
    F1_LARGE_DESYNC,
    F2_VELOCITY_DIRECTION_CHANGES_THRESHOLD,
    F3_TIMING_SHORT_THRESHOLD,
    F3_TIMING_LONG_THRESHOLD,
    F4_PAUSE_VELOCITY_THRESHOLD,
    F4_PAUSE_FRAMES_THRESHOLD,
    F5_HITCH_VELOCITY_FRACTION,
    F6_FOLLOWTHROUGH_WRIST_SHOULDER_MARGIN,
    F6_FOLLOWTHROUGH_MIN_FRAMES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contact_window(phases: PhaseResult, n: int) -> list[int]:
    c = phases.contact
    return list(range(max(0, c - CONTACT_WINDOW_FRAMES),
                      min(n, c + CONTACT_WINDOW_FRAMES + 1)))


def _avg(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _std(vals: list[float]) -> float:
    return float(np.std(vals)) if vals else 0.0


def _get(metrics, idx, attr, default=0.0):
    if 0 <= idx < len(metrics):
        return getattr(metrics[idx], attr, default)
    return default


# ---------------------------------------------------------------------------
# STABILITY rules  S1–S4
# ---------------------------------------------------------------------------

def rule_S1(metrics, phases, baseline):
    """S1: Weight not transferring to front foot."""
    n = len(metrics)
    cw = _contact_window(phases, n)
    if not cw:
        return []

    front_knee_avg = _avg([_get(metrics, i, "front_knee_angle") for i in cw])
    back_knee_avg  = _avg([_get(metrics, i, "back_knee_angle") for i in cw])
    hip_x_contact  = _avg([_get(metrics, i, "hip_centre_x") for i in cw])
    hip_x_setup    = baseline["setup"]["hip_centre_x_mean"]

    indicators = 0
    details    = []

    if front_knee_avg > S1_FRONT_KNEE_STRAIGHT_THRESHOLD:
        indicators += 1
        details.append(f"front knee {front_knee_avg:.0f}° (locked, >{S1_FRONT_KNEE_STRAIGHT_THRESHOLD:.0f}°)")

    if back_knee_avg < S1_BACK_KNEE_BENT_THRESHOLD:
        indicators += 1
        details.append(f"back knee {back_knee_avg:.0f}° (still bent)")

    hip_shift = abs(hip_x_contact - hip_x_setup)
    if hip_shift < S1_HIP_SHIFT_THRESHOLD:
        indicators += 1
        details.append(f"hip shift {hip_shift:.3f} (< {S1_HIP_SHIFT_THRESHOLD})")

    if indicators == 0:
        return []

    deduction = {1: 4, 2: 7, 3: 10}[min(indicators, 3)]
    return [Fault(
        rule_id="S1",
        fault="Weight not transferring to front foot",
        deduction=deduction,
        detail="; ".join(details),
        feedback=(
            "Your weight is staying too much on the back foot. Move your body through "
            "the ball so your hips travel with your front side. A bent front knee loaded "
            "with your weight gives you the platform to drive through the ball."
        ),
    )]


def rule_S2(metrics, phases, baseline):
    """S2: Post-contact instability."""
    n = len(metrics)
    start = phases.follow_through_start
    ft_frames = list(range(start, min(start + FOLLOW_THROUGH_ANALYSIS_FRAMES, n)))
    if len(ft_frames) < 4:
        return []

    hip_xs  = [_get(metrics, i, "hip_centre_x") for i in ft_frames]
    head_xs = [_get(metrics, i, "head_x")       for i in ft_frames]

    hip_std  = _std(hip_xs)
    head_std = _std(head_xs)

    if hip_std > S2_POST_CONTACT_HIP_STD or head_std > S2_POST_CONTACT_HEAD_STD:
        return [Fault(
            rule_id="S2",
            fault="Post-contact instability",
            deduction=7,
            detail=f"hip_std={hip_std:.4f}, head_std={head_std:.4f}",
            feedback=(
                "You are losing your balance after contact. The best players are set at "
                "impact and can hold their finish. If you are stumbling or correcting after "
                "the shot, your base was not stable enough at impact."
            ),
        )]
    return []


def rule_S3(metrics, phases, baseline, front_side="left"):
    """S3: Hips drifting outside the base."""
    n = len(metrics)
    fault_frames = []
    for i in range(phases.backlift_start, min(phases.follow_through_start + 5, n)):
        hip_x        = _get(metrics, i, "hip_centre_x")
        front_ankle_x = _get(metrics, i, "front_ankle_x")
        # Hip drifting outside the base = hip moving past front ankle
        # toward the off-side. Direction depends on handedness:
        #   RHB (front_side="left"):  front ankle is screen-left, drift = hip_x < front_ankle_x
        #   LHB (front_side="right"): front ankle is screen-right, drift = hip_x > front_ankle_x
        drift = abs(hip_x - front_ankle_x)
        if front_side == "left":
            drifting = hip_x < front_ankle_x
        else:
            drifting = hip_x > front_ankle_x
        if drift > S3_HIP_DRIFT_TOLERANCE and drifting:
            fault_frames.append(i)

    if len(fault_frames) >= 3:
        return [Fault(
            rule_id="S3",
            fault="Hips drifting outside the base",
            deduction=5,
            detail=f"drift detected at {len(fault_frames)} frames",
            feedback=(
                "Your hips are moving outside your base. Keep your body supported by "
                "your base through the strike."
            ),
        )]
    return []


def rule_S4(metrics, phases, baseline):
    """S4: Post-contact body rotation continues."""
    n = len(metrics)
    c = phases.contact
    check_end = min(c + S4_POST_CONTACT_ROTATION_FRAMES, n - 1)

    if c >= n - 2 or check_end >= n:
        return []

    sh_at_contact  = _get(metrics, c, "shoulder_openness")
    hip_at_contact = _get(metrics, c, "hip_openness")
    sh_post        = _get(metrics, check_end, "shoulder_openness")
    hip_post       = _get(metrics, check_end, "hip_openness")

    # Only flag if openness INCREASES after contact (body still rotating forward).
    # Closing/wrapping in the follow-through is acceptable and natural.
    sh_change  = sh_post  - sh_at_contact   # positive = opening more
    hip_change = hip_post - hip_at_contact

    if sh_change > S4_ROTATION_THRESHOLD or hip_change > S4_ROTATION_THRESHOLD:
        return [Fault(
            rule_id="S4",
            fault="Post-contact body rotation",
            deduction=5,
            detail=f"shoulder_change={sh_change:.1f}°, hip_change={hip_change:.1f}°",
            feedback=(
                "Your body is still rotating after contact. The best players are set by "
                "impact and then let the hands continue through the ball."
            ),
        )]
    return []


# ---------------------------------------------------------------------------
# TRACKING rules  T1–T5
# ---------------------------------------------------------------------------

def rule_T1(metrics, phases, baseline):
    """T1: Head falling outside the line at contact."""
    n = len(metrics)
    cw = _contact_window(phases, n)
    if not cw:
        return []

    head_offsets = [abs(_get(metrics, i, "head_offset")) for i in cw]
    avg_offset = _avg(head_offsets)

    if avg_offset > T1_HEAD_OFFSET_THRESHOLD:
        return [Fault(
            rule_id="T1",
            fault="Head falling outside the line",
            deduction=8,
            detail=f"avg_head_offset={avg_offset:.4f} (threshold {T1_HEAD_OFFSET_THRESHOLD})",
            feedback=(
                "Your head is falling outside the line at contact. Keep your head over "
                "the ball so your eyes stay on a stable tracking plane."
            ),
        )]
    return []


def rule_T2(metrics, phases, baseline):
    """T2: Early head movement before front foot down."""
    n = len(metrics)
    setup_end     = phases.setup_end
    backlift_start = phases.backlift_start

    if backlift_start >= n:
        return []

    setup_frames    = [i for i in range(max(0, setup_end - 10), setup_end + 1) if i < n]
    backlift_frames = [i for i in range(backlift_start, min(backlift_start + 8, n))]

    if not setup_frames or not backlift_frames:
        return []

    setup_head_offset    = _avg([_get(metrics, i, "head_offset") for i in setup_frames])
    backlift_head_offset = _avg([_get(metrics, i, "head_offset") for i in backlift_frames])

    change = abs(backlift_head_offset - setup_head_offset)

    if change > T2_EARLY_HEAD_CHANGE_THRESHOLD:
        return [Fault(
            rule_id="T2",
            fault="Early head movement",
            deduction=7,
            detail=f"head_offset change={change:.4f} (threshold {T2_EARLY_HEAD_CHANGE_THRESHOLD})",
            feedback=(
                "Your head is moving before your feet. Let your feet move first and "
                "keep the head quieter for longer."
            ),
        )]
    return []


def rule_T3(metrics, phases, baseline):
    """T3: Head not still at contact."""
    n = len(metrics)
    cw = _contact_window(phases, n)
    if len(cw) < 3:
        return []

    head_xs = [_get(metrics, i, "head_x") for i in cw]
    head_ys = [_get(metrics, i, "head_y") for i in cw]

    var = float(np.var(head_xs)) + float(np.var(head_ys))

    if var > T3_HEAD_STILLNESS_VARIANCE:
        return [Fault(
            rule_id="T3",
            fault="Head moving at impact",
            deduction=6,
            detail=f"head_position_variance={var:.6f} (threshold {T3_HEAD_STILLNESS_VARIANCE})",
            feedback=(
                "Your head is moving at impact. A still head helps your eyes stay "
                "locked onto the ball."
            ),
        )]
    return []


def rule_T4(metrics, phases, baseline):
    """T4: Eyes not level at contact."""
    n = len(metrics)
    cw = _contact_window(phases, n)
    if not cw:
        return []

    avg_tilt = _avg([_get(metrics, i, "eye_tilt") for i in cw])

    if avg_tilt > T4_EYE_TILT_THRESHOLD:
        return [Fault(
            rule_id="T4",
            fault="Eyes not level through impact",
            deduction=4,
            detail=f"avg_eye_tilt={avg_tilt:.4f} (threshold {T4_EYE_TILT_THRESHOLD})",
            feedback=(
                "Your eyes are not level through impact. Level eyes give you a much "
                "cleaner visual plane to track the ball."
            ),
        )]
    return []


def rule_T5(metrics, phases, baseline):
    """T5: Head not composed in setup."""
    n = len(metrics)
    setup_frames = list(range(0, min(phases.setup_end + 1, n)))
    if len(setup_frames) < 5:
        return []

    head_xs = [_get(metrics, i, "head_x") for i in setup_frames]
    head_ys = [_get(metrics, i, "head_y") for i in setup_frames]

    var = float(np.var(head_xs)) + float(np.var(head_ys))

    if var > T5_SETUP_HEAD_VARIANCE:
        return [Fault(
            rule_id="T5",
            fault="Head moving in setup",
            deduction=3,
            detail=f"setup_head_variance={var:.6f} (threshold {T5_SETUP_HEAD_VARIANCE})",
            feedback=(
                "Your head is moving in your setup. Start from a quieter head position "
                "so you can pick the ball up earlier."
            ),
        )]
    return []


# ---------------------------------------------------------------------------
# ACCESS rules  A1–A6
# ---------------------------------------------------------------------------

def rule_A1(metrics, phases, baseline):
    """A1: Bat path going around the body."""
    n = len(metrics)
    hp = phases.hands_peak
    cw = _contact_window(phases, n)
    if not cw or hp >= n:
        return []

    hp_m = metrics[hp]
    peak_spread    = abs(hp_m.wrist_x_left - hp_m.wrist_x_right)
    contact_spread = _avg([abs(_get(metrics, i, "wrist_x_left") -
                               _get(metrics, i, "wrist_x_right")) for i in cw])
    spread_increase = contact_spread - peak_spread

    if spread_increase > A1_WRIST_SPREAD_THRESHOLD:
        return [Fault(
            rule_id="A1",
            fault="Bat path going around the body",
            deduction=8,
            detail=f"wrist_spread_increase={spread_increase:.4f} (threshold {A1_WRIST_SPREAD_THRESHOLD})",
            feedback=(
                "Your bat is coming around your body rather than through the ball. "
                "Work on sending the hands straighter into the hitting zone."
            ),
        )]
    return []


def rule_A2(metrics, phases, baseline):
    """A2: Contact too close to body (arms fully stretched)."""
    n = len(metrics)
    cw = _contact_window(phases, n)
    if not cw:
        return []

    avg_elbow = _avg([_get(metrics, i, "front_elbow_angle") for i in cw])

    if avg_elbow > A2_ELBOW_STRAIGHT_THRESHOLD:
        return [Fault(
            rule_id="A2",
            fault="Contact too close to body / arms fully stretched",
            deduction=7,
            detail=f"front_elbow_angle={avg_elbow:.1f}° (threshold >{A2_ELBOW_STRAIGHT_THRESHOLD}°)",
            feedback=(
                "You are making contact too close to your body. Create more space out "
                "in front so you can strike the ball with freedom and extend through it."
            ),
        )]
    return []


def rule_A3(metrics, phases, baseline):
    """A3: Restricted contact window (compression too short)."""
    n = len(metrics)
    c = phases.contact

    # Count frames in and after the contact window where wrists continue
    # moving downward (vy > -0.001 = descending or flat = bat driving through).
    # Start from the start of the contact window so natural follow-through
    # deceleration is captured even when contact sits at the velocity zero-crossing.
    cw_start = max(0, c - CONTACT_WINDOW_FRAMES)
    compression_count = 0
    for i in range(cw_start, min(cw_start + 15, n)):
        vy = _get(metrics, i, "wrist_velocity_y")
        if vy > -0.001:   # still going down or flat
            compression_count += 1
        else:
            break  # wrists have started rising = follow-through begun

    ref_compression = baseline.get("follow_through", {}).get("compression_frames", 3)

    if compression_count < A3_COMPRESSION_MIN_FRAMES:
        return [Fault(
            rule_id="A3",
            fault="Restricted contact window",
            deduction=6,
            detail=f"compression_frames={compression_count} (ref={ref_compression})",
            feedback=(
                "Your contact window is too short. The best players keep force on the "
                "ball for longer by driving their hands through the hitting zone. Work "
                "on extending through the ball rather than coming up too quickly."
            ),
        )]
    return []


def rule_A4(metrics, phases, baseline):
    """A4: Torso leaning sideways at contact."""
    n = len(metrics)
    cw = _contact_window(phases, n)
    if not cw:
        return []

    avg_lean = _avg([_get(metrics, i, "torso_lean") for i in cw])

    if avg_lean > A4_TORSO_LEAN_THRESHOLD:
        return [Fault(
            rule_id="A4",
            fault="Torso leaning sideways at contact",
            deduction=6,
            detail=f"avg_torso_lean={avg_lean:.4f} (threshold {A4_TORSO_LEAN_THRESHOLD})",
            feedback=(
                "Your body is leaning sideways at impact. Keep your torso upright "
                "through the shot so your hands can work freely through the ball."
            ),
        )]
    return []


def rule_A5(metrics, phases, baseline):
    """A5: Shoulders lagging behind hips at contact."""
    n = len(metrics)
    cw = _contact_window(phases, n)
    if not cw:
        return []

    gap = _avg([_get(metrics, i, "shoulder_hip_gap") for i in cw])

    if gap < A5_SHOULDER_HIP_GAP_THRESHOLD:
        return [Fault(
            rule_id="A5",
            fault="Shoulders lagging behind hips at contact",
            deduction=4,
            detail=f"shoulder_hip_gap={gap:.1f}° (threshold {A5_SHOULDER_HIP_GAP_THRESHOLD}°)",
            feedback=(
                "Your shoulders are lagging behind your hips at impact. Work on sending "
                "your whole body — hips and shoulders together — as one block of energy "
                "toward the ball."
            ),
        )]
    return []


def rule_A6(metrics, phases, baseline):
    """A6: Opening up too early (before contact).

    Fires if shoulder/hip openness INCREASES significantly from the setup
    baseline TO the hands peak — meaning the batter has rotated toward the
    bowler ON THE WAY UP in the backlift (too early commitment).

    A batter who gets MORE side-on (decreasing openness) at hands peak and
    then opens into contact has the correct coil-and-unwind mechanics.
    """
    n  = len(metrics)
    hp = phases.hands_peak
    if hp >= n:
        return []

    ref_setup = baseline.get("setup", {})
    setup_sh  = ref_setup.get("shoulder_openness_mean", 77.0)
    setup_hip = ref_setup.get("hip_openness_mean", 84.0)

    sh_at_peak  = _get(metrics, hp, "shoulder_openness")
    hip_at_peak = _get(metrics, hp, "hip_openness")

    # Fault if openness has INCREASED from setup to hands peak
    # (body is MORE open at peak than at start = opened up during backlift)
    sh_increase  = sh_at_peak  - setup_sh
    hip_increase = hip_at_peak - setup_hip

    # Allow a small tolerance; only flag clearly larger openness
    fault_sh  = sh_increase  > (setup_sh  * (1.0 - A6_EARLY_OPEN_SHOULDER_FRACTION))
    fault_hip = hip_increase > (setup_hip * (1.0 - A6_EARLY_OPEN_HIP_FRACTION))

    if fault_sh or fault_hip:
        return [Fault(
            rule_id="A6",
            fault="Opening up too early",
            deduction=4,
            detail=f"setup_sh={setup_sh:.1f}°→peak={sh_at_peak:.1f}° (Δ{sh_increase:+.1f}°); "
                   f"setup_hip={setup_hip:.1f}°→peak={hip_at_peak:.1f}° (Δ{hip_increase:+.1f}°)",
            feedback=(
                "You are opening up too early. Your body is facing the bowler before "
                "you have made contact, which means your power has already gone. Stay "
                "side-on longer and let the hands lead into contact."
            ),
        )]
    return []


# ---------------------------------------------------------------------------
# FLOW rules  F1–F6
# ---------------------------------------------------------------------------

def rule_F1(metrics, phases, baseline):
    """F1: Hands Peak / Front Foot Down desync."""
    diff = abs(phases.hands_peak_vs_ffd_diff)
    diff_ms = abs(phases.hands_peak_vs_ffd_ms)

    if diff <= 2:
        return []

    if diff >= F1_LARGE_DESYNC:
        deduction = 12
    elif diff >= F1_MEDIUM_DESYNC:
        deduction = 8
    else:
        deduction = 5

    is_hands_late = phases.hands_peak_vs_ffd_diff < 0
    feedback = (
        "Your hands are peaking before your front foot lands. This breaks the flow of "
        "the movement. Focus on landing and peaking together."
        if is_hands_late else
        "Your front foot is landing before your hands are ready. Work on syncing the "
        "step and swing into one movement."
    )

    return [Fault(
        rule_id="F1",
        fault="Hands Peak / Front Foot Down desync",
        deduction=deduction,
        detail=f"diff={phases.hands_peak_vs_ffd_diff:+d} frames ({diff_ms:.0f}ms)",
        feedback=feedback,
    )]


def rule_F2(metrics, phases, baseline):
    """F2: Jerky acceleration (velocity direction changes)."""
    n = len(metrics)
    bl = phases.backlift_start
    c  = phases.contact

    if c <= bl + 2:
        return []

    vels = [_get(metrics, i, "wrist_velocity_y") for i in range(bl, c + 1)]
    if len(vels) < 4:
        return []

    # Count sign changes in velocity (each change = jerkiness)
    direction_changes = 0
    for j in range(1, len(vels)):
        if vels[j - 1] * vels[j] < 0 and abs(vels[j]) > 0.004:
            direction_changes += 1

    if direction_changes > F2_VELOCITY_DIRECTION_CHANGES_THRESHOLD:
        return [Fault(
            rule_id="F2",
            fault="Jerky acceleration",
            deduction=5,
            detail=f"velocity_direction_changes={direction_changes} (threshold >{F2_VELOCITY_DIRECTION_CHANGES_THRESHOLD})",
            feedback=(
                "Your swing has a hitch in it. The bat should move in one smooth flow "
                "rather than in separate bursts."
            ),
        )]
    return []


def rule_F3(metrics, phases, baseline):
    """F3: Movement timing out of range."""
    ref_frames = baseline.get("timing", {}).get("backlift_to_contact_frames", 15)
    actual_frames = phases.backlift_to_contact_frames

    if ref_frames <= 0:
        return []

    ratio = actual_frames / ref_frames

    if ratio < F3_TIMING_SHORT_THRESHOLD:
        return [Fault(
            rule_id="F3",
            fault="Shot rushed (timing too short)",
            deduction=5,
            detail=f"backlift_to_contact={actual_frames}f vs ref={ref_frames}f (ratio={ratio:.2f})",
            feedback=(
                "The shot looks rushed — everything is happening in a burst. Give "
                "yourself time to build smoothly into the shot."
            ),
        )]

    if ratio > F3_TIMING_LONG_THRESHOLD:
        return [Fault(
            rule_id="F3",
            fault="Shot started too early (timing too long)",
            deduction=5,
            detail=f"backlift_to_contact={actual_frames}f vs ref={ref_frames}f (ratio={ratio:.2f})",
            feedback=(
                "You are starting your movement too early. The best players move late "
                "and smooth. A more compact, efficient swing gives you better timing."
            ),
        )]
    return []


def rule_F4(metrics, phases, baseline):
    """F4: Pause at the top of the backswing."""
    n = len(metrics)
    hp = phases.hands_peak
    # Only look at a small window around the peak
    window = list(range(max(0, hp - 2), min(n, hp + 4)))

    if len(window) < 3:
        return []

    vels = [abs(_get(metrics, i, "wrist_velocity_y")) for i in window]
    pause_frames = sum(1 for v in vels if v < F4_PAUSE_VELOCITY_THRESHOLD)

    if pause_frames >= F4_PAUSE_FRAMES_THRESHOLD:
        return [Fault(
            rule_id="F4",
            fault="Pause at the top of the backswing",
            deduction=5,
            detail=f"pause_frames={pause_frames} (threshold ≥{F4_PAUSE_FRAMES_THRESHOLD})",
            feedback=(
                "Your hands are pausing at the top of the backswing. Try to make the "
                "lift and downswing feel like one connected movement."
            ),
        )]
    return []


def rule_F5(metrics, phases, baseline):
    """F5: Mid-downswing hitch (between Hands Peak + 3 and Contact)."""
    n = len(metrics)
    hp = phases.hands_peak
    c  = phases.contact

    downswing_start = hp + 3
    if downswing_start >= c - 1:
        return []

    vels = [abs(_get(metrics, i, "wrist_velocity_y"))
            for i in range(downswing_start, c)]

    if len(vels) < 3:
        return []

    avg_vel = float(np.mean(vels)) if vels else 0.0
    if avg_vel < 0.001:
        return []   # barely any downswing to measure

    hitch_threshold = avg_vel * F5_HITCH_VELOCITY_FRACTION
    hitch_frames = sum(1 for v in vels if v < hitch_threshold)

    if hitch_frames >= 2:
        return [Fault(
            rule_id="F5",
            fault="Mid-downswing hitch",
            deduction=4,
            detail=f"hitch_frames={hitch_frames}, avg_downswing_vel={avg_vel:.4f}",
            feedback=(
                "The transition from lift to strike has a hitch in it. The best swings "
                "flow up and down as one movement without interruption."
            ),
        )]
    return []


def rule_F6(metrics, phases, baseline):
    """F6: No natural follow-through."""
    n = len(metrics)
    c  = phases.contact
    ft = phases.follow_through_start

    if ft >= n - 2:
        return []

    ft_frames = list(range(ft, min(ft + FOLLOW_THROUGH_ANALYSIS_FRAMES, n)))
    if len(ft_frames) < 2:
        return []

    # Shoulder height at contact (reference for "above shoulder" check)
    sh_y = _get(metrics, c, "shoulder_mid_y")

    wrist_ys = [_get(metrics, i, "wrist_height") for i in ft_frames]

    # Wrists should rise above shoulder height in follow-through
    # (smaller Y = higher in frame = above shoulder)
    wrists_above_shoulder = any(
        wy < sh_y - F6_FOLLOWTHROUGH_WRIST_SHOULDER_MARGIN
        for wy in wrist_ys
    )

    # Also check follow-through duration
    ft_duration = len([i for i in ft_frames
                        if _get(metrics, i, "wrist_velocity_y") < 0])

    if not wrists_above_shoulder or ft_duration < F6_FOLLOWTHROUGH_MIN_FRAMES:
        return [Fault(
            rule_id="F6",
            fault="No natural follow-through",
            deduction=3,
            detail=f"wrists_above_shoulder={wrists_above_shoulder}, ft_duration={ft_duration}f",
            feedback=(
                "Your follow-through is being cut off. Let the hands continue naturally "
                "through the ball after impact."
            ),
        )]
    return []


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

# All rules in specification order
_ALL_RULES = [
    # Stability
    ("stability", rule_S1),
    ("stability", rule_S2),
    ("stability", rule_S3),
    ("stability", rule_S4),
    # Tracking
    ("tracking", rule_T1),
    ("tracking", rule_T2),
    ("tracking", rule_T3),
    ("tracking", rule_T4),
    ("tracking", rule_T5),
    # Access
    ("access", rule_A1),
    ("access", rule_A2),
    ("access", rule_A3),
    ("access", rule_A4),
    ("access", rule_A5),
    ("access", rule_A6),
    # Flow
    ("flow", rule_F1),
    ("flow", rule_F2),
    ("flow", rule_F3),
    ("flow", rule_F4),
    ("flow", rule_F5),
    ("flow", rule_F6),
]


def run_all_rules(
    metrics: list[FrameMetrics],
    phases: PhaseResult,
    baseline: dict,
    front_side: str = "left",
) -> dict[str, list[Fault]]:
    """
    Run all 21 coaching rules.
    Returns dict keyed by pillar name with a list of detected faults.

    Args:
        front_side: "left" for right-handed batter, "right" for left-handed.
    """
    results: dict[str, list[Fault]] = {
        "access": [], "tracking": [], "stability": [], "flow": []
    }
    rules_evaluated = 0
    rules_failed = []

    for pillar, rule_fn in _ALL_RULES:
        try:
            if rule_fn is rule_S3:
                faults = rule_fn(metrics, phases, baseline, front_side=front_side)
            else:
                faults = rule_fn(metrics, phases, baseline)
            results[pillar].extend(faults)
            rules_evaluated += 1
        except Exception as e:
            # Never crash the whole pipeline on a single rule
            rule_id = rule_fn.__name__.replace("rule_", "")
            rules_failed.append({"rule_id": rule_id, "error": str(e)})
            print(f"  Warning: rule {rule_fn.__name__} raised {e}")

    # Attach evaluation health as metadata
    results["_evaluation"] = {
        "rules_total": len(_ALL_RULES),
        "rules_evaluated": rules_evaluated,
        "rules_failed": rules_failed,
    }

    return results
