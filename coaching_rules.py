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
    A1_FULL_MARKS_THRESHOLD,
    A1_PARTIAL_THRESHOLD,
    A1_SIGNIFICANT_THRESHOLD,
    A1_FULL_MARKS_DEDUCTION,
    A1_PARTIAL_DEDUCTION,
    A1_SIGNIFICANT_DEDUCTION,
    A1_SEVERE_DEDUCTION,
    SUSPEND_A1,
    SUSPEND_A3,
    A3_FULL_MARKS_MIN,
    A3_SEVERE_DEDUCTION,
    A5_IDEAL_MIN,
    A5_IDEAL_MAX,
    A5_PARTIAL_OUTER,
    A5_SIGNIFICANT_OUTER,
    A5_FULL_MARKS_DEDUCTION,
    A5_PARTIAL_DEDUCTION,
    A5_SIGNIFICANT_DEDUCTION,
    A5_SEVERE_DEDUCTION,
    # DELETED 2026-04-29: A6 — downstream consequence, not independent
    # Tracking
    T2_MINOR_THRESHOLD,
    T2_SIGNIFICANT_THRESHOLD,
    T2_SEVERE_THRESHOLD,
    T2_FULL_MARKS_DEDUCTION,
    T2_PARTIAL_DEDUCTION,
    T2_SIGNIFICANT_DEDUCTION,
    T2_SEVERE_DEDUCTION,
    # Stability
    S1_FULL_MARKS_THRESHOLD,
    S1_PARTIAL_THRESHOLD,
    S1_SIGNIFICANT_THRESHOLD,
    S1_FULL_MARKS_DEDUCTION,
    S1_PARTIAL_DEDUCTION,
    S1_SIGNIFICANT_DEDUCTION,
    S1_SEVERE_DEDUCTION,
    S2_FULL_MARKS_THRESHOLD,
    S2_PARTIAL_THRESHOLD,
    S2_SIGNIFICANT_THRESHOLD,
    S2_FULL_MARKS_DEDUCTION,
    S2_PARTIAL_DEDUCTION,
    S2_SIGNIFICANT_DEDUCTION,
    S2_SEVERE_DEDUCTION,
    S3_HIP_DRIFT_TOLERANCE,
    S4_POST_CONTACT_ROTATION_FRAMES,
    S4_FULL_MARKS_THRESHOLD,
    S4_PARTIAL_THRESHOLD,
    S4_SIGNIFICANT_THRESHOLD,
    S4_FULL_MARKS_DEDUCTION,
    S4_PARTIAL_DEDUCTION,
    S4_SIGNIFICANT_DEDUCTION,
    S4_SEVERE_DEDUCTION,
    # Flow
    F1_FULL_MARKS_THRESHOLD,
    F1_PARTIAL_THRESHOLD,
    F1_SIGNIFICANT_THRESHOLD,
    F1_FULL_MARKS_DEDUCTION,
    F1_PARTIAL_DEDUCTION,
    F1_SIGNIFICANT_DEDUCTION,
    F1_SEVERE_DEDUCTION,
    F3_IDEAL_RATIO,
    F3_FULL_MARKS_DEVIATION,
    F3_PARTIAL_DEVIATION,
    F3_SIGNIFICANT_DEVIATION,
    F3_FULL_MARKS_DEDUCTION,
    F3_PARTIAL_DEDUCTION,
    F3_SIGNIFICANT_DEDUCTION,
    F3_SEVERE_DEDUCTION,
    F4_PAUSE_VELOCITY_THRESHOLD,
    F4_FULL_MARKS_THRESHOLD,
    F4_PARTIAL_THRESHOLD,
    F4_SIGNIFICANT_THRESHOLD,
    F4_FULL_MARKS_DEDUCTION,
    F4_PARTIAL_DEDUCTION,
    F4_SIGNIFICANT_DEDUCTION,
    F4_SEVERE_DEDUCTION,
    F5_HITCH_VELOCITY_FRACTION,
    F5_HITCH_DEDUCTION,
    SUSPEND_F6,
    F6_POST_CONTACT_OFFSET,
    F6_ANALYSIS_FRAMES,
    F6_V2_POST_CONTACT_WINDOW,
    F6_V2_METRIC,
    F6_FULL_MARKS_THRESHOLD,
    F6_PARTIAL_THRESHOLD,
    F6_SIGNIFICANT_THRESHOLD,
    F6_FULL_MARKS_DEDUCTION,
    F6_PARTIAL_DEDUCTION,
    F6_SIGNIFICANT_DEDUCTION,
    F6_SEVERE_DEDUCTION,
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


def _band_high(value: float, elite_max: float, good_max: float, average_max: float,
               deductions: tuple[int, int, int, int] = (0, 3, 5, 7)) -> int:
    if value <= elite_max:
        return deductions[0]
    if value <= good_max:
        return deductions[1]
    if value <= average_max:
        return deductions[2]
    return deductions[3]


def _band_low(value: float, elite_min: float, good_min: float, average_min: float,
              deductions: tuple[int, int, int, int] = (0, 3, 5, 7)) -> int:
    if value >= elite_min:
        return deductions[0]
    if value >= good_min:
        return deductions[1]
    if value >= average_min:
        return deductions[2]
    return deductions[3]


def _band_distance(value: float, elite_tol: float, good_tol: float, average_tol: float,
                   deductions: tuple[int, int, int, int] = (0, 3, 5, 7)) -> int:
    abs_value = abs(value)
    if abs_value <= elite_tol:
        return deductions[0]
    if abs_value <= good_tol:
        return deductions[1]
    if abs_value <= average_tol:
        return deductions[2]
    return deductions[3]


def _cap(total: int, cap: int) -> int:
    return min(total, cap)


# ---------------------------------------------------------------------------
# STABILITY rules  S1–S4
# ---------------------------------------------------------------------------

def rule_S1(metrics, phases, baseline):
    """S1: Weight not transferring to front foot."""
    # SUSPENDED 2026-04-29: width is proxy-of-proxy for balance; revisit when direct balance measurement available
    return []
    # Old logic: penalised high hip shift alongside front/back knee checks.
    # New logic: invert the signal and only penalise low hip shift because more shift is positive.
    n = len(metrics)
    cw = _contact_window(phases, n)
    if not cw:
        return []

    hip_x_contact  = _avg([_get(metrics, i, "hip_centre_x") for i in cw])
    hip_x_setup    = baseline["setup"]["hip_centre_x_mean"]
    hip_shift = abs(hip_x_contact - hip_x_setup)
    if hip_shift >= S1_FULL_MARKS_THRESHOLD:
        deduction = S1_FULL_MARKS_DEDUCTION
    elif hip_shift >= S1_PARTIAL_THRESHOLD:
        deduction = S1_PARTIAL_DEDUCTION
    elif hip_shift >= S1_SIGNIFICANT_THRESHOLD:
        deduction = S1_SIGNIFICANT_DEDUCTION
    else:
        deduction = S1_SEVERE_DEDUCTION

    if deduction == 0:
        return []

    return [Fault(
        rule_id="S1",
        fault="Weight not transferring to front foot",
        deduction=deduction,
        detail=f"hip_shift={hip_shift:.3f}",
        feedback=(
            "Your weight is staying too much on the back foot. Move your body through "
            "the ball so your hips travel with your front side. A bent front knee loaded "
            "with your weight gives you the platform to drive through the ball."
        ),
    )]


def rule_S2(metrics, phases, baseline):
    """S2: Post-contact instability."""
    # SUSPENDED 2026-05-11: Inverted after F5b detection improvements.
    # S2 fire rate: Elite > Beginner (should be opposite).
    # Root cause: thresholds calibrated on incorrect HP/FFD frames.
    # Redesign scheduled for R4-equivalent task once detection stable.
    return []
    # Old logic: high post-contact movement triggered a heavier graduated penalty.
    # New logic: keep the signal but cap it at a low maximum deduction.
    n = len(metrics)
    start = phases.follow_through_start
    ft_frames = list(range(start, min(start + FOLLOW_THROUGH_ANALYSIS_FRAMES, n)))
    if len(ft_frames) < 4:
        return []

    hip_xs  = [_get(metrics, i, "hip_centre_x") for i in ft_frames]
    head_xs = [_get(metrics, i, "head_x")       for i in ft_frames]

    hip_std  = _std(hip_xs)
    head_std = _std(head_xs)
    instability = max(hip_std, head_std)
    if instability <= S2_FULL_MARKS_THRESHOLD:
        deduction = S2_FULL_MARKS_DEDUCTION
    elif instability <= S2_PARTIAL_THRESHOLD:
        deduction = S2_PARTIAL_DEDUCTION
    elif instability <= S2_SIGNIFICANT_THRESHOLD:
        deduction = S2_SIGNIFICANT_DEDUCTION
    else:
        deduction = S2_SEVERE_DEDUCTION

    if deduction:
        return [Fault(
            rule_id="S2",
            fault="Post-contact instability",
            deduction=deduction,
            detail=f"hip_std={hip_std:.4f}, head_std={head_std:.4f}, max={instability:.4f}",
            feedback=(
                "You are losing your balance after contact. The best players are set at "
                "impact and can hold their finish. If you are stumbling or correcting after "
                "the shot, your base was not stable enough at impact."
            ),
        )]
    return []


def rule_S3(metrics, phases, baseline, front_side="left"):
    """S3: Hips drifting outside the base."""
    # Old logic: penalised drift count and magnitude.
    # New logic: SUSPENDED because the measured distribution is misleading and inverted by outliers.
    return []


def rule_S4(metrics, phases, baseline):
    """S4: Post-contact body rotation continues."""
    # Old logic: used absolute rotation, which penalised negative recoil/settling.
    # New logic: positive rotation = forward rotation continuing after contact.
    n = len(metrics)
    c = phases.contact
    check_end = min(c + S4_POST_CONTACT_ROTATION_FRAMES, n - 1)

    if c >= n - 2 or check_end >= n:
        return []

    sh_at_contact  = _get(metrics, c, "shoulder_openness")
    hip_at_contact = _get(metrics, c, "hip_openness")
    sh_post        = _get(metrics, check_end, "shoulder_openness")
    hip_post       = _get(metrics, check_end, "hip_openness")

    sh_change  = sh_post  - sh_at_contact   # positive = opening more
    hip_change = hip_post - hip_at_contact
    rotation = max(sh_change, hip_change)
    forward_rotation = max(0.0, rotation)
    band_value = abs(forward_rotation)

    # REVISED 2026-05-14: fixed sign convention (positive = forward rotation
    # continuing post-contact). Bands tightened at upper end.
    if band_value < S4_FULL_MARKS_THRESHOLD:
        deduction = S4_FULL_MARKS_DEDUCTION
    elif band_value < S4_PARTIAL_THRESHOLD:
        deduction = S4_PARTIAL_DEDUCTION
    elif band_value < S4_SIGNIFICANT_THRESHOLD:
        deduction = S4_SIGNIFICANT_DEDUCTION
    else:
        deduction = S4_SEVERE_DEDUCTION

    if deduction:
        return [Fault(
            rule_id="S4",
            fault="Post-contact body rotation",
            deduction=deduction,
            detail=f"shoulder_change={sh_change:.1f}°, hip_change={hip_change:.1f}°, forward_rotation={forward_rotation:.1f}°",
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
    # Old logic: penalised larger head offsets.
    # New logic: SUSPENDED because the measured distribution is inverted (Elite > Beginner).
    return []


def rule_T2(metrics, phases, baseline):
    """T2: Early head movement before front foot down."""
    # Old logic: one binary cutoff on head movement between setup and backlift.
    # New logic: apply the measured four-band thresholds directly.
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
    if change < T2_MINOR_THRESHOLD:
        deduction = T2_FULL_MARKS_DEDUCTION
    elif change < T2_SIGNIFICANT_THRESHOLD:
        deduction = T2_PARTIAL_DEDUCTION
    elif change < T2_SEVERE_THRESHOLD:
        deduction = T2_SIGNIFICANT_DEDUCTION
    else:
        deduction = T2_SEVERE_DEDUCTION

    if deduction:
        return [Fault(
            rule_id="T2",
            fault="Early head movement",
            deduction=deduction,
            detail=f"head_offset change={change:.4f}",
            feedback=(
                "Your head is moving before your feet. Let your feet move first and "
                "keep the head quieter for longer."
            ),
        )]
    return []


# DELETED 2026-04-29: T3/T4/T5 — not measurable from side-on video
# def rule_T3(metrics, phases, baseline):
#     """T3: Head not still at contact."""
#     return []
#
#
# def rule_T4(metrics, phases, baseline):
#     """T4: Eyes not level at contact."""
#     return []
#
#
# def rule_T5(metrics, phases, baseline):
#     """T5: Head not composed in setup."""
#     return []


# ---------------------------------------------------------------------------
# ACCESS rules  A1–A6
# ---------------------------------------------------------------------------

def rule_A1(metrics, phases, baseline):
    """A1: Bat path going around the body."""
    if SUSPEND_A1:
        return []

    # Old logic: penalised large positive wrist spread increases.
    # New logic: use the sign directly so negative values are treated as bat wrapping.
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
    if spread_increase > A1_FULL_MARKS_THRESHOLD:
        deduction = A1_FULL_MARKS_DEDUCTION
    elif spread_increase > A1_PARTIAL_THRESHOLD:
        deduction = A1_PARTIAL_DEDUCTION
    elif spread_increase > A1_SIGNIFICANT_THRESHOLD:
        deduction = A1_SIGNIFICANT_DEDUCTION
    else:
        deduction = A1_SEVERE_DEDUCTION

    if deduction:
        return [Fault(
            rule_id="A1",
            fault="Bat path going around the body",
            deduction=deduction,
            detail=f"wrist_spread_increase={spread_increase:.4f}",
            feedback=(
                "Your bat is coming around your body rather than through the ball. "
                "Work on sending the hands straighter into the hitting zone."
            ),
        )]
    return []


def rule_A2(metrics, phases, baseline):
    """A2: Contact too close to body (arms fully stretched)."""
    # Old logic: penalised larger elbow angles at contact.
    # New logic: SUSPENDED because the elbow angle distribution is non-monotonic across tiers.
    return []


def rule_A3(metrics, phases, baseline):
    """A3: Restricted contact window (compression too short)."""
    if SUSPEND_A3:
        # SUSPENDED 2026-05-13: D5 discrimination_score=0.45; adds noise, not signal.
        return []

    # Old logic: one binary cutoff on compression frames.
    # New logic: three-state graduation because the measured values are only 0, 1, or 2.
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

    # A3 FIXED 2026-04-25: collapsed to two bands; else branch was unreachable.
    # Previous code had A3_PARTIAL_MIN=0, A3_PARTIAL_DEDUCTION=0, making both
    # reachable branches return 0 — the deduction=3 else required count < 0.
    if compression_count >= A3_FULL_MARKS_MIN:
        return []

    return [Fault(
        rule_id="A3",
        fault="Restricted contact window",
        deduction=A3_SEVERE_DEDUCTION,
        detail=f"compression_frames={compression_count}",
        feedback=(
            "Your contact window is too short. The best players keep force on the "
            "ball for longer by driving their hands through the hitting zone. Work "
            "on extending through the ball rather than coming up too quickly."
        ),
    )]


def rule_A4(metrics, phases, baseline):
    """A4: Torso leaning sideways at contact."""
    # Old logic: penalised more torso lean.
    # New logic: SUSPENDED because elite players lean more than beginners in measured data.
    return []


def rule_A5(metrics, phases, baseline):
    """A5: Shoulders lagging behind hips at contact."""
    # Old logic: one-direction cutoff on shoulder-hip gap.
    # New logic: treat the ideal band as a target range and penalise deviation on either side.
    n = len(metrics)
    cw = _contact_window(phases, n)
    if not cw:
        return []

    gap = _avg([_get(metrics, i, "shoulder_hip_gap") for i in cw])
    deviation = max(0.0, A5_IDEAL_MIN - gap) + max(0.0, gap - A5_IDEAL_MAX)
    if deviation == 0:
        deduction = A5_FULL_MARKS_DEDUCTION
    elif deviation <= A5_PARTIAL_OUTER:
        deduction = A5_PARTIAL_DEDUCTION
    elif deviation <= A5_SIGNIFICANT_OUTER:
        deduction = A5_SIGNIFICANT_DEDUCTION
    else:
        deduction = A5_SEVERE_DEDUCTION

    if deduction:
        return [Fault(
            rule_id="A5",
            fault="Shoulders lagging behind hips at contact",
            deduction=deduction,
            detail=f"shoulder_hip_gap={gap:.1f}°, deviation={deviation:.1f}°",
            feedback=(
                "Your shoulders are lagging behind your hips at impact. Work on sending "
                "your whole body — hips and shoulders together — as one block of energy "
                "toward the ball."
            ),
        )]
    return []


# DELETED 2026-04-29: A6 — downstream consequence, not independent
# def rule_A6(metrics, phases, baseline):
#     """A6: Opening up too early (before contact).
#
#     Fires if shoulder/hip openness INCREASES significantly from the setup
#     baseline TO the hands peak — meaning the batter has rotated toward the
#     bowler ON THE WAY UP in the backlift (too early commitment).
#
#     A batter who gets MORE side-on (decreasing openness) at hands peak and
#     then opens into contact has the correct coil-and-unwind mechanics.
#     """
#     # Old logic: two binary checks using setup fractions.
#     # New logic: negative values are fine; only positive early opening is penalised in measured bands.
#     n  = len(metrics)
#     hp = phases.hands_peak
#     if hp >= n:
#         return []
#
#     ref_setup = baseline.get("setup", {})
#     setup_sh  = ref_setup.get("shoulder_openness_mean", 77.0)
#     setup_hip = ref_setup.get("hip_openness_mean", 84.0)
#
#     sh_at_peak  = _get(metrics, hp, "shoulder_openness")
#     hip_at_peak = _get(metrics, hp, "hip_openness")
#
#     # Fault if openness has INCREASED from setup to hands peak
#     # (body is MORE open at peak than at start = opened up during backlift)
#     sh_increase  = sh_at_peak  - setup_sh
#     hip_increase = hip_at_peak - setup_hip
#     sh_fraction = sh_increase / setup_sh if setup_sh else 0.0
#     hip_fraction = hip_increase / setup_hip if setup_hip else 0.0
#     openness = max(sh_fraction, hip_fraction)
#     if openness < A6_FULL_MARKS_THRESHOLD:
#         deduction = A6_FULL_MARKS_DEDUCTION
#     elif openness < A6_PARTIAL_THRESHOLD:
#         deduction = A6_PARTIAL_DEDUCTION
#     elif openness < A6_SIGNIFICANT_THRESHOLD:
#         deduction = A6_SIGNIFICANT_DEDUCTION
#     else:
#         deduction = A6_SEVERE_DEDUCTION
#
#     if deduction:
#         return [Fault(
#             rule_id="A6",
#             fault="Opening up too early",
#             deduction=deduction,
#             detail=f"setup_sh={setup_sh:.1f}°→peak={sh_at_peak:.1f}° (Δ{sh_increase:+.1f}°); "
#                    f"setup_hip={setup_hip:.1f}°→peak={hip_at_peak:.1f}° (Δ{hip_increase:+.1f}°); "
#                    f"fraction={openness:.2%}",
#             feedback=(
#                 "You are opening up too early. Your body is facing the bowler before "
#                 "you have made contact, which means your power has already gone. Stay "
#                 "side-on longer and let the hands lead into contact."
#             ),
#         )]
#     return []

# ---------------------------------------------------------------------------
# FLOW rules  F1–F6
# ---------------------------------------------------------------------------

def rule_F1(metrics, phases, baseline):
    """F1: Hands Peak / Front Foot Down desync."""
    # F1 SUSPENDED 2026-05-11: Hands_peak/FFD sync fires 4/4 Elite in auto,
    # 0/4 in validated. FFD MAE 7.0f corrupts sync measurement.
    # Redesign: tighten FFD detection or use more stable sync proxy.
    return []
    # Old logic: coarse desync buckets.
    # New logic: exact measured frame thresholds with a modest maximum weight.
    diff = abs(phases.hands_peak_vs_ffd_diff)
    diff_ms = abs(phases.hands_peak_vs_ffd_ms)
    if diff <= F1_FULL_MARKS_THRESHOLD:
        deduction = F1_FULL_MARKS_DEDUCTION
    elif diff <= F1_PARTIAL_THRESHOLD:
        deduction = F1_PARTIAL_DEDUCTION
    elif diff <= F1_SIGNIFICANT_THRESHOLD:
        deduction = F1_SIGNIFICANT_DEDUCTION
    else:
        deduction = F1_SEVERE_DEDUCTION

    if deduction == 0:
        return []

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
    # Old logic: penalised more direction changes.
    # New logic: SUSPENDED because the measured distribution is inverted (Elite > Beginner).
    return []


def rule_F3(metrics, phases, baseline):
    """F3: Movement timing out of range."""
    # F3 SUSPENDED 2026-05-11: backlift_to_contact_frames produces negative values on
    # Elite (backlift_start detected after contact). Root cause: backlift_start inherits
    # setup_frame errors (MAE 26.4f). Metric is invalid until setup/backlift detection
    # is stable. Redesign: replace metric with HP-to-contact frames once stable.
    return []
    # Old logic: binary short/long thresholds that both gave the same deduction.
    # New logic: direct deviation-from-ideal bands based on the measured ratio distribution.
    ref_frames = baseline.get("timing", {}).get("backlift_to_contact_frames", 15)
    actual_frames = phases.backlift_to_contact_frames

    if ref_frames <= 0:
        return []

    ratio = actual_frames / ref_frames
    deviation = abs(ratio - F3_IDEAL_RATIO)
    if deviation <= F3_FULL_MARKS_DEVIATION:
        deduction = F3_FULL_MARKS_DEDUCTION
    elif deviation <= F3_PARTIAL_DEVIATION:
        deduction = F3_PARTIAL_DEDUCTION
    elif deviation <= F3_SIGNIFICANT_DEVIATION:
        deduction = F3_SIGNIFICANT_DEDUCTION
    else:
        deduction = F3_SEVERE_DEDUCTION

    if deduction == 0:
        return []

    if ratio < 1.0:
        fault = "Shot rushed (timing too short)"
        feedback = (
            "The shot looks rushed — everything is happening in a burst. Give "
            "yourself time to build smoothly into the shot."
        )
    else:
        fault = "Shot started too early (timing too long)"
        feedback = (
            "You are starting your movement too early. The best players move late "
            "and smooth. A more compact, efficient swing gives you better timing."
        )

    return [Fault(
        rule_id="F3",
        fault=fault,
        deduction=deduction,
        detail=f"backlift_to_contact={actual_frames}f vs ref={ref_frames}f (ratio={ratio:.2f})",
        feedback=feedback,
    )]


def rule_F4(metrics, phases, baseline):
    """F4: Pause at the top of the backswing."""
    # Old logic: binary cutoff on the number of near-zero velocity frames.
    # New logic: exact graduated pause-frame bands from measured distributions.
    n = len(metrics)
    hp = phases.hands_peak
    # Only look at a small window around the peak
    window = list(range(max(0, hp - 2), min(n, hp + 4)))

    if len(window) < 3:
        return []

    vels = [abs(_get(metrics, i, "wrist_velocity_y")) for i in window]
    pause_frames = sum(1 for v in vels if v < F4_PAUSE_VELOCITY_THRESHOLD)
    if pause_frames <= F4_FULL_MARKS_THRESHOLD:
        deduction = F4_FULL_MARKS_DEDUCTION
    elif pause_frames <= F4_PARTIAL_THRESHOLD:
        deduction = F4_PARTIAL_DEDUCTION
    else:
        deduction = F4_SEVERE_DEDUCTION

    if deduction:
        return [Fault(
            rule_id="F4",
            fault="Pause at the top of the backswing",
            deduction=deduction,
            detail=f"pause_frames={pause_frames}",
            feedback=(
                "Your hands are pausing at the top of the backswing. Try to make the "
                "lift and downswing feel like one connected movement."
            ),
        )]
    return []


def rule_F5(metrics, phases, baseline):
    """F5: Mid-downswing hitch (between Hands Peak + 3 and Contact)."""
    # SUSPENDED 2026-05-11: Inverted after F5b detection improvements.
    # F5 fire rate: Elite > Beginner (should be opposite).
    # Root cause: thresholds calibrated on incorrect HP/FFD frames.
    # Redesign scheduled for R4-equivalent task once detection stable.
    return []
    # Old logic: one binary cutoff on hitch frames.
    # New logic: keep it binary because the measured distribution is effectively 0 or 1.
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
    deduction = 0 if hitch_frames == 0 else F5_HITCH_DEDUCTION

    if deduction:
        return [Fault(
            rule_id="F5",
            fault="Mid-downswing hitch",
            deduction=deduction,
            detail=f"hitch_frames={hitch_frames}, avg_downswing_vel={avg_vel:.4f}",
            feedback=(
                "The transition from lift to strike has a hitch in it. The best swings "
                "flow up and down as one movement without interruption."
            ),
        )]
    return []


def rule_F6(metrics, phases, baseline):
    """F6: No natural follow-through."""
    if SUSPEND_F6:
        # RE-SUSPENDED 2026-05-14: duration-based F6 was flat after follow_through v2.
        return []

    # Previous duration-based F6 abandoned 2026-05-15:
    # ft_duration = max(0, phases.follow_through_start - phases.contact)
    # if ft_duration >= F6_FULL_MARKS_THRESHOLD: ...

    # F6 V2: post-contact wrist velocity (contact anchor only)
    n = len(metrics)
    c = phases.contact
    if c >= n:
        return []

    window_end = min(c + F6_V2_POST_CONTACT_WINDOW, n - 1)
    window_frames = [
        c + i
        for i in range(1, F6_V2_POST_CONTACT_WINDOW + 1)
        if c + i <= window_end
    ]
    if len(window_frames) < 3:
        return []

    speeds = [abs(_get(metrics, i, F6_V2_METRIC)) for i in window_frames]
    mean_speed = sum(speeds) / len(speeds)

    if mean_speed >= F6_FULL_MARKS_THRESHOLD:
        deduction = F6_FULL_MARKS_DEDUCTION
    elif mean_speed >= F6_PARTIAL_THRESHOLD:
        deduction = F6_PARTIAL_DEDUCTION
    elif mean_speed >= F6_SIGNIFICANT_THRESHOLD:
        deduction = F6_SIGNIFICANT_DEDUCTION
    else:
        deduction = F6_SEVERE_DEDUCTION

    if deduction:
        return [Fault(
            rule_id="F6",
            fault="No natural follow-through",
            deduction=deduction,
            detail=f"mean_post_contact_wrist_speed={mean_speed:.6f}, frames={len(window_frames)}",
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
    # DELETED 2026-04-29: T3/T4/T5 — not measurable from side-on video
    # ("tracking", rule_T3),
    # ("tracking", rule_T4),
    # ("tracking", rule_T5),
    # Access
    ("access", rule_A1),
    ("access", rule_A2),
    ("access", rule_A3),
    ("access", rule_A4),
    ("access", rule_A5),
    # DELETED 2026-04-29: A6 — downstream consequence, not independent
    # ("access", rule_A6),
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


def collect_rule_measurements(
    metrics: list[FrameMetrics],
    phases: PhaseResult,
    baseline: dict,
    front_side: str = "left",
) -> dict[str, float | int | None]:
    """Export one primary raw measurement per rule for calibration analysis."""
    n = len(metrics)
    cw = _contact_window(phases, n)
    measurements: dict[str, float | int | None] = {
        "S1_hip_shift": None,
        "S2_post_contact_instability_std": None,
        "S3_hip_drift_frames": None,
        "S4_post_contact_rotation_deg": None,
        "T1_head_offset": None,
        "T2_early_head_change": None,
        "T3_head_position_variance": None,
        "T4_eye_tilt": None,
        "T5_setup_head_variance": None,
        "A1_wrist_spread_increase": None,
        "A2_elbow_angle_deg": None,
        "A3_compression_frames": None,
        "A4_torso_lean": None,
        "A5_shoulder_hip_gap_deg": None,
        "A6_early_open_fraction": None,
        "F1_sync_frames": None,
        "F2_velocity_direction_changes": None,
        "F3_timing_ratio": None,
        "F4_pause_frames": None,
        "F5_hitch_frames": None,
        "F6_follow_through_frames": None,
    }

    if cw:
        hip_x_contact = _avg([_get(metrics, i, "hip_centre_x") for i in cw])
        hip_x_setup = baseline["setup"]["hip_centre_x_mean"]
        measurements["S1_hip_shift"] = round(abs(hip_x_contact - hip_x_setup), 6)
        measurements["T1_head_offset"] = round(_avg([abs(_get(metrics, i, "head_offset")) for i in cw]), 6)
        head_xs = [_get(metrics, i, "head_x") for i in cw]
        head_ys = [_get(metrics, i, "head_y") for i in cw]
        measurements["T3_head_position_variance"] = round(float(np.var(head_xs)) + float(np.var(head_ys)), 6)
        measurements["T4_eye_tilt"] = round(_avg([_get(metrics, i, "eye_tilt") for i in cw]), 6)
        measurements["A2_elbow_angle_deg"] = round(_avg([_get(metrics, i, "front_elbow_angle") for i in cw]), 3)
        measurements["A4_torso_lean"] = round(_avg([_get(metrics, i, "torso_lean") for i in cw]), 6)
        measurements["A5_shoulder_hip_gap_deg"] = round(_avg([_get(metrics, i, "shoulder_hip_gap") for i in cw]), 3)

    setup_end = phases.setup_end
    backlift_start = phases.backlift_start
    setup_frames = [i for i in range(max(0, setup_end - 10), setup_end + 1) if i < n]
    backlift_frames = [i for i in range(backlift_start, min(backlift_start + 8, n))]
    if setup_frames:
        head_xs = [_get(metrics, i, "head_x") for i in range(0, min(setup_end + 1, n))]
        head_ys = [_get(metrics, i, "head_y") for i in range(0, min(setup_end + 1, n))]
        if len(head_xs) >= 5:
            measurements["T5_setup_head_variance"] = round(float(np.var(head_xs)) + float(np.var(head_ys)), 6)
    if setup_frames and backlift_frames:
        setup_head_offset = _avg([_get(metrics, i, "head_offset") for i in setup_frames])
        backlift_head_offset = _avg([_get(metrics, i, "head_offset") for i in backlift_frames])
        measurements["T2_early_head_change"] = round(abs(backlift_head_offset - setup_head_offset), 6)

    hp = phases.hands_peak
    if cw and hp < n:
        peak_spread = abs(_get(metrics, hp, "wrist_x_left") - _get(metrics, hp, "wrist_x_right"))
        contact_spread = _avg([
            abs(_get(metrics, i, "wrist_x_left") - _get(metrics, i, "wrist_x_right"))
            for i in cw
        ])
        measurements["A1_wrist_spread_increase"] = round(contact_spread - peak_spread, 6)

        ref_setup = baseline.get("setup", {})
        setup_sh = ref_setup.get("shoulder_openness_mean", 77.0)
        setup_hip = ref_setup.get("hip_openness_mean", 84.0)
        sh_at_peak = _get(metrics, hp, "shoulder_openness")
        hip_at_peak = _get(metrics, hp, "hip_openness")
        sh_fraction = (sh_at_peak - setup_sh) / setup_sh if setup_sh else 0.0
        hip_fraction = (hip_at_peak - setup_hip) / setup_hip if setup_hip else 0.0
        measurements["A6_early_open_fraction"] = round(max(sh_fraction, hip_fraction), 6)

    c = phases.contact
    cw_start = max(0, c - CONTACT_WINDOW_FRAMES)
    compression_count = 0
    for i in range(cw_start, min(cw_start + 15, n)):
        if _get(metrics, i, "wrist_velocity_y") > -0.001:
            compression_count += 1
        else:
            break
    measurements["A3_compression_frames"] = compression_count

    ft_frames = list(range(phases.follow_through_start, min(phases.follow_through_start + FOLLOW_THROUGH_ANALYSIS_FRAMES, n)))
    if len(ft_frames) >= 4:
        hip_std = _std([_get(metrics, i, "hip_centre_x") for i in ft_frames])
        head_std = _std([_get(metrics, i, "head_x") for i in ft_frames])
        measurements["S2_post_contact_instability_std"] = round(max(hip_std, head_std), 6)
    f6_window_frames = [
        phases.contact + i
        for i in range(1, F6_V2_POST_CONTACT_WINDOW + 1)
        if phases.contact + i < n
    ]
    if len(f6_window_frames) >= 3:
        measurements["F6_follow_through_frames"] = round(
            _avg([abs(_get(metrics, i, F6_V2_METRIC)) for i in f6_window_frames]),
            6,
        )

    drift_frames = 0
    for i in range(phases.backlift_start, min(phases.follow_through_start + 5, n)):
        hip_x = _get(metrics, i, "hip_centre_x")
        front_ankle_x = _get(metrics, i, "front_ankle_x")
        drift = abs(hip_x - front_ankle_x)
        drifting = hip_x < front_ankle_x if front_side == "left" else hip_x > front_ankle_x
        if drift > S3_HIP_DRIFT_TOLERANCE and drifting:
            drift_frames += 1
    measurements["S3_hip_drift_frames"] = drift_frames

    check_end = min(c + S4_POST_CONTACT_ROTATION_FRAMES, n - 1)
    if c < n - 2 and check_end < n:
        sh_change = _get(metrics, check_end, "shoulder_openness") - _get(metrics, c, "shoulder_openness")
        hip_change = _get(metrics, check_end, "hip_openness") - _get(metrics, c, "hip_openness")
        measurements["S4_post_contact_rotation_deg"] = round(max(sh_change, hip_change), 3)

    measurements["F1_sync_frames"] = abs(phases.hands_peak_vs_ffd_diff)

    if c > backlift_start + 2:
        vels = [_get(metrics, i, "wrist_velocity_y") for i in range(backlift_start, c + 1)]
        if len(vels) >= 4:
            direction_changes = 0
            for j in range(1, len(vels)):
                if vels[j - 1] * vels[j] < 0 and abs(vels[j]) > 0.004:
                    direction_changes += 1
            measurements["F2_velocity_direction_changes"] = direction_changes

    ref_frames = baseline.get("timing", {}).get("backlift_to_contact_frames", 15)
    if ref_frames > 0:
        measurements["F3_timing_ratio"] = round(phases.backlift_to_contact_frames / ref_frames, 6)

    if hp < n:
        pause_window = list(range(max(0, hp - 2), min(n, hp + 4)))
        if len(pause_window) >= 3:
            vels = [abs(_get(metrics, i, "wrist_velocity_y")) for i in pause_window]
            measurements["F4_pause_frames"] = sum(1 for v in vels if v < F4_PAUSE_VELOCITY_THRESHOLD)

        downswing_start = hp + 3
        if downswing_start < c - 1:
            vels = [abs(_get(metrics, i, "wrist_velocity_y")) for i in range(downswing_start, c)]
            if len(vels) >= 3:
                avg_vel = float(np.mean(vels)) if vels else 0.0
                if avg_vel >= 0.001:
                    hitch_threshold = avg_vel * F5_HITCH_VELOCITY_FRACTION
                    measurements["F5_hitch_frames"] = sum(1 for v in vels if v < hitch_threshold)
                else:
                    measurements["F5_hitch_frames"] = 0

    return measurements
