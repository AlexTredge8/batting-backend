"""
BattingIQ Phase 2 — Phase Detector
=====================================
State machine that labels each frame with its batting phase.

Key insight (end-on camera):
  wrist_height (Y) decreasing = hands going UP (backlift)
  wrist_height (Y) increasing = hands going DOWN (downswing)
  wrist_velocity_y negative   = hands rising
  wrist_velocity_y positive   = hands falling / in downswing
  wrist_velocity_y sign +→-   = CONTACT (local max Y = lowest physical point
                                          before follow-through rise)
  wrist_velocity_y sign -→+   = HANDS PEAK (local min Y = highest physical point)

Phase sequence:
  SETUP → BACKLIFT_STARTS → [HANDS_PEAK] → [FRONT_FOOT_DOWN] → CONTACT → FOLLOW_THROUGH
  HANDS_PEAK and FRONT_FOOT_DOWN occur around the same time.
"""

import numpy as np
from models import BattingPhase, FrameMetrics, PhaseResult
from config import (
    SETUP_MIN_STABLE_FRAMES,
    BACKLIFT_WRIST_RISE_THRESHOLD,
    BACKLIFT_CONSECUTIVE_FRAMES,
    FRONT_ANKLE_LANDED_VEL_THRESHOLD,
    STRIDE_WIDTH_INCREASE_THRESHOLD,
    FRONT_ANKLE_Z_VEL_THRESHOLD,
    CONTACT_WINDOW_FRAMES,
    SYNC_TOLERANCE_FRAMES,
)

_SMALL = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_baseline(metrics: list[FrameMetrics], end_frame: int) -> dict:
    """Compute baseline statistics from the detected frames in [0..end_frame]."""
    frames = [m for m in metrics[:end_frame + 1] if m.detected]
    if not frames:
        frames = [metrics[0]]
    return {
        "wrist_height_mean":      float(np.mean([f.wrist_height      for f in frames])),
        "wrist_height_std":       float(np.std( [f.wrist_height      for f in frames])),
        "stance_width_mean":      float(np.mean([f.stance_width      for f in frames])),
        "head_offset_mean":       float(np.mean([f.head_offset       for f in frames])),
        "hip_centre_x_mean":      float(np.mean([f.hip_centre_x      for f in frames])),
        "shoulder_openness_mean": float(np.mean([f.shoulder_openness for f in frames])),
        "hip_openness_mean":      float(np.mean([f.hip_openness      for f in frames])),
    }


def _smooth_velocities(values: list[float], window: int = 3) -> list[float]:
    """Simple moving average."""
    out = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(float(np.mean(values[lo:hi])))
    return out


_SETUP_SEED_FRAMES = 15   # frames used to establish baseline before backlift search


def _find_setup_end(metrics: list[FrameMetrics]) -> int:
    """
    Setup = frames before the backlift begins.
    We fix the baseline from the first _SETUP_SEED_FRAMES detected frames,
    then find when the wrist first deviates significantly from that baseline.
    """
    n = len(metrics)

    # Collect seed frames
    seed = [m for m in metrics[:min(_SETUP_SEED_FRAMES * 3, n)] if m.detected]
    if len(seed) < 3:
        return min(SETUP_MIN_STABLE_FRAMES, n - 10)

    seed = seed[:_SETUP_SEED_FRAMES]
    seed_wh_mean = float(np.mean([m.wrist_height for m in seed]))

    # Setup ends when wrist height drops below (seed_mean - threshold) for
    # BACKLIFT_CONSECUTIVE_FRAMES consecutive frames
    threshold_y = seed_wh_mean - BACKLIFT_WRIST_RISE_THRESHOLD
    consecutive = 0

    for i in range(1, n - 5):
        if metrics[i].wrist_height <= threshold_y:
            consecutive += 1
            if consecutive >= BACKLIFT_CONSECUTIVE_FRAMES:
                # Setup ended at the start of this run
                return max(0, i - consecutive)
        else:
            consecutive = 0

    # Fallback: use seed size
    return min(_SETUP_SEED_FRAMES * 2, n - 10)


def _find_backlift_start(
    metrics: list[FrameMetrics],
    setup_end: int,
    baseline: dict,
) -> int:
    """
    Backlift starts = first frame after setup where wrist_velocity_y is
    consistently negative (hands rising) for BACKLIFT_CONSECUTIVE_FRAMES.
    """
    n = len(metrics)
    baseline_wh = baseline["wrist_height_mean"]
    threshold   = baseline_wh - BACKLIFT_WRIST_RISE_THRESHOLD  # smaller Y = higher

    consecutive = 0
    first_cand  = setup_end + 1

    for i in range(setup_end + 1, n - 5):
        vy = metrics[i].wrist_velocity_y
        above = metrics[i].wrist_height <= threshold
        rising = vy < -0.002   # definitely moving upward

        if above or rising:
            if consecutive == 0:
                first_cand = i
            consecutive += 1
            if consecutive >= BACKLIFT_CONSECUTIVE_FRAMES:
                return first_cand
        else:
            consecutive = 0

    return min(setup_end + 5, n - 10)


def _find_hands_peak(
    metrics: list[FrameMetrics],
    backlift_start: int,
) -> int:
    """
    Hands Peak = the local minimum of wrist_height (physical highest point)
    in the FIRST half of the remaining stroke.  We constrain the search to
    the first 45% of frames after backlift start so the follow-through peak
    (which is physically higher) doesn't dominate.

    If no clear minimum is found via velocity sign-reversal, fall back to
    the position minimum within the constrained window.
    """
    n = len(metrics)
    # Constrain search to first 45% of remaining frames after backlift
    max_search = max(15, int((n - backlift_start) * 0.45))
    search_end = min(backlift_start + max_search, n - 3)

    vys = [metrics[i].wrist_velocity_y for i in range(backlift_start, search_end + 1)]
    vys_smooth = _smooth_velocities(vys, window=5)

    # Find first sign reversal: any negative → any positive
    for j in range(2, len(vys_smooth) - 1):
        prev = vys_smooth[j - 1]
        curr = vys_smooth[j]
        if prev < 0.0 and curr > 0.001:
            return backlift_start + j

    # Fallback: minimum wrist_height (highest physical position) in window
    whs = [metrics[i].wrist_height for i in range(backlift_start, search_end + 1)]
    return backlift_start + int(np.argmin(whs))


def _find_front_foot_down(
    metrics: list[FrameMetrics],
    backlift_start: int,
    hands_peak: int,
    setup_baseline: dict,
) -> int:
    """
    FFD = the FIRST frame (after backlift begins) where:
      (a) stance_width has clearly grown from the setup baseline, AND
      (b) front_ankle_vy has dropped to near zero (foot just landed).

    Search window: backlift_start → hands_peak + 20.
    """
    n = len(metrics)
    baseline_sw = setup_baseline["stance_width_mean"]
    search_start = backlift_start
    search_end   = min(hands_peak + 20, n - 2)

    for i in range(search_start, search_end + 1):
        sw = metrics[i].stance_width
        vy = metrics[i].front_ankle_vy

        sw_grown   = (sw - baseline_sw) > STRIDE_WIDTH_INCREASE_THRESHOLD
        foot_quiet = abs(vy) < FRONT_ANKLE_LANDED_VEL_THRESHOLD

        if sw_grown and foot_quiet:
            return i

    # Fallback: find when stance is widest (peak of the stride)
    sws = [metrics[i].stance_width for i in range(search_start, search_end + 1)]
    if sws:
        return search_start + int(np.argmax(sws))
    return min(hands_peak + 5, n - 1)


def _find_contact(
    metrics: list[FrameMetrics],
    hands_peak: int,
) -> int:
    """
    Contact = first frame AFTER Hands Peak where wrist_velocity_y transitions
    from positive (hands descending) to negative (follow-through rising).
    This is the local MAXIMUM of wrist_height (bat at its lowest physical point).

    If no clear sign reversal, fall back to local max of wrist_height post-peak.
    """
    n = len(metrics)
    search_start = hands_peak + 2
    search_end   = min(hands_peak + 50, n - 2)

    if search_start >= n - 1:
        return n - 3

    vys = [metrics[i].wrist_velocity_y for i in range(search_start, search_end + 1)]
    vys_smooth = _smooth_velocities(vys, window=3)

    # Find first sign reversal: positive → negative (contact = wrists start rising).
    # Return j-1 (the LAST downswing frame = true contact, not first upswing frame).
    for j in range(1, len(vys_smooth) - 1):
        prev = vys_smooth[j - 1]
        curr = vys_smooth[j]
        if prev > 0.002 and curr < 0.0:
            return search_start + max(0, j - 1)

    # Fallback: local maximum of wrist_height (physical lowest point)
    whs = [metrics[i].wrist_height for i in range(search_start, search_end + 1)]
    if whs:
        return search_start + int(np.argmax(whs))

    return min(hands_peak + 10, n - 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_phases(metrics: list[FrameMetrics], fps: float) -> PhaseResult:
    """
    Run the full phase detection state machine.
    Returns a PhaseResult with frame labels and key event indices.
    """
    n = len(metrics)
    labels = [BattingPhase.UNKNOWN] * n

    # --- Setup ---
    setup_end = _find_setup_end(metrics)
    baseline  = _setup_baseline(metrics, setup_end)

    for i in range(0, setup_end + 1):
        labels[i] = BattingPhase.SETUP

    # --- Backlift starts ---
    backlift_start = _find_backlift_start(metrics, setup_end, baseline)
    backlift_start = max(backlift_start, setup_end + 1)

    # Frames between setup_end and backlift_start → extend setup label
    for i in range(setup_end + 1, min(backlift_start, n)):
        labels[i] = BattingPhase.SETUP

    # --- Hands Peak ---
    hands_peak = _find_hands_peak(metrics, backlift_start)
    hands_peak = max(hands_peak, backlift_start + 2)

    for i in range(backlift_start, hands_peak):
        labels[i] = BattingPhase.BACKLIFT_STARTS
    if hands_peak < n:
        labels[hands_peak] = BattingPhase.HANDS_PEAK

    # --- Front Foot Down ---
    front_foot_down = _find_front_foot_down(metrics, backlift_start, hands_peak, baseline)
    front_foot_down = max(backlift_start + 2, min(front_foot_down, hands_peak + 10))
    if 0 <= front_foot_down < n:
        labels[front_foot_down] = BattingPhase.FRONT_FOOT_DOWN

    # --- Contact ---
    contact = _find_contact(metrics, hands_peak)
    contact = max(hands_peak + 2, min(contact, n - 3))

    cw_lo = max(0, contact - CONTACT_WINDOW_FRAMES)
    cw_hi = min(n, contact + CONTACT_WINDOW_FRAMES + 1)
    for i in range(cw_lo, cw_hi):
        labels[i] = BattingPhase.CONTACT

    # Frames between Hands Peak and Contact (downswing)
    for i in range(hands_peak + 1, cw_lo):
        labels[i] = BattingPhase.HANDS_PEAK  # downswing, labelled as post-peak

    # --- Follow-Through ---
    follow_through_start = contact + CONTACT_WINDOW_FRAMES + 1
    for i in range(follow_through_start, n):
        labels[i] = BattingPhase.FOLLOW_THROUGH

    # Fill remaining UNKNOWN
    for i in range(n):
        if labels[i] == BattingPhase.UNKNOWN:
            labels[i] = labels[i - 1] if i > 0 else BattingPhase.SETUP

    # --- Timing ---
    diff_frames = hands_peak - front_foot_down   # + = peak after FFD = feet early
    diff_ms     = round(diff_frames / fps * 1000, 1)
    backlift_to_contact = contact - backlift_start

    return PhaseResult(
        phase_labels=labels,
        setup_start=0,
        setup_end=setup_end,
        backlift_start=backlift_start,
        hands_peak=hands_peak,
        front_foot_down=front_foot_down,
        contact=contact,
        follow_through_start=follow_through_start,
        hands_peak_vs_ffd_diff=diff_frames,
        hands_peak_vs_ffd_ms=diff_ms,
        backlift_to_contact_frames=backlift_to_contact,
        fps=fps,
    )


def print_phase_summary(phase_result: PhaseResult, fps: float) -> None:
    """Pretty-print phase detection results."""
    pr = phase_result
    def ft(f): return f"{f} ({f/fps:.2f}s)"

    print("\n--- Phase Detection ---")
    print(f"  Setup:            frames 0–{pr.setup_end}   ({ft(pr.setup_end)})")
    print(f"  Backlift starts:  frame  {ft(pr.backlift_start)}")
    print(f"  Hands Peak:       frame  {ft(pr.hands_peak)}")
    print(f"  Front Foot Down:  frame  {ft(pr.front_foot_down)}")
    print(f"  Contact:          frame  {ft(pr.contact)}")
    print(f"  Follow-Through:   frame  {ft(pr.follow_through_start)}+")

    diff    = pr.hands_peak_vs_ffd_diff
    abs_d   = abs(diff)
    if abs_d <= 2:
        sync_label = "IN SYNC"
    elif diff > 0:
        sync_label = f"Peak AFTER FFD by {abs_d} frames ({abs(pr.hands_peak_vs_ffd_ms):.0f}ms) — feet early"
    else:
        sync_label = f"Peak BEFORE FFD by {abs_d} frames ({abs(pr.hands_peak_vs_ffd_ms):.0f}ms) — hands late"

    print(f"  Sync (Peak vs FFD): {diff:+d} frames → {sync_label}")
    print(f"  Backlift→Contact:   {pr.backlift_to_contact_frames} frames "
          f"({pr.backlift_to_contact_frames / fps * 1000:.0f}ms)")
