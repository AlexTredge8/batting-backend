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
    USE_SETUP_V1,
    USE_SETUP_V2,
    USE_SETUP_V4,
    SETUP_WINDOW_START_PCT,
    SETUP_PRE_HANDSTART_BUFFER,
    SETUP_SMOOTH_WINDOW,
    SETUP_MIN_LANDMARK_CHANGES,
    SETUP_SUSTAINED_FRAMES,
    SETUP_SEARCH_WINDOW_PCT,
    SETUP_SEARCH_FALLBACK_PCT,
    SETUP_STILLNESS_THRESHOLD,
    SETUP_MIN_STILLNESS_FRAMES,
    SETUP_MOTION_RISE_FRAMES,
    SETUP_V4_SEARCH_BEFORE_HANDS_PEAK,
    SETUP_V4_STILLNESS_THRESHOLD,
    SETUP_V4_MIN_STILL_FRAMES,
    BACKLIFT_WRIST_RISE_THRESHOLD,
    BACKLIFT_CONSECUTIVE_FRAMES,
    FRONT_ANKLE_LANDED_VEL_THRESHOLD,
    STRIDE_WIDTH_INCREASE_THRESHOLD,
    FRONT_ANKLE_Z_VEL_THRESHOLD,
    CONTACT_WINDOW_FRAMES,
    SYNC_TOLERANCE_FRAMES,
    USE_CONTACT_V1_ORIGINAL,
    USE_AUDIO_CONTACT,
    AUDIO_CONTACT_MIN_CONFIDENCE,
    USE_HANDS_PEAK_V1,
    USE_HANDS_PEAK_V3,
    HANDS_PEAK_WINDOW_START_PCT,
    HANDS_PEAK_WINDOW_END_PCT,
    HANDS_PEAK_MIN_RISE,
    HANDS_PEAK_SMOOTH_WINDOW,
    HANDS_PEAK_V3_SEARCH_BEFORE_CONTACT,
    HANDS_PEAK_V3_MIN_OFFSET,
    USE_FOLLOW_THROUGH_V1,
    FOLLOW_THROUGH_WINDOW_PCT,
    FOLLOW_THROUGH_POST_CONTACT_OFFSET,
    FOLLOW_THROUGH_SMOOTH_WINDOW,
    FOLLOW_THROUGH_MIN_RISE,
)

try:
    from audio_contact import detect_contact_from_video
except Exception:  # pragma: no cover - graceful fallback if optional deps are absent
    detect_contact_from_video = None

_SMALL = 1e-9
_CONTACT_CONSENSUS_WINDOW = 8
_CONTACT_CONFIDENCE_HIGH_SPAN = 3
_CONTACT_CONFIDENCE_MEDIUM_SPAN = 6


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
_SETUP_V2_SEED_FRAMES = 5
_SETUP_V2_SIGNAL_THRESHOLDS = {
    "knee_drop": {"delta": 0.0040, "span": 0.0045, "step": 0.0008},
    "hip_drop": {"delta": 0.0030, "span": 0.0035, "step": 0.0006},
    "shoulder_drop": {"delta": 0.0025, "span": 0.0030, "step": 0.0005},
    "forward_weight": {"delta": 0.0020, "span": 0.0025, "step": 0.0004},
}


def _setup_seed_indices(metrics: list[FrameMetrics], seed_frames: int) -> list[int]:
    indices = [
        idx
        for idx, metric in enumerate(metrics[:min(len(metrics), seed_frames * 3)])
        if metric.detected
    ]
    return indices[:seed_frames]


def _find_setup_end_v1(metrics: list[FrameMetrics]) -> int:
    """Original wrist-led setup detector retained as a fallback."""
    n = len(metrics)

    seed = [m for m in metrics[:min(_SETUP_SEED_FRAMES * 3, n)] if m.detected]
    if len(seed) < 3:
        return min(SETUP_MIN_STABLE_FRAMES, n - 10)

    seed = seed[:_SETUP_SEED_FRAMES]
    seed_wh_mean = float(np.mean([m.wrist_height for m in seed]))
    threshold_y = seed_wh_mean - BACKLIFT_WRIST_RISE_THRESHOLD
    consecutive = 0

    for i in range(1, n - 5):
        if metrics[i].wrist_height <= threshold_y:
            consecutive += 1
            if consecutive >= BACKLIFT_CONSECUTIVE_FRAMES:
                return max(0, i - consecutive)
        else:
            consecutive = 0

    return min(_SETUP_SEED_FRAMES * 2, n - 10)


def _build_setup_signal(
    metrics: list[FrameMetrics],
    attr: str,
    threshold_key: str,
    seed_indices: list[int],
) -> dict:
    series = [float(getattr(metric, attr)) for metric in metrics]
    seed_values = [series[idx] for idx in seed_indices]
    seed_deltas = [
        abs(series[curr] - series[prev])
        for prev, curr in zip(seed_indices, seed_indices[1:])
    ] or [0.0]

    threshold_floor = _SETUP_V2_SIGNAL_THRESHOLDS[threshold_key]
    baseline = float(np.mean(seed_values))
    motion_noise = float(np.median(seed_deltas))
    delta_threshold = max(threshold_floor["delta"], motion_noise * 4.0)
    step_threshold = max(threshold_floor["step"], motion_noise * 1.5)
    span_threshold = max(
        threshold_floor["span"],
        delta_threshold * 0.85,
        step_threshold * max(2.5, SETUP_SUSTAINED_FRAMES - 1),
    )

    pre_frames = max(1, SETUP_SUSTAINED_FRAMES - 2)
    post_frames = max(1, SETUP_SUSTAINED_FRAMES - pre_frames - 1)
    required_consistent_steps = max(2, min(3, SETUP_SUSTAINED_FRAMES - 2))
    active = [False] * len(metrics)

    for idx in range(pre_frames, len(metrics) - post_frames):
        start = idx - pre_frames
        end = idx + post_frames
        window = series[start : end + 1]
        steps = [window[j + 1] - window[j] for j in range(len(window) - 1)]
        span_change = window[-1] - window[0]
        if abs(span_change) < span_threshold:
            continue

        direction = 1.0 if span_change >= 0 else -1.0
        aligned_steps = [step * direction for step in steps]
        consistent_steps = sum(step >= step_threshold for step in aligned_steps)
        reversals = sum(step <= -(step_threshold * 0.5) for step in aligned_steps)
        displacement = abs(window[-1] - baseline)
        active[idx] = (
            displacement >= delta_threshold
            and consistent_steps >= required_consistent_steps
            and reversals == 0
        )

    return {
        "active": active,
        "baseline": baseline,
        "delta_threshold": delta_threshold,
        "step_threshold": step_threshold,
        "span_threshold": span_threshold,
    }


def _find_setup_end_v2(metrics: list[FrameMetrics]) -> int:
    """
    Detect setup from sustained multi-landmark stance changes rather than a
    single wrist-height threshold.
    """
    n = len(metrics)
    seed_indices = _setup_seed_indices(metrics, _SETUP_V2_SEED_FRAMES)
    if len(seed_indices) < 3:
        return _find_setup_end_v1(metrics)

    pre_frames = max(1, SETUP_SUSTAINED_FRAMES - 2)
    post_frames = max(1, SETUP_SUSTAINED_FRAMES - pre_frames - 1)
    signals = {
        "knee_drop": _build_setup_signal(metrics, "knee_mid_y", "knee_drop", seed_indices),
        "hip_drop": _build_setup_signal(metrics, "hip_mid_y", "hip_drop", seed_indices),
        "shoulder_drop": _build_setup_signal(metrics, "shoulder_mid_y", "shoulder_drop", seed_indices),
        "forward_weight": _build_setup_signal(metrics, "forward_weight", "forward_weight", seed_indices),
    }

    search_start = max(seed_indices[-1], SETUP_MIN_STABLE_FRAMES - 1, pre_frames)
    search_end = n - post_frames
    for idx in range(search_start, search_end):
        active_signals = [
            name
            for name, signal in signals.items()
            if signal["active"][idx]
        ]
        if len(active_signals) >= SETUP_MIN_LANDMARK_CHANGES:
            return idx

    return _find_setup_end_v1(metrics)


def _find_setup_end_v3_motion_onset(
    metrics: list[FrameMetrics],
) -> tuple[int, str]:
    """
    Setup v3 — motion-onset detector (F4.5, 2026-05-09).

    Setup frame = LAST frame within the search window where:
      (a) 5-frame rolling mean of |wrist_velocity_y| is below
          SETUP_STILLNESS_THRESHOLD, AND
      (b) the previous SETUP_MIN_STILLNESS_FRAMES frames are all also
          below the threshold (sustained stillness before this frame), AND
      (c) within the next SETUP_MOTION_RISE_FRAMES frames, the rolling
          mean rises above SETUP_STILLNESS_THRESHOLD (motion has begun).
          NOTE: this uses 1× not 2× the threshold — backlift velocities
          for slow-backlift Beginner batters barely exceed the still
          threshold, so a 2× gate would silently fail for the whole tier.

    Primary window:  frames 0 → SETUP_SEARCH_WINDOW_PCT  of video (default 70%).
    Fallback window: frames 0 → SETUP_SEARCH_FALLBACK_PCT of video (default 85%).
    Ultimate fallback: frame of minimum rolling velocity in fallback window,
      confidence="low".

    Returns (frame_idx, confidence):
      "high" — candidate found in primary window.
      "low"  — fallback or extended search path used.
    """
    n = len(metrics)
    half_w = 2  # 5-frame centred window = indices i-2 … i+2

    # Pre-compute 5-frame rolling mean of |wrist_velocity_y| for every frame.
    abs_vels = [abs(metrics[i].wrist_velocity_y) for i in range(n)]
    rolling_vel: list[float] = []
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        rolling_vel.append(float(np.mean(abs_vels[lo:hi])))

    thresh = SETUP_STILLNESS_THRESHOLD   # 0.005 by default
    k     = SETUP_MIN_STILLNESS_FRAMES   # 5 frames of sustained stillness
    m     = SETUP_MOTION_RISE_FRAMES     # 10 frames to see motion onset

    def _find_candidates(limit: int) -> list[int]:
        """All frames up to `limit` satisfying conditions (a)–(c)."""
        scan_limit = min(limit, n - m - 2)
        candidates = []
        for i in range(k, scan_limit):
            # (a) currently still
            if rolling_vel[i] >= thresh:
                continue
            # (b) sustained stillness over previous k frames
            if any(rolling_vel[j] >= thresh for j in range(i - k, i)):
                continue
            # (c) motion begins within next m frames (1× threshold, not 2×)
            if not any(rolling_vel[j] >= thresh
                       for j in range(i + 1, min(n, i + m + 1))):
                continue
            candidates.append(i)
        return candidates

    # Primary window
    primary_limit = max(k + m + 5, int(SETUP_SEARCH_WINDOW_PCT * n))
    candidates = _find_candidates(primary_limit)
    if candidates:
        return candidates[-1], "high"

    # Fallback window
    fallback_limit = max(primary_limit + 1, int(SETUP_SEARCH_FALLBACK_PCT * n))
    candidates = _find_candidates(fallback_limit)
    if candidates:
        return candidates[-1], "low"

    # Ultimate fallback: argmin of rolling velocity in fallback window
    ult_limit = max(1, min(fallback_limit, n - 3))
    min_idx = int(np.argmin(rolling_vel[:ult_limit]))
    return max(0, min_idx), "low"


def _find_setup_end_v4_stillness_anchored(
    metrics: list[FrameMetrics],
    hands_peak_frame: int,
) -> tuple[int, str]:
    """
    Setup v4 — hands-peak-anchored stillness detector (2026-05-11).

    Setup is the last stable address frame before the backlift.  Since
    hands_peak v3 is contact-anchored and no longer depends on setup, search
    backward from hands_peak for the last run of low multi-landmark motion.

    Motion score per frame =
      |wrist_velocity_y| + Δhip_openness + Δshoulder_openness

    Returns (frame_idx, confidence).
    """
    n = len(metrics)
    if n == 0:
        return 0, "low"

    hp = max(0, min(int(hands_peak_frame), n - 1))
    window_end = min(n - 1, hp - 2)
    window_start = max(0, hp - SETUP_V4_SEARCH_BEFORE_HANDS_PEAK)

    if window_end < window_start:
        return window_start, "low"

    motion_scores: list[float] = []
    for i in range(window_start, window_end + 1):
        curr = metrics[i]
        prev = metrics[i - 1] if i > 0 else curr
        score = (
            abs(curr.wrist_velocity_y)
            + abs(curr.hip_openness - prev.hip_openness)
            + abs(curr.shoulder_openness - prev.shoulder_openness)
        )
        motion_scores.append(float(score))

    smoothed: list[float] = []
    for i in range(len(motion_scores)):
        lo = max(0, i - 1)
        hi = min(len(motion_scores), i + 2)
        smoothed.append(float(np.mean(motion_scores[lo:hi])))

    min_run = max(1, SETUP_V4_MIN_STILL_FRAMES)
    last_run_end: int | None = None
    run_start: int | None = None

    for rel_idx, score in enumerate(smoothed):
        is_still = score < SETUP_V4_STILLNESS_THRESHOLD
        if is_still and run_start is None:
            run_start = rel_idx
        if not is_still and run_start is not None:
            if rel_idx - run_start >= min_run:
                last_run_end = rel_idx - 1
            run_start = None

    if run_start is not None and len(smoothed) - run_start >= min_run:
        last_run_end = len(smoothed) - 1

    if last_run_end is None:
        return window_start, "low"

    setup_frame = window_start + last_run_end
    confidence = "high" if setup_frame < hp - 5 else "low"
    return setup_frame, confidence




def detect_setup_v2(
    metrics: list[FrameMetrics],
    hands_start_up_frame: int,
    n_frames: int,
) -> tuple[int, str]:
    """
    Setup v2 — min-velocity detector (2026-05-13).

    Setup is the most physically still wrist-Y frame before hands start moving
    upward.  The available per-frame wrist-Y metric is wrist_height, which is
    the existing average wrist Y signal from metrics_calculator.py.
    """
    n = max(0, min(int(n_frames), len(metrics)))
    if n == 0:
        return 0, "low"

    hand_start = max(0, min(int(hands_start_up_frame), n - 1))
    start = int(SETUP_WINDOW_START_PCT * n)
    start = max(0, min(start, n - 1))
    end = max(hand_start - SETUP_PRE_HANDSTART_BUFFER, start + 3)
    end = max(start, min(end, n - 1))

    if end <= start:
        fallback = max(hand_start - 3, 0)
        return fallback, "low"

    wrist_y = [float(metrics[i].wrist_height) for i in range(start, end + 1)]
    if len(wrist_y) < 2:
        fallback = max(hand_start - 3, 0)
        return fallback, "low"

    deltas = [0.0]
    deltas.extend(abs(wrist_y[i] - wrist_y[i - 1]) for i in range(1, len(wrist_y)))

    window = max(1, int(SETUP_SMOOTH_WINDOW))
    half = window // 2
    smoothed: list[float] = []
    for rel_idx in range(len(deltas)):
        lo = max(0, rel_idx - half)
        hi = min(len(deltas), rel_idx + half + 1)
        smoothed.append(float(np.mean(deltas[lo:hi])))

    candidate = start + int(np.argmin(smoothed))
    if candidate <= hand_start - 3:
        return candidate, "high"

    return max(hand_start - 3, 0), "low"

def _find_setup_end(
    metrics: list[FrameMetrics],
    hands_peak_frame: int | None = None,
    hands_start_up_frame: int | None = None,
) -> tuple[int, str]:
    """
    Dispatch setup detection to the active version.

    Returns (frame_idx, confidence).

    USE_SETUP_V1=True  → original wrist-position-threshold detector (v1).
    USE_SETUP_V2=True  → legacy multi-landmark sustained-change detector.
    USE_SETUP_V4=True  → hands-peak-anchored stillness detector (v4).
    default            → min-velocity detector (v2, 2026-05-13).
    """
    if USE_SETUP_V1:
        return _find_setup_end_v1(metrics), "high"
    if USE_SETUP_V4 and hands_peak_frame is not None:
        return _find_setup_end_v4_stillness_anchored(metrics, hands_peak_frame)
    if USE_SETUP_V2:
        return _find_setup_end_v2(metrics), "low"
    if hands_start_up_frame is not None:
        return detect_setup_v2(metrics, hands_start_up_frame, len(metrics))
    return _find_setup_end_v1(metrics), "low"


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


def _rolling_min(values: list[float], window: int) -> list[float]:
    """Centered rolling minimum for jitter suppression."""
    half = window // 2
    n = len(values)
    result = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(float(min(values[lo:hi])))
    return result


def _find_hands_peak_v1_internal(
    metrics: list[FrameMetrics],
    backlift_start: int,
) -> tuple[int, str]:
    """
    Hands Peak v1 — velocity sign-reversal primary, position-minimum fallback.
    Extracted as a standalone function so it can be called directly as a
    bootstrap for v3 (contact detection happens before v3 HP detection).

    Search window: backlift_start → backlift_start + 45% of remaining frames.
    Always returns confidence="high" (v1 has no plausibility gate).
    """
    n = len(metrics)
    max_search = max(15, int((n - backlift_start) * 0.45))
    search_end = min(backlift_start + max_search, n - 3)

    vys = [metrics[i].wrist_velocity_y for i in range(backlift_start, search_end + 1)]
    vys_smooth = _smooth_velocities(vys, window=5)

    for j in range(2, len(vys_smooth) - 1):
        prev = vys_smooth[j - 1]
        curr = vys_smooth[j]
        if prev < 0.0 and curr > 0.001:
            return backlift_start + j, "high"

    whs = [metrics[i].wrist_height for i in range(backlift_start, search_end + 1)]
    return backlift_start + int(np.argmin(whs)), "high"


def _find_hands_peak_v3_contact_anchored(
    metrics: list[FrameMetrics],
    contact_frame: int,
    contact_confidence: str,
    contact_diagnostics: dict,
    setup_end: int = 0,
) -> tuple[int, str]:
    """
    Hands Peak v3 — backward search from audio-resolved contact (F5b, 2026-05-10).

    Ground-truth analysis of all 17 videos shows that hands_peak occurs
    5–10 frames before contact (mean=6.9, std=1.4) across all tiers and
    shot types.  Contact is resolved by audio (MAE ~2 frames for 16/17
    videos), giving an anchor that is entirely independent of setup_end
    or backlift_start.

    Signal: argmin of 5-frame rolling-minimum of wrist_height in window.
    Window: [contact - HANDS_PEAK_V3_SEARCH_BEFORE_CONTACT,
             contact - HANDS_PEAK_V3_MIN_OFFSET]

    Confidence rules:
      "low" if:
        (a) contact was obtained via pose fallback (audio unavailable), OR
        (b) contact_confidence is "low", OR
        (c) candidate is at the very edge of the window (within 1 frame of
            window_start or window_end — suggests peak is outside window).
      "high" otherwise.

    Fallback: if the contact frame is invalid or window is degenerate,
      fall back to v2 setup-anchored search with confidence="low".
    """
    n = len(metrics)

    # --- Compute search window ---
    window_end   = contact_frame - HANDS_PEAK_V3_MIN_OFFSET
    window_start = contact_frame - HANDS_PEAK_V3_SEARCH_BEFORE_CONTACT

    # Clamp to valid frame range
    window_start = max(0, window_start)
    window_end   = min(n - 3, window_end)

    # Guard: degenerate or invalid window → fall back to v2
    if (contact_frame < 0 or contact_frame >= n
            or window_end <= window_start):
        # v2 fallback
        w_start = max(0, setup_end + int(HANDS_PEAK_WINDOW_START_PCT * n))
        w_end   = min(n - 3, setup_end + int(HANDS_PEAK_WINDOW_END_PCT * n))
        if w_end <= w_start:
            w_end = min(n - 3, w_start + 15)
        whs        = [metrics[i].wrist_height for i in range(w_start, w_end + 1)]
        whs_smooth = _rolling_min(whs, HANDS_PEAK_SMOOTH_WINDOW)
        candidate  = w_start + int(np.argmin(whs_smooth))
        return candidate, "low"

    # --- Primary signal: argmin of smoothed wrist_height in window ---
    whs        = [metrics[i].wrist_height for i in range(window_start, window_end + 1)]
    whs_smooth = _rolling_min(whs, HANDS_PEAK_SMOOTH_WINDOW)
    rel_idx    = int(np.argmin(whs_smooth))
    candidate  = window_start + rel_idx

    # --- Confidence evaluation ---
    # (a) audio availability: pose_fallback method means audio was unavailable
    contact_method  = contact_diagnostics.get("method", "")
    audio_failed    = contact_method == "pose_fallback"

    # (b) contact itself was low-confidence
    contact_low     = contact_confidence == "low"

    # (c) candidate is at the window edge (peak outside window)
    at_edge = (rel_idx == 0 or rel_idx >= len(whs) - 1)

    confidence = "low" if (audio_failed or contact_low or at_edge) else "high"

    return candidate, confidence


def _find_hands_peak(
    metrics: list[FrameMetrics],
    backlift_start: int,
    setup_end: int = 0,
) -> tuple[int, str]:
    """
    Dispatch to the active hands_peak detector version.

    Returns (frame_idx, confidence) where confidence is "high" or "low".

    v1 (USE_HANDS_PEAK_V1=True): velocity sign-reversal primary, position fallback.
        Search window: backlift_start → backlift_start + 45% of remaining frames.

    v3 (USE_HANDS_PEAK_V3=True, USE_HANDS_PEAK_V1=False): contact-anchored backward
        search. Called directly from detect_phases() after contact is resolved —
        this dispatch branch is NOT used for v3 (only v1/v2 go through here).

    v2 (default): adaptive position-minimum primary with rolling-minimum smoothing.
        Search window: setup_end + 5% of total frames → setup_end + 45% of total frames.
        Plausibility check: wrist must rise ≥ HANDS_PEAK_MIN_RISE above setup baseline.
    """
    n = len(metrics)

    if USE_HANDS_PEAK_V1:
        return _find_hands_peak_v1_internal(metrics, backlift_start)

    # v2: adaptive window anchored to setup_end + percentage of total video
    w_start = max(0, setup_end + int(HANDS_PEAK_WINDOW_START_PCT * n))
    w_end   = min(n - 3, setup_end + int(HANDS_PEAK_WINDOW_END_PCT * n))

    # Guard: window must be non-degenerate
    if w_end <= w_start:
        w_end = min(n - 3, w_start + 15)

    whs        = [metrics[i].wrist_height for i in range(w_start, w_end + 1)]
    whs_smooth = _rolling_min(whs, HANDS_PEAK_SMOOTH_WINDOW)
    candidate  = w_start + int(np.argmin(whs_smooth))

    # Plausibility: wrist must have risen at least HANDS_PEAK_MIN_RISE above setup
    setup_wh     = metrics[min(setup_end, n - 1)].wrist_height
    candidate_wh = metrics[candidate].wrist_height
    confidence   = "high" if (candidate_wh <= setup_wh - HANDS_PEAK_MIN_RISE) else "low"

    return candidate, confidence


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


def _find_contact_with_diagnostics(
    metrics: list[FrameMetrics],
    hands_peak: int,
) -> tuple[int, dict]:
    if USE_CONTACT_V1_ORIGINAL:
        return _find_contact_with_diagnostics_original(metrics, hands_peak)
    return _find_contact_with_diagnostics_cleaned(metrics, hands_peak)


def _find_contact_with_diagnostics_original(
    metrics: list[FrameMetrics],
    hands_peak: int,
) -> tuple[int, dict]:
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
        candidate = max(0, n - 3)
        return candidate, {
            "mode": "insufficient_frames",
            "search_start": search_start,
            "search_end": search_end,
            "candidate": candidate,
        }

    vys = [metrics[i].wrist_velocity_y for i in range(search_start, search_end + 1)]
    vys_smooth = _smooth_velocities(vys, window=3)

    # Find first sign reversal: positive → negative (contact = wrists start rising).
    # Return j-1 (the LAST downswing frame = true contact, not first upswing frame).
    for j in range(1, len(vys_smooth) - 1):
        prev = vys_smooth[j - 1]
        curr = vys_smooth[j]
        if prev > 0.002 and curr < 0.0:
            candidate = search_start + max(0, j - 1)
            return candidate, {
                "mode": "sign_reversal",
                "search_start": search_start,
                "search_end": search_end,
                "candidate": candidate,
                "smoothed_velocity_prev": round(float(prev), 6),
                "smoothed_velocity_curr": round(float(curr), 6),
            }

    # Fallback: local maximum of wrist_height (physical lowest point)
    whs = [metrics[i].wrist_height for i in range(search_start, search_end + 1)]
    if whs:
        candidate = search_start + int(np.argmax(whs))
        return candidate, {
            "mode": "wrist_height_fallback",
            "search_start": search_start,
            "search_end": search_end,
            "candidate": candidate,
            "fallback_reason": "no_clear_velocity_sign_reversal",
        }

    candidate = min(hands_peak + 10, n - 2)
    return candidate, {
        "mode": "empty_search_window",
        "search_start": search_start,
        "search_end": search_end,
        "candidate": candidate,
    }


def _find_contact_with_diagnostics_cleaned(
    metrics: list[FrameMetrics],
    hands_peak: int,
) -> tuple[int, dict]:
    """
    Cleaned contact finder that keeps the wrist-led signal but widens the
    search to be less fragile when Hands Peak is detected too early.
    """
    n = len(metrics)
    hands_peak_start = hands_peak + 2
    search_start_fallback = int(0.35 * n)
    search_start = max(hands_peak_start, search_start_fallback)
    search_end = min(hands_peak + 80, n - 2)

    if search_start >= n - 1 or search_start > search_end:
        candidate = max(0, min(n - 3, max(hands_peak + 10, search_start_fallback)))
        return candidate, {
            "mode": "insufficient_frames",
            "search_start": search_start,
            "search_end": search_end,
            "candidate": candidate,
            "hands_peak_start": hands_peak_start,
            "search_start_fallback": search_start_fallback,
        }

    vys = [metrics[i].wrist_velocity_y for i in range(search_start, search_end + 1)]
    vys_smooth = _smooth_velocities(vys, window=3)

    for j in range(1, len(vys_smooth) - 1):
        prev = vys_smooth[j - 1]
        curr = vys_smooth[j]
        if prev > 0.005 and curr < 0.0:
            candidate = search_start + max(0, j - 1)
            return candidate, {
                "mode": "sign_reversal",
                "search_start": search_start,
                "search_end": search_end,
                "candidate": candidate,
                "hands_peak_start": hands_peak_start,
                "search_start_fallback": search_start_fallback,
                "smoothed_velocity_prev": round(float(prev), 6),
                "smoothed_velocity_curr": round(float(curr), 6),
            }

    whs = [metrics[i].wrist_height for i in range(search_start, search_end + 1)]
    if whs:
        candidate = search_start + int(np.argmax(whs))
        return candidate, {
            "mode": "wrist_height_fallback",
            "search_start": search_start,
            "search_end": search_end,
            "candidate": candidate,
            "hands_peak_start": hands_peak_start,
            "search_start_fallback": search_start_fallback,
            "fallback_reason": "no_clear_velocity_sign_reversal",
        }

    candidate = min(hands_peak + 10, n - 2)
    return candidate, {
        "mode": "empty_search_window",
        "search_start": search_start,
        "search_end": search_end,
        "candidate": candidate,
        "hands_peak_start": hands_peak_start,
        "search_start_fallback": search_start_fallback,
    }


def _resolve_contact_consensus(
    metrics: list[FrameMetrics],
    hands_peak: int,
) -> tuple[int, str, dict]:
    if USE_CONTACT_V1_ORIGINAL:
        return _resolve_contact_consensus_original(metrics, hands_peak)
    return _resolve_contact_consensus_cleaned(metrics, hands_peak)


def _resolve_contact_consensus_original(
    metrics: list[FrameMetrics],
    hands_peak: int,
) -> tuple[int, str, dict]:
    """
    Resolve contact from three signals:
      A: wrist velocity reversal / wrist-height fallback anchor
      B: front elbow maximum extension
      C: wrist speed minimum after the peak downswing speed
    """
    n = len(metrics)
    signal_a, signal_a_diag = _find_contact_with_diagnostics_original(metrics, hands_peak)

    window_start = max(hands_peak + 2, signal_a - _CONTACT_CONSENSUS_WINDOW)
    window_end = min(n - 2, signal_a + _CONTACT_CONSENSUS_WINDOW)
    if window_start > window_end:
        window_start = max(0, min(signal_a, n - 2))
        window_end = window_start

    def _window_items(start: int, end: int) -> list[int]:
        return list(range(max(0, start), min(n - 2, end) + 1))

    window_items = _window_items(window_start, window_end)
    if not window_items:
        window_items = [max(0, min(signal_a, n - 2))]

    signal_b = max(window_items, key=lambda i: (metrics[i].front_elbow_angle, -i))
    signal_b_angle = metrics[signal_b].front_elbow_angle

    speed_start = max(window_start, hands_peak + 1)
    speed_items = _window_items(speed_start, window_end) or window_items
    peak_speed_idx = max(
        speed_items,
        key=lambda i: (metrics[i].wrist_speed, -i),
    )

    decel_items = _window_items(peak_speed_idx, window_end) or [peak_speed_idx]
    signal_c = min(
        decel_items,
        key=lambda i: (metrics[i].wrist_speed, i),
    )
    signal_c_speed = metrics[signal_c].wrist_speed

    candidates = [signal_a, signal_b, signal_c]
    sorted_candidates = sorted(candidates)
    contact = sorted_candidates[1]
    span = max(candidates) - min(candidates)
    if span <= _CONTACT_CONFIDENCE_HIGH_SPAN:
        confidence = "high"
    elif span <= _CONTACT_CONFIDENCE_MEDIUM_SPAN:
        confidence = "medium"
    else:
        confidence = "low"

    diagnostics = {
        "window": {"start": window_start, "end": window_end},
        "signals": {
            "wrist_velocity_reversal": {
                "frame": signal_a,
                "mode": signal_a_diag.get("mode"),
                "candidate": signal_a_diag.get("candidate", signal_a),
                "search_start": signal_a_diag.get("search_start"),
                "search_end": signal_a_diag.get("search_end"),
            },
            "front_elbow_target": {
                "frame": signal_b,
                "angle_deg": round(float(signal_b_angle), 3),
                "selection": "max_extension",
            },
            "wrist_speed_decel": {
                "frame": signal_c,
                "peak_speed_frame": peak_speed_idx,
                "speed": round(float(signal_c_speed), 6),
            },
        },
        "candidates": candidates,
        "span": span,
        "chosen": contact,
        "confidence": confidence,
    }
    return contact, confidence, diagnostics


def _resolve_contact_consensus_cleaned(
    metrics: list[FrameMetrics],
    hands_peak: int,
) -> tuple[int, str, dict]:
    """
    Resolve contact from two wrist-led signals:
      A: wrist velocity reversal / wrist-height fallback anchor
      C: wrist speed minimum after the peak downswing speed

    Signal C is retained as a quality check, but Signal A remains the chosen
    contact even when the signals disagree.
    """
    n = len(metrics)
    signal_a, signal_a_diag = _find_contact_with_diagnostics_cleaned(metrics, hands_peak)

    window_start = max(hands_peak + 2, signal_a - _CONTACT_CONSENSUS_WINDOW)
    window_end = min(n - 2, signal_a + _CONTACT_CONSENSUS_WINDOW)
    if window_start > window_end:
        window_start = max(0, min(signal_a, n - 2))
        window_end = window_start

    def _window_items(start: int, end: int) -> list[int]:
        return list(range(max(0, start), min(n - 2, end) + 1))

    window_items = _window_items(window_start, window_end)
    if not window_items:
        window_items = [max(0, min(signal_a, n - 2))]

    speed_start = max(window_start, hands_peak + 1)
    speed_items = _window_items(speed_start, window_end) or window_items
    peak_speed_idx = max(
        speed_items,
        key=lambda i: (metrics[i].wrist_speed, -i),
    )

    decel_items = _window_items(peak_speed_idx, window_end) or [peak_speed_idx]
    signal_c = min(
        decel_items,
        key=lambda i: (metrics[i].wrist_speed, i),
    )
    signal_c_speed = metrics[signal_c].wrist_speed

    disagreement = abs(signal_a - signal_c)
    contact = signal_a
    if disagreement <= 3:
        confidence = "high"
    elif disagreement <= 8:
        confidence = "medium"
    else:
        confidence = "low"

    diagnostics = {
        "window": {"start": window_start, "end": window_end},
        "signals": {
            "wrist_velocity_reversal": {
                "frame": signal_a,
                "mode": signal_a_diag.get("mode"),
                "candidate": signal_a_diag.get("candidate", signal_a),
                "search_start": signal_a_diag.get("search_start"),
                "search_end": signal_a_diag.get("search_end"),
            },
            "wrist_speed_decel": {
                "frame": signal_c,
                "peak_speed_frame": peak_speed_idx,
                "speed": round(float(signal_c_speed), 6),
                "selection": "quality_check_only",
            },
        },
        "candidates": [signal_a, signal_c],
        "span": disagreement,
        "chosen": contact,
        "confidence": confidence,
    }
    return contact, confidence, diagnostics


def _resolve_contact_anchor(
    metrics: list[FrameMetrics],
    hands_peak: int,
    fps: float,
    video_path: str | None = None,
) -> tuple[int, str, dict]:
    n = len(metrics)
    n_original_frames = max(n, (_metric_index_to_orig_frame(metrics, n - 1) + 1) if n else 0)
    pose_contact, pose_confidence, pose_diagnostics = _resolve_contact_consensus(metrics, hands_peak)
    pose_contact = max(hands_peak + 2, min(pose_contact, n - 3))

    if not USE_AUDIO_CONTACT:
        return pose_contact, pose_confidence, pose_diagnostics

    if detect_contact_from_video is None:
        pose_diagnostics = dict(pose_diagnostics or {})
        pose_diagnostics["method"] = "pose_fallback"
        pose_diagnostics["reason"] = "audio_module_unavailable"
        pose_diagnostics["audio_confidence"] = 0.0
        return pose_contact, pose_confidence, pose_diagnostics

    audio_original_frame, audio_confidence, audio_diagnostics = detect_contact_from_video(
        video_path,
        fps,
        n_original_frames,
    )
    audio_frame = None
    if audio_original_frame is not None:
        audio_frame = _nearest_metric_index_for_orig_frame(metrics, int(audio_original_frame))
        audio_frame = max(0, min(int(audio_frame), n - 3))

    if audio_frame is not None and audio_confidence >= 0.8:
        return audio_frame, "high", {
            "method": "audio",
            "reason": "high_confidence_audio_onset",
            "audio_confidence": float(audio_confidence),
            "audio_frame": audio_frame,
            "audio_original_frame": int(audio_original_frame),
            "pose_candidate": pose_contact,
            "pose_confidence": pose_confidence,
            "window": {
                "start": max(0, audio_frame - CONTACT_WINDOW_FRAMES),
                "end": min(n - 1, audio_frame + CONTACT_WINDOW_FRAMES),
            },
            "signals": {
                "audio_onset": {
                    "frame": audio_frame,
                    "original_frame": int(audio_original_frame),
                    "confidence": round(float(audio_confidence), 3),
                },
                "pose_consensus": {
                    "frame": pose_contact,
                    "confidence": pose_confidence,
                },
            },
            "candidates": [audio_frame, pose_contact],
            "span": abs(audio_frame - pose_contact),
            "chosen": audio_frame,
            "audio_diagnostics": audio_diagnostics,
            "pose_diagnostics": pose_diagnostics,
        }

    if audio_frame is not None and AUDIO_CONTACT_MIN_CONFIDENCE <= audio_confidence < 0.8:
        agreement = abs(audio_frame - pose_contact)
        if agreement <= 15:
            method = "audio_pose_agree"
            confidence = "medium"
            reason = "mid_confidence_audio_matches_pose"
            resolved_audio_confidence = float(audio_confidence)
        else:
            method = "audio_preferred"
            confidence = "medium"
            reason = "mid_confidence_audio_preferred_over_pose"
            resolved_audio_confidence = 0.6
        return audio_frame, confidence, {
            "method": method,
            "reason": reason,
            "audio_confidence": float(resolved_audio_confidence),
            "audio_frame": audio_frame,
            "audio_original_frame": int(audio_original_frame),
            "pose_candidate": pose_contact,
            "pose_confidence": pose_confidence,
            "window": {
                "start": max(0, audio_frame - CONTACT_WINDOW_FRAMES),
                "end": min(n - 1, audio_frame + CONTACT_WINDOW_FRAMES),
            },
            "signals": {
                "audio_onset": {
                    "frame": audio_frame,
                    "original_frame": int(audio_original_frame),
                    "confidence": round(float(resolved_audio_confidence), 3),
                },
                "pose_consensus": {
                    "frame": pose_contact,
                    "confidence": pose_confidence,
                },
            },
            "candidates": [audio_frame, pose_contact],
            "span": agreement,
            "chosen": audio_frame,
            "audio_diagnostics": audio_diagnostics,
            "pose_diagnostics": pose_diagnostics,
        }

    pose_diagnostics = dict(pose_diagnostics or {})
    pose_diagnostics["method"] = "pose_fallback"
    pose_diagnostics["reason"] = (
        audio_diagnostics.get("status")
        or (audio_diagnostics.get("extract") or {}).get("status")
        or "audio_unavailable"
    )
    pose_diagnostics["audio_confidence"] = float(audio_confidence or 0.0)
    pose_diagnostics["audio_frame"] = audio_frame
    pose_diagnostics["audio_original_frame"] = audio_original_frame
    pose_diagnostics["audio_diagnostics"] = audio_diagnostics
    pose_diagnostics["pose_candidate"] = pose_contact
    return pose_contact, pose_confidence, pose_diagnostics


def _metric_index_to_orig_frame(metrics: list[FrameMetrics], metric_idx: int) -> int:
    if not metrics:
        return 0
    idx = max(0, min(metric_idx, len(metrics) - 1))
    return int(metrics[idx].frame_idx)


def _nearest_metric_index_for_orig_frame(metrics: list[FrameMetrics], orig_frame: int) -> int:
    """Return the metric index whose original frame is closest to orig_frame."""
    if not metrics:
        return 0
    best_idx = 0
    best_distance = None
    for idx, metric in enumerate(metrics):
        distance = abs(int(metric.frame_idx) - int(orig_frame))
        if best_distance is None or distance < best_distance:
            best_idx = idx
            best_distance = distance
    return best_idx


def _rebuild_phase_labels(phases: PhaseResult, metrics: list[FrameMetrics]) -> list[BattingPhase]:
    n = len(metrics)
    if n == 0:
        return []

    setup_end = max(0, min(phases.setup_end, n - 1))
    backlift_start = max(setup_end, min(phases.backlift_start, n - 1))
    hands_peak = max(backlift_start, min(phases.hands_peak, n - 1))
    front_foot_down = max(backlift_start, min(phases.front_foot_down, n - 1))
    contact = max(0, min(phases.contact, n - 1))
    follow_through_start = max(contact, min(phases.follow_through_start, n - 1))

    labels = [BattingPhase.UNKNOWN] * n
    for i in range(0, setup_end + 1):
        labels[i] = BattingPhase.SETUP
    for i in range(backlift_start, hands_peak):
        labels[i] = BattingPhase.BACKLIFT_STARTS
    if 0 <= hands_peak < n:
        labels[hands_peak] = BattingPhase.HANDS_PEAK
    if 0 <= front_foot_down < n:
        labels[front_foot_down] = BattingPhase.FRONT_FOOT_DOWN

    cw_lo = max(0, contact - CONTACT_WINDOW_FRAMES)
    cw_hi = min(n, contact + CONTACT_WINDOW_FRAMES + 1)
    for i in range(cw_lo, cw_hi):
        labels[i] = BattingPhase.CONTACT
    for i in range(follow_through_start, n):
        labels[i] = BattingPhase.FOLLOW_THROUGH
    return labels





def _find_follow_through_v1(contact_frame: int, n_frames: int) -> tuple[int, str]:
    """Original follow-through detector: fixed offset after contact window."""
    if n_frames <= 0:
        return 0, "low"
    return max(0, min(int(contact_frame) + CONTACT_WINDOW_FRAMES + 1, n_frames - 1)), "high"


def detect_follow_through_v2(
    metrics: list[FrameMetrics],
    contact_frame: int,
    n_frames: int,
) -> tuple[int, str]:
    """
    Follow-through v2 — post-contact wrist-height peak (2026-05-13).

    The available wrist-Y signal is metrics[i].wrist_height. Smaller image Y
    means the hands are physically higher, so the post-contact minimum is the
    follow-through peak.
    """
    n = max(0, min(int(n_frames), len(metrics)))
    if n == 0:
        return 0, "low"

    contact = max(0, min(int(contact_frame), n - 1))
    fallback = max(0, min(contact + FOLLOW_THROUGH_POST_CONTACT_OFFSET, n - 1))
    start = fallback
    end = min(contact + int(FOLLOW_THROUGH_WINDOW_PCT * n), n - 1)
    start = max(0, min(start, n - 1))
    end = max(0, min(end, n - 1))

    if end - start + 1 < 3:
        return fallback, "low"

    wrist_y = [float(metrics[i].wrist_height) for i in range(start, end + 1)]
    window = max(1, int(FOLLOW_THROUGH_SMOOTH_WINDOW))
    smoothed: list[float] = []
    for rel_idx in range(len(wrist_y)):
        lo = max(0, rel_idx - window + 1)
        hi = rel_idx + 1
        smoothed.append(float(min(wrist_y[lo:hi])))

    candidate = start + int(np.argmin(smoothed))
    contact_y = float(metrics[contact].wrist_height)
    candidate_y = float(metrics[candidate].wrist_height)
    confidence = "high" if candidate_y <= contact_y - FOLLOW_THROUGH_MIN_RISE else "low"
    return candidate, confidence


def _find_follow_through(
    metrics: list[FrameMetrics],
    contact_frame: int,
    n_frames: int,
) -> tuple[int, str]:
    if USE_FOLLOW_THROUGH_V1:
        return _find_follow_through_v1(contact_frame, n_frames)
    return detect_follow_through_v2(metrics, contact_frame, n_frames)

def enforce_anchor_ordering(phases: PhaseResult, metrics: list[FrameMetrics]) -> PhaseResult:
    """Enforce monotonic anchor ordering after all detectors have run."""
    n = len(metrics)
    ordered = [
        ("setup_frame", "setup_end", "setup_confidence"),
        ("hands_start_up_frame", "backlift_start", None),
        ("hands_peak_frame", "hands_peak", "hands_peak_confidence"),
        ("front_foot_down_frame", "front_foot_down", None),
        ("contact_frame", "contact", "contact_confidence"),
        ("follow_through_frame", "follow_through_start", None),
    ]
    nudges: list[dict] = []

    contact_method = str((phases.contact_diagnostics or {}).get("method") or "")
    protect_contact = contact_method.startswith("audio")

    for prev_item, curr_item in zip(ordered, ordered[1:]):
        prev_name, prev_attr, prev_conf = prev_item
        curr_name, curr_attr, conf_attr = curr_item
        prev_value = int(getattr(phases, prev_attr))
        curr_value = int(getattr(phases, curr_attr))
        if curr_value > prev_value:
            continue

        # Preserve the stronger downstream anchors and move their less reliable
        # predecessors back when ordering is violated.
        if curr_name == "hands_peak_frame":
            old_prev = prev_value
            new_prev = max(0, curr_value - 1)
            setattr(phases, prev_attr, new_prev)
            nudges.append({
                "anchor": prev_name,
                "from": old_prev,
                "to": new_prev,
                "delta": new_prev - old_prev,
                "reason": f"preserve hands_peak at {curr_value}",
            })
            continue

        if curr_name == "contact_frame" and protect_contact:
            old_prev = prev_value
            new_prev = max(0, curr_value - 1)
            setattr(phases, prev_attr, new_prev)
            nudges.append({
                "anchor": prev_name,
                "from": old_prev,
                "to": new_prev,
                "delta": new_prev - old_prev,
                "reason": f"preserve audio contact at {curr_value}",
            })
            continue

        old_value = curr_value
        new_value = min(n - 1, prev_value + 1)
        setattr(phases, curr_attr, new_value)
        nudges.append({
            "anchor": curr_name,
            "from": old_value,
            "to": new_value,
            "delta": new_value - old_value,
            "reason": f"must follow {prev_name}",
        })

    phases.ordering_guard_log = nudges
    phases.anchor_confidence_overrides = {nudge["anchor"]: "low" for nudge in nudges}
    phases.hands_peak_vs_ffd_diff = phases.hands_peak - phases.front_foot_down
    phases.hands_peak_vs_ffd_ms = round((phases.hands_peak_vs_ffd_diff / (phases.fps or 30.0)) * 1000, 1)
    phases.backlift_to_contact_frames = phases.contact - phases.backlift_start
    phases.follow_through_start = min(
        n - 1,
        max(phases.follow_through_start, phases.contact + CONTACT_WINDOW_FRAMES + 1),
    )
    phases.phase_labels = _rebuild_phase_labels(phases, metrics)
    return phases

def apply_contact_override(
    phases: PhaseResult,
    metrics: list[FrameMetrics],
    contact_original_frame: int | None = None,
) -> PhaseResult:
    """
    Convert the automatically estimated contact into a resolved contact.

    If contact_original_frame is provided, the resolved contact is treated as a
    manual/validated frame. Otherwise the resolved contact simply mirrors the
    automatic estimate and remains labelled as estimated.
    """
    phases.estimated_contact_frame = int(phases.contact)
    phases.estimated_contact_original_frame = _metric_index_to_orig_frame(metrics, phases.contact)
    phases.estimated_contact_confidence = phases.contact_confidence
    phases.estimated_contact_candidates = dict(phases.contact_candidates or {})
    phases.estimated_contact_window = dict(phases.contact_window or {})
    phases.estimated_contact_diagnostics = dict(phases.contact_diagnostics or {})

    if contact_original_frame is None:
        resolved_metric_idx = int(phases.estimated_contact_frame)
        resolved_original_frame = int(phases.estimated_contact_original_frame)
        resolved_source = "auto"
        resolved_status = "estimated"
    else:
        resolved_metric_idx = _nearest_metric_index_for_orig_frame(metrics, int(contact_original_frame))
        resolved_original_frame = int(contact_original_frame)
        resolved_source = "manual"
        resolved_status = "validated"
        phases.contact_confidence = "validated"

    phases.contact = resolved_metric_idx
    phases.resolved_contact_frame = resolved_metric_idx
    phases.resolved_contact_original_frame = resolved_original_frame
    phases.resolved_contact_source = resolved_source
    phases.resolved_contact_status = resolved_status

    cw_lo = max(0, resolved_metric_idx - CONTACT_WINDOW_FRAMES)
    cw_hi = min(len(metrics), resolved_metric_idx + CONTACT_WINDOW_FRAMES + 1)
    phases.contact_window = {"start": cw_lo, "end": max(cw_lo, cw_hi - 1)}
    phases.backlift_to_contact_frames = resolved_metric_idx - phases.backlift_start
    if contact_original_frame is not None:
        phases.follow_through_start = min(
            len(metrics) - 1,
            max(0, resolved_metric_idx + CONTACT_WINDOW_FRAMES + 1),
        )
        phases.follow_through_confidence = "low"

    phases.phase_labels = _rebuild_phase_labels(phases, metrics)

    return phases


def apply_anchor_overrides(
    phases: PhaseResult,
    metrics: list[FrameMetrics],
    anchor_original_frames: dict[str, int | None] | None = None,
) -> PhaseResult:
    """
    Apply original-frame overrides for any of the six anchor frames.

    Supported keys:
      setup_frame, hands_start_up_frame, front_foot_down_frame,
      hands_peak_frame, contact_frame, follow_through_frame
    """
    if not anchor_original_frames:
        return apply_contact_override(phases, metrics, None)

    def _override_metric_idx(field: str, current_value: int) -> int:
        original_frame = anchor_original_frames.get(field)
        if original_frame is None:
            return current_value
        return _nearest_metric_index_for_orig_frame(metrics, int(original_frame))

    phases.setup_end = _override_metric_idx("setup_frame", phases.setup_end)
    phases.backlift_start = max(phases.setup_end, _override_metric_idx("hands_start_up_frame", phases.backlift_start))
    phases.front_foot_down = _override_metric_idx("front_foot_down_frame", phases.front_foot_down)
    phases.hands_peak = _override_metric_idx("hands_peak_frame", phases.hands_peak)

    phases.hands_peak_vs_ffd_diff = phases.hands_peak - phases.front_foot_down
    phases.hands_peak_vs_ffd_ms = round((phases.hands_peak_vs_ffd_diff / (phases.fps or 30.0)) * 1000, 1)

    phases = apply_contact_override(
        phases,
        metrics,
        contact_original_frame=anchor_original_frames.get("contact_frame"),
    )

    follow_through_override = anchor_original_frames.get("follow_through_frame")
    if follow_through_override is not None:
        phases.follow_through_start = _nearest_metric_index_for_orig_frame(metrics, int(follow_through_override))
        phases.phase_labels = _rebuild_phase_labels(phases, metrics)

    if anchor_original_frames.get("contact_frame") is not None:
        phases.contact_confidence = "validated"

    return phases


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_phases(
    metrics: list[FrameMetrics],
    fps: float,
    verbose: bool = False,
    video_path: str | None = None,
) -> PhaseResult:
    """
    Run the full phase detection state machine.
    Returns a PhaseResult with frame labels and key event indices.

    Args:
        verbose: if True, log diagnostic detail about phase detection decisions.
    """
    n = len(metrics)
    labels = [BattingPhase.UNKNOWN] * n

    # --- Setup bootstrap ---
    # v4 setup is anchored backward from hands_peak, but the legacy HP fallback
    # still needs an initial setup/backlift estimate. Keep that bootstrap on the
    # legacy path, then replace setup after HP is known.
    if USE_SETUP_V1 or USE_SETUP_V2:
        setup_end, setup_confidence = _find_setup_end(metrics, None)
    else:
        setup_end, setup_confidence = _find_setup_end_v1(metrics), "high"
    baseline  = _setup_baseline(metrics, setup_end)
    backlift_start = _find_backlift_start(metrics, setup_end, baseline)
    backlift_start = max(backlift_start, setup_end + 1)

    # --- Hands Peak and Contact ---
    # v3 path: contact must be resolved first (backward anchor).
    #   Audio contact is independent of HP, so bootstrapping with v1 HP only
    #   affects the pose-fallback contact path (1/17 videos in current dataset).
    # v1/v2 path: HP first, then contact (original order).
    if not USE_HANDS_PEAK_V1 and USE_HANDS_PEAK_V3:
        # Bootstrap HP via v1 for pose-contact fallback; audio ignores it.
        bootstrap_hp, _ = _find_hands_peak_v1_internal(metrics, backlift_start)
        bootstrap_hp = max(bootstrap_hp, backlift_start + 2)
        contact, contact_confidence, contact_diagnostics = _resolve_contact_anchor(
            metrics,
            bootstrap_hp,
            fps,
            video_path=video_path,
        )
        hands_peak, hands_peak_confidence = _find_hands_peak_v3_contact_anchored(
            metrics,
            contact,
            contact_confidence,
            contact_diagnostics,
            setup_end,
        )
    else:
        hands_peak, hands_peak_confidence = _find_hands_peak(metrics, backlift_start, setup_end)
        contact, contact_confidence, contact_diagnostics = _resolve_contact_anchor(
            metrics,
            hands_peak,
            fps,
            video_path=video_path,
        )

    # v3 anchors on contact (backward), so backlift_start (which is itself
    # setup-coupled and frequently detected AFTER the true HP) must NOT clamp
    # the v3 result upward.  The v3 window already guarantees
    # HP <= contact - HANDS_PEAK_V3_MIN_OFFSET.
    # For v1/v2 (forward search), the backlift guard remains.
    if USE_HANDS_PEAK_V1 or not USE_HANDS_PEAK_V3:
        hands_peak = max(hands_peak, backlift_start + 2)
    else:
        # v3 safety: don't let HP be at or past contact
        hands_peak = min(hands_peak, contact - 1)
        hands_peak = max(hands_peak, 0)
    # --- Final setup after hands_start_up bootstrap and resolved hands_peak ---
    setup_end, setup_confidence = _find_setup_end(
        metrics,
        hands_peak,
        hands_start_up_frame=backlift_start,
    )
    baseline = _setup_baseline(metrics, setup_end)
    backlift_start = _find_backlift_start(metrics, setup_end, baseline)
    backlift_start = max(setup_end + 1, min(backlift_start, hands_peak - 1))
    if verbose:
        print(f"  [phase] setup_end={setup_end} confidence={setup_confidence} (baseline wrist_h={baseline['wrist_height_mean']:.4f})")
        print(f"  [phase] backlift_start={backlift_start}")
        wh = metrics[hands_peak].wrist_height if hands_peak < n else 0
        print(f"  [phase] hands_peak={hands_peak} (wrist_h={wh:.4f}, confidence={hands_peak_confidence})")

    for i in range(0, setup_end + 1):
        labels[i] = BattingPhase.SETUP
    for i in range(setup_end + 1, min(backlift_start, n)):
        labels[i] = BattingPhase.SETUP
    for i in range(backlift_start, hands_peak):
        labels[i] = BattingPhase.BACKLIFT_STARTS
    if hands_peak < n:
        labels[hands_peak] = BattingPhase.HANDS_PEAK

    # --- Front Foot Down ---
    front_foot_down = _find_front_foot_down(metrics, backlift_start, hands_peak, baseline)
    front_foot_down = max(backlift_start + 2, min(front_foot_down, hands_peak + 10))
    if 0 <= front_foot_down < n:
        labels[front_foot_down] = BattingPhase.FRONT_FOOT_DOWN
    if verbose:
        print(f"  [phase] front_foot_down={front_foot_down}")
    if verbose:
        print(
            f"  [phase] contact={contact} "
            f"(confidence={contact_confidence}, candidates={contact_diagnostics['candidates']}, span={contact_diagnostics['span']})"
        )

    cw_lo = max(0, contact - CONTACT_WINDOW_FRAMES)
    cw_hi = min(n, contact + CONTACT_WINDOW_FRAMES + 1)
    for i in range(cw_lo, cw_hi):
        labels[i] = BattingPhase.CONTACT

    # Frames between Hands Peak and Contact (downswing)
    for i in range(hands_peak + 1, cw_lo):
        labels[i] = BattingPhase.HANDS_PEAK  # downswing, labelled as post-peak

    # --- Follow-Through ---
    follow_through_start, follow_through_confidence = _find_follow_through(metrics, contact, n)
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

    signals = contact_diagnostics.get("signals") or {}
    contact_candidates = {
        "spread": contact_diagnostics.get("span"),
        "method": contact_diagnostics.get("method"),
    }
    if "wrist_velocity_reversal" in signals:
        contact_candidates["c_a"] = signals["wrist_velocity_reversal"]["frame"]
    if "wrist_speed_decel" in signals:
        contact_candidates["c_c"] = signals["wrist_speed_decel"]["frame"]
    if "audio_onset" in signals:
        contact_candidates["audio"] = signals["audio_onset"]["frame"]
    if "pose_consensus" in signals:
        contact_candidates["pose"] = signals["pose_consensus"]["frame"]
    if "front_elbow_target" in signals:
        contact_candidates["c_b"] = signals["front_elbow_target"]["frame"]
        contact_candidates["selection_reason"] = "median_of_three_signal_consensus"
    elif "audio_onset" in signals:
        contact_candidates["selection_reason"] = contact_diagnostics.get("reason", "audio_primary")
    else:
        contact_candidates["selection_reason"] = "signal_a_primary_signal_c_quality_check"

    phases = PhaseResult(
        phase_labels=labels,
        setup_start=0,
        setup_end=setup_end,
        setup_confidence=setup_confidence,
        backlift_start=backlift_start,
        hands_peak=hands_peak,
        hands_peak_confidence=hands_peak_confidence,
        front_foot_down=front_foot_down,
        contact=contact,
        follow_through_start=follow_through_start,
        follow_through_confidence=follow_through_confidence,
        hands_peak_vs_ffd_diff=diff_frames,
        hands_peak_vs_ffd_ms=diff_ms,
        backlift_to_contact_frames=backlift_to_contact,
        fps=fps,
        contact_confidence=contact_confidence,
        contact_candidates=contact_candidates,
        contact_window=contact_diagnostics["window"],
        contact_diagnostics=contact_diagnostics,
    )
    phases = enforce_anchor_ordering(phases, metrics)
    return apply_contact_override(phases, metrics)


def print_phase_summary(phase_result: PhaseResult, fps: float) -> None:
    """Pretty-print phase detection results."""
    pr = phase_result
    def ft(f): return f"{f} ({f/fps:.2f}s)"

    print("\n--- Phase Detection ---")
    print(f"  Setup:            frames 0–{pr.setup_end}   ({ft(pr.setup_end)})")
    print(f"  Backlift starts:  frame  {ft(pr.backlift_start)}")
    print(f"  Hands Peak:       frame  {ft(pr.hands_peak)}")
    print(f"  Front Foot Down:  frame  {ft(pr.front_foot_down)}")
    print(f"  Contact:          frame  {ft(pr.contact)} [{pr.contact_confidence}]")
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
