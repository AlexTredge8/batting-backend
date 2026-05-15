"""
BattingIQ Phase 2 — Configuration
==================================
All thresholds, constants and tunable parameters in one place.
Adjust these based on reference baseline calibration.

Coordinate system (MediaPipe, normalised 0-1):
  X: screen left → right
  Y: screen top → bottom   (higher hands = smaller Y)
  Z: depth from camera (negative = closer to camera)
"""

import os

RULES_VERSION = "2.0.0"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default

# ---------------------------------------------------------------------------
# Camera / handedness
# ---------------------------------------------------------------------------
# Right-handed batter from bowler's end:
#   Front foot  = LEFT  ankle / knee / elbow
#   Back foot   = RIGHT ankle / knee / elbow
# Left-handed batter from bowler's end:
#   Front foot  = RIGHT ankle / knee / elbow
#   Back foot   = LEFT  ankle / knee / elbow
#
# Mapping:  handedness="right" → front_side="left"
#           handedness="left"  → front_side="right"
FRONT_SIDE = "left"           # default for right-handed batters
DEFAULT_HANDEDNESS = "right"  # used when API caller doesn't specify

# LOCAL_MODE=True keeps source resolution and processes every frame.
# LOCAL_MODE=False preserves the lighter Railway-style processing path.
LOCAL_MODE = _env_bool("LOCAL_MODE", False)

# Set True for deterministic frame-by-frame inference.
# False uses temporal tracking (non-deterministic across sessions).
# Changed 2026-05-13: cross-session determinism failure confirmed in D4.
# Source: temporal tracking state varies on session init.
MEDIAPIPE_STATIC_IMAGE_MODE = _env_bool("MEDIAPIPE_STATIC_IMAGE_MODE", True)

# ---------------------------------------------------------------------------
# Setup detector version
# ---------------------------------------------------------------------------
# USE_SETUP_V1 = True  → original wrist-position-threshold detector (v1)
# USE_SETUP_V4 = True  → hands-peak-anchored stillness detector (v4, restored 2026-05-14)
# USE_SETUP_V1 = False → min-velocity detector (v2, rejected 2026-05-14)
# USE_SETUP_V2 = True  → multi-landmark sustained-change detector (deprecated, was net worse)
USE_SETUP_V4 = _env_bool("USE_SETUP_V4", True)
USE_SETUP_V1 = _env_bool("USE_SETUP_V1", False)
# REVERTED 2026-05-14: v2 MAE 28.94 vs v4 MAE 8.29.
# Bootstrapped dependency chain (setup needs hands_start_up which needs setup)
# introduced compounding error. v4 stillness-anchored approach is superior.
# Do not re-attempt v2 without resolving the circular dependency cleanly.
USE_SETUP_V2 = _env_bool("USE_SETUP_V2", False)
SETUP_WINDOW_START_PCT = float(os.getenv("SETUP_WINDOW_START_PCT", "0.02"))
SETUP_PRE_HANDSTART_BUFFER = int(os.getenv("SETUP_PRE_HANDSTART_BUFFER", "2"))
SETUP_SMOOTH_WINDOW = int(os.getenv("SETUP_SMOOTH_WINDOW", "5"))
SETUP_MIN_LANDMARK_CHANGES = int(os.getenv("SETUP_MIN_LANDMARK_CHANGES", "3"))
SETUP_SUSTAINED_FRAMES = int(os.getenv("SETUP_SUSTAINED_FRAMES", "4"))
SETUP_V4_SEARCH_BEFORE_HANDS_PEAK = int(os.getenv("SETUP_V4_SEARCH_BEFORE_HANDS_PEAK", "40"))
# SETUP_V4_THRESHOLD TUNED 2026-05-11: raised from 0.02 to 0.65
# Reason: 6+ videos hit frame-0 fallback at 0.02;
# min observed motion in validated-setup windows was ~0.20-0.63.
SETUP_V4_STILLNESS_THRESHOLD = float(os.getenv("SETUP_V4_STILLNESS_THRESHOLD", "0.65"))
SETUP_V4_MIN_STILL_FRAMES = int(os.getenv("SETUP_V4_MIN_STILL_FRAMES", "2"))

# v3 motion-onset parameters.
# NOTE: spec suggested 0.30/0.50 primary/fallback, but dataset analysis shows truth
# setup ranges from 8%–68% of video duration; 0.70/0.85 covers all 17 videos.
# The spec's 2× motion threshold was also dropped: backlift velocities for slow
# Beginner batters barely exceed the stillness threshold, so a 1× condition is
# used (velocity simply exceeds the threshold, not 2× it).
SETUP_SEARCH_WINDOW_PCT   = float(os.getenv("SETUP_SEARCH_WINDOW_PCT",   "0.70"))
SETUP_SEARCH_FALLBACK_PCT = float(os.getenv("SETUP_SEARCH_FALLBACK_PCT", "0.85"))
SETUP_STILLNESS_THRESHOLD    = float(os.getenv("SETUP_STILLNESS_THRESHOLD",    "0.005"))
SETUP_MIN_STILLNESS_FRAMES   = int(  os.getenv("SETUP_MIN_STILLNESS_FRAMES",   "5"))
SETUP_MOTION_RISE_FRAMES     = int(  os.getenv("SETUP_MOTION_RISE_FRAMES",     "10"))

SETUP_DETECTOR_VERSION = (
    "v1_wrist_threshold" if USE_SETUP_V1
    else ("v4_stillness_anchored" if USE_SETUP_V4
          else ("v2_multi_landmark_legacy" if USE_SETUP_V2
                else "v2_min_velocity"))
)

# ---------------------------------------------------------------------------
# Hands Peak detector version
# ---------------------------------------------------------------------------
# USE_HANDS_PEAK_V1 = True  → original velocity-reversal detector (v1)
# USE_HANDS_PEAK_V1 = False, USE_HANDS_PEAK_V3 = True  → contact-anchored backward search (v3, F5b 2026-05-10)
# USE_HANDS_PEAK_V1 = False, USE_HANDS_PEAK_V3 = False → adaptive position-minimum detector (v2, deprecated)
#
# Priority rule: V1 wins if USE_HANDS_PEAK_V1=True.
#                Else V3 if USE_HANDS_PEAK_V3=True.
#                Else V2.
USE_HANDS_PEAK_V1 = _env_bool("USE_HANDS_PEAK_V1", False)  # v3 evaluated in F5b (2026-05-10)
USE_HANDS_PEAK_V3 = _env_bool("USE_HANDS_PEAK_V3", True)   # contact-anchored backward search (F5b)

# v2 adaptive window: percentage of total frames relative to setup_end
HANDS_PEAK_WINDOW_START_PCT = 0.05   # search begins 5% of video after setup
HANDS_PEAK_WINDOW_END_PCT   = 0.45   # search ends  45% of video after setup
HANDS_PEAK_MIN_RISE         = 0.03   # wrist must rise at least this much above setup (normalised Y)
HANDS_PEAK_SMOOTH_WINDOW    = 5      # rolling-minimum window for jitter suppression

# v3 contact-anchored backward search window
# HP occurs 5–10 frames before contact in all 17 ground-truth videos (mean=6.9, std=1.4).
# Window: contact - SEARCH_BEFORE_CONTACT  →  contact - MIN_OFFSET
HANDS_PEAK_V3_SEARCH_BEFORE_CONTACT = int(os.getenv("HANDS_PEAK_V3_SEARCH_BEFORE_CONTACT", "15"))
HANDS_PEAK_V3_MIN_OFFSET            = int(os.getenv("HANDS_PEAK_V3_MIN_OFFSET",            "4"))

HANDS_PEAK_DETECTOR_VERSION = (
    "v1_velocity_reversal" if USE_HANDS_PEAK_V1
    else ("v3_contact_anchored" if USE_HANDS_PEAK_V3
          else "v2_adaptive")
)  # currently v3_contact_anchored (F5b 2026-05-10)

# Version identifier for the contact detector currently running in production.
# Every automatic estimate should record this so validated corrections can be
# evaluated against the exact heuristic/model version that produced them.
# v1_original: 3-signal median (A=wrist_vel, B=elbow_angle, C=wrist_decel)
# v1_cleaned: 2-signal (A=wrist_vel, C=wrist_decel), B removed
USE_CONTACT_V1_ORIGINAL = _env_bool("USE_CONTACT_V1_ORIGINAL", True)
CONTACT_DETECTOR_VERSION = os.getenv(
    "CONTACT_DETECTOR_VERSION",
    "v1_original" if USE_CONTACT_V1_ORIGINAL else "v1_cleaned",
)
USE_AUDIO_CONTACT = _env_bool("USE_AUDIO_CONTACT", True)
# raised from 0.3 to 0.5 to reduce false onset detection
AUDIO_ONSET_DELTA = float(os.getenv("AUDIO_ONSET_DELTA", "0.5"))
AUDIO_CONTACT_MIN_CONFIDENCE = float(os.getenv("AUDIO_CONTACT_MIN_CONFIDENCE", "0.5"))
# Temporal filter bounds (fraction of video duration)
AUDIO_TEMPORAL_FILTER_START = float(os.getenv("AUDIO_TEMPORAL_FILTER_START", "0.20"))
AUDIO_TEMPORAL_FILTER_END = float(os.getenv("AUDIO_TEMPORAL_FILTER_END", "0.85"))
# N-aware multi-onset strategy (see audio_contact.py for full rationale).
# N=2 → AUDIO_STRATEGY_N2:      "last"    = bat-on-ball for bowling machine videos
# N≥3 → AUDIO_STRATEGY_N3_PLUS: "loudest" = avoids follow-through misfire (Cam T_LH)
AUDIO_MIN_ONSET_GAP_SECONDS = float(os.getenv("AUDIO_MIN_ONSET_GAP_SECONDS", "0.15"))
AUDIO_STRATEGY_N2 = os.getenv("AUDIO_STRATEGY_N2", "last")
AUDIO_STRATEGY_N3_PLUS = os.getenv("AUDIO_STRATEGY_N3_PLUS", "loudest")

# ---------------------------------------------------------------------------
# Phase Detection thresholds
# ---------------------------------------------------------------------------

# SETUP phase: minimum number of stable frames to confirm stance
SETUP_MIN_STABLE_FRAMES = 5

# SETUP → BACKLIFT: wrist Y must drop (rise in image) by at least this
# much below the setup baseline mean before we call Backlift Started.
# (Y decreasing means hands are going UP)
BACKLIFT_WRIST_RISE_THRESHOLD = 0.02    # normalised Y units

# Must persist for this many consecutive frames before transition
BACKLIFT_CONSECUTIVE_FRAMES = 3

# BACKLIFT → HANDS PEAK: velocity sign flip (positive → negative in Y)
# i.e. wrist Y was decreasing, now increasing → peak has passed
# Smoothing window for velocity calculation
VELOCITY_SMOOTH_WINDOW = 3

# FRONT FOOT DOWN: front ankle Y velocity near zero (foot has settled)
FRONT_ANKLE_LANDED_VEL_THRESHOLD = 0.005   # pixels/frame (normalised)
# AND stance width must have grown from setup by at least:
STRIDE_WIDTH_INCREASE_THRESHOLD = 0.01     # normalised units
# OR front ankle Z change has stopped
FRONT_ANKLE_Z_VEL_THRESHOLD = 0.005

# HANDS PEAK ↔ FRONT FOOT DOWN sync tolerance (ideal = 0, allowed = ±2)
SYNC_TOLERANCE_FRAMES = 2

# CONTACT detection: look for maximum wrist deceleration in this window
# after Hands Peak (prevents false triggers on the way up)
CONTACT_SEARCH_START_OFFSET = 3   # frames after Hands Peak

# If no clear deceleration spike, fall back to wrist-height local minimum
CONTACT_DECEL_MIN_RATIO = 0.5     # decel must be this fraction of max decel

# Contact window ±frames for rule evaluation
CONTACT_WINDOW_FRAMES = 2

# When contact confidence is low, damp contact-derived deductions rather
# than pretending the exact contact estimate is trustworthy.
CONTACT_CONFIDENCE_LOW_WEIGHT = 0.70

# FOLLOW-THROUGH: frames to analyse after contact
FOLLOW_THROUGH_ANALYSIS_FRAMES = 15
USE_FOLLOW_THROUGH_V1 = _env_bool("USE_FOLLOW_THROUGH_V1", False)
FOLLOW_THROUGH_WINDOW_PCT = float(os.getenv("FOLLOW_THROUGH_WINDOW_PCT", "0.30"))
FOLLOW_THROUGH_POST_CONTACT_OFFSET = int(os.getenv("FOLLOW_THROUGH_POST_CONTACT_OFFSET", "3"))
FOLLOW_THROUGH_SMOOTH_WINDOW = int(os.getenv("FOLLOW_THROUGH_SMOOTH_WINDOW", "3"))
FOLLOW_THROUGH_MIN_RISE = float(os.getenv("FOLLOW_THROUGH_MIN_RISE", "0.02"))
FOLLOW_THROUGH_DETECTOR_VERSION = (
    "v1_contact_offset" if USE_FOLLOW_THROUGH_V1 else "v2_post_contact_peak"
)

# ---------------------------------------------------------------------------
# Metric smoothing
# ---------------------------------------------------------------------------
METRICS_SMOOTH_WINDOW = 3

# Gap filling: if detection gap exceeds this many consecutive frames,
# mark the filled metrics as low_confidence (forward-fill still applies
# but downstream consumers can filter or weight accordingly)
MAX_CONFIDENT_GAP_FRAMES = 10

# ---------------------------------------------------------------------------
# Coaching Rule Thresholds
# ---------------------------------------------------------------------------

# ---- ACCESS ----------------------------------------------------------------

# A1: Bat path going around the body
# Lateral wrist spread increase from peak to contact (normalised X units)
# Reference spread increase ≈ 0.051; threshold set above that
# A1: Wrist spread / bat path wrapping
# SUSPENDED 2026-05-14: wrist_spread_increase is camera-angle-sensitive.
# On-drives filmed side-on show wrists converging through the hitting zone,
# identical to "bat going around the body" fault. Cam T_Ondrive Elite.mov
# receives maximum 10pt deduction despite correct technique.
# D7 rule health: flat (discrimination_score -0.85, inverted on Elite).
# Redesign requires a 3D or overhead camera angle, or a different proxy.
SUSPEND_A1 = True
A1_FULL_MARKS_THRESHOLD = 0.005
A1_PARTIAL_THRESHOLD = 0.0
A1_SIGNIFICANT_THRESHOLD = -0.020
A1_FULL_MARKS_DEDUCTION = 0
A1_PARTIAL_DEDUCTION = 3
A1_SIGNIFICANT_DEDUCTION = 6
A1_SEVERE_DEDUCTION = 10

# A2: Elbow angle at contact
# SUSPENDED: elbow angle distribution is non-monotonic across tiers.
# Average (142°) > Elite (119°). Needs coaching review before re-enabling.

# A3: Compression frames
# A3 FIXED 2026-04-25: collapsed to two bands; else branch was unreachable.
# Previous A3_PARTIAL_MIN=0, A3_PARTIAL_DEDUCTION=0 made both branches → 0.
SUSPEND_A3 = True
# SUSPENDED 2026-05-13: discrimination_score 0.45 in D5 rule health audit.
# Fire rate: Beginner 0.40, Elite 0.25. Adding noise, not signal.
# Revisit after setup detection is stable (A3 measures contact position
# relative to body — setup baseline affects the measurement geometry).
A3_FULL_MARKS_MIN = 1
A3_SEVERE_DEDUCTION = 3

# A4: Torso lean
# SUSPENDED: Elite players lean more than beginners in measured data.

# A5: Shoulder-hip gap
A5_IDEAL_MIN = -25.0
A5_IDEAL_MAX = -10.0
A5_PARTIAL_OUTER = 15.0
A5_SIGNIFICANT_OUTER = 25.0
A5_FULL_MARKS_DEDUCTION = 0
A5_PARTIAL_DEDUCTION = 3
A5_SIGNIFICANT_DEDUCTION = 6
A5_SEVERE_DEDUCTION = 10

# DELETED 2026-04-29: A6 — downstream consequence, not independent
# A6_FULL_MARKS_THRESHOLD = 0.04
# A6_PARTIAL_THRESHOLD = 0.08
# A6_SIGNIFICANT_THRESHOLD = 0.12
# A6_FULL_MARKS_DEDUCTION = 0
# A6_PARTIAL_DEDUCTION = 3
# A6_SIGNIFICANT_DEDUCTION = 6
# A6_SEVERE_DEDUCTION = 9

# ---- TRACKING --------------------------------------------------------------

# T1: Head offset
# SUSPENDED: measured distribution is inverted (Elite > Beginner).

# T2: Early head change
# REVISED 2026-05-14: raised minimum threshold from previous value to 0.038.
# R1 data: Good Club range 0.025–0.037 (all fire under old threshold).
# New threshold ensures Good Club fire rate drops to 0.00 while Beginner
# fire rate stays ≥0.80. Average values (0.008–0.026) correctly score 0.
T2_MINOR_THRESHOLD = 0.038
T2_SIGNIFICANT_THRESHOLD = 0.055
T2_SEVERE_THRESHOLD = 0.080
T2_FULL_MARKS_THRESHOLD = T2_MINOR_THRESHOLD
T2_PARTIAL_THRESHOLD = T2_SIGNIFICANT_THRESHOLD
T2_FULL_MARKS_DEDUCTION = 0
T2_PARTIAL_DEDUCTION = 3
T2_SIGNIFICANT_DEDUCTION = 6
T2_SEVERE_DEDUCTION = 9

# DELETED 2026-04-29: T3/T4/T5 — not measurable from side-on video
# T3: Head position variance
# T4: Eye tilt
# T5: Setup head variance

# ---- STABILITY -------------------------------------------------------------

# S1: Hip shift
S1_FULL_MARKS_THRESHOLD = 0.14
S1_PARTIAL_THRESHOLD = 0.08
S1_SIGNIFICANT_THRESHOLD = 0.04
S1_FULL_MARKS_DEDUCTION = 0
S1_PARTIAL_DEDUCTION = 3
S1_SIGNIFICANT_DEDUCTION = 6
S1_SEVERE_DEDUCTION = 10

# S2: Post-contact instability
# SUSPENDED 2026-05-11: Inverted after F5b detection improvements.
# S2 fire rate: Elite > Beginner (should be opposite).
# Root cause: thresholds calibrated on incorrect HP/FFD frames.
# Redesign scheduled for R4-equivalent task once detection stable.
S2_FULL_MARKS_THRESHOLD = 0.010
S2_PARTIAL_THRESHOLD = 0.020
S2_SIGNIFICANT_THRESHOLD = 0.040
S2_FULL_MARKS_DEDUCTION = 0
S2_PARTIAL_DEDUCTION = 2
S2_SIGNIFICANT_DEDUCTION = 4
S2_SEVERE_DEDUCTION = 6

# S3: Hip drift frames
# SUSPENDED: misleading due to elite outlier inflation.
S3_HIP_DRIFT_TOLERANCE = 0.04

# S4: Post-contact body rotation
S4_POST_CONTACT_ROTATION_FRAMES = 5
# REVISED 2026-05-14: fixed sign convention (positive = forward rotation
# continuing post-contact). Bands tightened at upper end.
S4_FULL_MARKS_THRESHOLD = 5.0
S4_PARTIAL_THRESHOLD = 12.0
S4_SIGNIFICANT_THRESHOLD = 22.0
S4_FULL_MARKS_DEDUCTION = 0
S4_PARTIAL_DEDUCTION = 4
S4_SIGNIFICANT_DEDUCTION = 8
S4_SEVERE_DEDUCTION = 12

# ---- FLOW ------------------------------------------------------------------

# F1: Sync frames
# F1 SUSPENDED 2026-05-11: Hands_peak/FFD sync fires 4/4 Elite in auto, 0/4 in
# validated. FFD MAE 7.0f corrupts sync measurement.
# Redesign: tighten FFD detection or use more stable sync proxy.
F1_FULL_MARKS_THRESHOLD = 2
F1_PARTIAL_THRESHOLD = 3
F1_SIGNIFICANT_THRESHOLD = 5
F1_FULL_MARKS_DEDUCTION = 0
F1_PARTIAL_DEDUCTION = 2
F1_SIGNIFICANT_DEDUCTION = 4
F1_SEVERE_DEDUCTION = 7

# F2: Velocity direction changes
# SUSPENDED: measured distribution is inverted (Elite > Beginner).

# F3: Timing ratio
# F3 SUSPENDED 2026-05-11: backlift_to_contact_frames produces negative values on
# Elite (backlift_start detected after contact). Root cause: backlift_start inherits
# setup_frame errors (MAE 26.4f). Metric is invalid until setup/backlift detection
# is stable. Redesign: replace metric with HP-to-contact frames once stable.
F3_IDEAL_RATIO = 1.0
F3_FULL_MARKS_DEVIATION = 0.15
F3_PARTIAL_DEVIATION = 0.30
F3_SIGNIFICANT_DEVIATION = 0.45
F3_FULL_MARKS_DEDUCTION = 0
F3_PARTIAL_DEDUCTION = 3
F3_SIGNIFICANT_DEDUCTION = 6
F3_SEVERE_DEDUCTION = 9

# F4: Pause frames
# REVISED 2026-05-14: raised minimum threshold from 3→5 frames.
# D7 data: Elite and Good Club plateau at 3 frames (normal technique).
# Increased deduction amounts to preserve discrimination power at
# the tighter threshold range.
F4_PAUSE_VELOCITY_THRESHOLD = 0.003
F4_THRESHOLD_MINOR = 5
F4_THRESHOLD_SEVERE = 6
F4_FULL_MARKS_THRESHOLD = F4_THRESHOLD_MINOR - 1
F4_PARTIAL_THRESHOLD = F4_THRESHOLD_MINOR
F4_SIGNIFICANT_THRESHOLD = F4_THRESHOLD_SEVERE - 1
F4_FULL_MARKS_DEDUCTION = 0
F4_PARTIAL_DEDUCTION = 6
F4_SIGNIFICANT_DEDUCTION = 6
F4_SEVERE_DEDUCTION = 12

# F5: Mid-downswing hitch
# SUSPENDED 2026-05-11: Inverted after F5b detection improvements.
# F5 fire rate: Elite > Beginner (should be opposite).
# Root cause: thresholds calibrated on incorrect HP/FFD frames.
# Redesign scheduled for R4-equivalent task once detection stable.
F5_HITCH_VELOCITY_FRACTION = 0.30
F5_HITCH_DEDUCTION = 5

# F6: Follow-through velocity
# SUSPENDED 2026-05-15: keep deployment on the calibrated D7/R1 rule set.
# Duration-based F6 was stale after follow_through v2; velocity-based F6
# requires further calibration before reactivation.
SUSPEND_F6 = True
F6_POST_CONTACT_OFFSET = 3   # retained for legacy measurement compatibility
F6_ANALYSIS_FRAMES     = 15  # retained for legacy measurement compatibility
F6_V2_POST_CONTACT_WINDOW = 8
F6_V2_METRIC = "wrist_speed"
# F6 V2: contact-anchor-only measurement. Does not use follow_through
# anchor. Measures mean post-contact wrist speed in 8 frames after contact.
# Higher speed = batter continued through the shot (good).
# Thresholds derived from Beginner/Elite median separation.
# Previous duration-based F6 abandoned: follow_through MAE 10.65 too
# high for reliable duration measurement.
F6_FULL_MARKS_THRESHOLD = 0.014323
F6_PARTIAL_THRESHOLD = 0.015563
F6_SIGNIFICANT_THRESHOLD = 0.015560
F6_FULL_MARKS_DEDUCTION = 0
F6_PARTIAL_DEDUCTION = 4
F6_SIGNIFICANT_DEDUCTION = 7
F6_SEVERE_DEDUCTION = 10

# ---------------------------------------------------------------------------
# Scoring bands and traffic lights
# ---------------------------------------------------------------------------
PILLAR_MAX = 25
PILLAR_RULE_MAX_DEDUCTION = {
    "access": {
        # A1 suspended 2026-05-14: camera-angle-sensitive wrist spread proxy
        "A5": 10,
    },
    "tracking": {
        "T2": 9,
    },
    "stability": {
        # S2 suspended 2026-05-11: inverted after F5b
        "S4": 12,
    },
    "flow": {
        # F1 suspended 2026-05-11: HP/FFD sync corrupted by FFD detection errors
        # F3 suspended 2026-05-11: backlift_to_contact_frames invalid (negative on Elite)
        "F4": 12,
        # F5 suspended 2026-05-11: inverted after F5b
        # F6 suspended 2026-05-15: not included in active flow capacity
    },
}
TRAFFIC_GREEN_MIN = 20
TRAFFIC_AMBER_MIN = 12  # 12–19 = Amber
# < 12 = Red

SCORE_BANDS = [
    (85, 100, "Excellent"),
    (70, 84, "Good"),
    (55, 69, "Developing"),
    (40, 54, "Work Needed"),
    (0,  39, "Fundamentals"),
]

# Tiebreak priority for priority fix (lower index = higher priority)
PILLAR_TIEBREAK_ORDER = ["stability", "tracking", "access", "flow"]

# ---------------------------------------------------------------------------
# Reference baseline path
# ---------------------------------------------------------------------------
REFERENCE_BASELINE_PATH = "reference/reference_baseline.json"
