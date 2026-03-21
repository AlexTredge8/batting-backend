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

# FOLLOW-THROUGH: frames to analyse after contact
FOLLOW_THROUGH_ANALYSIS_FRAMES = 15

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
A1_WRIST_SPREAD_THRESHOLD = 0.07      # > this = fault

# A2: Contact too close to body
# Front elbow angle at contact — > this = arms too straight (bad)
# Reference front elbow ≈ 97°; fault is when arms are nearly fully extended
A2_ELBOW_STRAIGHT_THRESHOLD = 160.0   # degrees

# A3: Restricted contact window
# Frames wrists drive forward after contact before rising
# Reference compression ≈ 3 frames; fault if shorter than that
A3_COMPRESSION_MIN_FRAMES = 2        # < this = fault
A3_COMPRESSION_GOOD_FRAMES = 4       # >= this = no deduction

# A4: Torso leaning sideways at contact
# Reference torso_lean ≈ 0.087; set threshold comfortably above reference
A4_TORSO_LEAN_THRESHOLD = 0.12       # normalised X units

# A5: Shoulders lagging behind hips
# Reference shoulder_hip_gap = -27°; fault only when much more extreme
A5_SHOULDER_HIP_GAP_THRESHOLD = -35.0   # degrees (must be more extreme than reference)

# A6: Opening up too early
# Shoulder or hip openness exceeds contact value BEFORE backlift start
A6_EARLY_OPEN_SHOULDER_FRACTION = 0.90   # fraction of ref contact openness
A6_EARLY_OPEN_HIP_FRACTION = 0.90

# ---- TRACKING --------------------------------------------------------------

# T1: Head falling outside the line at contact
# Reference head_offset ≈ 0.133; set threshold above reference
T1_HEAD_OFFSET_THRESHOLD = 0.18      # normalised X units (absolute)

# T2: Early head movement — head offset change between setup and backlift starts
T2_EARLY_HEAD_CHANGE_THRESHOLD = 0.06

# T3: Head not still at contact — variance in ±2 frame window
T3_HEAD_STILLNESS_VARIANCE = 0.0005  # combined X+Y variance

# T4: Eyes not level at contact
T4_EYE_TILT_THRESHOLD = 0.035        # normalised Y difference between eyes

# T5: Head not composed in setup
T5_SETUP_HEAD_VARIANCE = 0.0008      # head position variance during setup

# ---- STABILITY -------------------------------------------------------------

# S1: Weight not transferring to front foot
# Reference front knee ≈ 156°; threshold above reference
S1_FRONT_KNEE_STRAIGHT_THRESHOLD = 168.0   # > this = too straight (locked)
S1_BACK_KNEE_BENT_THRESHOLD = 90.0         # < this = still bent (weight back)
# Reference hip shift = -0.067; threshold below absolute value of reference
S1_HIP_SHIFT_THRESHOLD = 0.03              # hip centre must shift this much

# S2: Post-contact instability
S2_POST_CONTACT_HIP_STD = 0.022
S2_POST_CONTACT_HEAD_STD = 0.022

# S3: Hips drifting outside base (hip X outside front ankle X)
S3_HIP_DRIFT_TOLERANCE = 0.04

# S4: Post-contact body rotation
S4_POST_CONTACT_ROTATION_FRAMES = 5
S4_ROTATION_THRESHOLD = 12.0          # degrees of further rotation allowed

# ---- FLOW ------------------------------------------------------------------

# F1: Hands Peak / Front Foot Down desync (frames)
F1_SMALL_DESYNC = 5     # ±3-5 frames → -5
F1_MEDIUM_DESYNC = 8    # ±6-8 frames → -8
F1_LARGE_DESYNC = 9     # ±9+ frames  → -12

# F2: Jerky acceleration — velocity direction changes between backlift and contact
F2_VELOCITY_DIRECTION_CHANGES_THRESHOLD = 3

# F3: Movement timing out of range (backlift start → contact duration in frames)
# Reference sets the target; these are tolerance multipliers
F3_TIMING_SHORT_THRESHOLD = 0.65   # < 65% of reference = too rushed
F3_TIMING_LONG_THRESHOLD = 1.45    # > 145% of reference = too slow

# F4: Pause at the top — wrist velocity near zero at Hands Peak
F4_PAUSE_VELOCITY_THRESHOLD = 0.003   # normalised units/frame
F4_PAUSE_FRAMES_THRESHOLD = 5         # > this many frames near-zero = fault

# F5: Mid-downswing hitch
# Wrist velocity drops below this fraction of average downswing velocity
F5_HITCH_VELOCITY_FRACTION = 0.30

# F6: No natural follow-through
# Wrists should rise above shoulder height in follow-through
F6_FOLLOWTHROUGH_WRIST_SHOULDER_MARGIN = 0.02   # wrist_y must be < shoulder_y + this
F6_FOLLOWTHROUGH_MIN_FRAMES = 3   # must continue at least this many frames

# ---------------------------------------------------------------------------
# Scoring bands and traffic lights
# ---------------------------------------------------------------------------
PILLAR_MAX = 25
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
