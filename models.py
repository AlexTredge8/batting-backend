"""
BattingIQ Phase 2 — Data Models
=================================
Dataclasses and enums used throughout the pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BattingPhase(str, Enum):
    SETUP = "setup"
    BACKLIFT_STARTS = "backlift_starts"
    HANDS_PEAK = "hands_peak"
    FRONT_FOOT_DOWN = "front_foot_down"
    CONTACT = "contact"
    FOLLOW_THROUGH = "follow_through"
    UNKNOWN = "unknown"


class TrafficLight(str, Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


# ---------------------------------------------------------------------------
# Per-frame data from pose extractor
# ---------------------------------------------------------------------------

@dataclass
class RawLandmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class FramePose:
    """Raw MediaPipe landmarks for one frame."""
    frame_idx: int
    timestamp_s: float
    landmarks: Optional[list[RawLandmark]]  # None if no pose detected

    @property
    def detected(self) -> bool:
        return self.landmarks is not None

    def lm(self, idx: int) -> RawLandmark:
        return self.landmarks[idx]


# ---------------------------------------------------------------------------
# Per-frame calculated metrics
# ---------------------------------------------------------------------------

@dataclass
class FrameMetrics:
    """All coaching-relevant metrics calculated from one frame's landmarks."""
    frame_idx: int
    timestamp_s: float
    detected: bool

    # Wrist position
    wrist_height: float = 0.0           # avg Y of both wrists (smaller = higher)
    wrist_x_left: float = 0.0
    wrist_x_right: float = 0.0
    wrist_x_mid: float = 0.0

    # Wrist velocity (frame-to-frame delta, calculated in post-pass)
    wrist_velocity_y: float = 0.0       # +ve = moving down, -ve = moving up
    wrist_velocity_x: float = 0.0
    wrist_speed: float = 0.0            # magnitude of 2D velocity
    wrist_forward_velocity: float = 0.0 # Z velocity (toward bowler = +ve for end-on)

    # Stance
    stance_width: float = 0.0           # X dist between ankles

    # Ankle positions
    front_ankle_x: float = 0.0
    front_ankle_y: float = 0.0
    front_ankle_z: float = 0.0
    front_ankle_vy: float = 0.0         # frame-to-frame Y velocity
    front_ankle_vz: float = 0.0         # frame-to-frame Z velocity
    back_ankle_x: float = 0.0
    back_ankle_y: float = 0.0

    # Head position
    head_offset: float = 0.0            # nose X - hip midpoint X
    head_x: float = 0.0
    head_y: float = 0.0
    head_vx: float = 0.0                # frame-to-frame lateral velocity
    head_vy: float = 0.0                # frame-to-frame vertical velocity

    # Eyes
    eye_tilt: float = 0.0               # |left_eye_y - right_eye_y|

    # Shoulder / hip openness
    shoulder_openness: float = 0.0      # degrees from side-on
    hip_openness: float = 0.0
    shoulder_hip_gap: float = 0.0       # shoulder_openness - hip_openness

    # Shoulder midpoint
    shoulder_mid_x: float = 0.0
    shoulder_mid_y: float = 0.0

    # Hip midpoint
    hip_mid_x: float = 0.0
    hip_mid_y: float = 0.0

    # Torso lean (horizontal distance between shoulder and hip midpoints)
    torso_lean: float = 0.0

    # Knee angles
    front_knee_angle: float = 0.0
    back_knee_angle: float = 0.0

    # Elbow angles
    front_elbow_angle: float = 0.0
    back_elbow_angle: float = 0.0

    # Hip centre X (for stability checks)
    hip_centre_x: float = 0.0
    hip_centre_vx: float = 0.0

    # Quality flags
    low_confidence: bool = False   # True if gap-filled beyond MAX_CONFIDENT_GAP_FRAMES


# ---------------------------------------------------------------------------
# Phase detection results
# ---------------------------------------------------------------------------

@dataclass
class PhaseResult:
    """Output of the phase detector."""
    phase_labels: list[BattingPhase]          # one label per frame

    # Key event frames
    setup_start: int = 0
    setup_end: int = 0
    backlift_start: int = 0
    hands_peak: int = 0
    front_foot_down: int = 0
    contact: int = 0
    follow_through_start: int = 0

    # Timing
    hands_peak_vs_ffd_diff: int = 0           # signed (+ = peak before FFD)
    hands_peak_vs_ffd_ms: float = 0.0
    backlift_to_contact_frames: int = 0
    fps: float = 30.0
    contact_confidence: str = "high"          # "high" | "medium" | "low"
    contact_candidates: dict = field(default_factory=dict)
    contact_window: dict = field(default_factory=dict)
    contact_diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Coaching faults
# ---------------------------------------------------------------------------

@dataclass
class Fault:
    rule_id: str                 # e.g. "A1", "T3"
    fault: str                   # short label
    deduction: int               # points deducted from pillar
    detail: str                  # numeric detail for the coach report
    feedback: str                # player-facing text


# ---------------------------------------------------------------------------
# Pillar scores
# ---------------------------------------------------------------------------

@dataclass
class PillarScore:
    name: str                    # "access" | "tracking" | "stability" | "flow"
    score: int
    max_score: int = 25
    status: TrafficLight = TrafficLight.GREEN
    faults: list[Fault] = field(default_factory=list)
    positives: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------

@dataclass
class BattingIQResult:
    battingiq_score: int
    score_band: str
    pillars: dict[str, PillarScore]     # keyed by pillar name
    priority_fix: Optional[Fault]
    development_notes: list[str]
    phases: PhaseResult
    metadata: dict
    handedness: str = "right"           # "right" or "left"
    handedness_source: str = "default"  # "api", "auto", or "default"
