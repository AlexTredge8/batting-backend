"""
Tests for the contact-frame consensus logic.

Validates:
1. The detector resolves contact from three signals and records diagnostics.
2. Confidence buckets map correctly from candidate disagreement.
3. Existing phase sequencing still marks contact and follow-through in the right places.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import BattingPhase, FrameMetrics
from phase_detector import detect_phases


def _metric(
    idx: int,
    *,
    wrist_height: float,
    wrist_velocity_y: float,
    front_elbow_angle: float,
    wrist_speed: float,
    stance_width: float,
    front_ankle_vy: float = 0.0,
    head_offset: float = 0.0,
    hip_centre_x: float = 0.5,
    shoulder_openness: float = 0.0,
    hip_openness: float = 0.0,
) -> FrameMetrics:
    return FrameMetrics(
        frame_idx=idx,
        timestamp_s=idx / 30.0,
        detected=True,
        wrist_height=wrist_height,
        wrist_velocity_y=wrist_velocity_y,
        wrist_speed=wrist_speed,
        stance_width=stance_width,
        front_ankle_vy=front_ankle_vy,
        head_offset=head_offset,
        hip_centre_x=hip_centre_x,
        shoulder_openness=shoulder_openness,
        hip_openness=hip_openness,
        front_elbow_angle=front_elbow_angle,
    )


def _build_metrics(
    *,
    elbow_peak_idx: int,
    speed_min_idx: int,
) -> list[FrameMetrics]:
    n = 26
    metrics = []
    wrist_heights = [
        0.50, 0.50, 0.50, 0.50, 0.50,
        0.45, 0.40, 0.35, 0.30,
        0.33, 0.37, 0.42, 0.48, 0.55, 0.62, 0.68, 0.72, 0.70, 0.66, 0.61, 0.56, 0.50, 0.46, 0.42, 0.39, 0.36,
    ]
    wrist_velocities = [
        0.00, 0.00, 0.00, 0.00, 0.00,
        -0.05, -0.05, -0.04, -0.03,
        0.06, 0.05, 0.04, 0.03, 0.01, -0.02, -0.03, -0.02, -0.02, -0.02, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03,
    ]
    front_elbows = [150.0] * n
    front_elbows[elbow_peak_idx] = 160.0
    if elbow_peak_idx - 1 >= 0:
        front_elbows[elbow_peak_idx - 1] = 158.0
    if elbow_peak_idx + 1 < n:
        front_elbows[elbow_peak_idx + 1] = 157.0

    wrist_speeds = [1.4] * n
    wrist_speeds[9] = 1.9
    wrist_speeds[10] = 2.1
    wrist_speeds[11] = 2.4
    wrist_speeds[12] = 1.6
    wrist_speeds[13] = 1.0
    wrist_speeds[14] = 0.8
    wrist_speeds[15] = 0.6
    wrist_speeds[16] = 0.5
    wrist_speeds[17] = 0.4
    wrist_speeds[18] = 0.6
    wrist_speeds[19] = 0.8
    wrist_speeds[20] = 1.0
    wrist_speeds[21] = 1.2
    wrist_speeds[speed_min_idx] = 0.1

    stance_widths = [0.20] * n
    for i in range(10, n):
        stance_widths[i] = 0.34

    front_ankle_vys = [0.03] * n
    for i in range(10, n):
        front_ankle_vys[i] = 0.0

    for i in range(n):
        metrics.append(
            _metric(
                i,
                wrist_height=wrist_heights[i],
                wrist_velocity_y=wrist_velocities[i],
                front_elbow_angle=front_elbows[i],
                wrist_speed=wrist_speeds[i],
                stance_width=stance_widths[i],
                front_ankle_vy=front_ankle_vys[i],
                head_offset=0.05 if i < 10 else 0.06,
                hip_centre_x=0.50 if i < 10 else 0.56,
                shoulder_openness=20.0 if i < 10 else 25.0,
                hip_openness=15.0 if i < 10 else 18.0,
            )
        )
    return metrics


def test_contact_consensus_high_confidence():
    metrics = _build_metrics(elbow_peak_idx=13, speed_min_idx=13)
    phases = detect_phases(metrics, fps=30.0)

    assert phases.contact_confidence == "high"
    assert phases.contact_diagnostics["signals"]["wrist_velocity_reversal"]["frame"] == 13
    assert phases.contact_diagnostics["signals"]["front_elbow_target"]["frame"] == 13
    assert phases.contact_diagnostics["signals"]["wrist_speed_decel"]["frame"] == 13
    assert phases.contact_candidates["spread"] == 0
    assert phases.contact == 13
    assert phases.phase_labels[13] == BattingPhase.CONTACT
    assert phases.phase_labels[16] == BattingPhase.FOLLOW_THROUGH


def test_contact_consensus_medium_confidence():
    metrics = _build_metrics(elbow_peak_idx=16, speed_min_idx=17)
    phases = detect_phases(metrics, fps=30.0)

    assert phases.contact_confidence == "medium"
    assert phases.contact_diagnostics["candidates"] == [13, 16, 17]
    assert phases.contact == 16
    assert phases.contact_window["start"] <= phases.contact <= phases.contact_window["end"]
    assert phases.estimated_contact_frame == 16
    assert phases.resolved_contact_frame == 16
    assert phases.resolved_contact_source == "auto"
    assert phases.phase_labels[16] == BattingPhase.CONTACT


def test_contact_consensus_low_confidence():
    metrics = _build_metrics(elbow_peak_idx=18, speed_min_idx=21)
    phases = detect_phases(metrics, fps=30.0)

    assert phases.contact_confidence == "low"
    assert phases.contact_diagnostics["span"] == 8
    assert phases.contact_diagnostics["chosen"] == 18
    assert phases.contact_candidates["selection_reason"] == "median_of_three_signal_consensus"
    assert phases.contact == 18
    assert phases.phase_labels[18] == BattingPhase.CONTACT


if __name__ == "__main__":
    test_contact_consensus_high_confidence()
    test_contact_consensus_medium_confidence()
    test_contact_consensus_low_confidence()
    print("All contact consensus tests passed!")
