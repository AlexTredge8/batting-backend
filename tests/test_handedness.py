"""
Tests for handedness support.

Validates:
1. Side mapping produces correct front/back assignments
2. S3 drift measurement direction works for both RHB and LHB (rule itself is suspended)
3. Handedness flows through the pipeline and appears in reports
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics_calculator import _build_side_map, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ANKLE, RIGHT_ANKLE


def _handedness_to_front_side(handedness: str) -> str:
    """Local copy to avoid importing run_analysis (heavy transitive deps)."""
    return "left" if handedness == "right" else "right"


def test_side_map_right_handed():
    """Right-handed batter: front side is left (person's left faces bowler)."""
    sm = _build_side_map("left")
    assert sm["FRONT_SHOULDER"] == LEFT_SHOULDER
    assert sm["BACK_SHOULDER"] == RIGHT_SHOULDER
    assert sm["FRONT_ANKLE"] == LEFT_ANKLE
    assert sm["BACK_ANKLE"] == RIGHT_ANKLE
    print("  PASS: side_map right-handed")


def test_side_map_left_handed():
    """Left-handed batter: front side is right (person's right faces bowler)."""
    sm = _build_side_map("right")
    assert sm["FRONT_SHOULDER"] == RIGHT_SHOULDER
    assert sm["BACK_SHOULDER"] == LEFT_SHOULDER
    assert sm["FRONT_ANKLE"] == RIGHT_ANKLE
    assert sm["BACK_ANKLE"] == LEFT_ANKLE
    print("  PASS: side_map left-handed")


def test_handedness_to_front_side():
    assert _handedness_to_front_side("right") == "left"
    assert _handedness_to_front_side("left") == "right"
    print("  PASS: handedness_to_front_side")


def _drift_fixture(front_ankle_x: float, hip_x: float):
    from models import FrameMetrics, PhaseResult, BattingPhase

    n = 30
    metrics = []
    for i in range(n):
        m = FrameMetrics(frame_idx=i, timestamp_s=i / 30, detected=True)
        m.front_ankle_x = front_ankle_x
        m.hip_centre_x = hip_x
        metrics.append(m)

    phases = PhaseResult(
        phase_labels=[BattingPhase.SETUP] * 5 + [BattingPhase.BACKLIFT_STARTS] * 10 +
                     [BattingPhase.CONTACT] * 5 + [BattingPhase.FOLLOW_THROUGH] * 10,
        backlift_start=5,
        follow_through_start=20,
        contact=15,
    )
    baseline = {"setup": {"hip_centre_x_mean": hip_x}, "timing": {"backlift_to_contact_frames": 15}}
    return metrics, phases, baseline


def test_s3_rule_is_suspended():
    """S3 is intentionally suspended (2026-04-29): the rule must not deduct for any handedness."""
    from coaching_rules import rule_S3

    metrics, phases, baseline = _drift_fixture(front_ankle_x=0.4, hip_x=0.30)
    assert rule_S3(metrics, phases, baseline, front_side="left") == []
    assert rule_S3(metrics, phases, baseline, front_side="right") == []


def test_s3_measurement_direction_rhb():
    """RHB: hip drifting LEFT of the front ankle counts as drift (outside the base)."""
    from coaching_rules import collect_rule_measurements

    metrics, phases, baseline = _drift_fixture(front_ankle_x=0.4, hip_x=0.30)
    measurements = collect_rule_measurements(metrics, phases, baseline, front_side="left")
    assert measurements["S3_hip_drift_frames"] > 0


def test_s3_measurement_direction_lhb():
    """LHB: hip drifting RIGHT of the front ankle counts as drift (outside the base)."""
    from coaching_rules import collect_rule_measurements

    metrics, phases, baseline = _drift_fixture(front_ankle_x=0.6, hip_x=0.70)
    measurements = collect_rule_measurements(metrics, phases, baseline, front_side="right")
    assert measurements["S3_hip_drift_frames"] > 0


def test_s3_measurement_no_false_positive_lhb():
    """LHB: hip toward the body (left of front ankle) is NOT drift."""
    from coaching_rules import collect_rule_measurements

    metrics, phases, baseline = _drift_fixture(front_ankle_x=0.6, hip_x=0.50)
    measurements = collect_rule_measurements(metrics, phases, baseline, front_side="right")
    assert measurements["S3_hip_drift_frames"] == 0


def test_report_includes_handedness():
    """Report generator includes handedness fields."""
    from models import BattingIQResult, PillarScore, PhaseResult, BattingPhase, TrafficLight
    from report_generator import build_json_report

    phases = PhaseResult(
        phase_labels=[BattingPhase.SETUP] * 10,
        fps=30.0,
    )
    result = BattingIQResult(
        battingiq_score=80,
        score_band="Good",
        pillars={
            "access": PillarScore(name="access", score=20),
            "tracking": PillarScore(name="tracking", score=20),
            "stability": PillarScore(name="stability", score=20),
            "flow": PillarScore(name="flow", score=20),
        },
        priority_fix=None,
        development_notes=[],
        phases=phases,
        metadata={},
        handedness="left",
        handedness_source="api",
    )

    report = build_json_report(result)
    assert report["handedness"] == "left"
    assert report["handedness_source"] == "api"
    print("  PASS: report includes handedness")


if __name__ == "__main__":
    print("Running handedness tests...")
    test_side_map_right_handed()
    test_side_map_left_handed()
    test_handedness_to_front_side()
    test_s3_rule_is_suspended()
    test_s3_measurement_direction_rhb()
    test_s3_measurement_direction_lhb()
    test_s3_measurement_no_false_positive_lhb()
    test_report_includes_handedness()
    print("\nAll handedness tests passed!")
