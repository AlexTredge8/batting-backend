"""
Tests for handedness support.

Validates:
1. Side mapping produces correct front/back assignments
2. S3 directional check works for both RHB and LHB
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


def test_s3_direction_rhb():
    """S3 for RHB: hip drifting left of front ankle = fault."""
    from coaching_rules import rule_S3
    from models import FrameMetrics, PhaseResult, BattingPhase

    n = 30
    metrics = []
    for i in range(n):
        m = FrameMetrics(frame_idx=i, timestamp_s=i / 30, detected=True)
        # RHB: front ankle at x=0.4, hip drifts to x=0.3 (left of ankle = outside base)
        m.front_ankle_x = 0.4
        m.hip_centre_x = 0.30
        metrics.append(m)

    phases = PhaseResult(
        phase_labels=[BattingPhase.SETUP] * 5 + [BattingPhase.BACKLIFT_STARTS] * 10 +
                     [BattingPhase.CONTACT] * 5 + [BattingPhase.FOLLOW_THROUGH] * 10,
        backlift_start=5,
        follow_through_start=20,
        contact=15,
    )

    faults = rule_S3(metrics, phases, {}, front_side="left")
    assert len(faults) > 0, "S3 should fire for RHB with hip drifting left"
    print("  PASS: S3 direction RHB")


def test_s3_direction_lhb():
    """S3 for LHB: hip drifting right of front ankle = fault."""
    from coaching_rules import rule_S3
    from models import FrameMetrics, PhaseResult, BattingPhase

    n = 30
    metrics = []
    for i in range(n):
        m = FrameMetrics(frame_idx=i, timestamp_s=i / 30, detected=True)
        # LHB: front ankle at x=0.6, hip drifts to x=0.7 (right of ankle = outside base)
        m.front_ankle_x = 0.6
        m.hip_centre_x = 0.70
        metrics.append(m)

    phases = PhaseResult(
        phase_labels=[BattingPhase.SETUP] * 5 + [BattingPhase.BACKLIFT_STARTS] * 10 +
                     [BattingPhase.CONTACT] * 5 + [BattingPhase.FOLLOW_THROUGH] * 10,
        backlift_start=5,
        follow_through_start=20,
        contact=15,
    )

    faults = rule_S3(metrics, phases, {}, front_side="right")
    assert len(faults) > 0, "S3 should fire for LHB with hip drifting right"
    print("  PASS: S3 direction LHB")


def test_s3_no_false_positive_lhb():
    """S3 for LHB: hip drifting LEFT of front ankle is NOT a fault (that's toward the body)."""
    from coaching_rules import rule_S3
    from models import FrameMetrics, PhaseResult, BattingPhase

    n = 30
    metrics = []
    for i in range(n):
        m = FrameMetrics(frame_idx=i, timestamp_s=i / 30, detected=True)
        # LHB: front ankle at x=0.6, hip at x=0.5 (left of ankle = toward body, not outside)
        m.front_ankle_x = 0.6
        m.hip_centre_x = 0.50
        metrics.append(m)

    phases = PhaseResult(
        phase_labels=[BattingPhase.SETUP] * 5 + [BattingPhase.BACKLIFT_STARTS] * 10 +
                     [BattingPhase.CONTACT] * 5 + [BattingPhase.FOLLOW_THROUGH] * 10,
        backlift_start=5,
        follow_through_start=20,
        contact=15,
    )

    faults = rule_S3(metrics, phases, {}, front_side="right")
    assert len(faults) == 0, "S3 should NOT fire for LHB with hip toward body"
    print("  PASS: S3 no false positive LHB")


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
    test_s3_direction_rhb()
    test_s3_direction_lhb()
    test_s3_no_false_positive_lhb()
    test_report_includes_handedness()
    print("\nAll handedness tests passed!")
