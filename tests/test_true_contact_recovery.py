"""
Tests for resolved-vs-estimated contact recovery.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import BattingPhase, FrameMetrics, PhaseResult
from phase_detector import apply_contact_override
from report_generator import build_json_report
from scorer import _apply_contact_confidence_weight
from models import Fault, PillarScore, BattingIQResult, TrafficLight
from evaluate_contact_labels import evaluate_rows


def _metric(frame_idx: int) -> FrameMetrics:
    return FrameMetrics(
        frame_idx=frame_idx,
        timestamp_s=frame_idx / 30.0,
        detected=True,
        wrist_height=0.5,
        wrist_velocity_y=0.0,
        wrist_speed=0.0,
        front_elbow_angle=120.0,
    )


def _phases() -> PhaseResult:
    labels = [BattingPhase.SETUP] * 5 + [BattingPhase.HANDS_PEAK] * 5 + [BattingPhase.CONTACT] * 3 + [BattingPhase.FOLLOW_THROUGH] * 7
    return PhaseResult(
        phase_labels=labels,
        setup_end=4,
        backlift_start=5,
        hands_peak=9,
        front_foot_down=8,
        contact=12,
        follow_through_start=15,
        backlift_to_contact_frames=7,
        fps=30.0,
        contact_confidence="low",
        contact_candidates={"c_a": 12, "c_b": 12, "c_c": 14},
        contact_window={"start": 10, "end": 14},
        contact_diagnostics={"signals": {}, "candidates": [12, 12, 14], "span": 2},
    )


def test_auto_only_keeps_resolved_equal_to_estimate():
    metrics = [_metric(i * 2) for i in range(20)]
    phases = apply_contact_override(_phases(), metrics, None)

    assert phases.contact == 12
    assert phases.estimated_contact_frame == 12
    assert phases.resolved_contact_frame == 12
    assert phases.resolved_contact_original_frame == metrics[12].frame_idx
    assert phases.resolved_contact_source == "auto"
    assert phases.resolved_contact_status == "estimated"


def test_manual_override_changes_resolved_contact_and_follow_through():
    metrics = [_metric(i * 2) for i in range(20)]
    phases = apply_contact_override(_phases(), metrics, 31)

    assert phases.estimated_contact_frame == 12
    assert phases.contact == 15
    assert phases.resolved_contact_frame == 15
    assert phases.resolved_contact_original_frame == 31
    assert phases.resolved_contact_source == "manual"
    assert phases.resolved_contact_status == "validated"
    assert phases.follow_through_start == 18
    assert phases.contact_window == {"start": 13, "end": 17}
    assert phases.phase_labels[15] == BattingPhase.CONTACT
    assert phases.phase_labels[18] == BattingPhase.FOLLOW_THROUGH


def test_low_confidence_weight_is_skipped_for_manual_contact():
    phases = _phases()
    phases.resolved_contact_source = "manual"
    faults = [Fault(rule_id="A1", fault="Around body", deduction=8, detail="", feedback="")]

    weighted = _apply_contact_confidence_weight(faults, phases)
    assert weighted[0].deduction == 8


def test_report_marks_contact_as_estimated_or_validated():
    phases = apply_contact_override(_phases(), [_metric(i * 2) for i in range(20)], 31)
    result = BattingIQResult(
        battingiq_score=80,
        score_band="Good",
        pillars={
            "access": PillarScore(name="access", score=20, status=TrafficLight.GREEN),
            "tracking": PillarScore(name="tracking", score=20, status=TrafficLight.GREEN),
            "stability": PillarScore(name="stability", score=20, status=TrafficLight.GREEN),
            "flow": PillarScore(name="flow", score=20, status=TrafficLight.GREEN),
        },
        priority_fix=None,
        development_notes=[],
        phases=phases,
        metadata={},
    )

    report = build_json_report(result)
    assert report["phases"]["contact"]["frame"] == 15
    assert report["phases"]["contact"]["estimated_frame"] == 12
    assert report["phases"]["contact"]["source"] == "manual"
    assert report["phases"]["contact"]["status"] == "validated"
    assert report["phases"]["contact"]["resolved_original_frame"] == 31


def test_report_surfaces_contact_detector_version_in_contact_phase():
    phases = apply_contact_override(_phases(), [_metric(i * 2) for i in range(20)], None)
    result = BattingIQResult(
        battingiq_score=80,
        score_band="Good",
        pillars={
            "access": PillarScore(name="access", score=20, status=TrafficLight.GREEN),
            "tracking": PillarScore(name="tracking", score=20, status=TrafficLight.GREEN),
            "stability": PillarScore(name="stability", score=20, status=TrafficLight.GREEN),
            "flow": PillarScore(name="flow", score=20, status=TrafficLight.GREEN),
        },
        priority_fix=None,
        development_notes=[],
        phases=phases,
        metadata={"contact_detector_version": "contact-consensus-3signal-v1"},
    )

    report = build_json_report(result)
    assert report["metadata"]["contact_detector_version"] == "contact-consensus-3signal-v1"
    assert report["phases"]["contact"]["detector_version"] == "contact-consensus-3signal-v1"


def test_contact_evaluation_groups_by_detector_version_and_handedness():
    detailed, summary = evaluate_rows([
        {
            "upload_id": "u1",
            "estimated_contact_original_frame": "40",
            "validated_contact_frame": "44",
            "fps": "30",
            "filming_angle": "front-on",
            "handedness": "right",
            "shot_family": "front_foot_drive",
            "shot_variant": "straight",
            "detector_version": "v1",
            "estimated_confidence": "medium",
        },
        {
            "upload_id": "u2",
            "estimated_contact_original_frame": "52",
            "validated_contact_frame": "50",
            "fps": "60",
            "filming_angle": "front-on",
            "handedness": "left",
            "shot_family": "front_foot_drive",
            "shot_variant": "offside",
            "detector_version": "v2",
            "estimated_confidence": "high",
        },
    ])

    assert len(detailed) == 2
    assert summary["rows_compared"] == 2
    assert summary["by_detector_version"]["v1"]["count"] == 1
    assert summary["by_detector_version"]["v2"]["mean_absolute_error"] == 2.0
    assert summary["by_handedness"]["right"]["mean_signed_error"] == -4.0
    assert summary["by_shot_type"]["front_foot_drive:straight"]["count"] == 1
