"""
Tests for six-anchor overrides, suppression, and evaluation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_anchor_labels import evaluate_rows
from models import BattingPhase, Fault, FrameMetrics, PhaseResult, PillarScore, TrafficLight
from phase_detector import apply_anchor_overrides
from scorer import build_scores


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
        contact_confidence="high",
        contact_candidates={"c_a": 12},
        contact_window={"start": 10, "end": 14},
        contact_diagnostics={"signals": {}, "candidates": [12], "span": 0},
    )


def test_apply_anchor_overrides_replays_all_six_anchor_frames():
    metrics = [_metric(i * 2) for i in range(20)]
    phases = apply_anchor_overrides(
        _phases(),
        metrics,
        {
            "setup_frame": 6,
            "hands_start_up_frame": 10,
            "front_foot_down_frame": 14,
            "hands_peak_frame": 18,
            "contact_frame": 24,
            "follow_through_frame": 32,
        },
    )

    assert phases.setup_end == 3
    assert phases.backlift_start == 5
    assert phases.front_foot_down == 7
    assert phases.hands_peak == 9
    assert phases.contact == 12
    assert phases.resolved_contact_original_frame == 24
    assert phases.follow_through_start == 16
    assert phases.phase_labels[12] == BattingPhase.CONTACT
    assert phases.phase_labels[16] == BattingPhase.FOLLOW_THROUGH


def test_build_scores_suppresses_rules_bound_to_low_confidence_anchors():
    fault_map = {
        "access": [
            Fault(rule_id="A6", fault="Backlift shape", deduction=5, detail="", feedback=""),
            Fault(rule_id="A2", fault="Contact access", deduction=4, detail="", feedback=""),
        ],
        "tracking": [],
        "stability": [],
        "flow": [],
        "_evaluation": {"rules_total": 21, "rules_evaluated": 21, "rules_failed": []},
    }
    phases = _phases()
    result = build_scores(
        fault_map,
        phases,
        baseline={},
        video_meta={
            "anchor_confidence": {
                "setup_frame": "low",
                "hands_start_up_frame": "high",
                "hands_peak_frame": "high",
                "contact_frame": "high",
            },
        },
    )

    access_faults = result.pillars["access"].faults
    assert [fault.rule_id for fault in access_faults] == ["A2"]
    assert result.metadata["suppressed_rules"] == [
        {
            "rule_id": "A6",
            "anchor_keys": ["setup_frame"],
            "reason": "suppressed_due_to_low_anchor_confidence",
        }
    ]
    assert result.metadata["rule_evaluation"]["rules_suppressed"] == 1


def test_anchor_evaluation_groups_by_anchor_and_reports_adjudication():
    detailed, summary = evaluate_rows([
        {
            "upload_id": "u1",
            "truth_status": "truth_ready",
            "handedness": "right",
            "shot_family": "front_foot_drive",
            "shot_variant": "straight",
            "filming_angle": "front-on",
            "fps": "30",
            "detector_version": "anchor-heuristics-v1",
            "ai_setup_frame": "10",
            "ai_setup_frame_confidence": "medium",
            "truth_setup_frame": "12",
            "ai_contact_frame": "44",
            "ai_contact_frame_confidence": "high",
            "truth_contact_frame": "43",
        },
        {
            "upload_id": "u2",
            "truth_status": "needs_adjudication",
            "handedness": "left",
            "shot_family": "front_foot_drive",
            "shot_variant": "offside",
            "filming_angle": "front-on",
            "fps": "60",
            "detector_version": "anchor-heuristics-v1",
            "ai_setup_frame": "20",
            "ai_setup_frame_confidence": "low",
            "truth_setup_frame": "18",
        },
    ])

    assert len(detailed) == 3
    assert summary["anchor_observations"] == 3
    assert summary["needs_adjudication_count"] == 1
    assert summary["by_anchor"]["setup_frame"]["count"] == 2
    assert summary["by_anchor"]["contact_frame"]["mean_absolute_error"] == 1.0
    assert summary["by_handedness"]["right"]["count"] == 2
    assert summary["by_confidence_bucket"]["low"]["count"] == 1
