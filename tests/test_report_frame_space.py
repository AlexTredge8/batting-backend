"""
Report timings must be expressed in ORIGINAL video frames / milliseconds even
when the extractor subsampled frames (frame_step > 1), while ``frame`` keeps the
metric-index semantics every detector and rule uses.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import BattingIQResult, BattingPhase, PhaseResult, PillarScore, TrafficLight
from report_generator import build_json_report


def _result(anchor_frames: dict | None) -> BattingIQResult:
    phases = PhaseResult(
        phase_labels=[BattingPhase.SETUP] * 60,
        setup_end=10, backlift_start=13, hands_peak=34, front_foot_down=39,
        contact=40, follow_through_start=54,
        hands_peak_vs_ffd_diff=-5, hands_peak_vs_ffd_ms=-166.5,
        backlift_to_contact_frames=27, fps=30.0,
        resolved_contact_original_frame=80, estimated_contact_original_frame=80,
    )
    metadata = {"anchor_frames": anchor_frames} if anchor_frames else {}
    return BattingIQResult(
        battingiq_score=100, score_band="Excellent",
        pillars={n: PillarScore(name=n, score=25, status=TrafficLight.GREEN)
                 for n in ("access", "tracking", "stability", "flow")},
        priority_fix=None, development_notes=[], phases=phases, metadata=metadata,
    )


def test_subsampled_report_uses_original_frames_for_ms():
    # frame_step=2: metric index k ↔ original frame 2k
    anchors = {
        "setup_frame": {"metric_index": 10, "original_frame": 20},
        "hands_start_up_frame": {"metric_index": 13, "original_frame": 26},
        "hands_peak_frame": {"metric_index": 34, "original_frame": 68},
        "front_foot_down_frame": {"metric_index": 39, "original_frame": 78},
        "contact_frame": {"metric_index": 40, "original_frame": 80},
        "follow_through_frame": {"metric_index": 54, "original_frame": 108},
    }
    report = build_json_report(_result(anchors))
    ph = report["phases"]

    assert ph["hands_peak"]["frame"] == 34                 # metric index preserved
    assert ph["hands_peak"]["original_frame"] == 68
    assert ph["hands_peak"]["ms"] == round(68 / 30 * 1000, 1)
    assert ph["setup"]["end_ms"] == round(20 / 30 * 1000, 1)
    assert ph["contact"]["original_frame"] == 80
    assert ph["contact"]["ms"] == round(80 / 30 * 1000, 1)
    assert ph["follow_through"]["original_frame"] == 108

    timing = ph["timing"]
    assert timing["hands_peak_vs_ffd_frames"] == -5
    assert timing["hands_peak_vs_ffd_original_frames"] == -10
    assert timing["hands_peak_vs_ffd_ms"] == round(-10 / 30 * 1000, 1)
    assert timing["backlift_to_contact_original_frames"] == 80 - 26
    assert timing["backlift_to_contact_ms"] == round((80 - 26) / 30 * 1000, 1)


def test_full_rate_report_is_identity():
    report = build_json_report(_result(None))
    ph = report["phases"]
    assert ph["hands_peak"]["original_frame"] == 34
    assert ph["hands_peak"]["ms"] == round(34 / 30 * 1000, 1)
    assert ph["timing"]["hands_peak_vs_ffd_ms"] == round(-5 / 30 * 1000, 1)
