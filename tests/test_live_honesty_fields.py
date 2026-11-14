"""Capture warnings and pillar evidence added to the live response (Phase 0, Nov 2026). No score changes."""

from types import SimpleNamespace

import run_analysis


def _m(head_y, ankle_y):
    return SimpleNamespace(detected=True, head_y=head_y, front_ankle_y=ankle_y, back_ankle_y=ankle_y - 0.01)


def test_capture_checks_small_batter_and_short_clip():
    phases = SimpleNamespace(setup_end=12)
    small = [_m(0.50, 0.58)] * 20
    info, warnings = run_analysis._capture_checks(small, phases, {"duration_s": 1.4})
    assert info["batter_height_fraction"] == 0.08
    assert len(warnings) == 2
    assert "Move closer" in warnings[0] and "very short" in warnings[1]


def test_capture_checks_good_clip_has_no_warnings():
    phases = SimpleNamespace(setup_end=12)
    big = [_m(0.30, 0.80)] * 20
    info, warnings = run_analysis._capture_checks(big, phases, {"duration_s": 4.0})
    assert info["batter_height_fraction"] == 0.5
    assert warnings == []


def test_pillar_evidence_marks_unmeasured_and_suppressed():
    report = {
        "metadata": {
            "rule_measurements": {"A5_shoulder_hip_gap_deg": -30.0, "T2_early_head_change": None,
                                  "S4_post_contact_rotation_deg": 2.0, "F4_pause_frames": 3},
            "rule_evaluation": {"suppressed_rule_ids": ["S4"]},
        },
        "pillars": {
            "access": {"score": 17, "faults": [{"rule_id": "A5", "deduction": 3}]},
            "tracking": {"score": 25, "faults": []},
            "stability": {"score": 25, "faults": []},
            "flow": {"score": 25, "faults": []},
        },
    }
    run_analysis._annotate_pillar_evidence(report)
    p = report["pillars"]
    assert (p["access"]["basis"], p["access"]["assessed"]) == ("fault_found", True)
    assert (p["tracking"]["basis"], p["tracking"]["assessed"]) == ("not_measured", False)
    assert (p["stability"]["basis"], p["stability"]["assessed"]) == ("suppressed", False)
    assert (p["flow"]["basis"], p["flow"]["assessed"]) == ("no_fault_found", True)
    assert p["access"]["score"] == 17  # scores untouched
