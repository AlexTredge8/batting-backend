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


class _Phases:
    contact_confidence = "high"
    resolved_contact_source = "auto"
    backlift_to_contact_frames = 20

    def __getattr__(self, name):
        return None


def _faults(**pillars):
    from models import Fault
    out = {p: [] for p in ("access", "tracking", "stability", "flow")}
    for p, (rid, ded) in pillars.items():
        out[p] = [Fault(rule_id=rid, fault="x", deduction=ded, detail="", feedback="")]
    return out


def test_unknown_pillar_takes_average_of_measured_pillars():
    import scorer
    meas = {"A5_shoulder_hip_gap_deg": -30.0, "T2_early_head_change": None,
            "S4_post_contact_rotation_deg": 15.0, "F4_pause_frames": 6}
    res = scorer.build_scores(_faults(access=("A5", 6), stability=("S4", 8), flow=("F4", 12)),
                              _Phases(), {}, {"rule_measurements": meas})
    p = {k: v.score for k, v in res.pillars.items()}
    known = [p["access"], p["stability"], p["flow"]]
    assert p["tracking"] == round(sum(known) / 3)        # not 25
    assert res.battingiq_score == sum(p.values())
    assert res.metadata["imputed_pillars"]["tracking"]["raw_score"] == 25
    assert res.metadata["pillar_basis"]["tracking"] == "not_measured"
    assert res.pillars["tracking"].positives == []


def test_measured_no_fault_pillar_keeps_its_score():
    import scorer
    meas = {"A5_shoulder_hip_gap_deg": -30.0, "T2_early_head_change": 0.01,
            "S4_post_contact_rotation_deg": 2.0, "F4_pause_frames": 3}
    res = scorer.build_scores(_faults(access=("A5", 6)), _Phases(), {}, {"rule_measurements": meas})
    assert {k: v.score for k, v in res.pillars.items()}["tracking"] == 25
    assert res.metadata["imputed_pillars"] == {}


def test_report_annotation_reads_scorer_decision():
    report = {"metadata": {"pillar_basis": {"access": "fault_found", "tracking": "not_measured",
                                            "stability": "suppressed", "flow": "no_fault_found"},
                           "imputed_pillars": {"tracking": {"raw_score": 25}, "stability": {"raw_score": 25}}},
              "pillars": {k: {"score": 15, "faults": []} for k in ("access", "tracking", "stability", "flow")}}
    run_analysis._annotate_pillar_evidence(report)
    p = report["pillars"]
    assert p["tracking"]["assessed"] is False and p["tracking"]["imputed"] is True and p["tracking"]["raw_score"] == 25
    assert p["flow"]["assessed"] is True and p["flow"]["imputed"] is False
