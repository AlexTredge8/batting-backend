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


# --- Phase 3 filming gate (16 Nov 2026): low resolution and a tilted phone are warnings, not refusals ---

import math  # noqa: E402

import config  # noqa: E402


def _standing(tilt_deg: float, w: float = 1080, h: float = 1920, n: int = 60, fps: float = 30.0):
    """A batter standing still; the ankle-to-shoulder line is tilt_deg off vertical in pixels
    (positive = shoulders camera-right of the ankles)."""
    ank_x, ank_y, sh_y = 0.5, 0.80, 0.45
    sh_x = ank_x + math.tan(math.radians(tilt_deg)) * (ank_y - sh_y) * h / w
    return [SimpleNamespace(detected=True, timestamp_s=i / fps, head_y=0.40, front_ankle_y=ank_y, back_ankle_y=ank_y,
                            front_ankle_x=ank_x - 0.03, back_ankle_x=ank_x + 0.03, shoulder_mid_x=sh_x, shoulder_mid_y=sh_y)
            for i in range(n)]


def _meta(w=1080, h=1920):
    return {"duration_s": 4.0, "width": w, "height": h}


def test_low_resolution_is_a_warning_on_the_short_side():
    phases = SimpleNamespace(setup_end=12, contact=45)
    info, warnings = run_analysis._capture_checks(_standing(0.0), phases, _meta(480, 848))
    assert info["short_side_px"] == 480
    assert len(warnings) == 1 and "low resolution" in warnings[0] and "1080p" in warnings[0]
    for w, h in ((1080, 1920), (1920, 1080), (720, 1280), (1280, 720)):  # 720 on the short side is allowed
        assert run_analysis._capture_checks(_standing(0.0, w, h), phases, _meta(w, h))[1] == [], (w, h)
    assert run_analysis._capture_checks(_standing(0.0), phases, _meta(1280, 719))[1]  # just under
    assert config.CAPTURE_MIN_SHORT_SIDE_PX == 720


def test_tilted_phone_is_a_warning_over_three_degrees():
    phases = SimpleNamespace(setup_end=12, contact=45)
    for tilt in (5.0, -5.0, 3.5):
        info, warnings = run_analysis._capture_checks(_standing(tilt), phases, _meta())
        assert abs(info["setup_axis_tilt_deg"] - tilt) < 0.05
        assert len(warnings) == 1 and "phone looks tilted" in warnings[0] and "level" in warnings[0], tilt
    for tilt in (0.0, 2.0, -2.9):
        assert run_analysis._capture_checks(_standing(tilt), phases, _meta())[1] == [], tilt
    assert config.CAPTURE_MAX_TILT_DEG == 3.0


def test_tilt_uses_only_the_contact_anchored_set_up_window():
    frames = _standing(0.0)
    for m in frames[30:]:  # after the set-up window (1000 to 700 ms before contact at frame 45): a big lean
        m.shoulder_mid_x += 0.2
    info, warnings = run_analysis._capture_checks(frames, SimpleNamespace(setup_end=12, contact=45), _meta())
    assert abs(info["setup_axis_tilt_deg"]) < 0.05 and warnings == []


def test_tilt_unknown_without_contact_or_frame_size():
    _, warnings = run_analysis._capture_checks(_standing(8.0), SimpleNamespace(setup_end=12), _meta())
    assert warnings == []  # no contact: no window, so no tilt check
    info, warnings = run_analysis._capture_checks(_standing(8.0), SimpleNamespace(setup_end=12, contact=45), {"duration_s": 4.0})
    assert info["setup_axis_tilt_deg"] is None and info["short_side_px"] is None and warnings == []
    info, _ = run_analysis._capture_checks(_standing(8.0), SimpleNamespace(setup_end=12, contact=10), _meta())
    assert info["setup_axis_tilt_deg"] is None  # set-up before the clip starts
