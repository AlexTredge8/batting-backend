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


# From 16 Nov 2026 (v8) the tilt check uses the camera roll from the background's vertical lines, not
# the batter's standing axis (which also holds a real lean). The line detector is mocked here.

import numpy as np  # noqa: E402

import capture_geometry  # noqa: E402


def _lines(monkeypatch, angles, length=600.0, n_frames=5):
    """Mock the video read and the line detector: each frame gives one long line per angle (degrees from
    vertical, positive = top camera-right), spread across the picture."""
    calls = {"read": 0, "boxes": []}

    def fake_read(path, indices):
        calls["read"] += 1
        idx = list(indices) if indices is not None else list(range(n_frames))
        return idx, [np.zeros((1920, 1080, 3), np.uint8) for _ in idx]

    def fake_segments(frame, box=None):
        calls["boxes"].append(box)
        out = []
        for j, a in enumerate(angles):
            x0, y0 = 60.0 + 80 * j, 1500.0
            t = np.radians(a)
            out.append((x0, y0, x0 + length * np.sin(t), y0 - length * np.cos(t)))
        return np.array(out, float).reshape(-1, 4)

    monkeypatch.setattr(capture_geometry, "_read_frames", fake_read)
    monkeypatch.setattr(capture_geometry, "line_segments", fake_segments)
    return calls


def _poses(n=60):
    lm = [SimpleNamespace(x=0.5 + 0.01 * (k % 3), y=0.3 + 0.015 * k, visibility=0.9) for k in range(33)]
    return [SimpleNamespace(frame_idx=i, landmarks=lm) for i in range(n)]


def test_tilted_phone_is_a_warning_over_three_degrees(monkeypatch):
    phases = SimpleNamespace(setup_end=12, contact=45)
    for roll in (5.0, -5.0, 3.5):
        calls = _lines(monkeypatch, [roll + d for d in (-0.4, -0.2, 0.0, 0.0, 0.2, 0.4, 0.1)])
        info, warnings = run_analysis._capture_checks(_standing(0.0), phases, _meta(), video_path="clip.mp4",
                                                      frame_poses=_poses())
        assert abs(info["camera_roll_deg"] - roll) < 0.15 and info["roll_confidence"] == "ok"
        assert len(warnings) == 1 and "phone looks tilted" in warnings[0] and "level" in warnings[0], roll
        assert calls["read"] == 1 and all(b is not None for b in calls["boxes"])  # the batter is masked out
    for roll in (0.0, 2.0, -2.9):
        _lines(monkeypatch, [roll + d for d in (-0.4, -0.2, 0.0, 0.0, 0.2, 0.4, 0.1)])
        assert run_analysis._capture_checks(_standing(0.0), phases, _meta(), video_path="clip.mp4",
                                            frame_poses=_poses())[1] == [], roll
    assert config.CAPTURE_MAX_TILT_DEG == 3.0


def test_leaning_batter_with_a_level_phone_is_not_a_warning(monkeypatch):
    """The reference clip's case: the batter leans 7 degrees, the background is upright."""
    _lines(monkeypatch, [0.0, 0.3, -0.2, 0.1, 0.0, -0.1, 0.2])
    info, warnings = run_analysis._capture_checks(_standing(-7.0), SimpleNamespace(setup_end=12, contact=45), _meta(),
                                                  video_path="clip.mp4", frame_poses=_poses())
    assert abs(info["setup_axis_tilt_deg"] + 7.0) < 0.05  # still reported, for information
    assert abs(info["camera_roll_deg"]) < 0.2 and warnings == []


def test_no_tilt_warning_when_the_roll_is_unknown(monkeypatch):
    phases = SimpleNamespace(setup_end=12, contact=45)
    _lines(monkeypatch, [8.0, 8.2, 7.9])  # too few lines
    info, warnings = run_analysis._capture_checks(_standing(8.0), phases, _meta(), video_path="clip.mp4", frame_poses=_poses())
    assert info["camera_roll_deg"] is None and info["roll_confidence"] == "unknown" and warnings == []
    _lines(monkeypatch, [8.0, 2.0, 14.0, -5.0, 11.0, 6.0, 0.0])  # lines disagree (spread over 3 degrees)
    info, warnings = run_analysis._capture_checks(_standing(8.0), phases, _meta(), video_path="clip.mp4", frame_poses=_poses())
    assert info["roll_confidence"] == "unknown" and warnings == []
    info, warnings = run_analysis._capture_checks(_standing(8.0), phases, _meta())  # no video: no roll, no warning
    assert info["camera_roll_deg"] is None and warnings == []
    assert abs(info["setup_axis_tilt_deg"] - 8.0) < 0.05


def test_a_failed_read_never_breaks_the_analysis(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("cannot decode")
    monkeypatch.setattr(capture_geometry, "_read_frames", boom)
    info, warnings = run_analysis._capture_checks(_standing(0.0), SimpleNamespace(setup_end=12, contact=45), _meta(),
                                                  video_path="clip.mp4", frame_poses=_poses())
    assert info["roll_confidence"] == "unknown" and warnings == []


def test_axis_tilt_uses_only_the_contact_anchored_set_up_window():
    frames = _standing(0.0)
    for m in frames[30:]:  # after the set-up window (1000 to 700 ms before contact at frame 45): a big lean
        m.shoulder_mid_x += 0.2
    info, warnings = run_analysis._capture_checks(frames, SimpleNamespace(setup_end=12, contact=45), _meta())
    assert abs(info["setup_axis_tilt_deg"]) < 0.05 and warnings == []


def test_axis_tilt_unknown_without_contact_or_frame_size():
    info, _ = run_analysis._capture_checks(_standing(8.0), SimpleNamespace(setup_end=12), _meta())
    assert info["setup_axis_tilt_deg"] is None  # no contact: no window
    info, warnings = run_analysis._capture_checks(_standing(8.0), SimpleNamespace(setup_end=12, contact=45), {"duration_s": 4.0})
    assert info["setup_axis_tilt_deg"] is None and info["short_side_px"] is None and warnings == []
    info, _ = run_analysis._capture_checks(_standing(8.0), SimpleNamespace(setup_end=12, contact=10), _meta())
    assert info["setup_axis_tilt_deg"] is None  # set-up before the clip starts
