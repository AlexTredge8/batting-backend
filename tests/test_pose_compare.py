"""You vs best practice (pose_reference.py, pose_compare.py), on synthetic landmarks (no video needed)."""

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase5_checkpoints as p5  # noqa: E402
import pose_compare as pc  # noqa: E402
import pose_reference as pr  # noqa: E402
from test_phase5_checkpoints import ANCH, FPS, H, N, W, _figure, _lefty, _rotate  # noqa: E402

FRAMES = np.arange(N)


def _clip(arr, hand="right", roll=None):
    return p5.clip_from_arrays(arr, FRAMES, W, H, FPS, hand, "off", roll)


def _roll(deg):
    return {"roll_deg": deg, "confidence": "ok", "source": "vertical", "n_lines": 10}


def _close(a: dict, b: dict, atol: float) -> None:
    assert a.keys() == b.keys()
    for m in a:
        assert a[m].keys() == b[m].keys(), m
        for j in a[m]:
            assert np.allclose(a[m][j], b[m][j], atol=atol), (m, j, a[m][j], b[m][j])


def _scaled(arr, k, dx, dy):
    out = arr.copy()
    out[..., 0] = (arr[..., 0] * W * k + dx) / W
    out[..., 1] = (arr[..., 1] * H * k + dy) / H
    return out


# --- normalisation ---------------------------------------------------------------------------

def test_normalised_pose_is_translation_and_scale_invariant():
    base = pr.clip_poses(_clip(_figure()), ANCH)
    moved = pr.clip_poses(_clip(_scaled(_figure(), 1.4, -60.0, -150.0)), ANCH)
    assert base["setup"] and base["contact"]
    _close(base, moved, 1e-6)
    # the origin is the set-up ankle midpoint and the unit the body length
    assert np.allclose(np.mean([base["setup"]["front_ankle"], base["setup"]["back_ankle"]], axis=0), [0, 0], atol=1e-6)
    assert base["setup"]["nose"][1] == pytest.approx(-1.0, abs=1e-6)


def test_left_hander_is_mirrored_to_a_right_hander():
    rh = pr.clip_poses(_clip(_figure(head_setup_off=20)), ANCH)
    lh = pr.clip_poses(_clip(_lefty(_figure(head_setup_off=20)), "left"), ANCH)
    _close(rh, lh, 1e-6)


def test_camera_roll_is_removed():
    level = pr.clip_poses(_clip(_figure()), ANCH)
    rolled = pr.clip_poses(_clip(_rotate(_figure(), 3.0), roll=_roll(3.0)), ANCH)
    _close(level, rolled, 0.005)  # the 5-frame median is not exactly rotation-equivariant
    not_levelled = pr.clip_poses(_clip(_rotate(_figure(), 3.0)), ANCH)
    assert abs(not_levelled["setup"]["nose"][0] - level["setup"]["nose"][0]) > 0.03  # 3 degrees ~ 5% BL


def test_median_pose_across_clips():
    a = {"nose": [0.0, -1.0], "hands": [0.1, -0.5]}
    b = {"nose": [0.2, -1.0]}
    c = {"nose": [0.4, -0.9], "hands": [0.3, -0.5]}
    m, n = pr.median_pose([a, b, c])
    assert n == 3 and m["nose"] == [0.2, -1.0] and m["hands"] == [0.2, -0.5]


# --- the fault table -------------------------------------------------------------------------

def test_fault_table_covers_every_fault():
    rubric = json.loads((ROOT / "docs" / "phase1" / "rubric_v0_3.json").read_text())
    ids = {f["id"] for f in rubric["faults_in_chain_order"]} | {f["id"] for f in rubric["faults_retired"]} | {"F00"}
    tips = json.loads((ROOT / "docs" / "phase6" / "fault_tips_v0_3.json").read_text())["tips"]
    assert ids <= set(pc.FAULT_TABLE), ids - set(pc.FAULT_TABLE)
    assert set(tips) <= set(pc.FAULT_TABLE)
    measures = {m[0]: m for m in p5.MEASURES}
    for fid, spec in pc.FAULT_TABLE.items():
        if spec.get("unavailable"):
            assert spec["moment"] is None
            continue
        assert spec["moment"] in pr.MOMENTS, fid
        assert set(spec["joints"]) <= set(pr.JOINTS), fid
        assert spec["align"] in ("setup_ankles", "shoulders", "hips") or spec["align"] in pr.JOINTS, fid
        if spec.get("measurement"):
            m = measures[spec["measurement"]]
            assert m[2] == (spec.get("retired_into") or fid), (fid, m)  # the measurement belongs to this fault
            assert spec["subject"] and spec["action"]
        if not spec.get("nothing_to_fix"):
            assert spec["joints"], fid


def test_live_rules_map_to_faults():
    from config import PILLAR_RULE_MAX_DEDUCTION
    active = {r for rules in PILLAR_RULE_MAX_DEDUCTION.values() for r in rules}
    assert active <= set(pc.RULE_TO_FAULT)
    assert pc.RULE_TO_FAULT == {"T2": "F01", "A5": "F07", "S4": "F12", "F4": "F09"}
    assert pc.fault_for_priority_fix({"rule_id": "T2"})[0] == "F01"
    assert pc.fault_for_priority_fix(None)[0] == "F00"
    assert pc.fault_for_priority_fix({"rule_id": "Z9"})[0] is None
    # the same steps as the results page: no priority fix but a fault elsewhere is NOT "nothing to fix"
    report = {"priority_fix": None, "pillars": {
        "stability": {"assessed": False, "faults": []},
        "flow": {"assessed": True, "faults": [{"rule_id": "F4", "deduction": 6}]},
        "access": {"assessed": True, "faults": [{"rule_id": "A5", "deduction": 3}]}}}
    assert pc.fault_for_report(report)[0] == "F09"
    assert pc.fault_for_report({"priority_fault": {"id": "f05"}, "priority_fix": {"rule_id": "T2"}})[0] == "F05"
    none_measured = {"priority_fix": None, "pillars": {"flow": {"assessed": False, "faults": []}}}
    assert pc.fault_for_report(none_measured)[0] is None
    clean = {"priority_fix": None, "pillars": {"flow": {"assessed": True, "faults": []}}}
    assert pc.fault_for_report(clean)[0] == "F00"


# --- change_text -----------------------------------------------------------------------------

def _m(value, lo, hi, unit="% BL", direction="higher_worse", status="outside"):
    return {"value": value, "reference_range": [lo, hi], "unit": unit, "direction": direction, "status": status}


def test_change_text_formatting():
    t = pc.change_text("F16", _m(15.2, 2.1, 4.0))
    assert t == ("At contact the middle of your body is 15% of your height to the side of your front foot; "
                 "good drives are within 4%. Get your head and chest over your front foot as you hit the ball.")
    t = pc.change_text("F10", _m(-6.0, 16.2, 31.0, "% BL, off side +", "higher_better"))
    assert t.startswith("At contact your head is 6% of your height on the leg side of your hips, falling away; "
                        "good drives are between 16% and 31% of your height to the off side of your hips.")
    t = pc.change_text("F05", _m(-9.0, -4.4, 13.9, "% BL, drive side +", "higher_better"))
    assert "between 4% towards the leg side and 14% of your height towards the off side" in t
    t = pc.change_text("F07", _m(2.4, 0.5, 1.85, "ratio"))
    assert "2.4 times as wide as at set-up" in t and "at 1.9 times or less" in t
    t = pc.change_text("F06", _m(171.0, 153.0, 167.0, "degrees"))
    assert "your front knee is at 171 degrees" in t and "167 degrees or less" in t
    inside = pc.change_text("F16", _m(3.0, 2.1, 4.0, status="inside"))
    assert inside.endswith("so on this measure you are already in the good range.")
    assert "compare your back knee and back ankle" in pc.change_text("F18", None)
    assert pc.change_text("F00", None, "Nothing major to fix.") == "Nothing major to fix."


def test_status_is_outside_only_on_the_worse_side():
    assert pc.measurement_status(5.0, 1.0, 4.0, "higher_worse") == ("outside", "above")
    assert pc.measurement_status(0.5, 1.0, 4.0, "higher_worse") == ("inside", "below")
    assert pc.measurement_status(0.5, 1.0, 4.0, "higher_better") == ("outside", "below")
    assert pc.measurement_status(9.0, 1.0, 4.0, "higher_better") == ("inside", "above")


# --- the comparison and the pictures --------------------------------------------------------

def _reference(arrs_hands=(("right", None),)):
    poses = [pr.clip_poses(_clip(_figure()), ANCH)]
    ranges = {"M10": {"name": "Head over the ball at contact", "unit": "% BL, off side +", "direction": "higher_better",
                      "low": -2.0, "high": 6.0, "median": 2.0, "n": 8},
              "M05": {"name": "Front foot step", "unit": "% BL, drive side +", "direction": "higher_better",
                      "low": 2.0, "high": 12.0, "median": 6.0, "n": 8}}
    return pr.build_reference(poses * 3, ranges, {"pose_n_clips": 3})


def _noise_frame(seed=1):
    return np.random.default_rng(seed).integers(0, 255, (int(H), int(W), 3), dtype=np.uint8)


def _head_falls_away(arr, px=60.0):
    out = arr.copy()
    out[70:, 0, 0] += px / W  # the head moves to the leg side (camera-right) from foot-down on
    return out


def test_comparison_on_a_matching_player_marks_nothing():
    ref = _reference()
    out = pc.compute_pose_comparison(_figure(), FRAMES, ANCH, "right", FPS, W, H, "F10", reference=ref,
                                     frame_image=_noise_frame())
    assert out["available"] and out["moment"] == "contact" and out["frame"] == ANCH["contact_frame"]
    assert out["measurement"]["id"] == "M10" and out["measurement"]["status"] == "inside"
    assert not any(j["highlighted"] for j in out["joints"])
    assert out["reference"]["anonymised"] and not out["reference"]["provisional"]


def test_overlay_renders_with_the_differing_joint_marked():
    ref = _reference()
    frame = _noise_frame()
    out = pc.compute_pose_comparison(_head_falls_away(_figure()), FRAMES, ANCH, "right", FPS, W, H, "F10",
                                     reference=ref, frame_image=frame)
    assert out["measurement"]["status"] == "outside" and out["measurement"]["value"] < -2.0
    nose = next(j for j in out["joints"] if j["name"] == "nose")
    assert nose["highlighted"] and nose["difference_pct_height"] > 5
    assert nose["player_xy"][0] > nose["reference_xy"][0]  # the head is camera-right of where it should be
    assert out["change_text"].startswith("At contact your head is")
    import cv2
    for key in ("overlay_data_url", "side_by_side_data_url"):
        url = out["images"][key]
        assert url.startswith("data:image/jpeg;base64,")
        img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[1] == pc.OUT_WIDTH
    assert out["privacy"]["face_blurred"] is True
    assert frame.sum() == _noise_frame().sum()  # the caller's frame is not drawn on


def test_ghost_lands_on_a_left_handers_picture():
    ref = _reference()
    rh = pc.compute_pose_comparison(_head_falls_away(_figure()), FRAMES, ANCH, "right", FPS, W, H, "F10", reference=ref,
                                    render=False)
    lh = pc.compute_pose_comparison(_lefty(_head_falls_away(_figure())), FRAMES, ANCH, "left", FPS, W, H, "F10",
                                    reference=ref, render=False)
    assert lh["measurement"]["value"] == pytest.approx(rh["measurement"]["value"], abs=1e-6)
    for a, b in zip(rh["joints"], lh["joints"]):
        assert a["name"] == b["name"]
        assert b["player_xy"][0] == pytest.approx(1 - a["player_xy"][0], abs=1e-3)
        assert b["reference_xy"][0] == pytest.approx(1 - a["reference_xy"][0], abs=1e-3)


def test_ghost_follows_a_rolled_camera():
    ref = _reference()
    level = pc.compute_pose_comparison(_figure(), FRAMES, ANCH, "right", FPS, W, H, "F05", reference=ref, render=False)
    rolled = pc.compute_pose_comparison(_rotate(_figure(), 3.0), FRAMES, ANCH, "right", FPS, W, H, "F05",
                                        reference=ref, render=False, roll=_roll(3.0))
    assert rolled["measurement"]["value"] == pytest.approx(level["measurement"]["value"], abs=0.3)
    # in the rolled picture the ghost is rolled with the player, so it still sits on the player
    for j in rolled["joints"]:
        assert j["difference_pct_height"] < 1.0, j


def test_blur_face_changes_only_the_head():
    import cv2  # noqa: F401
    img = _noise_frame(3)
    before = img.copy()
    pts = _figure()[ANCH["contact_frame"], :, :2] * [W, H]
    assert pc.blur_face(img, pts)
    nx, ny = pts[0].astype(int)
    assert not np.array_equal(img[ny - 3:ny + 3, nx - 3:nx + 3], before[ny - 3:ny + 3, nx - 3:nx + 3])
    assert np.array_equal(img[700:, :], before[700:, :])  # the feet are untouched


def test_nothing_to_fix_and_not_available():
    ref = _reference()
    f00 = pc.compute_pose_comparison(_figure(), FRAMES, ANCH, "right", FPS, W, H, "F00", reference=ref,
                                     frame_image=_noise_frame())
    assert f00["available"] and f00["nothing_to_fix"] and f00["joints"] == [] and f00["measurement"] is None
    assert f00["change_text"].startswith("Nothing major to fix")
    assert f00["images"]["overlay_data_url"]
    f19 = pc.compute_pose_comparison(_figure(), FRAMES, ANCH, "right", FPS, W, H, "F19", reference=ref)
    assert not f19["available"] and "choosing which ball" in f19["reason"]
    none = pc.compute_pose_comparison(_figure(), FRAMES, ANCH, "right", FPS, W, H, "F10", reference={})
    assert not none["available"] and "not been built" in none["reason"]
    short = pc.compute_pose_comparison(_figure(), FRAMES, dict(ANCH, contact_frame=20), "right", FPS, W, H, "F10",
                                       reference=ref)
    assert not short["available"] and "set-up" in short["reason"]


def test_every_fault_runs_on_the_synthetic_figure():
    ref = _reference()
    for fid in pc.FAULT_TABLE:
        out = pc.compute_pose_comparison(_figure(), FRAMES, ANCH, "right", FPS, W, H, fid, reference=ref, render=False)
        assert out["fault"] == fid
        assert out["available"] or out["reason"], fid
        if out["available"]:
            assert out["change_text"] and out["moment"] in pr.MOMENTS


# --- the reference file and the live switch -------------------------------------------------

def test_reference_file_holds_only_averaged_figures_and_ranges():
    ref = pr.load_reference()
    assert ref and ref["version"] == "pose_reference_v1"
    allowed = {"version", "built", "provisional", "anonymised", "status", "privacy", "coordinates", "sources", "joints",
               "joint_labels", "bones", "moments", "measurement_ranges", "range_rule"}
    assert set(ref) <= allowed
    text = json.dumps(ref).lower()
    for word in ("viv", ".mov", ".mp4", "filename", "player_id", "data:image"):
        assert word not in text, word
    assert set(ref["moments"]) == set(pr.MOMENTS)
    for m in ref["moments"].values():
        assert set(m) == {"label", "definition", "n_clips", "joints"}
        assert set(m["joints"]) <= set(pr.JOINTS)
    if ref["sources"].get("pose_n_clips", 0) < pr.MIN_CLIPS_ANONYMISED:
        assert ref["provisional"] and not ref["anonymised"]
    for r in ref["measurement_ranges"].values():
        assert set(r) == {"name", "unit", "direction", "low", "high", "median", "n"} and r["low"] <= r["high"]


def test_reference_cli_builds_from_a_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for name, arr in (("Clip A Elite", _figure()), ("Clip B Elite", _figure(head_setup_off=10)),
                      ("Clip C Good", _figure())):
        np.savez_compressed(cache / f"{name}.npz", landmarks=arr.astype(np.float32), frame_idx=FRAMES, fps=FPS,
                            width=W, height=H)
    anchors = tmp_path / "anchors.csv"
    anchors.write_text("filename," + ",".join(p5.ANCHORS) + "\n" + "".join(
        f"{n}.mov," + ",".join(str(ANCH[k]) for k in p5.ANCHORS) + "\n" for n in ("Clip A Elite", "Clip B Elite", "Clip C Good")))
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("filename,tier_csv,handedness,drive_type\nClip A Elite.mov,Elite,right,off\n"
                     "Clip B Elite.mov,Elite,right,off\nClip C Good.mov,Good Club,right,off\n")
    table = tmp_path / "m.csv"
    table.write_text("filename,measurement_set,anchor_set,measurement,unit,direction,value\n" + "".join(
        f"Clip {c} {t},v8,validated,M10,\"% BL, off side +\",higher_better,{v}\n"
        for c, t, v in (("A", "Elite", 10), ("B", "Elite", 20), ("C", "Good", 30))))
    out = tmp_path / "ref.json"
    assert pr.main(["--cache-dir", str(cache), "--anchors", str(anchors), "--clip-sheet", str(sheet),
                    "--measurements", str(table), "--out", str(out)]) == 0
    ref = json.loads(out.read_text())
    assert ref["sources"]["pose_n_clips"] == 2 and ref["provisional"]  # Elite only by default: 2 < 3
    assert ref["measurement_ranges"]["M10"]["n"] == 3 and ref["measurement_ranges"]["M10"]["low"] == pytest.approx(12.0)
    assert "clip a" not in out.read_text().lower() and "clip b" not in out.read_text().lower()
    assert pr.main(["--cache-dir", str(cache), "--anchors", str(anchors), "--clip-sheet", str(sheet),
                    "--measurements", str(table), "--out", str(out), "--include-good-club"]) == 0
    ref = json.loads(out.read_text())
    assert ref["sources"]["pose_n_clips"] == 3 and ref["anonymised"] and not ref["provisional"]


def test_flag_off_adds_nothing_and_errors_never_break_the_analysis(monkeypatch):
    import config
    import run_analysis
    report = {"battingiq_score": 92, "priority_fix": {"rule_id": "T2"}, "pillars": {"tracking": {"score": 20}}}
    before = json.loads(json.dumps(report))
    monkeypatch.setattr(config, "POSE_COMPARE", "off")
    run_analysis._add_pose_comparison(report, [], {}, "nothing.mov")
    assert report == before
    monkeypatch.setattr(config, "POSE_COMPARE", "on")

    def boom(*a, **k):
        raise RuntimeError("bad frame")
    monkeypatch.setattr(pc, "pose_comparison_for_report", boom)
    run_analysis._add_pose_comparison(report, [], {}, "nothing.mov")
    assert report["pose_comparison"]["available"] is False and "bad frame" in report["pose_comparison"]["error"]
    assert {k: v for k, v in report.items() if k != "pose_comparison"} == before


def test_config_default_is_off():
    import config
    assert 'POSE_COMPARE = os.getenv("POSE_COMPARE", "off")' in (ROOT / "config.py").read_text()
    assert "off" in config.POSE_COMPARE_MODES and "on" in config.POSE_COMPARE_MODES
