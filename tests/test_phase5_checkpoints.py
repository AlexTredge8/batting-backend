"""Phase 5 checkpoint measurements and entry tests, on synthetic landmarks (no video needed)."""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import phase5_checkpoints as p5  # noqa: E402
import phase5_entry_tests as p5e  # noqa: E402

W, H, FPS, N = 480.0, 848.0, 30.0, 120
ANCH = {"setup_frame": 40, "hands_start_up_frame": 50, "front_foot_down_frame": 70, "hands_peak_frame": 66,
        "contact_frame": 78, "follow_through_frame": 100}


def _figure(head_jitter=0.0, head_lean=0.0, seed=0, peak=66, head_setup_off=0.0, trigger=0.0, trigger_at=45,
            com_lag=0.0) -> np.ndarray:
    """A right-hander, side-on, filmed from the bowler's end. x in pixels, off side = smaller x.
    peak: frame of the top of the backlift; head_setup_off: nose moved this many px to the off side
    throughout; trigger: back ankle steps this many px across and back over 8 frames from trigger_at;
    com_lag: hips and shoulders held this many px on the leg side of the front foot from foot-down."""
    rng = np.random.default_rng(seed)
    a = np.zeros((N, 33, 4))
    a[..., 3] = 1.0
    pts = {0: (250, 200), 11: (245, 260), 12: (255, 262), 13: (240, 330), 14: (262, 330), 15: (260, 380), 16: (262, 382),
           23: (248, 420), 24: (256, 420), 25: (246, 520), 26: (258, 520), 27: (245, 620), 28: (262, 620),
           29: (255, 630), 30: (270, 630), 31: (225, 632), 32: (240, 632), 7: (245, 195), 8: (256, 195)}
    for i, (x, y) in pts.items():
        a[:, i, 0], a[:, i, 1] = x, y
    # stride: front foot (left ankle 27) moves 40 px to the off side between hands-start-up and foot-down
    for f in range(N):
        t = np.clip((f - 50) / 20, 0, 1)
        a[f, [27, 29, 31], 0] -= 40 * t
        a[f, [27, 29, 31], 1] -= 30 * (1 - abs(2 * t - 1))  # the front foot lifts and lands at frame 70
        a[f, [11, 12, 23, 24], 0] += com_lag * t
        a[f, 0, 0] -= head_lean * t + head_setup_off
        # backlift: top hand rises 80 px to the peak, then comes down to contact (frame 78)
        up = np.clip((f - 50) / (peak - 50), 0, 1) - np.clip((f - peak) / (78 - peak), 0, 1)
        a[f, 15, 1] -= 80 * up
        a[f, [28, 30, 32], 0] -= trigger * (1 - abs(np.clip((f - trigger_at) / 4, 0, 2) - 1))
    a[:, 0, :2] += rng.normal(0, head_jitter, (N, 2))
    a[..., 0] /= W
    a[..., 1] /= H
    return a


def _save(tmp: Path, name: str, arr: np.ndarray) -> Path:
    p = tmp / f"{name}.npz"
    np.savez_compressed(p, landmarks=arr.astype(np.float32), frame_idx=np.arange(N), fps=FPS, width=W, height=H)
    return p


def _values(npz: Path, hand="right", drive="off") -> dict[str, float]:
    return {r["measurement"]: (np.nan if r["value"] == "" else r["value"])
            for r in p5.measure_clip(p5.load_clip(npz, hand, drive), ANCH)}


def test_all_measurements_compute_and_feet_check(tmp_path):
    clip = p5.load_clip(_save(tmp_path, "rh", _figure()), "right")
    rows = p5.measure_clip(clip, ANCH)
    assert [r["measurement"] for r in rows] == [f"M{i:02d}" for i in range(1, 18)]
    assert all(r["value"] != "" for r in rows), [r for r in rows if r["value"] == ""]
    assert rows[0]["toes_point_off"] == "yes"
    assert 410 < rows[0]["body_length_px"] < 430  # ankles at y=620, nose at y=200


def test_left_hander_mirror_matches_right_hander(tmp_path):
    rh = _figure()
    lh = rh.copy()
    lh[..., 0] = 1.0 - lh[..., 0]
    swap = [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32), (7, 8)]
    for i, j in swap:
        lh[:, [i, j]] = lh[:, [j, i]]
    a, b = _values(_save(tmp_path, "rh", rh), "right"), _values(_save(tmp_path, "lh", lh), "left")
    for m in a:
        assert np.isclose(float(a[m]), float(b[m]), atol=1e-6), m


def test_front_foot_step_to_off_side_is_positive_and_on_drive_flips(tmp_path):
    npz = _save(tmp_path, "rh", _figure())
    off, on = _values(npz, drive="off"), _values(npz, drive="on")
    assert float(off["M05"]) > 5
    assert np.isclose(float(on["M05"]), -float(off["M05"]))


def test_head_measurements_move_the_right_way(tmp_path):
    still, shaky = _values(_save(tmp_path, "a", _figure(0.0))), _values(_save(tmp_path, "b", _figure(4.0, seed=1)))
    assert float(shaky["M01"]) > float(still["M01"])
    lead = _values(_save(tmp_path, "c", _figure(head_lean=30)))
    assert float(lead["M04"]) > float(still["M04"])  # head moved across towards the line
    assert float(lead["M10"]) > float(still["M10"])


def test_times_scale_with_fps(tmp_path):
    clip = p5.load_clip(_save(tmp_path, "rh", _figure()), "right")
    assert clip.ms(300) == 9
    clip.fps = 60.0
    assert clip.ms(300) == 18


def test_entry_tests_pass_a_good_measurement_and_fail_a_noisy_one(tmp_path):
    rng = np.random.default_rng(0)
    clips = [f"clip{i:02d}.mov" for i in range(14)]
    rating = rng.integers(0, 3, len(clips)).astype(float)
    good = 10 - 3 * rating + rng.normal(0, 0.3, len(clips))   # higher_worse, tracks the rating
    noise = rng.normal(0, 1, len(clips))
    rows = []
    for k, c in enumerate(clips):
        for m, cp, v in (("M01", "C01", good[k]), ("M02", "C02", noise[k])):
            for s, jitter in (("validated", 0.0), ("auto", 0.05)):
                rows.append({"filename": c, "anchor_set": s, "measurement": m, "checkpoint": cp, "fault": "F" + cp[1:],
                             "pillar": "tracking", "unit": "% BL", "direction": "higher_worse", "value": v + jitter})
    meas = tmp_path / "m.csv"
    with meas.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rat = tmp_path / "r.csv"
    with rat.open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["filename", "C01", "C02"])
        for k, c in enumerate(clips):
            w.writerow([c, int(rating[k]), int(rating[k])])
    gt = tmp_path / "gt.csv"
    gt.write_text("filename,tracking_score\n" + "".join(f"{c},{20}\n" for c in clips))
    out = tmp_path / "out"
    p5e.main(["--measurements", str(meas), "--ratings", f"{rat}:coach", "--ground-truth", str(gt), "--out", str(out)])
    res = {r["measurement"]: r for r in csv.DictReader((out / "entry_tests.csv").open())}
    assert res["M01"]["test1_tracks"] == "pass" and res["M01"]["test3_auto_anchors"] == "pass"
    assert res["M01"]["verdict"].startswith("pass so far")
    assert res["M02"]["test1_tracks"] == "fail" and res["M02"]["verdict"] == "fail"


def test_uniform_rater_leaves_test_one_undecided(tmp_path):
    clips = [f"clip{i:02d}.mov" for i in range(14)]
    rows = [{"filename": c, "anchor_set": s, "measurement": "M01", "checkpoint": "C01", "fault": "F01", "pillar": "tracking",
             "unit": "% BL", "direction": "higher_worse", "value": float(i)} for i, c in enumerate(clips) for s in ("validated", "auto")]
    meas = tmp_path / "m.csv"
    with meas.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rat = tmp_path / "r.csv"
    rat.write_text("filename,C01\n" + "".join(f"{c},{1 if i == 0 else 2}\n" for i, c in enumerate(clips)))
    gt = tmp_path / "gt.csv"
    gt.write_text("filename,tracking_score\n")
    out = tmp_path / "out"
    p5e.main(["--measurements", str(meas), "--ratings", f"{rat}:model", "--ground-truth", str(gt), "--out", str(out)])
    res = next(csv.DictReader((out / "entry_tests.csv").open()))
    assert res["test1_tracks"].startswith("not decided")
    assert res["verdict"] == "robust; awaiting coach ratings"


def test_pause_is_interpolated_between_frames():
    sp = np.array([10, 10, 1, 1, 10, 10], float)  # under 2 between frames 1.89 and 3.11
    assert np.isclose(p5._longest_below(sp, 2.0, 30.0), 1000 * (3 + 1 / 9 - (1 + 8 / 9)) / 30)
    assert p5._longest_below(np.array([5, 5, 5.0]), 2.0, 30.0) == 0.0
    assert p5._longest_below(np.array([1, 1, 10, 10.0]), 2.0, 30.0) == 0.0  # set-up stillness, not a pause


def test_mirrored_video_labelled_right_handed_is_flipped_by_the_feet(tmp_path):
    rh = _figure()
    mirrored = rh.copy()
    mirrored[..., 0] = 1.0 - mirrored[..., 0]  # a selfie-mirrored right-hander looks left-handed
    for i, j in [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32), (7, 8)]:
        mirrored[:, [i, j]] = mirrored[:, [j, i]]
    cache = tmp_path / "cache"
    cache.mkdir()
    _save(cache, "plain", rh)
    _save(cache, "mirrored", mirrored)
    anchors = tmp_path / "a.csv"
    anchors.write_text("filename," + ",".join(p5.ANCHORS) + "\n" + "".join(
        f"{n}.mov," + ",".join(str(ANCH[k]) for k in p5.ANCHORS) + "\n" for n in ("plain", "mirrored")))
    sheet = tmp_path / "s.csv"
    sheet.write_text("filename,handedness,drive_type\nplain.mov,right,off\nmirrored.mov,right,off\n")
    out = tmp_path / "out"
    p5.main(["--cache-dir", str(cache), "--anchors", str(anchors), "--auto-table", str(tmp_path / "none.csv"),
             "--clip-sheet", str(sheet), "--out", str(out)])
    rows = list(csv.DictReader((out / "checkpoint_measurements.csv").open()))
    geo = {r["filename"]: (r["geometry_handedness"], r["toes_point_off"]) for r in rows}
    assert geo["plain"] == ("right", "yes") and geo["mirrored"] == ("left", "yes")
    val = {(r["filename"], r["measurement"]): r["value"] for r in rows}
    for m in [f"M{i:02d}" for i in range(1, 18)]:
        assert np.isclose(float(val[("plain", m)]), float(val[("mirrored", m)]), atol=1e-6), m


def test_time_below_is_smooth_and_interpolated():
    sp = np.array([10, 10, 1, 1, 10, 10], float)
    assert np.isclose(p5._time_below(sp, 2.0, 30.0), 1000 * (1 / 9 + 1 + 1 / 9) / 30)
    assert p5._time_below(np.array([5, 5, 5.0]), 2.0, 30.0) == 0.0


def test_hands_are_the_more_visible_wrist(tmp_path):
    arr = _figure()
    arr[:, 15, 3] = 0.1  # top wrist hidden: the other wrist becomes "the hands"
    assert p5.load_clip(_save(tmp_path, "h", arr), "right").hand == 16
    assert p5.load_clip(_save(tmp_path, "v", _figure()), "right").hand in (15, 16)


# --- v5: M14-M17 ---------------------------------------------------------------------

def test_m14_foot_landing_downswing_gap(tmp_path):
    clip = p5.load_clip(_save(tmp_path, "rh", _figure()), "right")
    land, why = p5._front_foot_lands(clip, ANCH["contact_frame"])
    assert why == "" and 68 <= land <= 72  # the foot lands at frame 70
    early = _values(_save(tmp_path, "early", _figure(peak=60)))  # downswing starts 10 frames before landing
    coupled = _values(_save(tmp_path, "coupled", _figure(peak=70)))  # downswing starts as the foot lands
    assert float(early["M14"]) > 200
    assert float(coupled["M14"]) < float(early["M14"]) and float(coupled["M14"]) < 80
    assert float(early["M14"]) >= 0 and float(coupled["M14"]) >= 0  # stored as the absolute gap


def test_m15_head_over_base_at_setup(tmp_path):
    centred = p5.load_clip(_save(tmp_path, "c", _figure()), "right")
    off = p5.load_clip(_save(tmp_path, "o", _figure(head_setup_off=30)), "right")
    leg = p5.load_clip(_save(tmp_path, "l", _figure(head_setup_off=-30)), "right")
    s0, s_off, s_leg = (p5._head_offset_setup(c, ANCH) for c in (centred, off, leg))
    assert s_off > s0 > s_leg  # off side positive
    v = {n: float(_values(_save(tmp_path, n, _figure(head_setup_off=d)))["M15"]) for n, d in (("c2", 0), ("o2", 30), ("l2", -30))}
    assert v["o2"] > v["c2"] or v["l2"] > v["c2"]
    assert min(v.values()) >= 0  # stored as the absolute offset


def test_m16_weight_over_front_foot(tmp_path):
    over, back = _values(_save(tmp_path, "a", _figure())), _values(_save(tmp_path, "b", _figure(com_lag=30)))
    assert float(back["M16"]) > float(over["M16"]) + 5


def test_m17_trigger(tmp_path):
    a = dict(ANCH, contact_frame=100)  # pre-stride window 70..86, after the synthetic stride has finished
    none = {r["measurement"]: r["value"] for r in p5.measure_clip(p5.load_clip(_save(tmp_path, "n", _figure()), "right"), a)}
    trig = {r["measurement"]: r["value"] for r in
            p5.measure_clip(p5.load_clip(_save(tmp_path, "t", _figure(trigger=15, trigger_at=72)), "right"), a)}
    assert float(none["M17"]) < 0.5
    assert float(trig["M17"]) > 2


def test_new_measurements_mirror_for_a_left_hander(tmp_path):
    kw = dict(peak=60, head_setup_off=20, trigger=15, trigger_at=72, com_lag=20)
    rh = _figure(**kw)
    lh = rh.copy()
    lh[..., 0] = 1.0 - lh[..., 0]
    for i, j in [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32), (7, 8)]:
        lh[:, [i, j]] = lh[:, [j, i]]
    for anchors in (ANCH, dict(ANCH, contact_frame=100)):
        a = {r["measurement"]: r["value"] for r in p5.measure_clip(p5.load_clip(_save(tmp_path, "rh", rh), "right"), anchors)}
        b = {r["measurement"]: r["value"] for r in p5.measure_clip(p5.load_clip(_save(tmp_path, "lh", lh), "left"), anchors)}
        for m in ("M14", "M15", "M16", "M17"):
            if a[m] == "":  # M14 with contact at 100: the foot has long landed, blank on both
                assert b[m] == "" and anchors["contact_frame"] == 100 and m == "M14", m
                continue
            assert np.isclose(float(a[m]), float(b[m]), atol=1e-6), m


def test_new_measurements_blank_with_a_reason_when_the_window_is_missing(tmp_path):
    clip = p5.load_clip(_save(tmp_path, "rh", _figure()), "right")
    early = {r["measurement"]: r for r in p5.measure_clip(clip, dict(ANCH, contact_frame=10))}  # no 1 s before contact
    for m in ("M14", "M15", "M17"):
        assert early[m]["value"] == "" and early[m]["blank_reason"], m
    late = {r["measurement"]: r for r in p5.measure_clip(clip, dict(ANCH, contact_frame=N + 30))}  # contact after the clip
    for m in ("M14", "M15", "M16", "M17"):
        assert late[m]["value"] == "" and late[m]["blank_reason"], m
    bl = p5.body_length(clip, ANCH)
    v, why = p5.m14_foot_downswing_coupled(p5.load_clip(_save(tmp_path, "still", _still_feet()), "right"), ANCH, bl)
    assert np.isnan(v) and why


def _still_feet() -> np.ndarray:
    arr = _figure()
    arr[:, [27, 28, 29, 30, 31, 32], 1] = arr[0, [27, 28, 29, 30, 31, 32], 1]  # the feet never lift
    return arr
