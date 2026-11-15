"""Phase 6 offline prototype (phase6_model.py): pairs, orientation, grouping, thresholds, F00."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import phase6_model as p6  # noqa: E402

META = {
    "MW": {"direction": "higher_worse", "checkpoint": "C01", "fault": "F01", "pillar": "tracking", "unit": ""},
    "MB": {"direction": "higher_better", "checkpoint": "C05", "fault": "F05", "pillar": "access", "unit": ""},
}


# ----------------------------------------------------------------------------- pairs

def test_pairs_from_scores_skips_ties_and_puts_the_higher_score_first():
    pairs = p6.pairs_from_scores({"a": 90, "b": 50, "c": 50, "d": float("nan")}, "alex_scores")
    got = {(p["winner"], p["loser"]) for p in pairs}
    assert got == {("a", "b"), ("a", "c")}  # b v c tie skipped, d has no score
    assert all(p["source"] == "alex_scores" for p in pairs)


def test_pairs_from_swipes_reads_left_right_and_normalises_names():
    rows = [{"left": "Drive_Average 2.MOV", "right": "Viv P_Offdrive Elite.mov", "better": "R"},
            {"left": "A.mp4", "right": "B.mp4", "better": "l"},
            {"left": "A.mp4", "right": "B.mp4", "better": ""}]
    pairs = p6.pairs_from_swipes(rows, "cam_swipes")
    assert [(p["winner"], p["loser"]) for p in pairs] == [("viv p_offdrive elite", "drive_average 2"), ("a", "b")]


def test_clip_key_matches_measurement_stems_and_filenames():
    assert p6.clip_key("Drive_Average 2.MOV") == p6.clip_key("Drive_Average 2") == "drive_average 2"
    assert p6.clip_key("Cam T_LH Offdrive Elite.mov") == "cam t_lh offdrive elite"


# ----------------------------------------------------------------------------- orientation

def test_orient_flips_only_higher_worse():
    assert p6.orient(3.0, "higher_worse") == -3.0
    assert p6.orient(3.0, "higher_better") == 3.0
    with pytest.raises(ValueError):
        p6.orient(1.0, "sideways")


def test_standardise_makes_higher_better_and_uses_training_scale():
    values = {"x": {"MW": 1.0, "MB": 10.0}, "y": {"MW": 3.0, "MB": 20.0},
              "z": {"MW": 5.0, "MB": 30.0}, "new": {"MW": float("nan"), "MB": 1000.0}}
    scaler = p6.fit_scaler(values, ["x", "y", "z"], ["MW", "MB"], META)
    z = p6.standardise(values, list(values), ["MW", "MB"], META, scaler)
    # MW is higher_worse: the clip with the smallest raw value is the best
    assert z["x"][0] > z["y"][0] > z["z"][0]
    assert z["x"][1] < z["y"][1] < z["z"][1]
    assert z["y"][0] == pytest.approx(0.0) and z["y"][1] == pytest.approx(0.0)  # median of training clips
    assert np.isnan(z["new"][0])
    assert z["new"][1] == p6.Z_CLIP  # clipped


def test_pair_design_is_symmetric_and_missing_gives_zero_difference():
    z = {"a": np.array([1.0, np.nan]), "b": np.array([0.0, 2.0])}
    X, y = p6.pair_design([{"winner": "a", "loser": "b", "source": "s"}], z)
    assert X.tolist() == [[1.0, 0.0], [-1.0, 0.0]]
    assert y.tolist() == [1.0, 0.0]
    assert p6.strength(np.array([2.0, np.nan]), np.array([0.5, 9.0])) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- model

def _synthetic(n=30, seed=0):
    rng = np.random.default_rng(seed)
    z = {f"c{i}": rng.normal(size=2) for i in range(n)}
    true_w = np.array([1.5, -0.7])
    s = {c: float(v @ true_w) for c, v in z.items()}
    pairs = p6.pairs_from_scores(s, "syn")
    return z, pairs


def test_fit_pairwise_recovers_signs_and_respects_the_constraint():
    z, pairs = _synthetic()
    X, y = p6.pair_design(pairs, z)
    w_free = p6.fit_pairwise(X, y, lam=0.01, nonneg=False, use_sklearn=False)
    assert w_free[0] > 0 > w_free[1]
    w_pos = p6.fit_pairwise(X, y, lam=0.01, nonneg=True)
    assert w_pos[0] > 0 and w_pos[1] == 0.0


def test_numpy_solver_matches_sklearn_when_installed():
    pytest.importorskip("sklearn")
    z, pairs = _synthetic(seed=3)
    X, y = p6.pair_design(pairs, z)
    w_np = p6.fit_pairwise(X, y, lam=0.1, nonneg=False, use_sklearn=False)
    w_sk = p6.fit_pairwise(X, y, lam=0.1, nonneg=False, use_sklearn=True)
    assert np.allclose(w_np, w_sk, atol=1e-4)


# ----------------------------------------------------------------------------- grouping

def test_lopo_never_trains_or_scales_on_the_held_out_player():
    player_of = {"a1": "P1", "a2": "P1", "b": "P2", "c": "P3", "d": "P4"}
    clips = sorted(player_of)
    folds = p6.lopo_folds(clips, player_of)
    assert len(folds) == 4
    for player, test, train in folds:
        assert {player_of[c] for c in test} == {player}
        assert player not in {player_of[c] for c in train}
        assert set(test) | set(train) == set(clips)
    fold_p1 = [f for f in folds if f[0] == "P1"][0]
    assert sorted(fold_p1[1]) == ["a1", "a2"]  # both clips of the player leave together

    scores = {"a1": 90, "a2": 80, "b": 70, "c": 60, "d": 50}
    pairs = p6.pairs_from_scores(scores, "s")
    for player, test, train in folds:
        tr = p6.training_pairs(pairs, train)
        p6.assert_no_leak(tr, test, train, player_of)  # does not raise
        assert all(player_of[p["winner"]] != player and player_of[p["loser"]] != player for p in tr)
    with pytest.raises(AssertionError):
        p6.assert_no_leak(pairs, ["a1"], ["b"], player_of)
    with pytest.raises(AssertionError):
        p6.assert_no_leak([], ["a1"], ["a2", "b"], player_of)  # same player in the scaling statistics


def test_run_lopo_strengths_come_from_models_that_never_saw_the_player():
    rng = np.random.default_rng(1)
    clips = [f"k{i}" for i in range(8)]
    player_of = {c: f"P{i // 2}" for i, c in enumerate(clips)}
    values = {c: {"MW": float(rng.normal()), "MB": float(rng.normal())} for c in clips}
    s = {c: values[c]["MB"] - values[c]["MW"] for c in clips}
    res = p6.run_lopo(values, META, ["MW", "MB"], clips, player_of, p6.pairs_from_scores(s, "s"))
    assert set(res["oof"]) == set(clips)
    assert set(res["weights"]) == {f"P{i}" for i in range(4)}
    for r in res["records"]:  # every held-out record involves the fold's player
        assert r["fold"] in (player_of[r["winner"]], player_of[r["loser"]])


# ----------------------------------------------------------------------------- diagnosis

CHAIN = [{"id": "F01", "checkpoint": "C01"}, {"id": "F13", "checkpoint": "C13"}, {"id": "F05", "checkpoint": "C05"}]
ENTRY = [
    {"measurement": "MW", "checkpoint": "C01", "pillar": "tracking", "test3_auto_anchors": "pass",
     "test0_coverage": "pass", "rho_auto": "0.9", "median_abs_auto_diff": "0.1", "iqr": "1"},
    {"measurement": "MB", "checkpoint": "C05", "pillar": "access", "test3_auto_anchors": "pass",
     "test0_coverage": "pass", "rho_auto": "0.95", "median_abs_auto_diff": "0.1", "iqr": "1"},
    {"measurement": "MX", "checkpoint": "C13", "pillar": "flow", "test3_auto_anchors": "fail",
     "test0_coverage": "pass", "rho_auto": "0.3", "median_abs_auto_diff": "0.1", "iqr": "1"},
]


def test_diagnosis_checkpoints_keep_chain_order_and_drop_failed_test3():
    cps = p6.diagnosis_checkpoints(CHAIN, ENTRY)
    assert [c["fault"] for c in cps] == ["F01", "F05"]
    assert [c["measurement"] for c in cps] == ["MW", "MB"]


def test_thresholds_fail_only_when_worse_than_75_percent_of_reference():
    ref = {f"r{i}": {"MW": float(i), "MB": float(i)} for i in range(1, 6)}  # 1..5
    cps = p6.diagnosis_checkpoints(CHAIN, ENTRY)
    thr = p6.fit_thresholds(ref, list(ref), cps, META, q=0.75)
    # MW higher_worse: raw 75th percentile of 1..5 is 4 -> oriented threshold -4
    assert thr["MW"] == pytest.approx(-4.0)
    # MB higher_better: raw 25th percentile is 2
    assert thr["MB"] == pytest.approx(2.0)
    assert p6.diagnose({"MW": 4.5, "MB": 3.0}, thr, cps, META)["failed"] == ["F01"]
    assert p6.diagnose({"MW": 3.9, "MB": 1.5}, thr, cps, META)["failed"] == ["F05"]
    assert p6.diagnose({"MW": 4.0, "MB": 2.0}, thr, cps, META)["failed"] == []  # at the threshold is not worse


def test_priority_is_earliest_failed_in_chain_and_f00_when_nothing_fails():
    cps = p6.diagnosis_checkpoints(CHAIN, ENTRY)
    thr = {"MW": -4.0, "MB": 2.0}
    both = p6.diagnose({"MW": 9.0, "MB": 0.0}, thr, cps, META)
    assert both["priority"] == "F01" and both["failed"] == ["F01", "F05"]
    clean = p6.diagnose({"MW": 1.0, "MB": 5.0}, thr, cps, META)
    assert clean["priority"] == p6.F00 and clean["failed"] == []
    missing = p6.diagnose({"MW": float("nan"), "MB": 5.0}, thr, cps, META)
    assert missing["priority"] == p6.F00 and missing["not_measured"] == ["F01"]


# ----------------------------------------------------------------------------- helpers

def test_spearman_handles_ties_and_constants():
    assert p6.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert p6.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(p6.spearman([1, 2, 3], [5, 5, 5]))
    assert p6.rankdata(np.array([3.0, 1.0, 3.0])).tolist() == [2.5, 1.0, 2.5]


def test_isotonic_fit_is_monotone_and_pools_violators():
    ux, fy = p6.isotonic_fit([1, 2, 3, 4], [10, 30, 20, 40])
    assert fy.tolist() == [10, 25, 25, 40]
    assert p6.isotonic_predict((ux, fy), 0) == 10 and p6.isotonic_predict((ux, fy), 9) == 40


def test_calibration_target_rescales_to_alex():
    alex = {"a": 20.0, "b": 50.0, "c": 80.0}
    cam = {"a": 0.0, "b": 50.0, "c": 100.0}
    t = p6.calibration_target(alex, cam, ["a", "b", "c"])
    assert t["b"] == pytest.approx(50.0)
    assert t["a"] == pytest.approx(20.0) and t["c"] == pytest.approx(80.0)


def test_cluster_bootstrap_weights_pairs_by_player_counts():
    player_of = {"a": "P1", "b": "P2"}
    recs = [{"winner": "a", "loser": "b", "m": 1.0}]
    est, lo, hi, n = p6.boot_accuracy(recs, "m", player_of, p6.player_resamples(["P1", "P2"], 50, 0))
    assert (est, n) == (1.0, 1)


# ----------------------------------------------------------------------------- the real inputs

def test_feature_rule_on_phase5_v3_entry_tests():
    entry = p6.read_csv(ROOT / "baselines/phase5_2026-11-15_v3/entry_tests.csv")
    feats = p6.select_features(entry, 4)
    assert feats == ["M05", "M01", "M11", "M12"]  # best automatic-key-frame rho per pillar
    six = p6.select_features(entry, 6)
    assert len(six) == 6 and "M09" not in six and "M13" not in six  # M09 fails test 3, M13 coverage


def test_real_inputs_line_up():
    d = p6.load_inputs()
    clips = [c for c in d["values"] if c in d["player"] and c in d["alex"]]
    assert len(clips) == 17
    pairs = p6.all_pairs(d)
    assert {p["source"] for p in pairs} == set(p6.SOURCES)
    assert all(p["winner"] in d["alex"] and p["loser"] in d["alex"] for p in pairs)
