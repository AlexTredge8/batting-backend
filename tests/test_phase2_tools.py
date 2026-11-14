"""Phase 2 tooling: blind pack (sheets only), pairwise fit, agreement report, on synthetic coaches."""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import phase2_agreement  # noqa: E402
import phase2_blind_pack  # noqa: E402
from phase2_pairs import fit_bradley_terry, schedule_pairs  # noqa: E402


def test_schedule_pairs_balanced_and_unique():
    ids = [f"C{i:02d}" for i in range(18)]
    pairs = schedule_pairs(ids, 60, random.Random(1))
    assert len(pairs) == 60
    assert len({frozenset(p) for p in pairs}) == 60
    counts = {i: 0 for i in ids}
    for a, b in pairs:
        counts[a] += 1
        counts[b] += 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_bradley_terry_recovers_order():
    rng = np.random.default_rng(0)
    ids = [f"C{i:02d}" for i in range(12)]
    true = np.linspace(-2, 2, 12)
    judgements = []
    for a, b in schedule_pairs(ids, 66, random.Random(0)) * 4:
        ia, ib = ids.index(a), ids.index(b)
        p = 1 / (1 + np.exp(true[ib] - true[ia]))
        judgements.append((a, b) if rng.random() < p else (b, a))
    theta, se, ssr = fit_bradley_terry(ids, judgements)
    rho = np.corrcoef(np.argsort(np.argsort(theta)), np.argsort(np.argsort(true)))[0, 1]
    assert rho > 0.9
    assert 0 < ssr <= 1
    assert np.all(np.isfinite(se))
    # a clip in no pair gets no se and does not wreck the reliability figure
    theta2, se2, ssr2 = fit_bradley_terry(ids + ["UNSEEN"], judgements)
    assert np.isnan(se2[-1]) and 0 < ssr2 <= 1


def test_icc_perfect_and_noisy():
    truth = np.linspace(20, 95, 15)
    perfect = np.column_stack([truth, truth, truth])
    assert abs(phase2_agreement.icc_2_1(perfect) - 1) < 1e-9
    rng = np.random.default_rng(0)
    noisy = perfect + rng.normal(0, 8, perfect.shape)
    assert 0.6 < phase2_agreement.icc_2_1(noisy) < 1


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_blind_pack_excludes_self_and_report_runs(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    names = [f"Clip{i}.mov" for i in range(10)]
    for n in names:
        (videos / n).write_bytes(b"x")
    scorer = {0: "Cam", 1: "Viv", 2: "Alex"}
    _write(tmp_path / "clip_sheet.csv",
           [{"filename": n, "scorer_in_clip": scorer.get(i, "")} for i, n in enumerate(names)],
           ["filename", "scorer_in_clip"])
    pack = tmp_path / "pack"
    phase2_blind_pack.main(["--video-dir", str(videos), "--out", str(pack), "--clip-sheet",
                            str(tmp_path / "clip_sheet.csv"), "--pairs-per-coach", "20", "--no-copy"])
    key = {r["clip_id"]: r for r in csv.DictReader((pack / "blind_key_PRIVATE.csv").open())}
    assert len(key) == 10
    assert all(cid.startswith("BQ-") for cid in key)
    for coach in ("Alex", "Cam", "Viv"):
        rows = list(csv.DictReader((pack / "sheets" / f"{coach}_score_sheet.csv").open()))
        assert len(rows) == 9
        assert all(key[r["clip_id"]]["scorer_in_clip"] != coach for r in rows)
        pair_ids = {i for r in csv.DictReader((pack / "pairs" / f"{coach}_pairs.csv").open())
                    for i in (r["left_id"], r["right_id"])}
        assert all(key[i]["scorer_in_clip"] != coach for i in pair_ids)

    # Fill the sheets with synthetic scores: truth + coach noise; a model with more noise.
    rng = np.random.default_rng(3)
    truth = {cid: 20 + 7 * i for i, cid in enumerate(sorted(key))}
    for coach, sd in (("Alex", 3), ("Cam", 4), ("Viv", 4), ("Model", 9)):
        path = pack / "sheets" / f"{coach}_score_sheet.csv"
        if coach == "Model":
            base = list(csv.DictReader((pack / "sheets" / "Alex_score_sheet.csv").open()))
            fields = list(base[0].keys())
        else:
            base = list(csv.DictReader(path.open()))
            fields = list(base[0].keys())
        filled = []
        for r in base:
            overall = float(np.clip(truth[r["clip_id"]] + rng.normal(0, sd), 0, 100))
            r.update({"overall": round(overall), "minutes_taken": 4, "priority_fault": "F05" if overall < 60 else "F10"})
            for p in phase2_agreement.PILLARS:
                r[p] = round(overall / 4 + rng.normal(0, 1))
            filled.append(r)
        _write(path, filled, fields)
    out = tmp_path / "report"
    assert phase2_agreement.main(["--sheets-dir", str(pack / "sheets"), "--key", str(pack / "blind_key_PRIVATE.csv"),
                                  "--out", str(out), "--original", str(tmp_path / "missing.csv")]) == 0
    report = (out / "agreement_report.md").read_text()
    assert "Coach-to-coach error" in report and "Model as a fourth coach" in report
    import json
    res = json.loads((out / "agreement_results.json").read_text())
    assert res["humans"] == ["Alex", "Cam", "Viv"] and res["models"] == ["Model"]
    assert 0 < res["ceiling_mae"]["overall"] < 12
