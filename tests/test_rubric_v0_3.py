"""Rubric v0.3: structure, tips, the checkpoint file's checkpoint list, code decoding and ratings reading."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import phase2_offline_pairs  # noqa: E402
import phase5_checkpoint_codes  # noqa: E402
from phase5_entry_tests import load_ratings  # noqa: E402

RUBRIC = json.loads((ROOT / "docs" / "phase1" / "rubric_v0_3.json").read_text())
V02 = json.loads((ROOT / "docs" / "phase1" / "rubric_v0_2.json").read_text())
TIPS = json.loads((ROOT / "docs" / "phase6" / "fault_tips_v0_3.json").read_text())
RATED = [c for c in RUBRIC["checkpoints"] if c.get("rated", True)]


def test_ids_c01_to_c16_kept_and_new_ids_follow():
    ids = [c["id"] for c in RUBRIC["checkpoints"]]
    assert len(ids) == len(set(ids))
    assert {f"C{i:02d}" for i in range(1, 17)} <= set(ids)
    assert all(re.fullmatch(r"C\d{2}", i) for i in ids)
    v02 = {c["id"]: c for c in V02["checkpoints"]}
    for c in RUBRIC["checkpoints"]:  # an unchanged checkpoint keeps its v0.2 text, stage and pillar
        if c["status"] == "agreed in v0.1":
            assert {k: c[k] for k in ("text", "stage", "pillar")} == {k: v02[c["id"]][k] for k in ("text", "stage", "pillar")}


def test_rated_checkpoints_at_most_18_in_chain_order_with_source_and_status():
    assert len(RATED) <= 18 and len(RATED) == RUBRIC["rated_per_clip"]
    stages = RUBRIC["chain_stages"]
    pos = [stages.index(c["stage"]) for c in RATED]
    assert pos == sorted(pos)
    for c in RUBRIC["checkpoints"]:
        assert c["source"] in {"Cam", "Radford", "consensus", "research"}
        assert c["status"] in {"agreed in v0.1", "proposed"}
        assert c["pillar"] in {"access", "tracking", "stability", "flow"}
    for c in RUBRIC["checkpoints"]:
        if not c.get("rated", True):
            assert c.get("merged_into") in {r["id"] for r in RATED}


def test_evidence_changes_applied():
    by = {c["id"]: c for c in RUBRIC["checkpoints"]}
    assert "slips is fine" in by["C03"]["text"] and "do not mark it down" in by["C03"]["text"]
    assert by["C07"]["text"].startswith("Front shoulder stays aligned to the shot until contact")
    assert "stride" in by["C07"]["text"]
    assert "settle of the front foot" in by["C09"]["text"]
    assert "just inside" in by["C05"]["text"]


def test_faults_follow_rated_checkpoints_and_every_fault_has_one_sentence_tip():
    faults = RUBRIC["faults_in_chain_order"]
    assert [f["checkpoint"] for f in faults] == [c["id"] for c in RATED]
    assert set(TIPS["tips"]) == {f["id"] for f in faults} | {"F00"}
    for fid, tip in TIPS["tips"].items():
        assert tip.endswith(".") and tip.count(". ") == 0 and len(tip) < 200, fid
    assert RUBRIC["score_sheet_fields"][1:1 + len(RATED)] == [c["id"] for c in RATED]


def test_checkpoint_file_lists_rated_checkpoints_and_defaults_to_v0_3():
    cps = phase2_offline_pairs.rated_checkpoints(RUBRIC)
    assert [c[0] for c in cps] == [c["id"] for c in RATED]
    assert "C14" not in [c[0] for c in cps]
    assert len(phase2_offline_pairs.rated_checkpoints(V02)) == 16  # older rubrics: every checkpoint
    src = (ROOT / "phase2_offline_pairs.py").read_text()
    assert '"rubric_v0_3.json"' in src


def _code(build: str, items: str, n: int) -> str:
    body = ".".join(["CIQ2", build, "Qp", "Cam", np.base_repr(n, 36).lower(), items])
    return body + "." + np.base_repr(sum(map(ord, body)) % 36, 36).lower()


def test_decoder_uses_the_key_checkpoint_order(tmp_path):
    order = [c["id"] for c in RATED]
    ids = ["D01", "D02"]
    r1 = "".join("210x"[i % 4] for i in range(len(order)))
    r2 = "".join("x012"[i % 4] for i in range(len(order)))
    code = _code("ABC123", "0" + r1 + "1" + r2, len(order))
    d = phase5_checkpoint_codes.parse_checkpoint(code, ids, "ABC123", order)
    assert d["answers"]["D01"]["C17"] == r1[order.index("C17")]
    assert d["answers"]["D02"]["C12"] == ("X" if r2[-1] == "x" else r2[-1])
    assert set(d["answers"]["D01"]) == set(order)
    # an older key (no checkpoint list) reads positions as C01..Cn
    old = phase5_checkpoint_codes.parse_checkpoint(code, ids, "ABC123")
    assert set(old["answers"]["D01"]) == {f"C{i:02d}" for i in range(1, len(order) + 1)}

    key = tmp_path / "key.csv"
    with key.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["clip_id", "filename", "build_id", "rubric", "checkpoints"])
        w.writeheader()
        for cid, fn in zip(ids, ["a.mp4", "b.mov"]):
            w.writerow({"clip_id": cid, "filename": fn, "build_id": "ABC123", "rubric": "0.3-draft", "checkpoints": " ".join(order)})
    codes = tmp_path / "codes.txt"
    codes.write_text("Cam: " + code + "\n")
    out = tmp_path / "ratings.csv"
    assert phase5_checkpoint_codes.main(["--codes", str(codes), "--key", str(key), "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert list(rows[0])[4:] == order
    assert rows[0]["filename"] == "a.mp4" and rows[0]["C17"] == r1[order.index("C17")]


def test_entry_tests_read_c01_to_c12_from_a_wider_ratings_file(tmp_path):
    order = [c["id"] for c in RATED]
    path = tmp_path / "ratings.csv"
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["rater", "role", "level", "filename", *order, "overall", "priority_fault"])
        w.writeheader()
        for rater, shift in (("K01", 0), ("K02", 1)):
            row = {"rater": rater, "role": "qualified_coach", "level": "p", "filename": "Clip One.mp4", "overall": "70", "priority_fault": "F05"}
            row.update({cp: str((int(cp[1:]) + shift) % 3) for cp in order})
            row["C08"] = "X"
            w.writerow(row)
    ratings, kinds = load_ratings([f"{path}:coach"])
    got = ratings["clip one"]
    assert kinds == ["coach"]
    for i in range(1, 13):
        cp = f"C{i:02d}"
        if cp == "C08":
            assert cp not in got  # X is left out
        else:
            assert got[cp] == np.mean([i % 3, (i + 1) % 3]), cp
    assert "overall" not in got and "priority_fault" not in got
