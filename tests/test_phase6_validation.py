"""The "Check the AI's tips" file: suggestions (phase6_suggestions.py), the page (phase2_offline_pairs.py
--validate-out) and the answer codes (phase6_validation_codes.py)."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import phase2_offline_pairs  # noqa: E402
import phase6_suggestions as sug  # noqa: E402
import phase6_validation_codes as vc  # noqa: E402

RUBRIC = json.loads((ROOT / "docs" / "phase1" / "rubric_v0_3.json").read_text())
TIPS = json.loads((ROOT / "docs" / "phase6" / "fault_tips_v0_3.json").read_text())
NAMES = [r["filename"] for r in csv.DictReader((ROOT / "coach_ground_truth_from_screenshot.csv").open(encoding="utf-8-sig"))]


# ---------- suggestions ----------

def test_fault_ids_mapped_to_v0_3():
    valid, retired = sug.v03_faults()
    assert valid[-1] == "F00" and len(valid) == 19 and "F14" not in valid
    assert retired == {"F14": "F07"}
    assert sug.to_v03("F14", valid, retired) == "F07"
    assert sug.to_v03("f5", valid, retired) == "F05"
    assert sug.to_v03("none", valid, retired) == "" and sug.to_v03("", valid, retired) == ""
    assert sug.to_v03("F99", valid, retired) == ""


def test_sources_read_from_the_repo_baselines():
    src = sug.load_sources()
    assert {s: len(v) for s, v in src.items()} == {"phase6_measurements": 17, "gemini_checkpoints": 15, "gemini_absolute": 15, "stills_model": 14}
    assert src["phase6_measurements"]["oscar v_beginner"] == ("F00", "")
    assert src["phase6_measurements"]["cam t_lh offdrive elite"][0] == "F01"
    assert src["gemini_checkpoints"]["cam t_ondrive elite"][0] == "F07"
    assert src["gemini_absolute"]["oscar v_beginner"] == ("F09", "F07")
    assert src["stills_model"]["alext_ondrive good"] == ("F04", "F07")
    valid, _ = sug.v03_faults()
    assert all(m in valid and (not s or s in valid) for v in src.values() for m, s in v.values())


def test_random_source_per_clip_is_seeded_balanced_and_only_from_sources_with_an_answer():
    src = sug.load_sources()
    rows = sug.suggest(NAMES, src, seed=7)
    assert rows == sug.suggest(NAMES, src, seed=7)  # fixed seed: same file every time
    assert [r["source"] for r in rows] != [r["source"] for r in sug.suggest(NAMES, src, seed=8)]
    assert [r["filename"] for r in rows] == NAMES
    for r in rows:
        k = sug.key_of(r["filename"])
        assert r["source"] in sug.SOURCES and k in src[r["source"]]
        assert r["suggested_fault"] == src[r["source"]][k][0] == r[r["source"]]
    counts = [sum(r["source"] == s for r in rows) for s in sug.SOURCES]
    assert sum(counts) == 18 and max(counts) - min(counts) <= 2
    best = sug.suggest(NAMES, src, pick="best")
    assert {r["source"] for r in best} == {"phase6_measurements", "gemini_checkpoints"}  # Female_Average has no Phase 6 answer


def test_suggestions_cli_writes_the_csv(tmp_path):
    out = tmp_path / "s.csv"
    assert sug.main(["--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert list(rows[0])[:4] == ["filename", "suggested_fault", "source", "second_fault"]
    assert len(rows) == 18 and all(r["suggested_fault"] for r in rows)


# ---------- the page ----------

def test_validation_faults_chain_order_then_f00_with_tips():
    faults = phase2_offline_pairs.validation_faults(RUBRIC, TIPS)
    assert [f[0] for f in faults] == [f["id"] for f in RUBRIC["faults_in_chain_order"]] + ["F00"]
    assert faults[-1][2] == "Nothing major to fix" and all(f[3] for f in faults)
    assert {f[1] for f in faults[:-1]} == set(RUBRIC["chain_stages"])


def _fake_build(tmp_path, monkeypatch, suggestions, extra=()):
    monkeypatch.setattr(phase2_offline_pairs, "find_video", lambda d, n: Path(n))
    monkeypatch.setattr(phase2_offline_pairs, "encode_clip", lambda *a, **k: (b"not-a-real-mp4", {"frames": 1, "fps": 30.0, "size": "2x2"}))
    html = tmp_path / "BattingIQ_check_the_AI.html"
    args = ["--video-dir", str(tmp_path), "--validate-out", str(html), "--suggestions", str(suggestions), *extra]
    assert phase2_offline_pairs.main(args) == 0
    keys = sorted(tmp_path.glob("*_key_PRIVATE.csv"))  # one key, named after the first file asked for
    assert len(keys) == 1
    return html.read_text(), list(csv.DictReader(keys[0].open()))


def _write_suggestions(path, rows):
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_validate_out_builds_page_without_the_source(tmp_path, monkeypatch):
    rows = sug.suggest(NAMES, sug.load_sources())
    _write_suggestions(tmp_path / "s.csv", rows)
    html, key = _fake_build(tmp_path, monkeypatch, tmp_path / "s.csv")
    assert "default-src 'none'" in html and "__PAYLOAD__" not in html and "__SUGG__" not in html and "__FAULTS__" not in html
    for s in sug.SOURCES:  # the page never says which AI made a suggestion
        assert s not in html
    assert "cam_main_fault" not in html
    shown = json.loads(re.search(r"const SUGG = (\{.*?\});", html).group(1))
    by_name = {r["filename"]: r for r in rows}
    assert len(key) == 18 and {k["clip_id"] for k in key} == set(shown)
    for k in key:
        assert k["suggested_fault"] == shown[k["clip_id"]] == by_name[k["filename"]]["suggested_fault"]
        assert k["source"] == by_name[k["filename"]]["source"]
        assert k["phase6_measurements"] == by_name[k["filename"]]["phase6_measurements"]
    faults = json.loads(re.search(r"const FAULTS = (\[.*?\]);", html).group(1))
    assert faults[-1][:3] == ["F00", "", "Nothing major to fix"]


def test_validate_out_needs_suggestions_for_every_clip(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="needs --suggestions"):
        phase2_offline_pairs.main(["--video-dir", str(tmp_path), "--validate-out", str(tmp_path / "v.html")])
    rows = sug.suggest(NAMES, sug.load_sources())
    _write_suggestions(tmp_path / "short.csv", rows[:-1])
    with pytest.raises(SystemExit, match="No AI suggestion"):
        _fake_build(tmp_path, monkeypatch, tmp_path / "short.csv")
    rows[0]["suggested_fault"] = "F14"  # an old ID must be mapped before it reaches the page
    _write_suggestions(tmp_path / "old.csv", rows)
    with pytest.raises(SystemExit, match="not in the rubric"):
        _fake_build(tmp_path, monkeypatch, tmp_path / "old.csv")


def test_other_files_unchanged_when_validate_out_is_not_given(tmp_path, monkeypatch):
    monkeypatch.setattr(phase2_offline_pairs, "find_video", lambda d, n: Path(n))
    monkeypatch.setattr(phase2_offline_pairs, "encode_clip", lambda *a, **k: (b"x", {"frames": 1, "fps": 30.0, "size": "2x2"}))
    assert phase2_offline_pairs.main(["--video-dir", str(tmp_path), "--quick-out", str(tmp_path / "q.html")]) == 0
    key = list(csv.DictReader((tmp_path / "q_key_PRIVATE.csv").open()))
    assert list(key[0]) == ["clip_id", "filename", "build_id", "bytes", "frames", "fps", "size"]


def test_validate_and_checkpoint_files_share_clips_and_key(tmp_path, monkeypatch):
    rows = sug.suggest(NAMES, sug.load_sources())
    _write_suggestions(tmp_path / "s.csv", rows)
    html, key = _fake_build(tmp_path, monkeypatch, tmp_path / "s.csv", ["--checkpoint-out", str(tmp_path / "cp.html")])
    build = re.search(r'"build": "([0-9A-F]{6})"', html).group(1)
    assert build in (tmp_path / "cp.html").read_text() and key[0]["build_id"] == build
    assert key[0]["checkpoints"] and key[0]["source"]


# ---------- answer codes ----------

def _code(build: str, items: list[tuple[int, str, str, str]], role="Q", level="p", name="Alex") -> str:
    """The page's answerCode(): item = clip index + verdict + fault number base 36 (z) + confidence."""
    f36 = lambda f: np.base_repr(int(f[1:]), 36).lower() if f else "z"  # noqa: E731
    body = ".".join(["VIQ1", build, role + level, name, "".join(np.base_repr(i, 36).lower() + v + f36(f) + c for i, v, f, c in items)])
    return body + "." + np.base_repr(sum(map(ord, body)) % 36, 36).lower()


def test_parse_validation_code():
    ids = [f"D{i:02d}" for i in range(1, 19)]
    sugg = {i: "F05" for i in ids}
    code = _code("ABC123", [(0, "y", "", "h"), (17, "n", "F19", "x"), (5, "p", "F00", "l"), (3, "x", "", "m")])
    d = vc.parse_validation("From Alex: " + code + " thanks", ids, "ABC123", sugg)
    assert d["role"] == "qualified_coach" and d["level"] == "county_academy_plus" and d["name"] == "Alex"
    assert d["answers"]["D01"] == {"verdict": "yes", "alex_fault": "F05", "confidence": "very"}
    assert d["answers"]["D18"] == {"verdict": "no", "alex_fault": "F19", "confidence": ""}
    assert d["answers"]["D06"] == {"verdict": "partly", "alex_fault": "F00", "confidence": "not_very"}
    assert d["answers"]["D04"] == {"verdict": "cannot_judge", "alex_fault": "", "confidence": "fairly"}
    with pytest.raises(ValueError, match="checksum"):
        vc.parse_validation(code[:-1] + ("0" if code[-1] != "0" else "1"), ids, "ABC123", sugg)
    with pytest.raises(ValueError, match="build"):
        vc.parse_validation(code, ids, "FFFFFF", sugg)
    with pytest.raises(ValueError, match="without a main fault"):
        vc.parse_validation(_code("ABC123", [(0, "n", "", "x")]), ids, "ABC123", sugg)


def test_decode_cli_reports_agreement_per_source_and_against_cam(tmp_path, capsys):
    rows = sug.suggest(NAMES, sug.load_sources())
    key = [{"clip_id": f"D{i + 1:02d}", "filename": r["filename"], "build_id": "ABC123", **{k: v for k, v in r.items() if k != "filename"}}
           for i, r in enumerate(rows)]
    with (tmp_path / "key.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    cam = {sug.key_of(r["filename"]): r["main_fault"] for r in csv.DictReader((vc.CAM).open())}
    items = []
    for i, k in enumerate(key):  # agree with the AI on the first 9, then name Cam's fault (or F00) on the rest
        if i < 9:
            items.append((i, "y", "", "h"))
        else:
            want = sug.to_v03(cam.get(sug.key_of(k["filename"]), "F00"), *sug.v03_faults()) or "F00"
            items.append((i, "n" if want != k["suggested_fault"] else "y", "" if want == k["suggested_fault"] else want, "m"))
    code = _code("ABC123", items)
    (tmp_path / "codes.txt").write_text("hello\n" + code + "\n" + code + "\n")
    assert vc.main(["--codes", str(tmp_path / "codes.txt"), "--key", str(tmp_path / "key.csv"), "--out", str(tmp_path / "out.csv")]) == 0
    out = list(csv.DictReader((tmp_path / "out.csv").open()))
    assert len(out) == 18 and list(out[0]) == vc.FIELDS  # duplicate code dropped
    assert all(r["alex_fault"] == r["suggested_fault"] for r in out if r["verdict"] == "yes")
    text = capsys.readouterr().out
    assert "1 rater(s), 18 clip answers" in text and "AI's main fault right:" in text
    for s in sug.SOURCES:
        assert s in text
    assert "this rater v Cam (exact)" in text and "Measurement-based (Phase 6) suggestion right when shown" in text


def test_template_code_format_matches_decoder():
    t = (ROOT / "docs" / "phase6" / "offline_validate_template.html").read_text()
    assert '"VIQ1"' in t and "toString(36)" in t and "default-src 'none'" in t
    for label in ("Yes, that's the main thing", "Partly — it's there but not the main thing", "Can't judge this clip",
                  "Not very", "Fairly", "Very", "1×", "0.5×", "0.25×", "Restart"):
        assert label in t
    assert "localStorage" in t and "http" not in t.replace("http-equiv", "")
