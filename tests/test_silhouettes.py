"""Fault cartoon pictures (silhouettes.py, assets/silhouettes): every fault has its pair and an index
entry, every SVG parses, and the pictures are static (no per-player data, no images of people)."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import silhouettes as sil  # noqa: E402

ASSETS = ROOT / "assets" / "silhouettes"
SVG_NS = "{http://www.w3.org/2000/svg}"


def _rubric_faults():
    r = json.loads((ROOT / "docs" / "phase1" / "rubric_v0_3.json").read_text())
    return [f["id"] for f in r["faults_in_chain_order"]], [f["id"] for f in r.get("faults_retired", [])]


def _index():
    return json.loads((ASSETS / "index.json").read_text())


def test_every_fault_has_a_pair_and_an_index_entry():
    faults, retired = _rubric_faults()
    idx = _index()
    assert len(faults) == 18 and "F14" in retired
    for fid in faults:
        e = idx[fid]
        assert e["view"] in ("front_on", "side_on", "top_down", "pitch_side"), fid
        assert e["caption_fault"] and e["caption_correct"], fid
        assert 2 <= len(e["caption_fault"].split()) <= 6 and 2 <= len(e["caption_correct"].split()) <= 6, fid
        assert e["cue"]["type"] and e["cue"]["fault"] and e["cue"]["correct"], fid
        assert e["alt"], fid
        for key in ("fault", "correct", "pair"):
            assert e["files"][key] == f"{fid}_{key}.svg", (fid, key)
            assert (ASSETS / e["files"][key]).is_file(), (fid, key)
    assert idx["F14"]["same_as"] == "F07"
    f00 = idx["F00"]
    assert f00["files"]["fault"] is None and (ASSETS / f00["files"]["correct"]).is_file()
    assert (ASSETS / f00["files"]["pair"]).is_file() and (ASSETS / f00["files"]["correct_finish"]).is_file()


def test_moments_match_the_comparison_table():
    import pose_compare as pc
    idx = _index()
    for fid, spec in pc.FAULT_TABLE.items():
        e = idx[fid]
        if e.get("same_as"):
            e = idx[e["same_as"]]
        assert e["moment"] == spec["moment"], fid


def test_every_svg_parses_and_is_self_contained():
    files = sorted(ASSETS.glob("*.svg"))
    assert len(files) >= 3 * 18 + 3
    for f in files:
        text = f.read_text(encoding="utf-8")
        root = ET.fromstring(text)
        assert root.tag == f"{SVG_NS}svg", f.name
        assert root.get("viewBox"), f.name
        # static drawings only: no embedded or linked images, no scripts, no external references
        for bad in ("<image", "<script", "href=", "base64", "@import", "url(http"):
            assert bad not in text, (f.name, bad)
        if f.name.endswith(("_fault.svg", "_correct.svg", "_correct_finish.svg")):
            assert root.find(f".//{SVG_NS}text") is None, f.name   # mirror-safe: no words in the single panels
            assert root.find(f"{SVG_NS}title") is not None and root.find(f"{SVG_NS}desc") is not None, f.name


def test_pair_files_carry_labels_and_captions():
    idx = _index()
    for fid, e in idx.items():
        if fid.startswith("_") or e.get("same_as"):
            continue
        text = (ASSETS / e["files"]["pair"]).read_text(encoding="utf-8")
        assert "This" in text
        if e["caption_fault"]:
            assert "Not this" in text
        for word in (e["caption_fault"] or "").split()[:1] + e["caption_correct"].split()[:1]:
            assert word.replace("&", "&amp;") in text, (fid, word)


def test_built_assets_are_up_to_date(tmp_path):
    """The committed SVGs and index are exactly what silhouettes.py draws now (rebuild with
    `python silhouettes.py` after changing a pose)."""
    sil.build(tmp_path, png=False)
    for f in tmp_path.glob("*.svg"):
        assert f.read_text(encoding="utf-8") == (ASSETS / f.name).read_text(encoding="utf-8"), f.name
    built = json.loads((tmp_path / "index.json").read_text())
    for e in built.values():   # the committed index also lists the PNGs
        if isinstance(e, dict) and "files" in e:
            e["files"] = {k: v for k, v in e["files"].items() if not k.endswith("_png")}
    committed = _index()
    for e in committed.values():
        if isinstance(e, dict) and "files" in e:
            e["files"] = {k: v for k, v in e["files"].items() if not k.endswith("_png")}
    assert built == committed


def test_pose_is_a_parameter_set():
    """A fault picture is the correct pose with parameters pushed, not a drawing: changing one angle
    moves only the joints below it in the chain."""
    p = sil.FRONT_CONTACT
    a = sil.joints(p)
    b = sil.joints(sil.Pose(**{**p.__dict__, "head": p.head + 30}))
    assert a["head"] != b["head"]
    assert all(a[k] == b[k] for k in a if k != "head")
    assert sil.joints(p)["front_ankle"][1] == sil.ANKLE_H - p.ff_depth
