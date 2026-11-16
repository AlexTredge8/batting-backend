"""Phase 3 clip check: schema, ground truth, comparison logic and cost guard (HTTP mocked; nothing sent)."""

from __future__ import annotations

import base64
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import phase3_clip_check as cc  # noqa: E402
from phase0_baseline_table import key_of, load_csv  # noqa: E402

SHEET = {key_of(r["filename"]): r for r in load_csv(cc.CLIP_SHEET)}
COACH = {key_of(r["filename"]): r for r in load_csv(cc.COACH_GT)}
NAMES = [r["filename"] for r in load_csv(cc.COACH_GT)]


def make_offline(tmp_path: Path) -> tuple[Path, Path]:
    """A fake swipe file (tiny stand-in clips) and its private key, as phase2_offline_pairs.py writes them."""
    clips, rows = [], []
    for i, name in enumerate(NAMES, 1):
        cid = f"D{i:02d}"
        clips.append({"id": cid, "b64": base64.b64encode(f"clip-{cid}".encode()).decode()})
        rows.append({"clip_id": cid, "filename": name, "build_id": "ABC123", "bytes": 10, "frames": 90, "fps": 30.0, "size": "362x640"})
    html = tmp_path / "BattingIQ_which_drive.html"
    html.write_text("<script>const DATA = " + json.dumps({"build": "ABC123", "pairs": 40, "clips": clips}) + ";</script>")
    key = tmp_path / "BattingIQ_which_drive_key_PRIVATE.csv"
    with key.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return html, key


def truths() -> dict:
    return {n: cc.truth_for(SHEET.get(key_of(n)), COACH.get(key_of(n)), n) for n in NAMES}


def perfect_answer(name: str) -> dict:
    """What a model that agreed with every piece of ground truth would say."""
    t = cc.truth_for(SHEET.get(key_of(name)), COACH.get(key_of(name)), name)
    ans = {"shot_type": "front-foot drive", "drive_direction": "off", "camera_position": "behind the bowler",
           "batter_handedness": "right", "feed": "unclear", "venue": "other", "whole_batter_visible": "yes",
           "phone_level": "yes", "number_of_people_visible": 1, "multiple_shots_in_clip": "no", "batter_size": "medium",
           "usable_reason": "fine", "usable_for_analysis": "yes"}
    for f, tr in t.items():
        ans[f] = (2 if tr.expected == "2+" else int(tr.expected)) if f == "number_of_people_visible" else tr.expected
    return ans


def reply(ans: dict, prompt: int = 3000, out: int = 120, thoughts: int = 400) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(ans)}]}}],
            "usageMetadata": {"promptTokenCount": prompt, "candidatesTokenCount": out, "thoughtsTokenCount": thoughts}}


def name_of(body: dict, key: Path) -> str:
    by_b64 = {base64.b64encode(f"clip-{r['clip_id']}".encode()).decode(): r["filename"] for r in load_csv(key)}
    return by_b64[next(p for p in body["contents"][0]["parts"] if "inlineData" in p)["inlineData"]["data"]]


# --- schema and requests ----------------------------------------------------

def test_schema_has_every_field_with_the_asked_answers():
    props = cc.SCHEMA["properties"]
    assert list(props) == list(cc.FIELDS) == cc.SCHEMA["required"] == cc.SCHEMA["propertyOrdering"]
    assert props["shot_type"]["enum"] == ["front-foot drive", "back-foot shot", "defensive", "other"]
    assert props["drive_direction"]["enum"] == ["off", "straight", "on", "not a drive"]
    assert props["camera_position"]["enum"] == ["behind the bowler", "side-on", "behind the batter", "other"]
    assert props["batter_handedness"]["enum"] == ["right", "left"]
    assert props["feed"]["enum"] == ["bowling machine", "bowler", "throw-down", "unclear"]
    assert props["venue"]["enum"] == ["indoor net lane", "sports hall", "outdoor net", "middle of a ground", "other"]
    assert props["phone_level"]["enum"] == ["yes", "tilted"]
    assert props["batter_size"]["enum"] == ["small", "medium", "large"]
    assert props["number_of_people_visible"] == {"type": "INTEGER"}
    assert props["usable_reason"] == {"type": "STRING"}
    for f in ("whole_batter_visible", "multiple_shots_in_clip", "usable_for_analysis"):
        assert props[f]["enum"] == ["yes", "no"]
    for f in cc.FIELDS:
        assert f in cc.PROMPT


def test_requests_uncropped_8fps_two_runs_run_one_first(tmp_path, capsys):
    html, key = make_offline(tmp_path)
    args = cc.parse_args(["--clips-html", str(html), "--key", str(key), "--out", str(tmp_path / "o")])
    assert not args.crop and args.runs == 2 and args.video_fps == 8.0 and args.temperature == 0.0 and args.budget_gbp == 0.80
    clips = cc.v1.clips_from_html(html)
    names = {r["clip_id"]: r["filename"] for r in load_csv(key)}
    jobs = cc.build_jobs(names, clips, {c: 3.0 for c in names}, args)
    assert len(jobs) == 36
    assert [j.meta["run"] for j in jobs] == [1] * 18 + [2] * 18
    assert [j.meta["filename"] for j in jobs[:18]] == [j.meta["filename"] for j in jobs[18:]]
    body = jobs[0].body
    video = body["contents"][0]["parts"][0]
    assert video["videoMetadata"] == {"fps": 8.0}
    assert base64.b64decode(video["inlineData"]["data"]) == clips[jobs[0].clip_ids[0]]  # sent as it is, not cropped
    cfg = body["generationConfig"]
    assert cfg["temperature"] == 0.0 and cfg["responseSchema"] is cc.SCHEMA and cfg["thinkingConfig"] == {"thinkingLevel": "low"}
    assert all(set(j.meta) == {"filename", "run"} for j in jobs)  # no clip codes in what is written out
    assert 0 < jobs[0].est_usd < 0.05


def test_dry_run_sends_nothing_and_fits_the_cap(tmp_path, capsys):
    html, key = make_offline(tmp_path)

    def no_post(url, body):
        raise AssertionError("dry run must not send")
    out = tmp_path / "o"
    assert cc.main(["--clips-html", str(html), "--key", str(key), "--out", str(out), "--dry-run"], post=no_post) == 0
    assert not out.exists()
    text = capsys.readouterr().out
    assert "36 requests (18 clips x 2 runs)" in text and "about 36 of 36 requests fit" in text


def test_out_inside_repo_must_be_under_baselines():
    with pytest.raises(SystemExit):
        cc.parse_args(["--clips-html", "x", "--key", "y", "--out", str(ROOT / "results" / "cc")])


# --- ground truth -----------------------------------------------------------

def test_ground_truth_from_the_clip_sheet_and_known_facts():
    t = truths()
    cam = t["Cam T_Offdrive Elite.mp4"]
    assert (cam["batter_handedness"].expected, cam["batter_handedness"].source) == ("left", cc.CONFIRMED)
    assert (cam["drive_direction"].expected, cam["drive_direction"].source) == ("off", cc.FILENAME)
    assert cam["batter_size"].expected == "small" and cam["batter_size"].source == cc.MEASURED  # 19.3 %
    assert (cam["feed"].expected, cam["feed"].source) == ("bowling machine", cc.HUB)
    assert (cam["camera_position"].expected, cam["camera_position"].source) == ("behind the bowler", cc.HUB)
    assert t["Cam T_Ondrive Elite.mov"]["batter_size"].expected == "large"  # 40.4 %
    assert t["Alex T_Straightdrive Good.MOV"]["drive_direction"].expected == "straight"
    assert t["Straight Drive_Beginner.mp4"]["drive_direction"].source == cc.FILENAME
    assert t["Drive_Average 2.MOV"]["drive_direction"].source == cc.HUB
    assert t["Drive_Average.MOV"]["camera_position"].expected == "other"  # hub: angle 20-30 degrees
    assert t["Dan P_Offdrive Average.MOV"]["venue"].expected == "outdoor net"
    ollie = t["Ollie P_Offdrive Average.mov"]
    assert (ollie["number_of_people_visible"].expected, ollie["number_of_people_visible"].source) == ("2+", cc.KNOWN)
    assert ollie["usable_for_analysis"].expected == "no" and "batter_size" not in ollie  # height measurement failed
    db4 = t["Drive_Beginner 4.MOV"]
    assert db4["usable_for_analysis"].expected == "no" and "batter_size" not in db4
    assert t["Female_Average.mp4"]["usable_for_analysis"].expected == "no"
    assert t["Viv P_Offdrive Elite.mov"]["usable_for_analysis"].expected == "yes"
    assert t["Drive_Beginner 2.MOV"]["number_of_people_visible"].expected == "2+"
    assert all(x["shot_type"].expected == "front-foot drive" and x["multiple_shots_in_clip"].expected == "no" for x in t.values())
    for x in t.values():  # no ground truth for these two
        assert "whole_batter_visible" not in x and "phone_level" not in x


def test_matches_people_count_and_strings():
    assert cc.matches("number_of_people_visible", "2+", 3) is True
    assert cc.matches("number_of_people_visible", "2+", 1) is False
    assert cc.matches("number_of_people_visible", "1", 1) is True
    assert cc.matches("number_of_people_visible", "1", "") is None
    assert cc.matches("feed", "throw-down", "Throw-down") is True
    assert cc.matches("feed", "throw-down", "unclear") is False
    assert cc.size_band(19.9) == "small" and cc.size_band(20) == "medium" and cc.size_band(40) == "large"


# --- comparison and verdicts --------------------------------------------------

def results_for(answers_by_run: dict[int, dict[str, dict]]) -> list[dict]:
    out, i = [], 0
    for run, by_name in answers_by_run.items():
        for name, ans in by_name.items():
            out.append({"i": i, "job": cc.v2.Job("clip_check", {"filename": name, "run": run}, {}, 0.0), "answer": ans,
                        "usage": {"prompt": 1, "candidates": 1, "thoughts": 0}, "error": ""})
            i += 1
    return out


def test_perfect_and_consistent_answers_are_reliable():
    perfect = {n: perfect_answer(n) for n in NAMES}
    answers, comparison, summary, stats = cc.analyse(results_for({1: perfect, 2: perfect}), NAMES, truths(), 2)
    by = {r["field"]: r for r in summary}
    assert stats == {"clips_answered": 18, "clips_both_runs": 18, "errors": 0}
    assert len(answers) == 36 and len(comparison) == 18 * len(cc.COMPARED)
    assert by["batter_handedness"]["agreement"] == 1.0 and by["batter_handedness"]["consistency"] == 1.0
    assert by["batter_handedness"]["verdict"] == "reliable enough for an automatic check"
    assert by["feed"]["verdict"].startswith("provisional (ground truth is a hub guess)")
    assert by["camera_position"]["verdict"].startswith("provisional")
    assert by["number_of_people_visible"]["verdict"].startswith("provisional")  # 17 of 18 are hub guesses
    assert not by["drive_direction"]["verdict"].startswith("provisional")  # mostly from filenames
    assert "only one right answer" in by["multiple_shots_in_clip"]["verdict"]
    assert by["whole_batter_visible"]["verdict"] == "consistent, but accuracy unknown (no ground truth)"
    assert by["batter_size"]["answers_scored"] == 32  # 16 clips with a measurement x 2 runs
    use, soft, no = cc.recommendation(summary)
    assert "batter_handedness" in use and "usable_for_analysis" in use and "batter_size" in use
    assert "camera_position" in soft and "multiple_shots_in_clip" in soft
    assert "whole_batter_visible" in no and "phone_level" in no


def test_inconsistent_runs_and_wrong_answers_are_flagged():
    run1 = {n: perfect_answer(n) for n in NAMES}
    run2 = {n: dict(a) for n, a in run1.items()}
    for i, n in enumerate(NAMES):
        run2[n]["phone_level"] = "tilted" if i % 2 else "yes"           # flips on half the clips
        run2[n]["batter_handedness"] = "left" if run1[n]["batter_handedness"] == "right" else "right"  # always wrong in run 2
        run1[n]["venue"] = "other"                                        # wrong in run 1 for every clip
    answers, comparison, summary, _ = cc.analyse(results_for({1: run1, 2: run2}), NAMES, truths(), 2)
    by = {r["field"]: r for r in summary}
    assert by["phone_level"]["consistency"] == pytest.approx(9 / 18)
    assert by["phone_level"]["verdict"] == "not reliable (answers change between identical runs)"
    assert by["batter_handedness"]["agreement"] == pytest.approx(0.5)
    assert by["batter_handedness"]["consistency"] == 0.0
    assert by["venue"]["agreement"] == pytest.approx(0.5) and by["venue"]["verdict"].startswith("not reliable")
    assert set(by["batter_handedness"]["disagreeing_clips"].split("; ")) == set(NAMES)
    row = next(r for r in comparison if r["field"] == "batter_handedness" and r["filename"] == NAMES[0])
    assert row["agrees"] == "1/0" and row["consistent"] == 0 and row["source"] == cc.CONFIRMED


def test_bad_or_missing_answers_are_not_counted():
    run1 = {n: perfect_answer(n) for n in NAMES}
    run1[NAMES[0]] = {**run1[NAMES[0]], "feed": "catapult", "number_of_people_visible": "lots"}
    results = results_for({1: run1})
    results.append({"i": 99, "job": cc.v2.Job("clip_check", {"filename": NAMES[1], "run": 2}, {}, 0.0), "answer": {},
                    "usage": {}, "error": "Gemini API error 500: x"})
    answers, comparison, summary, stats = cc.analyse(results, NAMES, truths(), 2)
    assert stats["errors"] == 1 and stats["clips_both_runs"] == 0
    first = next(a for a in answers if a["filename"] == NAMES[0])
    assert first["feed"] == "" and first["number_of_people_visible"] == ""
    by = {r["field"]: r for r in summary}
    assert by["feed"]["answers_scored"] == 17 and by["feed"]["clips_both_runs"] == 0


# --- cost guard and end to end ------------------------------------------------

def test_cost_guard_stops_at_the_cap_and_run_one_covers_every_clip(tmp_path):
    html, key = make_offline(tmp_path)
    calls = []

    def fake_post(url, body):
        calls.append(name_of(body, key))
        # 3000 prompt + 120 answer + 400 thinking tokens = USD 0.0153 = GBP 0.01224 per request
        return reply(perfect_answer(calls[-1]))

    out = tmp_path / "o"
    rc = cc.main(["--clips-html", str(html), "--key", str(key), "--out", str(out), "--model", "gemini-test",
                  "--workers", "1", "--budget-gbp", "0.25"], post=fake_post)
    assert rc == 0
    assert 18 <= len(calls) < 36 and len(set(calls)) == 18  # every clip once before any is asked twice
    rows = load_csv(out / "answers.csv")
    assert len(rows) == len(calls)
    summary = (out / "summary.md").read_text()
    assert "Stopped by the cost guard" in summary
    spent = float(summary.split("spent about GBP ")[1].split(" ")[0])
    assert spent <= 0.25


def test_end_to_end_with_mocked_gemini_writes_the_summary_and_no_clip_codes(tmp_path):
    html, key = make_offline(tmp_path)

    def fake_post(url, body):
        assert url.endswith("/models/gemini-test:generateContent")
        return reply(perfect_answer(name_of(body, key)))

    out = tmp_path / "o"
    assert cc.main(["--clips-html", str(html), "--key", str(key), "--out", str(out), "--model", "gemini-test"], post=fake_post) == 0
    assert {p.name for p in out.iterdir()} == {"answers.csv", "comparison.csv", "field_summary.csv", "summary.md", "responses.jsonl"}
    summary = (out / "summary.md").read_text()
    assert "| field | ground truth | agreement |" in summary and "## Recommendation" in summary
    assert "unconfirmed" in summary and "Requests sent 36 of 36" in summary
    assert len(load_csv(out / "answers.csv")) == 36
    for f in out.iterdir():
        text = f.read_text()
        assert not any(f'"D{i:02d}"' in text or f",D{i:02d}," in text for i in range(1, 19)), f.name


def test_rejected_key_exits_3(tmp_path):
    html, key = make_offline(tmp_path)

    def fake_post(url, body):
        raise cc.v2.KeyRejected("Google rejected the API key (400: API key not valid)")
    rc = cc.main(["--clips-html", str(html), "--key", str(key), "--out", str(tmp_path / "o"), "--model", "m", "--workers", "1"],
                 post=fake_post)
    assert rc == cc.v2.KEY_REJECTED_EXIT
