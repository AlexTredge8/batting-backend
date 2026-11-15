"""Video judge v2: request building, crop box, A/B randomisation, cost guard and retries (HTTP mocked; nothing sent)."""

from __future__ import annotations

import base64
import csv
import io
import json
import random
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import phase5_video_judge_v2 as v2  # noqa: E402
from phase0_baseline_table import key_of, load_csv  # noqa: E402

GT = ROOT / "coach_ground_truth_from_screenshot.csv"
NAMES = [r["filename"] for r in load_csv(GT)]
ALEX = {r["filename"]: float(r["overall_score"]) for r in load_csv(GT)}


def make_offline(tmp_path: Path) -> tuple[Path, Path]:
    """A fake offline HTML (tiny stand-in clips) and its private key, as phase2_offline_pairs.py writes them."""
    clips, rows = [], []
    for i, name in enumerate(NAMES, 1):
        cid = f"D{i:02d}"
        clips.append({"id": cid, "b64": base64.b64encode(f"clip-{cid}".encode()).decode()})
        rows.append({"clip_id": cid, "filename": name, "build_id": "ABC123", "bytes": 10, "frames": 75, "fps": 30.0, "size": "362x640"})
    html = tmp_path / "BattingIQ_which_drive.html"
    html.write_text("<script>const DATA = " + json.dumps({"build": "ABC123", "pairs": 40, "clips": clips}) + ";</script>")
    key = tmp_path / "BattingIQ_which_drive_key_PRIVATE.csv"
    with key.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return html, key


def ctx_and_args(tmp_path: Path, **kw):
    html, key = make_offline(tmp_path)
    ctx = v2.load_context(html, key, GT, v2.CAM_QUICK)
    args = SimpleNamespace(video_fps=8.0, max_pairs=None, seed=1, refs_per_level=1, thinking=None, media_resolution="default",
                           temperature=0.0, usd_per_mtok_in=2.5, usd_per_mtok_out=15.0)
    for k, v in kw.items():
        setattr(args, k, v)
    return ctx, args


# --- request building -------------------------------------------------------

def test_targets_exclude_the_three_references(tmp_path):
    ctx, _ = ctx_and_args(tmp_path)
    assert len(ctx.targets) == 15
    assert not {key_of(ctx.names[c]) for c in ctx.targets} & {key_of(f) for f in v2.v1.REFERENCES}


def test_pairwise_requests(tmp_path):
    ctx, args = ctx_and_args(tmp_path)
    jobs = v2.build_pairwise_jobs(ctx, ctx.clips, args)
    assert len(jobs) == 105
    assert len({frozenset((j.meta["file_a"], j.meta["file_b"])) for j in jobs}) == 105
    parts = jobs[0].body["contents"][0]["parts"]
    videos = [p for p in parts if "inlineData" in p]
    assert len(videos) == 2 and all(p["videoMetadata"] == {"fps": 8.0} for p in videos)
    a, b = jobs[0].clip_ids
    assert base64.b64decode(videos[0]["inlineData"]["data"]) == ctx.clips[a]
    assert base64.b64decode(videos[1]["inlineData"]["data"]) == ctx.clips[b]
    assert jobs[0].body["generationConfig"]["responseSchema"]["properties"]["better"]["enum"] == ["A", "B"]
    assert "thinkingConfig" not in jobs[0].body["generationConfig"]
    assert all(j.est_usd > 0 for j in jobs)
    # outputs carry filenames, never the private clip codes
    assert not any(str(v).startswith("D") and len(str(v)) == 3 for j in jobs for v in j.meta.values())
    args.max_pairs, args.thinking, args.media_resolution = 40, "low", "high"
    jobs = v2.build_pairwise_jobs(ctx, ctx.clips, args)
    assert len(jobs) == 40
    cfg = jobs[0].body["generationConfig"]
    assert cfg["thinkingConfig"] == {"thinkingLevel": "low"} and cfg["mediaResolution"] == "MEDIA_RESOLUTION_HIGH"


def test_absolute_refs_per_level_exclude_the_clip_judged(tmp_path):
    ctx, args = ctx_and_args(tmp_path)
    for cid in ctx.targets:
        refs1 = v2.choose_refs(cid, ctx, 1)
        refs2 = v2.choose_refs(cid, ctx, 2)
        assert cid not in refs1 and cid not in refs2
        assert len(refs1) == 4 and len(refs2) == 8 and len(set(refs2)) == 8
        tiers = [ctx.gt[key_of(ctx.names[r])]["tier"] for r in refs2]
        assert tiers == ["Elite", "Elite", "Good Club", "Good Club", "Average", "Average", "Beginner", "Beginner"]
        # the three original references come first in their tier
        assert {ctx.names[r] for r in refs1} >= set(v2.v1.REFERENCES)
    args.refs_per_level = 2
    jobs = v2.build_absolute_jobs(ctx, ctx.clips, args)
    assert len(jobs) == 15
    parts = jobs[0].body["contents"][0]["parts"]
    assert sum("inlineData" in p for p in parts) == 9
    assert "You will first see 8 reference drives" in jobs[0].body["systemInstruction"]["parts"][0]["text"]
    assert jobs[0].meta["filename"] not in jobs[0].meta["refs"].split(";")


def test_checkpoint_requests_cover_the_12_checkpoints_in_4_groups(tmp_path):
    ctx, args = ctx_and_args(tmp_path)
    groups = v2.checkpoint_groups()
    assert len(groups) == 4
    assert [c["id"] for _, g in groups for c in g] == v2.CPS
    jobs = v2.build_checkpoint_jobs(ctx, ctx.clips, args)
    assert len(jobs) == 60
    props = jobs[1].body["generationConfig"]["responseSchema"]["properties"]
    assert set(props) == {"C04", "C05", "C06", "C07", "note"}
    assert props["C04"]["enum"] == ["yes", "partly", "no", "cannot_see"]


# --- crop box ---------------------------------------------------------------

def body(x0, y0, x1, y1, vis=0.9):
    f = np.zeros((33, 4), dtype=float)
    f[:, 0] = np.linspace(x0, x1, 33)
    f[:, 1] = np.linspace(y0, y1, 33)
    f[:, 3] = vis
    return f


def test_crop_box_union_padded_even_and_clamped():
    frames = [body(0.40, 0.30, 0.50, 0.80), body(0.42, 0.28, 0.55, 0.82), body(0.45, 0.30, 0.52, 0.85)]
    frames.append(np.full((33, 4), np.nan))            # no pose in one frame
    frames.append(body(0.0, 0.0, 1.0, 1.0, vis=0.1))   # low-visibility junk is ignored
    x, y, w, h = v2.crop_box(np.array(frames), 400, 600, pad=0.25)
    # union x 0.40-0.55 (w 0.15), y 0.28-0.85 (h 0.57); 25% pad on each side
    assert x == pytest.approx((0.40 - 0.0375) * 400, abs=1.5)
    assert y == pytest.approx((0.28 - 0.1425) * 600, abs=1.5)
    assert x + w == pytest.approx((0.55 + 0.0375) * 400, abs=2.5)
    assert y + h == pytest.approx(min(1.0, 0.85 + 0.1425) * 600, abs=2.5)
    assert w % 2 == 0 and h % 2 == 0 and x + w <= 400 and y + h <= 600
    # near the edge the box is clamped to the frame
    x, y, w, h = v2.crop_box(np.array([body(0.0, 0.1, 0.3, 0.99)] * 3), 400, 600)
    assert x == 0 and y + h <= 600 and w % 2 == 0


def test_crop_box_drops_a_frame_on_someone_else_and_needs_a_pose():
    good = [body(0.40, 0.30, 0.50, 0.80)] * 5
    stray = body(0.85, 0.05, 0.95, 0.30)  # pose jumped to another person for one frame
    with_stray = v2.crop_box(np.array(good + [stray]), 400, 600)
    assert with_stray == v2.crop_box(np.array(good), 400, 600)
    assert v2.crop_box(np.full((10, 33, 4), np.nan), 400, 600) is None


def test_scaled_size_only_shrinks():
    assert v2.scaled_size(1080, 1920, 720) == (404, 720)
    assert v2.scaled_size(132, 300, 720) == (132, 300)


# --- position bias ----------------------------------------------------------

def test_pair_order_randomises_sides_and_keeps_prefixes_balanced():
    ids = [f"clip{i:02d}" for i in range(15)]
    order = v2.balanced_pair_order(ids, None, random.Random(7))
    assert len(order) == 105 and len({frozenset(p[:2]) for p in order}) == 105
    for a, b, swapped in order:
        assert swapped == (a > b)
    n_swapped = sum(s for _, _, s in order)
    assert 35 <= n_swapped <= 70  # random sides, about half
    assert order == v2.balanced_pair_order(ids, None, random.Random(7))  # reproducible from the seed
    assert order != v2.balanced_pair_order(ids, None, random.Random(8))
    for k in (10, 30, 45, 60):
        count = {i: 0 for i in ids}
        for a, b, _ in order[:k]:
            count[a] += 1
            count[b] += 1
        assert max(count.values()) - min(count.values()) <= 2
    first_shown = {i: 0 for i in ids}
    for a, _, _ in order:
        first_shown[a] += 1
    assert max(first_shown.values()) < 14  # no clip is always shown first


def test_pairwise_analysis_uses_winner_not_position():
    files = sorted(ALEX)[:6]
    best = sorted(files, key=lambda f: -ALEX[f])
    results = []
    for n, (a, b, swapped) in enumerate(v2.balanced_pair_order(files, None, random.Random(3)), 1):
        better = "A" if ALEX[a] > ALEX[b] else "B"
        job = v2.Job("pairwise", {"pair_no": n, "file_a": a, "file_b": b, "swapped": swapped}, {}, 0.0)
        results.append({"job": job, "answer": {"better": better, "confidence": "high", "fault_a": "F00", "fault_b": "F04"},
                        "usage": {}, "error": ""})
    rows, scale, stats = v2.analyse_pairwise(results, files, ALEX, {})
    assert [r["filename"] for r in scale][:2] == best[:2]
    assert stats["pick_agreement_alex"] == 1.0
    assert stats["spearman_alex"] > 0.9
    assert 0 < stats["share_A"] < 1


# --- cost guard, retries, key rejection --------------------------------------

def test_cost_guard_stops_before_the_cap():
    guard = v2.CostGuard(budget_gbp=0.10, usd_in=2.5, usd_out=15.0, gbp_per_usd=0.8)
    jobs = [v2.Job("checkpoints", {"filename": f"f{i}", "group": 1, "group_name": "g", "checkpoints": []}, {}, est_usd=0.01)
            for i in range(50)]
    calls = []

    def fake_post(url, body):
        calls.append(url)
        # 2000 prompt + 300 answer + 1200 thinking tokens = USD 0.0275 = GBP 0.022 per request
        return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}],
                "usageMetadata": {"promptTokenCount": 2000, "candidatesTokenCount": 300, "thoughtsTokenCount": 1200}}

    results = v2.run_jobs(jobs, "gemini-test", guard, workers=1, post=fake_post, log=lambda s: None)
    assert guard.spent_gbp <= 0.10
    assert len(calls) == len(results) == 4  # 4 x 0.022 = 0.088; a 5th would pass 0.10
    assert guard.tokens == {"prompt": 8000, "candidates": 1200, "thoughts": 4800}
    assert guard.ratio == pytest.approx(0.0275 / 0.01)  # estimates scaled up after the first answer
    assert guard.refused >= 1


def test_cost_guard_holds_with_requests_in_flight():
    guard = v2.CostGuard(budget_gbp=0.05, gbp_per_usd=1.0)
    held = [guard.reserve(0.02) for _ in range(3)]
    assert held[:2] == [0.02, 0.02] and held[2] is None  # 0.06 reserved would pass 0.05
    guard.settle(held[0], 0.02, {"prompt": 1000, "candidates": 100, "thoughts": 0})
    assert guard.spent_usd == pytest.approx((1000 * 2.5 + 100 * 15) / 1e6)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code: int, message: str, headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", headers or {}, io.BytesIO(json.dumps({"error": {"message": message}}).encode()))


def test_post_retries_429_and_5xx_then_succeeds(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    seen, sleeps = [], []
    replies = [http_error(429, "slow down", {"Retry-After": "7"}), http_error(503, "busy"), b'{"ok": 1}']

    def opener(req, timeout):
        seen.append(req.get_header("X-goog-api-key"))
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return FakeResponse(r)

    assert v2.post_json("https://x", {"a": 1}, opener=opener, sleep=sleeps.append) == {"ok": 1}
    assert seen == ["AIza-test"] * 3
    assert 7 <= sleeps[0] < 8 and len(sleeps) == 2


def test_post_rejected_key_is_clear(monkeypatch):
    def opener(req, timeout):
        raise http_error(400, "API key not valid. Please pass a valid API key.")
    with pytest.raises(v2.KeyRejected, match="rejected the API key"):
        v2.post_json("https://x", {}, opener=opener, sleep=lambda s: None)
    with pytest.raises(v2.ApiError):
        v2.post_json("https://x", {}, opener=lambda r, timeout: (_ for _ in ()).throw(http_error(400, "bad schema")), sleep=lambda s: None)


def test_unsupported_thinking_setting_is_dropped_and_resent():
    job = v2.Job("checkpoints", {"filename": "f", "group": 1, "group_name": "g", "checkpoints": ["C01"]},
                 {"generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}}}, 0.001)
    bodies = []

    def fake_post(url, body):
        bodies.append(json.loads(json.dumps(body)))
        if "thinkingConfig" in body["generationConfig"]:
            raise v2.ApiError(400, "Thinking level is not supported for this model.")
        return {"candidates": [{"content": {"parts": [{"text": '{"C01": "no", "note": "x"}'}]}}], "usageMetadata": {}}

    res = v2.run_jobs([job], "m", v2.CostGuard(1.0), workers=1, post=fake_post, log=lambda s: None)
    assert len(bodies) == 2 and res[0]["answer"]["C01"] == "no" and not res[0]["error"]


def test_main_dry_run_sends_nothing(tmp_path, capsys):
    html, key = make_offline(tmp_path)
    def no_post(url, body):
        raise AssertionError("dry run must not send")
    for mode in ("pairwise", "absolute", "checkpoints"):
        assert v2.main(["--mode", mode, "--clips-html", str(html), "--key", str(key), "--out", str(tmp_path / mode), "--dry-run"], post=no_post) == 0
        assert not (tmp_path / mode).exists()
    assert "Estimated cost about GBP" in capsys.readouterr().out


def test_main_pairwise_end_to_end_with_mocked_gemini(tmp_path):
    html, key = make_offline(tmp_path)
    by_b64 = {}
    for r in load_csv(key):
        by_b64[base64.b64encode(f"clip-{r['clip_id']}".encode()).decode()] = r["filename"]

    def fake_post(url, body):
        a, b = [by_b64[p["inlineData"]["data"]] for p in body["contents"][0]["parts"] if "inlineData" in p]
        better = "A" if ALEX[a] >= ALEX[b] else "B"
        ans = {"fault_a": "F04", "fault_b": "F05", "reason": "x", "better": better, "confidence": "high"}
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(ans)}]}}],
                "usageMetadata": {"promptTokenCount": 3000, "candidatesTokenCount": 80, "thoughtsTokenCount": 500}}

    out = tmp_path / "out"
    rc = v2.main(["--mode", "pairwise", "--clips-html", str(html), "--key", str(key), "--out", str(out),
                  "--model", "gemini-test", "--max-pairs", "30", "--workers", "3"], post=fake_post)
    assert rc == 0
    rows = load_csv(out / "pairwise_judgements.csv")
    assert len(rows) == 30 and {r["swapped"] for r in rows} == {"0", "1"}
    summary = (out / "summary.md").read_text()
    assert "Spearman vs Alex" in summary and "Position bias" in summary
    # no private clip codes anywhere in what would be committed
    for f in out.iterdir():
        text = f.read_text()
        assert not any(f'"D{i:02d}"' in text or f",D{i:02d}," in text for i in range(1, 19)), f.name


def test_checkpoint_analysis_earliest_failed_checkpoint_is_the_main_fault():
    def res(fname, group, cps, answers):
        return {"job": v2.Job("checkpoints", {"filename": fname, "group": group, "group_name": "g", "checkpoints": cps}, {}, 0),
                "answer": dict(zip(cps, answers)), "usage": {}, "error": ""}
    results = [res("a.mov", 1, ["C01", "C02", "C03"], ["yes", "partly", "yes"]),
               res("a.mov", 2, ["C04", "C05", "C06", "C07"], ["no", "no", "yes", "cannot_see"]),
               res("b.mov", 1, ["C01", "C02", "C03"], ["yes", "partly", "yes"])]
    _, per_clip, stats = v2.analyse_checkpoints(results, ["a.mov", "b.mov"], {}, {}, {"a.mov": ("F04", "F05"), "b.mov": ("F00", "")})
    a, b = per_clip
    assert a["main_fault"] == "F04" and a["C07"] == "X" and a["complete"] == 0
    assert b["main_fault"] == "F00" and b["main_fault_incl_partly"] == "F02"
    assert stats["main_fault_matches_cam"] == 1.0
    assert stats["all_ratings"] == {"2": 5, "1": 2, "0": 2, "X": 1}
