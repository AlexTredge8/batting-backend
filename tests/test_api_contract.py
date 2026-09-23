"""
HTTP contract for POST /analyse (the path the Lovable frontend uses).

Runs the FastAPI app with the heavy pipeline monkeypatched out so the contract
can be checked in milliseconds: public media URLs + inline data URLs for the
storyboard stills, annotated_video_url, analysis_quality/warnings passthrough,
and the duration guard.
"""
import io
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import api


def _fake_pipeline(video_path: str, output_dir: str = None, **kwargs):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "input_battingiq_annotated.mp4").write_bytes(b"\x00" * 16)
    stills = []
    for i, phase in enumerate(("setup", "contact")):
        p = out / f"storyboard_{phase}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        stills.append({"index": i, "phase": phase, "label": phase.upper(), "path": str(p),
                       "original_frame_idx": 10 * i, "timestamp_ms": 333.0 * i})
    return {
        "battingiq_score": 84, "score_band": "Good", "pillars": {}, "phases": {},
        "metadata": {"processing_mode": "full_rate_calibrated"},
        "analysis_quality": {"audio_available": True, "reliable": True},
        "warnings": [],
        "storyboard_frames": stills,
        "_annotated_video": str(out / "input_battingiq_annotated.mp4"),
        "_storyboard": None,
        "_storyboard_frames": stills,
        "_storyboard_keyframes": {k: ("QUJD" if k in ("setup", "contact") else None)
                                  for k in ("setup", "hands_start_up", "front_foot_down",
                                            "hands_peak", "contact", "follow_through")},
    }


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "run_full_analysis", _fake_pipeline)
    monkeypatch.setattr(api, "RESULTS_DIR", tmp_path / "results")
    api.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "_probe_video", lambda p: {"fps": 30.0, "frames": 90, "duration_s": 3.0})
    return TestClient(api.app)


def _post(client, filename="clip.mov"):
    return client.post("/analyse", files={"file": (filename, io.BytesIO(b"fake"), "video/quicktime")},
                       data={"handedness": "right"})


def test_analyse_returns_public_media_and_quality(client):
    resp = _post(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["battingiq_score"] == 84
    assert body["job_id"]
    assert body["annotated_video_url"].startswith(f"/results/{body['job_id']}/output/")
    assert body["analysis_quality"]["audio_available"] is True
    assert body["warnings"] == []
    # Frontend contract (main 24b6135): six base64 JPEG keyframes keyed by phase.
    keyframes = body["storyboard_frames"]
    assert set(keyframes) == {"setup", "hands_start_up", "front_foot_down",
                              "hands_peak", "contact", "follow_through"}
    assert keyframes["contact"] == "QUJD"
    # Rich per-still items live in metadata, with public URLs and no leaked paths.
    frames = body["metadata"]["storyboard_frame_items"]
    assert len(frames) == 2
    for fr in frames:
        assert "path" not in fr, "internal filesystem paths must not leak"
        assert fr["url"].startswith(f"/results/{body['job_id']}/output/storyboard_")
        assert fr["data_url"].startswith("data:image/png;base64,")
    for key in ("_annotated_video", "_storyboard", "_storyboard_frames", "_storyboard_keyframes"):
        assert key not in body
    # the file the URL points at is actually served
    served = client.get(body["annotated_video_url"])
    assert served.status_code == 200


def test_analyse_rejects_unsupported_extension(client):
    resp = _post(client, filename="clip.txt")
    assert resp.status_code == 400


def test_analyse_rejects_overlong_clip(client, monkeypatch):
    monkeypatch.setattr(api, "_probe_video", lambda p: {"fps": 30.0, "frames": 9000, "duration_s": 300.0})
    resp = _post(client)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "video_too_long"
    assert detail["duration_s"] == 300.0


def test_analyse_bad_anchor_json_is_400_not_500(client):
    resp = client.post("/analyse", files={"file": ("clip.mov", io.BytesIO(b"fake"), "video/quicktime")},
                       data={"anchor_frames_json": "{not json"})
    assert resp.status_code == 400


def test_diag_reports_processing_mode(client):
    body = client.get("/diag").json()
    assert body["processing"]["mode"] in {"full_rate_calibrated", "fast_subsampled"}
    assert "ffmpeg_available" in body["processing"]
