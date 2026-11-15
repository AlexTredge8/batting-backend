"""
RTMPose fallback trigger logic (pose_extractor._apply_pose_fallback), with mocked extractors.

No MediaPipe or RTMPose inference runs here: poses are synthetic FramePose lists and the
RTMPose extractor is a stub, so these tests run without rtmlib/onnxruntime installed.
"""
import builtins
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pose_extractor as pe  # noqa: E402
from models import FramePose, RawLandmark  # noqa: E402

W, H = 480, 848


def _pose(frame_idx: int, nose_x: float = 0.5, vis: float = 0.9) -> FramePose:
    lms = [RawLandmark(x=0.5, y=0.5, z=0.0, visibility=vis) for _ in range(33)]
    lms[0] = RawLandmark(x=nose_x, y=0.2, z=0.0, visibility=vis)   # nose
    lms[27] = RawLandmark(x=0.45, y=0.8, z=0.0, visibility=vis)    # ankles: body length 0.6*H px
    lms[28] = RawLandmark(x=0.55, y=0.8, z=0.0, visibility=vis)
    return FramePose(frame_idx=frame_idx, timestamp_s=round(frame_idx / 30, 4), landmarks=lms)


def _clip(n: int = 40, missing: int = 0, jumps: int = 0) -> list[FramePose]:
    poses = [_pose(i) for i in range(n)]
    for i in range(missing):
        poses[i] = FramePose(frame_idx=i, timestamp_s=round(i / 30, 4), landmarks=None)
    # a jump = the nose moves 0.4 frame widths (0.4*480 = 192 px > 0.25 * 509 px body length)
    # for one frame and back: 2 steps over the limit per excursion
    for j in range(jumps):
        k = 20 + 4 * j
        poses[k] = _pose(k, nose_x=0.9)
    return poses


def _meta(poses):
    det = sum(p.landmarks is not None for p in poses)
    return {"fps": 30.0, "width": W, "height": H, "frame_step": 1, "frames_processed": len(poses),
            "frames_with_pose": det, "detection_rate": round(100 * det / len(poses), 1)}


class StubRTM:
    """Stand-in for _extract_with_rtmpose: returns full-detection poses on the same frames."""

    def __init__(self, poses=None, exc=None):
        self.calls = 0
        self.poses, self.exc = poses, exc

    def __call__(self, video_path, frame_step, scale, verbose=True):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return (self.poses if self.poses is not None else _clip(40)), {}


@pytest.fixture(autouse=True)
def _no_video(monkeypatch):
    monkeypatch.setattr(pe, "_decoded_frame_size", lambda path: (W, H))


def _run(monkeypatch, mode, poses, stub):
    monkeypatch.setattr(pe, "POSE_FALLBACK", mode)
    return pe._apply_pose_fallback("clip.mp4", poses, _meta(poses), verbose=False, rtm_extract=stub)


def test_default_off_never_calls_rtmpose(monkeypatch):
    stub = StubRTM()
    poses = _clip(missing=30)  # 25% detection: would trigger if the fallback were on
    out, meta = _run(monkeypatch, "off", poses, stub)
    assert stub.calls == 0
    assert out is poses
    assert meta["pose_backend"] == "mediapipe"
    assert meta["pose_fallback"]["mode"] == "off"
    assert meta["pose_fallback"]["triggered"] is False
    assert meta["detection_rate"] == 25.0


def test_config_default_is_off():
    import config
    if "POSE_FALLBACK" not in os.environ:
        assert config.POSE_FALLBACK == "off"
    assert config.POSE_FALLBACK_MIN_DETECTION == pytest.approx(0.90) or "POSE_FALLBACK_MIN_DETECTION" in os.environ


def test_good_clip_not_triggered(monkeypatch):
    stub = StubRTM()
    poses = _clip(missing=2)  # 95% detection, no jumps
    out, meta = _run(monkeypatch, "rtmpose", poses, stub)
    assert stub.calls == 0
    assert out is poses
    assert meta["pose_backend"] == "mediapipe"
    assert meta["pose_fallback"]["reason"] == "not triggered"
    assert meta["pose_fallback"]["nose_jumps_mediapipe"] == 0


def test_low_detection_triggers_rtmpose(monkeypatch):
    stub = StubRTM()
    poses = _clip(missing=16)  # 60% detection
    out, meta = _run(monkeypatch, "rtmpose", poses, stub)
    assert stub.calls == 1
    assert meta["pose_backend"] == "rtmpose_fallback"
    fb = meta["pose_fallback"]
    assert fb["triggered"] is True
    assert "detection_rate 60.0% below 90%" in fb["reason"]
    assert fb["detection_rate_mediapipe"] == 60.0
    assert fb["detection_rate_rtmpose"] == 100.0
    assert meta["detection_rate"] == 100.0 and meta["frames_with_pose"] == 40
    assert [p.frame_idx for p in out] == [p.frame_idx for p in poses]


def test_threshold_boundary(monkeypatch):
    stub = StubRTM()
    _, meta = _run(monkeypatch, "rtmpose", _clip(missing=4), stub)  # exactly 90%: not below
    assert stub.calls == 0 and meta["pose_backend"] == "mediapipe"
    _, meta = _run(monkeypatch, "rtmpose", _clip(missing=5), stub)  # 87.5%
    assert stub.calls == 1 and meta["pose_backend"] == "rtmpose_fallback"


def test_jump_rule(monkeypatch):
    poses = _clip(jumps=2)
    jumps, body_length = pe._count_nose_jumps(poses, W, H)
    assert body_length == pytest.approx(0.6 * H, abs=0.5)
    assert jumps == 4  # out and back, twice

    # one jump step only (person switch that stays): does not fire, the rule needs more than one
    one = _clip()
    for k in range(30, 40):
        one[k] = _pose(k, nose_x=0.9)
    assert pe._count_nose_jumps(one, W, H)[0] == 1
    stub = StubRTM()
    _, meta = _run(monkeypatch, "rtmpose", one, stub)
    assert stub.calls == 0 and meta["pose_fallback"]["reason"] == "not triggered"

    stub = StubRTM()
    _, meta = _run(monkeypatch, "rtmpose", poses, stub)
    assert stub.calls == 1
    assert meta["pose_backend"] == "rtmpose_fallback"
    assert "4 nose jumps" in meta["pose_fallback"]["reason"]
    assert meta["pose_fallback"]["nose_jumps_rtmpose"] == 0


def test_gaps_and_low_visibility_are_not_jumps():
    poses = _clip()
    poses[20] = FramePose(frame_idx=20, timestamp_s=0.667, landmarks=None)
    poses[21] = _pose(21, nose_x=0.9)  # far away, but after a gap
    poses[30] = _pose(30, nose_x=0.9, vis=0.1)  # low visibility nose is dropped
    assert pe._count_nose_jumps(poses, W, H)[0] == 1  # only 21 -> 22 (back to centre)


def test_rtmlib_missing_keeps_mediapipe(monkeypatch):
    poses = _clip(missing=20)
    stub = StubRTM(exc=ModuleNotFoundError("No module named 'rtmlib'"))
    out, meta = _run(monkeypatch, "rtmpose", poses, stub)
    assert out is poses
    assert meta["pose_backend"] == "mediapipe"
    assert meta["detection_rate"] == 50.0
    assert meta["pose_fallback"]["triggered"] is True
    assert "not installed" in meta["pose_fallback"]["error"]


def test_real_import_failure_is_graceful(monkeypatch):
    """The real extractor with rtmlib unimportable: ImportError is caught, server keeps going."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rtmlib" or name.startswith("rtmlib."):
            raise ModuleNotFoundError("No module named 'rtmlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(pe, "_RTM_MODELS", {})
    monkeypatch.setattr(pe, "POSE_FALLBACK", "rtmpose")
    poses = _clip(missing=20)
    out, meta = pe._apply_pose_fallback("clip.mp4", poses, _meta(poses), verbose=False)
    assert out is poses
    assert meta["pose_backend"] == "mediapipe"
    assert "not installed" in meta["pose_fallback"]["error"]


def test_rtmpose_runtime_failure_keeps_mediapipe(monkeypatch):
    poses = _clip(missing=20)
    out, meta = _run(monkeypatch, "rtmpose", poses, StubRTM(exc=RuntimeError("onnx session failed")))
    assert out is poses
    assert meta["pose_backend"] == "mediapipe"
    assert "RuntimeError: onnx session failed" in meta["pose_fallback"]["error"]


def test_rtmpose_worse_keeps_mediapipe(monkeypatch):
    poses = _clip(missing=10)  # 75%
    worse = _clip(missing=20)  # 50%
    out, meta = _run(monkeypatch, "rtmpose", poses, StubRTM(poses=worse))
    assert out is poses
    assert meta["pose_backend"] == "mediapipe"
    assert meta["pose_fallback"]["detection_rate_rtmpose"] == 50.0
    assert "not better" in meta["pose_fallback"]["error"]


def test_forced_mode_always_uses_rtmpose(monkeypatch):
    poses = _clip()
    stub = StubRTM()
    out, meta = _run(monkeypatch, "rtmpose_always", poses, stub)
    assert stub.calls == 1
    assert out is not poses
    assert meta["pose_backend"] == "rtmpose_fallback"
    assert meta["pose_fallback"]["reason"].startswith("forced")
    assert "rtmpose_ms_per_frame" in meta["pose_fallback"]


def test_unknown_mode_keeps_mediapipe(monkeypatch):
    stub = StubRTM()
    out, meta = _run(monkeypatch, "yolo", _clip(missing=30), stub)
    assert stub.calls == 0
    assert meta["pose_backend"] == "mediapipe"
    assert "unknown POSE_FALLBACK" in meta["pose_fallback"]["reason"]


def test_rtm_landmarks_mapping_and_proxies():
    k = np.zeros((26, 2))
    s = np.full(26, 1.3)  # scores above 1 are clipped
    k[0] = (240, 100)    # nose
    k[9] = (100, 400)    # left wrist (Halpe 9 -> MediaPipe 15)
    k[20] = (200, 800)   # left big toe (Halpe 20 -> MediaPipe 31)
    lms = pe._rtm_landmarks(k, s, W, H)
    assert len(lms) == 33 and all(np.isfinite([lm.x, lm.y]).all() for lm in lms)
    assert (lms[0].x, lms[0].y, lms[0].z, lms[0].visibility) == (0.5, pytest.approx(100 / H), 0.0, 1.0)
    assert lms[15].x == pytest.approx(100 / W) and lms[31].y == pytest.approx(800 / H)
    # hand points without an RTMPose equivalent sit on the wrist, invisible
    assert lms[19].x == lms[15].x and lms[19].visibility == 0.0
    assert lms[9].visibility == 0.0


def test_pick_person_tallest_then_nearest():
    short = np.column_stack([np.full(26, 100.0), np.linspace(400, 500, 26)])
    tall = np.column_stack([np.full(26, 300.0), np.linspace(100, 700, 26)])
    kps = np.stack([short, tall])
    scores = np.ones((2, 26))
    assert pe._pick_person(kps, scores, None) == 1
    assert pe._pick_person(kps, scores, short + 5) == 0
    assert pe._pick_person(np.zeros((0, 26, 2)), np.zeros((0, 26)), None) is None
