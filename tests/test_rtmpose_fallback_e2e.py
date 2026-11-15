"""
Reference clip forced through RTMPose (POSE_FALLBACK=rtmpose_always), end to end.

Opt-in: needs the optional packages (rtmlib, onnxruntime) and ~40 s of CPU, so it runs only with
BATTINGIQ_E2E_RTMPOSE=1. Results on 15 Nov 2026 (docs/phase5/rtmpose_fallback.md): score 92
(same as MediaPipe), anchors 61/64/77/72/81/102.

    BATTINGIQ_E2E_RTMPOSE=1 python -m pytest tests/test_rtmpose_fallback_e2e.py -q
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "test_batting.mov"

pytestmark = pytest.mark.skipif(
    os.environ.get("BATTINGIQ_E2E_RTMPOSE") != "1" or not VIDEO.exists()
    or importlib.util.find_spec("rtmlib") is None,
    reason="set BATTINGIQ_E2E_RTMPOSE=1 (with rtmlib + onnxruntime installed) to run",
)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    import pose_extractor
    from run_analysis import run_full_analysis

    mp = pytest.MonkeyPatch()
    mp.setattr(pose_extractor, "POSE_FALLBACK", "rtmpose_always")
    try:
        yield run_full_analysis(str(VIDEO), output_dir=str(tmp_path_factory.mktemp("rtm")))
    finally:
        mp.undo()


def test_forced_rtmpose_backend_recorded(report):
    md = report["metadata"]
    assert md["pose_backend"] == "rtmpose_fallback"
    assert report["analysis_quality"]["pose_backend"] == "rtmpose_fallback"
    fb = md["pose_fallback"]
    assert fb["triggered"] and fb["reason"].startswith("forced")
    assert fb["detection_rate_mediapipe"] >= 95 and fb["detection_rate_rtmpose"] >= 95
    assert md["detection_rate"] == fb["detection_rate_rtmpose"]


def test_forced_rtmpose_keeps_reference_result(report):
    anchors = {k: v["original_frame"] for k, v in report["metadata"]["anchor_frames"].items()}
    assert abs(anchors["contact_frame"] - 81) <= 1
    assert anchors["setup_frame"] < anchors["hands_peak_frame"] < anchors["contact_frame"] \
        < anchors["follow_through_frame"]
    assert report["score_band"] == "Excellent"
    assert len(report["_storyboard_frames"]) == 6
