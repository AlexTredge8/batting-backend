"""
End-to-end regression on the bundled reference clip (test_batting.mov).

Skipped unless BATTINGIQ_E2E=1 because it runs MediaPipe (~15-20s). Pins the
calibrated full-rate anchors so any change to extraction, phase detection or
processing mode that moves them is caught immediately.

    BATTINGIQ_E2E=1 python -m pytest tests/test_reference_video_e2e.py -q
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "test_batting.mov"

pytestmark = pytest.mark.skipif(
    os.environ.get("BATTINGIQ_E2E") != "1" or not VIDEO.exists(),
    reason="set BATTINGIQ_E2E=1 to run the MediaPipe end-to-end regression",
)

# Calibrated full-rate anchors for test_batting.mov (original video frames), measured
# 2026-09-04 with the mp.tasks heavy landmarker, frame_step=1, audio contact enabled.
EXPECTED_ANCHORS = {
    "setup_frame": 60,
    "hands_start_up_frame": 61,
    "front_foot_down_frame": 77,
    "hands_peak_frame": 73,
    "contact_frame": 81,
    "follow_through_frame": 109,
}
ANCHOR_TOLERANCE = 1  # frames


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    os.environ.pop("FAST_MODE", None)
    os.environ.pop("LOCAL_MODE", None)
    from run_analysis import run_full_analysis

    out = tmp_path_factory.mktemp("e2e")
    return run_full_analysis(str(VIDEO), output_dir=str(out))


def test_runs_full_rate_calibrated_path(report):
    md = report["metadata"]
    assert md["processing_mode"] == "full_rate_calibrated"
    assert md["frame_step"] == 1
    # container metadata may over-report frame count; decoded frames must be ~all of them
    assert md["frames_processed"] >= 0.8 * md["total_frames"]
    assert md["detection_rate"] >= 95.0


def test_anchors_match_calibrated_reference(report):
    anchors = {k: v["original_frame"] for k, v in report["metadata"]["anchor_frames"].items()}
    for key, expected in EXPECTED_ANCHORS.items():
        assert abs(anchors[key] - expected) <= ANCHOR_TOLERANCE, f"{key}: {anchors[key]} vs {expected}"
    # ordering sanity
    assert anchors["setup_frame"] < anchors["hands_start_up_frame"] < anchors["hands_peak_frame"] \
        < anchors["contact_frame"] < anchors["follow_through_frame"]


def test_reference_scores_excellent(report):
    assert report["battingiq_score"] >= 85
    assert report["score_band"] == "Excellent"


def test_media_and_quality_block_present(report):
    assert report["_annotated_video"] and Path(report["_annotated_video"]).stat().st_size > 10_000
    assert len(report["_storyboard_frames"]) == 6
    for fr in report["_storyboard_frames"]:
        assert Path(fr["path"]).exists()
    q = report["analysis_quality"]
    assert q["processing_mode"] == "full_rate_calibrated"
    assert q["contact_method"].startswith("audio"), "ffmpeg/audio must be available for the reference run"
    assert report["phases"]["contact"]["original_frame"] == report["metadata"]["anchor_frames"]["contact_frame"]["original_frame"]


def test_storyboard_keyframes_contract(report):
    keys = ("setup", "hands_start_up", "front_foot_down", "hands_peak", "contact", "follow_through")
    kf = report["storyboard_frames"]
    assert tuple(kf) == keys
    anchors = report["metadata"]["anchor_frames"]
    for key in keys:
        entry = kf[key]
        assert entry["available"] is True, key
        assert entry["image"].startswith("data:image/jpeg;base64,")
        assert len(entry["image"]) > 5_000
        assert isinstance(entry["frame"], int) and entry["frame"] > 0, key
        assert "_path" not in entry
    # contact keyframe is pinned to the resolved contact frame
    assert kf["contact"]["frame"] == anchors["contact_frame"]["original_frame"]
