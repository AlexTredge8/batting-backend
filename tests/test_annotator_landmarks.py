"""
The annotator must draw the landmarks the ANALYSIS used (no second pose model),
and must map every original frame to the nearest processed frame's landmarks.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FramePose, RawLandmark
from video_annotator import _build_landmark_lookup, _draw_pose_overlay


def _pose(frame_idx: int, x: float) -> FramePose:
    lms = [RawLandmark(x=x, y=0.5, z=0.0, visibility=0.99) for _ in range(33)]
    return FramePose(frame_idx=frame_idx, timestamp_s=frame_idx / 30, landmarks=lms)


def test_landmark_lookup_maps_to_nearest_prior_processed_frame():
    lookup = _build_landmark_lookup([_pose(0, 0.1), _pose(2, 0.2), _pose(4, 0.4)])
    assert lookup[0][0].x == 0.1
    assert lookup[1][0].x == 0.1      # between processed frames → prior frame
    assert lookup[2][0].x == 0.2
    assert lookup[3][0].x == 0.2
    assert lookup[4][0].x == 0.4
    assert lookup[10][0].x == 0.4     # past last processed frame → last frame


def test_landmark_lookup_keeps_undetected_frames_as_none():
    undetected = FramePose(frame_idx=1, timestamp_s=1 / 30, landmarks=None)
    lookup = _build_landmark_lookup([_pose(0, 0.1), undetected])
    assert lookup[0] is not None
    assert lookup[1] is None


def test_empty_lookup_signals_fallback():
    assert _build_landmark_lookup(None) == {}
    assert _build_landmark_lookup([]) == {}


def test_draw_pose_overlay_accepts_raw_landmarks():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    _draw_pose_overlay(frame, _pose(0, 0.5).landmarks)
    assert int((frame != 0).any(axis=2).sum()) > 50, "skeleton/markers should have been drawn"
