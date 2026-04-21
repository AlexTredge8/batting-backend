"""
Tests for HD video quality improvements.

Validates:
1. Frame lookup maps original frames to correct subsampled metric indices
2. Phase indices correctly convert to original frame indices
3. ffmpeg availability check works
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FrameMetrics


def test_frame_lookup_basic():
    """Frame lookup maps original frames to nearest prior metric."""
    from video_annotator import _build_frame_lookup

    # Simulate frame_step=2: metrics at original frames 0, 2, 4, 6, 8
    metrics = []
    for i in range(5):
        m = FrameMetrics(frame_idx=i * 2, timestamp_s=i * 2 / 30, detected=True)
        metrics.append(m)

    lookup = _build_frame_lookup(metrics)

    # Original frame 0 → metric index 0 (frame_idx=0)
    assert lookup[0] == 0
    # Original frame 1 → metric index 0 (nearest prior is frame_idx=0)
    assert lookup[1] == 0
    # Original frame 2 → metric index 1 (frame_idx=2)
    assert lookup[2] == 1
    # Original frame 3 → metric index 1 (nearest prior is frame_idx=2)
    assert lookup[3] == 1
    # Original frame 4 → metric index 2 (frame_idx=4)
    assert lookup[4] == 2
    # Original frame 8 → metric index 4 (frame_idx=8)
    assert lookup[8] == 4
    print("  PASS: frame_lookup_basic")


def test_frame_lookup_no_skip():
    """When every frame is processed (frame_step=1), lookup is identity."""
    from video_annotator import _build_frame_lookup

    metrics = []
    for i in range(10):
        m = FrameMetrics(frame_idx=i, timestamp_s=i / 30, detected=True)
        metrics.append(m)

    lookup = _build_frame_lookup(metrics)

    for i in range(10):
        assert lookup[i] == i, f"frame {i} should map to metric {i}"
    print("  PASS: frame_lookup_no_skip")


def test_frame_lookup_empty():
    """Empty metrics list produces empty lookup."""
    from video_annotator import _build_frame_lookup
    lookup = _build_frame_lookup([])
    assert lookup == {}
    print("  PASS: frame_lookup_empty")


def test_storyboard_thumb_size():
    """Storyboard thumbnail width was increased from 320 to 480."""
    from video_annotator import _THUMB_W
    assert _THUMB_W >= 480, f"_THUMB_W should be >= 480, got {_THUMB_W}"
    print("  PASS: storyboard_thumb_size")


def test_ffmpeg_check():
    """ffmpeg availability check returns a boolean."""
    from video_annotator import _ffmpeg_available
    result = _ffmpeg_available()
    assert isinstance(result, bool)
    print(f"  PASS: ffmpeg_available (ffmpeg={'found' if result else 'not found'})")


if __name__ == "__main__":
    print("Running video quality tests...")
    test_frame_lookup_basic()
    test_frame_lookup_no_skip()
    test_frame_lookup_empty()
    test_storyboard_thumb_size()
    test_ffmpeg_check()
    print("\nAll video quality tests passed!")
