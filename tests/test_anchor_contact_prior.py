"""Contact-anchored key frames (ANCHOR_MODE=contact_prior)."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchor_contact_prior import GAPS_MS, contact_prior_anchors  # noqa: E402


def _metrics(n=200, still=(40, 60)):
    out = []
    for i in range(n):
        moving = not (still[0] <= i <= still[1])
        out.append(SimpleNamespace(frame_idx=i, detected=True, head_x=0.5 + (0.01 * (i % 2) if moving else 0),
                                   head_y=0.3, head_offset=0.0))
    return out


def test_gaps_from_contact_at_30fps():
    a = contact_prior_anchors(_metrics(), 100, 30.0)
    assert a["front_foot_down_frame"] == 100 + round(GAPS_MS["front_foot_down_frame"] * 30 / 1000)
    assert a["follow_through_frame"] == 100 + round(GAPS_MS["follow_through_frame"] * 30 / 1000)
    assert a["hands_start_up_frame"] >= a["setup_frame"]


def test_setup_moves_to_the_stillest_window():
    a = contact_prior_anchors(_metrics(still=(60, 72)), 100, 30.0)  # prior set-up is frame 71
    assert 60 <= a["setup_frame"] <= 72


def test_clamped_to_the_clip():
    a = contact_prior_anchors(_metrics(n=40), 5, 30.0)
    assert all(0 <= v <= 39 for v in a.values())
