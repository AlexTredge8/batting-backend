"""Camera roll from the background (capture_geometry, 16 Nov 2026), on drawn pictures (no video needed)."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capture_geometry as cg  # noqa: E402

W, H = 480, 848


def _picture(roll=0.0, n_vertical=8, horizontals=(), batter_lean=None, batter_box=(200, 250, 300, 600)):
    """Light background with dark upright lines (net poles, door edges), the whole picture rolled by `roll`
    degrees (positive = the top of an upright line turns camera-right). horizontals: angles (degrees,
    positive = right end lower) of extra near-horizontal lines. batter_lean: draw a thick leaning bar
    (the batter) inside batter_box at that angle from vertical."""
    img = np.full((H, W, 3), 200, np.uint8)
    xs = np.linspace(30, W - 30, n_vertical) if n_vertical else []
    for x in xs:
        if batter_box[0] - 10 <= x <= batter_box[2] + 10:
            continue
        p = cg.rotate_points(np.array([[x, 120.0], [x, 760.0]]), roll, (W / 2, H / 2))
        cv2.line(img, tuple(int(round(v)) for v in p[0]), tuple(int(round(v)) for v in p[1]), (40, 40, 40), 3, cv2.LINE_AA)
    for j, a in enumerate(horizontals):
        y = 650 + 30 * j
        t = np.radians(a)
        cv2.line(img, (40, int(y - 200 * np.tan(t))), (440, int(y + 200 * np.tan(t))), (40, 40, 40), 3, cv2.LINE_AA)
    if batter_lean is not None:
        x0, y0, x1, y1 = batter_box
        cx, t = (x0 + x1) / 2, np.radians(batter_lean)
        top = (int(cx + (y1 - y0) * np.sin(t) / 2), y0 + 10)
        bot = (int(cx - (y1 - y0) * np.sin(t) / 2), y1 - 10)
        cv2.line(img, top, bot, (0, 0, 0), 25)
    return img


def test_rotate_points_sign_matches_the_standing_axis_convention():
    p = cg.rotate_points(np.array([[0.0, 0.0], [0.0, -100.0]]), 3.0, (0.0, 0.0))  # an upright line, top at y=-100
    assert p[1, 0] > 0  # top turns camera-right for a positive roll
    back = cg.rotate_points(p, -3.0, (0.0, 0.0))
    assert np.allclose(back, [[0, 0], [0, -100]])


def test_level_camera_and_three_degree_rolls_are_recovered():
    for roll in (0.0, 3.0, -3.0, 1.5):
        r = cg.camera_roll_deg([_picture(roll)] * 5)
        assert r["confidence"] == "ok" and r["source"] == "vertical", (roll, r)
        assert abs(r["roll_deg"] - roll) < 0.3, (roll, r)
        assert r["n_lines"] >= 6  # the same lines in 5 frames are counted once


def test_the_batter_is_masked_out():
    """A lean batter in front of a level background: masked, he does not move the roll."""
    pics = [_picture(0.0, n_vertical=8, batter_lean=-12.0)] * 5
    box = (180, 230, 320, 620)
    masked = cg.camera_roll_deg(pics, boxes=[box] * 5)
    assert masked["confidence"] == "ok" and abs(masked["roll_deg"]) < 0.3 and masked["masked_frames"] == 5
    segs = cg.line_segments(pics[0], None)
    a, _, _ = cg._angles(segs, True)
    assert (np.abs(a + 12.0) < 2).any()  # unmasked, the batter's line is found
    assert not (np.abs(cg._angles(cg.line_segments(pics[0], box), True)[0] + 12.0) < 2).any()


def test_unknown_with_too_few_lines_or_lines_that_disagree():
    r = cg.camera_roll_deg([_picture(2.0, n_vertical=3)] * 5)
    assert r["confidence"] == "unknown" and r["roll_deg"] is None
    segs = [np.array([[50 + 60 * j, 700, 50 + 60 * j + 500 * np.sin(np.radians(a)), 700 - 500 * np.cos(np.radians(a))]
                      for j, a in enumerate((-8, -4, 0, 3, 7, 10, -10))])]
    r = cg.roll_from_segments(segs, W, H)
    assert r["n_lines"] == 7 and r["iqr_deg"] > 3 and r["confidence"] == "unknown" and r["roll_deg"] is None
    assert r["raw_roll_deg"] is not None  # kept for information
    assert cg.camera_roll_deg([])["source"] == "none"


def test_horizontal_lines_only_as_a_fallback_and_said_so():
    pic = _picture(0.0, n_vertical=0, horizontals=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0))
    r = cg.roll_from_segments([np.array([[40, 600 + 20 * j, 440, 600 + 20 * j + 400 * np.tan(np.radians(2.0))]
                                         for j in range(7)])], W, H)
    assert r["source"] == "horizontal_fallback" and r["confidence"] == "ok" and abs(r["roll_deg"] - 2.0) < 0.05
    assert cg.camera_roll_deg([pic] * 3)["source"] in ("horizontal_fallback", "none")
    # with enough upright lines the horizontals are not used
    both = cg.camera_roll_deg([_picture(0.0, horizontals=(5.0,) * 6)] * 3)
    assert both["source"] == "vertical" and abs(both["roll_deg"]) < 0.3


def test_frame_sample_and_batter_box():
    assert cg.roll_frame_indices(110) == [0, 11, 22, 32, 43]
    assert cg.roll_frame_indices(3) == [0]
    assert cg.roll_frame_indices(0) == []
    lm = np.zeros((33, 4))
    lm[:, 0], lm[:, 1], lm[:, 3] = np.linspace(0.4, 0.6, 33), np.linspace(0.3, 0.7, 33), 1.0
    x0, y0, x1, y1 = cg.batter_box(lm, W, H)
    assert (x0, x1) == (int(np.floor(0.4 * W - 0.2 * 0.2 * W)), int(np.ceil(0.6 * W + 0.2 * 0.2 * W)))
    assert y0 < 0.3 * H and y1 > 0.7 * H
    lm[:, 3] = 0.1
    assert cg.batter_box(lm, W, H) is None  # nothing visible enough


def test_boxes_ignored_when_the_frame_size_does_not_match():
    r = cg.camera_roll_deg([_picture(0.0)] * 2, boxes=[(0, 0, W, H)] * 2, box_frame_size=(H, W))
    assert "mask_skipped" in r and r["confidence"] == "ok"
