"""Feet-based handedness in the live pipeline (run_analysis._handedness_from_feet)."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_analysis import _capture_checks, _handedness_from_feet  # noqa: E402


def _frames(toe_dx: float, n: int = 20, vis: float = 0.9):
    out = []
    for _ in range(n):
        lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=vis) for _ in range(33)]
        for heel, toe in ((29, 31), (30, 32)):
            lm[heel] = SimpleNamespace(x=0.5, y=0.9, z=0.0, visibility=vis)
            lm[toe] = SimpleNamespace(x=0.5 + toe_dx, y=0.9, z=0.0, visibility=vis)
        out.append(SimpleNamespace(landmarks=lm))
    return out


def test_toes_camera_left_is_right_handed():
    assert _handedness_from_feet(_frames(-0.05))["handedness"] == "right"


def test_toes_camera_right_is_left_handed_geometry():
    assert _handedness_from_feet(_frames(0.05))["handedness"] == "left"


def test_unsure_when_feet_hidden_or_flat():
    assert _handedness_from_feet(_frames(-0.05, vis=0.2))["handedness"] is None
    assert _handedness_from_feet(_frames(0.002))["handedness"] is None
    assert _handedness_from_feet([SimpleNamespace(landmarks=None)] * 10)["handedness"] is None


def test_disagreement_becomes_a_plain_warning():
    meta = {"duration_s": 4.0, "handedness_check": {"feet": "left", "used": "right", "disagrees": True}}
    _, warnings = _capture_checks([], SimpleNamespace(setup_end=0), meta)
    assert any("mirrored" in w for w in warnings)
