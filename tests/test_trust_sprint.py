"""
Focused tests for the trust sprint changes.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FrameMetrics


def _make_metric(i: int, vy: float = 0.0, speed: float = 0.0, elbow: float = 120.0, wrist_h: float = 0.0) -> FrameMetrics:
    return FrameMetrics(
        frame_idx=i,
        timestamp_s=i / 30,
        detected=True,
        wrist_velocity_y=vy,
        wrist_speed=speed,
        front_elbow_angle=elbow,
        wrist_height=wrist_h,
    )


def test_contact_consensus_high_confidence():
    from phase_detector import _resolve_contact_consensus

    metrics = [_make_metric(i) for i in range(20)]
    hands_peak = 4

    # Candidate A via sign reversal around frame 9
    velocity_map = {
        6: 0.01, 7: 0.02, 8: 0.03, 9: 0.02, 10: -0.01, 11: -0.02,
    }
    for idx, vy in velocity_map.items():
        metrics[idx].wrist_velocity_y = vy

    # Candidate B via max extension at frame 9
    for idx in range(6, 18):
        metrics[idx].front_elbow_angle = 120.0
    metrics[9].front_elbow_angle = 168.0

    # Candidate C via peak speed at 8 then decel minimum at 10
    for idx in range(6, 18):
        metrics[idx].wrist_speed = 0.4
    metrics[8].wrist_speed = 1.0
    metrics[10].wrist_speed = 0.05

    contact, confidence, diagnostics = _resolve_contact_consensus(metrics, hands_peak)
    assert contact == 9
    assert confidence == "high"
    assert diagnostics["candidates"] == [9, 9, 10]


def test_contact_consensus_low_confidence():
    from phase_detector import _resolve_contact_consensus

    metrics = [_make_metric(i) for i in range(30)]
    hands_peak = 4

    # Candidate A around frame 12
    velocity_map = {
        6: 0.01, 7: 0.02, 8: 0.03, 9: 0.04, 10: 0.05, 11: 0.03, 12: 0.02, 13: -0.01,
    }
    for idx, vy in velocity_map.items():
        metrics[idx].wrist_velocity_y = vy

    # Signal B earliest in window, signal C latest in window -> large spread
    for idx in range(6, 21):
        metrics[idx].front_elbow_angle = 120.0
        metrics[idx].wrist_speed = 0.5
    metrics[6].front_elbow_angle = 175.0
    metrics[10].wrist_speed = 1.2
    metrics[20].wrist_speed = 0.01

    contact, confidence, diagnostics = _resolve_contact_consensus(metrics, hands_peak)
    assert contact == 13
    assert confidence == "low"
    assert diagnostics["span"] > 6


def test_processing_settings_local_and_railway():
    from pose_extractor import _processing_settings

    local_step, local_scale, _ = _processing_settings(30.0, 1280, local_mode=True)
    railway_step, railway_scale, _ = _processing_settings(30.0, 1280, local_mode=False)

    assert local_step == 1
    assert local_scale == 1.0
    assert railway_step == 2
    assert railway_scale < 1.0


def test_batch_analyse_writes_csv():
    import batch_analyse

    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        (base / "clip1.mp4").write_bytes(b"video")
        (base / "clip2.mov").write_bytes(b"video")

        original = batch_analyse.run_full_analysis

        def fake_run_full_analysis(video_path: str, output_dir: str = None):
            return {
                "battingiq_score": 77,
                "pillars": {
                    "access": {"score": 18, "faults": [{"rule_id": "A1", "deduction": 4}]},
                    "tracking": {"score": 20, "faults": []},
                    "stability": {"score": 19, "faults": [{"rule_id": "S1", "deduction": 6}]},
                    "flow": {"score": 20, "faults": []},
                },
                "phases": {"contact": {"frame": 42, "confidence": "medium"}},
            }

        batch_analyse.run_full_analysis = fake_run_full_analysis
        argv = sys.argv
        try:
            sys.argv = ["batch_analyse.py", str(base)]
            exit_code = batch_analyse.main()
        finally:
            batch_analyse.run_full_analysis = original
            sys.argv = argv

        csv_path = base / "batch_results.csv"
        assert exit_code == 0
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "clip1.mp4" in content
        assert "contact_confidence" in content


if __name__ == "__main__":
    test_contact_consensus_high_confidence()
    test_contact_consensus_low_confidence()
    test_processing_settings_local_and_railway()
    test_batch_analyse_writes_csv()
    print("All trust sprint tests passed!")
