"""Tests for the setup-frame v2 detector."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FrameMetrics
from phase_detector import _find_setup_end_v1, _find_setup_end_v2


def _metric(
    idx: int,
    *,
    wrist_height: float,
    shoulder_mid_y: float,
    hip_mid_y: float,
    knee_mid_y: float,
    forward_weight: float,
) -> FrameMetrics:
    return FrameMetrics(
        frame_idx=idx,
        timestamp_s=idx / 30.0,
        detected=True,
        wrist_height=wrist_height,
        shoulder_mid_y=shoulder_mid_y,
        hip_mid_y=hip_mid_y,
        knee_mid_y=knee_mid_y,
        forward_weight=forward_weight,
    )


def _with_velocities(metrics: list[FrameMetrics]) -> list[FrameMetrics]:
    for idx in range(1, len(metrics)):
        curr = metrics[idx]
        prev = metrics[idx - 1]
        curr.shoulder_mid_vy = curr.shoulder_mid_y - prev.shoulder_mid_y
        curr.hip_mid_vy = curr.hip_mid_y - prev.hip_mid_y
        curr.knee_mid_vy = curr.knee_mid_y - prev.knee_mid_y
        curr.forward_weight_v = curr.forward_weight - prev.forward_weight
        curr.wrist_velocity_y = curr.wrist_height - prev.wrist_height
    return metrics


def _build_gradual_setup_metrics() -> list[FrameMetrics]:
    metrics: list[FrameMetrics] = []
    for idx in range(28):
        if idx < 5:
            metrics.append(
                _metric(
                    idx,
                    wrist_height=0.50,
                    shoulder_mid_y=0.300,
                    hip_mid_y=0.500,
                    knee_mid_y=0.700,
                    forward_weight=0.020,
                )
            )
            continue

        setup_progress = min(max(idx - 4, 0), 4)
        wrist_progress = min(max(idx - 10, 0), 4)
        metrics.append(
            _metric(
                idx,
                wrist_height=0.50 - (0.008 * wrist_progress),
                shoulder_mid_y=0.300 + (0.0020 * setup_progress),
                hip_mid_y=0.500 + (0.0025 * setup_progress),
                knee_mid_y=0.700 + (0.0035 * setup_progress),
                forward_weight=0.020 + (0.0025 * setup_progress),
            )
        )
    return _with_velocities(metrics)


def _build_single_signal_spike_metrics() -> list[FrameMetrics]:
    metrics: list[FrameMetrics] = []
    for idx in range(28):
        shoulder_mid_y = 0.300
        if 5 <= idx <= 7:
            shoulder_mid_y += 0.006
        wrist_height = 0.50 if idx < 12 else 0.47
        metrics.append(
            _metric(
                idx,
                wrist_height=wrist_height,
                shoulder_mid_y=shoulder_mid_y,
                hip_mid_y=0.500,
                knee_mid_y=0.700,
                forward_weight=0.020,
            )
        )
    return _with_velocities(metrics)


def test_setup_v2_detects_sustained_multi_landmark_change_before_wrist_threshold():
    metrics = _build_gradual_setup_metrics()

    setup_v1 = _find_setup_end_v1(metrics)
    setup_v2 = _find_setup_end_v2(metrics)

    assert setup_v2 <= 7
    assert setup_v2 < setup_v1


def test_setup_v2_does_not_fire_on_single_landmark_spike():
    metrics = _build_single_signal_spike_metrics()

    setup_v1 = _find_setup_end_v1(metrics)
    setup_v2 = _find_setup_end_v2(metrics)

    assert setup_v2 >= setup_v1
