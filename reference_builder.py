"""
BattingIQ Phase 2 — Reference Baseline Builder
=================================================
Processes the gold standard video to produce reference/reference_baseline.json.
This baseline is used by coaching rules to scale deductions relative to the
ideal technique.

Run standalone:
    python reference_builder.py test_batting.mov
"""

import json
import sys
import numpy as np
from pathlib import Path

from pose_extractor import extract_poses
from metrics_calculator import calculate_metrics
from phase_detector import detect_phases, print_phase_summary
from models import BattingPhase
from config import REFERENCE_BASELINE_PATH, FOLLOW_THROUGH_ANALYSIS_FRAMES


def _avg(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _frames_for_phase(labels: list, phase: BattingPhase) -> list[int]:
    return [i for i, l in enumerate(labels) if l == phase]


def build_reference_baseline(video_path: str, output_path: str = None) -> dict:
    """
    Process the gold standard video and write reference_baseline.json.
    Returns the baseline dict.
    """
    print(f"\nBuilding reference baseline from: {video_path}")

    frame_poses, meta = extract_poses(video_path, verbose=True)
    fps = meta["fps"]
    metrics = calculate_metrics(frame_poses, fps)
    phases  = detect_phases(metrics, fps, video_path=video_path)
    print_phase_summary(phases, fps)

    labels = phases.phase_labels
    n      = len(metrics)

    def get_m(idx): return metrics[idx] if 0 <= idx < n else metrics[-1]

    # ---- Per-phase metric collections ----

    setup_frames = _frames_for_phase(labels, BattingPhase.SETUP)
    backlift_frames = _frames_for_phase(labels, BattingPhase.BACKLIFT_STARTS)
    contact_idx = phases.contact
    contact_window = list(range(
        max(0, contact_idx - 2),
        min(n, contact_idx + 3)
    ))
    ft_frames = list(range(
        min(phases.follow_through_start, n - 1),
        min(phases.follow_through_start + FOLLOW_THROUGH_ANALYSIS_FRAMES, n)
    ))

    def metric_at_window(frames: list[int], attr: str) -> list[float]:
        return [getattr(metrics[i], attr) for i in frames if 0 <= i < n]

    # Setup baseline
    setup_wrist_height_mean = _avg(metric_at_window(setup_frames, "wrist_height"))
    setup_stance_width_mean = _avg(metric_at_window(setup_frames, "stance_width"))
    setup_shoulder_openness_mean = _avg(metric_at_window(setup_frames, "shoulder_openness"))
    setup_hip_openness_mean = _avg(metric_at_window(setup_frames, "hip_openness"))
    setup_head_offset_mean = _avg(metric_at_window(setup_frames, "head_offset"))
    setup_hip_centre_x_mean = _avg(metric_at_window(setup_frames, "hip_centre_x"))

    # Hands Peak
    hp_m = get_m(phases.hands_peak)

    # Contact frame metrics
    contact_shoulder_openness = _avg(metric_at_window(contact_window, "shoulder_openness"))
    contact_hip_openness = _avg(metric_at_window(contact_window, "hip_openness"))
    contact_front_knee_angle = _avg(metric_at_window(contact_window, "front_knee_angle"))
    contact_back_knee_angle = _avg(metric_at_window(contact_window, "back_knee_angle"))
    contact_front_elbow_angle = _avg(metric_at_window(contact_window, "front_elbow_angle"))
    contact_back_elbow_angle = _avg(metric_at_window(contact_window, "back_elbow_angle"))
    contact_wrist_x_mid = _avg(metric_at_window(contact_window, "wrist_x_mid"))
    contact_hip_centre_x = _avg(metric_at_window(contact_window, "hip_centre_x"))
    contact_head_offset = _avg(metric_at_window(contact_window, "head_offset"))
    contact_torso_lean = _avg(metric_at_window(contact_window, "torso_lean"))
    contact_shoulder_hip_gap = _avg(metric_at_window(contact_window, "shoulder_hip_gap"))
    contact_wrist_height = _avg(metric_at_window(contact_window, "wrist_height"))

    # Compression window (A3): frames from contact until wrist stops going forward/down
    compression_frames = 0
    for i in range(contact_idx + 1, min(contact_idx + 15, n)):
        if metrics[i].wrist_velocity_y > -0.002:  # still descending or flat
            compression_frames += 1
        else:
            break
    compression_frames = max(compression_frames, 3)

    # Movement timing (F3)
    backlift_to_contact_frames = phases.backlift_to_contact_frames

    # Follow-through wrist height (F6)
    ft_wrist_heights = metric_at_window(ft_frames, "wrist_height")
    ft_wrist_min = float(min(ft_wrist_heights)) if ft_wrist_heights else 0.3
    ft_shoulder_y = _avg(metric_at_window(contact_window, "shoulder_mid_y"))

    # Wrist spread at peak vs contact (A1)
    peak_wrist_spread = abs(hp_m.wrist_x_left - hp_m.wrist_x_right)
    contact_wrist_spread = _avg([abs(metrics[i].wrist_x_left - metrics[i].wrist_x_right)
                                  for i in contact_window])

    # Stance width at FFD
    ffd_stance_width = get_m(phases.front_foot_down).stance_width

    baseline = {
        "video": meta["video_name"],
        "fps": fps,
        "total_frames": meta["total_frames"],
        "detection_rate": meta["detection_rate"],

        "phases": {
            "setup_end": phases.setup_end,
            "backlift_start": phases.backlift_start,
            "hands_peak": phases.hands_peak,
            "front_foot_down": phases.front_foot_down,
            "contact": phases.contact,
            "follow_through_start": phases.follow_through_start,
            "hands_peak_vs_ffd_diff": phases.hands_peak_vs_ffd_diff,
            "backlift_to_contact_frames": backlift_to_contact_frames,
        },

        "setup": {
            "wrist_height_mean": round(setup_wrist_height_mean, 5),
            "stance_width_mean": round(setup_stance_width_mean, 5),
            "shoulder_openness_mean": round(setup_shoulder_openness_mean, 3),
            "hip_openness_mean": round(setup_hip_openness_mean, 3),
            "head_offset_mean": round(setup_head_offset_mean, 5),
            "hip_centre_x_mean": round(setup_hip_centre_x_mean, 5),
        },

        "hands_peak": {
            "wrist_height": round(hp_m.wrist_height, 5),
            "wrist_x_mid": round(hp_m.wrist_x_mid, 5),
            "wrist_spread": round(peak_wrist_spread, 5),
            "shoulder_openness": round(hp_m.shoulder_openness, 3),
        },

        "contact": {
            "shoulder_openness": round(contact_shoulder_openness, 3),
            "hip_openness": round(contact_hip_openness, 3),
            "shoulder_hip_gap": round(contact_shoulder_hip_gap, 3),
            "front_knee_angle": round(contact_front_knee_angle, 3),
            "back_knee_angle": round(contact_back_knee_angle, 3),
            "front_elbow_angle": round(contact_front_elbow_angle, 3),
            "back_elbow_angle": round(contact_back_elbow_angle, 3),
            "wrist_x_mid": round(contact_wrist_x_mid, 5),
            "wrist_spread": round(contact_wrist_spread, 5),
            "wrist_height": round(contact_wrist_height, 5),
            "hip_centre_x": round(contact_hip_centre_x, 5),
            "head_offset": round(contact_head_offset, 5),
            "torso_lean": round(contact_torso_lean, 5),
            "hip_shift_from_setup": round(contact_hip_centre_x - setup_hip_centre_x_mean, 5),
        },

        "follow_through": {
            "wrist_min_y": round(ft_wrist_min, 5),
            "shoulder_y_at_contact": round(ft_shoulder_y, 5),
            "compression_frames": compression_frames,
        },

        "timing": {
            "backlift_to_contact_frames": backlift_to_contact_frames,
            "backlift_to_contact_ms": round(backlift_to_contact_frames / fps * 1000, 1),
        },
    }

    # Write to disk
    out_path = Path(output_path or REFERENCE_BASELINE_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nReference baseline saved → {out_path}")
    print(f"  Contact shoulder openness : {contact_shoulder_openness:.1f}°")
    print(f"  Contact front knee angle  : {contact_front_knee_angle:.1f}°")
    print(f"  Contact front elbow angle : {contact_front_elbow_angle:.1f}°")
    print(f"  Backlift→Contact duration : {backlift_to_contact_frames} frames")
    print(f"  Compression frames        : {compression_frames}")

    return baseline


def load_reference_baseline(path: str = None) -> dict:
    p = Path(path or REFERENCE_BASELINE_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"Reference baseline not found at {p}. "
            "Run: python reference_builder.py test_batting.mov"
        )
    with open(p) as f:
        return json.load(f)


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "test_batting.mov"
    build_reference_baseline(video)
