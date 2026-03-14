"""
BattingIQ Phase 2 — Pose Extractor
=====================================
Runs MediaPipe BlazePose on every frame of a video and returns
a list of FramePose objects with raw landmark data.
"""

import cv2
import mediapipe as mp
from pathlib import Path
from models import FramePose, RawLandmark

mp_pose = mp.solutions.pose


MAX_PROCESS_FPS = 15   # subsample to this rate — plenty for cricket phase detection
MAX_PROCESS_WIDTH = 640  # downscale wide frames before MediaPipe inference


def extract_poses(video_path: str, verbose: bool = True) -> tuple[list[FramePose], dict]:
    """
    Run BlazePose on every frame of *video_path*.

    Returns:
        (frame_poses, video_meta)
        frame_poses: list of FramePose, one per frame
        video_meta:  dict with fps, width, height, total_frames, duration_s
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Subsample: process at most MAX_PROCESS_FPS frames per second
    frame_step = max(1, round(fps / MAX_PROCESS_FPS))

    # Downscale factor for MediaPipe input (keeps aspect ratio)
    scale = min(1.0, MAX_PROCESS_WIDTH / width) if width > MAX_PROCESS_WIDTH else 1.0
    proc_w = int(width * scale)
    proc_h = int(height * scale)

    video_meta = {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_s": round(total_frames / fps, 3),
        "video_name": path.name,
    }

    if verbose:
        print(f"  Extracting poses: {path.name}")
        print(f"  {width}x{height} @ {fps:.0f}fps — {total_frames} frames ({video_meta['duration_s']:.1f}s)")
        print(f"  Processing: every {frame_step} frame(s), scaled to {proc_w}x{proc_h}")

    frame_poses: list[FramePose] = []
    detected_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
    ) as pose_model:

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            if scale < 1.0:
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_model.process(rgb)

            if results.pose_landmarks:
                detected_count += 1
                landmarks = [
                    RawLandmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=lm.visibility,
                    )
                    for lm in results.pose_landmarks.landmark
                ]
            else:
                landmarks = None

            frame_poses.append(FramePose(
                frame_idx=frame_idx,
                timestamp_s=round(frame_idx / fps, 4),
                landmarks=landmarks,
            ))
            frame_idx += 1

    cap.release()

    detection_rate = round(detected_count / total_frames * 100, 1) if total_frames else 0
    video_meta["frames_with_pose"] = detected_count
    video_meta["detection_rate"] = detection_rate

    if verbose:
        print(f"  Detection rate: {detection_rate}% ({detected_count}/{total_frames} frames)")

    return frame_poses, video_meta
