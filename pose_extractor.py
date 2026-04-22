"""
BattingIQ Phase 2 - Pose Extractor
===================================
Runs MediaPipe pose estimation on every processed frame of a video and returns
a list of FramePose objects with raw landmark data.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

from config import LOCAL_MODE
from models import FramePose, RawLandmark

mp_pose = mp.solutions.pose

MAX_PROCESS_FPS = 15
MAX_PROCESS_WIDTH = 640
LOCAL_MODEL_COMPLEXITY = 1
RAILWAY_MODEL_COMPLEXITY = 1

POSE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
POSE_LANDMARKER_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "pose_landmarker_heavy.task"


def _processing_settings(fps: float, width: int, local_mode: bool = LOCAL_MODE) -> tuple[int, float, int]:
    """Return (frame_step, scale, model_complexity) for the current runtime mode."""
    if local_mode:
        return 1, 1.0, LOCAL_MODEL_COMPLEXITY

    frame_step = max(1, round(fps / MAX_PROCESS_FPS))
    scale = min(1.0, MAX_PROCESS_WIDTH / width) if width > MAX_PROCESS_WIDTH else 1.0
    return frame_step, scale, RAILWAY_MODEL_COMPLEXITY


def _ensure_pose_landmarker_model() -> Path:
    """Return the bundled pose landmarker task file, downloading it if needed."""
    if POSE_LANDMARKER_MODEL_PATH.exists():
        return POSE_LANDMARKER_MODEL_PATH

    POSE_LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(POSE_LANDMARKER_MODEL_URL, POSE_LANDMARKER_MODEL_PATH)
    return POSE_LANDMARKER_MODEL_PATH


def _build_video_meta(path: Path, fps: float, width: int, height: int, total_frames: int, frame_step: int) -> dict:
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_s": round(total_frames / fps, 3),
        "video_name": path.name,
        "frame_step": frame_step,
        "local_mode": LOCAL_MODE,
    }


def _extract_with_tasks(video_path: str, verbose: bool = True) -> tuple[list[FramePose], dict]:
    """Extract poses using the Tasks CPU delegate path."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_step, scale, _ = _processing_settings(fps, width, local_mode=LOCAL_MODE)
    proc_w = int(width * scale)
    proc_h = int(height * scale)
    video_meta = _build_video_meta(path, fps, width, height, total_frames, frame_step)

    if verbose:
        print(f"  Extracting poses: {path.name}")
        print(f"  {width}x{height} @ {fps:.0f}fps - {total_frames} frames ({video_meta['duration_s']:.1f}s)")
        print(
            f"  Processing: every {frame_step} frame(s), scaled to {proc_w}x{proc_h}, "
            f"mode={'LOCAL' if LOCAL_MODE else 'RAILWAY'}"
        )

    model_path = _ensure_pose_landmarker_model()
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(model_path),
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    frame_poses: list[FramePose] = []
    detected_count = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
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
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.pose_landmarks:
                detected_count += 1
                landmarks = [
                    RawLandmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=getattr(lm, "visibility", 0.0),
                    )
                    for lm in results.pose_landmarks[0]
                ]
            else:
                landmarks = None

            frame_poses.append(
                FramePose(
                    frame_idx=frame_idx,
                    timestamp_s=round(frame_idx / fps, 4),
                    landmarks=landmarks,
                )
            )
            frame_idx += 1

    cap.release()

    frames_processed = len(frame_poses)
    detection_rate = round(detected_count / frames_processed * 100, 1) if frames_processed else 0
    video_meta["frames_with_pose"] = detected_count
    video_meta["frames_processed"] = frames_processed
    video_meta["detection_rate"] = detection_rate

    if verbose:
        print(f"  Detection rate: {detection_rate}% ({detected_count}/{frames_processed} processed frames)")

    return frame_poses, video_meta


def _extract_with_legacy_pose(video_path: str, verbose: bool = True) -> tuple[list[FramePose], dict]:
    """Fallback extractor using the legacy Solutions API."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_step, scale, model_complexity = _processing_settings(fps, width, local_mode=LOCAL_MODE)
    proc_w = int(width * scale)
    proc_h = int(height * scale)
    video_meta = _build_video_meta(path, fps, width, height, total_frames, frame_step)

    if verbose:
        print(f"  Extracting poses: {path.name}")
        print(f"  {width}x{height} @ {fps:.0f}fps - {total_frames} frames ({video_meta['duration_s']:.1f}s)")
        print(
            f"  Processing: every {frame_step} frame(s), scaled to {proc_w}x{proc_h}, "
            f"mode={'LOCAL' if LOCAL_MODE else 'RAILWAY'}"
        )

    frame_poses: list[FramePose] = []
    detected_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
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

            frame_poses.append(
                FramePose(
                    frame_idx=frame_idx,
                    timestamp_s=round(frame_idx / fps, 4),
                    landmarks=landmarks,
                )
            )
            frame_idx += 1

    cap.release()

    frames_processed = len(frame_poses)
    detection_rate = round(detected_count / frames_processed * 100, 1) if frames_processed else 0
    video_meta["frames_with_pose"] = detected_count
    video_meta["frames_processed"] = frames_processed
    video_meta["detection_rate"] = detection_rate

    if verbose:
        print(f"  Detection rate: {detection_rate}% ({detected_count}/{frames_processed} processed frames)")

    return frame_poses, video_meta


def extract_poses(video_path: str, verbose: bool = True) -> tuple[list[FramePose], dict]:
    """
    Run pose extraction on every frame of *video_path*.

    Returns:
        (frame_poses, video_meta)
        frame_poses: list of FramePose, one per frame
        video_meta:  dict with fps, width, height, total_frames, duration_s
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        return _extract_with_tasks(video_path, verbose=verbose)
    except Exception as tasks_exc:
        if verbose:
            print(f"  Warning: pose landmarker tasks path failed ({tasks_exc}); falling back to solutions API")

    return _extract_with_legacy_pose(video_path, verbose=verbose)
