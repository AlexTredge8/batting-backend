"""
BattingIQ Phase 2 - Pose Extractor
===================================
Runs MediaPipe pose estimation on every processed frame of a video and returns
a list of FramePose objects with raw landmark data.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from config import (
    LOCAL_MODE,
    MEDIAPIPE_STATIC_IMAGE_MODE,
    POSE_FALLBACK,
    POSE_FALLBACK_JUMP_BL,
    POSE_FALLBACK_MAX_JUMPS,
    POSE_FALLBACK_MIN_DETECTION,
    POSE_FALLBACK_MODES,
    POSE_FALLBACK_RTM_MODE,
    POSE_RUNNING_MODE,
    PROCESSING_MODE,
)
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
        "effective_fps": round(fps / max(1, frame_step), 4),
        "local_mode": LOCAL_MODE,
        "processing_mode": PROCESSING_MODE,
        "pose_running_mode": POSE_RUNNING_MODE,
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
            f"mode={PROCESSING_MODE}"
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
        running_mode=RunningMode.IMAGE if POSE_RUNNING_MODE == "image" else RunningMode.VIDEO,
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
            if POSE_RUNNING_MODE == "image":
                results = landmarker.detect(mp_image)
            else:
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
            f"mode={PROCESSING_MODE}"
        )

    frame_poses: list[FramePose] = []
    detected_count = 0

    with mp_pose.Pose(
        static_image_mode=MEDIAPIPE_STATIC_IMAGE_MODE,
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


def _extract_with_mediapipe(video_path: str, verbose: bool = True) -> tuple[list[FramePose], dict]:
    try:
        return _extract_with_tasks(video_path, verbose=verbose)
    except Exception as tasks_exc:
        if verbose:
            print(f"  Warning: pose landmarker tasks path failed ({tasks_exc}); falling back to solutions API")

    return _extract_with_legacy_pose(video_path, verbose=verbose)


# ---------------------------------------------------------------------------
# RTMPose fallback (POSE_FALLBACK; default off). See docs/phase5/rtmpose_fallback.md.
# ---------------------------------------------------------------------------

# Halpe-26 (RTMPose body+feet) -> MediaPipe Pose 33 slots. Both use the person's own left/right.
# Same mapping as phase5_pose_compare.py, which produced the comparison this fallback rests on.
HALPE_TO_MP = {0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24,
               13: 25, 14: 26, 15: 27, 16: 28, 24: 29, 25: 30, 20: 31, 21: 32}
# MediaPipe slots RTMPose has no point for: filled with a neighbouring point at visibility 0, so
# every FramePose still has 33 finite landmarks and visibility-gated code (the annotator) skips them.
MP_PROXY_SLOTS = {1: 2, 3: 2, 4: 5, 6: 5, 9: 0, 10: 0, 17: 15, 19: 15, 21: 15, 18: 16, 20: 16, 22: 16}
_NOSE, _LANK, _RANK = 0, 27, 28
_JUMP_MIN_VISIBILITY = 0.2

_RTM_MODELS: dict[str, object] = {}
_RTM_LOCK = threading.Lock()


def _detection_fraction(frame_poses: list[FramePose]) -> float:
    return sum(fp.landmarks is not None for fp in frame_poses) / len(frame_poses) if frame_poses else 0.0


def _count_nose_jumps(frame_poses: list[FramePose], width: float, height: float,
                      jump_bl: float = None) -> tuple[int, float | None]:
    """Frames where the nose moves more than *jump_bl* body lengths from the previous frame.

    Body length = median (ankle height - nose height) in pixels. Points under 0.2 visibility and
    frames without a pose are dropped, so a gap never counts as a jump (as in phase5_pose_compare).
    """
    jump_bl = POSE_FALLBACK_JUMP_BL if jump_bl is None else jump_bl
    n = len(frame_poses)
    nose = np.full((n, 2), np.nan)
    ankle_y = np.full(n, np.nan)
    for i, fp in enumerate(frame_poses):
        lms = fp.landmarks
        if not lms or len(lms) <= _RANK:
            continue
        if lms[_NOSE].visibility >= _JUMP_MIN_VISIBILITY:
            nose[i] = (lms[_NOSE].x * width, lms[_NOSE].y * height)
        ys = [lms[k].y * height for k in (_LANK, _RANK) if lms[k].visibility >= _JUMP_MIN_VISIBILITY]
        if ys:
            ankle_y[i] = float(np.mean(ys))
    lengths = ankle_y - nose[:, 1]
    if n < 2 or not np.isfinite(lengths).any():
        return 0, None
    body_length = float(np.nanmedian(lengths))
    if not np.isfinite(body_length) or body_length <= 0:
        return 0, None
    steps = np.linalg.norm(np.diff(nose, axis=0), axis=1)
    with np.errstate(invalid="ignore"):
        jumps = int(np.nansum(steps > jump_bl * body_length))
    return jumps, round(body_length, 1)


def _fallback_reasons(detection_fraction: float, nose_jumps: int) -> list[str]:
    reasons = []
    if detection_fraction < POSE_FALLBACK_MIN_DETECTION:
        reasons.append(
            f"detection_rate {detection_fraction:.1%} below {POSE_FALLBACK_MIN_DETECTION:.0%}"
        )
    if nose_jumps > POSE_FALLBACK_MAX_JUMPS:
        reasons.append(
            f"{nose_jumps} nose jumps over {POSE_FALLBACK_JUMP_BL:.0%} of body length "
            f"(more than {POSE_FALLBACK_MAX_JUMPS})"
        )
    return reasons


def _decoded_frame_size(video_path: str) -> tuple[int, int] | None:
    """(width, height) of the first decoded frame, after the same orientation handling."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        ok, frame = cap.read()
        return (frame.shape[1], frame.shape[0]) if ok else None
    finally:
        cap.release()


def pose_fallback_status() -> dict:
    """What /diag reports about the fallback, without importing rtmlib or loading models."""
    import importlib.util
    import os

    hub = Path(os.path.expanduser(os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"),
                                                                        "rtmlib")))) / "hub" / "checkpoints"
    return {
        "mode": POSE_FALLBACK,
        "rtm_mode": POSE_FALLBACK_RTM_MODE,
        "min_detection": POSE_FALLBACK_MIN_DETECTION,
        "jump_body_lengths": POSE_FALLBACK_JUMP_BL,
        "max_jumps": POSE_FALLBACK_MAX_JUMPS,
        "rtmlib_installed": importlib.util.find_spec("rtmlib") is not None,
        "onnxruntime_installed": importlib.util.find_spec("onnxruntime") is not None,
        "model_files_cached": sorted(f.name for f in hub.glob("*.onnx")) if hub.is_dir() else [],
        "model_loaded": sorted(_RTM_MODELS),
    }


def _get_rtm_model(mode: str):
    """rtmlib BodyWithFeet, built once per process. Imported here so the server starts without it."""
    with _RTM_LOCK:
        model = _RTM_MODELS.get(mode)
        if model is None:
            from rtmlib import BodyWithFeet  # optional dependency (see requirements.txt)

            model = BodyWithFeet(mode=mode, backend="onnxruntime", device="cpu")
            _RTM_MODELS[mode] = model
        return model


def _pick_person(kps: np.ndarray, scores: np.ndarray, prev: np.ndarray | None) -> int | None:
    """The batter: tallest person in the first frame with a detection, then the nearest skeleton."""
    if len(kps) == 0:
        return None
    if prev is None:
        good = scores > 0.3
        heights = [np.ptp(k[g, 1]) if g.sum() >= 5 else 0 for k, g in zip(kps, good)]
        return int(np.argmax(heights))
    d = [np.nanmean(np.linalg.norm(k - prev, axis=1)) for k in kps]
    return int(np.argmin(d))


def _rtm_landmarks(k: np.ndarray, s: np.ndarray, w: int, h: int) -> list[RawLandmark]:
    lms: list[RawLandmark | None] = [None] * 33
    for hi, mi in HALPE_TO_MP.items():
        lms[mi] = RawLandmark(x=float(k[hi, 0]) / w, y=float(k[hi, 1]) / h, z=0.0,
                              visibility=float(np.clip(s[hi], 0.0, 1.0)))
    for mi, src in MP_PROXY_SLOTS.items():
        lms[mi] = RawLandmark(x=lms[src].x, y=lms[src].y, z=0.0, visibility=0.0)
    return lms


def _extract_with_rtmpose(video_path: str, frame_step: int, scale: float,
                          verbose: bool = True) -> tuple[list[FramePose], dict]:
    """RTMPose (rtmlib BodyWithFeet) on the same frames MediaPipe saw, in FramePose format."""
    model = _get_rtm_model(POSE_FALLBACK_RTM_MODE)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # same orientation as the MediaPipe pass
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_poses: list[FramePose] = []
    prev = None
    frame_idx = 0
    size = None
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue
            if scale < 1.0:
                frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
            size = (w, h)
            kps, scores = model(frame)
            kps = np.asarray(kps) if kps is not None else np.zeros((0, 26, 2))
            scores = np.asarray(scores) if scores is not None else np.zeros((0, 26))
            i = _pick_person(kps, scores, prev) if len(kps) else None
            landmarks = None
            if i is not None:
                k, s = kps[i], scores[i]
                jump = prev is not None and np.nanmedian(np.linalg.norm(k - prev, axis=1)) > 0.25 * h
                if not jump:
                    prev = k
                landmarks = _rtm_landmarks(k, s, w, h)
            frame_poses.append(FramePose(frame_idx=frame_idx, timestamp_s=round(frame_idx / fps, 4),
                                         landmarks=landmarks))
            frame_idx += 1
    finally:
        cap.release()
    return frame_poses, {"decoded_size": size}


def _align_to(reference: list[FramePose], other: list[FramePose]) -> list[FramePose]:
    """*other* re-indexed onto *reference*'s frame_idx list (missing frames have no pose)."""
    by_idx = {fp.frame_idx: fp for fp in other}
    return [by_idx.get(fp.frame_idx) or FramePose(frame_idx=fp.frame_idx, timestamp_s=fp.timestamp_s,
                                                  landmarks=None)
            for fp in reference]


def _apply_pose_fallback(video_path: str, frame_poses: list[FramePose], video_meta: dict,
                         verbose: bool = True,
                         rtm_extract=None) -> tuple[list[FramePose], dict]:
    """Decide whether to replace the MediaPipe poses with RTMPose; record the decision in video_meta."""
    mode = POSE_FALLBACK
    mp_frac = _detection_fraction(frame_poses)
    video_meta["pose_backend"] = "mediapipe"
    video_meta["pose_fallback"] = {
        "mode": mode,
        "triggered": False,
        "reason": None,
        "detection_rate_mediapipe": round(mp_frac * 100, 1),
        "detection_rate_rtmpose": None,
    }
    fb = video_meta["pose_fallback"]
    if mode == "off":
        return frame_poses, video_meta
    if mode not in POSE_FALLBACK_MODES:
        fb["reason"] = f"unknown POSE_FALLBACK value {mode!r}; MediaPipe kept"
        return frame_poses, video_meta

    size = _decoded_frame_size(video_path) or (video_meta.get("width") or 1, video_meta.get("height") or 1)
    mp_jumps, body_length = _count_nose_jumps(frame_poses, *size)
    fb.update({"nose_jumps_mediapipe": mp_jumps, "body_length_px": body_length,
               "thresholds": {"min_detection": POSE_FALLBACK_MIN_DETECTION,
                              "jump_body_lengths": POSE_FALLBACK_JUMP_BL,
                              "max_jumps": POSE_FALLBACK_MAX_JUMPS}})
    forced = mode == "rtmpose_always"
    reasons = ["forced (POSE_FALLBACK=rtmpose_always)"] if forced else _fallback_reasons(mp_frac, mp_jumps)
    if not reasons:
        fb["reason"] = "not triggered"
        return frame_poses, video_meta

    fb["triggered"] = True
    fb["reason"] = "; ".join(reasons)
    if verbose:
        print(f"  Pose fallback: {fb['reason']} -> re-extracting with RTMPose ({POSE_FALLBACK_RTM_MODE})")
    extract = rtm_extract or _extract_with_rtmpose
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        rtm_poses, _ = extract(video_path, video_meta.get("frame_step") or 1,
                               _processing_settings(video_meta.get("fps") or 30.0,
                                                    video_meta.get("width") or 0)[1],
                               verbose=verbose)
    except ImportError as exc:
        fb["error"] = f"rtmlib/onnxruntime not installed ({exc}); MediaPipe kept"
        if verbose:
            print(f"  Pose fallback skipped: {fb['error']}")
        return frame_poses, video_meta
    except Exception as exc:  # noqa: BLE001 - any RTMPose failure keeps the MediaPipe result
        fb["error"] = f"RTMPose failed ({type(exc).__name__}: {exc}); MediaPipe kept"
        if verbose:
            print(f"  Pose fallback skipped: {fb['error']}")
        return frame_poses, video_meta
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    rtm_poses = _align_to(frame_poses, rtm_poses)
    rtm_frac = _detection_fraction(rtm_poses)
    rtm_jumps, _ = _count_nose_jumps(rtm_poses, *size)
    n = max(1, len(rtm_poses))
    fb.update({
        "detection_rate_rtmpose": round(rtm_frac * 100, 1),
        "nose_jumps_rtmpose": rtm_jumps,
        "rtmpose_mode": POSE_FALLBACK_RTM_MODE,
        "rtmpose_seconds": round(wall, 2),
        "rtmpose_ms_per_frame": round(1000 * wall / n, 1),
        "rtmpose_cpu_ms_per_frame": round(1000 * cpu / n, 1),
    })
    if not forced and (rtm_frac < mp_frac or rtm_jumps > mp_jumps):
        fb["error"] = "RTMPose was not better (lower detection rate or more nose jumps); MediaPipe kept"
        if verbose:
            print(f"  Pose fallback not used: {fb['error']}")
        return frame_poses, video_meta

    detected = sum(fp.landmarks is not None for fp in rtm_poses)
    video_meta["pose_backend"] = "rtmpose_fallback"
    video_meta["frames_with_pose"] = detected
    video_meta["frames_processed"] = len(rtm_poses)
    video_meta["detection_rate"] = round(rtm_frac * 100, 1)
    if verbose:
        print(f"  Pose backend: RTMPose, detection rate {video_meta['detection_rate']}% "
              f"(MediaPipe {fb['detection_rate_mediapipe']}%), {fb['rtmpose_ms_per_frame']} ms/frame")
    return rtm_poses, video_meta


def extract_poses(video_path: str, verbose: bool = True) -> tuple[list[FramePose], dict]:
    """
    Run pose extraction on every frame of *video_path*.

    MediaPipe first; with POSE_FALLBACK=rtmpose a clip where MediaPipe lost the batter (low
    detection rate) or jumped between people is re-extracted with RTMPose. video_meta records
    pose_backend ("mediapipe" or "rtmpose_fallback") and pose_fallback (reason, both rates).

    Returns:
        (frame_poses, video_meta)
        frame_poses: list of FramePose, one per frame
        video_meta:  dict with fps, width, height, total_frames, duration_s
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    t0 = time.perf_counter()
    frame_poses, video_meta = _extract_with_mediapipe(video_path, verbose=verbose)
    video_meta["pose_seconds_mediapipe"] = round(time.perf_counter() - t0, 2)
    return _apply_pose_fallback(video_path, frame_poses, video_meta, verbose=verbose)
