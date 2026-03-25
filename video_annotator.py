"""
BattingIQ Phase 2 — Video Annotator
======================================
Produces an annotated video with:
  - Phase label in top-left corner
  - Live metrics panel (right side)
  - Traffic-light flash at the Contact window
  - BattingIQ watermark
  - Pillar score bars

Output uses H.264 encoding via ffmpeg for high quality web playback.
Falls back to mp4v if ffmpeg is unavailable.
"""

import cv2
import mediapipe as mp
import numpy as np
import subprocess
import tempfile
import shutil
from pathlib import Path

from models import BattingIQResult, BattingPhase, TrafficLight, FrameMetrics, PhaseResult

mp_pose = mp.solutions.pose

# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,   0,   0)
C_GREEN   = (50, 200, 50)
C_AMBER   = (30, 165, 255)   # orange in BGR
C_RED     = (50,  50, 230)
C_BLUE    = (220, 120, 40)
C_YELLOW  = (0,  220, 220)
C_CYAN    = (200, 200, 0)
C_DARK    = (20,  20,  20)

# Phase → colour
PHASE_COLOURS = {
    BattingPhase.SETUP:           C_WHITE,
    BattingPhase.BACKLIFT_STARTS: C_CYAN,
    BattingPhase.HANDS_PEAK:      C_YELLOW,
    BattingPhase.FRONT_FOOT_DOWN: C_GREEN,
    BattingPhase.CONTACT:         C_RED,
    BattingPhase.FOLLOW_THROUGH:  C_BLUE,
    BattingPhase.UNKNOWN:         C_WHITE,
}

PHASE_LABELS = {
    BattingPhase.SETUP:           "SETUP",
    BattingPhase.BACKLIFT_STARTS: "BACKLIFT",
    BattingPhase.HANDS_PEAK:      "HANDS PEAK",
    BattingPhase.FRONT_FOOT_DOWN: "FRONT FOOT DOWN",
    BattingPhase.CONTACT:         "CONTACT",
    BattingPhase.FOLLOW_THROUGH:  "FOLLOW-THROUGH",
    BattingPhase.UNKNOWN:         "",
}

STATUS_COLOURS = {
    TrafficLight.GREEN: C_GREEN,
    TrafficLight.AMBER: C_AMBER,
    TrafficLight.RED:   C_RED,
}

POSE_CONNECTION_COLOR = (80, 220, 80)
POSE_LANDMARK_COLOR = (245, 245, 245)
POSE_MARKER_OUTLINE = (0, 0, 0)
POSE_MARKER_FILL = C_YELLOW
POSE_VISIBILITY_THRESHOLD = 0.35
KEY_POSE_LANDMARKS = (
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
)


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def _build_pose_model(static_image_mode: bool = False):
    """Build the pose model used by the renderer."""
    return mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
    )


def _landmark_sequence(pose_landmarks):
    """Return a simple indexable landmark sequence from either pose API."""
    if pose_landmarks is None:
        return ()
    if hasattr(pose_landmarks, "landmark"):
        return pose_landmarks.landmark
    return pose_landmarks


def _detect_pose_landmarks(pose_model, frame):
    """Detect pose landmarks for one frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb_frame)
    return results.pose_landmarks


# ---------------------------------------------------------------------------
# Frame index mapping
# ---------------------------------------------------------------------------

def _build_frame_lookup(metrics: list[FrameMetrics]) -> dict:
    """
    Build a mapping from original video frame indices to metric list indices.

    Metrics are subsampled (e.g., every 2nd frame at 30fps → 15fps).
    Each FrameMetrics stores its original frame_idx. This function creates
    a dict so every original frame can find its nearest prior metric.

    Returns: {original_frame_idx: metric_list_index}
    """
    lookup = {}
    last_metric_idx = 0
    if not metrics:
        return lookup

    # Get all original frame indices that have metrics
    metric_orig_frames = [m.frame_idx for m in metrics]

    # For each original frame, find the nearest metric at or before it
    mi = 0
    max_orig = metric_orig_frames[-1] + 100  # cover some frames past last metric
    for orig_f in range(max_orig):
        # Advance metric index if next metric is at or before this frame
        while mi + 1 < len(metric_orig_frames) and metric_orig_frames[mi + 1] <= orig_f:
            mi += 1
        lookup[orig_f] = mi

    return lookup


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _text(frame, text, pos, scale=0.55, colour=C_WHITE, thickness=1, bold=False):
    t = thickness + (1 if bold else 0)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, t, cv2.LINE_AA)


def _panel_bg(frame, x, y, w, h, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), C_DARK, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _traffic_dot(frame, cx, cy, r, colour):
    cv2.circle(frame, (cx, cy), r, colour, -1)
    cv2.circle(frame, (cx, cy), r, C_WHITE, 1)


def _draw_pose_overlay(frame, pose_landmarks):
    """
    Draw the pose skeleton plus a few bold key joint markers.

    The extra joint circles make the BattingIQ overlay visibly read as an
    annotated biomechanics video rather than just a metrics HUD.
    """
    landmarks = _landmark_sequence(pose_landmarks)
    if not landmarks:
        return

    height, width = frame.shape[:2]
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx = getattr(connection, "start", None)
        end_idx = getattr(connection, "end", None)
        if start_idx is None or end_idx is None:
            start_idx, end_idx = connection

        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue

        start_lm = landmarks[start_idx]
        end_lm = landmarks[end_idx]
        if getattr(start_lm, "visibility", 1.0) < POSE_VISIBILITY_THRESHOLD:
            continue
        if getattr(end_lm, "visibility", 1.0) < POSE_VISIBILITY_THRESHOLD:
            continue

        start_pt = (int(start_lm.x * width), int(start_lm.y * height))
        end_pt = (int(end_lm.x * width), int(end_lm.y * height))
        cv2.line(frame, start_pt, end_pt, POSE_CONNECTION_COLOR, 4, cv2.LINE_AA)

    for lm in landmarks:
        if getattr(lm, "visibility", 1.0) < POSE_VISIBILITY_THRESHOLD:
            continue
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(frame, (x, y), 3, POSE_LANDMARK_COLOR, -1)

    marker_radius = max(6, min(height, width) // 96)
    outline_radius = marker_radius + 3

    for landmark in KEY_POSE_LANDMARKS:
        lm_idx = landmark.value
        if lm_idx >= len(landmarks):
            continue
        lm = landmarks[lm_idx]
        if getattr(lm, "visibility", 1.0) < POSE_VISIBILITY_THRESHOLD:
            continue

        x = int(lm.x * width)
        y = int(lm.y * height)

        cv2.circle(frame, (x, y), outline_radius, POSE_MARKER_OUTLINE, 3)
        cv2.circle(frame, (x, y), marker_radius, POSE_MARKER_FILL, -1)
        cv2.circle(frame, (x, y), max(2, marker_radius // 3), C_WHITE, -1)


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def _reencode_h264(input_path: str, output_path: str, fps: float) -> bool:
    """
    Re-encode a video to H.264 MP4 using ffmpeg.
    Returns True on success, False on failure.
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-movflags", "+faststart",
            "-an",  # no audio
            str(output_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, timeout=300,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

def annotate_video(
    video_path: str,
    result: BattingIQResult,
    metrics: list[FrameMetrics],
    output_path: str,
) -> None:
    """
    Re-process the video, draw overlays, write annotated output.
    A lightweight MediaPipe pass is used here to restore visible pose markers.

    Uses H.264 encoding via ffmpeg for high quality output.
    Falls back to mp4v if ffmpeg is not available.
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for annotation: {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build frame index lookup: original frame → metric list index
    frame_lookup = _build_frame_lookup(metrics)

    phases = result.phases
    labels = phases.phase_labels
    n_labels = len(labels)

    pillar_names = ["access", "tracking", "stability", "flow"]

    # Determine output strategy: ffmpeg H.264 or fallback mp4v
    use_ffmpeg = _ffmpeg_available()

    if use_ffmpeg:
        # Write to temporary MJPG AVI, then re-encode to H.264
        tmp_dir = tempfile.mkdtemp(prefix="battingiq_")
        tmp_path = Path(tmp_dir) / "raw.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        if use_ffmpeg:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Could not open video writer for annotated output: {output_path}")

    frame_idx = 0
    pose_model = _build_pose_model()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Map original frame index to subsampled metric index
            metric_idx = frame_lookup.get(frame_idx, None)
            m = metrics[metric_idx] if metric_idx is not None and metric_idx < len(metrics) else None
            label_idx = metric_idx if metric_idx is not None else 0
            label = labels[label_idx] if label_idx < n_labels else BattingPhase.UNKNOWN
            ph_colour = PHASE_COLOURS.get(label, C_WHITE)

            # Re-run pose estimation for visible body markers on the annotated frame.
            pose_landmarks = _detect_pose_landmarks(pose_model, frame)
            if pose_landmarks:
                _draw_pose_overlay(frame, pose_landmarks)

            # --- Contact flash (red overlay) ---
            if label == BattingPhase.CONTACT:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # --- Phase label (top left) ---
            ph_text = PHASE_LABELS.get(label, "")
            if ph_text:
                _panel_bg(frame, 5, 5, 230, 34)
                _text(frame, ph_text, (12, 28), scale=0.65, colour=ph_colour, bold=True)

            # --- Metrics panel (right side) ---
            px = width - 210
            panel_h = 200
            _panel_bg(frame, px - 5, 5, 215, panel_h)

            row = 25
            lh  = 24

            def metric_line(label_txt, val_txt, colour=C_WHITE):
                nonlocal row
                _text(frame, label_txt, (px, row), scale=0.45, colour=(180, 180, 180))
                _text(frame, val_txt,   (px + 120, row), scale=0.48, colour=colour, bold=True)
                row += lh

            if m:
                metric_line("Shoulder",  f"{m.shoulder_openness:.0f}°")
                metric_line("Hip open",  f"{m.hip_openness:.0f}°")
                metric_line("Head off",  f"{m.head_offset:+.3f}")
                metric_line("Eye tilt",  f"{m.eye_tilt:.3f}")
                metric_line("Frt knee",  f"{m.front_knee_angle:.0f}°")
                metric_line("Frt elbow", f"{m.front_elbow_angle:.0f}°")
                metric_line("Wrist H",   f"{m.wrist_height:.3f}")
                metric_line("Stance W",  f"{m.stance_width:.3f}")

            # --- Pillar scores (bottom right) ---
            by = height - 130
            _panel_bg(frame, px - 5, by, 215, 125)

            row = by + 20
            for pname in pillar_names:
                p = result.pillars.get(pname)
                if not p:
                    continue
                col = STATUS_COLOURS.get(p.status, C_WHITE)
                _text(frame, f"{pname[:4].upper()}", (px, row), scale=0.45, colour=(180, 180, 180))
                bar_len = int(p.score / p.max_score * 80)
                cv2.rectangle(frame, (px + 45, row - 12), (px + 45 + bar_len, row + 2), col, -1)
                _text(frame, f"{p.score}", (px + 135, row), scale=0.48, colour=col, bold=True)
                row += lh

            # BattingIQ score
            score_str = f"BattingIQ: {result.battingiq_score}  [{result.score_band}]"
            _panel_bg(frame, 5, height - 38, len(score_str) * 11, 32)
            _text(frame, score_str, (12, height - 14), scale=0.65,
                  colour=C_GREEN if result.battingiq_score >= 70 else C_AMBER, bold=True)

            # Frame counter
            _text(frame, f"f{frame_idx}", (width - 55, height - 10),
                  scale=0.4, colour=(120, 120, 120))

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        pose_model.close()

    if use_ffmpeg:
        # Re-encode temp AVI to H.264 MP4
        success = _reencode_h264(str(tmp_path), str(output_path), fps)
        # Clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if success and output_path.exists():
            print(f"  Annotated video (H.264) → {output_path}")
        else:
            print(f"  Warning: ffmpeg re-encode failed, falling back to mp4v")
            _annotate_fallback(video_path, result, metrics, output_path,
                               frame_lookup, fps, width, height)
            if not output_path.exists():
                raise RuntimeError(f"Annotated video generation failed: {output_path}")
    else:
        if not output_path.exists():
            raise RuntimeError(f"Annotated video generation failed: {output_path}")
        print(f"  Annotated video (mp4v) → {output_path}")


def _annotate_fallback(video_path, result, metrics, output_path,
                       frame_lookup, fps, width, height):
    """Fallback annotation using mp4v when ffmpeg is unavailable."""
    # Already written above with mp4v — this path only runs if ffmpeg fails
    # after we've already written to temp. In that case the mp4v path
    # was not used. Re-run the annotation with mp4v codec.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for fallback annotation: {video_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open fallback video writer: {output_path}")

    pose_model = _build_pose_model()

    phases = result.phases
    labels = phases.phase_labels
    n_labels = len(labels)
    pillar_names = ["access", "tracking", "stability", "flow"]

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            metric_idx = frame_lookup.get(frame_idx, None)
            m = metrics[metric_idx] if metric_idx is not None and metric_idx < len(metrics) else None
            label_idx = metric_idx if metric_idx is not None else 0
            label = labels[label_idx] if label_idx < n_labels else BattingPhase.UNKNOWN
            ph_colour = PHASE_COLOURS.get(label, C_WHITE)

            pose_landmarks = _detect_pose_landmarks(pose_model, frame)
            if pose_landmarks:
                _draw_pose_overlay(frame, pose_landmarks)

            if label == BattingPhase.CONTACT:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            ph_text = PHASE_LABELS.get(label, "")
            if ph_text:
                _panel_bg(frame, 5, 5, 230, 34)
                _text(frame, ph_text, (12, 28), scale=0.65, colour=ph_colour, bold=True)

            px = width - 210
            _panel_bg(frame, px - 5, 5, 215, 200)
            row = 25
            lh = 24

            def metric_line(label_txt, val_txt, colour=C_WHITE):
                nonlocal row
                _text(frame, label_txt, (px, row), scale=0.45, colour=(180, 180, 180))
                _text(frame, val_txt,   (px + 120, row), scale=0.48, colour=colour, bold=True)
                row += lh

            if m:
                metric_line("Shoulder",  f"{m.shoulder_openness:.0f}°")
                metric_line("Hip open",  f"{m.hip_openness:.0f}°")
                metric_line("Head off",  f"{m.head_offset:+.3f}")
                metric_line("Eye tilt",  f"{m.eye_tilt:.3f}")
                metric_line("Frt knee",  f"{m.front_knee_angle:.0f}°")
                metric_line("Frt elbow", f"{m.front_elbow_angle:.0f}°")
                metric_line("Wrist H",   f"{m.wrist_height:.3f}")
                metric_line("Stance W",  f"{m.stance_width:.3f}")

            by = height - 130
            _panel_bg(frame, px - 5, by, 215, 125)
            row = by + 20
            for pname in pillar_names:
                p = result.pillars.get(pname)
                if not p:
                    continue
                col = STATUS_COLOURS.get(p.status, C_WHITE)
                _text(frame, f"{pname[:4].upper()}", (px, row), scale=0.45, colour=(180, 180, 180))
                bar_len = int(p.score / p.max_score * 80)
                cv2.rectangle(frame, (px + 45, row - 12), (px + 45 + bar_len, row + 2), col, -1)
                _text(frame, f"{p.score}", (px + 135, row), scale=0.48, colour=col, bold=True)
                row += lh

            score_str = f"BattingIQ: {result.battingiq_score}  [{result.score_band}]"
            _panel_bg(frame, 5, height - 38, len(score_str) * 11, 32)
            _text(frame, score_str, (12, height - 14), scale=0.65,
                  colour=C_GREEN if result.battingiq_score >= 70 else C_AMBER, bold=True)
            _text(frame, f"f{frame_idx}", (width - 55, height - 10),
                  scale=0.4, colour=(120, 120, 120))

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        pose_model.close()

    if not output_path.exists():
        raise RuntimeError(f"Annotated video fallback failed to write output: {output_path}")

    print(f"  Annotated video (mp4v fallback) → {output_path}")


# ---------------------------------------------------------------------------
# Storyboard
# ---------------------------------------------------------------------------

# Per-phase: which metrics to show and how to format them
_PHASE_METRICS = {
    BattingPhase.SETUP:           [("shoulder_openness", "Shoulder",  "{:.0f}°"),
                                   ("stance_width",      "Stance W",  "{:.3f}")],
    BattingPhase.BACKLIFT_STARTS: [("wrist_height",      "Wrist H",   "{:.3f}"),
                                   ("back_knee_angle",   "Back Knee", "{:.0f}°")],
    BattingPhase.HANDS_PEAK:      [("wrist_height",      "Wrist H",   "{:.3f}"),
                                   ("shoulder_openness", "Shoulder",  "{:.0f}°")],
    BattingPhase.FRONT_FOOT_DOWN: [("front_knee_angle",  "Frt Knee",  "{:.0f}°"),
                                   ("head_offset",       "Head Off",  "{:+.3f}")],
    BattingPhase.CONTACT:         [("shoulder_openness", "Shoulder",  "{:.0f}°"),
                                   ("eye_tilt",          "Eye Tilt",  "{:.3f}")],
    BattingPhase.FOLLOW_THROUGH:  [("hip_openness",      "Hip Open",  "{:.0f}°"),
                                   ("shoulder_openness", "Shoulder",  "{:.0f}°")],
}

_THUMB_W  = 480   # px per panel (increased from 320 for HD quality)
_LABEL_H  = 80    # label bar height below the frame (scaled with thumb)
_PAD      = 8     # gap between panels


def generate_storyboard(
    video_path: str,
    result: BattingIQResult,
    metrics: list[FrameMetrics],
    output_path: str,
) -> None:
    """
    Extract the 6 key phase frames, annotate each with the pose skeleton
    and 2 phase-specific metrics, then stitch into a single horizontal strip.

    Phase indices are in subsampled metric space — this function converts
    them to original video frame indices for correct frame extraction.
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap     = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for storyboard generation: {video_path}")
    total   = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    thumb_h = int(orig_h * _THUMB_W / orig_w)

    pr = result.phases

    # Phase indices are in metric-list space. Convert to original frame indices
    # so we seek to the correct video position.
    def _to_orig_frame(metric_idx: int) -> int:
        """Convert a metric list index to the original video frame index."""
        idx = max(0, min(int(metric_idx), len(metrics) - 1))
        return metrics[idx].frame_idx if idx < len(metrics) else 0

    key_frames = [
        (pr.setup_end,            BattingPhase.SETUP,           "SETUP"),
        (pr.backlift_start,       BattingPhase.BACKLIFT_STARTS, "BACKLIFT"),
        (pr.hands_peak,           BattingPhase.HANDS_PEAK,      "HANDS PEAK"),
        (pr.front_foot_down,      BattingPhase.FRONT_FOOT_DOWN, "FRONT FOOT"),
        (pr.contact,              BattingPhase.CONTACT,         "CONTACT"),
        (pr.follow_through_start, BattingPhase.FOLLOW_THROUGH,  "FOLLOW-THROUGH"),
    ]

    panels = []
    pose_model = _build_pose_model(static_image_mode=True)

    try:
        for metric_idx, phase, label in key_frames:
            # Convert metric-space index to original video frame
            orig_frame = _to_orig_frame(metric_idx)
            orig_frame = max(0, min(orig_frame, total - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

            # Pose skeleton overlay
            pose_landmarks = _detect_pose_landmarks(pose_model, frame)
            if pose_landmarks:
                _draw_pose_overlay(frame, pose_landmarks)

            frame = cv2.resize(frame, (_THUMB_W, thumb_h))

            # Panel = frame + label bar
            panel = np.zeros((thumb_h + _LABEL_H, _THUMB_W, 3), dtype=np.uint8)
            panel[:thumb_h] = frame

            ph_colour = PHASE_COLOURS.get(phase, C_WHITE)

            # Coloured left-edge accent + dark label bar
            cv2.rectangle(panel, (0, thumb_h), (_THUMB_W, thumb_h + _LABEL_H), (18, 18, 18), -1)
            cv2.rectangle(panel, (0, thumb_h), (4, thumb_h + _LABEL_H), ph_colour, -1)

            # Phase name
            _text(panel, label, (10, thumb_h + 22), scale=0.55, colour=ph_colour, bold=True)

            # Timestamp (use original frame for accurate timing)
            fps_val = pr.fps or 30.0
            ts = f"{orig_frame / fps_val:.2f}s"
            _text(panel, ts, (_THUMB_W - 70, thumb_h + 22), scale=0.40, colour=(110, 110, 110))

            # 2 key metrics — use metric at the phase index
            mi = max(0, min(int(metric_idx), len(metrics) - 1))
            m = metrics[mi] if mi < len(metrics) else None
            if m:
                x_cursor = 10
                for attr, lbl, fmt in _PHASE_METRICS.get(phase, []):
                    val = getattr(m, attr, None)
                    if val is not None:
                        txt = f"{lbl}: {fmt.format(val)}"
                        _text(panel, txt, (x_cursor, thumb_h + 55),
                              scale=0.40, colour=(190, 190, 190))
                        x_cursor += _THUMB_W // 2

            panels.append(panel)
    finally:
        pose_model.close()

    cap.release()

    # Stitch 6 panels into one horizontal strip
    row_h   = thumb_h + _LABEL_H
    spacer  = np.zeros((row_h, _PAD, 3), dtype=np.uint8)
    strips  = []
    for i, p in enumerate(panels):
        if i > 0:
            strips.append(spacer)
        strips.append(p)

    storyboard = np.hstack(strips)
    if not cv2.imwrite(str(output_path), storyboard):
        raise RuntimeError(f"Could not write storyboard image: {output_path}")
    print(f"  Storyboard → {output_path}")
