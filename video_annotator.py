"""
BattingIQ Phase 2 — Video Annotator
======================================
Produces an annotated video with:
  - MediaPipe pose skeleton overlay
  - Phase label in top-left corner
  - Live metrics panel (right side)
  - Traffic-light flash at the Contact window
  - BattingIQ watermark
"""

import cv2
import mediapipe as mp
from pathlib import Path

from models import BattingIQResult, BattingPhase, TrafficLight, FrameMetrics, PhaseResult

mp_pose   = mp.solutions.pose
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

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
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    phases = result.phases
    labels = phases.phase_labels
    n      = len(labels)

    # Pre-compute pillar summary for the side panel
    pillar_names = ["access", "tracking", "stability", "flow"]

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
    ) as pose_model:

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = pose_model.process(rgb)

            m = metrics[frame_idx] if frame_idx < len(metrics) else None
            label = labels[frame_idx] if frame_idx < n else BattingPhase.UNKNOWN
            ph_colour = PHASE_COLOURS.get(label, C_WHITE)

            # --- Pose skeleton ---
            if pose_result.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    pose_result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

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
            lh  = 24   # line height

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

    cap.release()
    writer.release()
    print(f"  Annotated video → {output_path}")
