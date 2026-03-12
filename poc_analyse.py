"""
BattingIQ - Phase 1 Proof of Concept
=====================================
Processes a cricket batting video filmed from the bowler's end.
Extracts pose landmarks using MediaPipe BlazePose.
Calculates coaching-relevant measurements.
Outputs an annotated video + JSON summary.

Usage:
    python poc_analyse.py path/to/video.mp4
"""

import sys
import json
import math
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# --- Landmark Helpers ---
# MediaPipe BlazePose landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_WRIST = 15
RIGHT_WRIST = 16


def get_landmark(landmarks, idx):
    """Extract x, y, z, visibility for a landmark."""
    lm = landmarks.landmark[idx]
    return {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}


def calc_angle_degrees(a, b, c):
    """Calculate angle at point b given three points (each with x, y)."""
    ab = math.sqrt((b["x"] - a["x"])**2 + (b["y"] - a["y"])**2)
    bc = math.sqrt((c["x"] - b["x"])**2 + (c["y"] - b["y"])**2)
    ac = math.sqrt((c["x"] - a["x"])**2 + (c["y"] - a["y"])**2)
    if ab * bc == 0:
        return 0
    cos_angle = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
    cos_angle = max(-1, min(1, cos_angle))  # clamp for floating point
    return math.degrees(math.acos(cos_angle))


def calc_lateral_offset(nose, left_hip, right_hip):
    """
    Calculate how far the head is offset laterally from the midpoint
    of the hips. From bowler's end, this shows if the batter's head
    is falling to off side or leg side.
    
    Returns: offset in normalised coordinates (negative = off side for RHB)
    """
    hip_midpoint_x = (left_hip["x"] + right_hip["x"]) / 2
    return nose["x"] - hip_midpoint_x


def calc_shoulder_angle(left_shoulder, right_shoulder):
    """
    Calculate shoulder rotation from bowler's end.
    If shoulders are square to the bowler, the angle is ~0.
    If opening up, the x-difference increases.
    
    Returns: angle in degrees from vertical (0 = perfectly side-on)
    """
    dx = abs(right_shoulder["x"] - left_shoulder["x"])
    dy = abs(right_shoulder["y"] - left_shoulder["y"])
    if dy == 0:
        return 90
    return math.degrees(math.atan2(dx, dy))


def calc_stance_width(left_ankle, right_ankle):
    """
    Stance width as normalised horizontal distance between ankles.
    """
    return abs(left_ankle["x"] - right_ankle["x"])


# --- Frame Analysis ---
def analyse_frame(landmarks):
    """
    Extract all coaching-relevant measurements from one frame.
    """
    nose = get_landmark(landmarks, NOSE)
    l_shoulder = get_landmark(landmarks, LEFT_SHOULDER)
    r_shoulder = get_landmark(landmarks, RIGHT_SHOULDER)
    l_hip = get_landmark(landmarks, LEFT_HIP)
    r_hip = get_landmark(landmarks, RIGHT_HIP)
    l_knee = get_landmark(landmarks, LEFT_KNEE)
    r_knee = get_landmark(landmarks, RIGHT_KNEE)
    l_ankle = get_landmark(landmarks, LEFT_ANKLE)
    r_ankle = get_landmark(landmarks, RIGHT_ANKLE)
    l_wrist = get_landmark(landmarks, LEFT_WRIST)
    r_wrist = get_landmark(landmarks, RIGHT_WRIST)

    # Key measurements from bowler's end
    head_offset = calc_lateral_offset(nose, l_hip, r_hip)
    shoulder_openness = calc_shoulder_angle(l_shoulder, r_shoulder)
    stance_width = calc_stance_width(l_ankle, r_ankle)

    # Wrist height (proxy for bat position / hands)
    avg_wrist_height = (l_wrist["y"] + r_wrist["y"]) / 2

    # Knee bend - angle at each knee
    l_knee_angle = calc_angle_degrees(l_hip, l_knee, l_ankle)
    r_knee_angle = calc_angle_degrees(r_hip, r_knee, r_ankle)

    return {
        "head_lateral_offset": round(head_offset, 4),
        "shoulder_openness_deg": round(shoulder_openness, 1),
        "stance_width": round(stance_width, 4),
        "avg_wrist_height": round(avg_wrist_height, 4),
        "left_knee_angle": round(l_knee_angle, 1),
        "right_knee_angle": round(r_knee_angle, 1),
        "nose_visibility": round(nose["visibility"], 2),
    }


# --- Video Processing ---
def process_video(video_path, output_dir=None):
    """
    Process a batting video:
    1. Run BlazePose on each frame
    2. Collect measurements
    3. Output annotated video + JSON results
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        sys.exit(1)

    if output_dir is None:
        output_dir = video_path.parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path.name}")
    print(f"Resolution: {width}x{height} | FPS: {fps} | Frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.1f}s")
    print()

    # Output video with pose overlay
    output_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Collect frame-by-frame data
    all_frame_data = []
    frames_with_pose = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,       # highest accuracy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            frame_data = {"frame": frame_num, "timestamp_s": round(frame_num / fps, 3)}

            if results.pose_landmarks:
                frames_with_pose += 1

                # Draw pose on frame
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # Extract measurements
                measurements = analyse_frame(results.pose_landmarks)
                frame_data["measurements"] = measurements

                # Draw key info on frame
                cv2.putText(
                    frame,
                    f"Head offset: {measurements['head_lateral_offset']:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.putText(
                    frame,
                    f"Shoulder open: {measurements['shoulder_openness_deg']:.1f} deg",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.putText(
                    frame,
                    f"Stance width: {measurements['stance_width']:.3f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
            else:
                frame_data["measurements"] = None

            all_frame_data.append(frame_data)
            out_writer.write(frame)

            frame_num += 1
            if frame_num % 30 == 0:
                print(f"  Processed {frame_num}/{total_frames} frames...")

    cap.release()
    out_writer.release()

    # --- Summary Statistics ---
    measured_frames = [f for f in all_frame_data if f["measurements"] is not None]

    if not measured_frames:
        print("\n⚠️  No pose detected in any frame!")
        print("Check that the batter is clearly visible in the video.")
        return

    head_offsets = [f["measurements"]["head_lateral_offset"] for f in measured_frames]
    shoulder_angles = [f["measurements"]["shoulder_openness_deg"] for f in measured_frames]
    stance_widths = [f["measurements"]["stance_width"] for f in measured_frames]

    summary = {
        "video": video_path.name,
        "total_frames": total_frames,
        "frames_with_pose": frames_with_pose,
        "detection_rate": round(frames_with_pose / total_frames * 100, 1),
        "duration_s": round(total_frames / fps, 1),
        "measurements_summary": {
            "head_lateral_offset": {
                "mean": round(np.mean(head_offsets), 4),
                "min": round(np.min(head_offsets), 4),
                "max": round(np.max(head_offsets), 4),
                "std": round(np.std(head_offsets), 4),
            },
            "shoulder_openness_deg": {
                "mean": round(np.mean(shoulder_angles), 1),
                "min": round(np.min(shoulder_angles), 1),
                "max": round(np.max(shoulder_angles), 1),
            },
            "stance_width": {
                "mean": round(np.mean(stance_widths), 4),
                "min": round(np.min(stance_widths), 4),
                "max": round(np.max(stance_widths), 4),
            },
        },
    }

    # Save results
    json_path = output_dir / f"{video_path.stem}_analysis.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print(f"\n{'='*50}")
    print(f"BATTINGIQ - PHASE 1 PROOF OF CONCEPT")
    print(f"{'='*50}")
    print(f"Pose detected in {summary['detection_rate']}% of frames")
    print()
    print(f"HEAD POSITION (lateral offset from hips)")
    print(f"  Average: {summary['measurements_summary']['head_lateral_offset']['mean']:.4f}")
    print(f"  Range:   {summary['measurements_summary']['head_lateral_offset']['min']:.4f} to {summary['measurements_summary']['head_lateral_offset']['max']:.4f}")
    print()
    print(f"SHOULDER OPENNESS (0=side-on, 90=fully open)")
    print(f"  Average: {summary['measurements_summary']['shoulder_openness_deg']['mean']:.1f}°")
    print(f"  Range:   {summary['measurements_summary']['shoulder_openness_deg']['min']:.1f}° to {summary['measurements_summary']['shoulder_openness_deg']['max']:.1f}°")
    print()
    print(f"STANCE WIDTH")
    print(f"  Average: {summary['measurements_summary']['stance_width']['mean']:.4f}")
    print()
    print(f"Output files:")
    print(f"  Annotated video: {output_video_path}")
    print(f"  JSON analysis:   {json_path}")
    print(f"{'='*50}")

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python poc_analyse.py <video_path>")
        print("Example: python poc_analyse.py my_batting.mp4")
        sys.exit(1)

    process_video(sys.argv[1])
