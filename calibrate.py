#!/usr/bin/env python3
"""
BattingIQ Calibration Script

Sends each ground truth video to the Railway API and outputs a CSV of model scores.
Compare these against your coach consensus scores in Excel to find calibration gaps.

Usage:
    python3 calibrate.py /path/to/battingiq-ground-truth/videos/

Output:
    model_scores.csv in the current directory
"""

import sys
import os
import csv
import time

try:
    import requests
except ImportError:
    print("ERROR: 'requests' library not found.")
    print("Install it with:  pip3 install requests")
    sys.exit(1)

API_URL = "https://web-production-e9c26.up.railway.app/analyse"
TIMEOUT = 120  # seconds per video
VIDEO_EXTENSIONS = {".mp4", ".mov", ".MP4", ".MOV", ".m4v", ".mp"}
OUTPUT_FILE = "model_scores.csv"
LOW_DETECTION_THRESHOLD = 85.0


def resolve_folder(path):
    # iCloud Drive on macOS appends a colon to paths — strip it
    path = path.rstrip(":").rstrip("/").rstrip("\\")
    return os.path.abspath(path)


def find_videos(folder):
    files = []
    for f in os.listdir(folder):
        _, ext = os.path.splitext(f)
        if ext in VIDEO_EXTENSIONS:
            files.append(f)
    return sorted(files)


def analyse_video(video_path):
    with open(video_path, "rb") as f:
        response = requests.post(
            API_URL,
            files={"file": (os.path.basename(video_path), f, "video/mp4")},
            data={"handedness": "right"},
            timeout=TIMEOUT,
        )
    response.raise_for_status()
    return response.json()


def extract_scores(data):
    pillars = data.get("pillars", {})
    detection_rate = data.get("metadata", {}).get("detection_rate", None)
    return {
        "model_overall": data.get("battingiq_score", ""),
        "model_access": pillars.get("access", {}).get("score", ""),
        "model_tracking": pillars.get("tracking", {}).get("score", ""),
        "model_stability": pillars.get("stability", {}).get("score", ""),
        "model_flow": pillars.get("flow", {}).get("score", ""),
        "detection_rate": round(detection_rate, 1) if detection_rate is not None else "",
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 calibrate.py /path/to/videos/folder/")
        sys.exit(1)

    folder = resolve_folder(sys.argv[1])

    if not os.path.isdir(folder):
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)

    videos = find_videos(folder)

    if not videos:
        print(f"No video files found in: {folder}")
        sys.exit(1)

    print(f"\nFound {len(videos)} video(s) in {folder}")
    print(f"Results will be written to: {OUTPUT_FILE}\n")

    rows = []
    fieldnames = [
        "filename",
        "model_overall",
        "model_access",
        "model_tracking",
        "model_stability",
        "model_flow",
        "detection_rate",
        "error",
    ]

    for i, filename in enumerate(videos, start=1):
        video_path = os.path.join(folder, filename)
        print(f"[{i}/{len(videos)}] Processing: {filename} ...", end=" ", flush=True)

        start = time.time()
        try:
            data = analyse_video(video_path)
            scores = extract_scores(data)
            elapsed = time.time() - start

            warning = ""
            detection = scores.get("detection_rate")
            if detection != "" and float(detection) < LOW_DETECTION_THRESHOLD:
                warning = f" ⚠ Low detection rate ({detection}%) — scores may be unreliable"

            print(f"done ({elapsed:.0f}s) — BattingIQ: {scores['model_overall']}{warning}")

            rows.append({"filename": filename, **scores, "error": ""})

        except requests.exceptions.Timeout:
            elapsed = time.time() - start
            print(f"TIMEOUT after {elapsed:.0f}s")
            rows.append({
                "filename": filename,
                "model_overall": "", "model_access": "", "model_tracking": "",
                "model_stability": "", "model_flow": "", "detection_rate": "",
                "error": "API timeout",
            })

        except requests.exceptions.HTTPError as e:
            print(f"HTTP ERROR {e.response.status_code}")
            rows.append({
                "filename": filename,
                "model_overall": "", "model_access": "", "model_tracking": "",
                "model_stability": "", "model_flow": "", "detection_rate": "",
                "error": f"HTTP {e.response.status_code}",
            })

        except Exception as e:
            print(f"ERROR: {e}")
            rows.append({
                "filename": filename,
                "model_overall": "", "model_access": "", "model_tracking": "",
                "model_stability": "", "model_flow": "", "detection_rate": "",
                "error": str(e),
            })

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    success = sum(1 for r in rows if r["error"] == "")
    failed = len(rows) - success

    print(f"\n{'='*50}")
    print(f"Done. {success}/{len(rows)} videos analysed successfully.")
    if failed:
        print(f"{failed} failed — check the 'error' column in {OUTPUT_FILE}")
    print(f"Output saved to: {os.path.abspath(OUTPUT_FILE)}")
    print(f"{'='*50}\n")
    print("Next step: open model_scores.csv and paste the columns into your Excel")
    print("alongside the coach consensus scores, then add gap columns (coach - model).\n")


if __name__ == "__main__":
    main()
