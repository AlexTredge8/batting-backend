#!/usr/bin/env python3
"""
Phase 2: one offline HTML file for the "which drive is better?" session, with the
clips built in, so anyone can take part without uploading or installing anything.

Each clip is trimmed to the shot (validated set-up - 0.5 s to follow-through + 0.7 s),
scaled down, stripped of sound and metadata, and (by default) has the batter's face
blurred using the pose landmarks. The clips get random codes (D01..D18); the
code -> filename key is written next to the HTML as *_key_PRIVATE.csv. Keep that private.

The person opens the file in a browser (double-click on a laptop), says what kind of
judge they are, and picks the better drive in about 40 pairs (swipe, tap or arrow keys).
At the end the page shows a short answer code to copy into a message to Alex. Nothing
leaves their device unless they send that code. `phase2_pairs.py from-codes` turns
the codes into pairs sheets for the fit.

PRIVACY: the file IS the footage. Whoever receives it can forward it. Face blur and
trimming reduce what it shows; only send it to people you would show the clips to.

  python phase2_offline_pairs.py --video-dir "<clips>" --out ~/Desktop/BattingIQ_which_drive.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

from phase0_baseline_table import find_video, key_of, load_csv  # noqa: E402

TEMPLATE = SCRIPT_DIR / "docs" / "phase2" / "offline_pairs_template.html"
NOSE, L_EAR, R_EAR, L_SH, R_SH = 0, 7, 8, 11, 12


def landmarks_for(video: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(landmarks [n,33,4], frame_idx [n]) from the Phase 0 pose cache, else run the pose extractor."""
    npz = next((p for p in cache_dir.glob("*.npz") if p.stem.casefold() == video.stem.casefold()), None) if cache_dir.exists() else None
    if npz is not None:
        d = np.load(npz)
        return d["landmarks"], d["frame_idx"]
    import pose_extractor
    frames, _ = pose_extractor._extract_with_tasks(str(video), verbose=False)
    arr = np.full((len(frames), 33, 4), np.nan, dtype=np.float32)
    for i, fp in enumerate(frames):
        if fp.landmarks:
            arr[i] = [[lm.x, lm.y, lm.z, lm.visibility] for lm in fp.landmarks]
    return arr, np.array([f.frame_idx for f in frames])


def encode_clip(video: Path, start: int, end: int, max_side: int, blur: bool, cache_dir: Path) -> tuple[bytes, dict]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # same orientation as pose_extractor, so landmarks line up
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lm = landmarks_for(video, cache_dir) if blur else None
    by_frame = {int(f): i for i, f in enumerate(lm[1])} if lm is not None else {}
    frames, idx, last_head = [], 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if start <= idx <= end:
            h, w = frame.shape[:2]
            if blur:
                i = by_frame.get(idx)
                pts = lm[0][i] if i is not None else None
                if pts is not None and not np.isnan(pts[[NOSE, L_SH, R_SH], :2]).any():
                    cx, cy = pts[NOSE, 0] * w, pts[NOSE, 1] * h
                    sh = np.hypot((pts[L_SH, 0] - pts[R_SH, 0]) * w, (pts[L_SH, 1] - pts[R_SH, 1]) * h)
                    ear = np.hypot((pts[L_EAR, 0] - pts[R_EAR, 0]) * w, (pts[L_EAR, 1] - pts[R_EAR, 1]) * h) \
                        if not np.isnan(pts[[L_EAR, R_EAR], :2]).any() else 0
                    r = max(10.0, 0.45 * sh, 0.9 * ear, 0.035 * max(w, h))
                    last_head = (cx, cy, r)
                if last_head is not None:  # a missed frame keeps the last head position rather than showing the face
                    cx, cy, r = last_head
                    x0, y0 = max(0, int(cx - r)), max(0, int(cy - 1.2 * r))
                    x1, y1 = min(w, int(cx + r)), min(h, int(cy + 1.0 * r))
                    if x1 > x0 and y1 > y0:
                        roi = frame[y0:y1, x0:x1]
                        small = cv2.resize(roi, (max(1, (x1 - x0) // 10), max(1, (y1 - y0) // 10)), interpolation=cv2.INTER_LINEAR)
                        mask = np.zeros(roi.shape[:2], np.uint8)
                        cv2.ellipse(mask, ((x1 - x0) // 2, (y1 - y0) // 2), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 255, -1)
                        pix = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
                        roi[mask > 0] = cv2.GaussianBlur(pix, (0, 0), 3)[mask > 0]
            scale = min(1.0, max_side / max(h, w))
            nw, nh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
            frames.append(cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA))
        idx += 1
        if idx > end:
            break
    cap.release()
    if not frames and start > 0:
        print(f"  {video.name}: shot window {start}-{end} is past the end of the file; using the whole clip")
        return encode_clip(video, 0, 10 ** 6, max_side, blur, cache_dir)
    if not frames:
        raise SystemExit(f"No frames read from {video.name}")
    nh, nw = frames[0].shape[:2]
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is not on PATH")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:  # MP4 with faststart needs a seekable output, not a pipe
        dst = Path(tmp) / "clip.mp4"
        out = subprocess.run([ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{nw}x{nh}", "-r", f"{fps:.3f}",
                              "-i", "-", "-an", "-map_metadata", "-1", "-c:v", "libx264", "-preset", "slow", "-crf", "27",
                              "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
                             input=b"".join(f.tobytes() for f in frames), capture_output=True, check=False)
        if out.returncode != 0:
            raise SystemExit(f"ffmpeg failed on {video.name}: {out.stderr.decode()[-300:]}")
        data = dst.read_bytes()
    return data, {"frames": len(frames), "fps": round(fps, 2), "size": f"{nw}x{nh}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--out", required=True, help="the HTML file to write; must be outside the repo")
    parser.add_argument("--ground-truth", default=str(SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"))
    parser.add_argument("--anchor-truth", default=str(SCRIPT_DIR / "anchor_truth.csv"))
    parser.add_argument("--cache-dir", default=str(SCRIPT_DIR / "calibration_output" / "pose_cache"))
    parser.add_argument("--pairs", type=int, default=40, help="pairs each person judges (about 20 seconds each)")
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--no-blur", action="store_true", help="keep faces visible (only for a coaches-only file)")
    args = parser.parse_args(argv)

    out = Path(args.out).expanduser().resolve()
    if SCRIPT_DIR in out.parents:
        raise SystemExit("Write the file outside the repo: it holds the clips.")
    video_dir = Path(args.video_dir).expanduser()
    anchors = {key_of(r["filename"]): r for r in load_csv(Path(args.anchor_truth))}
    names = [r["filename"] for r in load_csv(Path(args.ground_truth))]
    rng = secrets.SystemRandom()
    codes = [f"D{i:02d}" for i in range(1, len(names) + 1)]
    rng.shuffle(codes)
    build_id = secrets.token_hex(3).upper()

    clips, key_rows = [], []
    for name, code in sorted(zip(names, codes), key=lambda t: t[1]):
        video = find_video(video_dir, name)
        if video is None:
            raise SystemExit(f"Clip not found: {name}")
        a = anchors.get(key_of(name), {})
        fps_guess = 30
        start = max(0, int(a["setup_frame"]) - fps_guess // 2) if a.get("setup_frame", "").strip() else 0
        end = int(a["follow_through_frame"]) + int(0.7 * fps_guess) if a.get("follow_through_frame", "").strip() else 10 ** 6
        data, info = encode_clip(video, start, end, args.max_side, not args.no_blur, Path(args.cache_dir))
        clips.append({"id": code, "b64": base64.b64encode(data).decode()})
        key_rows.append({"clip_id": code, "filename": name, "build_id": build_id, "bytes": len(data), **info})
        print(f"{code}  {len(data) / 1024:6.0f} KB  {info['size']}  {info['frames']} frames")

    payload = {"build": build_id, "pairs": args.pairs, "clips": clips}
    html = TEMPLATE.read_text().replace("/*__PAYLOAD__*/null", json.dumps(payload))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    key_path = out.with_name(out.stem + "_key_PRIVATE.csv")
    with key_path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(key_rows[0].keys()))
        w.writeheader()
        w.writerows(key_rows)
    print(f"\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB, build {build_id}, {args.pairs} pairs per person, "
          f"faces {'visible' if args.no_blur else 'blurred'})\nKEEP PRIVATE: {key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
