#!/usr/bin/env python3
"""
Phase 0 clip sheet: measured columns.

For each clip in the coach ground-truth CSV this measures container facts
(ffprobe, or ffmpeg -i as a fallback), the md5, the pose detection rate and
the batter height in pixels at set-up, and writes them into clip_sheet.csv.
Columns that a person fills (player_id, feed, line, venue, ...) are kept
as they are: re-running never overwrites them.

It also saves per-frame landmarks to calibration_output/pose_cache/<stem>.npz
(git-ignored) for phase0_wrong_plane.py.

Privacy: the repo is public. clip_sheet.csv holds pseudonymous player IDs
only. Names, ages, junior status and consent stay in player_key_PRIVATE.csv
next to the clips, which .gitignore excludes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

from phase0_baseline_table import find_video, key_of, load_csv, md5_of  # noqa: E402

MEASURED = [
    "duration_s", "avg_fps", "nb_frames", "implied_fps", "codec", "width", "height", "rotation", "bitrate_kbps",
    "audio_streams", "audio_codec", "audio_sample_rate", "md5", "pose_backend", "pose_detection_rate",
    "batter_height_px", "batter_height_pct", "height_window",
]
ALEX = [
    "player_id", "scorer_in_clip", "handedness", "drive_type", "feed", "line", "venue",
    "camera_position", "others_in_frame", "flags_confirmed",
]
COLUMNS = ["filename", "tier_csv", "coach_overall", "split", *MEASURED, *ALEX, "flags_hub",
           "hub_secs", "hub_fps", "hub_batter_px"]

# The hub's first reading (13 Nov 2026): secs, fps, batter height px from one frame, flags.
HUB = {
    "Cam T_LH Offdrive Elite.mov": (4.2, 30, 1400, ""),
    "Cam T_Offdrive Elite.mp4": (5.4, 30, 215, ""),
    "Cam T_Ondrive Elite.mov": (5.5, 30, 1000, ""),
    "Viv P_Offdrive Elite.mov": (3.7, 30, 260, "same file as reference clip"),
    "Alex T_Straightdrive Good.MOV": (1.8, 30, 290, "under 2 s"),
    "AlexT_Ondrive Good.MOV": (2.2, 30, 270, ""),
    "Cam T_Offdrive Good.mp4": (8.4, 30, 380, ""),
    "Ollie P_Offdrive Average.mov": (3.9, 39, 140, "no audio; two figures; small; black bars"),
    "Dan P_Offdrive Average.MOV": (3.2, 30, 235, ""),
    "Drive_Average 2.MOV": (3.7, 30, 215, ""),
    "Drive_Average.MOV": (4.4, 30, 370, "angle; others in shot"),
    "Female_Average.mp4": (10.6, 30, 145, "small; filmed_correctly = N"),
    "Drive_Beginner 2.MOV": (3.1, 30, 390, "angle; others in shot"),
    "Drive_Beginner 3.MOV": (1.7, 26, 510, "angle; under 2 s"),
    "Drive_Beginner 4.MOV": (2.2, 26, 295, ""),
    "Drive_beginner.MOV": (1.2, 26, 470, "angle; under 2 s; 42 frames"),
    "Oscar V_Beginner.mp4": (5.9, 30, 235, ""),
    "Straight Drive_Beginner.mp4": (6.0, 30, 135, "small"),
}
NOSE, L_ANKLE, R_ANKLE = 0, 27, 28


def drive_type_from_name(filename: str) -> str:
    stem = Path(filename).stem.lower().replace(" ", "")
    for token, label in (("offdrive", "off"), ("ondrive", "on"), ("straightdrive", "straight")):
        if token in stem:
            return label
    return ""


def probe(video: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        res = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)],
                             capture_output=True, text=True, check=False)
        try:
            data = json.loads(res.stdout or "{}")
        except json.JSONDecodeError:
            data = {}
        streams = data.get("streams", [])
        v = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        rot = v.get("tags", {}).get("rotate")
        for side in v.get("side_data_list", []) or []:
            if "rotation" in side:
                rot = side["rotation"]
        num, _, den = (v.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else None
        w, h = v.get("width"), v.get("height")
        if rot is not None and abs(int(float(rot))) % 180 == 90 and w and h:
            w, h = h, w
        out.update({
            "duration_s": round(float(data.get("format", {}).get("duration", 0) or 0), 3),
            "avg_fps": round(fps, 3) if fps else None,
            "nb_frames": v.get("nb_frames"),
            "codec": v.get("codec_name"),
            "width": w, "height": h,
            "rotation": rot if rot is not None else 0,
            "bitrate_kbps": round(int(data.get("format", {}).get("bit_rate", 0) or 0) / 1000),
            "audio_streams": len(audio),
            "audio_codec": audio[0].get("codec_name") if audio else "",
            "audio_sample_rate": audio[0].get("sample_rate") if audio else "",
        })
        return out

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return out
    err = subprocess.run([ffmpeg, "-hide_banner", "-i", str(video)], capture_output=True, text=True, check=False).stderr
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if dur:
        out["duration_s"] = round(int(dur[1]) * 3600 + int(dur[2]) * 60 + float(dur[3]), 3)
    br = re.search(r"bitrate: (\d+) kb/s", err)
    out["bitrate_kbps"] = int(br[1]) if br else None
    vid = re.search(r"Video: (\w+).*?, (\d{2,5})x(\d{2,5}).*?([\d.]+) fps", err)
    if vid:
        out.update({"codec": vid[1], "width": int(vid[2]), "height": int(vid[3]), "avg_fps": float(vid[4])})
    rot = re.search(r"rotat\w* (?:of )?(-?[\d.]+)", err)
    out["rotation"] = float(rot[1]) if rot else 0
    if rot and abs(int(float(rot[1]))) % 180 == 90 and out.get("width"):
        out["width"], out["height"] = out["height"], out["width"]
    audio = re.findall(r"Audio: (\w+), (\d+) Hz", err)
    out["audio_streams"] = len(audio)
    out["audio_codec"] = audio[0][0] if audio else ""
    out["audio_sample_rate"] = audio[0][1] if audio else ""
    return out


def pose_facts(video: Path, setup_frame: int | None, cache_dir: Path) -> dict[str, Any]:
    import numpy as np
    import pose_extractor

    backend = "tasks_heavy"
    try:
        frames, meta = pose_extractor._extract_with_tasks(str(video), verbose=False)
    except Exception as exc:  # record, do not hide, a fallback
        backend = f"legacy_solutions ({type(exc).__name__})"
        frames, meta = pose_extractor._extract_with_legacy_pose(str(video), verbose=False)

    width, height = meta.get("width"), meta.get("height")
    arr = np.full((len(frames), 33, 4), np.nan, dtype=np.float32)
    idx = np.array([f.frame_idx for f in frames], dtype=np.int32)
    for i, fp in enumerate(frames):
        if fp.landmarks:
            arr[i] = [[lm.x, lm.y, lm.z, lm.visibility] for lm in fp.landmarks]
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"{video.stem}.npz", landmarks=arr, frame_idx=idx,
                        fps=meta.get("fps"), width=width, height=height)

    detected = [i for i, fp in enumerate(frames) if fp.landmarks]
    if setup_frame is not None:
        window = [i for i, fp in enumerate(frames) if setup_frame - 10 <= fp.frame_idx <= setup_frame and fp.landmarks]
        label = f"anchor setup {setup_frame - 10}..{setup_frame}"
    else:
        window = detected[:15]
        label = "first 15 detected frames (no_anchor_row)"
    heights = [(max(arr[i, L_ANKLE, 1], arr[i, R_ANKLE, 1]) - arr[i, NOSE, 1]) * height for i in window]
    px = round(float(median(heights))) if heights else None
    return {
        "pose_backend": backend,
        "pose_detection_rate": meta.get("detection_rate"),
        "batter_height_px": px,
        "batter_height_pct": round(100 * px / height, 1) if px and height else None,
        "height_window": label,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-dir", default=None, help="Clip folder. Omit to write the sheet without measured columns.")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "clip_sheet.csv"))
    parser.add_argument("--notes", default=str(SCRIPT_DIR / "clip_sheet_notes.md"))
    parser.add_argument("--ground-truth", default=str(SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"))
    parser.add_argument("--anchor-truth", default=str(SCRIPT_DIR / "anchor_truth.csv"))
    parser.add_argument("--split", default=str(SCRIPT_DIR / "heldout_split.csv"))
    parser.add_argument("--cache-dir", default=str(SCRIPT_DIR / "calibration_output" / "pose_cache"))
    parser.add_argument("--skip-pose", action="store_true")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    existing = {key_of(r["filename"]): r for r in load_csv(out_path)} if out_path.exists() else {}
    anchors = {key_of(r["filename"]): r for r in load_csv(Path(args.anchor_truth))}
    split = {key_of(r["filename"]): r.get("split", "") for r in load_csv(Path(args.split))}
    video_dir = Path(args.video_dir).expanduser() if args.video_dir else None

    rows = []
    for gt in load_csv(Path(args.ground_truth)):
        name = gt["filename"]
        k = key_of(name)
        old = existing.get(k, {})
        row = {c: old.get(c, "") for c in COLUMNS}
        hub = HUB.get(name, (None, None, None, ""))
        row.update({"filename": name, "tier_csv": gt.get("tier", ""), "coach_overall": gt.get("overall_score", ""),
                    "split": split.get(k, ""), "hub_secs": hub[0], "hub_fps": hub[1], "hub_batter_px": hub[2]})
        if not row["drive_type"]:
            row["drive_type"] = drive_type_from_name(name)
        if not row["scorer_in_clip"]:  # pre-fill from the filename; Alex confirms
            low = Path(name).stem.lower().replace(" ", "")
            row["scorer_in_clip"] = next((who for tag, who in (("camt", "Cam"), ("vivp", "Viv"), ("alext", "Alex"))
                                          if low.startswith(tag)), "")
        if not row["handedness"] and "_lh" in Path(name).stem.lower():
            row["handedness"] = "left"
        video = find_video(video_dir, name) if video_dir else None
        flags = [f for f in hub[3].split("; ") if f]
        if video is not None:
            row.update(probe(video))
            row["md5"] = md5_of(video)
            try:  # frames per container second; differs from avg_fps on variable-frame-rate files
                row["implied_fps"] = round(float(row["nb_frames"]) / float(row["duration_s"]), 2)
            except (TypeError, ValueError, ZeroDivisionError):
                row["implied_fps"] = ""
            if not args.skip_pose:
                setup = anchors.get(k, {}).get("setup_frame", "")
                row.update(pose_facts(video, int(setup) if setup.strip() else None, Path(args.cache_dir)))
            try:
                if float(row.get("duration_s") or 99) < 2 and "under 2 s" not in flags:
                    flags.append("under 2 s")
                if float(row.get("batter_height_pct") or 100) < 20 and "small" not in flags:
                    flags.append("small")
                if str(row.get("audio_streams")) == "0" and "no audio" not in flags:
                    flags.append("no audio")
                if row.get("implied_fps") and row.get("avg_fps") and abs(float(row["implied_fps"]) - float(row["avg_fps"])) > 1:
                    flags.append(f"frame-rate mismatch ({row['implied_fps']} vs {row['avg_fps']} fps)")
            except ValueError:
                pass
        elif video_dir is not None:
            flags.append("FILE NOT FOUND")
        row["flags_hub"] = "; ".join(flags)
        rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    notes = ["# Clip sheet notes", "",
             "Measured batter height (median nose-to-lower-ankle over the 11 frames ending at the validated set-up) "
             "against the hub's one-frame estimate. Differences above 25% are listed.", ""]
    flagged = []
    for r in rows:
        try:
            m, h = float(r["batter_height_px"]), float(r["hub_batter_px"])
        except (TypeError, ValueError):
            continue
        if h and abs(m - h) / h > 0.25:
            flagged.append(f"- {r['filename']}: measured {m:.0f} px, hub {h:.0f} px ({100 * (m - h) / h:+.0f}%)")
    notes += flagged or ["- None (or measured heights not yet available)."]
    Path(args.notes).write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} clips) and {args.notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
