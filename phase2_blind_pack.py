#!/usr/bin/env python3
"""
Phase 2 blind pack.

Builds the pack the coaches score blind:
  clips/<ID>.mp4                 every clip, renamed to a random ID, container metadata
                                 stripped (names, dates, location), audio kept
  <coach>_scoring.html, <coach>_pairs.html  offline pages the coaches use (phase2_coach_pages.py)
  sheets/<coach>_score_sheet.csv one row per clip in a fresh random order per coach;
                                 a coach never gets a clip of themselves (clip_sheet.csv
                                 column scorer_in_clip)
  pairs/<coach>_pairs.csv        pairwise session (phase2_pairs.py schedule)
  blind_key_PRIVATE.csv          ID -> filename. Keep it private; never commit it.
  README_for_coaches.txt         instructions

Write the pack OUTSIDE the repo (default: next to the clip folder). The repo is
public and the pack holds clips of players, some of them juniors.
"""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from phase0_baseline_table import find_video, key_of, load_csv  # noqa: E402
from phase2_pairs import schedule_pairs  # noqa: E402

RUBRIC = SCRIPT_DIR / "docs" / "phase1" / "rubric_v0.json"
ID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I


def new_ids(n: int, rng: secrets.SystemRandom) -> list[str]:
    ids: set[str] = set()
    while len(ids) < n:
        ids.add("BQ-" + "".join(rng.choice(ID_ALPHABET) for _ in range(4)))
    return sorted(ids, key=lambda _: rng.random())


def remux_without_metadata(src: Path, dst: Path) -> str:
    """Stream copy into MP4 with all container metadata removed. Falls back to a re-encode."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is not on PATH")
    base = [ffmpeg, "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-map", "0:a?", "-map_metadata", "-1",
            "-map_chapters", "-1", "-metadata", "creation_time=", "-movflags", "+faststart"]
    tag = ["-tag:v", "hvc1"] if _is_hevc(src) else []  # Apple players need hvc1 for HEVC in MP4
    copy = subprocess.run(base + ["-c", "copy", *tag, str(dst)],
                          capture_output=True, text=True, check=False)
    if copy.returncode == 0:
        return "copy"
    enc = subprocess.run(base + ["-c:v", "libx264", "-crf", "16", "-preset", "slow", "-c:a", "aac", str(dst)],
                         capture_output=True, text=True, check=False)
    if enc.returncode != 0:
        raise SystemExit(f"ffmpeg failed on {src.name}: {enc.stderr[-400:]}")
    return "re-encoded"


def _is_hevc(src: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return False
    res = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                          "-of", "csv=p=0", str(src)], capture_output=True, text=True, check=False)
    return res.stdout.strip() == "hevc"


def sheet_fields() -> list[str]:
    rubric = json.loads(RUBRIC.read_text())
    return ["order", *rubric["score_sheet_fields"], "notes"]


def build(args: argparse.Namespace) -> int:
    rng = secrets.SystemRandom()
    video_dir = Path(args.video_dir).expanduser()
    out = Path(args.out).expanduser() if args.out else video_dir.parent / f"blind_pack_{date.today().isoformat()}"
    if SCRIPT_DIR in out.resolve().parents or out.resolve() == SCRIPT_DIR:
        raise SystemExit("Write the pack outside the repo: it holds player clips and the private key.")
    sheet = load_csv(Path(args.clip_sheet))
    clips = [r for r in sheet if find_video(video_dir, r["filename"])]
    if len(clips) != len(sheet):
        missing = {r["filename"] for r in sheet} - {r["filename"] for r in clips}
        raise SystemExit("Clips not found: " + ", ".join(sorted(missing)))
    coaches = [c.strip() for c in args.coaches.split(",") if c.strip()]

    for sub in ("clips", "sheets", "pairs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ids = new_ids(len(clips), rng)
    key_rows = []
    for clip_id, row in zip(ids, clips):
        src = find_video(video_dir, row["filename"])
        how = "skipped" if args.no_copy else remux_without_metadata(src, out / "clips" / f"{clip_id}.mp4")
        key_rows.append({"clip_id": clip_id, "filename": row["filename"], "scorer_in_clip": row.get("scorer_in_clip", ""),
                         "copy_mode": how})
    with (out / "blind_key_PRIVATE.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(key_rows[0].keys()))
        w.writeheader()
        w.writerows(key_rows)

    fields = sheet_fields()
    for coach in coaches:
        mine = [k for k in key_rows if k["scorer_in_clip"].casefold() != coach.casefold()]
        rng.shuffle(mine)
        with (out / "sheets" / f"{coach}_score_sheet.csv").open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=fields)
            w.writeheader()
            for i, k in enumerate(mine, 1):
                w.writerow({"order": i, "clip_id": k["clip_id"]})
        pairs = schedule_pairs([k["clip_id"] for k in mine], args.pairs_per_coach, rng)
        with (out / "pairs" / f"{coach}_pairs.csv").open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=["pair_no", "left_id", "right_id", "better", "confidence", "notes"])
            w.writeheader()
            for i, (left, right) in enumerate(pairs, 1):
                w.writerow({"pair_no": i, "left_id": left, "right_id": right})

    (out / "README_for_coaches.txt").write_text(f"""Batting IQ blind scoring pack ({date.today().isoformat()})

1. Score on your own. Do not discuss clips or scores with the other coaches until everyone has sent their sheets back.
2. Open <your name>_scoring.html in Chrome or Safari (double-click it; keep it in this folder so it can find the clips). It shows one clip at a time with the rubric. Work in the order given; it is a random order made for you. At the end, click "Download results" and send the file to Alex.
3. For each clip: rate C01-C12 as 2 (done well), 1 (partly), 0 (clear fault) or X (cannot judge from this clip). Then give the four pillar scores (0-25) and an overall score (0-100) on its own; do not add up the pillars. Then pick ONE priority fault (F01-F12), your confidence, and the minutes you took.
4. Open <your name>_pairs.html: two drives side by side; click the better one (or press L or R). About 25 minutes for 60 pairs. Do it on a different day from the scoring. Then "Download results" and send the file to Alex.
5. The clip names tell you nothing on purpose. If you recognise a player, say so in "notes" and score anyway.
6. Some clips show junior players. Do not share, upload or forward any clip.
""", encoding="utf-8")
    from phase2_coach_pages import build as build_pages
    build_pages(out)
    print(f"Pack written to {out}\n  clips: {len(key_rows)}  coaches: {', '.join(coaches)}  pairs per coach: {args.pairs_per_coach}")
    print(f"  KEEP PRIVATE: {out / 'blind_key_PRIVATE.csv'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--out", default=None, help="Pack folder; must be outside the repo.")
    parser.add_argument("--clip-sheet", default=str(SCRIPT_DIR / "clip_sheet.csv"))
    parser.add_argument("--coaches", default="Alex,Cam,Viv")
    parser.add_argument("--pairs-per-coach", type=int, default=60)
    parser.add_argument("--no-copy", action="store_true", help="Write sheets and key only (testing).")
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
