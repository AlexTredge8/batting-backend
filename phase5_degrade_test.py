#!/usr/bin/env python3
"""
Phase 5 entry test 2, replacement: capture-degradation test (15 Nov 2026).

The plan's two-phone net session is not going ahead (Alex, 15 Nov: real uploads will often be
poor phone video from the front). Test 2 instead asks: does each measurement give the same
answer when the SAME clip is made worse in the ways a poor phone or poor filming makes it worse?

Variants of every clip (ffmpeg, from the original file):
  lowres  360 px tall, heavy compression (CRF 32)            - cheap phone, WhatsApp-style copy
  lowfps  half the frame rate                                 - older phone, low light
  tilt    rotated 3 degrees and cropped 8% at each edge, then  - phone held off-level, closer
          scaled back to the original size
Each variant goes through the production pose path (MediaPipe, same settings) and the Phase 5
measurements, with the validated key frames moved to the same moment in time.

Pass (fixed 15 Nov 2026, before any variant was run; same rule as the two-phone test):
  for every variant, the median |variant - original| across clips <= 0.25 x the between-clip
  IQR of the original values.

  python phase5_degrade_test.py --video-dir <clips> --out baselines/phase5_degrade_<date>
      [--lowres-height 360]

--lowres-height (added 16 Nov 2026) sets the height of the lowres variant. The default, 360, is the
registered test and stays the default for the record. Any other value names the variant
"lowres<height>" (its own pose cache), and never scales a clip up: a clip already at or below that
height is only re-compressed. All 18 calibration clips are portrait, so the height is the LONG side:
--lowres-height 1280 gives 720 px on the short side, the Phase 3 gate's minimum
(docs/phase3/capture_standard.md); --lowres-height 720 would give about 406 px, below the gate.
Re-registering test 2 at the gate's minimum is a proposal awaiting Alex (docs/phase5/README.md).
From measurement set v8 (16 Nov 2026) each clip and each variant is levelled by its own camera roll,
estimated from the background of that video (capture_geometry.camera_roll_deg; cached as
<stem>.roll.json next to its pose cache). So the tilt variant is judged on the roll the estimate finds
in the rotated video (about +3 degrees), not on a roll the test already knows.
Writes tables only. Variant videos and landmarks stay in calibration_output/degrade/ (not committed).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
from phase0_baseline_table import find_video, key_of, load_csv  # noqa: E402

ANCHORS = ["setup_frame", "hands_start_up_frame", "front_foot_down_frame", "hands_peak_frame",
           "contact_frame", "follow_through_frame"]
VARIANTS = {
    "lowres": ["-vf", "scale=-2:360", "-crf", "32"],
    "lowfps": None,  # filled per clip: half the source frame rate
    "tilt": ["-vf", "rotate=3*PI/180:fillcolor=black,crop=iw*0.84:ih*0.84,scale=trunc(iw/0.84/2)*2:trunc(ih/0.84/2)*2", "-crf", "20"],
}
MAX_IQR_SHARE = 0.25
LOWRES_HEIGHT_REGISTERED = 360


def lowres_variant(height: int) -> tuple[str, list[str]]:
    """(variant name, ffmpeg args) for the lowres variant at this height. The registered 360 px variant
    keeps its exact name and arguments; any other height gets its own name and is never scaled up."""
    if height == LOWRES_HEIGHT_REGISTERED:
        return "lowres", VARIANTS["lowres"]
    return f"lowres{height}", ["-vf", f"scale=-2:'min(ih,{height})'", "-crf", "32"]


def _fps(path: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(path))
    f = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(f)


def make_variant(src: Path, dst: Path, name: str, lowres_height: int = LOWRES_HEIGHT_REGISTERED) -> float:
    """Writes the variant; returns its frame rate."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is not on PATH")
    src_fps = _fps(src)
    if name.startswith("lowres"):
        args = lowres_variant(lowres_height)[1]
    else:
        args = VARIANTS[name] or ["-vf", f"fps={src_fps / 2:.4f}", "-crf", "20"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(src), *args, "-c:v", "libx264", "-preset", "veryfast",
                          "-pix_fmt", "yuv420p", "-an", str(dst)], capture_output=True)
    if out.returncode != 0:
        raise SystemExit(f"ffmpeg failed on {src.name} ({name}): {out.stderr.decode()[-300:]}")
    return _fps(dst)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=str(SCRIPT_DIR / "calibration_output" / "degrade"))
    ap.add_argument("--mp-cache", default=str(SCRIPT_DIR / "calibration_output" / "pose_cache"))
    ap.add_argument("--anchors", default=str(SCRIPT_DIR / "anchor_truth.csv"))
    ap.add_argument("--clip-sheet", default=str(SCRIPT_DIR / "clip_sheet.csv"))
    ap.add_argument("--variants", default="lowres,lowfps,tilt")
    ap.add_argument("--lowres-height", type=int, default=LOWRES_HEIGHT_REGISTERED,
                    help="height of the lowres variant (default 360, the registered test; see the notes above)")
    args = ap.parse_args(argv)

    import warnings
    warnings.simplefilter("ignore")
    from phase0_clip_sheet import pose_facts
    import phase5_checkpoints

    vids, work, out = Path(args.video_dir).expanduser(), Path(args.work), Path(args.out)
    anchors = {key_of(r["filename"]): r for r in load_csv(Path(args.anchors)) if all(r.get(a, "").strip() for a in ANCHORS)}
    sheet = load_csv(Path(args.clip_sheet))
    sheet_by = {key_of(r["filename"]): r for r in sheet}
    lowres_name = lowres_variant(args.lowres_height)[0]
    variants = [lowres_name if v == "lowres" else v for v in args.variants.split(",") if v]

    # original measurements (validated key frames) from the production pose cache
    orig_dir = work / "original"
    phase5_checkpoints.main(["--cache-dir", args.mp_cache, "--anchors", args.anchors, "--auto-table", str(work / "none.csv"),
                             "--clip-sheet", args.clip_sheet, "--out", str(orig_dir), "--video-dir", str(vids)])

    rows_anchor, rows_sheet = [], []
    for k, a in sorted(anchors.items()):
        src = find_video(vids, a["filename"])
        if src is None:
            print("  not found:", a["filename"])
            continue
        src_fps = _fps(src)
        for v in variants:
            dst = work / "videos" / f"{src.stem}__{v}.mp4"
            npz = work / "pose_cache" / f"{dst.stem}.npz"
            if not npz.exists():
                print(f"  {v}: {src.name}", flush=True)
                vfps = make_variant(src, dst, v, args.lowres_height)
                pose_facts(dst, None, work / "pose_cache")
            else:
                vfps = float(np.load(npz)["fps"])
            scale = vfps / src_fps if v == "lowfps" else 1.0
            rows_anchor.append({"filename": dst.name, **{x: int(round(int(a[x]) * scale)) for x in ANCHORS}})
            base = sheet_by.get(k, {})
            rows_sheet.append({"filename": dst.name, "handedness": base.get("handedness", "right"),
                               "drive_type": base.get("drive_type", "off")})

    for path, rows in ((work / "variant_anchors.csv", rows_anchor), (work / "variant_sheet.csv", rows_sheet)):
        with path.open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    var_dir = work / "variants"
    phase5_checkpoints.main(["--cache-dir", str(work / "pose_cache"), "--anchors", str(work / "variant_anchors.csv"),
                             "--auto-table", str(work / "none.csv"), "--clip-sheet", str(work / "variant_sheet.csv"),
                             "--out", str(var_dir), "--video-dir", str(work / "videos")])

    def values(path: Path) -> dict[tuple[str, str], float]:
        d = {}
        for r in load_csv(path):
            if r["anchor_set"] == "validated" and r["value"] != "":
                d[(key_of(r["filename"]), r["measurement"])] = float(r["value"])
        return d
    orig = values(orig_dir / "checkpoint_measurements.csv")
    var = values(var_dir / "checkpoint_measurements.csv")
    meas = sorted({m for _, m in orig})
    res = []
    for m in meas:
        ov = [val for (k, mm), val in orig.items() if mm == m]
        iqr = float(np.subtract(*np.percentile(ov, [75, 25]))) if len(ov) >= 4 else np.nan
        for v in variants:
            d = [abs(var[(f"{k}__{v}", m)] - orig[(k, m)]) for (k, mm) in orig if mm == m and (f"{k}__{v}", m) in var]
            cov = len(d) / max(1, len(ov))
            md = median(d) if d else np.nan
            ok = bool(d) and np.isfinite(iqr) and md <= MAX_IQR_SHARE * iqr
            res.append({"measurement": m, "variant": v, "clips": len(d), "coverage_vs_original": round(cov, 2),
                        "median_abs_diff": round(md, 3) if d else "", "iqr": round(iqr, 3) if np.isfinite(iqr) else "",
                        "share_of_iqr": round(md / iqr, 2) if d and iqr else "", "pass": "pass" if ok else "fail"})
    out.mkdir(parents=True, exist_ok=True)
    with (out / "degrade_test.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    lines = ["# Phase 5 test 2: does the measurement survive worse filming?", "",
             "Same clip, made worse three ways; validated key frames; production pose path. Pass: median change <= 0.25 x the spread between clips.", "",
             "| Meas. | " + " | ".join(f"{v} (change/spread)" for v in variants) + " | All pass |", "|---|" + "---|" * (len(variants) + 1)]
    for m in meas:
        rr = {r["variant"]: r for r in res if r["measurement"] == m}
        cells = [f"{rr[v]['share_of_iqr']}: {rr[v]['pass']}" for v in variants]
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {'yes' if all(rr[v]['pass'] == 'pass' for v in variants) else 'no'} |")
    (out / "degrade_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
