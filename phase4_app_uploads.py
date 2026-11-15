#!/usr/bin/env python3
"""
Phase 4: download the app's own uploads to the Mac, drop duplicates, and list the new clips.

  python phase4_app_uploads.py --manifest ~/Downloads/app_uploads_manifest.tsv \
      --clips "<the 18 calibration clips folder>" --out "<folder outside the repo>"

Duplicates are dropped in two steps:
  1. identical files (same MD5);
  2. re-uploads of a calibration clip or of another upload: similar length (within 20%; header frame counts are unreliable) and
     the same picture at 25%, 50% and 75% of the clip (32x32 greyscale thumbnails, correlation
     >= 0.90 on all three, best of +-2 neighbouring frames).
Writes uploads_report.csv next to the downloads (kept on the Mac; it lists people's names).
Nothing is uploaded anywhere.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".webm", ".avi", ".mkv"}


def signature(path: Path) -> tuple[float, list[np.ndarray]] | None:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if n <= 0:
        cap.release()
        return None
    thumbs = []
    for q in (0.25, 0.5, 0.75):
        group = []
        for off in (-2, -1, 0, 1, 2):  # a few neighbouring frames: re-encodes shift frames by one or two
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n * q) + off))
            ok, fr = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(cv2.resize(fr, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(float).ravel()
            group.append((g - g.mean()) / (g.std() + 1e-6))
        if not group:
            cap.release()
            return None
        thumbs.append(group)
    cap.release()
    return n / fps, thumbs


def same(a, b) -> bool:
    # lengths from frame counts are unreliable in some .mov files (Phase 0), so allow 20%
    if a is None or b is None or abs(a[0] - b[0]) > max(0.3, 0.2 * max(a[0], b[0])):
        return False
    # each quartile: best match between the two clips' neighbouring frames
    return all(max(float(np.mean(x * y)) for x in ga for y in gb) >= 0.90 for ga, gb in zip(a[1], b[1]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clips", required=True, help="folder with the 18 calibration clips")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out).expanduser()
    (out / "all").mkdir(parents=True, exist_ok=True)
    (out / "new").mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(Path(args.manifest).expanduser(), encoding="utf-8"), delimiter="\t"))
    print(f"{len(rows)} uploads in the manifest")
    for r in rows:
        ext = Path(r["url"].split("?")[0]).suffix.lower() or ".mov"
        dst = out / "all" / f"{r['id']}{ext}"
        r["file"] = dst
        if not dst.exists() or dst.stat().st_size == 0:
            # curl uses the Mac's own certificates (Python's urllib often lacks them on macOS)
            res = subprocess.run(["curl", "-fsSL", "--retry", "2", "-o", str(dst), r["url"]], capture_output=True, text=True)
            if res.returncode != 0:
                dst.unlink(missing_ok=True)
                r["status"] = f"download failed: {res.stderr.strip()[:120]}"
                print(f"  {r['id']} {r['name']}: download failed ({res.stderr.strip()[:120]})")
                continue
        r["md5"] = hashlib.md5(dst.read_bytes()).hexdigest()

    known = []
    for p in sorted(Path(args.clips).expanduser().iterdir()):
        if p.suffix.lower() in VIDEO_EXT:
            known.append((p.name, signature(p)))
    print(f"{len(known)} calibration clips to compare against")

    seen_md5: dict[str, str] = {}
    kept: list[tuple[str, tuple]] = []
    for r in rows:
        if r.get("status"):
            continue
        if r["md5"] in seen_md5:
            r["status"], r["duplicate_of"] = "duplicate (identical file)", seen_md5[r["md5"]]
            continue
        seen_md5[r["md5"]] = r["id"]
        sig = signature(r["file"])
        r["seconds"] = round(sig[0], 2) if sig else ""
        if sig is None:
            r["status"] = "unreadable video"
            continue
        hit = next((name for name, s in known if same(sig, s)), None)
        if hit:
            r["status"], r["duplicate_of"] = "re-upload of a calibration clip", hit
            continue
        hit = next((kid for kid, s in kept if same(sig, s)), None)
        if hit:
            r["status"], r["duplicate_of"] = "re-upload of another upload", hit
            continue
        kept.append((r["id"], sig))
        r["status"] = "NEW"
        link = out / "new" / r["file"].name
        if not link.exists():
            link.write_bytes(r["file"].read_bytes())

    with (out / "uploads_report.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["id", "name", "created", "seconds", "status", "duplicate_of", "md5"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    for k, v in Counter(r.get("status", "") for r in rows).most_common():
        print(f"  {v:3d}  {k}")
    print(f"New clips copied to {out / 'new'}. Report: {out / 'uploads_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
