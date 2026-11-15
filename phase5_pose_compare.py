#!/usr/bin/env python3
"""
Pose-model comparison: MediaPipe Pose (current, heavy) against RTMPose (open source, Apache-2.0,
CPU via ONNX Runtime; body + feet, 26 points). Approved by Alex 15 Nov 2026 as an evaluation;
production keeps MediaPipe unless this shows a clear win.

Step 1 (extract): RTMPose landmarks for each clip, saved in the same .npz layout as the
MediaPipe pose cache (33 MediaPipe slots; points RTMPose does not have are NaN). The batter is
the tallest person in the first frame with a detection, then followed frame to frame (nearest
skeleton), so a second person in the picture is not picked up.
Step 2 (compare), per pose model:
  - detection rate (frames with the batter found);
  - jitter at set-up: median frame-to-frame movement of nose, shoulders and hips while the
    batter stands still (1000-700 ms before validated contact), % body length (lower = steadier);
  - jumps: frames where the nose moves more than 25% of body length in one frame (a sign of
    switching to another person or a bad frame);
  - the Phase 5 measurements and entry tests (coverage, automatic key frames, model ratings).

  python phase5_pose_compare.py --video-dir <clips> --out baselines/phase5_pose_compare_<date>
      [--mp-cache calibration_output/pose_cache] [--rtm-cache calibration_output/pose_cache_rtm]
      [--mode balanced|performance|lightweight]

Writes only tables and a report; landmarks stay in calibration_output/ (not committed).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase0_baseline_table import find_video, key_of, load_csv  # noqa: E402

# Halpe-26 (RTMPose body+feet) -> MediaPipe Pose 33 slots. Both use the person's own left/right.
HALPE_TO_MP = {0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24,
               13: 25, 14: 26, 15: 27, 16: 28, 24: 29, 25: 30, 20: 31, 21: 32}
NOSE, LSH, RSH, LHIP, RHIP, LANK, RANK = 0, 11, 12, 23, 24, 27, 28


def _pick(kps: np.ndarray, scores: np.ndarray, prev: np.ndarray | None) -> int | None:
    if len(kps) == 0:
        return None
    if prev is None:
        good = scores > 0.3
        heights = [np.ptp(k[g, 1]) if g.sum() >= 5 else 0 for k, g in zip(kps, good)]
        return int(np.argmax(heights))
    d = [np.nanmean(np.linalg.norm(k - prev, axis=1)) for k in kps]
    return int(np.argmin(d))


def extract_rtm(video: Path, out_npz: Path, mode: str) -> None:
    import cv2
    from rtmlib import BodyWithFeet
    model = BodyWithFeet(mode=mode, backend="onnxruntime", device="cpu")
    cap = cv2.VideoCapture(str(video))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # same orientation as pose_extractor
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, idx, prev, w, h = [], 0, None, None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        kps, scores = model(frame)
        arr = np.full((33, 4), np.nan, dtype=np.float32)
        i = _pick(np.asarray(kps), np.asarray(scores), prev) if kps is not None and len(kps) else None
        if i is not None:
            k, s = np.asarray(kps)[i], np.asarray(scores)[i]
            jump = prev is not None and np.nanmedian(np.linalg.norm(k - prev, axis=1)) > 0.25 * h
            if not jump:
                prev = k
            for hi, mi in HALPE_TO_MP.items():
                arr[mi] = [k[hi, 0] / w, k[hi, 1] / h, 0.0, float(np.clip(s[hi], 0, 1))]
        rows.append(arr)
        idx += 1
    cap.release()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, landmarks=np.stack(rows), frame_idx=np.arange(idx), fps=fps, width=w, height=h)


def quality(npz: Path, contact: int | None) -> dict:
    d = np.load(npz)
    a = d["landmarks"].astype(float)
    w, h, fps = float(d["width"]), float(d["height"]), float(d["fps"])
    xy = np.stack([a[..., 0] * w, a[..., 1] * h], axis=-1)
    xy[a[..., 3] < 0.2] = np.nan
    det = float(np.mean(np.isfinite(xy[:, NOSE, 0])))
    bl = np.nanmedian(np.nanmean(xy[:, [LANK, RANK], 1], axis=1) - xy[:, NOSE, 1])
    nose_step = np.linalg.norm(np.diff(xy[:, NOSE], axis=0), axis=1)
    jumps = int(np.nansum(nose_step > 0.25 * bl)) if np.isfinite(bl) else ""
    jit = ""
    if contact is not None and np.isfinite(bl):
        s0, s1 = contact - int(round(fps)), contact - int(round(0.7 * fps))
        seg = xy[max(0, s0): max(0, s1) + 1][:, [NOSE, LSH, RSH, LHIP, RHIP]]
        if len(seg) > 3:
            st = np.linalg.norm(np.diff(seg, axis=0), axis=2)
            jit = round(100 * float(np.nanmedian(st)) / bl, 3)
    return {"detection_rate": round(det, 3), "body_length_px": round(float(bl), 1) if np.isfinite(bl) else "",
            "setup_jitter_pct_bl": jit, "nose_jumps": jumps}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mp-cache", default=str(SCRIPT_DIR / "calibration_output" / "pose_cache"))
    ap.add_argument("--rtm-cache", default=str(SCRIPT_DIR / "calibration_output" / "pose_cache_rtm"))
    ap.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    ap.add_argument("--ground-truth", default=str(SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"))
    ap.add_argument("--anchors", default=str(SCRIPT_DIR / "anchor_truth.csv"))
    args = ap.parse_args(argv)

    vids, rtm_cache, mp_cache, out = Path(args.video_dir).expanduser(), Path(args.rtm_cache), Path(args.mp_cache), Path(args.out)
    names = [r["filename"] for r in load_csv(Path(args.ground_truth))]
    for n in names:
        v = find_video(vids, n)
        if v is None:
            print("  not found:", n)
            continue
        dst = rtm_cache / f"{v.stem}.npz"
        if not dst.exists():
            print("  RTMPose:", v.name, flush=True)
            extract_rtm(v, dst, args.mode)

    contact = {key_of(r["filename"]): int(r["contact_frame"]) for r in load_csv(Path(args.anchors)) if r.get("contact_frame", "").strip()}
    out.mkdir(parents=True, exist_ok=True)
    qrows = []
    for label, cache in (("mediapipe", mp_cache), ("rtmpose", rtm_cache)):
        for npz in sorted(cache.glob("*.npz")):
            qrows.append({"pose_model": label, "filename": npz.stem, **quality(npz, contact.get(key_of(npz.stem)))})
    with (out / "pose_quality.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(qrows[0].keys()))
        w.writeheader()
        w.writerows(qrows)

    import phase5_checkpoints
    import phase5_entry_tests
    for label, cache in (("mediapipe", mp_cache), ("rtmpose", rtm_cache)):
        phase5_checkpoints.main(["--cache-dir", str(cache), "--out", str(out / label)])
        phase5_entry_tests.main(["--measurements", str(out / label / "checkpoint_measurements.csv"), "--out", str(out / label)])

    def summ(label: str) -> dict:
        q = [r for r in qrows if r["pose_model"] == label]
        num = lambda k: [float(r[k]) for r in q if r[k] != ""]  # noqa: E731
        tests = load_csv(out / label / "entry_tests.csv")
        return {"clips": len(q), "det": np.mean(num("detection_rate")), "jit": np.median(num("setup_jitter_pct_bl")),
                "jumps": int(sum(num("nose_jumps"))), "t3": sum(r["test3_auto_anchors"] == "pass" for r in tests),
                "cov": np.mean([float(r["coverage"]) for r in tests]),
                "worst": min(q, key=lambda r: float(r["detection_rate"]))}
    s = {k: summ(k) for k in ("mediapipe", "rtmpose")}
    lines = ["# Pose model comparison: MediaPipe against RTMPose", "",
             f"RTMPose mode: {args.mode}. Same clips, same key frames, same measurements and tests.", "",
             "| | MediaPipe (current) | RTMPose |", "|---|---|---|",
             f"| Clips | {s['mediapipe']['clips']} | {s['rtmpose']['clips']} |",
             f"| Mean detection rate | {s['mediapipe']['det']:.0%} | {s['rtmpose']['det']:.0%} |",
             f"| Worst clip | {s['mediapipe']['worst']['filename']} {float(s['mediapipe']['worst']['detection_rate']):.0%} | "
             f"{s['rtmpose']['worst']['filename']} {float(s['rtmpose']['worst']['detection_rate']):.0%} |",
             f"| Set-up jitter (median, % body length per frame; lower is steadier) | {s['mediapipe']['jit']:.2f} | {s['rtmpose']['jit']:.2f} |",
             f"| Nose jumps over 25% body length (all clips) | {s['mediapipe']['jumps']} | {s['rtmpose']['jumps']} |",
             f"| Measurements passing the automatic-key-frame test (of 13) | {s['mediapipe']['t3']} | {s['rtmpose']['t3']} |",
             f"| Mean measurement coverage | {s['mediapipe']['cov']:.0%} | {s['rtmpose']['cov']:.0%} |", "",
             "Per clip: `pose_quality.csv`. Per measurement: `mediapipe/entry_tests.md`, `rtmpose/entry_tests.md`.",
             "Key frames come from the MediaPipe pipeline in both columns, so this compares landmarks, not detectors."]
    (out / "pose_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
