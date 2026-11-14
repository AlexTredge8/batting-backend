#!/usr/bin/env python3
"""
Phase 2: inputs for the vision-language model as a fourth coach (LOCAL ONLY).

For each clip in a blind pack this writes, under <pack>/model_inputs/<ID>/:
  still_<anchor>.jpg   the six storyboard moments (validated anchors from
                       anchor_truth.csv; the Phase 0 auto anchors where no truth row exists)
  strip_<frame>.jpg    every frame from contact - 6 to contact + 6
and one prompt.md for the whole pack: the rubric, the output format, and the
instruction to name a frame for every fault.

Raw video is not sent, because Google's documented default samples video at one frame per second.
NOTHING IS SENT ANYWHERE. Sending these images to a model provider needs, first,
parental consent for junior players and a check of the provider's data terms
(Ask-first item). Reference clips (3-4 scored examples) are chosen by Alex and
added to the prompt by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from phase0_baseline_table import ANCHORS, find_video, key_of, load_csv  # noqa: E402

RUBRIC = SCRIPT_DIR / "docs" / "phase1" / "rubric_v0.json"
STRIP_HALF = 6


def extract_frames(video: Path, frames: list[int], out_paths: list[Path]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is not on PATH")
    expr = "+".join(f"eq(n\\,{f})" for f in frames)
    tmp = out_paths[0].parent / "_tmp_%03d.jpg"
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(video), "-vf", f"select='{expr}'", "-vsync", "0",
                    "-q:v", "2", str(tmp)], check=True)
    made = sorted(out_paths[0].parent.glob("_tmp_*.jpg"))
    order = sorted(range(len(frames)), key=lambda i: frames[i])
    for src, i in zip(made, order):
        src.rename(out_paths[i])


def anchors_for(filename: str, truth: dict[str, dict[str, str]], baseline_reports: Path | None) -> dict[str, int] | None:
    row = truth.get(key_of(filename))
    if row and all(row.get(a, "").strip() for a in ANCHORS):
        return {a: int(row[a]) for a in ANCHORS}
    if baseline_reports:
        for p in baseline_reports.glob("*_battingiq.json"):
            if p.name.casefold().startswith(Path(filename).stem.casefold() + "_battingiq"):
                af = json.loads(p.read_text())["metadata"]["anchor_frames"]
                return {a: int(af[a]["original_frame"]) for a in ANCHORS}
    return None


def prompt_text() -> str:
    r = json.loads(RUBRIC.read_text())
    cps = "\n".join(f"- {c['id']} ({c['pillar']}): {c['text']}" for c in r["checkpoints"])
    faults = "\n".join(f"- {f['id']}: {f['text']}" for f in r["faults_in_chain_order"])
    anchors = "\n".join(f"- {p}: 5 = {a['5']}; 15 = {a['15']}; 25 = {a['25']}" for p, a in r["pillar_anchors"].items())
    return f"""# Score one cricket drive as a qualified coach (rubric v{r['rubric_version']})

You see one front-foot drive, filmed from behind the bowler. Images: six moments (set-up, backlift
starts, front foot down, hands peak, contact, follow-through), then every frame from 6 before to 6
after contact. Score THIS shot, not the player.

Rate each checkpoint 2 (done well), 1 (partly), 0 (clear fault) or X (cannot judge from these images):
{cps}

Do not mark down what this camera cannot show: {"; ".join(r["not_judged_from_this_camera"])}.

Pillar anchors (0-25 each):
{anchors}

Faults, in chain order (pick ONE priority fault: the one that, fixed first, helps most):
{faults}

Rules: name the image (file name) that shows each fault. If you cannot see it, rate X. Score the
overall (0-100) on its own; do not add up the pillars.

[Alex adds 3-4 scored reference clips here: their six stills and his scores.]

Answer with JSON only:
{{"clip_id": "...", "C01": "2|1|0|X", "...": "...", "C12": "...", "access": 0-25, "tracking": 0-25,
"stability": 0-25, "flow": 0-25, "overall": 0-100, "priority_fault": "F01-F12", "confidence": "low|medium|high",
"evidence": {{"F..": "image file name"}}}}
"""


def _write_references(pack: Path, key: list[dict], ground_truth: Path) -> None:
    """Three reference clips with Alex's original scores (highest, middle and lowest overall among clips with
    inputs). The model sees them as worked examples, so they are left out of its own marking."""
    gt = {key_of(r["filename"]): r for r in load_csv(ground_truth)}
    have = [k for k in key if (pack / "model_inputs" / k["clip_id"] / "still_contact.jpg").exists() and key_of(k["filename"]) in gt]
    have.sort(key=lambda k: int(gt[key_of(k["filename"])]["overall_score"]))
    if len(have) < 6:
        return
    picks = [have[-1], have[len(have) // 2], have[0]]
    refs = []
    for k in picks:
        g = gt[key_of(k["filename"])]
        refs.append({"clip_id": k["clip_id"], "overall": int(g["overall_score"]),
                     **{p: int(g[f"{p}_score"]) for p in ("access", "tracking", "stability", "flow")}})
    (pack / "model_inputs" / "references.json").write_text(json.dumps(refs, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", required=True, help="blind pack folder (with blind_key_PRIVATE.csv)")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--anchor-truth", default=str(SCRIPT_DIR / "anchor_truth.csv"))
    parser.add_argument("--baseline-reports", default=None, help="baselines/phase0_<date>/reports/auto")
    parser.add_argument("--ground-truth", default=str(SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"))
    args = parser.parse_args(argv)

    pack = Path(args.pack).expanduser()
    truth = {key_of(r["filename"]): r for r in load_csv(Path(args.anchor_truth))}
    reports = Path(args.baseline_reports) if args.baseline_reports else None
    with (pack / "blind_key_PRIVATE.csv").open(encoding="utf-8-sig", newline="") as h:
        key = list(csv.DictReader(h))
    done, skipped = 0, []
    for row in key:
        video = find_video(Path(args.video_dir).expanduser(), row["filename"])
        anchors = anchors_for(row["filename"], truth, reports)
        if video is None or anchors is None:
            skipped.append(row["clip_id"])
            continue
        out = pack / "model_inputs" / row["clip_id"]
        out.mkdir(parents=True, exist_ok=True)
        c = anchors["contact_frame"]
        strip = list(range(max(0, c - STRIP_HALF), c + STRIP_HALF + 1))
        frames = [anchors[a] for a in ANCHORS] + strip
        paths = [out / f"still_{a.replace('_frame', '')}.jpg" for a in ANCHORS] + [out / f"strip_{f:04d}.jpg" for f in strip]
        uniq = sorted(set(frames))
        tmp_paths = [out / f"_f{f:04d}.jpg" for f in uniq]
        extract_frames(video, uniq, tmp_paths)
        for f, p in zip(frames, paths):
            src = out / f"_f{f:04d}.jpg"
            if src.exists():
                shutil.copyfile(src, p)
        for p in tmp_paths:
            p.unlink(missing_ok=True)
        done += 1
    (pack / "model_inputs" / "prompt.md").write_text(prompt_text(), encoding="utf-8")
    _write_references(pack, key, Path(args.ground_truth))
    print(f"Model inputs for {done} clips in {pack / 'model_inputs'}" + (f"; skipped (no anchors or file): {', '.join(skipped)}" if skipped else ""))
    print("Nothing was sent. Sending needs parental consent for juniors and a check of the provider's data terms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
