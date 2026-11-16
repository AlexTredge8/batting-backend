#!/usr/bin/env python3
"""
Phase 3 clip check: can Gemini answer the clip-level questions the app's filming gate needs?

Gemini failed as a technique judge (baselines/phase5_video_judge_v2_2026-11-16/README.md). This
asks it something easier: one structured question set per clip about what the clip shows (shot,
camera position, handedness, feed, venue, whole batter visible, phone level, people in frame,
one shot or several, batter size, usable or not). Every clip in the swipe file (18), uncropped,
at 8 frames/s, temperature 0, two runs per clip (36 requests) to measure run-to-run consistency.

The answers are compared with what the repo already knows:
  - clip_sheet.csv: handedness (confirmed by Alex), drive_type (from the filename where it names
    the shot, else a hub guess), batter_height_pct (measured), and feed, venue, camera_position,
    others_in_frame (the hub's guesses from 14 Nov: NOT confirmed; reported, but labelled so);
  - known facts: Ollie P has two people in frame; Drive_Beginner 4 has poor pose detection;
    Female_Average was marked "not filmed correctly" by the coach;
  - the swipe file itself: every clip is one front-foot drive, trimmed to that one shot.

Per field: agreement with that ground truth, run-to-run consistency, and a verdict on whether the
field is reliable enough for an automatic filming check. Writes answers.csv, comparison.csv,
field_summary.csv, summary.md and responses.jsonl (filenames only: no clip codes, no video).

Reuses phase5_video_judge_v2.py (clip loading, optional crop, cost guard, retries, model choice,
key handling, videoMetadata fps) by import; that file is not changed.

  python phase3_clip_check.py --clips-html BattingIQ_which_drive.html --key BattingIQ_which_drive_key_PRIVATE.csv \\
      --out baselines/phase3_clip_check_<date> [--dry-run]

Key: GEMINI_API_KEY (x-goog-api-key header). Paid-tier key only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import phase5_video_judge_v2 as v2  # noqa: E402
from phase0_baseline_table import key_of, load_csv  # noqa: E402

v1 = v2.v1
CLIP_SHEET = SCRIPT_DIR / "clip_sheet.csv"
COACH_GT = SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"
GBP_PER_USD = v2.GBP_PER_USD
ANSWER_TOKENS = 350  # thirteen short fields
SIZE_BANDS = (20.0, 40.0)  # batter height as % of picture height: small < 20 <= medium < 40 <= large (gate v0: >= 20)

# Field name -> allowed answers (None = integer or free text). Order = order in the answer.
FIELDS: dict[str, list[str] | None] = {
    "shot_type": ["front-foot drive", "back-foot shot", "defensive", "other"],
    "drive_direction": ["off", "straight", "on", "not a drive"],
    "camera_position": ["behind the bowler", "side-on", "behind the batter", "other"],
    "batter_handedness": ["right", "left"],
    "feed": ["bowling machine", "bowler", "throw-down", "unclear"],
    "venue": ["indoor net lane", "sports hall", "outdoor net", "middle of a ground", "other"],
    "whole_batter_visible": ["yes", "no"],
    "phone_level": ["yes", "tilted"],
    "number_of_people_visible": None,
    "multiple_shots_in_clip": ["yes", "no"],
    "batter_size": ["small", "medium", "large"],
    "usable_reason": None,
    "usable_for_analysis": ["yes", "no"],
}
COMPARED = [f for f in FIELDS if f != "usable_reason"]

SCHEMA = {
    "type": "OBJECT",
    "properties": {name: ({"type": "STRING", "enum": allowed} if allowed else
                          {"type": "INTEGER"} if name == "number_of_people_visible" else {"type": "STRING"})
                   for name, allowed in FIELDS.items()},
    "required": list(FIELDS),
    "propertyOrdering": list(FIELDS),
}

PROMPT = """You are checking a short phone video of a cricket batter at practice, before any coaching
analysis is run on it. Do not judge technique. Answer only what the video shows. Faces are blurred
on purpose; the clip may be scaled down and has no sound.

Answer every field:
- shot_type: the shot played. front-foot drive = steps forward onto the front foot and drives the ball
  along the ground or in the air with a full swing; back-foot shot = weight goes back (pull, cut,
  back-foot punch); defensive = blocks the ball with little or no swing; other = anything else (leave,
  miss, no ball faced).
- drive_direction: if it is a drive, where it goes relative to the batter: off = through the off side
  (the side the batter faces in the stance: cover / extra cover); straight = back past the bowler
  (mid-off to mid-on); on = through the leg side (mid-on / midwicket). Otherwise "not a drive".
- camera_position: behind the bowler = the camera looks down the pitch at the batter from the bowler's
  end, in line with the stumps (within about 15 degrees); side-on = level with the batter, looking
  across the pitch (from point or square leg); behind the batter = from behind the batter's stumps,
  looking towards the bowler; other = anything else, including from the bowler's end but clearly off
  to one side (more than about 15 degrees).
- batter_handedness: right = right-handed (left shoulder towards the bowler, right hand lower on the
  handle); left = left-handed.
- feed: how the ball was delivered: bowling machine; bowler (a bowler running in and bowling);
  throw-down (someone throwing or using a ball-thrower from a short distance); unclear if you cannot tell.
- venue: indoor net lane (a netted lane indoors, usually artificial pitch and lights); sports hall
  (a large indoor hall, not a dedicated net lane); outdoor net (outdoor practice nets); middle of a
  ground (an open cricket field); other.
- whole_batter_visible: yes if the batter's whole body (head to feet) and bat stay in the picture for
  the whole shot; no if any part is cut off by the edge of the picture or hidden for part of it.
- phone_level: yes if the picture is upright (vertical lines such as net poles, stumps and walls look
  vertical); tilted if the picture is visibly rotated (more than about 5 degrees).
- number_of_people_visible: everyone you can see at any point, including the batter, the bowler or
  thrower, and people in other lanes or in the background, even if only partly visible.
- multiple_shots_in_clip: yes if the batter faces more than one ball in this clip; no if only one.
- batter_size: how tall the batter is in the picture in the stance (head to feet) compared with the
  picture's height: small = under a fifth; medium = a fifth to two fifths; large = over two fifths.
- usable_reason: one short line: what, if anything, would stop a coach or a pose-tracking program
  analysing this batter's technique from this clip.
- usable_for_analysis: yes if this clip is good enough to analyse the batter's technique (one batter,
  clearly visible, whole body, from a suitable angle, one shot); no otherwise."""

ASK = "Answer the questions about this clip."


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

CONFIRMED, MEASURED, KNOWN, BY_SET, FILENAME, HUB = (
    "confirmed", "measured", "known", "by construction", "filename", "hub guess, not confirmed")

FEED_MAP = {"machine": "bowling machine", "bowler": "bowler", "throw-down": "throw-down"}
VENUE_MAP = {"indoor net lane": "indoor net lane", "sports hall": "sports hall", "outdoor nets": "outdoor net",
             "outdoor net": "outdoor net"}
TWO_PEOPLE_KNOWN = {key_of("Ollie P_Offdrive Average.mov")}
POOR_DETECTION_KNOWN = {key_of("Drive_Beginner 4.MOV")}


@dataclass
class Truth:
    expected: str          # the right answer for the comparison ("" = none)
    source: str            # where it comes from (CONFIRMED, HUB, ...)
    note: str = ""


def size_band(pct: float) -> str:
    lo, hi = SIZE_BANDS
    return "small" if pct < lo else "medium" if pct < hi else "large"


def direction_in_filename(name: str) -> str | None:
    s = Path(name).stem.lower().replace(" ", "")
    for token, d in (("offdrive", "off"), ("ondrive", "on"), ("straightdrive", "straight")):
        if token in s:
            return d
    return None


def truth_for(sheet_row: dict | None, coach_row: dict | None, filename: str) -> dict[str, Truth]:
    """Ground truth per compared field for one clip. Fields with no ground truth are left out."""
    r = sheet_row or {}
    k = key_of(filename)
    t: dict[str, Truth] = {
        "shot_type": Truth("front-foot drive", BY_SET, "every clip in the swipe file is a front-foot drive"),
        "multiple_shots_in_clip": Truth("no", BY_SET, "clips are trimmed to one shot"),
    }
    named = direction_in_filename(filename)
    if named:
        t["drive_direction"] = Truth(named, FILENAME)
    elif r.get("drive_type"):
        t["drive_direction"] = Truth(r["drive_type"].strip().lower(), HUB)
    if r.get("handedness"):
        t["batter_handedness"] = Truth(r["handedness"].strip().lower(), CONFIRMED)
    cam = (r.get("camera_position") or "").strip().lower()
    if cam:
        t["camera_position"] = Truth("other" if cam.startswith("angle") else "behind the bowler" if "behind bowler" in cam else cam,
                                     HUB, r["camera_position"])
    feed = (r.get("feed") or "").strip().lower()
    if feed in FEED_MAP:
        t["feed"] = Truth(FEED_MAP[feed], HUB)
    venue = (r.get("venue") or "").strip().lower()
    if venue in VENUE_MAP:
        t["venue"] = Truth(VENUE_MAP[venue], HUB)
    others = (r.get("others_in_frame") or "").strip().upper()
    if k in TWO_PEOPLE_KNOWN:
        t["number_of_people_visible"] = Truth("2+", KNOWN, "two people in frame")
    elif others in ("Y", "N"):
        t["number_of_people_visible"] = Truth("2+" if others == "Y" else "1", HUB)
    try:
        pct = float(r.get("batter_height_pct") or "")
    except ValueError:
        pct = float("nan")
    if not math.isnan(pct) and pct > 0:
        t["batter_size"] = Truth(size_band(pct), MEASURED, f"{pct:.1f}% (nose to ankle)")
    if k in TWO_PEOPLE_KNOWN:
        t["usable_for_analysis"] = Truth("no", KNOWN, "two people in frame")
    elif k in POOR_DETECTION_KNOWN:
        t["usable_for_analysis"] = Truth("no", KNOWN, f"pose found in {r.get('pose_detection_rate') or '?'}% of frames")
    elif coach_row and (coach_row.get("filmed_correctly") or "").upper() == "N":
        t["usable_for_analysis"] = Truth("no", CONFIRMED, "coach: not filmed correctly")
    elif coach_row:
        t["usable_for_analysis"] = Truth("yes", CONFIRMED, "coach scored it as filmed correctly")
    return t


def matches(field: str, expected: str, answer) -> bool | None:
    """Does one answer agree with the ground truth? None when there is no answer."""
    if answer in (None, ""):
        return None
    if field == "number_of_people_visible":
        try:
            n = int(answer)
        except (TypeError, ValueError):
            return None
        return n >= 2 if expected == "2+" else n == int(expected)
    return str(answer).strip().lower() == expected


def same(a, b) -> bool | None:
    if a in (None, "") or b in (None, ""):
        return None
    return str(a).strip().lower() == str(b).strip().lower()


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

def estimate_usd(secs: float, fps: float, thinking: str | None, media_resolution: str | None,
                 usd_in: float, usd_out: float) -> float:
    tpf = v2.TOKENS_PER_FRAME.get(media_resolution or "default", 100)
    tokens_in = (len(PROMPT) + len(ASK)) / 3.5 + math.ceil(secs * fps) * tpf
    tokens_out = ANSWER_TOKENS + v2.THOUGHT_TOKENS.get(thinking, 1500)
    return (tokens_in * usd_in + tokens_out * usd_out) / 1e6


def build_jobs(names: dict[str, str], clips: dict[str, bytes], secs: dict[str, float], args) -> list[v2.Job]:
    """One request per clip per run; all of run 1 first, so a run stopped by the cost guard still covers every clip once."""
    order = sorted(names, key=lambda c: names[c].casefold())
    jobs = []
    for run in range(1, args.runs + 1):
        for cid in order:
            body = {
                "systemInstruction": {"parts": [{"text": PROMPT}]},
                "contents": [{"role": "user", "parts": [v2.video_part(clips[cid], args.video_fps), {"text": ASK}]}],
                "generationConfig": v2.generation_config(SCHEMA, args.thinking, args.media_resolution, args.temperature),
            }
            est = estimate_usd(secs.get(cid, 4.0), args.video_fps, args.thinking, args.media_resolution,
                               args.usd_per_mtok_in, args.usd_per_mtok_out)
            jobs.append(v2.Job("clip_check", {"filename": names[cid], "run": run}, body, est, [cid]))
    return jobs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

AGREE_OK, CONSIST_OK = 0.90, 0.90          # reliable enough for an automatic check
AGREE_SOFT, CONSIST_SOFT = 0.75, 0.85      # good enough for a soft prompt, not a block
MIN_SCORED = 5


def clean_answer(ans: dict) -> dict:
    out = {}
    for f, allowed in FIELDS.items():
        v = ans.get(f)
        if v is None:
            out[f] = ""
        elif f == "number_of_people_visible":
            try:
                out[f] = int(v)
            except (TypeError, ValueError):
                out[f] = ""
        elif allowed is not None:
            s = str(v).strip().lower()
            out[f] = s if s in allowed else ""
        else:
            out[f] = re.sub(r"\s+", " ", str(v)).strip()
    return out


def analyse(results: list[dict], files: Sequence[str], truths: dict[str, dict[str, Truth]], runs: int
            ) -> tuple[list[dict], list[dict], list[dict], dict]:
    """answers rows, comparison rows (clip x field), field summary rows, overall stats."""
    by_clip: dict[str, dict[int, dict]] = {f: {} for f in files}
    answers = []
    for res in results:
        fname, run = res["job"].meta["filename"], res["job"].meta["run"]
        a = clean_answer(res["answer"]) if res["answer"] else {}
        if a:
            by_clip.setdefault(fname, {})[run] = a
        u = res.get("usage") or {}
        answers.append({"filename": fname, "run": run, **{f: a.get(f, "") for f in FIELDS}, "error": res.get("error", ""),
                        "prompt_tokens": u.get("prompt", ""), "answer_tokens": u.get("candidates", ""),
                        "thinking_tokens": u.get("thoughts", "")})

    comparison, per_field = [], {f: {"match": [], "match_conf": [], "consistent": [], "sources": Counter(),
                                     "truth_classes": set(), "answers": Counter(), "misses": []} for f in COMPARED}
    for fname in files:
        got = by_clip.get(fname, {})
        t = truths.get(fname, {})
        for f in COMPARED:
            vals = [got.get(r, {}).get(f, "") for r in range(1, runs + 1)]
            tr = t.get(f)
            ms = [matches(f, tr.expected, v) if tr else None for v in vals]
            cons = same(vals[0], vals[1]) if runs >= 2 else None
            row = {"filename": fname, "field": f, "expected": tr.expected if tr else "", "source": tr.source if tr else "none",
                   "truth_note": tr.note if tr else ""}
            row.update({f"run{r}": vals[r - 1] for r in range(1, runs + 1)})
            row.update(consistent="" if cons is None else int(cons),
                       agrees="/".join("" if m is None else str(int(m)) for m in ms))
            comparison.append(row)
            pf = per_field[f]
            for v in vals:
                if v != "":
                    pf["answers"][str(v)] += 1
            if cons is not None:
                pf["consistent"].append(cons)
            if tr:
                pf["sources"][tr.source] += 1
                pf["truth_classes"].add(tr.expected)
                for m in ms:
                    if m is not None:
                        pf["match"].append(m)
                        if tr.source != HUB:
                            pf["match_conf"].append(m)
                if any(m is False for m in ms):
                    pf["misses"].append(fname)

    summary = []
    for f in COMPARED:
        pf = per_field[f]
        agree = _share(pf["match"])
        consist = _share(pf["consistent"])
        unconfirmed = pf["sources"][HUB] * 2 > sum(pf["sources"].values())  # mostly hub guesses
        summary.append({
            "field": f,
            "truth_source": "; ".join(f"{s} ({n})" for s, n in pf["sources"].most_common()) or "none",
            "agreement": agree, "answers_scored": len(pf["match"]),
            "agreement_excl_hub_guesses": _share(pf["match_conf"]), "answers_scored_excl_hub": len(pf["match_conf"]),
            "consistency": consist, "clips_both_runs": len(pf["consistent"]),
            "answer_counts": "; ".join(f"{k}: {n}" for k, n in sorted(pf["answers"].items())),
            "disagreeing_clips": "; ".join(pf["misses"]),
            "verdict": verdict(agree, len(pf["match"]), consist, len(pf["consistent"]), bool(unconfirmed),
                               len(pf["truth_classes"]), f),
        })
    stats = {"clips_answered": sum(1 for f in files if by_clip.get(f)),
             "clips_both_runs": sum(1 for f in files if len(by_clip.get(f, {})) >= min(runs, 2)),
             "errors": sum(1 for r in results if r.get("error"))}
    return answers, comparison, summary, stats


def _share(xs: list[bool]) -> float | None:
    return sum(xs) / len(xs) if xs else None


GATE_FIELDS = ("camera_position", "whole_batter_visible", "phone_level", "number_of_people_visible",
               "multiple_shots_in_clip", "batter_size", "usable_for_analysis", "batter_handedness")


def verdict(agree: float | None, n_scored: int, consist: float | None, n_consist: int, unconfirmed: bool,
            truth_classes: int, field: str) -> str:
    """Plain verdict for one field. Strict on purpose: a filming gate that is wrong turns players away."""
    if consist is not None and n_consist >= MIN_SCORED and consist < CONSIST_SOFT:
        return "not reliable (answers change between identical runs)"
    if agree is None or n_scored < MIN_SCORED:
        if consist is None:
            return "no data"
        return ("consistent, but accuracy unknown (no ground truth)" if consist >= CONSIST_OK
                else "accuracy unknown (no ground truth); not consistent enough")
    tail = ""
    if truth_classes < 2:
        tail = "; only one right answer in this set, so this measures false alarms only, not whether it catches the problem"
    pre = "provisional (ground truth is a hub guess): " if unconfirmed else ""
    if agree >= AGREE_OK and (consist is None or consist >= CONSIST_OK):
        return pre + "reliable enough for an automatic check" + tail
    if agree >= AGREE_SOFT and (consist is None or consist >= CONSIST_SOFT):
        return pre + "use only as a soft prompt, not a block" + tail
    return pre + "not reliable" + tail


def _fmt(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def recommendation(summary: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """(use as a block, use as a soft prompt, do not use) among the fields a filming gate would check."""
    use, soft, no = [], [], []
    for row in summary:
        if row["field"] not in GATE_FIELDS:
            continue
        v = row["verdict"]
        if "reliable enough" in v and "only one right answer" not in v and not v.startswith("provisional"):
            use.append(row["field"])
        elif "reliable enough" in v or "soft prompt" in v:
            soft.append(row["field"])
        else:
            no.append(row["field"])
    return use, soft, no


def summary_md(args, model: str, guard: v2.CostGuard, n_jobs: int, summary: list[dict], stats: dict) -> str:
    lines = [f"# Phase 3 clip check with Gemini ({time.strftime('%Y-%m-%d')})", "",
             f"Model {model}, {args.video_fps:g} frames/s, temperature {args.temperature:g}, {args.runs} runs per clip"
             + (f", thinking {args.thinking}" if args.thinking else "") + (", cropped around the batter" if args.crop else ", uncropped") + ".",
             f"Requests sent {guard.requests} of {n_jobs}; spent about GBP {guard.spent_gbp:.2f} (cap {args.budget_gbp:.2f}; "
             f"tokens: prompt {guard.tokens['prompt']}, answer {guard.tokens['candidates']}, thinking {guard.tokens['thoughts']}).",
             f"Clips answered {stats['clips_answered']}, both runs {stats['clips_both_runs']}; errors {stats['errors']}.", ""]
    if guard.refused:
        lines += ["**Stopped by the cost guard before all requests were sent.**", ""]
    lines += ["Agreement = share of answers (both runs) that match the ground truth. Consistency = share of clips "
              "where run 1 and run 2 gave the same answer. Fields whose only ground truth is the hub's 14 Nov guesses "
              "(feed, venue, camera position, others in frame) are **unconfirmed**: their agreement is with a guess, not a fact.",
              f"Reliable enough for an automatic check = agreement >= {AGREE_OK:.2f} and consistency >= {CONSIST_OK:.2f}; "
              f"soft prompt = >= {AGREE_SOFT:.2f} and >= {CONSIST_SOFT:.2f}; at least {MIN_SCORED} answers scored.", "",
              "| field | ground truth | agreement | excl. hub guesses | consistency | verdict |", "|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r['field']} | {r['truth_source']} | {_fmt(r['agreement'])} (n={r['answers_scored']}) | "
                     f"{_fmt(r['agreement_excl_hub_guesses'])} (n={r['answers_scored_excl_hub']}) | "
                     f"{_fmt(r['consistency'])} (n={r['clips_both_runs']}) | {r['verdict']} |")
    lines += ["", "Answer counts and clips that disagreed with the ground truth are in field_summary.csv; every "
              "answer next to its ground truth is in comparison.csv.", ""]
    notes = {r["field"]: r for r in summary}
    lines += ["## Notes on the ground truth", "",
              "- camera_position: the hub's 'angle (~20-30 deg)' clips are expected to be answered 'other' (the prompt "
              "asks for 'other' when the camera is more than about 15 degrees off the line of the stumps).",
              "- number_of_people_visible: compared as '1' or '2 or more'. Ollie P (two people) is a known fact; the rest "
              "use the hub's others_in_frame guess.",
              f"- batter_size: bands from the measured batter_height_pct (nose to ankle, so a little under full height): "
              f"small < {SIZE_BANDS[0]:g}%, medium < {SIZE_BANDS[1]:g}%, large above. Gate v0 asks for at least {SIZE_BANDS[0]:g}%. "
              "Ollie P and Drive_Beginner 4 have no usable measurement and are left out.",
              "- usable_for_analysis: 'no' for Ollie P (two people), Drive_Beginner 4 (poor pose detection) and "
              "Female_Average (coach: not filmed correctly); 'yes' for the clips the coach scored as filmed correctly.",
              "- shot_type and multiple_shots_in_clip have one right answer for every clip (all are single front-foot "
              "drives), so they can show false alarms but not whether the model catches a wrong shot or a long clip.",
              "- whole_batter_visible and phone_level have no ground truth: only consistency is measured.", ""]
    use, soft, no = recommendation(summary)
    lines += ["## Recommendation", ""]
    lines.append("- Use in the live gate as an automatic check: " + (", ".join(use) if use else "none of the fields"))
    lines.append("- Use only as a soft prompt (ask the player to confirm, never block): " + (", ".join(soft) if soft else "none"))
    lines.append("- Do not use: " + (", ".join(no) if no else "none"))
    lines += ["", "Gate checks that already have a measured source (batter size from pose, clip length, sound, resolution, "
              "frame rate) should stay measured; a model answer is only worth adding where the pipeline cannot measure "
              "the thing at all (camera angle, other people in frame, a tilted phone). With 18 clips and 2 runs, one "
              "wrong answer moves agreement by about 3 points: treat these as a first look, to be re-checked on the "
              "Phase 3 net-session footage before any check blocks a real upload."]
    if notes.get("batter_handedness", {}).get("verdict", "").startswith("reliable"):
        lines += ["", "Handedness from the model agrees with Alex's confirmed labels; it could back up the feet check."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-html", required=True)
    ap.add_argument("--key", required=True, help="the offline file's *_key_PRIVATE.csv (clip code -> filename)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-sheet", default=str(CLIP_SHEET))
    ap.add_argument("--coach-truth", default=str(COACH_GT))
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=2, help="runs per clip (consistency needs 2)")
    ap.add_argument("--video-fps", type=float, default=v1.VIDEO_FPS)
    ap.add_argument("--crop", action="store_true", help="crop around the batter first (off by default: the questions are about the whole picture)")
    ap.add_argument("--crop-pad", type=float, default=0.25)
    ap.add_argument("--crop-max-side", type=int, default=720)
    ap.add_argument("--crop-cache", default=None)
    ap.add_argument("--limit-clips", type=int, default=None, help="only the first N clips (a cheap smoke test)")
    ap.add_argument("--thinking", choices=["low", "high"], default="low", help="thinkingLevel (Gemini 3); low keeps the cost down")
    ap.add_argument("--media-resolution", choices=["default", "low", "medium", "high"], default="default")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget-gbp", type=float, default=0.80)
    ap.add_argument("--usd-per-mtok-in", type=float, default=v2.USD_PER_MTOK_IN)
    ap.add_argument("--usd-per-mtok-out", type=float, default=v2.USD_PER_MTOK_OUT)
    ap.add_argument("--dry-run", action="store_true", help="build every request and estimate the cost; send nothing")
    args = ap.parse_args(argv)
    if args.runs < 1:
        ap.error("--runs must be at least 1")
    out = Path(args.out).resolve()
    if args.crop_cache and SCRIPT_DIR in Path(args.crop_cache).resolve().parents:
        ap.error("--crop-cache holds video: put it outside the repo")
    if SCRIPT_DIR in out.parents and not str(out.relative_to(SCRIPT_DIR)).startswith("baselines"):
        ap.error("--out inside the repo must be under baselines/")
    return args


def main(argv: list[str] | None = None, post: Callable[[str, dict], dict] | None = None) -> int:
    args = parse_args(argv)
    clips = v1.clips_from_html(Path(args.clips_html))
    key_rows = load_csv(Path(args.key))
    names = {r["clip_id"]: r["filename"] for r in key_rows if r["clip_id"] in clips}
    if not names:
        raise SystemExit("No clips: does the key file belong to this HTML file?")
    if args.limit_clips:
        names = dict(sorted(names.items(), key=lambda kv: kv[1].casefold())[:args.limit_clips])
    secs = {r["clip_id"]: float(r["frames"]) / float(r["fps"]) for r in key_rows
            if r.get("frames") and r.get("fps") and float(r["fps"]) > 0}
    sheet = {key_of(r["filename"]): r for r in load_csv(Path(args.clip_sheet))}
    coach = {key_of(r["filename"]): r for r in load_csv(Path(args.coach_truth))} if Path(args.coach_truth).exists() else {}
    files = sorted(names.values(), key=str.casefold)
    truths = {f: truth_for(sheet.get(key_of(f)), coach.get(key_of(f)), f) for f in files}
    missing = [f for f in files if key_of(f) not in sheet]
    if missing:
        print(f"Note: not in the clip sheet (no ground truth beyond the clip set): {', '.join(missing)}")

    sent = dict(clips)
    if args.crop:
        cache = Path(args.crop_cache).expanduser() if args.crop_cache else None
        for cid in sorted(names, key=lambda c: names[c]):
            sent[cid], info = v2.crop_clip(clips[cid], args.crop_pad, args.crop_max_side, cache)
            print(f"  {names[cid]}: {info['src_size']} -> {info['out_size']} (crop {info['crop']})")

    jobs = build_jobs(names, sent, secs, args)
    est = sum(j.est_usd for j in jobs) * GBP_PER_USD
    sizes = [len(json.dumps(j.body)) / 1e6 for j in jobs]
    fits, run = 0, 0.0
    for j in jobs:
        run += j.est_usd * GBP_PER_USD
        if run > args.budget_gbp:
            break
        fits += 1
    print(f"Clip check: {len(jobs)} requests ({len(names)} clips x {args.runs} runs) at {args.video_fps:g} frames/s, "
          f"{'cropped' if args.crop else 'uncropped'}, thinking {args.thinking or 'model default'}; "
          f"request size {min(sizes):.2f}-{max(sizes):.2f} MB. Estimated cost about GBP {est:.2f} "
          f"(priced high on purpose; cap {args.budget_gbp:.2f}: about {fits} of {len(jobs)} requests fit).")
    if max(sizes) > 18:
        print("Warning: a request is over 18 MB; Gemini refuses inline requests over about 20 MB.")
    if args.dry_run:
        print("Dry run: nothing sent.")
        return 0

    try:
        if os.environ.get("GEMINI_API_KEY") is None and post is None:
            print("Note: GEMINI_API_KEY is not set; relying on a proxy to add the key.")
        model = args.model or v1.pick_model(v1.list_models())
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return v2.KEY_REJECTED_EXIT if "rejected" in str(exc) else 1
    print(f"Model: {model}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    guard = v2.CostGuard(args.budget_gbp, args.usd_per_mtok_in, args.usd_per_mtok_out)
    (out / "responses.jsonl").unlink(missing_ok=True)
    try:
        results = v2.run_jobs(jobs, model, guard, args.workers, post=post, jsonl=out / "responses.jsonl")
    except v2.KeyRejected as exc:
        print(exc, file=sys.stderr)
        return v2.KEY_REJECTED_EXIT

    answers, comparison, summary, stats = analyse(results, files, truths, args.runs)
    v2.write_csv(out / "answers.csv", answers)
    v2.write_csv(out / "comparison.csv", comparison)
    v2.write_csv(out / "field_summary.csv", [{k: (_fmt(v, 3) if isinstance(v, float) or v is None else v) for k, v in r.items()}
                                              for r in summary])
    (out / "summary.md").write_text(summary_md(args, model, guard, len(jobs), summary, stats), encoding="utf-8")
    use, soft, no = recommendation(summary)
    print(f"Spent about GBP {guard.spent_gbp:.2f}. Automatic check: {', '.join(use) or 'none'}; "
          f"soft prompt: {', '.join(soft) or 'none'}. Wrote {out}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
