#!/usr/bin/env python3
"""
Video-model judge, round 2: three experiments on the Gemini set-up of phase5_video_judge.py.

Run 1 (1 frame/s) ranked the 15 clips at random (Spearman 0.10 with Alex). Run 2 (8 frames/s)
reached 0.67 but its scores were bimodal (about 95 or about 30). Coaches agree with each other
at 0.87-0.95 on ORDER, so these experiments ask for order, or for narrow yes/no questions,
rather than one absolute number:

  --mode pairwise     two clips per request: "which drive is better, A or B?", confidence and
                      the main fault of each. All pairs of the 15 non-reference clips (105) or a
                      balanced subset (--max-pairs; any prefix of the order is balanced, so a run
                      stopped by the cost guard is still usable). A/B order is random per pair
                      (--seed) and recorded, so position bias cancels and can be measured.
                      Bradley-Terry fit (phase2_pairs.fit_bradley_terry); Spearman vs Alex and
                      Cam; agreement of each pick with Alex's score order.
  --mode absolute     as phase5_video_judge.py (score sheet per clip), with --refs-per-level 1|2
                      reference clips per tier (Elite, Good Club, Average, Beginner), never the
                      clip being judged.
  --mode checkpoints  four requests per clip, one per stage group of rubric v0's 12 checkpoints,
                      each checkpoint answered yes / partly / no / cannot_see. Writes C01-C12
                      (2/1/0/X) and the earliest failed checkpoint as the main fault; agreement
                      with Cam's main fault and the spread of the ratings.

  --crop              crop each clip around the batter first: MediaPipe pose on the clip, union
                      box of the body over the clip, padded 25 %, cropped and re-encoded (H.264,
                      up to 720 px on the long side) with ffmpeg. Works with every mode.

Every mode: a hard cost guard (--budget-gbp, default 1.2) on usageMetadata (prompt, answer and
thinking tokens) at prices set above Google's list price; --dry-run builds every request and
prints counts and the estimated cost without sending anything; retries with backoff on 429/5xx;
exit code 3 if Google rejects the key. Outputs (CSVs + summary.md) name clips by filename only:
no clip codes (the code -> filename key is private) and no video.

  python phase5_video_judge_v2.py --mode pairwise --crop --clips-html BattingIQ_which_drive.html \\
      --key BattingIQ_which_drive_key_PRIVATE.csv --out baselines/phase5_video_judge_v2_<date>/pairwise_crop [--dry-run]

Key: GEMINI_API_KEY (sent in the x-goog-api-key header). Paid-tier key only (see phase5_video_judge.py).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import phase5_video_judge as v1  # noqa: E402
from phase0_baseline_table import key_of, load_csv  # noqa: E402
from phase2_pairs import fit_bradley_terry  # noqa: E402

API = v1.API
CPS = v1.CPS
FAULTS = [f"F{i:02d}" for i in range(13)]
TIERS = ("Elite", "Good Club", "Average", "Beginner")  # high to low, as the references are shown
CAM_QUICK = SCRIPT_DIR / "baselines" / "phase2_preliminary_2026-11-14" / "cam_quick_check.csv"
GBP_PER_USD = v1.GBP_PER_USD  # 0.80: above the real rate, so the guard errs high

# Cost assumptions, all set high on purpose. Gemini 3 Pro preview lists USD 2 / 12 per million
# input / output tokens (prompts under 200k). Run 2 (8 frames/s, gemini-3.1-pro-preview) counted
# about 66-70 prompt tokens per sampled frame; Gemini 2.5 counts 258 at default resolution.
USD_PER_MTOK_IN = 2.5
USD_PER_MTOK_OUT = 15.0  # answer and thinking tokens are both billed as output
TOKENS_PER_FRAME = {"default": 100, "low": 100, "medium": 100, "high": 300}
THOUGHT_TOKENS = {None: 1500, "low": 600, "high": 2000}  # run 1's spend implies about 1200 per request
ANSWER_TOKENS = {"pairwise": 250, "absolute": 400, "checkpoints": 200}
RETRY_CODES = {429, 500, 502, 503, 504}
KEY_REJECTED_EXIT = 3


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Context:
    clips: dict[str, bytes]          # clip code -> mp4 bytes (from the offline HTML)
    names: dict[str, str]            # clip code -> filename (private key)
    secs: dict[str, float]           # clip code -> duration in seconds
    gt: dict[str, dict]              # key_of(filename) -> ground-truth row (Alex)
    cam: dict[str, dict]             # key_of(filename) -> Cam's quick-check row
    targets: list[str]               # clip codes judged (all but the three original references)

    def alex(self, cid: str) -> float | None:
        r = self.gt.get(key_of(self.names[cid]))
        return float(r["overall_score"]) if r and r.get("overall_score") else None

    def cam_score(self, cid: str) -> float | None:
        r = self.cam.get(key_of(self.names[cid]))
        return float(r["overall"]) if r and r.get("overall") not in (None, "") else None


def load_context(clips_html: Path, key_csv: Path, ground_truth: Path, cam_csv: Path, limit: int | None = None) -> Context:
    clips = v1.clips_from_html(clips_html)
    key_rows = load_csv(key_csv)
    names = {r["clip_id"]: r["filename"] for r in key_rows if r["clip_id"] in clips}
    secs = {r["clip_id"]: float(r["frames"]) / float(r["fps"]) for r in key_rows
            if r.get("frames") and r.get("fps") and float(r["fps"]) > 0}
    gt = {key_of(r["filename"]): r for r in load_csv(ground_truth)}
    cam = {key_of(r["filename"]): r for r in load_csv(cam_csv)} if cam_csv.exists() else {}
    refs = {key_of(f) for f in v1.REFERENCES}
    targets = sorted(cid for cid in names if key_of(names[cid]) not in refs)
    if limit:
        targets = targets[:limit]
    if not targets:
        raise SystemExit("No clips to judge: does the key file belong to this HTML file?")
    return Context(clips, names, secs, gt, cam, targets)


# ---------------------------------------------------------------------------
# Crop around the batter
# ---------------------------------------------------------------------------

def crop_box(landmarks: np.ndarray, width: int, height: int, pad: float = 0.25, min_vis: float = 0.5,
             min_frames: int = 3) -> tuple[int, int, int, int] | None:
    """Union box of the body over the clip, padded, in pixels: (x, y, w, h), even sizes, inside the frame.

    landmarks: [frames, 33, 4] normalised (x, y, z, visibility), NaN where no pose was found.
    Frames whose body centre jumps more than one body size from the median (pose briefly on
    someone else) are left out. None if too few frames have a pose."""
    boxes = []
    for f in np.asarray(landmarks, dtype=float):
        ok = ~np.isnan(f[:, 0]) & ~np.isnan(f[:, 1]) & (np.nan_to_num(f[:, 3]) >= min_vis)
        if ok.sum() < 8:
            continue
        xs, ys = np.clip(f[ok, 0], 0, 1), np.clip(f[ok, 1], 0, 1)
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    if len(boxes) < min_frames:
        return None
    b = np.array(boxes)
    centres = (b[:, :2] + b[:, 2:]) / 2
    size = np.maximum(np.median(b[:, 2:] - b[:, :2], axis=0), 0.05)
    keep = np.all(np.abs(centres - np.median(centres, axis=0)) <= size, axis=1)
    b = b[keep]
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = max(0.0, x0 - pad * bw) * width, min(1.0, x1 + pad * bw) * width
    y0, y1 = max(0.0, y0 - pad * bh) * height, min(1.0, y1 + pad * bh) * height
    x, y = int(math.floor(x0)), int(math.floor(y0))
    w, h = int(math.ceil(x1)) - x, int(math.ceil(y1)) - y
    w, h = min(w, width - x), min(h, height - y)
    w, h = w - w % 2, h - h % 2
    if w < 32 or h < 32:
        return None
    return x, y, w, h


def scaled_size(w: int, h: int, max_side: int) -> tuple[int, int]:
    """Down-scale only, to max_side on the long side; even sizes for H.264."""
    s = min(1.0, max_side / max(w, h))
    return max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)


def _pose_landmarks(mp4: Path) -> tuple[np.ndarray, int, int]:
    import cv2
    import pose_extractor
    cap = cv2.VideoCapture(str(mp4))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    frames, _ = pose_extractor._extract_with_tasks(str(mp4), verbose=False)
    arr = np.full((len(frames), 33, 4), np.nan, dtype=np.float32)
    for i, fp in enumerate(frames):
        if fp.landmarks:
            arr[i] = [[lm.x, lm.y, lm.z, lm.visibility] for lm in fp.landmarks]
    return arr, width, height


def crop_clip(data: bytes, pad: float = 0.25, max_side: int = 720, cache_dir: Path | None = None) -> tuple[bytes, dict]:
    """The clip cropped around the batter, and what was done. Falls back to the clip as it is."""
    tag = hashlib.sha1(data + f"{pad}:{max_side}".encode()).hexdigest()[:16]
    if cache_dir is not None and (cache_dir / f"{tag}.mp4").exists():
        return (cache_dir / f"{tag}.mp4").read_bytes(), json.loads((cache_dir / f"{tag}.json").read_text())
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is not on PATH (needed for --crop). On a Mac: brew install ffmpeg")
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.mp4", Path(tmp) / "out.mp4"
        src.write_bytes(data)
        lms, width, height = _pose_landmarks(src)
        box = crop_box(lms, width, height, pad=pad)
        info = {"src_size": f"{width}x{height}", "pose_frames": int((~np.isnan(lms[:, 0, 0])).sum()), "frames": len(lms)}
        if box is None:
            info.update(crop="none (too few frames with a pose)", out_size=f"{width}x{height}")
            return data, info
        x, y, w, h = box
        sw, sh = scaled_size(w, h, max_side)
        out = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(src), "-vf", f"crop={w}:{h}:{x}:{y},scale={sw}:{sh}:flags=lanczos",
                              "-an", "-map_metadata", "-1", "-c:v", "libx264", "-preset", "slow", "-crf", "23",
                              "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], capture_output=True, check=False)
        if out.returncode != 0:
            raise SystemExit(f"ffmpeg crop failed: {out.stderr.decode()[-300:]}")
        cropped = dst.read_bytes()
    info.update(crop=f"{x},{y},{w},{h}", out_size=f"{sw}x{sh}", area_share=round(w * h / (width * height), 3))
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{tag}.mp4").write_bytes(cropped)
        (cache_dir / f"{tag}.json").write_text(json.dumps(info))
    return cropped, info


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

@dataclass
class Job:
    kind: str
    meta: dict                       # filenames, order, group: everything the analysis needs (no clip codes)
    body: dict
    est_usd: float
    clip_ids: list[str] = field(default_factory=list)


def generation_config(schema: dict, thinking: str | None, media_resolution: str | None, temperature: float = 0.0) -> dict:
    cfg = {"temperature": temperature, "responseMimeType": "application/json", "responseSchema": schema}
    if thinking:
        cfg["thinkingConfig"] = {"thinkingLevel": thinking}
    if media_resolution and media_resolution != "default":
        cfg["mediaResolution"] = f"MEDIA_RESOLUTION_{media_resolution.upper()}"
    return cfg


def video_part(data: bytes, fps: float) -> dict:
    return v1.video_part(data, fps)


def estimate_usd(text_chars: int, video_secs: Sequence[float], fps: float, kind: str, thinking: str | None,
                 media_resolution: str | None, usd_in: float, usd_out: float) -> float:
    tpf = TOKENS_PER_FRAME.get(media_resolution or "default", 100)
    tokens_in = text_chars / 3.5 + sum(math.ceil(s * fps) * tpf for s in video_secs)
    tokens_out = ANSWER_TOKENS[kind] + THOUGHT_TOKENS.get(thinking, 1500)
    return (tokens_in * usd_in + tokens_out * usd_out) / 1e6


def _rubric() -> dict:
    return json.loads(v1.RUBRIC.read_text())


def _rubric_lists(r: dict) -> tuple[str, str]:
    cps = "\n".join(f"- {c['id']} ({c['stage']}): {c['text']}" for c in r["checkpoints"])
    faults = "\n".join(f"- {f['id']}: {f['text']}" for f in r["faults_in_chain_order"])
    return cps, faults


PAIR_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "fault_a": {"type": "STRING", "enum": FAULTS},
        "fault_b": {"type": "STRING", "enum": FAULTS},
        "reason": {"type": "STRING"},
        "better": {"type": "STRING", "enum": ["A", "B"]},
        "confidence": {"type": "STRING", "enum": ["low", "medium", "high"]},
    },
    "required": ["fault_a", "fault_b", "reason", "better", "confidence"],
    "propertyOrdering": ["fault_a", "fault_b", "reason", "better", "confidence"],
}


def pairwise_prompt() -> str:
    r = _rubric()
    cps, faults = _rubric_lists(r)
    return f"""You are a qualified cricket coach comparing two front-foot drives, each filmed from behind the
bowler (rubric v{r['rubric_version']}). Judge the TECHNIQUE of each shot, not the player's age, size or kit,
the video quality, or where the ball went. Faces are blurred on purpose.

What a good drive does:
{cps}

Do not judge what this camera cannot show: {"; ".join(r["not_judged_from_this_camera"])}.

Faults in chain order (F00 = no real fault). The main fault is the one that, fixed first, would help
the player most (usually the earliest in the chain):
{faults}

You will see drive A, then drive B. Which one is shown first is random: do not favour either position.
Give each drive's main fault, then say which drive is technically better. You must choose one."""


def balanced_pair_order(ids: Sequence[str], max_pairs: int | None, rng: random.Random) -> list[tuple[str, str, bool]]:
    """Pairs in an order where every prefix is balanced (each clip seen about equally often).
    Returns (shown_first, shown_second, swapped): swapped is True when the first clip shown is the
    later one in sorted order. Orientation is random per pair (rng), so position bias cancels."""
    pool = list(itertools.combinations(sorted(ids), 2))
    rng.shuffle(pool)
    n = len(pool) if max_pairs is None else min(max_pairs, len(pool))
    count = {i: 0 for i in ids}
    out = []
    while len(out) < n and pool:
        best = min(pool, key=lambda p: (count[p[0]] + count[p[1]], max(count[p[0]], count[p[1]])))
        pool.remove(best)
        swapped = rng.random() < 0.5
        a, b = (best[1], best[0]) if swapped else best
        out.append((a, b, swapped))
        count[a] += 1
        count[b] += 1
    return out


def build_pairwise_jobs(ctx: Context, clips: dict[str, bytes], args) -> list[Job]:
    system = pairwise_prompt()
    ask = "Which drive is technically better, A or B?"
    jobs = []
    for n, (a, b, swapped) in enumerate(balanced_pair_order(ctx.targets, args.max_pairs, random.Random(args.seed)), 1):
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": "Drive A."}, video_part(clips[a], args.video_fps),
                                                    {"text": "Drive B."}, video_part(clips[b], args.video_fps), {"text": ask}]}],
            "generationConfig": generation_config(PAIR_SCHEMA, args.thinking, args.media_resolution, args.temperature),
        }
        est = estimate_usd(len(system) + len(ask) + 20, [ctx.secs.get(a, 4.0), ctx.secs.get(b, 4.0)], args.video_fps,
                           "pairwise", args.thinking, args.media_resolution, args.usd_per_mtok_in, args.usd_per_mtok_out)
        jobs.append(Job("pairwise", {"pair_no": n, "file_a": ctx.names[a], "file_b": ctx.names[b], "swapped": swapped},
                        body, est, [a, b]))
    return jobs


def choose_refs(target: str, ctx: Context, per_level: int) -> list[str]:
    """per_level reference clips per tier, never the target. The three original references come
    first in their tier; then the clips whose score is closest to the tier's median."""
    orig = {key_of(f) for f in v1.REFERENCES}
    by_key = {key_of(n): cid for cid, n in ctx.names.items()}
    out = []
    for tier in TIERS:
        rows = [r for r in ctx.gt.values() if r.get("tier") == tier and key_of(r["filename"]) in by_key
                and by_key[key_of(r["filename"])] != target and r.get("overall_score")]
        if not rows:
            continue
        med = float(np.median([float(r["overall_score"]) for r in rows]))
        rows.sort(key=lambda r: (key_of(r["filename"]) not in orig, abs(float(r["overall_score"]) - med), r["filename"]))
        out += [by_key[key_of(r["filename"])] for r in rows[:per_level]]
    return out


def absolute_prompt(n_refs: int) -> str:
    text = v1.prompt_text()
    return text.replace("You will first see three reference drives", f"You will first see {n_refs} reference drives")


def build_absolute_jobs(ctx: Context, clips: dict[str, bytes], args) -> list[Job]:
    jobs = []
    for cid in ctx.targets:
        refs = choose_refs(cid, ctx, args.refs_per_level)
        system = absolute_prompt(len(refs))
        parts, text_chars = [], len(system)
        for n, rid in enumerate(refs, 1):
            g = ctx.gt[key_of(ctx.names[rid])]
            note = (f"A qualified coach scored reference drive {n} ({g['tier']}): overall {g['overall_score']}; "
                    + ", ".join(f"{p} {g[f'{p}_score']}" for p in v1.PILLARS) + ".")
            parts += [{"text": f"Reference drive {n}."}, video_part(clips[rid], args.video_fps), {"text": note}]
            text_chars += len(note) + 20
        parts += [{"text": "Now score this drive."}, video_part(clips[cid], args.video_fps)]
        body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": parts}],
                "generationConfig": generation_config(v1.SCHEMA, args.thinking, args.media_resolution, args.temperature)}
        est = estimate_usd(text_chars, [ctx.secs.get(c, 4.0) for c in refs + [cid]], args.video_fps, "absolute",
                           args.thinking, args.media_resolution, args.usd_per_mtok_in, args.usd_per_mtok_out)
        jobs.append(Job("absolute", {"filename": ctx.names[cid], "refs": ";".join(ctx.names[r] for r in refs)}, body, est, refs + [cid]))
    return jobs


STAGE_GROUP = {"Set-up": 0, "Backlift": 0, "Movement": 1, "Downswing": 2, "Contact": 2, "Finish": 3}
GROUP_NAMES = ["Set-up and backlift", "Movement to the ball", "Downswing and contact", "Finish"]
ANSWER_TO_RATING = {"yes": "2", "partly": "1", "no": "0", "cannot_see": "X"}


def checkpoint_groups(rubric: dict | None = None) -> list[tuple[str, list[dict]]]:
    r = rubric or _rubric()
    groups: list[list[dict]] = [[] for _ in GROUP_NAMES]
    for c in r["checkpoints"]:
        groups[STAGE_GROUP[c["stage"]]].append(c)
    return [(GROUP_NAMES[i], g) for i, g in enumerate(groups) if g]


def checkpoint_schema(cps: list[dict]) -> dict:
    ids = [c["id"] for c in cps]
    return {"type": "OBJECT",
            "properties": {**{i: {"type": "STRING", "enum": list(ANSWER_TO_RATING)} for i in ids}, "note": {"type": "STRING"}},
            "required": ids + ["note"], "propertyOrdering": ids + ["note"]}


def checkpoint_prompt(group: str, cps: list[dict], rubric: dict) -> str:
    qs = "\n".join(f"- {c['id']}: {c['text']}? (what this camera shows: {c.get('visible', 'yes')})" for c in cps)
    return f"""You are a qualified cricket coach watching one front-foot drive filmed from behind the bowler.
Faces are blurred on purpose. Look only at this part of the shot: {group}.

Answer each question about THIS shot:
- yes: done well; a coach would not mention it
- partly: partly done, or a small fault
- no: a clear fault
- cannot_see: this clip does not show it (camera angle, out of frame, too fast)
{qs}

Do not judge what this camera cannot show: {"; ".join(rubric["not_judged_from_this_camera"])}.
Be as strict as a coach marking a club or junior player; do not answer yes by default.
In "note", say in one sentence what you saw."""


def build_checkpoint_jobs(ctx: Context, clips: dict[str, bytes], args) -> list[Job]:
    r = _rubric()
    jobs = []
    for cid in ctx.targets:
        for g, (group, cps) in enumerate(checkpoint_groups(r), 1):
            system = checkpoint_prompt(group, cps, r)
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [video_part(clips[cid], args.video_fps),
                                                            {"text": f"Answer the {len(cps)} questions for this drive."}]}],
                    "generationConfig": generation_config(checkpoint_schema(cps), args.thinking, args.media_resolution, args.temperature)}
            est = estimate_usd(len(system) + 50, [ctx.secs.get(cid, 4.0)], args.video_fps, "checkpoints",
                               args.thinking, args.media_resolution, args.usd_per_mtok_in, args.usd_per_mtok_out)
            jobs.append(Job("checkpoints", {"filename": ctx.names[cid], "group": g, "group_name": group,
                                            "checkpoints": [c["id"] for c in cps]}, body, est, [cid]))
    return jobs


# ---------------------------------------------------------------------------
# Sending, with retries and the cost guard
# ---------------------------------------------------------------------------

class KeyRejected(Exception):
    pass


class ApiError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(f"Gemini API error {code}: {message}")
        self.code = code


def _error_message(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode(errors="replace")
    except Exception:  # noqa: BLE001
        return ""
    try:
        return json.loads(raw).get("error", {}).get("message", raw[:300])
    except (json.JSONDecodeError, AttributeError):
        return raw[:300]


def _key_problem(code: int, msg: str) -> bool:
    return code in (401, 403) or (code == 400 and ("API key" in msg or "API_KEY" in msg))


def post_json(url: str, body: dict, *, timeout: int = 300, attempts: int = 6,
              sleep: Callable[[float], None] = time.sleep, opener: Callable | None = None) -> dict:
    """POST with the key header; backoff on 429 and 5xx (Retry-After respected); KeyRejected on a bad key."""
    opener = opener or urllib.request.urlopen
    data = json.dumps(body).encode()
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": v1._key()})
        try:
            with opener(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            msg = _error_message(exc)
            if _key_problem(exc.code, msg):
                raise KeyRejected(f"Google rejected the API key ({exc.code}: {msg}). Copy the whole key again from "
                                  "aistudio.google.com > Get API key (it starts with AIza), then re-run.") from None
            if exc.code in RETRY_CODES and attempt < attempts - 1:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                wait = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 4.0 * 2 ** attempt)
                sleep(wait + random.uniform(0, 1))
                continue
            raise ApiError(exc.code, msg) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt < attempts - 1:
                sleep(min(60.0, 4.0 * 2 ** attempt))
                continue
            raise ApiError(0, str(exc)) from None
    raise ApiError(0, "no answer after retries")


def _drop_unsupported(body: dict, message: str) -> bool:
    """Remove thinkingConfig / mediaResolution when the model says it does not take them. True if removed."""
    cfg, msg, dropped = body.get("generationConfig", {}), message.lower(), False
    if "thinking" in msg and "thinkingConfig" in cfg:
        cfg.pop("thinkingConfig")
        dropped = True
    if ("media_resolution" in msg or "mediaresolution" in msg) and "mediaResolution" in cfg:
        cfg.pop("mediaResolution")
        dropped = True
    return dropped


def usage_of(resp: dict) -> dict:
    u = resp.get("usageMetadata", {}) or {}
    return {"prompt": int(u.get("promptTokenCount", 0) or 0), "candidates": int(u.get("candidatesTokenCount", 0) or 0),
            "thoughts": int(u.get("thoughtsTokenCount", 0) or 0)}


def answer_of(resp: dict) -> dict | None:
    try:
        parts = resp["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        ans = json.loads(text)
        return ans if isinstance(ans, dict) else None
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None


class CostGuard:
    """Hard cap on spend. Before a request, its estimate (scaled up by the worst actual/estimate
    ratio seen so far) is reserved; the request is refused if spent + reserved + it would pass the
    cap. After it, the reservation is replaced by the cost from usageMetadata."""

    def __init__(self, budget_gbp: float, usd_in: float = USD_PER_MTOK_IN, usd_out: float = USD_PER_MTOK_OUT,
                 gbp_per_usd: float = GBP_PER_USD):
        self.budget_gbp, self.usd_in, self.usd_out, self.gbp_per_usd = budget_gbp, usd_in, usd_out, gbp_per_usd
        self.spent_usd = 0.0
        self.reserved_usd = 0.0
        self.ratio = 1.0
        self.requests = 0
        self.refused = 0
        self.tokens = {"prompt": 0, "candidates": 0, "thoughts": 0}
        self._lock = threading.Lock()

    def cost_usd(self, usage: dict) -> float:
        return (usage.get("prompt", 0) * self.usd_in + (usage.get("candidates", 0) + usage.get("thoughts", 0)) * self.usd_out) / 1e6

    @property
    def spent_gbp(self) -> float:
        return self.spent_usd * self.gbp_per_usd

    def reserve(self, est_usd: float) -> float | None:
        with self._lock:
            need = est_usd * self.ratio
            if (self.spent_usd + self.reserved_usd + need) * self.gbp_per_usd > self.budget_gbp:
                self.refused += 1
                return None
            self.reserved_usd += need
            return need

    def settle(self, reserved: float, est_usd: float, usage: dict | None) -> float:
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - reserved)
            if usage is None:
                return 0.0
            actual = self.cost_usd(usage)
            self.spent_usd += actual
            self.requests += 1
            for k in self.tokens:
                self.tokens[k] += usage.get(k, 0)
            if est_usd > 0:
                self.ratio = max(self.ratio, actual / est_usd)
            return actual


def run_jobs(jobs: list[Job], model: str, guard: CostGuard, workers: int = 4, log: Callable[[str], None] = print,
             post: Callable[[str, dict], dict] | None = None, jsonl: Path | None = None) -> list[dict]:
    """Send the jobs in order (up to `workers` at a time) until the guard refuses one. Returns one
    result per job sent: {"job", "answer", "usage", "error"}."""
    post = post or (lambda url, body: post_json(url, body))
    url = f"{API}/models/{model}:generateContent"
    results: list[dict] = []
    lock = threading.Lock()
    stop = threading.Event()
    sink = jsonl.open("a", encoding="utf-8") if jsonl else None

    def one(i: int, job: Job) -> dict | None:
        if stop.is_set():
            return None
        held = guard.reserve(job.est_usd)
        if held is None:
            stop.set()
            return None
        usage, ans, err = None, None, ""
        try:
            try:
                resp = post(url, job.body)
            except ApiError as exc:
                if not (exc.code == 400 and _drop_unsupported(job.body, str(exc))):
                    raise
                log(f"  the model refused an optional setting; resending without it ({exc})")
                resp = post(url, job.body)
            usage, ans = usage_of(resp), answer_of(resp)
            if ans is None:
                err = f"unreadable answer: {json.dumps(resp)[:200]}"
        except ApiError as exc:
            err = str(exc)
        finally:
            cost = guard.settle(held, job.est_usd, usage)
        res = {"i": i, "job": job, "answer": ans or {}, "usage": usage or {}, "error": err, "cost_usd": cost}
        with lock:
            results.append(res)
            if len(results) >= 3 and all(x["error"] for x in results):
                stop.set()  # the first three all failed: the request itself is wrong; do not spend on the rest
                log(f"The first {len(results)} requests all failed ({err}); stopping.")
            if sink:
                sink.write(json.dumps({"kind": job.kind, **job.meta, "answer": ans, "usage": usage, "error": err}) + "\n")
                sink.flush()
        label = job.meta.get("filename") or f"pair {job.meta.get('pair_no')}"
        log(f"  {len(results):3d}/{len(jobs)} {label}: {'ERROR ' + err if err else 'ok'} (spent ~GBP {guard.spent_gbp:.2f})")
        return res

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(one, i, j) for i, j in enumerate(jobs)]
            try:
                for f in as_completed(futures):
                    f.result()
            except KeyRejected:
                stop.set()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
    finally:
        if sink:
            sink.close()
    if guard.refused:
        log(f"Budget cap (GBP {guard.budget_gbp:.2f}) reached: stopped after {len(results)} of {len(jobs)} requests.")
    return sorted(results, key=lambda r: r["i"])


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(len(v), dtype=float)
    for val in np.unique(v):  # average ranks for ties
        m = v == val
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and not (isinstance(a, float) and math.isnan(a))]
    if len(pairs) < 3:
        return None
    a, b = (np.array(t, dtype=float) for t in zip(*pairs))
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def order_agreement(model: dict[str, float], coach: dict[str, float]) -> tuple[float | None, int]:
    """Share of clip pairs (coach scores differ) that the model orders the same way; a model tie counts half."""
    names = [n for n in model if n in coach and model[n] is not None]
    agree, n = 0.0, 0
    for a, b in itertools.combinations(names, 2):
        dc = coach[a] - coach[b]
        if dc == 0:
            continue
        dm = model[a] - model[b]
        n += 1
        agree += 1.0 if dm * dc > 0 else 0.5 if dm == 0 else 0.0
    return (agree / n if n else None), n


def rank_metrics(model: dict[str, float], alex: dict[str, float], cam: dict[str, float]) -> dict:
    out = {}
    for who, coach in (("alex", alex), ("cam", cam)):
        names = [n for n in model if n in coach]
        out[f"spearman_{who}"] = spearman([model[n] for n in names], [coach[n] for n in names])
        out[f"n_{who}"] = len(names)
        out[f"order_agreement_{who}"], out[f"order_pairs_{who}"] = order_agreement(model, coach)
    return out


def analyse_pairwise(results: list[dict], files: list[str], alex: dict[str, float], cam: dict[str, float]) -> tuple[list[dict], list[dict], dict]:
    rows, judgements = [], []
    for r in results:
        m, a = r["job"].meta, r["answer"]
        better = a.get("better", "")
        winner = m["file_a"] if better == "A" else m["file_b"] if better == "B" else ""
        loser = m["file_b"] if better == "A" else m["file_a"] if better == "B" else ""
        if winner:
            judgements.append((winner, loser))
        agree_alex = ""
        if winner and alex.get(winner) is not None and alex.get(loser) is not None and alex[winner] != alex[loser]:
            agree_alex = int(alex[winner] > alex[loser])
        agree_cam = ""
        if winner and cam.get(winner) is not None and cam.get(loser) is not None and cam[winner] != cam[loser]:
            agree_cam = int(cam[winner] > cam[loser])
        rows.append({"pair_no": m["pair_no"], "drive_a": m["file_a"], "drive_b": m["file_b"], "swapped": int(m["swapped"]),
                     "better": better, "winner": winner, "confidence": a.get("confidence", ""),
                     "fault_a": a.get("fault_a", ""), "fault_b": a.get("fault_b", ""),
                     "alex_a": alex.get(m["file_a"], ""), "alex_b": alex.get(m["file_b"], ""),
                     "agree_alex": agree_alex, "agree_cam": agree_cam, "reason": a.get("reason", ""),
                     **{f"{k}_tokens": r["usage"].get(k, "") for k in ("prompt", "candidates", "thoughts")}, "error": r["error"]})
    theta, se, ssr = fit_bradley_terry(files, judgements) if judgements else (np.zeros(len(files)), np.full(len(files), np.nan), float("nan"))
    model = {f: float(t) for f, t in zip(files, theta)}
    scale = []
    for rank, i in enumerate(np.argsort(-theta), 1):
        f = files[i]
        scale.append({"rank": rank, "filename": f, "theta": round(float(theta[i]), 3),
                      "se": "" if np.isnan(se[i]) else round(float(se[i]), 3),
                      "wins": sum(1 for w, _ in judgements if w == f), "games": sum(1 for w, l in judgements if f in (w, l)),
                      "alex": alex.get(f, ""), "cam": cam.get(f, "")})
    answered = [r for r in rows if r["better"]]
    n_a = sum(1 for r in answered if r["better"] == "A")
    stats = {"pairs_sent": len(rows), "pairs_answered": len(answered), "ssr": ssr,
             "share_A": n_a / len(answered) if answered else None,
             "position_z": (n_a - len(answered) / 2) / math.sqrt(len(answered) / 4) if answered else None,
             **rank_metrics(model, alex, cam)}
    for who in ("alex", "cam"):
        vals = [r[f"agree_{who}"] for r in rows if r[f"agree_{who}"] != ""]
        stats[f"pick_agreement_{who}"] = sum(vals) / len(vals) if vals else None
        stats[f"pick_pairs_{who}"] = len(vals)
        hi = [r[f"agree_{who}"] for r in rows if r[f"agree_{who}"] != "" and r["confidence"] == "high"]
        stats[f"pick_agreement_{who}_high_conf"] = sum(hi) / len(hi) if hi else None
        stats[f"pick_pairs_{who}_high_conf"] = len(hi)
    return rows, scale, stats


def analyse_absolute(results: list[dict], alex: dict[str, float], cam: dict[str, float]) -> tuple[list[dict], dict]:
    rows = []
    for r in results:
        a = r["answer"]
        rows.append({"filename": r["job"].meta["filename"], **{k: a.get(k, "") for k in v1.SCHEMA["properties"]},
                     "refs": r["job"].meta["refs"], **{f"{k}_tokens": r["usage"].get(k, "") for k in ("prompt", "candidates", "thoughts")},
                     "error": r["error"]})
    model = {r["filename"]: float(r["overall"]) for r in rows if str(r["overall"]).strip() not in ("", "None")}
    hist = {f"{lo}-{lo + 19}": sum(1 for v in model.values() if lo <= v <= lo + 19 + (1 if lo == 80 else 0)) for lo in range(0, 100, 20)}
    return rows, {"clips_scored": len(model), "histogram": hist, **rank_metrics(model, alex, cam)}


def earliest_fault(ratings: dict[str, str], level: str = "0") -> str:
    """F code of the earliest checkpoint (chain order C01..C12) rated `level` ("0", or "1" for partly); F00 if none."""
    for i, c in enumerate(CPS, 1):
        if ratings.get(c) == level:
            return f"F{i:02d}"
    return "F00"


def analyse_checkpoints(results: list[dict], files: list[str], alex: dict[str, float], cam: dict[str, float],
                        cam_faults: dict[str, tuple[str, str]]) -> tuple[list[dict], list[dict], dict]:
    ratings: dict[str, dict[str, str]] = {f: {} for f in files}
    answers = []
    for r in results:
        m, a = r["job"].meta, r["answer"]
        for c in m["checkpoints"]:
            if a.get(c) in ANSWER_TO_RATING:
                ratings[m["filename"]][c] = ANSWER_TO_RATING[a[c]]
        answers.append({"filename": m["filename"], "group": m["group"], "group_name": m["group_name"],
                        **{c: a.get(c, "") for c in m["checkpoints"]}, "note": a.get("note", ""),
                        **{f"{k}_tokens": r["usage"].get(k, "") for k in ("prompt", "candidates", "thoughts")}, "error": r["error"]})
    per_clip, model = [], {}
    for f in files:
        rt = ratings[f]
        if not rt:
            continue
        judged = [int(v) for v in rt.values() if v in ("0", "1", "2")]
        score = 100 * sum(judged) / (2 * len(judged)) if judged else None
        if score is not None and len(rt) == len(CPS):
            model[f] = score
        main = earliest_fault(rt, "0")
        lenient = main if main != "F00" else earliest_fault(rt, "1")
        cm, cs = cam_faults.get(f, ("", ""))
        per_clip.append({"filename": f, **{c: rt.get(c, "") for c in CPS}, "complete": int(len(rt) == len(CPS)),
                         "checkpoint_score": "" if score is None else round(score, 1), "main_fault": main,
                         "main_fault_incl_partly": lenient, "failed": ";".join(c for c in CPS if rt.get(c) == "0"),
                         "cam_main_fault": cm, "cam_second_fault": cs, "alex": alex.get(f, ""), "cam": cam.get(f, "")})
    with_cam = [p for p in per_clip if p["cam_main_fault"]]
    stats = {"clips_rated": len(per_clip), "clips_complete": sum(p["complete"] for p in per_clip), **rank_metrics(model, alex, cam)}
    if with_cam:
        def share(pred):
            return sum(1 for p in with_cam if pred(p)) / len(with_cam)
        stats.update({
            "cam_clips": len(with_cam),
            "main_fault_matches_cam": share(lambda p: p["main_fault"] == p["cam_main_fault"]),
            "main_fault_incl_partly_matches_cam": share(lambda p: p["main_fault_incl_partly"] == p["cam_main_fault"]),
            "main_fault_is_cam_main_or_second": share(lambda p: p["main_fault"] in (p["cam_main_fault"], p["cam_second_fault"])),
            "cam_fault_checkpoint_rated_no": share(lambda p: p["cam_main_fault"] != "F00"
                                                   and p.get(f"C{int(p['cam_main_fault'][1:]):02d}") == "0"),
            "cam_f00_and_model_f00": share(lambda p: p["cam_main_fault"] == "F00" and p["main_fault"] == "F00"),
        })
    spread = {c: {lvl: sum(1 for p in per_clip if p[c] == lvl) for lvl in ("2", "1", "0", "X", "")} for c in CPS}
    stats["spread"] = spread
    stats["all_ratings"] = {lvl: sum(s[lvl] for s in spread.values()) for lvl in ("2", "1", "0", "X")}
    return answers, per_clip, stats


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _fmt(v, digits: int = 2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def summary_md(args, model: str, guard: CostGuard, n_jobs: int, stats: dict, crop_rows: list[dict]) -> str:
    lines = [f"# Video-model judge v2: {args.mode}{' + crop' if args.crop else ''} ({time.strftime('%Y-%m-%d')})", "",
             f"Model {model}, {args.video_fps:g} frames/s, temperature {args.temperature:g}"
             + (f", thinking {args.thinking}" if args.thinking else "")
             + (f", media resolution {args.media_resolution}" if args.media_resolution != "default" else "")
             + (f", refs per tier {args.refs_per_level}" if args.mode == "absolute" else "")
             + (f", seed {args.seed}" if args.mode == "pairwise" else "") + ".",
             f"Requests sent {guard.requests} of {n_jobs}; spent about GBP {guard.spent_gbp:.2f} (cap {args.budget_gbp:.2f}; "
             f"tokens: prompt {guard.tokens['prompt']}, answer {guard.tokens['candidates']}, thinking {guard.tokens['thoughts']}; "
             f"priced at USD {args.usd_per_mtok_in}/{args.usd_per_mtok_out} per million in/out, GBP {GBP_PER_USD}/USD).", ""]
    if guard.refused:
        lines += ["**Stopped by the cost guard before all requests were sent.**", ""]
    lines += [f"- Spearman vs Alex: {_fmt(stats.get('spearman_alex'))} (n={stats.get('n_alex')}); "
              f"vs Cam: {_fmt(stats.get('spearman_cam'))} (n={stats.get('n_cam')})",
              f"- Clip-pair order agreement (model order vs coach order over all clip pairs): Alex {_fmt(stats.get('order_agreement_alex'))} "
              f"({stats.get('order_pairs_alex')} pairs), Cam {_fmt(stats.get('order_agreement_cam'))} ({stats.get('order_pairs_cam')} pairs)"]
    if args.mode == "pairwise":
        lines += [f"- Pairs answered {stats['pairs_answered']} of {stats['pairs_sent']} sent; Bradley-Terry scale separation reliability {_fmt(stats['ssr'])}",
                  f"- Each pick vs Alex's score order: {_fmt(stats['pick_agreement_alex'])} of {stats['pick_pairs_alex']} "
                  f"(high confidence only: {_fmt(stats['pick_agreement_alex_high_conf'])} of {stats['pick_pairs_alex_high_conf']}); "
                  f"vs Cam: {_fmt(stats['pick_agreement_cam'])} of {stats['pick_pairs_cam']}",
                  f"- Position bias: drive A (shown first) picked in {_fmt(stats['share_A'])} of answers (z = {_fmt(stats['position_z'])}; "
                  "A/B order was random per pair, so a bias adds noise but not a direction)"]
    elif args.mode == "absolute":
        lines += [f"- Clips scored {stats['clips_scored']}; overall score histogram: "
                  + ", ".join(f"{k}: {v}" for k, v in stats["histogram"].items())]
    else:
        lines += [f"- Clips rated {stats['clips_rated']} ({stats['clips_complete']} with all 12 checkpoints); "
                  "checkpoint score = share of available marks, used for the Spearman above",
                  "- All ratings: " + ", ".join(f"{k}: {v}" for k, v in stats["all_ratings"].items()) + " (2 yes, 1 partly, 0 no, X cannot see)"]
        if stats.get("cam_clips"):
            lines += [f"- Main fault (earliest checkpoint rated no) = Cam's main fault: {_fmt(stats['main_fault_matches_cam'])} of {stats['cam_clips']} clips; "
                      f"counting 'partly' when nothing is rated no: {_fmt(stats['main_fault_incl_partly_matches_cam'])}; "
                      f"= Cam's main or second fault: {_fmt(stats['main_fault_is_cam_main_or_second'])}; "
                      f"Cam's main-fault checkpoint rated no: {_fmt(stats['cam_fault_checkpoint_rated_no'])}"]
        lines += ["", "| checkpoint | yes (2) | partly (1) | no (0) | cannot see (X) |", "|---|---|---|---|---|"]
        lines += [f"| {c} | {s['2']} | {s['1']} | {s['0']} | {s['X']} |" for c, s in stats["spread"].items()]
    if crop_rows:
        shares = [r["area_share"] for r in crop_rows if r.get("area_share")]
        lines += ["", f"Crop: {sum(1 for r in crop_rows if r.get('area_share'))} of {len(crop_rows)} clips cropped around the batter"
                  + (f" (crop keeps {min(shares):.0%}-{max(shares):.0%} of the frame area)" if shares else "")
                  + f"; pad {args.crop_pad:.0%}, long side up to {args.crop_max_side} px. Boxes in crop_boxes.csv."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["pairwise", "absolute", "checkpoints"], required=True)
    ap.add_argument("--clips-html", required=True)
    ap.add_argument("--key", required=True, help="the offline file's *_key_PRIVATE.csv (clip code -> filename)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ground-truth", default=str(SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"))
    ap.add_argument("--cam", default=str(CAM_QUICK), help="Cam's quick check (overall, main_fault per filename)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--video-fps", type=float, default=v1.VIDEO_FPS)
    ap.add_argument("--crop", action="store_true", help="crop each clip around the batter (pose box + 25%%) before sending")
    ap.add_argument("--crop-pad", type=float, default=0.25)
    ap.add_argument("--crop-max-side", type=int, default=720)
    ap.add_argument("--crop-cache", default=None, help="folder to keep cropped clips between runs (outside the repo; delete after)")
    ap.add_argument("--max-pairs", type=int, default=None, help="pairwise: a balanced subset of this many pairs (default all)")
    ap.add_argument("--seed", type=int, default=20261115, help="pairwise: seed for the pair order and the A/B sides")
    ap.add_argument("--refs-per-level", type=int, choices=[1, 2], default=1, help="absolute: reference clips per tier")
    ap.add_argument("--limit-clips", type=int, default=None, help="judge only the first N clips (a cheap smoke test)")
    ap.add_argument("--thinking", choices=["low", "high"], default=None, help="thinkingLevel (Gemini 3); default: the model's own")
    ap.add_argument("--media-resolution", choices=["default", "low", "medium", "high"], default="default")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 as in runs 1-2 (Google suggests 1.0 for Gemini 3)")
    ap.add_argument("--workers", type=int, default=4, help="requests in flight at once")
    ap.add_argument("--budget-gbp", type=float, default=1.2)
    ap.add_argument("--usd-per-mtok-in", type=float, default=USD_PER_MTOK_IN)
    ap.add_argument("--usd-per-mtok-out", type=float, default=USD_PER_MTOK_OUT)
    ap.add_argument("--dry-run", action="store_true", help="build every request and estimate the cost; send nothing")
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    if args.crop_cache and SCRIPT_DIR in Path(args.crop_cache).resolve().parents:
        ap.error("--crop-cache holds video: put it outside the repo")
    if SCRIPT_DIR in out.parents and not str(out.relative_to(SCRIPT_DIR)).startswith("baselines"):
        ap.error("--out inside the repo must be under baselines/")
    return args


BUILDERS = {"pairwise": build_pairwise_jobs, "absolute": build_absolute_jobs, "checkpoints": build_checkpoint_jobs}


def main(argv: list[str] | None = None, post: Callable[[str, dict], dict] | None = None) -> int:
    args = parse_args(argv)
    ctx = load_context(Path(args.clips_html), Path(args.key), Path(args.ground_truth), Path(args.cam), args.limit_clips)
    alex = {ctx.names[c]: ctx.alex(c) for c in ctx.names if ctx.alex(c) is not None}
    cam = {ctx.names[c]: ctx.cam_score(c) for c in ctx.names if ctx.cam_score(c) is not None}
    cam_faults = {name: (ctx.cam[key_of(name)].get("main_fault", ""), ctx.cam[key_of(name)].get("second_fault", ""))
                  for name in ctx.names.values() if key_of(name) in ctx.cam}

    clips, crop_rows = dict(ctx.clips), []
    if args.crop:
        cache = Path(args.crop_cache).expanduser() if args.crop_cache else None
        needed = set(ctx.targets)
        if args.mode == "absolute":
            for cid in ctx.targets:
                needed.update(choose_refs(cid, ctx, args.refs_per_level))
        print(f"Cropping {len(needed)} clips around the batter (pose + ffmpeg)...")
        for cid in sorted(needed, key=lambda c: ctx.names[c]):
            clips[cid], info = crop_clip(ctx.clips[cid], args.crop_pad, args.crop_max_side, cache)
            crop_rows.append({"filename": ctx.names[cid], "bytes_before": len(ctx.clips[cid]), "bytes_after": len(clips[cid]), **info})
            print(f"  {ctx.names[cid]}: {info['src_size']} -> {info['out_size']} (crop {info['crop']})")

    jobs = BUILDERS[args.mode](ctx, clips, args)
    est = sum(j.est_usd for j in jobs) * GBP_PER_USD
    sizes = [len(json.dumps(j.body)) / 1e6 for j in jobs]
    fits, run = 0, 0.0
    for j in jobs:
        run += j.est_usd * GBP_PER_USD
        if run > args.budget_gbp:
            break
        fits += 1
    print(f"Mode {args.mode}{' + crop' if args.crop else ''}: {len(jobs)} requests over {len(ctx.targets)} clips at {args.video_fps:g} frames/s; "
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
        return KEY_REJECTED_EXIT if "rejected" in str(exc) else 1
    print(f"Model: {model}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    guard = CostGuard(args.budget_gbp, args.usd_per_mtok_in, args.usd_per_mtok_out)
    (out / "responses.jsonl").unlink(missing_ok=True)  # raw answers of this run only (filenames, no clip codes, no video)
    try:
        results = run_jobs(jobs, model, guard, args.workers, post=post, jsonl=out / "responses.jsonl")
    except KeyRejected as exc:
        print(exc, file=sys.stderr)
        return KEY_REJECTED_EXIT

    if args.mode == "pairwise":
        files = sorted(ctx.names[c] for c in ctx.targets)
        rows, scale, stats = analyse_pairwise(results, files, alex, cam)
        write_csv(out / "pairwise_judgements.csv", rows)
        write_csv(out / "pairwise_scale.csv", scale)
    elif args.mode == "absolute":
        rows, stats = analyse_absolute(results, alex, cam)
        write_csv(out / "video_judge_scores.csv", rows)
        write_csv(out / "video_judge_checkpoints.csv", [{"filename": r["filename"], **{c: r[c] for c in CPS}} for r in rows])
    else:
        files = sorted(ctx.names[c] for c in ctx.targets)
        answers, per_clip, stats = analyse_checkpoints(results, files, alex, cam, cam_faults)
        write_csv(out / "checkpoint_answers.csv", answers)
        write_csv(out / "checkpoint_ratings.csv", [{"filename": p["filename"], **{c: p[c] for c in CPS}} for p in per_clip])
        write_csv(out / "checkpoint_faults.csv", per_clip)
    write_csv(out / "crop_boxes.csv", crop_rows)
    (out / "summary.md").write_text(summary_md(args, model, guard, len(jobs), stats, crop_rows), encoding="utf-8")
    print(f"Spent about GBP {guard.spent_gbp:.2f}. Spearman vs Alex {_fmt(stats.get('spearman_alex'))}, vs Cam {_fmt(stats.get('spearman_cam'))}. "
          f"Wrote {out}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
