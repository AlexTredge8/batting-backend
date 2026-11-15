#!/usr/bin/env python3
"""
Video-capable model as a second judge (plan, "held back" option; approved by Alex 15 Nov 2026).

Sends each clip as video, with the rubric and three reference clips scored by Alex, to Google's
Gemini API, and gets back the same sheet the stills model filled in: 12 checkpoint ratings,
four pillars, overall, priority fault, second fault, confidence. Each clip is judged --runs
times so run-to-run consistency can be measured.

Clips come from an offline coach file (the swipe, quick or checkpoint HTML): trimmed to the shot,
640 px, no sound or metadata, faces blurred. No original video is sent.

Privacy and cost:
  - Use a key from a Google Cloud project WITH BILLING ENABLED. On the paid tier Google does not
    use prompts or outputs to improve its products; on the free tier it does, and people may
    read them. Several clips show juniors: free-tier keys are not acceptable.
  - --budget-gbp stops the run before the estimated spend passes the cap (default 4; Alex's
    approved cap for model tests is under 5).

  python phase5_video_judge.py --clips-html BattingIQ_which_drive.html --key which_drive_key_PRIVATE.csv \
      --out baselines/phase5_video_judge_<date> [--model gemini-...] [--runs 2] [--dry-run]

The API key is never read by this script: the session's network proxy adds it to requests to
generativelanguage.googleapis.com. Set GEMINI_API_KEY to send it yourself (e.g. on a Mac).
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase0_baseline_table import key_of, load_csv  # noqa: E402

API = "https://generativelanguage.googleapis.com/v1beta"
RUBRIC = SCRIPT_DIR / "docs" / "phase1" / "rubric_v0.json"
REFERENCES = ["Viv P_Offdrive Elite.mov", "Dan P_Offdrive Average.MOV", "Drive_beginner.MOV"]  # same three the stills model saw
CPS = [f"C{i:02d}" for i in range(1, 13)]
PILLARS = ["access", "tracking", "stability", "flow"]
GBP_PER_USD = 0.80


def clips_from_html(path: Path) -> dict[str, bytes]:
    text = path.read_text(encoding="utf-8")
    i = text.index("const DATA = ") + len("const DATA = ")
    data, _ = json.JSONDecoder().raw_decode(text[i:])
    return {c["id"]: base64.b64decode(c["b64"]) for c in data["clips"]}


def prompt_text() -> str:
    r = json.loads(RUBRIC.read_text())
    cps = "\n".join(f"- {c['id']} ({c['stage']}, {c['pillar']}): {c['text']}" for c in r["checkpoints"])
    faults = "\n".join(f"- {f['id']}: {f['text']}" for f in r["faults_in_chain_order"])
    anchors = "\n".join(f"- {p}: 5 = {a['5']}; 15 = {a['15']}; 25 = {a['25']}" for p, a in r["pillar_anchors"].items())
    return f"""You are a qualified cricket coach scoring one front-foot drive from a short video, filmed from
behind the bowler (rubric v{r['rubric_version']}). Score THIS shot, not the player. Faces are blurred on purpose.

Rate each checkpoint 2 (done well), 1 (partly, or a small fault), 0 (clear fault) or X (cannot judge from this clip):
{cps}

Do not mark down what this camera cannot show: {"; ".join(r["not_judged_from_this_camera"])}.

Pillar anchors (0-25 each):
{anchors}

Faults in chain order. Pick ONE priority fault: the one that, fixed first, would help this player most
(usually the earliest in the chain). F00 means no real fault.
{faults}

Use the whole range. A raw beginner can score 15-40 overall; club average 50-70; good club 70-85; elite 85+.
You will first see three reference drives with the scores a qualified coach gave them; use them to calibrate.
Score the overall (0-100) on its own; do not add up the pillars."""


SCHEMA = {
    "type": "OBJECT",
    "properties": {
        **{c: {"type": "STRING", "enum": ["2", "1", "0", "X"]} for c in CPS},
        **{p: {"type": "INTEGER"} for p in PILLARS},
        "overall": {"type": "INTEGER"},
        "priority_fault": {"type": "STRING", "enum": [f"F{i:02d}" for i in range(13)]},
        "second_fault": {"type": "STRING", "enum": [f"F{i:02d}" for i in range(13)] + ["none"]},
        "confidence": {"type": "STRING", "enum": ["low", "medium", "high"]},
        "reason": {"type": "STRING"},
    },
    "required": CPS + PILLARS + ["overall", "priority_fault", "second_fault", "confidence", "reason"],
    "propertyOrdering": CPS + PILLARS + ["overall", "priority_fault", "second_fault", "confidence", "reason"],
}


def _post(url: str, body: dict, timeout: int = 300) -> dict:
    headers = {"Content-Type": "application/json"}
    headers["x-goog-api-key"] = os.environ.get("GEMINI_API_KEY", "set-by-proxy")
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            msg = exc.read().decode()[:400]
            if exc.code in (429, 500, 503) and attempt < 3:
                time.sleep(5 * 2 ** attempt)
                continue
            raise SystemExit(f"Gemini API error {exc.code}: {msg}")
    raise SystemExit("Gemini API: no answer after retries")


def list_models() -> list[str]:
    req = urllib.request.Request(f"{API}/models?pageSize=200",
                                 headers={"x-goog-api-key": os.environ.get("GEMINI_API_KEY", "set-by-proxy")})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return [m["name"].split("/", 1)[1] for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])]


def pick_model(models: list[str]) -> str:
    """The newest stable 'pro' model (no preview/exp/tts/image variants)."""
    def ver(name: str) -> tuple:
        m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
        return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)
    pro = [m for m in models if "-pro" in m and not re.search(r"preview|exp|tts|image|latest|vision", m)]
    pro = pro or [m for m in models if "-pro" in m]
    if not pro:
        raise SystemExit("No Gemini 'pro' model available to this key. Pass --model.")
    return sorted(pro, key=lambda n: (ver(n), -len(n)))[-1]


def video_part(data: bytes) -> dict:
    return {"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(data).decode()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-html", required=True)
    ap.add_argument("--key", required=True, help="the offline file's *_key_PRIVATE.csv (clip code -> filename)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ground-truth", default=str(SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--budget-gbp", type=float, default=4.0)
    ap.add_argument("--usd-per-mtok-in", type=float, default=2.5, help="price assumption for the cost guard (set high, not low)")
    ap.add_argument("--usd-per-mtok-out", type=float, default=15.0)
    ap.add_argument("--dry-run", action="store_true", help="build the requests and estimate the cost; send nothing")
    args = ap.parse_args(argv)

    clips = clips_from_html(Path(args.clips_html))
    key = {r["clip_id"]: r["filename"] for r in load_csv(Path(args.key))}
    by_name = {key_of(f): cid for cid, f in key.items()}
    gt = {key_of(r["filename"]): r for r in load_csv(Path(args.ground_truth))}
    refs = [(by_name[key_of(f)], gt[key_of(f)]) for f in REFERENCES if key_of(f) in by_name and key_of(f) in gt]
    targets = sorted(cid for cid in clips if cid in key and key_of(key[cid]) not in {key_of(f) for f in REFERENCES})

    ref_parts = []
    for n, (cid, g) in enumerate(refs, 1):
        ref_parts += [{"text": f"Reference drive {n}."}, video_part(clips[cid]),
                      {"text": f"A qualified coach scored reference drive {n}: overall {g['overall_score']}; "
                               + ", ".join(f"{p} {g[f'{p}_score']}" for p in PILLARS) + "."}]
    base_bytes = sum(len(clips[c]) for c, _ in refs)
    # Gemini counts video at roughly 300 tokens per second; the clips are a few seconds each.
    est_tokens = lambda b: 300 * max(1.0, b / 60_000) * 1.0  # noqa: E731  (about 60 KB per second at these settings)
    per_call_in = 3000 + sum(est_tokens(len(clips[c])) for c, _ in refs)
    est_usd = len(targets) * args.runs * ((per_call_in + est_tokens(max(len(v) for v in clips.values()))) * args.usd_per_mtok_in + 800 * args.usd_per_mtok_out) / 1e6
    print(f"{len(targets)} clips x {args.runs} runs, {len(refs)} references ({base_bytes / 1e6:.1f} MB). "
          f"Estimated cost about GBP {est_usd * GBP_PER_USD:.2f} (cap {args.budget_gbp:.2f}).")
    if args.dry_run:
        return 0

    model = args.model or pick_model(list_models())
    print(f"Model: {model}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, spent_usd = [], 0.0
    for run in range(1, args.runs + 1):
        for cid in targets:
            body = {
                "systemInstruction": {"parts": [{"text": prompt_text()}]},
                "contents": [{"role": "user", "parts": ref_parts + [{"text": "Now score this drive."}, video_part(clips[cid])]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": SCHEMA},
            }
            resp = _post(f"{API}/models/{model}:generateContent", body)
            use = resp.get("usageMetadata", {})
            spent_usd += (use.get("promptTokenCount", 0) * args.usd_per_mtok_in
                          + (use.get("candidatesTokenCount", 0) + use.get("thoughtsTokenCount", 0)) * args.usd_per_mtok_out) / 1e6
            try:
                ans = json.loads(resp["candidates"][0]["content"]["parts"][0]["text"])
            except (KeyError, IndexError, json.JSONDecodeError):
                ans = {"reason": f"unreadable answer: {json.dumps(resp)[:200]}"}
            rows.append({"filename": key[cid], "run": run, "model": model, **{k: ans.get(k, "") for k in SCHEMA["properties"]},
                         "prompt_tokens": use.get("promptTokenCount", ""), "output_tokens": use.get("candidatesTokenCount", "")})
            print(f"  run {run} {cid}: overall {ans.get('overall')} fault {ans.get('priority_fault')} (spent ~GBP {spent_usd * GBP_PER_USD:.2f})")
            if spent_usd * GBP_PER_USD > args.budget_gbp:
                print("Budget cap reached: stopping.")
                break
        else:
            continue
        break

    with (out / "video_judge_scores.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # first run as a checkpoint-ratings sheet for phase5_entry_tests.py --ratings <file>:video_model
    with (out / "video_judge_checkpoints.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["filename", *CPS])
        w.writeheader()
        for r in rows:
            if r["run"] == 1:
                w.writerow({"filename": r["filename"], **{c: r[c] for c in CPS}})
    print(f"Spent about GBP {spent_usd * GBP_PER_USD:.2f}. Wrote {out / 'video_judge_scores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
