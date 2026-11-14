#!/usr/bin/env python3
"""
Phase 0 baseline table.

Reads the outputs of batch_calibration_compare.py (auto, validated and an
auto repeat) and writes one dated table of per-clip overall and pillar error
against the coach scores, plus a summary.

Read-only on outputs: this script never runs the pipeline. It imports
config only for RULES_VERSION and PILLAR_RULE_MAX_DEDUCTION.

Usage:
  python phase0_baseline_table.py --run-dir calibration_output/phase0_2026-11-14 \
      --out-dir baselines/phase0_2026-11-14 --video-dir "<clip folder>"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import config  # noqa: E402

PILLARS = ("access", "tracking", "stability", "flow")
TIER_ORDER = ("Beginner", "Average", "Good Club", "Elite")
ANCHORS = (
    "setup_frame",
    "hands_start_up_frame",
    "front_foot_down_frame",
    "hands_peak_frame",
    "contact_frame",
    "follow_through_frame",
)
RULE_MEASUREMENT_KEY = {
    "A5": "A5_shoulder_hip_gap_deg",
    "T2": "T2_early_head_change",
    "S4": "S4_post_contact_rotation_deg",
    "F4": "F4_pause_frames",
}
SCORE_BANDS = ((0, 39, "<40"), (40, 59, "40-59"), (60, 79, "60-79"), (80, 100, ">=80"))
MODES = ("auto", "validated")
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _int_or_none(value: Any) -> int | None:
    text = _clean(value)
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{k.strip(): _clean(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def key_of(filename: str) -> str:
    return Path(filename).stem.casefold()


def tier_in_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()
    for token, tier in (("elite", "Elite"), ("good", "Good Club"), ("average", "Average"), ("beginner", "Beginner")):
        if token in stem:
            return tier
    return ""


def pillar_rule_map() -> dict[str, str]:
    """pillar -> the single active rule id (the active set has one rule per pillar)."""
    out: dict[str, str] = {}
    for pillar in PILLARS:
        rules = list(config.PILLAR_RULE_MAX_DEDUCTION.get(pillar, {}).keys())
        out[pillar] = ";".join(rules)
    return out


def find_report(mode_dir: Path, filename: str) -> Path | None:
    runs = mode_dir / "local_runs"
    if not runs.exists():
        return None
    wanted = key_of(filename)
    for child in runs.iterdir():
        if child.is_dir() and child.name.casefold() == wanted:
            candidates = list(child.glob("*_battingiq.json"))
            if candidates:
                return candidates[0]
    return None


def find_video(video_dir: Path | None, filename: str) -> Path | None:
    if video_dir is None or not video_dir.exists():
        return None
    wanted_name = Path(filename).name.casefold()
    wanted_stem = key_of(filename)
    stem_match = None
    for path in video_dir.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in VIDEO_SUFFIXES:
            continue
        if path.name.casefold() == wanted_name:
            return path
        if path.stem.casefold() == wanted_stem and stem_match is None:
            stem_match = path
    return stem_match


def audio_stream_count(video: Path | None) -> int | None:
    if video is None:
        return None
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=False,
        )
        return len([line for line in result.stdout.splitlines() if line.strip()])
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    result = subprocess.run([ffmpeg, "-hide_banner", "-i", str(video)], capture_output=True, text=True, check=False)
    return len(re.findall(r"Stream #\S+.*Audio:", result.stderr))


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Per-report extraction
# ---------------------------------------------------------------------------

def extract(report: dict[str, Any], pillar_rules: dict[str, str]) -> dict[str, Any]:
    pillars = report.get("pillars") or {}
    md = report.get("metadata") or {}
    measurements = md.get("rule_measurements") or {}
    evaluation = md.get("rule_evaluation") or {}
    suppressed = set(evaluation.get("suppressed_rule_ids") or [])
    anchors = md.get("anchor_frames") or {}
    conf = md.get("anchor_confidence") or {}
    diag = md.get("phase_diagnostics") or {}

    row: dict[str, Any] = {"sys_overall": _int_or_none(report.get("battingiq_score"))}
    pillar_sum = 0
    for pillar in PILLARS:
        info = pillars.get(pillar) or {}
        score = _int_or_none(info.get("score"))
        row[f"sys_{pillar}"] = score
        pillar_sum += score or 0
    row["pillar_sum_matches"] = "Y" if row["sys_overall"] == pillar_sum else "N"

    for pillar in PILLARS:
        rule = pillar_rules[pillar]
        rule_ids = rule.split(";") if rule else []
        faults = (pillars.get(pillar) or {}).get("faults") or []
        deduction = sum(int(f.get("deduction") or 0) for f in faults if f.get("rule_id") in rule_ids)
        measurement_values = [measurements.get(RULE_MEASUREMENT_KEY.get(r, ""), None) for r in rule_ids]
        measurement = measurement_values[0] if len(measurement_values) == 1 else ";".join(str(v) for v in measurement_values)
        if any(r in suppressed for r in rule_ids):
            status = "suppressed"
        elif all(v is None for v in measurement_values):
            status = "not_measured"
        elif any(f.get("rule_id") in rule_ids for f in faults):
            status = "fault"
        else:
            status = "no_fault"
        row[f"{pillar}_rule"] = rule
        row[f"{pillar}_measurement"] = measurement
        row[f"{pillar}_deduction"] = deduction
        row[f"{pillar}_status"] = status

    for anchor in ANCHORS:
        info = anchors.get(anchor) or {}
        row[f"{anchor}_sys"] = _int_or_none(info.get("original_frame", info.get("metric_index")))
        row[f"{anchor}_conf"] = conf.get(anchor, "")

    row["contact_method"] = diag.get("contact_method", "")
    row["contact_confidence"] = diag.get("contact_confidence", "")
    for key in ("fps", "width", "height", "total_frames", "duration_s", "detection_rate"):
        row[key] = md.get(key)
    row["priority_fix_rule"] = (report.get("priority_fix") or {}).get("rule_id", "") if report.get("priority_fix") else ""
    row["handedness_used"] = report.get("handedness", "")
    row["_measurements"] = {r: measurements.get(k) for r, k in RULE_MEASUREMENT_KEY.items()}
    return row


def determinism_signature(row: dict[str, Any]) -> dict[str, Any]:
    sig = {"overall": row["sys_overall"]}
    for pillar in PILLARS:
        sig[pillar] = row[f"sys_{pillar}"]
    for anchor in ANCHORS:
        sig[anchor] = row[f"{anchor}_sys"]
    for rule, value in row["_measurements"].items():
        sig[rule] = value
    return sig


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def band_of(score: int | None) -> str:
    if score is None:
        return ""
    for low, high, label in SCORE_BANDS:
        if low <= score <= high:
            return label
    return ""


def fmt(value: Any, nd: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def spearman(a: pd.Series, b: pd.Series) -> float | None:
    frame = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(frame) < 3:
        return None
    return float(frame["a"].rank().corr(frame["b"].rank()))


def loo_constant_mae(values: pd.Series) -> float | None:
    """MAE of predicting each clip as the mean of the other clips."""
    values = values.dropna().astype(float)
    n = len(values)
    if n < 2:
        return None
    total = values.sum()
    preds = (total - values) / (n - 1)
    return float((preds - values).abs().mean())


def md_table(header: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(c) for c in row) + " |")
    return lines


def mode_summary(df: pd.DataFrame, mode: str, reference_names: set[str]) -> list[str]:
    d = df[df["mode"] == mode].copy()
    lines = [f"## Mode: {mode}", ""]
    if d.empty:
        return lines + ["No rows.", ""]

    subsets = [("all clips", d), ("without reference clip", d[~d["filename"].isin(reference_names)])]
    rows = []
    for label, sub in subsets:
        rows.append([label, len(sub), sub["err_overall"].abs().mean(), *[sub[f"err_{p}"].abs().mean() for p in PILLARS]])
    lines += ["**Mean absolute error vs coach (points)**", ""]
    lines += md_table(["Set", "n", "Overall", *[p.title() for p in PILLARS]], rows)
    lines.append("")

    # Constant benchmark
    bench = [loo_constant_mae(d["coach_overall"]), *[loo_constant_mae(d[f"coach_{p}"]) for p in PILLARS]]
    system = [d["err_overall"].abs().mean(), *[d[f"err_{p}"].abs().mean() for p in PILLARS]]
    verdict = [
        "n/a" if (s is None or b is None or pd.isna(s)) else ("beats" if s < b else "does not beat")
        for s, b in zip(system, bench)
    ]
    lines += ["**Against a constant scorer (each clip predicted as the mean coach score of the other clips)**", ""]
    lines += md_table(
        ["Measure", "System MAE", "Constant MAE", "Verdict"],
        [[name, s, b, v] for name, s, b, v in zip(["Overall", *[p.title() for p in PILLARS]], system, bench, verdict)],
    )
    lines.append("")

    # Bias
    bias_rows = []
    for tier in TIER_ORDER:
        sub = d[d["tier"] == tier]
        if not sub.empty:
            mean_bias = sub["err_overall"].mean()
            bias_rows.append([f"tier {tier}", len(sub), mean_bias, "ABOVE 5" if abs(mean_bias) > 5 else ""])
    for _, _, label in SCORE_BANDS:
        sub = d[d["coach_band"] == label]
        if not sub.empty:
            mean_bias = sub["err_overall"].mean()
            bias_rows.append([f"coach band {label}", len(sub), mean_bias, "ABOVE 5" if abs(mean_bias) > 5 else ""])
    lines += ["**Mean signed bias, overall (system minus coach)**", ""]
    lines += md_table(["Group", "n", "Bias", "Flag"], bias_rows)
    lines.append("")

    rho = spearman(d["sys_overall"], d["coach_overall"])
    within5 = int((d["err_overall"].abs() <= 5).sum())
    lines.append(f"- Spearman rank correlation, system vs coach overall: {fmt(rho)} (n = {len(d)})")
    lines.append(f"- Within ±5 overall (continuity only): {within5}/{len(d)}")

    means = {t: d[d['tier'] == t]['sys_overall'].mean() for t in TIER_ORDER if not d[d['tier'] == t].empty}
    ordered = [means[t] for t in TIER_ORDER if t in means]
    monotonic = all(a < b for a, b in zip(ordered, ordered[1:]))
    lines.append("- Tier means (sanity check only): " + ", ".join(f"{t} {fmt(m, 1)}" for t, m in means.items())
                 + f"; monotonic: {'yes' if monotonic else 'no'}")

    # Status counts
    status_counts: dict[str, int] = {}
    free = []
    for _, r in d.iterrows():
        for p in PILLARS:
            status = r[f"{p}_status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            coach = r[f"coach_{p}"]
            if status == "no_fault" and coach is not None and not pd.isna(coach) and coach <= 15:
                free.append(f"{r['filename']} ({p}, coach {int(coach)})")
    lines.append("- Pillar results by status: " + ", ".join(f"{k} {v}" for k, v in sorted(status_counts.items())))
    lines.append(f"- Free 25s (no_fault while coach pillar ≤ 15): {len(free)}" + (": " + "; ".join(free) if free else ""))

    pf = d["priority_fix_rule"].replace("", "none").value_counts().to_dict()
    lines.append("- System priority fix per clip: " + ", ".join(f"{k} {v}" for k, v in pf.items())
                 + ". No coach priority fix exists to compare against (coach_notes is empty).")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir).expanduser() if args.video_dir else None

    gt_rows = [r for r in load_csv(Path(args.ground_truth)) if r.get("filmed_correctly", "Y").upper() == "Y"]
    excluded = [r["filename"] for r in load_csv(Path(args.ground_truth)) if r.get("filmed_correctly", "Y").upper() != "Y"]
    anchor_truth = {key_of(r["filename"]): r for r in load_csv(Path(args.anchor_truth))}
    split = {key_of(r["filename"]): r.get("split", "tuning") for r in load_csv(Path(args.split))} if Path(args.split).exists() else {}
    pillar_rules = pillar_rule_map()
    active_rules = ";".join(r for p in PILLARS for r in config.PILLAR_RULE_MAX_DEDUCTION.get(p, {}))

    manifest_path = run_dir / "dual_run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    git_commit = manifest.get("git_commit") or ""
    run_date = (manifest.get("run_timestamp") or date.today().isoformat())[:10]

    rows: list[dict[str, Any]] = []
    det_rows: list[dict[str, Any]] = []
    clip_meta: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    for gt in gt_rows:
        filename = gt["filename"]
        video = find_video(video_dir, filename)
        clip_meta[filename] = {
            "audio_streams": audio_stream_count(video),
            "md5": md5_of(video) if video else None,
            "path": str(video) if video else None,
        }
        auto_sig = None
        for mode in MODES:
            report_path = find_report(run_dir / mode, filename)
            if report_path is None:
                missing.append(f"{mode}:{filename}")
                continue
            ext = extract(json.loads(report_path.read_text()), pillar_rules)
            row: dict[str, Any] = {
                "run_date": run_date,
                "git_commit": git_commit,
                "rules_version": config.RULES_VERSION,
                "active_rules": active_rules,
                "mode": mode,
                "filename": filename,
                "tier": gt.get("tier", ""),
                "tier_in_filename": tier_in_filename(filename),
                "split": split.get(key_of(filename), "tuning"),
                "handedness_used": ext["handedness_used"],
                "coach_overall": _int_or_none(gt.get("overall_score")),
            }
            for p in PILLARS:
                row[f"coach_{p}"] = _int_or_none(gt.get(f"{p}_score"))
            row["sys_overall"] = ext["sys_overall"]
            for p in PILLARS:
                row[f"sys_{p}"] = ext[f"sys_{p}"]
            row["err_overall"] = None if row["sys_overall"] is None or row["coach_overall"] is None else row["sys_overall"] - row["coach_overall"]
            for p in PILLARS:
                s, c = row[f"sys_{p}"], row[f"coach_{p}"]
                row[f"err_{p}"] = None if s is None or c is None else s - c
            for p in PILLARS:
                for suffix in ("rule", "measurement", "deduction", "status"):
                    row[f"{p}_{suffix}"] = ext[f"{p}_{suffix}"]
            truth = anchor_truth.get(key_of(filename), {})
            for a in ANCHORS:
                sys_frame = ext[f"{a}_sys"]
                truth_frame = _int_or_none(truth.get(a))
                row[f"{a}_sys"] = sys_frame
                row[f"{a}_truth"] = truth_frame
                row[f"{a}_delta"] = None if sys_frame is None or truth_frame is None else sys_frame - truth_frame
                row[f"{a}_conf"] = ext[f"{a}_conf"]
            row["contact_method"] = ext["contact_method"]
            row["contact_confidence"] = ext["contact_confidence"]
            row["audio_streams"] = clip_meta[filename]["audio_streams"]
            for key in ("fps", "width", "height", "total_frames", "duration_s", "detection_rate"):
                row[key] = ext[key]
            row["priority_fix_rule"] = ext["priority_fix_rule"]
            row["pillar_sum_matches"] = ext["pillar_sum_matches"]
            row["repeat_identical"] = ""
            if mode == "auto":
                auto_sig = determinism_signature(ext)
                repeat_path = find_report(run_dir / "auto_repeat", filename)
                if repeat_path is not None:
                    rep_sig = determinism_signature(extract(json.loads(repeat_path.read_text()), pillar_rules))
                    diffs = [k for k in auto_sig if auto_sig[k] != rep_sig[k]]
                    row["repeat_identical"] = "Y" if not diffs else "N"
                    det_rows.append({"filename": filename, "identical": row["repeat_identical"],
                                     "fields_differing": ";".join(diffs),
                                     **{f"auto_{k}": v for k, v in auto_sig.items()},
                                     **{f"repeat_{k}": v for k, v in rep_sig.items()}})
            rows.append(row)

    if not rows:
        print("No reports found under", run_dir, file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df["coach_band"] = df["coach_overall"].map(band_of)
    table_cols = [c for c in df.columns if c != "coach_band"]
    df[table_cols].to_csv(out_dir / "baseline_table.csv", index=False)
    if det_rows:
        pd.DataFrame(det_rows).to_csv(out_dir / "determinism.csv", index=False)

    reference_names = {r["filename"] for r in gt_rows if split.get(key_of(r["filename"])) == "reference"}

    # Summary
    env = {
        "python": platform.python_version(),
        "mediapipe": _pkg_version("mediapipe"),
        "os": platform.platform(),
        "machine": platform.machine(),
        "ffmpeg": shutil.which("ffmpeg") or "NOT ON PATH",
    }
    lines = [
        f"# Phase 0 baseline — {run_date}",
        "",
        f"- Commit: `{git_commit}`; RULES_VERSION {config.RULES_VERSION}; active rules {active_rules.replace(';', ', ')}",
        f"- Environment: Python {env['python']}, MediaPipe {env['mediapipe']}, {env['os']} ({env['machine']}), ffmpeg {env['ffmpeg']}",
        f"- Clips: {len({r['filename'] for r in rows})}. Excluded: {', '.join(excluded) or 'none'} (filmed_correctly = N; no anchor truth).",
        f"- Reference clip (not held out): {', '.join(sorted(reference_names)) or 'none'}.",
        "- Coach-to-coach error: not yet measured (Phase 2). No comparison with targets is made here.",
        "",
    ]
    if missing:
        lines += ["**Missing reports:** " + ", ".join(missing), ""]
    for mode in MODES:
        lines += mode_summary(df, mode, reference_names)

    both = df.pivot_table(index="filename", columns="mode", values="sys_overall", aggfunc="first")
    if set(MODES).issubset(both.columns):
        delta = (both["auto"] - both["validated"]).abs().dropna().sort_values(ascending=False)
        lines += ["## Auto against validated", "",
                  f"- Mean |overall difference|: {fmt(delta.mean())} points (n = {len(delta)})",
                  "- Largest: " + ", ".join(f"{k} ({int(v)})" for k, v in delta.head(3).items()), ""]
    if det_rows:
        same = sum(1 for r in det_rows if r["identical"] == "Y")
        varied = sorted({f for r in det_rows for f in r["fields_differing"].split(";") if f})
        lines += ["## Determinism", "", f"- {same}/{len(det_rows)} clips identical across auto and auto_repeat."
                  + (f" Fields that varied: {', '.join(varied)}." if varied else ""), ""]
    lh_path = out_dir / "handedness_check.csv"
    if lh_path.exists():
        lh = load_csv(lh_path)
        changed = [r for r in lh if r.get("any_difference") == "Y"]
        lines += ["## Left-hander check", "",
                  f"- {len(lh)} runs of the left-handed clip with handedness=left; "
                  + ("no field differs from the right-handed run." if not changed
                     else "differences: " + "; ".join(f"{r['mode']}: {r['fields_differing']}" for r in changed)), ""]

    (out_dir / "baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    run_manifest = dict(manifest)
    run_manifest.update({
        "rules_version": config.RULES_VERSION,
        "active_rules": active_rules,
        "environment": env,
        "video_dir": str(video_dir) if video_dir else None,
        "clips": clip_meta,
    })
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print((out_dir / "baseline_summary.md").read_text())
    return 0


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # pragma: no cover
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ground-truth", default=str(SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"))
    parser.add_argument("--anchor-truth", default=str(SCRIPT_DIR / "anchor_truth.csv"))
    parser.add_argument("--split", default=str(SCRIPT_DIR / "heldout_split.csv"))
    parser.add_argument("--video-dir", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
