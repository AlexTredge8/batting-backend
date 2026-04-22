#!/usr/bin/env python3
"""
Extract the six detected anchor frames for each filmed-correctly calibration video
and build a self-contained HTML review sheet.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "anchor_frame_review"
DEFAULT_GROUND_TRUTH = SCRIPT_DIR / "ground_truth_scores.csv"
FALLBACK_GROUND_TRUTH = SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"
DEFAULT_VIDEO_DIR = SCRIPT_DIR / "ground_truth_videos"
DEFAULT_CACHED_REPORT_ROOT = SCRIPT_DIR / "calibration_output" / "local_runs"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
TIER_ORDER = {"Elite": 0, "Good Club": 1, "Average": 2, "Beginner": 3}
ANCHOR_KEYS = (
    "setup_frame",
    "hands_start_up_frame",
    "front_foot_down_frame",
    "hands_peak_frame",
    "contact_frame",
    "follow_through_frame",
)

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "battingiq-mplconfig"))
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("LOCAL_MODE", "1")

import pose_extractor  # noqa: E402

pose_extractor.LOCAL_MODEL_COMPLEXITY = 2

from run_analysis import run_full_analysis  # noqa: E402


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_int(value: Any) -> int | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def _is_truthy(value: Any) -> bool:
    return _clean_text(value).casefold() in {"y", "yes", "true", "1"}


def _resolve_ground_truth_path(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    if path.name == DEFAULT_GROUND_TRUTH.name and FALLBACK_GROUND_TRUTH.exists():
        return FALLBACK_GROUND_TRUTH.resolve()
    raise FileNotFoundError(f"Ground-truth file not found: {path}")


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({key: _clean_text(value) for key, value in row.items()})
        return rows


def _discover_videos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES
    )


def _discover_cached_reports(root: Path) -> list[Path]:
    reports: list[Path] = []
    if not root.exists():
        return reports
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        json_reports = sorted(path.glob("*_battingiq.json"))
        if json_reports:
            reports.append(json_reports[0])
    return reports


def _build_cached_lookup(reports: list[Path]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    exact_lookup: dict[str, list[Path]] = {}
    stem_lookup: dict[str, list[Path]] = {}
    for report in reports:
        exact_lookup.setdefault(report.name.casefold(), []).append(report)
        stem_lookup.setdefault(report.parent.name.casefold(), []).append(report)
    return exact_lookup, stem_lookup


def _match_cached_report(
    filename: str,
    exact_lookup: dict[str, list[Path]],
    stem_lookup: dict[str, list[Path]],
) -> Path | None:
    raw_name = _clean_text(filename)
    if not raw_name:
        return None

    stem = Path(raw_name).stem
    name_key = f"{stem}_battingiq.json".casefold()
    stem_key = Path(raw_name).name.casefold()
    candidates = exact_lookup.get(name_key) or stem_lookup.get(stem_key) or []
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return sorted(candidates, key=lambda path: (path.name.casefold(), str(path.parent).casefold()))[0]


def _load_cached_report(report_path: Path) -> dict[str, Any]:
    return json.loads(report_path.read_text(encoding="utf-8"))


def _build_video_lookup(videos: list[Path]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    exact_lookup: dict[str, list[Path]] = {}
    stem_lookup: dict[str, list[Path]] = {}
    for video in videos:
        exact_lookup.setdefault(video.name.casefold(), []).append(video)
        stem_lookup.setdefault(video.stem.casefold(), []).append(video)
    return exact_lookup, stem_lookup


def _match_video(
    filename: str,
    exact_lookup: dict[str, list[Path]],
    stem_lookup: dict[str, list[Path]],
) -> Path | None:
    raw_name = _clean_text(filename)
    if not raw_name:
        return None

    name_key = Path(raw_name).name.casefold()
    stem_key = Path(raw_name).stem.casefold()
    candidates = exact_lookup.get(name_key) or stem_lookup.get(stem_key) or []
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    desired_name = Path(raw_name).name.casefold()
    for candidate in candidates:
        if candidate.name.casefold() == desired_name:
            return candidate
    return sorted(candidates, key=lambda path: (path.name.casefold(), str(path.parent).casefold()))[0]


def _format_timestamp(frame_idx: int, fps: float) -> str:
    seconds = frame_idx / (fps or 30.0)
    return f"{seconds:.2f}s"


def _confidence_class(confidence: str) -> str:
    value = _clean_text(confidence).casefold()
    return value if value in {"high", "medium", "low"} else "unknown"


def _display_anchor_name(anchor_key: str) -> str:
    return anchor_key.removesuffix("_frame").replace("_", " ")


def _image_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _read_frame_with_fallback(cap: cv2.VideoCapture, frame_idx: int) -> tuple[bool, Any | None, int]:
    """Read the requested frame, with a few nearby retries if decode lands off target."""
    for offset in (0, -1, 1, -2, 2, -3, 3):
        candidate = max(0, frame_idx + offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ret, frame = cap.read()
        if ret and frame is not None:
            return True, frame, candidate
    return False, None, frame_idx


def _extract_anchor_images(
    video_path: Path,
    anchor_frames: dict[str, dict[str, int]],
    fps: float,
    video_stem: str,
    frames_dir: Path,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        for anchor_key in ANCHOR_KEYS:
            anchor_info = anchor_frames.get(anchor_key) or {}
            frame_idx = int(anchor_info.get("original_frame", 0))
            output_path = frames_dir / f"{video_stem}__{anchor_key.removesuffix('_frame')}.png"

            ret, frame, resolved_frame_idx = _read_frame_with_fallback(cap, frame_idx)
            if not ret or frame is None:
                raise RuntimeError(f"Could not read frame {frame_idx} for {video_path.name}")

            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Could not write frame image: {output_path}")

            output_rows.append(
                {
                    "anchor_key": anchor_key,
                    "anchor_label": _display_anchor_name(anchor_key),
                    "frame_idx": resolved_frame_idx,
                    "timestamp": _format_timestamp(resolved_frame_idx, fps),
                    "confidence": _clean_text((anchor_info or {}).get("confidence", "")),
                    "image_path": output_path,
                    "image_data": _image_data_uri(output_path),
                }
            )
    finally:
        cap.release()

    return output_rows


def _build_html(rows: list[dict[str, Any]]) -> str:
    tier_styles = {
        "Elite": "tier-elite",
        "Good Club": "tier-good",
        "Average": "tier-average",
        "Beginner": "tier-beginner",
    }

    body_rows: list[str] = []
    for row in rows:
        anchor_cells = []
        for anchor in row["anchors"]:
            confidence = _confidence_class(anchor["confidence"])
            anchor_cells.append(
                f"""
                <td class="anchor-cell">
                  <div class="anchor-card">
                    <div class="anchor-name">{html.escape(anchor["anchor_label"])}</div>
                    <img src="{anchor["image_data"]}" alt="{html.escape(anchor["anchor_label"])} frame">
                    <div class="frame-meta">Frame {anchor["frame_idx"]}</div>
                    <div class="frame-meta">{html.escape(anchor["timestamp"])}</div>
                    <div class="confidence confidence-{html.escape(confidence)}">{html.escape(anchor["confidence"] or "n/a")}</div>
                  </div>
                </td>
                """.strip()
            )

        body_rows.append(
            f"""
            <tr class="{tier_styles.get(row['tier'], '')}">
              <td class="filename">{html.escape(row["filename"])}</td>
              <td><span class="tier-badge {tier_styles.get(row['tier'], 'tier-unknown')}">{html.escape(row["tier"])}</span></td>
              <td class="manual-score">{html.escape(str(row["manual_overall"]))}</td>
              {''.join(anchor_cells)}
            </tr>
            """.strip()
        )

    style = """
    :root {
      --bg: #f6f7fb;
      --panel: #ffffff;
      --text: #132238;
      --muted: #5f6b7a;
      --border: #dbe2ea;
      --elite: #1b7f4a;
      --good: #1f5fbf;
      --average: #c17a16;
      --beginner: #b23b31;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(31, 95, 191, 0.08), transparent 30%),
        radial-gradient(circle at top right, rgba(27, 127, 74, 0.08), transparent 28%),
        var(--bg);
    }
    .page {
      padding: 24px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: -0.02em;
    }
    .subtitle {
      margin: 0 0 18px;
      color: var(--muted);
    }
    .summary {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin: 0 0 20px;
    }
    .summary-card {
      background: rgba(255,255,255,0.88);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 14px;
      min-width: 160px;
      box-shadow: 0 10px 30px rgba(20, 30, 50, 0.05);
    }
    .summary-card .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .summary-card .value {
      font-size: 20px;
      font-weight: 700;
      margin-top: 4px;
    }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: rgba(255,255,255,0.82);
      box-shadow: 0 20px 60px rgba(20, 30, 50, 0.08);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 1600px;
    }
    thead th {
      position: sticky;
      top: 0;
      background: #eef3f8;
      text-align: left;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      padding: 12px 10px;
      border-bottom: 1px solid var(--border);
      z-index: 1;
    }
    tbody td {
      border-top: 1px solid var(--border);
      padding: 10px;
      vertical-align: top;
    }
    tbody tr:hover {
      background: rgba(31, 95, 191, 0.03);
    }
    .filename {
      min-width: 220px;
      font-weight: 600;
    }
    .manual-score {
      font-weight: 700;
      font-size: 18px;
      min-width: 90px;
    }
    .tier-badge {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      color: white;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .tier-elite { background: var(--elite); }
    .tier-good { background: var(--good); }
    .tier-average { background: var(--average); }
    .tier-beginner { background: var(--beginner); }
    .tier-unknown { background: #687282; }
    .anchor-cell {
      width: 200px;
      min-width: 200px;
    }
    .anchor-card {
      display: flex;
      flex-direction: column;
      gap: 6px;
      align-items: center;
      text-align: center;
    }
    .anchor-name {
      font-size: 13px;
      font-weight: 700;
      text-transform: capitalize;
    }
    .anchor-card img {
      width: 200px;
      height: auto;
      display: block;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #fff;
    }
    .frame-meta {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.2;
    }
    .confidence {
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .confidence-high { background: #e3f4ea; color: #1b7f4a; }
    .confidence-medium { background: #fff1d8; color: #a86d0f; }
    .confidence-low { background: #f9e0de; color: #b23b31; }
    .confidence-unknown { background: #e8edf3; color: #415063; }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BattingIQ Anchor Frame Review</title>
  <style>{style}</style>
</head>
<body>
  <div class="page">
    <h1>BattingIQ Anchor Frame Review</h1>
    <p class="subtitle">Each row shows the six detected anchors, extracted from the original video frames for a fast human check.</p>
    <div class="summary">
      <div class="summary-card"><div class="label">Videos</div><div class="value">{len(rows)}</div></div>
      <div class="summary-card"><div class="label">Frames exported</div><div class="value">{len(rows) * len(ANCHOR_KEYS)}</div></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Video</th>
            <th>Tier</th>
            <th>Manual Overall</th>
            <th>Setup</th>
            <th>Hands Start Up</th>
            <th>Front Foot Down</th>
            <th>Hands Peak</th>
            <th>Contact</th>
            <th>Follow Through</th>
          </tr>
        </thead>
        <tbody>
          {''.join(body_rows)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract anchor frames and build an HTML review page.")
    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH.name),
        help="Path to the ground-truth CSV. Defaults to ground_truth_scores.csv in this directory.",
    )
    parser.add_argument(
        "--video-dir",
        default=str(DEFAULT_VIDEO_DIR),
        help="Directory containing the source batting videos.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for the review output.",
    )
    parser.add_argument(
        "--cached-report-root",
        default=str(DEFAULT_CACHED_REPORT_ROOT),
        help="Directory containing cached *_battingiq.json reports for fallback anchor metadata.",
    )
    args = parser.parse_args()

    ground_truth_path = _resolve_ground_truth_path((SCRIPT_DIR / args.ground_truth).expanduser())
    video_dir = Path(args.video_dir).expanduser()
    if not video_dir.is_absolute():
        cwd_candidate = Path.cwd() / video_dir
        video_dir = cwd_candidate if cwd_candidate.exists() else (SCRIPT_DIR / video_dir)
    video_dir = video_dir.resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    frames_dir = output_dir / "frames"

    rows = _load_rows(ground_truth_path)
    rows = [row for row in rows if _is_truthy(row.get("filmed_correctly"))]
    if not rows:
        raise SystemExit(f"No filmed-correctly rows found in {ground_truth_path}")

    videos = _discover_videos(video_dir)
    if not videos:
        raise SystemExit(f"No supported video files found in {video_dir}")

    exact_lookup, stem_lookup = _build_video_lookup(videos)
    cached_root = Path(args.cached_report_root).expanduser()
    if not cached_root.is_absolute():
        cached_root = (SCRIPT_DIR / cached_root).resolve()
    cached_reports = _discover_cached_reports(cached_root)
    cached_exact_lookup, cached_stem_lookup = _build_cached_lookup(cached_reports)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[dict[str, Any]] = []
    failures: list[str] = []

    for row in rows:
        filename = _clean_text(row.get("filename"))
        tier = _clean_text(row.get("tier")) or "Unknown"
        manual_overall = _coerce_int(row.get("overall_score"))
        video_path = _match_video(filename, exact_lookup, stem_lookup)
        if video_path is None:
            failures.append(f"{filename}: video not found in {video_dir}")
            continue

        source = "fresh"
        try:
            report = run_full_analysis(str(video_path), output_dir=str(output_dir / "analysis_runs" / video_path.stem))
        except Exception:
            cached_report = _match_cached_report(filename, cached_exact_lookup, cached_stem_lookup)
            if cached_report is None:
                failures.append(f"{filename}: analysis failed and no cached report found")
                continue
            report = _load_cached_report(cached_report)
            source = "cached"

        metadata = report.get("metadata", {}) or {}
        anchor_frames = metadata.get("anchor_frames", {}) or {}
        anchor_confidence = metadata.get("anchor_confidence", {}) or {}
        fps = float(metadata.get("fps") or report.get("phases", {}).get("contact", {}).get("fps") or 30.0)

        try:
            anchor_rows = _extract_anchor_images(video_path, anchor_frames, fps, video_path.stem, frames_dir)
        except Exception as exc:  # pragma: no cover - best-effort batch processing
            failures.append(f"{filename}: {exc}")
            continue

        for anchor_row in anchor_rows:
            anchor_row["confidence"] = _clean_text(anchor_confidence.get(anchor_row["anchor_key"], anchor_row["confidence"]))

        extracted.append(
            {
                "filename": filename or video_path.name,
                "tier": tier,
                "manual_overall": manual_overall if manual_overall is not None else "n/a",
                "tier_sort": TIER_ORDER.get(tier, 99),
                "anchors": anchor_rows,
            }
        )
        print(f"Processed {video_path.name} ({source})")

    extracted.sort(key=lambda row: (row["tier_sort"], row["filename"].casefold()))

    html_path = output_dir / "frame_review.html"
    html_path.write_text(_build_html(extracted), encoding="utf-8")

    print()
    print(f"Videos processed: {len(extracted)}")
    print(f"HTML review: {html_path}")
    print(f"Frames directory: {frames_dir}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
