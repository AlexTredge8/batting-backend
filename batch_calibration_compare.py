#!/usr/bin/env python3
"""
Run BattingIQ calibration against coach ground-truth scores and produce
local-vs-Railway comparison outputs with tier separation diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "battingiq-mplconfig"))
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
OUTPUT_DIR = SCRIPT_DIR / "calibration_output"
DEFAULT_GROUND_TRUTH = SCRIPT_DIR / "ground_truth_scores.csv"
FALLBACK_GROUND_TRUTH = SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"
DEFAULT_VIDEO_DIR = SCRIPT_DIR / "ground_truth_videos"
DEFAULT_CACHED_REPORT_ROOT = SCRIPT_DIR / "calibration_batch_output"
RAILWAY_DEFAULT = "https://web-production-e9c26.up.railway.app"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
PILLARS = ("access", "tracking", "stability", "flow")
TIER_ORDER = ("Beginner", "Average", "Good Club", "Elite")
REQUIRED_COLUMNS = {
    "filename",
    "tier",
    "access_score",
    "tracking_score",
    "stability_score",
    "flow_score",
    "overall_score",
    "filmed_correctly",
    "coach_notes",
}
OPTIONAL_CAM_COLUMNS = {
    "cam_access_score",
    "cam_tracking_score",
    "cam_stability_score",
    "cam_flow_score",
    "cam_overall_score",
    "cam_notes",
}
LOCAL_FULL_QUALITY_MODEL_COMPLEXITY = 2
HTTP_TIMEOUT_SECONDS = 60


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalise_tier(value: Any) -> str:
    return _clean_text(value)


def _is_filmed_correctly(value: Any) -> bool:
    text = _clean_text(value).casefold()
    return text in {"y", "yes", "true", "1"}


def _normalise_confidence(value: Any) -> str:
    text = _clean_text(value).casefold()
    if text in {"high", "medium", "low"}:
        return text
    return text


def _resolve_input_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if not candidate.is_absolute():
        cwd_candidate = Path.cwd() / candidate
        if cwd_candidate.exists():
            return cwd_candidate.resolve()

        script_candidate = SCRIPT_DIR / candidate
        if script_candidate.exists():
            return script_candidate.resolve()

    if candidate.name == DEFAULT_GROUND_TRUTH.name and FALLBACK_GROUND_TRUTH.exists():
        return FALLBACK_GROUND_TRUTH.resolve()

    raise FileNotFoundError(f"Ground-truth file not found: {path_value}")


def _load_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [str(column).strip() for column in df.columns]

    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise SystemExit(
            f"{path} is missing required columns: {', '.join(missing)}"
        )

    for column in df.columns:
        df[column] = df[column].map(_clean_text)

    return df


def _discover_videos(video_root: Path) -> list[Path]:
    return sorted(
        path
        for path in video_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES
    )


def _build_video_lookup(videos: list[Path]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    exact_lookup: dict[str, list[Path]] = {}
    stem_lookup: dict[str, list[Path]] = {}
    for video in videos:
        exact_lookup.setdefault(video.name.casefold(), []).append(video)
        stem_lookup.setdefault(video.stem.casefold(), []).append(video)
    return exact_lookup, stem_lookup


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


def _match_cached_report(filename: str, exact_lookup: dict[str, list[Path]], stem_lookup: dict[str, list[Path]]) -> Path | None:
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
    with report_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _match_video(filename: str, exact_lookup: dict[str, list[Path]], stem_lookup: dict[str, list[Path]]) -> Path | None:
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


def _prepare_local_runner() -> Any:
    os.environ["LOCAL_MODE"] = "1"
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    import pose_extractor  # noqa: WPS433
    from run_analysis import run_full_analysis  # noqa: WPS433

    pose_extractor.LOCAL_MODEL_COMPLEXITY = LOCAL_FULL_QUALITY_MODEL_COMPLEXITY
    return run_full_analysis


def _safe_sum(values: list[int | None]) -> int | None:
    if any(value is None for value in values):
        return None
    return int(sum(int(value) for value in values if value is not None))


def _extract_report_scores(report: dict[str, Any]) -> dict[str, Any]:
    pillars = report.get("pillars") or {}
    phases = report.get("phases") or {}
    contact = phases.get("contact") or {}
    metadata = report.get("metadata") or {}

    pillar_scores = {
        pillar: _coerce_int((pillars.get(pillar) or {}).get("score"))
        for pillar in PILLARS
    }
    overall_from_pillars = _safe_sum([pillar_scores[pillar] for pillar in PILLARS])

    raw_overall = _coerce_int(report.get("battingiq_score"))
    detector_version = (
        metadata.get("detector_version")
        or metadata.get("contact_detector_version")
        or contact.get("detector_version")
        or ""
    )

    confidence = (
        metadata.get("estimated_contact_confidence")
        or (metadata.get("phase_diagnostics") or {}).get("estimated_contact_confidence")
        or contact.get("confidence")
        or ""
    )

    return {
        "battingiq_score": raw_overall,
        "pillar_scores": pillar_scores,
        "overall_from_pillars": overall_from_pillars,
        "detector_version": _clean_text(detector_version),
        "anchor_detector_version": _clean_text(metadata.get("anchor_detector_version")),
        "contact_confidence": _normalise_confidence(confidence),
    }


def _run_local_analysis(video_path: Path, output_root: Path, run_full_analysis: Any) -> dict[str, Any]:
    output_dir = output_root / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_full_analysis(str(video_path), output_dir=str(output_dir))


def _run_local_analysis_with_fallback(
    video_path: Path,
    output_root: Path,
    run_full_analysis: Any,
    cached_exact_lookup: dict[str, list[Path]],
    cached_stem_lookup: dict[str, list[Path]],
    allow_cached_fallback: bool,
    logger: logging.Logger,
) -> tuple[dict[str, Any] | None, str]:
    try:
        return _run_local_analysis(video_path, output_root, run_full_analysis), "fresh"
    except Exception as exc:  # pragma: no cover - batch path should continue on failure
        logger.exception("LOCAL FAILED %s: %s", video_path.name, exc)
        if not allow_cached_fallback:
            return None, "failed"

        cached_report = _match_cached_report(video_path.name, cached_exact_lookup, cached_stem_lookup)
        if cached_report is None:
            logger.warning("No cached local report found for %s", video_path.name)
            return None, "failed"

        logger.warning("Falling back to cached local report for %s -> %s", video_path.name, cached_report)
        return _load_cached_report(cached_report), "cached"


def _run_live_analysis(video_path: Path, api_base: str, logger: logging.Logger) -> dict[str, Any]:
    endpoint = f"{api_base.rstrip('/')}/analyse"
    last_error: Exception | None = None

    for attempt in (1, 2):
        files: dict[str, Any] | None = None
        try:
            with video_path.open("rb") as handle:
                files = {
                    "file": (video_path.name, handle, {
                        ".mp4": "video/mp4",
                        ".mov": "video/quicktime",
                        ".avi": "video/x-msvideo",
                        ".mkv": "video/x-matroska",
                        ".m4v": "video/x-m4v",
                        ".webm": "video/webm",
                    }.get(video_path.suffix.casefold(), "application/octet-stream"))
                }
                response = requests.post(
                    endpoint,
                    files=files,
                    timeout=HTTP_TIMEOUT_SECONDS,
                    headers={"User-Agent": "BattingIQ calibration compare/1.0"},
                )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == 1:
                logger.warning(
                    "Railway request failed for %s on attempt %s: %s",
                    video_path.name,
                    attempt,
                    exc,
                )
                time.sleep(1.5)
                continue
            break

    assert last_error is not None
    raise RuntimeError(f"Railway analysis failed for {video_path.name}: {last_error}") from last_error


def _build_row(
    row: pd.Series,
    local_result: dict[str, Any] | None,
    railway_result: dict[str, Any] | None,
    local_source: str,
    railway_source: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    manual_scores = {pillar: _coerce_int(row.get(f"{pillar}_score")) for pillar in PILLARS}
    manual_overall = _coerce_int(row.get("overall_score"))

    local_scores = (local_result or {}).get("pillar_scores", {})
    railway_scores = (railway_result or {}).get("pillar_scores", {})

    local_overall = _safe_sum([local_scores.get(pillar) for pillar in PILLARS])
    railway_overall = _safe_sum([railway_scores.get(pillar) for pillar in PILLARS])

    row_dict: dict[str, Any] = {
        "filename": _clean_text(row.get("filename")),
        "tier": _normalise_tier(row.get("tier")),
        "filmed_correctly": _clean_text(row.get("filmed_correctly")).upper() or "Y",
        "manual_overall": manual_overall,
        "local_overall": local_overall,
        "railway_overall": railway_overall,
        "local_gap": None if local_overall is None or manual_overall is None else local_overall - manual_overall,
        "railway_gap": None if railway_overall is None or manual_overall is None else railway_overall - manual_overall,
        "local_vs_railway_drift": None
        if local_overall is None or railway_overall is None
        else local_overall - railway_overall,
        "local_detector_version": (local_result or {}).get("detector_version", ""),
        "railway_detector_version": (railway_result or {}).get("detector_version", ""),
        "local_contact_confidence": (local_result or {}).get("contact_confidence", ""),
        "railway_contact_confidence": (railway_result or {}).get("contact_confidence", ""),
        "local_source": local_source,
        "railway_source": railway_source,
        "notes": _clean_text(row.get("coach_notes")),
        "analysis_status": _analysis_status(local_result, railway_result),
    }

    for pillar in PILLARS:
        row_dict[f"{pillar}_manual"] = manual_scores[pillar]
        row_dict[f"{pillar}_local"] = local_scores.get(pillar)
        row_dict[f"{pillar}_railway"] = railway_scores.get(pillar)
        row_dict[f"{pillar}_local_gap"] = (
            None
            if manual_scores[pillar] is None or local_scores.get(pillar) is None
            else int(local_scores[pillar]) - int(manual_scores[pillar])
        )
        row_dict[f"{pillar}_railway_gap"] = (
            None
            if manual_scores[pillar] is None or railway_scores.get(pillar) is None
            else int(railway_scores[pillar]) - int(manual_scores[pillar])
        )

    cam_present = any(column in row.index for column in OPTIONAL_CAM_COLUMNS)
    if cam_present:
        cam_scores = {
            pillar: _coerce_int(row.get(f"cam_{pillar}_score"))
            for pillar in PILLARS
        }
        cam_overall = _coerce_int(row.get("cam_overall_score"))
        row_dict["cam_access"] = cam_scores["access"]
        row_dict["cam_tracking"] = cam_scores["tracking"]
        row_dict["cam_stability"] = cam_scores["stability"]
        row_dict["cam_flow"] = cam_scores["flow"]
        row_dict["cam_overall"] = cam_overall
        row_dict["cam_access_minus_manual"] = (
            None if cam_scores["access"] is None or manual_scores["access"] is None
            else cam_scores["access"] - manual_scores["access"]
        )
        row_dict["cam_tracking_minus_manual"] = (
            None if cam_scores["tracking"] is None or manual_scores["tracking"] is None
            else cam_scores["tracking"] - manual_scores["tracking"]
        )
        row_dict["cam_stability_minus_manual"] = (
            None if cam_scores["stability"] is None or manual_scores["stability"] is None
            else cam_scores["stability"] - manual_scores["stability"]
        )
        row_dict["cam_flow_minus_manual"] = (
            None if cam_scores["flow"] is None or manual_scores["flow"] is None
            else cam_scores["flow"] - manual_scores["flow"]
        )
        row_dict["cam_overall_minus_manual"] = (
            None if cam_overall is None or manual_overall is None else cam_overall - manual_overall
        )
        row_dict["cam_notes"] = _clean_text(row.get("cam_notes"))

    if local_result and local_result.get("battingiq_score") is not None and local_overall is not None:
        raw_local = int(local_result["battingiq_score"])
        if raw_local != local_overall:
            logger.warning(
                "Local raw battingiq_score (%s) != pillar sum (%s) for %s",
                raw_local,
                local_overall,
                row_dict["filename"],
            )
    if railway_result and railway_result.get("battingiq_score") is not None and railway_overall is not None:
        raw_railway = int(railway_result["battingiq_score"])
        if raw_railway != railway_overall:
            logger.warning(
                "Railway raw battingiq_score (%s) != pillar sum (%s) for %s",
                raw_railway,
                railway_overall,
                row_dict["filename"],
            )

    return row_dict


def _analysis_status(local_result: dict[str, Any] | None, railway_result: dict[str, Any] | None) -> str:
    if local_result and railway_result:
        return "both_success"
    if local_result and not railway_result:
        return "local_only"
    if railway_result and not local_result:
        return "railway_only"
    return "both_failed"


def _build_output_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        "filename",
        "tier",
        "filmed_correctly",
        "manual_overall",
        "local_overall",
        "railway_overall",
        "local_gap",
        "railway_gap",
        "local_vs_railway_drift",
        "access_manual",
        "access_local",
        "access_railway",
        "access_local_gap",
        "access_railway_gap",
        "tracking_manual",
        "tracking_local",
        "tracking_railway",
        "tracking_local_gap",
        "tracking_railway_gap",
        "stability_manual",
        "stability_local",
        "stability_railway",
        "stability_local_gap",
        "stability_railway_gap",
        "flow_manual",
        "flow_local",
        "flow_railway",
        "flow_local_gap",
        "flow_railway_gap",
        "local_detector_version",
        "railway_detector_version",
        "local_contact_confidence",
        "railway_contact_confidence",
        "local_source",
        "railway_source",
        "notes",
        "analysis_status",
    ]
    for extra in [
        "cam_access",
        "cam_tracking",
        "cam_stability",
        "cam_flow",
        "cam_overall",
        "cam_access_minus_manual",
        "cam_tracking_minus_manual",
        "cam_stability_minus_manual",
        "cam_flow_minus_manual",
        "cam_overall_minus_manual",
        "cam_notes",
    ]:
        if extra in df.columns:
            columns.append(extra)
    return columns


def _prepare_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric_columns = [
        "manual_overall",
        "local_overall",
        "railway_overall",
        "local_gap",
        "railway_gap",
        "local_vs_railway_drift",
        "access_manual",
        "access_local",
        "access_railway",
        "access_local_gap",
        "access_railway_gap",
        "tracking_manual",
        "tracking_local",
        "tracking_railway",
        "tracking_local_gap",
        "tracking_railway_gap",
        "stability_manual",
        "stability_local",
        "stability_railway",
        "stability_local_gap",
        "stability_railway_gap",
        "flow_manual",
        "flow_local",
        "flow_railway",
        "flow_local_gap",
        "flow_railway_gap",
        "cam_access",
        "cam_tracking",
        "cam_stability",
        "cam_flow",
        "cam_overall",
        "cam_access_minus_manual",
        "cam_tracking_minus_manual",
        "cam_stability_minus_manual",
        "cam_flow_minus_manual",
        "cam_overall_minus_manual",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _group_tier_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tier in list(TIER_ORDER) + sorted(set(df["tier"].dropna().astype(str)) - set(TIER_ORDER)):
        tier_df = df[df["tier"] == tier]
        if tier_df.empty:
            continue
        rows.append(
            {
                "tier": tier,
                "video_count": int(len(tier_df)),
                "mean_manual_score": tier_df["manual_overall"].mean(),
                "mean_local_score": tier_df["local_overall"].mean(),
                "mean_railway_score": tier_df["railway_overall"].mean(),
                "mean_local_gap": tier_df["local_gap"].mean(),
                "mean_railway_gap": tier_df["railway_gap"].mean(),
                "min_manual": tier_df["manual_overall"].min(),
                "max_manual": tier_df["manual_overall"].max(),
                "min_local": tier_df["local_overall"].min(),
                "max_local": tier_df["local_overall"].max(),
                "min_railway": tier_df["railway_overall"].min(),
                "max_railway": tier_df["railway_overall"].max(),
                "score_range_local": tier_df["local_overall"].max() - tier_df["local_overall"].min(),
                "score_range_railway": tier_df["railway_overall"].max() - tier_df["railway_overall"].min(),
            }
        )
    return pd.DataFrame(rows)


def _fmt(value: Any, precision: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}" if isinstance(value, float) else str(value)
    return str(value)


def _summary_lines(df: pd.DataFrame, tier_summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    total_rows = len(df)
    local_rows = int(df["local_overall"].notna().sum())
    railway_rows = int(df["railway_overall"].notna().sum())
    both_rows = int(((df["local_overall"].notna()) & (df["railway_overall"].notna())).sum())
    both_failed = int(((df["local_overall"].isna()) & (df["railway_overall"].isna())).sum())

    lines.append("BattingIQ calibration comparison summary")
    lines.append(f"Total processed videos: {total_rows}")
    lines.append(f"Local results present: {local_rows}")
    lines.append(f"Railway results present: {railway_rows}")
    lines.append(f"Rows with both analyses: {both_rows}")
    lines.append(f"Rows with both analyses missing: {both_failed}")
    if "local_source" in df.columns:
        cached_local = int((df["local_source"] == "cached").sum())
        lines.append(f"Cached local fallbacks used: {cached_local}")
    lines.append("")

    if df["local_gap"].notna().any():
        lines.append(f"Mean local gap vs manual: {_fmt(df['local_gap'].mean())}")
    if df["railway_gap"].notna().any():
        lines.append(f"Mean railway gap vs manual: {_fmt(df['railway_gap'].mean())}")
    if df["local_vs_railway_drift"].notna().any():
        lines.append(f"Mean local vs railway drift: {_fmt(df['local_vs_railway_drift'].mean())}")
        if df["local_vs_railway_drift"].notna().sum() > 1:
            lines.append(f"Drift standard deviation: {_fmt(df['local_vs_railway_drift'].std(ddof=0))}")
    lines.append("")

    lines.append("Tier separation analysis (local model)")
    local_mins = tier_summary.set_index("tier")["min_local"].to_dict() if not tier_summary.empty else {}
    for tier in TIER_ORDER:
        lines.append(f"  Lowest {tier}: {_fmt(local_mins.get(tier))}")
    if "Beginner" in local_mins and "Good Club" in local_mins:
        lines.append(
            f"  Beginner -> Good Club gap: {_fmt(local_mins['Good Club'] - local_mins['Beginner'])}"
        )
    if "Average" in local_mins and "Elite" in local_mins:
        lines.append(
            f"  Average -> Elite gap: {_fmt(local_mins['Elite'] - local_mins['Average'])}"
        )
    lines.append("")

    lines.append("Score compression analysis (local model)")
    for tier in TIER_ORDER:
        tier_row = tier_summary[tier_summary["tier"] == tier]
        if tier_row.empty:
            continue
        score_range = tier_row.iloc[0]["score_range_local"]
        lines.append(f"  {tier}: range {_fmt(score_range)}")
        if pd.notna(score_range) and float(score_range) < 20:
            lines.append("    -> Compression warning: spread is below the 20-point target.")
    lines.append("")

    lines.append("Railway vs local drift")
    if df["local_vs_railway_drift"].notna().any():
        drift = df["local_vs_railway_drift"].dropna()
        max_pos = drift.idxmax()
        max_neg = drift.idxmin()
        pos_row = df.loc[max_pos]
        neg_row = df.loc[max_neg]
        lines.append(f"  Mean local - railway drift: {_fmt(drift.mean())}")
        lines.append(f"  Std dev drift: {_fmt(drift.std(ddof=0))}")
        lines.append(
            f"  Max positive drift: {_fmt(pos_row['local_vs_railway_drift'])} ({pos_row['filename']})"
        )
        lines.append(
            f"  Max negative drift: {_fmt(neg_row['local_vs_railway_drift'])} ({neg_row['filename']})"
        )
    else:
        lines.append("  No paired local/railway scores available.")
    lines.append("")

    if any(col in df.columns for col in ("cam_access", "cam_overall")):
        lines.append("Reviewer comparison (Cam vs Alex manual)")
        reviewer_cols = [
            "cam_access_minus_manual",
            "cam_tracking_minus_manual",
            "cam_stability_minus_manual",
            "cam_flow_minus_manual",
            "cam_overall_minus_manual",
        ]
        for column in reviewer_cols:
            if column in df.columns and df[column].notna().any():
                lines.append(f"  Mean {column.replace('_', ' ')}: {_fmt(df[column].mean())}")
    lines.append("")

    recommendations: list[str] = []
    if "Beginner" in local_mins and "Good Club" in local_mins:
        gap = local_mins["Good Club"] - local_mins["Beginner"]
        if pd.notna(gap) and gap < 20:
            recommendations.append(
                "Score compression indicates the binary rules are still too flat; graduate the thresholds."
            )
        elif pd.notna(gap) and gap < 25:
            recommendations.append(
                "Tier separation is improving but still below the 25-point target between Beginner and Good Club."
            )
    if "Average" in local_mins and "Elite" in local_mins:
        gap = local_mins["Elite"] - local_mins["Average"]
        if pd.notna(gap) and gap < 25:
            recommendations.append(
                "Average-to-Elite separation is still under the 25-point calibration target."
            )
    if df["local_vs_railway_drift"].notna().any():
        mean_drift = float(df["local_vs_railway_drift"].dropna().mean())
        if abs(mean_drift) > 3:
            recommendations.append(
                "Railway/local drift is material; verify the optimized inference path is not changing score semantics."
            )
    if not recommendations:
        recommendations.append("Calibration looks stable enough for further threshold tuning.")

    lines.append("Recommendations")
    for recommendation in recommendations:
        lines.append(f"  - {recommendation}")

    return lines


def _write_outputs(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = output_dir / "battingiq_calibration_comparison_detailed.csv"
    tier_summary_path = output_dir / "battingiq_calibration_tier_summary.csv"
    metrics_path = output_dir / "battingiq_calibration_metrics.txt"

    detailed_columns = _build_output_columns(df)
    df.reindex(columns=detailed_columns).to_csv(detailed_path, index=False, encoding="utf-8")

    stats_df = df[~(df["local_overall"].isna() & df["railway_overall"].isna())].copy()
    tier_summary = _group_tier_summary(stats_df)
    tier_summary.to_csv(tier_summary_path, index=False, encoding="utf-8")

    metrics_lines = _summary_lines(stats_df, tier_summary)
    metrics_path.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    return detailed_path, tier_summary_path, metrics_path


def _setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("batch_calibration_compare")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(output_dir / "batch_calibration.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local and Railway BattingIQ scores against coach ground truth.")
    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH.name),
        help="Path to the coach ground-truth CSV. Defaults to ground_truth_scores.csv in the script directory.",
    )
    parser.add_argument(
        "--video-dir",
        default=str(DEFAULT_VIDEO_DIR),
        help="Directory containing the source batting videos. Defaults to ground_truth_videos next to the script.",
    )
    parser.add_argument(
        "--api-base",
        default=RAILWAY_DEFAULT,
        help="Railway API base URL for live comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for calibration outputs.",
    )
    parser.add_argument(
        "--cached-report-root",
        default=str(DEFAULT_CACHED_REPORT_ROOT),
        help="Directory containing cached *_battingiq.json reports for fallback local results.",
    )
    parser.add_argument(
        "--allow-cached-local-fallback",
        action="store_true",
        help="Use cached local JSON reports if fresh local pose extraction fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ground_truth_path = _resolve_input_path(args.ground_truth)
    video_dir = Path(args.video_dir).expanduser()
    if not video_dir.is_absolute():
        cwd_candidate = Path.cwd() / video_dir
        if cwd_candidate.exists():
            video_dir = cwd_candidate
        else:
            video_dir = (SCRIPT_DIR / video_dir).resolve()

    if not video_dir.exists() or not video_dir.is_dir():
        raise SystemExit(
            f"Video directory not found: {video_dir}\n"
            f"Provide --video-dir pointing at the raw calibration videos or create {DEFAULT_VIDEO_DIR}."
        )

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(output_dir)

    logger.info("Ground truth CSV: %s", ground_truth_path)
    logger.info("Video directory: %s", video_dir)
    logger.info("Railway API: %s", args.api_base)
    logger.info("Cached report root: %s", args.cached_report_root)

    run_full_analysis = _prepare_local_runner()
    ground_truth_df = _load_ground_truth(ground_truth_path)
    videos = _discover_videos(video_dir)
    if not videos:
        raise SystemExit(f"No supported video files found in {video_dir}")

    exact_lookup, stem_lookup = _build_video_lookup(videos)
    cached_root = Path(args.cached_report_root).expanduser()
    if not cached_root.is_absolute():
        cached_root = (SCRIPT_DIR / cached_root).resolve()
    cached_reports = _discover_cached_reports(cached_root)
    cached_exact_lookup, cached_stem_lookup = _build_cached_lookup(cached_reports)

    processed_rows: list[dict[str, Any]] = []
    skipped_not_filmed = 0
    skipped_missing_video = 0
    local_success = 0
    railway_success = 0

    for index, row in ground_truth_df.iterrows():
        filename = _clean_text(row.get("filename"))
        tier = _normalise_tier(row.get("tier"))
        filmed = _is_filmed_correctly(row.get("filmed_correctly"))

        if not filmed:
            skipped_not_filmed += 1
            logger.info("SKIPPED %s [%s] - filmed_correctly != Y", filename, tier)
            continue

        video_path = _match_video(filename, exact_lookup, stem_lookup)
        if video_path is None:
            skipped_missing_video += 1
            logger.error("SKIPPED %s [%s] - file not found in %s", filename, tier, video_dir)
            continue

        logger.info("PROCESSING %s [%s] -> %s", filename, tier, video_path)
        local_result: dict[str, Any] | None = None
        railway_result: dict[str, Any] | None = None

        local_report, local_source = _run_local_analysis_with_fallback(
            video_path,
            output_dir / "local_runs",
            run_full_analysis,
            cached_exact_lookup,
            cached_stem_lookup,
            args.allow_cached_local_fallback,
            logger,
        )
        if local_report is not None:
            local_result = _extract_report_scores(local_report)
            local_success += 1
            logger.info("LOCAL OK %s [%s] (%s)", filename, tier, local_source)

        railway_source = "failed"
        try:
            railway_report = _run_live_analysis(video_path, args.api_base, logger)
            railway_result = _extract_report_scores(railway_report)
            railway_success += 1
            railway_source = "live"
            logger.info("RAILWAY OK %s [%s]", filename, tier)
        except Exception as exc:  # pragma: no cover - batch path should continue on failure
            logger.exception("RAILWAY FAILED %s [%s]: %s", filename, tier, exc)

        processed_rows.append(_build_row(row, local_result, railway_result, local_source, railway_source, logger))

    if not processed_rows:
        raise SystemExit("No calibration rows were processed. Check the input CSV and video directory.")

    detailed_df = _prepare_dataframe(processed_rows)
    detailed_path, tier_summary_path, metrics_path = _write_outputs(detailed_df, output_dir)

    valid_rows = detailed_df[~(detailed_df["local_overall"].isna() & detailed_df["railway_overall"].isna())]
    if valid_rows.empty:
        raise SystemExit("All processed videos failed both local and Railway analysis.")

    logger.info("Processed rows: %s", len(detailed_df))
    logger.info("Local successes: %s", local_success)
    logger.info("Railway successes: %s", railway_success)
    logger.info("Skipped not-filmed-correctly rows: %s", skipped_not_filmed)
    logger.info("Skipped missing-video rows: %s", skipped_missing_video)
    logger.info("Detailed CSV: %s", detailed_path)
    logger.info("Tier summary CSV: %s", tier_summary_path)
    logger.info("Metrics text: %s", metrics_path)

    print()
    print("\n".join(_summary_lines(valid_rows, _group_tier_summary(valid_rows))))
    print()
    print(f"Detailed CSV: {detailed_path}")
    print(f"Tier summary CSV: {tier_summary_path}")
    print(f"Metrics text: {metrics_path}")
    print(f"Log file: {output_dir / 'batch_calibration.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
