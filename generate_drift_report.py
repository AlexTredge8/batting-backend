#!/usr/bin/env python3
"""
Generate anchor and score drift diagnostics from a dual-mode calibration run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = SCRIPT_DIR / "calibration_output" / "diagnostics_d1_drift"
ANCHOR_KEYS = (
    "setup_frame",
    "hands_start_up_frame",
    "front_foot_down_frame",
    "hands_peak_frame",
    "contact_frame",
    "follow_through_frame",
)
PILLARS = ("access", "tracking", "stability", "flow")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    text = _clean_text(value)
    if not text:
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def _mean_abs(values: list[int | None]) -> float | None:
    usable = [abs(int(value)) for value in values if value is not None]
    if not usable:
        return None
    return float(sum(usable) / len(usable))


def _json_report_by_filename(mode_dir: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    local_runs_dir = mode_dir / "local_runs"
    if not local_runs_dir.exists():
        return reports

    for report_path in sorted(local_runs_dir.glob("*/*_battingiq.json")):
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        metadata = report.get("metadata") or {}
        filename = _clean_text(metadata.get("video_name"))
        if not filename:
            filename = report_path.name.replace("_battingiq.json", "")
        reports[filename] = report
    return reports


def _detailed_rows_by_filename(mode_dir: Path) -> dict[str, dict[str, Any]]:
    detailed_path = mode_dir / "battingiq_calibration_comparison_detailed.csv"
    if not detailed_path.exists():
        return {}
    df = pd.read_csv(detailed_path, dtype=str, keep_default_na=False)
    return {
        _clean_text(row.get("filename")): {key: _clean_text(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
        if _clean_text(row.get("filename"))
    }


def _extract_anchor_frame(report: dict[str, Any], anchor_key: str) -> int | None:
    metadata = report.get("metadata") or {}
    anchor_frames = metadata.get("anchor_frames") or {}
    anchor_info = anchor_frames.get(anchor_key) or {}
    return _coerce_int(anchor_info.get("original_frame"))


def _extract_pillar_score(report: dict[str, Any], pillar: str) -> int | None:
    pillars = report.get("pillars") or {}
    pillar_info = pillars.get(pillar) or {}
    return _coerce_int(pillar_info.get("score"))


def _build_row(
    filename: str,
    tier: str,
    auto_report: dict[str, Any] | None,
    validated_report: dict[str, Any] | None,
    auto_detail: dict[str, Any] | None,
    validated_detail: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    row: dict[str, Any] = {
        "filename": filename,
        "tier": tier,
    }
    notes: list[str] = []

    for mode_name, detail_row in (("auto", auto_detail), ("validated", validated_detail)):
        if detail_row is None:
            notes.append(f"{mode_name}_detailed_row_missing")
            continue
        local_source = _clean_text(detail_row.get("local_source"))
        analysis_status = _clean_text(detail_row.get("analysis_status"))
        if local_source and local_source != "fresh":
            notes.append(f"{mode_name}_local_source={local_source}")
        if analysis_status and analysis_status != "both_success":
            notes.append(f"{mode_name}_analysis_status={analysis_status}")

    if auto_report is None and validated_report is not None:
        notes.append("auto_local_report_missing")
    if validated_report is None and auto_report is not None:
        notes.append("validated_local_report_missing")
    if auto_report is None and validated_report is None:
        notes.append("both_local_reports_missing")

    frame_deltas: list[int | None] = []
    for anchor_key in ANCHOR_KEYS:
        auto_frame = _extract_anchor_frame(auto_report, anchor_key) if auto_report else None
        validated_frame = _extract_anchor_frame(validated_report, anchor_key) if validated_report else None
        delta = None
        if auto_frame is not None and validated_frame is not None:
            delta = auto_frame - validated_frame
        row[anchor_key.replace("_frame", "_frame_delta")] = delta
        frame_deltas.append(delta)

    row["mean_abs_frame_delta"] = _mean_abs(frame_deltas)

    auto_overall = _coerce_int((auto_report or {}).get("battingiq_score"))
    validated_overall = _coerce_int((validated_report or {}).get("battingiq_score"))
    row["auto_overall"] = auto_overall
    row["validated_overall"] = validated_overall
    row["overall_score_delta"] = (
        None if auto_overall is None or validated_overall is None else auto_overall - validated_overall
    )

    for pillar in PILLARS:
        auto_score = _extract_pillar_score(auto_report, pillar) if auto_report else None
        validated_score = _extract_pillar_score(validated_report, pillar) if validated_report else None
        row[f"auto_{pillar}"] = auto_score
        row[f"validated_{pillar}"] = validated_score
        row[f"{pillar}_score_delta"] = (
            None if auto_score is None or validated_score is None else auto_score - validated_score
        )

    return row, notes


def _build_drift_dataframe(auto_dir: Path, validated_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    auto_reports = _json_report_by_filename(auto_dir)
    validated_reports = _json_report_by_filename(validated_dir)
    auto_details = _detailed_rows_by_filename(auto_dir)
    validated_details = _detailed_rows_by_filename(validated_dir)

    filenames = sorted(
        set(auto_reports)
        | set(validated_reports)
        | set(auto_details)
        | set(validated_details)
    )
    rows: list[dict[str, Any]] = []
    issues: list[str] = []

    for filename in filenames:
        auto_detail = auto_details.get(filename)
        validated_detail = validated_details.get(filename)
        tier = _clean_text((auto_detail or {}).get("tier")) or _clean_text((validated_detail or {}).get("tier"))
        row, notes = _build_row(
            filename,
            tier,
            auto_reports.get(filename),
            validated_reports.get(filename),
            auto_detail,
            validated_detail,
        )
        rows.append(row)
        if notes:
            issues.append(f"{filename}: {', '.join(notes)}")

    return pd.DataFrame(rows), issues


def _build_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tier in sorted(df["tier"].dropna().astype(str).unique()):
        tier_df = df[df["tier"] == tier].copy()
        summary_row: dict[str, Any] = {
            "tier": tier,
            "n_videos": int(len(tier_df)),
        }
        for anchor_key in ANCHOR_KEYS:
            column = anchor_key.replace("_frame", "_frame_delta")
            summary_row[f"mean_abs_frame_delta_{anchor_key.replace('_frame', '')}"] = (
                tier_df[column].abs().mean() if column in tier_df.columns else None
            )
        summary_row["mean_signed_overall_score_delta"] = tier_df["overall_score_delta"].mean()
        summary_row["mean_abs_overall_score_delta"] = tier_df["overall_score_delta"].abs().mean()
        for pillar in PILLARS:
            summary_row[f"mean_signed_{pillar}_delta"] = tier_df[f"{pillar}_score_delta"].mean()
        rows.append(summary_row)
    return pd.DataFrame(rows)


def _format_float(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"


def _worst_anchor_name(anchor_means: dict[str, float | None]) -> str | None:
    usable = [(anchor_key, value) for anchor_key, value in anchor_means.items() if value is not None and not pd.isna(value)]
    if not usable:
        return None
    return max(usable, key=lambda item: float(item[1]))[0]


def _score_sensitive_note(score_sensitive_df: pd.DataFrame) -> str:
    if score_sensitive_df.empty:
        return "Score-sensitive videos do not appear to cluster in a particular tier because none exceeded the threshold."

    counts = score_sensitive_df["tier"].value_counts()
    top_tier = str(counts.index[0])
    top_count = int(counts.iloc[0])
    if len(counts) == 1:
        return f"Score-sensitive videos are concentrated in {top_tier} ({top_count} of {top_count})."

    second_count = int(counts.iloc[1])
    if top_count > second_count:
        return f"Score-sensitive videos skew toward {top_tier} ({top_count} videos), suggesting possible tier-biased detection error there."

    tiers = ", ".join(f"{tier} ({int(count)})" for tier, count in counts.items())
    return f"Score-sensitive videos are spread across tiers rather than clearly clustered: {tiers}."


def _build_summary_text(df: pd.DataFrame, summary_df: pd.DataFrame, issues: list[str]) -> str:
    lines: list[str] = []
    lines.append("Sign convention: positive frame delta means auto detected LATER than validated; negative means auto detected EARLIER.")
    lines.append("Score deltas are reported as auto minus validated.")
    lines.append("")

    overall_anchor_means = {
        anchor_key: df[anchor_key.replace("_frame", "_frame_delta")].abs().mean()
        for anchor_key in ANCHOR_KEYS
    }
    worst_anchor = _worst_anchor_name(overall_anchor_means)
    if worst_anchor is None:
        lines.append("Single worst anchor overall: n/a")
    else:
        lines.append(
            f"Single worst anchor overall: {worst_anchor} ({_format_float(overall_anchor_means[worst_anchor])} mean absolute frames)"
        )
    lines.append("")

    lines.append("Worst anchor by tier:")
    for _, row in summary_df.iterrows():
        tier = _clean_text(row.get("tier"))
        anchor_means = {
            anchor_key: row.get(f"mean_abs_frame_delta_{anchor_key.replace('_frame', '')}")
            for anchor_key in ANCHOR_KEYS
        }
        worst_for_tier = _worst_anchor_name(anchor_means)
        if worst_for_tier is None:
            lines.append(f"- {tier}: n/a")
        else:
            lines.append(
                f"- {tier}: {worst_for_tier} ({_format_float(anchor_means[worst_for_tier])} mean absolute frames)"
            )
    lines.append("")

    if summary_df.empty or summary_df["mean_abs_overall_score_delta"].dropna().empty:
        lines.append("Tier with largest mean absolute overall score delta: n/a")
    else:
        max_index = summary_df["mean_abs_overall_score_delta"].astype(float).idxmax()
        max_row = summary_df.loc[max_index]
        lines.append(
            "Tier with largest mean absolute overall score delta: "
            f"{_clean_text(max_row['tier'])} ({_format_float(max_row['mean_abs_overall_score_delta'])})"
        )
    lines.append("")

    lines.append("Pillar with largest mean absolute score delta by tier:")
    for _, row in summary_df.iterrows():
        tier = _clean_text(row.get("tier"))
        pillar_abs = {
            pillar: abs(float(row[f"mean_signed_{pillar}_delta"]))
            for pillar in PILLARS
            if row.get(f"mean_signed_{pillar}_delta") is not None and not pd.isna(row.get(f"mean_signed_{pillar}_delta"))
        }
        if not pillar_abs:
            lines.append(f"- {tier}: n/a")
            continue
        worst_pillar = max(pillar_abs, key=pillar_abs.get)
        signed_value = row[f"mean_signed_{worst_pillar}_delta"]
        lines.append(f"- {tier}: {worst_pillar} ({_format_float(signed_value)} signed mean delta)")
    lines.append("")

    extreme_rows: list[str] = []
    for _, row in df.iterrows():
        exceeded = [
            anchor_key
            for anchor_key in ANCHOR_KEYS
            if _coerce_int(row.get(anchor_key.replace("_frame", "_frame_delta"))) is not None
            and abs(int(row[anchor_key.replace("_frame", "_frame_delta")])) > 8
        ]
        if exceeded:
            anchors = ", ".join(exceeded)
            extreme_rows.append(f"- {row['filename']} [{row['tier']}]: {anchors}")
    lines.append('Videos flagged "extreme" (any anchor delta > 8 frames):')
    if extreme_rows:
        lines.extend(extreme_rows)
    else:
        lines.append("- none")
    lines.append("")

    score_sensitive_df = df[df["overall_score_delta"].abs() > 10].copy()
    lines.append('Videos flagged "score-sensitive-to-detection" (|auto_overall - validated_overall| > 10):')
    if score_sensitive_df.empty:
        lines.append("- none")
    else:
        for _, row in score_sensitive_df.sort_values(["tier", "filename"]).iterrows():
            lines.append(
                f"- {row['filename']} [{row['tier']}]: overall delta {int(row['overall_score_delta'])}"
            )
    lines.append("")
    lines.append(_score_sensitive_note(score_sensitive_df))
    lines.append("")

    lines.append("Mode mismatch or processing issues:")
    if issues:
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def _write_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    df.reindex(columns=columns).to_csv(path, index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-video and per-tier drift diagnostics from a dual-mode run.")
    parser.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT),
        help="Run root containing auto/ and validated/ output directories.",
    )
    parser.add_argument(
        "--auto-dir",
        default=None,
        help="Optional explicit auto output directory. Overrides --run-root/auto.",
    )
    parser.add_argument(
        "--validated-dir",
        default=None,
        help="Optional explicit validated output directory. Overrides --run-root/validated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser()
    if not run_root.is_absolute():
        run_root = (SCRIPT_DIR / run_root).resolve()

    auto_dir = Path(args.auto_dir).expanduser() if args.auto_dir else run_root / "auto"
    validated_dir = Path(args.validated_dir).expanduser() if args.validated_dir else run_root / "validated"
    if not auto_dir.is_absolute():
        auto_dir = (SCRIPT_DIR / auto_dir).resolve()
    if not validated_dir.is_absolute():
        validated_dir = (SCRIPT_DIR / validated_dir).resolve()
    if not auto_dir.exists() or not validated_dir.exists():
        raise SystemExit(f"Expected readable auto and validated directories, got auto={auto_dir} validated={validated_dir}")

    drift_df, issues = _build_drift_dataframe(auto_dir, validated_dir)
    if drift_df.empty:
        raise SystemExit(f"No drift rows could be built from {run_root}")

    numeric_columns = [
        column
        for column in drift_df.columns
        if column not in {"filename", "tier"}
    ]
    for column in numeric_columns:
        drift_df[column] = pd.to_numeric(drift_df[column], errors="coerce")

    summary_df = _build_summary_dataframe(drift_df)

    drift_report_path = run_root / "drift_report.csv"
    drift_summary_path = run_root / "drift_summary.csv"
    drift_text_path = run_root / "drift_summary.txt"
    run_root.mkdir(parents=True, exist_ok=True)

    drift_report_columns = [
        "filename",
        "tier",
        "setup_frame_delta",
        "hands_start_up_frame_delta",
        "front_foot_down_frame_delta",
        "hands_peak_frame_delta",
        "contact_frame_delta",
        "follow_through_frame_delta",
        "mean_abs_frame_delta",
        "auto_overall",
        "validated_overall",
        "overall_score_delta",
        "auto_access",
        "validated_access",
        "access_score_delta",
        "auto_tracking",
        "validated_tracking",
        "tracking_score_delta",
        "auto_stability",
        "validated_stability",
        "stability_score_delta",
        "auto_flow",
        "validated_flow",
        "flow_score_delta",
    ]
    drift_summary_columns = [
        "tier",
        "n_videos",
        "mean_abs_frame_delta_setup",
        "mean_abs_frame_delta_hands_start_up",
        "mean_abs_frame_delta_front_foot_down",
        "mean_abs_frame_delta_hands_peak",
        "mean_abs_frame_delta_contact",
        "mean_abs_frame_delta_follow_through",
        "mean_signed_overall_score_delta",
        "mean_abs_overall_score_delta",
        "mean_signed_access_delta",
        "mean_signed_tracking_delta",
        "mean_signed_stability_delta",
        "mean_signed_flow_delta",
    ]

    _write_csv(drift_df.sort_values(["tier", "filename"]).reset_index(drop=True), drift_report_path, drift_report_columns)
    _write_csv(summary_df.sort_values("tier").reset_index(drop=True), drift_summary_path, drift_summary_columns)
    drift_text_path.write_text(_build_summary_text(drift_df, summary_df, issues), encoding="utf-8")

    print(f"Drift report CSV: {drift_report_path}")
    print(f"Drift summary CSV: {drift_summary_path}")
    print(f"Drift summary text: {drift_text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
