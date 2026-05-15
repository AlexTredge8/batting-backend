#!/usr/bin/env python3
"""Generate rule firing-health diagnostics for D3."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GROUND_TRUTH = SCRIPT_DIR / "ground_truth_scores.csv"
FALLBACK_GROUND_TRUTH = SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"
DEFAULT_AUTO_DIR = SCRIPT_DIR / "calibration_output" / "diagnostics_d1_drift" / "auto"
DEFAULT_VALIDATED_DIR = SCRIPT_DIR / "calibration_output" / "diagnostics_d1_drift" / "validated"
DEFAULT_RULES = SCRIPT_DIR / "coaching_rules.py"
TIER_ORDER = ["Beginner", "Average", "Good Club", "Elite"]
STATUS_ORDER = ["inverted", "flat", "sparse", "healthy"]


def normalize_name(value: str) -> str:
    stem = Path(value).stem
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def parse_rule_registry(path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()

    rule_details: dict[str, dict[str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("rule_"):
            rule_id = node.name.replace("rule_", "", 1)
            doc = ast.get_docstring(node) or rule_id
            first_line = doc.strip().splitlines()[0]
            rule_name = first_line.split(":", 1)[1].strip() if ":" in first_line else first_line.strip()
            snippet = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            rule_details[rule_id] = {
                "rule_name": rule_name,
                "source": snippet,
            }

    registry: list[dict[str, str]] = []
    measurement_keys: dict[str, str] = {}

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_ALL_RULES" for target in node.targets)
            and isinstance(node.value, ast.List)
        ):
            for item in node.value.elts:
                if not isinstance(item, ast.Tuple) or len(item.elts) != 2:
                    continue
                pillar_node, fn_node = item.elts
                if not (
                    isinstance(pillar_node, ast.Constant)
                    and isinstance(pillar_node.value, str)
                    and isinstance(fn_node, ast.Name)
                    and fn_node.id.startswith("rule_")
                ):
                    continue
                rule_id = fn_node.id.replace("rule_", "", 1)
                detail = rule_details[rule_id]
                registry.append(
                    {
                        "rule_id": rule_id,
                        "pillar": pillar_node.value,
                        "rule_name": detail["rule_name"],
                        "status": "suspended" if "SUSPENDED" in detail["source"] else "active",
                    }
                )
            break

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "collect_rule_measurements":
            for child in ast.walk(node):
                if isinstance(child, ast.Dict):
                    for key_node in child.keys:
                        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                            key = key_node.value
                            if "_" in key:
                                rule_id = key.split("_", 1)[0]
                                measurement_keys.setdefault(rule_id, key)
            break

    return registry, measurement_keys


def load_tiers(ground_truth_path: Path, fallback_detailed_csv: Path | None = None) -> tuple[dict[str, str], str, dict[str, int]]:
    if ground_truth_path.exists():
        source_path = ground_truth_path
    elif FALLBACK_GROUND_TRUTH.exists():
        source_path = FALLBACK_GROUND_TRUTH
    else:
        raise FileNotFoundError("Neither ground_truth_scores.csv nor coach_ground_truth_from_screenshot.csv exists.")

    mapping: dict[str, str] = {}
    tier_counts = {tier: 0 for tier in TIER_ORDER}
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        filmed_correctly = row.get("filmed_correctly")
        if filmed_correctly and filmed_correctly.upper() != "Y":
            continue
        filename = row.get("filename")
        tier = row.get("tier")
        if not filename or tier not in tier_counts:
            continue
        mapping[normalize_name(filename)] = tier
        tier_counts[tier] += 1

    if mapping:
        return mapping, str(source_path), tier_counts

    if fallback_detailed_csv and fallback_detailed_csv.exists():
        with fallback_detailed_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            filename = row.get("filename")
            tier = row.get("tier")
            if not filename or tier not in tier_counts:
                continue
            mapping[normalize_name(filename)] = tier
            tier_counts[tier] += 1
        if mapping:
            return mapping, str(fallback_detailed_csv), tier_counts

    raise RuntimeError("Unable to derive tier labels from available inputs.")


def discover_runs(mode_dir: Path, tier_map: dict[str, str], measurement_keys: dict[str, str]) -> list[dict]:
    runs: list[dict] = []
    json_paths = sorted(mode_dir.glob("local_runs/*/*_battingiq.json"))
    if not json_paths:
        raise FileNotFoundError(f"No *_battingiq.json files found under {mode_dir}")

    unmatched: list[str] = []
    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        candidates = [
            metadata.get("video_name", ""),
            json_path.parent.name,
            json_path.name.replace("_battingiq.json", ""),
        ]

        tier = None
        matched_name = None
        for candidate in candidates:
            normalized = normalize_name(candidate)
            if normalized in tier_map:
                matched_name = normalized
                tier = tier_map[normalized]
                break

        if not tier:
            unmatched.append(str(json_path))
            continue

        deductions = defaultdict(float)
        for pillar_data in payload.get("pillars", {}).values():
            for fault in pillar_data.get("faults", []):
                rule_id = fault.get("rule_id")
                if rule_id:
                    deductions[rule_id] += float(fault.get("deduction", 0) or 0)

        raw_measurements = metadata.get("rule_measurements", {}) or {}
        measurements: dict[str, float | None] = {}
        for rule_id, key in measurement_keys.items():
            value = raw_measurements.get(key)
            measurements[rule_id] = None if value is None else float(value)

        runs.append(
            {
                "path": str(json_path),
                "tier": tier,
                "match_key": matched_name,
                "deductions": dict(deductions),
                "measurements": measurements,
            }
        )

    if unmatched:
        raise RuntimeError("Could not match these run files to a tier label:\n" + "\n".join(unmatched))

    return runs


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_row(rule: dict[str, str], runs: list[dict], tier_counts: dict[str, int]) -> dict[str, object]:
    deductions_by_tier: dict[str, list[float]] = {tier: [] for tier in TIER_ORDER}
    for run in runs:
        deductions_by_tier[run["tier"]].append(float(run["deductions"].get(rule["rule_id"], 0.0)))

    stats: dict[str, float] = {}
    for tier in TIER_ORDER:
        values = deductions_by_tier[tier]
        total = tier_counts[tier]
        fired = sum(1 for value in values if value > 0)
        key_base = tier.lower().replace(" ", "_")
        stats[f"{key_base}_fire_rate"] = fired / total if total else 0.0
        stats[f"{key_base}_mean_deduction"] = mean(values)

    beginner_fire_rate = stats["beginner_fire_rate"]
    elite_fire_rate = stats["elite_fire_rate"]
    beginner_mean = stats["beginner_mean_deduction"]
    elite_mean = stats["elite_mean_deduction"]
    discrimination = beginner_mean - elite_mean

    if rule["status"] == "suspended":
        status_flag = "suspended"
    elif elite_fire_rate > beginner_fire_rate + 0.25 or discrimination < -1.0:
        status_flag = "inverted"
    elif beginner_fire_rate < 0.4 and elite_fire_rate < 0.4:
        status_flag = "sparse"
    elif abs(beginner_fire_rate - elite_fire_rate) < 0.25 and abs(discrimination) < 1.0:
        status_flag = "flat"
    else:
        status_flag = "healthy"

    return {
        "rule_id": rule["rule_id"],
        "rule_name": rule["rule_name"],
        "pillar": rule["pillar"],
        "registry_status": rule["status"],
        "beginner_fire_rate": beginner_fire_rate,
        "average_fire_rate": stats["average_fire_rate"],
        "good_club_fire_rate": stats["good_club_fire_rate"],
        "elite_fire_rate": elite_fire_rate,
        "beginner_mean_deduction": beginner_mean,
        "elite_mean_deduction": elite_mean,
        "discrimination_score": discrimination,
        "status_flag": status_flag,
    }


def compute_measurement_summary(rule_id: str, runs: list[dict]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for tier in ("Beginner", "Elite"):
        values = [
            float(run["measurements"][rule_id])
            for run in runs
            if run["tier"] == tier and run["measurements"].get(rule_id) is not None
        ]
        summary[tier] = mean(values) if values else None
    return summary


def format_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.3f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "rule_id",
        "rule_name",
        "beginner_fire_rate",
        "average_fire_rate",
        "good_club_fire_rate",
        "elite_fire_rate",
        "beginner_mean_deduction",
        "elite_mean_deduction",
        "discrimination_score",
        "status_flag",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {}
            for name in fieldnames:
                value = row[name]
                output[name] = f"{value:.3f}" if isinstance(value, float) else value
            writer.writerow(output)


def build_summary(
    rows: list[dict[str, object]],
    validated_rows: list[dict[str, object]],
    auto_runs: list[dict],
    validated_runs: list[dict],
    tier_source: str,
    tier_counts: dict[str, int],
) -> str:
    active_rows = [row for row in rows if row["status_flag"] != "suspended"]
    suspended_rows = [row for row in rows if row["status_flag"] == "suspended"]

    ids_by_status = {
        status: [row["rule_id"] for row in active_rows if row["status_flag"] == status]
        for status in STATUS_ORDER
    }
    suspended_ids = [row["rule_id"] for row in suspended_rows]

    validated_status_map = {row["rule_id"]: row["status_flag"] for row in validated_rows}
    auto_status_map = {row["rule_id"]: row["status_flag"] for row in active_rows}
    detection_noise = [
        rule_id
        for rule_id, auto_status in auto_status_map.items()
        if validated_status_map.get(rule_id) == "healthy" and auto_status in {"flat", "inverted"}
    ]

    rules_to_investigate = ids_by_status["inverted"] + ids_by_status["flat"] + ids_by_status["sparse"]

    pillar_breakdown: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pillar_breakdown[row["pillar"]][row["status_flag"]].append(row["rule_id"])

    suspended_signal_lines = []
    for row in suspended_rows:
        rule_id = row["rule_id"]
        auto_measure = compute_measurement_summary(rule_id, auto_runs)
        validated_measure = compute_measurement_summary(rule_id, validated_runs)
        suspended_signal_lines.append(
            f"- {rule_id}: auto beginner/elite={format_float(auto_measure['Beginner'])}/{format_float(auto_measure['Elite'])}; "
            f"validated beginner/elite={format_float(validated_measure['Beginner'])}/{format_float(validated_measure['Elite'])}"
        )

    lines = [
        "Rule Health Audit (D3)",
        "",
        f"Tier label source: {tier_source}",
        (
            "Code reality check: coaching_rules.py currently registers "
            f"{len(active_rows)} active rules and {len(suspended_rows)} suspended rules "
            "(13 active + 8 suspended), even though the task text says 14 active / 7 suspended."
        ),
        "Tier counts used: " + ", ".join(f"{tier}={tier_counts[tier]}" for tier in TIER_ORDER),
        "",
        "Status counts:",
        f"- healthy: {len(ids_by_status['healthy'])}",
        f"- flat: {len(ids_by_status['flat'])}",
        f"- inverted: {len(ids_by_status['inverted'])}",
        f"- sparse: {len(ids_by_status['sparse'])}",
        f"- suspended: {len(suspended_ids)}",
        "",
        "Rule IDs by category:",
        f"- healthy: {', '.join(ids_by_status['healthy']) or 'none'}",
        f"- flat: {', '.join(ids_by_status['flat']) or 'none'}",
        f"- inverted: {', '.join(ids_by_status['inverted']) or 'none'}",
        f"- sparse: {', '.join(ids_by_status['sparse']) or 'none'}",
        f"- suspended: {', '.join(suspended_ids) or 'none'}",
        "",
        "Rules to investigate first:",
        f"- {', '.join(rules_to_investigate) or 'none'}",
        "",
        "Validated-vs-auto comparison:",
        "- Healthy on validated but flat/inverted on auto: " + (
            ", ".join(
                f"{rule_id} (validated healthy -> auto {auto_status_map[rule_id]})"
                for rule_id in detection_noise
            )
            if detection_noise
            else "none"
        ),
        "",
        "Suspended rule measurement signal (means from metadata.rule_measurements):",
        *suspended_signal_lines,
        "",
        "Per-pillar breakdown:",
    ]

    for pillar in ("access", "tracking", "stability", "flow"):
        status_parts = []
        for status in ("healthy", "flat", "inverted", "sparse", "suspended"):
            ids = pillar_breakdown[pillar].get(status, [])
            status_parts.append(f"{status}={','.join(ids) if ids else 'none'}")
        lines.append(f"- {pillar}: " + " | ".join(status_parts))

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto-dir", type=Path, default=DEFAULT_AUTO_DIR)
    parser.add_argument("--validated-dir", type=Path, default=DEFAULT_VALIDATED_DIR)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--rules-file", type=Path, default=DEFAULT_RULES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry, measurement_keys = parse_rule_registry(args.rules_file)

    auto_detailed_csv = args.auto_dir / "battingiq_calibration_comparison_detailed.csv"
    tier_map, tier_source, tier_counts = load_tiers(args.ground_truth, auto_detailed_csv)

    auto_runs = discover_runs(args.auto_dir, tier_map, measurement_keys)
    validated_runs = discover_runs(args.validated_dir, tier_map, measurement_keys)

    rows = [compute_row(rule, auto_runs, tier_counts) for rule in registry]
    validated_rows = [compute_row(rule, validated_runs, tier_counts) for rule in registry]

    csv_path = args.auto_dir / "rule_health.csv"
    summary_path = args.auto_dir / "rule_health_summary.txt"

    write_csv(csv_path, rows)
    summary_text = build_summary(rows, validated_rows, auto_runs, validated_runs, tier_source, tier_counts)
    summary_path.write_text(summary_text, encoding="utf-8")

    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
