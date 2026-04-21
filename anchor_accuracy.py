"""
Utilities for six-anchor metadata, overrides, and rule dependency handling.
"""

from __future__ import annotations

from typing import Any

from models import FrameMetrics, PhaseResult


ANCHOR_KEYS = (
    "setup_frame",
    "hands_start_up_frame",
    "front_foot_down_frame",
    "hands_peak_frame",
    "contact_frame",
    "follow_through_frame",
)

ANCHOR_CONFIDENCE_LOW = {"low", "missing", "unresolved"}
ANCHOR_DETECTOR_VERSION = "anchor-heuristics-v1"

RULE_ANCHOR_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "A1": ("hands_peak_frame", "contact_frame"),
    "A2": ("contact_frame",),
    "A3": ("contact_frame", "follow_through_frame"),
    "A4": ("contact_frame",),
    "A5": ("contact_frame",),
    "A6": ("setup_frame", "hands_start_up_frame", "hands_peak_frame"),
    "T1": ("contact_frame",),
    "T2": ("setup_frame", "hands_start_up_frame"),
    "T3": ("contact_frame",),
    "T4": ("contact_frame",),
    "T5": ("setup_frame",),
    "S1": ("contact_frame",),
    "S2": ("contact_frame", "follow_through_frame"),
    "S3": ("hands_start_up_frame", "follow_through_frame"),
    "S4": ("contact_frame", "follow_through_frame"),
    "F1": ("hands_peak_frame", "front_foot_down_frame"),
    "F2": ("hands_start_up_frame", "contact_frame"),
    "F3": ("hands_start_up_frame", "contact_frame"),
    "F4": ("hands_peak_frame",),
    "F5": ("hands_peak_frame", "contact_frame"),
    "F6": ("contact_frame", "follow_through_frame"),
}


def _metric_index_to_orig_frame(metrics: list[FrameMetrics], metric_idx: int) -> int:
    if not metrics:
        return 0
    idx = max(0, min(metric_idx, len(metrics) - 1))
    return int(metrics[idx].frame_idx)


def _metric_index_confidence(metrics: list[FrameMetrics], metric_idx: int) -> str:
    if not metrics or metric_idx < 0 or metric_idx >= len(metrics):
        return "missing"

    metric = metrics[metric_idx]
    if not metric.detected or metric.low_confidence:
        return "low"

    neighbour_range = range(max(0, metric_idx - 1), min(len(metrics), metric_idx + 2))
    low_neighbours = sum(
        1
        for idx in neighbour_range
        if not metrics[idx].detected or metrics[idx].low_confidence
    )
    return "medium" if low_neighbours else "high"


def build_anchor_frames(phases: PhaseResult, metrics: list[FrameMetrics]) -> dict[str, dict[str, int]]:
    anchors = {
        "setup_frame": phases.setup_end,
        "hands_start_up_frame": phases.backlift_start,
        "front_foot_down_frame": phases.front_foot_down,
        "hands_peak_frame": phases.hands_peak,
        "contact_frame": phases.resolved_contact_frame or phases.contact,
        "follow_through_frame": phases.follow_through_start,
    }
    return {
        key: {
            "metric_index": int(metric_idx),
            "original_frame": _metric_index_to_orig_frame(metrics, int(metric_idx)),
        }
        for key, metric_idx in anchors.items()
    }


def build_anchor_confidence(phases: PhaseResult, metrics: list[FrameMetrics]) -> dict[str, str]:
    confidences = {
        "setup_frame": _metric_index_confidence(metrics, phases.setup_end),
        "hands_start_up_frame": _metric_index_confidence(metrics, phases.backlift_start),
        "front_foot_down_frame": _metric_index_confidence(metrics, phases.front_foot_down),
        "hands_peak_frame": _metric_index_confidence(metrics, phases.hands_peak),
        "contact_frame": phases.contact_confidence or _metric_index_confidence(metrics, phases.contact),
        "follow_through_frame": _metric_index_confidence(metrics, phases.follow_through_start),
    }
    if phases.resolved_contact_source == "manual":
        confidences["contact_frame"] = "validated"
    return confidences


def build_anchor_quality_summary(anchor_confidence: dict[str, str]) -> dict[str, Any]:
    low_anchors = [key for key, value in anchor_confidence.items() if value in ANCHOR_CONFIDENCE_LOW]
    medium_anchors = [key for key, value in anchor_confidence.items() if value == "medium"]
    return {
        "low_confidence_count": len(low_anchors),
        "medium_confidence_count": len(medium_anchors),
        "low_confidence_anchors": low_anchors,
        "medium_confidence_anchors": medium_anchors,
        "all_high_confidence": len(low_anchors) == 0 and len(medium_anchors) == 0,
    }


def should_suppress_rule(rule_id: str, anchor_confidence: dict[str, str] | None) -> tuple[bool, list[str]]:
    dependencies = list(RULE_ANCHOR_DEPENDENCIES.get(rule_id, ()))
    if not dependencies or not anchor_confidence:
        return False, []

    failing = [anchor for anchor in dependencies if anchor_confidence.get(anchor) in ANCHOR_CONFIDENCE_LOW]
    return bool(failing), failing
