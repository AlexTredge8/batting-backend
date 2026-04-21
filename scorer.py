"""
BattingIQ Phase 2 — Scorer
============================
Converts raw fault lists into pillar scores, traffic lights,
BattingIQ score, score band, and priority fix selection.
"""

from dataclasses import replace
from typing import Optional

from anchor_accuracy import should_suppress_rule
from models import Fault, PillarScore, BattingIQResult, TrafficLight, PhaseResult
from config import (
    CONTACT_CONFIDENCE_LOW_WEIGHT,
    PILLAR_MAX,
    TRAFFIC_GREEN_MIN,
    TRAFFIC_AMBER_MIN,
    SCORE_BANDS,
    PILLAR_TIEBREAK_ORDER,
)

_CONTACT_DERIVED_RULE_IDS = {
    "S1", "S2", "S4",
    "T1", "T3", "T4",
    "A1", "A2", "A3", "A4", "A5",
    "F6",
}


def _traffic_light(score: int) -> TrafficLight:
    if score >= TRAFFIC_GREEN_MIN:
        return TrafficLight.GREEN
    if score >= TRAFFIC_AMBER_MIN:
        return TrafficLight.AMBER
    return TrafficLight.RED


def _score_band(total: int) -> str:
    for lo, hi, label in SCORE_BANDS:
        if lo <= total <= hi:
            return label
    return "Fundamentals"


def _score_pillar(faults: list[Fault]) -> int:
    total_deduction = sum(f.deduction for f in faults)
    return max(0, PILLAR_MAX - total_deduction)


def _apply_contact_confidence_weight(faults: list[Fault], phases: PhaseResult) -> list[Fault]:
    """Softly reduce contact-derived deductions when contact confidence is low."""
    if phases.resolved_contact_source == "manual":
        return faults
    if phases.contact_confidence != "low":
        return faults

    weighted: list[Fault] = []
    for fault in faults:
        if fault.rule_id in _CONTACT_DERIVED_RULE_IDS:
            weighted_deduction = max(1, int(round(fault.deduction * CONTACT_CONFIDENCE_LOW_WEIGHT)))
            weighted.append(replace(fault, deduction=weighted_deduction))
        else:
            weighted.append(fault)
    return weighted


def _apply_anchor_confidence_suppression(
    faults: list[Fault],
    anchor_confidence: dict[str, str] | None,
    suppressed_rules: list[dict],
) -> list[Fault]:
    if not faults or not anchor_confidence:
        return faults

    filtered: list[Fault] = []
    seen_rule_ids = {item.get("rule_id") for item in suppressed_rules}

    for fault in faults:
        should_suppress, anchor_keys = should_suppress_rule(fault.rule_id, anchor_confidence)
        if not should_suppress:
            filtered.append(fault)
            continue

        if fault.rule_id not in seen_rule_ids:
            suppressed_rules.append({
                "rule_id": fault.rule_id,
                "anchor_keys": anchor_keys,
                "reason": "suppressed_due_to_low_anchor_confidence",
            })
            seen_rule_ids.add(fault.rule_id)

    return filtered


def _select_priority_fix(pillars: dict[str, PillarScore]) -> Optional[Fault]:
    """
    Priority fix = highest deduction fault in the lowest-scoring pillar.
    Tiebreak order: stability > tracking > access > flow.
    """
    # Find minimum score across pillars
    min_score = min(p.score for p in pillars.values())

    # Get all pillars at the minimum score, ordered by tiebreak priority
    candidates = [
        name for name in PILLAR_TIEBREAK_ORDER
        if name in pillars and pillars[name].score == min_score
    ]
    if not candidates:
        return None

    # In the lowest-scoring pillar, pick the fault with the highest deduction
    worst_pillar = candidates[0]
    faults = pillars[worst_pillar].faults
    if not faults:
        return None

    return max(faults, key=lambda f: f.deduction)


def _development_notes(pillars: dict[str, PillarScore], phases: PhaseResult, baseline: dict) -> list[str]:
    """Generate development notes for players with good scores."""
    notes = []

    flow = pillars.get("flow")
    if flow and flow.score >= 20:
        ref_bl_frames = baseline.get("timing", {}).get("backlift_to_contact_frames", 15)
        actual_frames = phases.backlift_to_contact_frames
        if actual_frames < ref_bl_frames * 0.85:
            notes.append(
                "Your technique is flowing well. As you develop, a fuller backlift "
                "could generate more power while maintaining your timing."
            )

    access = pillars.get("access")
    if access and access.score >= 22:
        notes.append(
            "Your access looks clean. Focus on maintaining that elbow space "
            "through the ball as you face faster bowling."
        )

    stability = pillars.get("stability")
    if stability and stability.score >= 22:
        notes.append(
            "Good base stability. Work on making that planted finish a habit "
            "against full-pitched deliveries."
        )

    return notes


def build_scores(
    fault_map: dict[str, list[Fault]],
    phases: PhaseResult,
    baseline: dict,
    video_meta: dict,
    handedness: str = "right",
    handedness_source: str = "default",
) -> BattingIQResult:
    """
    Build the complete BattingIQResult from raw faults.
    """
    pillars: dict[str, PillarScore] = {}
    video_meta = dict(video_meta or {})
    anchor_confidence = (
        video_meta.get("anchor_confidence")
        if isinstance(video_meta.get("anchor_confidence"), dict)
        else None
    )
    suppressed_rules: list[dict] = []

    pillar_positives = {
        "access": [
            "Good elbow bend at contact — you are creating space through the ball."
        ],
        "tracking": [
            "Head position looks composed — eyes tracking well."
        ],
        "stability": [
            "Good weight transfer — your body is moving through the shot."
        ],
        "flow": [
            "Smooth swing sequence — the bat is working as one connected movement."
        ],
    }

    # Extract rule evaluation health (if present)
    evaluation = fault_map.pop("_evaluation", None)

    for name in ["access", "tracking", "stability", "flow"]:
        pillar_faults = _apply_anchor_confidence_suppression(
            fault_map.get(name, []),
            anchor_confidence,
            suppressed_rules,
        )
        faults = _apply_contact_confidence_weight(pillar_faults, phases)
        score  = _score_pillar(faults)
        light  = _traffic_light(score)
        # Only include positives if the pillar is green
        positives = pillar_positives[name] if light == TrafficLight.GREEN else []
        pillars[name] = PillarScore(
            name=name,
            score=score,
            max_score=PILLAR_MAX,
            status=light,
            faults=faults,
            positives=positives,
        )

    total = sum(p.score for p in pillars.values())
    band  = _score_band(total)
    priority = _select_priority_fix(pillars)
    dev_notes = _development_notes(pillars, phases, baseline)

    # Include rule evaluation health in metadata
    if evaluation:
        evaluation = dict(evaluation)
        evaluation["rules_suppressed"] = len(suppressed_rules)
        evaluation["suppressed_rule_ids"] = [item["rule_id"] for item in suppressed_rules]
        video_meta["rule_evaluation"] = evaluation
    if suppressed_rules:
        video_meta["suppressed_rules"] = suppressed_rules
        video_meta["suppressed_rule_notice"] = (
            "Some rule deductions were suppressed because their anchor frames were low confidence or unresolved."
        )
    video_meta["contact_confidence"] = phases.contact_confidence
    video_meta["estimated_contact_confidence"] = phases.estimated_contact_confidence
    video_meta["contact_candidates"] = phases.contact_candidates
    video_meta["contact_window"] = phases.contact_window
    video_meta["estimated_contact_frame"] = phases.estimated_contact_frame
    video_meta["estimated_contact_original_frame"] = phases.estimated_contact_original_frame
    video_meta["resolved_contact_frame"] = phases.resolved_contact_frame or phases.contact
    video_meta["resolved_contact_original_frame"] = phases.resolved_contact_original_frame
    video_meta["resolved_contact_source"] = phases.resolved_contact_source
    video_meta["resolved_contact_status"] = phases.resolved_contact_status
    video_meta.setdefault("detector_version", video_meta.get("contact_detector_version"))
    video_meta.setdefault("contact_detector_version", video_meta.get("detector_version"))
    if phases.contact_confidence == "low":
        video_meta["contact_notice"] = (
            "Contact confidence is low for this video, so contact-derived deductions "
            "have been softened."
        )
        video_meta["contact_confidence_weight"] = CONTACT_CONFIDENCE_LOW_WEIGHT
    if phases.resolved_contact_source == "manual":
        video_meta["contact_notice"] = (
            "Contact frame was manually validated and pinned for storyboard and scoring."
        )

    return BattingIQResult(
        battingiq_score=total,
        score_band=band,
        pillars=pillars,
        priority_fix=priority,
        development_notes=dev_notes,
        phases=phases,
        metadata=video_meta,
        handedness=handedness,
        handedness_source=handedness_source,
    )
