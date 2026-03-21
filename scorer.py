"""
BattingIQ Phase 2 — Scorer
============================
Converts raw fault lists into pillar scores, traffic lights,
BattingIQ score, score band, and priority fix selection.
"""

from models import Fault, PillarScore, BattingIQResult, TrafficLight, PhaseResult
from config import (
    PILLAR_MAX,
    TRAFFIC_GREEN_MIN,
    TRAFFIC_AMBER_MIN,
    SCORE_BANDS,
    PILLAR_TIEBREAK_ORDER,
)


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


def _select_priority_fix(pillars: dict[str, PillarScore]) -> Fault | None:
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

    for name in ["access", "tracking", "stability", "flow"]:
        faults = fault_map.get(name, [])
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
