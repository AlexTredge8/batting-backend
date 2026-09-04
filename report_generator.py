"""
BattingIQ Phase 2 — Report Generator
=======================================
Assembles the final JSON report from a BattingIQResult.
"""

import json
from pathlib import Path
from models import BattingIQResult, Fault, PhaseResult


def _fault_to_dict(f: Fault) -> dict:
    return {
        "rule_id":  f.rule_id,
        "fault":    f.fault,
        "deduction": f.deduction,
        "detail":   f.detail,
        "feedback": f.feedback,
    }


def _phases_to_dict(pr: PhaseResult, anchor_frames: dict | None = None) -> dict:
    """
    Serialise phase anchors.

    ``frame`` values are metric-list indices (the space every detector and rule
    works in). ``original_frame`` values are real video frame numbers and every
    millisecond value is derived from those, so timings stay correct even when
    the extractor subsamples frames (frame_step > 1).
    """
    fps = pr.fps or 30.0
    anchor_frames = anchor_frames or {}

    def orig(key: str, metric_idx: int) -> int:
        info = anchor_frames.get(key) or {}
        original = info.get("original_frame")
        return int(original) if original is not None else int(metric_idx)

    def ms_of(original_frame: int) -> float:
        return round(original_frame / fps * 1000, 1)

    setup_of    = orig("setup_frame", pr.setup_end)
    backlift_of = orig("hands_start_up_frame", pr.backlift_start)
    hp_of       = orig("hands_peak_frame", pr.hands_peak)
    ffd_of      = orig("front_foot_down_frame", pr.front_foot_down)
    contact_of  = int(pr.resolved_contact_original_frame or orig("contact_frame", pr.contact))
    ft_of       = orig("follow_through_frame", pr.follow_through_start)

    sync_diff = pr.hands_peak_vs_ffd_diff
    sync_label = (
        "in_sync" if abs(sync_diff) <= 2
        else ("hands_late" if sync_diff < 0 else "feet_early")
    )

    return {
        "setup":            {"start": 0, "end": pr.setup_end, "original_frame": setup_of,
                             "start_ms": 0, "end_ms": ms_of(setup_of),
                             "confidence": pr.setup_confidence},
        "backlift_starts":  {"frame": pr.backlift_start, "original_frame": backlift_of,
                             "ms": ms_of(backlift_of)},
        "hands_peak":       {"frame": pr.hands_peak, "original_frame": hp_of, "ms": ms_of(hp_of),
                             "confidence": pr.hands_peak_confidence},
        "front_foot_down":  {"frame": pr.front_foot_down, "original_frame": ffd_of,
                             "ms": ms_of(ffd_of)},
        "contact":          {
            "frame": pr.contact,
            "original_frame": contact_of,
            "ms": ms_of(contact_of),
            "source": pr.resolved_contact_source,
            "status": pr.resolved_contact_status,
            "estimated_frame": pr.estimated_contact_frame,
            "estimated_original_frame": pr.estimated_contact_original_frame,
            "resolved_original_frame": pr.resolved_contact_original_frame,
            "confidence": pr.contact_confidence,
            "candidates": pr.contact_candidates,
            "window": pr.contact_window,
            "diagnostics": pr.contact_diagnostics,
        },
        "follow_through":   {"start": pr.follow_through_start, "original_frame": ft_of,
                             "start_ms": ms_of(ft_of),
                             "confidence": pr.follow_through_confidence},
        "timing": {
            "hands_peak_vs_ffd_frames": sync_diff,
            "hands_peak_vs_ffd_original_frames": hp_of - ffd_of,
            "hands_peak_vs_ffd_ms": round((hp_of - ffd_of) / fps * 1000, 1),
            "sync_status": sync_label,
            "backlift_to_contact_frames": pr.backlift_to_contact_frames,
            "backlift_to_contact_original_frames": contact_of - backlift_of,
            "backlift_to_contact_ms": round((contact_of - backlift_of) / fps * 1000, 1),
        },
    }


def build_json_report(result: BattingIQResult) -> dict:
    """Return the full report as a Python dict (JSON-serialisable)."""
    storyboard_generation = result.metadata.get("storyboard_generation", {}) if result.metadata else {}
    metadata = dict(result.metadata or {})
    report = {
        "battingiq_score": result.battingiq_score,
        "score_band": result.score_band,
        "handedness": result.handedness,
        "handedness_source": result.handedness_source,
        "pillars": {},
        "priority_fix": _fault_to_dict(result.priority_fix) if result.priority_fix else None,
        "development_notes": result.development_notes,
        "phases": _phases_to_dict(result.phases, metadata.get("anchor_frames")),
        "metadata": metadata,
        "storyboard_frames": storyboard_generation.get("frames", []),
    }
    if metadata.get("contact_detector_version"):
        report["phases"].setdefault("contact", {})
        report["phases"]["contact"]["detector_version"] = metadata["contact_detector_version"]

    for name, p in result.pillars.items():
        report["pillars"][name] = {
            "score": p.score,
            "max":   p.max_score,
            "status": p.status.value,
            "faults": [_fault_to_dict(f) for f in p.faults],
            "positives": p.positives,
        }

    return report


def save_json_report(result: BattingIQResult, output_path: str) -> None:
    report = build_json_report(result)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(report, fh, indent=2)


def print_report(result: BattingIQResult) -> None:
    """Print a human-readable summary to stdout."""
    r = result
    print()
    print("=" * 60)
    print(f"  BATTINGIQ SCORE: {r.battingiq_score}/100  [{r.score_band.upper()}]")
    print("=" * 60)

    for name in ["access", "tracking", "stability", "flow"]:
        p = r.pillars[name]
        bar = "#" * p.score + "." * (p.max_score - p.score)
        print(f"  {name.upper():12s} [{bar}] {p.score}/{p.max_score} {p.status.value.upper()}")
        for f in p.faults:
            print(f"    - [{f.rule_id}] {f.fault}  (-{f.deduction})")

    if r.priority_fix:
        pf = r.priority_fix
        print()
        print("  PRIORITY FIX:")
        print(f"    [{pf.rule_id}] {pf.feedback}")

    pr = r.phases
    fps = pr.fps or 30.0
    print()
    print("  PHASE DETECTION:")
    print(f"    Setup      : 0–{pr.setup_end} ({pr.setup_end/fps:.2f}s)")
    print(f"    Backlift   : frame {pr.backlift_start} ({pr.backlift_start/fps:.2f}s)")
    print(f"    Hands Peak : frame {pr.hands_peak} ({pr.hands_peak/fps:.2f}s)")
    print(f"    Front Foot : frame {pr.front_foot_down} ({pr.front_foot_down/fps:.2f}s)")
    print(f"    Contact    : frame {pr.contact} ({pr.contact/fps:.2f}s)")
    sync = pr.hands_peak_vs_ffd_diff
    print(f"    Peak vs FFD: {sync:+d} frames  {'IN SYNC' if abs(sync)<=2 else 'OUT OF SYNC'}")

    if r.development_notes:
        print()
        print("  DEVELOPMENT NOTES:")
        for note in r.development_notes:
            print(f"    • {note}")

    print("=" * 60)
