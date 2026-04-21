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


def _phases_to_dict(pr: PhaseResult) -> dict:
    fps = pr.fps or 30.0

    def ms(f): return round(f / fps * 1000, 1)
    resolved_contact_ms = round((pr.resolved_contact_original_frame or pr.contact) / fps * 1000, 1)

    sync_diff = pr.hands_peak_vs_ffd_diff
    sync_label = (
        "in_sync" if abs(sync_diff) <= 2
        else ("hands_late" if sync_diff < 0 else "feet_early")
    )

    return {
        "setup":            {"start": 0,              "end": pr.setup_end,
                             "start_ms": 0,           "end_ms": ms(pr.setup_end)},
        "backlift_starts":  {"frame": pr.backlift_start, "ms": ms(pr.backlift_start)},
        "hands_peak":       {"frame": pr.hands_peak,     "ms": ms(pr.hands_peak)},
        "front_foot_down":  {"frame": pr.front_foot_down,"ms": ms(pr.front_foot_down)},
        "contact":          {
            "frame": pr.contact,
            "ms": resolved_contact_ms,
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
        "follow_through":   {"start": pr.follow_through_start,
                             "start_ms": ms(pr.follow_through_start)},
        "timing": {
            "hands_peak_vs_ffd_frames": sync_diff,
            "hands_peak_vs_ffd_ms":     pr.hands_peak_vs_ffd_ms,
            "sync_status":              sync_label,
            "backlift_to_contact_frames": pr.backlift_to_contact_frames,
            "backlift_to_contact_ms":     ms(pr.backlift_to_contact_frames),
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
        "phases": _phases_to_dict(result.phases),
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
