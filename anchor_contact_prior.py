"""
Contact-anchored key frames (switch: ANCHOR_MODE=contact_prior; default off).

Phase 5 finding (15 Nov 2026, baselines/phase5_anchor_candidates_2026-11-15): with automatic
contact, "contact + typical gap" is closer to the validated key frames than the current
detectors for hands-start-up, front-foot-down, hands-peak and follow-through (within 3 frames
71/76/76/47% against 59/59/53/18%, leave one clip out). For set-up, the stillest 300 ms of head
and hips within +-400 ms of the prior is best (43% within 3 frames against 29%).

Gaps are medians over the 17 calibration clips' validated key frames, in ms from contact.
Contact itself is unchanged (audio onset where available).
"""

from __future__ import annotations

GAPS_MS = {
    "setup_frame": -964,
    "hands_start_up_frame": -467,
    "front_foot_down_frame": -167,
    "hands_peak_frame": -231,
    "follow_through_frame": 560,
}
SETUP_SEARCH_MS = 400
SETUP_STILL_MS = 300


def contact_prior_anchors(metrics, contact_idx: int, fps: float) -> dict[str, int]:
    """Original-frame numbers for the five non-contact key frames, from the contact metric index."""
    if not metrics:
        return {}
    n = len(metrics)
    k_orig = int(metrics[max(0, min(n - 1, contact_idx))].frame_idx)
    last = int(metrics[-1].frame_idx)
    out = {}
    for key, gap in GAPS_MS.items():
        out[key] = max(0, min(last, k_orig + int(round(gap * fps / 1000.0))))
    setup = _stillest_setup(metrics, out["setup_frame"], fps)
    if setup is not None:
        out["setup_frame"] = setup
    # keep the order the pipeline expects
    out["hands_start_up_frame"] = max(out["hands_start_up_frame"], out["setup_frame"])
    return out


def _stillest_setup(metrics, prior_orig: int, fps: float) -> int | None:
    """Middle of the stillest SETUP_STILL_MS (head + hips) within +-SETUP_SEARCH_MS of the prior."""
    by_orig = {int(m.frame_idx): i for i, m in enumerate(metrics)}
    lo, hi = prior_orig - int(SETUP_SEARCH_MS * fps / 1000), prior_orig + int(SETUP_SEARCH_MS * fps / 1000)
    idx = [by_orig[f] for f in range(lo, hi + 1) if f in by_orig]
    if len(idx) < 4:
        return None
    move = []
    for a, b in zip(idx, idx[1:]):
        ma, mb = metrics[a], metrics[b]
        if not (ma.detected and mb.detected):
            move.append(None)
            continue
        hip_a, hip_b = ma.head_x - ma.head_offset, mb.head_x - mb.head_offset
        move.append(abs(mb.head_x - ma.head_x) + abs(mb.head_y - ma.head_y) + abs(hip_b - hip_a))
    half = max(1, int(SETUP_STILL_MS * fps / 1000 / 2))
    best, best_i = None, None
    for c in range(half, len(move) - half):
        seg = [v for v in move[c - half: c + half] if v is not None]
        if len(seg) >= half:
            s = sum(seg) / len(seg)
            if best is None or s < best:
                best, best_i = s, c
    return int(metrics[idx[best_i]].frame_idx) if best_i is not None else None
