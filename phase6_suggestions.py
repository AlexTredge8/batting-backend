#!/usr/bin/env python3
"""
Phase 6: the AI's suggested main fault for each clip, for the "Check the AI's tips" file
(phase2_offline_pairs.py --validate-out ... --suggestions <this CSV>).

Four AI sources already in the repo can name a main fault. In priority order:
  1. phase6_measurements  the Phase 6 prototype diagnosis from the pose measurements
                          (baselines/phase6_prototype_2026-11-15/clip_predictions.csv, priority_fault)
  2. gemini_checkpoints   the video model rating each checkpoint, main fault = first failed in the chain
                          (baselines/phase5_video_judge_v2_2026-11-16/checkpoints_crop/checkpoint_faults.csv)
  3. gemini_absolute      the video model scoring the clip with reference clips, priority_fault
                          (baselines/phase5_video_judge_v2_2026-11-16/absolute_crop_refs2/video_judge_scores.csv)
  4. stills_model         the stills model as a fourth coach, model_priority_fault
                          (baselines/phase2_preliminary_2026-11-14/model_vs_alex.csv)

Fault IDs are mapped to rubric v0.3 (F14 is part of F07 there; "none" means no fault).

To limit anchoring and to learn which source agrees with the coach most, each clip's suggestion
comes from a source picked AT RANDOM (fixed seed) among the sources that have an answer for that
clip. The draw is balanced: clips are taken in a random order and each takes a random source among
the least-used ones it has, so every source gets a similar number of clips. --pick best uses the
priority order instead. The source is written here and in the private key, never on the page.

  python phase6_suggestions.py --out /tmp/suggestions.csv            # seed 20261116
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase0_baseline_table import key_of, load_csv  # noqa: E402

SOURCES = ["phase6_measurements", "gemini_checkpoints", "gemini_absolute", "stills_model"]
SOURCE_FILES = {
    "phase6_measurements": "baselines/phase6_prototype_2026-11-15/clip_predictions.csv",
    "gemini_checkpoints": "baselines/phase5_video_judge_v2_2026-11-16/checkpoints_crop/checkpoint_faults.csv",
    "gemini_absolute": "baselines/phase5_video_judge_v2_2026-11-16/absolute_crop_refs2/video_judge_scores.csv",
    "stills_model": "baselines/phase2_preliminary_2026-11-14/model_vs_alex.csv",
}
RUBRIC = SCRIPT_DIR / "docs" / "phase1" / "rubric_v0_3.json"
FIELDS = ["filename", "suggested_fault", "source", "second_fault", *SOURCES]
DEFAULT_SEED = 20261116


def v03_faults(rubric_path: Path = RUBRIC) -> tuple[list[str], dict[str, str]]:
    """(fault IDs in chain order then F00, retired ID -> v0.3 ID)."""
    rub = json.loads(Path(rubric_path).read_text())
    ids = [f["id"] for f in rub["faults_in_chain_order"]] + ["F00"]
    retired = {}
    for f in rub.get("faults_retired", []):
        now = f.get("now", "").split()
        retired[f["id"]] = next((w for w in now if w.startswith("F") and w[1:].isdigit()), "")
    return ids, retired


def to_v03(fault: str | None, valid: list[str], retired: dict[str, str]) -> str:
    """A fault ID as rubric v0.3 has it ('' for none, unknown or missing)."""
    f = (fault or "").strip().upper()
    if f in ("", "NONE", "NAN"):
        return ""
    if f.startswith("F") and f[1:].isdigit():
        f = f"F{int(f[1:]):02d}"
    f = retired.get(f, f)
    return f if f in valid else ""


def _cp_to_fault(cp: str) -> str:
    cp = cp.strip().upper()
    return "F" + cp[1:] if cp.startswith("C") and cp[1:].isdigit() else ""


def load_sources(root: Path = SCRIPT_DIR, rubric_path: Path = RUBRIC) -> dict[str, dict[str, tuple[str, str]]]:
    """source -> {clip key: (main fault, second fault)} in v0.3 IDs; a clip without a usable main fault is left out."""
    valid, retired = v03_faults(rubric_path)
    v = lambda f: to_v03(f, valid, retired)  # noqa: E731
    out: dict[str, dict[str, tuple[str, str]]] = {s: {} for s in SOURCES}
    for source in SOURCES:
        path = Path(root) / SOURCE_FILES[source]
        if not path.exists():
            continue
        for r in load_csv(path):
            if (r.get("error") or "").strip():
                continue
            if source == "phase6_measurements":
                main = v(r.get("priority_fault"))
                failed = [v(f) for f in (r.get("failed_checkpoints") or "").split()]
                second = next((f for f in failed if f and f != main), "")
            elif source == "gemini_checkpoints":
                if (r.get("complete") or "1").strip() not in ("1", "1.0", "True", "true"):
                    continue
                main = v(r.get("main_fault"))
                failed = [v(_cp_to_fault(c)) for c in (r.get("failed") or "").replace(";", " ").split()]
                second = next((f for f in failed if f and f != main), "")
                if not second and v(r.get("main_fault_incl_partly")) != main:
                    second = v(r.get("main_fault_incl_partly"))
            elif source == "gemini_absolute":
                main, second = v(r.get("priority_fault")), v(r.get("second_fault"))
            else:
                main, second = v(r.get("model_priority_fault")), v(r.get("model_second_fault"))
            if main:
                out[source][key_of(r["filename"])] = (main, "" if second == main else second)
    return out


def suggest(filenames: list[str], sources: dict[str, dict[str, tuple[str, str]]], seed: int = DEFAULT_SEED,
            pick: str = "random") -> list[dict[str, str]]:
    """One row per clip (in the order given). pick='random': balanced random source per clip; 'best': priority order."""
    rng = random.Random(seed)
    chosen: dict[str, str] = {}
    if pick == "best":
        for name in filenames:
            chosen[name] = next((s for s in SOURCES if key_of(name) in sources[s]), "")
    else:
        used = {s: 0 for s in SOURCES}
        order = list(filenames)
        rng.shuffle(order)
        for name in order:
            have = [s for s in SOURCES if key_of(name) in sources[s]]
            if not have:
                chosen[name] = ""
                continue
            least = min(used[s] for s in have)
            s = rng.choice([s for s in have if used[s] == least])
            used[s] += 1
            chosen[name] = s
    rows = []
    for name in filenames:
        k, s = key_of(name), chosen[name]
        main, second = sources[s][k] if s else ("", "")
        rows.append({"filename": name, "suggested_fault": main, "source": s, "second_fault": second,
                     **{src: sources[src].get(k, ("", ""))[0] for src in SOURCES}})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="CSV to write: filename,suggested_fault,source,second_fault + each source's answer")
    ap.add_argument("--ground-truth", default=str(SCRIPT_DIR / "coach_ground_truth_from_screenshot.csv"), help="the clip list (same as the offline files)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--pick", choices=["random", "best"], default="random")
    args = ap.parse_args(argv)

    names = [r["filename"] for r in load_csv(Path(args.ground_truth))]
    sources = load_sources()
    rows = suggest(names, sources, args.seed, args.pick)
    missing = [r["filename"] for r in rows if not r["source"]]
    if missing:
        raise SystemExit("No AI source has a main fault for: " + ", ".join(missing))
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    counts = {s: sum(r["source"] == s for r in rows) for s in SOURCES}
    print(f"{len(rows)} suggestions -> {out} (pick {args.pick}, seed {args.seed})")
    print("  clips per source: " + ", ".join(f"{s} {n}" for s, n in counts.items()))
    print("  clips each source has an answer for: " + ", ".join(f"{s} {len(sources[s])}" for s in SOURCES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
