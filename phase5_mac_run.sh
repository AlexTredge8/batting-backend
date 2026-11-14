#!/usr/bin/env bash
# Phase 5: checkpoint measurements on the 18 clips (the clips live only on Alex's Mac).
#
#   cd ~/Documents/battingiq-phase0 && git pull && ./phase5_mac_run.sh ["<clip folder>"]
#
# Uses the pose cache from the Phase 0 run (calibration_output/pose_cache). Re-extracts
# only clips missing from it. Pushes only the measurement tables and the entry-test
# report: no videos, frames or landmarks leave the Mac. About 1-5 minutes.
set -euo pipefail

BRANCH="claude/batting-iq-phase-0-375kjr"
DEFAULT_VIDS="$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:"
VIDS="${1:-$DEFAULT_VIDS}"
DATE="$(date +%F)"
CACHE="calibration_output/pose_cache"
OUT="baselines/phase5_${DATE}"
cd "$(dirname "$0")"

say()  { printf '\n==> %s\n' "$*"; }
fail() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || fail "run this on branch $BRANCH (git checkout $BRANCH)"
[ -x .venv310/bin/python ] || fail "no .venv310 here: run ./phase0_mac_run.sh once first"
PY=.venv310/bin/python
if [ ! -d "$VIDS" ]; then
  ALT="$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:/"
  [ -d "$ALT" ] && VIDS="$ALT" || fail "clip folder not found: $VIDS  (pass it as the first argument)"
fi

say "Pose cache: re-extract any clip that is missing (same pose path as production)"
mkdir -p "$CACHE"
"$PY" - "$VIDS" "$CACHE" <<'PYEOF'
import os, sys
from pathlib import Path
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
from phase0_baseline_table import find_video, load_csv
from phase0_clip_sheet import pose_facts
vids, cache = Path(sys.argv[1]), Path(sys.argv[2])
have = {p.stem.casefold() for p in cache.glob("*.npz")}
done = 0
for r in load_csv(Path("coach_ground_truth_from_screenshot.csv")):
    v = find_video(vids, r["filename"])
    if v is None:
        print("  not found:", r["filename"]); continue
    if v.stem.casefold() not in have:
        print("  extracting", v.name); pose_facts(v, None, cache); done += 1
print(f"  {len(list(cache.glob('*.npz')))} clips cached ({done} new)")
PYEOF

say "Measurements (validated and automatic key frames)"
"$PY" phase5_checkpoints.py --cache-dir "$CACHE" --out "$OUT" 2>&1 | grep -v -i warning

say "Entry tests (checkpoint ratings: the model for now; coaches' ratings replace it later)"
"$PY" phase5_entry_tests.py --measurements "$OUT/checkpoint_measurements.csv" --out "$OUT" > /dev/null
cat "$OUT/entry_tests.md"

say "Commit and push the tables (no landmarks, frames or video)"
git add "$OUT"
git commit -q -m "Phase 5 checkpoint measurements on the clips (${DATE})" || echo "nothing new to commit"
ok=0
for i in 1 2 3 4; do git pull -q --rebase origin "$BRANCH" && git push -q -u origin "$BRANCH" && { ok=1; break; } || sleep $((2 ** i)); done
[ "$ok" = 1 ] && say "Done. Pushed $OUT" || echo "Push failed: run  git push -u origin $BRANCH  by hand."
