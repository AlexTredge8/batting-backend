#!/usr/bin/env bash
# Phase 5, current measurement set: measurements, the worse-filming test (test 2) and the entry tests.
#
#   cd ~/Documents/battingiq-phase0 && git pull && ./phase5_mac_next.sh
#
# About 10-15 minutes (three degraded copies of each clip go through the pose model).
# Pushes only tables; degraded videos and landmarks stay in calibration_output/.
set -euo pipefail
cd "$(dirname "$0")"
./mac_pin_env.sh
BRANCH="claude/batting-iq-phase-0-375kjr"
VIDS="${1:-$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:}"
[ -d "$VIDS" ] || VIDS="$VIDS/"
[ -d "$VIDS" ] || { echo "clip folder not found: $VIDS (pass it as the first argument)"; exit 1; }
PY=.venv310/bin/python
export PATH="$PWD/.venv310/bin:$PATH"
SET="$(sed -n 's/^MEASUREMENT_SET = "\(v[0-9]*\)".*/\1/p' phase5_checkpoints.py)"
OUT="baselines/phase5_$(date +%F)_${SET}"
echo "==> Measurements ($SET)"
"$PY" phase5_checkpoints.py --out "$OUT" --video-dir "$VIDS" 2>&1 | grep -v -i warning
echo "==> Worse-filming test (one line per clip and variant)"
"$PY" phase5_degrade_test.py --video-dir "$VIDS" --out "$OUT" 2>&1 | grep -E "^  (lowres|lowfps|tilt):|^\|" || true
echo "==> Entry tests"
"$PY" phase5_entry_tests.py --measurements "$OUT/checkpoint_measurements.csv" --degrade "$OUT/degrade_test.csv" --out "$OUT" > /dev/null
cat "$OUT/entry_tests.md"
git add "$OUT"
git commit -q -m "Phase 5 ${SET} with the worse-filming test ($(date +%F))" || echo "nothing new to commit"
ok=0
for i in 1 2 3 4; do git pull -q --rebase origin "$BRANCH" && git push -q -u origin "$BRANCH" && { ok=1; break; } || sleep $((2 ** i)); done
[ "$ok" = 1 ] && echo "==> Done. Pushed $OUT" || echo "Push failed: run  git push -u origin $BRANCH  by hand."
