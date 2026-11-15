#!/usr/bin/env bash
# Before/after for contact-anchored key frames on the 17 clips (full production path, automatic).
#
#   cd ~/Documents/battingiq-phase0 && git pull && ./phase5_anchor_prior_mac.sh
#
# About 15-25 minutes. The live app is not changed: the switch is set only for this run.
set -euo pipefail
cd "$(dirname "$0")"
BRANCH="claude/batting-iq-phase-0-375kjr"
VIDS="${1:-$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:}"
[ -d "$VIDS" ] || VIDS="$VIDS/"
[ -d "$VIDS" ] || { echo "clip folder not found: $VIDS (pass it as the first argument)"; exit 1; }
PY=.venv310/bin/python
export PATH="$PWD/.venv310/bin:$PATH"
RUN="calibration_output/anchor_prior_$(date +%F)"
OUT="baselines/phase5_anchor_prior_$(date +%F)"
mkdir -p calibration_output
echo "==> Batch with ANCHOR_MODE=contact_prior (17 clips)"
ANCHOR_MODE=contact_prior "$PY" batch_calibration_compare.py --ground-truth coach_ground_truth_from_screenshot.csv \
  --video-dir "$VIDS" --skip-railway --export-measurements --output-dir "$RUN" > "$RUN.log" 2>&1 \
  || { tail -30 "$RUN.log"; echo "batch failed"; exit 1; }
"$PY" phase5_anchor_prior_compare.py --run-dir "$RUN" --out "$OUT"
git add "$OUT"
git commit -q -m "Contact-anchored key frames: before/after on the clips ($(date +%F))" || echo "nothing new to commit"
ok=0
for i in 1 2 3 4; do git pull -q --rebase origin "$BRANCH" && git push -q -u origin "$BRANCH" && { ok=1; break; } || sleep $((2 ** i)); done
[ "$ok" = 1 ] && echo "==> Done. Pushed $OUT" || echo "Push failed: run  git push -u origin $BRANCH  by hand."
