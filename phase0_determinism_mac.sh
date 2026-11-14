#!/usr/bin/env bash
# Same clip, same score? Runs the 6 clips that changed between runs on the Mac (plus 2 controls)
# 3 times each in fresh processes, in pose "video" mode and "image" mode, then commits and pushes.
#   cd ~/Documents/battingiq-phase0 && git pull --rebase && ./phase0_determinism_mac.sh
# About 25 minutes. Keep the Mac awake (caffeinate runs while it works).
set -euo pipefail
DEFAULT_VIDS="$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:"
VIDS="${1:-$DEFAULT_VIDS}"
cd "$(dirname "$0")"
[ -d "$VIDS" ] || VIDS="$VIDS/"
[ -d "$VIDS" ] || { echo "STOP: clip folder not found: $VIDS"; exit 1; }
PY=.venv310/bin/python
export PATH="$PWD/.venv310/bin:$PATH"
OUT="baselines/phase0_$(date +%F)_determinism"
caffeinate -i -w $$ >/dev/null 2>&1 &
"$PY" phase0_determinism.py --video-dir "$VIDS" --out "$OUT" 2>&1 | grep -vE '^(W0000|I0000|E0000)|INFO|WARNING'
git add "$OUT"
git commit -q -m "Determinism check by pose running mode on the Mac ($(date +%F))"
git pull -q --rebase origin claude/batting-iq-phase-0-375kjr && git push -q origin claude/batting-iq-phase-0-375kjr && echo "Pushed. Tell Claude it is done."
