#!/usr/bin/env bash
# Phase 2 on the Mac, one command (run after phase0_mac_run.sh):
#   cd ~/Documents/battingiq-phase0 && git pull --rebase && ./phase2_mac_pack.sh ["<clip folder>"]
# Builds the blind pack NEXT TO the clip folder (never in the repo), one zip per coach with only the clips
# that coach may see, and the still frames for the fourth-coach test. Nothing is uploaded or pushed.
set -euo pipefail
DEFAULT_VIDS="$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:"
VIDS="${1:-$DEFAULT_VIDS}"
cd "$(dirname "$0")"
[ -d "$VIDS" ] || VIDS="$VIDS/"
[ -d "$VIDS" ] || { echo "STOP: clip folder not found: $VIDS"; exit 1; }
PY=.venv310/bin/python
[ -x "$PY" ] || { echo "STOP: run ./phase0_mac_run.sh first (it sets up Python)"; exit 1; }
export PATH="$PWD/.venv310/bin:$PATH"
PACK="$(cd "$VIDS/.." && pwd)/blind_pack_$(date +%F)"
[ -e "$PACK" ] && { echo "STOP: $PACK already exists. Keep it (its IDs may already be with the coaches) or move it away first."; exit 1; }

"$PY" phase2_blind_pack.py --video-dir "$VIDS" --out "$PACK"
REPORTS="$(ls -d baselines/phase0_*/reports/auto 2>/dev/null | tail -1 || true)"
"$PY" phase2_model_pack.py --pack "$PACK" --video-dir "$VIDS" ${REPORTS:+--baseline-reports "$REPORTS"}
OFFLINE="$PACK/BattingIQ_which_drive.html"
"$PY" phase2_offline_pairs.py --video-dir "$VIDS" --out "$OFFLINE"

echo
echo "Done. In $PACK:"
echo "  BattingIQ_which_drive.html                the offline file for anyone: clips built in, faces blurred, no sound"
echo "  for_Alex.zip, for_Cam.zip, for_Viv.zip   send each coach their own zip (AirDrop or a private Drive link)"
echo "  model_inputs/                             choose this folder on the Fourth coach test page"
echo "  blind_key_PRIVATE.csv, *_key_PRIVATE.csv  KEEP PRIVATE: never send them to anyone"
open "$PACK" 2>/dev/null || true
