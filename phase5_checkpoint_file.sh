#!/usr/bin/env bash
# Builds the offline checkpoint coach check (the rated checkpoints of rubric v0.3, 2/1/0/X per clip) on the Desktop,
# with faces blurred. The *_key_PRIVATE.csv next to it maps clip codes to filenames: keep it, never send it.
#
#   cd ~/Documents/battingiq-phase0 && git pull && ./phase5_checkpoint_file.sh
set -euo pipefail
cd "$(dirname "$0")"
./mac_pin_env.sh
VIDS="${1:-$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:}"
[ -d "$VIDS" ] || VIDS="$VIDS/"
[ -d "$VIDS" ] || { echo "clip folder not found: $VIDS (pass it as the first argument)"; exit 1; }
[ -x .venv310/bin/python ] || { echo "no .venv310 here: run ./phase0_mac_run.sh once first"; exit 1; }
export PATH="$PWD/.venv310/bin:$PATH"
.venv310/bin/python phase2_offline_pairs.py --video-dir "$VIDS" --checkpoint-out "$HOME/Desktop/BattingIQ_checkpoints.html"
echo
echo "Done. Send ~/Desktop/BattingIQ_checkpoints.html to Cam (OneDrive link or attachment)."
echo "Keep ~/Desktop/BattingIQ_checkpoints_key_PRIVATE.csv: send it to no one; upload it to Claude only when the codes come back."
