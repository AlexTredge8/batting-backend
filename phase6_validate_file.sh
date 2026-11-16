#!/usr/bin/env bash
# Builds the offline "Check the AI's tips" file on the Desktop: for each clip the AI's main fault and its tip,
# and Alex answers yes / partly / no (about 10 to 15 seconds a clip). Faces are blurred.
# The AI's suggestions come from phase6_suggestions.py (one AI source per clip, picked at random with a fixed seed).
# The *_key_PRIVATE.csv next to it maps clip codes to filenames and says which AI source made each suggestion:
# keep it, never send it, and do not open it before you have done the check (it would tell you which AI is which).
#
#   cd ~/Documents/battingiq-phase0 && git pull && ./phase6_validate_file.sh
set -euo pipefail
cd "$(dirname "$0")"
./mac_pin_env.sh
VIDS="${1:-$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:}"
[ -d "$VIDS" ] || VIDS="$VIDS/"
[ -d "$VIDS" ] || { echo "clip folder not found: $VIDS (pass it as the first argument)"; exit 1; }
[ -x .venv310/bin/python ] || { echo "no .venv310 here: run ./phase0_mac_run.sh once first"; exit 1; }
export PATH="$PWD/.venv310/bin:$PATH"
SUGG="$(mktemp "${TMPDIR:-/tmp}/biq_suggestions.XXXXXX")"
trap 'rm -f "$SUGG"' EXIT
.venv310/bin/python phase6_suggestions.py --out "$SUGG"
.venv310/bin/python phase2_offline_pairs.py --video-dir "$VIDS" --suggestions "$SUGG" \
  --validate-out "$HOME/Desktop/BattingIQ_check_the_AI.html"
echo
echo "Done. Open ~/Desktop/BattingIQ_check_the_AI.html (double-click), or AirDrop it to your phone."
echo "At the end, paste the answer code into your chat with Claude."
echo "Keep ~/Desktop/BattingIQ_check_the_AI_key_PRIVATE.csv: send it to no one; upload it to Claude only with the code."
