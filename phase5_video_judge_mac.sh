#!/usr/bin/env bash
# Video-model judge (Gemini) on the clips in the swipe file you sent out, run from the Mac.
#
#   cd ~/Documents/battingiq-phase0 && git pull && ./phase5_video_judge_mac.sh
#
# Asks for your Gemini API key (paste it; nothing shows on screen; it is not saved).
# Use a key from a project with billing linked (paid tier). Stops itself at GBP 4.
# Sends only the trimmed, face-blurred clips inside the swipe file. Pushes only the scores.
set -euo pipefail
cd "$(dirname "$0")"
BRANCH="claude/batting-iq-phase-0-375kjr"
HTML="${1:-$(ls -t "$HOME"/Desktop/BattingIQ_which_drive.html "$HOME"/Downloads/BattingIQ_which_drive*.html 2>/dev/null | head -1)}"
KEY="${2:-$(ls -t "$HOME"/Desktop/BattingIQ_which_drive_key_PRIVATE.csv 2>/dev/null | head -1)}"
[ -f "$HTML" ] || { echo "Swipe file not found. Run: ./phase5_video_judge_mac.sh <path to BattingIQ_which_drive.html> <path to its _key_PRIVATE.csv>"; exit 1; }
[ -f "$KEY" ] || { echo "Key file not found next to it. Pass both paths as above."; exit 1; }
grep -q "$(sed -n '2p' "$KEY" | cut -d, -f3)" "$HTML" || { echo "The key file is for a different build than the swipe file."; exit 1; }
echo "Swipe file: $HTML"
printf "Paste your Gemini API key and press Enter (hidden): "
read -rs GEMINI_API_KEY; echo
export GEMINI_API_KEY
OUT="baselines/phase5_video_judge_$(date +%F)"
.venv310/bin/python phase5_video_judge.py --clips-html "$HTML" --key "$KEY" --out "$OUT" ${GEMINI_MODEL:+--model "$GEMINI_MODEL"}
unset GEMINI_API_KEY
git add "$OUT"
git commit -q -m "Video-model judge scores ($(date +%F))" || echo "nothing new to commit"
ok=0
for i in 1 2 3 4; do git pull -q --rebase origin "$BRANCH" && git push -q -u origin "$BRANCH" && { ok=1; break; } || sleep $((2 ** i)); done
[ "$ok" = 1 ] && echo "==> Done. Pushed $OUT" || echo "Push failed: run  git push -u origin $BRANCH  by hand."
