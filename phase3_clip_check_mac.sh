#!/usr/bin/env bash
# Phase 3 clip check (Gemini), run from the Mac: can the video model answer the filming gate's
# clip-level questions (camera position, people in frame, whole batter visible, phone level,
# batter size, handedness, one shot, usable or not)?
#
#   cd ~/Documents/battingiq-phase0 && git pull && ./phase3_clip_check_mac.sh
#
# 18 clips x 2 runs = 36 requests, uncropped, 8 frames/s, temperature 0.     cap GBP 0.80
# It stops itself at the cap (cost from Google's token counts, priced above list price).
# Asks for your Gemini API key once (paste it; nothing shows on screen; it is not saved).
# Use a key from a project with billing linked (paid tier). Sends only the trimmed, face-blurred
# clips inside the swipe file. Pushes only the results (no video, no key file).
set -euo pipefail
cd "$(dirname "$0")"
./mac_pin_env.sh
BRANCH="claude/batting-iq-phase-0-375kjr"
PY=".venv310/bin/python"
HTML="${1:-$(ls -t "$HOME"/Downloads/BattingIQ_which_drive*.html "$HOME"/Desktop/BattingIQ_which_drive.html 2>/dev/null | head -1 || true)}"
KEY="${2:-$(ls -t "$HOME"/Desktop/BattingIQ_which_drive_key_PRIVATE.csv "$HOME"/Downloads/BattingIQ_which_drive_key_PRIVATE.csv 2>/dev/null | head -1 || true)}"
[ -f "$HTML" ] || { echo "Swipe file not found. Run: ./phase3_clip_check_mac.sh <path to BattingIQ_which_drive.html> <path to its _key_PRIVATE.csv>"; exit 1; }
[ -f "$KEY" ] || { echo "Key file not found. Pass both paths as above."; exit 1; }
grep -q "$(sed -n '2p' "$KEY" | cut -d, -f3)" "$HTML" || { echo "The key file is for a different build than the swipe file."; exit 1; }
echo "Swipe file: $HTML"
echo "Key file:   $KEY"

OUT="baselines/phase3_clip_check_$(date +%F)"
BUDGET="0.80"
trap 'unset GEMINI_API_KEY' EXIT
COMMON=(--clips-html "$HTML" --key "$KEY" --out "$OUT" --budget-gbp "$BUDGET" ${GEMINI_MODEL:+--model "$GEMINI_MODEL"})

echo "==> Checking the requests and the cost (free)"
"$PY" phase3_clip_check.py "${COMMON[@]}" --dry-run

printf "Paste your Gemini API key and press Enter (hidden): "
read -rs GEMINI_API_KEY; echo
export GEMINI_API_KEY

echo; echo "==> Clip check: 18 clips x 2 runs (cap GBP $BUDGET)"
set +e
"$PY" phase3_clip_check.py "${COMMON[@]}"
rc=$?
set -e
unset GEMINI_API_KEY
if [ "$rc" = 3 ]; then echo "Google rejected the key: nothing was sent."; exit 3; fi
[ "$rc" = 0 ] || echo "==> The clip check failed (exit $rc); pushing whatever results it wrote."

[ -d "$OUT" ] || { echo "No results to push."; exit 1; }
git add "$OUT"
git commit -q -m "Phase 3 clip check with Gemini: 18 clips x 2 runs ($(date +%F))" || echo "nothing new to commit"
ok=0
for i in 1 2 3 4; do git pull -q --rebase origin "$BRANCH" && git push -q -u origin "$BRANCH" && { ok=1; break; } || sleep $((2 ** i)); done
[ "$ok" = 1 ] && echo "==> Done. Pushed $OUT (see summary.md there)" || echo "Push failed: run  git push -u origin $BRANCH  by hand."
