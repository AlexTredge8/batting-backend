#!/usr/bin/env bash
# Video-model judge, round 2 (Gemini), run from the Mac: three experiments on the swipe-file clips.
#
#   cd ~/Documents/battingiq-phase0 && git pull && ./phase5_video_judge_v2_mac.sh
#
#   1. pairwise + crop     "which drive is better?" on up to 70 balanced pairs   cap GBP 1.50
#   2. absolute + crop     score sheet, 2 reference clips per tier (8 refs)      cap GBP 0.90
#   3. checkpoints + crop  12 yes/partly/no questions in 4 requests per clip     cap GBP 1.00
#                                                                         total cap GBP 3.40
# Each run stops itself at its cap (cost from Google's token counts, priced above list price).
# Asks for your Gemini API key once (paste it; nothing shows on screen; it is not saved).
# Use a key from a project with billing linked (paid tier). Sends only the trimmed, face-blurred
# clips inside the swipe file, cropped around the batter. Pushes only the results (no video, no key file).
set -euo pipefail
cd "$(dirname "$0")"
./mac_pin_env.sh
BRANCH="claude/batting-iq-phase-0-375kjr"
PY=".venv310/bin/python"
HTML="${1:-$(ls -t "$HOME"/Downloads/BattingIQ_which_drive*.html "$HOME"/Desktop/BattingIQ_which_drive.html 2>/dev/null | head -1 || true)}"
KEY="${2:-$(ls -t "$HOME"/Desktop/BattingIQ_which_drive_key_PRIVATE.csv "$HOME"/Downloads/BattingIQ_which_drive_key_PRIVATE.csv 2>/dev/null | head -1 || true)}"
[ -f "$HTML" ] || { echo "Swipe file not found. Run: ./phase5_video_judge_v2_mac.sh <path to BattingIQ_which_drive.html> <path to its _key_PRIVATE.csv>"; exit 1; }
[ -f "$KEY" ] || { echo "Key file not found. Pass both paths as above."; exit 1; }
grep -q "$(sed -n '2p' "$KEY" | cut -d, -f3)" "$HTML" || { echo "The key file is for a different build than the swipe file."; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is needed to crop the clips: brew install ffmpeg"; exit 1; }
echo "Swipe file: $HTML"
echo "Key file:   $KEY"

OUT="baselines/phase5_video_judge_v2_$(date +%F)"
CROPS="$(mktemp -d)"   # cropped clips live here only for this run (outside the repo), then are deleted
trap 'rm -rf "$CROPS"; unset GEMINI_API_KEY' EXIT
COMMON=(--clips-html "$HTML" --key "$KEY" --crop --crop-cache "$CROPS" ${GEMINI_MODEL:+--model "$GEMINI_MODEL"})

echo "==> Cropping the clips and checking the requests (free, about a minute)"
"$PY" phase5_video_judge_v2.py --mode absolute --refs-per-level 2 --out "$OUT/absolute_crop_refs2" "${COMMON[@]}" --dry-run

printf "Paste your Gemini API key and press Enter (hidden): "
read -rs GEMINI_API_KEY; echo
export GEMINI_API_KEY

FAILED=""
run() {  # run <name> <args...>: a failed experiment does not stop the next; a rejected key stops all
  local name="$1"; shift
  echo; echo "==> $name"
  set +e
  "$PY" phase5_video_judge_v2.py "$@" "${COMMON[@]}"
  local rc=$?
  set -e
  if [ "$rc" = 3 ]; then echo "Google rejected the key: nothing more was sent."; exit 3; fi
  [ "$rc" = 0 ] || { echo "==> $name failed (exit $rc); going on."; FAILED="$FAILED $name"; }
}
run "1/3 pairwise + crop" --mode pairwise --max-pairs 70 --budget-gbp 1.50 --out "$OUT/pairwise_crop"
run "2/3 absolute + crop, 2 refs per tier" --mode absolute --refs-per-level 2 --budget-gbp 0.90 --out "$OUT/absolute_crop_refs2"
run "3/3 checkpoints + crop" --mode checkpoints --thinking low --budget-gbp 1.00 --out "$OUT/checkpoints_crop"
unset GEMINI_API_KEY

[ -d "$OUT" ] || { echo "No results to push."; exit 1; }
git add "$OUT"
git commit -q -m "Video-model judge v2: pairwise, absolute with more refs, checkpoints, all cropped ($(date +%F))" || echo "nothing new to commit"
ok=0
for i in 1 2 3 4; do git pull -q --rebase origin "$BRANCH" && git push -q -u origin "$BRANCH" && { ok=1; break; } || sleep $((2 ** i)); done
[ -n "$FAILED" ] && echo "Failed:$FAILED (see the messages above)."
[ "$ok" = 1 ] && echo "==> Done. Pushed $OUT (see the summary.md in each folder)" || echo "Push failed: run  git push -u origin $BRANCH  by hand."
