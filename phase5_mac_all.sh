#!/usr/bin/env bash
# Everything that needs the clips, in one go (about 20-40 minutes; the pose comparison is the slow part):
#   1. Phase 5 v3 measurements and entry tests
#   2. contact-anchored key frames against the current detectors
#   3. Cam's checkpoint file on the Desktop (faces blurred)
#   4. MediaPipe against RTMPose
#
#   cd ~/Documents/battingiq-phase0 && git pull && ./phase5_mac_all.sh
#
# Pushes only tables and reports. Leave the laptop awake (plugged in) until it says ALL DONE.
set -euo pipefail
cd "$(dirname "$0")"
BRANCH="claude/batting-iq-phase-0-375kjr"
DATE="$(date +%F)"
PY=.venv310/bin/python
push() { local ok=0; for i in 1 2 3 4; do git pull -q --rebase origin "$BRANCH" && git push -q -u origin "$BRANCH" && { ok=1; break; } || sleep $((2 ** i)); done
         [ "$ok" = 1 ] || echo "Push failed: run  git push -u origin $BRANCH  by hand."; }

echo "==> 1/4 Phase 5 measurements"
./phase5_mac_run.sh "$@"

echo; echo "==> 2/4 Key frames: contact-anchored against current"
OUT="baselines/phase5_anchor_candidates_${DATE}"
"$PY" phase5_anchor_candidates.py --out "$OUT" 2>&1 | grep -v -i warning
git add "$OUT" && { git commit -q -m "Contact-anchored key-frame candidates on the clips (${DATE})" || true; } && push

echo; echo "==> 3/4 Cam's checkpoint file"
./phase5_checkpoint_file.sh "$@"

echo; echo "==> 4/4 Pose models (slow: one line per clip)"
./phase5_pose_compare_mac.sh "$@"

echo; echo "ALL DONE. Results are pushed. Cam's file: ~/Desktop/BattingIQ_checkpoints.html"
