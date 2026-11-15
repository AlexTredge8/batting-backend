#!/usr/bin/env bash
# Pose-model comparison on the 18 clips: MediaPipe (current) against RTMPose (open source, CPU).
#
#   cd ~/Documents/battingiq-phase0 && git pull && ./phase5_pose_compare_mac.sh
#
# Installs rtmlib + onnxruntime into .venv310 (first run downloads about 200 MB of model files
# from download.openmmlab.com). About 15-30 minutes. Pushes only tables and a report.
set -euo pipefail
cd "$(dirname "$0")"
BRANCH="claude/batting-iq-phase-0-375kjr"
VIDS="${1:-$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:}"
[ -d "$VIDS" ] || VIDS="$VIDS/"
[ -d "$VIDS" ] || { echo "clip folder not found: $VIDS (pass it as the first argument)"; exit 1; }
[ -x .venv310/bin/python ] || { echo "no .venv310 here: run ./phase0_mac_run.sh once first"; exit 1; }
PY=.venv310/bin/python
echo "==> Installing rtmlib and onnxruntime"
"$PY" -m pip install -q rtmlib onnxruntime
OUT="baselines/phase5_pose_compare_$(date +%F)"
echo "==> Comparing (RTMPose extraction prints one line per clip)"
"$PY" phase5_pose_compare.py --video-dir "$VIDS" --out "$OUT" 2>&1 | grep -v -i "warning"
git add "$OUT"
git commit -q -m "Pose model comparison on the clips ($(date +%F))" || echo "nothing new to commit"
ok=0
for i in 1 2 3 4; do git pull -q --rebase origin "$BRANCH" && git push -q -u origin "$BRANCH" && { ok=1; break; } || sleep $((2 ** i)); done
[ "$ok" = 1 ] && echo "==> Done. Pushed $OUT" || echo "Push failed: run  git push -u origin $BRANCH  by hand."
