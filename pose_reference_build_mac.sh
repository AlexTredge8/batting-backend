#!/usr/bin/env bash
# Rebuild the "best practice" reference figure (reference/pose_reference_v1.json) from the Elite clips.
#
#   cd ~/Documents/battingiq-phase0 && git pull && ./pose_reference_build_mac.sh [--include-good-club] ["<clip folder>"]
#
# Uses the Phase 0/5 pose cache (calibration_output/pose_cache), the validated key frames (anchor_truth.csv)
# and clip_sheet.csv. Elite clips only unless --include-good-club is given. The measurement ranges come from
# the newest Phase 5 measurement table for the current measurement set (Elite and Good Club clips).
# The JSON holds only averaged, normalised stick-figure coordinates and value ranges: no names, clip names,
# frames, images or per-clip values. Only that JSON is committed and pushed. Under a minute.
set -euo pipefail
cd "$(dirname "$0")"
./mac_pin_env.sh
BRANCH="claude/batting-iq-phase-0-375kjr"
PY=.venv310/bin/python
CACHE="calibration_output/pose_cache"
OUT="reference/pose_reference_v1.json"
EXTRA=()
VIDS="$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:"
for arg in "$@"; do
  case "$arg" in
    --include-good-club) EXTRA+=(--include-good-club) ;;
    *) VIDS="$arg" ;;
  esac
done
[ -d "$VIDS" ] || VIDS="$VIDS/"

fail() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || fail "run this on branch $BRANCH (git checkout $BRANCH)"
[ -x "$PY" ] || fail "no .venv310 here: run ./phase0_mac_run.sh once first"
[ -d "$CACHE" ] || fail "no pose cache at $CACHE: run ./phase5_mac_run.sh once first"

SET="$(sed -n 's/^MEASUREMENT_SET = "\(v[0-9]*\)".*/\1/p' phase5_checkpoints.py)"
TABLE="$(ls -1d baselines/phase5_*_"$SET"/checkpoint_measurements.csv 2>/dev/null | sort | tail -n 1 || true)"
[ -n "$TABLE" ] || TABLE="baselines/phase5_2026-11-16_v8/checkpoint_measurements.csv"
[ -f "$TABLE" ] || fail "no Phase 5 measurement table found ($TABLE)"

echo "==> Building the reference figure (ranges from $TABLE)"
VIDARGS=()
[ -d "$VIDS" ] && VIDARGS=(--video-dir "$VIDS")  # only to work out a camera roll not cached yet
"$PY" pose_reference.py --cache-dir "$CACHE" --anchors anchor_truth.csv --clip-sheet clip_sheet.csv \
  --measurements "$TABLE" --out "$OUT" ${VIDARGS[@]+"${VIDARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | grep -v -i warning

echo "==> Checking the file holds no clip names"
"$PY" - "$OUT" <<'PYEOF'
import csv, json, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8").casefold()
names = [Path(r["filename"]).stem.casefold() for r in csv.DictReader(open("clip_sheet.csv", encoding="utf-8-sig"))]
bad = [n for n in names if n and n in text]
if bad:
    sys.exit(f"STOP: clip names found in the reference file: {bad}")
ref = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"  ok: {ref['sources']['pose_n_clips']} clips averaged, anonymised={ref['anonymised']}, provisional={ref['provisional']}")
PYEOF

git add "$OUT"
if git diff --cached --quiet -- "$OUT"; then
  echo "==> The reference figure has not changed: nothing to commit"
else
  git commit -q -m "Best-practice reference figure rebuilt on the Mac ($(date +%F))" -- "$OUT"
fi
ok=0
for i in 1 2 3 4; do git pull -q --rebase --autostash origin "$BRANCH" && git push -q -u origin "$BRANCH" && { ok=1; break; } || sleep $((2 ** i)); done
[ "$ok" = 1 ] && echo "==> Done. Pushed $OUT" || echo "Push failed: run  git push -u origin $BRANCH  by hand."
