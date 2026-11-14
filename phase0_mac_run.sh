#!/usr/bin/env bash
# Phase 0, all clip-dependent steps in one run (the clips live only on Alex's Mac).
#
#   git clone -b claude/batting-iq-phase-0-375kjr https://github.com/AlexTredge8/batting-backend.git ~/Documents/battingiq-phase0
#   cd ~/Documents/battingiq-phase0 && ./phase0_mac_run.sh ["<clip folder>"]
#
# Measurement only: no scoring, rule, threshold or detector change. Nothing
# leaves the Mac except the final git push of the CSV/JSON/markdown results
# (no videos, frames or landmarks). Works with macOS bash 3.2.
set -euo pipefail

BRANCH="claude/batting-iq-phase-0-375kjr"
DEFAULT_VIDS="$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:"
VIDS="${1:-$DEFAULT_VIDS}"
REF_MD5="0a1b3a74e1f123a97085c2fafbaa12ea"
DATE="$(date +%F)"
RUN="calibration_output/phase0_${DATE}"
OUT="baselines/phase0_${DATE}"
cd "$(dirname "$0")"

say()  { printf '\n==> %s\n' "$*"; }
fail() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }
md5_of() { if command -v md5 >/dev/null 2>&1; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }

say "Checks: branch, clip folder"
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || fail "run this on branch $BRANCH (git checkout $BRANCH)"
if [ ! -d "$VIDS" ]; then
  # Finder shows '/' in a folder name as ':'; try the trailing-slash spelling too.
  ALT="$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:/"
  [ -d "$ALT" ] && VIDS="$ALT" || fail "clip folder not found: $VIDS  (pass it as the first argument)"
fi
echo "Clips: $VIDS"
echo "Commit: $(git rev-parse HEAD)"

say "Python 3.10 environment (.venv310)"
if [ ! -x .venv310/bin/python ]; then
  if command -v python3.10 >/dev/null 2>&1; then python3.10 -m venv .venv310
  elif command -v uv >/dev/null 2>&1; then uv venv -p 3.10 .venv310
  elif command -v pyenv >/dev/null 2>&1 && pyenv prefix 3.10.13 >/dev/null 2>&1; then "$(pyenv prefix 3.10.13)/bin/python" -m venv .venv310
  else fail "no Python 3.10 found. Install one with: brew install python@3.10  (or: brew install uv)"; fi
fi
PY=.venv310/bin/python
"$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip >/dev/null 2>&1 || true
if "$PY" -m pip --version >/dev/null 2>&1; then "$PY" -m pip install -q -r requirements-dev.txt
else uv pip install -q -p "$PY" -r requirements-dev.txt; fi
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version' || fail "venv is not Python 3.10"

say "ffmpeg on PATH (audio contact needs it; without it every clip falls back to pose contact)"
export PATH="$PWD/.venv310/bin:$PATH"
if ! command -v ffmpeg >/dev/null 2>&1; then
  ln -sf "$("$PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" .venv310/bin/ffmpeg
fi
ffmpeg -version | head -1
echo "ffmpeg: $(command -v ffmpeg)"

say "Pre-flight 1/3: unit tests"
"$PY" -m pytest tests -q 2>&1 | tail -1
"$PY" -m pytest tests -q >/dev/null 2>&1 || fail "unit tests fail"
say "Pre-flight 2/3: reference-clip end-to-end test (pins anchors 60/61/77/73/81/109)"
BATTINGIQ_E2E=1 "$PY" -m pytest tests/test_reference_video_e2e.py -q 2>&1 | tail -1
BATTINGIQ_E2E=1 "$PY" -m pytest tests/test_reference_video_e2e.py -q >/dev/null 2>&1 || fail "e2e reference test fails: this Mac does not reproduce the calibrated path"
say "Pre-flight 3/3: clips present, reference md5"
"$PY" - "$VIDS" <<'PYEOF'
import sys
from pathlib import Path
from phase0_baseline_table import find_video, load_csv
vids = Path(sys.argv[1])
rows = load_csv(Path("coach_ground_truth_from_screenshot.csv"))
missing = [r["filename"] for r in rows if find_video(vids, r["filename"]) is None]
print(f"matched {len(rows) - len(missing)}/{len(rows)} clips")
if missing:
    sys.exit("missing: " + ", ".join(missing))
PYEOF
VIV="$(find "$VIDS" -iname 'Viv P_Offdrive Elite.mov' | head -1)"
[ "$(md5_of "$VIV")" = "$REF_MD5" ] || fail "Viv P_Offdrive Elite.mov md5 differs from test_batting.mov"
echo "Viv md5 = $REF_MD5 (same file as test_batting.mov)"

mkdir -p calibration_output
START=$(date +%s)
say "Batch 1/2: auto + validated (17 clips, local, no Railway, no cached fallback)"
"$PY" batch_calibration_compare.py --ground-truth coach_ground_truth_from_screenshot.csv \
  --video-dir "$VIDS" --anchor-truth anchor_truth.csv --dual-mode --skip-railway --export-measurements \
  --output-dir "$RUN" > "$RUN.batch1.log" 2>&1 || { tail -30 "$RUN.batch1.log"; fail "batch 1 failed"; }
say "Batch 2/2: auto repeat (determinism)"
"$PY" batch_calibration_compare.py --ground-truth coach_ground_truth_from_screenshot.csv \
  --video-dir "$VIDS" --skip-railway --export-measurements \
  --output-dir "$RUN/auto_repeat" > "$RUN.batch2.log" 2>&1 || { tail -30 "$RUN.batch2.log"; fail "batch 2 failed"; }
for m in auto validated auto_repeat; do
  grep -q "Local successes: 17" "$RUN/$m/batch_calibration.log" || fail "$m: not 17 local successes (see $RUN/$m/batch_calibration.log)"
  grep -q "Cached local fallbacks used: 0" "$RUN/$m/battingiq_calibration_metrics.txt" || fail "$m: cached fallback used"
done
echo "Batches done in $(( ($(date +%s) - START) / 60 )) min"

mkdir -p "$OUT"
say "Left-hander check (Cam T_LH with handedness=left)"
"$PY" phase0_lh_check.py --run-dir "$RUN" --out-dir "$OUT" --video-dir "$VIDS" 2>&1 | grep -E '^(auto|validated):' || fail "left-hander check failed"

say "Baseline table"
"$PY" phase0_baseline_table.py --run-dir "$RUN" --out-dir "$OUT" --video-dir "$VIDS" > "$OUT/.table.log" 2>&1 || { cat "$OUT/.table.log"; fail "table failed"; }
rm -f "$OUT/.table.log"
for m in auto validated auto_repeat; do
  mkdir -p "$OUT/raw/$m" "$OUT/reports/$m"
  for f in battingiq_calibration_comparison_detailed.csv battingiq_calibration_tier_summary.csv \
           battingiq_calibration_metrics.txt rule_measurements.csv batch_calibration.log \
           tuning_summary.csv tuning_metrics.txt heldout_summary.csv heldout_metrics.txt; do
    [ -f "$RUN/$m/$f" ] && cp "$RUN/$m/$f" "$OUT/raw/$m/"
  done
  find "$RUN/$m/local_runs" -name '*_battingiq.json' -exec cp {} "$OUT/reports/$m/" \;
done
[ -f "$RUN/dual_run_manifest.json" ] && cp "$RUN/dual_run_manifest.json" "$OUT/raw/"

say "Clip sheet: measured columns + pose cache"
"$PY" phase0_clip_sheet.py --video-dir "$VIDS" 2>&1 | grep -E '^Wrote' || fail "clip sheet failed"

say "Wrong-plane test"
"$PY" phase0_wrong_plane.py --measurements "$RUN/validated/rule_measurements.csv" --out-dir "$OUT/wrong_plane" > /dev/null 2>&1 \
  || fail "wrong-plane test failed"

if [ "${PHASE0_NO_PUSH:-0}" = 1 ]; then
  say "PHASE0_NO_PUSH=1: skipping commit and push"
else
say "Commit and push (results only: CSV, JSON, markdown)"
git add "$OUT" clip_sheet.csv clip_sheet_notes.md
git commit -q -m "Phase 0 baseline run on the 18 clips (${DATE})

Four-rule scorer, auto + validated + auto repeat, 17 anchored clips;
left-hander check; clip sheet measured columns; wrong-plane test.
Generated by phase0_mac_run.sh at $(git rev-parse --short HEAD)."
ok=0
for i in 1 2 3 4; do git push -u origin "$BRANCH" && { ok=1; break; } || sleep $((2 ** i)); done
[ "$ok" = 1 ] || echo "Push failed: run  git push -u origin $BRANCH  by hand."
fi

say "Done in $(( ($(date +%s) - START) / 60 )) min. Summary:"
cat "$OUT/baseline_summary.md"
echo
cat "$OUT/wrong_plane/wrong_plane_report.md"
