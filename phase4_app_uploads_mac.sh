#!/usr/bin/env bash
# Downloads the app's uploads to the Mac and keeps only new clips (not re-uploads of the 18).
#   1. Save app_uploads_manifest.tsv (sent in the Claude session) to ~/Downloads
#   2. cd ~/Documents/battingiq-phase0 && git pull && ./phase4_app_uploads_mac.sh
# Everything stays on the Mac, outside the repo. Nothing is committed.
set -euo pipefail
cd "$(dirname "$0")"
BASE="$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders"
VIDS="$BASE/battingiq-groundtruth:/videos:"; [ -d "$VIDS" ] || VIDS="$VIDS/"
MAN="${1:-$HOME/Downloads/app_uploads_manifest.tsv}"
[ -f "$MAN" ] || { echo "Manifest not found: $MAN (save it from the Claude session to Downloads)"; exit 1; }
.venv310/bin/python phase4_app_uploads.py --manifest "$MAN" --clips "$VIDS" --out "$BASE/app_uploads"
echo; echo "Open: $BASE/app_uploads/new  (watch them; tell Claude which are front-foot drives)"
