#!/usr/bin/env bash
# Downloads the app's uploads to the Mac and keeps only new clips (not re-uploads of the 18).
#   1. Download the videos from Supabase Storage (battingiq-uploads) into ~/Downloads/app_uploads_raw
#   2. cd ~/Documents/battingiq-phase0 && git pull && ./phase4_app_uploads_mac.sh
# Everything stays on the Mac, outside the repo. Nothing is committed.
set -euo pipefail
cd "$(dirname "$0")"
BASE="$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders"
VIDS="$BASE/battingiq-groundtruth:/videos:"; [ -d "$VIDS" ] || VIDS="$VIDS/"
SRC="${1:-$HOME/Downloads/app_uploads_raw}"
# The uploads bucket is private, so links do not work: download the videos from the Supabase
# dashboard (Storage > battingiq-uploads) into ~/Downloads/app_uploads_raw, then run this.
[ -d "$SRC" ] || { echo "Folder not found: $SRC. Download the videos from Supabase Storage into it first."; exit 1; }
.venv310/bin/python phase4_app_uploads.py --local-dir "$SRC" --clips "$VIDS" --out "$BASE/app_uploads"
echo; echo "Open: $BASE/app_uploads/new  (watch them; tell Claude which are front-foot drives)"
