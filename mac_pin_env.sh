#!/usr/bin/env bash
# Undoes what a plain "pip install rtmlib" does to .venv310: it pulls OpenCV 5 and NumPy 2,
# which moved the reference key frames by one frame. Only packages at OpenCV >= 5 or
# NumPy >= 2 are changed (back to the versions the calibration used); nothing else is touched.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv310/bin/python
FIX="$("$PY" - <<'PYEOF'
import importlib.metadata as m
fix = []
def ver(p):
    try:
        return m.version(p)
    except m.PackageNotFoundError:
        return None
if (ver("numpy") or "1").split(".")[0] != "1":
    fix.append("numpy>=1.24.0,<2")
for p in ("opencv-python", "opencv-contrib-python", "opencv-python-headless"):
    v = ver(p)
    if v and int(v.split(".")[0]) >= 5:
        fix.append(f"{p}<5")
print(" ".join(fix))
PYEOF
)"
if [ -n "$FIX" ]; then
  echo "==> Restoring pinned versions in .venv310: $FIX"
  # shellcheck disable=SC2086
  "$PY" -m pip install -q --force-reinstall --no-deps $FIX
fi
