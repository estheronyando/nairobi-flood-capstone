#!/usr/bin/env bash
# Nairobi flood susceptibility - setup and launch (macOS / Linux).
# Finds a Python 3.12+ and hands over to scripts/bootstrap.py.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)' 2>/dev/null; then
        PY="$c"; break
    fi
done

if [ -z "$PY" ]; then
    echo "No Python 3.12 or newer found. The pinned rasterio/pyproj/xgboost/shap/numpy"
    echo "versions publish no wheels for anything older."
    exit 1
fi

echo "Using $PY"
exec "$PY" scripts/bootstrap.py "$@"
