#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/venv" ]; then
  VENV_DIR="$SCRIPT_DIR/venv"
else
  VENV_DIR="$SCRIPT_DIR/.venv"
fi
PYTHON_CMD=""

ensure_macos_lightgbm_runtime() {
  if [ "$(uname -s)" != "Darwin" ]; then
    return
  fi

  if [ ! -f "$SCRIPT_DIR/requirements.txt" ] || ! grep -Eq '^lightgbm([<=>[:space:]]|$)' "$SCRIPT_DIR/requirements.txt"; then
    return
  fi

  if [ -f "/opt/homebrew/opt/libomp/lib/libomp.dylib" ] || \
     [ -f "/usr/local/opt/libomp/lib/libomp.dylib" ] || \
     [ -f "/opt/local/lib/libomp/libomp.dylib" ]; then
    return
  fi

  echo "[pipeline] macOS detected: LightGBM needs the OpenMP runtime (libomp)."
  if command -v brew >/dev/null 2>&1; then
    echo "[pipeline] installing libomp with Homebrew"
    brew install libomp
  else
    echo "LightGBM cannot start because libomp.dylib is missing." >&2
    echo "Install Homebrew, then run: brew install libomp" >&2
    echo "Or remove 'lightgbm' from config/model_config.yaml enabled_models and stacking estimators." >&2
    exit 1
  fi
}

echo "[pipeline] working dir: $SCRIPT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=python
else
  echo "Python not found. Install Python 3.8+ and retry." >&2
  exit 1
fi

echo "[pipeline] using: $($PYTHON_CMD --version 2>&1)"

ensure_macos_lightgbm_runtime

CREATED_VENV=0
if [ ! -d "$VENV_DIR" ]; then
  echo "[pipeline] creating virtualenv at $VENV_DIR"
  $PYTHON_CMD -m venv "$VENV_DIR"
  CREATED_VENV=1
fi

echo "[pipeline] activating venv"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

if [ "$CREATED_VENV" -eq 1 ]; then
  echo "[pipeline] installing requirements"
  pip install --upgrade pip
  if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    pip install -r "$SCRIPT_DIR/requirements.txt"
  fi
else
  echo "[pipeline] using existing virtualenv; dependency install skipped"
fi

echo "[pipeline] setting numeric thread limits"
export LOKY_MAX_CPU_COUNT=${LOKY_MAX_CPU_COUNT:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export KMA_FETCH_AWS=${KMA_FETCH_AWS:-0}

if [ "$#" -eq 0 ]; then
  set -- --fetch all
fi

echo "[pipeline] running unified pipeline"
python "$SCRIPT_DIR/src/pipeline.py" "$@"

echo "[pipeline] done. Artifacts saved under: artifacts/ and output/"
ls -la "$SCRIPT_DIR/artifacts" || true
ls -la "$SCRIPT_DIR/output" || true

echo "[pipeline] To fetch fresh source data, pass: --fetch water, --fetch weather, or --fetch all."
