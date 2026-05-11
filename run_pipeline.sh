#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_CMD=""

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

if [ ! -d "$VENV_DIR" ]; then
  echo "[pipeline] creating virtualenv at $VENV_DIR"
  $PYTHON_CMD -m venv "$VENV_DIR"
fi

echo "[pipeline] activating venv"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[pipeline] upgrading pip and installing requirements"
pip install --upgrade pip
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  pip install -r "$SCRIPT_DIR/requirements.txt"
fi

echo "[pipeline] setting numeric thread limits"
export LOKY_MAX_CPU_COUNT=${LOKY_MAX_CPU_COUNT:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

echo "[pipeline] running weather fetch (src/weather_api.py)"
python "$SCRIPT_DIR/src/weather_api.py"

echo "[pipeline] preparing model input (src/data_prep.py)"
python "$SCRIPT_DIR/src/data_prep.py"

echo "[pipeline] running training pipeline (train.py)"
python "$SCRIPT_DIR/train.py"

echo "[pipeline] optional plotting (plot_model_comparison.py)"
if [ -f "$SCRIPT_DIR/plot_model_comparison.py" ]; then
  python "$SCRIPT_DIR/plot_model_comparison.py" || true
fi

echo "[pipeline] done. Artifacts saved under: artifacts/ and output/"
ls -la "$SCRIPT_DIR/artifacts" || true
ls -la "$SCRIPT_DIR/output" || true

echo "[pipeline] To re-run without explainability (faster), set SKIP_EXPLAIN=1 and run again."
