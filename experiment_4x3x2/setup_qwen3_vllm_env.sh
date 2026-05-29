#!/bin/bash
set -euo pipefail

BASE=${BASE:-/home/users/grad/2025/25t9801/projects/LLm-pipelin/experiment_4x3x2}
ENV_DIR=${ENV_DIR:-.venv_vllm_qwen3}
VLLM_SPEC=${VLLM_SPEC:-"vllm>=0.9.0"}

cd "$BASE"

if [[ -d "$ENV_DIR" ]]; then
  echo "Environment already exists: $BASE/$ENV_DIR"
else
  python3 -m venv "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install "$VLLM_SPEC"

python - <<'PY'
import vllm
print(f"vLLM version: {vllm.__version__}")
PY

echo "Qwen3 vLLM environment is ready: $BASE/$ENV_DIR"
