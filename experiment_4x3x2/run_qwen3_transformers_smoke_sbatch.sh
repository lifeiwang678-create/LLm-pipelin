#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=shard:1
#SBATCH -J qwen3_tf_smoke
#SBATCH -o /home/users/grad/2025/25t9801/logs/qwen3_tf_smoke_%j.out
#SBATCH -e /home/users/grad/2025/25t9801/logs/qwen3_tf_smoke_%j.err
#SBATCH --time=02:00:00

set -euo pipefail

BASE=${BASE:-/home/users/grad/2025/25t9801/projects/LLm-pipelin/experiment_4x3x2}
LOGROOT=${LOGROOT:-/home/users/grad/2025/25t9801/logs/qwen3_tf_smoke_$(date +%Y%m%d%H%M%S)}
QWEN3_ENV=${QWEN3_ENV:-.venv_vllm_qwen3_cu121}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3-8b}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
API_URL="http://${HOST}:${PORT}/v1"
API_KEY=${API_KEY:-lm-studio}
CHAT_TEMPLATE_KWARGS_JSON=${CHAT_TEMPLATE_KWARGS_JSON:-'{"enable_thinking": false}'}
CONCURRENCY=${CONCURRENCY:-1}
SMOKE_DIRECT_REPETITIONS=${SMOKE_DIRECT_REPETITIONS:-2}
SMOKE_PIPELINE_BALANCED_PER_LABEL=${SMOKE_PIPELINE_BALANCED_PER_LABEL:-1}

mkdir -p "$LOGROOT"
cd "$BASE"

echo "Job started at $(date)"
echo "Base: $BASE"
echo "Log dir: $LOGROOT"
echo "Model path: $MODEL_PATH"
echo "Served model name: $SERVED_MODEL_NAME"
echo "Qwen3 env: $QWEN3_ENV"
echo "API URL: $API_URL"

export MPLCONFIGDIR="${LOGROOT}/mplconfig"
mkdir -p "$MPLCONFIGDIR"

PYTHON="$BASE/$QWEN3_ENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Qwen3 environment: $BASE/$QWEN3_ENV"
  exit 1
fi

"$PYTHON" - <<'PY'
import torch
import transformers

print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}")
print(f"transformers: {transformers.__version__}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$LOGROOT/nvidia_smi_start.txt" || true
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
fi

SERVER_PID=""
cleanup () {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if curl -fsS "${API_URL}/models" > /dev/null 2>&1; then
  echo "A server is already responding on ${API_URL}/models."
  echo "Stop it first or change PORT."
  exit 1
fi

nohup "$PYTHON" serve_qwen3_transformers_openai.py \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --device cuda \
  --max-model-len 8192 \
  > "$LOGROOT/server.log" 2>&1 &
SERVER_PID=$!

echo "Waiting for Qwen3 transformers server..."
READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Qwen3 transformers server exited before becoming ready."
    tail -160 "$LOGROOT/server.log" || true
    exit 1
  fi
  if curl -fsS "${API_URL}/models" > /dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 10
done

if [[ "$READY" -ne 1 ]]; then
  echo "Qwen3 transformers server did not become ready."
  tail -160 "$LOGROOT/server.log" || true
  exit 1
fi

echo "Qwen3 transformers server ready at $(date)"
curl -fsS "${API_URL}/models" > "$LOGROOT/models.json" || true

.venv/bin/python smoke_qwen3_json.py \
  --api-url "$API_URL" \
  --api-key "$API_KEY" \
  --model "$SERVED_MODEL_NAME" \
  --output-dir "$LOGROOT/results" \
  --report-json "$LOGROOT/qwen3_smoke_report.json" \
  --direct-repetitions "$SMOKE_DIRECT_REPETITIONS" \
  --pipeline-balanced-per-label "$SMOKE_PIPELINE_BALANCED_PER_LABEL" \
  --pipeline-concurrency "$CONCURRENCY" \
  --chat-template-kwargs-json "$CHAT_TEMPLATE_KWARGS_JSON" \
  2>&1 | tee "$LOGROOT/smoke.log"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$LOGROOT/nvidia_smi_end.txt" || true
fi

echo "Qwen3 transformers smoke completed successfully at $(date)"
echo "Report: $LOGROOT/qwen3_smoke_report.json"
