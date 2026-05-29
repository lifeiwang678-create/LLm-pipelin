#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=shard:1
#SBATCH -J qwen3_smoke
#SBATCH -o /home/users/grad/2025/25t9801/logs/qwen3_smoke_%j.out
#SBATCH -e /home/users/grad/2025/25t9801/logs/qwen3_smoke_%j.err
#SBATCH --time=02:00:00

set -euo pipefail

BASE=${BASE:-/home/users/grad/2025/25t9801/projects/LLm-pipelin/experiment_4x3x2}
LOGROOT=${LOGROOT:-/home/users/grad/2025/25t9801/logs/qwen3_smoke_$(date +%Y%m%d%H%M%S)}
QWEN3_VLLM_ENV=${QWEN3_VLLM_ENV:-.venv_vllm_qwen3_cu121}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3-8b}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
API_URL="http://${HOST}:${PORT}/v1"
API_KEY=${API_KEY:-lm-studio}
CHAT_TEMPLATE_KWARGS_JSON=${CHAT_TEMPLATE_KWARGS_JSON:-'{"enable_thinking": false}'}
CONCURRENCY=${CONCURRENCY:-2}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.25}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-8}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
SMOKE_DIRECT_REPETITIONS=${SMOKE_DIRECT_REPETITIONS:-3}
SMOKE_PIPELINE_BALANCED_PER_LABEL=${SMOKE_PIPELINE_BALANCED_PER_LABEL:-1}

mkdir -p "$LOGROOT"
cd "$BASE"

echo "Job started at $(date)"
echo "Base: $BASE"
echo "Log dir: $LOGROOT"
echo "Model path: $MODEL_PATH"
echo "Served model name: $SERVED_MODEL_NAME"
echo "Qwen3 vLLM env: $QWEN3_VLLM_ENV"
echo "API URL: $API_URL"
echo "Chat template kwargs: $CHAT_TEMPLATE_KWARGS_JSON"

export MPLCONFIGDIR="${LOGROOT}/mplconfig"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -x "$QWEN3_VLLM_ENV/bin/vllm" ]]; then
  echo "Missing Qwen3 vLLM environment: $BASE/$QWEN3_VLLM_ENV"
  echo "Create it first with: ./setup_qwen3_vllm_env.sh"
  exit 1
fi

QWEN3_VLLM_PYTHON="$BASE/$QWEN3_VLLM_ENV/bin/python"
QWEN3_VLLM_BIN="$BASE/$QWEN3_VLLM_ENV/bin/vllm"

"$QWEN3_VLLM_PYTHON" - <<'PY'
from importlib.metadata import version

def as_tuple(text):
    parts = []
    for item in text.split(".")[:3]:
        digits = "".join(ch for ch in item if ch.isdigit())
        parts.append(int(digits or 0))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)

v = version("vllm")
print(f"vLLM version: {v}")
if as_tuple(v) < (0, 8, 5):
    raise SystemExit("Qwen3 smoke requires vLLM >= 0.8.5 for Qwen3 model support.")
PY

SUPPORTS_DEFAULT_CHAT_TEMPLATE_KWARGS=$(
  "$QWEN3_VLLM_PYTHON" - <<'PY'
import contextlib
import io

try:
    with contextlib.redirect_stdout(io.StringIO()):
        from vllm.entrypoints.openai.cli_args import BaseFrontendArgs
except Exception:
    print("0")
else:
    print("1" if "default_chat_template_kwargs" in getattr(BaseFrontendArgs, "__dataclass_fields__", {}) else "0")
PY
)
echo "Supports server default chat_template_kwargs: $SUPPORTS_DEFAULT_CHAT_TEMPLATE_KWARGS"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$LOGROOT/nvidia_smi_start.txt" || true
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
fi

VLLM_PID=""
VLLM_EXTRA_ARGS=()
if [[ "$SUPPORTS_DEFAULT_CHAT_TEMPLATE_KWARGS" == "1" ]]; then
  VLLM_EXTRA_ARGS+=(--default-chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_JSON")
else
  echo "Using request-level chat_template_kwargs from smoke_qwen3_json.py."
fi

cleanup () {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if curl -fsS "${API_URL}/models" > /dev/null 2>&1; then
  echo "A server is already responding on ${API_URL}/models."
  echo "Stop it first or change PORT."
  exit 1
fi

nohup "$QWEN3_VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
  "${VLLM_EXTRA_ARGS[@]}" \
  > "$LOGROOT/vllm.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for Qwen3 vLLM..."
READY=0
for _ in $(seq 1 120); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Qwen3 vLLM exited before becoming ready."
    tail -120 "$LOGROOT/vllm.log" || true
    exit 1
  fi
  if curl -fsS "${API_URL}/models" > /dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 10
done

if [[ "$READY" -ne 1 ]]; then
  echo "Qwen3 vLLM did not become ready."
  tail -120 "$LOGROOT/vllm.log" || true
  exit 1
fi

echo "Qwen3 vLLM ready at $(date)"
curl -fsS "${API_URL}/models" > "$LOGROOT/vllm_models.json" || true

source .venv/bin/activate

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

echo "Qwen3 smoke completed successfully at $(date)"
echo "Report: $LOGROOT/qwen3_smoke_report.json"
