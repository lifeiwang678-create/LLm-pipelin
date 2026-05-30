#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=shard:1
#SBATCH -J vllm_shard_bench
#SBATCH -o /home/users/grad/2025/25t9801/logs/vllm_shard_bench_%j.out
#SBATCH -e /home/users/grad/2025/25t9801/logs/vllm_shard_bench_%j.err
#SBATCH --time=04:00:00

set -euo pipefail

BASE=${BASE:-/home/users/grad/2025/25t9801/projects/LLm-pipelin/experiment_4x3x2}
LOGROOT=${LOGROOT:-/home/users/grad/2025/25t9801/logs/vllm_shard_bench_$(date +%Y%m%d%H%M%S)}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen2.5-7b-instruct}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
API_URL="http://${HOST}:${PORT}/v1"
API_KEY=${API_KEY:-lm-studio}
CONCURRENCY_LEVELS=${CONCURRENCY_LEVELS:-"1 4 8 16 32 64 96"}
REQUESTS_PER_LEVEL=${REQUESTS_PER_LEVEL:-64}
PROMPT_CHARS=${PROMPT_CHARS:-2500}
MAX_TOKENS=${MAX_TOKENS:-64}

mkdir -p "$LOGROOT"
cd "$BASE"

echo "Job started at $(date)"
echo "Base: $BASE"
echo "Log dir: $LOGROOT"
echo "Model: $MODEL_PATH"
echo "Served model name: $SERVED_MODEL_NAME"
echo "API URL: $API_URL"
echo "Concurrency levels: $CONCURRENCY_LEVELS"
echo "Requests per level: $REQUESTS_PER_LEVEL"
echo "Prompt chars: $PROMPT_CHARS"
echo "Max tokens: $MAX_TOKENS"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$LOGROOT/nvidia_smi_start.txt" || true
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
else
  echo "nvidia-smi not found in job environment."
fi

source .venv_vllm_cu121/bin/activate

VLLM_PID=""

cleanup () {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_vllm () {
  local log_file=$1
  local ready=0
  for _ in $(seq 1 120); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM exited before becoming ready."
      tail -100 "$log_file" || true
      return 1
    fi
    if curl -fsS "${API_URL}/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 10
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "vLLM did not become ready."
    tail -100 "$log_file" || true
    return 1
  fi
}

run_config () {
  local name=$1
  local gpu_util=$2
  local max_num_seqs=$3
  local max_num_batched_tokens=$4
  local max_model_len=$5

  local config_dir="$LOGROOT/$name"
  local server_log="$config_dir/vllm.log"
  mkdir -p "$config_dir"

  echo "=================================================="
  echo "Config: $name"
  echo "gpu_memory_utilization=$gpu_util max_num_seqs=$max_num_seqs max_num_batched_tokens=$max_num_batched_tokens max_model_len=$max_model_len"

  if curl -fsS "${API_URL}/models" >/dev/null 2>&1; then
    echo "A server is already responding on ${API_URL}/models."
    echo "Stop it first or change PORT."
    return 1
  fi

  nohup vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype bfloat16 \
    --gpu-memory-utilization "$gpu_util" \
    --max-model-len "$max_model_len" \
    --max-num-seqs "$max_num_seqs" \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    > "$server_log" 2>&1 &
  VLLM_PID=$!

  if ! wait_for_vllm "$server_log"; then
    echo "$name,START_FAILED" >> "$LOGROOT/status.csv"
    cleanup
    VLLM_PID=""
    sleep 10
    return 0
  fi

  curl -fsS "${API_URL}/models" > "$config_dir/models.json" || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$config_dir/nvidia_smi_ready.txt" || true
  fi

  .venv/bin/python benchmark_vllm_batch.py \
    --api-url "$API_URL" \
    --api-key "$API_KEY" \
    --model "$SERVED_MODEL_NAME" \
    --concurrency $CONCURRENCY_LEVELS \
    --requests-per-level "$REQUESTS_PER_LEVEL" \
    --prompt-chars "$PROMPT_CHARS" \
    --max-tokens "$MAX_TOKENS" \
    --output-jsonl "$config_dir/results.jsonl" \
    --output-csv "$config_dir/summary.csv" \
    2>&1 | tee "$config_dir/client.log"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$config_dir/nvidia_smi_end.txt" || true
  fi

  echo "$name,OK" >> "$LOGROOT/status.csv"
  cleanup
  VLLM_PID=""
  sleep 20
}

echo "config,status" > "$LOGROOT/status.csv"

run_config shard_util040_seqs032_tokens08192 0.40 32 8192 8192
run_config shard_util045_seqs064_tokens16384 0.45 64 16384 8192
run_config shard_util050_seqs096_tokens24576 0.50 96 24576 8192

echo "Benchmark finished at $(date)"
echo "Results saved in: $LOGROOT"
