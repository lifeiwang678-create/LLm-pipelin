#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=shard:1
#SBATCH -J llm_nomultiagent
#SBATCH -o /home/users/grad/2025/25t9801/logs/llm_nomultiagent_%j.out
#SBATCH -e /home/users/grad/2025/25t9801/logs/llm_nomultiagent_%j.err
#SBATCH --time=12:00:00

# Runs the 4x?x2 grid over 3 datasets, EXCLUDING the multi_agent (majority-vote)
# LM method. By default LMS="direct few_shot" (the remaining 2 of the 3 methods),
# so the grid is 4 inputs x 2 LMs x 2 outputs x 3 datasets = 48 combos.
#
# Modes (set SUBSET_LEVEL):
#   SUBSET_LEVEL=debug   -> small smoke subset (partial)
#   SUBSET_LEVEL=main    -> full dataset
#
# Everything is configurable via env; to re-enable multi_agent later:
#   LMS="direct few_shot multi_agent" ./run_3datasets_no_multiagent_sbatch.sh

set -uo pipefail

BASE=${BASE:-/workspace/LLm-pipelin/experiment_4x3x2}
SUBSET_LEVEL=${SUBSET_LEVEL:-debug}
LOGROOT=${LOGROOT:-/workspace/logs/llm_nomultiagent_${SUBSET_LEVEL}_$(date +%Y%m%d%H%M%S)}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen2.5-7b-instruct}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
API_URL="http://${HOST}:${PORT}/v1"
API_KEY=${API_KEY:-lm-studio}
CONCURRENCY=${CONCURRENCY:-8}
FEW_SHOT_TRAIN_SUBSET_LEVEL=${FEW_SHOT_TRAIN_SUBSET_LEVEL:-pilot}
FEW_SHOT_EXAMPLE_SUBJECTS=${FEW_SHOT_EXAMPLE_SUBJECTS:-3}
LOG_EVERY=${LOG_EVERY:-1}

# --- experiment grid (all overridable via env) ---
DATASETS=${DATASETS:-"WESAD HHAR DREAMT"}
INPUTS=${INPUTS:-"raw_data feature_description encoded_time_series extra_knowledge"}
LMS=${LMS:-"direct few_shot"}                 # multi_agent intentionally excluded
OUTPUTS=${OUTPUTS:-"label_only label_explanation"}

# --- vLLM / server ---
REUSE_EXISTING_SERVER=${REUSE_EXISTING_SERVER:-0}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-16}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
VLLM_DATA_PARALLEL_SIZE=${VLLM_DATA_PARALLEL_SIZE:-1}
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}

mkdir -p "$LOGROOT"
cd "$BASE"

TOTAL=$(( $(echo $DATASETS | wc -w) * $(echo $INPUTS | wc -w) * $(echo $LMS | wc -w) * $(echo $OUTPUTS | wc -w) ))

echo "Job started at $(date)"
echo "Base: $BASE"
echo "Log dir: $LOGROOT"
echo "API URL: $API_URL"
echo "Client concurrency: $CONCURRENCY"
echo "Evaluation subset (mode): $SUBSET_LEVEL"
echo "Few-shot train subset: $FEW_SHOT_TRAIN_SUBSET_LEVEL"
echo "Datasets: $DATASETS"
echo "Inputs:   $INPUTS"
echo "LMs:      $LMS   (multi_agent excluded by default)"
echo "Outputs:  $OUTPUTS"
echo "Total combinations: $TOTAL"

export MPLCONFIGDIR="${LOGROOT}/mplconfig"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CUDA_COMPAT_DIR=${CUDA_COMPAT_DIR:-/usr/local/cuda-13.0/compat}
if [[ -e "$CUDA_COMPAT_DIR/libcuda.so.1" ]]; then
  export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR:${LD_LIBRARY_PATH:-}"
fi
mkdir -p "$MPLCONFIGDIR"

# pre-check evaluation subset caches for the requested grid
for dataset in $DATASETS; do
  for input in $INPUTS; do
    subset_cache="Processed/LLMSubsets/${dataset}/${SUBSET_LEVEL}/${dataset}_${input}_${SUBSET_LEVEL}_samples.pkl"
    if [[ ! -s "$subset_cache" ]]; then
      echo "Missing fixed evaluation subset cache: $subset_cache"
      echo "Run prepare_data_subsets.py and prepare_subset_inputs.py before this script."
      exit 1
    fi
  done
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$LOGROOT/nvidia_smi_start.txt" || true
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 5 \
    > "$LOGROOT/gpu_usage.csv" 2>/dev/null &
  GPU_MONITOR_PID=$!
fi

source .venv/bin/activate

VLLM_PID=""
GPU_MONITOR_PID="${GPU_MONITOR_PID:-}"

cleanup () {
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if curl -fsS "${API_URL}/models" > /dev/null 2>&1; then
  if [[ "$REUSE_EXISTING_SERVER" == "1" ]]; then
    echo "Reusing existing server on ${API_URL}/models (skipping vLLM startup)."
    READY=1
  else
    echo "A server is already responding on ${API_URL}/models."
    echo "Stop it first, change PORT, or set REUSE_EXISTING_SERVER=1."
    exit 1
  fi
else
  nohup vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$VLLM_DTYPE" \
    --data-parallel-size "$VLLM_DATA_PARALLEL_SIZE" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    > "$LOGROOT/vllm.log" 2>&1 &
  VLLM_PID=$!

  echo "Waiting for vLLM..."
  READY=0
  for _ in $(seq 1 120); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM exited before becoming ready."
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
    echo "vLLM failed to become ready."
    tail -120 "$LOGROOT/vllm.log" || true
    exit 1
  fi
fi

echo "vLLM ready at $(date)"
curl -fsS "${API_URL}/models" > "$LOGROOT/vllm_models.json" || true

source .venv/bin/activate

echo "dataset,input,lm,output,status,start_time,end_time,log" > "$LOGROOT/status.csv"

failures=0
COUNT=0
RUN_START=$(date +%s)

fmt_hms () {
  local s=$1
  printf '%d:%02d:%02d' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

print_progress () {
  local now elapsed avg remaining eta_clock filled bar pct width i
  now=$(date +%s)
  elapsed=$((now - RUN_START))
  pct=$((COUNT * 100 / TOTAL))
  if (( COUNT > 0 )); then
    avg=$((elapsed / COUNT))
    remaining=$((avg * (TOTAL - COUNT)))
  else
    avg=0; remaining=0
  fi
  eta_clock=$(date -d "@$((now + remaining))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "n/a")
  width=30
  filled=$((pct * width / 100))
  bar=""
  for ((i = 0; i < width; i++)); do
    if (( i < filled )); then bar+="#"; else bar+="-"; fi
  done
  echo ">>> PROGRESS [${bar}] ${COUNT}/${TOTAL} (${pct}%) | elapsed $(fmt_hms ${elapsed}) | avg $(fmt_hms ${avg})/task | remaining $(fmt_hms ${remaining}) | ETA ${eta_clock}"
}

run_one () {
  local dataset=$1
  local input=$2
  local lm=$3
  local output=$4
  local start_time
  local end_time
  local log_file
  local eval_cache
  local train_cache
  local extra_args=()
  local status=0

  start_time=$(date +%Y-%m-%dT%H:%M:%S)
  log_file="$LOGROOT/${dataset}_${input}_${lm}_${output}.log"
  eval_cache="Processed/LLMSubsets/${dataset}/${SUBSET_LEVEL}/${dataset}_${input}_${SUBSET_LEVEL}_samples.pkl"
  train_cache="$eval_cache"

  echo "=================================================="
  echo "Running: dataset=$dataset input=$input lm=$lm output=$output"
  echo "Log: $log_file"

  if [[ "$lm" == "few_shot" ]]; then
    train_cache="Processed/LLMSubsets/${dataset}/${FEW_SHOT_TRAIN_SUBSET_LEVEL}/${dataset}_${input}_${FEW_SHOT_TRAIN_SUBSET_LEVEL}_samples.pkl"
    if [[ ! -s "$train_cache" ]]; then
      echo "Missing few-shot train subset cache: $train_cache"
      echo "Refusing to fall back to eval subset cache because that can leak evaluation samples into few-shot examples."
      status=1
      end_time=$(date +%Y-%m-%dT%H:%M:%S)
      failures=$((failures + 1))
      echo "FAILED: dataset=$dataset input=$input lm=$lm output=$output status=$status"
      echo "$dataset,$input,$lm,$output,$status,$start_time,$end_time,$log_file" >> "$LOGROOT/status.csv"
      return
    fi
    extra_args+=(
      --few-shot-example-selection leave_one_subject_out
      --few-shot-example-subjects "$FEW_SHOT_EXAMPLE_SUBJECTS"
      --few-shot-examples-per-subject-per-label 1
      --few-shot-n-per-class 1
      --train-input-cache-file "$train_cache"
    )
    if [[ "$input" == "raw_data" ]]; then
      extra_args+=(--few-shot-example-max-chars 500)
    elif [[ "$input" == "encoded_time_series" ]]; then
      extra_args+=(--few-shot-example-max-chars 300)
    else
      extra_args+=(--few-shot-example-max-chars 800)
    fi
  fi

  if [[ "$lm" == "multi_agent" ]]; then
    extra_args+=(--multi-agent-intermediate-max-tokens 128)
  fi

  set +e
  python main.py \
    -dataset "$dataset" \
    -Input "$input" \
    -LM "$lm" \
    -output "$output" \
    --use-input-cache \
    --input-cache-dir Processed \
    --subject-split all \
    --eval-input-cache-file "$eval_cache" \
    --api-url "$API_URL" \
    --api-key "$API_KEY" \
    -llm "$SERVED_MODEL_NAME" \
    --concurrency "$CONCURRENCY" \
    --log-every "$LOG_EVERY" \
    "${extra_args[@]}" \
    > "$log_file" 2>&1
  status=$?
  set -e

  end_time=$(date +%Y-%m-%dT%H:%M:%S)
  if [[ "$status" -ne 0 ]]; then
    failures=$((failures + 1))
    echo "FAILED: dataset=$dataset input=$input lm=$lm output=$output status=$status"
  else
    echo "OK: dataset=$dataset input=$input lm=$lm output=$output"
  fi
  echo "$dataset,$input,$lm,$output,$status,$start_time,$end_time,$log_file" >> "$LOGROOT/status.csv"
}

for dataset in $DATASETS; do
  for input in $INPUTS; do
    for lm in $LMS; do
      for output in $OUTPUTS; do
        run_one "$dataset" "$input" "$lm" "$output"
        COUNT=$((COUNT + 1))
        print_progress
      done
    done
  done
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$LOGROOT/nvidia_smi_end.txt" || true
fi

echo "All $TOTAL combinations finished at $(date)"
echo "Failures: $failures"
echo "Logs saved in: $LOGROOT"

if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
