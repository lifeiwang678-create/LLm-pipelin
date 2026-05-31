#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=shard:1
#SBATCH -J llm_72_debug
#SBATCH -o /home/users/grad/2025/25t9801/logs/llm_72_debug_%j.out
#SBATCH -e /home/users/grad/2025/25t9801/logs/llm_72_debug_%j.err
#SBATCH --time=08:00:00

set -euo pipefail

BASE=${BASE:-/home/users/grad/2025/25t9801/projects/LLm-pipelin/experiment_4x3x2}
LOGROOT=${LOGROOT:-/home/users/grad/2025/25t9801/logs/llm_72_debug_$(date +%Y%m%d%H%M%S)}
API_URL=${API_URL:-https://api.openai.com/v1}
API_KEY=${API_KEY:-${OPENAI_API_KEY:-}}
LLM_MODEL=${LLM_MODEL:-gpt-5.4-mini}
CONCURRENCY=${CONCURRENCY:-8}
SUBSET_LEVEL=${SUBSET_LEVEL:-debug}
FEW_SHOT_TRAIN_SUBSET_LEVEL=${FEW_SHOT_TRAIN_SUBSET_LEVEL:-pilot}
FEW_SHOT_EXAMPLE_SUBJECTS=${FEW_SHOT_EXAMPLE_SUBJECTS:-3}
LOG_EVERY=${LOG_EVERY:-1}

if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: API_KEY is not set. Export API_KEY or OPENAI_API_KEY before running this script."
  exit 1
fi

mkdir -p "$LOGROOT"
cd "$BASE"

echo "Job started at $(date)"
echo "Base: $BASE"
echo "Log dir: $LOGROOT"
echo "API URL: $API_URL"
echo "LLM model: $LLM_MODEL"
echo "Client concurrency: $CONCURRENCY"
echo "Evaluation subset: $SUBSET_LEVEL"
echo "Few-shot train subset: $FEW_SHOT_TRAIN_SUBSET_LEVEL"
echo "Few-shot example subjects: $FEW_SHOT_EXAMPLE_SUBJECTS"

export MPLCONFIGDIR="${LOGROOT}/mplconfig"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$MPLCONFIGDIR"

for dataset in WESAD HHAR DREAMT; do
  for input in raw_data feature_description encoded_time_series extra_knowledge; do
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

GPU_MONITOR_PID="${GPU_MONITOR_PID:-}"

cleanup () {
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

source .venv/bin/activate

echo "dataset,input,lm,output,status,start_time,end_time,log" > "$LOGROOT/status.csv"

failures=0

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
    -llm "$LLM_MODEL" \
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

for dataset in WESAD HHAR DREAMT; do
  for input in raw_data feature_description encoded_time_series extra_knowledge; do
    for lm in direct few_shot multi_agent; do
      for output in label_only label_explanation; do
        run_one "$dataset" "$input" "$lm" "$output"
      done
    done
  done
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$LOGROOT/nvidia_smi_end.txt" || true
fi

echo "All 72 debug combinations finished at $(date)"
echo "Failures: $failures"
echo "Logs saved in: $LOGROOT"

if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
