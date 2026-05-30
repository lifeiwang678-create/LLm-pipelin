#!/bin/bash
# Local (non-SLURM) adaptation of run_wesad_remaining5_sbatch.sh
# - single .venv (vllm + project), runs on one local H100
# - reads input caches from Processed/LLMSubsets/WESAD/<SUBSET>/
# Usage:  SUBSET=debug ./run_wesad_remaining5_local.sh
set -uo pipefail

BASE=/workspace/LLm-pipelin/experiment_4x3x2
SUBSET=${SUBSET:-debug}                       # debug(6) | pilot(100) | main(320)
DP=${DP:-8}                                    # data-parallel replicas (use all 8 H100)
GPU_MEM=${GPU_MEM:-0.90}                       # per-GPU memory utilization
MAX_NUM_SEQS=${MAX_NUM_SEQS:-512}             # max batch size per replica
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-32768}
CONCURRENCY=${CONCURRENCY:-512}               # client concurrency (fills DP*batch)
LOGROOT=$BASE/logs/wesad_remaining5_${SUBSET}_$(date +%Y%m%d%H%M%S)

mkdir -p "$LOGROOT"
cd "$BASE"

echo "Job started at $(date)"
echo "Subset: $SUBSET | DP=$DP | max_num_seqs=$MAX_NUM_SEQS | concurrency=$CONCURRENCY"
echo "Log dir: $LOGROOT"

source .venv/bin/activate
# Use ALL local GPUs (do not pin CUDA_VISIBLE_DEVICES) so DP can spread across them.

VLLM_PID=""
cleanup () {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if curl -fsS http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
  echo "A server is already responding on http://127.0.0.1:8000/v1/models."
  echo "Reusing it (will NOT start a new vLLM)."
else
  nohup vllm serve Qwen/Qwen2.5-7B-Instruct \
    --served-model-name qwen2.5-7b-instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --data-parallel-size "$DP" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len 8192 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    > "$LOGROOT/vllm.log" 2>&1 &
  VLLM_PID=$!

  echo "Waiting for vLLM (PID $VLLM_PID)..."
  READY=0
  for i in $(seq 1 120); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM process exited before becoming ready."
      tail -100 "$LOGROOT/vllm.log"
      exit 1
    fi
    if curl -fsS http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
      echo "vLLM is ready at $(date)"
      READY=1
      break
    fi
    sleep 15
  done

  if [ "$READY" -ne 1 ]; then
    echo "vLLM failed to start."
    tail -100 "$LOGROOT/vllm.log"
    exit 1
  fi
fi

run_one () {
  INPUT=$1
  OUTPUT=$2
  CACHE="Processed/LLMSubsets/WESAD/${SUBSET}/WESAD_${INPUT}_${SUBSET}_samples.pkl"

  if [ ! -f "$CACHE" ]; then
    echo "Missing cache: $CACHE"
    return 1
  fi

  LOG="$LOGROOT/WESAD_${INPUT}_multi_agent_${OUTPUT}.log"
  echo "=================================================="
  echo "Running: Input=$INPUT | LM=multi_agent | Output=$OUTPUT"
  echo "Log: $LOG"

  python main.py \
    -dataset WESAD \
    -Input "$INPUT" \
    -LM multi_agent \
    -output "$OUTPUT" \
    --use-input-cache \
    --input-cache-file "$CACHE" \
    --api-url http://127.0.0.1:8000/v1 \
    -llm qwen2.5-7b-instruct \
    --concurrency "$CONCURRENCY" \
    --multi-agent-intermediate-max-tokens 128 \
    --log-every 20 \
    > "$LOG" 2>&1

  STATUS=$?
  echo "Finished: Input=$INPUT | LM=multi_agent | Output=$OUTPUT | status=$STATUS"
  echo "Finished: Input=$INPUT | LM=multi_agent | Output=$OUTPUT | status=$STATUS" >> "$LOGROOT/status.txt"
}

run_one feature_description label_explanation
run_one encoded_time_series label_only
run_one encoded_time_series label_explanation
run_one extra_knowledge label_only
run_one extra_knowledge label_explanation

echo "Remaining 5 experiments finished at $(date)"
echo "Logs saved in: $LOGROOT"
