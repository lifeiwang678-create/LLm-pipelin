#!/bin/bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -J wesad_remain5
#SBATCH -o /home/users/grad/2025/25t9801/logs/wesad_remaining5_%j.out
#SBATCH -e /home/users/grad/2025/25t9801/logs/wesad_remaining5_%j.err
#SBATCH --time=12:00:00

BASE=/home/users/grad/2025/25t9801/projects/LLm-pipelin/experiment_4x3x2
LOGROOT=/home/users/grad/2025/25t9801/logs/wesad_remaining5_$(date +%Y%m%d%H%M%S)
CONCURRENCY=${CONCURRENCY:-64}

mkdir -p "$LOGROOT"
cd "$BASE"

echo "Job started at $(date)"
echo "Log dir: $LOGROOT"
echo "Client concurrency: $CONCURRENCY"

source .venv_vllm_cu121/bin/activate

VLLM_PID=""

cleanup () {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if curl -fsS http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
  echo "A server is already responding on http://127.0.0.1:8000/v1/models."
  echo "Stop it first or change the port in this script."
  exit 1
fi

nohup vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5-7b-instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 8192 \
  --max-num-seqs 96 \
  --max-num-batched-tokens 24576 \
  > "$LOGROOT/vllm.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM..."
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
  sleep 30
done

if [ "$READY" -ne 1 ]; then
  echo "vLLM failed to start."
  tail -100 "$LOGROOT/vllm.log"
  exit 1
fi

source .venv/bin/activate

run_one () {
  INPUT=$1
  OUTPUT=$2

  if [ "$INPUT" = "feature_description" ]; then
    CACHE="Processed/WESAD_feature_description_samples.pkl"
  elif [ "$INPUT" = "encoded_time_series" ]; then
    CACHE="Processed/WESAD_encoded_time_series_samples.pkl"
  elif [ "$INPUT" = "extra_knowledge" ]; then
    CACHE="Processed/WESAD_extra_knowledge_samples.pkl"
  else
    echo "Unknown input: $INPUT"
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
