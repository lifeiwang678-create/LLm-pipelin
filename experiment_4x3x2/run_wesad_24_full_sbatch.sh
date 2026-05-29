#!/bin/bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -J wesad_24_full
#SBATCH -o /home/users/grad/2025/25t9801/logs/wesad_24_full_%j.out
#SBATCH -e /home/users/grad/2025/25t9801/logs/wesad_24_full_%j.err
#SBATCH --time=12:00:00

BASE=/home/users/grad/2025/25t9801/projects/LLm-pipelin/experiment_4x3x2
LOGROOT=/home/users/grad/2025/25t9801/logs/wesad_24_full_$(date +%Y%m%d%H%M%S)
CONCURRENCY=${CONCURRENCY:-64}
SUBSET_LEVEL=${SUBSET_LEVEL:-main}

mkdir -p "$LOGROOT"
cd "$BASE"

echo "Job started at $(date)"
echo "Log dir: $LOGROOT"
echo "Client concurrency: $CONCURRENCY"
echo "Evaluation subset: $SUBSET_LEVEL"

GPU_MONITOR_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$LOGROOT/nvidia_smi_start.txt" || true
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 5 \
    > "$LOGROOT/gpu_usage.csv" 2>/dev/null &
  GPU_MONITOR_PID=$!
fi

source .venv_vllm_cu121/bin/activate

VLLM_PID=""

cleanup () {
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
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
  --disable-log-requests \
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

curl -fsS http://127.0.0.1:8000/v1/models > "$LOGROOT/vllm_models.json" || true

source .venv/bin/activate

run_one () {
  INPUT=$1
  LM=$2
  OUTPUT=$3

  if [ "$INPUT" = "raw_data" ]; then
    CACHE="Processed/LLMSubsets/WESAD/${SUBSET_LEVEL}/WESAD_raw_data_${SUBSET_LEVEL}_samples.pkl"
  elif [ "$INPUT" = "feature_description" ]; then
    CACHE="Processed/LLMSubsets/WESAD/${SUBSET_LEVEL}/WESAD_feature_description_${SUBSET_LEVEL}_samples.pkl"
  elif [ "$INPUT" = "encoded_time_series" ]; then
    CACHE="Processed/LLMSubsets/WESAD/${SUBSET_LEVEL}/WESAD_encoded_time_series_${SUBSET_LEVEL}_samples.pkl"
  elif [ "$INPUT" = "extra_knowledge" ]; then
    CACHE="Processed/LLMSubsets/WESAD/${SUBSET_LEVEL}/WESAD_extra_knowledge_${SUBSET_LEVEL}_samples.pkl"
  else
    echo "Unknown input: $INPUT"
    return 1
  fi
  if [ ! -s "$CACHE" ]; then
    echo "Missing fixed evaluation subset cache: $CACHE"
    echo "Run prepare_data_subsets.py and prepare_subset_inputs.py first."
    return 1
  fi

  LOG="$LOGROOT/WESAD_${INPUT}_${LM}_${OUTPUT}.log"

  echo "=================================================="
  echo "Running: Input=$INPUT | LM=$LM | Output=$OUTPUT"
  echo "Log: $LOG"

  EXTRA_ARGS=""

  if [ "$LM" = "few_shot" ]; then
    if [ "$INPUT" = "raw_data" ]; then
      EXTRA_ARGS="--few-shot-example-selection leave_one_subject_out --few-shot-example-subjects 5 --few-shot-examples-per-subject-per-label 1 --few-shot-n-per-class 1 --few-shot-example-max-chars 500"
    elif [ "$INPUT" = "encoded_time_series" ]; then
      EXTRA_ARGS="--few-shot-example-selection leave_one_subject_out --few-shot-example-subjects 5 --few-shot-examples-per-subject-per-label 1 --few-shot-n-per-class 1 --few-shot-example-max-chars 300"
    else
      EXTRA_ARGS="--few-shot-example-selection leave_one_subject_out --few-shot-example-subjects 5 --few-shot-examples-per-subject-per-label 1 --few-shot-n-per-class 1 --few-shot-example-max-chars 800"
    fi
    EXTRA_ARGS="$EXTRA_ARGS --train-input-cache-file $CACHE"
  fi

  if [ "$LM" = "multi_agent" ]; then
    EXTRA_ARGS="--multi-agent-intermediate-max-tokens 128"
  fi

  python main.py \
    -dataset WESAD \
    -Input "$INPUT" \
    -LM "$LM" \
    -output "$OUTPUT" \
    --use-input-cache \
    --input-cache-dir Processed \
    --subject-split all \
    --eval-input-cache-file "$CACHE" \
    --api-url http://127.0.0.1:8000/v1 \
    -llm qwen2.5-7b-instruct \
    --concurrency "$CONCURRENCY" \
    --log-every 20 \
    $EXTRA_ARGS \
    > "$LOG" 2>&1

  STATUS=$?
  echo "Finished: Input=$INPUT | LM=$LM | Output=$OUTPUT | status=$STATUS"
  echo "Finished: Input=$INPUT | LM=$LM | Output=$OUTPUT | status=$STATUS" >> "$LOGROOT/status.txt"
}

for LM in direct few_shot multi_agent; do
  for INPUT in raw_data feature_description encoded_time_series extra_knowledge; do
    for OUTPUT in label_only label_explanation; do
      run_one "$INPUT" "$LM" "$OUTPUT"
    done
  done
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$LOGROOT/nvidia_smi_end.txt" || true
fi

echo "All experiments finished at $(date)"
echo "Logs saved in: $LOGROOT"
