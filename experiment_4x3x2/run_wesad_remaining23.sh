#!/usr/bin/env bash

mkdir -p ~/logs/wesad_24

API_URL="http://127.0.0.1:8000/v1"
MODEL="qwen2.5-7b-instruct"
CONCURRENCY="${CONCURRENCY:-64}"
PYTHON="${PYTHON:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
fi

inputs=("raw_data" "feature_description" "encoded_time_series" "extra_knowledge")
lms=("direct" "few_shot" "multi_agent")
outputs=("label_only" "label_explanation")

cache_file() {
  case "$1" in
    raw_data) echo "Processed/WESAD_raw_data_samples.pkl" ;;
    feature_description) echo "Processed/WESAD_feature_description_samples.pkl" ;;
    encoded_time_series) echo "Processed/WESAD_encoded_time_series_samples.pkl" ;;
    extra_knowledge) echo "Processed/WESAD_extra_knowledge_samples.pkl" ;;
  esac
}

summary=~/logs/wesad_24/summary.csv
echo "input,lm,output,status,log" > "$summary"
echo "Client concurrency: $CONCURRENCY"

for input in "${inputs[@]}"; do
  for lm in "${lms[@]}"; do
    for output in "${outputs[@]}"; do

      if [[ "$input" == "feature_description" && "$lm" == "direct" && "$output" == "label_only" ]]; then
        echo "skip finished: $input $lm $output"
        continue
      fi

      log=~/logs/wesad_24/WESAD_${input}_${lm}_${output}_$(date +%Y%m%d%H%M%S).log

      echo "=================================================="
      echo "Running: Input=$input | LM=$lm | Output=$output"
      echo "Log: $log"

      extra_args=()

      if [[ "$lm" == "few_shot" ]]; then
        extra_args+=(
          --train-subjects S2
          --test-subjects S3 S4 S5 S6 S7 S8 S9 S10 S11 S13 S14 S15 S16 S17
          --few-shot-n-per-class 1
          --few-shot-example-max-chars 800
        )
      fi

      if [[ "$lm" == "multi_agent" ]]; then
        extra_args+=(--multi-agent-intermediate-max-tokens 128)
      fi

      "$PYTHON" main.py \
        -dataset WESAD \
        -Input "$input" \
        -LM "$lm" \
        -output "$output" \
        --use-input-cache \
        --input-cache-file "$(cache_file "$input")" \
        --api-url "$API_URL" \
        -llm "$MODEL" \
        --concurrency "$CONCURRENCY" \
        --log-every 20 \
        "${extra_args[@]}" \
        > "$log" 2>&1

      status=$?
      echo "$input,$lm,$output,$status,$log" >> "$summary"
      echo "Finished: status=$status"

    done
  done
done

echo "All remaining WESAD experiments finished."
echo "Summary: $summary"
