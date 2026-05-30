#!/bin/bash
# Run the no-multiagent grid against the Qwen3.6 vLLM load balancer.
# Results -> Results/Qwen3.6-35B-A3B/
# Usage:  SUBSET_LEVEL=main ./run_exp_qwen36.sh      (debug | main)
exec env \
  PORT="${PORT:-8080}" REUSE_EXISTING_SERVER=1 \
  MODEL_PATH=Qwen/Qwen3.6-35B-A3B-FP8 \
  SERVED_MODEL_NAME=qwen3.6-35b-a3b \
  MODEL_TAG=Qwen3.6-35B-A3B \
  "$(dirname "$0")/run_3datasets_no_multiagent_sbatch.sh" "$@"
