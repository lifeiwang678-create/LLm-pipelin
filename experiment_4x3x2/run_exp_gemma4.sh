#!/bin/bash
# Run the no-multiagent grid against the Gemma-4 vLLM load balancer.
# Results -> Results/gemma-4-26B-A4B-it/
# Prereq: a vLLM server/LB for google/gemma-4-26B-A4B-it must be serving on $PORT,
#         exposing served-model-name "gemma-4-26b-a4b-it" (see run_multi_vllm_lb.sh).
# Usage:  SUBSET_LEVEL=main ./run_exp_gemma4.sh      (debug | main)
exec env \
  PORT="${PORT:-8080}" REUSE_EXISTING_SERVER=1 \
  MODEL_PATH=google/gemma-4-26B-A4B-it \
  SERVED_MODEL_NAME=gemma-4-26b-a4b-it \
  MODEL_TAG=gemma-4-26B-A4B-it \
  "$(dirname "$0")/run_3datasets_no_multiagent_sbatch.sh" "$@"
