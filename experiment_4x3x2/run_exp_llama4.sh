#!/bin/bash
# Run the no-multiagent grid against the Llama-4-Scout vLLM load balancer.
# Results -> Results/Llama-4-Scout-17B-16E/
# Prereq: a vLLM server/LB for the Scout int4 (w4a16) model must be serving on
#         $PORT, exposing served-model-name "llama-4-scout-17b-16e".
# Usage:  SUBSET_LEVEL=main ./run_exp_llama4.sh      (debug | main)
exec env \
  PORT="${PORT:-8080}" REUSE_EXISTING_SERVER=1 \
  MODEL_PATH=RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16 \
  SERVED_MODEL_NAME=llama-4-scout-17b-16e \
  MODEL_TAG=Llama-4-Scout-17B-16E \
  "$(dirname "$0")/run_3datasets_no_multiagent_sbatch.sh" "$@"
