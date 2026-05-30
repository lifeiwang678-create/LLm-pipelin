#!/bin/bash
# Launch N single-GPU vLLM instances (TP=1, one per GPU) behind an nginx
# round-robin load balancer. Best layout for a model that fits on one GPU:
# pure replicas -> max throughput, no TP/EP sharding headaches.
#
#   instance i: GPU i -> 127.0.0.1:$((BASE_PORT+i))
#   nginx LB           -> 127.0.0.1:${LB_PORT:-8080}  (round-robin over all)
#
# Notes for Qwen3.6-35B-A3B-FP8 on this box:
#   - VLLM_USE_DEEP_GEMM=0 : the flashinfer DeepGEMM FP8 cubin fails to load
#     here and crashes at inference; fall back to Triton FP8 kernels.
#   - --enforce-eager      : skip CUDA-graph capture (also disables torch.compile,
#     so parallel instance startup won't race on the compile cache).
set -uo pipefail

BASE=${BASE:-/workspace/LLm-pipelin/experiment_4x3x2}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.6-35B-A3B-FP8}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.6-35b-a3b}
NUM_GPUS=${NUM_GPUS:-8}
BASE_PORT=${BASE_PORT:-8000}
LB_PORT=${LB_PORT:-8080}
VLLM_DTYPE=${VLLM_DTYPE:-auto}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-256}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
ENFORCE_EAGER=${ENFORCE_EAGER:-1}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
LOGROOT=${LOGROOT:-/workspace/logs/multi_vllm_$(date +%Y%m%d%H%M%S)}

mkdir -p "$LOGROOT" "$LOGROOT/nginx"
cd "$BASE"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

eager_flag=""
[[ "$ENFORCE_EAGER" == "1" ]] && eager_flag="--enforce-eager"

echo "Multi-vLLM launcher at $(date)"
echo "Model: $MODEL_PATH | instances: $NUM_GPUS | ports: $BASE_PORT..$((BASE_PORT+NUM_GPUS-1)) | LB: $LB_PORT"
echo "enforce_eager=$ENFORCE_EAGER VLLM_USE_DEEP_GEMM=$VLLM_USE_DEEP_GEMM max_num_seqs=$VLLM_MAX_NUM_SEQS"
echo "Log dir: $LOGROOT"

PIDS=()
NGINX_RUNNING=0
cleanup () {
  [[ "$NGINX_RUNNING" == "1" ]] && nginx -c "$LOGROOT/nginx.conf" -s quit 2>/dev/null || true
  for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

# launch all instances in parallel (eager mode -> no compile-cache race)
for ((i=0; i<NUM_GPUS; i++)); do
  port=$((BASE_PORT + i))
  CUDA_VISIBLE_DEVICES=$i VLLM_CACHE_ROOT="$LOGROOT/cache_$i" nohup vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 --port "$port" \
    --dtype "$VLLM_DTYPE" --tensor-parallel-size 1 $eager_flag \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    > "$LOGROOT/vllm_$i.log" 2>&1 &
  PIDS+=($!)
  echo "instance $i -> GPU $i, port $port, pid ${PIDS[$i]}"
done

# wait for all to become ready
all_ready=1
for ((i=0; i<NUM_GPUS; i++)); do
  port=$((BASE_PORT + i))
  ready=0
  for _ in $(seq 1 120); do
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      echo "instance $i (port $port) exited before ready."; tail -40 "$LOGROOT/vllm_$i.log" || true; break
    fi
    curl -fsS "http://127.0.0.1:${port}/v1/models" -o /dev/null 2>/dev/null && { echo "instance $i READY ($port) at $(date +%T)"; ready=1; break; }
    sleep 10
  done
  [[ "$ready" != "1" ]] && { echo "instance $i FAILED to become ready"; all_ready=0; }
done
[[ "$all_ready" != "1" ]] && { echo "Not all instances came up; aborting."; exit 1; }

# nginx round-robin LB over all instances.
# Temp dirs go under /tmp (local fs): nginx chowns them and the /workspace
# network volume forbids chown. 'user root;' avoids chown-to-nobody failures.
NGX_TMP="/tmp/nginx_lb_${LB_PORT}"
mkdir -p "$NGX_TMP/proxy_temp" "$NGX_TMP/client_temp"
{
  echo "user root;"
  echo "worker_processes auto;"
  echo "error_log $LOGROOT/nginx/error.log warn;"
  echo "pid $LOGROOT/nginx/nginx.pid;"
  echo "events { worker_connections 8192; }"
  echo "http {"
  echo "  access_log off;"
  echo "  proxy_temp_path $NGX_TMP/proxy_temp;"
  echo "  client_body_temp_path $NGX_TMP/client_temp;"
  echo "  upstream vllm_backends {"
  for ((i=0; i<NUM_GPUS; i++)); do echo "    server 127.0.0.1:$((BASE_PORT+i));"; done
  echo "  }"
  echo "  server {"
  echo "    listen $LB_PORT;"
  echo "    client_max_body_size 64m;"
  echo "    location / {"
  echo "      proxy_pass http://vllm_backends;"
  echo "      proxy_http_version 1.1;"
  echo "      proxy_set_header Connection \"\";"
  echo "      proxy_read_timeout 600s;"
  echo "      proxy_send_timeout 600s;"
  echo "      proxy_buffering off;"
  echo "    }"
  echo "  }"
  echo "}"
} > "$LOGROOT/nginx.conf"

nginx -t -c "$LOGROOT/nginx.conf" || { echo "nginx config test failed"; exit 1; }
nginx -c "$LOGROOT/nginx.conf"
NGINX_RUNNING=1
sleep 2
if curl -fsS "http://127.0.0.1:${LB_PORT}/v1/models" -o /dev/null 2>/dev/null; then
  echo "LB READY at http://127.0.0.1:${LB_PORT}/v1 (round-robin over $NUM_GPUS instances) at $(date)"
else
  echo "LB not responding"; exit 1
fi

echo "ALL READY at $(date). Keeping $NUM_GPUS instances alive."
wait
