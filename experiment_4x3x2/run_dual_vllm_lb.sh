#!/bin/bash
# Launch TWO tensor-parallel vLLM instances (TP=4 each) across 8 GPUs, with an
# nginx round-robin load balancer in front. Used for FP8 block-quantized MoE
# models that cannot be sharded 8 ways (block_n=128 divisibility), so we run
# 2x TP=4 replicas to use all 8 cards.
#
#   instance A: GPUs 0-3  -> 127.0.0.1:8000
#   instance B: GPUs 4-7  -> 127.0.0.1:8001
#   nginx LB              -> 127.0.0.1:${LB_PORT:-8080}  (round-robin)
set -uo pipefail

BASE=${BASE:-/workspace/LLm-pipelin/experiment_4x3x2}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.6-35B-A3B-FP8}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.6-35b-a3b}
TP=${TP:-4}
GPUS_A=${GPUS_A:-0,1,2,3}
GPUS_B=${GPUS_B:-4,5,6,7}
PORT_A=${PORT_A:-8000}
PORT_B=${PORT_B:-8001}
LB_PORT=${LB_PORT:-8080}
VLLM_DTYPE=${VLLM_DTYPE:-auto}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-512}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}
LOGROOT=${LOGROOT:-/workspace/logs/dual_vllm_$(date +%Y%m%d%H%M%S)}

mkdir -p "$LOGROOT" "$LOGROOT/nginx"
cd "$BASE"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Dual-vLLM launcher starting at $(date)"
echo "Model: $MODEL_PATH | TP=$TP | A=$GPUS_A:$PORT_A | B=$GPUS_B:$PORT_B | LB=$LB_PORT"
echo "Log dir: $LOGROOT"

A_PID=""; B_PID=""; NGINX_RUNNING=0
cleanup () {
  [[ "$NGINX_RUNNING" == "1" ]] && nginx -c "$LOGROOT/nginx.conf" -s quit 2>/dev/null || true
  [[ -n "$A_PID" ]] && kill "$A_PID" 2>/dev/null || true
  [[ -n "$B_PID" ]] && kill "$B_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

serve_one () {
  local gpus=$1 port=$2 tag=$3
  # Distinct VLLM_CACHE_ROOT per instance: two vLLM processes of the same model
  # otherwise share the same torch.compile/AOT cache path (model+rank hash) and
  # deadlock writing it concurrently.
  CUDA_VISIBLE_DEVICES=$gpus VLLM_CACHE_ROOT="$LOGROOT/vllm_cache_${tag}" nohup vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$port" \
    --dtype "$VLLM_DTYPE" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    > "$LOGROOT/vllm_${tag}.log" 2>&1 &
  echo $!
}

wait_ready () {
  local port=$1 pid=$2 tag=$3
  for _ in $(seq 1 120); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "vLLM $tag exited before ready."; tail -80 "$LOGROOT/vllm_${tag}.log" || true; return 1
    fi
    curl -fsS "http://127.0.0.1:${port}/v1/models" -o /dev/null 2>/dev/null && { echo "Instance $tag READY ($port) at $(date)"; return 0; }
    sleep 10
  done
  echo "vLLM $tag failed to become ready."; tail -80 "$LOGROOT/vllm_${tag}.log" || true; return 1
}

# Staggered startup: bring A fully up first, then B. Avoids two concurrent
# torch.compile passes thrashing CPU/mem and any first-run cache races.
echo "Starting instance A (GPUs $GPUS_A, port $PORT_A)..."
A_PID=$(serve_one "$GPUS_A" "$PORT_A" "A")
wait_ready "$PORT_A" "$A_PID" "A" || exit 1
echo "Starting instance B (GPUs $GPUS_B, port $PORT_B)..."
B_PID=$(serve_one "$GPUS_B" "$PORT_B" "B")
wait_ready "$PORT_B" "$B_PID" "B" || exit 1

# nginx round-robin LB
cat > "$LOGROOT/nginx.conf" <<EOF
worker_processes auto;
error_log $LOGROOT/nginx/error.log warn;
pid $LOGROOT/nginx/nginx.pid;
events { worker_connections 8192; }
http {
  access_log off;
  proxy_temp_path $LOGROOT/nginx/proxy_temp;
  client_body_temp_path $LOGROOT/nginx/client_temp;
  upstream vllm_backends {
    server 127.0.0.1:$PORT_A;
    server 127.0.0.1:$PORT_B;
  }
  server {
    listen $LB_PORT;
    client_max_body_size 64m;
    location / {
      proxy_pass http://vllm_backends;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_read_timeout 600s;
      proxy_send_timeout 600s;
      proxy_buffering off;
    }
  }
}
EOF

nginx -t -c "$LOGROOT/nginx.conf" || { echo "nginx config test failed"; exit 1; }
nginx -c "$LOGROOT/nginx.conf"
NGINX_RUNNING=1
sleep 2
if curl -fsS "http://127.0.0.1:${LB_PORT}/v1/models" -o /dev/null 2>/dev/null; then
  echo "Load balancer READY at http://127.0.0.1:${LB_PORT}/v1 (round-robin over $PORT_A,$PORT_B)"
else
  echo "LB not responding"; exit 1
fi

echo "ALL READY at $(date). Keeping instances alive (Ctrl-C / screen quit to stop)."
# keep the two vLLM processes in the foreground of this session
wait "$A_PID" "$B_PID"
