#!/usr/bin/env bash
set -euo pipefail
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-30000}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen35-sglang-demo-debug}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
HF_TOKEN="${HF_TOKEN:-}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$HF_CACHE_DIR"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --shm-size 16g \
  -p "$PORT:30000" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -v "$PROJECT_ROOT/assets:/workspace/assets" \
  -e HF_TOKEN="$HF_TOKEN" \
  lmsysorg/sglang:latest-cu130-runtime \
  python3 -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 1 \
    --mem-fraction-static 0.8 \
    --context-length 32768 \
    --attention-backend triton \
    --reasoning-parser qwen3

echo "$CONTAINER_NAME"
