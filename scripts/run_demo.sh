#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-30000}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen35-sglang-demo}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
HF_TOKEN="${HF_TOKEN:-}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$HF_CACHE_DIR"
python3 "$PROJECT_ROOT/scripts/generate_test_image.py"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d --rm \
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
    --reasoning-parser qwen3

echo "Started container: $CONTAINER_NAME"
echo "Waiting for server readiness..."
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Server is ready"
    break
  fi
  sleep 5
  if [[ "$i" == "180" ]]; then
    echo "Server did not become ready in time" >&2
    docker logs "$CONTAINER_NAME" || true
    exit 1
  fi
done

python3 "$PROJECT_ROOT/scripts/text_request.py" --api-base "http://127.0.0.1:${PORT}/v1" --model "$MODEL_NAME" | tee "$PROJECT_ROOT/outputs/text_response.json"
python3 "$PROJECT_ROOT/scripts/vision_request.py" --api-base "http://127.0.0.1:${PORT}/v1" --model "$MODEL_NAME" --image "$PROJECT_ROOT/assets/demo_image.png" | tee "$PROJECT_ROOT/outputs/vision_response.json"

echo "Demo finished successfully. Outputs saved under $PROJECT_ROOT/outputs"
