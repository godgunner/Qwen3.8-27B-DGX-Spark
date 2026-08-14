#!/usr/bin/env bash
# Qwen3.8-27B BF16 on vLLM — full-precision reference lane.
# Tested on one NVIDIA DGX Spark (GB10, 128 GB unified memory).
# BF16 is bandwidth-bound: use it for parity checks / quality baselines,
# and prefer NVFP4 or FP8 for daily serving.
set -euo pipefail

IMAGE="${IMAGE:-vllm-node}"
PORT="${PORT:-8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODS="$HERE/../mods"

docker rm -f qwen38-27b 2>/dev/null || true
docker run -d --name qwen38-27b \
  --gpus all --net=host --ipc=host --privileged \
  --ulimit nofile=1048576:1048576 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_VERSION=5.8.0 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$MODS/fix-qwen3.6-chat-template:/workspace/mods/fix-qwen3.6-chat-template:ro" \
  "$IMAGE" \
  vllm serve Qwen/Qwen3.8-27B \
    --served-model-name Qwen3.8-27B \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len 262144 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.55 \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
    --max-num-seqs 4 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --load-format fastsafetensors \
    --attention-backend flashinfer \
    --trust-remote-code \
    --mm-encoder-tp-mode data \
    -tp 1

echo "launched qwen38-27b (BF16) on :$PORT"
echo "  logs:    docker logs -f qwen38-27b"
echo "  verify:  $HERE/verify.sh $PORT Qwen3.8-27B"
