#!/usr/bin/env bash
# Qwen3.8-27B NVFP4 (unsloth, W4A4) on vLLM — daily driver, best speed/byte.
# Tested on one NVIDIA DGX Spark (GB10, 128 GB unified memory):
# currently running with ~123 GB system memory in use, no swap, util 0.55.
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
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.triton:/root/.triton" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$MODS/fix-qwen3.6-chat-template:/workspace/mods/fix-qwen3.6-chat-template:ro" \
  "$IMAGE" \
  vllm serve unsloth/Qwen3.8-27B-NVFP4 \
    --served-model-name Qwen3.8-27B-NVFP4 \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len 262144 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.75 \
    --quantization compressed-tensors \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --max-num-seqs 4 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --attention-backend triton_attn \
    --trust-remote-code \
    --mm-encoder-tp-mode data \
    -tp 1

# Auto-prewarm (backgrounded): waits for /health, then fires 1,2,3,4
# concurrent tiny requests so the per-batch-shape Triton kernels compile
# BEFORE the first real turn (fixes the first-request 1-3 min JIT stall).
( "$HERE/prewarm.sh" "$PORT" Qwen3.8-27B-NVFP4 4 ) >> /tmp/prewarm_qwen38-27b.log 2>&1 &

echo "launched qwen38-27b (NVFP4) on :$PORT"
echo "  logs:    docker logs -f qwen38-27b"
echo "  prewarm: running in background — see /tmp/prewarm_qwen38-27b.log"
echo "  verify:  $HERE/verify.sh $PORT Qwen3.8-27B-NVFP4"
