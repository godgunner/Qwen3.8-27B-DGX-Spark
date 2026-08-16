#!/usr/bin/env bash
# prewarm.sh — compile Triton kernels for batch sizes 1..N before real traffic.
#
# WHY: This hybrid Gated-DeltaNet + MTP model JIT-compiles a set of bespoke
# Triton kernels (mamba_align, eagle_*, causal_conv1d, delta_rule_update,
# topk_topp) the FIRST time each batch shape is hit. vLLM's own startup
# warmup only compiles the single-request shape, so the first time you run
# 2 or 3 chats concurrently, those shapes compile on the fly and every
# request drops to ~0.1-0.8 tok/s until compilation finishes (then it stays
# fast). A plain dense model reuses one compiled kernel across batch sizes,
# so it never shows this stall — this model does.
#
# This script fires short concurrent requests at 1, 2, 3, ... N in-flight
# once /health is up, forcing each batch shape to compile up front. Run it
# right after launch (or on container start) and the first real concurrent
# turn is fast.
#
# Usage: ./scripts/prewarm.sh [port] [model] [max_concurrency]
#        ./scripts/prewarm.sh 8000 Qwen3.8-27B-NVFP4 4
set -uo pipefail

PORT="${1:-8000}"
MODEL="${2:-Qwen3.8-27B-NVFP4}"
MAX_CONC="${3:-4}"
BASE="http://127.0.0.1:$PORT"

echo "waiting for /health ..."
for i in $(seq 1 300); do
  if curl -fsS "$BASE/health" >/dev/null 2>&1; then
    echo "server healthy."
    break
  fi
  sleep 2
done

if ! curl -fsS "$BASE/health" >/dev/null 2>&1; then
  echo "server not healthy at $BASE after waiting — aborting prewarm." >&2
  exit 1
fi

# Fire MAX_CONC tiny concurrent requests. They all hit the same new batch
# shape at once, so the kernels for that width compile in parallel. Keep
# max_tokens small — the goal is to touch the kernels, not generate output.
echo "prewarming up to $MAX_CONC concurrent requests (non-thinking, short) ..."

req_body() {
  cat <<JSON
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "Reply with the single word: ok."}],
  "max_tokens": 8,
  "temperature": 0.0,
  "chat_template_kwargs": {"enable_thinking": false}
}
JSON
}

for conc in $(seq 1 "$MAX_CONC"); do
  echo "  -> concurrency $conc"
  pids=()
  for _ in $(seq 1 "$conc"); do
    curl -fsS "$BASE/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "$(req_body)" >/dev/null 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
done

echo "prewarm done — first concurrent turn should now be fast."
echo "tip: the compiled kernels persist in ~/.triton; they survive restarts"
echo "     only if that directory is a mounted volume (run-recipe.py mounts it"
echo "     by default; use a bind mount for standalone docker run)."
