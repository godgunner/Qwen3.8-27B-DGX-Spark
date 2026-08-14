#!/usr/bin/env bash
# verify.sh — don't trust the launch, verify what you actually got.
# Checks: HTTP health, served model name, a timed non-thinking probe (tok/s),
# and the server's own MTP acceptance metrics.
# Usage: ./scripts/verify.sh [port] [model]
set -uo pipefail

PORT="${1:-8000}"
MODEL="${2:-Qwen3.8-27B}"
BASE="http://127.0.0.1:$PORT"

echo "== 1. health =="
if curl -fsS "$BASE/health" >/dev/null 2>&1; then
  echo "OK: /health returned 200"
else
  echo "FAIL: server is not healthy at $BASE"
  exit 1
fi

echo
echo "== 2. served model name =="
curl -fsS "$BASE/v1/models" | python3 -c "import sys,json; print([m['id'] for m in json.load(sys.stdin)['data']])" \
  || echo "WARN: could not read /v1/models"

echo
echo "== 3. timed probe (non-thinking, 400 max tokens) =="
START=$(date +%s.%N)
RESP=$(curl -fsS "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Explain speculative decoding in five short sentences.\"}],
    \"max_tokens\": 400,
    \"temperature\": 0.7,
    \"top_p\": 0.8,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }") || { echo "FAIL: /v1/chat/completions errored"; exit 1; }
END=$(date +%s.%N)

python3 - "$RESP" "$START" "$END" <<'PY'
import json, sys
resp, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
r = json.loads(resp)
usage = r.get("usage", {})
gen = usage.get("completion_tokens", 0)
dt = max(end - start, 1e-9)
print(f"  completion_tokens = {gen}")
print(f"  end-to-end        = {dt:.2f}s  ->  {gen/dt:.2f} tok/s (includes HTTP + prefill)")
rc = (r.get("choices") or [{}])[0].get("message", {}).get("reasoning_content")
print(f"  reasoning_content = {'<absent/empty>' if not rc else '<present, ' + str(len(rc)) + ' chars>'}")
PY

echo
echo "== 4. MTP acceptance (server-side, since start) =="
METRICS=$(curl -fsS "$BASE/metrics" 2>/dev/null | grep -E '^vllm:spec_decode_num_(drafted|accepted)_tokens_total')
if [[ -z "$METRICS" ]]; then
  echo "  NOT PRESENT — speculative decoding metrics are missing."
  echo "  If you launched with --speculative-config mtp, the spec config did not take."
else
  echo "$METRICS"
  DRAFTED=$(echo "$METRICS" | grep draft | awk '{print $2}')
  ACCEPTED=$(echo "$METRICS" | grep accepted | awk '{print $2}')
  if python3 - "$DRAFTED" "$ACCEPTED" <<'PY'
import sys
d, a = float(sys.argv[1]), float(sys.argv[2])
if d > 0:
    print(f"  cumulative draft acceptance = {a/d*100:.1f}%  (accepted {a:.0f} / drafted {d:.0f})")
else:
    print("  no draft tokens yet")
PY
  then :; fi
fi

echo
echo "done — compare this against the launch you asked for."
