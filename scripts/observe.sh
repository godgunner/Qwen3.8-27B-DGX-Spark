#!/usr/bin/env bash
# observe.sh — sample vLLM /metrics during a test stage and show the deltas.
#
# Usage: ./scripts/observe.sh [port] [interval_s] [duration_s]
#        ./scripts/observe.sh 8000 10 120    # every 10 s for 2 minutes
#
# Shows per interval:
#   - tok/s aggregate        (completion tokens delta / dt, all streams)
#   - running / waiting reqs
#   - KV cache usage %
#   - prefix-cache hit rate  (hits / queries)
#   - MTP acceptance %       (accepted / drafted, cumulative)
#   - system RAM + swap used
#
# Run it while a stage (C1/C2/C3) is steady. Record the steady-state rows.
set -uo pipefail

PORT="${1:-8000}"
INTERVAL="${2:-10}"
DURATION="${3:-120}"
BASE="http://127.0.0.1:$PORT"

metric() { curl -fsS "$BASE/metrics" 2>/dev/null | grep -E "^$1" | head -1 | awk '{print $2}'; }

echo "observing $BASE every ${INTERVAL}s for ${DURATION}s ..."
prev_tok=""
prev_t=""
prev_phit=0; prev_pq=0
end=$(( $(date +%s) + DURATION ))

while [ "$(date +%s)" -lt "$end" ]; do
  t=$(date +%s)
  tok=$(metric '^vllm:generation_tokens_total' | awk '{print $1}')
  # fallback: request success counter if token counter name differs
  [[ -z "$tok" ]] && tok=$(metric '^vllm:requests_success_total' | awk '{print $1}')
  running=$(metric '^vllm:num_requests_running')
  waiting=$(metric '^vllm:num_requests_waiting')
  kv=$(metric '^vllm:kv_cache_usage_perc' | head -1)
  phit=$(metric '^vllm:gpu_prefix_cache_hits_total' | awk '{print $1}')
  pq=$(metric '^vllm:gpu_prefix_cache_queries_total' | awk '{print $1}')
  drafted=$(metric '^vllm:spec_decode_num_drafted_tokens_total' | awk '{print $1}')
  accepted=$(metric '^vllm:spec_decode_num_accepted_tokens_total' | awk '{print $1}')

  line="[$(date +%H:%M:%S)] run=$running wait=$waiting kv=${kv:-?}%"
  if [[ -n "$tok" && -n "$prev_tok" ]]; then
    dt=$(( t - prev_t )); dt=$(( dt > 0 ? dt : 1 ))
    line="$line agg=$(( (tok - prev_tok) / dt )) tok/s"
  fi
  if [[ -n "$phit" && -n "$pq" && "$pq" -gt 0 ]]; then
    line="$line prefix=$(( (phit - prev_phit) * 100 / ( (pq - prev_pq) > 0 ? (pq - prev_pq) : 1 ) ))%"
  fi
  if [[ -n "$accepted" && -n "$drafted" && "$drafted" -gt 0 ]]; then
    line="$line mtp_acc=$(awk -v a="$accepted" -v d="$drafted" 'BEGIN{printf "%.2f", a/d}')"
  fi
  mem=$(free -g | awk '/^Mem:/{m=$3} /^Swap:/{s=$3} END{print "ram="m"G swap="s"G"}')
  line="$line $mem"
  echo "$line"

  prev_tok="$tok"; prev_t="$t"; prev_phit="$phit"; prev_pq="$pq"
  sleep "$INTERVAL"
done

echo "done. Record the steady-state rows into TESTING.md."
