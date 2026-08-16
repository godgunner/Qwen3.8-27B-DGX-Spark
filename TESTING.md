# Testing strategy & tracker — Qwen3.8-27B on one DGX Spark

Goal: measure per-stream and aggregate throughput at **1, 2, and 3
concurrency for each quant (NVFP4, FP8, BF16)**, using fixed real-world
workloads, with every result recorded in the tracker below. One variable per
run; record the vLLM version and launch config with every row.

## Controlled workloads (fixed across all lanes)

| ID | Chat | Prompt (keep byte-identical across quants) | Thinking |
|---|---|---|---|
| W1 | Dev work | the real dev task being run right now (paste its first user turn here once pinned: `PENDING`) | default (on) |
| W2 | LLM benchmark | the fixed benchmark prompt (pin it here: `PENDING`) | default (on) |
| W3 | Pelican SVG | "Create an SVG of a pelican riding a bicycle" (or the exact prompt in use) | off (`chat_template_kwargs: {enable_thinking: false}`) |

Rules:
- Same prompt, same sampling, same client for every lane — only the quant/config changes.
- W3 is non-thinking on purpose: it isolates pure decode (no thinking-burst noise).
- Let each stage settle ~2 minutes before sampling numbers (drop the first
  30 s of each stage — it can still contain prefill/JIT transients).

## Procedure per lane (one quant at a time, fresh start)

```bash
# 0. Record environment
docker logs qwen38-27b 2>&1 | head -5   # note vLLM version line
docker logs qwen38-27b 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency' | head -2

# 1. Verify the launch actually came up right
./scripts/verify.sh 8000 Qwen3.8-27B-NVFP4      # (or -FP8 / no suffix for BF16)

# 2. PREWARM — mandatory before C2/C3 (compiles per-batch-shape Triton kernels)
./scripts/prewarm.sh 8000 Qwen3.8-27B-NVFP4 4

# 3. Start the observer (logs aggregate metrics every 10 s)
./scripts/observe.sh 8000 10 20 &   # 10 s interval, 20 s duration...
# (re-run per stage, or use: while true; do ...; done variant; see script)

# 4. C1: run W1 alone. Steady-state 2 min. Record.
# 5. C2: add W2 (keep W1 alive). Steady-state 2 min. Record.
# 6. C3: add W3 (keep both alive). Steady-state 2 min. Record.
```

After each stage, capture:

```bash
# per-request tok/s from the server log
docker logs --since 30m qwen38-27b 2>&1 | grep -E 'Avg generation throughput|Running: ' | tail -20

# MTP acceptance delta since start (server-side truth)
curl -s 127.0.0.1:8000/metrics | grep -E '^vllm:spec_decode_num_(drafted|accepted)_tokens_total'

# memory — confirm no swap
free -g
```

## Metrics to record per row

| Metric | Source |
|---|---|
| Per-stream tok/s (steady state) | server log `Avg generation throughput` lines |
| Aggregate tok/s | sum of concurrent per-stream rates, or `observe.sh` counter deltas |
| Mean MTP acceptance | Δaccepted/Δdrafted from `/metrics` (per stage) |
| KV cache usage % | `observe.sh` (`vllm:kv_cache_usage_perc`) |
| Prefix-cache hit rate | `observe.sh` (prefix_cache counters) |
| System RAM / swap | `free -g` after the run |
| vLLM version + backend + util + K | startup log (record once per lane) |

## Concurrency interpretation guardrails

- **JIT stall ≠ steady state.** The first 2-wide/3-wide step compiles
  Triton kernels per batch shape (0.1–0.8 tok/s spike). That is what
  `prewarm.sh` removes. If it happens, the row is invalid — re-run.
- **Per-stream SHOULD drop as concurrency rises** on a shared
  memory-bandwidth wall; **aggregate is the metric that should scale**
  (~1.7–2.5× from C1→C3 expected). Do not chase per-stream numbers with
  `max_num_seqs`.
- MTP's per-stream lift shrinks as concurrency rises (draft forwards stop
  overlapping idle GPU time). Expected, not a bug.
- One variable per run. Don't change K, util, and backend in the same row.

## Tracker

Legend: ✅ tested · ⏳ pending · ❌ invalid run (JIT stall / config changed mid-run) · n/a

Baseline config = K5 (BF16/FP8) / K3 (NVFP4), as in `recipes/`. "Tuned" =
the user's current working variant (FP8: spec-off, util 0.80, seqs 2).

### C1 (single concurrency, W1)

| Lane | Config | Per-stream tok/s | MTP mean acc | KV % | Prefix hit | RAM/swap | Date |
|---|---|---|---|---|---|---|---|
| NVFP4 | baseline (K3, u0.75) | ✅ ~20–29 (probe, non-real-workload) | ✅ 3.3–4.0 | ✅ ~15–22% | ✅ ~35–43% | ✅ ~123 GB, no swap | 08-15 |
| NVFP4 | baseline (K3, u0.75), W1 real | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |
| FP8 | baseline (K5, u0.55), W1 real | ⏳ (single-chat 27 tok/s measured earlier at K5, thinking-heavy — log as ✅ in notes if reused) | ✅ ~99.6% draft accept (thinking tokens) | ⏳ | ⏳ | ⏳ | |
| BF16 | baseline (K5, u0.55), W1 real | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |

### C2 (W1 + W2)

| Lane | Config | Per-stream tok/s | Aggregate tok/s | MTP mean acc | KV % | RAM/swap | Date |
|---|---|---|---|---|---|---|---|
| NVFP4 | baseline | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |
| FP8 | baseline | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |
| BF16 | baseline | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |

### C3 (W1 + W2 + W3)

| Lane | Config | Per-stream tok/s | Aggregate tok/s | MTP mean acc | KV % | RAM/swap | Date |
|---|---|---|---|---|---|---|---|
| NVFP4 | baseline | ✅ observed 15–32 (raw session, pre-prewarm discipline) | ✅ observed ~45–60 (same) | ⏳ | ✅ ~15–22% | ✅ no swap | 08-15 |
| NVFP4 | baseline, clean (post-prewarm) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |
| FP8 | baseline | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |
| BF16 | baseline | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |

### Follow-up experiments (after the matrix is filled)

| # | Experiment | Purpose | Status |
|---|---|---|---|
| E1 | K1 vs K2 vs K3 at C3 (NVFP4 first) | find the concurrency-optimal draft depth | ⏳ |
| E2 | FP8 tuned variant (spec-off, u0.80, seqs 2) at C1/C2/C3 | compare against FP8 baseline | ⏳ |
| E3 | Cold vs warm `~/.triton` restart at C3 | quantify the JIT stall (should be ~30 s–min) | ⏳ |
| E4 | K5 NVFP4 single-chat vs K3 | confirm the 1.65× single-stream figure | ⏳ |
| E5 | util 0.85/0.90 NVFP4 | KV headroom vs OOM margin | ⏳ |

## Known results so far (carry-over evidence)

- NVFP4 single-stream ~20–29 tok/s @ K3, mean MTP acceptance 3.3–4.0.
- NVFP4 C3 raw-session: per-stream 15–32, aggregate ~45–60 (post-JIT).
- FP8 single-chat 27 tok/s @ K5 (thinking-heavy traffic).
- BF16 K8 acceptance curve → K5 decision (see `evidence/bf16-acceptance-analysis.md`).
- FP8 working recipe drifted to a spec-off/u0.80/seqs-2 experiment (see
  E2) — matrix rows for FP8 must state which config they ran.
