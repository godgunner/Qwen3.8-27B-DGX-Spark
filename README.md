# Qwen3.8-27B on one NVIDIA DGX Spark — tested vLLM recipes

Serve Qwen3.8-27B (hybrid Gated-DeltaNet VLM, 262K native context, native MTP
head) on a **single NVIDIA DGX Spark (GB10, 128 GB unified memory)** — in the
quant that fits your use case, with speculative decoding on, thinking effort
under control, and a verifier that proves what you actually launched.

**This is a one-DGX-Spark verified setup.** Every launch command here was
run on one GB10 node; evidence (startup logs, MTP acceptance data, memory
behavior) lives in [`evidence/`](./evidence/).

## Status: measured vs. not tested

Honesty rule for this repo: any number we have not measured on this hardware
is marked **NOT TESTED** and updated as we measure it.

| Item | Status |
|---|---|
| BF16 lane launches + serves (vLLM 0.23.1 nightly, GB10) | ✅ tested — [`evidence/`](./evidence/) |
| BF16 MTP acceptance at K8 | ✅ tested — [`evidence/`](./evidence/) |
| FlashInfer + fp8 KV on sm121 (patched image) | ✅ tested — log line in [`evidence/`](./evidence/) |
| Triton-attn + fp8 KV on sm121 (NVFP4 lane) | ✅ tested — log line in [`evidence/`](./evidence/) |
| NVFP4 single-request decode (util 0.75, K3) | ✅ tested — **~20–29 tok/s**, mean MTP acceptance 3.3–4.0 ([`evidence/`](./evidence/)) |
| NVFP4 3-concurrent decode (util 0.75, K3) | ✅ tested — **~15–32 tok/s aggregate** after warmup ([`evidence/`](./evidence/)) |
| `gpu_memory_utilization` no-swap point (NVFP4) | ✅ tested — **0.75 ≈ 123 GB / 128 GB, no swap** |
| JIT warmup stall on first turn | ✅ observed + mitigated — inline prewarm built into all recipes — see [Concurrency & the JIT stall](#concurrency--the-jit-warmup-stall) |
| NVFP4 single-stream steady decode (agentic, K3) | ✅ tested — **~12–15 tok/s**, mean MTP acceptance 2.2–3.5, KV 5–7% |
| FP8 single-request tok/s at K5 | **NOT TESTED** — pending |
| FP8 lane end-to-end | **NOT TESTED** — pending |
| Prefix-cache hit rate in production | ✅ observed ~80% (BF16 K8), ~35–43% (NVFP4 K3 multi-chat) |
| Concurrency headroom at util 0.75 (KV tokens) | ✅ tested — **1.81M KV tokens, 6.92× at 262K** |

## The three lanes

| Lane | Recipe | Script | Model | Notes |
|---|---|---|---|---|
| **NVFP4** ⭐ daily driver | [`recipes/qwen3.8-27b-nvfp4.yaml`](./recipes/qwen3.8-27b-nvfp4.yaml) | [`scripts/vllm-nvfp4.sh`](./scripts/vllm-nvfp4.sh) | `unsloth/Qwen3.8-27B-NVFP4` | Smallest weights → fastest bandwidth-bound decode |
| **FP8** (quality) | [`recipes/qwen3.8-27b-fp8.yaml`](./recipes/qwen3.8-27b-fp8.yaml) | [`scripts/vllm-fp8.sh`](./scripts/vllm-fp8.sh) | `Qwen/Qwen3.8-27B-FP8` | Official Qwen calibration; best quality-per-byte |
| **BF16** (reference) | [`recipes/qwen3.8-27b.yaml`](./recipes/qwen3.8-27b.yaml) | [`scripts/vllm-bf16.sh`](./scripts/vllm-bf16.sh) | `Qwen/Qwen3.8-27B` | Full precision; parity checks & quality baseline |

Single-request decode on GB10 is **memory-bandwidth-bound**: smaller weights ≈
proportionally faster decode, and the MTP head claws back the rest.

## Why each flag

The common launch (all three lanes) and the reasoning:

| Flag | Why |
|---|---|
| `--kv-cache-dtype fp8` | Halves KV footprint. Verified working on sm121 with FlashInfer (see below). On this hybrid model only 16 of 64 layers are full attention — the rest are Gated DeltaNet with constant-size state — so KV is cheap to begin with. |
| `--attention-backend` | Either works on sm121 with fp8 KV — both verified in startup logs: `flashinfer-native` (BF16/FP8 image) and `xqa`/triton (NVFP4). The NVFP4 lane runs `triton_attn`; the BF16/FP8 lanes run `flashinfer`. Cost on both: spec-decode forces `cudagraph_mode=PIECEWISE` — cosmetic. |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | The checkpoint ships its own MTP head (`mtp.*` tensors) — no draft model. **K3 is the concurrency-friendly default**: it still raises single-stream throughput ~1.65× (measured on this box: 3.3–4.0 mean acceptance, 20–29 tok/s single). We started at K8 (positions 4–8 mostly didn't pay), then K5 (single-chat max), then **K3** because every spec token multiplies the per-step draft forwards that a concurrent batch must wait on — K3 keeps the single-chat lift while staying fast under 2–3 parallel requests. Tune per workload: K5 if strictly single-user, K3 for concurrent, 0 to disable. |
| `--reasoning-parser qwen3` | Without it the entire thinking trace arrives inline in `content` (the chat template emits no opening `think` tag, only a closing one). Any client that splits on `reasoning_content` silently breaks. Non-negotiable for agentic use. |
| `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | Without these the server never emits structured `tool_calls`; every agentic client silently degrades to text parsing. |
| `--gpu-memory-utilization 0.75` | **Tested no-swap point: ~123 GB of 128 GB system memory in use, no swap** (NVFP4). GB10 unified memory means the "GPU pool" is system RAM; a driver-level OOM can wedge the whole node, not just the process — err low if unsure. At 0.75 the NVFP4 lane reserves **64.27 GiB KV cache = 1.81M tokens** (6.92× at 262K). KV scales linearly with this value. |
| `--max-num-seqs 4` | Single-user box: admit a few streams, keep batch-1 decode latency. |
| `--max-model-len 262144` | Native window. (Optional 1M via YaRN: **NOT TESTED** on this box.) |
| `--enable-chunked-prefill` | Smooths long-prompt prefill against decode; standard on vLLM v1. |
| `--enable-prefix-caching` | Observed ~80% hit rate in an agentic session — real, repeated-system-prompt workloads benefit a lot. (Mamba-layer support in this build is experimental per vLLM's own startup warning.) |
| `--load-format fastsafetensors` | Multi-shard parallel load; 18 shards in ~12 s in our log. |
| `--served-model-name Qwen3.8-27B*` | Short, stable name for clients. Without it, `HF_HUB_OFFLINE` makes vLLM serve the local HF cache path as the model id. |
| `--mm-encoder-tp-mode data` / `-tp 1` / `--distributed-executor-backend ray` | Single-node no-ops retained from the cluster-capable image; harmless. |
| `--quantization compressed-tensors` (NVFP4 only) | Makes vLLM load the unsloth NVFP4 (W4A4) weights explicitly. Drop it if a future checkpoint auto-detects. |
| `--trust-remote-code` | Model ships custom code paths. |
| `HF_HUB_OFFLINE=1` | Serve from the local HF cache only — no network at inference time. |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` (NVFP4) | Marlin kernel flag for this quant path. |

## Chat template mod

`mods/fix-qwen3.6-chat-template/` carries a fixed `chat_template.jinja` that
supports `enable_thinking`, `reasoning_effort`, and `preserve_thinking`
kwargs. Apply with `--chat-template <path>` when you need to override the
served template.

**Thinking effort is controllable per request** (verified in the template):

```json
{"chat_template_kwargs": {"reasoning_effort": "medium"}}
{"chat_template_kwargs": {"enable_thinking": false}}
```

Heads-up: this model's default effort is aggressive — a single one-shot
creative task produced **75,291 reasoning tokens** before the first output
line. For latency-sensitive calls, drop the effort or disable thinking.

## Recommended sampling (model card)

| Mode | temp | top_p | top_k | presence |
|---|---|---|---|---|
| Thinking (default) | 1.0 | 0.95 | 20 | 0.0 |
| Instruct (no think) | 0.7 | 0.80 | 20 | 1.5 |

vLLM auto-picks up the thinking-mode values from `generation_config.json`
(the startup "sampling parameters overridden" warning is correct behavior).
Note: vLLM warns that `min_p`/`logit_bias` don't work with speculative decoding.

## Concurrency & the JIT warmup stall

**Symptom (observed, real):** with 3 chats running in parallel, the first
concurrent turn drops every request to **~0.1–0.8 tok/s for ~30 s–a few
minutes**, then recovers to full speed (measured **15–32 tok/s** at 3-wide
after warmup). A plain dense model doesn't do this.

**Cause:** this hybrid Gated-DeltaNet + MTP model JIT-compiles a set of
bespoke Triton kernels (`precopy_mamba_align_fused`, `eagle_*`,
`causal_conv1d_*`, `fused_sigmoid_gating_delta_rule_update`, `topk_topp`, …)
**once per batch shape**. vLLM's own startup warmup only compiles the
single-request shape, so the *first time* you go 2-wide or 3-wide, those
batch shapes compile on the fly and every in-flight request pays for it.
The `WARNING … Triton kernel JIT compilation during inference: … This causes
a latency spike; consider extending warmup` lines in the log are exactly
this — they stop appearing once the shapes are compiled.

**Fixes, in order of effect:**

1. **Inline prewarm — now built into every recipe.** Each `recipes/*.yaml`
   `command` block launches a backgrounded prewarm subshell *alongside* the
   vLLM server: it waits for `/health`, then fires 1, 2, 3, 4 concurrent
   tiny non-thinking requests (`/tmp/prewarm_<lane>.log` inside the
   container). So launching a recipe via `run-recipe.py` is self-warming —
   no extra step. The standalone `scripts/vllm-*.sh` launchers do the same
   thing automatically via `scripts/prewarm.sh`. Run
   **`./scripts/prewarm.sh 8000 <model> 4`** manually if you launch without
   either mechanism.
2. **Mount `~/.triton` (and `~/.cache/vllm`, `~/.cache/flashinfer`)** so the
   compiled kernels and torch.compile artifacts survive container restarts.
   All three standalone scripts mount these; the `run-recipe.py` launcher
   mounts `~/.triton` by default (opt-out `--no-cache-dirs`). With the cache
   warm, only a *new* batch width you've never hit will compile.
3. **K3 instead of K5** — fewer draft forwards per step, so even a slow
   compile step blocks less. (See the speculative-config row above.)

**Measured (2026-08-16, NVFP4, vLLM 0.27.2rc1.dev126, K3, util 0.75):**
without the inline prewarm, a single Copilot chat's first turn hit the stall
exactly as predicted — `POST /v1/messages 200` at 12:31:45, then
`precopy_mamba_align_fused_kernel` / `eagle_*` / `expand_kernel` JIT warnings
through 12:33:35, and the 10 s engine samples read **10.7 → 13.4 → 19.4 →
15.8 tok/s** only after the last `_topk_topp`-family kernel compiled. Steady
single-stream decode settled at **~12–15 tok/s, mean MTP acceptance 2.2–3.5,
KV 5–7%, prefix-cache hit 47–64%** for agentic traffic. (Note the 404s at
12:31:45: the client had `unsloth/Qwen3.6-27B-NVFP4` selected — a model-name
mismatch on the client side, not a server fault.)

**Not a bug, not a concurrency limit:** at util 0.75 the NVFP4 lane has
**1.81M KV tokens** (6.92× at 262K) and `--max-num-seqs 4`, so 2–3 concurrent
chats fit comfortably — the stall is purely the one-time kernel compile, and
it's what the warmup warnings were telling us to fix.

## Verify what you launched (don't trust the launch)

```bash
./scripts/verify.sh 8000 Qwen3.8-27B-NVFP4
```

Prints: HTTP health, served model name, a timed 400-token non-thinking probe
(tok/s), and the server's cumulative MTP acceptance (`vllm:spec_decode_num_*`).
If acceptance metrics are absent, your spec config didn't take. If
`reasoning_content` is empty on a thinking request, your reasoning parser
isn't set.

## Provenance

- Single NVIDIA DGX Spark (GB10, 128 GB unified memory). vLLM
  `0.23.1rc1.dev1431+ge9d9c7a39.d20260723` (BF16/FP8 logs) and
  `0.27.2rc1.dev88+gaa3100357.d20260814` (NVFP4 log), patched-image builds.
- Measured 2026-08-14/15. Evidence: [`evidence/bf16-acceptance-analysis.md`](./evidence/bf16-acceptance-analysis.md).
- Not affiliated with NVIDIA or Qwen. Community notes, updated as tests land.
