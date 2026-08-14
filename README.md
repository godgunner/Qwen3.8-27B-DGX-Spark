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
| FlashInfer + fp8 KV on sm121 (our patched image) | ✅ tested — log line in [`evidence/`](./evidence/) |
| NVFP4 lane serving (unsloth checkpoint, util 0.55, no swap, ~123 GB used) | ✅ tested — current running state |
| `gpu_memory_utilization 0.55` = no swap | ✅ tested |
| NVFP4 / FP8 single-request tok/s | **NOT TESTED** — pending |
| MTP acceptance at K5 (post-tuning) | **NOT TESTED** — pending fresh log |
| FP8 lane end-to-end | **NOT TESTED** — pending |
| Prefix-cache hit rate in production | ✅ observed ~80% in the K8 session ([`evidence/`](./evidence/)) |
| Concurrency headroom at util 0.55 (KV tokens) | **NOT TESTED** — pending startup log |

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
| `--attention-backend flashinfer` | Our patched vLLM image's FlashInfer build **natively serves fp8 KV on sm121** (`kv_cache_dtype=torch.float8_e4m3fn, arch=sm121` in the startup log). No triton fallback needed. Cost: spec-decode forces `cudagraph_mode=PIECEWISE` — cosmetic. |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'` | The checkpoint ships its own MTP head (`mtp.*` tensors) — no draft model. **K5 chosen from measured K8 acceptance data**: positions 4–5 still pay (0.17–0.67, up to ~0.79 in code), positions 6–8 mostly don't (0.02–0.33). vLLM's own startup warning about repeated MTP forwards at K>1 is the mechanism; K5 trims the dead weight. Exact K5 delta: **NOT TESTED** yet. |
| `--reasoning-parser qwen3` | Without it the entire thinking trace arrives inline in `content` (the chat template emits no opening `think` tag, only a closing one). Any client that splits on `reasoning_content` silently breaks. Non-negotiable for agentic use. |
| `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | Without these the server never emits structured `tool_calls`; every agentic client silently degrades to text parsing. |
| `--gpu-memory-utilization 0.55` | **Tested value: ~123 GB of 128 GB system memory in use, no swap** (NVFP4, current). GB10 unified memory means the "GPU pool" is system RAM; a driver-level OOM can wedge the whole node, not just the process — err low if unsure. KV cache scales linearly with this value. |
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

- Single NVIDIA DGX Spark (GB10, 128 GB unified memory), vLLM
  `0.23.1rc1.dev1431+ge9d9c7a39.d20260723` (patched FlashInfer image).
- Measured 2026-08-14/15. Evidence: [`evidence/bf16-acceptance-analysis.md`](./evidence/bf16-acceptance-analysis.md).
- Not affiliated with NVIDIA or Qwen. Community notes, updated as tests land.
