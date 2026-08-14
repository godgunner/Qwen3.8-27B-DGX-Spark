# Evidence — BF16 lane (single DGX Spark, GB10)

## Startup log (abridged)

vLLM `0.23.1rc1.dev1431+ge9d9c7a39.d20260723`, model `Qwen/Qwen3.8-27B`,
resolved architecture `Qwen3_5ForConditionalGeneration` (hybrid Gated-DeltaNet
full attention + 16 full-attention layers), drafter `Qwen3_5MTP`.

Key lines and why each one matters:

```text
INFO [cache.py:285] Using fp8 data type to store kv cache. It reduces the GPU memory footprint and boosts the performance...
```
fp8 KV cache active — halves KV footprint vs bf16.

```text
WARNING [speculative.py:901] Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate
```
This is the launch the acceptance data below was taken on: **K8**. It is the
smoking gun for why we dropped to K5 — positions 6–8 in the data below
rarely pay for themselves.

```text
INFO [cuda.py:482] Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN'].
INFO [flashinfer.py:822] FlashInfer resolved query dtypes: prefill=torch.bfloat16, decode=torch.bfloat16, decode_backend=flashinfer-native, kv_cache_dtype=torch.float8_e4m3fn, arch=sm121
```
**FlashInfer natively serves fp8 KV on sm121 (GB10)** with our patched build.
No triton fallback needed.

```text
WARNING [compilation.py:1443] CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE); setting cudagraph_mode=PIECEWISE
```
Expected: spec-decode + FlashInfer forces PIECEWISE CUDA graphs. Cosmetic.

```text
INFO [gpu_worker.py:578] Available KV cache memory: 22.72 GiB
INFO [kv_cache_utils.py:2214] GPU KV cache size: 586,374 tokens
INFO [kv_cache_utils.py:2215] Maximum concurrency for 262,144 tokens per request: 2.24x
INFO [gpu_worker.py:821] Free memory on device (116.05/121.69 GiB) on startup. Desired GPU memory utilization is (0.6, 73.01 GiB). Actual usage is 60.69 GiB for consumed memory...
```
That launch used `gpu_memory_utilization 0.6` → only 22.72 GiB of KV cache
→ ~2.24× concurrency at the full 262K window. This is why the recipe now
runs at **0.55 as the tested safe value with no swap** (see below) — and why
the KV budget scales linearly with it: more headroom = more concurrent 262K
requests or a larger effective context under load.

Note: this startup was captured with util 0.6 *before* the final tuning. The
final tested value on this machine is **0.55 with ~123 GB of 128 GB system
memory in use and no swap** (NVFP4 lane, current). The 0.60→0.55 KV-cache
implications above are directional.

```text
INFO [api_server.py:677] Starting vLLM server on http://0.0.0.0:8000
INFO [model.py:1550] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}`.
```
Server up; the generation-config override warning is correct behavior
(thinking-mode sampling defaults baked into the checkpoint).

## Speculative decoding acceptance (K8 launch, mixed agentic traffic)

Sampled from the server log over a multi-minute agentic session
(`/v1/messages`, 1–2 concurrent requests, 80% prefix-cache hit rate):

```text
Per-position acceptance rate (selected samples, positions 1..8):
0.712, 0.577, 0.385, 0.231, 0.173, 0.115, 0.058, 0.038   mean acceptance length 3.29
0.750, 0.500, 0.404, 0.250, 0.192, 0.096, 0.077, 0.077   mean acceptance length 3.35
0.740, 0.500, 0.320, 0.240, 0.180, 0.140, 0.120, 0.100   mean acceptance length 3.34
0.827, 0.577, 0.404, 0.288, 0.154, 0.077, 0.019, 0.019   mean acceptance length 3.37
0.865, 0.712, 0.596, 0.481, 0.385, 0.327, 0.250, 0.154   mean acceptance length 4.77
0.720, 0.640, 0.520, 0.420, 0.340, 0.320, 0.240, 0.220   mean acceptance length 4.42
0.885, 0.731, 0.577, 0.500, 0.462, 0.385, 0.308, 0.288   mean acceptance length 5.13
0.939, 0.818, 0.818, 0.818, 0.636, 0.545, 0.515, 0.485   mean acceptance length 6.58 (prompt burst)
0.827, 0.788, 0.769, 0.731, 0.596, 0.577, 0.519, 0.423   mean acceptance length 6.23 (code burst)
0.904, 0.846, 0.788, 0.712, 0.673, 0.635, 0.596, 0.596   mean acceptance length 6.75 (code burst)
```

Read-off:

- **Positions 1–3:** consistently 0.32–0.89 — always pay.
- **Positions 4–5:** 0.17–0.67 average, spikes to 0.46–0.79 in code-heavy
  segments — worth keeping; the model's own warning about repeated MTP
  forwards is the price we pay for these two.
- **Positions 6–8:** 0.02–0.33, mostly below 0.17 outside code bursts —
  this is the dead weight K8 was paying verification cost for.
- Mean acceptance length 2.5–5.1 at K8 (higher on code, lower on prose).

**Decision:** `num_speculative_tokens 8 → 5`. Keep the positions that pay,
drop the three that mostly burn. A/B against 4/6 recommended; not yet
re-measured, so treat the exact K5 delta as **not tested** until a fresh
acceptance log is captured.

## Memory behavior

| Config | Observed |
|---|---|
| NVFP4, util 0.55, current | **~123 GB of 128 GB system memory in use, no swap** — the tested safe point |
| BF16, util 0.6, startup log above | 60.69 GiB consumed (weights + non-torch) before KV; KV got 22.72 GiB |

Full raw log: `evidence/bf16-startup-raw.log`
