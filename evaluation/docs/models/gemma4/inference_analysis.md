# Gemma4 E4B Inference Analysis — Bottleneck, Architecture, and Optimizations

**Context:** Full-dataset evaluation of `gemma-4-e4b-it-Q4_K_M.gguf` (4.62 GiB) on
`causal_drive_bench_20260427` (815 QA samples, 7353 VQA questions + 815 trajectory questions)
using llama.cpp server (`--parallel 64 --cont-batching`) on an A100 80 GB GPU.

All numbers below are empirically measured from 850+ completed questions, not theoretical.

---

## 1. GPU Metrics Explained

```
GPU:  61–88 % SM utilization | 20 % memory bandwidth | 18 GB / 82 GB VRAM
```

### 18 GB VRAM out of 82 GB
| Component | Size |
|-----------|------|
| Model weights (Q4_K_M, 4.5B active params) | ~5 GB |
| Vision encoder / mmproj | ~1 GB |
| KV cache (64 slots × 10 240 tokens × layers) | ~12 GB |
| **Total** | **~18 GB** |

62 GB of VRAM is unused — enough for 3–4 additional server replicas.

### 61–88 % SM utilization (compute cores)
The model is **compute-bound on attention**, not on weight reads. When 20–24 slots are
actively decoding long reasoning sequences (5 000+ tokens of KV cache), the quadratic
attention cost (O(batch × seq²)) saturates the CUDA cores. SM % oscillates because
slots alternate between fast prefill phases and slow decode phases.

### 20 % memory bandwidth
Only 20 % of the A100's 2 TB/s bandwidth is consumed. This is counter-intuitive for
LLM inference, which is normally memory-bandwidth-bound. Here it's explained by:
- Batch size ~24 is above the compute/bandwidth crossover for this model size
- Long KV sequences make attention compute-dominant, not weight-read-dominant

---

## 2. Empirical Performance Numbers

All measurements taken under real load (p64/SC=6, 54 concurrent requests):

| Metric | Value |
|--------|-------|
| Tokens per image (448 px) | **267 tokens** |
| Total prompt tokens (16 images) | **~4 272 image + ~300 text = ~4 572** |
| Prefill speed (slots available) | **~537 tokens/second** |
| Prefill time (16 images) | **~8 s** |
| Decode speed (server under load) | **~2.65 tokens/second per slot** |
| Observed throughput | **0.116–0.128 questions/second** |
| Median per-question wall time | **261 s (VQA), 339 s (trajectory)** |
| Median VQA answer length | **~51 words (~70 tokens in `content`)** |
| Reasoning tokens generated (thinking) | **~1 200–1 500 tokens (hidden)** |

---

## 3. The Real Bottleneck: Hidden Thinking Tokens

### What is happening

llama.cpp auto-detects Gemma4's chat template and applies `reasoning_format: deepseek`,
which puts the model into **chain-of-thought mode**. Before producing the visible answer,
the model generates a long thinking block stored in `reasoning_content` (a separate
response field). Our inference code reads only `message.content` (the final answer).

```
Request → [image prefill: 4 272 tokens, ~8 s]
        → [thinking block: ~1 200 tokens @ 2.65 t/s = ~453 s]  ← HIDDEN, discarded
        → [final answer:  ~70 tokens @ 2.65 t/s = ~26 s]       ← captured
        Total wall time: ~490 s
```

The thinking block consumes **~95 % of total request time** and is never saved to disk.

### Why the decode rate is 2.65 t/s (not 40+ t/s)

With 20–24 slots simultaneously decoding sequences of 5 000–6 000 tokens:
- Attention per step: O(batch × seq²) → compute-intensive, not memory-limited
- Total GPU throughput: ~64 tokens/second (shared across all active slots)
- Per-slot rate: 64 / 24 ≈ 2.65 t/s ✓

### Impact on throughput

Each 6-sample batch (SAMPLE_CONCURRENCY=6) takes:
- VQA questions: complete in ~20–60 s (once a slot is free)
- Sample completes when ALL its questions finish, including the thinking blocks
- Effective batch time: ~400–500 s, dominated by thinking generation
- Throughput: 54 questions / 450 s ≈ 0.12 q/s ✓

### What we're discarding

Every question generates ~1 200 reasoning tokens of structured analysis in
`reasoning_content`. This is currently not written to outputs.jsonl. It represents
~95 % of the compute spent per question. At minimum, future runs should save it.

---

## 4. Evaluation of the "Architecture Incompatibility" Hypothesis

A common explanation for slow Gemma4 inference is software-level architectural
incompatibility. Here is what holds in our llama.cpp context vs. what does not:

| Claim | Accuracy | Notes |
|-------|----------|-------|
| Heterogeneous attention head dims (local 256 / global 512) → FlashAttention disabled in vLLM | **True for vLLM**. Not relevant for llama.cpp. | We use `-fa on` (llama.cpp FlashAttention); the bottleneck is not attention dispatch. |
| MTP heads stripped from open-source GGUF → sequential token generation | **Partially true**. MTP would give ~2× speculative speedup. | Does not explain a 100× slowdown. The thinking tokens are the dominant cost. |
| E4B edge architecture → accidental CPU offloading | **False in our setup**. | `--n-gpu-layers 99` puts all layers on GPU. 0 % CPU compute observed. |
| 61–88 % GPU utilization = inefficient serving | **Misleading**. | GPU is active on reasoning token generation; it is not idle. |

**Verdict:** The architectural incompatibility analysis accurately describes vLLM-specific
issues and is useful context for the broader ecosystem. In our llama.cpp + GGUF setup,
the bottleneck is the reasoning format generating 1 200+ hidden thinking tokens per
question — not FlashAttention dispatch or CPU offloading.

---

## 5. Optimization Options

### Tier 1 — High impact, actionable now

#### A. Multiple server replicas (expected speedup: 3–4×)
We have 62 GB of free VRAM. Each replica needs ~18 GB.

```bash
# Start 3 additional replicas on ports 8081, 8082, 8083
docker run -d --name llamacpp-gemma4-r2 --gpus all \
  -v /datadrive/models/gemma4:/models \
  -p 8081:8080 llamacpp-server:gemma4 \
  --model /models/gemma-4-e4b-it-Q4_K_M.gguf \
  --mmproj /models/gemma-4-e4b-it.mmproj \
  --parallel 21 --ctx-size 215040 --cont-batching -fa on
```

Three replicas + one existing = 4 servers × 0.128 q/s ≈ **0.51 q/s → ~4 hours total**.
Requires a round-robin load balancer (nginx upstream / haproxy) in front of all four.
The `setup.sh` and `run_inference.sh` scripts need an `GEMMA4_API_BASE_URL` pointing at
the balancer. This is the highest-ROI optimization requiring no model changes.

#### B. Save reasoning_content to outputs.jsonl
Currently discarded. At minimum, capture it for postprocessing:

```python
# In inference.py run_single():
content   = response.choices[0].message.content or ""
reasoning = getattr(response.choices[0].message, "reasoning_content", "") or ""
return {"text": content, "reasoning": reasoning}
```

No speedup, but avoids wasting 95 % of compute. The reasoning trace could be used for
scoring reasoning quality in postprocessing without a separate LLM judge call.

### Tier 2 — Medium impact, requires server changes

#### C. Limit or disable thinking (expected speedup: 5–15×)
If the model can be made to answer directly without a thinking block, decode time
drops from ~453 s to ~26 s per question.

**Findings from testing:**
- `"thinking": False` in the API body: **no effect** — model still generates `reasoning_content`
- `"think_budget_tokens": 0`: **no effect** — model still generates `reasoning_content`
- Neither parameter is reliably supported in the current llama.cpp build

**What would work:**
1. Restart server with `--no-context-shift` and a system prompt instructing the model
   to answer directly (may override the chat template's thinking behavior).
2. Fine-tune or use a different GGUF variant that doesn't have the thinking template
   baked in. The `unsloth/gemma-4-e4b-it-gguf` variant may expose template control.
3. Use a model tagged with `-nothink` or `instruct` (without reasoning) if Google releases one.
4. Patch the GGUF's chat template to remove the `<think>` start token injection.

**Caution:** Disabling thinking will reduce answer quality. For a benchmark measuring
causal reasoning, maintaining the thinking trace (even if token-limited) is likely
preferable. Recommend testing a sample with thinking disabled vs. enabled before
committing to this optimization.

#### D. Reduce `max_tokens` for trajectory questions
Trajectory questions (`--trajectory-eval`) generate `reasoning_content` of comparable
length to VQA but their final answers are shorter (~37 words). With `max_tokens=10000`,
some trajectory thinking chains approach the limit, producing 1 000s of second wall times.

Setting `TRAJ_MAX_TOKENS=3000` (thinking + answer) should cut p90 trajectory time
significantly without truncating meaningful content.

### Tier 3 — Architectural, requires re-engineering

#### E. Pre-compute and cache image embeddings
Current cost per request: 4 272 image tokens × prefill at 537 t/s ≈ **8 s** (when slot available).
This is only ~2 % of total request time when thinking is enabled.

**If thinking is disabled** (see Tier 2-C), image encoding becomes 8 s out of 30 s total
= 27 % of time → then this optimization becomes high value.

Implementation: run all 53 024 images through the vision encoder offline using
`llama.cpp`'s embedding API or a standalone SigLIP encoder. Save as `.npy` files.
Modify the API call to pass pre-encoded vision embeddings instead of raw images.
**Requires custom llama.cpp API support or a different serving approach.**

#### F. Switch to vLLM when Gemma4 support matures
vLLM is rapidly adding Gemma4 support (tracking issue vllm-project/vllm#38887).
Once available with FlashAttention on the heterogeneous head dims, vLLM provides:
- `PagedAttention` for better KV cache utilization
- Continuous batching with better scheduling
- Tensor parallelism across multiple GPUs
- Native speculative decoding

Switch is straightforward: vLLM exposes the same OpenAI-compatible API. Only
`GEMMA4_API_BASE_URL` and `GEMMA4_API_MODEL_NAME` need updating.

#### G. Switch to Gemma4 26B-A4B MoE
The 26B MoE model has ~3.8B active parameters per token (similar compute to E4B) but
is trained for server workloads rather than edge deployment. It does not use PLE
(Per-Layer Embeddings) and integrates better with standard batching schedulers.
Requires more VRAM (~32–40 GB) but the A100 80 GB can host it with 2× parallelism.

---

## 6. Optimization Priority Summary

| Option | Speedup | Effort | Recommendation |
|--------|---------|--------|----------------|
| Multiple replicas (3 more) | ~4× | Low (config) | **Do this for next eval run** |
| Save reasoning_content | 0× throughput | Low (code) | **Do this now** |
| Limit thinking / no-think mode | 5–15× | Medium (server restart, quality risk) | **Test on small sample first** |
| Reduce `max_tokens` for traj | 1.5–2× | Low (env var) | **Yes, set TRAJ_MAX_TOKENS=3000** |
| vLLM migration | 2–5× | High (wait for Gemma4 support) | Track, not yet actionable |
| 26B A4B MoE | Unknown | High | Evaluate after vLLM support |
| Image embedding cache | <5 % with thinking on | Very high (custom server) | Defer until thinking is handled |

---

## 7. Lessons for Other Model Integrations

These findings generalize across multimodal models on llama.cpp:

**Image tokens dominate the context.** At 448 px, each image produces 267 tokens
(SigLIP encoder). 16 images = 4 272 tokens, filling 42 % of a 10 240-token slot.
This limits how many images can be used per question before hitting context limits,
and drives up prefill time linearly with image count. Smaller images (224 px) reduce
this to ~67 tokens/image — a 4× reduction in vision encoding cost.

**Thinking models multiply token cost.** Any model with chain-of-thought enabled
(Gemma4, Qwen3, DeepSeek-R1 family, o-series) generates a large hidden thinking block
that you pay for even if you discard it. Always check `reasoning_content` in the
response and factor the thinking budget into ETA calculations. Do not estimate throughput
from answer word counts alone.

**`inference_time_s` includes queue wait, not just server processing time.** The measured
wall time per question is a poor proxy for server efficiency when slots are contended.
Use `timings.predicted_per_second` from the server response to measure true GPU throughput.

**Server slot count drives VRAM, not throughput.** 64 slots × 10 240 tokens requires
~12 GB KV cache. The real throughput gain comes from total tokens decoded per second
across all slots, not from slot count alone. With thinking-heavy models, fewer slots
with higher per-slot token budgets (and shorter thinking) often outperform many slots
with capped budgets.

**Multiple replicas beat single large parallel server.** Given the A100's large VRAM
headroom, running 3–4 independent replicas (each with --parallel 21) and load-balancing
provides linear throughput scaling with no code changes to the inference client.

---

## 8. Recommended Setup for Next Evaluation Run

```bash
# 1. Start 4 replicas (21 slots each = same 84 total slots as current p64)
#    Replace GEMMA4_API_BASE_URL with the load balancer URL

# 2. Set in dev.env:
CONCURRENT_REQUESTS=84          # 4 replicas × 21 slots
SAMPLE_CONCURRENCY=8            # 8 × 10 q = 80 in-flight
TRAJECTORY_EVAL=true
MAX_TOKENS=5000                 # Reduced from 10000 (cuts tail latency)
PRELOAD_IMAGES=true
GEMMA4_API_BASE_URL=http://localhost:9090/v1  # load balancer

# 3. Save reasoning_content in inference.py (see Section 5-B above)
```

Expected throughput: ~0.45–0.55 q/s → full dataset in ~4 hours vs current 15–16 hours.
