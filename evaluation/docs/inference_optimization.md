# Inference Optimization Notes

Performance tuning guide for the Gemma4 (and API-based) evaluation pipeline.
Derived from profiling the Gemma4 E4B full-dataset run on A100 80 GB (April 2026).

---

## Bottleneck Model

The API-based inference pipeline (Gemma4) has this time breakdown per sample:

```
For each sample (~12 questions = 8 VQA + 4 trajectory):

  save_prompts()        ~0.1 s   — reads JSON, writes image paths (no pixel work)
  image encode × 12    ~1.5 s   — PIL open + LANCZOS resize + JPEG + base64 per question
  API latency × 12     ~37 s ea — GPU inference, serialized by server n_parallel slots

Wall clock per sample ≈ 12 × 37 s / n_parallel  (GPU is the bottleneck)
```

The GPU is nearly 100% busy once the request queue is full.
All code-side optimizations target eliminating idle gaps in that queue.

---

## Implemented Optimizations (inference.py)

### 1. In-RAM Image Cache (`self._image_cache`)

**Problem:** Each of the ~12 questions per sample calls `_build_vision_messages`, which
calls `_resize_and_encode` for all 16 images. The same 16 images are encoded up to 12 times.

**Fix:** A shared dict `(str(abs_path), image_size) → base64_string` on the
`Gemma4Inference` instance. All threads read/write it without a lock (CPython dict
operations are atomic; last-writer-wins is fine for identical re-encodes).

**Effect:** Within a sample, only the first 4 concurrent threads encode the 16 images.
Threads 5–12 get cache hits. Saves ~7/12 of encoding work per sample.

---

### 2. Parallel Prompt Pre-generation (`PROMPT_GEN_WORKERS`)

**Problem:** The original loop called `save_prompts()` / `save_trajectory_prompts()` inline,
interleaved with inference. This serialized text-only I/O with GPU wait time and re-wrote
files on every resume.

**Fix:** Before the inference loop, all `prompts.jsonl` and `trajectory_prompts.jsonl` files
are generated in parallel via `ThreadPoolExecutor(max_workers=PROMPT_GEN_WORKERS)`.
Files that already exist are skipped (safe on resume). The inference loop then only runs
`run_from_jsonl` — no prompt generation in the hot path.

**Tuning:** `PROMPT_GEN_WORKERS=64` (default). `save_prompts` is disk-I/O + libjpeg decode
(GIL-releasing), so 64 threads scale well on NVMe. Peak RAM: ~8 GB (64 workers × 16 images
× ~8 MB PIL arrays). Full-dataset pre-gen for 3314 samples completes in ~30–60 s.

---

### 3. Image RAM Preload (`PRELOAD_IMAGES=true`)

**Problem:** Even with the lazy cache, the first 4 concurrent threads per sample still pay
the full encode cost (~1.5 s) before the first API call goes out.

**Fix:** Before inference starts, `preload_image_cache()` enumerates all unique image paths
across the entire dataset and encodes them in parallel. By inference time, every
`_build_vision_messages` call is a dict lookup with zero disk or PIL work.

**Dataset facts (causal_drive_bench_20260427, causal_nuscenes):**
- 3314 samples × 16 images = **53,024 unique images** (zero cross-sample sharing confirmed)
- Base64 JPEG at 448×448: ~40 KB each → **~2.1 GB RAM**
- Preload time with 64 workers: **~60–120 s** (depends on disk cache state)

**Env vars:**
```bash
PRELOAD_IMAGES=true      # enable (default: false)
PRELOAD_WORKERS=64       # parallelism for preload (default: 64)
```

**When to use:** Always for full-dataset runs. Skip for quick single-scene tests where
startup cost exceeds savings.

---

### 4. Sample-Level Concurrency (`SAMPLE_CONCURRENCY`)

**Problem:** Even with `CONCURRENT_REQUESTS=64` matching `--parallel 64`, the per-sample
Python loop submits only one sample's questions at a time (~9 VQA + 1 traj = 10 questions).
The server's 64 parallel slots are only ~15% filled; the GPU sees idle gaps between samples.

**Fix:** The main loop now wraps sample processing in a `ThreadPoolExecutor` sized by
`SAMPLE_CONCURRENCY`. With `SAMPLE_CONCURRENCY=K`, K samples are processed simultaneously,
putting K × ~10 questions into the server queue at once.

**Throughput math** (measured on A100, Gemma4 E4B):

| Config | Slots in use | avg/q | Wall throughput |
|---|---|---|---|
| SC=1, `--parallel 64` | ~10 / 64 | 71s | ~0.85 q/s |
| SC=6, `--parallel 64` | ~60 / 64 | 71s | ~0.85 q/s |
| SC=19, `--parallel 192` | ~190 / 192 | ~150s* | ~1.3 q/s |

*Server image-encode bottleneck: 190 slots × 16 images × ~100ms = 5 min prefill startup
for the first round. Subsequent rounds are faster once the first wave completes.

**Important constraint:** Setting SAMPLE_CONCURRENCY very high causes the server to
process all image prefills serially (llama.cpp encodes one image at a time per slot),
creating a startup "bubble" where no questions complete for several minutes. The sweet
spot depends on `--parallel N` and response length:
- For `--parallel 64`: `SAMPLE_CONCURRENCY=6` fills slots without startup bubble
- For `--parallel 192`: `SAMPLE_CONCURRENCY=19` for maximum theoretical throughput,
  but accept ~5 min startup bubble per round

**Env vars:**
```bash
SAMPLE_CONCURRENCY=6   # number of samples to process in parallel (default: 1)
CONCURRENT_REQUESTS=64  # API worker threads per run_from_jsonl call; set ≥ avg_q_per_sample
```

**Safety:** Each sample writes to its own independent output files, so concurrent
processing is safe. Resume logic (`run_from_jsonl(resume=True)`) is thread-safe since
each sample has its own `outputs.jsonl` path.

---

### 5. Concurrent VQA + Trajectory Inference per Sample

**Problem:** The original loop called `run_from_jsonl` for VQA, waited for completion,
then called it again for trajectory. The server queue emptied between the two calls
(~0.5 s idle per sample × 3314 samples ≈ 27 min total).

**Fix:** When both `do_vqa` and `do_traj` are true, both `run_from_jsonl` calls are
submitted to a 2-thread `ThreadPoolExecutor` and run concurrently. VQA and trajectory
prompts are independent (trajectory doesn't use VQA outputs), so this is safe.

**Effect:** VQA questions and trajectory questions are all in the server queue simultaneously,
eliminating the inter-call idle gap.

---

### 5. Concurrent API Requests (`CONCURRENT_REQUESTS`)

**Background:** Gemma4 uses an OpenAI-compatible HTTP server (llama.cpp). Python-side
concurrency is via `ThreadPoolExecutor` inside `run_from_jsonl`.

**Server parallelism:** The llama.cpp server is started with `--parallel N` slots.
Each slot holds one in-flight request's KV cache. The `docker model bench` output
(April 2026, via Docker model runner at n_parallel≈4) confirmed:

```
Concurrency 1: 132.49 tok/s
Concurrency 2: 203.43 tok/s  (1.54×)
Concurrency 4: 271.28 tok/s  (2.05×)  ← plateau for n_parallel=4
Concurrency 8: 267.52 tok/s  (2.02×)  ← server queue builds up, no gain
```

These numbers reflect the Docker model runner which internally used `n_parallel≈4`.
With **`--parallel 64`** on the direct llama.cpp server (see §Server Setup below),
the plateau shifts to 64 concurrent requests.

**Current setting:** `CONCURRENT_REQUESTS=64` matches `--parallel 64` server.

---

## Server Setup: Direct llama.cpp Container (`--parallel 64`)

The Docker model runner daemon (`docker-model-runner`, port 12434) does not expose
`--parallel` configuration. Throughput is capped at ~4 concurrent slots.

**Solution (implemented 2026-04-28):** build a minimal image from the model runner's
own llama-server binary and run it directly with explicit flags:

```dockerfile
FROM docker/model-runner:latest-cuda
USER root
RUN chmod +x /app/bin/llama-server
ENV LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH
ENTRYPOINT ["/app/bin/llama-server"]
```

```bash
# Build once
docker build -t llamacpp-server:gemma4 /tmp/llamacpp-build/

# Start server (idempotent — see start_llamacpp_server.sh)
docker run -d \
  --name llamacpp-gemma4-p64 \
  --gpus all --ipc=host \
  -v /datadrive/models/gemma4:/models:ro \
  -p 8080:8080 \
  llamacpp-server:gemma4 \
  -m /models/gemma-4-e4b-it-Q4_K_M.gguf \
  --mmproj /models/gemma-4-e4b-it.mmproj \
  --port 8080 --host 0.0.0.0 \
  --parallel 64 --cont-batching \
  --ctx-size 655360 \       # 64 slots × 10240 tokens each
  --n-gpu-layers 99 \
  -fa on
```

**Model files** (copied from Docker model runner volume):
- `/datadrive/models/gemma4/gemma-4-e4b-it-Q4_K_M.gguf` — 4.8 GB model
- `/datadrive/models/gemma4/gemma-4-e4b-it.mmproj` — 945 MB vision projector

**Script:** `eval_20260427/start_llamacpp_server.sh`

---

## GPU VRAM Budget (A100 80 GB)

Model weights: **3002 MiB** (Gemma4 E4B Q4_K_M, on CUDA0)
Vision projector: **944 MiB** (CLIP mmproj)

**Actual KV allocation at `--parallel 64`, `--ctx-size 655360` (10240 ctx/slot):**

| KV cache component | Size |
|---|---|
| Non-SWA (4 global-attn layers, 64 seqs × 10240 ctx) | 10240 MiB |
| SWA (20 sliding-window layers, 64 seqs × 1024 window) | 2560 MiB |
| Compute buffer | 570 MiB |
| **Total GPU used** | **~18 GB** |
| **Free for other work** | **~63 GB** |

The Gemma4 SWA architecture is far more VRAM-efficient than the initial estimate
(46.5 GB from a naive calculation). The SWA layers only cache a 1024-token window,
not the full 10K context.

Estimated KV budget table (updated with actual measurements):

| `--parallel N` | Actual KV (SWA arch) | Total GPU | Fits in 80 GB? |
|---|---|---|---|
| 4 (old model-runner) | ~0.8 GB | ~5.5 GB | ✓ |
| 64 (current) | ~12.8 GB | ~18 GB | ✓ |
| 128 | ~25.6 GB | ~31 GB | ✓ |
| 256 | ~51.2 GB | ~56 GB | ✓ |

**To change throughput:** edit `--parallel N` in `start_llamacpp_server.sh`, restart
the container, and set `CONCURRENT_REQUESTS=N` in `restart_inference.sh`.

With `--parallel 64` + `CONCURRENT_REQUESTS=64`, throughput scales linearly with
GPU compute capacity (~271 tok/s × 64/4 ≈ 4336 tok/s theoretical upper bound;
actual depends on batch scheduling and memory bandwidth).

---

## Environment Variables Reference

| Variable | Default | Effect |
|---|---|---|
| `CONCURRENT_REQUESTS` | `4` | Python-side API concurrency per `run_from_jsonl` call; set ≥ avg questions/sample |
| `SAMPLE_CONCURRENCY` | `1` | Number of samples processed in parallel; K×questions_per_sample questions in flight |
| `PRELOAD_IMAGES` | `false` | Pre-encode all dataset images into RAM before inference (~60s, ~2.1 GB) |
| `PRELOAD_WORKERS` | `64` | Thread count for in-memory image preload |
| `PERSISTENT_IMAGE_CACHE` | `""` | Path to persistent cache dir (built by `build_image_cache.py`); loads in ~2s |
| `PROMPT_GEN_WORKERS` | `64` | Thread count for parallel prompts.jsonl generation |
| `MAX_TOKENS` | `512` | Max tokens per response (use 10000 for full CoT) |
| `IMAGE_SIZE` | `448` | Resize images to this square pixel size before encoding |
| `TRAJECTORY_EVAL` | `false` | Enable trajectory (BTE/CTE) evaluation alongside VQA |

---

## Diminishing Returns / What Doesn't Help

- **Increasing `CONCURRENT_REQUESTS` beyond `n_parallel`:** requests just queue; no GPU gain.
- **Pre-encoding images without `PRELOAD_IMAGES`:** lazy cache still encodes once per 4
  concurrent workers at sample start; savings are marginal vs preload.
- **Loading dataset JSON into RAM:** state.json / frames.json / qa files are tiny (<100 KB
  each). Not a bottleneck.
- **Cross-sample global queue:** inter-sample Python overhead is ~0.5 s/sample × 3314 = 27 min,
  vs 90-hour total runtime = 0.5%. Optimization 4 (concurrent VQA+traj) captures most of this.

---

## Full-Dataset Run Reference (eval_20260427)

```
Dataset:    causal_drive_bench_20260427 / causal_nuscenes
Samples:    815 valid (851 scenes; only samples with qa/ folders count)
Questions:  7285 VQA + 814 trajectory answered
Model:      Gemma4 E4B (gemma-4-e4b-it-Q4_K_M.gguf, 4.8 GB; mmproj 945 MB)
GPU:        A100 80 GB PCIe  (~18 GB used with --parallel 64)
Server:     llamacpp-gemma4-p64 (port 8080), --parallel 64, --cont-batching

Phase 1 server:  Docker model runner (n_parallel≈4) — 815 samples processed
Phase 2 server:  Direct llama.cpp (--parallel 64) — confirmed all 815 done, 0 empty

Outputs:    /datadrive/datasets/cdb_outputs_20260427/gemma4_e4b_20260427_234037
Scripts:
  eval_20260427/start_llamacpp_server.sh  — start direct llama.cpp server
  eval_20260427/restart_inference.sh      — resume with all optimizations
  eval_20260427/monitor.sh                — check progress
  eval_20260427/run_postprocess.sh        — run after inference completes
```
