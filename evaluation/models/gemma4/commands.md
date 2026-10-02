# Gemma 4 — Run Commands

## Server Options

There are two ways to run the Gemma4 model server.  Choose based on your setup:

| | Option A — Docker model runner | Option B — Direct llama.cpp (recommended for full runs) |
|---|---|---|
| **Setup** | `docker model pull ai/gemma4:e4b` | `bash evaluation/models/gemma4/setup.sh` |
| **n_parallel** | ≈ 4 (fixed by daemon) | Configurable (default: 64) |
| **Throughput** | ~4 concurrent requests | ~64 concurrent requests (16× faster) |
| **VRAM (E4B)** | ~5.5 GB | ~18 GB (A100 has 63 GB free) |
| **API endpoint** | `http://localhost:12434/v1` | `http://localhost:8080/v1` |
| **Model name** | `ai/gemma4:e4b` | `model.gguf` |
| **CONCURRENT_REQUESTS** | 4 | 64 |

> Option B was validated on an A100 80 GB with the E4B variant and achieved
> ~1.3 questions/second wall-clock throughput on the full 815-sample dataset.

---

## Prerequisites

Fill in credentials and paths:
```bash
cp evaluation/models/gemma4/dev.env.example /path/to/dev.env
# Edit /path/to/dev.env:
#   GEMMA4_MODEL_TAG    — model variant: e2b | e4b | 26b-a4b | 31b
#   CAUSAL_BENCH_DIR    — host path to causal_drive_bench/
#   CDB_OUTPUTS_DIR     — host path for inference outputs
#   CDB_REPORTS_DIR     — host path for evaluation reports
#   RAW_DATA_DIR        — host path to raw open-source datasets
```

The `dev.env` file can live anywhere on the host. Pass its path to every
`docker compose` command via `--env-file`. No HuggingFace token is required —
model weights are shipped inside the `ai/gemma4` Docker image.

---

## 1. Pull the Gemma 4 Model Image (first time per variant)

The model server image contains the weights. Pull it once per variant before
running inference.

### Option A — Docker model runner (Docker Desktop / Engine 28+)
```bash
docker model pull ai/gemma4:e2b
docker model pull ai/gemma4:e4b
docker model pull ai/gemma4:26b-a4b
docker model pull ai/gemma4:31b
```

### Option B — Standard Docker pull (Linux without Desktop)
```bash
docker pull ai/gemma4:${GEMMA4_MODEL_TAG:-e2b}
```

### Option C — Via the weight-downloader compose service
```bash
GEMMA4_MODEL_TAG=e2b \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/gemma4/docker-compose.yml \
  --profile setup run --rm weight-downloader
```

Repeat for each variant you want to benchmark.

---

## 2. Build the Evaluation Wrapper Image

Only the `inference/postprocess` container needs building — the model server
image (`ai/gemma4`) is pre-built.

```bash
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/gemma4/docker-compose.yml \
  build
```

---

## 3. Sanity-Check the Model Server (optional)

Starts the Gemma 4 server and sends a test image+text request.

```bash
GEMMA4_MODEL_TAG=e2b \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/gemma4/docker-compose.yml \
  --profile original up
```

Successful output ends with: `[Test] Gemma 4 API is working correctly.`

---

## 4. Run Inference

> The `gemma4-server` service starts automatically via `depends_on`.

### Full dataset (default)
```bash
GEMMA4_MODEL_TAG=e2b \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/gemma4/docker-compose.yml \
  run --rm inference
```

### Subset (e.g. first 10 scenes)
```bash
GEMMA4_MODEL_TAG=e2b MODE=subset SUBSET_SIZE=10 \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/gemma4/docker-compose.yml \
  run --rm inference
```

### Single scene
```bash
GEMMA4_MODEL_TAG=e2b MODE=single SCENE=nuscenes-scene-0159 \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/gemma4/docker-compose.yml \
  run --rm inference
```

### Resume an interrupted run
```bash
GEMMA4_MODEL_TAG=e2b \
RUN_DIR=/workspace/data/cdb_outputs/gemma4_e2b_20260421_120000 \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/gemma4/docker-compose.yml \
  run --rm inference
```

Outputs are written to a **timestamped run directory**:
```
$CDB_OUTPUTS_DIR/gemma4_<tag>_<YYYYMMDD_HHMMSS>/causal_nuscenes/<scene_id>/<sample_id>/
  prompts.jsonl   — per-question prompt payloads
  outputs.jsonl   — raw model responses + timing
```

### Switching model variants
Change `GEMMA4_MODEL_TAG` per run. Each variant writes to its own directory:
```
gemma4_e2b_<timestamp>/
gemma4_e4b_<timestamp>/
gemma4_26b_a4b_<timestamp>/
gemma4_31b_<timestamp>/
```

---

## 5. Post-Process Outputs

Set `RUN_DIR` to the timestamped directory created by inference.

```bash
RUN_DIR=/workspace/data/cdb_outputs/gemma4_e2b_20260421_120000 \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/gemma4/docker-compose.yml \
  --profile postprocess run --rm postprocess
```

With LLM judge enabled:
```bash
RUN_DIR=/workspace/data/cdb_outputs/gemma4_e2b_20260421_120000 \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/gemma4/docker-compose.yml \
  --profile postprocess \
  run --rm -e USE_LLM_JUDGE=true postprocess
```

---

## 6. High-Throughput Setup (Direct llama.cpp, Recommended for Full Runs)

The Docker model runner daemon caps `n_parallel` at ~4, limiting throughput.
For full-dataset runs, use the direct llama.cpp server which supports `--parallel 64`.

### 6a. Get GGUF Weights

**Option A — Extract from existing Docker model runner volume (if `docker model pull` was used):**
```bash
# Find the volume
VOL=$(docker volume ls -q | grep model-runner | head -1)
mkdir -p /datadrive/models/gemma4

# Copy GGUF (first .gguf found in the volume)
docker run --rm -v "$VOL:/src:ro" -v "/datadrive/models/gemma4:/dst" alpine \
  sh -c 'find /src -name "*.gguf" | head -1 | xargs -I{} cp {} /dst/gemma-4-e4b-it-Q4_K_M.gguf'

# Copy mmproj
docker run --rm -v "$VOL:/src:ro" -v "/datadrive/models/gemma4:/dst" alpine \
  sh -c 'find /src -name "*.mmproj" | head -1 | xargs -I{} cp {} /dst/gemma-4-e4b-it.mmproj'
```

**Option B — Download from HuggingFace:**
```bash
pip install huggingface_hub
mkdir -p /datadrive/models/gemma4

huggingface-cli download unsloth/gemma-4-e4b-it-GGUF \
  gemma-4-e4b-it-Q4_K_M.gguf --local-dir /datadrive/models/gemma4

# Vision projector (mmproj)
huggingface-cli download unsloth/gemma-4-e4b-it-GGUF \
  mmproj-gemma-4-e4b-it-f16.gguf --local-dir /datadrive/models/gemma4
mv /datadrive/models/gemma4/mmproj-gemma-4-e4b-it-f16.gguf \
   /datadrive/models/gemma4/gemma-4-e4b-it.mmproj
```

### 6b. Build and Start the Server

```bash
# Build llamacpp-server:gemma4 image + verify weights + start container (idempotent)
bash evaluation/models/gemma4/setup.sh

# Override parallelism (optional):
PARALLEL_SLOTS=128 bash evaluation/models/gemma4/setup.sh

# Check server
curl http://localhost:8080/v1/models
docker logs -f llamacpp-gemma4
```

**VRAM budget (Gemma4 E4B, A100 80 GB):**

| `--parallel N` | Total VRAM | Free (80 GB) |
|---|---|---|
| 4 (model runner) | ~5.5 GB | ~75 GB |
| 64 (default) | ~18 GB | ~63 GB |
| 128 | ~31 GB | ~50 GB |
| 256 | ~56 GB | ~25 GB |

*(Gemma4 SWA architecture: only a 1024-token window is cached for 20 of 24 KV-heavy layers)*

### 6c. Run Inference Against the Direct Server

```bash
cd /path/to/neurips-dataset
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python evaluation/models/gemma4/inference.py \
  --bench-dir /datadrive/datasets/causal_drive_bench/causal_nuscenes \
  --output-base /datadrive/datasets/cdb_outputs \
  --api-base-url http://localhost:8080/v1 \
  --model-name model.gguf \
  --mode full \
  --max-tokens 10000 \
  --image-size 448 \
  --trajectory-eval
```

Key env vars for maximum throughput:
```bash
export CONCURRENT_REQUESTS=64    # must match --parallel N
export PRELOAD_IMAGES=true       # pre-encode all images into RAM (~2.1 GB, ~60s startup)
export PROMPT_GEN_WORKERS=64     # parallel prompts.jsonl generation
```

### 6d. Via Docker Compose (llamacpp profile)

```bash
# Start server via compose
WEIGHTS_DIR=/datadrive/models/gemma4 \
PARALLEL_SLOTS=64 CTX_SIZE=655360 \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/gemma4/docker-compose.yml \
  --profile llamacpp up -d llamacpp-server

# Run inference against it (--no-deps = don't start gemma4-server)
GEMMA4_API_BASE_URL=http://localhost:8080/v1 \
GEMMA4_API_MODEL_NAME=model.gguf \
CONCURRENT_REQUESTS=64 \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/gemma4/docker-compose.yml \
  run --rm --no-deps inference
```

---

## 7. Run Inference Directly (local dev, no Docker)

```bash
cd /path/to/neurips-dataset
export PYTHONPATH="$PYTHONPATH:$(pwd)"
pip install openai pillow numpy tqdm

# Start model server separately (e.g. docker run -p 8080:8080 ai/gemma4:e2b)

python evaluation/models/gemma4/inference.py \
  --bench-dir /datadrive/datasets/causal_drive_bench/causal_nuscenes \
  --output-base /datadrive/datasets/cdb_outputs \
  --api-base-url http://localhost:8080/v1 \
  --model-name ai/gemma4:e2b \
  --mode subset --subset-size 5
```

---

## 8. Run Postprocess Directly (local dev)

```bash
python evaluation/models/gemma4/postprocess.py \
  --bench-dir /datadrive/datasets/causal_drive_bench/causal_nuscenes \
  --run-dir /datadrive/datasets/cdb_outputs/gemma4_e2b_20260421_120000 \
  --mode full

# Reprocess all runs under cdb_outputs_dir (auto-detects datasets):
python evaluation/models/gemma4/postprocess.py --all-runs
```

---

## 9. Orchestrator (multi-model evaluation)

```bash
python evaluation/orchestrate.py \
  --model gemma4_e2b \
  --dataset causal_nuscenes \
  --mode subset --subset-size 20
```

---

## Useful Docker Operations

### View live logs of a running container
```bash
docker logs -f <container_id_or_name>
```

### Exec into the inference container
```bash
docker exec -it <container_id_or_name> /bin/bash
```

### Exec into the model server container
```bash
docker exec -it <gemma4_server_container> /bin/sh
```

### Check model server logs
```bash
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/gemma4/docker-compose.yml \
  logs gemma4-server
```

### Check model server health
```bash
# From host:
curl http://localhost:8080/v1/models

# From within the compose network (exec into inference container):
curl http://gemma4-server:8080/v1/models
```

### Inspect a stopped container's logs
```bash
docker ps -a                          # find container ID
docker logs <container_id>
```

### Remove stopped containers and dangling images
```bash
docker container prune -f
docker image prune -f
```

### List all pulled model images
```bash
docker images ai/gemma4
```

---

## Output Files

| Path | Description |
|------|-------------|
| `<run_dir>/causal_nuscenes/<scene>/<sample>/prompts.jsonl` | Per-question prompt payloads |
| `<run_dir>/causal_nuscenes/<scene>/<sample>/outputs.jsonl` | Per-question raw API responses + timing |
| `<run_dir>/causal_nuscenes/<scene>/<sample>/report.json` | Tier 1: per-sample metrics |
| `<run_dir>/causal_nuscenes/report.json` | Tier 2: per-dataset metrics |
| `<run_dir>/report.json` | Tier 3: per-run metrics |
| `<run_dir>/inference.log` | Structured inference log |

### outputs.jsonl schema
```json
{
  "scene_id": "nuscenes-scene-0159",
  "sample_id": "SAMPLED_0",
  "question_id": "Q1",
  "prompt_id": "000001",
  "raw_output": {"text": "Answer: A\nThe pedestrian in front is directly blocking..."},
  "inference_time_s": 1.87,
  "timestamp": "2026-04-22T10:00:00Z"
}
```

### report.json schema (Tier 3)
```json
{
  "overall": {"accuracy": 0.71, "n": 450, "correct": 320},
  "per_qa_type": {
    "ladder":     {"accuracy": 0.68, "n": 200, "correct": 136},
    "dormant":    {"accuracy": 0.79, "n": 150, "correct": 119},
    "distractor": {"accuracy": 0.65, "n": 100, "correct": 65}
  }
}
```

---

## Model Variant Summary

| Tag | Effective Params | Context | Modalities | VRAM (approx) |
|-----|-----------------|---------|------------|---------------|
| `e2b` | 2.3B (5.1B w/ embeddings) | 128K | Text + Image + Audio | ~6 GB |
| `e4b` | 4.5B (8B w/ embeddings) | 128K | Text + Image + Audio | ~10 GB |
| `26b-a4b` | 3.8B active / 25.2B total (MoE) | 256K | Text + Image | ~20 GB |
| `31b` | 30.7B dense | 256K | Text + Image | ~40 GB |
