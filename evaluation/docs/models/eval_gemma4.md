# Gemma 4 Integration — Design Decisions and Notes

**Model**: Google Gemma 4 family (E2B, E4B, 26B A4B, 31B Dense)  
**Source**: `hub.docker.com/r/ai/gemma4`  
**Modalities**: Text + Image (all variants); + Audio (E2B, E4B only)  
**Context**: 128K tokens (E2B, E4B) / 256K tokens (26B, 31B)  
**Integration path**: OpenAI-compatible HTTP API (not in-process model loading)

---

## Architecture Overview

**Docker Desktop (Mac/Windows):**
```
┌─────────────────────────────────────────────────────────────┐
│  docker compose up                                           │
│  ┌─────────────────────┐   HTTP/OpenAI API  ┌────────────┐ │
│  │   gemma4-server      │ ◄──────────────── │ inference  │ │
│  │  ai/gemma4:<tag>     │   :8080/v1        │ container  │ │
│  │  (GPU, llama.cpp)    │                   │ (no GPU)   │ │
│  └─────────────────────┘                   └────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Linux (no Docker Desktop) — actual production setup:**
```
Host OS
  ├── docker-model-plugin daemon  :12434  (model runner, serves ai/gemma4 weights)
  │
  └── docker compose (with override file)
        ┌─────────────────────────┐   healthcheck   ┌─────────────────────────┐
        │  gemma4-server (curl    │ ─────────────►  │  localhost:12434/v1     │
        │  keepalive container)   │   pings daemon  │  model runner daemon    │
        └─────────────────────────┘                 └─────────────────────────┘
                                                             ▲
        ┌─────────────────────────┐   HTTP/OpenAI API       │
        │  inference container    │ ───────────────────────►│
        │  (Python wrapper)       │  host.docker.internal   │
        └─────────────────────────┘      :12434/v1          │
```

Unlike alpamayo_1_5 (which loads model weights directly in Python), Gemma 4
runs as a **separate model server** managed by the Docker model runner daemon.
The evaluation wrapper only needs network access to the server — no GPU, no
model weights in the inference container.

---

## Key Design Decisions

### 1. API-Based Inference vs. In-Process Loading

**Decision**: Use Docker's `ai/gemma4` image as a model server; call via OpenAI-compatible API.

**Why**: The `ai/gemma4` Docker image packages the model + llama.cpp runtime as a
self-contained server. Attempting to load Gemma 4 via `transformers` from HuggingFace
would require a separate weights download and is not the distribution method Docker
provides. The API approach is also more robust: the model server manages memory, VRAM,
and KV cache independently of our Python process.

**Trade-off**: Round-trip HTTP latency (~10–50 ms/call on localhost) vs. Python function
call latency. For VQA with large images, model compute time (1–30s) dominates.

---

### 2. `load_model()` Repurposed as API Client Initialisation

**Decision**: `load_model(weights_path)` initialises an `openai.OpenAI` client
instead of loading local weights. The `weights_path` argument (kept for interface
compatibility with the orchestrator) is reinterpreted as an optional API base URL.

**Why**: The abstract base class requires `load_model(weights_path: str)`. Repurposing
this argument avoids a base class change while preserving the CLI convention. The
`--api-base-url` argument is the preferred way to specify the endpoint; `--weights-path`
is an accepted alias for backwards compatibility with `orchestrate.py`.

---

### 3. Multi-Variant Support via `GEMMA4_MODEL_TAG`

**Decision**: All four model variants share a single docker-compose file, dataloader,
inference, and postprocess script. The selected variant is controlled by `GEMMA4_MODEL_TAG`
(e.g. `e2b`, `e4b`, `26b-a4b`, `31b`). The run directory name is derived from the tag
(`gemma4_e2b_<timestamp>`, etc.).

**Why**: The evaluation code is identical across variants — only the model image and
(optionally) image resolution differ. A shared compose file reduces duplication and
makes cross-variant comparison straightforward: run the same config with different tags
and compare run directories side by side.

**How to use**: Set `GEMMA4_MODEL_TAG` in `dev.env` before each run. Four variants
registered in `eval_config.yaml` as `gemma4_e2b`, `gemma4_e4b`, `gemma4_26b_a4b`, and
`gemma4_31b`.

---

### 4. Image Encoding Strategy

**Decision**: Images are NOT base64-encoded in `prompts.jsonl`. Relative paths are
stored (same as alpamayo_1_5). `run_single()` loads, resizes, and encodes images at
call time.

**Why**: 16 images × ~100 KB base64 each = ~1.6 MB per question in JSONL. For a
benchmark with hundreds of questions this would create multi-GB prompt files. Storing
paths keeps JSONL compact; the model server accepts the base64 in the HTTP request
without needing to save it to disk.

**Image resolution**: Default 448×448 pixels. At this size, 16 images consume roughly
4,000–5,000 image tokens (at ~280 tokens/image), well within the 128K context limit.
Override with `IMAGE_SIZE=224` for faster lower-VRAM runs or `IMAGE_SIZE=672` for
higher visual quality.

---

### 5. Camera-Major Image Ordering with Text Labels

**Decision**: Images are grouped camera-first (all 4 timesteps for each camera before
moving to the next), with a text label ("Front camera views (T-1.5s → T=0):") inserted
before each group.

**Why**: Camera-major ordering matches CausalDriveBench convention (same as alpamayo_1_5).
Text labels provide spatial context that pure image position cannot convey in a flat
message array. Without labels, the model has no way to know which of the 16 images is
from which camera.

**Alternative considered**: Time-major ordering (all cameras at T-1.5s, then T-1.0s, …).
Rejected — camera-major is the convention established by alpamayo models; keeping it
consistent enables direct comparison of prompts.

---

### 6. Thinking Mode Enabled

**Decision**: The system prompt ends with `<|think|>` to enable Gemma 4's internal
reasoning mode.

**Why**: Causal reasoning benchmarks explicitly test *why* events happen, not just *what*
happened. Internal chain-of-thought helps the model reason about causal chains before
committing to an answer. The Docker Hub page documents `<|think|>` as the mechanism to
activate this.

**Post-processing**: The base class `parse_answer()` already strips `<think>...</think>`
blocks. `Gemma4Postprocessor._preprocess_text()` additionally strips the control-token
variant `<|thinking|>...</|thinking|>` in case the model uses that form.

**Critical**: Thinking mode consumes tokens from the same `MAX_TOKENS` budget. With
`MAX_TOKENS=512`, the entire budget is used by the reasoning trace and `content` is
empty. Set `MAX_TOKENS=10000` (see `dev.env`). See Known Issues #2 for diagnosis steps.

---

### 7. State Context as Text (Not Trajectory Arrays)

**Decision**: Ego history and agent positions are formatted as human-readable text in
`state_summary` and `ego_history_text`, included in the text portion of the vision
message.

**Why**: Gemma 4 has no special trajectory input head (unlike Alpamayo 1.5 which accepts
`ego_history_xyz` arrays). Text is the natural modality for structured tabular data in
a general-purpose VLM. The base dataloader's `interpolate_ego_history()` is not called
because Gemma 4 doesn't benefit from a densified position array.

---

### 8. Lightweight Evaluation Container

**Decision**: The inference container uses `python:3.12-slim` with only `openai`, `pillow`,
`numpy`, and `tqdm`. No CUDA, no HuggingFace, no model-specific packages.

**Why**: All GPU work is delegated to `gemma4-server`. The evaluation wrapper is a thin
HTTP client. Keeping the image small (~300 MB) speeds up builds and reduces CI overhead.
`pip` is used instead of `uv` because there is no complex dependency tree to resolve.

---

### 9. Weight Downloader via Docker Socket

**Decision**: `weight-downloader` uses `docker:latest` with `/var/run/docker.sock`
mounted to run `docker pull ai/gemma4:<tag>`.

**Why**: Gemma 4 weights are distributed through the Docker model runner — not as
standard Docker Hub images. The "download" step uses `docker model pull`, not
`docker pull`. Mounting the Docker socket is the standard pattern for compose services
that need to interact with the host Docker daemon.

**Security note**: Mounting the socket grants the container root-equivalent access to
the host Docker daemon. Only run `weight-downloader` from a trusted environment.

**Linux note**: On Linux without Docker Desktop, skip `weight-downloader` entirely.
Instead pull manually: `docker model pull ai/gemma4` (E2B) or via HuggingFace GGUF
paths for other variants (see Linux Setup section). `docker pull ai/gemma4:<tag>` will
return 404 — it goes to Docker Hub, not the model runner registry.

---

## Learnings

1. **Docker model runner port uncertainty**: The exact port and API path of `ai/gemma4`
   images depends on the Docker runtime version. `GEMMA4_API_BASE_URL` and `GEMMA4_PORT`
   env vars allow the endpoint to be configured without rebuilding. The `original` service
   and healthcheck probe `/v1/models` to catch misconfiguration early.

2. **Image size vs. context window**: 16 images at 448×448 with 280 tokens/image uses
   ~4,480 tokens. At 560 tokens/image that doubles to ~8,960 — still fine for 128K
   context but noticeably slower. Monitor token usage in inference.log if hitting
   context limits.

3. **Temperature 1.0 matters**: Google's recommendation for Gemma 4 sampling is
   temperature=1.0 with top_p=0.95, top_k=64. Deviating from this (especially lowering
   temperature) can reduce diversity in reasoning and slightly degrade MCQ accuracy on
   harder causal questions.

4. **Model name in API calls**: The `model` parameter must match what the model runner
   registers, which differs from the pull command. E2B is pulled as `ai/gemma4` but
   registers as `docker.io/ai/gemma4:latest`. HuggingFace GGUF models register as
   `hf.co/<org>/<repo>:Q4_K_M`. Always verify with `docker model list` after pulling
   and set `GEMMA4_API_MODEL_NAME` in `dev.env` to the registered name.

---

## Recommendations

1. **Start with E2B for development**: Fastest inference, lowest VRAM. Validate the
   prompt format and scoring pipeline before running the larger variants.

2. **Use `--mode subset --subset-size 20` first**: Full benchmark runs can take hours
   even for API-based inference (per-question sequential processing). A 20-scene subset
   gives a quick accuracy estimate.

3. **Enable thinking for ladder/dormant, disable for distractor**: Thinking mode costs
   extra tokens and latency. For distractor questions (which test recognition of
   irrelevant agents), thinking may not improve accuracy. Consider running two variants:
   one with thinking enabled and one without, then compare.

4. **Cross-variant comparison**: Use the same `CAUSAL_BENCH_DIR` and `--mode full` across
   all four variants for a fair comparison. The `eval_config.yaml` registers all four
   variants; the orchestrator can run them sequentially.

5. **Monitor server VRAM**: The 31B dense model requires ~40 GB VRAM. Ensure the host
   GPU has sufficient memory before pulling that image. The 26B A4B MoE model achieves
   similar quality with ~20 GB due to sparse expert activation.

---

## Linux Setup (No Docker Desktop)

On Linux servers without Docker Desktop, Gemma 4 is served via the
**Docker model runner** daemon — a separate component from the standard Docker engine.

### Step 1: Install docker-model-plugin (one-time)

```bash
sudo apt-get install -y docker-model-plugin
```

This installs the model runner daemon. It starts automatically and listens on
`localhost:12434` (not the normal Docker port 2376). Verify with:

```bash
curl http://localhost:12434/v1/models
```

### Step 2: Pull model weights

**E2B** (default, ~6 GB):
```bash
docker model pull ai/gemma4
# Registers as: docker.io/ai/gemma4:latest
```

**E4B** (~5 GB / 4.62 GiB, 7.52B Q4_K_M):
```bash
docker model pull hf.co/unsloth/gemma-4-E4B-it-GGUF
# Registers as: huggingface.co/unsloth/gemma-4-e4b-it-gguf
```

**26B-A4B MoE** (~16 GB / 15.70 GiB, 25.23B Q4_K_M):
```bash
docker model pull hf.co/unsloth/gemma-4-26B-A4B-it-GGUF
# Registers as: huggingface.co/unsloth/gemma-4-26b-a4b-it-gguf
```

**31B Dense** (~17 GB / 17.05 GiB, 30.70B Q4_K_M):
```bash
docker model pull hf.co/unsloth/gemma-4-31B-it-GGUF
# Registers as: huggingface.co/unsloth/gemma-4-31b-it-gguf
```

> **Note:** `docker model pull` (model runner) is NOT the same as `docker pull` (standard
> registry). The `ai/gemma4:e2b` / `e4b` / `26b-a4b` / `31b` tags do not exist on Docker
> Hub — they are model runner tags. Running `docker pull ai/gemma4:e2b` returns 404.

After pulling, verify registration:
```bash
docker model list
```

The API model name reported here is what must match `GEMMA4_API_MODEL_NAME` in `dev.env`.
For e2b it is `docker.io/ai/gemma4:latest`. For HuggingFace GGUF models, it is typically
`hf.co/<org>/<repo>:Q4_K_M` (check `docker model list` output to confirm exact name).

### Step 3: Use the compose override file

The standard `docker-compose.yml` uses `provider: type: model` to declare `gemma4-server`.
On Linux with Docker Compose v5+, this creates **no real container**, so `depends_on:
service_healthy` never resolves and inference never starts.

Use the override file that replaces `gemma4-server` with a keepalive curl container
whose healthcheck pings the model runner daemon directly:

```bash
# Override file location:
/datadrive/code/ssk/neurips_expts/eval/docker-compose.gemma4.model-runner.yml
```

Include this override on **every** inference and postprocess run:

```bash
cd /datadrive/code/ssk/neurips-dataset
MODE=single SCENE=nuscenes-scene-0001 \
docker compose \
  -f evaluation/models/gemma4/docker-compose.yml \
  -f /datadrive/code/ssk/neurips_expts/eval/docker-compose.gemma4.model-runner.yml \
  --env-file /datadrive/code/env_files/models/gemma4/dev.env \
  run --rm inference
```

---

## Known Issues and Fixes

### Issue 1: `ai/gemma4:e2b` tag returns 404

The named variant tags (`e2b`, `e4b`, `26b-a4b`, `31b`) are model runner tags, not
Docker Hub registry tags. `docker pull ai/gemma4:e2b` hits Docker Hub and returns 404.

**Fix:** Use `docker model pull ai/gemma4` (no tag) for E2B — it downloads the default
e2b variant, registering as `docker.io/ai/gemma4:latest`. For other variants, use the
HuggingFace GGUF paths shown in the Linux Setup section above.

---

### Issue 2: Empty model responses with `max_tokens=512`

The system prompt ends with `<|think|>`, enabling Gemma 4 chain-of-thought mode. The
model generates a reasoning trace in `reasoning_content` before producing `content`.
With `MAX_TOKENS=512`, the entire budget is consumed by the reasoning trace; the final
answer in `content` is an empty string. The `finish_reason` is `"length"`.

**Fix:** Set `MAX_TOKENS=10000` in `dev.env`. Verified minimum: 2048 is sufficient for
single questions. 10000 provides comfortable headroom for complex multi-step reasoning.

---

### Issue 3: Docker Compose v5 `provider: type: model` creates no container

In Docker Compose v5.0.2, `provider: type: model` blocks declare a model dependency but
create no running container. The `depends_on: gemma4-server: condition: service_healthy`
check can never succeed, so the inference container never starts.

**Fix:** The override file (`docker-compose.gemma4.model-runner.yml`) replaces
`gemma4-server` with a `curlimages/curl` keepalive container. Its healthcheck pings
`http://host.docker.internal:12434/v1/models` (the model runner API) directly, so the
dependency resolves correctly when the model runner daemon is ready.

---

### Issue 4: Missing `pyyaml` — `ModuleNotFoundError: No module named 'yaml'`

The evaluation wrapper imports `CDBConfig` which uses `yaml`. `pyyaml` was not listed
in `requirements.in`.

**Fix:** Added `pyyaml>=6.0` to `evaluation/models/gemma4/requirements.in` and
`requirements.txt`. Rebuild with `docker compose build --no-cache` after editing
requirements files.

---

### Issue 5: Missing `anthropic` — `ModuleNotFoundError: No module named 'anthropic'`

The postprocessor invokes the LLM judge (`USE_LLM_JUDGE=true` with `azure_foundry`
provider), which requires the `anthropic` SDK. It was not listed in requirements.

**Fix:** Added `anthropic>=0.40` to `evaluation/models/gemma4/requirements.in` and
`requirements.txt`. Rebuild with `docker compose build --no-cache`.

---

### Issue 6: LLM judge 404 `api_not_supported` — double `/v1` in URL

**Symptom:** All judge calls fail silently; `llm_judge_score` is absent from reports.  
**Root cause:** `FOUNDRY_BASE_URL` ends in `/v1`. `AnthropicFoundry(base_url=FOUNDRY_BASE_URL)`
appends its own `/v1/messages` → URL becomes `.../anthropic/v1/v1/messages` → 404.

**Fix:** `evaluation/common/llm_judge.py` strips the trailing `/v1` before passing to the SDK.

### Issue 7: LLM judge `ValueError: base_url and resource are mutually exclusive`

**Symptom:** `ValueError: base_url and resource are mutually exclusive` when both
`ANTHROPIC_FOUNDRY_RESOURCE` env var and an explicit `base_url=` parameter are present.  
**Root cause:** `ANTHROPIC_FOUNDRY_RESOURCE` may be set globally on the host. Passing
`base_url=` at the same time triggers this SDK conflict.  
**Fix:** `llm_judge.py` now checks for `ANTHROPIC_FOUNDRY_RESOURCE` first; if set, calls
`AnthropicFoundry(api_key=...)` with no `base_url` (SDK reads resource from env). All env
files must include `ANTHROPIC_FOUNDRY_RESOURCE=your-resource`. Ensure
`ANTHROPIC_FOUNDRY_RESOURCE` is in the compose `common-env` section.

### Issue 8: `--force` flag cannot be passed via `docker compose run`

**Symptom:** `docker compose run --rm postprocess -- --force` → OCI error: `exec: "--"`.  
**Root cause:** Docker Compose passes extra args as the entrypoint, not appended to command.  
**Fix:** `evaluation/common/base_postprocess.py` supports `FORCE` env var:
`default=os.environ.get("FORCE", "").lower() == "true"`. Use `FORCE=true RUN_DIR=... docker compose ...`.

---

### Issue 9: `nuscenes-scene-0512` has no QA data in 20260423 dataset

The scene directory exists but `qa/` folders are absent from all 4 SAMPLED_* subdirs.
The QA generation pipeline has not processed this scene in the 20260423 dataset.

**Fix:** Use `nuscenes-scene-0001` for development (complete QA coverage).

---

## Commands Quick Reference

### Build evaluation image (one-time per code change)

```bash
cd /datadrive/code/ssk/neurips-dataset
docker compose \
  -f evaluation/models/gemma4/docker-compose.yml \
  --env-file /datadrive/code/env_files/models/gemma4/dev.env \
  build
```

### Inference — single scene

```bash
cd /datadrive/code/ssk/neurips-dataset
MODE=single SCENE=nuscenes-scene-0001 \
docker compose \
  -f evaluation/models/gemma4/docker-compose.yml \
  -f /datadrive/code/ssk/neurips_expts/eval/docker-compose.gemma4.model-runner.yml \
  --env-file /datadrive/code/env_files/models/gemma4/dev.env \
  run --rm inference
```

### Inference — full dataset

```bash
cd /datadrive/code/ssk/neurips-dataset
MODE=full \
docker compose \
  -f evaluation/models/gemma4/docker-compose.yml \
  -f /datadrive/code/ssk/neurips_expts/eval/docker-compose.gemma4.model-runner.yml \
  --env-file /datadrive/code/env_files/models/gemma4/dev.env \
  run --rm inference
```

### Postprocess

```bash
cd /datadrive/code/ssk/neurips-dataset
RUN_DIR=/workspace/data/cdb_outputs/gemma4_e2b_<YYYYMMDD_HHMMSS> \
MODE=full \
docker compose \
  -f evaluation/models/gemma4/docker-compose.yml \
  -f /datadrive/code/ssk/neurips_expts/eval/docker-compose.gemma4.model-runner.yml \
  --env-file /datadrive/code/env_files/models/gemma4/dev.env \
  --profile postprocess run --rm postprocess
```

### Switch model variant

Edit `dev.env`:
```
GEMMA4_MODEL_TAG=e4b
GEMMA4_API_MODEL_NAME=hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M   # verify via docker model list
```

Then re-run inference and postprocess with the same compose commands above. The run
directory name will automatically include the tag (`gemma4_e4b_<timestamp>`).

### Check model runner health

```bash
curl http://localhost:12434/v1/models
docker model list
```

### Check if thinking mode is consuming all tokens (diagnosis)

```bash
curl -s http://localhost:12434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"docker.io/ai/gemma4:latest","max_tokens":50,"messages":[{"role":"user","content":"Say hi"}]}' \
  | python3 -m json.tool | grep -E "content|finish_reason|reasoning"
```
If `content` is empty and `finish_reason` is `"length"`, increase `MAX_TOKENS`.

See `evaluation/models/gemma4/commands.md` for legacy Docker-Desktop commands and
`/datadrive/code/ssk/neurips_expts/eval/cmd_gemma4.md` for per-variant run commands.

---

## Files Created

| File | Purpose |
|------|---------|
| `Dockerfile` | Lightweight Python 3.12 evaluation wrapper (no GPU/CUDA) |
| `dev.env.example` | Environment variable template (model tag, API URL, paths) |
| `docker-compose.yml` | Services: gemma4-server, inference, weight-downloader, original, postprocess |
| `commands.md` | Full Docker command reference |
| `system_prompt.txt` | System prompt with `<|think|>` for reasoning mode |
| `dataloader.py` | Camera-major image paths + text state summary |
| `inference.py` | OpenAI-client-based inference, base64 image encoding |
| `postprocess.py` | Answer parsing with Gemma 4 thinking-token cleanup |
| `original_test.py` | API connectivity sanity check |
| `requirements.in` | Unpinned source dependencies |
| `requirements.txt` | Pinned dependencies for reproducible builds |
