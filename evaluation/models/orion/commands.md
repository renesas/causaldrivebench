# Commands — Orion

Quick reference for development and evaluation commands.

> **GPU memory requirements:**
> - FP32 inference: 32 GB+ (tested on A100)
> - FP16 inference: 17 GB+ (set `MODEL_CONFIG` to `orion_stage3_fp16.py`)

---

## Prerequisites

```bash
cp evaluation/models/orion/dev.env.example evaluation/models/orion/dev.env
# Edit dev.env and set:
#   HF_TOKEN, WEIGHTS_DIR, RAW_DATA_DIR, CAUSAL_BENCH_DIR, CDB_OUTPUTS_DIR
#   MODEL_CHECKPOINT, PRETRAIN_QFORMER_PATH, EVA_PROJ_PATH
```

All `docker compose` commands must be run from the model directory:

```bash
cd evaluation/models/orion/
```

The `--env-file` flag lets you keep `dev.env` anywhere on the host:

```bash
docker compose --env-file /path/to/dev.env <subcommand>
```

---

## 1. Unit Tests

```bash
# From repo root (no Docker required):
/home/shreedhar_kodate/.local/bin/pytest evaluation/models/orion/tests/ -v

# Dataloader tests only
/home/shreedhar_kodate/.local/bin/pytest evaluation/models/orion/tests/test_dataloader.py -v

# Postprocessor tests only
/home/shreedhar_kodate/.local/bin/pytest evaluation/models/orion/tests/test_postprocess.py -v

# Inside the container:
docker compose run --rm --entrypoint "" inference \
    uv run python -m pytest /workspace/repo/evaluation/models/orion/tests/ -v
```

---

## 2. Build Docker Image

```bash
# From repo root (builds CUDA extensions — can take 10-20 min)
cd ../../..
docker compose -f evaluation/models/orion/docker-compose.yml build

# Clean rebuild
docker compose -f evaluation/models/orion/docker-compose.yml build --no-cache
```

---

## 3. Download Model Weights

Downloads `poleyzdk/Orion` and `exiawsh/pretrain_qformer` from HuggingFace:

```bash
docker compose --env-file /path/to/dev.env --profile setup run --rm weight-downloader
```

> **Manual step required:** Download `eva02_petr_proj.pth` from
> https://github.com/NVlabs/OmniDrive/releases/tag/v1.0
> and place it at `WEIGHTS_DIR/eva02_petr_proj.pth`.

---

## 4. Run Original Model Inference (Sanity Check)

Runs `orion_dist_eval.sh` with FP16 config on 1 GPU:

```bash
docker compose --env-file /path/to/dev.env --profile original run --rm original
```

---

## 5. Run Benchmark Inference

Full dataset:

```bash
docker compose --env-file /path/to/dev.env run --rm inference
```

Random subset:

```bash
MODE=subset SUBSET_SIZE=5 docker compose --env-file /path/to/dev.env run --rm inference
```

Single scene:

```bash
MODE=single SCENE=nuscenes-scene-0001 docker compose --env-file /path/to/dev.env run --rm inference
```

The inference service creates a timestamped run directory:
```
/workspace/data/cdb_outputs/orion_<YYYYMMDD_HHMMSS>/
```

### Resume an interrupted run
```bash
RUN_DIR=/workspace/data/cdb_outputs/orion_<timestamp> \
docker compose --env-file /path/to/dev.env run --rm inference
```

---

## 6. Post-processing / Metrics

```bash
RUN_DIR=/workspace/data/cdb_outputs/orion_20260410_120000 \
  docker compose --env-file /path/to/dev.env --profile postprocess run --rm postprocess
```

Reports are written to:
- `<run_dir>/<dataset>/<scene>/<sample>/report.json` — Tier 1: per-sample
- `<run_dir>/<dataset>/report.json` — Tier 2: per-dataset
- `<run_dir>/report.json` — Tier 3: per-run

### Optional: LLM judge scoring

To score open-ended answers with an LLM judge, add `--use-llm-judge` and pick a
dispatch path:

```bash
# Anthropic Message Batches API (50% cheaper; provider must be anthropic)
RUN_DIR=/workspace/data/cdb_outputs/orion_<timestamp> \
  docker compose --env-file /path/to/dev.env --profile postprocess run --rm postprocess \
    --use-llm-judge --judge-batch-api --judge-provider anthropic

# Concurrent thread-pool (all providers; tune to provider rate limit)
RUN_DIR=/workspace/data/cdb_outputs/orion_<timestamp> \
  docker compose --env-file /path/to/dev.env --profile postprocess run --rm postprocess \
    --use-llm-judge --judge-provider azure_foundry --judge-concurrency 20
```

`--judge-concurrency` defaults to 20; set to 1 for serial. Ignored when
`--judge-batch-api` is set. Provider API keys come from the env file
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_FOUNDRY_API_KEY` +
`FOUNDRY_BASE_URL`, or `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`).

---

## 7. Interactive Development (Jupyter)

```bash
docker compose --env-file /path/to/dev.env --profile jupyter up jupyter
# Open: http://localhost:8888
```

---

## 8. Interactive Shell (Debugging)

```bash
docker compose --env-file /path/to/dev.env run --rm --entrypoint bash inference
```

Inside the shell:

```bash
nvidia-smi
ls /workspace/weights/Orion/         # Verify Orion checkpoint
ls /workspace/weights/pretrain_qformer/

# Run benchmark inference manually
uv run python inference.py --mode single --scene nuscenes-scene-0001 \
  --output-base /workspace/data/cdb_outputs \
  --model-config /workspace/model/projects/configs/Orion/orion_stage3_fp16.py

# Run distributed eval directly (single GPU)
cd /workspace/model
bash adzoo/orion/orion_dist_eval.sh \
  adzoo/orion/configs/orion_stage3_fp16.py \
  /workspace/weights/Orion/Orion.pth \
  1
```

---

## 9. Docker Logs

```bash
docker compose --env-file /path/to/dev.env logs -f inference
docker compose --env-file /path/to/dev.env logs --tail=100 inference
```

---

## 10. Exec into a Running Container

```bash
docker ps
docker exec -it <container_name> bash
```

---

## Output File Reference

| File | Description |
|------|-------------|
| `/workspace/data/cdb_outputs/orion_<timestamp>/` | Run directory |
| `/workspace/data/cdb_outputs/orion_<timestamp>/<dataset>/<scene_id>/<sample_id>/prompts.jsonl` | Per-question prompt payloads |
| `/workspace/data/cdb_outputs/orion_<timestamp>/<dataset>/<scene_id>/<sample_id>/outputs.jsonl` | Per-question raw outputs + timing |
| `/workspace/data/cdb_outputs/orion_<timestamp>/<dataset>/<scene_id>/<sample_id>/report.json` | Tier 1: per-sample metrics |
| `/workspace/data/cdb_outputs/orion_<timestamp>/<dataset>/report.json` | Tier 2: per-dataset metrics |
| `/workspace/data/cdb_outputs/orion_<timestamp>/report.json` | Tier 3: per-run metrics |
| `/workspace/data/cdb_outputs/orion_<timestamp>/inference.log` | Structured inference log |
