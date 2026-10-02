# Commands — Doe-1

Quick reference for development and evaluation commands.

---

## Prerequisites

Copy and fill in `dev.env` before running any service:

```bash
cp evaluation/models/doe/dev.env.example evaluation/models/doe/dev.env
# Edit dev.env and set: WEIGHTS_DIR, RAW_DATA_DIR, CAUSAL_BENCH_DIR, CDB_OUTPUTS_DIR
# Set MODEL_CHECKPOINT and VQVAE_PATH (manual downloads — see below).
```

All `docker compose` commands below must be run from the model directory:

```bash
cd evaluation/models/doe/
```

The `--env-file` flag lets you keep `dev.env` anywhere on the host:

```bash
docker compose --env-file /path/to/dev.env <subcommand>
```

> **IMPORTANT — Manual weight downloads required:**
>
> Doe-1 weights are **not** on HuggingFace.
>
> 1. **Doe-1 checkpoint**: https://cloud.tsinghua.edu.cn/d/86e47c05830e4caaa40e/
>    → place under `WEIGHTS_DIR/doe1/`
>
> 2. **Chameleon VQVAE tokenizer**: https://github.com/facebookresearch/chameleon
>    → place under `WEIGHTS_DIR/chameleon/tokenizer/`
>
> 3. **OmniDrive nuScenes annotations** (val_infos.pkl):
>    https://github.com/NVlabs/OmniDrive/releases/tag/v1.0

---

## 1. Unit Tests

Run locally (from repo root):

```bash
pytest evaluation/models/doe/tests/ -v
```

Inside the container:

```bash
docker compose run --rm --entrypoint "" inference \
    python -m pytest /workspace/repo/evaluation/models/doe/tests/ -v
```

---

## 2. Build Docker Image

```bash
# From repo root
cd ../../..
docker compose -f evaluation/models/doe/docker-compose.yml build

# Clean rebuild
docker compose -f evaluation/models/doe/docker-compose.yml build --no-cache
```

---

## 3. Download Model Weights

The `weight-downloader` service prints manual download instructions (no automated download available):

```bash
docker compose --env-file /path/to/dev.env --profile setup run --rm weight-downloader
```

---

## 4. Run Original Model Inference (Sanity Check)

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

Outputs are written to:
```
/workspace/data/cdb_outputs/doe_<timestamp>/<dataset>/<scene_id>/<sample_id>/prompts.jsonl
/workspace/data/cdb_outputs/doe_<timestamp>/<dataset>/<scene_id>/<sample_id>/outputs.jsonl
```

### Resume an interrupted run
```bash
RUN_DIR=/workspace/data/cdb_outputs/doe_<timestamp> \
docker compose --env-file /path/to/dev.env run --rm inference
```

---

## 6. Post-processing / Metrics

```bash
RUN_DIR=/workspace/data/cdb_outputs/doe_<timestamp> \
docker compose --env-file /path/to/dev.env --profile postprocess run --rm postprocess
```

Reports are written to:
- `<run_dir>/<dataset>/<scene>/<sample>/report.json` — Tier 1: per-sample
- `<run_dir>/<dataset>/report.json` — Tier 2: per-dataset
- `<run_dir>/report.json` — Tier 3: per-run

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
ls /workspace/weights/doe1/          # Verify Doe-1 checkpoint
ls /workspace/weights/chameleon/     # Verify VQVAE tokenizer

# Run original inference/eval.py directly
cd /workspace/model
CUDA_VISIBLE_DEVICES=0 python inference/eval.py \
  --anno_path /workspace/data/raw_data/nuscenes/val_infos.pkl \
  --nusc_path /workspace/data/raw_data/nuscenes \
  --save_path /workspace/data/cdb_outputs/doe_debug \
  --model_path /workspace/weights/doe1/model.pth \
  --task qa
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
| `/workspace/data/cdb_outputs/doe_<timestamp>/<dataset>/<scene_id>/<sample_id>/prompts.jsonl` | Per-question prompt payloads |
| `/workspace/data/cdb_outputs/doe_<timestamp>/<dataset>/<scene_id>/<sample_id>/outputs.jsonl` | Per-question raw outputs + timing |
| `/workspace/data/cdb_outputs/doe_<timestamp>/<dataset>/<scene_id>/<sample_id>/report.json` | Tier 1: per-sample metrics |
| `/workspace/data/cdb_outputs/doe_<timestamp>/<dataset>/report.json` | Tier 2: per-dataset metrics |
| `/workspace/data/cdb_outputs/doe_<timestamp>/report.json` | Tier 3: per-run metrics |
| `/workspace/data/cdb_outputs/doe_<timestamp>/inference.log` | Structured inference log |
