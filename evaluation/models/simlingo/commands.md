# Commands — SimLingo

Quick reference for development and evaluation commands.

> **NOTE:** SimLingo's closed-loop evaluation requires CARLA 0.9.15 (not included
> in this image). This setup covers language-model inference only.
> The InternVL2-1B vision backbone is downloaded automatically from HuggingFace
> at first inference if not already cached.

---

## Prerequisites

```bash
cp evaluation/models/simlingo/dev.env.example evaluation/models/simlingo/dev.env
# Edit dev.env and set: HF_TOKEN, WEIGHTS_DIR, RAW_DATA_DIR, CAUSAL_BENCH_DIR, CDB_OUTPUTS_DIR
```

All `docker compose` commands must be run from the model directory:

```bash
cd evaluation/models/simlingo/
```

The `--env-file` flag lets you keep `dev.env` anywhere on the host:

```bash
docker compose --env-file /path/to/dev.env <subcommand>
```

---

## 1. Unit Tests

Run locally (from repo root):

```bash
pytest evaluation/models/simlingo/tests/ -v
```

Inside the container:

```bash
docker compose run --rm --entrypoint "" inference \
    uv run python -m pytest /workspace/repo/evaluation/models/simlingo/tests/ -v
```

---

## 2. Build Docker Image

```bash
# From repo root
cd ../../..
docker compose -f evaluation/models/simlingo/docker-compose.yml build

# Clean rebuild
docker compose -f evaluation/models/simlingo/docker-compose.yml build --no-cache
```

---

## 3. Download Model Weights

Downloads `RenzKa/simlingo` from HuggingFace:

```bash
docker compose --env-file /path/to/dev.env --profile setup run --rm weight-downloader
```

> The `OpenGVLab/InternVL2-1B` vision backbone is downloaded automatically
> by the transformers library at first inference (requires `HF_TOKEN` if gated).

---

## 4. Run Original Model Inference (Sanity Check)

Runs the language evaluation script (no CARLA required):

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
/workspace/data/cdb_outputs/simlingo_<timestamp>/<dataset>/<scene_id>/<sample_id>/prompts.jsonl
/workspace/data/cdb_outputs/simlingo_<timestamp>/<dataset>/<scene_id>/<sample_id>/outputs.jsonl
```

### Resume an interrupted run
```bash
RUN_DIR=/workspace/data/cdb_outputs/simlingo_<timestamp> \
docker compose --env-file /path/to/dev.env run --rm inference
```

---

## 6. Post-processing / Metrics

```bash
RUN_DIR=/workspace/data/cdb_outputs/simlingo_<timestamp> \
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
ls /workspace/weights/simlingo/      # Verify weights

# Run language eval directly
cd /workspace/model
python simlingo_training/eval.py \
  model_path=/workspace/weights/simlingo
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
| `/workspace/data/cdb_outputs/simlingo_<timestamp>/<dataset>/<scene_id>/<sample_id>/prompts.jsonl` | Per-question prompt payloads |
| `/workspace/data/cdb_outputs/simlingo_<timestamp>/<dataset>/<scene_id>/<sample_id>/outputs.jsonl` | Per-question raw outputs + timing |
| `/workspace/data/cdb_outputs/simlingo_<timestamp>/<dataset>/<scene_id>/<sample_id>/report.json` | Tier 1: per-sample metrics |
| `/workspace/data/cdb_outputs/simlingo_<timestamp>/<dataset>/report.json` | Tier 2: per-dataset metrics |
| `/workspace/data/cdb_outputs/simlingo_<timestamp>/report.json` | Tier 3: per-run metrics |
| `/workspace/data/cdb_outputs/simlingo_<timestamp>/inference.log` | Structured inference log |
