# Commands — Impromptu-VLA

Quick reference for development and evaluation commands.

> **Model variants:** Set `MODEL_VARIANT` in `dev.env` to select the checkpoint.
> Available: `3B_Base_finetune`, `3B_AD`, `3B_AD_finetune` (default),
> `7B_Base_finetune`, `7B_AD`, `7B_AD_finetune`.

---

## Prerequisites

```bash
cp evaluation/models/impromptu_vla/dev.env.example evaluation/models/impromptu_vla/dev.env
# Edit dev.env and set:
#   HF_TOKEN, MODEL_VARIANT, WEIGHTS_DIR, RAW_DATA_DIR, CAUSAL_BENCH_DIR, OUTPUT_DIR
```

All `docker compose` commands must be run from the model directory:

```bash
cd evaluation/models/impromptu_vla/
```

The `--env-file` flag lets you keep `dev.env` anywhere on the host:

```bash
docker compose --env-file /path/to/dev.env <subcommand>
```

---

## 1. Unit Tests

```bash
# From repo root (no Docker needed)
/home/shreedhar_kodate/.local/bin/pytest evaluation/models/impromptu_vla/tests/ -v

# Inside container
docker compose run --rm --entrypoint "" inference \
    uv run python -m pytest /workspace/repo/evaluation/models/impromptu_vla/tests/ -v
```

---

## 2. Build Docker Image

```bash
# From repo root
cd ../../..
docker compose -f evaluation/models/impromptu_vla/docker-compose.yml build

# Clean rebuild
docker compose -f evaluation/models/impromptu_vla/docker-compose.yml build --no-cache
```

---

## 3. Download Model Weights

Downloads the selected `MODEL_VARIANT` subfolder from `aaaaaap/ImpromptuVLAModel`:

```bash
docker compose --env-file /path/to/dev.env --profile setup run --rm weight-downloader
```

To download a different variant without editing `dev.env`:

```bash
MODEL_VARIANT=7B_AD_finetune docker compose --env-file /path/to/dev.env --profile setup run --rm weight-downloader
```

---

## 4. Run Original Model Inference (Sanity Check)

Runs `sglang_infer.py` via the original repo script:

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
MODE=subset docker compose --env-file /path/to/dev.env run --rm inference
```

Single scene:

```bash
MODE=single docker compose --env-file /path/to/dev.env run --rm inference
```

Outputs are written to: `$CDB_OUTPUTS_DIR/impromptu_vla_<timestamp>/<dataset>/<scene_id>/<sample_id>/`

### Resume an interrupted run
```bash
RUN_DIR=/workspace/data/cdb_outputs/impromptu_vla_<timestamp> \
docker compose --env-file /path/to/dev.env run --rm inference
```

---

## 6. Post-processing / Metrics

```bash
RUN_DIR=/workspace/data/cdb_outputs/impromptu_vla_20260410_120000 \
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
ls /workspace/weights/ImpromptuVLAModel/   # Verify downloaded variant

# Run sglang inference directly
cd /workspace/model
python train/inference_scripts/sglang_infer.py \
  --model_name_or_path /workspace/weights/ImpromptuVLAModel/3B_AD_finetune \
  --save_name /outputs/impromptu_debug \
  --template qwen2_vl \
  --tensor_parallel_size 1 \
  --data_parallel_size 1

# Alternative: vLLM backend
python train/inference_scripts/vllm_infer.py \
  --model_name_or_path /workspace/weights/ImpromptuVLAModel/3B_AD_finetune \
  --save_name /outputs/impromptu_debug_vllm \
  --template qwen2_vl
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
| `<run_dir>/<dataset>/<scene_id>/<sample_id>/prompts.jsonl` | Per-question prompt payloads |
| `<run_dir>/<dataset>/<scene_id>/<sample_id>/outputs.jsonl` | Per-question raw outputs + timing |
| `<run_dir>/<dataset>/<scene_id>/<sample_id>/report.json` | Tier 1: per-sample metrics |
| `<run_dir>/<dataset>/report.json` | Tier 2: per-dataset metrics |
| `<run_dir>/report.json` | Tier 3: per-run metrics |
| `<run_dir>/inference.log` | Structured inference log |

Where `<run_dir>` = `$CDB_OUTPUTS_DIR/impromptu_vla_<timestamp>/`
