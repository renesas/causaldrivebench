# Commands — <YOUR MODEL NAME>

Quick reference for all development and evaluation commands.
Replace `<model-name>` and `<model_folder>` throughout.

---

## Prerequisites

Copy and fill in `dev.env` before running any service:

```bash
cp evaluation/models/<model_folder>/dev.env.example /path/to/dev.env
# Edit and set: HF_TOKEN, WEIGHTS_DIR, RAW_DATA_DIR, CAUSAL_BENCH_DIR, OUTPUT_DIR
```

**Keep `dev.env` outside the repo** (it contains secrets). Pass it to compose with `--env-file`:

```bash
docker compose --env-file /path/to/dev.env <subcommand>
```

All `docker compose` commands below must be run from the **repo root** (build context)
or the model directory. Using `--env-file` works from anywhere:

```bash
cd evaluation/models/<model_folder>/
# or from repo root:
docker compose -f evaluation/models/<model_folder>/docker-compose.yml --env-file /path/to/dev.env <subcommand>
```

---

## 1. Unit Tests

Run inside the container (recommended — matches runtime deps):

```bash
# Dataloader tests
docker compose run --rm --entrypoint "" inference \
    python -m pytest tests/test_dataloader.py -v

# Postprocessor tests
docker compose run --rm --entrypoint "" inference \
    python -m pytest tests/test_postprocess.py -v

# All tests at once
docker compose run --rm --entrypoint "" inference \
    python -m pytest tests/ -v
```

Or locally if deps are installed (e.g. inside a uv environment):

```bash
python -m pytest tests/ -v
```

---

## 2. Build Docker Image

```bash
docker compose build
```

Force a clean rebuild (no cache):

```bash
docker compose build --no-cache
```

---

## 3. Download Model Weights

### HuggingFace-hosted models

```bash
docker compose --env-file /path/to/dev.env --profile setup run --rm weight-downloader
```

Weights are saved to `WEIGHTS_DIR` (mapped to `/workspace/weights` inside the container).
Requires `HF_TOKEN` if the model repo is gated.

### Manually-hosted models (non-HuggingFace)

Run the weight-downloader to see instructions, then download manually:

```bash
docker compose --env-file /path/to/dev.env --profile setup run --rm weight-downloader
# Follow the printed instructions to download from the model's cloud storage.
```

---

## 4. Run Original Model Inference (Sanity Check)

Runs the model vendor's own sample script to verify the image and weights are correct:

```bash
docker compose --env-file /path/to/dev.env --profile original run --rm original
```

---

## 5. Run Benchmark Inference

Full dataset (all scenes):

```bash
docker compose --env-file /path/to/dev.env run --rm inference
```

Random subset (N scenes):

```bash
MODE=subset SUBSET_SIZE=5 docker compose --env-file /path/to/dev.env run --rm inference
```

Single scene (fastest for debugging):

```bash
MODE=single SCENE=nuscenes-scene-0001 docker compose --env-file /path/to/dev.env run --rm inference
```

Outputs written to `OUTPUT_DIR`:
- `outputs/<dataset>/<scene>_<sample>/outputs.jsonl`
- `outputs/<dataset>/<scene>_<sample>/prompts.jsonl`

---

## 6. Run Post-processing / Metrics

```bash
docker compose --env-file /path/to/dev.env --profile postprocess run --rm postprocess
```

Report written to `OUTPUT_DIR/report.json`.

---

## 7. Interactive Development (Jupyter)

```bash
docker compose --env-file /path/to/dev.env --profile jupyter up jupyter
# Open: http://localhost:8888  (or the port in JUPYTER_PORT)
```

---

## 8. Interactive Shell (Debugging)

```bash
# Drop into a bash shell inside the inference container
docker compose --env-file /path/to/dev.env run --rm --entrypoint bash inference
```

Inside the shell:

```bash
# Verify GPU
nvidia-smi

# Check weights are mounted
ls /workspace/weights

# Run inference manually
python inference.py --mode single --scene nuscenes-scene-0001

# Run postprocess manually
python postprocess.py
```

## 9. Docker Logs

```bash
# Follow logs of a running container
docker compose --env-file /path/to/dev.env logs -f inference

# Last 200 lines from a stopped container
docker compose --env-file /path/to/dev.env logs --tail=200 inference
```

## 10. Exec into a Running Container

```bash
# Find the container name
docker ps

# Exec into it
docker exec -it <container_name> bash
```

---

## 9. Run via Orchestrator (Multi-model)

From the repo root:

```bash
python evaluation/orchestrate.py \
    --model <model_folder> \
    --dataset causal_nuscenes \
    --mode subset \
    --subset-size 10
```

---

## Output File Reference

| File | Description |
|------|-------------|
| `outputs/<dataset>/<scene>_<sample>/prompts.jsonl` | Per-question prompt payloads |
| `outputs/<dataset>/<scene>_<sample>/outputs.jsonl` | Per-question raw outputs + timing |
| `outputs/report.json` | Accuracy metrics by QA type |

### `outputs.jsonl` row schema

```json
{
  "sample_id": "nuscenes-scene-0001/SAMPLED_0",
  "question_id": "q_dl_001",
  "question_type": "dormant",
  "raw_output": "<model's raw text output>",
  "inference_time_s": 2.34,
  "timestamp": "2026-04-04T10:00:00Z"
}
```
