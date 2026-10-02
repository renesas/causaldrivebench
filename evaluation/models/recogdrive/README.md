# ReCogDrive — CausalDriveBench Evaluation

End-to-end evaluation of [ReCogDrive](https://github.com/xiaomi-research/recogdrive) (InternVL3-8B fine-tuned on driving data) on the CausalDriveBench benchmark.

---

## Architecture

```
InternVL3-8B (ViT + Qwen2 LM)
   fine-tuned on 582k driving samples
          ↓
    model.chat()  →  answer text
```

ReCogDrive has two components: a **VLM backbone** (InternVL3 fine-tuned for driving scene understanding) and a **diffusion planner head** (for NAVSIM trajectory prediction). For VQA evaluation we use the VLM backbone directly via `AutoModel + model.chat()`.

Camera setup: 4 cameras (`cam_front`, `cam_front_left`, `cam_front_right`, `cam_back`), 4 timesteps (`Tm1p5, Tm1p0, Tm0p5, Tp0p0`) = **16 images per sample**.
Images are resized to 448×448 with ImageNet normalisation.

---

## Environment Setup

### Prerequisites

- CUDA-capable GPU with SM ≥ 8 (Ampere or newer) for flash-attn — tested on A40 (46 GB, CUDA 12.x)
- Miniconda installed at `/root/miniconda3`
- ReCogDrive repo cloned at `/workspace/recogdrive`

### Create the conda environment

```bash
conda create -n recogdrive python=3.10 -y
conda activate recogdrive

# PyTorch with CUDA 12.4
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

# Core VLM dependencies
pip install transformers==4.40.2 tokenizers sentencepiece accelerate einops timm

# flash-attn (requires --no-build-isolation to find torch)
pip install flash-attn --no-build-isolation

# InternVL3 model code
pip install -e /workspace/recogdrive/internvl_chat --no-deps

# Evaluation framework
pip install jsonlines Pillow
```

### Download weights

The VLM weights are public (no HuggingFace token required):

```bash
huggingface-cli download owl10/ReCogDrive-VLM-8B \
    --local-dir /workspace/weights/ReCogDrive-VLM-8B
```

The `-VLM-8B` variant is the InternVL3-8B backbone fine-tuned on ~582k driving samples for 3 epochs. The `-8B-RL` / `-8B-IL` variants only contain the diffusion planner checkpoint and are **not** suitable for VQA.

---

## Running Inference

### Option A — Docker (recommended for production)

```bash
cp dev.env.example dev.env
# Set: WEIGHTS_DIR, CAUSAL_BENCH_DIR, CDB_OUTPUTS_DIR, CDB_REPORTS_DIR
```

```bash
# Full dataset
docker compose --env-file dev.env run --rm inference

# Subset (first N samples)
MODE=subset SUBSET_SIZE=5 docker compose --env-file dev.env run --rm inference

# Single scene
MODE=single SCENE=nuscenes-scene-0001 docker compose --env-file dev.env run --rm inference

# Resume interrupted run
RUN_DIR=/workspace/data/cdb_outputs/recogdrive_<timestamp> \
  docker compose --env-file dev.env run --rm inference
```

### Option B — Bare-metal (conda env)

```bash
cd /workspace/neurips-dataset/evaluation/models/recogdrive

/root/miniconda3/envs/recogdrive/bin/python inference.py \
  --bench-dir /workspace/nuscenes_qa_test \
  --weights-path /workspace/weights/ReCogDrive-VLM-8B \
  --output-base /workspace/cdb_outputs \
  --device cuda
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--weights-path` | from config | Path to local `ReCogDrive-VLM-8B` directory |
| `--mode` | `full` | `full` / `subset` / `single` |
| `--subset-size` | — | Number of samples for subset mode |
| `--scene` | — | Scene ID for single mode |
| `--run-dir` | — | Resume from existing run directory |
| `FRAME_ORDER` (env) | `Tm1p5,Tm1p0,Tm0p5,Tp0p0` | Comma-separated list of timesteps |

Outputs land in a timestamped directory:
```
/workspace/cdb_outputs/recogdrive_<YYYYMMDD_HHMMSS>/
```

---

## Post-processing

```bash
# Docker
RUN_DIR=/workspace/data/cdb_outputs/recogdrive_<timestamp> \
  docker compose --env-file dev.env --profile postprocess run --rm postprocess

# Bare-metal
/root/miniconda3/envs/recogdrive/bin/python postprocess.py \
  --run-dir /workspace/cdb_outputs/recogdrive_<timestamp> \
  --bench-dir /workspace/nuscenes_qa_test \
  --config /path/to/eval_config.yaml
```

Generates the standard 3-tier report hierarchy:

| File | Level | Contents |
|------|-------|----------|
| `<run>/<dataset>/<scene>/<sample>/report.json` | Tier 1 — sample | Per-question predictions, correct flags, metrics |
| `<run>/<dataset>/report.json` | Tier 2 — dataset | Aggregated metrics across all samples |
| `<run>/report.json` | Tier 3 — run | Full metrics: overall, per_qa_type, per_rung, binary, etc. |

---

## Viewing Reports

```bash
# Generate HTML
python postprocess.py --run-dir ... --bench-dir ... --config eval_config.yaml

# Serve
cd /workspace/cdb_reports && python3 -m http.server 8080
```

Then in VS Code: **Terminal → Ports tab → Forward Port 8080** → open `localhost:8080`.

---

## Benchmark Results

Full dataset: 72 scenes, 689 questions (NuScenes).

| Run | Overall | Distractor | Dormant | Ladder |
|-----|---------|------------|---------|--------|
| `recogdrive_20260422_144005` | **56.6%** | 47.7% | 62.1% | 66.2% |

Configuration: `owl10/ReCogDrive-VLM-8B`, `FRAME_ORDER=Tm1p5,Tm1p0,Tm0p5,Tp0p0` (16 images/sample), ~6.1s/question on A40.

**Note on binary questions:** The model shows a strong Yes-bias (predicts "Yes" ~63% of the time vs 23% ground-truth rate), resulting in binary accuracy of ~50.6% (near chance). MCQ accuracy is 84.4%. The overall score is dominated by the high MCQ performance.

---

## What Was Added / Changed

### `inference.py`
- Uses `AutoModel.from_pretrained` + `model.chat()` for InternVL3 VQA — no NAVSIM planner head needed.
- Applies InternVL3's standard image pipeline: `Resize(448) → ToTensor() → Normalize(ImageNet)`.
- Builds multi-image question string with `<image>\n` × N_images prefix (InternVL3 protocol).
- `use_flash_attn=True` — requires SM ≥ 8 (Ampere or newer).

### `dataloader.py`
- `FRAME_ORDER` configurable via env var (default: 4 timesteps = 16 images/sample).
- `build_image_paths()` handles both nested (`frame → cameras → cam`) and flat frame dicts.
- `format_sample()` loads images in TIME-MAJOR order; placeholder tensors for missing frames.

### `dev.env.example`
- Follows the `alpamayo_1_5` env var convention: `MODEL_REPO`, `CAUSAL_BENCH_DIR`, `CDB_OUTPUTS_DIR`, `CDB_REPORTS_DIR`, `RAW_DATA_DIR`, full LLM-judge section.
- `MODEL_REPO` points to `owl10/ReCogDrive-VLM-8B` (the VLM backbone, not the planner checkpoint).
