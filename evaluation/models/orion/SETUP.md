# Orion — Setup & Run Guide

## What Orion is

EVA-ViT + LLaMA-based autonomous driving model (xiaomi-mlab/Orion). Takes 4 cameras × 4 timesteps = 16 images and produces scene descriptions with detected object positions and driving behavior. Fine-tuned on Chat-B2D (Bench2Drive VQA dataset).


---

## 1. Model Weights

Three weight files are required. Place them under your `WEIGHTS_DIR`:

| File | Source |
|------|--------|
| `Orion/Orion.pth` | [poleyzdk/Orion](https://huggingface.co/poleyzdk/Orion) on HuggingFace |
| `pretrain_qformer/` | [exiawsh/pretrain_qformer](https://huggingface.co/exiawsh/pretrain_qformer) on HuggingFace |
| `eva02_petr_proj.pth` | [NVlabs/OmniDrive releases v1.0](https://github.com/NVlabs/OmniDrive/releases/tag/v1.0) — manual download |

The Docker weight-downloader pulls Orion and pretrain_qformer automatically:

```bash
cd evaluation/models/orion/
cp dev.env.example dev.env          # fill in HF_TOKEN and paths
docker compose --env-file dev.env --profile setup run --rm weight-downloader
```

`eva02_petr_proj.pth` must be downloaded manually from the OmniDrive GitHub release.

---

## 2. NuScenes Data

Two directories are needed:

| Path | Contents |
|------|----------|
| `raw_data/nuscenes/` | Full NuScenes dataset (images + calibration) |
| `raw_data/nuscenes/can_bus/` | Per-scene pose files `scene-XXXX_pose.json` with `utime`, `pos`, `orientation`, `vel`, `accel`, `rotation_rate` |

The can_bus files are used to build the 18-element ego-state vector the model expects.

---

## 3. Environment File

```bash
cp dev.env.example dev.env
```

Key variables to fill in:

```
HF_TOKEN=<your HuggingFace token>
WEIGHTS_DIR=./weights                          # host path for model weights
RAW_DATA_DIR=/path/to/raw_data                 # NuScenes data root
CAUSAL_BENCH_DIR=/path/to/causal_nuscenes      # benchmark scenes
CDB_OUTPUTS_DIR=./outputs
```

---

## 4. Build Docker Image

From the repo root (CUDA extensions compile — takes ~10–20 min on first build):

```bash
docker compose -f evaluation/models/orion/docker-compose.yml build
```

---


## 5. Full Benchmark Inference

```bash
# Full dataset
docker compose --env-file dev.env run --rm inference

# Random subset
MODE=subset SUBSET_SIZE=50 docker compose --env-file dev.env run --rm inference

# Single scene
MODE=single SCENE=nuscenes-scene-0001 docker compose --env-file dev.env run --rm inference
```

Outputs land in `CDB_OUTPUTS_DIR/orion_<YYYYMMDD_HHMMSS>/`.

---

## 6. Prompt Format

The model was fine-tuned on Chat-B2D and responds to its training question templates. The inference script uses the critical-object template to elicit scene-specific detections:

```
USER: <image>
You are driving a car.Where are the critical objects in the scene and what impact do they have on the ego vehicle?
ASSISTANT:
```

Conversation format: vicuna_v1 (system: "A chat between a curious user and an assistant…", sep=" ", sep2="</s>").
---

## 7. Known Issues Fixed During Setup

| Error | Fix |
|-------|-----|
| `ImportError: load_checkpoint from mmcv.runner` | Changed to `from mmcv.utils import load_checkpoint` |
| `ModuleNotFoundError: mmdet3d` | Changed to `from mmcv.models import build_model` |
| `pad_shape` 4-tuple `(640,640,3,3)` | Use `[img_shape] * n_cameras` not `img_shape + (3,)` |
| `KeyError: cam_intrinsic` | Tile 4-camera calibration matrices for all temporal frames → `(1, 16, 4, 4)` |
| `linalg.inv: Low precision dtypes not supported` | Pass `lidar2img` as `float32` (not FP16) |
| Empty responses from `generate()` | LLaVA returns only new tokens; decode `output_ids[0]` directly, no slicing |
| Identical outputs for all scenes | Wrong prompt format + temperature=0.1; use Chat-B2D critical-object template + temperature=0.7 |
