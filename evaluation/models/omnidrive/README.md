# OmniDrive — CausalDriveBench Evaluation

End-to-end evaluation of [OmniDrive](https://github.com/NVlabs/OmniDrive) (EVA-02-Large ViT + StreamPETR + TinyLlama 1.1B) on the CausalDriveBench benchmark.

---

## Architecture

```
EVA-02-Large ViT (1024-dim)
        ↓
    StreamPETR
        ↓
  TinyLlama 1.1B  →  answer text
```

Camera setup: 4 cameras (`cam_front`, `cam_front_left`, `cam_front_right`, `cam_back`), single timestep (`FRAME_ORDER=Tp0p0`).

---

## Environment Setup

### Prerequisites

- CUDA-capable GPU (tested on A40, CUDA 11.7 / driver 12.x)
- Miniconda installed at `/root/miniconda3`
- OmniDrive repo cloned at `/workspace/OmniDrive` with checkpoint at `ckpts/iter_10548.pth`

### Create the conda environment

The exact working environment is captured in `/workspace/omnidrive_env.yml` and `/workspace/omnidrive_pip_freeze.txt`.

```bash
# Create env from the yml snapshot (Python 3.9, torch 1.13.1+cu117)
conda env create -f /workspace/omnidrive_env.yml

conda activate omnidrive
```

If the yml is unavailable, install manually following `/workspace/OmniDrive/docs/setup.md`, with these fixes:

```bash
# flash-attn and editable installs need --no-build-isolation / setup.py develop
pip install flash-attn==2.3.6 --no-build-isolation

cd /workspace/OmniDrive/mmdetection3d
python setup.py develop          # NOT pip install -e .

cd /workspace/OmniDrive/OpenLane-V2
python setup.py develop

# networkx 2.2 is incompatible with Python 3.9 (fractions.gcd removed)
pip install "networkx==2.6"
```

---

## Running Inference

### Option A — Docker (recommended for production)

Copy and fill in `dev.env`:

```bash
cp dev.env.example dev.env
# Set: MODEL_CONFIG, MODEL_CHECKPOINT, BENCH_DIR, OUTPUT_BASE
```

```bash
# Full dataset
docker compose --env-file dev.env run --rm inference

# Subset
MODE=subset SUBSET_SIZE=5 docker compose --env-file dev.env run --rm inference

# Single scene
MODE=single SCENE=nuscenes-scene-0001 docker compose --env-file dev.env run --rm inference

# Resume interrupted run
RUN_DIR=/workspace/data/cdb_outputs/omnidrive_<timestamp> \
  docker compose --env-file dev.env run --rm inference
```

### Option B — Bare-metal (conda env)

```bash
cd /workspace/neurips-dataset/evaluation/models/omnidrive

FRAME_ORDER=Tp0p0 /root/miniconda3/envs/omnidrive/bin/python inference.py \
  --bench-dir /workspace/nuscenes_qa_test \
  --model-config /workspace/OmniDrive/projects/configs/OmniDrive/eva_base_tinyllama.py \
  --weights-path /workspace/OmniDrive/ckpts/iter_10548.pth \
  --output-base /workspace/cdb_outputs
```

Key env vars / flags:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_REPO` | `/workspace/OmniDrive` | Path to cloned OmniDrive repo (env: MODEL_REPO) |
| `FRAME_ORDER` | — | Must be `Tp0p0` (single timestep, 4 cameras) |
| `--mode` | `full` | `full` / `subset` / `single` |
| `--subset-size` | — | Number of samples for subset mode |
| `--scene` | — | Scene ID for single mode |
| `--run-dir` | — | Resume from existing run directory |

Outputs land in a timestamped directory:
```
/workspace/cdb_outputs/omnidrive_<YYYYMMDD_HHMMSS>/
```

---

## Post-processing

```bash
# Docker
RUN_DIR=/workspace/cdb_outputs/omnidrive_<timestamp> \
  docker compose --env-file dev.env --profile postprocess run --rm postprocess

# Bare-metal
/root/miniconda3/envs/omnidrive/bin/python postprocess.py \
  --run-dir /workspace/cdb_outputs/omnidrive_<timestamp> \
  --bench-dir /workspace/nuscenes_qa_test \
  --config /path/to/eval_config.yaml
```

Generates a 3-tier report hierarchy:

| File | Level | Contents |
|------|-------|----------|
| `<run>/<dataset>/<scene>/<sample>/report.json` | Tier 1 — sample | Per-question predictions, correct flags, metrics |
| `<run>/<dataset>/report.json` | Tier 2 — dataset | Aggregated metrics across all samples |
| `<run>/report.json` | Tier 3 — run | Full metrics: overall, per_qa_type, per_rung, binary, etc. |

To force regeneration of all sample reports:
```bash
python postprocess.py --run-dir ... --bench-dir ... --force
```

---

## Viewing Reports

### HTML report + local port forward

```bash
# Generate HTML (requires html.enabled: true in eval_config.yaml)
python postprocess.py --run-dir ... --bench-dir ... --config /tmp/eval_config_local.yaml

# Serve
cd /workspace/cdb_reports && python3 -m http.server 8080
```

Then in VS Code: **Terminal → Ports tab → Forward Port 8080** → open `localhost:8080` in your browser.

### JSON summary (quick)

```python
import json
r = json.load(open("/workspace/cdb_outputs/omnidrive_<timestamp>/report.json"))
m = r["metrics"]
print(f"Overall: {m['overall']['accuracy']:.1%}  ({m['overall']['correct']}/{m['overall']['n']})")
for k, v in m["per_qa_type"].items():
    print(f"  {k}: {v['accuracy']:.1%} (n={v['n']})")
```

---

## Benchmark Results

Full dataset: 72 scenes, 689 questions (NuScenes).

| Run | Overall | Distractor | Dormant | Ladder |
|-----|---------|------------|---------|--------|
| `omnidrive_20260421_160540` | **36.1%** | 41.4% | 27.3% | 34.8% |

Configuration: `FRAME_ORDER=Tp0p0`, `eva_base_tinyllama.py`, `iter_10548.pth`.

> **Note:** Images were not available on this pod (all 4 camera slots used placeholder tensors). The model ran vision-encoder + LM head on blank inputs — results reflect language-prior performance only.

---

## What Was Added / Changed

### `inference.py`
- **`OmniDriveDataLoader`** (`dataloader.py`): custom data loader that maps CausalDriveBench scene/sample structure to OmniDrive's mmdet3d pipeline. Handles camera ordering, frame ordering (`FRAME_ORDER` env var), calibration matrices, and graceful image-not-found fallback (placeholder tensors).
- **`FRAME_ORDER=Tp0p0`**: selects a single timestep with 4 cameras — the correct config for `eva_base_tinyllama.py` (EVA-02-Large backbone, not EVA-02-Base).
- Resolved merge conflict: duplicate `--bench-dir` arg and conflicting `parse_args()` blocks merged from `origin/main`.

### `postprocess.py`
- Migrated from a custom flat report generator to the shared `ModelPostprocessor` base class (`evaluation/common/base_postprocess.py`).
- Produces the standard 3-tier JSON report hierarchy used by all CausalDriveBench models.
- `OmniDrivePostprocessor` inherits `parse_answer()` directly — no model-specific token stripping needed (TinyLlama outputs plain text).
- Removed `generate_report_pdf.py` (replaced by the shared HTML reporter).

### Environment fixes
- `networkx==2.6` instead of 2.2 — `fractions.gcd` was removed in Python 3.9.
- `chardet==5.2.0` pinned — setup.py develop for mmdet3d/OpenLane-V2 installs a newer egg that breaks on Python 3.9 dataclass `slots=True`.
- `mmdet3d` and `OpenLane-V2` must be installed via `python setup.py develop`, not `pip install -e .` (build backend missing `build_editable` hook).
- `flash-attn==2.3.6` requires `--no-build-isolation` to find the already-installed torch.
