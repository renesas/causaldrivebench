# SimLingo Inference Setup

## Overview

SimLingo is a driving model built on InternVL2-1B (vision encoder + Qwen2-0.5B LLM)
with LoRA fine-tuning and a driving-specific waypoint prediction head trained via
DeepSpeed ZeRO Stage 2.

Two evaluation modes are supported:

- **VQA** — causal reasoning benchmark (binary / MCQ). Uses the InternVL2 backbone
  via `model.chat()`. The driving head is not used.
- **Trajectory** — BTE and CTE waypoint prediction. Uses the driving head
  (`DrivingAdaptor`) on top of LM features conditioned on the question prompt.
  Both BTE and CTE prompts are question-dependent; CTE additionally includes the
  counterfactual scenario text so the LM features carry that context into the
  driving head.

`traj_eval.py` is a module used internally by `inference.py --trajectory`. Its
standalone `main()` is superseded by the integrated pipeline described below.

---

## Environment

- **Python**: 3.8
- **CUDA**: 11.8+ recommended
- **PyTorch**: 2.6
- **Conda env**: `simlingo`

---

## System Dependencies

```bash
apt-get update
apt-get install -y git-lfs
git lfs install
```

---

## Python Dependencies

```bash
# Core
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Transformers — must be pinned to 4.40.0
# Newer versions introduce all_tied_weights_keys which InternVL2's custom
# modeling code does not implement, causing an AttributeError on load
pip install transformers==4.40.0

# HuggingFace ecosystem
pip install huggingface_hub safetensors accelerate

# InternVL2 / timm dependency
pip install timm einops

# SimLingo training package (required for trajectory inference)
# Clone from GitHub first:
#   git clone https://github.com/RenzKa/simlingo
#   cd simlingo && pip install -e .

# LoRA support
pip install peft

# Remove pyarrow if installed — causes GLIBCXX conflicts in this environment
pip uninstall pyarrow -y
```

---

## Downloading Model Weights

The weights are hosted on HuggingFace at `RenzKa/simlingo` (~9.68 GB).
They are stored via Git LFS so `git-lfs` must be installed before cloning.

```bash
git lfs install
git clone https://huggingface.co/RenzKa/simlingo simlingo_weights
```

The relevant checkpoint is at:
```
simlingo_weights/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
```

> **Important**: If `pytorch_model.pt` starts with `version https://git-`, the LFS
> pointer was not resolved. Run `git lfs pull` inside the `simlingo_weights` directory
> to fetch the actual weights.

```bash
cd simlingo_weights
git lfs pull
```

---

## Running Inference

### VQA only

```bash
conda activate simlingo
cd evaluation/models/simlingo

python inference.py \
    --run-dir /workspace/cdb/outputs/simlingo_<timestamp> \
    --bench-dir /workspace/raw_data/causal_nuscenes/nuscenes_full_ice \
    --weights-path /workspace/ckpts/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt \
    --dataset nuscenes \
    --mode full
```

### VQA + Trajectory (BTE and CTE)

Adding `--trajectory` runs a second pass after VQA using the driving head.
The VQA model is freed from GPU before the trajectory model loads.

```bash
python inference.py \
    --run-dir /workspace/cdb/outputs/simlingo_<timestamp> \
    --bench-dir /workspace/raw_data/causal_nuscenes/nuscenes_full_ice \
    --weights-path /workspace/ckpts/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt \
    --dataset nuscenes \
    --mode full \
    --trajectory
```

Both passes are resumable — already-completed samples are skipped automatically.
To resume an interrupted run, pass the same `--run-dir` and both passes will
pick up where they left off.

Other modes:
```bash
# Subset
python inference.py ... --mode subset --subset-size 50

# Single scene
python inference.py ... --mode single --scene nuscenes-scene-0001
```

### Background (nohup)

```bash
nohup python inference.py \
    --run-dir /workspace/cdb/outputs/simlingo_<timestamp> \
    --bench-dir /workspace/raw_data/causal_nuscenes/nuscenes_full_ice \
    --weights-path /workspace/ckpts/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt \
    --mode full \
    --trajectory \
    > /workspace/cdb/outputs/simlingo_<timestamp>/nohup.out 2>&1 &
```

---

## Post-processing

### VQA metrics (Tier 1–3 reports)

```bash
python postprocess.py \
    --run-dir /workspace/cdb/outputs/simlingo_<timestamp>
```

### VQA + LLM judge (batched, ~50% cheaper)

```bash
python postprocess.py \
    --run-dir /workspace/cdb/outputs/simlingo_<timestamp> \
    --use-llm-judge \
    --judge-batch-api \
    --judge-provider anthropic
```

### VQA + LLM judge (concurrent, all providers)

For non-anthropic providers (or when the Batch API is unavailable), use a thread
pool. Default concurrency is 20; tune to the provider's rate limit. Set to 1
for serial. Ignored when `--use-batch-api` is set.

```bash
python postprocess.py \
    --run-dir /workspace/cdb/outputs/simlingo_<timestamp> \
    --use-llm-judge \
    --judge-provider azure_foundry \
    --judge-concurrency 20
```

### Trajectory metrics (BTE + CTE ADE)

```bash
python postprocess.py \
    --trajectory-eval \
    --run-dir /workspace/cdb/outputs/simlingo_<timestamp> \
    --bench-dir /workspace/raw_data/causal_nuscenes/nuscenes_full_ice
```

---

## Output Files

| File | Description |
|------|-------------|
| `<run_dir>/nuscenes_full_ice/<scene>/<sample>/prompts.jsonl` | VQA prompt payloads |
| `<run_dir>/nuscenes_full_ice/<scene>/<sample>/outputs.jsonl` | VQA raw outputs |
| `<run_dir>/nuscenes_full_ice/<scene>/<sample>/report.json` | Per-sample VQA metrics |
| `<run_dir>/nuscenes_full_ice/<scene>/<sample>/trajectory_prompts.jsonl` | Trajectory prompt records + GT waypoints (BTE + CTE) |
| `<run_dir>/nuscenes_full_ice/<scene>/<sample>/trajectory_outputs.jsonl` | Driving head waypoint predictions per question |
| `<run_dir>/nuscenes_full_ice/<scene>/<sample>/trajectory_report.json` | Per-sample BTE/CTE ADE metrics |
| `<run_dir>/nuscenes_full_ice/report.json` | Dataset-level VQA aggregate |
| `<run_dir>/nuscenes_full_ice/trajectory_aggregate.json` | Dataset-level BTE/CTE ADE aggregate |
| `<run_dir>/report.json` | Run-level VQA aggregate |
| `<run_dir>/inference.log` | Structured inference log |

---

## Known Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `AttributeError: all_tied_weights_keys` | transformers too new | Pin to `transformers==4.40.0` |
| `invalid load key, 'v'` | File is safetensors format | Use `safetensors.torch.load_file()` |
| `version https://git-` in weights file | Git LFS pointer not resolved | Run `git lfs pull` |
| `GLIBCXX_3.4.31 not found` | pyarrow/libstdc++ mismatch | `pip uninstall pyarrow -y` |
| `HFValidationError: Repo id must be in form...` | Local path passed as HF repo ID | Pass directory, not `.pt` file path |
| `KeyError: pytorch-lightning_version` | Raw DeepSpeed shards passed to PL loader | Use manual `load_state_dict` instead of `load_from_checkpoint` |
| `_pickle.UnpicklingError: Weights only load` | PyTorch 2.6 default changed | Add `weights_only=False` to `torch.load` |
| `0 samples across 0 scene(s)` | `--bench-dir` points to parent dir | Pass full path ending in `nuscenes_full_ice` |
