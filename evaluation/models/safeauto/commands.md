# SafeAuto — Commands

Quick reference for running SafeAuto on CausalDriveBench.

All commands assume `conda activate safeauto` and are run from the repo root
(`/workspace/neurips-dataset/`).

---

## Prerequisites

```bash
# 1. Activate conda env (torch, transformers, flash-attn already installed)
conda activate safeauto

# 2. Download LanguageBind vision encoders (one-time, ~3 GB each)
bash evaluation/models/safeauto/setup.sh

# 3. Ensure raw_data_dir=/workspace is set in eval_config.yaml
#    (so raw_data/nuscenes/... paths resolve correctly)
```

---

## Unit Tests (no GPU)

```bash
conda activate safeauto
cd /workspace/neurips-dataset
pytest evaluation/models/safeauto/tests/ -v
```

---

## Inference — Sample Data

```bash
conda activate safeauto
cd /workspace/neurips-dataset

# Single scene (fastest, for verification)
python evaluation/models/safeauto/inference.py \
  --bench-dir evaluation/sample_data/causal_nuscenes \
  --mode single \
  --scene nuscenes-scene-0159 \
  --weights-path /workspace/ckpts/SafeAuto-DriveLM

# All samples in sample_data
python evaluation/models/safeauto/inference.py \
  --bench-dir evaluation/sample_data/causal_nuscenes \
  --mode full \
  --weights-path /workspace/ckpts/SafeAuto-DriveLM

# Resume an interrupted run
python evaluation/models/safeauto/inference.py \
  --bench-dir evaluation/sample_data/causal_nuscenes \
  --run-dir /path/to/safeauto_20260424_120000
```

---

## Post-process and Report

```bash
conda activate safeauto
cd /workspace/neurips-dataset

python evaluation/models/safeauto/postprocess.py \
  --run-dir /path/to/safeauto_20260424_120000 \
  --bench-dir evaluation/sample_data/causal_nuscenes

# View report
cat /path/to/safeauto_20260424_120000/report.json | python3 -m json.tool | head -40
```

### Optional: LLM judge scoring

```bash
# Anthropic batch API (50% cheaper, anthropic provider only)
python evaluation/models/safeauto/postprocess.py \
  --run-dir /path/to/safeauto_<timestamp> \
  --bench-dir evaluation/sample_data/causal_nuscenes \
  --use-llm-judge --judge-batch-api --judge-provider anthropic

# Concurrent thread-pool (all providers; tune to provider rate limit)
python evaluation/models/safeauto/postprocess.py \
  --run-dir /path/to/safeauto_<timestamp> \
  --bench-dir evaluation/sample_data/causal_nuscenes \
  --use-llm-judge --judge-provider azure_foundry --judge-concurrency 20
```

`--judge-concurrency` defaults to 20; set to 1 for serial. Ignored when
`--judge-batch-api` is set.

---

## Inference — Full nuScenes Dataset

```bash
conda activate safeauto
cd /workspace/neurips-dataset

python evaluation/models/safeauto/inference.py \
  --bench-dir /path/to/causal_drive_bench/causal_nuscenes \
  --mode full \
  --weights-path /workspace/ckpts/SafeAuto-DriveLM \
  --output-base /path/to/cdb_outputs
```

---

## Notes

- Batch size is always 1 (LLaVA generate is not batched across questions).
- The inference script changes `cwd` to `evaluation/models/safeauto/SafeAuto/`
  before loading the model — required for `./cache_dir/LanguageBind_*` resolution.
- Images are loaded from `/workspace/raw_data/nuscenes/...` via `raw_data_dir=/workspace`
  in `eval_config.yaml`.
- No RAG or MLN safety verification is used (inference-only, no retrieval DB).
