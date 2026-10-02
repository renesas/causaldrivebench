# Qwen2.5-VL Evaluation

This directory contains the implementation for evaluating **Qwen2.5-VL** (3B and 7B variants) on CausalDriveBench.

---

## 1. Setup

### **Weights**
The inference script uses HuggingFace `transformers`. You can use either:
- A HuggingFace model ID (e.g., `Qwen/Qwen2.5-VL-3B-Instruct`) — downloaded automatically.
- A local path to downloaded weights.

### **Dependencies**
```
torch, transformers, qwen_vl_utils, pillow
```

### **Config**
Create a local config override (copy from `evaluation/eval_config.yaml`):
```bash
cp evaluation/eval_config.yaml evaluation/eval_config.local.yaml
# Edit to fill in raw_data_dir, causal_drive_bench_dir, cdb_outputs_dir, etc.
```

---

## 2. Running Inference

### **Host**

Run from the repo root:
```bash
python evaluation/models/qwen25vl/inference.py \
    --config evaluation/eval_config.local.yaml \
    --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --mode subset \
    --subset-size 10
```

Key arguments:
- `--model-id`: HuggingFace ID or local path to weights.
- `--config`: Path to `eval_config.yaml` (auto-detected if omitted).
- `--mode`: `full`, `subset` (with `--subset-size N`), or `single` (with `--scene <scene-id>`).
- `--bench-dir`: Override the benchmark directory from config.
- `--output-base`: Override the output directory from config.

---

## 3. Post-processing & Metrics

Once inference is complete, compute accuracy metrics:

```bash
python evaluation/models/qwen25vl/postprocess.py \
    --config evaluation/eval_config.local.yaml \
    --run-dir /path/to/cdb_outputs/qwen25vl_YYYYMMDD_HHMMSS
```

---

## 4. Output Files

Each run writes:
```
<run_dir>/
├── inference.log
├── report.json                              # run-level metrics
└── causal_nuscenes/
    └── <scene_id>/
        └── <sample_id>/
            ├── prompts.jsonl                # per-question prompt payloads
            ├── outputs.jsonl                # per-question raw model outputs
            └── report.json                  # per-sample metrics
```
