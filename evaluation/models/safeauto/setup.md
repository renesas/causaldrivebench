# SafeAuto — Setup Guide

SafeAuto (ICML 2025) is a LLaVA-based VQA model fine-tuned on DriveLM (nuScenes multi-view) with PDCE safety loss. This guide walks through everything needed to run it inside the CausalDriveBench evaluation framework.

---

## Prerequisites

| Component | Location | Notes |
|-----------|----------|-------|
| SafeAuto repo | `evaluation/models/safeauto/SafeAuto/` | Clone from `https://github.com/AI-secure/SafeAuto` |
| Checkpoint | `/workspace/ckpts/SafeAuto-DriveLM/` | HuggingFace or direct download |
| LanguageBind encoders | `SafeAuto/cache_dir/` | Downloaded by `setup.sh` |
| nuScenes images | `/workspace/raw_data/nuscenes/` | Required for inference |
| conda env `safeauto` | — | Must have torch, transformers, flash-attn |

---

## Step 1 — Clone the SafeAuto repo

```bash
cd evaluation/models/safeauto
git clone https://github.com/AI-secure/SafeAuto SafeAuto
```

The inference script changes the working directory to `SafeAuto/` at load time (required because `builder.py` hardcodes `cache_dir='./cache_dir'` for LanguageBind path resolution).

---

## Step 2 — Create the conda environment

```bash
conda create -n safeauto python=3.10 -y
conda activate safeauto

# Core deps (install in this order to avoid version conflicts)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.5.8 --no-build-isolation
pip install transformers==4.37.0 accelerate sentencepiece

# SafeAuto's own requirements
pip install -r evaluation/models/safeauto/SafeAuto/requirements.txt

# Evaluation framework wrapper deps
pip install Pillow numpy tqdm jsonlines huggingface_hub
```

> If `flash-attn` build fails, try `pip install flash-attn --no-build-isolation` with the CUDA toolkit active.

---

## Step 3 — Download the checkpoint

The fine-tuned checkpoint should be placed at `/workspace/ckpts/SafeAuto-DriveLM/`. It is a full LLaVA model directory containing at minimum:

```
SafeAuto-DriveLM/
  config.json          # "X": ["Video", "Image"] — both towers required
  tokenizer.model
  tokenizer_config.json
  *.safetensors        # model weights
```

Download via HuggingFace Hub:

```bash
huggingface-cli download AI-secure/SafeAuto-DriveLM \
    --local-dir /workspace/ckpts/SafeAuto-DriveLM
```

---

## Step 4 — Download LanguageBind vision encoders

SafeAuto uses two LanguageBind towers (Image + Video). They must be present in `SafeAuto/cache_dir/` before the model can be loaded — the builder reads them at `__init__` time even with `delay_load=True`.

```bash
conda activate safeauto
bash evaluation/models/safeauto/setup.sh
```

This clones the two encoder repos (~3 GB each) via git-lfs:

```
SafeAuto/cache_dir/
  LanguageBind_Image/
  LanguageBind_Video_merge/
```

The script skips any encoder that is already present, so it is safe to re-run.

---

## Step 5 — Configure paths

`eval_config.yaml` (repo root) must have:

```yaml
raw_data_dir: /workspace/raw_data
```

This makes the framework resolve `raw_data/nuscenes/samples/...` paths from `frames.json` to `/workspace/raw_data/nuscenes/...`.

If your nuScenes images are not under a `samples/` subdirectory (flat layout), the inference script automatically strips the `/samples/` level as a fallback — no config change needed.

---

## Step 6 — Verify with unit tests (no GPU)

```bash
conda activate safeauto
cd /workspace/neurips-dataset
pytest evaluation/models/safeauto/tests/ -v
```

All 56 tests should pass without a GPU or a loaded model.

---

## Step 7 — Run inference on sample data

```bash
conda activate safeauto
cd /workspace/neurips-dataset

# Single scene (fastest verification)
python evaluation/models/safeauto/inference.py \
  --bench-dir evaluation/sample_data/causal_nuscenes \
  --mode single \
  --scene nuscenes-scene-0159 \
  --weights-path /workspace/ckpts/SafeAuto-DriveLM

# All scenes in sample_data
python evaluation/models/safeauto/inference.py \
  --bench-dir evaluation/sample_data/causal_nuscenes \
  --mode full \
  --weights-path /workspace/ckpts/SafeAuto-DriveLM
```

Outputs are written to `/workspace/cdb/outputs/safeauto_<timestamp>/`:

```
safeauto_20260424_120000/
  prompts.jsonl    # one line per question with conversation_text + image_paths
  outputs.jsonl    # one line per question with raw_output.text
  inference.log
```

---

## Step 8 — Post-process and generate report

```bash
conda activate safeauto
cd /workspace/neurips-dataset

python evaluation/models/safeauto/postprocess.py \
  --run-dir /workspace/cdb/outputs/safeauto_20260424_120000 \
  --bench-dir evaluation/sample_data/causal_nuscenes

cat /workspace/cdb/outputs/safeauto_20260424_120000/report.json | python3 -m json.tool | head -50
```

### Optional: LLM judge scoring

```bash
# Anthropic batch API (50% cheaper)
python evaluation/models/safeauto/postprocess.py \
  --run-dir /workspace/cdb/outputs/safeauto_20260424_120000 \
  --bench-dir evaluation/sample_data/causal_nuscenes \
  --use-llm-judge --judge-batch-api --judge-provider anthropic

# Concurrent thread-pool (all providers)
python evaluation/models/safeauto/postprocess.py \
  --run-dir /workspace/cdb/outputs/safeauto_20260424_120000 \
  --bench-dir evaluation/sample_data/causal_nuscenes \
  --use-llm-judge --judge-provider azure_foundry --judge-concurrency 20
```

`--judge-concurrency` defaults to 20; set to 1 for serial. Ignored when
`--judge-batch-api` is set.

---

## Architecture notes

**Model loading**: `inference.py` calls `os.chdir(SafeAuto/)` before `load_pretrained_model` so that `./cache_dir/LanguageBind_*` resolves correctly. All framework paths (`bench_dir`, `run_dir`, `weights_path`) are converted to absolute paths in `main()` before the chdir.

**Image selection**: SafeAuto uses only T=0 (`Tp0p0`) images from 4 cameras: `cam_front`, `cam_front_left`, `cam_front_right`, `cam_back`. The `cam_back_left` and `cam_back_right` cameras present in nuScenes are excluded to match DriveLM training.

**Prompt format**: Each question is wrapped in a DriveLM-style multi-image prompt:

```
Following multi-view images record the driving scenario:
CAM_FRONT: <image>, CAM_FRONT_LEFT: <image>, CAM_FRONT_RIGHT: <image>, CAM_BACK: <image>

<question text>

<format hint: Yes/No or A/B/C/D>
```

**Tokenization quirk**: The prompt is tokenized with `tokenizer_X_token(..., X_TOKEN_INDEX["VIDEO"])` even though the modality is `"image"`. This matches DriveLM training conventions — the VIDEO index (-201) is used as the image placeholder token.

**Expected output format**:
```
Answer: <Yes | No | A | B | C | D>
Reasoning: <explanation>
```

The postprocessor handles free-form alternatives (`"Yes, because..."`, `"The answer is A"`, `"B) ..."`) via regex fallback.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `OSError: Can't load configuration of './cache_dir/LanguageBind_Image'` | Encoders not downloaded | Run `setup.sh` |
| `KeyError: 'image'` in `processor["image"]` | `model_name` didn't contain `'llava'` | Inference hardcodes `"llava-SafeAuto-DriveLM"` — check `load_model()` |
| `Sample directory not found` | `bench_dir` was relative when `os.chdir` fired | Paths are `.resolve()`-d in `main()` before `load_model()` |
| Images resolve to `/raw_data/nuscenes/...` (root `/`) | `raw_data_dir` set to `/workspace` instead of `/workspace/raw_data` | Set `raw_data_dir: /workspace/raw_data` in `eval_config.yaml` |
| Inference completes in <1 s, no GPU activity | Image tensors empty → `model.generate` skipped | Check that image paths exist; inference logs a warning per missing file |
| `flash_attn` import error | Flash-attn not installed or wrong CUDA | `pip install flash-attn --no-build-isolation`; or set `attn_implementation=eager` |
