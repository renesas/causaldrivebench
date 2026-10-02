# Containerized Model Evaluation Framework

> **This document has been superseded by the modular docs in [`evaluation/docs/`](docs/).**
> It is retained for reference. New content should go in the appropriate docs/ file.
>
> | Topic | New location |
> |-------|-------------|
> | Architecture & design | [docs/architecture.md](docs/architecture.md) |
> | Adding a model | [docs/adding_a_model.md](docs/adding_a_model.md) |
> | Docker commands & services | [docs/docker_operations.md](docs/docker_operations.md) |
> | JSONL schemas & data formats | [docs/data_formats.md](docs/data_formats.md) |
> | Pipeline & orchestrator | [docs/evaluation_pipeline.md](docs/evaluation_pipeline.md) |

---

Complete guide for running and extending the causal scene understanding benchmark evaluation.

---

## 1. Framework Overview

```
evaluation/
├── eval_config.yaml          # dataset paths + model registry
├── orchestrate.py            # CLI: build → infer → postprocess → compare
├── outputs/                  # per-model inference outputs (gitignored content)
├── reports/                  # comparison JSON reports
├── common/
│   ├── dataset.py            # BenchmarkDataset — record + QA loader
│   ├── base_dataloader.py    # abstract ModelDataLoader
│   ├── base_inference.py     # abstract ModelInference
│   ├── base_postprocess.py   # abstract ModelPostprocessor
│   ├── metrics.py            # accuracy, confusion, comparison table
│   └── llm_judge.py          # LLM-based open-ended scoring
└── models/
    ├── _template/            # copy-paste starting point for new models
    ├── drivelm/              # DriveLM implementation
    ├── drivevlm/             # DriveVLM implementation
    └── llava_next/           # LLaVA-NeXT implementation
```

### Data flow

```
BenchmarkDataset
      |
      | load_record()
      v
ModelDataLoader.format_sample()   <- load images, state JSON
      |
      | get_prompt()  (per question)
      v
prompts.jsonl
      |
      | ModelInference.run_from_jsonl()
      v
outputs.jsonl  {sample_id, question_id, raw_output, ...}
      |
      | ModelPostprocessor.compute_metrics()
      v
report.json    {overall, per_qa_type, confusion}
      |
      | aggregate_model_metrics() + format_comparison_table()
      v
comparison_<timestamp>.json
```

---

## 2. Prerequisites

### Docker and NVIDIA runtime

```bash
# Docker Engine >= 24.x
docker --version

# NVIDIA Container Toolkit (allows GPU passthrough into containers)
sudo apt-get install -y nvidia-container-toolkit nvidia-docker2
sudo systemctl restart docker

# Verify GPU is visible inside a container
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Python (host-side, for orchestrate.py)

```bash
pip install pyyaml
```

---

## 3. Repository Structure

| Path | Purpose |
|------|---------|
| `eval_config.yaml` | Central config: dataset paths, model list, evaluation settings |
| `orchestrate.py` | Top-level CLI that drives the full pipeline |
| `common/dataset.py` | Loads records + QA files; returns structured dicts |
| `common/base_dataloader.py` | Abstract base — override `format_sample` + `get_prompt` |
| `common/base_inference.py` | Abstract base — override `load_model` + `run_single` |
| `common/base_postprocess.py` | Abstract base — override `parse_answer` |
| `common/metrics.py` | Pure-Python accuracy / confusion / comparison utilities |
| `common/llm_judge.py` | OpenAI-based judge for open-ended answers |
| `models/<name>/Dockerfile` | Container definition for the model |
| `models/<name>/docker-compose.yml` | Compose file with GPU + volume mounts |
| `models/<name>/setup.sh` | One-time weight download script |
| `models/<name>/requirements.in` | Direct Python dependencies |
| `models/<name>/dev.env.example` | Environment variable template |
| `models/<name>/dataloader.py` | `ModelDataLoader` subclass |
| `models/<name>/inference.py` | `ModelInference` subclass + `main()` |
| `models/<name>/postprocess.py` | `ModelPostprocessor` subclass + `main()` |

---

## 4. Quick Start — Run Existing Models

```bash
# From the repo root
cd /projects/hpcai/code/neurips-dataset

# Edit eval_config.yaml to point to your data
nano evaluation/eval_config.yaml

# Run the full pipeline on a 50-scene subset
python evaluation/orchestrate.py \
  --models drivelm \
  --dataset nuscenes \
  --mode subset \
  --subset-size 50 \
  --output-dir evaluation/outputs \
  --report-dir evaluation/reports

# Dry-run to see what commands would be executed
python evaluation/orchestrate.py \
  --models drivelm,llava_next \
  --dry-run

# Skip rebuild (images already built) and re-run post-processing only
python evaluation/orchestrate.py \
  --models drivelm \
  --skip-build \
  --skip-inference
```

---

## 5. Step-by-Step: Adding a New Model

### Step 1: Copy the template

```bash
cp -r evaluation/models/_template evaluation/models/my_model
```

### Step 2: Fill in setup.sh

Edit `setup.sh` to clone your model's repository and download weights.
The two most common patterns:

```bash
# HuggingFace Hub
huggingface-cli download ORG/REPO --local-dir "$WEIGHTS_DIR"

# GitHub + local install
git clone --depth 1 https://github.com/ORG/REPO "$MODEL_DIR"
pip install -e "$MODEL_DIR"
```

### Step 3: Write requirements.in

Add your model's direct Python dependencies. Avoid pinning transitive
dependencies here — pip-compile will handle that.

```
# requirements.in example
torch>=2.0.0
transformers>=4.40.0
flash-attn>=2.5.0
my-special-package==1.2.3
```

Compile to a locked file:
```bash
pip install pip-tools
pip-compile evaluation/models/my_model/requirements.in \
    -o evaluation/models/my_model/requirements.txt
```

### Step 4: Configure dev.env.example

Copy to `dev.env` and fill in values:

```bash
cp evaluation/models/my_model/dev.env.example evaluation/models/my_model/dev.env
# Edit dev.env — never commit this file
```

### Step 5: Customize Dockerfile

- Change `ARG CUDA_VERSION` to match your model's CUDA requirement.
- Add any system packages needed (e.g. `libsndfile1` for audio models).
- Decide whether to run `setup.sh` at build time (bakes weights into image)
  or at runtime (mount weights as a volume — recommended for large models).

```dockerfile
# Example: run setup at build time
RUN bash /workspace/eval/setup.sh

# Example: defer to runtime via entrypoint
ENTRYPOINT ["/bin/bash", "-c", "bash /workspace/eval/setup.sh && python inference.py"]
```

### Step 6: Configure docker-compose.yml

Set the correct image tag and verify your volume paths:

```yaml
services:
  inference:
    image: benchmark-my_model:latest
    # Volumes should match your RAW_DATA_DIR, CAUSAL_BENCH_DIR, OUTPUT_DIR, WEIGHTS_PATH
```

### Step 7: Implement dataloader.py

Subclass `ModelDataLoader` and implement two methods:

```python
class MyModelDataLoader(ModelDataLoader):
    def format_sample(self, record: dict) -> dict:
        # Load images, tokenise text, etc.
        # record["frames"]["Tp0p0"]["cam_front_path"] — front camera at T=0
        # record["frames"]["Tp0p0"]["state_json"]     — scene state dict
        return {"images": [...], "state_text": "..."}

    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        # Build the exact dict your model's generate() expects
        # Must include: sample_id, question_id, question_type, prompt
        return {
            "sample_id": ...,
            "question_id": question["id"],
            "question_type": ...,
            "prompt": [...],  # your model's format
        }
```

Key helper: `self.format_question_text(question)` builds the MCQ text block.

### Step 8: Implement inference.py

Subclass `ModelInference`:

```python
class MyModelInference(ModelInference):
    def load_model(self, weights_path: str) -> None:
        self.model = MyModel.from_pretrained(weights_path).to(self.device)
        self.tokenizer = MyTokenizer.from_pretrained(weights_path)

    def run_single(self, prompt_payload: dict) -> str:
        # Takes prompt_payload from get_prompt(), returns raw string
        inputs = self.tokenizer(prompt_payload["prompt"], return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=64)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
```

The base class provides `run_from_jsonl()` which handles JSONL I/O, progress
bars, resumption, and timing — you only need to implement `run_single`.

### Step 9: Implement postprocess.py

Subclass `ModelPostprocessor`:

```python
class MyModelPostprocessor(ModelPostprocessor):
    def parse_answer(self, raw_output: str, question: dict) -> str:
        # Extract the predicted letter from the model's raw text
        # Return one of: "A", "B", "C", "D" (or empty string if unclear)
        import re
        m = re.search(r"\b([A-D])\b", raw_output)
        return m.group(1).upper() if m else ""
```

`compute_metrics()` and `save_report()` are fully implemented in the base class.

### Step 10: Register in eval_config.yaml

```yaml
models:
  my_model:
    enabled: true
    description: "My awesome model"
    compose_file: models/my_model/docker-compose.yml
```

### Step 11: Test the pipeline (single scene first)

```bash
# Build the image
docker compose -f evaluation/models/my_model/docker-compose.yml build

# Test on one scene
python evaluation/orchestrate.py \
  --models my_model \
  --mode single \
  --scene <scene_name> \
  --dry-run   # review commands first

python evaluation/orchestrate.py \
  --models my_model \
  --mode single \
  --scene <scene_name>

# Check output
cat evaluation/outputs/my_model/outputs.jsonl | head -5
cat evaluation/outputs/my_model/report.json
```

---

## 6. Dataset Settings

### Full evaluation
```bash
python evaluation/orchestrate.py --models drivelm --mode full
```

### Random subset (N scenes)
```bash
python evaluation/orchestrate.py --models drivelm --mode subset --subset-size 100
```

### Single scene (debugging)
```bash
python evaluation/orchestrate.py \
    --models drivelm \
    --mode single \
    --scene scene_0001_nuscenes
```

---

## 7. Weights Download Patterns

### HuggingFace Hub

```bash
# In setup.sh or Dockerfile RUN step:
pip install huggingface_hub
huggingface-cli login --token "$HF_TOKEN"
huggingface-cli download ORG/MODEL_NAME --local-dir /workspace/weights

# Python API:
from huggingface_hub import snapshot_download
snapshot_download(repo_id="ORG/MODEL", local_dir="/workspace/weights")
```

### Google Drive (gdown)

```bash
pip install gdown
gdown <FILE_ID> -O /workspace/weights/model.pt
# For folder:
gdown --folder <FOLDER_ID> -O /workspace/weights/
```

### AWS S3

```bash
pip install awscli
aws s3 cp s3://my-bucket/path/to/weights/ /workspace/weights/ --recursive
# With credentials via env vars:
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... aws s3 cp s3://...
```

### Direct URL

```bash
wget -q --show-progress -O /workspace/weights/model.bin \
    https://example.com/path/to/model.bin
```

---

## 8. GPU Configuration

### All available GPUs (default in docker-compose.yml)

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### Specific GPU count

```yaml
devices:
  - driver: nvidia
    count: 2
    capabilities: [gpu]
```

### Specific GPU device IDs

```yaml
devices:
  - driver: nvidia
    device_ids: ["0", "1"]
    capabilities: [gpu]
```

### Memory limit (Docker)

```yaml
deploy:
  resources:
    limits:
      memory: 40g
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Multi-GPU inference via accelerate

Set in `dev.env`:
```
CUDA_VISIBLE_DEVICES=0,1
```

In `inference.py`:
```python
# load_model with device_map="auto" distributes across visible GPUs
self.model = AutoModel.from_pretrained(
    weights_path,
    device_map="auto",
    torch_dtype=torch.float16,
)
```

---

## 9. Troubleshooting

### "CUDA out of memory"
- Reduce `BATCH_SIZE` in `dev.env` (try `BATCH_SIZE=1`).
- Add `torch.cuda.empty_cache()` between batches in `run_single`.
- Use `torch_dtype=torch.float16` or `torch.bfloat16` when loading the model.
- Enable gradient checkpointing: `model.gradient_checkpointing_enable()`.

### "docker: Error response from daemon: could not select device driver"
```bash
# Reinstall NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### "No module named 'evaluation'"
The `PYTHONPATH` in the Dockerfile must include the repo root:
```dockerfile
ENV PYTHONPATH=/workspace/eval_root:/workspace/model:/workspace
```
Or add to `sys.path` at the top of your script:
```python
sys.path.insert(0, "/workspace/eval_root")
```

### "question_id not found in QA index" warnings
QA files may not exist for all scenes. Verify:
```bash
ls /causal_drive_bench/causal_nuscenes/<scene_name>_SAMPLED_0/
# Should contain: qa.json (unified), or ladder_qa.json, dl_qa.json, distractor_qa.json (legacy)
```

### Container exits immediately without error
Run interactively to debug:
```bash
docker compose -f models/my_model/docker-compose.yml run --rm --entrypoint bash inference
# Inside container:
python inference.py --mode single --scene test_scene
```

### Partial results / resume
`ModelInference.run_from_jsonl()` automatically resumes if `outputs.jsonl`
already exists. It skips `(sample_id, question_id)` pairs already written.

---

## 10. LLM Judge Setup for Open-Ended Metrics

Some questions (future benchmark extensions) may require open-ended scoring
rather than exact letter matching.

### Configuration

In `eval_config.yaml`:
```yaml
evaluation:
  use_llm_judge: true
  judge:
    provider: openai
    model: gpt-4o
```

### API key

```bash
# In dev.env:
OPENAI_API_KEY=sk-...
USE_LLM_JUDGE=true
JUDGE_MODEL=gpt-4o
```

### Standalone usage

```python
from evaluation.common.llm_judge import LLMJudge

judge = LLMJudge(provider="openai", model="gpt-4o")
result = judge.score_response(
    question="Why did the vehicle brake?",
    ground_truth="The vehicle braked because a pedestrian entered the crosswalk.",
    model_response="Because a person walked into the road.",
)
print(result)  # {"score": 8, "reasoning": "..."}
```

### Batch scoring with rate limiting

```python
records = [
    {"question": "...", "ground_truth": "...", "model_response": "..."},
    ...
]
scored = judge.score_batch(
    records,
    requests_per_minute=60,   # adjust for your API tier
    save_path="evaluation/outputs/my_model/judge_scores.jsonl",
)
print(LLMJudge.aggregate_scores(scored))
```

### Cost estimation

At ~500 tokens per call (question + GT + response + prompt), with GPT-4o:
- 1 000 questions ≈ 0.5M tokens ≈ ~$1.25 input + ~$0.25 output = ~$1.50
- 10 000 questions ≈ ~$15

Use `--mode subset --subset-size 200` for cost-effective development runs.
