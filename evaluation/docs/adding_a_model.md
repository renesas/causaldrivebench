# CausalDriveBench Adding a New Model

Step-by-step guide for integrating a new model into the CausalDriveBench evaluation framework.

---

## Step 1 — Copy the Template

```bash
cp -r evaluation/models/_template evaluation/models/<your_model>
```

Then rename the placeholder classes and files:

| File | Class to rename |
|------|----------------|
| `dataloader.py` | `TemplateDataLoader` → `<YourModel>DataLoader` |
| `inference.py` | `TemplateInference` → `<YourModel>Inference` |
| `postprocess.py` | `TemplatePostprocessor` → `<YourModel>Postprocessor` |

Search and replace `<model-name>` and `<model_folder>` in `docker-compose.yml`.

---

## Step 2 — Configure `dev.env`

```bash
cp evaluation/models/<your_model>/dev.env.example \
   evaluation/models/<your_model>/dev.env
# Edit dev.env — never commit this file
```

Key variables:

```bash
HF_TOKEN=hf_...              # Required for gated HuggingFace models
MODEL_REPO=org/repo-name     # HuggingFace repo identifier
WEIGHTS_DIR=./weights        # Host path; mapped to /workspace/weights
CAUSAL_BENCH_DIR=/path/to/bench   # Top-level bench dir (not dataset-specific)
CDB_OUTPUTS_DIR=./cdb_outputs     # Output base; timestamped run dirs created here
DEVICE=cuda
```

---

## Step 3 — Update `requirements.in`

Add only the packages your evaluation scripts need that are **not** already managed by the model's own `pyproject.toml`:

```
# requirements.in
Pillow>=10.0.0
numpy>=1.24.0
tqdm
jsonlines
# Add model-specific extras:
# flash-attn>=2.5.0
```

Compile to a pinned lockfile:

```bash
# With pip-tools
pip install pip-tools
pip-compile requirements.in -o requirements.txt --no-header

# Or with uv (recommended)
uv pip compile requirements.in -o requirements.txt
```

Re-run this whenever you add or change a dependency.

---

## Step 4 — Update the `Dockerfile`

Minimal changes needed:

1. Update `LABEL` fields.
2. Change the `git clone` URL to your model's repository.
3. Adjust `PYTHON_VERSION` if needed.
4. If your model uses `pyproject.toml` → keep `uv sync --active` (Option A).
   If not → switch to `uv pip install -r requirements.txt` (Option B).
5. Update the `WORKDIR` and `COPY` path at the bottom.

---

## Step 5 — Configure `docker-compose.yml`

Replace all `<model-name>` and `<model_folder>` placeholders.

Use **CLI args** for inference and postprocess commands (not ENV vars):

```yaml
inference:
  command:
    - uv
    - run
    - python
    - /workspace/repo/evaluation/models/<model>/inference.py
    - --bench-dir
    - /causal_drive_bench/${DATASET:-causal_nuscenes}
    - --output-base
    - /cdb_outputs
    - --weights-path
    - /workspace/weights
    - --device
    - ${DEVICE:-cuda}
    - --mode
    - ${MODE:-full}

postprocess:
  command:
    - uv
    - run
    - python
    - /workspace/repo/evaluation/models/<model>/postprocess.py
    - --bench-dir
    - /causal_drive_bench/${DATASET:-causal_nuscenes}
    - --run-dir
    - ${RUN_DIR:-/cdb_outputs/latest}
    - --mode
    - ${MODE:-full}
```

Update the `original` service command to run your model's own sample script:

```yaml
original:
  command: ["uv", "run", "python", "src/my_model/test_inference.py"]
```

The five services and when to use them:

| Service | Profile | Purpose |
|---------|---------|---------|
| `weight-downloader` | `setup` | Download model weights (run once) |
| `original` | `original` | Sanity-check vendor's own sample inference |
| `inference` | *(default)* | Run benchmark evaluation |
| `postprocess` | `postprocess` | Parse outputs, compute metrics |
| `jupyter` | `jupyter` | Interactive development |

---

## Step 6 — Implement `dataloader.py`

Subclass `ModelDataLoader` and implement two required methods plus one optional one:

```python
class MyModelDataLoader(ModelDataLoader):

    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        """Return image paths as a list in camera-major order.

        Each entry: {"path": str, "time_key": str, "camera_key": str}
        """
        image_paths = []
        frames = record.get("frames", {})
        for cam in CAMERA_ORDER:            # outer loop: cameras
            for frame_key in FRAME_ORDER:   # inner loop: timesteps
                path = frames.get(frame_key, {}).get(cam)
                if path:
                    image_paths.append({
                        "path": path,
                        "time_key": frame_key,
                        "camera_key": cam,
                    })
        return image_paths

    def format_sample(self, record: dict) -> dict:
        """Load images and state for one benchmark sample."""
        scene_id  = record["scene_id"]   # new API (was record["scene"])
        sample_id = record["sample_id"]  # new API (was record["sample"])
        state     = record.get("state") or {}  # new API (was frames["Tp0p0"]["state_json"])
        frames    = record.get("frames", {})
        # frames: {time_key → {camera_key → relative_path}}
        # camera_key is e.g. "cam_front" (no "_path" suffix)
        images = [...]  # load PIL.Images from frames
        return {
            "scene_id":  scene_id,
            "sample_id": sample_id,
            "images":    images,
            # Add any model-specific tensors or state:
            "state_summary": str(state),
        }

    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        """Return only model-specific fields (framework fields injected by base class)."""
        # Do NOT include: scene_id, sample_id, question_id, image_paths, qa_text, etc.
        # Those are injected automatically by iter_questions().
        return {
            # Add your model's prompt format extras here:
            # "system_prompt": "...",
            # "ego_history": [...],
        }
```

Key notes:
- `build_image_paths()` must return a **list** (not a dict). Framework injects it as `prompt["image_paths"]`.
- `get_prompt()` must return a **JSON-serialisable dict** with **only model-specific extras** (no numpy arrays, no PIL objects, no framework fields).
- Use `self.format_question_text(question)` to get the standard MCQ text block when needed.

---

## Step 7 — Implement `inference.py`

```python
MODEL_ID = "my_model"

class MyModelInference(ModelInference):

    def load_model(self, weights_path: str) -> None:
        self.model = MyModel.from_pretrained(weights_path).to(self.device)
        self.tokenizer = MyTokenizer.from_pretrained(weights_path)

    def run_single(self, prompt_payload: dict) -> dict:
        """Return a dict with at least a 'text' key."""
        # Image paths come as a list in camera-major order:
        image_paths: list = prompt_payload.get("image_paths", [])
        # Build lookup for flexible access:
        lookup = {
            (item["time_key"], item["camera_key"]): item["path"]
            for item in image_paths
        }
        qa_text: str = prompt_payload.get("qa_text", "")  # framework field

        # ... tokenize, run model.generate(), decode ...
        raw_output_string = "..."

        return {"text": raw_output_string}
```

The base class `run_from_jsonl()` handles JSONL I/O, progress, resumption, and timing.

**Timestamped run directory** (recommended, matching alpamayo_1_0):

```python
# In main():
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = output_base / f"{MODEL_ID}_{timestamp}"
setup_logging(run_dir / "inference.log")

causal_dataset_name = bench_dir.name  # e.g. "causal_nuscenes"

for sample in sample_ids:
    scene_id  = sample["scene_id"]
    sample_id = sample["sample_id"]
    sample_out_dir = run_dir / causal_dataset_name / scene_id / sample_id
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    loader.save_prompts([sample], sample_out_dir / "prompts.jsonl")
    inference.run_from_jsonl(
        sample_out_dir / "prompts.jsonl",
        sample_out_dir / "outputs.jsonl",
    )
```

**GPU-lazy loading**: check whether any work is needed before loading the model:

```python
def _has_pending_work(sample_ids, run_dir, causal_dataset_name) -> bool:
    for s in sample_ids:
        out = run_dir / causal_dataset_name / s["scene_id"] / s["sample_id"] / "outputs.jsonl"
        if not out.exists():
            return True
    return False

if _has_pending_work(sample_ids, run_dir, causal_dataset_name):
    inference.load_model(args.weights_path)
```

---

## Step 8 — Implement `postprocess.py`

```python
class MyModelPostprocessor(ModelPostprocessor):

    def parse_answer(self, raw_output, question: dict) -> str:
        """Extract predicted answer from raw model output (dict or str)."""
        text = self._extract_text(raw_output)  # handles both dict and str
        if not text or not text.strip():
            raise EmptyResponseException("Empty model response")
        # For binary questions
        if question.get("answer_format") == "binary":
            m = re.search(r"\bAnswer\s*:\s*(Yes|No)\b", text, re.IGNORECASE)
            return m.group(1).capitalize() if m else ""
        # For MCQ questions
        m = re.search(r"\bAnswer\s*:\s*([A-D])\b", text, re.IGNORECASE)
        return m.group(1).upper() if m else ""
```

**Collecting outputs** from the nested per-sample structure:

```python
def _collect_outputs(sample_ids, output_base):
    all_outputs, missing = [], []
    for sid in sample_ids:
        outputs_file = output_base / sid["scene_id"] / sid["sample_id"] / "outputs.jsonl"
        if not outputs_file.exists():
            missing.append(f"{sid['scene_id']}/{sid['sample_id']}")
        else:
            with outputs_file.open() as f:
                all_outputs.extend(json.loads(line) for line in f if line.strip())
    return all_outputs, missing
```

---

## Step 9 — Write `system_prompt.txt`

Start from the template's `system_prompt.txt` and add model-specific instructions after the `# TODO` line.

---

## Step 10 — Write Tests

Place tests in `evaluation/models/<your_model>/tests/`:

```
tests/
  __init__.py
  test_dataloader.py    # geometry, build_image_paths, format_sample, get_prompt, iter_questions
  test_postprocess.py   # parse_answer, _collect_outputs, compute_metrics, save_report
```

Use the factory pattern:

```python
# tests/test_dataloader.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # model dir

from dataloader import MyModelDataLoader

SAMPLE_DATA = Path(__file__).parents[4] / "evaluation/sample_data/causal_nuscenes"

def make_loader():
    """Factory: returns a loader ready for testing."""
    from evaluation.common.dataset import BenchmarkDataset
    ds = BenchmarkDataset(SAMPLE_DATA)
    return MyModelDataLoader(dataset=ds, model_cfg={})
```

Critical things to test:
- `build_image_paths()` returns a **list** (not dict) of dicts with `path`, `time_key`, `camera_key`
- `format_sample()` uses `record["scene_id"]` and `record["sample_id"]` (not `record["scene"]`)
- `format_sample()` reads state from `record["state"]` (not `frames["Tp0p0"]["state_json"]`)
- `get_prompt()` returns no framework fields (`scene_id`, `sample_id`, `image_paths`, `qa_text`, etc.)
- `get_prompt()` return value is JSON-serialisable (no numpy arrays, no PIL objects)
- `parse_answer()` handles both dict `{"text": "..."}` and legacy plain string
- `_collect_outputs()` uses nested `scene_id/sample_id/outputs.jsonl` paths

Run from repo root:

```bash
pytest evaluation/models/<your_model>/tests/ -v
```

Or inside the container:

```bash
docker compose run --rm inference \
    uv run python -m pytest /workspace/repo/evaluation/models/<your_model>/tests/ -v
```

---

## Step 11 — Register in `eval_config.yaml`

```yaml
models:
  my_model:
    enabled: true
    description: "My model description"
    compose_file: models/my_model/docker-compose.yml
```

---

## Step 12 — Smoke Test

```bash
cd evaluation/models/<your_model>

# Build
docker compose build

# Sanity-check with single scene
MODE=single SCENE=nuscenes-scene-0159 docker compose run --rm inference

# Post-process (set RUN_DIR to the timestamped directory printed by inference)
RUN_DIR=/cdb_outputs/my_model_20260410_120000 \
docker compose --profile postprocess run --rm postprocess

# Check outputs
ls /cdb_outputs/my_model_20260410_120000/causal_nuscenes/nuscenes-scene-0159/SAMPLED_0/
cat /cdb_outputs/my_model_20260410_120000/causal_nuscenes/nuscenes-scene-0159/SAMPLED_0/outputs.jsonl | head -3
cat /cdb_outputs/my_model_20260410_120000/report.json
```
