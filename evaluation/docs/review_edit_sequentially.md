# Sequential Code Review Guide

This document is a **reading order** for deeply understanding the CausalDriveBench (CDB) evaluation codebase and for implementing support for a new VLA (Vision-Language-Action) model.

Each phase builds on the previous. Follow the order the first time; use it as an index during edits.

---

## Phase 0 — Orient: Docs and Config (30 min)

Read these before any code. They define the vocabulary and contracts used everywhere.

### 0.1 `evaluation/docs/architecture.md`
- **What**: Top-level design philosophy and directory layout
- **Focus**: Factory pattern (base classes → concrete implementations), per-sample output structure, 4-tier report hierarchy

### 0.2 `evaluation/docs/data_formats.md`
- **What**: Every file schema in the benchmark and output directories
- **Focus**:
  - `frames.json` — time_key → camera_key → relative_path
  - `qa/<type>_qa.json` — question structure; required fields: `id`, `question`, `answer_format`, `correct_answer`, `reasoning`
  - `prompts.jsonl` — framework fields vs model-specific fields; `qa_text` has no ground truth
  - `outputs.jsonl` — `raw_output` must be `{"text": str, ...}` (not a plain string)
  - `report.json` — `schema_version`, `level`, `metrics`, `qa_results`
- **Critical invariant**: `qa_text` in prompts.jsonl is built by `_format_inference_text()` — it deliberately omits `correct_answer` and `reasoning` to prevent leakage into the model prompt.

### 0.3 `evaluation/eval_config.yaml`
- **What**: All runtime paths, model registry, QA types, LLM judge config, report schema version
- **Focus**:
  - `evaluation.qa_types` must use internal keys: `ladder`, `dormant`, `distractor` (NOT `active`)
  - `evaluation.judge` block — provider, model, api_key_env, azure_endpoint_env
  - `reporting.schema_version` — bump this to force regeneration of all cached sample reports

### 0.4 `evaluation/common/utils/config_reader.py` — `CDBConfig`
- **What**: Resolves all directory paths for host vs Docker runtime
- **Focus on these properties and methods**:
  - `dir_data_root`, `dir_raw_data`, `dir_causal_drive_bench`, `dir_cdb_outputs`, `dir_cdb_reports`
  - `resolve_frame_path(relative_path)` — the ONLY correct way to turn a frames.json path into an absolute path; `data_root` in frames.json is always ignored
  - `get_bench_dir(dataset_name)` → `causal_drive_bench/causal_<name>`
  - `judge_config` property → dict from `evaluation.judge` YAML block
- **Key invariant**: `EVAL_RUNTIME` env var selects host paths (from YAML) vs Docker paths (hardcoded constants). Both resolve to the same logical layout.

---

## Phase 1 — Dataset Layer (30 min)

### 1.1 `evaluation/common/dataset.py` — `BenchmarkDataset`

The bridge between raw benchmark files and the evaluation framework. All model dataloaders consume records from this class.

**Read in this order within the file**:

| Function | Purpose |
|---|---|
| `__init__(bench_dir)` | Validates the bench directory exists |
| `list_scenes()` | Discovers all `nuscenes-scene-XXXX/` dirs that have at least one `SAMPLED_N/` child |
| `list_samples(scene)` | Returns all `SAMPLED_N` names for a scene |
| `load_record_by_name(scene, sample)` | **Main entry point** — returns the full record dict (meta, state, calib, frames, qa) |
| `_load_frames(sample_dir)` | Parses `frames.json`; `data_root` is stripped; returns `{time_key → {cam_key → relative_path}}` |
| `_load_qa(sample_dir)` | Loads all three `qa/<type>_qa.json` files; maps `active_qa.json → "ladder"`, `dormant_qa.json → "dormant"`, `distractor_qa.json → "distractor"` |
| `_parse_qa_file(path, qa_type)` | Validates each question: requires `id`, `question`, `answer_format`, `correct_answer`, `reasoning`; skips invalids |
| `get_sample_ids(mode, ...)` | Returns `[{"scene_id": str, "sample_id": str, "sample_idx": int}, ...]`; used by inference/postprocess |
| `iter_records(mode, ...)` | Convenience generator over `load_record_by_name()` |

**Key schema returned by `load_record_by_name()`**:
```python
{
  "scene_id":  "nuscenes-scene-0159",
  "sample_id": "SAMPLED_0",
  "meta":      {...},           # meta.json, or {} if absent
  "state":     {...},           # state.json (ego history, agents, nav_command)
  "calib":     {...},           # calib.json (camera calibration matrices)
  "frames":    {                # frames.json — relative paths only
    "Tm1p5": {"cam_front": "raw_data/nuscenes/...", ...},
    "Tm1p0": {...},
    "Tm0p5": {...},
    "Tp0p0": {...},
  },
  "qa": {                       # loaded from qa/ subfolder
    "ladder":     [question_dict, ...],
    "dormant":    [question_dict, ...],
    "distractor": [question_dict, ...],
  }
}
```

---

## Phase 2 — Base Classes (1 hour)

These define the **contract** every model implementation must satisfy. Read the abstract methods first, then the concrete helpers.

### 2.1 `evaluation/common/base_dataloader.py` — `ModelDataLoader`

**Abstract methods (must implement per model)**:

| Method | Signature | Responsibility |
|---|---|---|
| `build_image_paths` | `(record) → List[Dict]` | Returns `[{"path": str, "time_key": str, "camera_key": str}, ...]` in the model's expected ordering. Stored verbatim in `prompts.jsonl`. |
| `format_sample` | `(record) → dict` | Heavy lifting: load PIL images, parse state, interpolate ego history. Called once per sample. Returns model-specific dict (must include `scene_id`, `sample_id`). |
| `get_prompt` | `(formatted_sample, question) → dict` | Builds model-specific fields for one question. Must NOT return framework fields (they are injected by the base class). |

**Concrete helpers (do not override unless necessary)**:

| Method | Purpose |
|---|---|
| `iter_questions(sample_ids)` | Outer loop: loads record → format_sample → per-question get_prompt → yields complete prompt record. Injected framework fields: `scene_id`, `sample_id`, `question_id`, `prompt_id`, `is_evaluated`, `image_paths`, `answer_format`, `question_text`, `qa_text` |
| `save_prompts(sample_ids, path)` | Writes prompts.jsonl by consuming `iter_questions()` |
| `format_question_text(question)` | Plain question text; no answer. Used for `question_text` field. |
| `_format_inference_text(question)` | Question + format hint (`Answer: Yes or No` / `Answer: A, B, C, or D`). Used for `qa_text`. **No ground truth** — prevents answer leakage. |
| `format_qa_text(question)` | Question + correct answer + reasoning. For few-shot/debug use only — **never use for `qa_text` in prompts.jsonl**. |
| `interpolate_ego_history(state, n_steps)` | Linear interpolation of ego XYZ from state.json. Falls back to all-zeros if state is None. |
| `build_chat_messages(system, user)` | Returns `[{role:system}, {role:user}]` in OpenAI format. |
| `handle_missing_frame(path, label)` | Returns path or None; logs debug message for missing frames. |

**Critical invariant — `iter_questions()` field precedence**: Framework fields always overwrite any same-named key from `get_prompt()`. Specifically, `qa_text` is always set by the base class using `_format_inference_text()`.

### 2.2 `evaluation/common/base_inference.py` — `ModelInference`

**Abstract methods**:

| Method | Signature | Responsibility |
|---|---|---|
| `load_model` | `(weights_path) → None` | Load weights into `self.model`; also set `self.tokenizer`, `self.processor` etc. |
| `run_single` | `(prompt_payload) → dict` | Run inference on one prompt; **must return `{"text": str, ...}`** — a plain string return is coerced but generates a warning. |

**Concrete helpers**:

| Method | Purpose |
|---|---|
| `run_from_jsonl(input_path, output_path, batch_size, resume)` | Main inference loop. Reads `prompts.jsonl`, skips already-processed `question_id`s (resume), calls `run_single()` per prompt, writes `outputs.jsonl`. Returns count of new outputs written. |
| `timing_stats` property | `{total_inferences, total_time_s, avg_time_s}` |

**Output schema written to `outputs.jsonl`**:
```json
{
  "scene_id": "nuscenes-scene-0159",
  "sample_id": "SAMPLED_0",
  "question_id": "CI1",
  "prompt_id": "000001",
  "raw_output": {"text": "Answer: No\nReasoning: ...", "waypoints": []},
  "inference_time_s": 2.34,
  "timestamp": "2026-04-17T21:00:00Z"
}
```

**Resume mechanism**: `run_from_jsonl` reads the existing `outputs.jsonl` to collect already-processed `question_id`s. Safe to kill and restart any time.

### 2.3 `evaluation/common/base_postprocess.py` — `ModelPostprocessor`

**Abstract method**:

| Method | Signature | Responsibility |
|---|---|---|
| `parse_answer` | `(raw_output, question) → str` | Extract the predicted answer from `raw_output` (dict or str). Return `"Yes"`/`"No"` for binary, `"A"`–`"D"` for MCQ. Return `""` on failure (no exception). |

**Concrete helpers**:

| Method | Purpose |
|---|---|
| `_load_qa_index(bench_dir)` | Global index: rglobs all `qa/*.json` across all scenes. **Do not use for CDB** — cross-scene ID collisions (CI1 etc. are reused). |
| `_load_sample_qa_index(sample_bench_dir)` | **Use this for CDB**: reads only the specific `<scene>/<sample>/qa/` folder; avoids ID collisions. |
| `compute_metrics(outputs_path, bench_dir)` | Used by older simple flows — calls `_load_qa_index` (global). Prefer model-specific `_process_dataset` which uses `_load_sample_qa_index`. |
| `_run_llm_judge(records, judge_cfg, score_all)` | Scores CoC reasoning with LLM judge. Pass `score_all=True` for CDB (all questions are binary/MCQ with CoC). |
| `save_report(metrics, path)` | Writes JSON + prints human-readable summary. |
| `_extract_text(raw_output)` | Extracts `raw_output["text"]` or falls back to `str(raw_output)`. |

### 2.4 `evaluation/common/metrics.py`

Pure Python, no GPU dependencies. Read the function signatures and docstrings:

| Function | Input | Output |
|---|---|---|
| `mcq_accuracy(predictions, ground_truth)` | two str lists | float [0,1] |
| `per_type_accuracy(records)` | list of `{type, predicted, ground_truth}` | `{type: {accuracy, n, correct}}` |
| `confusion_summary(records)` | same | `{matrix: {true: {pred: count}}, most_confused: [...]}` |
| `reasoning_score_summary(records)` | records with `llm_judge_score` | composite mean + per-dimension stats |

### 2.5 `evaluation/common/report_generator.py`

4-tier report hierarchy. Read the public functions only:

| Function | Tier | Writes to |
|---|---|---|
| `generate_sample_report(...)` | 1 (sample) | `<run>/<dataset>/<scene>/<sample>/report.json` |
| `generate_dataset_report(...)` | 2 (dataset) | `<run>/<dataset>/report.json` |
| `generate_run_report(...)` | 3 (run) | `<run>/report.json` |
| `generate_root_aggregate(...)` | 4 (root) | In-memory dict (passed to html_reporter) |
| `is_sample_report_fresh(path, schema_version)` | — | Returns True if report exists with matching `schema_version`. Incremental skip logic. |

**Incremental logic**: Only Tier 1 is skipped if fresh. Tiers 2-4 always regenerate (fast, read-only).

---

## Phase 3 — Alpamayo Reference Implementation (1–2 hours)

This is the canonical concrete implementation. Study it before writing any new model.

### 3.1 `evaluation/models/alpamayo_1_0/dataloader.py` — `AlpamayoDataLoader`

Read in this order:

**1. Module-level `CONFIG` dict** (lines ~46–63)
All tunable constants: frame order, camera order, placeholder image size, history interpolation settings. Pattern to copy for new models.

**2. `build_image_paths(record)`**
Camera-major ordering: outer loop = cameras, inner loop = timesteps → 16 image path dicts. Critical: this order must match what the model was trained on.

**3. `format_sample(record)`**
- Camera-major image loading: same nested loop as `build_image_paths`, but loading `PIL.Image` objects. Grey 1600×900 placeholder for missing frames (preserves `torch.stack()` compatibility).
- Ego history extraction from `state.json`: reads `ego.history` entries at t=-1.5, -1.0, -0.5; appends [0,0,0] for t=0 (ego-centric origin). Produces `(4, 3)` float32 array.
- `_compute_rotations()` derives `(4, 3, 3)` rotation matrices from forward differences.

**4. `get_prompt(formatted_sample, question)`**
Returns only `ego_history_xyz` and `ego_history_rot` as nested lists (JSON-serialisable). The base class injects all other fields.

**5. `interpolate_ego_history(xyz_sparse, rot_sparse, num_steps=16)`**
Upsamples 4 NuScenes timesteps → 16 model-expected steps via `np.interp`. Called at inference time (not stored in JSONL).

**6. `inject_qa_into_messages(messages, qa_text, system_prompt, override_system_prompt)`**
Modifies a deep copy of the Alpamayo-generated message list:
- System: replaces (or appends) the QA system prompt
- User: locates the text block containing `<|traj_history_end|>` and replaces the "generate driving trajectory" instruction with `qa_text`, preserving the trajectory token prefix

**7. `build_fallback_messages(...)`**
Used when `helper.create_message()` is unavailable. Does not include trajectory history tokens — for emergency fallback only.

### 3.2 `evaluation/models/alpamayo_1_0/inference.py` — `AlpamayoInference`

**1. `load_model(weights_path)`**
Loads Qwen3VL model + processor, Alpamayo helper, trajectory head. Sets `self.model`, `self.processor`, `self.helper`, `self.traj_head`.

**2. `run_single(prompt_payload)`** — the core method
Steps:
  1. Load 16 images from `prompt_payload["image_paths"]` via `cfg.resolve_frame_path()`
  2. Convert to `(16, 3, H, W)` uint8 torch.Tensor (camera-major; critical for correct token embedding)
  3. Interpolate ego history: `(4, 3)` → `(16, 3)` and `(4, 3, 3)` → `(16, 3, 3)` via `interpolate_ego_history()`
  4. Build base model messages via `helper.create_message(frames_tensor, ...)` — this is alpamayo's VLM rollout initialiser
  5. Inject QA: `inject_qa_into_messages(base_messages, qa_text, system_prompt, override_system_prompt=True)` — **must use `override_system_prompt=True`** to fully replace the alpamayo driving instruction
  6. Call `helper.sample_trajectories_from_data_with_vlm_rollout(...)` with `continue_final_message=True` (model continues from `<|cot_start|>` token)
  7. Extract chain-of-causation from `extra["cot"]`
  8. Return `{"text": cot_text}`

**3. `main()` / `parse_args()`**
Entry point: builds `AlpamayoDataLoader` + `AlpamayoInference`, generates prompts.jsonl per sample, then calls `run_from_jsonl()`.

**Critical adaptations documented in the module docstring** — read these first (lines 1–60).

### 3.3 `evaluation/models/alpamayo_1_0/postprocess.py` — `AlpamayoPostprocessor`

**1. `AlpamayoPostprocessor.parse_answer(raw_output, question)`**
Cascade of regex patterns:
1. Strip `<think>...</think>` wrapper if present; try both inner and full text
2. Binary: look for `Answer: Yes/No` → `answer is Yes/No` → standalone `Yes`/`No` at line start → any `Yes`/`No`
3. MCQ: `Answer: A/B/C/D` → `answer is A` → `Option A` → `A)` → standalone letter

**2. `_parse_sample_outputs(outputs_path, qa_index, postprocessor)`**
Reads `outputs.jsonl`, resolves question from `qa_index`, calls `parse_answer()`, computes `correct` (case-insensitive match). Returns `qa_results` list.

**3. `_build_judge_cfg(args, cfg)`**
Constructs the judge config dict. Uses `LLMJudge._PROVIDER_DEFAULT_*` dicts to fill defaults. Returns `{provider, model, api_key_env, azure_endpoint_env, azure_api_version}`.

**4. `_process_dataset(run_dir, dataset, bench_dir, ...)`**
The main per-run-per-dataset loop:
1. Iterates all `(scene_id, sample_id)` pairs
2. Skips if `is_sample_report_fresh(schema_version)` (incremental)
3. Builds per-sample QA index via `_load_sample_qa_index(bench_dir/scene_id/sample_id)` — **not global** to avoid cross-scene ID collisions
4. Calls `_parse_sample_outputs()` → `qa_results`
5. Optionally calls `_run_llm_judge(ji, judge_cfg, score_all=True)` — `score_all=True` because all CDB questions are binary/MCQ with CoC
6. Calls `generate_sample_report()` → writes Tier 1 report
7. Calls `generate_dataset_report()` → writes Tier 2 report

**5. `main()`**
Orchestrates across run dirs → datasets → samples → Tier 3 → HTML (Tier 4).

---

## Phase 4 — Environment and Container Setup (20 min)

### 4.1 `evaluation/models/alpamayo_1_0/dev.env.example`
All configurable variables. Copy to `dev.env` and fill in:
- `HF_TOKEN`, `WEIGHTS_DIR`, `RAW_DATA_DIR`, `CAUSAL_BENCH_DIR`, `CDB_OUTPUTS_DIR`, `CDB_REPORTS_DIR`
- **LLM Judge section**: `USE_LLM_JUDGE`, `JUDGE_PROVIDER`, `JUDGE_MODEL`, `ANTHROPIC_FOUNDRY_API_KEY`, `FOUNDRY_BASE_URL`

### 4.2 `evaluation/models/alpamayo_1_0/docker-compose.yml`
Key patterns:
- `env_file: ${ENV_FILE:-dev.env}` — loads `dev.env` for `${}` interpolation AND passes vars to the container. Override path with `ENV_FILE=/other/path`.
- `postprocess` service explicitly re-declares all LLM judge env vars (`ANTHROPIC_FOUNDRY_API_KEY`, etc.) so they are visible to Python's `os.environ`.
- `RUN_DIR` env var selects which output directory to postprocess.

---

## Phase 5 — LLM Judge (30 min, optional for new models)

### 5.1 `evaluation/common/llm_judge.py` — `LLMJudge`

**When to read**: When `use_llm_judge: true` or when debugging judge output.

| Section | Purpose |
|---|---|
| Module-level `_load_prompt()` + `JUDGE_SYSTEM_PROMPT` | Loads rubric from `evaluation/common/prompts/llm_judge_system.txt` at import time. Edit the `.txt` file to change the rubric. |
| `LLMJudge.__init__` class-level dicts | `_PROVIDER_DEFAULT_MODELS`, `_PROVIDER_DEFAULT_KEY_ENVS`, `_PROVIDER_DEFAULT_ENDPOINT_ENVS` — the single source of truth for per-provider defaults. |
| `_get_client()` | Lazy client init. `azure_foundry` → `AnthropicFoundry`; `azure_openai` → `AzureOpenAI`; `anthropic` → `Anthropic`; `openai` → `OpenAI`. |
| `score_response(question, ground_truth, model_response)` | Sends to judge model; parses JSON response; returns `{scores, composite_score, reasoning}`. |
| `score_batch(records, ...)` | Rate-limited batch scoring. |
| `aggregate_scores(scored_records)` | Summary stats over a batch. |

See `evaluation/docs/llm_judge.md` for setup, env vars, and config.

### 5.2 `evaluation/common/prompts/llm_judge_system.txt`
The four-dimension rubric (spatial_grounding, mechanism_identification, conditionality_awareness, answer_consistency). Edit to change scoring criteria.

---

## Phase 6 — Tests (20 min)

### 6.1 `evaluation/models/alpamayo_1_0/test_dataloader.py`
Smoke tests for `AlpamayoDataLoader`:
- `format_sample` produces correct image count and tensor shapes
- `build_image_paths` produces camera-major ordering
- `get_prompt` returns serialisable payload without framework fields

### 6.2 `evaluation/models/alpamayo_1_0/test_postprocess.py`
Unit tests for `AlpamayoPostprocessor.parse_answer`:
- All answer formats and extraction patterns
- Edge cases: empty response, think-tag wrapping, no clear answer

---

## Phase 7 — Implementing a New Model

After reading phases 0–3, copy the template and follow this sequence:

### 7.1 Start from `evaluation/models/_template/`
```bash
cp -r evaluation/models/_template evaluation/models/<your_model>/
```

### 7.2 Edit these files in this order

**Step 1 — `dataloader.py`**

| What to do | Where |
|---|---|
| Define `CONFIG` dict | model-specific frame order, camera order, image size |
| Implement `build_image_paths(record)` | return `[{"path", "time_key", "camera_key"}, ...]` ordered for your model |
| Implement `format_sample(record)` | load images, parse state (call `interpolate_ego_history` if needed), return dict with `scene_id`, `sample_id` |
| Implement `get_prompt(formatted_sample, question)` | return only model-specific fields; base class injects everything else |

**Critical**: Do NOT include `qa_text`, `question_text`, `image_paths`, `scene_id`, `sample_id`, `question_id`, or `answer_format` in `get_prompt()` return. These are injected by the base class and your values will be silently overwritten.

**Step 2 — `inference.py`**

| What to do | Notes |
|---|---|
| Subclass `ModelInference` | |
| Implement `load_model(weights_path)` | Set `self.model`, `self.processor`, etc. |
| Implement `run_single(prompt_payload)` | Must return `{"text": str}`. Load images from `prompt_payload["image_paths"]` using `cfg.resolve_frame_path()`. |
| Define `MODEL_ID` constant | Used to name the timestamped run directory |
| Write `main()` | Create `BenchmarkDataset` → `YourDataLoader` → call `save_prompts()` per sample → `run_from_jsonl()` |

**Step 3 — `postprocess.py`**

| What to do | Notes |
|---|---|
| Subclass `ModelPostprocessor` | |
| Implement `parse_answer(raw_output, question)` | Raise `EmptyResponseException` for empty output; return `""` for parse failure |
| Copy `_parse_sample_outputs()` from alpamayo template | Almost always reusable |
| Copy `_process_dataset()` from alpamayo template | Uses `_load_sample_qa_index()` (correct); change nothing else unless your model adds extra fields |
| Copy `main()` / `parse_args()` | Change `MODEL_ID` and class names |

**Step 4 — `dev.env.example`**

Add all model-specific env vars. Always include the LLM judge section (copy from alpamayo).

**Step 5 — `docker-compose.yml`**

- Use `env_file: ${ENV_FILE:-dev.env}` pattern (not a commented-out block)
- In `postprocess` service `environment:` block, explicitly declare all LLM judge vars:
  ```yaml
  - USE_LLM_JUDGE=${USE_LLM_JUDGE:-false}
  - ANTHROPIC_FOUNDRY_API_KEY=${ANTHROPIC_FOUNDRY_API_KEY:-}
  - FOUNDRY_BASE_URL=${FOUNDRY_BASE_URL:-}
  ```
- `--use-llm-judge` flag reads `USE_LLM_JUDGE` env var automatically

**Step 6 — `eval_config.yaml`**

Add your model to the `models:` registry:
```yaml
models:
  your_model:
    enabled: true
    description: "Your model description"
    compose_file: models/your_model/docker-compose.yml
```

---

## Quick Reference: Data Flow

```
eval_config.yaml + EVAL_RUNTIME
    │
    ▼
CDBConfig                    ← resolves all paths (host or docker)
    │
    ▼
BenchmarkDataset             ← loads records from bench_dir
    │  load_record_by_name(scene, sample)
    │  → {scene_id, sample_id, frames, state, calib, qa}
    ▼
ModelDataLoader              ← YOUR IMPL: build_image_paths, format_sample, get_prompt
    │  iter_questions(sample_ids)
    │  → prompt records (framework fields + model fields)
    │  save_prompts() → prompts.jsonl
    ▼
ModelInference               ← YOUR IMPL: load_model, run_single
    │  run_from_jsonl(prompts.jsonl, outputs.jsonl)
    │  → outputs.jsonl (raw_output per question)
    ▼
ModelPostprocessor           ← YOUR IMPL: parse_answer
    │  _load_sample_qa_index(sample_bench_dir)  ← scoped to sample to avoid ID collisions
    │  _parse_sample_outputs(outputs.jsonl, qa_index)
    │  → qa_results [predicted, ground_truth, correct, ...]
    │  _run_llm_judge(qa_results, judge_cfg, score_all=True)  ← optional CoC scoring
    ▼
report_generator             ← no changes needed
    │  generate_sample_report → Tier 1 report.json
    │  generate_dataset_report → Tier 2
    │  generate_run_report → Tier 3
    ▼
html_reporter                ← no changes needed
    generate_all_html → HTML reports in cdb_reports/
```

---

## Common Mistakes and How to Avoid Them

| Mistake | Symptom | Prevention |
|---|---|---|
| Using `active` instead of `ladder` in `qa_types` config | 5 questions per sample silently skipped; prompts.jsonl has only dormant + distractor | Always use `ladder` in `eval_config.yaml` |
| Returning `qa_text` from `get_prompt()` | Leaks ground-truth answer into model prompt; model outputs trivially correct answers | Never return framework fields from `get_prompt()` |
| Using `_load_qa_index(bench_dir)` for per-sample postprocessing | Wrong question text in reports (last alphabetical scene's questions win for each ID) | Use `_load_sample_qa_index(bench_dir/scene/sample)` |
| Calling `inject_qa_into_messages` with `override_system_prompt=False` | Original model driving instruction competes with QA question; model answers the wrong task | Always pass `override_system_prompt=True` for CDB |
| Using `score_all=False` for CDB questions | `_run_llm_judge` filters out all ladder/dormant/distractor types; `reasoning_score` metric is always empty | Pass `score_all=True` in `_process_dataset` |
| `run_single` returning a plain string | Base class coerces it to `{"text": str}` but CoC extraction in postprocess fails | Always return `{"text": cot_text, ...}` from `run_single` |
| `env_file` commented out in docker-compose | API keys from `dev.env` not available in container; LLM judge raises `EnvironmentError` | Use `env_file: ${ENV_FILE:-dev.env}` and explicit `environment:` block |
| Hardcoding `schema_version` in reports | Cached reports never regenerate after code changes | Set in `eval_config.yaml`; bump value after any schema change |

---

## File Index

| File | Phase | Role |
|---|---|---|
| `evaluation/eval_config.yaml` | 0.3 | Runtime configuration (paths, models, judge, schema version) |
| `evaluation/common/utils/config_reader.py` | 0.4 | `CDBConfig`: path resolution, host/docker switching |
| `evaluation/common/utils/errors.py` | — | Non-recoverable errors: `TensorShapeError`, `MissingEnvVarError` |
| `evaluation/common/utils/exceptions.py` | — | Recoverable exceptions: `EmptyResponseException`, `MissingFrameException` |
| `evaluation/common/dataset.py` | 1.1 | `BenchmarkDataset`: file loading, QA parsing, record schema |
| `evaluation/common/base_dataloader.py` | 2.1 | `ModelDataLoader` abstract base |
| `evaluation/common/base_inference.py` | 2.2 | `ModelInference` abstract base |
| `evaluation/common/base_postprocess.py` | 2.3 | `ModelPostprocessor` abstract base |
| `evaluation/common/metrics.py` | 2.4 | `mcq_accuracy`, `per_type_accuracy`, `confusion_summary` |
| `evaluation/common/report_generator.py` | 2.5 | Tier 1–4 report JSON generation |
| `evaluation/common/html_reporter.py` | — | HTML report rendering (no changes needed for new models) |
| `evaluation/common/llm_judge.py` | 5.1 | `LLMJudge`: multi-provider CoC scoring |
| `evaluation/common/prompts/llm_judge_system.txt` | 5.2 | Editable judge rubric |
| `evaluation/models/alpamayo_1_0/dataloader.py` | 3.1 | Reference: camera-major ordering, ego interpolation, message injection |
| `evaluation/models/alpamayo_1_0/inference.py` | 3.2 | Reference: VLM rollout, tensor format, QA injection |
| `evaluation/models/alpamayo_1_0/postprocess.py` | 3.3 | Reference: CoC parsing, per-sample QA index, judge wiring |
| `evaluation/models/alpamayo_1_0/dev.env.example` | 4.1 | Template: all env vars including LLM judge |
| `evaluation/models/alpamayo_1_0/docker-compose.yml` | 4.2 | Template: env_file pattern, judge env vars in postprocess service |
| `evaluation/models/_template/` | 7 | Skeleton for new model implementations |
