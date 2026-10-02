# Trajectory Evaluation — Design Document

> **Feature status:** implemented, tested (63 unit tests pass), dry-run validated on
> `nuscenes-scene-0072` (rung=3, CR1) and `nuscenes-scene-0074` (rung=2, CDE1).

## Contents

1. [Overview](#overview)
2. [Evaluation Modes](#evaluation-modes)
3. [Data Flow](#data-flow)
4. [File Schemas](#file-schemas)
5. [Ground Truth Extraction](#ground-truth-extraction)
6. [Prompt Format](#prompt-format)
7. [Output Parsing](#output-parsing)
8. [Metrics and Analysis](#metrics-and-analysis)
9. [Configuration](#configuration)
10. [Model Integration](#model-integration)
11. [CLI Usage](#cli-usage)
12. [Architecture Reference](#architecture-reference)
13. [Developer Guide](#developer-guide)

---

## Overview

Trajectory evaluation extends the CausalDriveBench evaluation framework with **ego-trajectory
prediction tasks**. It runs as an **optional, parallel pipeline** alongside the standard VQA
evaluation — using separate JSONL files, separate report files, and separate CLI flags — so it
has zero impact on existing VQA scores.

Three evaluation modes are defined:

| Mode | Abbr | Questions per sample | Ground truth source |
|------|------|---------------------|---------------------|
| Baseline Trajectory Evaluation | **BTE** | 1 | `state.json → ego.future` |
| Counterfactual Trajectory (Rung 2) | **CTE_R2** | 0..N | `alternate_trajectory.waypoints` |
| Counterfactual Trajectory (Rung 3) | **CTE_R3** | 0..N | `alternate_trajectory.waypoints` |

**Core metric:** Average Displacement Error (ADE) in metres.

---

## Evaluation Modes

### BTE — Baseline Trajectory Evaluation

- **One question per SAMPLED directory.**
- The model receives multi-camera images (past 1.5 s → now) and navigation context.
- Task: predict the ego vehicle's future waypoints at fixed timesteps.
- Ground truth: `state.json` → `ego.future` list of `{t, x, y, heading, speed}` dicts.
- Timesteps default to `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]` seconds.
- `question_id = "traj_BTE"`, `eval_type = "BTE"`.

### CTE — Counterfactual Trajectory Evaluation

- **One question per active_qa question that carries an `alternate_trajectory` field.**
- The model is given the VQA question (e.g. "Would you still stop?") **and** asked to predict
  the ego trajectory under the counterfactual scenario.
- The model must output **both** a VQA answer and waypoints in a single JSON block.
- Ground truth: `question["alternate_trajectory"]["waypoints"]` (excludes `t=0.0`).
- `question_id = "traj_CTE_{parent_question_id}"` (e.g. `traj_CTE_CR1`).
- `eval_type` encodes rung and question type: `"CTE_R2_CDE"`, `"CTE_R3_CR"`, etc.

**Rung mapping:**

| Rung | eval_type prefix | Description |
|------|-----------------|-------------|
| 2 | `CTE_R2_*` | Counterfactual direct-effect questions |
| 3 | `CTE_R3_*` | Counterfactual reasoning questions |

---

## Data Flow

```
benchmark records (state.json, active_qa.json)
        │
        ▼
ModelDataLoader.iter_trajectory_questions()
   ├─ extract_bte_ground_truth(state_json)   → BTE GT waypoints
   └─ find_trajectory_questions(active_qa)
        └─ extract_cte_ground_truth(question) → CTE GT waypoints + eval_type
        │
        ▼
save_trajectory_prompts() → trajectory_prompts.jsonl  (per sample dir)
        │
        ▼
ModelInference (trajectory branch)
   reads trajectory_prompts.jsonl
   runs model (same VQA interface, different qa_text)
        │
        ▼
trajectory_outputs.jsonl (per sample dir)
        │
        ▼
ModelPostprocessor.process_trajectory_outputs()
   ├─ parse_trajectory_output() → (pred_waypoints, vqa_answer)
   ├─ align_trajectories() → trim pred to GT range, interpolate GT
   ├─ compute_ade()
   ├─ analyze_bte() / analyze_cte()
   └─ build_trajectory_report() → trajectory_report.json (per sample dir)
        │
        ▼
aggregate_trajectory_reports() → trajectory_aggregate.json (per dataset dir)
```

**Key invariant:** Trajectory prompts/outputs/reports use separate filenames from VQA
(`trajectory_prompts.jsonl` vs `prompts.jsonl`, etc.) so both can coexist in the same sample
directory.

---

## File Schemas

### `trajectory_prompts.jsonl`

One JSON object per line. Each record includes:

```json
{
  "scene_id":          "nuscenes-scene-0072",
  "sample_id":         "SAMPLED_1",
  "question_id":       "traj_CTE_CR1",
  "prompt_id":         "000002",
  "eval_type":         "CTE_R3_CR",
  "trajectory_eval":   true,
  "answer_format":     "vqa+trajectory",
  "parent_question_id": "CR1",
  "parent_qa_type":    "ladder",
  "rung":              3,
  "question_type":     "CR",
  "gt_answer":         "Yes",
  "gt_timesteps":      [0.5, 1.0, 1.5],
  "gt_waypoints":      [{"t": 0.5, "x": 1.2, "y": 0.1}, ...],
  "vqa_question_text": "Would you still remain stopped?",
  "image_paths":       [...],
  "qa_text":           "<full prompt text passed to model>",
  "conversation":      [...]
}
```

BTE records use `"answer_format": "trajectory"`, `"parent_question_id": null`, and
omit `gt_answer`, `rung`, `question_type`, `vqa_question_text`.

### `trajectory_outputs.jsonl`

Same format as `outputs.jsonl`. Each record mirrors the prompt record plus:

```json
{
  "raw_output":   {"answer": "...", "extra": {"answer": "..."}},
  "latency_ms":   234,
  "is_evaluated": true,
  "error":        null
}
```

### `trajectory_report.json` (per sample)

```json
{
  "schema_version": "1.0",
  "level":          "sample",
  "scene_id":       "nuscenes-scene-0072",
  "sample_id":      "SAMPLED_1",
  "bte": {
    "question_id":    "traj_BTE",
    "eval_type":      "BTE",
    "parse_success":  true,
    "ade":            0.42,
    "n_waypoints_gt": 6,
    "n_waypoints_pred": 6,
    "n_eval_points":  6,
    "gt_t_range":     [0.5, 3.0],
    "analysis":       {"ade": 0.42, "grade": "EXCELLENT", "thresholds": {...}},
    "ground_truth_waypoints": [...],
    "predicted_waypoints":    [...]
  },
  "cte": [
    {
      "question_id":            "traj_CTE_CR1",
      "parent_question_id":     "CR1",
      "eval_type":              "CTE_R3_CR",
      "parse_success":          true,
      "ade":                    0.81,
      "n_waypoints_gt":         3,
      "n_waypoints_pred":       3,
      "n_eval_points":          3,
      "gt_t_range":             [0.5, 1.5],
      "vqa_answer_cte":         "Yes",
      "vqa_answer_original":    "Yes",
      "analysis": {
        "flags":             ["GOOD_COUNTERFACTUAL"],
        "grade":             "GOOD",
        "answer_match":      true,
        "cte_ade":           0.81,
        "bte_ade":           0.42,
        "cte_bte_mean_disp": 0.95,
        "thresholds":        {...}
      },
      "ground_truth_waypoints": [...],
      "predicted_waypoints":    [...]
    }
  ],
  "n_cte": 1,
  "summary": {
    "bte_ade":       0.42,
    "bte_grade":     "EXCELLENT",
    "cte_mean_ade":  0.81,
    "cte_grades":    ["GOOD"],
    "all_flags":     ["GOOD_COUNTERFACTUAL"]
  }
}
```

### `trajectory_aggregate.json` (per dataset)

Aggregates all sample-level trajectory reports:

```json
{
  "level":          "dataset",
  "causal_dataset": "causal_nuscenes",
  "n_samples":      50,
  "bte": {
    "n_evaluated":  48,
    "n_parse_fail": 2,
    "mean_ade":     0.73,
    "grade_counts": {"EXCELLENT": 12, "GOOD": 20, "FAIR": 10, "POOR": 6}
  },
  "cte": {
    "n_evaluated":  31,
    "n_parse_fail": 3,
    "mean_ade":     1.21,
    "grade_counts": {"EXCELLENT": 5, "GOOD": 12, "FLAGGED": 8, "INCONCLUSIVE": 6},
    "flag_counts":  {
      "ANSWER_MISMATCH": 4,
      "TRAJECTORY_SAME_AS_BTE": 3,
      "GOOD_COUNTERFACTUAL": 17,
      "FULL_MATCH": 5
    },
    "by_eval_type": {
      "CTE_R2_CDE": {"n": 15, "mean_ade": 1.10},
      "CTE_R3_CR":  {"n": 16, "mean_ade": 1.31}
    }
  }
}
```

---

## Ground Truth Extraction

### BTE Ground Truth

Source: `state.json → ego.future`.

The `ego.future` list contains waypoints at `t = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]` seconds
in ego-centric coordinates where `(0, 0)` is the vehicle's position at `T=0`.

```python
from evaluation.common.trajectory_eval import extract_bte_ground_truth
gt_wps = extract_bte_ground_truth(state_json)
# → [{"t": 0.5, "x": 1.2, "y": 0.0}, {"t": 1.0, "x": 2.5, "y": -0.1}, ...]
```

Raises `EvaluationException` if `ego.future` is absent or empty.

### CTE Ground Truth

Source: `active_qa.json → question["alternate_trajectory"]["waypoints"]`.

```json
"alternate_trajectory": {
  "frame":     "Tp0p0",
  "rung":      3,
  "waypoints": [
    {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "speed": 0.0},
    {"t": 0.5, "x": 0.3, "y": 0.0, "heading": 0.0, "speed": 0.5},
    {"t": 1.0, "x": 0.8, "y": 0.0, "heading": 0.0, "speed": 1.0},
    {"t": 1.5, "x": 1.5, "y": 0.0, "heading": 0.0, "speed": 1.5}
  ]
}
```

- `t=0.0` is excluded (current position is always `(0, 0)`; not a prediction target).
- The `rung` field drives `eval_type` (`CTE_R2_*` vs `CTE_R3_*`).
- The question's `id` suffix (e.g. `"CDE"`, `"CR"`, `"SC"`) drives the type tag.

```python
from evaluation.common.trajectory_eval import extract_cte_ground_truth
gt_wps, rung, eval_type = extract_cte_ground_truth(question)
# → ([{"t": 0.5, ...}, ...], 3, "CTE_R3_CR")
```

---

## Prompt Format

### BTE Prompt

Delivered via the model's VQA interface (`qa_text` field). The model sees:

```
You are an autonomous driving system with access to multi-camera images
from the past 1.5 seconds up to now (T=0).

Navigation command: STRAIGHT
Current ego speed:  8.30 m/s
Ego-centric frame:  (0, 0) = current position, +x = forward, +y = left

Task: Predict the ego vehicle's future trajectory waypoints.
Output timesteps (seconds): [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

Output ONLY the following JSON (replace x/y with your predictions):
{"waypoints": [{"t": 0.5, "x": 0.0, "y": 0.0}, {"t": 1.0, "x": 0.0, "y": 0.0}, ...]}
```

The model receives the same camera images as for VQA (multi-camera, 4 timesteps).

### CTE Prompt

Combines the VQA question with trajectory prediction in a single prompt:

```
You are an autonomous driving system with access to multi-camera images
from the past 1.5 seconds up to now (T=0).

Navigation command: STRAIGHT
Current ego speed:  0.00 m/s
Ego-centric frame:  (0, 0) = current position, +x = forward, +y = left

=== Causal Reasoning Question ===
Would you still remain stopped if the construction worker moved to the sidewalk?

Counterfactual scenario: Construction worker relocated to sidewalk

Task 1 — Answer the question (Yes or No).
Task 2 — Given your counterfactual reasoning, predict the ego vehicle's
trajectory under this scenario at timesteps: [0.5, 1.0, 1.5] seconds.

Output ONLY the following JSON (fill in answer and x/y values):
{"answer": "<Yes or No>", "waypoints": [{"t": 0.5, "x": 0.0, "y": 0.0}, ...]}
```

---

## Output Parsing

`parse_waypoints_from_text()` tries four strategies in order:

| Priority | Strategy | Matches |
|----------|---------|---------|
| 1 | Full CTE JSON | `{"answer": "...", "waypoints": [...]}` |
| 2 | BTE JSON | `{"waypoints": [...]}` |
| 3 | Bare array | `[{"t": ..., "x": ..., "y": ...}, ...]` |
| 4 | Heuristic | `t=X x=Y y=Z` or `t:X, x:Y, y:Z` patterns |

Returns `(waypoints_or_None, vqa_answer_or_None)`.

**Alpamayo 1.5 specific:** The `<|answer_end|>` token is stripped before parsing via
`Alpamayo15Postprocessor._preprocess_text()`.

---

## Metrics and Analysis

### ADE (Average Displacement Error)

```
ADE = (1/N) * Σ ||(pred_i - gt_i)||_2
```

where N is the number of **aligned** timesteps (pred trimmed to GT range; GT interpolated
to pred timesteps). No extrapolation beyond the GT horizon.

**Alignment algorithm:**
1. Trim prediction to GT time range `[gt_t_min, gt_t_max]` — no extrapolation.
2. Interpolate GT to the prediction's timesteps using `numpy.interp` (linear).
3. Pair the resulting arrays and compute L2 distances.

### BTE Grades

| Grade | Threshold |
|-------|-----------|
| EXCELLENT | ADE < 0.5 m |
| GOOD | ADE < 1.0 m |
| FAIR | ADE < 2.0 m |
| POOR | ADE ≥ 2.0 m |

### CTE Analysis Flags

Multiple flags may be set simultaneously:

| Flag | Condition |
|------|-----------|
| `ANSWER_MISMATCH` | CTE VQA answer ≠ original VQA answer |
| `TRAJECTORY_SAME_AS_BTE` | Mean CTE–BTE displacement < 0.30 m |
| `GOOD_COUNTERFACTUAL` | Trajectory distinct from BTE AND CTE ADE < 1.50 m |
| `FULL_MATCH` | Answers match + trajectory distinct + CTE ADE < 0.75 m |

### CTE Grades

| Grade | Conditions |
|-------|-----------|
| EXCELLENT | `FULL_MATCH` set |
| GOOD | `GOOD_COUNTERFACTUAL` set, no `ANSWER_MISMATCH` |
| FLAGGED | `ANSWER_MISMATCH` or `TRAJECTORY_SAME_AS_BTE` set |
| INCONCLUSIVE | None of the above |

### Cross-referencing original VQA answers

`process_trajectory_outputs()` reads `outputs.jsonl` alongside `trajectory_outputs.jsonl`.
It calls `parse_answer()` on the original VQA output for the parent question ID to get
`original_vqa_answer`, which feeds into `analyze_cte()` for `ANSWER_MISMATCH` checking.

---

## Configuration

`evaluation/eval_config.yaml` (under `evaluation:`):

```yaml
evaluation:
  use_trajectory_eval: false          # set true to enable by default
  trajectory_eval:
    bte_timesteps: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    eval_types: [BTE, CTE_R2, CTE_R3]
    bte_thresholds:
      excellent: 0.5
      good:      1.0
      fair:      2.0
    cte_thresholds:
      same_as_bte_mean_disp: 0.30
      good_ade:              1.50
      excellent_ade:         0.75
    max_generation_length: 1024
```

Threshold constants are also hardcoded in `trajectory_eval.py` (`BTE_THRESHOLDS`,
`CTE_THRESHOLDS`) as the authoritative source. The YAML values are for reference/documentation.

---

## Model Integration

All models reuse the base class machinery with minimal changes. Three methods must be
implemented or reviewed per model:

### 1. `DataLoader.get_trajectory_prompt()`

Override in the model's `dataloader.py`:

```python
def get_trajectory_prompt(
    self,
    formatted_sample: dict,
    traj_type: str,
    context_question: Optional[dict] = None,
    gt_waypoints: Optional[list] = None,
) -> dict:
    """Return model-specific prompt payload with qa_text set."""
    from evaluation.common.trajectory_eval import (
        build_bte_prompt_text, build_cte_prompt_text, BTE_TIMESTEPS,
    )
    state_json = formatted_sample.get("_state_json", {})
    gt_times = [w["t"] for w in gt_waypoints] if gt_waypoints else BTE_TIMESTEPS

    if traj_type == "BTE":
        qa_text = build_bte_prompt_text(state_json, gt_times)
    else:
        qa_text = build_cte_prompt_text(context_question, gt_times, state_json)

    # Build VQA-compatible message structure for this model
    return {
        "qa_text":      qa_text,
        "conversation": self._build_trajectory_conversation(qa_text, formatted_sample),
    }
```

Also ensure `format_sample()` stores `"_state_json"` in its return dict.

### 2. `Postprocessor.parse_trajectory_output()`

Override if the model needs pre-processing before waypoint parsing:

```python
def parse_trajectory_output(self, raw_output, prompt_record: dict) -> tuple:
    text = self._extract_text(raw_output)
    text = self._preprocess_text(text)          # e.g. strip <|answer_end|>
    return parse_waypoints_from_text(text)
```

The default in `base_postprocess.py` calls `parse_waypoints_from_text()` directly.

### 3. `Inference.main()` dispatch

Add `--trajectory-eval` and `--traj-only` CLI flags and dispatch:

```python
# In main():
do_vqa  = not args.traj_only
do_traj = args.trajectory_eval or args.traj_only

if do_vqa:
    run_from_jsonl(...)        # reads prompts.jsonl → outputs.jsonl

if do_traj:
    run_from_jsonl(
        prompts_file="trajectory_prompts.jsonl",
        outputs_file="trajectory_outputs.jsonl",
        ...
    )
```

### Alpamayo 1.5 implementation

- `dataloader.py`: `get_trajectory_prompt()` uses `create_vqa_message()` with the trajectory
  `qa_text`; images and ego history come from `formatted_sample` as usual.
- `postprocess.py`: `parse_trajectory_output()` strips `<|answer_end|>` then calls
  `parse_waypoints_from_text()`. `run_trajectory_postprocess()` is a standalone CLI entrypoint.
- `inference.py`: `--trajectory-eval` and `--traj-only` flags added; `main()` dispatches
  to separate `run_from_jsonl()` calls for VQA and trajectory.

---

## CLI Usage

### Data loading (generate trajectory_prompts.jsonl)

```bash
# Dry-run: generate prompts without running the model
python models/alpamayo_1_5/inference.py \
  --mode single --scene nuscenes-scene-0072 \
  --trajectory-eval --traj-only --dry-run \
  --run-dir /path/to/run_dir
```

### Inference (generate trajectory_outputs.jsonl)

```bash
# Run trajectory inference alongside VQA
python models/alpamayo_1_5/inference.py \
  --mode single --scene nuscenes-scene-0072 \
  --trajectory-eval

# Run trajectory inference only (skip VQA)
python models/alpamayo_1_5/inference.py \
  --mode single --scene nuscenes-scene-0072 \
  --traj-only
```

### Postprocessing (generate trajectory_report.json + aggregate)

```bash
python models/alpamayo_1_5/postprocess.py \
  --trajectory-eval \
  --run-dir /datadrive/datasets/cdb_outputs/alpamayo_1_5_20260424_120000

# With bench_dir override
python models/alpamayo_1_5/postprocess.py \
  --trajectory-eval \
  --run-dir /path/to/run_dir \
  --bench-dir /datadrive/datasets/causal_drive_bench_20260424/causal_nuscenes
```

### Docker

```bash
cd evaluation/models/alpamayo_1_5
TRAJECTORY_EVAL=true docker compose run --rm inference
RUN_DIR=... TRAJECTORY_EVAL=true docker compose --profile postprocess run --rm postprocess
```

---

## Architecture Reference

### Core module: `evaluation/common/trajectory_eval.py`

| Function | Purpose |
|----------|---------|
| `extract_bte_ground_truth(state_json)` | BTE GT from ego.future |
| `extract_cte_ground_truth(question)` | CTE GT from alternate_trajectory |
| `find_trajectory_questions(qa_data)` | Filter questions with alternate_trajectory |
| `get_eval_type(question)` | Derive eval_type string for a question |
| `interpolate_waypoints(gt, target_times)` | Linear interp within GT range |
| `align_trajectories(gt, pred)` | Trim + interpolate for ADE |
| `compute_ade(gt, pred)` | L2 mean error |
| `compute_mean_displacement(a, b)` | Mean L2 between two trajectories |
| `analyze_bte(ade)` | Grade with EXCELLENT/GOOD/FAIR/POOR |
| `analyze_cte(...)` | Flags + grade with CTE rubric |
| `parse_waypoints_from_text(text)` | 4-strategy waypoint extractor |
| `build_bte_prompt_text(state_json, times)` | BTE prompt string |
| `build_cte_prompt_text(question, times, state_json)` | CTE prompt string |
| `evaluate_bte_output(raw_text, gt, qid)` | Full BTE eval (parse + ADE + grade) |
| `evaluate_cte_output(raw_text, gt, ...)` | Full CTE eval (parse + ADE + flags) |
| `build_trajectory_report(...)` | Assemble trajectory_report.json dict |

### Base class extensions

| Class | File | New methods |
|-------|------|------------|
| `ModelDataLoader` | `base_dataloader.py` | `get_trajectory_prompt()`, `iter_trajectory_questions()`, `save_trajectory_prompts()` |
| `ModelPostprocessor` | `base_postprocess.py` | `parse_trajectory_output()`, `process_trajectory_outputs()`, `aggregate_trajectory_reports()` |

### Tests

`evaluation/tests/test_trajectory_eval.py` — 63 tests, 10 test classes:

- `TestExtractBteGroundTruth` (7 tests)
- `TestExtractCteGroundTruth` (8 tests)
- `TestFindTrajectoryQuestions` (5 tests)
- `TestInterpolateWaypoints` (5 tests)
- `TestAlignTrajectories` (6 tests)
- `TestComputeAde` (5 tests)
- `TestAnalyzeBte` (5 tests)
- `TestAnalyzeCte` (8 tests)
- `TestParseWaypointsFromText` (7 tests)
- `TestEvaluateBteOutput` (4 tests) + `TestEvaluateCteOutput` (3 tests)

---

## Developer Guide

### Adding trajectory eval to a new model

1. In `dataloader.py`:
   - Add `"_state_json": record.get("state", {})` to `format_sample()` return dict.
   - Implement `get_trajectory_prompt(formatted_sample, traj_type, context_question, gt_waypoints)`.
   - Build the model's native VQA message format using `qa_text` from
     `build_bte_prompt_text()` / `build_cte_prompt_text()`.

2. In `postprocess.py` (optional):
   - Override `parse_trajectory_output()` if the model adds tokens around JSON output.
   - Add `run_trajectory_postprocess(args, model_label)` calling
     `process_trajectory_outputs()` + `aggregate_trajectory_reports()`.

3. In `inference.py`:
   - Add `--trajectory-eval` / `--traj-only` CLI flags.
   - Dispatch to `run_from_jsonl()` with `trajectory_prompts.jsonl` → `trajectory_outputs.jsonl`.

4. Run tests:
   ```bash
   pytest evaluation/tests/test_trajectory_eval.py -v
   ```

### Dry-run validation

```bash
python evaluation/models/alpamayo_1_5/inference.py \
  --mode subset --subset-size 5 \
  --trajectory-eval --traj-only --dry-run \
  --bench-dir /datadrive/datasets/causal_drive_bench_20260424/causal_nuscenes \
  --run-dir /tmp/traj_dryrun
```

Check that `trajectory_prompts.jsonl` is created in each sample dir with correct
`eval_type`, `n_gt` waypoints, and `gt_timesteps`.

### Debugging parse failures

Set logging level to DEBUG to see which parse strategy matched (or failed):

```python
import logging
logging.basicConfig(level=logging.DEBUG)
from evaluation.common.trajectory_eval import parse_waypoints_from_text
wps, ans = parse_waypoints_from_text(model_raw_output)
```

`trajectory_report.json` entries with `"parse_success": false` include a
`"raw_text_excerpt"` (first 300 chars) for diagnosis.

### Validated test scenes

| Scene | Rung | Type | CTE questions |
|-------|------|------|---------------|
| `nuscenes-scene-0072` | 3 | CR | `traj_CTE_CR1` (n_gt=3, t=[0.5, 1.0, 1.5]) |
| `nuscenes-scene-0074` | 2 | CDE | `traj_CTE_CDE1` (n_gt=8, t=[0.5..4.0]) |

Both scenes produce `traj_BTE` with 6 waypoints at `t=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]`.
