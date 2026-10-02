# Alpamayo 1.5 Evaluation Integration — Design Decisions

## Overview

This document captures design decisions, key differences from Alpamayo 1.0, and
recommendations for future model integrations.

---

## 1. VQA Interface vs CoC Rollout

**Decision**: Use `helper.create_vqa_message()` + `model.generate_text()` instead of
`create_message()` + `sample_trajectories_from_data_with_vlm_rollout()`.

**Why**: Alpamayo 1.5 introduces dedicated VQA support with `<|question_start|>` and
`<|question_end|>` special tokens trained to elicit short, structured answers. The
`generate_text()` method returns `extra["answer"]` — a clean answer string without
trajectory tokens or CoC reasoning preamble. This is better suited to the binary/MCQ
benchmark than the CoC rollout which was designed for trajectory prediction with
reasoning traces.

**Trade-off**: The CoC rollout (1.0 approach) conditions on ego trajectory history,
which can anchor reasoning to ego motion context. `generate_text()` skips trajectory
conditioning. For QA accuracy this is a net benefit — trajectory tokens don't
contribute to answering "Is the cyclist causal to the ego?" questions.

**Alternative**: Use `create_message()` + QA injection + trajectory rollout (same as
1.0). This extracts `extra["cot"]` and can leverage ego history conditioning. Implement
if accuracy on trajectory-dependent questions is lower than expected with VQA approach.

---

## 2. Camera Indices for Display Name Annotations

**Decision**: Pass `camera_indices=torch.tensor([1, 0, 2, 4])` to `create_vqa_message()`.

**Why**: `helper._build_image_content()` annotates each camera group with its display
name ("Front camera:", "Front left camera:", etc.) when `camera_indices` is provided.
This matches the training format used by PhysicalAI-AV dataset loader and improves
image-text alignment for the VLM. The mapping from NuScenes camera names to Alpamayo
camera indices:

| NuScenes key    | Alpamayo index | Display name      |
|-----------------|---------------|-------------------|
| cam_front       | 1             | Front camera      |
| cam_front_left  | 0             | Front left camera |
| cam_front_right | 2             | Front right camera|
| cam_back        | 4             | Rear camera       |

**Caution**: If Alpamayo 1.5 adds more cameras (cam_back_left, cam_back_right) in
future datasets, extend `CAMERA_INDICES` in `dataloader.py` and `CAMERA_ORDER`.

---

## 3. No Ego History Conditioning in VQA Path

**Decision**: `generate_text()` does not receive `ego_history_xyz`/`ego_history_rot`.

**Why**: The VQA message format (`<|question_start|>...<|question_end|>`) doesn't
include `<|traj_history_start|>` tokens, so ego history is not tokenized. We still
store ego history in `prompts.jsonl` for compatibility and potential future use
(e.g., switching back to CoC rollout).

---

## 4. Output Field: `extra["answer"]` vs `extra["cot"]`

**Decision**: Extract `extra["answer"][0, 0]` as the raw text for postprocessing.

**Why**: `generate_text()` returns a dict with keys `"cot"`, `"meta_action"`, `"answer"`,
each of shape `[B, num_samples]` (numpy string arrays). For VQA questions, `"answer"`
contains the text between `<|answer_start|>` and `<|answer_end|>`. This is the clean,
parseable answer — no CoC prefix to strip.

**Postprocessor note**: The `<|answer_end|>` token may appear in the raw text if the
model generates it before the max token limit. `parse_answer()` strips it with regex
before parsing. This is a robustness guard not needed in 1.0.

---

## 5. `physical-ai-av` Dependency

**Observation**: `pyproject.toml` lists `physical-ai-av==0.2.0` which is an NVIDIA
proprietary package. If it is not available on PyPI, `uv sync --active` will fail
during Docker build.

**Recommendation**: Contact NVIDIA for `physical-ai-av` access, or remove it from
`pyproject.toml` before building since our evaluation scripts (`inference.py`,
`dataloader.py`) do not import it. The only user of `physical-ai-av` is
`load_physical_aiavdataset.py` which is called by the `original` service's
`test_inference.py`.

---

## 6. `--env-file` in Docker Commands

**Decision**: All Docker commands in `commands.md` use `--env-file /path/to/dev.env`
rather than assuming `dev.env` lives next to `docker-compose.yml`.

**Why**: Users may manage environment files in a central secrets directory separate from
the repo. Docker Compose's default `.env` lookup is directory-relative and would miss
files stored elsewhere. Explicit `--env-file` makes the path unambiguous and avoids
accidental credential exposure in the repo.

---

## 7. Inference Architecture: Per-Question Processing

**Decision**: Each QA question is processed independently by `run_single()` (batch_size=1
in practice).

**Why**: The framework enforces individual question processing for consistency — each
question must be answered independently of others to avoid context bleed. Even though
Alpamayo 1.5's `generate_text()` supports batching, we keep `batch_size=1` as the
recommended default.

---

## Learnings and Recommendations

### For Future Model Integrations

1. **Check for dedicated VQA interface first**. If the model has a VQA-specific entry
   point (like `create_vqa_message()` + `generate_text()`), prefer it over hacking
   trajectory rollout for QA. It's more aligned with training and produces cleaner outputs.

2. **Check for proprietary dataset dependencies**. Models released with training scripts
   may depend on closed datasets. Verify that evaluation-only imports can be isolated
   before building the Docker image.

3. **Camera name mapping is model-specific**. Always check the model's camera index
   scheme against the benchmark's camera names. A mismatched mapping silently degrades
   spatial grounding without producing errors.

4. **Store all intermediate state in `prompts.jsonl`**. Even if a model field (like
   `ego_history_xyz`) isn't used by the current inference path, keeping it in JSONL
   makes it easy to switch inference strategies without re-running the dataloader.

5. **Always strip model-specific special tokens before returning text**. Tokens like
   `<|answer_end|>` or `<|cot_end|>` can appear in truncated outputs. Stripping them
   in the postprocessor prevents regex matching failures downstream.

### Differences vs Alpamayo 1.0

| Aspect | Alpamayo 1.0 | Alpamayo 1.5 |
|--------|-------------|-------------|
| Package | `alpamayo_r1` | `alpamayo1_5` |
| Model class | `AlpamayoR1` | `Alpamayo1_5` |
| HF repo | `nvidia/Alpamayo-R1-10B` | `nvidia/Alpamayo-1.5-10B` |
| Inference path | `sample_trajectories_from_data_with_vlm_rollout()` | `generate_text()` |
| Message builder | `create_message()` + QA injection | `create_vqa_message()` |
| Output field | `extra["cot"]` | `extra["answer"]` |
| Ego conditioning | Yes (trajectory tokens) | No (VQA format) |
| Camera labels | No | Yes (`camera_indices`) |
| Special token strip | `<think>...</think>` | `<|answer_end|>` |
