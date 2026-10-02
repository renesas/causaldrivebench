# Alpamayo 1.5 — Evaluation Integration Design

Model: `nvidia/Alpamayo-1.5-10B`
Repo: `https://github.com/NVlabs/alpamayo1.5`
Implementation: `evaluation/models/alpamayo_1_5/`

See `evaluation/models/alpamayo_1_5/DESIGN.md` for the full decision log.
This document is the canonical reference version for the `evaluation/docs/` collection.

---

## 1. VQA Interface over CoC Rollout

**Decision**: Use `helper.create_vqa_message()` + `model.generate_text()` rather than
`create_message()` + `sample_trajectories_from_data_with_vlm_rollout()` (the approach
used in Alpamayo 1.0).

**Why**: Alpamayo 1.5 introduces dedicated VQA support via `<|question_start|>` and
`<|question_end|>` special tokens that the model was explicitly trained to respond to
with a short, structured answer. `generate_text()` returns `extra["answer"]` — the
text between `<|answer_start|>` and `<|answer_end|>` — which is directly parseable
without stripping a CoC reasoning preamble.

Using the trajectory rollout path (`sample_trajectories_from_data_with_vlm_rollout()`)
works but is suboptimal: it generates trajectory tokens alongside the answer, is slower,
and requires more VRAM (~40 GB for multi-trajectory vs ~24 GB for VQA).

**Trade-off**: `generate_text()` skips ego trajectory conditioning (no
`<|traj_history_start|>` tokens in the VQA message format). For binary/MCQ causal
questions this is acceptable — trajectory tokens don't contribute to answering
"Is the cyclist causal to the ego?" type questions. If accuracy on trajectory-dependent
questions is unexpectedly low, switch to the CoC rollout path (same as 1.0, just
change the import and method call).

**How it works**:
1. Build messages via `helper.create_vqa_message(image_frames, qa_text, camera_indices)`.
2. Tokenise with `processor.apply_chat_template(..., continue_final_message=True)`.
3. Call `model.generate_text({"tokenized_data": inputs}, ...)`.
4. Extract `extra["answer"][0, 0]` (shape `[B, num_samples]`; B=1, num_samples=1).

---

## 2. Camera Indices for Display Name Annotations

**Decision**: Pass `camera_indices=torch.tensor([1, 0, 2, 4])` to `create_vqa_message()`.

**Why**: `helper._build_image_content()` annotates each camera group with its display
name ("Front camera:", "Rear camera:", etc.) when `camera_indices` is provided. This
matches the message format used during training on the PhysicalAI-AV dataset and
improves image-text spatial grounding for the VLM.

NuScenes camera → Alpamayo 1.5 camera index mapping:

| NuScenes key    | Alpamayo index | Display name       |
|-----------------|---------------|--------------------|
| cam_front       | 1             | Front camera       |
| cam_front_left  | 0             | Front left camera  |
| cam_front_right | 2             | Front right camera |
| cam_back        | 4             | Rear camera        |

The indices follow the `CAMERA_DISPLAY_NAMES` dict in `helper.py`. Camera-major ordering
(all timesteps for one camera before the next) is preserved, so `camera_indices` has one
entry per camera group, not per frame.

**Caution**: If more cameras are added in future datasets, extend `CAMERA_INDICES` in
`dataloader.py` and `CAMERA_ORDER`.

---

## 3. Ego History Stored but Not Used in VQA Path

**Decision**: Ego history (`ego_history_xyz`, `ego_history_rot`) is computed in
`dataloader.py` and written to `prompts.jsonl`, but not passed to `generate_text()`.

**Why**: `generate_text()` accepts only `{"tokenized_data": inputs}` — no trajectory
tensors. The VQA message format (`<|question_start|>...<|question_end|>`) does not
include `<|traj_history_start|>` tokens, so ego history cannot be tokenised. Storing
it in JSONL preserves the option to switch to the CoC rollout path without re-running
the dataloader.

---

## 4. Output Field: `extra["answer"]` vs `extra["cot"]`

**Decision**: Extract `extra["answer"][0, 0]` from `generate_text()` output.

**Why**: `generate_text()` returns a dict mapping text-field names to numpy string
arrays of shape `[B, num_samples]`. For VQA inference, `extra["answer"]` contains
the clean text between `<|answer_start|>` and `<|answer_end|>` — no CoC reasoning
prefix to strip. `extra["cot"]` and `extra["meta_action"]` are also returned but unused
for benchmark evaluation.

---

## 5. Postprocessor: `<|answer_end|>` Token Stripping

**Decision**: Strip `<|answer_end|>` (and everything after it) from `raw_output["text"]`
before regex parsing.

**Why**: If the model generates `<|answer_end|>` before reaching `max_generation_length`,
the token appears in the extracted text. Naive regex matching then fails to find
"Answer: Yes" if it is preceded by or embedded near this token. The strip is a one-line
regex applied before the cascading answer extraction patterns. This guard is not
needed in 1.0 because `extra["cot"]` text does not contain `<|answer_end|>`.

---

## 6. `<think>` Stripping Retained as Guard

**Decision**: Keep `<think>...</think>` stripping in the postprocessor even though
`generate_text()` does not produce reasoning model wrappers.

**Why**: If a future Alpamayo 1.5 variant or fine-tune uses the reasoning model format,
the postprocessor will still work correctly without code changes. The strip is a no-op
when no `<think>` block is present.

---

## 7. `physical-ai-av` Dependency

**Observation**: `pyproject.toml` lists `physical-ai-av==0.2.0`, an NVIDIA proprietary
package. If it is not available on PyPI, `uv sync --active` fails during Docker build.

**Resolution**: Contact NVIDIA for `physical-ai-av` access, or remove the dependency
from `pyproject.toml` before building. Our evaluation scripts (`dataloader.py`,
`inference.py`, `postprocess.py`) do not import it. The only consumer is
`load_physical_aiavdataset.py`, called only by the `original` service's
`src/alpamayo1_5/test_inference.py`.

---

## 8. `--env-file` in All Docker Commands

**Decision**: All Docker commands use `--env-file /path/to/dev.env` rather than relying
on a `dev.env` file in the same directory as `docker-compose.yml`.

**Why**: `dev.env` contains credentials and should not be co-located with the repo for
teams that manage secrets centrally. Docker Compose's default `.env` lookup is
directory-relative and silently uses the wrong file if the path is wrong. Explicit
`--env-file` eliminates ambiguity.

---

## 9. Per-Question Processing (batch_size=1)

**Decision**: Each QA question is processed independently by `run_single()`.

**Why**: The benchmark requires each question to be answered in isolation — shared
context across questions in the same sample would constitute cross-contamination.
`generate_text()` supports batching natively, but `batch_size=1` is the recommended
default to ensure each question is answered from the same state.

---

## Differences vs Alpamayo 1.0

| Aspect | Alpamayo 1.0 | Alpamayo 1.5 |
|--------|-------------|-------------|
| Package | `alpamayo_r1` | `alpamayo1_5` |
| Model class | `AlpamayoR1` | `Alpamayo1_5` |
| HF repo | `nvidia/Alpamayo-R1-10B` | `nvidia/Alpamayo-1.5-10B` |
| Inference method | `sample_trajectories_from_data_with_vlm_rollout()` | `generate_text()` |
| Message builder | `create_message()` + QA injection | `create_vqa_message()` |
| Output field | `extra["cot"]` | `extra["answer"]` |
| Ego conditioning | Yes (trajectory tokens in message) | No (VQA format has no traj tokens) |
| Camera display labels | No | Yes (`camera_indices` tensor) |
| Special token to strip | `<think>...</think>` | `<|answer_end|>` |
| Min VRAM | ~24 GB | ~24 GB (VQA path) |

---

## Learnings Applicable to Future Model Integrations

1. **Check for a dedicated VQA interface.** If the model has VQA-specific special tokens
   and a matching generation method, prefer that over repurposing a trajectory rollout.
   It produces shorter, cleaner outputs and uses less VRAM.

2. **Verify proprietary dataset dependencies before building.** Models released with
   training scripts often depend on closed datasets. Isolate evaluation-only imports
   and confirm they don't chain-import proprietary packages.

3. **Camera name mapping is always model-specific.** A wrong mapping silently degrades
   spatial grounding without producing any error. Always cross-reference the model's
   `CAMERA_DISPLAY_NAMES` or equivalent against the benchmark's camera keys.

4. **Store all computed state in `prompts.jsonl`.** Even if a model field isn't used
   by the current inference path (e.g., `ego_history_xyz` here), keeping it in JSONL
   means switching inference strategies doesn't require re-running the dataloader.

5. **Strip model-specific special tokens before returning text.** Tokens like
   `<|answer_end|>` or `<|cot_end|>` can appear in truncated outputs and break
   downstream regex parsing. Strip them in `parse_answer()`, not in `run_single()`,
   so the raw model output is preserved in `outputs.jsonl` for debugging.
