# Alpamayo 1.0 — Evaluation Integration Design

Model: `nvidia/Alpamayo-R1-10B`
Repo: `https://github.com/NVlabs/alpamayo`
Implementation: `evaluation/models/alpamayo_1_0/`

---

## 1. Inference Path: CoC Rollout via `sample_trajectories_from_data_with_vlm_rollout()`

**Decision**: Run the model's trajectory rollout method and extract the chain-of-causation (CoC) text from `extra["cot"]`.

**Why**: Alpamayo R1 is a trajectory-prediction model whose primary interface is
`sample_trajectories_from_data_with_vlm_rollout()`. This method simultaneously
generates a reasoning trace (CoC) and predicts future trajectories. For the benchmark
we care only about the CoC text, which contains the model's structured answer to the
QA question. Using the trajectory rollout path keeps inference identical to the
model's trained usage and ensures the ego trajectory conditioning is active.

**How it works**:
1. Build messages via `helper.create_message(image_frames)` — produces system + user
   messages with 16 vision tokens and trajectory history tokens.
2. Inject the QA question via `inject_qa_into_messages()` — appends `qa_text` after
   the `<|traj_history_end|>` token in the user message and augments the system role
   with the evaluation system prompt.
3. Tokenise with `processor.apply_chat_template(..., continue_final_message=True)` —
   generation continues from `<|cot_start|>`.
4. Call `model.sample_trajectories_from_data_with_vlm_rollout()` with ego history.
5. Extract `extra["cot"]` and unwrap the nested list/array structure (batch=1, traj_sets=1,
   num_samples=1 → three levels of indexing).

---

## 2. Camera-Major Image Ordering

**Decision**: Load 16 images in camera-major order:
`[cam_front@Tm1p5, ..., cam_front@Tp0p0, cam_front_left@Tm1p5, ..., cam_back@Tp0p0]`

**Why**: The original `load_physical_aiavdataset.py` stacks images as
`(N_cameras, num_frames, 3, H, W)` then calls `.flatten(0, 1)`, which gives camera-major
order. `helper.create_message()` processes the resulting `(16, 3, H, W)` tensor and
embeds vision tokens in this order. Preserving this ordering ensures the positional
embeddings seen at inference match training.

**Concrete order** (outer loop: cameras, inner loop: timesteps):
```
CAMERA_ORDER = ["cam_front", "cam_front_left", "cam_front_right", "cam_back"]
FRAME_ORDER  = ["Tm1p5", "Tm1p0", "Tm0p5", "Tp0p0"]
```

---

## 3. Tensor Type Requirement for `helper.create_message()`

**Decision**: Convert all images to `torch.Tensor(N, C, H, W)` uint8 before passing
to `helper.create_message()`.

**Why**: The helper checks `frames.ndim == 4` and expects a PyTorch tensor. Passing
numpy arrays or PIL Images causes a crash. The pipeline converts each PIL image to
numpy via `np.array()`, permutes HWC → CHW, then stacks 16 frames into a single tensor.

---

## 4. Ego History Interpolation: 4 → 16 Steps

**Decision**: Linearly interpolate sparse NuScenes ego trajectory (4 steps at
-1.5, -1.0, -0.5, 0.0 s) to the 16 dense steps the model expects.

**Why**: The model was trained on the PhysicalAI-AV dataset at 10 Hz (16 steps over 1.5 s).
NuScenes provides only 4 ego history frames at 0.5 s intervals. We store the 4 sparse
values in `prompts.jsonl` and interpolate at inference time so the dataloader remains
dataset-agnostic.

**Rotation interpolation**: Per-element linear interpolation of 3×3 rotation matrices.
This is not geometrically correct (SLERP would be ideal) but sufficient in practice
because NuScenes heading changes between consecutive 0.5 s steps are small.

**Shape contract** after interpolation:
```
ego_history_xyz : (1, 1, 16, 3)   — unsqueeze twice from (16, 3)
ego_history_rot : (1, 1, 16, 3, 3) — unsqueeze twice from (16, 3, 3)
```

---

## 5. QA Injection into Trajectory Messages

**Decision**: Augment `helper.create_message()` output rather than building messages
from scratch.

**Why**: `create_message()` constructs the correct trajectory token sequence
(`<|traj_history_start|>` ... `<|traj_history_end|>`) and vision token layout.
Building this from scratch would require replicating fragile internals. Instead
`inject_qa_into_messages()` deep-copies the message list, replaces the driving
instruction in the user turn (after `<|traj_history_end|>`) with `qa_text`, and
augments the system role with the evaluation system prompt.

**Fallback**: If `helper.create_message()` fails (import error, API change), 
`build_fallback_messages()` constructs a minimal Qwen3VL message without trajectory
tokens. This loses ego conditioning but keeps inference running.

---

## 6. CoC Text Extraction

**Decision**: Unwrap `extra["cot"]` with three levels of indexing (`cot[0][0][0]`).

**Why**: `sample_trajectories_from_data_with_vlm_rollout()` returns `extra["cot"]` as
a nested structure of shape `[batch, traj_sets, num_traj_samples]`. With batch=1,
traj_sets=1, num_traj_samples=1, three index operations reach the raw string. The code
uses a loop that checks for list/array/tuple and indexes `[0]` up to three times,
then converts to `str()` — robust to shape changes.

---

## 7. Postprocessor: Cascading Regex for Robust Answer Extraction

**Decision**: Apply cascading regex patterns from most specific to least specific.

**Why**: The CoC is a long reasoning trace. The model is instructed to end with
`"Answer: Yes/No"` or `"Answer: A"`, but may produce slight variations
("answer is Yes", bare "Yes", etc.). Cascading from tight patterns to loose ones
maximises extraction rate without sacrificing precision on well-formed outputs.

**Binary cascade** (checked in order):
1. `Answer\s*:\s*(Yes|No)` — explicit label
2. `answer\s+is\s+(Yes|No)` — paraphrase
3. `^(Yes|No)[.,\s]` — line-start
4. `\b(Yes|No)\b` — bare word (last resort)

**MCQ cascade**:
1. `Answer\s*:\s*([A-D])` — explicit label
2. `answer\s+is\s+([A-D])` — paraphrase
3. `Option\s+([A-D])` — option reference
4. `([A-D])[).]` — letter with punctuation
5. `^\s*([A-D])\s*$` — lone letter on a line
6. `\b([A-D])\b` — bare letter (last resort)

The `<think>...</think>` block (reasoning model wrapper) is searched first when
present, falling back to the full text.

---

## 8. Placeholder Images for Missing Frames

**Decision**: Substitute a grey `(1600, 900)` RGB image for any missing camera frame.

**Why**: `torch.stack()` requires all tensors to have identical spatial dimensions.
A missing frame cannot be silently dropped — the model expects exactly 16 images.
Grey (128, 128, 128) is used because it is a neutral mid-range value that doesn't
strongly activate any visual features, making it a safe no-op.

---

## 9. GPU-Lazy Model Load

**Decision**: Check whether all `outputs.jsonl` files are non-empty before calling
`load_model()`.

**Why**: Loading AlpamayoR1 (22 GB, bfloat16) takes 30–60 s and occupies GPU memory
for the lifetime of the process. If inference is being re-run on a directory where
all samples are already complete (e.g., resume after accidental re-trigger), skipping
the load avoids unnecessary GPU allocation.

---

## 10. Output Schema

`raw_output` written to `outputs.jsonl`:
```json
{"text": "<think>\n...CoC reasoning...\n</think>\nAnswer: Yes"}
```

The `text` field is the raw CoC string extracted from `extra["cot"]`. The
`<think>...</think>` wrapper appears when the model is in reasoning mode.
`parse_answer()` handles both wrapped and unwrapped formats.
