"""
Alpamayo 1.0 DataLoader for the causal scene understanding benchmark.

Adapts multi-camera frames and state.json to the Alpamayo R1 model's
expected input format: 4 cameras × 4 timesteps image sequences + ego trajectory.

IMAGE ORDERING — CAMERA-MAJOR (matching original Alpamayo training format):
  Original load_physical_aiavdataset.py stacks as (N_cameras, num_frames, 3, H, W)
  then calls .flatten(0, 1) → gives camera-major sequence:
    [cam0@t0, cam0@t1, cam0@t2, cam0@t3, cam1@t0, ..., cam3@t3]  (16 images total)

  We replicate this ordering here so the processor token embedding is consistent
  with what the model was trained on.

EGO HISTORY:
  NuScenes provides 4 timesteps at [-1.5, -1.0, -0.5, 0.0]s.
  The Alpamayo model expects 16 history steps at 0.1s intervals.
  We store the 4 sparse values in JSONL and interpolate to 16 at inference time.

MOVED FROM INFERENCE:
  - interpolate_ego_history() / _interpolate_ego_history()
  - inject_qa_into_messages() / _inject_qa_into_messages()
  - build_fallback_messages() / _build_fallback_messages()
"""

from __future__ import annotations

import copy
import math
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_dataloader import ModelDataLoader
from evaluation.common.dataset import BenchmarkDataset
from evaluation.common.utils.config_reader import CDBConfig

# ---------------------------------------------------------------------------
# Config — update constants here without touching logic
# ---------------------------------------------------------------------------
CONFIG = {
    # Temporal frames in chronological order (oldest → current), matching frames.json keys
    "frame_order": ["Tm1p5", "Tm1p0", "Tm0p5", "Tp0p0"],
    # Camera names matching frames.json and the causal benchmark
    # Order matters: must correspond to Alpamayo camera index assignment
    "camera_order": ["cam_front", "cam_front_left", "cam_front_right", "cam_back"],
    # Absolute time offsets (seconds) for each frame_key relative to T=0
    "frame_times_s": [-1.5, -1.0, -0.5, 0.0],
    # Fallback image size (W, H) used by PIL.Image.new() when a camera file is missing.
    # NuScenes camera images are 1600×900 — placeholder must match so torch.stack()
    # does not fail on mismatched spatial dimensions when some frames are missing.
    "placeholder_image_size_wh": (1600, 900),
    # Ego history interpolation: NuScenes has 4 sparse steps; model expects 16
    "num_history_steps": 16,
    "history_t_start_s": -1.5,
    "history_t_end_s": 0.0,
    "nuscenes_frame_times_s": [-1.5, -1.0, -0.5, 0.0],
}

# Expose as module-level constants for import by inference.py and tests
FRAME_ORDER: List[str] = CONFIG["frame_order"]
CAMERA_ORDER: List[str] = CONFIG["camera_order"]
FRAME_TIMES: List[float] = CONFIG["frame_times_s"]
_PLACEHOLDER_SIZE: tuple = CONFIG["placeholder_image_size_wh"]  # (W, H)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _heading_to_rotation_matrix(theta: float) -> np.ndarray:
    """
    Convert a 2D heading angle (radians) to a 3×3 rotation matrix (Z-axis rotation).

    The resulting matrix rotates vectors from the ego frame (pointing along theta)
    to the world frame.
    """
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[c, -s, 0.0],
         [s,  c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _compute_rotations(positions: np.ndarray) -> np.ndarray:
    """
    Compute per-timestep 3×3 rotation matrices from (N, 3) ego positions.

    Method: forward differences between consecutive positions to infer heading.
      heading[i] = atan2(dy, dx)  where  dx,dy = positions[i+1] - positions[i]
    The last timestep reuses the most recently computed heading.
    Falls back to theta=0 (identity-like forward rotation) for stationary steps.

    Args:
        positions: (N, 3) float32 array of ego positions in ego-centric frame.

    Returns:
        (N, 3, 3) float32 rotation matrices.
    """
    n = len(positions)
    rotations = np.zeros((n, 3, 3), dtype=np.float32)
    theta = 0.0  # default: facing forward along +X

    for i in range(n - 1):
        dx = positions[i + 1, 0] - positions[i, 0]
        dy = positions[i + 1, 1] - positions[i, 1]
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            theta = math.atan2(dy, dx)
        rotations[i] = _heading_to_rotation_matrix(theta)

    rotations[n - 1] = _heading_to_rotation_matrix(theta)
    return rotations


# ---------------------------------------------------------------------------
# Ego trajectory interpolation (moved from inference.py)
# ---------------------------------------------------------------------------

def interpolate_ego_history(
    xyz_sparse: np.ndarray,             # (4, 3) at sparse NuScenes timestamps
    rot_sparse: np.ndarray,             # (4, 3, 3)
    num_steps: int = 16,
    t_sparse: Optional[np.ndarray] = None,
    t_start: float = -1.5,
    t_end: float = 0.0,
) -> tuple:
    """
    Linearly interpolate sparse NuScenes ego trajectory to model's expected density.

    NuScenes provides 4 timesteps at [-1.5, -1.0, -0.5, 0.0]s.
    The Alpamayo model expects *num_steps* history steps covering the same window.

    Note on rotation interpolation: We use per-element linear interpolation of the
    3×3 rotation matrices.  This is not geometrically correct (SLERP would be ideal)
    but gives valid-enough values for VLM trajectory conditioning, especially given
    that the NuScenes angular velocities are small between consecutive steps.

    Returns:
        xyz_dense  : (num_steps, 3) float32
        rot_dense  : (num_steps, 3, 3) float32
    """
    if t_sparse is None:
        t_sparse = np.array(CONFIG["nuscenes_frame_times_s"], dtype=np.float64)

    t_dense = np.linspace(t_start, t_end, num_steps, dtype=np.float64)

    # Interpolate XYZ: 3 dimensions independently
    xyz_dense = np.zeros((num_steps, 3), dtype=np.float32)
    for dim in range(3):
        xyz_dense[:, dim] = np.interp(t_dense, t_sparse, xyz_sparse[:, dim])

    # Interpolate rotation matrix: 9 elements independently
    rot_dense = np.zeros((num_steps, 3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            rot_dense[:, i, j] = np.interp(t_dense, t_sparse, rot_sparse[:, i, j])

    return xyz_dense, rot_dense


def _interpolate_ego_history(
    xyz_sparse: np.ndarray,
    rot_sparse: np.ndarray,
) -> tuple:
    """
    Backward-compat wrapper: interpolate 4 → CONFIG['num_history_steps'] steps.

    Use ``interpolate_ego_history()`` directly for more control.
    """
    return interpolate_ego_history(
        xyz_sparse,
        rot_sparse,
        num_steps=CONFIG["num_history_steps"],
        t_start=CONFIG["history_t_start_s"],
        t_end=CONFIG["history_t_end_s"],
    )


# ---------------------------------------------------------------------------
# Message building helpers (moved from inference.py)
# ---------------------------------------------------------------------------

def inject_qa_into_messages(
    messages: list,
    qa_text: str,
    system_prompt: str,
    override_system_prompt: bool = False,
) -> list:
    """
    Return a deep copy of `messages` with QA question injected.

    - System message: replaced with `system_prompt` (override=True) or appended.
    - User message: the text instruction after trajectory tokens is replaced with
      `qa_text`. Trajectory tokens (<|traj_history_start|>…<|traj_history_end|>)
      are preserved so the model retains ego-history conditioning.

    The conversation structure becomes:
      system:    [QA system prompt]
      user:      [16 vision tokens] + traj_history_tokens + QA_question
      assistant: <|cot_start|>  (model continues generating from here)
    """
    _TRAJ_END = "<|traj_history_end|>"
    messages = copy.deepcopy(messages)

    # --- System message ---
    for msg in messages:
        if msg.get("role") == "system":
            if override_system_prompt:
                msg["content"] = [{"type": "text", "text": system_prompt}]
            else:
                msg["content"].append({"type": "text", "text": f"\n\n{system_prompt}"})
            break

    # --- User message: replace driving instruction, keep trajectory tokens ---
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            replaced = False
            for block in content:
                if block.get("type") == "text":
                    text = block["text"]
                    if _TRAJ_END in text:
                        # Keep everything up to and including <|traj_history_end|>,
                        # replace the "output the chain-of-thought…" driving instruction
                        # with our QA question.
                        traj_prefix = text[: text.index(_TRAJ_END) + len(_TRAJ_END)]
                        block["text"] = f"{traj_prefix}\n\n{qa_text}"
                    else:
                        block["text"] = f"{text}\n\n{qa_text}"
                    replaced = True
                    break
            if not replaced:
                content.append({"type": "text", "text": qa_text})
            break

    return messages


def build_fallback_messages(
    image_tensors,     # torch.Tensor (N, 3, H, W) — imported lazily at call site
    qa_text: str,
    system_prompt: str,
) -> list:
    """
    Build a Qwen3VL-compatible message list without helper.create_message().

    Used when the Alpamayo helper is unavailable or fails (e.g. import error).
    Does NOT include trajectory history tokens — trajectory conditioning relies
    only on the model_inputs tensors passed separately.

    Args:
        image_tensors: (N, 3, H, W) torch.Tensor of uint8 image frames
        qa_text:       full formatted QA question + format instruction
        system_prompt: system role content string

    Returns:
        3-element message list compatible with processor.apply_chat_template()
    """
    image_content = [{"type": "image", "image": frame} for frame in image_tensors]

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": image_content + [
                {
                    "type": "text",
                    "text": (
                        "These are camera views (front, front-left, front-right, back) "
                        "from a self-driving vehicle across 4 consecutive timesteps "
                        "(T-1.5s, T-1.0s, T-0.5s, T=0 current), shown in camera-major "
                        "order: all 4 timesteps for each camera, then the next camera."
                        f"\n\n{qa_text}"
                    ),
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|cot_start|>"}],
        },
    ]


# Backward-compat aliases (old names used by inference.py before refactor)
_inject_qa_into_messages = inject_qa_into_messages
_build_fallback_messages = build_fallback_messages


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class AlpamayoDataLoader(ModelDataLoader):
    """
    DataLoader for the Alpamayo R1-10B model on the causal driving benchmark.

    Converts a BenchmarkDataset record into:
      - camera-major ordered image path list (via build_image_paths)
      - 16 PIL.Images in camera-major order (via format_sample, for in-memory use)
      - Ego trajectory arrays (4 sparse steps) that inference interpolates to 16
      - Human-readable state_summary for debugging

    Camera-major ordering:
      [cam_front@Tm1p5, cam_front@Tm1p0, ..., cam_back@Tp0p0]  (16 entries)
    """

    def __init__(
        self,
        dataset: BenchmarkDataset,
        model_cfg: dict,
        cfg: Optional[CDBConfig] = None,
    ) -> None:
        super().__init__(dataset, model_cfg)
        self.cfg = cfg
        print(
            f"[AlpamayoDataLoader] Initialized.\n"
            f"  CAMERA_ORDER = {CAMERA_ORDER}\n"
            f"  FRAME_ORDER  = {FRAME_ORDER}\n"
            f"  Image ordering: CAMERA-MAJOR (cam-all-frames, then next-cam)"
        )

    # ------------------------------------------------------------------
    # build_image_paths (NEW — abstract method from base class)
    # ------------------------------------------------------------------

    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        """
        Build the camera-major ordered image path list for Alpamayo.

        Returns paths in CAMERA-MAJOR order (all 4 timesteps for each camera
        before moving to the next camera), matching the training data ordering.

        Missing camera/frame combinations are silently omitted.

        Returns
        -------
        List of {"path": str, "time_key": str, "camera_key": str} dicts,
        up to 16 entries (4 cameras × 4 timesteps).
        """
        image_paths: List[Dict[str, str]] = []
        frames = record.get("frames", {})

        for cam in CAMERA_ORDER:           # outer: cameras (camera-major)
            for frame_key in FRAME_ORDER:  # inner: timesteps
                path = frames.get(frame_key, {}).get(cam)
                if path := self.handle_missing_frame(path, label=f"{frame_key}/{cam}"):
                    image_paths.append(
                        {"path": path, "time_key": frame_key, "camera_key": cam}
                    )

        return image_paths

    # ------------------------------------------------------------------
    # format_sample
    # ------------------------------------------------------------------

    def format_sample(self, record: dict) -> dict:
        """
        Convert a BenchmarkDataset record into Alpamayo model inputs.

        Uses new record schema:
          - record["scene_id"]  (was record["scene"])
          - record["sample_id"] (was record["sample"])
          - record["state"]     (was frames["Tp0p0"]["state_json"])
          - record["frames"]    {time_key → {cam_key → relative_path}}

        Image loading order (CAMERA-MAJOR, 16 total):
          for each camera in CAMERA_ORDER:
            for each frame_key in FRAME_ORDER:
              load image

        Ego trajectory:
          state.json provides history at t = [-1.5, -1.0, -0.5] plus T=0.
          Stored as (4, 3) float32 in ego-centric frame (T=0 is the origin).
          inference.py interpolates these 4 steps to 16 steps.

        Note: PIL images may be grey placeholders if paths are relative and the
        data root is not the current working directory.  Use CDBConfig to resolve
        absolute paths for production inference.

        Returns:
          scene_id        – str
          sample_id       – str
          images          – 16 PIL.Images (CAMERA-MAJOR order); may include placeholders
          ego_history_xyz – (4, 3) float32 ego positions at [-1.5,-1.0,-0.5, 0.0]s
          ego_history_rot – (4, 3, 3) float32 heading-derived rotation matrices
          state_summary   – str, human-readable scene description for logging
        """
        from PIL import Image  # lazy import

        scene_id = record["scene_id"]
        sample_id = record["sample_id"]
        frames = record.get("frames", {})

        # state is now at record top level (was frames["Tp0p0"]["state_json"])
        state_json = record.get("state") or {}

        nav_cmd = state_json.get("nav_command", "UNKNOWN")
        print(f"[format_sample] {scene_id}/{sample_id} | nav_command={nav_cmd}")

        # ------------------------------------------------------------------
        # Load multi-camera images — CAMERA-MAJOR order
        # ------------------------------------------------------------------
        images: List = []
        missing_count = 0

        for cam in CAMERA_ORDER:           # outer loop: cameras
            for frame_key in FRAME_ORDER:  # inner loop: timesteps
                # New dataset API: frames[time_key][cam_key] = relative_path
                rel_path = frames.get(frame_key, {}).get(cam)

                if rel_path:
                    abs_path = (
                        self.cfg.resolve_frame_path(rel_path)
                        if self.cfg is not None
                        else Path(rel_path)
                    )
                    if abs_path.exists():
                        try:
                            img = Image.open(abs_path).convert("RGB")
                        except Exception:
                            img = Image.new("RGB", _PLACEHOLDER_SIZE, color=(128, 128, 128))
                            missing_count += 1
                    else:
                        img = Image.new("RGB", _PLACEHOLDER_SIZE, color=(128, 128, 128))
                        missing_count += 1
                        print(f"[format_sample]   WARN: file missing → {abs_path}")
                else:
                    img = Image.new("RGB", _PLACEHOLDER_SIZE, color=(128, 128, 128))
                    missing_count += 1
                    print(f"[format_sample]   WARN: no path for {frame_key}/{cam}")

                images.append(img)

        n_total = len(CAMERA_ORDER) * len(FRAME_ORDER)
        print(
            f"[format_sample] Loaded {n_total - missing_count}/{n_total} images "
            f"({missing_count} placeholder(s))"
        )

        # ------------------------------------------------------------------
        # Ego history in ego-centric frame (T=0 = origin)
        # ------------------------------------------------------------------
        # state.json "ego.history" provides entries at t = -1.5, -1.0, -0.5
        # We append [0, 0, 0] for t=0 (by definition of ego-centric frame)
        ego = state_json.get("ego", {})
        history_entries = ego.get("history", [])

        # Build time → entry map for safe access (keys are float seconds)
        hist_map = {float(e["t"]): e for e in history_entries if isinstance(e, dict) and "t" in e}

        positions: List[List[float]] = []
        for t in FRAME_TIMES[:-1]:  # -1.5, -1.0, -0.5 (historical frames)
            e = hist_map.get(t, {})
            x, y = float(e.get("x", 0.0)), float(e.get("y", 0.0))
            positions.append([x, y, 0.0])
        positions.append([0.0, 0.0, 0.0])  # t=0 is the ego-centric origin

        ego_history_xyz = np.array(positions, dtype=np.float32)  # (4, 3)
        ego_history_rot = _compute_rotations(ego_history_xyz)     # (4, 3, 3)

        print(
            f"[format_sample] Ego XY history: {ego_history_xyz[:, :2].tolist()} "
            f"(speed={ego.get('speed', 0.0):.2f} m/s)"
        )

        # ------------------------------------------------------------------
        # Human-readable state summary (for logging / debugging)
        # ------------------------------------------------------------------
        agents = state_json.get("agents", {})
        agent_lines = [
            f"  {name} ({info.get('category', '?')}): "
            f"x={info.get('position', {}).get('x', 0):.1f}m "
            f"y={info.get('position', {}).get('y', 0):.1f}m "
            f"speed={info.get('speed', 0):.1f} m/s"
            for name, info in agents.items()
        ]
        state_summary = (
            f"Navigation: {nav_cmd} | Ego speed: {ego.get('speed', 0.0):.2f} m/s\n"
            + ("\n".join(agent_lines) if agent_lines else "  No nearby agents")
        )

        return {
            "scene_id":        scene_id,
            "sample_id":       sample_id,
            # 16 PIL.Images in CAMERA-MAJOR order — NOT serialised to JSONL
            "images":          images,
            # (4, 3) float32 ego positions at FRAME_TIMES, ego-centric
            "ego_history_xyz": ego_history_xyz,
            # (4, 3, 3) float32 rotation matrices derived from forward differences
            "ego_history_rot": ego_history_rot,
            # Human-readable summary for logs
            "state_summary":   state_summary,
        }

    # ------------------------------------------------------------------
    # get_prompt
    # ------------------------------------------------------------------

    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        """
        Build model-specific payload fields for one causal QA question.

        Framework fields (scene_id, sample_id, question_id, prompt_id,
        is_evaluated, question_json_file, answer_format, question_text,
        qa_text, image_paths) are always injected by the base class and
        must NOT be returned here.

        Returns model-specific fields only:
          - ego_history_xyz: (4, 3) list-of-lists for JSON serialisation
          - ego_history_rot: (4, 3, 3) list-of-lists for JSON serialisation

        inference.py reads these alongside the framework fields from prompts.jsonl.
        """
        scene_id = formatted_sample["scene_id"]
        sample_id = formatted_sample["sample_id"]
        question_id = question.get("id", "")
        answer_format = question.get("answer_format", "binary")

        print(
            f"[get_prompt] {scene_id}/{sample_id} | q={question_id} fmt={answer_format}"
        )

        return {
            # Ego trajectory stored as nested lists for JSON serialisability.
            # Shape: (4, 3) and (4, 3, 3); inference.py interpolates to (16, 3) etc.
            "ego_history_xyz": formatted_sample["ego_history_xyz"].tolist(),
            "ego_history_rot": formatted_sample["ego_history_rot"].tolist(),
        }
