"""
Alpamayo 1.5 DataLoader for the causal scene understanding benchmark.

Adapts multi-camera frames and state.json to the Alpamayo 1.5 model's
expected input format: 4 cameras × 4 timesteps image sequences + ego trajectory.

IMAGE ORDERING — CAMERA-MAJOR (matching Alpamayo training format):
  The dataset loader stacks as (N_cameras, num_frames, 3, H, W) then calls
  .flatten(0, 1) → gives camera-major sequence:
    [cam0@t0, cam0@t1, cam0@t2, cam0@t3, cam1@t0, ..., cam3@t3]  (16 images)

  We replicate this ordering so the processor token embedding is consistent
  with training.

CAMERA INDICES — NEW IN 1.5:
  Alpamayo 1.5's helper.create_vqa_message() accepts an optional camera_indices
  tensor that annotates each image group with its camera display name:
    0: Front left  | 1: Front  | 2: Front right
    3: Rear left   | 4: Rear   | 5: Rear right  | 6: Front telephoto

  NuScenes camera → Alpamayo 1.5 camera index mapping:
    cam_front       → 1 (Front camera)
    cam_front_left  → 0 (Front left camera)
    cam_front_right → 2 (Front right camera)
    cam_back        → 4 (Rear camera)

EGO HISTORY:
  NuScenes provides 4 timesteps at [-1.5, -1.0, -0.5, 0.0]s.
  The Alpamayo model expects 16 history steps at 0.1s intervals.
  We store the 4 sparse values in JSONL and interpolate to 16 at inference time.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_dataloader import ModelDataLoader
from evaluation.common.dataset import BenchmarkDataset
from evaluation.common.utils.config_reader import CDBConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = {
    "frame_order": ["Tm1p5", "Tm1p0", "Tm0p5", "Tp0p0"],
    "camera_order": ["cam_front", "cam_front_left", "cam_front_right", "cam_back"],
    "frame_times_s": [-1.5, -1.0, -0.5, 0.0],
    # NuScenes image resolution — placeholder must match so torch.stack() succeeds
    "placeholder_image_size_wh": (1600, 900),
    "num_history_steps": 16,
    "history_t_start_s": -1.5,
    "history_t_end_s": 0.0,
    "nuscenes_frame_times_s": [-1.5, -1.0, -0.5, 0.0],
    # Alpamayo 1.5 camera display-name indices (see helper.CAMERA_DISPLAY_NAMES)
    "camera_indices": {
        "cam_front":       1,
        "cam_front_left":  0,
        "cam_front_right": 2,
        "cam_back":        4,
    },
}

FRAME_ORDER: List[str] = CONFIG["frame_order"]
CAMERA_ORDER: List[str] = CONFIG["camera_order"]
FRAME_TIMES: List[float] = CONFIG["frame_times_s"]
CAMERA_INDICES: Dict[str, int] = CONFIG["camera_indices"]
_PLACEHOLDER_SIZE: tuple = CONFIG["placeholder_image_size_wh"]  # (W, H)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _heading_to_rotation_matrix(theta: float) -> np.ndarray:
    """Convert a 2D heading angle (radians) to a 3×3 Z-axis rotation matrix."""
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

    Uses forward differences between consecutive positions to infer heading.
    Falls back to theta=0 (forward-facing) for stationary steps.
    """
    n = len(positions)
    rotations = np.zeros((n, 3, 3), dtype=np.float32)
    theta = 0.0

    for i in range(n - 1):
        dx = positions[i + 1, 0] - positions[i, 0]
        dy = positions[i + 1, 1] - positions[i, 1]
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            theta = math.atan2(dy, dx)
        rotations[i] = _heading_to_rotation_matrix(theta)

    rotations[n - 1] = _heading_to_rotation_matrix(theta)
    return rotations


# ---------------------------------------------------------------------------
# Ego trajectory interpolation
# ---------------------------------------------------------------------------

def interpolate_ego_history(
    xyz_sparse: np.ndarray,
    rot_sparse: np.ndarray,
    num_steps: int = 16,
    t_sparse: Optional[np.ndarray] = None,
    t_start: float = -1.5,
    t_end: float = 0.0,
) -> tuple:
    """
    Linearly interpolate sparse NuScenes ego trajectory to model's expected density.

    NuScenes provides 4 timesteps at [-1.5, -1.0, -0.5, 0.0]s.
    The Alpamayo model expects num_steps history steps covering the same window.

    Rotation interpolation is per-element linear (not SLERP). This is an
    approximation but sufficient given small angular velocities between steps.

    Returns:
        xyz_dense : (num_steps, 3) float32
        rot_dense : (num_steps, 3, 3) float32
    """
    if t_sparse is None:
        t_sparse = np.array(CONFIG["nuscenes_frame_times_s"], dtype=np.float64)

    t_dense = np.linspace(t_start, t_end, num_steps, dtype=np.float64)

    xyz_dense = np.zeros((num_steps, 3), dtype=np.float32)
    for dim in range(3):
        xyz_dense[:, dim] = np.interp(t_dense, t_sparse, xyz_sparse[:, dim])

    rot_dense = np.zeros((num_steps, 3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            rot_dense[:, i, j] = np.interp(t_dense, t_sparse, rot_sparse[:, i, j])

    return xyz_dense, rot_dense


def _interpolate_ego_history(xyz_sparse: np.ndarray, rot_sparse: np.ndarray) -> tuple:
    """Convenience wrapper: interpolate 4 → CONFIG['num_history_steps'] steps."""
    return interpolate_ego_history(
        xyz_sparse,
        rot_sparse,
        num_steps=CONFIG["num_history_steps"],
        t_start=CONFIG["history_t_start_s"],
        t_end=CONFIG["history_t_end_s"],
    )


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class Alpamayo15DataLoader(ModelDataLoader):
    """
    DataLoader for the Alpamayo 1.5 model on the causal driving benchmark.

    Converts a BenchmarkDataset record into:
      - Camera-major ordered image path list (via build_image_paths)
      - 16 PIL.Images in camera-major order (via format_sample)
      - Ego trajectory arrays (4 sparse steps) that inference interpolates to 16
      - Camera index list matching Alpamayo 1.5's display name scheme

    Camera-major ordering (16 images total):
      [cam_front@Tm1p5, cam_front@Tm1p0, ..., cam_back@Tp0p0]
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
            f"[Alpamayo15DataLoader] Initialized.\n"
            f"  CAMERA_ORDER   = {CAMERA_ORDER}\n"
            f"  CAMERA_INDICES = {[CAMERA_INDICES[c] for c in CAMERA_ORDER]}\n"
            f"  FRAME_ORDER    = {FRAME_ORDER}\n"
            f"  Image ordering : CAMERA-MAJOR (all timesteps per camera)"
        )

    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        """
        Build the camera-major ordered image path list.

        Returns paths in CAMERA-MAJOR order (all 4 timesteps for each camera
        before moving to the next), matching the training data ordering.
        Missing camera/frame combinations are silently omitted.

        Returns
        -------
        List of {"path": str, "time_key": str, "camera_key": str}, up to 16 entries.
        """
        image_paths: List[Dict[str, str]] = []
        frames = record.get("frames", {})

        for cam in CAMERA_ORDER:
            for frame_key in FRAME_ORDER:
                path = frames.get(frame_key, {}).get(cam)
                if path := self.handle_missing_frame(path, label=f"{frame_key}/{cam}"):
                    image_paths.append(
                        {"path": path, "time_key": frame_key, "camera_key": cam}
                    )

        return image_paths

    def format_sample(self, record: dict) -> dict:
        """
        Convert a BenchmarkDataset record into Alpamayo 1.5 model inputs.

        Image loading order (CAMERA-MAJOR, 16 total):
          for each camera in CAMERA_ORDER:
            for each frame_key in FRAME_ORDER:
              load image

        Ego trajectory:
          state.json provides history at t = [-1.5, -1.0, -0.5] plus T=0.
          Stored as (4, 3) float32 in ego-centric frame (T=0 is the origin).
          inference.py interpolates these 4 steps to 16 steps.

        Returns:
          scene_id        – str
          sample_id       – str
          images          – 16 PIL.Images (CAMERA-MAJOR order); may include placeholders
          ego_history_xyz – (4, 3) float32 ego positions at [-1.5,-1.0,-0.5, 0.0]s
          ego_history_rot – (4, 3, 3) float32 heading-derived rotation matrices
          state_summary   – str, human-readable scene description
        """
        from PIL import Image

        scene_id = record["scene_id"]
        sample_id = record["sample_id"]
        frames = record.get("frames", {})
        state_json = record.get("state") or {}

        nav_cmd = state_json.get("nav_command", "UNKNOWN")
        print(f"[format_sample] {scene_id}/{sample_id} | nav_command={nav_cmd}")

        # Load images — CAMERA-MAJOR order
        images: List = []
        missing_count = 0

        for cam in CAMERA_ORDER:
            for frame_key in FRAME_ORDER:
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

        # Ego history in ego-centric frame (T=0 = origin)
        ego = state_json.get("ego", {})
        history_entries = ego.get("history", [])
        hist_map = {float(e["t"]): e for e in history_entries if isinstance(e, dict) and "t" in e}

        positions = []
        for t in FRAME_TIMES[:-1]:
            e = hist_map.get(t, {})
            positions.append([float(e.get("x", 0.0)), float(e.get("y", 0.0)), 0.0])
        positions.append([0.0, 0.0, 0.0])  # t=0 is ego-centric origin

        ego_history_xyz = np.array(positions, dtype=np.float32)  # (4, 3)
        ego_history_rot = _compute_rotations(ego_history_xyz)     # (4, 3, 3)

        print(
            f"[format_sample] Ego XY history: {ego_history_xyz[:, :2].tolist()} "
            f"(speed={ego.get('speed', 0.0):.2f} m/s)"
        )

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
            "images":          images,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "state_summary":   state_summary,
            "_state_json":     state_json,   # kept for trajectory prompt building
        }

    def get_trajectory_prompt(
        self,
        formatted_sample: dict,
        traj_type: str,
        context_question: Optional[dict] = None,
        gt_waypoints: Optional[list] = None,
    ) -> dict:
        """
        Build a trajectory prompt payload for Alpamayo 1.5.

        The trajectory question text is placed in ``qa_text`` so the existing
        VQA inference pipeline (create_vqa_message + generate_text) handles it
        without any changes.  ego_history_xyz/rot are included for completeness.

        Parameters
        ----------
        formatted_sample : dict from format_sample()
        traj_type        : "BTE", "CTE_R2_CDE", "CTE_R3_CR", etc.
        context_question : for CTE types, the source VQA question dict
        gt_waypoints     : [{t, x, y}, ...] used to derive prompt timesteps

        Returns
        -------
        dict with qa_text (trajectory question) + ego history arrays.
        """
        from evaluation.common.trajectory_eval import (
            BTE_TIMESTEPS,
            build_bte_prompt_text,
            build_cte_prompt_text,
        )

        state_json = formatted_sample.get("_state_json") or {}
        gt_times = [w["t"] for w in gt_waypoints] if gt_waypoints else BTE_TIMESTEPS

        if traj_type == "BTE":
            qa_text = build_bte_prompt_text(state_json, gt_times)
            print(
                f"[get_trajectory_prompt] BTE {formatted_sample['scene_id']}/"
                f"{formatted_sample['sample_id']} timesteps={gt_times}"
            )
        elif context_question is not None:
            qa_text = build_cte_prompt_text(context_question, gt_times, state_json)
            print(
                f"[get_trajectory_prompt] {traj_type} "
                f"{formatted_sample['scene_id']}/{formatted_sample['sample_id']} "
                f"q={context_question.get('id','?')} timesteps={gt_times}"
            )
        else:
            raise ValueError(f"context_question required for traj_type={traj_type!r}")

        return {
            "qa_text":         qa_text,
            "ego_history_xyz": formatted_sample["ego_history_xyz"].tolist(),
            "ego_history_rot": formatted_sample["ego_history_rot"].tolist(),
        }

    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        """
        Build model-specific payload fields for one causal QA question.

        Framework fields (scene_id, sample_id, question_id, prompt_id,
        is_evaluated, question_json_file, answer_format, question_text,
        qa_text, image_paths) are injected by the base class and must NOT
        be returned here.

        Returns model-specific fields only:
          ego_history_xyz : (4, 3) list-of-lists for JSON serialisation
          ego_history_rot : (4, 3, 3) list-of-lists for JSON serialisation

        inference.py reads these alongside the framework fields from prompts.jsonl.
        """
        question_id = question.get("id", "")
        answer_format = question.get("answer_format", "binary")

        print(
            f"[get_prompt] {formatted_sample['scene_id']}/{formatted_sample['sample_id']} "
            f"| q={question_id} fmt={answer_format}"
        )

        return {
            "ego_history_xyz": formatted_sample["ego_history_xyz"].tolist(),
            "ego_history_rot": formatted_sample["ego_history_rot"].tolist(),
        }
