"""
Gemma 4 DataLoader for the causal scene understanding benchmark.

Adapts multi-camera frames and state.json into a text+image prompt suitable
for the Gemma 4 OpenAI-compatible vision API.

IMAGE ORDERING — CAMERA-MAJOR (matching CausalDriveBench convention):
  Cameras are grouped; within each group all 4 timesteps appear in order:
    [cam_front@Tm1p5, cam_front@Tm1p0, cam_front@Tm0p5, cam_front@Tp0p0,
     cam_front_left@Tm1p5, ..., cam_back@Tp0p0]   (16 images total)

  Text labels ("Front camera (T-1.5s to T=0):") are injected into the message
  at inference time so Gemma 4 knows which images belong to which camera.

EGO HISTORY:
  NuScenes provides 3 history waypoints (t = -1.5, -1.0, -0.5s) + current (t=0).
  These are stored as a human-readable text block in prompts.jsonl and included
  in the API message as scene context.

INFERENCE:
  Images are NOT base64-encoded here.  Only relative paths are stored in JSONL.
  The Gemma4Inference.run_single() loads, resizes, and encodes them at call time,
  keeping prompts.jsonl compact regardless of image count.
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
    # Human-readable camera labels for the API prompt
    "camera_labels": {
        "cam_front":       "Front camera",
        "cam_front_left":  "Front-left camera",
        "cam_front_right": "Front-right camera",
        "cam_back":        "Rear camera",
    },
    # Placeholder image size for missing frames (NuScenes native resolution)
    "placeholder_image_size_wh": (1600, 900),
}

FRAME_ORDER: List[str] = CONFIG["frame_order"]
CAMERA_ORDER: List[str] = CONFIG["camera_order"]
FRAME_TIMES: List[float] = CONFIG["frame_times_s"]
CAMERA_LABELS: Dict[str, str] = CONFIG["camera_labels"]
_PLACEHOLDER_SIZE: tuple = CONFIG["placeholder_image_size_wh"]


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class Gemma4DataLoader(ModelDataLoader):
    """
    DataLoader for Gemma 4 on the causal driving benchmark.

    Produces:
      - Camera-major ordered image path list (image_paths in prompts.jsonl)
      - Human-readable scene state text (state_summary, ego_history_text)
        to serve as structured context alongside the visual input

    The model has no proprietary camera index scheme — context is conveyed
    entirely through text labels in the API message.
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
            f"[Gemma4DataLoader] Initialized.\n"
            f"  CAMERA_ORDER = {CAMERA_ORDER}\n"
            f"  FRAME_ORDER  = {FRAME_ORDER}\n"
            f"  Image order  : CAMERA-MAJOR (all timesteps per camera)"
        )

    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        """
        Build the camera-major ordered image path list.

        Returns paths in the same CAMERA-MAJOR order used by alpamayo_1_5,
        ensuring cross-model reproducibility in the benchmark harness.
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
        Convert a BenchmarkDataset record into Gemma 4 model inputs.

        Loads PIL images for validation (missing → grey placeholder).
        Constructs human-readable scene state text for the prompt.

        Returns
        -------
        dict with keys:
          scene_id         – str
          sample_id        – str
          images           – List[PIL.Image], 16 images, CAMERA-MAJOR order
          state_summary    – str, scene context for the API message
          ego_history_text – str, formatted ego waypoints
        """
        from PIL import Image

        scene_id = record["scene_id"]
        sample_id = record["sample_id"]
        frames = record.get("frames", {})
        state_json = record.get("state") or {}

        nav_cmd = state_json.get("nav_command", "UNKNOWN")
        print(f"[format_sample] {scene_id}/{sample_id} | nav_command={nav_cmd}")

        # ── Load images (CAMERA-MAJOR order) ──────────────────────────────
        images: list = []
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

                images.append(img)

        n_total = len(CAMERA_ORDER) * len(FRAME_ORDER)
        print(
            f"[format_sample] Loaded {n_total - missing_count}/{n_total} images"
            + (f" ({missing_count} placeholder(s))" if missing_count else "")
        )

        # ── Ego history text ───────────────────────────────────────────────
        ego = state_json.get("ego", {})
        history_entries = ego.get("history", [])
        hist_map = {
            float(e["t"]): e
            for e in history_entries
            if isinstance(e, dict) and "t" in e
        }

        ego_lines = []
        for t in FRAME_TIMES:
            e = hist_map.get(t, {})
            x = float(e.get("x", 0.0))
            y = float(e.get("y", 0.0))
            spd = float(e.get("speed", 0.0))
            label = "current" if t == 0.0 else f"t={t:+.1f}s"
            ego_lines.append(f"  {label}: x={x:.2f}m  y={y:.2f}m  speed={spd:.2f} m/s")

        ego_history_text = "Ego vehicle history (ego-centric frame, T=0 is origin):\n" + "\n".join(ego_lines)

        # ── State summary ──────────────────────────────────────────────────
        agents = state_json.get("agents", {})
        agent_lines = [
            f"  {name} ({info.get('category', '?')}): "
            f"x={info.get('position', {}).get('x', 0):.1f}m "
            f"y={info.get('position', {}).get('y', 0):.1f}m "
            f"speed={info.get('speed', 0):.1f} m/s"
            for name, info in agents.items()
        ]
        state_summary = (
            f"Navigation command: {nav_cmd}\n"
            f"Ego speed: {ego.get('speed', 0.0):.2f} m/s\n"
            + ego_history_text
            + "\n\nNearby agents:\n"
            + ("\n".join(agent_lines) if agent_lines else "  None detected")
        )

        return {
            "scene_id":         scene_id,
            "sample_id":        sample_id,
            "images":           images,
            "state_summary":    state_summary,
            "ego_history_text": ego_history_text,
            "_state_json":      state_json,
        }

    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        """
        Build model-specific serialisable fields for one causal QA question.

        Framework fields (scene_id, sample_id, question_id, prompt_id,
        answer_format, question_text, qa_text, image_paths) are injected by
        the base class and must NOT be returned here.

        Returns
        -------
        dict with:
          state_summary    – scene state text (ego + agents) for the API message
          ego_history_text – ego waypoints formatted as text
        """
        question_id = question.get("id", "")
        answer_format = question.get("answer_format", "binary")
        print(
            f"[get_prompt] {formatted_sample['scene_id']}/{formatted_sample['sample_id']}"
            f" | q={question_id} fmt={answer_format}"
        )

        return {
            "state_summary":    formatted_sample["state_summary"],
            "ego_history_text": formatted_sample["ego_history_text"],
        }

    def get_trajectory_prompt(
        self,
        formatted_sample: dict,
        traj_type: str,
        context_question: Optional[dict] = None,
        gt_waypoints: Optional[list] = None,
    ) -> dict:
        """
        Build a trajectory prompt payload for Gemma 4.

        Places the trajectory question text in ``qa_text`` so the existing
        OpenAI API inference pipeline handles it unchanged.
        ``state_summary`` passes through as the scene context block.
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
                f"q={context_question.get('id', '?')} timesteps={gt_times}"
            )
        else:
            raise ValueError(f"context_question required for traj_type={traj_type!r}")

        return {
            "qa_text":         qa_text,
            "state_summary":   formatted_sample.get("state_summary", ""),
            "ego_history_text": formatted_sample.get("ego_history_text", ""),
        }
