"""
Doe-1 DataLoader for the causal scene understanding benchmark.

Adapts NuScenes multi-camera frames and state.json to the Doe-1 model's
expected input format. Doe-1 (wzzheng/Doe) is a world model that uses a
discrete visual token representation via a Chameleon-based VQVAE.

IMAGE ORDERING — TIME-MAJOR, FRONT-FOCUSED:
  Doe-1's training uses NuScenes with emphasis on front cameras for VQA.
  We provide the front camera across all 4 timesteps (temporal context)
  plus front-left, front-right at the current frame (spatial context).

    [cam_front@Tm1p5, cam_front@Tm1p0, cam_front@Tm0p5, cam_front@Tp0p0,
     cam_front_left@Tp0p0, cam_front_right@Tp0p0]   (6 images total)

  The most recent front-camera frame is provided last so it is closest to
  the question context in the conversation window.

NOTE:
  Doe-1 weights are NOT on HuggingFace. Download from Tsinghua Cloud:
  https://cloud.tsinghua.edu.cn/d/86e47c05830e4caaa40e/
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from PIL import Image

from evaluation.common.base_dataloader import ModelDataLoader
from evaluation.common.dataset import BenchmarkDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    # Front camera temporal sequence (all 4 timesteps)
    "temporal_camera": "cam_front",
    "frame_order": ["Tm1p5", "Tm1p0", "Tm0p5", "Tp0p0"],
    # Additional cameras at current frame only (spatial context)
    "spatial_cameras_at_t0": ["cam_front_left", "cam_front_right"],
    "frame_times_s": [-1.5, -1.0, -0.5, 0.0],
    "image_size_wh": (1600, 900),
}

FRAME_ORDER: List[str] = CONFIG["frame_order"]
_TEMPORAL_CAM: str = CONFIG["temporal_camera"]
_SPATIAL_CAMS: List[str] = CONFIG["spatial_cameras_at_t0"]
_PLACEHOLDER_SIZE: tuple = CONFIG["image_size_wh"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_state_summary(state_json: dict) -> str:
    ego = state_json.get("ego", {})
    agents = state_json.get("agents", {})
    nav_cmd = state_json.get("nav_command", "UNKNOWN")
    agent_lines = [
        f"  {name} ({info.get('category', '?')}): "
        f"x={info.get('position', {}).get('x', 0):.1f}m "
        f"v={info.get('speed', 0):.1f}m/s"
        for name, info in agents.items()
    ]
    return (
        f"Nav: {nav_cmd} | Ego speed: {ego.get('speed', 0.0):.2f} m/s\n"
        + ("\n".join(agent_lines) if agent_lines else "  No nearby agents")
    )


def _ego_history_text(state_json: dict) -> str:
    ego = state_json.get("ego", {})
    history = ego.get("history", [])
    pts = [(float(e.get("t", 0)), e.get("x", 0), e.get("y", 0)) for e in history]
    pts.append((0.0, 0.0, 0.0))
    return "Ego: " + ", ".join(f"t={t:.1f}s→({x:.1f},{y:.1f})m" for t, x, y in pts)


def _load_image(path: Optional[str], placeholder_size: tuple) -> Image.Image:
    """Load PIL image from path, returning grey placeholder on failure."""
    if path and Path(path).exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", placeholder_size, color=(128, 128, 128))


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DoeDataLoader(ModelDataLoader):
    """
    DataLoader for Doe-1 on the causal driving benchmark.

    Provides 6 images: front camera × 4 timesteps + front-left and
    front-right at the current timestep, in TIME-MAJOR order.
    """

    def __init__(self, dataset: BenchmarkDataset, model_cfg: dict) -> None:
        super().__init__(dataset, model_cfg)
        n_total = len(FRAME_ORDER) + len(_SPATIAL_CAMS)
        print(
            f"[DoeDataLoader] Initialized.\n"
            f"  Temporal camera: {_TEMPORAL_CAM} × {len(FRAME_ORDER)} frames\n"
            f"  Spatial cameras at t=0: {_SPATIAL_CAMS}\n"
            f"  Total images per sample: {n_total}"
        )

    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        """
        Return ordered list of image path metadata dicts.

        Image sequence:
          cam_front × 4 timesteps (temporal), then cam_front_left / cam_front_right
          at Tp0p0 (spatial).
        """
        frames = record["frames"]
        paths: List[Dict[str, str]] = []

        # Temporal: front camera across all frames
        for frame_key in FRAME_ORDER:
            path = frames.get(frame_key, {}).get(_TEMPORAL_CAM)
            paths.append({
                "path": path or "",
                "time_key": frame_key,
                "camera_key": _TEMPORAL_CAM,
            })

        # Spatial: side cameras at t=0 only
        for cam in _SPATIAL_CAMS:
            path = frames.get("Tp0p0", {}).get(cam)
            paths.append({
                "path": path or "",
                "time_key": "Tp0p0",
                "camera_key": cam,
            })

        return paths

    def format_sample(self, record: dict) -> dict:
        """
        Load images and build Doe-1 inputs from a benchmark record.

        Image sequence (TIME-MAJOR, front-focused):
          cam_front@Tm1p5, cam_front@Tm1p0, cam_front@Tm0p5, cam_front@Tp0p0,
          cam_front_left@Tp0p0, cam_front_right@Tp0p0
        """
        scene_id = record["scene_id"]
        sample_id = record["sample_id"]
        frames = record["frames"]
        state_json = record.get("state") or {}

        images: List[Image.Image] = []
        missing = 0

        # Temporal sequence: front camera across all frames
        for frame_key in FRAME_ORDER:
            path = frames.get(frame_key, {}).get(_TEMPORAL_CAM)
            img = _load_image(path, _PLACEHOLDER_SIZE)
            if not (path and Path(path).exists()):
                missing += 1
            images.append(img)

        # Spatial context: side cameras at current frame only
        for cam in _SPATIAL_CAMS:
            path = frames.get("Tp0p0", {}).get(cam)
            img = _load_image(path, _PLACEHOLDER_SIZE)
            if not (path and Path(path).exists()):
                missing += 1
            images.append(img)

        n_total = len(images)
        print(f"[format_sample] {n_total - missing}/{n_total} images ({missing} placeholder(s))")

        return {
            "scene_id": scene_id,
            "sample_id": sample_id,
            "images": images,
            "state_summary": _build_state_summary(state_json),
            "ego_text": _ego_history_text(state_json),
        }

    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        """Build model-specific extras for one causal question."""
        answer_format = question.get("answer_format", "binary")
        preamble = question.get("preamble", "")
        ego_text = formatted_sample.get("ego_text", "")

        if answer_format == "binary":
            format_instruction = (
                "Answer with 'Yes' or 'No', then briefly explain.\n"
                "Format:\nAnswer: Yes\nReasoning: <explanation>"
            )
        else:
            format_instruction = (
                "Choose the correct option (A, B, C, or D) and explain.\n"
                "Format:\nAnswer: <letter>\nReasoning: <explanation>"
            )

        context = f"{ego_text}\n\n{preamble}" if preamble else ego_text
        prompt_context = f"{context}\n\n{format_instruction}".strip()

        return {
            "prompt_context": prompt_context,
        }
