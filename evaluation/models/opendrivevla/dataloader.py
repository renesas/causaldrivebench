"""
OpenDriveVLA DataLoader for the causal scene understanding benchmark.

Adapts NuScenes multi-camera frames and state.json to OpenDriveVLA's expected
input format. OpenDriveVLA (DriveVLA/OpenDriveVLA) is based on Qwen2.5-VL and
processes multi-view NuScenes images for VLA tasks.

IMAGE ORDERING — TIME-MAJOR, RECENT FRAMES:
  Qwen2.5-VL handles multiple images efficiently via dynamic resolution.
  We provide 2 timesteps (most recent: Tm0p5 and Tp0p0) × all 4 cameras
  for a balance of temporal context and inference efficiency:

    [cam_front@Tm0p5, cam_fl@Tm0p5, cam_fr@Tm0p5, cam_back@Tm0p5,
     cam_front@Tp0p0, cam_fl@Tp0p0, cam_fr@Tp0p0, cam_back@Tp0p0]  (8 images)

  The 2-frame window covers ~0.5s of motion — sufficient for causal questions
  about immediate actions. Extend FRAME_ORDER to ["Tm1p5","Tm1p0","Tm0p5","Tp0p0"]
  for full temporal context if GPU memory allows.

DATA ALIGNMENT:
  OpenDriveVLA's inference_drivevla.py expects NuScenes data processed through
  UniAD feature extraction (optionally). For our VQA eval, we use raw images
  directly with the Qwen2.5-VL processor.
"""

from __future__ import annotations

import sys
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
    # Most recent 2 frames (Qwen2.5-VL efficiency vs. context tradeoff)
    # Extend to all 4 frames for richer temporal context at cost of tokens.
    "frame_order": ["Tm0p5", "Tp0p0"],
    "camera_order": ["cam_front", "cam_front_left", "cam_front_right", "cam_back"],
    "frame_times_s": [-0.5, 0.0],
    "image_ordering": "time_major",
    "image_size_wh": (1600, 900),
}

FRAME_ORDER: List[str] = CONFIG["frame_order"]
CAMERA_ORDER: List[str] = CONFIG["camera_order"]
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
    if path and Path(path).exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", placeholder_size, color=(128, 128, 128))


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class OpenDriveVLADataLoader(ModelDataLoader):
    """
    DataLoader for OpenDriveVLA (Qwen2.5-VL) on the causal driving benchmark.

    Provides 2 frames × 4 cameras = 8 images in TIME-MAJOR order.
    Qwen2.5-VL processes these with its dynamic resolution visual encoder.
    """

    def __init__(self, dataset: BenchmarkDataset, model_cfg: dict) -> None:
        super().__init__(dataset, model_cfg)
        n_images = len(FRAME_ORDER) * len(CAMERA_ORDER)
        print(
            f"[OpenDriveVLADataLoader] Initialized.\n"
            f"  CAMERA_ORDER = {CAMERA_ORDER}\n"
            f"  FRAME_ORDER  = {FRAME_ORDER} (most recent 2 frames)\n"
            f"  Total images per sample: {n_images}"
        )

    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        """Build ordered list of image path metadata dicts (TIME-MAJOR)."""
        frames = record.get("frames", {})
        image_paths = []
        for frame_key in FRAME_ORDER:
            for cam in CAMERA_ORDER:
                path = frames.get(frame_key, {}).get(cam)
                if path:
                    image_paths.append({"path": path, "time_key": frame_key, "camera_key": cam})
        return image_paths

    def format_sample(self, record: dict) -> dict:
        """
        Load recent 2-frame × 4-camera image grid in TIME-MAJOR order.

        Sequence: all cameras at t=-0.5s, then all cameras at t=0.0s.
        """
        scene_id = record["scene_id"]
        sample_id = record["sample_id"]
        frames = record.get("frames", {})
        state_json = record.get("state") or {}

        print(f"[format_sample] {scene_id}/{sample_id} | nav={state_json.get('nav_command', '?')}")

        images: List[Image.Image] = []
        missing = 0

        for frame_key in FRAME_ORDER:
            for cam in CAMERA_ORDER:
                path = frames.get(frame_key, {}).get(cam)
                img = _load_image(path, _PLACEHOLDER_SIZE)
                if not (path and Path(path).exists()):
                    missing += 1
                images.append(img)

        n_total = len(FRAME_ORDER) * len(CAMERA_ORDER)
        print(f"[format_sample] {n_total - missing}/{n_total} images ({missing} placeholder(s))")

        return {
            "scene_id": scene_id,
            "sample_id": sample_id,
            "images": images,
            "state_summary": _build_state_summary(state_json),
            "ego_text": _ego_history_text(state_json),
        }

    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        """
        Build a QA prompt payload for one causal question.

        Returns only prompt_context (model-specific field).
        Framework fields are injected by the base class.
        """
        answer_format = question.get("answer_format", "binary")
        ego_text = formatted_sample.get("ego_text", "")
        preamble = question.get("preamble", "")

        if answer_format == "binary":
            format_instruction = (
                "Answer 'Yes' or 'No' and briefly explain.\n"
                "Format:\nAnswer: Yes\nReasoning: <explanation>"
            )
        else:
            format_instruction = (
                "Choose the correct option (A, B, C, or D) and explain.\n"
                "Format:\nAnswer: <letter>\nReasoning: <explanation>"
            )

        n_images = len(FRAME_ORDER) * len(CAMERA_ORDER)
        layout_desc = (
            f"You are shown {n_images} camera images from a self-driving vehicle "
            f"({len(FRAME_ORDER)} timesteps × {len(CAMERA_ORDER)} cameras: "
            f"{', '.join(CAMERA_ORDER)}), ordered past-to-present by timestep."
        )

        context_parts = [layout_desc, ego_text]
        if preamble:
            context_parts.append(preamble)
        context_parts.append(format_instruction)
        prompt_context = "\n\n".join(context_parts).strip()

        return {"prompt_context": prompt_context}
