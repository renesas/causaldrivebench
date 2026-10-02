"""
ReCogDrive DataLoader for the causal scene understanding benchmark.

Adapts NuScenes multi-camera frames and state.json to ReCogDrive's expected
input format. ReCogDrive (xiaomi-research/recogdrive) uses InternVL3 as its
VLM backbone with a diffusion planner head for NAVSIM planning evaluation.

IMAGE ORDERING — TIME-MAJOR, RECENT FRAMES:
  ReCogDrive processes NuScenes-derived NAVSIM scenes with multi-view images.
  We provide recent 2 timesteps × all 4 cameras (8 images total):

    [cam_front@Tm0p5, cam_fl@Tm0p5, cam_fr@Tm0p5, cam_back@Tm0p5,
     cam_front@Tp0p0, cam_fl@Tp0p0, cam_fr@Tp0p0, cam_back@Tp0p0]

  InternVL3 processes images at 448×448 per its CLIPImageProcessor.
  The model handles multi-image inputs via InternVL's <image> token protocol.

GPU REQUIREMENT:
  ReCogDrive requires GPU with SM >= 8 (Ampere or newer) for flash-attn.
  Use 8B variant (owl10/ReCogDrive-8B-RL) for best QA performance;
  2B variant (owl10/ReCogDrive-2B-RL) for memory-constrained setups.

INTERNVL MULTI-IMAGE CONVENTION:
  Each image requires one <image>\n token in the question string.
  inference.py builds the full InternVL-compatible question with image tokens.
"""

from __future__ import annotations

import os
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
    # 4 timesteps × 4 cameras = 16 images (configurable via FRAME_ORDER env var)
    "frame_order": os.environ.get("FRAME_ORDER", "Tm1p5,Tm1p0,Tm0p5,Tp0p0").split(","),
    "camera_order": ["cam_front", "cam_front_left", "cam_front_right", "cam_back"],
    "image_ordering": "time_major",
    # Native NuScenes resolution (InternVL3 resizes to 448×448 internally)
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

class ReCogDriveDataLoader(ModelDataLoader):
    """
    DataLoader for ReCogDrive (InternVL3) on the causal driving benchmark.

    Provides 2 frames × 4 cameras = 8 images in TIME-MAJOR order.
    Image tokens are prepended by inference.py following InternVL convention.
    """

    def __init__(self, dataset: BenchmarkDataset, model_cfg: dict) -> None:
        super().__init__(dataset, model_cfg)
        n_images = len(FRAME_ORDER) * len(CAMERA_ORDER)
        print(
            f"[ReCogDriveDataLoader] Initialized.\n"
            f"  CAMERA_ORDER = {CAMERA_ORDER}\n"
            f"  FRAME_ORDER  = {FRAME_ORDER}\n"
            f"  Total images per sample: {n_images} (InternVL3 processes at 448×448)"
        )

    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        """Build ordered list of image path metadata dicts (TIME-MAJOR)."""
        frames = record.get("frames", {})
        raw_data_dir = self.model_cfg.get("raw_data_dir", "")
        data_root = Path(raw_data_dir).parent if raw_data_dir else Path(".")

        image_paths = []
        for frame_key in FRAME_ORDER:
            frame = frames.get(frame_key, {})
            cameras = frame.get("cameras", frame)
            for cam in CAMERA_ORDER:
                path = cameras.get(cam)
                if path:
                    abs_path = str(data_root / path)
                    image_paths.append({"path": abs_path, "time_key": frame_key, "camera_key": cam})
        return image_paths

    def format_sample(self, record: dict) -> dict:
        """Load frames × cameras images in TIME-MAJOR order."""
        scene_id = record["scene_id"]
        sample_id = record["sample_id"]
        frames = record.get("frames", {})
        state_json = record.get("state") or {}
        raw_data_dir = self.model_cfg.get("raw_data_dir", "")
        data_root = Path(raw_data_dir).parent if raw_data_dir else Path(".")

        print(f"[format_sample] {scene_id}/{sample_id} | nav={state_json.get('nav_command', '?')}")

        images: List[Image.Image] = []
        missing = 0

        for frame_key in FRAME_ORDER:
            frame = frames.get(frame_key, {})
            cameras = frame.get("cameras", frame)
            for cam in CAMERA_ORDER:
                rel_path = cameras.get(cam)
                abs_path = str(data_root / rel_path) if rel_path else None
                img = _load_image(abs_path, _PLACEHOLDER_SIZE)
                if not (abs_path and Path(abs_path).exists()):
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

        Each image is labeled with its timestamp and camera view inline,
        following the DriveLM convention so the model knows which image is which.

        Returns prompt_context (model-specific field) with <image> tokens already
        embedded — inference.py must NOT prepend additional image tokens.
        """
        _FRAME_LABELS = {
            "Tm1p5": "t=-1.5s", "Tm1p0": "t=-1.0s",
            "Tm0p5": "t=-0.5s", "Tp0p0": "t=0.0s",
        }
        _CAM_LABELS = {
            "cam_front": "FRONT VIEW", "cam_front_left": "FRONT LEFT VIEW",
            "cam_front_right": "FRONT RIGHT VIEW", "cam_back": "BACK VIEW",
        }

        answer_format = question.get("answer_format", "binary")
        ego_text = formatted_sample.get("ego_text", "")
        preamble = question.get("preamble", "")

        # Build labeled image section matching DriveLM's fine-tuning format:
        # "[t=Xs] <VIEW NAME>:\n<image>" — familiar label style + temporal context
        image_lines = ["The following images are from a self-driving vehicle across multiple timesteps:"]
        for frame_key in FRAME_ORDER:
            t_label = _FRAME_LABELS.get(frame_key, frame_key)
            for cam in CAMERA_ORDER:
                cam_label = _CAM_LABELS.get(cam, cam.upper())
                image_lines.append(f"[{t_label}] <{cam_label}>:\n<image>")
        image_section = "\n".join(image_lines)

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

        context_parts = [image_section, ego_text]
        if preamble:
            context_parts.append(preamble)
        context_parts.append(format_instruction)
        prompt_context = "\n\n".join(context_parts).strip()

        return {"prompt_context": prompt_context}
