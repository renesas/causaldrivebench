"""
Qwen2.5-VL DataLoader for CausalDriveBench.

Image ordering: all 4 cameras × 4 timesteps = 16 images total.
Order: timestep-major (Tm1p5 → Tp0p0), within each timestep
all cameras in order: cam_front, cam_front_left, cam_front_right, cam_back.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from PIL import Image

from evaluation.common.base_dataloader import ModelDataLoader
from evaluation.common.dataset import BenchmarkDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FRAME_ORDER: List[str] = ["Tm1p5", "Tm1p0", "Tm0p5", "Tp0p0"]
ALL_CAMS: List[str] = ["cam_front", "cam_front_left", "cam_front_right", "cam_back"]
PLACEHOLDER_SIZE: tuple = (1600, 900)

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"


def _load_system_prompt() -> str:
    if _SYSTEM_PROMPT_PATH.exists():
        return _SYSTEM_PROMPT_PATH.read_text().strip()
    return (
        "You are an expert autonomous driving assistant. "
        "Answer with 'Answer: Yes/No' or 'Answer: A/B/C/D' then explain."
    )


SYSTEM_PROMPT = _load_system_prompt()


def _load_image(path: str | None, data_root: str = "") -> Image.Image:
    """Load PIL image; return grey placeholder on failure."""
    if path:
        full = Path(data_root) / path if data_root else Path(path)
        if full.exists():
            try:
                return Image.open(full).convert("RGB")
            except Exception:
                pass
    return Image.new("RGB", PLACEHOLDER_SIZE, color=(128, 128, 128))


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class Qwen25VLDataLoader(ModelDataLoader):
    """
    DataLoader for Qwen2.5-VL on CausalDriveBench.

    Provides 16 images per sample:
      4 timesteps (Tm1p5, Tm1p0, Tm0p5, Tp0p0) ×
      4 cameras (cam_front, cam_front_left, cam_front_right, cam_back)
    Ordering: timestep-major.
    """

    def __init__(self, dataset: BenchmarkDataset, model_cfg: dict) -> None:
        super().__init__(dataset, model_cfg)
        self.data_root = model_cfg.get("data_root", os.environ.get("RAW_DATA_DIR", ""))
        n_total = len(FRAME_ORDER) * len(ALL_CAMS)
        print(
            f"[Qwen25VLDataLoader] Initialized.\n"
            f"  Cameras: {ALL_CAMS}\n"
            f"  Timesteps: {FRAME_ORDER}\n"
            f"  Total images per sample: {n_total}\n"
            f"  data_root: {self.data_root or '(relative paths)'}"
        )

    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        frames = record["frames"]
        paths: List[Dict[str, str]] = []

        for time_key in FRAME_ORDER:
            for cam in ALL_CAMS:
                path = frames.get(time_key, {}).get(cam, "")
                paths.append({"path": path, "time_key": time_key, "camera_key": cam})

        return paths

    def format_sample(self, record: dict) -> dict:
        frames = record["frames"]
        images: List[Image.Image] = []
        missing = 0

        for time_key in FRAME_ORDER:
            for cam in ALL_CAMS:
                path = frames.get(time_key, {}).get(cam)
                img = _load_image(path, self.data_root)
                if not path or not (Path(self.data_root) / path if self.data_root else Path(path)).exists():
                    missing += 1
                images.append(img)

        n_total = len(images)
        if missing:
            print(f"[format_sample] {n_total - missing}/{n_total} images ({missing} placeholder(s))")

        return {
            "scene_id": record["scene_id"],
            "sample_id": record["sample_id"],
            "images": images,
        }

    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        """
        Build prompt text following CausalDriveBench eval rules.
        Images are NOT embedded here (PIL is not JSON-serializable).
        inference.py loads images from image_paths and inserts them at runtime.

        - Rung 0-1: question only (no preamble)
        - Rung 2-3: minimal counterfactual premise + question
        """
        rung = question.get("rung", 0)
        answer_format = question.get("answer_format", "binary")
        question_text = question.get("question", "")
        options = question.get("options", {})

        if answer_format == "binary":
            answer_instruction = 'Answer with "Yes" or "No", then explain your reasoning.\nFormat:\nAnswer: Yes\nReasoning: <explanation>'
        else:
            if isinstance(options, dict):
                opts_str = "\n".join(f"{k}) {v}" for k, v in options.items())
            else:
                opts_str = "\n".join(f"{chr(65+i)}) {v}" for i, v in enumerate(options))
            answer_instruction = f'Choose one option (A, B, C, or D), then explain your reasoning.\nFormat:\nAnswer: <letter>\nReasoning: <explanation>\n{opts_str}'

        if rung >= 2:
            premise = question.get("intervention_premise") or question.get("counterfactual_premise", "")
            if premise:
                user_text = f"Suppose the following: {premise}\n\nQuestion: {question_text}\n\n{answer_instruction}"
            else:
                user_text = f"Question: {question_text}\n\n{answer_instruction}"
        else:
            user_text = f"Question: {question_text}\n\n{answer_instruction}"

        return {"user_text": user_text}
