"""
SafeAuto DataLoader for the CausalDriveBench evaluation framework.

SafeAuto (ICML 2025) is a LLaVA-based model (LanguageBind image encoder +
LLaMA backbone) fine-tuned on the DriveLM dataset (nuScenes multi-view QA).

Input format:
  - 4 camera images at T=0 only (Tp0p0): front, front_left, front_right, back
  - DriveLM-style prompt with <image> tokens embedded in the human turn
  - Single-turn conversation per question (no multi-turn)

The prompt structure mirrors DriveLM training:
  "Following multi-view images record the driving scenario:
   CAM_FRONT: <image>, CAM_FRONT_LEFT: <image>, CAM_FRONT_RIGHT: <image>,
   CAM_BACK: <image>

   {question text}

   {format instruction}"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_dataloader import ModelDataLoader
from evaluation.common.dataset import BenchmarkDataset

# Only T=0 images — SafeAuto DriveLM does not use temporal history
_CAMERAS = ["cam_front", "cam_front_left", "cam_front_right", "cam_back"]

# CausalDriveBench camera key → DriveLM display label
_CAM_LABELS = {
    "cam_front":       "CAM_FRONT",
    "cam_front_left":  "CAM_FRONT_LEFT",
    "cam_front_right": "CAM_FRONT_RIGHT",
    "cam_back":        "CAM_BACK",
}


class SafeAutoDataLoader(ModelDataLoader):
    """
    DataLoader for SafeAuto on the CausalDriveBench nuScenes benchmark.

    Uses only the T=0 front/front-left/front-right/back camera images,
    matching the DriveLM training distribution.
    """

    def __init__(self, dataset: BenchmarkDataset, model_cfg: dict) -> None:
        super().__init__(dataset, model_cfg)

    # ------------------------------------------------------------------
    # build_image_paths
    # ------------------------------------------------------------------

    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        """
        Return T=0 image paths for the 4 nuScenes cameras.

        SafeAuto DriveLM was trained on single-timestep multi-view images;
        temporal frames are not used.
        """
        paths: List[Dict[str, str]] = []
        t0_frames = record.get("frames", {}).get("Tp0p0", {})

        for cam in _CAMERAS:
            path = t0_frames.get(cam)
            if path := self.handle_missing_frame(path, label=f"Tp0p0/{cam}"):
                paths.append({"path": path, "time_key": "Tp0p0", "camera_key": cam})

        return paths

    # ------------------------------------------------------------------
    # format_sample
    # ------------------------------------------------------------------

    def format_sample(self, record: dict) -> dict:
        """
        Lightweight: extract scene/sample IDs only.

        Images are loaded at inference time from the paths in prompts.jsonl.
        """
        return {
            "scene_id":  record["scene_id"],
            "sample_id": record["sample_id"],
        }

    # ------------------------------------------------------------------
    # get_prompt
    # ------------------------------------------------------------------

    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        """
        Build a DriveLM-style single-turn prompt for one causal QA question.

        The prompt embeds one <image> token per camera, matching how SafeAuto
        was trained on DriveLM.  inference.py passes this string through
        tokenizer_X_token() which replaces each <image> with the image feature
        index (-201, the VIDEO index as used during DriveLM training).

        Returns model-specific field only:
            conversation_text : str — full user turn with embedded <image> tokens
        """
        question_text = self.format_question_text(question)
        answer_format = question.get("answer_format", "binary")

        if answer_format == "binary":
            format_hint = "Answer Yes or No, followed by your descriptive reasoning."
        else:
            format_hint = (
                "Answer with the single correct letter (A, B, C, or D), "
                "followed by your descriptive reasoning."
            )

        image_block = ", ".join(
            f"{_CAM_LABELS[cam]}: <image>" for cam in _CAMERAS
        )

        conversation_text = (
            f"Following multi-view images record the driving scenario: {image_block}\n\n"
            f"{question_text}\n\n"
            f"{format_hint}"
        )

        return {"conversation_text": conversation_text}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset_name: str,
    dataset: BenchmarkDataset,
    model_cfg: dict,
) -> SafeAutoDataLoader:
    """
    Return a SafeAutoDataLoader for the given dataset.

    Only nuScenes and openscene are supported (both use the same 4-camera rig).
    Argoverse2 has a different camera set and would need a separate subclass.
    """
    if dataset_name not in ("nuscenes", "openscene"):
        raise ValueError(
            f"SafeAuto currently supports 'nuscenes' and 'openscene'; "
            f"got '{dataset_name}'"
        )
    return SafeAutoDataLoader(dataset=dataset, model_cfg=model_cfg)


def dataset_name_from_bench_dir(bench_dir_name: str) -> str:
    """E.g. 'causal_nuscenes' → 'nuscenes'."""
    if bench_dir_name.startswith("causal_"):
        return bench_dir_name[len("causal_"):]
    return bench_dir_name
