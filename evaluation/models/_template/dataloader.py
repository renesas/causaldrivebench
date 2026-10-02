"""
Template DataLoader — copy this file to your model directory and implement
the three abstract methods per the TODO comments.

Per-dataset factory pattern
---------------------------
Each source dataset (nuscenes, argoverse2, openscene) has different camera rigs.
Three lightweight subclasses pre-configure the camera layout for each dataset,
and ``make_dataloader()`` returns the correct one automatically.

    loader = make_dataloader("nuscenes", dataset, model_cfg)
    loader = make_dataloader("argoverse2", dataset, model_cfg)
    loader = make_dataloader("openscene", dataset, model_cfg)

Key customisation points
------------------------
1. ``build_image_paths(record)``
       Select which cameras and time steps your model needs.
       Override per-dataset if camera naming differs (already handled below).

2. ``format_sample(record)``
       Heavy data work: load PIL images, tokenise state, interpolate ego history.

3. ``get_prompt(formatted_sample, question)``
       Model-specific prompt payload (conversation turns, tokens, etc.).
       Framework fields are injected by the base class.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow running from repo root or within container
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_dataloader import ModelDataLoader
from evaluation.common.dataset import BenchmarkDataset, FRAMES

# ---------------------------------------------------------------------------
# Per-dataset camera rig configurations
# Camera keys must match those in the corresponding dataset's frames.json.
# ---------------------------------------------------------------------------

DATASET_CAMERAS: Dict[str, List[str]] = {
    "nuscenes": [
        "cam_front",
        "cam_front_left",
        "cam_front_right",
        "cam_back",
    ],
    "argoverse2": [
        "cam_front",
        "cam_front_left",
        "cam_front_right",
        "cam_back_left",
        "cam_back_right",
    ],
    "openscene": [
        "cam_front",
        "cam_front_left",
        "cam_front_right",
        "cam_back",
    ],
}

# System prompt (loaded once at class definition time)
_SYSTEM_PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"


def _load_system_prompt() -> str:
    if _SYSTEM_PROMPT_FILE.exists():
        return _SYSTEM_PROMPT_FILE.read_text().strip()
    return (
        "You are an expert autonomous-driving assistant. "
        "Answer each question with the letter of the correct choice (A/B/C/D) "
        "or Yes/No for binary questions, followed by a brief justification.\n"
        "Format:\nAnswer: <answer>\nReasoning: <explanation>"
    )


_SYSTEM_PROMPT = _load_system_prompt()


# ---------------------------------------------------------------------------
# Base template DataLoader
# ---------------------------------------------------------------------------


class TemplateDataLoader(ModelDataLoader):
    """
    Baseline DataLoader for a vision-language model.

    Loads camera images across temporal frames and formats a chat-style prompt.
    Subclassed per dataset to configure the correct camera keys.

    Class variable ``CAMERAS`` controls which cameras are included.
    TODO: Narrow to the specific cameras your model uses.
    """

    # Override in per-dataset subclasses (or here to change for all datasets)
    CAMERAS: List[str] = DATASET_CAMERAS["nuscenes"]

    def __init__(self, dataset: BenchmarkDataset, model_cfg: dict) -> None:
        super().__init__(dataset, model_cfg)
        # TODO: initialise your model's tokeniser / image processor here, e.g.
        # from transformers import AutoProcessor
        # self.processor = AutoProcessor.from_pretrained(model_cfg.get("weights_path", ""))

    # ------------------------------------------------------------------
    # build_image_paths
    # ------------------------------------------------------------------

    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        """
        Build the ordered image path list for this model's input format.

        Default ordering: camera-major, temporal-minor (all 4 time steps
        per camera before moving to the next camera).

        Uses ``self.CAMERAS`` which is set per-dataset subclass.

        TODO: Change the ordering or subset time_keys to match your model.
              E.g. for front-cam only: set CAMERAS = ["cam_front"]
              E.g. for temporal-major: swap the loop order below.

        Returns
        -------
        List of ``{"path": str, "time_key": str, "camera_key": str}`` dicts.
        Missing frames are silently omitted (not all datasets have all cameras).
        """
        image_paths: List[Dict[str, str]] = []
        frames = record.get("frames", {})

        for cam_key in self.CAMERAS:
            for time_key in FRAMES:
                path = frames.get(time_key, {}).get(cam_key)
                if path := self.handle_missing_frame(path, label=f"{time_key}/{cam_key}"):
                    image_paths.append(
                        {"path": path, "time_key": time_key, "camera_key": cam_key}
                    )

        return image_paths

    # ------------------------------------------------------------------
    # format_sample
    # ------------------------------------------------------------------

    def format_sample(self, record: dict) -> dict:
        """
        Convert a benchmark record into model inputs.

        Loads camera images for all cameras in ``self.CAMERAS``, stringifies
        state.json, and interpolates ego history.  Reused for all questions.

        TODO: Adapt image loading, ego-history steps, and any additional
              model-specific preprocessing.
        """
        from PIL import Image  # lazy import; move to top after customisation

        frames = record.get("frames", {})
        state = record.get("state")

        # --- Load PIL images (using relative paths as-is) -----------------
        # TODO: Resolve paths via CDBConfig.resolve_frame_path(path) to get
        #       absolute paths.  Here we attempt a direct open for simplicity.
        images: List = []
        for cam_key in self.CAMERAS:
            for time_key in FRAMES:
                rel_path = frames.get(time_key, {}).get(cam_key)
                if rel_path:
                    abs_path = Path(rel_path)
                    if abs_path.exists():
                        images.append(Image.open(abs_path).convert("RGB"))

        # --- Ego history (interpolated to 4 steps) ------------------------
        # TODO: change n_steps to match your model's expected input length
        ego_history_xyz = self.interpolate_ego_history(state, n_steps=4)

        # --- Stringified state -------------------------------------------
        state_text = (
            json.dumps(state, indent=2) if state else "(state not available)"
        )

        return {
            "scene_id":        record["scene_id"],
            "sample_id":       record["sample_id"],
            "images":          images,
            "ego_history_xyz": ego_history_xyz,
            "state_text":      state_text,
        }

    # ------------------------------------------------------------------
    # get_prompt
    # ------------------------------------------------------------------

    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        """
        Build a model-specific prompt payload for one question.

        Returns model-specific fields only; the base class always sets:
        scene_id, sample_id, question_id, prompt_id, is_evaluated,
        question_json_file, answer_format, question_text, qa_text, image_paths.

        TODO: Replace the conversation structure with your model's expected
              format (LLaVA-style, OpenAI-style, Qwen-style, etc.).
        """
        question_text = self.format_question_text(question)
        answer_format = question.get("answer_format", "binary")

        if answer_format == "binary":
            answer_instruction = "Answer Yes or No, then explain your reasoning."
        else:
            answer_instruction = (
                "Answer with the letter (A, B, C, or D), "
                "then explain your reasoning."
            )

        user_content = (
            f"Scene state summary:\n"
            f"{formatted_sample['state_text'][:500]}\n\n"
            f"{question_text}\n\n"
            f"{answer_instruction}\n"
            f"Format:\nAnswer: <answer>\nReasoning: <explanation>"
        )

        # TODO: replace "conversation" with your model's expected prompt key
        conversation = self.build_chat_messages(
            system_prompt=_SYSTEM_PROMPT,
            user_content=user_content,
        )

        return {
            "conversation": conversation,
            # TODO: add model-specific fields, e.g.:
            # "ego_history_xyz": formatted_sample["ego_history_xyz"],
        }


# ---------------------------------------------------------------------------
# Per-dataset subclasses
# ---------------------------------------------------------------------------


class NuScenesDataLoader(TemplateDataLoader):
    """
    DataLoader configured for NuScenes camera rig.

    Cameras: cam_front, cam_front_left, cam_front_right, cam_back
    """
    CAMERAS = DATASET_CAMERAS["nuscenes"]


class Argoverse2DataLoader(TemplateDataLoader):
    """
    DataLoader configured for Argoverse2 camera rig.

    Cameras: cam_front, cam_front_left, cam_front_right,
             cam_back_left, cam_back_right
    Note: Argoverse2 has no dedicated rear-center camera.
    """
    CAMERAS = DATASET_CAMERAS["argoverse2"]


class OpenSceneDataLoader(TemplateDataLoader):
    """
    DataLoader configured for OpenScene camera rig.

    Cameras: cam_front, cam_front_left, cam_front_right, cam_back
    """
    CAMERAS = DATASET_CAMERAS["openscene"]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

_DATALOADER_REGISTRY: Dict[str, type] = {
    "nuscenes":   NuScenesDataLoader,
    "argoverse2": Argoverse2DataLoader,
    "openscene":  OpenSceneDataLoader,
}


def make_dataloader(
    dataset_name: str,
    dataset: BenchmarkDataset,
    model_cfg: dict,
) -> TemplateDataLoader:
    """
    Return the appropriate DataLoader subclass for *dataset_name*.

    Parameters
    ----------
    dataset_name : ``"nuscenes"``, ``"argoverse2"``, or ``"openscene"``
    dataset      : BenchmarkDataset instance pointing at the bench_dir
    model_cfg    : dict with model configuration (qa_types, batch_size, …)

    Returns
    -------
    Instantiated DataLoader subclass with the correct camera configuration.

    Raises
    ------
    ValueError if *dataset_name* is not recognised.

    Example
    -------
        loader = make_dataloader("argoverse2", dataset, model_cfg)
    """
    cls = _DATALOADER_REGISTRY.get(dataset_name)
    if cls is None:
        known = list(_DATALOADER_REGISTRY)
        raise ValueError(
            f"Unknown dataset_name '{dataset_name}'. "
            f"Known datasets: {known}"
        )
    return cls(dataset=dataset, model_cfg=model_cfg)


def dataset_name_from_bench_dir(bench_dir_name: str) -> str:
    """
    Derive the dataset name from a causal bench dir folder name.

    E.g. ``"causal_nuscenes"`` → ``"nuscenes"``

    Parameters
    ----------
    bench_dir_name : last component of the bench_dir path, e.g. ``"causal_argoverse2"``
    """
    if bench_dir_name.startswith("causal_"):
        return bench_dir_name[len("causal_"):]
    return bench_dir_name
