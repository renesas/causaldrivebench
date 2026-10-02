"""
OmniDrive DataLoader for the causal scene understanding benchmark.

Adapts NuScenes multi-camera frames and state.json to OmniDrive's expected
input format: multi-view images across 4 timesteps (6 cameras in training,
4 available in our benchmark).

IMAGE ORDERING — TIME-MAJOR:
  OmniDrive processes temporal sequences via a PETR-based temporal transformer.
  We order images by timestep (oldest → newest), with all cameras at each step:
    [cam_front@Tm1p5, cam_fl@Tm1p5, cam_fr@Tm1p5, cam_back@Tm1p5,
     cam_front@Tm1p0, ..., cam_back@Tp0p0]   (16 images total)

  This matches the temporal-attention expectation: same-timestep images are
  adjacent, making it easier for the model to reason across viewpoints.

CAMERA ADAPTATION:
  OmniDrive trains on 6 NuScenes cameras; our benchmark provides 4.
  Mapping: cam_front, cam_front_left, cam_front_right, cam_back.
  cam_back_left and cam_back_right are unavailable — no placeholders are
  inserted since OmniDrive's VQA head conditions on whatever is provided.

EGO HISTORY:
  State.json provides 3 past positions plus the current origin.
  Stored for optional use in inference context string.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_dataloader import ModelDataLoader
from evaluation.common.dataset import BenchmarkDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    # Temporal frames — oldest to newest (TIME-MAJOR ordering)
    "frame_order": ["Tm1p5", "Tm1p0", "Tm0p5", "Tp0p0"],
    # Cameras available in causal bench (OmniDrive trains on 6; we have 4)
    "camera_order": ["cam_front", "cam_front_left", "cam_front_right", "cam_back"],
    "frame_times_s": [-1.5, -1.0, -0.5, 0.0],
    # TIME-MAJOR: outer loop = timesteps, inner loop = cameras
    # Produces groups of same-timestep images, aiding temporal attention
    "image_ordering": "time_major",
    "image_size_wh": (1600, 900),  # NuScenes native; model resizes internally
}

_frame_override = os.environ.get("FRAME_ORDER", "")
FRAME_ORDER: List[str] = _frame_override.split(",") if _frame_override else CONFIG["frame_order"]
CAMERA_ORDER: List[str] = CONFIG["camera_order"]
FRAME_TIMES: List[float] = [CONFIG["frame_times_s"][CONFIG["frame_order"].index(f)]
                             for f in FRAME_ORDER if f in CONFIG["frame_order"]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_state_summary(state_json: dict) -> str:
    """Human-readable scene summary for logging and debug."""
    ego = state_json.get("ego", {})
    agents = state_json.get("agents", {})
    nav_cmd = state_json.get("nav_command", "UNKNOWN")

    agent_lines = [
        f"  {name} ({info.get('category', '?')}): "
        f"x={info.get('position', {}).get('x', 0):.1f}m "
        f"y={info.get('position', {}).get('y', 0):.1f}m "
        f"v={info.get('speed', 0):.1f}m/s"
        for name, info in agents.items()
    ]
    return (
        f"Nav: {nav_cmd} | Ego speed: {ego.get('speed', 0.0):.2f} m/s\n"
        + ("\n".join(agent_lines) if agent_lines else "  No nearby agents")
    )


def _ego_history_text(state_json: dict) -> str:
    """Short ego trajectory string to embed in prompts."""
    ego = state_json.get("ego", {})
    history = ego.get("history", [])
    pts = [(float(e.get("t", 0)), e.get("x", 0), e.get("y", 0)) for e in history]
    pts.append((0.0, 0.0, 0.0))  # current origin
    coords = ", ".join(f"t={t:.1f}s→({x:.1f},{y:.1f})m" for t, x, y in pts)
    return f"Ego trajectory (ego-centric, x=forward, y=left): [{coords}]"


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class OmniDriveDataLoader(ModelDataLoader):
    """
    DataLoader for OmniDrive on the causal driving benchmark.

    Provides 4 cameras × 4 timesteps = 16 images in TIME-MAJOR order,
    plus a structured state summary and QA prompt payload.
    """

    def __init__(self, dataset: BenchmarkDataset, model_cfg: dict) -> None:
        super().__init__(dataset, model_cfg)
        print(
            f"[OmniDriveDataLoader] Initialized.\n"
            f"  CAMERA_ORDER = {CAMERA_ORDER}\n"
            f"  FRAME_ORDER  = {FRAME_ORDER}\n"
            f"  Image ordering: TIME-MAJOR (all cameras at each timestep)"
        )

    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        """
        Build ordered list of image path metadata dicts in TIME-MAJOR order.

        Returns a list of dicts with keys: path, time_key, camera_key.
        Order: all cameras at Tm1p5, then all at Tm1p0, Tm0p5, Tp0p0.
        """
        frames = record.get("frames", {})
        raw_data_dir = self.model_cfg.get("raw_data_dir", "")
        data_root = Path(raw_data_dir).parent if raw_data_dir else Path(".")

        image_paths = []
        for frame_key in FRAME_ORDER:
            for cam in CAMERA_ORDER:
                path = frames.get(frame_key, {}).get(cam)
                if path:
                    abs_path = str(data_root / path)
                    image_paths.append({"path": abs_path, "time_key": frame_key, "camera_key": cam})
        return image_paths

    def format_sample(self, record: dict) -> dict:
        """
        Convert a BenchmarkDataset record into OmniDrive model inputs.

        Image loading happens later in run_single() using the absolute paths
        stored in prompts.jsonl (built by build_image_paths).  Here we only
        extract lightweight scene metadata used by get_prompt().

        Returns:
          scene_id     – scene identifier
          sample_id    – sample identifier
          state_summary– human-readable scene description (for logging/debug)
          ego_text     – short trajectory string embedded in prompt context
        """
        scene_id = record["scene_id"]
        sample_id = record["sample_id"]
        state_json = record.get("state") or {}

        print(f"[format_sample] {scene_id}/{sample_id} | nav={state_json.get('nav_command', '?')}")

        return {
            "scene_id": scene_id,
            "sample_id": sample_id,
            "state_summary": _build_state_summary(state_json),
            "ego_text": _ego_history_text(state_json),
        }

    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        """
        Build a single QA prompt payload for one causal question.

        Returns only model-specific fields (framework fields injected by base class).
        """
        answer_format = question.get("answer_format", "binary")
        ego_text = formatted_sample.get("ego_text", "")
        preamble = question.get("preamble", "")

        if answer_format == "binary":
            format_instruction = (
                "Answer with 'Yes' or 'No', then explain your reasoning.\n"
                "Format:\nAnswer: Yes\nReasoning: <explanation>"
            )
        else:
            format_instruction = (
                "Answer with the letter of the correct option (A, B, C, or D).\n"
                "Format:\nAnswer: <letter>\nReasoning: <explanation>"
            )

        context = f"{ego_text}\n\n{preamble}" if preamble else ego_text
        prompt_context = f"{context}\n\n{format_instruction}".strip()

        return {
            "prompt_context": prompt_context,
            "bench_dir": self.model_cfg.get("bench_dir", ""),
        }
