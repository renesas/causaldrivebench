"""
SimLingo DataLoader for the causal scene understanding benchmark.

Adapts NuScenes multi-camera frames and state.json to SimLingo's expected
input format. SimLingo (RenzKa/simlingo) uses InternVL2 as its VLM backbone,
trained primarily on front-camera images from CARLA simulations.

IMAGE ORDERING — FRONT-CAMERA TEMPORAL SEQUENCE:
  SimLingo is trained on a front-facing camera in CARLA. For our NuScenes
  benchmark, we provide the front camera across all 4 timesteps to give
  temporal context, plus the current-frame side cameras for spatial awareness.

    [cam_front@Tm1p5, cam_front@Tm1p0, cam_front@Tm0p5, cam_front@Tp0p0]
    (4 images — front camera only)

  We use front-camera-only to best match SimLingo's training distribution.
  Side cameras can be added via CONFIG["include_side_cameras"] if needed.

CARLA → NUSCENES MAPPING:
  CARLA "front RGB" → cam_front (1600×900)
  SimLingo expects InternVL2-processed images at 448×448 per the processor.
  Resizing is handled by the InternVL2 processor in inference.py.
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
    # Front camera across all timesteps (primary temporal sequence)
    "frame_order": ["Tm1p5", "Tm1p0", "Tm0p5", "Tp0p0"],
    # Only front camera matches SimLingo's CARLA training distribution.
    # Set include_side_cameras=True to add spatial context cameras at t=0.
    "camera_order": ["cam_front"],
    "include_side_cameras": False,
    "side_cameras": ["cam_front_left", "cam_front_right"],
    "frame_times_s": [-1.5, -1.0, -0.5, 0.0],
    "image_size_wh": (1600, 900),
}

FRAME_ORDER: List[str] = CONFIG["frame_order"]
CAMERA_ORDER: List[str] = CONFIG["camera_order"]
_INCLUDE_SIDE: bool = CONFIG["include_side_cameras"]
_SIDE_CAMS: List[str] = CONFIG["side_cameras"]
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

class SimLingoDataLoader(ModelDataLoader):
    """
    DataLoader for SimLingo (InternVL2-based) on the causal driving benchmark.

    Provides front-camera images across 4 timesteps in TIME-MAJOR order.
    Matches SimLingo's CARLA training distribution (front-camera primary).
    """

    def __init__(self, dataset: BenchmarkDataset, model_cfg: dict) -> None:
        super().__init__(dataset, model_cfg)
        # ns_root: prepended when resolving frames.json paths (strips leading "samples/")
        self._ns_root: Optional[str] = model_cfg.get("ns_root")
        n_images = len(FRAME_ORDER) * len(CAMERA_ORDER) + (len(_SIDE_CAMS) if _INCLUDE_SIDE else 0)
        print(
            f"[SimLingoDataLoader] Initialized.\n"
            f"  Camera: {CAMERA_ORDER} × {len(FRAME_ORDER)} frames\n"
            f"  Side cameras at t=0: {_SIDE_CAMS if _INCLUDE_SIDE else 'disabled'}\n"
            f"  Total images per sample: {n_images}\n"
            f"  ns_root: {self._ns_root or '(not set — paths used as-is)'}"
        )

    def _resolve(self, path: Optional[str]) -> str:
        """Resolve frames.json path against ns_root (paths are relative to ns_root)."""
        if not path:
            return ""
        if self._ns_root:
            return str(Path(self._ns_root) / path)
        return path

    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        """
        Return ordered list of image path metadata dicts.

        Front camera temporal sequence, then optional side cameras at t=0.
        """
        frames = record["frames"]
        paths: List[Dict[str, str]] = []

        # Front camera temporal sequence
        for frame_key in FRAME_ORDER:
            for cam in CAMERA_ORDER:
                raw = frames.get(frame_key, {}).get(cam)
                paths.append({
                    "path": self._resolve(raw),
                    "time_key": frame_key,
                    "camera_key": cam,
                })

        # Optional side cameras at current frame
        if _INCLUDE_SIDE:
            for cam in _SIDE_CAMS:
                raw = frames.get("Tp0p0", {}).get(cam)
                paths.append({
                    "path": self._resolve(raw),
                    "time_key": "Tp0p0",
                    "camera_key": cam,
                })

        return paths

    def format_sample(self, record: dict) -> dict:
        """
        Load front-camera temporal sequence + optional side cameras.

        Primary: cam_front × 4 timesteps (time-major: oldest first)
        Optional: cam_front_left, cam_front_right at Tp0p0
        """
        scene_id = record["scene_id"]
        sample_id = record["sample_id"]
        frames = record["frames"]
        state_json = record.get("state") or {}

        images: List[Image.Image] = []
        missing = 0

        # Front camera temporal sequence
        for frame_key in FRAME_ORDER:
            for cam in CAMERA_ORDER:
                raw  = frames.get(frame_key, {}).get(cam)
                path = self._resolve(raw)
                img = _load_image(path, _PLACEHOLDER_SIZE)
                if not (path and Path(path).exists()):
                    missing += 1
                    print(f"[format_sample] MISSING {frame_key}/{cam}: raw={raw!r} resolved={path!r} ns_root={self._ns_root!r}")
                images.append(img)

        # Optional side cameras at current frame
        if _INCLUDE_SIDE:
            for cam in _SIDE_CAMS:
                path = self._resolve(frames.get("Tp0p0", {}).get(cam))
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
            "speed": float(state_json.get("ego", {}).get("speed", 0.0)),
            "nav_command": state_json.get("nav_command", "GO STRAIGHT"),
            "state_json": state_json,
        }

    def get_trajectory_prompt(
        self,
        formatted_sample: dict,
        eval_type: str,
        context_question: Optional[dict] = None,
        gt_waypoints: Optional[list] = None,
    ) -> dict:
        """
        Return model-specific payload for trajectory prompts (BTE and CTE).

        Uses the standardised prompt builders so both BTE and CTE prompts are
        question-dependent. For CTE the counterfactual question text is included
        in the prompt, conditioning the LM features that the driving head
        operates on. For BTE the prompt contains ego state and the waypoint
        output template.
        """
        from evaluation.common.trajectory_eval import (
            BTE_TIMESTEPS,
            build_bte_prompt_text,
            build_cte_prompt_text,
        )

        state_json  = formatted_sample.get("state_json", {})
        gt_timesteps = [w["t"] for w in gt_waypoints] if gt_waypoints else BTE_TIMESTEPS

        if eval_type == "BTE":
            qa_text = build_bte_prompt_text(state_json, gt_timesteps)
        else:
            qa_text = build_cte_prompt_text(
                context_question or {}, gt_timesteps, state_json
            )

        return {"qa_text": qa_text}

    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        """Build model-specific extras for one causal question."""
        answer_format = question.get("answer_format", "binary")
        preamble = question.get("preamble", "")
        ego_text = formatted_sample.get("ego_text", "")

        # InternVL2 / SimLingo responds well to concise prompts
        if answer_format == "binary":
            format_instruction = (
                "Answer 'Yes' or 'No' and briefly explain.\n"
                "Format:\nAnswer: Yes\nReasoning: <explanation>"
            )
        else:
            format_instruction = (
                "Select A, B, C, or D and explain your reasoning.\n"
                "Format:\nAnswer: <letter>\nReasoning: <explanation>"
            )

        context = f"{ego_text}\n\n{preamble}" if preamble else ego_text
        prompt_context = f"{context}\n\n{format_instruction}".strip()

        return {
            "prompt_context": prompt_context,
        }
