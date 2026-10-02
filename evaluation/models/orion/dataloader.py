"""
Orion DataLoader for the causal scene understanding benchmark.

Adapts NuScenes multi-camera frames and state.json to Orion's expected input
format. Orion (xiaomi-mlab/Orion) uses EVA-ViT + QFormer + LLaMA operating
on 4 cameras × 4 timesteps = 16 images in TIME-MAJOR order.

IMAGE ORDERING — TIME-MAJOR:
  Orion's temporal transformer processes images grouped by timestep:
    [cam_front@Tm1p5, cam_fl@Tm1p5, cam_fr@Tm1p5, cam_back@Tm1p5,
     cam_front@Tm1p0, ..., cam_back@Tp0p0]   (16 images total)

TRAJECTORY FIELDS (added per question):
  ego_future          — state.json ego.future (6 GT waypoints at t=0.5..3.0s,
                        cumulative x,y in T=0 ego frame). Used for BTE eval.
  ego_history         — state.json ego.history (t,x,y per past frame). Used
                        for R3 counterfactual frame transform in postprocess.
  trajectory_changes  — bool from question.trajectory_changes; signals that
                        a counterfactual intervention changes ego behavior.
  alternate_trajectory— question.meta.alternate_trajectory (GT for CF eval).
  counterfactual_premise — question intervention premise text for CF prompt.

GPU MEMORY:
  FP16 inference: 17 GB+  (use orion_stage3_fp16.py config)
  FP32 inference: 32 GB+
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from PIL import Image

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
    "image_ordering": "time_major",
    "image_size_wh": (1600, 900),
}

FRAME_ORDER: List[str] = CONFIG["frame_order"]
CAMERA_ORDER: List[str] = CONFIG["camera_order"]
FRAME_TIMES: List[float] = CONFIG["frame_times_s"]
_PLACEHOLDER_SIZE: tuple = CONFIG["image_size_wh"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_frame_path(rel_path: str, cfg: Optional[CDBConfig]) -> Optional[str]:
    """
    Resolve a relative image path to an absolute path.

    Handles two formats stored in frames.json:
      1. Standard: "raw_data/nuscenes/samples/..." → cfg.resolve_frame_path()
      2. Short:    "samples/CAM_FRONT/..."         → cfg.dir_raw_data/nuscenes/<path>
    Returns None if the file cannot be found.
    """
    if not rel_path:
        return None
    if cfg is not None:
        p = cfg.resolve_frame_path(rel_path)
        if p.exists():
            return str(p)
        # Short-path fallback: prepend nuscenes subdir
        p2 = cfg.dir_raw_data / "nuscenes" / rel_path
        if p2.exists():
            return str(p2)
    # Try as-is (absolute or relative to CWD)
    p3 = Path(rel_path)
    if p3.exists():
        return str(p3)
    return None


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


def _extract_counterfactual_premise(question: dict) -> str:
    """Return the counterfactual intervention premise text if present."""
    for field in ("intervention_premise", "counterfactual_premise", "preamble"):
        val = question.get(field, "")
        if val:
            return str(val)
    return ""


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class OrionDataLoader(ModelDataLoader):
    """
    DataLoader for Orion on the causal driving benchmark.

    Provides 4 cameras × 4 timesteps = 16 images in TIME-MAJOR order plus
    trajectory ground-truth and counterfactual fields for trajectory evaluation.
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
            f"[OrionDataLoader] Initialized.\n"
            f"  CAMERA_ORDER = {CAMERA_ORDER}\n"
            f"  FRAME_ORDER  = {FRAME_ORDER}\n"
            f"  Image ordering: TIME-MAJOR ({len(FRAME_ORDER) * len(CAMERA_ORDER)} images)"
        )

    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        """
        Build ordered list of image path metadata dicts in TIME-MAJOR order.

        Returns a list of dicts with keys: path, time_key, camera_key.
        Order: all cameras at Tm1p5, then all at Tm1p0, Tm0p5, Tp0p0.
        """
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
        Load 4×4 multi-view temporal image grid in TIME-MAJOR order.

        Also extracts ego_future (BTE GT) and ego_history (for R3 CF
        coordinate transforms) from state.json.
        """
        scene_id  = record["scene_id"]
        sample_id = record["sample_id"]
        frames    = record["frames"]
        state_json = record.get("state") or {}

        print(f"[format_sample] {scene_id}/{sample_id} | nav={state_json.get('nav_command', '?')}")

        images: List[Image.Image] = []
        missing = 0

        for frame_key in FRAME_ORDER:
            for cam in CAMERA_ORDER:
                path = frames.get(frame_key, {}).get(cam)
                abs_path = _resolve_frame_path(path, self.cfg) if path else None
                img = _load_image(abs_path, _PLACEHOLDER_SIZE)
                if not (abs_path and Path(abs_path).exists()):
                    missing += 1
                    if path:
                        print(f"[format_sample]   WARN: missing {abs_path}")
                images.append(img)

        n_total = len(FRAME_ORDER) * len(CAMERA_ORDER)
        print(f"[format_sample] {n_total - missing}/{n_total} images ({missing} placeholder(s))")

        # Ego future trajectory (ground truth for BTE evaluation)
        ego = state_json.get("ego", {})
        ego_future  = ego.get("future", [])   # [{t, x, y, speed}, …] 6 pts at t=0.5..3.0
        ego_history = ego.get("history", [])  # [{t, x, y, speed}, …] past frames

        meta = record.get("meta", {})
        return {
            "scene_id":       scene_id,
            "sample_id":      sample_id,
            "images":         images,
            "state_summary":  _build_state_summary(state_json),
            "ego_text":       _ego_history_text(state_json),
            "ego_future":     ego_future,
            "ego_history":    ego_history,
            "nuscenes_scene": meta.get("nuscenes_scene", ""),
            "timestamp":      meta.get("timestamp", 0),
            "calib":          record.get("calib") or {},
        }

    def get_trajectory_prompt(
        self,
        formatted_sample: dict,
        traj_type: str,
        context_question: Optional[dict] = None,
        gt_waypoints: Optional[list] = None,
    ) -> dict:
        """
        Build a trajectory prompt payload for Orion (BTE or CTE).

        Orion uses its own fixed planning prompt with the <waypoint_ego>
        special token, so we don't emit a textual qa_text here. Instead we
        carry the inputs the visual encoder + planning decoder need at
        inference time.

        For CTE we capture the counterfactual premise text so inference can
        plug it into Orion's CF planning template.
        """
        cf_premise = ""
        if traj_type != "BTE" and context_question is not None:
            cf_premise = _extract_counterfactual_premise(context_question)

        return {
            "nuscenes_scene":         formatted_sample.get("nuscenes_scene", ""),
            "timestamp":              formatted_sample.get("timestamp", 0),
            "calib":                  formatted_sample.get("calib", {}),
            "counterfactual_premise": cf_premise,
        }

    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        """
        Build a QA prompt payload for one causal question.

        Framework fields (scene_id, sample_id, question_id, image_paths,
        question_text, qa_text, answer_format) are injected by the base class.

        Model-specific fields returned here:
          prompt_context        — ego state + format instruction preamble
          nuscenes_scene        — for CAN-bus loading at inference time
          timestamp             — scene timestamp (µs)
          calib                 — calibration matrices
          ego_future            — GT future trajectory for BTE evaluation
          ego_history           — past ego poses for R3 CF frame transform
          trajectory_changes    — bool: does this CF question change ego behavior?
          alternate_trajectory  — GT counterfactual trajectory (or None)
          counterfactual_premise— text for the counterfactual inference prompt
        """
        answer_format = question.get("answer_format", "binary")
        preamble      = question.get("preamble", "")
        ego_text      = formatted_sample.get("ego_text", "")

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

        # Trajectory-specific fields
        trajectory_changes   = bool(question.get("trajectory_changes", False))
        alternate_trajectory = (
            question.get("meta", {}).get("alternate_trajectory")
            if trajectory_changes else None
        )
        cf_premise = _extract_counterfactual_premise(question) if trajectory_changes else ""

        return {
            "prompt_context":        prompt_context,
            "nuscenes_scene":        formatted_sample.get("nuscenes_scene", ""),
            "timestamp":             formatted_sample.get("timestamp", 0),
            "calib":                 formatted_sample.get("calib", {}),
            # Trajectory fields
            "ego_future":            formatted_sample.get("ego_future", []),
            "ego_history":           formatted_sample.get("ego_history", []),
            "trajectory_changes":    trajectory_changes,
            "alternate_trajectory":  alternate_trajectory,
            "counterfactual_premise": cf_premise,
        }
