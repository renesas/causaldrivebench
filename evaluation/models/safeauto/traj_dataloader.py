"""
SafeAuto trajectory prompt builder — BTE and CTE.

Produces three prompt types per sample:

  BTE (Behavioral Trajectory Evaluation)
    Predict the factual ego future from T=0 (6 waypoints, t=0.5–3.0 s).
    Ground truth: state.json ego.future.

  CTE_R2 (Counterfactual Trajectory Evaluation, Rung 2)
    For questions whose meta.alternate_trajectory.rung == 2 (present-time
    intervention, frame=T=0).  Predict 8 waypoints (t=0.5–4.0 s) in the T=0
    ego frame given the counterfactual scenario described in the question.
    Ground truth: alternate_trajectory.waypoints.

  CTE_R3 (Counterfactual Trajectory Evaluation, Rung 3)
    For questions whose meta.alternate_trajectory.rung == 3 (counterfactual
    past, frame=T=-1.5).  Predict 3 waypoints (t=0.5, 1.0, 1.5 s) relative to
    the T=-1.5 anchor in the T=-1.5 ego frame.
    Ground truth: alternate_trajectory.waypoints[1:] (skipping the t=0 anchor).

Image paths are taken from frames.json (same format as the QA dataloader) so
that SafeAutoInference.run_single() can resolve them via cfg.resolve_frame_path().
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_CAMERAS = ["cam_front", "cam_front_left", "cam_front_right", "cam_back"]
_CAM_LABELS: Dict[str, str] = {
    "cam_front":       "CAM_FRONT",
    "cam_front_left":  "CAM_FRONT_LEFT",
    "cam_front_right": "CAM_FRONT_RIGHT",
    "cam_back":        "CAM_BACK",
}

_BTE_TIMESTEPS   = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
_CTE_R2_TIMESTEPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
_CTE_R3_TIMESTEPS = [0.5, 1.0, 1.5]  # relative to T=-1.5 anchor

_QA_KINDS = ("active", "dormant", "distractor")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def _history_block(state: dict) -> str:
    ego = state.get("ego", {})
    history = ego.get("history", [])
    current_speed = float(ego.get("speed", 0.0))
    lines = [
        f"  t={float(pt['t']):.1f}s: ({float(pt['x']):.2f}, {float(pt['y']):.2f}) m,"
        f"  speed={float(pt['speed']):.2f} m/s"
        for pt in history
    ]
    lines.append(f"  t=0.0s: (0.00, 0.00) m [current],  speed={current_speed:.2f} m/s")
    return "\n".join(lines)


def _image_block(image_paths: List[Dict]) -> str:
    loaded_cams = [ip["camera_key"] for ip in image_paths]
    return ", ".join(
        f"{_CAM_LABELS[cam]}: <image>"
        for cam in _CAMERAS
        if cam in loaded_cams
    )


# ---------------------------------------------------------------------------
# Image path collection (from frames.json, compatible with run_single)
# ---------------------------------------------------------------------------

def _get_t0_image_paths(sample_dir: Path) -> List[Dict[str, str]]:
    """Load T=0 image paths from frames.json in the same format as the QA dataloader."""
    frames_data = _load_json(sample_dir / "frames.json")
    if frames_data is None:
        return []
    t0_frames = frames_data.get("frames", {}).get("Tp0p0", {})
    paths = []
    for cam in _CAMERAS:
        rel = t0_frames.get(cam)
        if rel:
            paths.append({"path": rel, "time_key": "Tp0p0", "camera_key": cam})
        else:
            logger.warning("[%s] No path for Tp0p0/%s", sample_dir.name, cam)
    return paths


# ---------------------------------------------------------------------------
# BTE prompt
# ---------------------------------------------------------------------------

def _build_bte_prompt(
    scene_id: str,
    sample_id: str,
    state: dict,
    image_paths: List[Dict],
) -> dict:
    nav_command = state.get("nav_command", "UNKNOWN")
    history_str = _history_block(state)
    img_block   = _image_block(image_paths)
    times_str   = ", ".join(f"t={t:.1f}" for t in _BTE_TIMESTEPS)

    conversation_text = (
        f"Following multi-view images record the driving scenario: {img_block}\n\n"
        f"Navigation command: {nav_command}\n\n"
        f"Ego vehicle past trajectory (ego-centric, +x=forward, +y=left):\n"
        f"{history_str}\n\n"
        f"Predict the ego vehicle's future trajectory for the next 3 seconds.\n"
        f"Output ONLY a JSON list with 6 waypoints at {times_str} seconds, like:\n"
        f'[{{"t": 0.5, "x": <float>, "y": <float>}}, '
        f'{{"t": 1.0, "x": <float>, "y": <float>}}, '
        f'{{"t": 1.5, "x": <float>, "y": <float>}}, '
        f'{{"t": 2.0, "x": <float>, "y": <float>}}, '
        f'{{"t": 2.5, "x": <float>, "y": <float>}}, '
        f'{{"t": 3.0, "x": <float>, "y": <float>}}]'
    )

    ego_future = state.get("ego", {}).get("future", [])
    gt_waypoints = [
        {"t": float(pt["t"]), "x": float(pt["x"]), "y": float(pt["y"])}
        for pt in ego_future
        if float(pt["t"]) in _BTE_TIMESTEPS
    ]

    return {
        "type":              "BTE",
        "scene_id":          scene_id,
        "sample_id":         sample_id,
        "question_id":       None,
        "image_paths":       image_paths,
        "conversation_text": conversation_text,
        "gt_waypoints":      gt_waypoints,
    }


# ---------------------------------------------------------------------------
# CTE prompts
# ---------------------------------------------------------------------------

def _build_cte_r2_prompt(
    scene_id: str,
    sample_id: str,
    state: dict,
    image_paths: List[Dict],
    question: dict,
    alt: dict,
) -> dict:
    nav_command  = state.get("nav_command", "UNKNOWN")
    history_str  = _history_block(state)
    img_block    = _image_block(image_paths)
    question_text = question.get("question", "")
    times_str    = ", ".join(f"t={t:.1f}" for t in _CTE_R2_TIMESTEPS)

    conversation_text = (
        f"Following multi-view images record the driving scenario: {img_block}\n\n"
        f"Navigation command: {nav_command}\n\n"
        f"Ego vehicle past trajectory (ego-centric at T=0, +x=forward, +y=left):\n"
        f"{history_str}\n\n"
        f"Counterfactual scenario: {question_text}\n\n"
        f"Given this counterfactual scenario, predict the ego vehicle's future "
        f"trajectory for the next 4 seconds.\n"
        f"Output ONLY a JSON list with 8 waypoints at {times_str} seconds "
        f"(ego-centric at T=0, +x=forward, +y=left), like:\n"
        f'[{{"t": 0.5, "x": <float>, "y": <float>}}, '
        f'{{"t": 1.0, "x": <float>, "y": <float>}}, ...]'
    )

    gt_waypoints = [
        {"t": float(wp["t"]), "x": float(wp["x"]), "y": float(wp["y"])}
        for wp in alt.get("waypoints", [])
    ]

    return {
        "type":              "CTE_R2",
        "scene_id":          scene_id,
        "sample_id":         sample_id,
        "question_id":       question.get("id"),
        "image_paths":       image_paths,
        "conversation_text": conversation_text,
        "gt_waypoints":      gt_waypoints,
    }


def _build_cte_r3_prompt(
    scene_id: str,
    sample_id: str,
    state: dict,
    image_paths: List[Dict],
    question: dict,
    alt: dict,
) -> dict:
    nav_command   = state.get("nav_command", "UNKNOWN")
    img_block     = _image_block(image_paths)
    question_text = question.get("question", "")
    times_str     = ", ".join(f"t={t:.1f}" for t in _CTE_R3_TIMESTEPS)

    # Anchor is at T=-1.5 in real time (1.5 s before the scene's T=0).
    # The ego position at T=-1.5 is available in state.json history.
    ego_history = state.get("ego", {}).get("history", [])
    anchor_entry = next((h for h in ego_history if float(h["t"]) == -1.5), None)
    anchor_speed = f"{float(anchor_entry['speed']):.2f}" if anchor_entry else "?"

    conversation_text = (
        f"Following multi-view images record the driving scenario: {img_block}\n\n"
        f"Navigation command: {nav_command}\n\n"
        f"Counterfactual scenario (counterfactual history starting 1.5 s ago): "
        f"{question_text}\n\n"
        f"In this alternate history, the ego starts at the position it occupied "
        f"1.5 seconds ago (T=-1.5s, speed={anchor_speed} m/s).\n"
        f"Predict the trajectory for the next 1.5 seconds from that anchor, "
        f"in the T=-1.5s ego frame (+x=forward, +y=left, anchor=(0,0)).\n"
        f"Output ONLY a JSON list with 3 waypoints at {times_str} seconds "
        f"(relative to the T=-1.5s anchor), like:\n"
        f'[{{"t": 0.5, "x": <float>, "y": <float>}}, '
        f'{{"t": 1.0, "x": <float>, "y": <float>}}, '
        f'{{"t": 1.5, "x": <float>, "y": <float>}}]'
    )

    # waypoints[0] is the anchor (t=0, x=0, y=0); evaluate on waypoints[1:]
    all_wps = alt.get("waypoints", [])
    gt_waypoints = [
        {"t": float(wp["t"]), "x": float(wp["x"]), "y": float(wp["y"])}
        for wp in all_wps
        if float(wp["t"]) > 0.0
    ]

    return {
        "type":              "CTE_R3",
        "scene_id":          scene_id,
        "sample_id":         sample_id,
        "question_id":       question.get("id"),
        "image_paths":       image_paths,
        "conversation_text": conversation_text,
        "gt_waypoints":      gt_waypoints,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_traj_prompts(
    bench_dir: Path,
    scene_id: str,
    sample_id: str,
) -> List[dict]:
    """
    Build all trajectory prompts (BTE + CTE_R2 + CTE_R3) for one sample.

    Returns an empty list if required files are missing.
    """
    sample_dir = bench_dir / scene_id / sample_id

    image_paths = _get_t0_image_paths(sample_dir)
    if not image_paths:
        logger.error("[%s/%s] No T=0 images found — skipping traj prompts", scene_id, sample_id)
        return []

    state = _load_json(sample_dir / "state.json")
    if state is None:
        logger.warning("[%s/%s] state.json missing — skipping traj prompts", scene_id, sample_id)
        return []

    prompts: List[dict] = []

    # BTE — one per sample
    prompts.append(_build_bte_prompt(scene_id, sample_id, state, image_paths))

    # CTE — one per question that has an alternate_trajectory
    for qa_kind in _QA_KINDS:
        qa_path = sample_dir / "qa" / f"{qa_kind}_qa.json"
        if not qa_path.exists():
            continue
        qa_data = _load_json(qa_path)
        if not qa_data:
            continue
        for q in qa_data.get("questions", []):
            alt = q.get("meta", {}).get("alternate_trajectory")
            if not alt:
                continue
            rung = int(alt.get("rung", 0))
            if rung == 2:
                prompts.append(_build_cte_r2_prompt(scene_id, sample_id, state, image_paths, q, alt))
            elif rung == 3:
                prompts.append(_build_cte_r3_prompt(scene_id, sample_id, state, image_paths, q, alt))
            else:
                logger.warning("[%s/%s] Unknown alt traj rung=%s for q=%s", scene_id, sample_id, rung, q.get("id"))

    return prompts


def iter_traj_prompts(
    bench_dir: Path,
    sample_ids: List[dict],
) -> Iterator[dict]:
    """Yield trajectory prompts for a list of sample_id dicts."""
    for sid in sample_ids:
        for prompt in build_traj_prompts(bench_dir, sid["scene_id"], sid["sample_id"]):
            yield prompt
