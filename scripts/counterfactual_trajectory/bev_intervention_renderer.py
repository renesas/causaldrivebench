#!/usr/bin/env python3
"""
BEV Intervention Renderer + Ego Trajectory Generator

Stage 4: Modify BEV image to show agent intervention (remove/reposition/speed_change)
Stage 5: Feed modified BEV + cameras to LLM for ego counterfactual trajectory

Usage:
  # Single scene
  python bev_intervention_renderer.py \
    --records /path/to/records/nuscenes_50 \
    --outputs /path/to/outputs/nuscenes_50 \
    --scene nuscenes-scene-0006_SAMPLED_3 \
    --provider claude --model claude-opus-4-6

  # All scenes in outputs dir
  python bev_intervention_renderer.py \
    --records /path/to/records/nuscenes_50 \
    --outputs /path/to/outputs/nuscenes_50 \
    --provider claude --model claude-opus-4-6
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ── BEV Constants (must match record builders) ───────────────────────────
BEV_SIZE = 1536
BEV_RANGE = 50.0  # metres each side of ego
BEV_SCALE = BEV_SIZE / (2 * BEV_RANGE)  # px/m
BEV_CENTER = BEV_SIZE // 2

# Agent box colors (BGR) — match record builders
COLOR_VEHICLE = (180, 200, 255)  # peach
COLOR_PED = (0, 255, 128)       # green
COLOR_CYCLE = (0, 200, 200)     # yellow-ish
COLOR_EGO = (255, 200, 0)       # cyan

# Intervention overlay colors (BGR)
COLOR_INTERVENTION = (0, 0, 255)       # red — for repositioned/changed agents
COLOR_TRAJ_DOTS = (255, 0, 255)        # magenta — for speed_change trajectory
COLOR_REMOVED = (80, 80, 80)           # dark grey — road fill for removed agents

# Default agent sizes (metres) — used when actual size unknown
DEFAULT_VEHICLE_SIZE = (4.5, 2.0)  # length, width
DEFAULT_PED_SIZE = (0.8, 0.8)
DEFAULT_CYCLE_SIZE = (2.0, 0.8)


def ego_to_bev(x, y):
    """Convert ego-local (x=fwd, y=left) to BEV pixel (col, row)."""
    px = int(BEV_CENTER - y * BEV_SCALE)
    py = int(BEV_CENTER - x * BEV_SCALE)
    return px, py


def get_agent_size(category):
    """Get (length, width) in metres based on category string."""
    cat = category.lower()
    if "pedestrian" in cat or "human" in cat:
        return DEFAULT_PED_SIZE
    elif "bicycle" in cat or "motorcycle" in cat or "cycle" in cat:
        return DEFAULT_CYCLE_SIZE
    return DEFAULT_VEHICLE_SIZE


def get_agent_color(category):
    """Get BGR color based on category string."""
    cat = category.lower()
    if "pedestrian" in cat or "human" in cat:
        return COLOR_PED
    elif "bicycle" in cat or "motorcycle" in cat or "cycle" in cat:
        return COLOR_CYCLE
    return COLOR_VEHICLE


def draw_rotated_box(img, cx, cy, length, width, yaw, color, thickness=-1, border_color=None, border_thickness=2):
    """Draw a rotated rectangle on BEV image.

    cx, cy: center in pixel coords
    length, width: in pixels
    yaw: rotation in radians (ego-local: 0 = forward = up on BEV)
    """
    # BEV: +x is up, so yaw=0 means pointing up
    # cv2 rotatedRect angle is in degrees, clockwise from horizontal
    # Convert: ego yaw → cv2 angle
    angle_deg = -math.degrees(yaw) + 90  # ego forward (yaw=0) → 90° in cv2 (pointing up)
    rect = ((cx, cy), (width, length), angle_deg)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if thickness == -1:
        cv2.fillPoly(img, [box], color)
    else:
        cv2.drawContours(img, [box], 0, color, thickness)
    if border_color:
        cv2.drawContours(img, [box], 0, border_color, border_thickness)


def remove_agent(bev_img, agent_x, agent_y, agent_yaw, category):
    """Remove an agent from BEV using inpainting."""
    length_m, width_m = get_agent_size(category)
    cx, cy = ego_to_bev(agent_x, agent_y)

    # Create mask slightly larger than agent
    mask = np.zeros(bev_img.shape[:2], dtype=np.uint8)
    pad = 4  # pixels padding
    length_px = length_m * BEV_SCALE + pad * 2
    width_px = width_m * BEV_SCALE + pad * 2

    angle_deg = -math.degrees(agent_yaw) + 90
    rect = ((cx, cy), (width_px, length_px), angle_deg)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.fillPoly(mask, [box], 255)

    # Inpaint
    result = cv2.inpaint(bev_img, mask, 10, cv2.INPAINT_TELEA)
    return result


def reposition_agent(bev_img, orig_x, orig_y, orig_yaw, new_x, new_y, new_yaw, category, label=""):
    """Remove agent from original position, draw at new position with intervention color."""
    # Step 1: Remove from original position
    bev_img = remove_agent(bev_img, orig_x, orig_y, orig_yaw, category)

    # Step 2: Draw at new position
    length_m, width_m = get_agent_size(category)
    cx, cy = ego_to_bev(new_x, new_y)
    length_px = length_m * BEV_SCALE
    width_px = width_m * BEV_SCALE

    draw_rotated_box(bev_img, cx, cy, length_px, width_px, new_yaw,
                     COLOR_INTERVENTION, thickness=-1,
                     border_color=(255, 255, 255), border_thickness=2)

    # Add label
    if label:
        cv2.putText(bev_img, label, (cx + int(width_px/2) + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return bev_img


def speed_change_agent(bev_img, agent_x, agent_y, agent_yaw, trajectory, category, label=""):
    """Draw trajectory dots showing where agent will be at future timesteps."""
    length_m, width_m = get_agent_size(category)

    # Draw trajectory dots
    pts = []
    for wp in trajectory:
        if isinstance(wp, dict):
            wx, wy = wp.get("x", 0), wp.get("y", 0)
        elif isinstance(wp, list) and len(wp) >= 2:
            wx, wy = wp[0], wp[1]
        else:
            continue
        px, py = ego_to_bev(wx, wy)
        pts.append((px, py))
        cv2.circle(bev_img, (px, py), 6, COLOR_TRAJ_DOTS, -1)

    # Connect dots
    for i in range(len(pts) - 1):
        cv2.line(bev_img, pts[i], pts[i + 1], COLOR_TRAJ_DOTS, 2, cv2.LINE_AA)

    # Draw agent box at final position with intervention color
    if trajectory:
        last = trajectory[-1]
        if isinstance(last, dict):
            fx, fy = last.get("x", agent_x), last.get("y", agent_y)
            fyaw = last.get("yaw", agent_yaw)
        else:
            fx, fy, fyaw = last[0], last[1], agent_yaw
        cx, cy = ego_to_bev(fx, fy)
        length_px = length_m * BEV_SCALE
        width_px = width_m * BEV_SCALE
        draw_rotated_box(bev_img, cx, cy, length_px, width_px, fyaw,
                         COLOR_INTERVENTION, thickness=-1,
                         border_color=(255, 255, 255), border_thickness=2)

    if label:
        px0, py0 = ego_to_bev(agent_x, agent_y)
        cv2.putText(bev_img, label, (px0 + 10, py0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TRAJ_DOTS, 1)

    return bev_img


def render_intervention_bev(scene_dir: Path, state_json: dict, intervention_config: dict,
                            temporal_anchor: str = "t0") -> np.ndarray:
    """Render the modified BEV with the intervention applied.

    Returns the modified BEV image (numpy array).
    """
    # Pick the right BEV based on temporal anchor
    if temporal_anchor == "observation_window":
        bev_path = scene_dir / "Tm1p5" / "bev.png"
    else:
        bev_path = scene_dir / "Tp0p0" / "bev.png"

    bev = cv2.imread(str(bev_path))
    if bev is None:
        raise FileNotFoundError(f"BEV not found: {bev_path}")

    agents = state_json.get("agents", {})

    for agent_id, config in intervention_config.items():
        if agent_id.startswith("_"):  # skip _usage etc
            continue
        if not isinstance(config, dict) or "action" not in config:
            continue

        action = config["action"]
        agent_data = agents.get(agent_id, {})
        category = agent_data.get("category", "vehicle")
        pos = agent_data.get("position", {})
        orig_x = pos.get("x", 0)
        orig_y = pos.get("y", 0)
        orig_yaw = pos.get("yaw", 0)

        if action == "remove":
            bev = remove_agent(bev, orig_x, orig_y, orig_yaw, category)
            # Add legend text
            cv2.putText(bev, f"REMOVED: {agent_id}", (20, BEV_SIZE - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        elif action == "reposition":
            cs = config.get("change_state", {})
            new_x = cs.get("x", orig_x)
            new_y = cs.get("y", orig_y)
            new_yaw = cs.get("yaw", orig_yaw)
            bev = reposition_agent(bev, orig_x, orig_y, orig_yaw,
                                   new_x, new_y, new_yaw, category, label=agent_id)
            cv2.putText(bev, f"REPOSITIONED: {agent_id}", (20, BEV_SIZE - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_INTERVENTION, 1)

        elif action == "speed_change":
            cs = config.get("change_state", {})
            trajectory = cs.get("trajectory", [])
            if not trajectory and cs.get("x") is not None:
                # Single waypoint speed change — compute simple trajectory
                new_speed = cs.get("speed", 0)
                yaw = cs.get("yaw", orig_yaw)
                trajectory = []
                for i in range(8):
                    t = (i + 1) * 0.5
                    trajectory.append({
                        "t": t,
                        "x": orig_x + new_speed * math.cos(yaw) * t,
                        "y": orig_y + new_speed * math.sin(yaw) * t,
                        "yaw": yaw,
                        "speed": new_speed
                    })
            bev = speed_change_agent(bev, orig_x, orig_y, orig_yaw,
                                     trajectory, category, label=agent_id)
            cv2.putText(bev, f"SPEED_CHANGE: {agent_id}", (20, BEV_SIZE - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TRAJ_DOTS, 1)

    return bev


def process_scene(scene_id: str, records_dir: Path, outputs_dir: Path,
                  provider=None, model: str = "claude-opus-4-6",
                  reasoning: str = "none", force: bool = False):
    """Process all nuplan_config CF results for a scene."""

    # Parse scene_id → scene_dir
    parts = scene_id.rsplit("_SAMPLED_", 1)
    if len(parts) != 2:
        print(f"  SKIP {scene_id}: can't parse scene_id")
        return 0
    scene_name, sample_idx = parts[0], f"SAMPLED_{parts[1]}"
    scene_dir = records_dir / scene_name / sample_idx

    if not scene_dir.exists():
        print(f"  SKIP {scene_id}: scene_dir not found")
        return 0

    # Load state.json
    state_path = scene_dir / "Tp0p0" / "state.json"
    if not state_path.exists():
        print(f"  SKIP {scene_id}: no state.json")
        return 0
    state_json = json.loads(state_path.read_text())

    # Find CF trajectory results for this scene
    cf_dir = outputs_dir / scene_id / "cf_trajectories"
    if not cf_dir.exists():
        return 0

    processed = 0
    for q_dir in sorted(cf_dir.iterdir()):
        if not q_dir.is_dir():
            continue

        # Find result.json (may be in q_dir/ or q_dir/<scene_id>/)
        result_path = q_dir / scene_id / "result.json"
        if not result_path.exists():
            result_path = q_dir / "result.json"
        if not result_path.exists():
            continue

        result = json.loads(result_path.read_text())
        gen = result.get("generator", {})

        # Check if this is a nuplan_config route (has per-agent action configs)
        has_agent_action = any(
            isinstance(v, dict) and "action" in v
            for k, v in gen.items() if not k.startswith("_")
        )
        if not has_agent_action:
            continue  # skip LLM-generated trajectories

        # Check if already processed
        modified_bev_path = result_path.parent / "bev_intervention.png"
        ego_traj_path = result_path.parent / "ego_cf_trajectory.json"
        if modified_bev_path.exists() and ego_traj_path.exists() and not force:
            processed += 1
            continue

        # Determine temporal anchor
        # Check manifest or infer from the question
        temporal_anchor = gen.get("temporal_anchor", "t0")

        # Extract intervention config (filter out _usage etc)
        intervention_config = {k: v for k, v in gen.items()
                              if isinstance(v, dict) and "action" in v}

        print(f"  [{scene_id}] Rendering intervention BEV: {list(intervention_config.keys())}")

        # Stage 4: Render modified BEV
        try:
            modified_bev = render_intervention_bev(
                scene_dir, state_json, intervention_config,
                temporal_anchor=temporal_anchor
            )
            cv2.imwrite(str(modified_bev_path), modified_bev)
            print(f"    → {modified_bev_path}")
        except Exception as e:
            print(f"    ERROR rendering BEV: {e}")
            continue

        # Stage 5: Generate ego trajectory (if provider available)
        if provider:
            try:
                ego_trajectory = generate_ego_trajectory(
                    scene_dir, state_json, modified_bev_path,
                    result, intervention_config, temporal_anchor,
                    provider, model, reasoning
                )
                with open(ego_traj_path, "w") as f:
                    json.dump(ego_trajectory, f, indent=2)
                print(f"    → {ego_traj_path}")
            except Exception as e:
                print(f"    ERROR generating trajectory: {e}")

        processed += 1

    return processed


def generate_ego_trajectory(scene_dir: Path, state_json: dict,
                            modified_bev_path: Path, result: dict,
                            intervention_config: dict, temporal_anchor: str,
                            provider, model: str, reasoning: str) -> dict:
    """Stage 5: Feed modified BEV + cameras to LLM for ego CF trajectory."""
    from common.scene_utils import image_content_block, prepare_timestep_blocks

    # Build image blocks: all timestep cameras + BEVs, then modified BEV last
    timestep_blocks = prepare_timestep_blocks(scene_dir, for_batch=False)

    # Add modified BEV as last image
    modified_bev_block = image_content_block(str(modified_bev_path),
                                             label="MODIFIED BEV (intervention applied)",
                                             for_batch=False)

    # Build text content
    question = result.get("question", "")
    answer = result.get("answer", "")
    gen = result.get("generator", {})
    reasoning_text = gen.get("reasoning", result.get("reasoning", ""))

    n_images = len([b for b in timestep_blocks if b.get("type") in ("image", "input_image")])

    text = f"""## Scene Data
{json.dumps(state_json, indent=2)}

## Counterfactual Question
{question}

## Answer
{answer}

## Intervention Config
{json.dumps(intervention_config, indent=2)}

## Image Order
Images 1-{n_images}: Original scene (cameras + BEV per timestep, T=-1.5 to T=0)
Image {n_images + 1}: MODIFIED BEV showing the intervention

LOOK at the MODIFIED BEV (last image). The intervention has been applied visually.
Generate ego's counterfactual trajectory as JSON with waypoints."""

    if temporal_anchor == "observation_window":
        text += """

This is a PAST counterfactual (observation_window). Generate 4 waypoints at T=-1.5, T=-1.0, T=-0.5, T=0.
Coordinates are ego-local at T=-1.5 frame."""
    else:
        text += """

This is a T=0 counterfactual. Generate 8 future waypoints at T=0.5, T=1.0, ... T=4.0.
Coordinates are ego-local at T=0 frame."""

    text += """

Return JSON:
```json
{
  "counterfactual_trajectory": [
    [x, y, heading, speed],
    ...
  ],
  "reasoning": "..."
}
```"""

    content_blocks = timestep_blocks + [modified_bev_block, {"type": "text", "text": text}]

    # Load system prompt
    prompts_dir = Path(__file__).parent.parent.parent / "prompts" / "CounterFactualTrajectory"
    if temporal_anchor == "observation_window":
        sys_path = prompts_dir / "CounterFactual_System_Prompt.md"
    else:
        sys_path = prompts_dir / "Intervention_System_Prompt.md"

    system_prompt = sys_path.read_text() if sys_path.exists() else "You are an autonomous driving trajectory planner."

    raw, usage = provider.call(system_prompt, content_blocks, max_tokens=8192)

    # Parse response
    from counterfactual_pipeline import extract_json
    result_data = extract_json(raw)
    result_data["usage"] = usage

    return result_data


def main():
    parser = argparse.ArgumentParser(
        description="BEV Intervention Renderer + Ego Trajectory Generator")
    parser.add_argument("--records", required=True, help="Records directory")
    parser.add_argument("--outputs", required=True, help="Outputs directory")
    parser.add_argument("--scene", default=None, help="Specific scene_id (default: all)")
    parser.add_argument("--provider", default=None, choices=["openai", "claude", "gemini"],
                        help="LLM provider for ego trajectory (omit for BEV render only)")
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--reasoning", default="none")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--render-only", action="store_true",
                        help="Only render BEV interventions, skip LLM trajectory generation")
    args = parser.parse_args()

    records_dir = Path(args.records)
    outputs_dir = Path(args.outputs)

    # Initialize provider if needed
    provider = None
    if args.provider and not args.render_only:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from common.llm_provider import get_provider
        from config_loader import load_config
        cfg = load_config() or {}
        api_key = None
        if args.provider == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY") or cfg.get("anthropic_api_key")
        elif args.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY") or cfg.get("api_key")
        provider = get_provider(args.provider, model=args.model,
                               reasoning=args.reasoning, api_key=api_key)

    # Find scenes to process
    if args.scene:
        scene_ids = [args.scene]
    else:
        # Find all scenes that have cf_trajectories/
        scene_ids = sorted(set(
            d.name for d in outputs_dir.iterdir()
            if d.is_dir() and (d / "cf_trajectories").exists()
        ))

    print(f"Processing {len(scene_ids)} scenes from {outputs_dir}")
    total = 0
    for scene_id in scene_ids:
        n = process_scene(scene_id, records_dir, outputs_dir,
                         provider=provider if not args.render_only else None,
                         model=args.model, reasoning=args.reasoning,
                         force=args.force)
        total += n

    print(f"\nDone. Processed {total} interventions across {len(scene_ids)} scenes.")


if __name__ == "__main__":
    main()
