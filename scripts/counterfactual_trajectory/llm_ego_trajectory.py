#!/usr/bin/env python3
"""
llm_ego_trajectory.py — Generate ego CF trajectories via LLM using modified BEV.

Takes bev_intervention.png (from trajectory_utils.py) + original cameras,
feeds to Claude, gets ego counterfactual trajectory. Saves alongside nuplan
results for side-by-side comparison.

Usage:
    # Single scene
    python llm_ego_trajectory.py \
        --records /path/to/records/openscene_50 \
        --outputs /path/to/outputs/openscene_50 \
        --scene openscene-2021.05.12.19.36.12_veh-35_002_SAMPLED_1 \
        --provider claude

    # All scenes
    python llm_ego_trajectory.py \
        --records /path/to/records/openscene_50 \
        --outputs /path/to/outputs/openscene_50 \
        --dataset openscene \
        --provider claude
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Add parent dirs for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "agent_intervention"


def load_prompt(name, dataset="nuscenes"):
    """Load prompt file, using av2_prompts/ for argoverse2."""
    if dataset == "argoverse2":
        p = PROMPTS_DIR / "av2_prompts" / name
        if p.exists() and p.stat().st_size > 0:
            return p.read_text()
    p = PROMPTS_DIR / name
    if p.exists():
        return p.read_text()
    raise FileNotFoundError(f"Prompt not found: {name}")


def fill_template(template, replacements):
    """Fill {{PLACEHOLDER}} and {{#if BLOCK}}...{{/if}} conditionals."""
    for key, val in replacements.items():
        template = template.replace(f"{{{{{key}}}}}", str(val))

    # Handle conditionals — match action type, also handle non-standard names
    action = replacements.get("ACTION_TYPE", "").upper()
    # Normalize: "BRAKE TO STOP" → treat as SPEED_CHANGE, "MODIFY_TRAJECTORY" → REPOSITION
    action_map = {
        "REMOVE": "REMOVE", "REPOSITION": "REPOSITION", "SPEED_CHANGE": "SPEED_CHANGE",
        "BRAKE TO STOP": "SPEED_CHANGE", "STOP": "SPEED_CHANGE",
        "MODIFY_TRAJECTORY": "REPOSITION", "MODIFY": "REPOSITION",
    }
    normalized_action = action_map.get(action, action)

    for atype in ["REMOVE", "REPOSITION", "SPEED_CHANGE"]:
        tag_open = "{{#if " + atype + "}}"
        tag_close = "{{/if}}"
        if atype == normalized_action:
            template = template.replace(tag_open, "").replace(tag_close, "")
        else:
            while tag_open in template:
                start = template.find(tag_open)
                if start == -1:
                    break
                end = template.find(tag_close, start)
                if end == -1:
                    break
                template = template[:start] + template[end + len(tag_close):]

    return template


def build_image_order_block(n_cams, temporal_anchor):
    """Build the image order description text."""
    cam_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK"]
    if n_cams == 5:
        cam_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"]

    ts_labels = ["T=-1.5", "T=-1.0", "T=-0.5", "T=0"]
    lines = []
    idx = 1
    for ts in ts_labels:
        for cam in cam_names:
            lines.append(f"Image {idx}: {cam} at {ts}")
            idx += 1
        lines.append(f"Image {idx}: BEV at {ts}")
        idx += 1
    return "\n".join(lines)


def process_scene(scene_id, records_dir, outputs_dir, dataset, provider, force=False):
    """Process all nuplan_config CF results for a scene via LLM."""
    from common.scene_utils import prepare_timestep_blocks, image_content_block

    parts = scene_id.rsplit("_SAMPLED_", 1)
    if len(parts) != 2:
        return 0
    scene_dir = records_dir / parts[0] / f"SAMPLED_{parts[1]}"
    if not scene_dir.exists():
        return 0

    cf_base = outputs_dir / scene_id / "cf_trajectories"
    if not cf_base.exists():
        return 0

    # Load state.json
    state_json = json.loads((scene_dir / "Tp0p0" / "state.json").read_text())
    nav_command = state_json.get("nav_command", "GO STRAIGHT")

    # Detect camera count
    tp0 = scene_dir / "Tp0p0"
    n_cams = 5 if (tp0 / "cam_back_right.jpg").exists() else 4

    processed = 0
    for q_dir in sorted(cf_base.iterdir()):
        if not q_dir.is_dir():
            continue

        # Find result.json
        scene_subdir = q_dir / scene_id if (q_dir / scene_id).exists() else q_dir
        result_path = scene_subdir / "result.json"
        if not result_path.exists():
            continue

        result = json.loads(result_path.read_text())
        gen = result.get("generator", {})

        # Only process nuplan_config results
        clean_config = {k: v for k, v in gen.items()
                        if isinstance(v, dict) and "action" in v}
        if not clean_config:
            continue

        # Check if LLM trajectory already exists
        llm_traj_path = scene_subdir / "llm_ego_trajectory.json"
        if llm_traj_path.exists() and not force:
            processed += 1
            continue

        # Check bev_intervention.png exists
        bev_int = scene_subdir / "bev_intervention.png"
        if not bev_int.exists():
            continue

        temporal_anchor = gen.get("temporal_anchor", "t0")
        question = result.get("question", "")
        action_type = list(clean_config.values())[0]["action"].upper()

        print(f"  [{scene_id}] LLM trajectory: {action_type} — {question[:50]}...")

        # Build image blocks: timestep cameras + BEVs
        ts_blocks = prepare_timestep_blocks(scene_dir, for_batch=False)

        if temporal_anchor == "observation_window":
            # Past: add 4 modified BEVs
            sys_prompt = load_prompt("CF_Ego_Trajectory_Past_System_Prompt.md", dataset)
            user_template = load_prompt("CF_Ego_Trajectory_Past_User_Prompt.md", dataset)

            modified_blocks = []
            for slug in ["Tm1p5", "Tm1p0", "Tm0p5", "Tp0p0"]:
                mbev = scene_subdir / f"bev_intervention_{slug}.png"
                if not mbev.exists():
                    mbev = bev_int  # fallback to single frame
                modified_blocks.append(
                    image_content_block(str(mbev), label=f"MODIFIED BEV {slug}", for_batch=False))

            all_blocks = ts_blocks + modified_blocks
        else:
            # T=0: add 1 modified BEV
            sys_prompt = load_prompt("CF_Ego_Trajectory_T0_System_Prompt.md", dataset)
            user_template = load_prompt("CF_Ego_Trajectory_T0_User_Prompt.md", dataset)

            modified_block = image_content_block(
                str(bev_int), label="MODIFIED BEV at T=0", for_batch=False)
            all_blocks = ts_blocks + [modified_block]

        # Fill user prompt template
        image_order = build_image_order_block(n_cams, temporal_anchor)
        user_text = fill_template(user_template, {
            "IMAGE_ORDER_BLOCK": image_order,
            "NAV_COMMAND": nav_command,
            "STATE_JSON": json.dumps(state_json, indent=2),
            "QUESTION": question,
            "INTERVENTION_CONFIG": json.dumps(clean_config, indent=2),
            "ACTION_TYPE": action_type,
        })

        content_blocks = all_blocks + [{"type": "text", "text": user_text}]

        # Call LLM
        try:
            raw, usage = provider.call(sys_prompt, content_blocks, max_tokens=8192)

            # Parse response
            from counterfactual_pipeline import extract_json
            llm_result = extract_json(raw)
            llm_result["_usage"] = usage
            llm_result["_temporal_anchor"] = temporal_anchor
            llm_result["_action_type"] = action_type

            with open(llm_traj_path, "w") as f:
                json.dump(llm_result, f, indent=2)
            print(f"    → {llm_traj_path}")
            processed += 1

        except Exception as e:
            print(f"    ERROR: {e}")

    return processed


def main():
    parser = argparse.ArgumentParser(description="Generate ego CF trajectories via LLM")
    parser.add_argument("--records", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--dataset", default="openscene",
                        choices=["nuscenes", "openscene", "argoverse2"])
    parser.add_argument("--scene", default=None, help="Specific scene_id")
    parser.add_argument("--provider", default="claude",
                        choices=["claude", "openai", "gemini"])
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--reasoning", default="none")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    records_dir = Path(args.records)
    outputs_dir = Path(args.outputs)

    # Init provider
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

    # Find scenes
    if args.scene:
        scene_ids = [args.scene]
    else:
        scene_ids = sorted(set(
            d.name for d in outputs_dir.iterdir()
            if d.is_dir() and (d / "cf_trajectories").exists()
        ))
    if args.max_samples:
        scene_ids = scene_ids[:args.max_samples]

    print(f"Processing {len(scene_ids)} scenes via {args.provider}/{args.model}")
    total = 0
    for scene_id in scene_ids:
        n = process_scene(scene_id, records_dir, outputs_dir, args.dataset,
                         provider, force=args.force)
        total += n

    print(f"\nDone. Generated {total} LLM ego trajectories across {len(scene_ids)} scenes.")


if __name__ == "__main__":
    main()
