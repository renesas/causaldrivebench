#!/usr/bin/env python3
"""
Prepare batch request files for OpenAI Batch API.

Creates JSONL files for each stage of the pipeline:
- Stage 1:   Causal graph generation (generator)
- Stage 2:   Causal graph refinement (discriminator)
- Stage 3a:  QA Active questions
- Stage 3b:  QA Dormant questions
- Stage 4cf: Counterfactual trajectories (CR Rung-3 questions)
- Stage 4ac: Activation counterfactual trajectories (AC-Binary Yes questions)

Usage:
    python prepare_batches.py --stage 1  --dataset nuscenes --records /path/to/records
    python prepare_batches.py --stage 2  --dataset nuscenes --records /path/to/records --graphs /path/to/graphs
    python prepare_batches.py --stage 3a --dataset nuscenes --records /path/to/records --graphs /path/to/graphs
    python prepare_batches.py --stage 3b --dataset nuscenes --records /path/to/records --graphs /path/to/graphs
    python prepare_batches.py --stage 4cf --dataset nuscenes --records /path/to/records --graphs /path/to/graphs --qa-results /path/to/qa
    python prepare_batches.py --stage 4ac --dataset nuscenes --records /path/to/records --graphs /path/to/graphs --qa-results /path/to/qa
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from config_loader import load_config
from common.scene_utils import (
    find_scene_directories,
    prepare_scene_content,
    create_batch_request,
    load_prompt,
    filter_state_json,
    state_to_natural_language,
    format_trajectory_as_waypoints,
    text_block,
    image_content_block,
)
from common.llm_provider import get_provider

import hashlib
import re

# Global map: sanitized_id → original_id. Saved alongside batch JSONL.
_ID_MAP = {}


def resolve_graph_path(graphs_dir: Path, scene_id: str) -> Optional[Path]:
    """Return the best available graph file: verified > edited > pruned."""
    for suffix in ("_verified.json", "_naren_edited.json", "_pruned.json"):
        p = graphs_dir / f"{scene_id}{suffix}"
        if p.exists():
            return p
    return None


def load_filter_set(filter_path: Optional[str]) -> Optional[set]:
    """Load an allowlist of scene keys from a diverse_subset.json manifest.
    Returns None (no filtering) if filter_path is not provided."""
    if not filter_path:
        return None
    p = Path(filter_path)
    if not p.exists():
        print(f"WARNING: filter file {p} not found — processing all scenes")
        return None
    data = json.loads(p.read_text())
    keys = {entry["key"] for entry in data.get("selected", [])}
    print(f"  Filter loaded: {len(keys)} scenes from {p.name}")
    return keys


def apply_filter(scene_dirs: List, filter_set: Optional[set]) -> List:
    """Filter scene_dirs to only those whose key is in filter_set."""
    if filter_set is None:
        return scene_dirs
    filtered = [
        d for d in scene_dirs
        if f"{d.parent.name}_{d.name}" in filter_set
    ]
    print(f"  Filtered {len(scene_dirs)} → {len(filtered)} scenes")
    return filtered


def _sanitize_id(s: str) -> str:
    """Sanitize custom_id for Claude batch API: only [a-zA-Z0-9_-], max 64 chars.
    Uses a short hash suffix to guarantee uniqueness even after sanitization."""
    clean = re.sub(r'[^a-zA-Z0-9_-]', '', s)
    if clean == s and len(s) <= 64:
        return s  # already valid
    h = hashlib.md5(s.encode()).hexdigest()[:8]
    sanitized = f"{clean[:55]}_{h}"
    _ID_MAP[sanitized] = s
    return sanitized

def save_id_map(output_dir: Path, dataset: str, stage: str):
    """Save the sanitized→original ID map if any IDs were sanitized."""
    if not _ID_MAP:
        return
    map_file = output_dir / f"{dataset}_{stage}_id_map.json"
    with open(map_file, "w") as f:
        json.dump(_ID_MAP, f, indent=2)
    _ID_MAP.clear()

# ── Prompt paths ──────────────────────────────────────────────────────────
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
GRAPH_PROMPTS  = PROMPTS_DIR / "causal_scene_graph"
QA_PROMPTS     = PROMPTS_DIR / "causal_qas"
CF_PROMPTS     = PROMPTS_DIR / "alternate_trajectories"
NUPLAN_PROMPTS    = PROMPTS_DIR / "nuplan_config"
VERIFY_PROMPTS    = PROMPTS_DIR / "graph_verification"
AGENT_INT_PROMPTS = PROMPTS_DIR / "agent_intervention"

# ── Camera variables for v5 system prompt template ────────────────────────
_DEFAULT_CAMERA_VARS = {
    # causal_scene_graph/v5/system_prompt.md
    "camera_images_desc": (
        "16 images — 4 timesteps (T=-1.5, T=-1.0, T=-0.5, T=0) × 4 views "
        "(CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK). "
        "CAM_BACK is unmirrored (left/right reversed vs front cameras). "
        "Compare same view across timesteps to infer motion/speed/trajectory."
    ),
    "bev_rear_mapping": "  • Behind on BEV (down) → visible in CAM_BACK",
    "cam_back_name": "CAM_BACK",
    "num_camera_images": "16",
    "image_ordering": (
        "T=-1.5: 1. CAM_FRONT, 2. CAM_FRONT_LEFT, 3. CAM_FRONT_RIGHT, 4. CAM_BACK, 5. BEV\n"
        "T=-1.0: 6. CAM_FRONT, 7. CAM_FRONT_LEFT, 8. CAM_FRONT_RIGHT, 9. CAM_BACK, 10. BEV\n"
        "T=-0.5: 11. CAM_FRONT, 12. CAM_FRONT_LEFT, 13. CAM_FRONT_RIGHT, 14. CAM_BACK, 15. BEV\n"
        "T=0:    16. CAM_FRONT, 17. CAM_FRONT_LEFT, 18. CAM_FRONT_RIGHT, 19. CAM_BACK, 20. BEV\n"
        "21: Scene state description (T=0)"
    ),
    # QA v5 system prompts
    "qa_camera_header": "16 camera images** — 4 views (front, front-left, front-right, back)",
    # CF CounterFactual_System_Prompt
    "cf_sys_image_ordering": (
        "T=-1.5: 1. CAM_FRONT, 2. CAM_FRONT_LEFT, 3. CAM_FRONT_RIGHT, 4. CAM_BACK\n"
        "T=-1.0: 5. CAM_FRONT, 6. CAM_FRONT_LEFT, 7. CAM_FRONT_RIGHT, 8. CAM_BACK\n"
        "T=-0.5: 9. CAM_FRONT, 10. CAM_FRONT_LEFT, 11. CAM_FRONT_RIGHT, 12. CAM_BACK\n"
        "T=0:    13. CAM_FRONT, 14. CAM_FRONT_LEFT, 15. CAM_FRONT_RIGHT, 16. CAM_BACK"
    ),
    # CF CounterFactual_User_Prompt
    "cf_usr_image_ordering": (
        "T=-1.5: [Image 1] CAM_FRONT, [Image 2] CAM_FRONT_LEFT, [Image 3] CAM_FRONT_RIGHT, [Image 4] CAM_BACK, [Image 5] BEV\n"
        "T=-1.0: [Image 6] CAM_FRONT, [Image 7] CAM_FRONT_LEFT, [Image 8] CAM_FRONT_RIGHT, [Image 9] CAM_BACK, [Image 10] BEV\n"
        "T=-0.5: [Image 11] CAM_FRONT, [Image 12] CAM_FRONT_LEFT, [Image 13] CAM_FRONT_RIGHT, [Image 14] CAM_BACK, [Image 15] BEV\n"
        "T=0:    [Image 16] CAM_FRONT, [Image 17] CAM_FRONT_LEFT, [Image 18] CAM_FRONT_RIGHT, [Image 19] CAM_BACK, [Image 20] BEV"
    ),
    # CF Intervention_User_Prompt
    "intervention_images_section": (
        "=== IMAGES ===\n"
        "[Image 1]: Front camera at T=0\n"
        "[Image 2]: Front-left camera at T=0\n"
        "[Image 3]: Front-right camera at T=0\n"
        "[Image 4]: Back camera at T=0\n"
        "[Image 5]: BEV at T=0 (labeled bounding boxes, lane markings, road layout)"
    ),
    "intervention_generation_instruction": (
        "Generate the counterfactual trajectory. Use the provided question to identify the "
        "intervention, and use the provided answer and reasoning to inform your binding analysis. "
        "Return ONLY valid JSON."
    ),
    # nuplan_config/system_prompt.md
    "nuplan_cam_listing": (
        "T=-1.5: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK\n"
        "T=-1.0: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK\n"
        "T=-0.5: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK\n"
        "T=0:    CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK"
    ),
    "nuplan_rear_cam_note": "CAM_BACK is unmirrored (left/right reversed vs front cameras).",
}

_AV2_CAMERA_VARS = {
    # causal_scene_graph/v5/system_prompt.md
    "camera_images_desc": (
        "20 images — 4 timesteps (T=-1.5, T=-1.0, T=-0.5, T=0) × 5 views "
        "(CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, CAM_BACK_LEFT). "
        "The two rear cameras cover the left-rear and right-rear respectively — "
        "together they provide full rear coverage. "
        "Compare same view across timesteps to infer motion/speed/trajectory."
    ),
    "bev_rear_mapping": (
        "  • Behind-right on BEV (lower-right) → visible in CAM_BACK_RIGHT\n"
        "  • Behind-left on BEV (lower-left) → visible in CAM_BACK_LEFT"
    ),
    "cam_back_name": "CAM_BACK_LEFT or CAM_BACK_RIGHT",
    "num_camera_images": "20",
    "image_ordering": (
        "T=-1.5: 1. CAM_FRONT, 2. CAM_FRONT_LEFT, 3. CAM_FRONT_RIGHT, 4. CAM_BACK_RIGHT, 5. CAM_BACK_LEFT, 6. BEV\n"
        "T=-1.0: 7. CAM_FRONT, 8. CAM_FRONT_LEFT, 9. CAM_FRONT_RIGHT, 10. CAM_BACK_RIGHT, 11. CAM_BACK_LEFT, 12. BEV\n"
        "T=-0.5: 13. CAM_FRONT, 14. CAM_FRONT_LEFT, 15. CAM_FRONT_RIGHT, 16. CAM_BACK_RIGHT, 17. CAM_BACK_LEFT, 18. BEV\n"
        "T=0:    19. CAM_FRONT, 20. CAM_FRONT_LEFT, 21. CAM_FRONT_RIGHT, 22. CAM_BACK_RIGHT, 23. CAM_BACK_LEFT, 24. BEV\n"
        "25: Scene state description (T=0)"
    ),
    # QA v5 system prompts
    "qa_camera_header": "20 camera images** — 5 views (front, front-left, front-right, back-right, back-left)",
    # CF CounterFactual_System_Prompt
    "cf_sys_image_ordering": (
        "T=-1.5: 1. CAM_FRONT, 2. CAM_FRONT_LEFT, 3. CAM_FRONT_RIGHT, 4. CAM_BACK_RIGHT, 5. CAM_BACK_LEFT\n"
        "T=-1.0: 6. CAM_FRONT, 7. CAM_FRONT_LEFT, 8. CAM_FRONT_RIGHT, 9. CAM_BACK_RIGHT, 10. CAM_BACK_LEFT\n"
        "T=-0.5: 11. CAM_FRONT, 12. CAM_FRONT_LEFT, 13. CAM_FRONT_RIGHT, 14. CAM_BACK_RIGHT, 15. CAM_BACK_LEFT\n"
        "T=0:    16. CAM_FRONT, 17. CAM_FRONT_LEFT, 18. CAM_FRONT_RIGHT, 19. CAM_BACK_RIGHT, 20. CAM_BACK_LEFT"
    ),
    # CF CounterFactual_User_Prompt
    "cf_usr_image_ordering": (
        "T=-1.5: [Image 1] CAM_FRONT, [Image 2] CAM_FRONT_LEFT, [Image 3] CAM_FRONT_RIGHT, [Image 4] CAM_BACK_RIGHT, [Image 5] CAM_BACK_LEFT, [Image 6] BEV\n"
        "T=-1.0: [Image 7] CAM_FRONT, [Image 8] CAM_FRONT_LEFT, [Image 9] CAM_FRONT_RIGHT, [Image 10] CAM_BACK_RIGHT, [Image 11] CAM_BACK_LEFT, [Image 12] BEV\n"
        "T=-0.5: [Image 13] CAM_FRONT, [Image 14] CAM_FRONT_LEFT, [Image 15] CAM_FRONT_RIGHT, [Image 16] CAM_BACK_RIGHT, [Image 17] CAM_BACK_LEFT, [Image 18] BEV\n"
        "T=0:    [Image 19] CAM_FRONT, [Image 20] CAM_FRONT_LEFT, [Image 21] CAM_FRONT_RIGHT, [Image 22] CAM_BACK_RIGHT, [Image 23] CAM_BACK_LEFT, [Image 24] BEV"
    ),
    # CF Intervention_User_Prompt
    "intervention_images_section": (
        "=== IMAGES (grouped by timestep) ===\n"
        "T=-1.5: [Image 1] CAM_FRONT, [Image 2] CAM_FRONT_LEFT, [Image 3] CAM_FRONT_RIGHT, [Image 4] CAM_BACK_RIGHT, [Image 5] CAM_BACK_LEFT, [Image 6] BEV\n"
        "T=-1.0: [Image 7] CAM_FRONT, [Image 8] CAM_FRONT_LEFT, [Image 9] CAM_FRONT_RIGHT, [Image 10] CAM_BACK_RIGHT, [Image 11] CAM_BACK_LEFT, [Image 12] BEV\n"
        "T=-0.5: [Image 13] CAM_FRONT, [Image 14] CAM_FRONT_LEFT, [Image 15] CAM_FRONT_RIGHT, [Image 16] CAM_BACK_RIGHT, [Image 17] CAM_BACK_LEFT, [Image 18] BEV\n"
        "T=0:    [Image 19] CAM_FRONT, [Image 20] CAM_FRONT_LEFT, [Image 21] CAM_FRONT_RIGHT, [Image 22] CAM_BACK_RIGHT, [Image 23] CAM_BACK_LEFT, [Image 24] BEV"
    ),
    "intervention_generation_instruction": (
        "Generate the counterfactual trajectory. The entity in question is currently constraining "
        "ego's behavior at T=0. The counterfactual asks: if this entity were removed/changed right "
        "now, how would ego drive over the next 4 seconds?\n\n"
        "Use the BEV frames to verify spatial layout and lane geometry. Use the provided answer and "
        "reasoning to inform your binding analysis. Return ONLY valid JSON."
    ),
    # nuplan_config/system_prompt.md
    "nuplan_cam_listing": (
        "T=-1.5: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, CAM_BACK_LEFT\n"
        "T=-1.0: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, CAM_BACK_LEFT\n"
        "T=-0.5: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, CAM_BACK_LEFT\n"
        "T=0:    CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, CAM_BACK_LEFT"
    ),
    "nuplan_rear_cam_note": "The two rear cameras (CAM_BACK_RIGHT, CAM_BACK_LEFT) cover the left-rear and right-rear respectively.",
}

BEV_SPATIAL_TEXT = (
    "**BEV orientation:** Ego is at the center, always facing UP.\n"
    "- Vertical axis = X axis: up = +x (ahead of ego), down = -x (behind ego).\n"
    "- Horizontal axis = Y axis: left in image = +y (left of ego), right in image = -y (right of ego).\n"
    "- This matches the state.json coordinate frame exactly.\n\n"
    "**BEV Visual Encoding:**\n"
    "- Ego vehicle: CYAN/BLUE oriented box at center with heading arrow.\n"
    "- Other vehicles: PEACH oriented boxes with heading arrows (if moving).\n"
    "- Pedestrians: GREEN small boxes with heading arrows (if moving).\n"
    "- Cyclists: YELLOW small boxes with heading arrows (if moving).\n"
    "- Heading arrows: drawn from the front edge of moving agents (speed > 0.1 m/s), indicating direction of travel.\n\n"
    "**Map layers:**\n"
    "- Drivable area: dark grey surface.\n"
    "- Lanes: medium grey fill with white boundary lines.\n"
    "- Lane direction arrows: light grey triangles along lane centerlines.\n"
    "- Pedestrian crossings: orange striped regions.\n"
    "- Walkways: dark green areas (sidewalks, separated from road).\n"
    "- Lane dividers: white lines. Road dividers: yellow lines."
)


# ── Stage 1: Generator ────────────────────────────────────────────────────

def prepare_stage1_batch(records_dir: Path, dataset: str, model: str,
                         reasoning: str, output_dir: Path,
                         max_samples: int = None,
                         graphs_dir: Path = None,
                         provider=None, layout: str = "v1") -> Path:
    """Prepare Stage 1 (causal graph generator) batch requests."""
    print("Preparing Stage 1: Causal Graph Generation")

    prompt_subdir = GRAPH_PROMPTS / layout
    system_prompt = (prompt_subdir / "system_prompt.md").read_text()
    camera_vars = _AV2_CAMERA_VARS if dataset == "argoverse2" else _DEFAULT_CAMERA_VARS
    for key, value in camera_vars.items():
        system_prompt = system_prompt.replace(f"{{{key}}}", value)
    scene_dirs = find_scene_directories(records_dir, dataset)
    if max_samples:
        scene_dirs = scene_dirs[:max_samples]
    print(f"  Found {len(scene_dirs)} scene samples  (layout={layout})")

    batch_file = output_dir / f"{dataset}_stage1_generator.jsonl"
    requests, skipped = [], 0

    for scene_dir in scene_dirs:
        content = prepare_scene_content(scene_dir, layout=layout, for_batch=True)
        scene_id = content["scene_id"]

        if graphs_dir and (graphs_dir / f"{scene_id}.json").exists():
            skipped += 1
            continue

        # Build content blocks based on layout
        if layout == "v2":
            content_blocks = list(content["bev_blocks"])
            content_blocks.extend(content["camera_blocks"])
        elif layout in ("v5", "v6"):
            content_blocks = list(content["camera_blocks"])
            content_blocks.extend(content["bev_blocks"])
        else:
            content_blocks = list(content["camera_blocks"])
            content_blocks.append(content["bev_block"])
        content_blocks.append(text_block(
            f"Scene state description:\n{content['state_nl']}", for_batch=True
        ))

        formatted_prompt = system_prompt.replace("{nav_command}", content["nav_command"])

        if provider:
            requests.append(provider.format_batch_request(
                custom_id=_sanitize_id(f"{scene_id}_stage1"),
                system_prompt=formatted_prompt,
                content_blocks=content_blocks,
            ))
        else:
            requests.append(create_batch_request(
                custom_id=_sanitize_id(f"{scene_id}_stage1"),
                model=model,
                system_prompt=formatted_prompt,
                content_blocks=content_blocks,
                reasoning=reasoning,
            ))

    with open(batch_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"  Wrote {len(requests)} requests → {batch_file}  (skipped {skipped} already done)")
    save_id_map(output_dir, dataset, "stage1")
    return batch_file


# ── Stage 2: Discriminator ────────────────────────────────────────────────

def prepare_stage2_batch(records_dir: Path, graphs_dir: Path, dataset: str,
                         model: str, reasoning: str, output_dir: Path,
                         max_samples: int = None,
                         provider=None, layout: str = "v1") -> Path:
    """Prepare Stage 2 (causal graph discriminator/refiner) batch requests."""
    print("Preparing Stage 2: Causal Graph Refinement")

    prompt_subdir = GRAPH_PROMPTS / layout
    refiner_prompt = (prompt_subdir / "refinement_prompt.md").read_text()
    scene_dirs = find_scene_directories(records_dir, dataset)
    if max_samples:
        scene_dirs = scene_dirs[:max_samples]

    batch_file = output_dir / f"{dataset}_stage2_discriminator.jsonl"
    requests, skipped = [], 0

    for scene_dir in scene_dirs:
        content = prepare_scene_content(scene_dir, layout=layout, for_batch=True)
        scene_id = content["scene_id"]
        raw_graph_path = graphs_dir / f"{scene_id}.json"

        if not raw_graph_path.exists():
            print(f"  Skipping {scene_id} — no raw graph")
            skipped += 1
            continue

        raw_graph = json.loads(raw_graph_path.read_text())

        if layout == "v2":
            content_blocks = list(content["bev_blocks"])
            content_blocks.extend(content["camera_blocks"])
        elif layout in ("v5", "v6"):
            content_blocks = list(content["camera_blocks"])
            content_blocks.extend(content["bev_blocks"])
        else:
            content_blocks = list(content["camera_blocks"])
            content_blocks.append(content["bev_block"])
        content_blocks.append(text_block(
            f"Scene state description:\n{content['state_nl']}", for_batch=True
        ))
        content_blocks.append(text_block(
            f"Raw causal graph to review:\n```json\n{json.dumps(raw_graph, indent=2)}\n```",
            for_batch=True,
        ))

        formatted_prompt = refiner_prompt.replace("{nav_command}", content["nav_command"])

        if provider:
            requests.append(provider.format_batch_request(
                custom_id=_sanitize_id(f"{scene_id}_stage2"),
                system_prompt=formatted_prompt,
                content_blocks=content_blocks,
            ))
        else:
            requests.append(create_batch_request(
                custom_id=_sanitize_id(f"{scene_id}_stage2"),
                model=model,
                system_prompt=formatted_prompt,
                content_blocks=content_blocks,
                reasoning=reasoning,
            ))

    with open(batch_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"  Wrote {len(requests)} requests → {batch_file}  (skipped {skipped})")
    save_id_map(output_dir, dataset, "stage2")
    return batch_file


# ── Stage 3a: QA Active ───────────────────────────────────────────────────

def prepare_stage3a_batch(records_dir: Path, graphs_dir: Path, dataset: str,
                          model: str, reasoning: str, output_dir: Path,
                          prompt_version: str = "v3", max_samples: int = None,
                          provider=None, layout: str = "v1",
                          filter_set: set = None) -> Path:
    """Prepare Stage 3a (QA Active) batch requests."""
    print("Preparing Stage 3a: QA Active Questions")

    prompts_dir   = QA_PROMPTS / prompt_version
    camera_vars = _AV2_CAMERA_VARS if dataset == "argoverse2" else _DEFAULT_CAMERA_VARS
    system_prompt = (prompts_dir / "Causal_Ladder_System_Prompt.md").read_text()
    user_template = (prompts_dir / "Causal_Ladder_User_Prompt.md").read_text()
    for key, value in camera_vars.items():
        system_prompt = system_prompt.replace(f"{{{key}}}", value)
        user_template = user_template.replace(f"{{{key}}}", value)

    scene_dirs = find_scene_directories(records_dir, dataset)
    scene_dirs = apply_filter(scene_dirs, filter_set)
    if max_samples:
        scene_dirs = scene_dirs[:max_samples]

    batch_file = output_dir / f"{dataset}_stage3a_active.jsonl"
    requests, skipped = [], 0

    for scene_dir in scene_dirs:
        content = prepare_scene_content(scene_dir, layout=layout, for_batch=True)
        scene_id = content["scene_id"]
        refined_graph_path = resolve_graph_path(graphs_dir, scene_id)

        if refined_graph_path is None:
            print(f"  Skipping {scene_id} — no graph found")
            skipped += 1
            continue

        graph = json.loads(refined_graph_path.read_text())
        graph.pop("changes", None)

        # Load pre-computed subgraph analysis (Markdown) if available
        subgraph_path = graphs_dir / f"{scene_id}_subgraphs.md"
        subgraph_md = subgraph_path.read_text() if subgraph_path.exists() else None

        # Skip if only ego node (no active edges for Active QA)
        non_ego_nodes = [n for n in graph.get("nodes", []) if n.get("id") != "ego"]
        if not non_ego_nodes:
            print(f"  Skipping {scene_id} — only ego node (no active edges for Active QA)")
            skipped += 1
            continue

        assert content["camera_blocks"], f"No camera blocks for {scene_id}"
        content_blocks = []
        if layout in ("v5", "v6"):
            content_blocks.extend(content["camera_blocks"])
            content_blocks.extend(content["bev_blocks"])
        elif layout == "v2":
            content_blocks.extend(content["bev_blocks"])
            content_blocks.extend(content["camera_blocks"])
        else:
            content_blocks.extend(content["camera_blocks"])
            content_blocks.append(content["bev_block"])
        filtered_state = filter_state_json(content["state_json"], graph.get("nodes", []))

        user_prompt = user_template.replace(
            "[paste scene graph JSON here]", json.dumps(graph, indent=2)
        )
        if filtered_state and "[paste filtered state JSON here]" in user_prompt:
            filtered_agent_labels = set(filtered_state.get("agents", {}).keys())
            filtered_state_nl = state_to_natural_language(content["state_json"], agents_filter=filtered_agent_labels)
            user_prompt = user_prompt.replace(
                "[paste filtered state JSON here]", filtered_state_nl
            )
        # Inject subgraph analysis into user prompt
        if subgraph_md and "[paste subgraph analysis here]" in user_prompt:
            user_prompt = user_prompt.replace(
                "[paste subgraph analysis here]", subgraph_md
            )
        elif "[paste subgraph analysis here]" in user_prompt:
            # No subgraph file — strip the placeholder section
            import re as _re
            user_prompt = _re.sub(
                r"\n?<SUBGRAPH_ANALYSIS>\n\[paste subgraph analysis here\]\n</SUBGRAPH_ANALYSIS>",
                "", user_prompt
            )
        content_blocks.append(text_block(user_prompt, for_batch=True))

        if provider:
            requests.append(provider.format_batch_request(
                custom_id=_sanitize_id(f"{scene_id}_stage3a"),
                system_prompt=system_prompt,
                content_blocks=content_blocks,
            ))
        else:
            requests.append(create_batch_request(
                custom_id=_sanitize_id(f"{scene_id}_stage3a"),
                model=model,
                system_prompt=system_prompt,
                content_blocks=content_blocks,
                reasoning=reasoning,
            ))

    with open(batch_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"  Wrote {len(requests)} requests → {batch_file}  (skipped {skipped})")
    save_id_map(output_dir, dataset, "stage3a")
    return batch_file


# ── Stage 3b: QA Dormant ──────────────────────────────────────────────────────

def prepare_stage3b_batch(records_dir: Path, graphs_dir: Path, dataset: str,
                          model: str, reasoning: str, output_dir: Path,
                          prompt_version: str = "v3", max_samples: int = None,
                          provider=None, layout: str = "v1",
                          filter_set: set = None) -> Path:
    """Prepare Stage 3b (QA Dormant) batch requests."""
    print("Preparing Stage 3b: QA Dormant Questions")

    prompts_dir   = QA_PROMPTS / prompt_version
    camera_vars = _AV2_CAMERA_VARS if dataset == "argoverse2" else _DEFAULT_CAMERA_VARS
    system_prompt = (prompts_dir / "Dormant_Nodes_System_Prmpt.md").read_text()
    user_template = (prompts_dir / "Dormant_Nodes_User_Prompt.md").read_text()
    for key, value in camera_vars.items():
        system_prompt = system_prompt.replace(f"{{{key}}}", value)
        user_template = user_template.replace(f"{{{key}}}", value)

    scene_dirs = find_scene_directories(records_dir, dataset)
    scene_dirs = apply_filter(scene_dirs, filter_set)
    if max_samples:
        scene_dirs = scene_dirs[:max_samples]

    batch_file = output_dir / f"{dataset}_stage3b_dormant.jsonl"
    requests, skipped = [], 0

    for scene_dir in scene_dirs:
        content = prepare_scene_content(scene_dir, layout=layout, for_batch=True)
        scene_id = content["scene_id"]
        refined_graph_path = resolve_graph_path(graphs_dir, scene_id)

        if refined_graph_path is None:
            print(f"  Skipping {scene_id} — no graph found")
            skipped += 1
            continue

        graph = json.loads(refined_graph_path.read_text())
        graph.pop("changes", None)

        # Skip if only ego node
        non_ego_nodes = [n for n in graph.get("nodes", []) if n.get("id") != "ego"]
        if not non_ego_nodes:
            print(f"  Skipping {scene_id} — only ego node (no dormant edges for Dormant QA)")
            skipped += 1
            continue

        # Skip scenes with no inactive edges (nothing to ask about in Dormant QA)
        has_inactive = any(
            str(e.get("active", True)).lower() != "true"
            for e in graph.get("ego_edges", [])
        )
        if not has_inactive:
            print(f"  Skipping {scene_id} — no inactive ego_edges")
            skipped += 1
            continue

        assert content["camera_blocks"], f"No camera blocks for {scene_id}"
        content_blocks = []
        if layout in ("v5", "v6"):
            content_blocks.extend(content["camera_blocks"])
            content_blocks.extend(content["bev_blocks"])
        elif layout == "v2":
            content_blocks.extend(content["bev_blocks"])
            content_blocks.extend(content["camera_blocks"])
        else:
            content_blocks.extend(content["camera_blocks"])
            content_blocks.append(content["bev_block"])
        filtered_state = filter_state_json(content["state_json"], graph.get("nodes", []))

        user_prompt = user_template.replace(
            "[paste scene graph JSON here]", json.dumps(graph, indent=2)
        )
        if filtered_state and "[paste filtered state JSON here]" in user_prompt:
            filtered_agent_labels = set(filtered_state.get("agents", {}).keys())
            filtered_state_nl = state_to_natural_language(content["state_json"], agents_filter=filtered_agent_labels)
            user_prompt = user_prompt.replace(
                "[paste filtered state JSON here]", filtered_state_nl
            )
        content_blocks.append(text_block(user_prompt, for_batch=True))

        if provider:
            requests.append(provider.format_batch_request(
                custom_id=_sanitize_id(f"{scene_id}_stage3b"),
                system_prompt=system_prompt,
                content_blocks=content_blocks,
            ))
        else:
            requests.append(create_batch_request(
                custom_id=_sanitize_id(f"{scene_id}_stage3b"),
                model=model,
                system_prompt=system_prompt,
                content_blocks=content_blocks,
                reasoning=reasoning,
            ))

    with open(batch_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"  Wrote {len(requests)} requests → {batch_file}  (skipped {skipped})")
    save_id_map(output_dir, dataset, "stage3b")
    return batch_file


# ── Stage 3c: Distractor Nodes QA ────────────────────────────────────────

def prepare_stage3c_batch(records_dir: Path, graphs_dir: Path, dataset: str,
                          model: str, reasoning: str, output_dir: Path,
                          prompt_version: str = "v4", max_samples: int = None,
                          provider=None, layout: str = "v1",
                          filter_set: set = None) -> Path:
    """Prepare Stage 3c (Distractor Nodes QA) batch requests."""
    print("Preparing Stage 3c: Distractor Nodes Questions")

    prompts_dir   = QA_PROMPTS / prompt_version
    camera_vars = _AV2_CAMERA_VARS if dataset == "argoverse2" else _DEFAULT_CAMERA_VARS
    system_prompt = (prompts_dir / "Distractor_Nodes_System_Prompt.md").read_text()
    user_template = (prompts_dir / "Distractor_Nodes_User_Prompt.md").read_text()
    for key, value in camera_vars.items():
        system_prompt = system_prompt.replace(f"{{{key}}}", value)
        user_template = user_template.replace(f"{{{key}}}", value)

    scene_dirs = find_scene_directories(records_dir, dataset)
    scene_dirs = apply_filter(scene_dirs, filter_set)
    if max_samples:
        scene_dirs = scene_dirs[:max_samples]

    batch_file = output_dir / f"{dataset}_stage3c_distractor.jsonl"
    requests, skipped = [], 0

    for scene_dir in scene_dirs:
        content = prepare_scene_content(scene_dir, layout=layout, for_batch=True)
        scene_id = content["scene_id"]
        refined_graph_path = resolve_graph_path(graphs_dir, scene_id)

        if refined_graph_path is None:
            print(f"  Skipping {scene_id} — no graph found")
            skipped += 1
            continue

        graph = json.loads(refined_graph_path.read_text())
        graph.pop("changes", None)

        # Skip scenes with no distractors
        if not graph.get("distractors"):
            print(f"  Skipping {scene_id} — no distractors")
            skipped += 1
            continue

        content_blocks = []
        if layout in ("v5", "v6"):
            content_blocks.extend(content["camera_blocks"])
            content_blocks.extend(content["bev_blocks"])
        elif layout == "v2":
            content_blocks.extend(content["bev_blocks"])
            content_blocks.extend(content["camera_blocks"])
        else:
            content_blocks.extend(content["camera_blocks"])
            content_blocks.append(content["bev_block"])
        filtered_state = filter_state_json(content["state_json"], graph.get("nodes", []))

        user_prompt = user_template.replace(
            "[paste scene graph JSON here]", json.dumps(graph, indent=2)
        )
        if filtered_state and "[paste filtered state JSON here]" in user_prompt:
            filtered_agent_labels = set(filtered_state.get("agents", {}).keys())
            filtered_state_nl = state_to_natural_language(content["state_json"], agents_filter=filtered_agent_labels)
            user_prompt = user_prompt.replace(
                "[paste filtered state JSON here]", filtered_state_nl
            )
        content_blocks.append(text_block(user_prompt, for_batch=True))

        if provider:
            requests.append(provider.format_batch_request(
                custom_id=_sanitize_id(f"{scene_id}_stage3c"),
                system_prompt=system_prompt,
                content_blocks=content_blocks,
            ))
        else:
            requests.append(create_batch_request(
                custom_id=_sanitize_id(f"{scene_id}_stage3c"),
                model=model,
                system_prompt=system_prompt,
                content_blocks=content_blocks,
                reasoning=reasoning,
            ))

    with open(batch_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"  Wrote {len(requests)} requests → {batch_file}  (skipped {skipped})")
    save_id_map(output_dir, dataset, "stage3c")
    return batch_file


# ── Stage 4: CF / AC Trajectories ────────────────────────────────────────

def _q_slug(question_text: str, max_len: int = 80) -> str:
    """Short filesystem-safe slug from a question string."""
    return (
        question_text[:max_len]
        .replace(" ", "_")
        .replace("?", "")
        .replace("/", "_")
        .replace("'", "")
        .replace('"', "")
        .strip("_")
    )


def _short_id(scene_id: str, prompt_type: str, q_idx: int) -> str:
    """Create a custom_id that fits within 64 chars (Claude batch limit).
    e.g. 'ns-0001_S0__cf__0' instead of full scene name + slug.
    """
    # Shorten scene_id: nuscenes-scene-0001_SAMPLED_0 → ns0001_S0
    short = scene_id
    short = short.replace("nuscenes-scene-", "ns")
    short = short.replace("openscene-scene-", "os")
    short = short.replace("argoverse2-scene-", "av")
    short = short.replace("_SAMPLED_", "_S")
    raw = f"{short}__{prompt_type}__{q_idx}"
    return _sanitize_id(raw)


def prepare_stage4_batch(records_dir: Path, graphs_dir: Path, qa_results_dir: Path,
                         dataset: str, model: str, reasoning: str, output_dir: Path,
                         prompt_type: str = "cf", max_samples: int = None,
                         provider=None, layout: str = "v1") -> Path:
    """Prepare Stage 4 CF or AC trajectory batch requests.

    prompt_type='cf' → uses Rung-3 CR questions from active_qa.json
    prompt_type='ac' → uses AC-Binary Yes questions from dormant_qa.json
    """
    label = "CF Trajectories"
    print(f"Preparing Stage 4cf: {label}")

    # Three prompt routes:
    # 1. entity_type=agent → nuPlan Config Gen (all 20 images, T=0 coords)
    # 2. entity_type=infrastructure, anchor=t0 → Intervention (T=0 images, future trajectory)
    # 3. entity_type=infrastructure, anchor=observation_window → CounterFactual (all 20 images, T=-1.5 coords)
    camera_vars = _AV2_CAMERA_VARS if dataset == "argoverse2" else _DEFAULT_CAMERA_VARS
    def _apply_vars(text):
        for key, value in camera_vars.items():
            text = text.replace(f"{{{key}}}", value)
        return text
    cf_t0_sys = _apply_vars((CF_PROMPTS / "Intervention_System_Prompt.md").read_text())
    cf_t0_usr = _apply_vars((CF_PROMPTS / "Intervention_User_Prompt.md").read_text())
    cf_obs_sys = _apply_vars((CF_PROMPTS / "CounterFactual_System_Prompt.md").read_text())
    cf_obs_usr = _apply_vars((CF_PROMPTS / "CounterFactual_User_Prompt.md").read_text())
    np_sys = _apply_vars((NUPLAN_PROMPTS / "system_prompt.md").read_text())
    np_usr = _apply_vars((NUPLAN_PROMPTS / "user_prompt.md").read_text())

    scene_dirs = find_scene_directories(records_dir, dataset)
    if max_samples:
        scene_dirs = scene_dirs[:max_samples]

    batch_file    = output_dir / f"{dataset}_stage4{prompt_type}.jsonl"
    manifest_file = output_dir / f"{dataset}_stage4{prompt_type}_manifest.json"
    requests, manifest, skipped = [], {}, 0

    for scene_dir in scene_dirs:
        content = prepare_scene_content(scene_dir, layout=layout, for_batch=True)
        scene_id = content["scene_id"]

        # Need the refined graph
        refined_graph_path = graphs_dir / f"{scene_id}_pruned.json"
        if not refined_graph_path.exists():
            print(f"  Skipping {scene_id} — no pruned graph")
            skipped += 1
            continue

        # Load questions with trajectory_changes=true from ALL QA sources
        qa_scene_dir = qa_results_dir / scene_id
        applicable = []
        for qa_file_name in ["active_qa.json", "dormant_qa.json", "distractor_qa.json"]:
            qa_file = qa_scene_dir / qa_file_name
            if not qa_file.exists():
                continue
            qa_data = json.loads(qa_file.read_text())
            for q in qa_data.get("questions", []):
                if not isinstance(q, dict):
                    continue
                cf = (q.get("meta") or {}).get("counterfactual_intervention")
                if cf and cf.get("trajectory_changes") == True:
                    applicable.append(q)

        if not applicable:
            print(f"  Skipping {scene_id} — no applicable {prompt_type.upper()} questions")
            skipped += 1
            continue

        # Load data once per scene
        graph = json.loads(refined_graph_path.read_text())
        graph.pop("changes", None)
        state_json = content["state_json"]
        nav_command = content["nav_command"]

        # Future trajectory (for t0 intervention)
        future_traj = state_json.get("ego", {}).get("future", [])
        future_formatted = format_trajectory_as_waypoints(future_traj)

        # Past trajectory (for observation_window — normalize at T=-1.5)
        ego_data = state_json.get("ego", {})
        history = ego_data.get("history", [])
        ego_speed = ego_data.get("speed", 0)
        # WP1=T=-1.5, WP2=T=-1.0, WP3=T=-0.5, WP4=T=0 (origin)
        obs_wps = []
        for h in history:
            obs_wps.append(f"[{h.get('x', 0)}, {h.get('y', 0)}, 0, {h.get('speed', 0)}]")
        obs_wps.append(f"[0, 0, 0, {ego_speed}]")
        obs_formatted = "\n".join(f"WP{i+1}: {wp}  (T={[-1.5, -1.0, -0.5, 0][i]}s)"
                                   for i, wp in enumerate(obs_wps))

        # Normalize state.json at T=-1.5 for observation_window prompts
        def normalize_state_at_t15(sj):
            """Re-center state.json coordinates at T=-1.5 ego position."""
            import copy, math
            sj = copy.deepcopy(sj)
            hist = sj.get("ego", {}).get("history", [])
            if not hist:
                return sj
            # T=-1.5 position in ego-T=0 frame
            ref = hist[0]
            rx, ry = ref.get("x", 0), ref.get("y", 0)
            # Shift everything so T=-1.5 is origin
            for h in hist:
                h["x"] = round(h["x"] - rx, 4)
                h["y"] = round(h["y"] - ry, 4)
            # Shift ego future
            for f in sj.get("ego", {}).get("future", []):
                f["x"] = round(f["x"] - rx, 4)
                f["y"] = round(f["y"] - ry, 4)
            # Shift agents
            for ag in sj.get("agents", {}).values():
                pos = ag.get("position", {})
                pos["x"] = round(pos.get("x", 0) - rx, 4)
                pos["y"] = round(pos.get("y", 0) - ry, 4)
                for h in ag.get("history", []):
                    h["x"] = round(h["x"] - rx, 4)
                    h["y"] = round(h["y"] - ry, 4)
            return sj

        # Build T=0 image blocks (for intervention prompts)
        from common.scene_utils import image_content_block as _icb
        tp0_dir = scene_dir / "Tp0p0"
        cam_list = ["cam_front", "cam_front_left", "cam_front_right", "cam_back"]
        t0_image_blocks = []
        for cam in cam_list:
            p = tp0_dir / f"{cam}.jpg"
            if p.exists():
                t0_image_blocks.append(_icb(str(p), f"{cam} at T=0", for_batch=True))
        bev_p = tp0_dir / "bev.png"
        if bev_p.exists():
            t0_image_blocks.append(_icb(str(bev_p), "BEV at T=0", for_batch=True))

        # Build all 20 image blocks (for observation_window prompts)
        all_image_blocks = []
        if layout in ("v5", "v6"):
            all_image_blocks = list(content["camera_blocks"]) + list(content["bev_blocks"])
        else:
            all_image_blocks = list(t0_image_blocks)

        traj_subdir = "cf_trajectories"

        # One batch request per question
        for q_idx, q in enumerate(applicable):
            question_text = q["question"]
            cf_answer     = str(q.get("correct_answer", ""))
            cf_reasoning  = str(q.get("reasoning", ""))
            slug          = _q_slug(question_text)
            custom_id     = _short_id(scene_id, "cf", q_idx)

            # Skip if result already exists
            result_path = qa_results_dir / scene_id / traj_subdir / slug / scene_id / "result.json"
            if result_path.exists():
                skipped += 1
                continue

            # Route by entity_type and temporal_anchor
            cf_meta = q.get("meta", {}).get("counterfactual_intervention", {})
            entity_type = cf_meta.get("target_entity_type", "infrastructure")
            anchor = cf_meta.get("temporal_anchor", "t0")

            if entity_type == "agent":
                # nuPlan Config Gen: agent intervention → planner re-simulation
                sys_prompt = np_sys
                usr_template = np_usr
                use_state = state_json
                use_images = all_image_blocks
                cf_metadata_json = json.dumps(cf_meta, indent=2)
                user_prompt = usr_template
                user_prompt = user_prompt.replace("{state_json}", json.dumps(use_state, indent=2))
                user_prompt = user_prompt.replace("{scene_graph_json}", json.dumps(graph, indent=2))
                user_prompt = user_prompt.replace("{cf_question}", question_text)
                user_prompt = user_prompt.replace("{cf_answer}", cf_answer)
                user_prompt = user_prompt.replace("{cf_reasoning}", cf_reasoning)
                user_prompt = user_prompt.replace("{cf_metadata_json}", cf_metadata_json)

            elif anchor == "observation_window":
                # Past intervention: all 20 images, obs trajectory, coords at T=-1.5
                sys_prompt = cf_obs_sys
                usr_template = cf_obs_usr
                use_state = normalize_state_at_t15(state_json)
                use_images = all_image_blocks
                user_prompt = usr_template
                user_prompt = user_prompt.replace("{state_json}", json.dumps(use_state, indent=2))
                user_prompt = user_prompt.replace("{causal_graph_json}", json.dumps(graph, indent=2))
                user_prompt = user_prompt.replace("{nav_command}", nav_command)
                user_prompt = user_prompt.replace("{cf_question}", question_text)
                user_prompt = user_prompt.replace("{cf_answer}", cf_answer)
                user_prompt = user_prompt.replace("{cf_reasoning}", cf_reasoning)
                # Inject obs trajectory
                wp_pattern = re.compile(
                    r'(=== ORIGINAL EGO TRAJECTORY.*?\n)(WP1:.*?WP\d:.*?\n)', re.DOTALL
                )
                if wp_pattern.search(user_prompt):
                    user_prompt = wp_pattern.sub(
                        lambda m: m.group(1) + obs_formatted + "\n", user_prompt
                    )

            else:
                # T=0 infrastructure intervention: T=0 images only, future trajectory
                sys_prompt = cf_t0_sys
                usr_template = cf_t0_usr
                use_state = state_json
                use_images = t0_image_blocks
                user_prompt = usr_template
                user_prompt = user_prompt.replace("{state_json}", json.dumps(use_state, indent=2))
                user_prompt = user_prompt.replace("{causal_graph_json}", json.dumps(graph, indent=2))
                user_prompt = user_prompt.replace("{nav_command}", nav_command)
                user_prompt = user_prompt.replace("{cf_question}", question_text)
                user_prompt = user_prompt.replace("{cf_answer}", cf_answer)
                user_prompt = user_prompt.replace("{cf_reasoning}", cf_reasoning)
                # Inject future trajectory
                wp_pattern = re.compile(
                    r'(=== ORIGINAL EGO TRAJECTORY.*?\n)(WP1:.*?WP\d:.*?\n)', re.DOTALL
                )
                if wp_pattern.search(user_prompt):
                    user_prompt = wp_pattern.sub(
                        lambda m: m.group(1) + future_formatted + "\n", user_prompt
                    )

            content_blocks = list(use_images) + [text_block(user_prompt, for_batch=True)]

            if provider:
                requests.append(provider.format_batch_request(
                    custom_id=custom_id,
                    system_prompt=sys_prompt,
                    content_blocks=content_blocks,
                ))
            else:
                requests.append(create_batch_request(
                    custom_id=custom_id,
                    model=model,
                    system_prompt=sys_prompt,
                    content_blocks=content_blocks,
                    reasoning=reasoning,
                ))

            # Manifest entry so process_results can reconstruct full context
            route = "nuplan_config" if entity_type == "agent" else (
                "llm_obs_window" if anchor == "observation_window" else "llm_t0")
            manifest[custom_id] = {
                "scene_id":        scene_id,
                "scene_dir":       str(scene_dir),
                "route":           route,
                "entity_type":     entity_type,
                "entity_id":       cf_meta.get("target_entity_id", ""),
                "temporal_anchor": anchor,
                "question":        question_text,
                "cf_answer":       cf_answer,
                "cf_reasoning":    cf_reasoning,
                "slug":            slug,
            }

        print(f"  {scene_id}: {len(applicable)} {prompt_type.upper()} question(s)")

    with open(batch_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Wrote {len(requests)} requests → {batch_file}  (skipped {skipped})")
    print(f"  Manifest          → {manifest_file}")
    save_id_map(output_dir, dataset, "stage4")
    return batch_file


# ── Stage 4np: nuPlan Config Generation ───────────────────────────────────

def prepare_stage4_nuplan_batch(records_dir: Path, graphs_dir: Path, qa_results_dir: Path,
                                dataset: str, model: str, reasoning: str, output_dir: Path,
                                max_samples: int = None, provider=None, layout: str = "v1") -> Path:
    """Prepare Stage 4np: nuPlan intervention config batch requests.

    Filters for agent-to-ego interventions with trajectory_changes=true only.
    Sources: CR (active_qa.json) and AC-Binary Yes (dormant_qa.json).
    """
    print("Preparing Stage 4np: nuPlan Config Generation")

    camera_vars = _AV2_CAMERA_VARS if dataset == "argoverse2" else _DEFAULT_CAMERA_VARS
    def _apply_vars(text):
        for key, value in camera_vars.items():
            text = text.replace(f"{{{key}}}", value)
        return text
    system_prompt = _apply_vars((NUPLAN_PROMPTS / "system_prompt.md").read_text())
    user_template = _apply_vars((NUPLAN_PROMPTS / "user_prompt.md").read_text())

    scene_dirs = find_scene_directories(records_dir, dataset)
    if max_samples:
        scene_dirs = scene_dirs[:max_samples]

    batch_file    = output_dir / f"{dataset}_stage4np.jsonl"
    manifest_file = output_dir / f"{dataset}_stage4np_manifest.json"
    requests, manifest, skipped = [], {}, 0

    for scene_dir in scene_dirs:
        content = prepare_scene_content(scene_dir, layout=layout, for_batch=True)
        scene_id = content["scene_id"]

        refined_graph_path = graphs_dir / f"{scene_id}_pruned.json"
        if not refined_graph_path.exists():
            print(f"  Skipping {scene_id} — no pruned graph")
            skipped += 1
            continue

        qa_scene_dir = qa_results_dir / scene_id
        graph = json.loads(refined_graph_path.read_text())
        graph.pop("changes", None)
        state_json = content["state_json"]
        nav_command = content["nav_command"]

        # Collect agent interventions from CR and AC
        agent_questions = []

        # CR from active_qa.json
        active_f = qa_scene_dir / "active_qa.json"
        if active_f.exists():
            for q in json.loads(active_f.read_text()).get("questions", []):
                if not isinstance(q, dict): continue
                if q.get("question_type") != "CR": continue
                meta = q.get("meta") or {}
                cr = meta.get("cr_intervention") or {}
                if cr.get("entity_type") == "agent" and cr.get("trajectory_changes") == True:
                    agent_questions.append(("CR", q, meta))

        # AC-Binary Yes from dormant_qa.json
        dl_f = qa_scene_dir / "dormant_qa.json"
        if dl_f.exists():
            for q in json.loads(dl_f.read_text()).get("questions", []):
                if not isinstance(q, dict): continue
                subtype = q.get("subtype", q.get("question_type", ""))
                if subtype != "AC-Binary": continue
                if q.get("correct_answer", "").strip().lower() != "yes": continue
                meta = q.get("meta") or {}
                ac = meta.get("ac_intervention") or {}
                eid = ac.get("entity_id", "")
                is_infra_id = eid.startswith("n") and (len(eid) < 4 or eid[1:].isdigit() or eid[1] == "_")
                if ac.get("entity_type") == "agent" and ac.get("trajectory_changes") == True and not is_infra_id:
                    agent_questions.append(("AC", q, meta))

        if not agent_questions:
            print(f"  Skipping {scene_id} — no agent interventions with trajectory_changes=true")
            skipped += 1
            continue

        # Build image blocks (shared across all questions for this scene)
        tp0_dir = scene_dir / "Tp0p0"
        from common.scene_utils import image_content_block as _icb

        image_blocks = []
        # 20 images: 4 cams + BEV per timestep (timestep-major)
        timestep_slugs = [("Tm1p5", "t=-1.5"), ("Tm1p0", "t=-1.0"), ("Tm0p5", "t=-0.5"), ("Tp0p0", "t=0")]
        cam_list = ["cam_front", "cam_front_left", "cam_front_right", "cam_back"]
        cam_labels = {"cam_front": "CAM_FRONT", "cam_front_left": "CAM_FRONT_LEFT",
                      "cam_front_right": "CAM_FRONT_RIGHT", "cam_back": "CAM_BACK"}
        for slug, t_label in timestep_slugs:
            slug_dir = scene_dir / slug
            for cam in cam_list:
                p = slug_dir / f"{cam}.jpg"
                if p.exists():
                    image_blocks.append(_icb(str(p), f"{cam_labels[cam]} at {t_label}", for_batch=True))
            bev_p = slug_dir / "bev.png"
            if bev_p.exists():
                image_blocks.append(_icb(str(bev_p), f"BEV at {t_label}", for_batch=True))

        # One request per agent question
        for q_idx, (q_type, q, meta) in enumerate(agent_questions):
            intervention = meta.get("cr_intervention") or meta.get("ac_intervention") or {}
            question_text = q["question"]
            cf_answer = str(q.get("correct_answer", ""))
            cf_reasoning = str(q.get("reasoning", ""))
            custom_id = _short_id(scene_id, "np", q_idx)

            # Fill user prompt
            user_prompt = user_template
            user_prompt = user_prompt.replace("{cf_question}", question_text)
            user_prompt = user_prompt.replace("{cf_answer}", cf_answer)
            user_prompt = user_prompt.replace("{cf_reasoning}", cf_reasoning)
            user_prompt = user_prompt.replace("{cf_metadata_json}", json.dumps(intervention, indent=2))
            user_prompt = user_prompt.replace("{state_json}", json.dumps(state_json, indent=2))
            user_prompt = user_prompt.replace("{scene_graph_json}", json.dumps(graph, indent=2))

            content_blocks = list(image_blocks) + [text_block(user_prompt, for_batch=True)]

            if provider:
                requests.append(provider.format_batch_request(
                    custom_id=custom_id,
                    system_prompt=system_prompt,
                    content_blocks=content_blocks,
                ))
            else:
                requests.append(create_batch_request(
                    custom_id=custom_id,
                    model=model,
                    system_prompt=system_prompt,
                    content_blocks=content_blocks,
                    reasoning=reasoning,
                ))

            manifest[custom_id] = {
                "scene_id": scene_id,
                "scene_dir": str(scene_dir),
                "q_type": q_type,
                "entity_id": intervention.get("entity_id", ""),
                "intervention_type": intervention.get("intervention", intervention.get("action", "activate")),
                "question": question_text,
                "cf_answer": cf_answer,
            }

        print(f"  {scene_id}: {len(agent_questions)} agent intervention(s)")

    with open(batch_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Wrote {len(requests)} requests → {batch_file}  (skipped {skipped})")
    print(f"  Manifest          → {manifest_file}")
    save_id_map(output_dir, dataset, "stage4np")
    return batch_file


# ── Stage 5: Agent Intervention LLM Ego Trajectory ───────────────────────

def _fill_template(template, replacements):
    """Fill {{PLACEHOLDER}} and {{#if BLOCK}}...{{/if}} conditionals."""
    for key, val in replacements.items():
        template = template.replace(f"{{{{{key}}}}}", str(val))
    action = replacements.get("ACTION_TYPE", "").upper()
    for atype in ["REMOVE", "REPOSITION", "SPEED_CHANGE"]:
        tag_open = "{{#if " + atype + "}}"
        tag_close = "{{/if}}"
        if atype == action:
            template = template.replace(tag_open, "").replace(tag_close, "")
        else:
            while tag_open in template:
                start = template.find(tag_open)
                if start == -1: break
                end = template.find(tag_close, start)
                if end == -1: break
                template = template[:start] + template[end + len(tag_close):]
    return template


def prepare_stage5_batch(records_dir: Path, graphs_dir: Path, qa_results_dir: Path,
                         dataset: str, model: str, reasoning: str, output_dir: Path,
                         max_samples: int = None, provider=None, layout: str = "v1") -> Path:
    """Prepare Stage 5: Agent Intervention LLM Ego Trajectory.

    Routes by apply_at_t:
      apply_at_t == 0.0  → T=0 prompts (1 modified BEV, 8 future waypoints)
      apply_at_t < 0     → Past prompts (4 modified BEVs, 4 past waypoints)

    Camera ordering in IMAGE_ORDER_BLOCK is built dynamically from detected cameras,
    so this works for nuscenes (4 cams) and argoverse2 (5 cams) without separate av2 prompts.
    """
    print("Preparing Stage 5: Agent Intervention LLM Ego Trajectories")

    t0_sys            = (AGENT_INT_PROMPTS / "CF_Ego_Trajectory_T0_System_Prompt.md").read_text()
    t0_usr_template   = (AGENT_INT_PROMPTS / "CF_Ego_Trajectory_T0_User_Prompt.md").read_text()
    past_sys          = (AGENT_INT_PROMPTS / "CF_Ego_Trajectory_Past_System_Prompt.md").read_text()
    past_usr_template = (AGENT_INT_PROMPTS / "CF_Ego_Trajectory_Past_User_Prompt.md").read_text()

    scene_dirs = find_scene_directories(records_dir, dataset)
    if max_samples:
        scene_dirs = scene_dirs[:max_samples]

    batch_file    = output_dir / f"{dataset}_stage5_ego_traj.jsonl"
    manifest_file = output_dir / f"{dataset}_stage5_ego_traj_manifest.json"
    requests, manifest, skipped = [], {}, 0

    for scene_dir in scene_dirs:
        content     = prepare_scene_content(scene_dir, layout=layout, for_batch=True)
        scene_id    = content["scene_id"]
        state_json  = content["state_json"]
        nav_command = content["nav_command"]

        cf_base = qa_results_dir / scene_id / "cf_trajectories"
        if not cf_base.exists():
            continue

        # Detect camera count from record — drives IMAGE_ORDER_BLOCK
        n_cams    = 5 if (scene_dir / "Tp0p0" / "cam_back_right.jpg").exists() else 4
        cam_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
        cam_names += ["CAM_BACK_RIGHT", "CAM_BACK_LEFT"] if n_cams == 5 else ["CAM_BACK"]

        q_idx = 0
        for q_dir in sorted(cf_base.iterdir()):
            if not q_dir.is_dir():
                continue

            scene_subdir = q_dir / scene_id if (q_dir / scene_id).exists() else q_dir
            result_path  = scene_subdir / "result.json"
            if not result_path.exists():
                continue

            result = json.loads(result_path.read_text())
            gen    = result.get("generator", {})

            # Only agent interventions (infrastructure handled by stage 4cf)
            clean_config = {k: v for k, v in gen.items()
                            if isinstance(v, dict) and v.get("action") in ("remove", "reposition", "speed_change")}
            if not clean_config:
                continue

            bev_int = scene_subdir / "bev_intervention.png"
            if not bev_int.exists():
                skipped += 1
                continue
            if (scene_subdir / "llm_ego_trajectory.json").exists():
                skipped += 1
                continue

            first_config = list(clean_config.values())[0]
            apply_at_t   = first_config.get("apply_at_t", 0.0)
            is_past      = apply_at_t < 0
            action_type  = first_config["action"]
            question     = result.get("question", "")
            custom_id    = _short_id(scene_id, "s5", q_idx)
            q_idx += 1

            # Original scene images
            if layout in ("v5", "v6"):
                img_blocks = list(content["camera_blocks"]) + list(content["bev_blocks"])
            else:
                img_blocks = list(content["camera_blocks"])
                if content.get("bev_block"):
                    img_blocks.append(content["bev_block"])

            if is_past:
                sys_prompt   = past_sys
                usr_template = past_usr_template
                for slug in ["Tm1p5", "Tm1p0", "Tm0p5", "Tp0p0"]:
                    mbev = scene_subdir / f"bev_intervention_{slug}.png"
                    if not mbev.exists():
                        mbev = bev_int
                    img_blocks.append(image_content_block(str(mbev),
                                      label=f"MODIFIED BEV {slug}", for_batch=True))
            else:
                sys_prompt   = t0_sys
                usr_template = t0_usr_template
                img_blocks.append(image_content_block(str(bev_int),
                                  label="MODIFIED BEV at T=0", for_batch=True))

            # Build image order dynamically (nuscenes=4 cams, argoverse2=5 cams)
            order_lines, idx = [], 1
            for ts in ["T=-1.5", "T=-1.0", "T=-0.5", "T=0"]:
                for cam in cam_names:
                    order_lines.append(f"Image {idx}: {cam} at {ts}")
                    idx += 1
                order_lines.append(f"Image {idx}: BEV at {ts}")
                idx += 1

            user_text = _fill_template(usr_template, {
                "IMAGE_ORDER_BLOCK":   "\n".join(order_lines),
                "NAV_COMMAND":         nav_command,
                "STATE_JSON":          json.dumps(state_json, indent=2),
                "QUESTION":            question,
                "INTERVENTION_CONFIG": json.dumps(clean_config, indent=2),
                "ACTION_TYPE":         action_type.upper(),
            })

            content_blocks = img_blocks + [text_block(user_text, for_batch=True)]

            if provider:
                requests.append(provider.format_batch_request(
                    custom_id=custom_id, system_prompt=sys_prompt,
                    content_blocks=content_blocks,
                ))
            else:
                requests.append(create_batch_request(
                    custom_id=custom_id, model=model,
                    system_prompt=sys_prompt,
                    content_blocks=content_blocks, reasoning=reasoning,
                ))

            manifest[custom_id] = {
                "scene_id":   scene_id,  "scene_dir":   str(scene_dir),
                "cf_subdir":  str(scene_subdir),
                "apply_at_t": apply_at_t, "action_type": action_type,
                "question":   question,   "slug":        q_dir.name,
            }

    with open(batch_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote {len(requests)} requests → {batch_file}  (skipped {skipped})")
    save_id_map(output_dir, dataset, "stage5")
    return batch_file


# ── Stage verify: Graph Verification ─────────────────────────────────────

def _condense_graph_for_verification(graph: dict) -> dict:
    """Return a token-minimised copy of a pruned graph for the verify stage.

    Strips fields that Haiku does not need for classification auditing:
    `mechanism`, `affected_state` (verbose), deep `motion` sub-dict, `_usage`.
    Keeps everything required to audit: id, type, category, label, state,
    position_relative_to_ego, visible_in, causal_status, effect_type, active,
    justification, rejection_reason.
    """
    import copy
    g = copy.deepcopy(graph)
    g.pop("_usage", None)
    g.pop("changes", None)
    for node in g.get("nodes", []):
        mot = node.get("motion", {})
        node["motion"] = {"is_static": mot.get("is_static", True)} if isinstance(mot, dict) else {}
    for edge in g.get("ego_edges", []) + g.get("inter_node_edges", []):
        edge.pop("affected_state", None)
        edge.pop("mechanism", None)
    return g


def prepare_stage_verify_batch(
    records_dir: Path,
    graphs_dir: Path,
    dataset: str,
    output_dir: Path,
    max_samples: int = None,
    provider=None,
    force: bool = False,
) -> Path:
    """Prepare Stage verify: Graph Verification batch requests.

    Sends a condensed pruned graph + cameras and BEV across all 4 timesteps
    (T=-1.5, T=-1.0, T=-0.5, T=0) to verify node/edge/distractor classifications.

    Output verification JSONs land flat in graphs_dir alongside *_pruned.json:
        {graphs_dir}/{scene_id}_verification.json

    Args:
        records_dir:  Path to records root directory.
        graphs_dir:   Path to outputs directory (contains *_pruned.json files
                      and receives *_verification.json outputs).
        dataset:      "nuscenes" | "openscene" | "argoverse2".
        output_dir:   Batch JSONL output directory (batch_jobs).
        max_samples:   Limit to first N scenes.
        provider:     LLMProvider instance — should be ClaudeProvider (Haiku).

    Returns:
        Path to the written JSONL file.
    """
    print("Preparing Stage verify: Graph Verification")
    system_prompt = (VERIFY_PROMPTS / "system_prompt.md").read_text()

    scene_dirs = find_scene_directories(records_dir, dataset)
    if max_samples:
        scene_dirs = scene_dirs[:max_samples]
    print(f"  Found {len(scene_dirs)} scene samples")

    batch_file = output_dir / f"{dataset}_stage_verify.jsonl"
    requests, skipped = [], 0

    # Rear camera names differ between datasets
    base_cams = ["cam_front", "cam_front_left", "cam_front_right"]
    rear_options = ["cam_back_right", "cam_back_left", "cam_back"]  # AV2 first, then nuScenes
    cam_label = {
        "cam_front":       "CAM_FRONT",
        "cam_front_left":  "CAM_FRONT_LEFT",
        "cam_front_right": "CAM_FRONT_RIGHT",
        "cam_back":        "CAM_BACK",
        "cam_back_right":  "CAM_BACK_RIGHT",
        "cam_back_left":   "CAM_BACK_LEFT",
    }

    for scene_dir in scene_dirs:
        content  = prepare_scene_content(scene_dir, layout="v1", for_batch=True)
        scene_id = content["scene_id"]

        # Idempotency: skip if already verified (unless --force)
        if not force and (graphs_dir / f"{scene_id}_verification.json").exists():
            skipped += 1
            continue

        # Need a graph (verified > edited > pruned)
        graph_path = resolve_graph_path(graphs_dir, scene_id)
        if graph_path is None:
            print(f"  Skipping {scene_id} — no graph found")
            skipped += 1
            continue

        graph = json.loads(graph_path.read_text())
        condensed = _condense_graph_for_verification(graph)

        tp0_dir = scene_dir / "Tp0p0"
        state_path = tp0_dir / "state.json"
        if not state_path.exists():
            print(f"  Skipping {scene_id} — missing Tp0p0/state.json")
            skipped += 1
            continue

        state_json = json.loads(state_path.read_text())

        desc_path = tp0_dir / "state_desc.json"
        state_nl = (
            json.loads(desc_path.read_text()).get("description", "")
            if desc_path.exists()
            else state_to_natural_language(state_json)
        )

        # Build multi-timestep image content blocks (cameras + BEV at each of 4 timesteps)
        timestep_slugs = [("Tm1p5", "T=-1.5"), ("Tm1p0", "T=-1.0"), ("Tm0p5", "T=-0.5"), ("Tp0p0", "T=0")]
        content_blocks = []
        missing_mandatory = False

        for slug, t_label in timestep_slugs:
            t_dir = scene_dir / slug
            is_t0 = (slug == "Tp0p0")

            for cam in base_cams:
                p = t_dir / f"{cam}.jpg"
                if not p.exists():
                    if is_t0:
                        print(f"  Skipping {scene_id} — missing T=0 {cam}.jpg")
                        missing_mandatory = True
                    continue
                content_blocks.append(image_content_block(str(p), label=f"{cam_label[cam]} ({t_label})", for_batch=True))
            if missing_mandatory:
                break

            rear_present = [c for c in rear_options if (t_dir / f"{c}.jpg").exists()]
            for cam in rear_present:
                content_blocks.append(image_content_block(
                    str(t_dir / f"{cam}.jpg"), label=f"{cam_label[cam]} ({t_label})", for_batch=True
                ))

            bev_path = t_dir / "bev.png"
            if bev_path.exists():
                content_blocks.append(image_content_block(str(bev_path), label=f"BEV ({t_label})", for_batch=True))

        if missing_mandatory:
            skipped += 1
            continue

        # User prompt: scene context + state_desc + condensed graph
        nav_command = state_json.get("nav_command", "UNKNOWN")
        n_nodes = len([n for n in condensed.get("nodes", []) if n.get("id") != "ego"])
        n_edges = len(condensed.get("ego_edges", [])) + len(condensed.get("inter_node_edges", []))
        n_dists = len(condensed.get("distractors", []))

        user_prompt = (
            f"Navigation command: {nav_command}\n\n"
            f"Scene state description (T=0):\n{state_nl}\n\n"
            f"=== CAUSAL GRAPH TO VERIFY ===\n"
            f"Graph summary: {n_nodes} non-ego node(s), {n_edges} edge(s), {n_dists} distractor(s).\n\n"
            f"```json\n{json.dumps(condensed, indent=2)}\n```\n\n"
            f"Verify all node classifications, edge causal_status/effect_type assignments, and "
            f"distractor rejection_reasons against the images above. Return ONLY the verification JSON."
        )
        content_blocks.append(text_block(user_prompt, for_batch=True))

        custom_id = _sanitize_id(f"{scene_id}_verify")

        if provider:
            requests.append(provider.format_batch_request(
                custom_id=custom_id,
                system_prompt=system_prompt,
                content_blocks=content_blocks,
                max_tokens=4096,
            ))
        else:
            requests.append(create_batch_request(
                custom_id=custom_id,
                model="claude-haiku-4-5-20251001",
                system_prompt=system_prompt,
                content_blocks=content_blocks,
            ))

    with open(batch_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"  Wrote {len(requests)} requests → {batch_file}  (skipped {skipped})")
    save_id_map(output_dir, dataset, "verify")
    return batch_file


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    # Pre-parse dataset to load config defaults before building full parser
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--dataset", choices=["nuscenes", "openscene", "argoverse2"])
    _known, _ = _pre.parse_known_args()
    _cfg     = load_config(_known.dataset) if _known.dataset else {}
    _top_cfg = load_config() or {}
    _batch   = _top_cfg.get("batch", {})

    parser = argparse.ArgumentParser(description="Prepare batch requests for pipeline stages")
    parser.add_argument("--stage", required=True,
                        choices=["1", "2", "3a", "3b", "3c", "4cf", "4ac", "4np", "5", "verify"],
                        help="Pipeline stage to prepare")
    parser.add_argument("--dataset", required=True,
                        choices=["nuscenes", "openscene", "argoverse2"])
    parser.add_argument("--records",    default=_cfg.get("records"), help="Path to records directory")
    parser.add_argument("--graphs",     default=_cfg.get("outputs"), help="Path to graphs directory (for stages 2+)")
    parser.add_argument("--qa-results", help="Path to QA results directory (for stages 4cf/4ac)")
    parser.add_argument("--output",     default=_batch.get("batch_dir", "batch_jobs"), help="Output directory")
    parser.add_argument("--model",      default=_top_cfg.get("model", "gpt-5.2"))
    parser.add_argument("--reasoning",  default=_top_cfg.get("reasoning", "none"))
    parser.add_argument("--prompt-version", default=_batch.get("qa_prompt_version", "v5"))
    parser.add_argument("--max-samples",    type=int)
    parser.add_argument("--provider",       choices=["openai", "claude", "gemini"],
                        default=_top_cfg.get("provider"), help="LLM provider")
    parser.add_argument("--layout",         choices=["v1", "v3", "v4", "v5", "v6"],
                        default=_batch.get("graph_prompt_version", "v5"),
                        help="Graph prompt version for stage 1")
    parser.add_argument("--force",          action="store_true",
                        help="Re-process scenes that already have output files (skips idempotency check)")
    parser.add_argument("--filter",         default=None,
                        help="Path to diverse_subset.json manifest — only process selected scenes")
    args = parser.parse_args()
    if not args.records:
        parser.error("--records is required (or set in config.yaml)")

    records_dir = Path(args.records)
    output_dir  = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Create provider instance if specified
    provider = None
    if args.provider:
        provider = get_provider(args.provider, model=args.model, reasoning=args.reasoning)

    # Load optional scene filter
    filter_set = load_filter_set(args.filter)

    if args.stage == "1":
        graphs_dir = Path(args.graphs) if args.graphs else None
        prepare_stage1_batch(records_dir, args.dataset, args.model,
                             args.reasoning, output_dir, args.max_samples,
                             graphs_dir=graphs_dir,
                             provider=provider, layout=args.layout)

    elif args.stage == "2":
        if not args.graphs:
            print("ERROR: --graphs required for stage 2"); sys.exit(1)
        prepare_stage2_batch(records_dir, Path(args.graphs), args.dataset,
                             args.model, args.reasoning, output_dir, args.max_samples,
                             provider=provider, layout=args.layout)

    elif args.stage == "3a":
        if not args.graphs:
            print("ERROR: --graphs required for stage 3a"); sys.exit(1)
        prepare_stage3a_batch(records_dir, Path(args.graphs), args.dataset,
                              args.model, args.reasoning, output_dir,
                              args.prompt_version, args.max_samples,
                              provider=provider, layout=args.layout,
                              filter_set=filter_set)

    elif args.stage == "3b":
        if not args.graphs:
            print("ERROR: --graphs required for stage 3b"); sys.exit(1)
        prepare_stage3b_batch(records_dir, Path(args.graphs), args.dataset,
                              args.model, args.reasoning, output_dir,
                              args.prompt_version, args.max_samples,
                              provider=provider, layout=args.layout,
                              filter_set=filter_set)

    elif args.stage == "3c":
        if not args.graphs:
            print("ERROR: --graphs required for stage 3c"); sys.exit(1)
        prepare_stage3c_batch(records_dir, Path(args.graphs), args.dataset,
                              args.model, args.reasoning, output_dir,
                              args.prompt_version, args.max_samples,
                              provider=provider, layout=args.layout,
                              filter_set=filter_set)

    elif args.stage in ("4cf", "4ac"):
        if not args.graphs:
            print(f"ERROR: --graphs required for stage {args.stage}"); sys.exit(1)
        if not args.qa_results:
            print(f"ERROR: --qa-results required for stage {args.stage}"); sys.exit(1)
        prompt_type = args.stage[1:]  # "cf" or "ac"
        prepare_stage4_batch(records_dir, Path(args.graphs), Path(args.qa_results),
                             args.dataset, args.model, args.reasoning, output_dir,
                             prompt_type, args.max_samples,
                             provider=provider, layout=args.layout)

    elif args.stage == "4np":
        if not args.graphs:
            print("ERROR: --graphs required for stage 4np"); sys.exit(1)
        if not args.qa_results:
            print("ERROR: --qa-results required for stage 4np"); sys.exit(1)
        prepare_stage4_nuplan_batch(records_dir, Path(args.graphs), Path(args.qa_results),
                                    args.dataset, args.model, args.reasoning, output_dir,
                                    args.max_samples, provider=provider, layout=args.layout)

    elif args.stage == "5":
        if not args.graphs:
            print("ERROR: --graphs required for stage 5"); sys.exit(1)
        if not args.qa_results:
            print("ERROR: --qa-results required for stage 5"); sys.exit(1)
        prepare_stage5_batch(records_dir, Path(args.graphs), Path(args.qa_results),
                             args.dataset, args.model, args.reasoning, output_dir,
                             args.max_samples, provider=provider, layout=args.layout)

    elif args.stage == "verify":
        if not args.graphs:
            print("ERROR: --graphs required for stage verify"); sys.exit(1)
        prepare_stage_verify_batch(
            records_dir=records_dir,
            graphs_dir=Path(args.graphs),
            dataset=args.dataset,
            output_dir=output_dir,
            max_samples=args.max_samples,
            provider=provider,
            force=args.force,
        )


if __name__ == "__main__":
    main()
