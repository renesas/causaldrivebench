"""
v3_pipeline.py — LangGraph two-stage causal graph pipeline.

Graph:  Generator  →  Discriminator/Fixer
        (Stage 1)      (Stage 2)

Shared state flows between nodes carrying BEV, state.json, and the graph.
generate_causal_graph.py is NOT modified.

Usage (causal_graph conda env):
  python v3_pipeline.py \
      --records ../STSBench/scenario_records \
      --model gpt-5.2 \
      --reasoning high

  # Stage 2 only (reuse existing raw graphs):
  python v3_pipeline.py \
      --records ../STSBench/scenario_records \
      --skip-stage1 --force
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, TypedDict

import cv2
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

load_dotenv()

# Provider instance — set once in main(), used by generator/discriminator nodes
_provider = None

# ── Constants ────────────────────────────────────────────────────────────

PROMPT_CAMS = ["cam_front", "cam_front_left", "cam_front_right", "cam_back"]
CAM_DISPLAY = {
    "cam_front": "CAM_FRONT",
    "cam_front_left": "CAM_FRONT_LEFT",
    "cam_front_right": "CAM_FRONT_RIGHT",
    "cam_back": "CAM_BACK",
}


# ── Shared State ─────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    """State that flows through the LangGraph pipeline."""
    # Inputs (set once, shared by all nodes)
    scene_dir: str
    scene_name: str
    bev_b64: str            # base64-encoded BEV image
    state_json: dict        # parsed state.json
    state_text: str          # state.json as formatted text
    nav_command: str
    camera_blocks: list     # content blocks for 16 camera images
    # Prompts
    generator_prompt: str
    refiner_prompt: str
    # Model config
    model: str
    s1_reasoning: str
    s2_reasoning: str
    # Outputs (populated by nodes)
    raw_graph: dict | None
    refined_graph: dict | None
    reasoning_trace: str
    # Token tracking
    s1_usage: dict
    s2_usage: dict
    error: str | None


# ── Helpers ──────────────────────────────────────────────────────────────

def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def image_content_block(path: str, label: str | None = None,
                        flip_horizontal: bool = False) -> dict:
    if label or flip_horizontal:
        img = cv2.imread(str(path))
        if img is not None:
            if flip_horizontal:
                img = cv2.flip(img, 1)
            if label:
                cv2.putText(img, label, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            _, buf = cv2.imencode(".jpg", img)
            b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")
            return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/png")
    b64 = load_image_b64(path)
    return {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}


def parse_slug_to_seconds(slug: str) -> float | None:
    m = re.match(r"T([mp])(\d+)p(\d+)", slug)
    if not m:
        return None
    sign = -1 if m.group(1) == "m" else 1
    return sign * (int(m.group(2)) + int(m.group(3)) / 10.0)


def find_temporal_frames(scene_dir: Path) -> list:
    tp0 = scene_dir / "Tp0p0"
    if not tp0.exists():
        return []
    timesteps = []
    for d in scene_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("T"):
            continue
        t = parse_slug_to_seconds(d.name)
        if t is not None:
            timesteps.append((d.name, t, d))
    timesteps.sort(key=lambda x: x[1])
    history = [(s, t, d) for s, t, d in timesteps if t < 0]
    t0 = [(s, t, d) for s, t, d in timesteps if t == 0.0]
    if not t0:
        return []
    recent_hist = history[-3:] if len(history) >= 3 else history
    padding = [None] * (3 - len(recent_hist))
    return padding + recent_hist + t0


def extract_json(raw: str) -> dict:
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        json_lines, inside = [], False
        for line in lines:
            if line.strip().startswith("```") and not inside:
                inside = True
                continue
            elif line.strip() == "```" and inside:
                inside = False
                continue
            if inside:
                json_lines.append(line)
        raw = "\n".join(json_lines).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON object/array from anywhere in the response
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = raw.find(start_char)
            end = raw.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end+1])
                except json.JSONDecodeError:
                    pass
        return {"_raw_response": raw}


# ── LangGraph Nodes ──────────────────────────────────────────────────────

def generator_node(state: PipelineState) -> dict:
    """
    Stage 1 — Generator.
    Sends 16 camera images + BEV + state.json to GPT with the system prompt.
    Produces the raw causal graph.
    """
    print(f"  [Generator] Building content blocks ...")
    scene_dir = Path(state["scene_dir"])

    # Camera images (16 frames: 4 cameras × 4 timesteps)
    content_blocks = list(state["camera_blocks"])  # already built

    # BEV image
    bev_path = scene_dir / "Tp0p0" / "bev.png"
    if bev_path.exists():
        content_blocks.append(image_content_block(str(bev_path)))
    else:
        content_blocks.append({"type": "input_text", "text": "[IMAGE MISSING: BEV at t0]"})

    # state.json as text
    content_blocks.append({
        "type": "input_text",
        "text": f"state.json:\n```json\n{state['state_text']}\n```"
    })

    # System prompt with nav_command
    system_prompt = state["generator_prompt"].replace("{nav_command}", state["nav_command"])

    n_images = sum(1 for b in content_blocks if b.get("type") in ("input_image", "image"))
    reasoning = state["s1_reasoning"]
    model = state["model"]
    print(f"  [Generator] Calling {model}  ({n_images} images + state.json)  reasoning={reasoning} ...")

    raw, usage_data = _provider.call(system_prompt, content_blocks, max_tokens=65536)
    result = extract_json(raw)
    print(f"  [Generator] Tokens: in={usage_data.get('input','?')} (image={usage_data.get('image',0)}) out={usage_data.get('output','?')}")

    if "_raw_response" in result:
        return {"raw_graph": None, "s1_usage": usage_data, "error": "Stage 1 returned invalid JSON"}

    return {"raw_graph": result, "s1_usage": usage_data, "error": None}


def discriminator_node(state: PipelineState) -> dict:
    """
    Stage 2 — Discriminator / Fixer.
    Receives the SAME context as Generator (16 cameras + BEV + state.json)
    plus the raw graph. Cross-checks and fixes.
    """
    raw_graph = state["raw_graph"]
    if raw_graph is None:
        return {"refined_graph": None, "error": "No raw graph to refine"}

    print(f"  [Discriminator] Reviewing raw graph ...")

    # Build content blocks — SAME images as Generator + state.json + raw graph
    content_blocks = []

    # 1. Camera images (same 16 frames the Generator saw)
    content_blocks.extend(state["camera_blocks"])

    # 2. BEV image (same as Generator)
    bev_path = Path(state["scene_dir"]) / "Tp0p0" / "bev.png"
    if bev_path.exists():
        content_blocks.append(image_content_block(str(bev_path)))
    else:
        content_blocks.append({"type": "input_text", "text": "[BEV image not available]"})

    # 3. state.json (same as Generator)
    content_blocks.append({
        "type": "input_text",
        "text": f"state.json:\n```json\n{state['state_text']}\n```"
    })

    # 4. Raw graph from Generator (the thing to review)
    raw_graph_clean = {k: v for k, v in raw_graph.items() if not k.startswith("_")}
    content_blocks.append({
        "type": "input_text",
        "text": f"Raw causal graph (Stage 1 output) to review:\n```json\n{json.dumps(raw_graph_clean, indent=2)}\n```"
    })

    # System prompt
    system_prompt = state["refiner_prompt"].replace("{nav_command}", state["nav_command"])

    model = state["model"]
    reasoning = state["s2_reasoning"]
    n_images = sum(1 for b in content_blocks if b.get("type") in ("input_image", "image"))
    print(f"  [Discriminator] Calling {model} ({n_images} images + state.json + raw graph)  reasoning={reasoning} ...")

    raw, usage_data = _provider.call(system_prompt, content_blocks, max_tokens=65536)
    result = extract_json(raw)
    print(f"  [Discriminator] Tokens: in={usage_data.get('input','?')} (image={usage_data.get('image',0)}) out={usage_data.get('output','?')}")

    trace = result.get("reasoning_trace", "") if isinstance(result, dict) else ""

    if "_raw_response" in result:
        return {"refined_graph": None, "s2_usage": usage_data, "reasoning_trace": trace,
                "error": "Stage 2 returned invalid JSON"}

    return {"refined_graph": result, "s2_usage": usage_data, "reasoning_trace": trace, "error": None}


def should_skip_stage2(state: PipelineState) -> str:
    """Conditional edge: skip discriminator if generator failed."""
    if state.get("error") or state.get("raw_graph") is None:
        return END
    return "discriminator"


# ── Build the LangGraph ──────────────────────────────────────────────────

def build_pipeline() -> Any:
    """Construct the Generator → Discriminator LangGraph."""
    graph = StateGraph(PipelineState)

    graph.add_node("generator", generator_node)
    graph.add_node("discriminator", discriminator_node)

    graph.set_entry_point("generator")
    graph.add_conditional_edges("generator", should_skip_stage2)
    graph.add_edge("discriminator", END)

    return graph.compile()


def build_stage2_only_pipeline() -> Any:
    """Construct a Discriminator-only graph (when --skip-stage1)."""
    graph = StateGraph(PipelineState)
    graph.add_node("discriminator", discriminator_node)
    graph.set_entry_point("discriminator")
    graph.add_edge("discriminator", END)
    return graph.compile()


# ── Scene Preparation ────────────────────────────────────────────────────

def prepare_scene_state(scene_dir: Path, generator_prompt: str, refiner_prompt: str,
                        model: str, s1_reasoning: str, s2_reasoning: str,
                        existing_raw_graph: dict | None = None) -> PipelineState:
    """Build the initial shared state for a scene."""
    tp0_dir = scene_dir / "Tp0p0"
    state_path = tp0_dir / "state.json"
    state_json = json.loads(state_path.read_text())

    # Build camera content blocks
    temporal = find_temporal_frames(scene_dir)
    slot_labels = ["t-3", "t-2", "t-1", "t0"]
    camera_blocks = []

    for cam in PROMPT_CAMS:
        cam_label = CAM_DISPLAY[cam]
        # Text separator so the model knows which camera group follows
        camera_blocks.append({
            "type": "input_text",
            "text": f"--- {cam_label}: 4 temporal frames ({', '.join(slot_labels)}) ---"
        })
        for i, entry in enumerate(temporal):
            if entry is None:
                camera_blocks.append(
                    {"type": "input_text", "text": f"[NULL: {cam.upper()} at {slot_labels[i]}]"})
                continue
            slug, t, d = entry
            img_path = d / f"{cam}.jpg"
            label = f"{cam_label} ({slot_labels[i]})"
            if img_path.exists():
                camera_blocks.append(
                    image_content_block(str(img_path), label=label))
            else:
                camera_blocks.append(
                    {"type": "input_text", "text": f"[IMAGE MISSING: {cam.upper()} at {slug}]"})

    # BEV as base64
    bev_path = tp0_dir / "bev.png"
    bev_b64 = load_image_b64(str(bev_path)) if bev_path.exists() else ""

    return PipelineState(
        scene_dir=str(scene_dir),
        scene_name=f"{scene_dir.parent.name}/{scene_dir.name}",
        bev_b64=bev_b64,
        state_json=state_json,
        state_text=json.dumps(state_json, indent=2),
        nav_command=state_json.get("nav_command", "UNKNOWN"),
        camera_blocks=camera_blocks,
        generator_prompt=generator_prompt,
        refiner_prompt=refiner_prompt,
        model=model,
        s1_reasoning=s1_reasoning,
        s2_reasoning=s2_reasoning,
        raw_graph=existing_raw_graph,
        refined_graph=None,
        reasoning_trace="",
        s1_usage={},
        s2_usage={},
        error=None,
    )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config_loader import load_config
    cfg = load_config() or {}

    parser = argparse.ArgumentParser(
        description="V3 LangGraph pipeline: Generator → Discriminator/Fixer"
    )
    parser.add_argument("--records", default=None, help="Path to scenario_records directory")
    parser.add_argument("--out", default=None, help="Output directory (default: <records>/../causal_graphs)")
    parser.add_argument("--provider", default=cfg.get("provider", "openai"),
                        choices=["openai", "gemini", "claude"], help="LLM provider")
    parser.add_argument("--model", default=cfg.get("model", "gpt-5.2"), help="Model to use")
    parser.add_argument("--reasoning", default=cfg.get("reasoning", "none"), choices=["none", "low", "medium", "high"])
    parser.add_argument("--stage1-reasoning", default=None, choices=["none", "low", "medium", "high"])
    parser.add_argument("--stage2-reasoning", default=None, choices=["none", "low", "medium", "high"])
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--scenes", nargs="*", default=None, help="Specific scenes to process")
    parser.add_argument("--samples", nargs="*", default=None,
                        help="Specific scene_SAMPLED_N combos (e.g. nuscenes-scene-0035_SAMPLED_1)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Skip Generator — use existing raw graph, only run Discriminator")
    args = parser.parse_args()

    # Initialize provider
    global _provider
    provider_name = args.provider or cfg.get("provider", "openai")

    api_key = None
    if provider_name == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or cfg.get("api_key")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    elif provider_name in ("gemini", "google"):
        api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or cfg.get("google_api_key")
    elif provider_name in ("claude", "anthropic"):
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY") or cfg.get("anthropic_api_key")

    if not api_key or api_key.startswith("${"):
        print(f"ERROR: Provide --api-key or set the appropriate env var for provider '{provider_name}'")
        sys.exit(1)

    from common.llm_provider import get_provider
    _provider = get_provider(provider_name, model=args.model, api_key=api_key,
                             reasoning=args.stage1_reasoning or args.reasoning)
    records_dir = Path(args.records)
    out_dir = Path(args.out) if args.out else records_dir.parent / "causal_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_reasoning = args.stage1_reasoning or args.reasoning
    s2_reasoning = args.stage2_reasoning or args.reasoning

    # Load prompts
    script_dir = Path(__file__).parent
    prompt_dir = script_dir.parent.parent / "prompts" / "causal_scene_graph" / "v1"
    generator_prompt = (prompt_dir / "system_prompt.md").read_text()
    refiner_prompt = (prompt_dir / "refinement_prompt.md").read_text()

    # Build the graph
    if args.skip_stage1:
        pipeline = build_stage2_only_pipeline()
        print("Pipeline: Discriminator only (--skip-stage1)")
    else:
        pipeline = build_pipeline()
        print("Pipeline: Generator → Discriminator")

    # Find scene directories
    samples_set = set(args.samples) if args.samples else None
    scene_dirs = []
    for scene_path in sorted(records_dir.iterdir()):
        if not scene_path.is_dir() or scene_path.name == "overview":
            continue
        if args.scenes and scene_path.name not in args.scenes:
            continue
        for maneuver_path in sorted(scene_path.iterdir()):
            if maneuver_path.is_dir() and (maneuver_path / "Tp0p0" / "state.json").exists():
                combo = f"{scene_path.name}_{maneuver_path.name}"
                if samples_set and combo not in samples_set:
                    continue
                scene_dirs.append(maneuver_path)

    print(f"Scenes: {len(scene_dirs)}  |  Model: {args.model}")
    print(f"Stage 1: {s1_reasoning}  |  Stage 2: {s2_reasoning}")
    print(f"Output: {out_dir}\n")

    # Accumulators
    total_s1_in, total_s1_out, total_s1_img = 0, 0, 0
    total_s2_in, total_s2_out, total_s2_img = 0, 0, 0
    processed = 0

    for scene_dir in scene_dirs:
        scene_name = scene_dir.parent.name
        maneuver = scene_dir.name
        raw_file = out_dir / f"{scene_name}_{maneuver}.json"
        refined_file = out_dir / f"{scene_name}_{maneuver}_refined.json"

        if refined_file.exists() and not args.force:
            print(f"SKIP {scene_name}/{maneuver} — exists (use --force)")
            continue

        print(f"{'='*60}")
        print(f"Scene: {scene_name} / {maneuver}")
        print(f"{'='*60}")

        # Load existing raw graph if skipping Stage 1
        existing_raw = None
        if args.skip_stage1:
            if raw_file.exists():
                existing_raw = json.loads(raw_file.read_text())
                print(f"  Loaded raw graph: {raw_file.name}")
            else:
                print(f"  SKIP — no raw graph found at {raw_file.name}")
                continue

        # Prepare shared state
        initial_state = prepare_scene_state(
            scene_dir, generator_prompt, refiner_prompt,
            args.model, s1_reasoning, s2_reasoning,
            existing_raw_graph=existing_raw,
        )

        # Run the LangGraph pipeline
        final_state = pipeline.invoke(initial_state)

        # Handle errors
        if final_state.get("error"):
            print(f"  ERROR: {final_state['error']}\n")
            continue

        # Save raw graph (from Stage 1)
        raw_graph = final_state.get("raw_graph")
        if raw_graph and not args.skip_stage1:
            with open(raw_file, "w") as f:
                raw_clean = {k: v for k, v in raw_graph.items() if not k.startswith("_")}
                json.dump(raw_clean, f, indent=2)
            print(f"  Raw graph → {raw_file.name}")

        # Save refined graph (from Stage 2)
        refined_graph = final_state.get("refined_graph")
        if refined_graph:
            state_json = final_state.get("state_json", {})

            # ── CRITICAL 2 Fix B: Validate effect_type values ──
            VALID_EFFECT_TYPES = {
                "OBSTRUCTS", "BOUNDS_LATERALLY", "LIMITS_SPEED",
                "GRANTS_RIGHT_OF_WAY", "REQUIRES_YIELD",
                "REQUIRES_GAP_ACCEPTANCE", "REQUIRES_FOLLOWING",
                "REQUIRES_EVASION", "COMMANDS", "DEGRADES",
            }
            EFFECT_TYPE_FIXES = {
                "LIMITS_ACCELERATION": "LIMITS_SPEED",
                "REGULATORY_ELEMENT": "REQUIRES_YIELD",
                "LIMITS_LANE_CHANGE": "BOUNDS_LATERALLY",
                "LEADING_VEHICLE": "REQUIRES_FOLLOWING",
                "CLOSING_SPEED": "LIMITS_SPEED",
            }
            for e in refined_graph.get("ego_edges", []):
                etype = e.get("effect_type", "")
                if etype not in VALID_EFFECT_TYPES:
                    fixed = EFFECT_TYPE_FIXES.get(etype, "LIMITS_SPEED")
                    print(f"  [Post-process] Fixed invalid effect_type: {etype} -> {fixed} (edge {e.get('source')} -> ego)")
                    e["effect_type"] = fixed

            # ── CRITICAL 3 Fix B: Remove ego_edges from behind-ego different-lane agents ──
            ego_lane = state_json.get("ego", {}).get("lane_id")
            agents_data = state_json.get("agents", {})
            kept_edges = []
            for e in refined_graph.get("ego_edges", []):
                src = e.get("source", "")
                if src in agents_data and ego_lane:
                    agent_info = agents_data[src]
                    agent_x = agent_info.get("position", {}).get("x", 0)
                    agent_lane = agent_info.get("lane_id")
                    if agent_x < 0 and agent_lane and agent_lane != ego_lane:
                        print(f"  [Post-process] Removed ego_edge: {src} -> ego (behind x={agent_x:.1f}, lane {agent_lane} != ego {ego_lane})")
                        continue
                kept_edges.append(e)
            if len(kept_edges) < len(refined_graph.get("ego_edges", [])):
                n_removed = len(refined_graph["ego_edges"]) - len(kept_edges)
                refined_graph["ego_edges"] = kept_edges
                print(f"  [Post-process] Removed {n_removed} behind-ego different-lane ego_edge(s)")

            # Post-process: remove truly disconnected nodes
            # Keep nodes reachable from ego-connected set via EITHER ego_edges OR inter_node_edges
            ego_edge_ids = set()
            for e in refined_graph.get("ego_edges", []):
                ego_edge_ids.add(e.get("source"))
                ego_edge_ids.add(e.get("target"))
            # Seed: ego, nav_cmd, and anything directly in ego_edges
            reachable = {"ego", "nav_cmd"} | ego_edge_ids
            # Build adjacency from inter_node_edges (undirected for reachability)
            inter_adj = {}
            for e in refined_graph.get("inter_node_edges", []):
                s, t = e.get("source"), e.get("target")
                inter_adj.setdefault(s, set()).add(t)
                inter_adj.setdefault(t, set()).add(s)
            # BFS: expand reachable set via inter_node_edges
            frontier = list(reachable)
            while frontier:
                nid = frontier.pop()
                for neighbor in inter_adj.get(nid, []):
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        frontier.append(neighbor)
            keep_ids = reachable
            before_count = len(refined_graph.get("nodes", []))
            refined_graph["nodes"] = [
                n for n in refined_graph.get("nodes", [])
                if n.get("id") in keep_ids
            ]
            removed_ids = set()
            if len(refined_graph["nodes"]) < before_count:
                all_kept = {n["id"] for n in refined_graph["nodes"]}
                removed_ids = {n.get("id") for n in final_state["refined_graph"].get("nodes", [])} - all_kept
                # Also remove inter_node_edges referencing removed nodes
                refined_graph["inter_node_edges"] = [
                    e for e in refined_graph.get("inter_node_edges", [])
                    if e.get("source") in all_kept and e.get("target") in all_kept
                ]
                for rid in sorted(removed_ids):
                    print(f"  [Post-process] Removed disconnected node: {rid}")

            refined_clean = {k: v for k, v in refined_graph.items() if not k.startswith("_") and k != "changes"}
            refined_clean["scene_id"] = scene_name
            with open(refined_file, "w") as f:
                json.dump(refined_clean, f, indent=2)
            print(f"  Refined graph → {refined_file.name}")

            # Save into Tp0p0 as v3 (don't overwrite v2's causal_graph_generated.json)
            tp0_out = scene_dir / "Tp0p0" / "causal_graph_v3.json"
            with open(tp0_out, "w") as f:
                json.dump(refined_clean, f, indent=2)
            print(f"  Viewer copy → {tp0_out}")

        # Print changes from Stage 2
        if refined_graph:
            changes = refined_graph.get("changes", [])
            if changes:
                print(f"  Stage 2 changes ({len(changes)}):")
                for c in changes:
                    action = c.get("action", "?")
                    ctype = c.get("type", "?")
                    reason = c.get("reason", "")
                    if action in ("REMOVED", "ADDED"):
                        src = c.get("source", c.get("id", "?"))
                        tgt = c.get("target", "")
                        effect = c.get("effect_type", "")
                        label = f"{src} → {tgt}" if tgt else src
                        if effect:
                            label += f" [{effect}]"
                        print(f"    {action} {ctype}: {label}")
                        print(f"      reason: {reason}")
                    elif action == "FIXED":
                        nid = c.get("id", "?")
                        field = c.get("field", "?")
                        print(f"    FIXED {ctype} {nid}.{field}: {c.get('old','?')} → {c.get('new','?')}")
                        print(f"      reason: {reason}")
                    elif action == "CHANGED":
                        src = c.get("source", "?")
                        tgt = c.get("target", "?")
                        field = c.get("field", "?")
                        print(f"    CHANGED {ctype} {src} → {tgt} .{field}: {c.get('old','?')} → {c.get('new','?')}")
                        print(f"      reason: {reason}")
                    else:
                        print(f"    {action}: {c}")
            else:
                print(f"  Stage 2: no changes")

            # Save changes log
            changes_file = out_dir / f"{scene_name}_{maneuver}_changes.json"
            with open(changes_file, "w") as f:
                json.dump(changes, f, indent=2)
            print(f"  Changes log → {changes_file.name}")

        # Accumulate usage
        s1u = final_state.get("s1_usage", {})
        s2u = final_state.get("s2_usage", {})
        total_s1_in += s1u.get("input", 0)
        total_s1_out += s1u.get("output", 0)
        total_s1_img += s1u.get("image", 0)
        total_s2_in += s2u.get("input", 0)
        total_s2_out += s2u.get("output", 0)
        total_s2_img += s2u.get("image", 0)
        processed += 1
        print()

    # Summary
    print(f"{'='*60}")
    print(f"Done. {processed} scenes processed.")
    print(f"{'='*60}")
    print(f"Stage 1 — in: {total_s1_in:,} (image={total_s1_img:,})  out: {total_s1_out:,}")
    print(f"Stage 2 — in: {total_s2_in:,} (image={total_s2_img:,})  out: {total_s2_out:,}")
    total_in = total_s1_in + total_s2_in
    total_out = total_s1_out + total_s2_out
    total_img = total_s1_img + total_s2_img
    cost = (total_in / 1e6 * 1.75) + (total_out / 1e6 * 14.00)
    print(f"Total  — in: {total_in:,} (image={total_img:,})  out: {total_out:,}")
    print(f"Estimated cost: ${cost:.2f}")


if __name__ == "__main__":
    main()
