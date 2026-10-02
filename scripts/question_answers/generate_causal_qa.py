"""
generate_causal_qa.py — VLM-based Causal Q&A Generation Pipeline

Takes a refined causal graph + BEV at T=0 and generates Q&A pairs using two prompts:

  Prompt 1 — Active QA (Rung 0–3):
      Active-edge Q&A: CaI, CA, AR, CB, BAS, CDE, CR, SC, NC, NIE, NDE
      Rung 3 counterfactual questions feed into the CF trajectory pipeline.

  Prompt 2 — Discrimination + Latent Activation (Dormant):
      Latent-node Q&A: DQ-Binary, DQ-Select, DQ-Contrast, AC-Binary, AC-Single, AC-Conflict, AC-Chain
      AC-Binary (Yes) questions feed into the AC trajectory pipeline.

Flow:
  causal_graph + BEV + state.json
       │
       ├──► Prompt 1 (Active QA)  ──► Rung 0-3 Q&A pairs
       │                                       │
       │                                  Rung 3 CF questions ──► counterfactual_pipeline.py --prompt-type cf
       │                                                           (Generator → Visualize)
       │
       └──► Prompt 2 (Dormant)    ──► DQ + AC Q&A pairs
                                       │
                                  AC-Binary (Yes) ──► counterfactual_pipeline.py --prompt-type ac
                                                       (AC Generator → Visualize)

  Both prompts always run. CF + AC trajectory pipelines are optional (--skip-cf-pipeline).

Usage:
  python generate_causal_qa.py \
      --records openscene_records \
      --graphs openscene_causal_graphs/v3 \
      --scene openscene-2021.05.12.19.36.12_veh-35_002_SAMPLED \
      --out causal_qa_results \
      --model gpt-5.2

  # Skip CF trajectory generation for Rung 3 questions:
  python generate_causal_qa.py ... --skip-cf-pipeline
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
# Provider instance — set in main()
_provider = None

load_dotenv()

# ── Prompt paths ────────────────────────────────────────────────────────
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
QA_PROMPTS = PROMPTS_DIR / "causal_qas"

# ── Camera / BEV constants ─────────────────────────────────────────────
PROMPT_CAMS = ["cam_front", "cam_front_left", "cam_front_right", "cam_back"]
CAM_DISPLAY = {
    "cam_front": "CAM_FRONT",
    "cam_front_left": "CAM_FRONT_LEFT",
    "cam_front_right": "CAM_FRONT_RIGHT",
    "cam_back": "CAM_BACK",
}

BEV_SPATIAL_TEXT = (
    "**BEV orientation:** Ego is at the center, always facing UP.\n"
    "- Vertical axis = X axis: up = +x (ahead of ego), down = -x (behind ego).\n"
    "- Horizontal axis = Y axis: left in image = +y (left of ego), right in image = -y (right of ego).\n"
    "- This matches the state.json coordinate frame exactly.\n"
    "- Infrastructure or agents at the BOTTOM of the BEV are BEHIND ego.\n\n"
    "**BEV Visual Encoding:**\n"
    "- Ego vehicle: RED oriented box at center.\n"
    "- Other vehicles: GREY oriented boxes.\n"
    "- Pedestrians/cyclists (VRU): GREEN small boxes.\n"
    "- Drivable area: dark surface. Lane boundaries: white/yellow lines.\n"
    "- Crosswalks: striped regions. Road edges: solid boundaries.\n"
    "- Lane direction arrows: light grey arrows along lane centerlines."
)


# ── Helpers ─────────────────────────────────────────────────────────────

def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def image_content_block(path: str) -> dict:
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/png")
    b64 = load_image_b64(path)
    return {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}


def build_scene_content_blocks(tp0_dir: Path, version: str) -> list:
    """Build content blocks based on prompt version.

    v1: no images (graph-only)
    v2: BEV image + BEV spatial text (no cameras)
    v3: 4 camera images + BEV + BEV spatial text
    """
    blocks = []

    if version == "v3":
        # 4 camera images at T=0
        for cam in PROMPT_CAMS:
            cam_path = tp0_dir / f"{cam}.jpg"
            if cam_path.exists():
                blocks.append({"type": "input_text", "text": f"--- {CAM_DISPLAY[cam]} (t0) ---"})
                blocks.append(image_content_block(str(cam_path)))
            else:
                blocks.append({"type": "input_text",
                               "text": f"[IMAGE MISSING: {CAM_DISPLAY[cam]} at t0]"})

    if version in ("v2", "v3"):
        # BEV with spatial orientation
        bev_path = tp0_dir / "bev.png"
        if bev_path.exists():
            blocks.append({"type": "input_text",
                           "text": f"--- BEV (Bird's Eye View at t0) ---\n{BEV_SPATIAL_TEXT}"})
            blocks.append(image_content_block(str(bev_path)))

    n_imgs = sum(1 for b in blocks if b["type"] == "input_image")
    if n_imgs:
        print(f"  Scene context: {n_imgs} images")
    else:
        print(f"  Scene context: graph-only (no images)")

    return blocks


def filter_state_for_graph(state_json: dict, causal_graph: dict) -> dict:
    """Filter state.json to only include agents that appear as nodes in the causal graph."""
    graph_node_ids = {n["id"] for n in causal_graph.get("nodes", [])}
    filtered = dict(state_json)  # shallow copy
    if "agents" in filtered:
        filtered["agents"] = {
            k: v for k, v in filtered["agents"].items()
            if k in graph_node_ids
        }
    return filtered


def extract_json_from_response(raw: str) -> list | dict:
    """Extract JSON from VLM response, handling markdown fences and scratchpad."""
    import re
    raw = raw.strip()
    # Strip scratchpad if present (Dormant prompt produces one before JSON)
    if "END SCRATCHPAD" in raw:
        raw = raw[raw.index("END SCRATCHPAD") + len("END SCRATCHPAD"):].strip()
    # Extract last markdown code block (model may output analysis text before JSON)
    code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if code_blocks:
        raw = code_blocks[-1].strip()
    # Try parsing as-is
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try finding first [ or {
        for i, ch in enumerate(raw):
            if ch in "[{":
                try:
                    return json.loads(raw[i:])
                except json.JSONDecodeError:
                    pass
        return {"_raw_response": raw}


def find_scene_dir(records_dir: Path, scene_name: str) -> Path | None:
    for candidate in records_dir.iterdir():
        if not candidate.is_dir():
            continue
        for md in candidate.iterdir():
            if not md.is_dir():
                continue
            full_name = f"{candidate.name}_{md.name}"
            if full_name == scene_name or candidate.name == scene_name:
                return md
    return None


def find_graph_file(graphs_dir: Path, scene_name: str) -> Path | None:
    for f in graphs_dir.glob("*_pruned.json"):
        if scene_name in f.stem or re.sub(r'_SAMPLED(_\d+)?$', '', scene_name) in f.stem:
            return f
    return None


def call_vlm(system_prompt: str, content_blocks: list, model: str, reasoning: str,
             max_tokens: int = 16384) -> tuple[str, dict]:
    """Call VLM via the configured provider. Returns (raw_text, usage_dict)."""
    return _provider.call(system_prompt, content_blocks, max_tokens=max_tokens)


# ── Prompt 1: Active QA ────────────────────────────────────────────

def generate_active_qa(
    scene_blocks: list,
    causal_graph: dict,
    model: str,
    reasoning: str,
    prompt_version: str = "v3",
    filtered_state: dict | None = None,
) -> dict:
    """
    Generate Active QA Q&A pairs (Rung 0–3) from causal graph + scene images.
    Returns: {"questions": [...], "usage": {...}, "error": ...}
    """
    prompts_dir = QA_PROMPTS / prompt_version
    system_prompt = (prompts_dir / "Causal_Ladder_System_Prompt.md").read_text()

    # Build user prompt — inject scene graph into template
    user_template = (prompts_dir / "Causal_Ladder_User_Prompt.md").read_text()
    user_prompt = user_template.replace(
        "[paste scene graph JSON here]",
        json.dumps(causal_graph, indent=2),
    )
    # v3 user prompt has state.json placeholder
    if filtered_state is not None and "[paste filtered state JSON here]" in user_prompt:
        user_prompt = user_prompt.replace(
            "[paste filtered state JSON here]",
            json.dumps(filtered_state, indent=2),
        )

    # Scene images (cameras + BEV) followed by the text prompt
    content_blocks = list(scene_blocks)
    content_blocks.append({"type": "input_text", "text": user_prompt})

    print(f"  [Active QA] Calling {model} ...")
    raw, usage = call_vlm(system_prompt, content_blocks, model, reasoning)
    print(f"  [Active QA] Tokens: in={usage.get('input', '?')} (image={usage.get('image', 0)}) out={usage.get('output', '?')}")

    result = extract_json_from_response(raw)
    if isinstance(result, dict) and "_raw_response" in result:
        return {"questions": [], "usage": usage, "error": raw}

    questions = result if isinstance(result, list) else result.get("questions", [result])
    # Filter out non-question items (e.g. structure enumeration preamble)
    questions = [q for q in questions if isinstance(q, dict) and "question" in q]
    return {"questions": questions, "usage": usage, "error": None}


# ── Prompt 2: Discrimination + Latent Activation ──────────────────────

def generate_dormant_qa(
    scene_blocks: list,
    causal_graph: dict,
    model: str,
    reasoning: str,
    prompt_version: str = "v3",
    filtered_state: dict | None = None,
) -> dict:
    """
    Generate DQ + AC Q&A pairs from causal graph + scene images.
    Returns: {"questions": [...], "usage": {...}, "error": ...}
    """
    prompts_dir = QA_PROMPTS / prompt_version
    # v4 renamed Dormant to Dormant_Nodes
    sys_file = prompts_dir / "Dormant_Nodes_System_Prmpt.md"
    if not sys_file.exists():
        sys_file = prompts_dir / "Dormant_System_Prompt.md"
    system_prompt = sys_file.read_text()

    usr_file = prompts_dir / "Dormant_Nodes_User_Prompt.md"
    if not usr_file.exists():
        usr_file = prompts_dir / "Dormant_User_Prompt.md"
    user_template = usr_file.read_text()
    user_prompt = user_template.replace(
        "[paste scene graph JSON here]",
        json.dumps(causal_graph, indent=2),
    )
    # v3 user prompt has state.json placeholder
    if filtered_state is not None and "[paste filtered state JSON here]" in user_prompt:
        user_prompt = user_prompt.replace(
            "[paste filtered state JSON here]",
            json.dumps(filtered_state, indent=2),
        )

    # Scene images (cameras + BEV) followed by the text prompt
    content_blocks = list(scene_blocks)
    content_blocks.append({"type": "input_text", "text": user_prompt})

    print(f"  [Dormant] Calling {model} ...")
    raw, usage = call_vlm(system_prompt, content_blocks, model, reasoning)
    print(f"  [Dormant] Tokens: in={usage.get('input', '?')} (image={usage.get('image', 0)}) out={usage.get('output', '?')}")

    result = extract_json_from_response(raw)
    if isinstance(result, dict) and "_raw_response" in result:
        return {"questions": [], "usage": usage, "error": raw}

    questions = result if isinstance(result, list) else result.get("questions", [result])
    return {"questions": questions, "usage": usage, "error": None}


# ── Prompt 3: Distractor Nodes ────────────────────────────────────────

def generate_distractor_qa(
    scene_blocks: list,
    causal_graph: dict,
    model: str,
    reasoning: str,
    prompt_version: str = "v4",
    filtered_state: dict | None = None,
) -> dict:
    """
    Generate Distractor Nodes Q&A pairs from causal graph + scene images.
    Returns: {"questions": [...], "usage": {...}, "error": ...}
    """
    prompts_dir = QA_PROMPTS / prompt_version
    system_prompt = (prompts_dir / "Distractor_Nodes_System_Prompt.md").read_text()

    user_template = (prompts_dir / "Distractor_Nodes_User_Prompt.md").read_text()
    user_prompt = user_template.replace(
        "[paste scene graph JSON here]",
        json.dumps(causal_graph, indent=2),
    )
    if filtered_state is not None and "[paste filtered state JSON here]" in user_prompt:
        user_prompt = user_prompt.replace(
            "[paste filtered state JSON here]",
            json.dumps(filtered_state, indent=2),
        )

    content_blocks = list(scene_blocks)
    content_blocks.append({"type": "input_text", "text": user_prompt})

    print(f"  [Distractor] Calling {model} ...")
    raw, usage = call_vlm(system_prompt, content_blocks, model, reasoning)
    print(f"  [Distractor] Tokens: in={usage.get('input', '?')} (image={usage.get('image', 0)}) out={usage.get('output', '?')}")

    result = extract_json_from_response(raw)
    if isinstance(result, dict) and "_raw_response" in result:
        return {"questions": [], "usage": usage, "error": raw}

    questions = result if isinstance(result, list) else result.get("questions", [result])
    questions = [q for q in questions if isinstance(q, dict) and "question" in q]
    return {"questions": questions, "usage": usage, "error": None}


# ── Extract Rung 3 CF questions from Active QA output ─────────────

def extract_cf_questions(active_questions: list) -> list[dict]:
    """Pull out CR (Counterfactual Reasoning) questions for the CF trajectory pipeline.

    Only CR questions involve a concrete state change intervention that can be
    simulated as a counterfactual trajectory. SC, NC, NIE, NDE are answered
    analytically and do not need trajectory generation.
    """
    cf_questions = []
    for q in active_questions:
        rung = q.get("rung")
        q_type = q.get("question_type", "")
        if rung == 3 and q_type == "CR":
            cf_questions.append(q)
    return cf_questions


# ── Extract AC-Binary questions from Dormant output ─────────────────────────

def extract_ac_questions(dl_questions: list, entity_type: str = None) -> list[dict]:
    """Pull out AC-Binary questions where correct_answer is Yes (behavior would change).

    These are activation counterfactuals for inactive elements — analogous to
    CR questions but for latent nodes becoming active.

    entity_type: if set, filter by ac_intervention.entity_type ('infrastructure' or 'agent').
                 'infrastructure' → LLM trajectory pipeline.
                 'agent' → nuPlan planner re-simulation (future).
                 None → return all AC-Binary Yes questions.
    """
    ac_questions = []
    for q in dl_questions:
        subtype = q.get("subtype", q.get("question_type", ""))
        correct = q.get("correct_answer", "").strip().lower()
        if subtype == "AC-Binary" and correct == "yes":
            # Filter by entity_type if specified
            if entity_type is not None:
                intervention = q.get("meta", {}).get("ac_intervention", {})
                if intervention and intervention.get("entity_type") != entity_type:
                    continue
            ac_questions.append(q)
    return ac_questions


# ── Run trajectory pipeline for a single question ────────────────────────

def run_trajectory_pipeline_for_question(
    question_text: str,
    scene_name: str,
    records_dir: Path,
    graphs_dir: Path,
    out_dir: Path,
    model: str,
    reasoning: str,
    prompt_type: str = "cf",
    force: bool = False,
    cf_answer: str = "",
    cf_reasoning: str = "",
    provider: str = "openai",
) -> dict | None:
    """Shell out to counterfactual_pipeline.py. Returns result dict or None.
    prompt_type: 'cf' for CR questions, 'ac' for AC-Binary questions."""
    q_slug = (
        question_text[:80]
        .replace(" ", "_")
        .replace("?", "")
        .replace("/", "_")
        .replace("'", "")
        .replace('"', "")
    )

    q_out_dir = out_dir / q_slug
    result_path = q_out_dir / scene_name / "result.json"

    if result_path.exists() and not force:
        print(f"      SKIP: {result_path} exists")
        return json.loads(result_path.read_text())

    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "counterfactual_trajectory" / "counterfactual_pipeline.py"),
        "--records", str(records_dir),
        "--graphs", str(graphs_dir),
        "--scene", scene_name,
        "--question", question_text,
        "--cf-answer", cf_answer,
        "--cf-reasoning", cf_reasoning,
        "--out", str(q_out_dir),
        "--provider", provider,
        "--model", model,
        "--reasoning", reasoning,
        "--prompt-type", prompt_type,
    ]
    if force:
        cmd.append("--force")

    label = "AC" if prompt_type == "ac" else "CF"
    print(f"      Running {label} pipeline ...")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"      {label} pipeline FAILED:\n{proc.stderr[-300:]}")
        return None

    if result_path.exists():
        return json.loads(result_path.read_text())
    return None


# ── Main ────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config_loader import load_config
    cfg = load_config() or {}

    parser = argparse.ArgumentParser(
        description="Causal Q&A generation: Active QA (Rung 0-3) + Dormant (DQ+AC) + CF trajectory pipeline")
    parser.add_argument("--records", default=None, help="Scenario records directory")
    parser.add_argument("--graphs", default=None, help="Causal graphs directory (*_pruned.json)")
    parser.add_argument("--scene", required=True, help="Scene name")
    parser.add_argument("--out", default="causal_qa_results", help="Output directory")
    parser.add_argument("--provider", default=cfg.get("provider", "openai"),
                        choices=["openai", "gemini", "claude"], help="LLM provider")
    parser.add_argument("--model", default=cfg.get("model", "gpt-5.2"), help="Model to use")
    parser.add_argument("--reasoning", default=cfg.get("reasoning", "none"),
                        choices=["none", "low", "medium", "high"])
    parser.add_argument("--skip-cf-pipeline", action="store_true",
                        help="Generate Q&A only, skip CF trajectory generation")
    parser.add_argument("--only", choices=["active", "dormant"],
                        help="Run only one prompt type (default: both)")
    parser.add_argument("--prompt-version", default="v3", choices=["v1", "v2", "v3", "v4"],
                        help="Prompt version: v1=original, v2=MCQ+no-graph-terms, v3=v2+images+state, v4=v3+distractor")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    # Initialize provider
    global _provider
    provider_name = args.provider
    api_key = None
    if provider_name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or cfg.get("api_key")
        if api_key and not api_key.startswith("${"):
            os.environ["OPENAI_API_KEY"] = api_key
    elif provider_name in ("gemini", "google"):
        api_key = os.environ.get("GOOGLE_API_KEY") or cfg.get("google_api_key")
    elif provider_name in ("claude", "anthropic"):
        api_key = os.environ.get("ANTHROPIC_API_KEY") or cfg.get("anthropic_api_key")

    from common.llm_provider import get_provider
    _provider = get_provider(provider_name, model=args.model, api_key=api_key,
                             reasoning=args.reasoning)

    records_dir = Path(args.records)
    graphs_dir = Path(args.graphs)

    # ── Locate scene ──
    scene_dir = find_scene_dir(records_dir, args.scene)
    if scene_dir is None:
        print(f"ERROR: Scene '{args.scene}' not found in {records_dir}")
        sys.exit(1)

    graph_file = find_graph_file(graphs_dir, args.scene)
    if graph_file is None:
        print(f"ERROR: No refined graph for '{args.scene}' in {graphs_dir}")
        sys.exit(1)

    print(f"Scene dir:    {scene_dir}")
    print(f"Causal graph: {graph_file}")

    pv = args.prompt_version
    print(f"Prompt version: {pv}")

    # ── Load data ──
    tp0_dir = scene_dir / "Tp0p0"
    state_json = json.loads((tp0_dir / "state.json").read_text())
    causal_graph = json.loads(graph_file.read_text())
    causal_graph.pop("changes", None)  # Strip refinement audit log

    # Filter state.json to only include agents in the graph (for v3)
    filtered_state = filter_state_for_graph(state_json, causal_graph) if pv == "v3" else None

    # Build scene content blocks based on prompt version
    scene_blocks = build_scene_content_blocks(tp0_dir, pv)

    out_dir = Path(args.out) / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    run_active = args.only is None or args.only == "active"
    run_dl = args.only is None or args.only == "dormant"
    run_distractor = (args.only is None or args.only == "distractor") and pv == "v4"

    # ══════════════════════════════════════════════════════════════════════
    #  Prompt 1: Active QA (Rung 0-3)
    # ══════════════════════════════════════════════════════════════════════
    active_result = {"questions": [], "usage": {}, "error": None}
    if run_active:
        print(f"\n{'='*60}")
        print(f"  Prompt 1: Active QA (Rung 0-3)")
        print(f"{'='*60}")

        active_path = out_dir / "active_qa.json"
        if active_path.exists() and not args.force:
            print(f"  SKIP: {active_path} exists (use --force)")
            active_result = json.loads(active_path.read_text())
        else:
            active_result = generate_active_qa(scene_blocks, causal_graph, args.model, args.reasoning,
                                                prompt_version=pv, filtered_state=filtered_state)
            with open(active_path, "w") as f:
                json.dump(active_result, f, indent=2)
            print(f"  Saved → {active_path}")

        n_q = len(active_result.get("questions", []))
        print(f"  Active QA questions: {n_q}")
        if active_result.get("error"):
            print(f"  ERROR: {str(active_result['error'])[:200]}")

    # ══════════════════════════════════════════════════════════════════════
    #  Prompt 2: Discrimination + Latent Activation
    # ══════════════════════════════════════════════════════════════════════
    dl_result = {"questions": [], "usage": {}, "error": None}
    if run_dl:
        print(f"\n{'='*60}")
        print(f"  Prompt 2: Discrimination + Latent Activation (Dormant)")
        print(f"{'='*60}")

        # Skip Dormant if there are no inactive ego_edges (nothing latent to ask about)
        has_inactive = any(str(e.get("active", True)).lower() != "true" for e in causal_graph.get("ego_edges", []))
        dl_path = out_dir / "dormant_qa.json"
        if not has_inactive:
            print(f"  SKIP Dormant: no inactive ego_edges in this scene")
            dl_result = {"questions": [], "usage": {}, "error": None}
            with open(dl_path, "w") as f:
                json.dump(dl_result, f, indent=2)
        elif dl_path.exists() and not args.force:
            print(f"  SKIP: {dl_path} exists (use --force)")
            dl_result = json.loads(dl_path.read_text())
        else:
            dl_result = generate_dormant_qa(scene_blocks, causal_graph, args.model, args.reasoning,
                                        prompt_version=pv, filtered_state=filtered_state)
            with open(dl_path, "w") as f:
                json.dump(dl_result, f, indent=2)
            print(f"  Saved → {dl_path}")

        n_dl = len(dl_result.get("questions", []))
        print(f"  Dormant questions: {n_dl}")
        if dl_result.get("error"):
            print(f"  ERROR: {str(dl_result['error'])[:200]}")

    # ══════════════════════════════════════════════════════════════════════
    #  Prompt 3: Distractor Nodes (v4 only)
    # ══════════════════════════════════════════════════════════════════════
    distractor_result = {"questions": [], "usage": {}, "error": None}
    if run_distractor:
        print(f"\n{'='*60}")
        print(f"  Prompt 3: Distractor Nodes")
        print(f"{'='*60}")

        has_distractors = bool(causal_graph.get("distractors"))
        distractor_path = out_dir / "distractor_qa.json"
        if not has_distractors:
            print(f"  SKIP Distractor: no distractors in this scene")
            distractor_result = {"questions": [], "usage": {}, "error": None}
            with open(distractor_path, "w") as f:
                json.dump(distractor_result, f, indent=2)
        elif distractor_path.exists() and not args.force:
            print(f"  SKIP: {distractor_path} exists (use --force)")
            distractor_result = json.loads(distractor_path.read_text())
        else:
            distractor_result = generate_distractor_qa(scene_blocks, causal_graph, args.model, args.reasoning,
                                                        prompt_version=pv, filtered_state=filtered_state)
            with open(distractor_path, "w") as f:
                json.dump(distractor_result, f, indent=2)
            print(f"  Saved → {distractor_path}")

        n_dist = len(distractor_result.get("questions", []))
        print(f"  Distractor questions: {n_dist}")
        if distractor_result.get("error"):
            print(f"  ERROR: {str(distractor_result['error'])[:200]}")

    # ══════════════════════════════════════════════════════════════════════
    #  Extract Rung 3 CF questions → run CF trajectory pipeline
    # ══════════════════════════════════════════════════════════════════════
    cf_results = []
    cf_questions = extract_cf_questions(active_result.get("questions", [])) if run_active else []

    if not args.skip_cf_pipeline and cf_questions:
        print(f"\n{'='*60}")
        print(f"  CF Trajectory Pipeline ({len(cf_questions)} Rung 3 questions)")
        print(f"{'='*60}")

        cf_out_dir = out_dir / "cf_trajectories"

        for i, cfq in enumerate(cf_questions):
            q_text = cfq["question"]
            print(f"\n  [{i+1}/{len(cf_questions)}] {q_text}")

            cf_result = run_trajectory_pipeline_for_question(
                question_text=q_text,
                scene_name=args.scene,
                records_dir=records_dir,
                graphs_dir=graphs_dir,
                out_dir=cf_out_dir,
                model=args.model,
                reasoning=args.reasoning,
                prompt_type="cf",
                force=args.force,
                cf_answer=str(cfq.get("correct_answer", "")),
                cf_reasoning=str(cfq.get("reasoning", "")),
                provider=args.provider,
            )
            cf_results.append({
                "active_question": cfq,
                "cf_pipeline_result": cf_result,
            })

        cf_results_path = out_dir / "rung3_cf_trajectories.json"
        with open(cf_results_path, "w") as f:
            json.dump(cf_results, f, indent=2)
        print(f"\n  Saved CF results → {cf_results_path}")
    elif not cf_questions:
        print(f"\n  No Rung 3 CF questions found in Active QA output.")
        # Clear stale CF artifacts from previous runs
        cf_results_path = out_dir / "rung3_cf_trajectories.json"
        if cf_results_path.exists():
            cf_results_path.unlink()
        cf_stale_dir = out_dir / "cf_trajectories"
        if cf_stale_dir.exists():
            import shutil
            shutil.rmtree(cf_stale_dir)
    else:
        print(f"\n  Skipping CF pipeline (--skip-cf-pipeline)")

    # ══════════════════════════════════════════════════════════════════════
    #  Extract AC-Binary (Yes) questions → run AC trajectory pipeline
    # ══════════════════════════════════════════════════════════════════════
    ac_results = []
    # For now, only run LLM trajectory pipeline for infrastructure-type activations
    # Agent-type activations will use nuPlan planner re-simulation (future)
    ac_questions = extract_ac_questions(dl_result.get("questions", []), entity_type="infrastructure") if run_dl else []

    if not args.skip_cf_pipeline and ac_questions:
        print(f"\n{'='*60}")
        print(f"  AC Trajectory Pipeline ({len(ac_questions)} AC-Binary Yes questions)")
        print(f"{'='*60}")

        ac_out_dir = out_dir / "ac_trajectories"

        for i, acq in enumerate(ac_questions):
            q_text = acq["question"]
            print(f"\n  [{i+1}/{len(ac_questions)}] {q_text}")

            ac_result = run_trajectory_pipeline_for_question(
                question_text=q_text,
                scene_name=args.scene,
                records_dir=records_dir,
                graphs_dir=graphs_dir,
                out_dir=ac_out_dir,
                model=args.model,
                reasoning=args.reasoning,
                prompt_type="ac",
                force=args.force,
                cf_answer=str(acq.get("correct_answer", "")),
                cf_reasoning=str(acq.get("reasoning", "")),
                provider=args.provider,
            )
            ac_results.append({
                "dl_question": acq,
                "ac_pipeline_result": ac_result,
            })

        ac_results_path = out_dir / "ac_binary_trajectories.json"
        with open(ac_results_path, "w") as f:
            json.dump(ac_results, f, indent=2)
        print(f"\n  Saved AC results → {ac_results_path}")
    elif not ac_questions:
        print(f"\n  No AC-Binary (Yes) questions found in Dormant output.")
        # Clear stale AC artifacts from previous runs
        ac_results_path = out_dir / "ac_binary_trajectories.json"
        if ac_results_path.exists():
            ac_results_path.unlink()
        ac_stale_dir = out_dir / "ac_trajectories"
        if ac_stale_dir.exists():
            import shutil
            shutil.rmtree(ac_stale_dir)
    else:
        print(f"\n  Skipping AC pipeline (--skip-cf-pipeline)")

    # ══════════════════════════════════════════════════════════════════════
    #  Combined output
    # ══════════════════════════════════════════════════════════════════════
    combined = {
        "scene": args.scene,
        "model": args.model,
        "active_qa": active_result.get("questions", []),
        "dormant_qa": dl_result.get("questions", []),
        "rung3_cf_questions": cf_questions,
        "rung3_cf_results": cf_results if cf_results else None,
        "ac_binary_questions": ac_questions,
        "ac_binary_results": ac_results if ac_results else None,
        "distractor_questions": distractor_result.get("questions", []),
        "usage": {
            "active": active_result.get("usage", {}),
            "dl": dl_result.get("usage", {}),
            "distractor": distractor_result.get("usage", {}),
        },
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(combined, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {args.scene}")
    print(f"{'='*60}")
    qs = active_result.get("questions", [])
    by_rung = {}
    for q in qs:
        r = q.get("rung", "?")
        by_rung[r] = by_rung.get(r, 0) + 1
    print(f"  Active QA: {len(qs)} questions  {dict(sorted(by_rung.items(), key=lambda x: str(x[0])))}")

    qs_dl = dl_result.get("questions", [])
    by_cat = {}
    for q in qs_dl:
        c = q.get("subtype", q.get("category", "?"))
        by_cat[c] = by_cat.get(c, 0) + 1
    print(f"  Dormant:       {len(qs_dl)} questions  {by_cat}")

    qs_dist = distractor_result.get("questions", [])
    print(f"  Distractor:    {len(qs_dist)} questions")

    print(f"  Rung 3 CF:     {len(cf_questions)} questions → {len(cf_results)} pipeline runs")
    print(f"  AC-Binary:     {len(ac_questions)} questions → {len(ac_results)} pipeline runs")

    u_active = active_result.get("usage", {})
    u_dl = dl_result.get("usage", {})
    u_dist = distractor_result.get("usage", {})
    total_in = u_active.get("input", 0) + u_dl.get("input", 0) + u_dist.get("input", 0)
    total_out = u_active.get("output", 0) + u_dl.get("output", 0) + u_dist.get("output", 0)
    total_img = u_active.get("image", 0) + u_dl.get("image", 0) + u_dist.get("image", 0)
    if total_in:
        cost = total_in * 2.50 / 1_000_000 + total_out * 12.50 / 1_000_000
        print(f"  Tokens: in={total_in:,} (image={total_img:,})  out={total_out:,}  ~${cost:.2f} (Q&A only)")

    print(f"  Output: {out_dir / 'result.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
