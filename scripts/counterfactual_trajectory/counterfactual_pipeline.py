"""
counterfactual_pipeline.py — Generate counterfactual trajectories for autonomous driving.

Flow:  Generate (LLM)  →  Visualize (code)  →  Save

Takes a causal scene graph + counterfactual question, generates a
counterfactual trajectory, and visualises it on BEV + front camera.

Usage:
  python counterfactual_pipeline.py \
      --records openscene_records \
      --graphs openscene_causal_graphs/v3 \
      --scene openscene-2021.05.12.23.36.44_veh-35_011_SAMPLED \
      --question "What if ego did not slow down for vehicle_1?" \
      --out counterfactual_results
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Provider instance — set in main()
_provider = None

# ── Constants ────────────────────────────────────────────────────────────

PROMPT_CAMS = ["cam_front", "cam_front_left", "cam_front_right", "cam_back"]
CAM_DISPLAY = {
    "cam_front": "CAM_FRONT",
    "cam_front_left": "CAM_FRONT_LEFT",
    "cam_front_right": "CAM_FRONT_RIGHT",
    "cam_back": "CAM_BACK",
}

# BEV constants (must match build scripts)
BEV_SIZE = 1536
BEV_RANGE = 50.0
BEV_SCALE = BEV_SIZE / (2 * BEV_RANGE)  # 15.36 px/m
BEV_CENTER = BEV_SIZE // 2               # 768

# Trajectory colors (BGR for OpenCV)
TRAJ_COLOR_BGR = (0, 0, 255)  # RED for counterfactual trajectory
ORIG_COLOR_BGR = (0, 200, 0)  # GREEN for original trajectory

CF_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "CounterFactualTrajectory"
AC_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "AC_Trajectory"

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


# ── Helpers ──────────────────────────────────────────────────────────────

def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def image_content_block(path: str, label: str | None = None) -> dict:
    if label:
        img = cv2.imread(str(path))
        if img is not None:
            cv2.putText(img, label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            _, buf = cv2.imencode(".jpg", img)
            b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")
            return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/png")
    b64 = load_image_b64(path)
    return {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}


def extract_json(raw: str) -> dict:
    raw = raw.strip()
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Extract from markdown fences — collect ALL valid blocks, return the last one
    lines = raw.split("\n")
    json_lines, inside = [], False
    last_valid = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") and not inside:
            inside = True
            json_lines = []
            continue
        elif stripped == "```" and inside:
            inside = False
            candidate = "\n".join(json_lines)
            try:
                last_valid = json.loads(candidate)
            except json.JSONDecodeError:
                pass
            json_lines = []
            continue
        if inside:
            json_lines.append(line)
    # Try any unclosed fence
    if json_lines:
        try:
            last_valid = json.loads("\n".join(json_lines))
        except json.JSONDecodeError:
            pass
    if last_valid is not None:
        return last_valid
    # Last resort: find outermost { ... } braces
    start = raw.find("{")
    if start != -1:
        depth, end = 0, start
        for i in range(start, len(raw)):
            if raw[i] == "{": depth += 1
            elif raw[i] == "}": depth -= 1
            if depth == 0:
                end = i + 1
                break
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    return {"_raw_response": raw}


def ego_local_to_bev(x, y):
    """Convert ego-local (x=fwd, y=left) to BEV pixel (col, row)."""
    px = int(BEV_CENTER - y * BEV_SCALE)
    py = int(BEV_CENTER - x * BEV_SCALE)
    return px, py


def project_to_cam(x, y, z, calib_cam):
    """Project ego-local 3D point to camera pixel. Returns (u, v) or None."""
    R = np.array(calib_cam["R_ego2cam"])
    t = np.array(calib_cam["t_ego2cam"])
    K = np.array(calib_cam["intrinsic"])

    pt_ego = np.array([x, y, z])
    pt_cam = R @ pt_ego + t

    if pt_cam[2] <= 0.1:
        return None

    u = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
    v = K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]
    return int(u), int(v)


# ── BEV + Camera Trajectory Drawing ─────────────────────────────────────

def draw_trajectory_on_bev(bev_img, waypoints, color, thickness=2, radius=4):
    """Draw waypoints on BEV image. waypoints: list of [x, y, heading, speed]."""
    pts = []
    for wp in waypoints:
        px, py = ego_local_to_bev(wp[0], wp[1])
        pts.append((px, py))
        cv2.circle(bev_img, (px, py), radius, color, -1)
    for i in range(len(pts) - 1):
        cv2.line(bev_img, pts[i], pts[i + 1], color, thickness)


def draw_trajectory_on_cam(cam_img, waypoints, color, calib_cam, thickness=2, radius=4):
    """Draw waypoints on camera image. waypoints: list of [x, y, heading, speed].
    Projects a ground corridor (left/right offset) for visibility at distance."""
    h, w = cam_img.shape[:2]
    pts = []
    left_pts = []
    right_pts = []
    corridor_half_w = 0.9  # meters half-width of drawn corridor

    for wp in waypoints:
        center = project_to_cam(wp[0], wp[1], 0.0, calib_cam)
        lp = project_to_cam(wp[0], wp[1] + corridor_half_w, 0.0, calib_cam)
        rp = project_to_cam(wp[0], wp[1] - corridor_half_w, 0.0, calib_cam)
        if center is not None and 0 <= center[0] < w and 0 <= center[1] < h:
            pts.append(center)
        if lp is not None and 0 <= lp[0] < w and 0 <= lp[1] < h:
            left_pts.append(lp)
        if rp is not None and 0 <= rp[0] < w and 0 <= rp[1] < h:
            right_pts.append(rp)

    # Draw filled corridor polygon (semi-transparent)
    if len(left_pts) >= 2 and len(right_pts) >= 2:
        poly = np.array(left_pts + right_pts[::-1], dtype=np.int32)
        overlay = cam_img.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.35, cam_img, 0.65, 0, cam_img)

    # Draw center line and circles
    for pt in pts:
        cv2.circle(cam_img, pt, radius, color, -1)
    for i in range(len(pts) - 1):
        cv2.line(cam_img, pts[i], pts[i + 1], color, thickness)


# ── Generate ─────────────────────────────────────────────────────────────

def format_trajectory_as_waypoints(trajectory: list) -> str:
    """Format ego.future list into labeled WP1-WP8 lines with computed heading."""
    times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    lines = []
    n = len(trajectory)
    for i, wp in enumerate(trajectory):
        x = wp.get("x", 0.0)
        y = wp.get("y", 0.0)
        speed = wp.get("speed", 0.0)
        # Compute heading from direction of travel
        if i < n - 1:
            dx = trajectory[i + 1].get("x", x) - x
            dy = trajectory[i + 1].get("y", y) - y
        elif i > 0:
            dx = x - trajectory[i - 1].get("x", x)
            dy = y - trajectory[i - 1].get("y", y)
        else:
            dx, dy = 1.0, 0.0
        heading = math.atan2(dy, dx)
        t = times[i] if i < len(times) else wp.get("t", (i + 1) * 0.5)
        lines.append(f"WP{i+1}: [{x:.4f}, {y:.4f}, {heading:.4f}, {speed:.4f}]  (T+{t}s)")
    return "\n".join(lines)


def generate(scene_dir: Path, state_text: str, causal_graph_text: str,
             original_trajectory: list, question: str, nav_command: str,
             model: str, reasoning: str, prompt_type: str = "cf",
             cf_answer: str = "", cf_reasoning: str = "") -> tuple[dict | None, dict]:
    """Call the LLM to generate a counterfactual trajectory.
    prompt_type: 'cf' for CR questions, 'ac' for AC-Binary questions."""
    prompts_dir = AC_PROMPTS_DIR if prompt_type == "ac" else CF_PROMPTS_DIR
    sys_file = "AC_Trajectory_Generator.md" if prompt_type == "ac" else "Generator_System_Prompt.md"
    user_file = "AC_User_Prompt.md" if prompt_type == "ac" else "Generator_Prompt.md"
    label = "AC" if prompt_type == "ac" else "CF"

    print(f"  [{label} Generate] Building content blocks ...")

    # Camera images
    tp0 = scene_dir / "Tp0p0"
    content_blocks = []
    for cam in PROMPT_CAMS:
        cam_label = CAM_DISPLAY[cam]
        img_path = tp0 / f"{cam}.jpg"
        if img_path.exists():
            content_blocks.append({"type": "input_text", "text": f"--- {cam_label} T0 ---"})
            content_blocks.append(image_content_block(str(img_path)))

    # BEV
    bev_path = tp0 / "bev.png"
    if bev_path.exists():
        content_blocks.append({"type": "input_text",
                               "text": f"--- BEV (Bird's Eye View at T0) ---\n{BEV_SPATIAL_TEXT}"})
        content_blocks.append(image_content_block(str(bev_path), label="BEV T0"))
    else:
        content_blocks.append({"type": "input_text", "text": "[BEV image not available]"})

    # Format original trajectory as labeled WP lines
    traj_formatted = format_trajectory_as_waypoints(original_trajectory)

    # User prompt — fill all placeholders
    user_prompt = (prompts_dir / user_file).read_text()
    user_prompt = user_prompt.replace("{state_json}", state_text)
    user_prompt = user_prompt.replace("{causal_graph_json}", causal_graph_text)
    user_prompt = user_prompt.replace("{nav_command}", nav_command)
    user_prompt = user_prompt.replace("{cf_question}", question)
    user_prompt = user_prompt.replace("{cf_answer}", cf_answer)
    user_prompt = user_prompt.replace("{cf_reasoning}", cf_reasoning)

    # Inject actual trajectory values into the WP1-WP8 section
    import re as _re
    wp_block_pattern = _re.compile(
        r'(=== ORIGINAL EGO TRAJECTORY.*?\n)(WP1:.*?WP8:.*?\n)',
        _re.DOTALL
    )
    if wp_block_pattern.search(user_prompt):
        user_prompt = wp_block_pattern.sub(
            lambda m: m.group(1) + traj_formatted + "\n", user_prompt
        )
    else:
        # Fallback: append formatted trajectory after the section header if pattern not found
        user_prompt = user_prompt.replace(
            "{original_trajectory}", traj_formatted
        )

    content_blocks.append({"type": "input_text", "text": user_prompt})

    system_prompt = (prompts_dir / sys_file).read_text()

    n_images = sum(1 for b in content_blocks if b.get("type") in ("input_image", "image"))
    print(f"  [{label} Generate] Calling {model}  ({n_images} images)  reasoning={reasoning} ...")

    raw, usage = _provider.call(system_prompt, content_blocks, max_tokens=65536)
    result = extract_json(raw)
    print(f"  [{label} Generate] Tokens: in={usage.get('input','?')} (image={usage.get('image',0)}) out={usage.get('output','?')}")

    if "_raw_response" in result:
        print(f"  [{label} Generate] ERROR: Invalid JSON response")
        return None, usage

    return result, usage


# ── Visualize ────────────────────────────────────────────────────────────

def visualize(scene_dir: Path, out_dir: Path, gen_output: dict,
              original_trajectory: list, calib: dict | None,
              temporal_anchor: str = "t0"):
    """Draw separate BEV and camera images for original and counterfactual trajectories."""
    print(f"  [Visualize] Drawing trajectories (anchor={temporal_anchor}) ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    cf_trajectory = gen_output.get("counterfactual_trajectory", [])
    calib_cam = (calib or {}).get("cam_front")

    # Original waypoints as [x, y, heading, speed]
    orig_waypoints = [[wp["x"], wp["y"], 0, wp.get("speed", 0)] for wp in original_trajectory]

    # Use T=-1.5 BEV for observation_window (alternate history), T=0 for t0
    if temporal_anchor == "observation_window":
        bev_path = scene_dir / "Tm1p5" / "bev.png"
    else:
        bev_path = scene_dir / "Tp0p0" / "bev.png"
    bev_base = cv2.imread(str(bev_path))
    if bev_base is None:
        bev_base = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.uint8)

    # BEV — original only (GREEN)
    bev_orig = bev_base.copy()
    draw_trajectory_on_bev(bev_orig, orig_waypoints, ORIG_COLOR_BGR, thickness=3, radius=5)
    cv2.imwrite(str(out_dir / "bev_original.png"), bev_orig)

    # BEV — counterfactual only (RED)
    if cf_trajectory:
        bev_cf = bev_base.copy()
        draw_trajectory_on_bev(bev_cf, cf_trajectory, TRAJ_COLOR_BGR, thickness=3, radius=5)
        cv2.imwrite(str(out_dir / "bev_counterfactual.png"), bev_cf)

    # BEV — combined (GREEN original + RED counterfactual)
    bev_combined = bev_base.copy()
    draw_trajectory_on_bev(bev_combined, orig_waypoints, ORIG_COLOR_BGR, thickness=3, radius=5)
    if cf_trajectory:
        draw_trajectory_on_bev(bev_combined, cf_trajectory, TRAJ_COLOR_BGR, thickness=2, radius=4)
    cv2.imwrite(str(out_dir / "bev_overlay.png"), bev_combined)

    # ── Front camera projections ──
    if temporal_anchor == "observation_window":
        cam_path = scene_dir / "Tm1p5" / "cam_front.jpg"
    else:
        cam_path = scene_dir / "Tp0p0" / "cam_front.jpg"
    cam_base = cv2.imread(str(cam_path))

    if cam_base is not None and calib_cam:
        # Camera — original only (GREEN)
        cam_orig = cam_base.copy()
        draw_trajectory_on_cam(cam_orig, orig_waypoints, ORIG_COLOR_BGR, calib_cam,
                               thickness=6, radius=12)
        cv2.imwrite(str(out_dir / "cam_original.png"), cam_orig)

        # Camera — counterfactual only (RED)
        if cf_trajectory:
            cam_cf = cam_base.copy()
            draw_trajectory_on_cam(cam_cf, cf_trajectory, TRAJ_COLOR_BGR, calib_cam,
                                   thickness=5, radius=10)
            cv2.imwrite(str(out_dir / "cam_counterfactual.png"), cam_cf)

        # Camera — combined
        cam_combined = cam_base.copy()
        draw_trajectory_on_cam(cam_combined, orig_waypoints, ORIG_COLOR_BGR, calib_cam,
                               thickness=6, radius=12)
        if cf_trajectory:
            draw_trajectory_on_cam(cam_combined, cf_trajectory, TRAJ_COLOR_BGR, calib_cam,
                                   thickness=5, radius=10)
        cv2.imwrite(str(out_dir / "cam_proj.png"), cam_combined)

    print(f"  [Visualize] Saved to {out_dir}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual trajectory pipeline (Generate → Visualize)")
    parser.add_argument("--records", required=True, help="Path to scenario_records directory")
    parser.add_argument("--graphs", required=True, help="Path to causal graphs directory (*_pruned.json)")
    parser.add_argument("--scene", required=True, help="Scene name to process")
    parser.add_argument("--question", required=True, help="Counterfactual question")
    parser.add_argument("--cf-answer", default="", help="Binary answer to the CF question (Yes/No)")
    parser.add_argument("--cf-reasoning", default="", help="Reasoning behind the CF answer")
    parser.add_argument("--out", default="counterfactual_results", help="Output directory")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "gemini", "claude"], help="LLM provider")
    parser.add_argument("--model", default="gpt-5.2", help="Model to use")
    parser.add_argument("--reasoning", default="none", choices=["none", "low", "medium", "high"])
    parser.add_argument("--prompt-type", default="cf", choices=["cf", "ac"],
                        help="Prompt type: cf=CR counterfactual, ac=AC activation")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    # Initialize provider
    global _provider
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common.llm_provider import get_provider
    from config_loader import load_config
    cfg = load_config() or {}

    api_key = None
    if args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or cfg.get("api_key") or None
    elif args.provider in ("gemini", "google"):
        api_key = (os.environ.get("GOOGLE_API_KEY")
                   or os.environ.get("GEMINI_API_KEY")
                   or cfg.get("google_api_key") or None)
    elif args.provider in ("claude", "anthropic"):
        api_key = (os.environ.get("ANTHROPIC_API_KEY")
                   or cfg.get("anthropic_api_key") or None)

    _provider = get_provider(args.provider, model=args.model, api_key=api_key,
                             reasoning=args.reasoning)

    label = "AC Trajectory" if args.prompt_type == "ac" else "Counterfactual"
    print(f"{label} Pipeline: Generate → Visualize")
    print(f"Provider: {args.provider}  |  Model: {args.model}  |  Reasoning: {args.reasoning}")
    print(f"Question: {args.question}")
    print()

    # ── Find scene directory ──
    records_dir = Path(args.records)
    scene_name = args.scene

    scene_dir = None
    for candidate in records_dir.iterdir():
        if not candidate.is_dir():
            continue
        dir_name = candidate.name
        maneuver_dirs = [d for d in candidate.iterdir() if d.is_dir()]
        for md in maneuver_dirs:
            full_name = f"{dir_name}_{md.name}"
            if full_name == scene_name or dir_name == scene_name:
                scene_dir = md
                break
        if scene_dir:
            break

    if scene_dir is None:
        print(f"ERROR: Could not find scene '{scene_name}' in {records_dir}")
        sys.exit(1)

    print(f"Scene directory: {scene_dir}")

    # ── Find causal graph ──
    graphs_dir = Path(args.graphs)
    graph_file = None
    for f in graphs_dir.glob("*_pruned.json"):
        if scene_name in f.stem or re.sub(r'_SAMPLED(_\d+)?$', '', scene_name) in f.stem:
            graph_file = f
            break

    if graph_file is None:
        print(f"ERROR: Could not find pruned causal graph for '{scene_name}' in {graphs_dir}")
        sys.exit(1)

    print(f"Causal graph: {graph_file}")

    # ── Output directory ──
    out_dir = Path(args.out) / scene_name
    if (out_dir / "result.json").exists() and not args.force:
        print(f"SKIP: {out_dir / 'result.json'} already exists (use --force)")
        sys.exit(0)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load scene data ──
    tp0_dir = scene_dir / "Tp0p0"
    state_json = json.loads((tp0_dir / "state.json").read_text())
    state_text = json.dumps(state_json, indent=2)

    causal_graph = json.loads(graph_file.read_text())
    causal_graph.pop("changes", None)
    causal_graph_text = json.dumps(causal_graph, indent=2)

    original_trajectory = state_json.get("ego", {}).get("future", [])
    nav_command = state_json.get("nav_command", "UNKNOWN")

    # Load camera calibration if available
    calib = None
    calib_path = tp0_dir / "calib.json"
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
    else:
        print("  WARNING: No calib.json found — camera projections will be skipped")

    # ── Generate ──
    print(f"\n{'='*60}")
    print(f"Scene: {scene_name}")
    print(f"{'='*60}")

    gen_output, usage = generate(
        scene_dir=scene_dir,
        state_text=state_text,
        causal_graph_text=causal_graph_text,
        original_trajectory=original_trajectory,
        question=args.question,
        nav_command=nav_command,
        model=args.model,
        reasoning=args.reasoning,
        prompt_type=args.prompt_type,
        cf_answer=args.cf_answer,
        cf_reasoning=args.cf_reasoning,
    )

    # ── Visualize ──
    if gen_output:
        anchor = gen_output.get("temporal_anchor", "t0")
        visualize(scene_dir, out_dir, gen_output, original_trajectory, calib,
                  temporal_anchor=anchor)

    # ── Save ──
    if gen_output:
        with open(out_dir / "generator.json", "w") as f:
            json.dump(gen_output, f, indent=2)
        print(f"  Generator output → {out_dir / 'generator.json'}")

    result = {
        "scene": scene_name,
        "question": args.question,
        "model": args.model,
        "generator": gen_output,
        "usage": usage,
        "error": None if gen_output else "Generator returned invalid JSON",
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Result → {out_dir / 'result.json'}")

    # ── Token summary ──
    total_in = usage.get("input", 0)
    total_out = usage.get("output", 0)
    total_img = usage.get("image", 0)
    print(f"\n{'='*60}")
    print(f"Tokens — in: {total_in:,} (image={total_img:,})  out: {total_out:,}")
    cost = total_in * 1.75 / 1_000_000 + total_out * 14.00 / 1_000_000
    print(f"Estimated cost: ${cost:.2f}")
    print(f"{'='*60}")

    if not gen_output:
        print(f"\nERROR: Generator returned invalid JSON")
        sys.exit(1)


if __name__ == "__main__":
    main()
