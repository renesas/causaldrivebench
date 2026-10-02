"""
Shared utilities for scene processing across batch and regular pipelines.

Content block formats:
  Internal (provider-agnostic):
    text  → {"type": "text", "text": "..."}
    image → {"type": "image", "mime": "image/jpeg", "data": "<base64>"}

  OpenAI Batch (Chat Completions):
    text  → {"type": "text", "text": "..."}
    image → {"type": "image_url", "image_url": {"url": "data:..."}}

  OpenAI Responses API:
    text  → {"type": "input_text", "text": "..."}
    image → {"type": "input_image", "image_url": "data:..."}
"""

import base64
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import cv2


# ── Content Block Formats ────────────────────────────────────────────────

INTERNAL = "internal"          # provider-agnostic (new default)
OPENAI_RESPONSES = "responses" # OpenAI Responses API (legacy for_batch=False)
OPENAI_BATCH = "batch"         # OpenAI Batch/Chat Completions (legacy for_batch=True)


def _resolve_format(for_batch: bool = False, fmt: str = None) -> str:
    """Resolve content block format from legacy for_batch flag or explicit fmt."""
    if fmt:
        return fmt
    return OPENAI_BATCH if for_batch else OPENAI_RESPONSES


# ── Image Processing ─────────────────────────────────────────────────────

def load_image_b64(path: str) -> str:
    """Load image as base64 string."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


MAX_IMAGE_DIM = 2000  # Claude multi-image request limit


def image_content_block(path: str, label: Optional[str] = None,
                       flip_horizontal: bool = False,
                       for_batch: bool = False,
                       fmt: str = None) -> Dict:
    """Create an image content block for API calls.

    fmt="internal"  → provider-agnostic: {"type":"image","mime":...,"data":...}
    fmt="batch"     → Chat Completions:  {"type":"image_url","image_url":{"url":"data:..."}}
    fmt="responses" → Responses API:     {"type":"input_image","image_url":"data:..."}
    for_batch       → legacy compat (True→batch, False→responses)

    Images exceeding MAX_IMAGE_DIM (2000px) are resized to fit.
    """
    fmt = _resolve_format(for_batch, fmt)

    img = cv2.imread(str(path))
    if img is not None:
        h, w = img.shape[:2]
        needs_resize = max(h, w) > MAX_IMAGE_DIM
        if label or flip_horizontal or needs_resize:
            if needs_resize:
                scale = MAX_IMAGE_DIM / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            if flip_horizontal:
                img = cv2.flip(img, 1)
            if label:
                cv2.putText(img, label, (10, 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            ext = Path(path).suffix.lower().lstrip(".")
            if ext == "png":
                _, buf = cv2.imencode(".png", img)
                b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")
                return _format_image_block(b64, "image/png", fmt)
            else:
                _, buf = cv2.imencode(".jpg", img)
                b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")
                return _format_image_block(b64, "image/jpeg", fmt)

    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/png")
    b64 = load_image_b64(path)
    return _format_image_block(b64, mime, fmt)


def _format_image_block(b64: str, mime: str, fmt: str) -> Dict:
    """Format an image block according to the target format."""
    if fmt == INTERNAL:
        return {"type": "image", "mime": mime, "data": b64}
    elif fmt == OPENAI_BATCH:
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    else:  # OPENAI_RESPONSES
        return {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}


# ── JSON Processing ──────────────────────────────────────────────────────

def extract_json(text: str) -> Dict:
    """Extract the last valid JSON object from an LLM response.

    Returns the LAST valid JSON found. Multi-phase prompts emit intermediate
    JSON blocks (e.g. a decision block) before the final authoritative output;
    always taking the last block gives the correct result for both single-block
    and multi-block responses.

    Priority order:
      1. Fenced code blocks (``` or ```json) — most reliable signal
      2. Bare outermost { } blocks — fallback for unfenced responses
    """
    candidates = []

    # Pass 1: fenced blocks — handles both ```json and ``` delimiters
    for m in re.finditer(r'```(?:json)?\s*\n?([\s\S]*?)\n?\s*```', text):
        try:
            candidates.append(json.loads(m.group(1).strip()))
        except (json.JSONDecodeError, ValueError):
            pass

    if candidates:
        return candidates[-1]

    # Pass 2: stack-based scan for bare JSON objects (O(n), handles nesting)
    stack, start = [], None
    in_string, escape = False, False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if not stack:
                start = i
            stack.append(ch)
        elif ch == '}' and stack:
            stack.pop()
            if not stack:
                try:
                    candidates.append(json.loads(text[start:i + 1]))
                except (json.JSONDecodeError, ValueError):
                    pass
                start = None

    if candidates:
        return candidates[-1]

    return {"_raw_response": text, "_error": "No valid JSON found in response"}


def format_state_json(state_json: Dict, ego_only: bool = False) -> str:
    """Format state.json for prompt inclusion."""
    if ego_only:
        reduced = {
            "nav_command": state_json.get("nav_command"),
            "ego": state_json.get("ego", {}),
            "n_agents": len(state_json.get("agents", {})),
        }
        return json.dumps(reduced, indent=2)
    return json.dumps(state_json, indent=2)


def filter_state_json(state_json: Dict, nodes: List[Dict]) -> Dict:
    """Filter state.json to only include agents referenced in the graph."""
    node_labels = {n.get("label", n.get("id", "")) for n in nodes}

    filtered_agents = {}
    for label, agent in state_json.get("agents", {}).items():
        if label in node_labels or any(label.lower() in nl.lower() for nl in node_labels):
            filtered_agents[label] = agent

    return {
        "nav_command": state_json.get("nav_command"),
        "ego": state_json.get("ego"),
        "agents": filtered_agents,
    }


# ── Natural Language State Encoding ─────────────────────────────────────

def _nl_speed_category(speed_mps: float) -> str:
    if speed_mps < 0.5:    return "stopped"
    elif speed_mps < 3.0:  return "slow"
    elif speed_mps < 7.0:  return "moderate speed"
    elif speed_mps < 12.0: return "fast speed"
    else:                  return "highway speed"


def _nl_motion_trend(speeds: list) -> str:
    """speeds: oldest → newest, with current speed appended as last entry."""
    if not speeds or len(speeds) < 2:
        return "constant speed"
    first, last = speeds[0], speeds[-1]
    if first <= 0.5 and last > 0.5:  return "starting"
    if first > 0.5  and last <= 0.5: return "stopping"
    if last - first > 0.5:           return "accelerating"
    if first - last > 0.5:           return "decelerating"
    return "constant speed"


def _nl_direction_word(x: float, y: float) -> str:
    """ego-local frame (+x=forward, +y=left). Returns 8-octant direction word."""
    if x == 0.0 and y == 0.0:
        return "at ego position"
    angle = math.atan2(y, x)
    abs_a = abs(angle)
    if abs_a < math.radians(22.5):                          return "ahead"
    if abs_a > math.radians(157.5):                         return "behind"
    if math.radians(67.5) < abs_a < math.radians(112.5):
        return "to the left" if y > 0 else "to the right"
    if abs_a < math.radians(67.5):
        return "ahead-left" if y > 0 else "ahead-right"
    return "behind-left" if y > 0 else "behind-right"


def _nl_distance_bucket(dist_m: float) -> str:
    if dist_m < 5:  return f"very close (~{dist_m:.0f}m)"
    if dist_m < 15: return f"close (~{dist_m:.0f}m)"
    if dist_m < 30: return f"~{dist_m:.0f}m away"
    if dist_m < 50: return f"far (~{dist_m:.0f}m)"
    return                  f"distant (~{dist_m:.0f}m)"


def _nl_heading_relation(yaw_rad: float, ax: float = 0.0, ay: float = 0.0) -> str:
    """Classify agent heading relative to ego.

    Two-step classification:

    1. Yaw thresholds select the broad bucket:
         |yaw| < 45°    aligned with ego   (same-dir / approaching)
         |yaw| > 135°   opposite to ego    (oncoming / moving-away)
         otherwise       lateral            (crossing)

    2. Dot product of heading · bearing refines aligned / opposite
       into ahead vs behind:
         bearing = (ax, ay) / dist   (ego → agent unit vector)
         heading = (cos yaw, sin yaw)
         dot = bearing · heading     (>0 receding, <0 closing)

    Labels (6):
        ahead and heading away from ego     |yaw|<45°,  dot >= 0
        behind and heading toward ego       |yaw|<45°,  dot <  0
        ahead and heading toward ego        |yaw|>135°, dot <= 0
        behind and heading away from ego    |yaw|>135°, dot >  0
        crossing right-to-left              45°<|yaw|<135°, yaw > 0
        crossing left-to-right              45°<|yaw|<135°, yaw < 0
    """
    yaw = (yaw_rad + math.pi) % (2 * math.pi) - math.pi
    abs_yaw = abs(yaw)

    # Dot product: heading · bearing  (>0 receding, <0 approaching)
    hx, hy = math.cos(yaw), math.sin(yaw)
    dist = math.hypot(ax, ay)
    dot = (ax * hx + ay * hy) / dist if dist > 0.5 else 0.0

    if abs_yaw < math.radians(45):                   # aligned with ego
        return "ahead and heading away from ego" if dot >= 0 \
          else "behind and heading toward ego"

    if abs_yaw > math.radians(135):                  # opposite to ego
        return "behind and heading away from ego" if dot > 0 \
          else "ahead and heading toward ego"

    # Lateral — crossing direction from yaw sign (+y = left in BEV)
    return "crossing right-to-left" if yaw > 0 else "crossing left-to-right"


def _nl_short_category(category: str) -> str:
    cat = category.lower()
    if "pedestrian" in cat or "human" in cat:              return "pedestrian"
    if "bicycle" in cat or "cyclist" in cat:               return "cyclist"
    if "motorcycle" in cat:                                return "motorcyclist"
    if "truck" in cat or "bus" in cat or "trailer" in cat: return "large vehicle"
    if "car" in cat or "vehicle" in cat:                   return "vehicle"
    return category


def state_to_natural_language(state_json: Dict, agents_filter: Optional[set] = None) -> str:
    """Convert state.json numerical data into a natural language scene description.

    Replaces the raw JSON block in LLM prompts so the model cannot shortcut by
    copying exact coordinates/speeds and must instead use visual inputs for
    spatial perception.

    Args:
        state_json:     Parsed state.json dict (ego + agents + nav_command).
        agents_filter:  If provided, only agents whose label is in this set are
                        included (used by Stage 3 filtered view). None = show all.
    """
    lines = ["=== SCENE STATE AT T=0 ==="]

    ego = state_json.get("ego", {}) or {}
    ego_speed = ego.get("speed", 0.0) or 0.0
    nav_command = (state_json.get("nav_command", "") or "").strip() or "UNKNOWN"
    hist_speeds = [h.get("speed", 0.0) for h in (ego.get("history") or []) if isinstance(h, dict)]
    ego_trend = _nl_motion_trend(hist_speeds + [ego_speed])
    lines.append(f"Ego: {_nl_speed_category(ego_speed)}, {ego_trend}. Navigation command: {nav_command}.")

    agents = state_json.get("agents", {}) or {}
    if not agents:
        lines.append("\nNo agents in scene.")
        return "\n".join(lines)
    lines.append("\nAgents:")

    for label, agent in agents.items():
        if agents_filter is not None and label not in agents_filter:
            continue
        pos = agent.get("position", {}) or {}
        ax     = pos.get("x", 0.0) or 0.0
        ay     = pos.get("y", 0.0) or 0.0
        a_yaw  = pos.get("yaw", 0.0) or 0.0
        a_speed = agent.get("speed", 0.0) or 0.0
        dist = math.sqrt(ax ** 2 + ay ** 2)

        ag_hist = agent.get("history")
        if ag_hist:
            ag_speeds = [h.get("speed", 0.0) for h in ag_hist if isinstance(h, dict)] + [a_speed]
            ag_trend = _nl_motion_trend(ag_speeds)
        else:
            ag_trend = ""

        trend_suffix = f", {ag_trend}" if ag_trend and ag_trend != "constant speed" else ""
        cat = agent.get("category", "unknown")
        lines.append(
            f"  - {label} [{cat}] ({_nl_short_category(cat)}): "
            f"{_nl_distance_bucket(dist)} {_nl_direction_word(ax, ay)}. "
            f"{_nl_heading_relation(a_yaw, ax, ay).capitalize()}, {_nl_speed_category(a_speed)}{trend_suffix}."
        )

    if agents_filter is not None:
        omitted = sum(1 for k in agents if k not in agents_filter)
        if omitted:
            lines.append(f"  ({omitted} additional agent(s) not relevant to this graph — omitted)")

    return "\n".join(lines)


# ── Scene Discovery ──────────────────────────────────────────────────────

def find_temporal_frames(scene_dir: Path) -> List[Optional[Path]]:
    """Find temporal frame directories (Tm1p5, Tm1p0, Tm0p5, Tp0p0).

    Returns a fixed-length list of 4 entries. Missing history frames are None;
    this matches the sequential v3_pipeline behaviour (NULL padding).
    """
    slugs = ["Tm1p5", "Tm1p0", "Tm0p5", "Tp0p0"]
    frames = []
    for slug in slugs:
        frame_dir = scene_dir / slug
        frames.append(frame_dir if frame_dir.exists() else None)
    return frames


def find_scene_directories(records_dir: Path, dataset: Optional[str] = None) -> List[Path]:
    """Find all valid SAMPLED_N scene directories in records."""
    scene_dirs = []

    for scene_path in sorted(records_dir.iterdir()):
        if not scene_path.is_dir() or scene_path.name == "overview":
            continue

        if dataset and not scene_path.name.startswith(f"{dataset}-"):
            continue

        # Find all SAMPLED_N subdirectories (multi-sample naming)
        for sampled_dir in sorted(scene_path.glob("SAMPLED_*")):
            if sampled_dir.is_dir() and (sampled_dir / "Tp0p0" / "state.json").exists():
                scene_dirs.append(sampled_dir)

    return scene_dirs


# ── Scene Content Preparation ────────────────────────────────────────────

def prepare_camera_blocks(scene_dir: Path,
                         cams: List[str] = None,
                         cam_labels: Dict[str, str] = None,
                         for_batch: bool = False,
                         fmt: str = None) -> List[Dict]:
    """Prepare camera image content blocks (camera-major ordering).

    Matches the sequential v3_pipeline layout:
      for each camera → for each temporal frame → image block
    with text separators between camera groups and NULL padding for missing frames.
    """
    if cams is None:
        cams = ["cam_front", "cam_front_left", "cam_front_right", "cam_back"]

    if cam_labels is None:
        cam_labels = {
            "cam_front": "CAM_FRONT",
            "cam_front_left": "CAM_FRONT_LEFT",
            "cam_front_right": "CAM_FRONT_RIGHT",
            "cam_back": "CAM_BACK",
        }

    temporal = find_temporal_frames(scene_dir)
    slot_labels = ["t-3", "t-2", "t-1", "t0"]
    camera_blocks = []

    for cam in cams:
        cam_label = cam_labels[cam]
        camera_blocks.append(text_block(
            f"--- {cam_label}: 4 temporal frames ({', '.join(slot_labels)}) ---",
            for_batch=for_batch, fmt=fmt))

        for j, frame_dir in enumerate(temporal):
            slot = slot_labels[j] if j < len(slot_labels) else f"t{j}"
            is_t0 = (slot == "t0")
            if frame_dir is None:
                camera_blocks.append(text_block(
                    f"[NULL: {cam.upper()} at {slot}]",
                    for_batch=for_batch, fmt=fmt))
                continue
            label = f"{cam_label} ({slot})"
            cam_path = frame_dir / f"{cam}.jpg"
            if not cam_path.exists():
                if is_t0:
                    raise FileNotFoundError(
                        f"Missing T0 camera image: {cam_path}\n"
                        f"Scene may be corrupt or build_records incomplete."
                    )
                camera_blocks.append(text_block(
                    f"[IMAGE MISSING: {cam.upper()} at {slot}]",
                    for_batch=for_batch, fmt=fmt))
                continue
            camera_blocks.append(image_content_block(
                str(cam_path), label=label, flip_horizontal=False,
                for_batch=for_batch, fmt=fmt))

    if not camera_blocks:
        raise RuntimeError(
            f"No camera images found in {scene_dir} — "
            f"check that build_records completed successfully."
        )

    return camera_blocks


def prepare_bev_blocks_v2(scene_dir: Path,
                          for_batch: bool = False,
                          fmt: str = None) -> List[Dict]:
    """Prepare 4 temporal BEV image blocks (t-3 to t0), all centered on T=0 ego.
    Used by v2 pipeline where BEV exists at each timestep.
    """
    temporal = find_temporal_frames(scene_dir)
    slot_labels = ["t-3", "t-2", "t-1", "t0"]
    bev_blocks = [text_block(
        f"--- BEV: 4 temporal frames ({', '.join(slot_labels)}), "
        f"all centered on T=0 ego position ---",
        for_batch=for_batch, fmt=fmt)]

    for j, frame_dir in enumerate(temporal):
        slot = slot_labels[j]
        if frame_dir is None:
            bev_blocks.append(text_block(
                f"[NULL: BEV at {slot}]", for_batch=for_batch, fmt=fmt))
            continue
        bev_path = frame_dir / "bev.png"
        if bev_path.exists():
            bev_blocks.append(image_content_block(
                str(bev_path), label=f"BEV ({slot})",
                for_batch=for_batch, fmt=fmt))
        else:
            bev_blocks.append(text_block(
                f"[IMAGE MISSING: BEV at {slot}]", for_batch=for_batch, fmt=fmt))
    return bev_blocks


def prepare_timestep_blocks(scene_dir: Path,
                            for_batch: bool = False,
                            fmt: str = None) -> List[Dict]:
    """Prepare 20 image blocks in timestep-major order (matching v5 prompt).

    Order: for each timestep → CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, BEV
    T=-1.5: images 1-5, T=-1.0: 6-10, T=-0.5: 11-15, T=0: 16-20
    """
    # Detect available cameras from T=0 frame
    tp0 = scene_dir / "Tp0p0"
    base_cams = ["cam_front", "cam_front_left", "cam_front_right"]
    # Detect rear cameras — AV2 has cam_back_right + cam_back_left, others have cam_back
    rear_cams = ["cam_back_right", "cam_back_left", "cam_back"]
    cams = base_cams + [c for c in rear_cams if (tp0 / f"{c}.jpg").exists()]
    cam_labels = {
        "cam_front": "CAM_FRONT",
        "cam_front_left": "CAM_FRONT_LEFT",
        "cam_front_right": "CAM_FRONT_RIGHT",
        "cam_back": "CAM_BACK",
        "cam_back_right": "CAM_BACK_RIGHT",
        "cam_back_left": "CAM_BACK_LEFT",
    }
    temporal = find_temporal_frames(scene_dir)
    ts_labels = ["T=-1.5", "T=-1.0", "T=-0.5", "T=0"]
    blocks = []

    for j, frame_dir in enumerate(temporal):
        ts = ts_labels[j]
        if frame_dir is None:
            for cam in cams:
                blocks.append(text_block(
                    f"[NULL: {cam_labels[cam]} at {ts}]",
                    for_batch=for_batch, fmt=fmt))
            blocks.append(text_block(
                f"[NULL: BEV at {ts}]",
                for_batch=for_batch, fmt=fmt))
            continue

        # 4 cameras for this timestep
        for cam in cams:
            cam_path = frame_dir / f"{cam}.jpg"
            label = f"{cam_labels[cam]} ({ts})"
            if cam_path.exists():
                blocks.append(image_content_block(
                    str(cam_path), label=label,
                    for_batch=for_batch, fmt=fmt))
            else:
                blocks.append(text_block(
                    f"[IMAGE MISSING: {cam_labels[cam]} at {ts}]",
                    for_batch=for_batch, fmt=fmt))

        # BEV for this timestep
        bev_path = frame_dir / "bev.png"
        label = f"BEV ({ts})"
        if bev_path.exists():
            blocks.append(image_content_block(
                str(bev_path), label=label,
                for_batch=for_batch, fmt=fmt))
        else:
            blocks.append(text_block(
                f"[IMAGE MISSING: BEV at {ts}]",
                for_batch=for_batch, fmt=fmt))

    return blocks


def prepare_camera_blocks_v2(scene_dir: Path,
                              cams: List[str] = None,
                              cam_labels: Dict[str, str] = None,
                              for_batch: bool = False,
                              fmt: str = None) -> List[Dict]:
    """Prepare 4 camera image blocks at T=0 only.
    Used by v2 pipeline where cameras only exist at Tp0p0.
    """
    if cams is None:
        cams = ["cam_front", "cam_front_left", "cam_front_right", "cam_back"]
    if cam_labels is None:
        cam_labels = {
            "cam_front": "CAM_FRONT",
            "cam_front_left": "CAM_FRONT_LEFT",
            "cam_front_right": "CAM_FRONT_RIGHT",
            "cam_back": "CAM_BACK",
        }

    tp0_dir = scene_dir / "Tp0p0"
    camera_blocks = [text_block(
        "--- Camera images at T=0 ---", for_batch=for_batch, fmt=fmt)]

    for cam in cams:
        cam_path = tp0_dir / f"{cam}.jpg"
        label = f"{cam_labels[cam]} (t0)"
        if cam_path.exists():
            camera_blocks.append(image_content_block(
                str(cam_path), label=label, for_batch=for_batch, fmt=fmt))
        else:
            camera_blocks.append(text_block(
                f"[IMAGE MISSING: {cam_labels[cam]} at t0]",
                for_batch=for_batch, fmt=fmt))
    return camera_blocks


def prepare_scene_content(scene_dir: Path,
                         include_graph: bool = False,
                         graph_path: Optional[Path] = None,
                         version: str = "v3",
                         layout: str = "v1",
                         for_batch: bool = False,
                         fmt: str = None) -> Dict[str, Any]:
    """Prepare all content for a scene (cameras, BEV, state.json, optional graph).

    layout="v1": 16 camera images (4 cams x 4 timesteps) + 1 BEV at T=0
    layout="v2": 4 temporal BEV images (t-3 to t0) + 4 camera images (T=0 only)
    """
    tp0_dir = scene_dir / "Tp0p0"

    state_path = tp0_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state.json: {state_path}")
    state_json = json.loads(state_path.read_text())

    if layout == "v2":
        bev_blocks = prepare_bev_blocks_v2(scene_dir, for_batch=for_batch, fmt=fmt)
        camera_blocks = prepare_camera_blocks_v2(scene_dir, for_batch=for_batch, fmt=fmt)
        bev_block = None  # not used in v2
    elif layout == "v5":
        # v5: timestep-major — 4 timesteps × (4 cams + 1 BEV) = 20 images
        # Order: T=-1.5(FRONT,FL,FR,BACK,BEV), T=-1.0(...), T=-0.5(...), T=0(...)
        timestep_blocks = prepare_timestep_blocks(scene_dir, for_batch=for_batch, fmt=fmt)
        camera_blocks = timestep_blocks  # all 20 images in one list
        bev_blocks = []  # already interleaved in camera_blocks
        bev_block = None
    else:
        camera_blocks = prepare_camera_blocks(scene_dir, for_batch=for_batch, fmt=fmt)
        bev_blocks = None  # not used in v1
        bev_path = tp0_dir / "bev.png"
        if bev_path.exists():
            bev_block = image_content_block(str(bev_path), for_batch=for_batch, fmt=fmt)
        else:
            bev_block = text_block("[BEV image not available]", for_batch=for_batch, fmt=fmt)

    # Load graph if needed
    graph = None
    if include_graph and graph_path and graph_path.exists():
        graph = json.loads(graph_path.read_text())

    desc_path = tp0_dir / "state_desc.json"
    if desc_path.exists():
        state_nl = json.loads(desc_path.read_text())["description"]
    else:
        state_nl = state_to_natural_language(state_json)

    return {
        "scene_name": scene_dir.parent.name,
        "maneuver":   scene_dir.name,
        "scene_id":   f"{scene_dir.parent.name}_{scene_dir.name}",
        "scene_dir":  str(scene_dir),
        "state_json": state_json,
        "state_text": format_state_json(state_json),
        "state_nl":   state_nl,
        "camera_blocks": camera_blocks,
        "bev_block":  bev_block,      # v1 only (single BEV)
        "bev_blocks": bev_blocks,     # v2 only (4 temporal BEVs)
        "nav_command": state_json.get("nav_command", ""),
        "graph": graph,
        "layout": layout,
        "n_cameras": len(camera_blocks),
        "has_bev": True,
    }


# ── Batch Request Formatting ─────────────────────────────────────────────

def create_batch_request(custom_id: str,
                        model: str,
                        system_prompt: str,
                        content_blocks: List[Dict],
                        max_tokens: int = 16384,
                        reasoning: Optional[str] = None) -> Dict:
    """Create a single batch request in OpenAI Chat Completions Batch API format.

    Content blocks must use Chat Completions format:
      text  → {"type": "text", "text": "..."}
      image → {"type": "image_url", "image_url": {"url": "data:..."}}
    Use for_batch=True in image_content_block() and text_block() to get the right format.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_blocks}
    ]

    body = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }

    if reasoning and model.startswith(("o1", "gpt-5")):
        body["reasoning_effort"] = reasoning

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body
    }


# ── Batch Response Processing ────────────────────────────────────────────

def parse_batch_response(response_line: str) -> Tuple[str, Dict, Optional[str]]:
    """Parse a single line from batch API response file.
    Returns: (custom_id, result_dict, error_msg)
    """
    try:
        data = json.loads(response_line)
        custom_id = data.get("custom_id", "")

        if data.get("error"):
            return custom_id, {}, data["error"].get("message", "Unknown error")

        response = data.get("response", {})
        if response.get("status_code") != 200:
            return custom_id, {}, f"HTTP {response.get('status_code')}"

        # Extract the actual content
        body = response.get("body", {})
        choices = body.get("choices", [])
        if not choices:
            return custom_id, {}, "No choices in response"

        content = choices[0].get("message", {}).get("content", "")
        result = extract_json(content)

        # Wrap lists so we can attach metadata
        if isinstance(result, list):
            result = {"_items": result}

        # Add token usage if available
        if "usage" in body:
            result["_usage"] = body["usage"]

        return custom_id, result, None

    except Exception as e:
        return "", {}, str(e)


# ── Scene State Management ───────────────────────────────────────────────

def save_scene_state(scene_dir: Path, state_file: str, data: Dict):
    """Save intermediate state for a scene."""
    state_dir = Path(scene_dir) / ".batch_state"
    state_dir.mkdir(exist_ok=True)

    state_path = state_dir / state_file
    with open(state_path, "w") as f:
        json.dump(data, f, indent=2)


def load_scene_state(scene_dir: Path, state_file: str) -> Optional[Dict]:
    """Load intermediate state for a scene."""
    state_path = Path(scene_dir) / ".batch_state" / state_file
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return None


# ── Prompt Loading ────────────────────────────────────────────────────────

def load_prompt(prompt_path: Path, replacements: Dict[str, str] = None) -> str:
    """Load and optionally format a prompt file."""
    prompt = prompt_path.read_text()

    if replacements:
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)

    return prompt


def text_block(text: str, for_batch: bool = False, fmt: str = None) -> Dict:
    """Create a text content block in the correct format for the target API.

    fmt="internal"  → {"type":"text","text":"..."}
    fmt="batch"     → {"type":"text","text":"..."}
    fmt="responses" → {"type":"input_text","text":"..."}
    """
    fmt = _resolve_format(for_batch, fmt)
    if fmt == OPENAI_RESPONSES:
        return {"type": "input_text", "text": text}
    return {"type": "text", "text": text}


def get_raw_frames(sample_dir: Path, cfg: dict, data_root: str = None) -> dict:
    """
    Build a frames dict {timestep: {cam_name: path}} using the original sensor
    images from the dataset dataroot rather than the annotated copies in the
    records directory.

    Args:
        sample_dir: Path to a SAMPLED_N directory (contains meta.json).
        cfg:        Top-level config dict (from load_config()).
        data_root:  If given, paths are returned relative to this root.
                    If None, absolute paths are returned.

    Returns:
        Dict mapping timestep slug → {cam_name → path string}.
    """
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {sample_dir}")

    meta = json.loads(meta_path.read_text())

    dataset = meta.get("dataset", "")
    if dataset == "nuscenes":
        return _raw_frames_nuscenes(meta, cfg, data_root)
    elif dataset == "openscene":
        return _raw_frames_openscene(meta, cfg, data_root)
    elif dataset == "argoverse2":
        return _raw_frames_argoverse2(meta, cfg, data_root)
    raise ValueError(f"Unknown dataset '{dataset}' in {meta_path}")


# Temporal offsets shared by all datasets (matches build_*_records.py)
_TEMPORAL_OFFSETS = [(-3, "Tm1p5"), (-2, "Tm1p0"), (-1, "Tm0p5"), (0, "Tp0p0")]


def _make_relative(abs_path: Path, data_root: str = None) -> str:
    """Return path string, relative to data_root if given."""
    if data_root:
        root = Path(data_root)
        try:
            return str(abs_path.relative_to(root))
        except ValueError:
            return str(abs_path)
    return str(abs_path)


_nuscenes_cache = {}  # keyed by ver_dir string → (samples, sd_index)

def _raw_frames_nuscenes(meta: dict, cfg: dict, data_root: str = None) -> dict:
    ns_cfg   = cfg.get("nuscenes", {})
    dataroot = ns_cfg.get("data", "")
    version  = ns_cfg.get("version", "v1.0-trainval")
    if not dataroot:
        return {}

    ver_dir = Path(dataroot) / version
    cache_key = str(ver_dir)

    if cache_key in _nuscenes_cache:
        samples, sd_index = _nuscenes_cache[cache_key]
    else:
        try:
            samples  = {s["token"]: s for s in json.loads((ver_dir / "sample.json").read_text())}
            sd_index: Dict[str, Dict[str, str]] = {}
            for sd in json.loads((ver_dir / "sample_data.json").read_text()):
                if not sd["is_key_frame"]:
                    continue
                tok   = sd["sample_token"]
                fname = sd["filename"]
                parts = fname.split("/")
                if len(parts) < 2:
                    continue
                channel = parts[1]
                sd_index.setdefault(tok, {})[channel] = fname
            _nuscenes_cache[cache_key] = (samples, sd_index)
        except Exception:
            return {}

    CAM_MAP = {
        "CAM_FRONT":       "cam_front",
        "CAM_FRONT_LEFT":  "cam_front_left",
        "CAM_FRONT_RIGHT": "cam_front_right",
        "CAM_BACK":        "cam_back",
    }
    TIMESTEPS = ["Tp0p0", "Tm0p5", "Tm1p0", "Tm1p5"]

    t0_token = meta.get("t0_sample_token", "")
    if not t0_token or t0_token not in samples:
        return {}

    frames: Dict[str, Dict[str, str]] = {}
    tok = t0_token
    for ts in TIMESTEPS:
        chans = sd_index.get(tok, {})
        frame = {}
        for ch, fname in chans.items():
            if ch not in CAM_MAP:
                continue
            frame[CAM_MAP[ch]] = _make_relative(Path(dataroot) / fname, data_root)
        if frame:
            frames[ts] = frame
        tok = samples.get(tok, {}).get("prev", "")
        if not tok:
            break
    return frames


def _raw_frames_openscene(meta: dict, cfg: dict, data_root: str = None) -> dict:
    import pickle

    os_cfg   = cfg.get("openscene", {})
    blobs    = os_cfg.get("blobs", "")
    pkl_path = meta.get("pkl_path", "")
    t0_idx   = meta.get("t0_frame_idx")
    if not blobs or not pkl_path or t0_idx is None:
        return {}

    CAM_MAP = {
        "CAM_F0": "cam_front",
        "CAM_L0": "cam_front_left",
        "CAM_R0": "cam_front_right",
        "CAM_B0": "cam_back",
    }

    try:
        with open(pkl_path, "rb") as f:
            pkl_frames = pickle.load(f)
    except Exception:
        return {}

    frames: Dict[str, Dict[str, str]] = {}
    for offset, slug in _TEMPORAL_OFFSETS:
        fidx = t0_idx + offset
        if fidx < 0 or fidx >= len(pkl_frames):
            continue
        pf = pkl_frames[fidx]
        frame = {}
        for os_cam, our_cam in CAM_MAP.items():
            if os_cam not in pf.get("cams", {}):
                continue
            data_path = pf["cams"][os_cam].get("data_path", "")
            if not data_path:
                continue
            abs_path = Path(blobs) / data_path
            frame[our_cam] = _make_relative(abs_path, data_root)
        if frame:
            frames[slug] = frame
    return frames


def _raw_frames_argoverse2(meta: dict, cfg: dict, data_root: str = None) -> dict:
    import bisect

    raw_data_dir = meta.get("raw_data_dir", "")
    t0_idx       = meta.get("t0_frame_idx")
    if not raw_data_dir or t0_idx is None:
        return {}

    scene_dir = Path(raw_data_dir)

    CAM_MAP = {
        "ring_front_center": "cam_front",
        "ring_front_left":   "cam_front_left",
        "ring_front_right":  "cam_front_right",
        "ring_rear_right":   "cam_back_right",
        "ring_rear_left":    "cam_back_left",
    }

    # Load annotation timestamps and subsample to 2Hz (same as build_argoverse2_records.py)
    import pandas as pd
    try:
        ann_df = pd.read_feather(scene_dir / "annotations.feather")
        ann_timestamps = sorted(ann_df.timestamp_ns.unique())
        ann_2hz = ann_timestamps[::5]  # 10Hz → 2Hz
    except Exception:
        return {}

    # Camera timestamps per channel
    cam_timestamps: Dict[str, list] = {}
    for av_cam in CAM_MAP:
        cam_dir = scene_dir / "sensors" / "cameras" / av_cam
        if cam_dir.exists():
            cam_timestamps[av_cam] = sorted(int(f.stem) for f in cam_dir.glob("*.jpg"))
        else:
            cam_timestamps[av_cam] = []

    frames: Dict[str, Dict[str, str]] = {}
    for offset, slug in _TEMPORAL_OFFSETS:
        ts_idx = t0_idx + offset
        if ts_idx < 0 or ts_idx >= len(ann_2hz):
            continue
        target_ns = ann_2hz[ts_idx]
        frame = {}
        for av_cam, our_cam in CAM_MAP.items():
            cts = cam_timestamps.get(av_cam, [])
            if not cts:
                continue
            # Find closest camera timestamp (same logic as build_argoverse2_records.py)
            idx = bisect.bisect_left(cts, target_ns)
            candidates = []
            if idx > 0:
                candidates.append(cts[idx - 1])
            if idx < len(cts):
                candidates.append(cts[idx])
            best = min(candidates, key=lambda t: abs(t - target_ns))
            if abs(best - target_ns) > 100 * 1e6:  # 100ms threshold
                continue
            abs_path = scene_dir / "sensors" / "cameras" / av_cam / f"{best}.jpg"
            frame[our_cam] = _make_relative(abs_path, data_root)
        if frame:
            frames[slug] = frame
    return frames



def format_trajectory_as_waypoints(trajectory: list) -> str:
    """Format ego.future list [{t,x,y,speed},...] as labeled WP1-WP8 lines with heading."""
    times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    lines = []
    n = len(trajectory)
    for i, wp in enumerate(trajectory):
        x = wp.get("x", 0.0)
        y = wp.get("y", 0.0)
        speed = wp.get("speed", 0.0)
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