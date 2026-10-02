#!/usr/bin/env python3
"""
trajectory_utils.py — BEV intervention rendering using map-only canvas

Loads bev_map.png (map without agents), then redraws agents with modifications.
No inpainting — pixel-perfect results matching the original record builders.

Three intervention types:
  REMOVE:       Skip the target agent, draw all others
  REPOSITION:   Draw target at new position (red), others at original
  SPEED_CHANGE: Draw target trajectory dots (magenta), others at original

Also draws ego original trajectory (green) and CF trajectory (red) as dot overlays.

Usage:
    from trajectory_utils import render_intervention_bev, draw_ego_trajectories
"""

import math
import json
from pathlib import Path

import cv2
import numpy as np

# ── BEV Constants (must match record builders) ───────────────────────────
BEV_SIZE = 1536
BEV_RANGE = 50.0
BEV_SUPERSAMPLE = 2
BEV_SCALE = BEV_SIZE / (2 * BEV_RANGE)  # 15.36 px/m at final resolution
BEV_CENTER = BEV_SIZE // 2

# ── Agent Colors (BGR, match record builders) ────────────────────────────
COLOR_EGO = (255, 200, 0)
COLOR_VEHICLE = (180, 200, 255)
COLOR_PED = (0, 255, 128)
COLOR_CYCLE = (0, 200, 200)

# ── Intervention Colors ──────────────────────────────────────────────────
COLOR_INTERVENTION = (0, 0, 255)    # red — repositioned agent
COLOR_SPEED_DOTS = (255, 0, 255)    # magenta — speed change trajectory
COLOR_EGO_ORIG = (0, 255, 0)       # green — ego original trajectory
COLOR_EGO_CF = (0, 0, 255)         # red — ego counterfactual trajectory


def _agent_color(category):
    cat = (category or "").lower()
    if "pedestrian" in cat or "human" in cat:
        return COLOR_PED
    if "bicycle" in cat or "motorcycle" in cat or "cycle" in cat:
        return COLOR_CYCLE
    return COLOR_VEHICLE


def _agent_dims(category):
    """Return (length, width) in metres."""
    cat = (category or "").lower()
    if "pedestrian" in cat or "human" in cat:
        return (0.8, 0.8)
    if "bicycle" in cat or "motorcycle" in cat or "cycle" in cat:
        return (2.0, 0.8)
    return (4.5, 2.0)


def ego_to_bev(x, y):
    """Ego-local coords → BEV pixel (col, row)."""
    px = int(BEV_CENTER - y * BEV_SCALE)
    py = int(BEV_CENTER - x * BEV_SCALE)
    return px, py


def _draw_box(canvas, x, y, l, w, yaw, color, speed=0.0, label="", fill=False):
    """Draw agent box on BEV canvas. Replicates record builder logic exactly."""
    cos_a, sin_a = math.cos(yaw), math.sin(yaw)
    l2, w2 = l / 2, w / 2
    corners = [
        (x + cos_a*l2 - sin_a*w2, y + sin_a*l2 + cos_a*w2),
        (x + cos_a*l2 + sin_a*w2, y + sin_a*l2 - cos_a*w2),
        (x - cos_a*l2 + sin_a*w2, y - sin_a*l2 - cos_a*w2),
        (x - cos_a*l2 - sin_a*w2, y - sin_a*l2 + cos_a*w2),
    ]
    bev_corners = np.array([ego_to_bev(cx, cy) for cx, cy in corners], dtype=np.int32)

    if fill:
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [bev_corners.reshape(-1, 1, 2)], color)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
    cv2.polylines(canvas, [bev_corners.reshape(-1, 1, 2)], True, color, 1)

    # Front edge highlight
    front_pts = np.array([ego_to_bev(*corners[0]), ego_to_bev(*corners[1])], dtype=np.int32)
    cv2.polylines(canvas, [front_pts.reshape(-1, 1, 2)], False,
                  tuple(min(255, c + 80) for c in color), 2)

    # Heading arrow (only for moving agents)
    if speed > 0.1:
        fcx = int((bev_corners[0][0] + bev_corners[1][0]) / 2)
        fcy = int((bev_corners[0][1] + bev_corners[1][1]) / 2)
        bcx = int((bev_corners[2][0] + bev_corners[3][0]) / 2)
        bcy = int((bev_corners[2][1] + bev_corners[3][1]) / 2)
        dx, dy = fcx - bcx, fcy - bcy
        h_len = max(1, math.sqrt(dx**2 + dy**2))
        arrow_len = 18
        ax = int(fcx + dx / h_len * arrow_len)
        ay = int(fcy + dy / h_len * arrow_len)
        arrow_color = tuple(min(255, c + 60) for c in color)
        cv2.arrowedLine(canvas, (fcx, fcy), (ax, ay), arrow_color, 2, tipLength=0.35)

    # Label
    if label:
        bcx = int(np.mean(bev_corners[:, 0]))
        bcy = int(np.min(bev_corners[:, 1])) - 4
        cv2.putText(canvas, label, (bcx - 4, bcy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def _draw_dots(canvas, waypoints, color, radius=5, thickness=2):
    """Draw trajectory dots + connecting line."""
    pts = []
    for wp in waypoints:
        if isinstance(wp, dict):
            x, y = wp.get("x", 0), wp.get("y", 0)
        elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
            x, y = wp[0], wp[1]
        else:
            continue
        px, py = ego_to_bev(x, y)
        pts.append((px, py))
        cv2.circle(canvas, (px, py), radius, color, -1)
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)


# ── Main API ─────────────────────────────────────────────────────────────

def render_intervention_bev(bev_map_path, state_json, intervention_config,
                            ego_speed=0.0, skip_agents=None):
    """Render a fresh BEV with agents, applying interventions.

    Args:
        bev_map_path: Path to bev_map.png (map-only canvas)
        state_json: Parsed state.json dict
        intervention_config: {agent_id: {action, change_state, ...}}
        ego_speed: Ego speed for heading arrow
        skip_agents: Set of agent_ids to skip entirely (for REMOVE)

    Returns: modified BEV image (numpy array)
    """
    canvas = cv2.imread(str(bev_map_path))
    if canvas is None:
        raise FileNotFoundError(f"bev_map.png not found: {bev_map_path}")

    skip = skip_agents or set()
    agents = state_json.get("agents", {})

    # Clean intervention config
    interventions = {k: v for k, v in intervention_config.items()
                     if isinstance(v, dict) and v.get("action") in ("remove", "reposition", "speed_change")}

    # Collect agents to remove
    for agent_id, config in interventions.items():
        if config["action"] == "remove":
            skip.add(agent_id)

    # Draw ego box
    _draw_box(canvas, 0, 0, 4.5, 2.0, 0, COLOR_EGO,
              speed=ego_speed, label="EGO", fill=True)

    # Draw all non-intervened agents at original positions
    for agent_id, agent in agents.items():
        if agent_id in skip:
            continue
        if agent_id in interventions and interventions[agent_id]["action"] != "remove":
            continue  # will be drawn separately below

        pos = agent.get("position", {})
        cat = agent.get("category", "vehicle")
        lm, wm = _agent_dims(cat)
        color = _agent_color(cat)
        _draw_box(canvas, pos.get("x", 0), pos.get("y", 0), lm, wm,
                  pos.get("yaw", 0), color, speed=agent.get("speed", 0),
                  label=agent_id)

    # Draw intervened agents
    for agent_id, config in interventions.items():
        action = config["action"]
        agent = agents.get(agent_id, {})
        cat = agent.get("category", "vehicle")
        lm, wm = _agent_dims(cat)
        pos = agent.get("position", {})

        if action == "remove":
            pass  # already skipped

        elif action == "reposition":
            cs = config.get("change_state", {})
            _draw_box(canvas, cs.get("x", 0), cs.get("y", 0), lm, wm,
                      cs.get("yaw", 0), COLOR_INTERVENTION,
                      speed=cs.get("speed", 0), label=f"{agent_id} [MOVED]", fill=True)

        elif action == "speed_change":
            cs = config.get("change_state", {})
            trajectory = cs.get("trajectory", [])

            if not trajectory:
                # Single state — compute straight-line future dots
                new_speed = cs.get("speed", 0)
                yaw = cs.get("yaw", pos.get("yaw", 0))
                ox, oy = pos.get("x", 0), pos.get("y", 0)
                trajectory = []
                for i in range(1, 9):
                    t = i * 0.5
                    trajectory.append({
                        "t": t,
                        "x": ox + new_speed * math.cos(yaw) * t,
                        "y": oy + new_speed * math.sin(yaw) * t,
                        "yaw": yaw, "speed": new_speed
                    })

            # Draw trajectory dots
            _draw_dots(canvas, trajectory, COLOR_SPEED_DOTS, radius=6, thickness=2)

            # Draw agent at final position
            if trajectory:
                last = trajectory[-1]
                lx = last["x"] if isinstance(last, dict) else last[0]
                ly = last["y"] if isinstance(last, dict) else last[1]
                lyaw = (last.get("yaw", pos.get("yaw", 0))
                        if isinstance(last, dict) else pos.get("yaw", 0))
                lspeed = (last.get("speed", 0)
                          if isinstance(last, dict) else 0)
                _draw_box(canvas, lx, ly, lm, wm, lyaw, COLOR_SPEED_DOTS,
                          speed=lspeed, label=f"{agent_id} [SPEED]")

    # Add legend
    _add_legend(canvas, interventions)

    return canvas


def draw_ego_trajectories(bev_img, original_wps=None, cf_wps=None):
    """Draw ego trajectories on a BEV image. Returns modified copy."""
    canvas = bev_img.copy()
    if original_wps:
        _draw_dots(canvas, original_wps, COLOR_EGO_ORIG, radius=5, thickness=3)
    if cf_wps:
        _draw_dots(canvas, cf_wps, COLOR_EGO_CF, radius=4, thickness=2)
    return canvas


def _add_legend(canvas, interventions):
    """Add intervention legend to bottom of BEV."""
    h = canvas.shape[0]
    for agent_id, config in interventions.items():
        action = config["action"].upper()
        txt = f"INTERVENTION: {action} {agent_id}"
        cv2.putText(canvas, txt, (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        break  # just show first
    cv2.putText(canvas, "Green=ego original | Red=ego CF | Magenta=agent speed",
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (200, 200, 200), 1, cv2.LINE_AA)


# ── Multi-Frame Rendering (Past / Observation Window) ────────────────────

TIMESTEP_SLUGS = [("Tm1p5", -1.5), ("Tm1p0", -1.0), ("Tm0p5", -0.5), ("Tp0p0", 0.0)]


def _get_agent_pos_at_t(agent, t):
    """Get agent position at timestep t from history or T=0 position."""
    if t == 0.0:
        pos = agent.get("position", {})
        return pos.get("x", 0), pos.get("y", 0), pos.get("yaw", 0), agent.get("speed", 0)
    for h in agent.get("history", []):
        if abs(h["t"] - t) < 0.01:
            return h.get("x", 0), h.get("y", 0), h.get("yaw", agent.get("position", {}).get("yaw", 0)), h.get("speed", 0)
    return None


def _compute_speed_change_pos_at_t(agent, config, t):
    """For speed_change: compute where agent would be at time t with new speed."""
    cs = config.get("change_state", {})
    trajectory = cs.get("trajectory", [])

    # If trajectory has entries, find closest to t
    if trajectory:
        for wp in trajectory:
            if abs(wp["t"] - t) < 0.01:
                return wp.get("x", 0), wp.get("y", 0), wp.get("yaw", 0), wp.get("speed", 0)
        # Interpolate or use closest
        closest = min(trajectory, key=lambda wp: abs(wp["t"] - t))
        return closest.get("x", 0), closest.get("y", 0), closest.get("yaw", 0), closest.get("speed", 0)

    # Single state — linear extrapolation from T=-1.5
    new_speed = cs.get("speed", 0)
    yaw = cs.get("yaw", agent.get("position", {}).get("yaw", 0))
    # Start from T=-1.5 position
    hist = agent.get("history", [])
    if hist:
        start = hist[0]
        sx, sy = start.get("x", 0), start.get("y", 0)
    else:
        pos = agent.get("position", {})
        sx, sy = pos.get("x", 0), pos.get("y", 0)
    dt = t - (-1.5)
    return sx + new_speed * math.cos(yaw) * dt, sy + new_speed * math.sin(yaw) * dt, yaw, new_speed


def render_intervention_multi_frame(scene_dir, state_json, intervention_config, ego_speed=0.0):
    """Render 4 modified BEVs (T=-1.5 to T=0) for observation_window interventions.

    Returns list of 4 (slug, modified_bev) tuples.
    """
    agents = state_json.get("agents", {})
    interventions = {k: v for k, v in intervention_config.items()
                     if isinstance(v, dict) and v.get("action") in ("remove", "reposition", "speed_change")}

    ego_history = state_json.get("ego", {}).get("history", [])
    results = []

    for slug, t_val in TIMESTEP_SLUGS:
        bev_map_path = scene_dir / slug / "bev_map.png"
        if not bev_map_path.exists():
            continue

        canvas = cv2.imread(str(bev_map_path))
        if canvas is None:
            continue

        # Determine ego position at this timestep
        if t_val == 0.0:
            ego_x, ego_y, ego_yaw = 0, 0, 0
            ego_spd = ego_speed
        else:
            ego_pos = None
            for h in ego_history:
                if abs(h["t"] - t_val) < 0.01:
                    ego_pos = h
                    break
            if ego_pos:
                ego_x, ego_y = ego_pos.get("x", 0), ego_pos.get("y", 0)
                ego_yaw = 0  # ego history doesn't have yaw, assume forward
                ego_spd = ego_pos.get("speed", 0)
            else:
                ego_x, ego_y, ego_yaw, ego_spd = 0, 0, 0, 0

        # Draw ego
        _draw_box(canvas, ego_x, ego_y, 4.5, 2.0, ego_yaw, COLOR_EGO,
                  speed=ego_spd, label="EGO", fill=True)

        # Collect which agents to skip (REMOVE)
        skip = set()
        for agent_id, config in interventions.items():
            if config["action"] == "remove":
                skip.add(agent_id)

        # Draw non-intervened agents at their position for this timestep
        for agent_id, agent in agents.items():
            if agent_id in skip or agent_id in interventions:
                continue
            pos_at_t = _get_agent_pos_at_t(agent, t_val)
            if pos_at_t is None:
                continue
            ax, ay, ayaw, aspeed = pos_at_t
            cat = agent.get("category", "vehicle")
            lm, wm = _agent_dims(cat)
            color = _agent_color(cat)
            _draw_box(canvas, ax, ay, lm, wm, ayaw, color,
                      speed=aspeed, label=agent_id)

        # Draw intervened agents
        for agent_id, config in interventions.items():
            action = config["action"]
            agent = agents.get(agent_id, {})
            cat = agent.get("category", "vehicle")
            lm, wm = _agent_dims(cat)

            if action == "remove":
                pass  # already skipped

            elif action == "reposition":
                cs = config.get("change_state", {})
                _draw_box(canvas, cs.get("x", 0), cs.get("y", 0), lm, wm,
                          cs.get("yaw", 0), COLOR_INTERVENTION,
                          speed=cs.get("speed", 0),
                          label=f"{agent_id} [MOVED]", fill=True)

            elif action == "speed_change":
                pos = _compute_speed_change_pos_at_t(agent, config, t_val)
                if pos:
                    ax, ay, ayaw, aspeed = pos
                    _draw_box(canvas, ax, ay, lm, wm, ayaw, COLOR_SPEED_DOTS,
                              speed=aspeed, label=f"{agent_id} [SPEED]")

        # Draw ego original trajectory (green dots) — full history
        _draw_dots(canvas, ego_history, COLOR_EGO_ORIG, radius=5, thickness=3)

        # Legend
        _add_legend(canvas, interventions)

        # Timestamp label
        label = f"T={t_val:+.1f}s" if t_val != 0 else "T=0 (now)"
        cv2.putText(canvas, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        results.append((slug, canvas))

    return results


# ── Batch Processing ─────────────────────────────────────────────────────

def process_cf_result(result_path, records_dir, force=False):
    """Process a single CF result.json — render intervention BEV from bev_map.png.

    Creates:
      - bev_intervention.png: intervention + ego original trajectory (green)
      - bev_intervention_overlay.png: + ego CF trajectory (red) if available

    Returns True if processed.
    """
    result_path = Path(result_path)
    result = json.loads(result_path.read_text())
    gen = result.get("generator", {})

    # Only process nuplan_config results (have per-agent action configs)
    clean_config = {k: v for k, v in gen.items()
                    if isinstance(v, dict) and v.get("action") in ("remove", "reposition", "speed_change")}
    if not clean_config:
        return False

    # Parse scene_id → scene_dir
    scene_id = result.get("scene", "")
    parts = scene_id.rsplit("_SAMPLED_", 1)
    if len(parts) != 2:
        return False
    scene_dir = records_dir / parts[0] / f"SAMPLED_{parts[1]}"
    if not scene_dir.exists():
        return False

    out_dir = result_path.parent
    bev_int_path = out_dir / "bev_intervention.png"
    if bev_int_path.exists() and not force:
        return True

    # Determine temporal anchor
    temporal_anchor = gen.get("temporal_anchor", "t0")

    # Load state.json
    state_json = json.loads((scene_dir / "Tp0p0" / "state.json").read_text())
    ego_speed = state_json.get("ego", {}).get("speed", 0)
    cf_traj = gen.get("counterfactual_trajectory", [])

    if temporal_anchor == "observation_window":
        # Past: render 4 modified BEVs
        frames = render_intervention_multi_frame(
            scene_dir, state_json, clean_config, ego_speed=ego_speed)
        if not frames:
            print(f"  WARN: no bev_map.png found for multi-frame")
            return False
        for slug, bev in frames:
            cv2.imwrite(str(out_dir / f"bev_intervention_{slug}.png"), bev)
        # Also save the T=-1.5 one as the main intervention BEV
        if frames:
            cv2.imwrite(str(bev_int_path), frames[0][1])
    else:
        # T=0: render single modified BEV
        bev_map_path = scene_dir / "Tp0p0" / "bev_map.png"
        if not bev_map_path.exists():
            print(f"  WARN: {bev_map_path} not found, skipping")
            return False

        modified = render_intervention_bev(
            bev_map_path, state_json, clean_config, ego_speed=ego_speed)

        # Draw ego original future trajectory (green)
        ego_future = state_json.get("ego", {}).get("future", [])
        modified = draw_ego_trajectories(modified, original_wps=ego_future)

        cv2.imwrite(str(bev_int_path), modified)

        # If CF trajectory exists, make overlay
        if cf_traj:
            overlay = draw_ego_trajectories(modified, cf_wps=cf_traj)
            cv2.imwrite(str(out_dir / "bev_intervention_overlay.png"), overlay)

    return True


def _load_bev_and_state(result_path, records_dir):
    """Load BEV image and state.json for a CF result. Returns (bev, state, scene_dir) or None."""
    result = json.loads(Path(result_path).read_text())
    scene_id = result.get("scene", "")
    parts = scene_id.rsplit("_SAMPLED_", 1)
    if len(parts) != 2:
        return None
    scene_dir = records_dir / parts[0] / f"SAMPLED_{parts[1]}"
    bev_path = scene_dir / "Tp0p0" / "bev.png"
    state_path = scene_dir / "Tp0p0" / "state.json"
    if not bev_path.exists() or not state_path.exists():
        return None
    bev = cv2.imread(str(bev_path))
    state = json.loads(state_path.read_text())
    return bev, state, scene_dir


def render_original_trajectory(result_path, records_dir, force=False):
    """Draw ego original future trajectory (green) on BEV → bev_original.png."""
    out_dir = Path(result_path).parent
    out_img = out_dir / "bev_original.png"
    if out_img.exists() and not force:
        return True
    data = _load_bev_and_state(result_path, records_dir)
    if not data:
        return False
    bev, state, _ = data
    ego_future = state.get("ego", {}).get("future", [])
    vis = bev.copy()
    _draw_dots(vis, ego_future, COLOR_EGO_ORIG, radius=5, thickness=3)
    cv2.putText(vis, "Green = Ego Original Trajectory", (10, vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_img), vis)
    return True


def render_llm_trajectory(result_path, records_dir, force=False):
    """Read llm_ego_trajectory.json and draw on BEV → bev_llm_trajectory.png.
    Green = original, Red = LLM CF.
    """
    out_dir = Path(result_path).parent
    llm_path = out_dir / "llm_ego_trajectory.json"
    out_img = out_dir / "bev_llm_trajectory.png"
    if not llm_path.exists():
        return False
    if out_img.exists() and not force:
        return True
    llm = json.loads(llm_path.read_text())
    llm_traj = llm.get("counterfactual_trajectory", [])
    if not llm_traj:
        return False
    data = _load_bev_and_state(result_path, records_dir)
    if not data:
        return False
    bev, state, _ = data
    ego_future = state.get("ego", {}).get("future", [])
    vis = bev.copy()
    _draw_dots(vis, ego_future, COLOR_EGO_ORIG, radius=5, thickness=3)
    _draw_dots(vis, llm_traj, COLOR_EGO_CF, radius=4, thickness=2)
    cv2.putText(vis, "Green=Original  Red=Claude CF", (10, vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_img), vis)
    return True


def process_all(outputs_dir, records_dir, force=False):
    """Process all CF results: render intervention BEVs + LLM trajectory overlays."""
    outputs_dir, records_dir = Path(outputs_dir), Path(records_dir)

    results = list(set(
        list(outputs_dir.rglob("cf_trajectories/*/result.json")) +
        list(outputs_dir.rglob("cf_trajectories/*/*/result.json"))
    ))

    print(f"Found {len(results)} CF trajectory results in {outputs_dir}")
    bev_count = orig_count = llm_count = 0
    for rp in sorted(results):
        if process_cf_result(rp, records_dir, force=force):
            bev_count += 1
        if render_original_trajectory(rp, records_dir, force=force):
            orig_count += 1
        if render_llm_trajectory(rp, records_dir, force=force):
            llm_count += 1

    print(f"Done. {bev_count} intervention BEVs, {orig_count} original, {llm_count} LLM trajectory overlays.")
    return bev_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render intervention BEVs from CF results")
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    process_all(args.outputs, args.records, force=args.force)
