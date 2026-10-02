"""
Render clean causal graph PNGs from scene JSON files using Graphviz.
Supports both formats:
  - Hand-crafted (flat "edges" array)
  - VLM-generated ("ego_edges" + "inter_node_edges")

Also generates an HTML interactive visualization (D3.js force graph).

Usage:
    python render_causal_graph.py scene_0018.json scene_0093.json ...
    python render_causal_graph.py *.json
    python render_causal_graph.py --stitch          # stitch all PNGs into overview grid
"""
import json
import os
import sys
import textwrap
from pathlib import Path

import graphviz

# Ensure conda env dot binary is on PATH
DOT_DIR = str(Path(sys.executable).parent)
if DOT_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = DOT_DIR + ":" + os.environ.get("PATH", "")

# ── Node type → styling ───────────────────────────────────────────────────
TYPE_STYLE = {
    "EGO_VEHICLE": {
        "fillcolor": "#E74C3C", "fontcolor": "white",
        "shape": "box", "style": "filled,bold,rounded",
        "penwidth": "3", "width": "2.0", "height": "0.8",
    },
    "NAVIGATION_COMMAND": {
        "fillcolor": "#3498DB", "fontcolor": "white",
        "shape": "box", "style": "filled,rounded",
    },
    "TRAFFIC_CONTROL": {
        "fillcolor": "#9B59B6", "fontcolor": "white",
        "shape": "box", "style": "filled,rounded",
    },
    "ROAD_BOUNDARY_GEOMETRY": {
        "fillcolor": "#1ABC9C", "fontcolor": "white",
        "shape": "box", "style": "filled,rounded",
    },
    "OTHER_VEHICLE": {
        "fillcolor": "#2ECC71", "fontcolor": "white",
        "shape": "box", "style": "filled,rounded",
    },
    "VULNERABLE_ROAD_USER": {
        "fillcolor": "#F39C12", "fontcolor": "white",
        "shape": "box", "style": "filled,rounded",
    },
    "ENVIRONMENTAL_CONDITION": {
        "fillcolor": "#7F8C8D", "fontcolor": "white",
        "shape": "box", "style": "filled,rounded",
    },
    "ROAD_OBSTACLE": {
        "fillcolor": "#E67E22", "fontcolor": "white",
        "shape": "box", "style": "filled,rounded",
    },
    "EMERGENCY_VEHICLE_ACTIVE": {
        "fillcolor": "#C0392B", "fontcolor": "white",
        "shape": "box", "style": "filled,rounded",
    },
}

TYPE_LABEL = {
    "EGO_VEHICLE": "Ego Vehicle",
    "NAVIGATION_COMMAND": "Navigation",
    "TRAFFIC_CONTROL": "Traffic Control",
    "ROAD_BOUNDARY_GEOMETRY": "Road Geometry",
    "OTHER_VEHICLE": "Other Vehicle",
    "VULNERABLE_ROAD_USER": "VRU (Ped/Cyclist)",
    "ENVIRONMENTAL_CONDITION": "Environment",
    "ROAD_OBSTACLE": "Road Obstacle",
    "EMERGENCY_VEHICLE_ACTIVE": "Emergency Vehicle",
}

# Edge effect_type → color
EFFECT_COLORS = {
    "COMMANDS_HEADING":        "#3498DB",
    "COMMANDS_SPEED_PROFILE":  "#3498DB",
    "GRANTS_RIGHT_OF_WAY":     "#2ECC71",
    "BOUNDS_LATERALLY":        "#1ABC9C",
    "REQUIRES_YIELD":          "#E74C3C",
    "REQUIRES_GAP_ACCEPTANCE": "#E67E22",
    "REQUIRES_FOLLOWING":      "#F39C12",
    "HALTS":                   "#C0392B",
    "BLOCKS_PATH":             "#C0392B",
    "LIMITS_SPEED":            "#E74C3C",
    "DEGRADES_VISIBILITY":     "#95A5A6",
}

# Inter-node relation → color
RELATION_COLORS = {
    "TRIGGERS":             "#E74C3C",
    "MODIFIES":             "#3498DB",
    "OCCLUDES":             "#95A5A6",
    "CONSTRAINS_RESPONSE":  "#E67E22",
    "CAUSES_STATE_OF":      "#9B59B6",
}


def wrap(text, width=32):
    return "\\n".join(textwrap.wrap(text, width))


def _escape_html(text):
    """Escape special characters for Graphviz HTML labels."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _normalize_edges(data):
    """
    Normalize both JSON formats into a unified edges list.
    Returns (ego_edges, inter_edges) where each is a list of dicts.
    """
    # New format: ego_edges + inter_node_edges
    if "ego_edges" in data:
        ego_edges = data.get("ego_edges", [])
        inter_edges = data.get("inter_node_edges", [])
        return ego_edges, inter_edges

    # Old format: flat "edges" array (all are ego edges, no inter-node)
    return data.get("edges", []), []


def render(json_path):
    with open(json_path) as f:
        data = json.load(f)

    ego_state = data.get("ego_state", {})
    nodes_data = data["nodes"]
    ego_edges, inter_edges = _normalize_edges(data)

    scene_name = Path(json_path).stem.replace("_", "-")
    scene_id = data.get("scene_id", scene_name)
    nav = ego_state.get("nav_command", "")
    speed = ego_state.get("speed_mps", "")

    # New format fields
    accel = ego_state.get("inferred_acceleration", ego_state.get("planned_speed", ""))
    heading = ego_state.get("inferred_heading_change", ego_state.get("planned_path", ""))

    title = (
        f'<<FONT POINT-SIZE="22"><B>{scene_name}</B></FONT><BR/>'
        f'<FONT POINT-SIZE="14">Nav: {nav}  |  {accel} / {heading}  '
        f'|  Speed: {speed} m/s</FONT>>'
    )

    # ── Find connected nodes (skip orphans with zero edges) ──────────
    connected_ids = set()
    for e in ego_edges:
        connected_ids.add(e.get("source", e.get("from", "")))
        connected_ids.add(e.get("target", e.get("to", "")))
    for e in inter_edges:
        connected_ids.add(e.get("source", ""))
        connected_ids.add(e.get("target", ""))
    # Always keep ego
    connected_ids.add("ego")

    g = graphviz.Digraph(
        name=scene_name,
        format="png",
        engine="dot",
        graph_attr={
            "bgcolor": "#0f0f1a",
            "label": title,
            "labelloc": "t",
            "fontcolor": "white",
            "fontname": "Helvetica",
            "rankdir": "TB",
            "pad": "0.8",
            "nodesep": "0.5",
            "ranksep": "0.9",
            "dpi": "150",
            "splines": "true",
            "concentrate": "true",
            "size": "18,12!",
        },
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "11",
            "penwidth": "1.2",
            "color": "#333355",
            "margin": "0.15,0.08",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "8",
            "fontcolor": "#ccccdd",
            "color": "#555577",
            "penwidth": "1.0",
            "arrowsize": "0.8",
        },
    )

    # ── Nodes (only connected ones) ──────────────────────────────────
    node_type_map = {}
    for n in nodes_data:
        nid = n["id"]
        if nid not in connected_ids:
            continue
        ntype = n.get("type", "EGO_VEHICLE" if nid == "ego" else "OTHER_VEHICLE")
        node_type_map[nid] = ntype
        style = TYPE_STYLE.get(ntype, TYPE_STYLE["OTHER_VEHICLE"])

        if nid == "ego" or ntype == "EGO_VEHICLE":
            label = f'<<B>EGO</B><BR/><FONT POINT-SIZE="9">{_escape_html(nav)} | {speed} m/s</FONT>>'
            g.node(nid, label=label, **style)
        else:
            nlabel = n.get("label", nid)
            # Support both "position" (old) and "position_relative_to_ego" (new)
            pos = n.get("position", n.get("position_relative_to_ego", ""))
            state = n.get("state", "")
            lines = [f'<B>{_escape_html(nlabel)}</B>']
            if state:
                lines.append(f'<FONT POINT-SIZE="8" COLOR="#cccccc">{_escape_html(state)}</FONT>')
            if pos:
                lines.append(f'<FONT POINT-SIZE="7" COLOR="#999999">{_escape_html(pos)}</FONT>')
            html_label = "<" + "<BR/>".join(lines) + ">"
            g.node(nid, label=html_label, **style)

    # Ensure "ego" node exists
    if "ego" not in node_type_map:
        node_type_map["ego"] = "EGO_VEHICLE"
        style = TYPE_STYLE["EGO_VEHICLE"]
        label = f'<<B>EGO</B><BR/><FONT POINT-SIZE="9">{_escape_html(nav)} | {speed} m/s</FONT>>'
        g.node("ego", label=label, **style)

    # ── Ego Edges ────────────────────────────────────────────────────
    for e in ego_edges:
        src = e.get("source", e.get("from", ""))
        dst = e.get("target", e.get("to", ""))
        effect = e.get("effect_type", "")
        active = e.get("active", True)
        priority = e.get("priority_level", 2)
        condition = e.get("condition", "")

        effect_short = effect.replace("_", " ").title()
        label_parts = [effect_short]
        if condition:
            cond_short = condition[:40] + ("..." if len(condition) > 40 else "")
            label_parts.append(f"if: {cond_short}")

        label = wrap("\\n".join(label_parts), 28)

        edge_color = EFFECT_COLORS.get(effect, "#555577")
        edge_attrs = {"label": label, "color": edge_color}

        if not active:
            edge_attrs["style"] = "dashed"
            edge_attrs["color"] = edge_color + "88"
            edge_attrs["fontcolor"] = "#888899"
        else:
            edge_attrs["penwidth"] = str(0.8 + int(priority) * 0.3)

        g.edge(src, dst, **edge_attrs)

    # ── Inter-Node Edges ─────────────────────────────────────────────
    for e in inter_edges:
        src = e.get("source", "")
        dst = e.get("target", "")
        relation = e.get("relation", "")

        rel_short = relation.replace("_", " ").title()
        label = wrap(rel_short, 28)

        edge_color = RELATION_COLORS.get(relation, "#777799")
        edge_attrs = {
            "label": label,
            "color": edge_color,
            "style": "dotted",
            "penwidth": "1.5",
            "fontcolor": "#aaaacc",
        }

        g.edge(src, dst, **edge_attrs)

    # ── Legend ─────────────────────────────────────────────────────────
    used_types = set(node_type_map.values())
    legend_types = [t for t in TYPE_STYLE if t in used_types]

    with g.subgraph(name="cluster_legend") as leg:
        leg.attr(
            label="  Node Types  ", fontcolor="#999999", fontsize="11",
            fontname="Helvetica",
            style="dashed", color="#333355", bgcolor="#14142a",
        )
        prev = None
        for ntype in legend_types:
            style = TYPE_STYLE[ntype]
            lbl = TYPE_LABEL.get(ntype, ntype)
            lid = f"_leg_{ntype}"
            ls = {k: v for k, v in style.items()
                  if k not in ("width", "height", "penwidth")}
            leg.node(lid, label=lbl,
                     fontsize="9", width="0.1", height="0.1", **ls)
            if prev:
                leg.edge(prev, lid, style="invis")
            prev = lid

    out_path = str(Path(json_path).with_suffix(""))
    g.render(out_path, cleanup=True)
    print(f"  {out_path}.png")
    return out_path + ".png"


def _img_to_b64(img_path):
    """Read an image file and return base64 data-URI string."""
    import base64
    p = Path(img_path)
    if not p.exists():
        return ""
    ext = p.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, ext)
    with open(p, "rb") as f:
        return f"data:image/{mime};base64,{base64.b64encode(f.read()).decode()}"


def _copy_img(img_path, assets_dir, filename):
    """Copy an image file to assets_dir and return the relative URL."""
    import shutil
    p = Path(img_path)
    if not p.exists():
        return ""
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    dst = assets_dir / filename
    shutil.copy2(str(p), str(dst))
    return f"assets/{filename}"


def _load_d3_tag():
    """Return a <script> tag for D3.js using a relative path (served via HTTP)."""
    return '<script src="../d3.v7.min.js"></script>'


def create_html_visualization(json_path, output_file=None, scene_dir=None, model_label=None):
    """
    Create an interactive HTML visualization with:
      - Top: BEV (left) + 4-camera × N-timestep grid (right) showing temporal history
      - Bottom: D3 force-directed causal graph
    scene_dir: path to the scenario_records/<scene>/<maneuver>/ folder
    model_label: e.g. "Cosmos-Reason2-8B" or "GPT-5.2" — shown in header
    Images are copied to an assets/ subfolder alongside the HTML for fast loading.
    """
    import re
    d3_script_tag = _load_d3_tag()
    with open(json_path) as f:
        graph_data = json.load(f)

    # Normalize to new format
    if "ego_edges" not in graph_data and "edges" in graph_data:
        graph_data["ego_edges"] = graph_data.pop("edges")
        graph_data["inter_node_edges"] = []

    if output_file is None:
        output_file = str(Path(json_path).with_suffix(".html"))

    scene_id = graph_data.get("scene_id", Path(json_path).stem)
    model_tag = f" [{model_label}]" if model_label else ""
    display_title = f"{scene_id}{model_tag}"

    # Assets directory alongside the output HTML
    assets_dir = Path(output_file).parent / "assets"

    # ── Discover timestep folders and copy images to assets ──────────
    CAMS = ["cam_front", "cam_front_left", "cam_front_right", "cam_back"]
    CAM_LABELS = {"cam_front": "FRONT", "cam_front_left": "FRONT LEFT",
                  "cam_front_right": "FRONT RIGHT", "cam_back": "BACK"}
    img_bev = ""
    # timesteps: list of (slug_label, {cam: img_url})
    timesteps = []

    if scene_dir:
        sd = Path(scene_dir)
        stem = Path(output_file).stem  # e.g. scene-0018_left_turn
        img_bev = _copy_img(sd / "Tp0p0" / "bev.png", assets_dir, f"{stem}_bev.png")

        # Find all T* subdirs, sort by time
        def _slug_to_seconds(slug):
            s = slug.replace("Tp", "+").replace("Tm", "-").replace("p", ".")
            try:
                return float(s)
            except ValueError:
                return 999

        t_dirs = []
        for d in sorted(sd.iterdir()):
            if d.is_dir() and re.match(r"T[mp]\d", d.name):
                t_dirs.append(d)
        t_dirs.sort(key=lambda d: _slug_to_seconds(d.name))

        for td in t_dirs:
            # Human-readable label: Tm1p5 → T-1.5s, Tp0p0 → T=0.0s
            sec = _slug_to_seconds(td.name)
            label = f"T={sec:+.1f}s" if sec != 0 else "T=0 (now)"
            cam_imgs = {}
            for cam in CAMS:
                cam_imgs[cam] = _copy_img(td / f"{cam}.jpg", assets_dir,
                                          f"{stem}_{td.name}_{cam}.jpg")
            timesteps.append((label, td.name, cam_imgs))

    n_cols = len(timesteps) if timesteps else 1

    # Build camera grid HTML rows: one row per camera, cells are timesteps
    cam_grid_html = ""
    for cam in CAMS:
        for i, (label, slug, cam_imgs) in enumerate(timesteps):
            is_t0 = "Tp0p0" in slug
            border = "border: 2px solid #3498DB;" if is_t0 else ""
            cam_grid_html += f'''            <div class="cam-cell" style="{border}">
                <img src="{cam_imgs.get(cam, '')}" alt="{cam} {label}">
                <div class="cam-label">{CAM_LABELS[cam]}<br/>{label}</div>
            </div>\n'''

    # Model label color mapping
    model_color = "#F39C12"  # default orange
    if model_label:
        if "cosmos" in model_label.lower():
            model_color = "#00D4AA"  # teal/green for Cosmos
        elif "gpt" in model_label.lower():
            model_color = "#AB68FF"  # purple for GPT

    json_pretty = json.dumps(graph_data, indent=2)
    # Escape backticks and backslashes for JS template literal
    json_for_js = json_pretty.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Scene: {display_title}</title>
    {d3_script_tag}
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f1a; color: #eee; padding: 15px; }}
        h1 {{ text-align: center; font-size: 26px; margin-bottom: 4px; color: #fff; letter-spacing: 0.5px; }}
        .model-badge {{
            display: inline-block; padding: 3px 14px; border-radius: 20px; font-size: 13px;
            font-weight: 700; letter-spacing: 0.5px; margin-left: 8px; vertical-align: middle;
            background: {model_color}22; color: {model_color}; border: 1px solid {model_color}55;
        }}
        .subtitle {{ text-align: center; color: #888; font-size: 13px; margin-bottom: 14px; }}
        .stats-bar {{
            display: flex; justify-content: center; gap: 20px; margin-bottom: 14px; flex-wrap: wrap;
        }}
        .stat {{
            background: #16213e; border: 1px solid #333; border-radius: 6px;
            padding: 6px 16px; font-size: 12px; text-align: center;
        }}
        .stat .val {{ font-size: 18px; font-weight: bold; color: #3498DB; }}

        /* ── Scene view: BEV left + temporal camera grid right ── */
        .scene-view {{
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 4px;
            max-width: 1600px;
            margin: 0 auto 12px auto;
            background: #16213e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 4px;
        }}
        .bev-panel {{
            display: flex; align-items: center; justify-content: center;
            background: #111; border-radius: 6px; overflow: hidden;
            width: 380px; height: 380px;
        }}
        .bev-panel img {{ width: 100%; height: 100%; object-fit: contain; }}
        .bev-title {{
            position: absolute; top: 4px; left: 6px; font-size: 11px; font-weight: bold;
            color: #3498DB; background: rgba(0,0,0,0.7); padding: 2px 6px; border-radius: 3px;
        }}

        /* Camera temporal grid */
        .cam-grid {{
            display: grid;
            grid-template-columns: repeat({n_cols}, 1fr);
            grid-template-rows: repeat(4, 1fr);
            gap: 2px;
        }}
        .cam-cell {{
            background: #111; border-radius: 3px; overflow: hidden;
            position: relative; min-height: 0;
        }}
        .cam-cell img {{ width: 100%; height: 100%; object-fit: contain; background: #111; }}
        .cam-label {{
            position: absolute; bottom: 2px; left: 3px; font-size: 8px;
            color: #fff; background: rgba(0,0,0,0.75); padding: 1px 4px; border-radius: 2px;
            line-height: 1.2;
        }}

        /* ── Graph container ── */
        #graph-container {{
            background: #16213e; border: 1px solid #333; border-radius: 8px;
            max-width: 1600px; height: 650px; margin: 0 auto;
            position: relative; overflow: hidden;
        }}
        .node-EGO_VEHICLE {{ fill: #E74C3C; stroke: #c0392b; }}
        .node-VULNERABLE_ROAD_USER {{ fill: #F39C12; stroke: #e67e22; }}
        .node-OTHER_VEHICLE {{ fill: #2ECC71; stroke: #27ae60; }}
        .node-TRAFFIC_CONTROL {{ fill: #9B59B6; stroke: #8e44ad; }}
        .node-ROAD_BOUNDARY_GEOMETRY {{ fill: #1ABC9C; stroke: #16a085; }}
        .node-ENVIRONMENTAL_CONDITION {{ fill: #7F8C8D; stroke: #6c7a7d; }}
        .node-NAVIGATION_COMMAND {{ fill: #3498DB; stroke: #2980b9; }}
        .node-ROAD_OBSTACLE {{ fill: #E67E22; stroke: #d35400; }}
        .node-EMERGENCY_VEHICLE_ACTIVE {{ fill: #C0392B; stroke: #a93226; }}
        .node-label {{ font-size: 10px; pointer-events: none; text-anchor: middle; fill: white; font-weight: bold; }}
        .edge-label {{ font-size: 8px; fill: #999; text-anchor: middle; }}
        .link-ego {{ stroke: #E74C3C; }}
        .link-inter {{ stroke: #3498DB; stroke-dasharray: 5,5; }}
        .link-active {{ opacity: 0.9; }}
        .link-latent {{ opacity: 0.3; stroke-dasharray: 3,3; }}
        /* Focus mode: dim everything, then un-dim connected elements */
        .focus-active line.dimmed {{ opacity: 0.04; }}
        .focus-active text.edge-label.dimmed {{ opacity: 0; }}
        .focus-active circle.dimmed {{ opacity: 0.15; }}
        .focus-active text.node-label.dimmed {{ opacity: 0.15; }}
        #tooltip {{
            position: absolute; background: rgba(0,0,0,0.95); color: white;
            border-radius: 6px; opacity: 0; font-size: 12px;
            min-width: 280px; overflow: hidden;
            pointer-events: none;
            border: 1px solid #555; z-index: 100; line-height: 1.6;
            display: flex; flex-direction: column;
        }}
        #tooltip.pinned {{
            pointer-events: auto; opacity: 1 !important;
            border-color: #3498DB; box-shadow: 0 0 12px rgba(52,152,219,0.4);
        }}
        #tooltip .tip-header {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px 12px; background: #1a1a2e; border-bottom: 1px solid #333;
            cursor: grab; user-select: none; flex-shrink: 0;
        }}
        #tooltip .tip-header:active {{ cursor: grabbing; }}
        #tooltip .tip-header .title {{ font-size: 13px; font-weight: bold; }}
        #tooltip .close-btn {{
            cursor: pointer; color: #999; font-size: 16px; font-weight: bold;
            padding: 0 4px; border-radius: 3px;
        }}
        #tooltip .close-btn:hover {{ color: white; }}
        #tooltip .tip-body {{
            padding: 10px 14px; overflow-y: auto; flex: 1;
        }}
        #tooltip .resize-handle {{
            position: absolute; bottom: 0; right: 0; width: 20px; height: 20px;
            cursor: nwse-resize; flex-shrink: 0;
        }}
        #tooltip .resize-handle::after {{
            content: ''; position: absolute; bottom: 3px; right: 3px;
            width: 0; height: 0;
            border-style: solid; border-width: 0 0 10px 10px;
            border-color: transparent transparent #3498DB transparent;
        }}
        #tooltip::-webkit-scrollbar {{ width: 6px; }}
        #tooltip::-webkit-scrollbar-track {{ background: #222; border-radius: 3px; }}
        #tooltip::-webkit-scrollbar-thumb {{ background: #555; border-radius: 3px; }}
        #tooltip::-webkit-scrollbar-thumb:hover {{ background: #777; }}
        #tooltip .tip-body::-webkit-scrollbar {{ width: 6px; }}
        #tooltip .tip-body::-webkit-scrollbar-track {{ background: #222; border-radius: 3px; }}
        #tooltip .tip-body::-webkit-scrollbar-thumb {{ background: #555; border-radius: 3px; }}
        #legend {{
            position: absolute; top: 10px; right: 10px; background: rgba(22,33,62,0.95);
            padding: 10px; border: 1px solid #444; border-radius: 4px; font-size: 10px; z-index: 50;
        }}
        .legend-item {{ margin: 3px 0; display: flex; align-items: center; }}
        .legend-color {{ width: 14px; height: 14px; margin-right: 6px; border-radius: 50%; border: 1px solid #555; }}

        /* ── JSON Viewer Panel ── */
        .json-panel {{
            max-width: 1600px; margin: 14px auto 0 auto;
        }}
        .json-toggle {{
            background: #16213e; border: 1px solid #333; border-radius: 8px 8px 0 0;
            padding: 10px 20px; cursor: pointer; display: flex; align-items: center;
            justify-content: space-between; user-select: none; transition: background 0.2s;
        }}
        .json-toggle:hover {{ background: #1a3050; }}
        .json-toggle .label {{ font-size: 14px; font-weight: 600; }}
        .json-toggle .arrow {{ font-size: 12px; color: #888; transition: transform 0.3s; }}
        .json-toggle.open .arrow {{ transform: rotate(180deg); }}
        .json-toggle:not(.open) {{ border-radius: 8px; }}
        .json-content {{
            display: none; background: #0d1117; border: 1px solid #333; border-top: none;
            border-radius: 0 0 8px 8px; overflow: hidden; position: relative;
        }}
        .json-content.open {{ display: block; }}
        .json-actions {{
            display: flex; gap: 8px; padding: 8px 12px; background: #161b22; border-bottom: 1px solid #333;
        }}
        .json-actions button {{
            background: #21262d; border: 1px solid #444; color: #ccc; padding: 4px 12px;
            border-radius: 4px; cursor: pointer; font-size: 12px; transition: background 0.2s;
        }}
        .json-actions button:hover {{ background: #30363d; color: #fff; }}
        .json-actions button.copied {{ background: #238636; border-color: #2ea043; color: #fff; }}
        #json-viewer {{
            padding: 16px; font-family: 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
            font-size: 12px; line-height: 1.5; overflow: auto; max-height: 500px;
            white-space: pre; color: #c9d1d9; tab-size: 2;
        }}
        /* JSON syntax highlighting */
        .json-key {{ color: #7ee787; }}
        .json-string {{ color: #a5d6ff; }}
        .json-number {{ color: #f8b86c; }}
        .json-bool {{ color: #ff7b72; }}
        .json-null {{ color: #8b949e; }}
        .json-bracket {{ color: #ccc; }}
    </style>
</head>
<body>
    <h1>{scene_id} {f'<span class="model-badge">{model_label}</span>' if model_label else ''}</h1>
    <div class="stats-bar">
        <div class="stat"><div class="val">{len(graph_data.get('nodes', []))}</div>Nodes</div>
        <div class="stat"><div class="val">{len(graph_data.get('ego_edges', []))}</div>Ego Edges</div>
        <div class="stat"><div class="val">{len(graph_data.get('inter_node_edges', []))}</div>Inter Edges</div>
        <div class="stat"><div class="val">{n_cols}</div>Timesteps</div>
    </div>

    <!-- ── Scene View: BEV + Temporal Camera Grid ── -->
    <div class="scene-view">
        <div class="bev-panel" style="position:relative">
            <img src="{img_bev}" alt="BEV">
            <div class="bev-title">BEV @ T=0</div>
        </div>
        <div class="cam-grid">
{cam_grid_html}        </div>
    </div>

    <!-- ── Causal Graph ── -->
    <div id="graph-container">
        <div id="legend">
            <div class="legend-item"><div class="legend-color" style="background:#E74C3C"></div>Ego</div>
            <div class="legend-item"><div class="legend-color" style="background:#3498DB"></div>Nav</div>
            <div class="legend-item"><div class="legend-color" style="background:#9B59B6"></div>Traffic Ctrl</div>
            <div class="legend-item"><div class="legend-color" style="background:#1ABC9C"></div>Road Geom</div>
            <div class="legend-item"><div class="legend-color" style="background:#2ECC71"></div>Vehicle</div>
            <div class="legend-item"><div class="legend-color" style="background:#F39C12"></div>VRU</div>
            <div class="legend-item"><div class="legend-color" style="background:#7F8C8D"></div>Env</div>
            <hr style="border-color:#444; margin:5px 0">
            <div class="legend-item"><svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#E74C3C" stroke-width="2"/></svg>&nbsp;Active</div>
            <div class="legend-item"><svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#E74C3C" stroke-width="2" stroke-dasharray="3,3" opacity="0.3"/></svg>&nbsp;Latent</div>
            <div class="legend-item"><svg width="24" height="2"><line x1="0" y1="1" x2="24" y2="1" stroke="#3498DB" stroke-width="2" stroke-dasharray="5,5"/></svg>&nbsp;Inter</div>
        </div>
        <div id="tooltip"></div>
    </div>

    <!-- ── JSON Viewer ── -->
    <div class="json-panel">
        <div class="json-toggle" onclick="toggleJson(this)">
            <span class="label">Causal Graph JSON</span>
            <span class="arrow">&#9660;</span>
        </div>
        <div class="json-content" id="json-content">
            <div class="json-actions">
                <button onclick="copyJson(this)">Copy JSON</button>
                <button onclick="downloadJson()">Download .json</button>
            </div>
            <div id="json-viewer"></div>
        </div>
    </div>

    <script>
        const graphData = {json.dumps(graph_data)};
        const nodes = graphData.nodes.map((n, i) => ({{ ...n, index: i }}));
        const nodeMap = {{}};
        nodes.forEach(n => {{ nodeMap[n.id] = n; }});
        const links = [];
        (graphData.ego_edges || []).forEach(e => {{
            if (nodeMap[e.source] && nodeMap[e.target])
                links.push({{ source: nodeMap[e.source], target: nodeMap[e.target], type:'ego',
                    effect: e.effect_type, active: e.active, priority: e.priority_level,
                    justification: e.justification }});
        }});
        (graphData.inter_node_edges || []).forEach(e => {{
            if (nodeMap[e.source] && nodeMap[e.target])
                links.push({{ source: nodeMap[e.source], target: nodeMap[e.target], type:'inter',
                    relation: e.relation, justification: e.justification }});
        }});
        const container = d3.select("#graph-container");
        const rect = container.node().getBoundingClientRect();
        const width = rect.width || 1600, height = rect.height || 650;
        const svg = container.append("svg").attr("width", "100%").attr("height", "100%")
            .attr("viewBox", `0 0 ${{width}} ${{height}}`);
        const gEl = svg.append("g");
        svg.call(d3.zoom().extent([[0,0],[width,height]]).scaleExtent([0.1,10])
            .on("zoom", ev => gEl.attr("transform", ev.transform)));
        svg.append("defs").selectAll("marker").data(["arrow-ego","arrow-inter"])
            .enter().append("marker").attr("id", d=>d).attr("viewBox","0 -5 10 10")
            .attr("refX",28).attr("refY",0).attr("markerWidth",6).attr("markerHeight",6)
            .attr("orient","auto").append("path").attr("d","M0,-5L10,0L0,5")
            .attr("fill", d => d==="arrow-ego" ? "#E74C3C" : "#3498DB");
        const sim = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d=>d.id).distance(160))
            .force("charge", d3.forceManyBody().strength(-700))
            .force("center", d3.forceCenter(width/2, height/2))
            .force("collision", d3.forceCollide().radius(45));
        const link = gEl.append("g").selectAll("line").data(links).enter().append("line")
            .attr("class", d => `link-${{d.type}} link-${{d.active!==false?'active':'latent'}}`)
            .attr("stroke-width", d => d.type==='ego' ? 3 : 2)
            .attr("marker-end", d => `url(#arrow-${{d.type}})`);
        const edgeLbl = gEl.append("g").selectAll("text").data(links).enter().append("text")
            .attr("class","edge-label").text(d => d.effect||d.relation||"");
        const node = gEl.append("g").selectAll("circle").data(nodes).enter().append("circle")
            .attr("r", d => d.id==="ego"?28:20)
            .attr("class", d => `node-${{d.type||'OTHER_VEHICLE'}}`)
            .attr("stroke-width", 2).call(drag(sim));
        const nodeLbl = gEl.append("g").selectAll("text").data(nodes).enter().append("text")
            .attr("class","node-label").text(d => d.id);
        const tooltip = d3.select("#tooltip");
        let pinned = false;
        function closeTooltip() {{
            pinned = false;
            tooltip.classed("pinned", false).style("opacity", 0).html("");
        }}
        function buildTooltipHTML(d) {{
            const outgoing = links.filter(l => l.source.id===d.id);
            const incoming = links.filter(l => l.target.id===d.id);
            function edgeRow(l, dir) {{
                const other = dir==='out' ? l.target.id : l.source.id;
                const arrow = dir==='out' ? '&rarr;' : '&larr;';
                const color = dir==='out' ? '#E74C3C' : '#2ECC71';
                return `<div style="margin:4px 0;padding:6px 0;border-bottom:1px solid #333">`
                    +`<span style="color:${{color}};font-weight:bold">${{arrow}} ${{other}}</span> `
                    +`<span style="color:#aaa">[${{l.effect||l.relation}}]</span><br/>`
                    +`<span style="color:#ccc">${{l.justification||''}}</span></div>`;
            }}
            let outHTML = outgoing.map(l => edgeRow(l, 'out')).join('');
            let inHTML  = incoming.map(l => edgeRow(l, 'in')).join('');
            let edgesHTML = '';
            if (outHTML) edgesHTML += `<div style="margin-top:6px"><b style="color:#E74C3C">Outgoing (${{outgoing.length}})</b></div>` + outHTML;
            if (inHTML)  edgesHTML += `<div style="margin-top:6px"><b style="color:#2ECC71">Incoming (${{incoming.length}})</b></div>` + inHTML;
            return `<div class="tip-header"><span class="title">${{d.label}}</span>`
                +`<span class="close-btn">&times;</span></div>`
                +`<div class="tip-body">`
                +`Type: ${{d.type}}<br/>`
                +`${{d.state?'State: '+d.state+'<br/>':''}}`
                +`Pos: ${{d.position_relative_to_ego||d.position||''}}`
                +`${{edgesHTML?'<hr style="border-color:#555;margin:6px 0">'+edgesHTML:''}}`
                +`</div><div class="resize-handle"></div>`;
        }}
        let focusedNode = null;
        function focusNode(d) {{
            focusedNode = d.id;
            svg.classed("focus-active", true);
            const connected = new Set([d.id]);
            links.forEach(l => {{
                const sid = typeof l.source === 'object' ? l.source.id : l.source;
                const tid = typeof l.target === 'object' ? l.target.id : l.target;
                if (sid === d.id || tid === d.id) {{ connected.add(sid); connected.add(tid); }}
            }});
            link.classed("dimmed", l => {{
                const sid = typeof l.source === 'object' ? l.source.id : l.source;
                const tid = typeof l.target === 'object' ? l.target.id : l.target;
                return sid !== d.id && tid !== d.id;
            }});
            edgeLbl.classed("dimmed", l => {{
                const sid = typeof l.source === 'object' ? l.source.id : l.source;
                const tid = typeof l.target === 'object' ? l.target.id : l.target;
                return sid !== d.id && tid !== d.id;
            }});
            node.classed("dimmed", n => !connected.has(n.id));
            nodeLbl.classed("dimmed", n => !connected.has(n.id));
        }}
        function clearFocus() {{
            focusedNode = null;
            svg.classed("focus-active", false);
            link.classed("dimmed", false);
            edgeLbl.classed("dimmed", false);
            node.classed("dimmed", false);
            nodeLbl.classed("dimmed", false);
        }}
        node.on("mouseenter", function(ev,d) {{
            if (pinned) return;
            tooltip.style("opacity",1).style("left",(ev.offsetX+15)+"px").style("top",(ev.offsetY-10)+"px")
                .html(buildTooltipHTML(d));
        }}).on("mousemove", function(ev) {{
            if (pinned) return;
            tooltip.style("left",(ev.offsetX+15)+"px").style("top",(ev.offsetY-10)+"px");
        }}).on("mouseleave", () => {{
            if (!pinned) tooltip.style("opacity",0);
        }}).on("click", function(ev,d) {{
            ev.stopPropagation();
            pinned = true;
            tooltip.classed("pinned", true).style("opacity",1)
                .style("left",(ev.offsetX+15)+"px").style("top",(ev.offsetY-10)+"px")
                .html(buildTooltipHTML(d));
            if (focusedNode === d.id) {{ clearFocus(); }}
            else {{ focusNode(d); }}
        }});
        svg.on("click", () => {{ if (pinned) closeTooltip(); clearFocus(); }});
        // ── Tooltip drag-to-move + drag-to-resize ──
        (function() {{
            const tip = document.getElementById('tooltip');
            let mode = null, startX, startY, startL, startT, startW, startH;
            tip.addEventListener('click', function(e) {{
                if (e.target.classList.contains('close-btn')) {{ closeTooltip(); e.stopPropagation(); }}
            }});
            tip.addEventListener('mousedown', function(e) {{
                if (e.target.classList.contains('resize-handle')) {{
                    mode = 'resize';
                }} else if (e.target.closest('.tip-header') && !e.target.classList.contains('close-btn')) {{
                    mode = 'drag';
                }} else {{ return; }}
                e.preventDefault(); e.stopPropagation();
                startX = e.clientX; startY = e.clientY;
                startW = tip.offsetWidth; startH = tip.offsetHeight;
                startL = tip.offsetLeft; startT = tip.offsetTop;
            }});
            document.addEventListener('mousemove', function(e) {{
                if (!mode) return;
                if (mode === 'resize') {{
                    tip.style.width  = Math.max(220, startW + (e.clientX - startX)) + 'px';
                    tip.style.height = Math.max(120, startH + (e.clientY - startY)) + 'px';
                }} else if (mode === 'drag') {{
                    tip.style.left = (startL + e.clientX - startX) + 'px';
                    tip.style.top  = (startT + e.clientY - startY) + 'px';
                }}
            }});
            document.addEventListener('mouseup', function() {{ mode = null; }});
        }})();
        sim.on("tick", () => {{
            link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
                .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
            edgeLbl.attr("x",d=>(d.source.x+d.target.x)/2).attr("y",d=>(d.source.y+d.target.y)/2);
            node.attr("cx",d=>d.x).attr("cy",d=>d.y);
            nodeLbl.attr("x",d=>d.x).attr("y",d=>d.y+4);
        }});
        function drag(sim) {{
            return d3.drag()
                .on("start", (ev) => {{ if(!ev.active) sim.alphaTarget(0.3).restart(); ev.subject.fx=ev.subject.x; ev.subject.fy=ev.subject.y; }})
                .on("drag", (ev) => {{ ev.subject.fx=ev.x; ev.subject.fy=ev.y; }})
                .on("end", (ev) => {{ if(!ev.active) sim.alphaTarget(0); ev.subject.fx=null; ev.subject.fy=null; }});
        }}

        // ── JSON Viewer ──
        const rawJson = `{json_for_js}`;
        function syntaxHighlight(json) {{
            return json.replace(/("(\\\\u[a-zA-Z0-9]{{4}}|\\\\[^u]|[^\\\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(\.\d+)?([eE][+-]?\d+)?)/g,
                function(match) {{
                    let cls = 'json-number';
                    if (/^"/.test(match)) {{
                        cls = /:$/.test(match) ? 'json-key' : 'json-string';
                    }} else if (/true|false/.test(match)) {{
                        cls = 'json-bool';
                    }} else if (/null/.test(match)) {{
                        cls = 'json-null';
                    }}
                    return '<span class="' + cls + '">' + match + '</span>';
                }});
        }}
        document.getElementById('json-viewer').innerHTML = syntaxHighlight(rawJson);

        function toggleJson(el) {{
            el.classList.toggle('open');
            document.getElementById('json-content').classList.toggle('open');
        }}
        function copyJson(btn) {{
            navigator.clipboard.writeText(rawJson).then(() => {{
                btn.classList.add('copied');
                btn.textContent = 'Copied!';
                setTimeout(() => {{ btn.classList.remove('copied'); btn.textContent = 'Copy JSON'; }}, 2000);
            }});
        }}
        function downloadJson() {{
            const blob = new Blob([rawJson], {{type: 'application/json'}});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = '{scene_id}.json';
            a.click();
        }}
    </script>
</body>
</html>"""

    with open(output_file, "w") as f:
        f.write(html_content)
    print(f"  HTML → {output_file}")
    return output_file


def create_index_html(html_dir, scene_files, model_label=None):
    """Create an index.html that links to all scene HTMLs."""
    links_html = ""
    for name, filename in sorted(scene_files):
        links_html += f'        <a href="{filename}">{name}</a>\n'

    model_tag = f" — {model_label}" if model_label else ""
    index = f"""<!DOCTYPE html>
<html>
<head>
    <title>STSBench — Causal Scene Graphs{model_tag}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f1a; color: #eee;
               display: flex; flex-direction: column; align-items: center; padding: 40px; }}
        h1 {{ margin-bottom: 30px; }}
        a {{ display: block; color: #3498DB; font-size: 18px; margin: 8px 0; text-decoration: none;
             padding: 12px 30px; background: #16213e; border: 1px solid #333; border-radius: 6px;
             transition: background 0.2s; min-width: 350px; text-align: center; }}
        a:hover {{ background: #1a3050; }}
    </style>
</head>
<body>
    <h1>STSBench — Causal Scene Graphs{model_tag}</h1>
{links_html}
</body>
</html>"""
    idx_path = Path(html_dir) / "index.html"
    with open(idx_path, "w") as f:
        f.write(index)
    print(f"  Index → {idx_path}")
    return str(idx_path)


def create_unified_index(out_dir, cosmos_dir=None, gpt_dir=None):
    """Create a unified index.html that shows both Cosmos and GPT graphs side-by-side."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect scenes from each model dir
    def _collect_scenes(d, prefix):
        scenes = []
        if d and Path(d).exists():
            for f in sorted(Path(d).glob("scene-*.html")):
                scene_name = f.stem  # e.g. scene-0018_left_turn
                scenes.append((scene_name, f"{prefix}/{f.name}"))
        return scenes

    cosmos_scenes = _collect_scenes(cosmos_dir, "cosmos")
    gpt_scenes = _collect_scenes(gpt_dir, "gpt")

    # Build a merged list of unique scene names
    all_scene_names = sorted(set(s[0] for s in cosmos_scenes) | set(s[0] for s in gpt_scenes))
    cosmos_map = {s[0]: s[1] for s in cosmos_scenes}
    gpt_map = {s[0]: s[1] for s in gpt_scenes}

    # Build table rows
    rows_html = ""
    for name in all_scene_names:
        # Parse scene info
        parts = name.split("_", 1)
        scene_id = parts[0] if parts else name
        maneuver = parts[1].replace("_", " ").title() if len(parts) > 1 else ""

        cosmos_link = f'<a href="{cosmos_map[name]}" class="btn cosmos">View</a>' if name in cosmos_map else '<span class="na">--</span>'
        gpt_link = f'<a href="{gpt_map[name]}" class="btn gpt">View</a>' if name in gpt_map else '<span class="na">--</span>'

        rows_html += f"""            <tr>
                <td class="scene-name">{scene_id}</td>
                <td class="maneuver">{maneuver}</td>
                <td>{cosmos_link}</td>
                <td>{gpt_link}</td>
            </tr>\n"""

    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>STSBench — Causal Scene Graphs</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f1a; color: #eee;
            padding: 40px 20px; min-height: 100vh;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{
            text-align: center; font-size: 28px; margin-bottom: 8px;
            background: linear-gradient(135deg, #3498DB, #00D4AA);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .tagline {{ text-align: center; color: #666; font-size: 14px; margin-bottom: 30px; }}
        .model-chips {{
            display: flex; justify-content: center; gap: 16px; margin-bottom: 24px;
        }}
        .chip {{
            padding: 6px 18px; border-radius: 20px; font-size: 13px; font-weight: 600;
        }}
        .chip.cosmos {{ background: #00D4AA22; color: #00D4AA; border: 1px solid #00D4AA55; }}
        .chip.gpt {{ background: #AB68FF22; color: #AB68FF; border: 1px solid #AB68FF55; }}
        table {{
            width: 100%; border-collapse: collapse; background: #16213e;
            border: 1px solid #333; border-radius: 8px; overflow: hidden;
        }}
        thead th {{
            padding: 14px 16px; text-align: left; font-size: 13px;
            color: #888; border-bottom: 2px solid #333; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.5px;
        }}
        tbody td {{
            padding: 12px 16px; border-bottom: 1px solid #222;
        }}
        tbody tr:hover {{ background: #1a3050; }}
        tbody tr:last-child td {{ border-bottom: none; }}
        .scene-name {{ font-weight: 700; color: #fff; font-size: 15px; }}
        .maneuver {{ color: #888; font-size: 13px; }}
        .btn {{
            display: inline-block; padding: 5px 16px; border-radius: 5px;
            text-decoration: none; font-size: 13px; font-weight: 600;
            transition: all 0.2s;
        }}
        .btn.cosmos {{ background: #00D4AA22; color: #00D4AA; border: 1px solid #00D4AA44; }}
        .btn.cosmos:hover {{ background: #00D4AA44; }}
        .btn.gpt {{ background: #AB68FF22; color: #AB68FF; border: 1px solid #AB68FF44; }}
        .btn.gpt:hover {{ background: #AB68FF44; }}
        .na {{ color: #444; font-size: 13px; }}
        .footer {{ text-align: center; color: #444; font-size: 11px; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>STSBench Causal Scene Graphs</h1>
        <div class="tagline">NuScenes scenario mining with causal graph generation</div>
        <div class="model-chips">
            <div class="chip cosmos">Cosmos-Reason2-8B</div>
            <div class="chip gpt">GPT-5.2</div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Scene</th>
                    <th>Maneuver</th>
                    <th>Cosmos-Reason2-8B</th>
                    <th>GPT-5.2</th>
                </tr>
            </thead>
            <tbody>
{rows_html}            </tbody>
        </table>
        <div class="footer">
            {len(all_scene_names)} scenes | {len(cosmos_scenes)} Cosmos | {len(gpt_scenes)} GPT
        </div>
    </div>
</body>
</html>"""
    idx_path = out_dir / "index.html"
    with open(idx_path, "w") as f:
        f.write(index_html)
    print(f"  Unified index → {idx_path}")
    return str(idx_path)


def stitch_overview(png_dir, out_dir, cols=2):
    """Stitch all causal graph PNGs in a directory into a grid overview image."""
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow required for stitching. pip install Pillow")
        return

    png_dir = Path(png_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(png_dir.glob("scene_*.png"))
    if not pngs:
        print("No scene_*.png files found to stitch.")
        return

    images = []
    for p in pngs:
        img = Image.open(p)
        images.append((p.stem, img))
    print(f"  Stitching {len(images)} graphs into overview grid ...")

    # Resize all to same width
    max_w = max(im.width for _, im in images)
    resized = []
    for name, im in images:
        if im.width != max_w:
            ratio = max_w / im.width
            im = im.resize((max_w, int(im.height * ratio)), Image.LANCZOS)
        resized.append((name, im))

    # Arrange into grid
    rows_list = [resized[i:i+cols] for i in range(0, len(resized), cols)]
    row_heights = []
    for row in rows_list:
        row_heights.append(max(im.height for _, im in row))

    total_w = max_w * cols
    total_h = sum(row_heights)
    canvas = Image.new("RGB", (total_w, total_h), (15, 15, 26))

    y_off = 0
    for row, rh in zip(rows_list, row_heights):
        x_off = 0
        for name, im in row:
            canvas.paste(im, (x_off, y_off))
            x_off += max_w
        y_off += rh

    out_path = out_dir / "causal_graphs_overview.png"
    canvas.save(str(out_path))
    print(f"  Overview saved → {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render causal scene graph visualizations")
    parser.add_argument("files", nargs="*", help="JSON files to render")
    parser.add_argument("--html", action="store_true", help="Generate HTML with embedded BEV + cameras + D3 graph")
    parser.add_argument("--records", default=None, help="scenario_records root dir (auto-finds scene dirs for images)")
    parser.add_argument("--stitch", action="store_true", help="Stitch all PNGs into overview grid")
    parser.add_argument("--stitch-dir", default=None, help="Directory containing PNGs to stitch")
    parser.add_argument("--overview-dir", default=None, help="Output dir for overview")
    parser.add_argument("--index", action="store_true", help="Generate index.html linking all scene HTMLs")
    parser.add_argument("--model-label", default=None, help="Model label shown in header (e.g. 'Cosmos-Reason2-8B', 'GPT-5.2')")
    parser.add_argument("--unified-index", default=None,
                        help="Output dir for unified index.html (needs --cosmos-dir and --gpt-dir)")
    parser.add_argument("--cosmos-dir", default=None, help="Directory with Cosmos HTML files")
    parser.add_argument("--gpt-dir", default=None, help="Directory with GPT HTML files")
    args = parser.parse_args()

    scene_html_files = []

    # Render individual graphs
    for fp in args.files:
        print(f"Rendering {fp} ...")
        render(fp)
        if args.html:
            # Try to find matching scene_dir in records
            scene_dir = None
            if args.records:
                stem = Path(fp).stem
                # Strip common suffixes that don't correspond to directory names
                for suffix in ("_refined", "_raw"):
                    if stem.endswith(suffix):
                        stem = stem[:-len(suffix)]
                        break
                # Strategy 1: split on first _ (nuScenes: scene-0018_left_turn → scene-0018/left_turn)
                parts = stem.split("_", 1)
                if len(parts) == 2:
                    candidate = Path(args.records) / parts[0] / parts[1]
                    if candidate.exists():
                        scene_dir = str(candidate)
                # Strategy 2: rsplit on last _ (OpenScene/Argoverse: name_SAMPLED → name/SAMPLED)
                if scene_dir is None:
                    parts = stem.rsplit("_", 1)
                    if len(parts) == 2:
                        candidate = Path(args.records) / parts[0] / parts[1]
                        if candidate.exists():
                            scene_dir = str(candidate)
            html_path = create_html_visualization(fp, scene_dir=scene_dir, model_label=args.model_label)
            scene_html_files.append((Path(fp).stem, Path(html_path).name))

    # Create index.html
    if args.index and scene_html_files:
        html_dir = str(Path(args.files[0]).parent) if args.files else "."
        create_index_html(html_dir, scene_html_files, model_label=args.model_label)

    # Create unified index
    if args.unified_index:
        create_unified_index(args.unified_index,
                             cosmos_dir=args.cosmos_dir, gpt_dir=args.gpt_dir)

    # Stitch overview
    if args.stitch:
        png_src = Path(args.stitch_dir) if args.stitch_dir else Path(__file__).parent
        overview_dst = Path(args.overview_dir) if args.overview_dir else Path(__file__).parent.parent / "STSBench" / "scenario_records" / "overview"
        stitch_overview(png_src, overview_dst)
