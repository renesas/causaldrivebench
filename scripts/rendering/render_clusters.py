#!/usr/bin/env python3
"""Render clustering results as HTML with links to scene viewer.

Renders flat clustering: direct group cards sorted by size, filters,
separate singleton section, per-group collapse.

Usage:
    python tmp/render_clusters_full.py --path tmp/clusters.json
    python tmp/render_clusters_full.py --path clusters.json --base-url http://.../outputs/nuscenes_full_final
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ── Optional auto-annotation helpers (graph description per group) ──────


_CAT_MAP = [
    (("pedestrian", "bicycle", "motorcycle"), "vru"),
    (("truck", "trailer"), "truck"),
    (("bus",), "bus"),
    (("car", "van", "suv"), "car"),
    (("signal",), "signal"),
    (("crosswalk", "crossing"), "crosswalk"),
    (("cone", "barrier", "construction", "gate"), "obstacle"),
    (("sign",), "sign"),
]


def _cat(c):
    c = (c or "").lower()
    for kws, canon in _CAT_MAP:
        if any(kw in c for kw in kws):
            return canon
    return "other"


def _effect_short(et):
    return (et or "").replace("CONSTRAINS_", "").replace("_PROCEED", "").replace("INHIBITS", "INHIBIT").replace("GRANTS", "GRANT")


def _weather_tag(env):
    env = (env or "").lower()
    tags = []
    if "night" in env or "dark" in env: tags.append("night")
    if "rain" in env or "wet" in env: tags.append("wet")
    if "fog" in env or "haze" in env or "mist" in env: tags.append("fog")
    if not tags and ("clear" in env or "daytime" in env):
        return "clear-daytime"
    return "+".join(sorted(tags)) or "other"


def _road_tag(rg):
    rg = (rg or "").lower()
    if "roundabout" in rg: return "roundabout"
    if "intersection" in rg or "junction" in rg: return "intersection"
    if "parking" in rg or "service area" in rg: return "parking"
    if "highway" in rg or "expressway" in rg: return "highway"
    if "multi-lane" in rg or "divided road" in rg: return "multi-lane"
    return "road"


def _describe_graph(graph, nav_command="?"):
    """Return description dict for a single graph (loaded JSON)."""
    import sys as _sys
    repo = Path(__file__).resolve().parent.parent
    if str(repo / "causal_graph_extraction") not in _sys.path:
        _sys.path.insert(0, str(repo / "causal_graph_extraction"))
    from extract_subgraphs import extract_subgraphs

    node_cats = {n["id"]: n.get("category", "") for n in graph.get("nodes", [])}
    for d in graph.get("distractors", []) or []:
        if "id" in d:
            node_cats[d["id"]] = d.get("category", "")

    ego_edges = graph.get("ego_edges", [])
    try:
        sub = extract_subgraphs(graph, active_only=True, include_dormant_in_confounding=False)
    except Exception:
        sub = {}

    n_direct = len(sub.get("direct", []) or [])
    n_chain = len(sub.get("chain", []) or [])
    n_col = len(sub.get("collider", []) or [])
    n_cf = len(sub.get("confounding_observed", []) or [])
    n_cfu = len(sub.get("confounding_unobserved", []) or [])

    if not ego_edges: topology = "empty"
    elif n_cf or n_cfu: topology = "confounding"
    elif n_col: topology = "collider"
    elif n_chain: topology = "chain"
    elif n_direct: topology = "direct"
    else: topology = "other"

    n_active = sum(1 for e in ego_edges if e.get("active") or str(e.get("causal_status", "")).startswith("active"))
    n_dormant = len(ego_edges) - n_active
    if not ego_edges: activity = "no_edges"
    elif n_active > 0 and n_dormant > 0: activity = "mixed"
    elif n_active > 0: activity = "active"
    else: activity = "inactive"

    parts = []
    if n_direct: parts.append(f"direct×{n_direct}")
    if n_chain: parts.append(f"chain×{n_chain}")
    if n_col: parts.append(f"collider×{n_col}")
    if n_cf: parts.append(f"confounding×{n_cf}")
    if n_cfu: parts.append(f"confounding_unobs×{n_cfu}")
    n_inter = len(graph.get("inter_node_edges", []) or [])
    if n_inter:
        parts.append(f"{n_inter} inter")
    details = " + ".join(parts) if parts else "(no edges)"

    # Per-edge role
    direct_role_by_src = {d["source"]: d.get("confounding_role", "independent")
                          for d in sub.get("direct", []) or []}
    chain_distals = {c["distal"] for c in sub.get("chain", []) or []}
    collider_parents = {p for c in sub.get("collider", []) or [] for p in c.get("parents", [])}
    cfu_nodes = {n for c in sub.get("confounding_unobserved", []) or [] for n in c.get("nodes", [])}

    def _role(src_id):
        if src_id in direct_role_by_src:
            r = direct_role_by_src[src_id]
            if r == "confounder": return "confounder"
            if r == "mediator": return "mediator"
            if src_id in collider_parents: return "collider_parent"
            if src_id in cfu_nodes: return "conf_unobs"
            return "independent"
        if src_id in chain_distals: return "chain_distal"
        return "other"

    edge_lines = []
    for e in ego_edges:
        st = "active" if e.get("active") or str(e.get("causal_status", "")).startswith("active") else "dormant"
        edge_lines.append({
            "status": st,
            "effect": _effect_short(e.get("effect_type", "")),
            "source": _cat(node_cats.get(e.get("source", ""), "")),
            "role": _role(e.get("source", "")),
        })

    inter_lines = []
    for e in graph.get("inter_node_edges", []) or []:
        inter_lines.append({
            "relation": e.get("relation", "UNK"),
            "src": _cat(node_cats.get(e.get("source", ""), "")),
            "tgt": _cat(node_cats.get(e.get("target", ""), "")),
        })

    ctx = graph.get("scene_context", {}) or {}
    return {
        "topology": topology,
        "activity": activity,
        "topology_details": details,
        "edges": edge_lines,
        "inter_edges": inter_lines,
        "weather": _weather_tag(ctx.get("environment_conditions", "")),
        "road": _road_tag(ctx.get("road_geometry", "")),
        "nav": nav_command,
    }


def _auto_annotate(tree, graph_dir, records_dir, force=False, graph_suffix="_verified"):
    """In-memory annotate tree with descriptions. If force, overwrite existing."""
    import re
    graph_dir = Path(graph_dir)
    records_dir = Path(records_dir)

    def _nav_of(key):
        m = re.match(r"(nuscenes-scene-\d+)_(SAMPLED_\d+)", key)
        if not m: return "?"
        p = records_dir / m.group(1) / m.group(2) / "meta.json"
        if not p.exists(): return "?"
        with open(p) as f:
            return json.load(f).get("nav_command", "?")

    def _desc_for_group(group):
        # Aggregate over members
        w_c, r_c, n_c = Counter(), Counter(), Counter()
        for m in group["members"]:
            key = m["key"]
            p = graph_dir / f"{key}{graph_suffix}.json"
            if not p.exists():
                continue
            with open(p) as f:
                g = json.load(f)
            ctx = g.get("scene_context", {}) or {}
            w_c[_weather_tag(ctx.get("environment_conditions", ""))] += 1
            r_c[_road_tag(ctx.get("road_geometry", ""))] += 1
            n_c[_nav_of(key)] += 1
        # Graph description from leader
        leader_path = graph_dir / f'{group["leader"]}{graph_suffix}.json'
        leader_desc = {}
        if leader_path.exists():
            with open(leader_path) as f:
                lg = json.load(f)
            leader_desc = _describe_graph(lg, nav_command=n_c.most_common(1)[0][0] if n_c else "?")
        leader_desc.update({
            "weather": (w_c.most_common(1)[0][0] if w_c else "?"),
            "road": (r_c.most_common(1)[0][0] if r_c else "?"),
            "nav": (n_c.most_common(1)[0][0] if n_c else "?"),
            "weather_mix": dict(w_c),
            "road_mix": dict(r_c),
            "nav_mix": dict(n_c),
        })
        return leader_desc

    for fam in tree.get("families", []):
        for bkt in fam.get("buckets", []):
            for grp in bkt.get("groups", []):
                if force or "description" not in grp:
                    grp["description"] = _desc_for_group(grp)
                for sg in grp.get("sub_groups", []) or []:
                    if force or "description" not in sg:
                        sg["description"] = _desc_for_group(sg)


def render(clusters_path: str, output_path: str, base_url: str,
           graph_dir: str = None, records_dir: str = None,
           reannotate: bool = False, graph_suffix: str = "_verified") -> None:
    with open(clusters_path) as f:
        data = json.load(f)

    meta = data["metadata"]
    families = data.get("families", [])

    # Auto-annotate if descriptions missing (or forced)
    if graph_dir and records_dir:
        first_group = next((g for fam in families for bkt in fam.get("buckets", []) for g in bkt.get("groups", [])), None)
        needs_annot = first_group is not None and (reannotate or "description" not in first_group)
        if needs_annot:
            why = "forced re-annotation" if reannotate else "descriptions missing"
            print(f"{why} — annotating from {graph_dir}...")
            _auto_annotate(data, graph_dir, records_dir, force=reannotate, graph_suffix=graph_suffix)
            with open(clusters_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Annotated {clusters_path}")

    # Flatten all groups
    all_groups = []
    for fam in families:
        for bkt in fam["buckets"]:
            for grp in bkt["groups"]:
                all_groups.append(grp)

    # Sort by (topology, activity, road, weather, nav, -size) so structurally
    # similar groups sit together. Topology uses an explicit order by complexity
    # (direct → chain → collider → confounding → empty → other); other fields
    # fall back to alphabetical. Size is the final tiebreaker (largest first).
    _TOPO_ORDER = {"direct": 0, "chain": 1, "collider": 2, "confounding": 3, "empty": 4, "other": 5}

    def _sort_key(g):
        d = g.get("description") or {}
        topo = d.get("topology", "other")
        return (
            _TOPO_ORDER.get(topo, 99),
            d.get("activity", ""),
            d.get("road", ""),
            d.get("weather", ""),
            d.get("nav", ""),
            -g["size"],
            g["id"],
        )

    all_groups.sort(key=_sort_key)
    multi = [g for g in all_groups if g["size"] > 1]
    singles = [g for g in all_groups if g["size"] == 1]

    L = []
    L.append("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Clustering Results</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f1a;color:#d0d0e0;padding:20px;line-height:1.5}
h1{color:#fff;margin-bottom:6px;font-size:1.5em}
h2{color:#e0e0f0;margin:24px 0 10px;font-size:1.1em;border-bottom:1px solid #2a2a4a;padding-bottom:4px}
.sub{color:#888;font-size:.85em;margin-bottom:16px}
.cards{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 20px}
.card{background:#16213e;border-radius:8px;padding:10px 16px;text-align:center;min-width:85px}
.card .n{font-size:1.5em;font-weight:700}
.card .l{font-size:.7em;color:#999;text-transform:uppercase;letter-spacing:.5px}
.c1 .n{color:#c084fc} .c2 .n{color:#60a5fa} .c3 .n{color:#22d3ee}
.c4 .n{color:#facc15} .c5 .n{color:#4ade80} .c6 .n{color:#f87171}
.bar{background:#16213e;border-radius:8px;padding:10px 14px;margin:12px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:10}
.bar input[type=text]{background:#0f0f1a;border:1px solid #3b3b6b;border-radius:4px;color:#e0e0f0;padding:6px 10px;font-size:.85em;width:260px}
.bar select{background:#0f0f1a;border:1px solid #3b3b6b;border-radius:4px;color:#e0e0f0;padding:6px 10px;font-size:.85em}
.bar label{font-size:.8em;color:#aaa;cursor:pointer}
.bar input[type=checkbox]{margin-right:4px;vertical-align:middle}
.act{font-size:.8em;color:#60a5fa;cursor:pointer}
.act:hover{text-decoration:underline}
.spacer{flex:1}
.grp{background:#131325;border:1px solid #252545;border-radius:8px;margin:6px 0;overflow:hidden}
.grp-h{padding:8px 14px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;user-select:none;gap:10px}
.grp-h:hover{background:#1a1a35}
.grp-title{font-size:.85em;font-weight:600;display:flex;align-items:center;gap:8px}
.grp-title .gid{color:#60a5fa;font-family:monospace;background:#1e3a5f;padding:2px 7px;border-radius:4px;font-weight:700}
.grp-title .gsize{color:#facc15;font-weight:700}
.grp.multi .grp-title::after{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:#60a5fa;margin-left:6px}
.grp.single .grp-title::after{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:#4ade80;margin-left:6px}
.grp-body{display:none;padding:4px 14px 10px;border-top:1px solid #2a2a4a}
.grp.open .grp-body{display:block}
.mem{padding:3px 0;font-size:.82em}
.mem a{color:#60a5fa;text-decoration:none;font-family:monospace}
.mem a:hover{text-decoration:underline}
.leader{color:#facc15;font-weight:700;font-size:.7em;margin-left:8px;padding:1px 6px;background:#3b2f1f;border-radius:3px}
.hidden{display:none !important}
.section-title{color:#888;font-size:.8em;text-transform:uppercase;letter-spacing:.5px;margin:20px 0 8px}
#singles-wrap.collapsed .grp{display:none}
.toggle-singles{color:#60a5fa;cursor:pointer;font-size:.85em}
.subgrp{background:#0f0f1a;border:1px solid #1f1f3a;border-radius:6px;margin:5px 0;padding:4px 10px}
.subgrp-h{display:flex;align-items:center;gap:8px;padding:6px 4px 8px;border-bottom:1px dashed #252545;margin-bottom:6px;flex-wrap:wrap}
.sgid{color:#a78bfa;font-family:monospace;background:#261f3e;padding:1px 6px;border-radius:3px;font-size:.78em;font-weight:600}
.ssize{color:#facc15;font-weight:600;font-size:.78em}
.desc{padding:4px 14px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;font-size:.78em}
.desc-lg{padding:6px 14px 8px;border-bottom:1px solid #1f1f3a;background:#0d1020}
.desc-sm{padding:2px 0 4px;font-size:.74em}
.tag-graph{background:#1e3a5f;color:#93c5fd;padding:3px 8px;border-radius:4px;font-family:monospace;font-size:.9em}
.tag-ctx-weather{background:#1a2f3a;color:#7dd3fc;padding:3px 10px;border-radius:4px;font-size:.85em;font-weight:600}
.tag-ctx-road{background:#1f3b2d;color:#86efac;padding:3px 10px;border-radius:4px;font-size:.85em;font-weight:600}
.tag-ctx-nav{background:#3b2f1f;color:#fcd34d;padding:3px 10px;border-radius:4px;font-size:.85em;font-weight:600}
.tag-topo{padding:3px 9px;border-radius:4px;font-weight:700;font-size:.78em;text-transform:uppercase;letter-spacing:.4px}
.tag-topo-direct{background:#1e3a5f;color:#93c5fd}
.tag-topo-chain{background:#3b2f1f;color:#fcd34d}
.tag-topo-collider{background:#3b1f2b;color:#fca5a5}
.tag-topo-confounding{background:#261f3e;color:#c4b5fd}
.tag-topo-empty{background:#2a2a3a;color:#94a3b8}
.tag-topo-other{background:#3a2a2a;color:#fca5a5}
.tag-details{background:#16213e;color:#a0a0d0;padding:3px 8px;border-radius:4px;font-family:monospace;font-size:.85em}
.tag-act{padding:3px 9px;border-radius:4px;font-weight:700;font-size:.72em;text-transform:uppercase;letter-spacing:.3px}
.tag-act-active{background:#1a3a1a;color:#4ade80}
.tag-act-inactive{background:#3a1a1a;color:#f87171}
.tag-act-mixed{background:#2a3a1a;color:#facc15}
.desc-head{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:6px}
.edge-rows{display:flex;flex-direction:column;gap:3px;background:#0a0a18;border-radius:4px;padding:6px 8px;font-size:.78em}
.edge-row{display:flex;gap:6px;align-items:center;font-family:monospace}
.tag-st{padding:1px 6px;border-radius:3px;font-size:.78em;font-weight:600;min-width:58px;text-align:center}
.tag-st-active{background:#1a3a1a;color:#4ade80}
.tag-st-dormant{background:#3a1a1a;color:#f87171}
.tag-role{padding:1px 6px;border-radius:3px;font-size:.78em;font-weight:600;min-width:95px;text-align:center}
.tag-role-independent{background:#1e3a5f;color:#93c5fd}
.tag-role-mediator{background:#261f3e;color:#c4b5fd}
.tag-role-confounder{background:#3b1f2b;color:#fca5a5}
.tag-role-collider_parent{background:#3b2f1f;color:#fcd34d}
.tag-role-chain_distal{background:#1f3b2d;color:#86efac}
.tag-role-conf_unobs{background:#3a2a3a;color:#e0b5d5}
.tag-role-other{background:#2a2a3a;color:#94a3b8}
.tag-role-inter{background:#16213e;color:#93c5fd}
.tag-src{background:#181828;color:#e0e0f0;padding:1px 6px;border-radius:3px;font-weight:600}
.tag-arrow{color:#666}
.tag-eff{color:#a0a0d0;font-size:.88em}
.pipeline-banner{background:#0d1020;border:1px solid #252545;border-radius:8px;padding:10px 14px;margin:14px 0;font-size:.82em}
.pl-section{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:4px 0}
.pl-head{color:#c084fc;font-weight:700;text-transform:uppercase;letter-spacing:.5px;font-size:.78em;min-width:90px}
.pl-step{background:#16213e;color:#93c5fd;padding:3px 10px;border-radius:4px;font-size:.88em}
.pl-k{background:#1f3b2d;color:#86efac;padding:3px 9px;border-radius:4px;font-family:monospace;font-size:.85em}
.subdesc{display:inline-flex;gap:6px;flex-wrap:wrap;margin-left:8px}
.sub-leader{margin-left:auto;color:#666;font-size:.72em;font-family:monospace}
</style></head><body>""")

    # Detect which clustering script produced this clusters.json
    kernel_tag = meta.get("kernel")  # topology_filtering sets this; robust filter doesn't
    include_dormant = meta.get("include_dormant")

    # Header
    if kernel_tag == "K_topology":
        title = "Topology-Only Graph Clustering"
    else:
        title = "Hierarchical Graph Clustering"
    L.append(f"<h1>{title}</h1>")
    L.append(f'<div class="sub">{meta["n_samples"]} samples &rarr; {meta["n_groups"]} groups '
             f'({meta["n_multi_groups"]} multi, {meta["n_singletons"]} singleton, largest={meta["largest_group"]})</div>')

    # Pipeline info banner — shape depends on source script
    if kernel_tag == "K_topology":
        dormant_note = ("include dormant parents" if include_dormant
                        else "active parents only") if include_dormant is not None else ""
        dormant_html = (f'<span class="pl-step">{dormant_note}</span>'
                        if dormant_note else "")
        L.append(f'''<div class="pipeline-banner">
<div class="pl-section">
  <span class="pl-head">Pipeline</span>
  <span class="pl-step">Agglomerative complete-link + elbow-cut (floor 0.30)</span>
  <span class="pl-step">counts log-bucketed (1, 2, 3–5, 6+)</span>
  {dormant_html}
</div>
<div class="pl-section">
  <span class="pl-head">Kernel</span>
  <span class="pl-k">K_topology — subgraph pattern × source-agent histogram</span>
  <span style="color:#888;margin-left:6px">(direct / chain / collider / conf_obs / conf_unobs)</span>
</div>
</div>''')
    else:
        L.append('''<div class="pipeline-banner">
<div class="pl-section">
  <span class="pl-head">Pipeline</span>
  <span class="pl-step">Agglomerative complete-link on averaged 10-kernel distance (elbow-cut)</span>
</div>
<div class="pl-section">
  <span class="pl-head">10 Kernels</span>
  <span class="pl-k">K1 topology</span>
  <span class="pl-k">K2 status×effect×source</span>
  <span class="pl-k">K3 road</span>
  <span class="pl-k">K4 nav</span>
  <span class="pl-k">K5 behavior</span>
  <span class="pl-k">K6a active-sem</span>
  <span class="pl-k">K6b dormant-sem</span>
  <span class="pl-k">K7 inter</span>
  <span class="pl-k">K8 weather</span>
  <span class="pl-k">K9 distractor</span>
</div>
</div>''')

    L.append('<div class="cards">')
    for val, lbl, c in [
        (meta["n_samples"], "Samples", "c1"),
        (meta["n_groups"], "Groups", "c4"),
        (meta["n_multi_groups"], "Multi", "c5"),
        (meta["n_singletons"], "Singleton", "c6"),
        (meta["largest_group"], "Largest", "c2"),
        (f'{meta.get("elbow_threshold", 0):.3f}', "Threshold", "c3"),
    ]:
        L.append(f'<div class="card {c}"><div class="n">{val}</div><div class="l">{lbl}</div></div>')
    L.append("</div>")

    # Filter bar (sticky)
    L.append("""<div class="bar">
<input type="text" id="q" placeholder="Search key or group id..." oninput="filt()">
<select id="sizeFilter" onchange="filt()">
  <option value="all">All sizes</option>
  <option value="multi">Multi only (size > 1)</option>
  <option value="single">Singletons only</option>
  <option value="big">Big (size ≥ 5)</option>
  <option value="huge">Huge (size ≥ 20)</option>
</select>
<label><input type="checkbox" id="showSingles" checked onchange="toggleSingles()"> Show singletons</label>
<div class="spacer"></div>
<span class="act" onclick="tog(true)">Expand All</span>
<span class="act" onclick="tog(false)">Collapse All</span>
</div>""")

    # Helper to format description
    def _edge_row(e):
        """One row per ego-edge: status | role | source → ego | effect"""
        st = e.get("status", "?")
        role = e.get("role", "?")
        src = e.get("source", "?")
        eff = e.get("effect", "?")
        return (f'<div class="edge-row">'
                f'<span class="tag-st tag-st-{st}">{st}</span>'
                f'<span class="tag-role tag-role-{role}">{role}</span>'
                f'<span class="tag-src">{src}</span>'
                f'<span class="tag-arrow">&rarr; ego</span>'
                f'<span class="tag-eff">{eff}</span>'
                f'</div>')

    def _inter_row(e):
        return (f'<div class="edge-row">'
                f'<span class="tag-role tag-role-inter">inter</span>'
                f'<span class="tag-src">{e["src"]}</span>'
                f'<span class="tag-arrow">&rarr;</span>'
                f'<span class="tag-src">{e["tgt"]}</span>'
                f'<span class="tag-eff">{e["relation"]}</span>'
                f'</div>')

    def _desc_html(desc, large=False):
        """Full group-level description: topology + per-edge structural rows + context."""
        if not desc: return ""
        size_cls = "desc-lg" if large else "desc-sm"
        w, r, n = desc.get("weather", "?"), desc.get("road", "?"), desc.get("nav", "?")
        topology = desc.get("topology", "?")
        activity = desc.get("activity", "")
        topo_details = desc.get("topology_details", "")
        edges = desc.get("edges", [])
        inter_edges = desc.get("inter_edges", [])
        activity_html = f'<span class="tag-act tag-act-{activity}">{activity}</span>' if activity and activity != "no_edges" else ""

        edge_rows_html = "".join(_edge_row(e) for e in edges) if isinstance(edges, list) else ""
        inter_rows_html = "".join(_inter_row(e) for e in inter_edges) if isinstance(inter_edges, list) else ""
        edges_block = ""
        if edge_rows_html or inter_rows_html:
            edges_block = f'<div class="edge-rows">{edge_rows_html}{inter_rows_html}</div>'
        else:
            edges_block = '<div class="edge-rows"><em style="color:#666">no causal edges</em></div>'

        return (f'<div class="desc {size_cls}">'
                f'<div class="desc-head">'
                f'<span class="tag-topo tag-topo-{topology}">{topology}</span>'
                f'{activity_html}'
                f'<span class="tag-details">{topo_details}</span>'
                f'<span class="tag-ctx-weather">{w}</span>'
                f'<span class="tag-ctx-road">{r}</span>'
                f'<span class="tag-ctx-nav">{n}</span>'
                f'</div>'
                f'{edges_block}'
                f'</div>')

    def _subdesc_html(desc):
        """Sub-group description: ONLY context tags (graph is same as parent)."""
        if not desc: return ""
        w = desc.get("weather", "?")
        r = desc.get("road", "?")
        n = desc.get("nav", "?")
        w_mix = desc.get("weather_mix", {})
        r_mix = desc.get("road_mix", {})
        n_mix = desc.get("nav_mix", {})
        # Show mix counts if not uniform
        def _tag(val, mix, cls):
            if len(mix) > 1:
                parts = ", ".join(f"{k}×{v}" for k, v in sorted(mix.items(), key=lambda x: -x[1]))
                return f'<span class="{cls}" title="{parts}">{val} <span style="opacity:.7">({len(mix)} variants)</span></span>'
            return f'<span class="{cls}">{val}</span>'
        return (f'<div class="subdesc">'
                f'{_tag(w, w_mix, "tag-ctx-weather")}'
                f'{_tag(r, r_mix, "tag-ctx-road")}'
                f'{_tag(n, n_mix, "tag-ctx-nav")}'
                f'</div>')

    # Multi-sample groups
    L.append('<h2>Multi-Sample Groups</h2>')
    L.append('<div id="multi-wrap">')
    for grp in multi:
        sub_groups = grp.get("sub_groups") or []
        n_sub = len(sub_groups)
        desc = grp.get("description", {})
        L.append(f'<div class="grp multi" data-size="{grp["size"]}">')
        L.append(f'<div class="grp-h" onclick="this.parentElement.classList.toggle(\'open\')">')
        L.append(f'<span class="grp-title"><span class="gid">{grp["id"]}</span>'
                 f'<span class="gsize">{grp["size"]}</span> samples')
        if n_sub > 1:
            L.append(f' <span style="color:#888;font-size:.85em">&bull; {n_sub} sub-groups</span>')
        L.append(f'</span>')
        L.append(f'<span style="color:#888;font-size:.8em">leader: {grp["leader"]}</span>')
        L.append("</div>")
        L.append(_desc_html(desc, large=True))
        L.append('<div class="grp-body">')

        if sub_groups:
            for sg in sub_groups:
                sg_desc = sg.get("description", {})
                L.append(f'<div class="subgrp" data-size="{sg["size"]}">')
                L.append(f'<div class="subgrp-h">'
                         f'<span class="sgid">{sg["id"]}</span>'
                         f'<span class="ssize">{sg["size"]}</span>'
                         f'{_subdesc_html(sg_desc)}'
                         f'<span class="sub-leader">leader: {sg["leader"]}</span>'
                         f'</div>')
                for mem in sg["members"]:
                    key = mem["key"]
                    url = f"{base_url}/{key}.html"
                    leader_tag = '<span class="leader">LEADER</span>' if mem.get("is_leader") else ""
                    L.append(f'<div class="mem"><a href="{url}" target="_blank">{key}</a>{leader_tag}</div>')
                L.append("</div>")
        else:
            for mem in grp["members"]:
                key = mem["key"]
                url = f"{base_url}/{key}.html"
                leader_tag = '<span class="leader">LEADER</span>' if mem.get("is_leader") else ""
                L.append(f'<div class="mem"><a href="{url}" target="_blank">{key}</a>{leader_tag}</div>')
        L.append("</div></div>")
    L.append("</div>")

    # Singletons — separate section
    L.append('<h2>Singletons <span class="toggle-singles" onclick="toggleSinglesSection()" id="singles-label">(hide)</span></h2>')
    L.append('<div id="singles-wrap">')
    for grp in singles:
        key = grp["members"][0]["key"]
        url = f"{base_url}/{key}.html"
        L.append(f'<div class="grp single" data-size="1">')
        L.append(f'<div class="grp-h">')
        L.append(f'<span class="grp-title"><span class="gid">{grp["id"]}</span><span class="gsize">1</span> sample</span>')
        L.append(f'<a href="{url}" target="_blank" style="color:#60a5fa;font-family:monospace;font-size:.8em">{key}</a>')
        L.append("</div></div>")
    L.append("</div>")

    # JavaScript
    L.append("""<script>
function filt() {
    const q = document.getElementById('q').value.toLowerCase();
    const sz = document.getElementById('sizeFilter').value;
    document.querySelectorAll('.grp').forEach(g => {
        const size = parseInt(g.dataset.size);
        const text = g.textContent.toLowerCase();
        let pass = true;
        if (q && !text.includes(q)) pass = false;
        if (sz === 'multi' && size < 2) pass = false;
        if (sz === 'single' && size !== 1) pass = false;
        if (sz === 'big' && size < 5) pass = false;
        if (sz === 'huge' && size < 20) pass = false;
        g.classList.toggle('hidden', !pass);
    });
}
function tog(o) {
    document.querySelectorAll('.grp').forEach(g => {
        if (o) g.classList.add('open');
        else g.classList.remove('open');
    });
}
function toggleSingles() {
    const show = document.getElementById('showSingles').checked;
    document.getElementById('singles-wrap').style.display = show ? '' : 'none';
    document.getElementById('singles-label').style.display = show ? '' : 'none';
    document.querySelectorAll('h2')[1].style.display = show ? '' : 'none';
}
function toggleSinglesSection() {
    const wrap = document.getElementById('singles-wrap');
    const label = document.getElementById('singles-label');
    wrap.classList.toggle('collapsed');
    label.textContent = wrap.classList.contains('collapsed') ? '(show)' : '(hide)';
}
// Collapse singletons by default (there are many)
document.addEventListener('DOMContentLoaded', () => {
    toggleSinglesSection();
});
</script></body></html>""")

    with open(output_path, "w") as f:
        f.write("\n".join(L))
    print(f"Done: {output_path}  (multi: {len(multi)}, singletons: {len(singles)})")


def main():
    p = argparse.ArgumentParser(description="Render clustering results as HTML")
    p.add_argument("--path", "--clusters", dest="clusters", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--base-url", default="http://w.x.y.z:8999/outputs/nuscenes_full_final")
    p.add_argument("--graph-dir", default=None,
                   help="Dir with graph JSONs (for auto-annotation if descriptions missing)")
    p.add_argument("--records-dir", default=None,
                   help="Dir with scene records for nav_command lookup")
    p.add_argument("--graph-suffix", default="_verified",
                   help="Graph file suffix used in --graph-dir (default: _verified). "
                        "Use _fixed for fix_causal_graphs outputs. "
                        "Leading underscore is optional.")
    p.add_argument("--reannotate", action="store_true",
                   help="Force re-annotation, overwriting cached descriptions in clusters.json")
    args = p.parse_args()

    suffix = args.graph_suffix if args.graph_suffix.startswith("_") else "_" + args.graph_suffix
    clusters_path = Path(args.clusters)
    output_path = args.output or str(clusters_path.parent / "clustering_results.html")
    render(str(clusters_path), output_path, args.base_url,
           graph_dir=args.graph_dir, records_dir=args.records_dir,
           reannotate=args.reannotate, graph_suffix=suffix)


if __name__ == "__main__":
    main()
