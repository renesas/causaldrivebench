#!/usr/bin/env python3
"""Render 2-level QA-kernel clustering as HTML with links to scene viewer.

Matches the hierarchy produced by scripts/graph_filtering/qa_kernel_filter.py:

  Level 1 families — grouped by QA + topology kernels
      K1 = 0.25*K_active + 0.25*K_dormant + 0.35*K_distractor + 0.25*K_topology
  Level 2 groups    — scene/context refinement inside each family
      K2 = mean(K_road, K_weather, K_nav, K_behavior)

Top-level cards are families; each family expands into its L2 groups.
Each L2 group has a leader (the selected representative).

Usage:
    python scripts/rendering/render_qa_clusters.py --clusters clusters.json
    python scripts/rendering/render_qa_clusters.py --clusters clusters.json \
        --graph-dir /path/to/graphs --records-dir /path/to/records \
        --base-url http://host:port/outputs/nuscenes_full_final
"""

import argparse
import json
from collections import Counter
from pathlib import Path


# ── Coarse taggers (shared look with render_clusters.py) ────────────────


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
    return ((et or "")
            .replace("CONSTRAINS_", "")
            .replace("_PROCEED", "")
            .replace("INHIBITS", "INHIBIT")
            .replace("GRANTS", "GRANT"))


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


# ── Graph-level description (for family leaders) ────────────────────────


def _describe_graph(graph, nav_command="?"):
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

    n_active = sum(1 for e in ego_edges
                   if e.get("active") or str(e.get("causal_status", "")).startswith("active"))
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
    if n_inter: parts.append(f"{n_inter} inter")
    details = " + ".join(parts) if parts else "(no edges)"

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

    # Distractor profile — important for K_distractor family signature
    distractors = graph.get("distractors", []) or []
    d_by_cat = Counter(_cat(d.get("category", "")) for d in distractors)
    d_summary = ", ".join(f"{k}×{v}" for k, v in d_by_cat.most_common()) if d_by_cat else "none"

    ctx = graph.get("scene_context", {}) or {}
    return {
        "topology": topology,
        "activity": activity,
        "topology_details": details,
        "edges": edge_lines,
        "inter_edges": inter_lines,
        "n_distractors": len(distractors),
        "distractor_mix": dict(d_by_cat),
        "distractor_summary": d_summary,
        "weather": _weather_tag(ctx.get("environment_conditions", "")),
        "road": _road_tag(ctx.get("road_geometry", "")),
        "nav": nav_command,
    }


# ── Auto-annotate families (L1) and groups (L2) ─────────────────────────


def _nav_of(key, records_dir: Path):
    import re
    m = re.match(r"(nuscenes-scene-\d+)_(SAMPLED_\d+)", key)
    if not m: return "?"
    p = records_dir / m.group(1) / m.group(2) / "meta.json"
    if not p.exists(): return "?"
    with open(p) as f:
        return json.load(f).get("nav_command", "?")


def _annotate(tree, graph_dir: Path, records_dir: Path, force=False):
    """Attach 'description' (family-level, uses leader graph) and
    per-group 'description' (scene-context aggregated)."""
    graph_dir = Path(graph_dir)
    records_dir = Path(records_dir)

    def _group_ctx(group):
        w_c, r_c, n_c = Counter(), Counter(), Counter()
        for m in group["members"]:
            key = m["key"]
            p = graph_dir / f"{key}_verified.json"
            if not p.exists():
                continue
            with open(p) as f:
                g = json.load(f)
            ctx = g.get("scene_context", {}) or {}
            w_c[_weather_tag(ctx.get("environment_conditions", ""))] += 1
            r_c[_road_tag(ctx.get("road_geometry", ""))] += 1
            n_c[_nav_of(key, records_dir)] += 1
        return {
            "weather": w_c.most_common(1)[0][0] if w_c else "?",
            "road": r_c.most_common(1)[0][0] if r_c else "?",
            "nav": n_c.most_common(1)[0][0] if n_c else "?",
            "weather_mix": dict(w_c),
            "road_mix": dict(r_c),
            "nav_mix": dict(n_c),
        }

    def _family_desc(fam):
        # Use the leader of the largest L2 group as the family graph exemplar.
        all_groups = [g for b in fam.get("buckets", []) for g in b.get("groups", [])]
        if not all_groups:
            return {}
        biggest = max(all_groups, key=lambda g: g["size"])
        leader_key = biggest["leader"]
        leader_path = graph_dir / f"{leader_key}_verified.json"
        if not leader_path.exists():
            return {}
        with open(leader_path) as f:
            lg = json.load(f)
        nav = _nav_of(leader_key, records_dir)
        desc = _describe_graph(lg, nav_command=nav)
        return desc

    for fam in tree.get("families", []):
        if force or "description" not in fam:
            fam["description"] = _family_desc(fam)
        for bkt in fam.get("buckets", []):
            for grp in bkt.get("groups", []):
                if force or "description" not in grp:
                    grp["description"] = _group_ctx(grp)


# ── HTML helpers ────────────────────────────────────────────────────────


def _edge_row(e):
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


def _family_desc_html(desc):
    """Large block: topology, activity, per-edge rows, distractor profile."""
    if not desc:
        return ""
    topology = desc.get("topology", "?")
    activity = desc.get("activity", "")
    topo_details = desc.get("topology_details", "")
    edges = desc.get("edges", []) or []
    inter_edges = desc.get("inter_edges", []) or []
    n_dist = desc.get("n_distractors", 0)
    d_summary = desc.get("distractor_summary", "none")
    activity_html = (f'<span class="tag-act tag-act-{activity}">{activity}</span>'
                     if activity and activity != "no_edges" else "")

    edge_rows_html = "".join(_edge_row(e) for e in edges)
    inter_rows_html = "".join(_inter_row(e) for e in inter_edges)
    if edge_rows_html or inter_rows_html:
        edges_block = f'<div class="edge-rows">{edge_rows_html}{inter_rows_html}</div>'
    else:
        edges_block = '<div class="edge-rows"><em style="color:#666">no causal edges</em></div>'

    return (f'<div class="desc desc-lg">'
            f'<div class="desc-head">'
            f'<span class="tag-topo tag-topo-{topology}">{topology}</span>'
            f'{activity_html}'
            f'<span class="tag-details">{topo_details}</span>'
            f'<span class="tag-dist">distractors: {n_dist} ({d_summary})</span>'
            f'</div>'
            f'{edges_block}'
            f'</div>')


def _group_ctx_html(desc):
    """Inline context-only tags (road / weather / nav) with mix tooltips."""
    if not desc:
        return ""
    w = desc.get("weather", "?")
    r = desc.get("road", "?")
    n = desc.get("nav", "?")
    w_mix = desc.get("weather_mix", {})
    r_mix = desc.get("road_mix", {})
    n_mix = desc.get("nav_mix", {})

    def _tag(val, mix, cls):
        if len(mix) > 1:
            parts = ", ".join(f"{k}×{v}" for k, v in sorted(mix.items(), key=lambda x: -x[1]))
            return (f'<span class="{cls}" title="{parts}">{val}'
                    f' <span style="opacity:.7">({len(mix)})</span></span>')
        return f'<span class="{cls}">{val}</span>'

    return (f'<div class="subdesc">'
            f'{_tag(r, r_mix, "tag-ctx-road")}'
            f'{_tag(w, w_mix, "tag-ctx-weather")}'
            f'{_tag(n, n_mix, "tag-ctx-nav")}'
            f'</div>')


# ── Main renderer ───────────────────────────────────────────────────────


def render(clusters_path: str, output_path: str, base_url: str,
           graph_dir: str = None, records_dir: str = None,
           reannotate: bool = False) -> None:
    with open(clusters_path) as f:
        data = json.load(f)

    meta = data["metadata"]
    families = data.get("families", [])

    if graph_dir and records_dir:
        first_fam = families[0] if families else None
        needs = first_fam is not None and (reannotate or "description" not in first_fam)
        if needs:
            why = "forced re-annotation" if reannotate else "descriptions missing"
            print(f"{why} — annotating from {graph_dir}...")
            _annotate(data, Path(graph_dir), Path(records_dir), force=reannotate)
            with open(clusters_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Annotated {clusters_path}")

    # Sort families by complexity then size
    _TOPO_ORDER = {"direct": 0, "chain": 1, "collider": 2, "confounding": 3, "empty": 4, "other": 5}

    def _fam_key(fam):
        d = fam.get("description") or {}
        return (
            _TOPO_ORDER.get(d.get("topology", "other"), 99),
            d.get("activity", ""),
            -fam["size"],
            fam["label"],
        )

    families_sorted = sorted(families, key=_fam_key)
    multi_families = [f for f in families_sorted if f["size"] > 1]
    singleton_families = [f for f in families_sorted if f["size"] == 1]

    w = meta.get("l1_weights", {})
    w_active = w.get("K_active", 0.25)
    w_dormant = w.get("K_dormant", 0.25)
    w_distractor = w.get("K_distractor", 0.35)
    w_topology = w.get("K_topology", 0.25)

    L = []
    L.append("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>QA-Kernel Clustering</title>
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
.fam{background:#131325;border:1px solid #252545;border-radius:8px;margin:8px 0;overflow:hidden}
.fam-h{padding:10px 14px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;user-select:none;gap:10px}
.fam-h:hover{background:#1a1a35}
.fam-title{font-size:.9em;font-weight:600;display:flex;align-items:center;gap:8px}
.fam-title .fid{color:#c084fc;font-family:monospace;background:#261f3e;padding:2px 8px;border-radius:4px;font-weight:700}
.fam-title .fsize{color:#facc15;font-weight:700}
.fam.multi .fam-title::after{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:#c084fc;margin-left:6px}
.fam.single .fam-title::after{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:#4ade80;margin-left:6px}
.fam-body{display:none;padding:4px 14px 10px;border-top:1px solid #2a2a4a}
.fam.open .fam-body{display:block}
.grp{background:#0f0f1a;border:1px solid #1f1f3a;border-radius:6px;margin:5px 0}
.grp-h{padding:6px 12px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;cursor:pointer;user-select:none}
.grp-h:hover{background:#161630}
.gid{color:#60a5fa;font-family:monospace;background:#1e3a5f;padding:2px 7px;border-radius:3px;font-size:.78em;font-weight:700}
.gsize{color:#facc15;font-weight:600;font-size:.8em}
.grp-body{display:none;padding:4px 12px 8px;border-top:1px dashed #252545}
.grp.open .grp-body{display:block}
.mem{padding:3px 0;font-size:.82em}
.mem a{color:#60a5fa;text-decoration:none;font-family:monospace}
.mem a:hover{text-decoration:underline}
.leader{color:#facc15;font-weight:700;font-size:.7em;margin-left:8px;padding:1px 6px;background:#3b2f1f;border-radius:3px}
.hidden{display:none !important}
.toggle-singles{color:#60a5fa;cursor:pointer;font-size:.85em}
#singles-wrap.collapsed .fam{display:none}
.desc{padding:4px 14px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;font-size:.78em}
.desc-lg{padding:6px 14px 8px;border-bottom:1px solid #1f1f3a;background:#0d1020}
.tag-topo{padding:3px 9px;border-radius:4px;font-weight:700;font-size:.78em;text-transform:uppercase;letter-spacing:.4px}
.tag-topo-direct{background:#1e3a5f;color:#93c5fd}
.tag-topo-chain{background:#3b2f1f;color:#fcd34d}
.tag-topo-collider{background:#3b1f2b;color:#fca5a5}
.tag-topo-confounding{background:#261f3e;color:#c4b5fd}
.tag-topo-empty{background:#2a2a3a;color:#94a3b8}
.tag-topo-other{background:#3a2a2a;color:#fca5a5}
.tag-details{background:#16213e;color:#a0a0d0;padding:3px 8px;border-radius:4px;font-family:monospace;font-size:.85em}
.tag-dist{background:#2a1f3a;color:#d8b4fe;padding:3px 8px;border-radius:4px;font-size:.82em}
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
.tag-ctx-weather{background:#1a2f3a;color:#7dd3fc;padding:2px 8px;border-radius:4px;font-size:.82em;font-weight:600}
.tag-ctx-road{background:#1f3b2d;color:#86efac;padding:2px 8px;border-radius:4px;font-size:.82em;font-weight:600}
.tag-ctx-nav{background:#3b2f1f;color:#fcd34d;padding:2px 8px;border-radius:4px;font-size:.82em;font-weight:600}
.subdesc{display:inline-flex;gap:6px;flex-wrap:wrap}
.leader-tag{margin-left:auto;color:#888;font-size:.72em;font-family:monospace}
.pipeline-banner{background:#0d1020;border:1px solid #252545;border-radius:8px;padding:10px 14px;margin:14px 0;font-size:.82em}
.pl-section{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:4px 0}
.pl-head{color:#c084fc;font-weight:700;text-transform:uppercase;letter-spacing:.5px;font-size:.78em;min-width:90px}
.pl-step{background:#16213e;color:#93c5fd;padding:3px 10px;border-radius:4px;font-size:.88em}
.pl-k{background:#1f3b2d;color:#86efac;padding:3px 9px;border-radius:4px;font-family:monospace;font-size:.85em}
.pl-k-l2{background:#261f3e;color:#c4b5fd}
.pl-w{color:#facc15;font-family:monospace;font-size:.82em;margin-left:4px}
</style></head><body>""")

    # Header
    L.append("<h1>QA-Kernel Hierarchical Clustering</h1>")
    L.append(f'<div class="sub">{meta["n_samples"]} samples &rarr; '
             f'{meta["n_families"]} L1 families &rarr; {meta["n_groups"]} L2 groups '
             f'({meta["n_multi_groups"]} multi, {meta["n_singletons"]} singleton, '
             f'largest={meta["largest_group"]})</div>')

    # Pipeline banner
    L.append(f'''<div class="pipeline-banner">
<div class="pl-section">
  <span class="pl-head">Pipeline</span>
  <span class="pl-step">Two-level agglomerative complete-link + elbow-cut</span>
</div>
<div class="pl-section">
  <span class="pl-head">Level 1</span>
  <span class="pl-k">K_active<span class="pl-w">× {w_active}</span></span>
  <span class="pl-k">K_dormant<span class="pl-w">× {w_dormant}</span></span>
  <span class="pl-k">K_distractor<span class="pl-w">× {w_distractor}</span></span>
  <span class="pl-k">K_topology<span class="pl-w">× {w_topology}</span></span>
</div>
<div class="pl-section">
  <span class="pl-head">Level 2</span>
  <span class="pl-k pl-k-l2">K_road</span>
  <span class="pl-k pl-k-l2">K_weather</span>
  <span class="pl-k pl-k-l2">K_nav</span>
  <span class="pl-k pl-k-l2">K_behavior</span>
  <span style="color:#888;margin-left:6px">(mean, inside each L1 family)</span>
</div>
</div>''')

    # Stats cards
    L.append('<div class="cards">')
    for val, lbl, c in [
        (meta["n_samples"], "Samples", "c1"),
        (meta["n_families"], "L1 Families", "c2"),
        (meta["n_groups"], "L2 Groups", "c4"),
        (meta["n_multi_groups"], "Multi", "c5"),
        (meta["n_singletons"], "Singleton", "c6"),
        (f'{meta.get("l1_threshold", 0):.3f}', "L1 Threshold", "c3"),
    ]:
        L.append(f'<div class="card {c}"><div class="n">{val}</div><div class="l">{lbl}</div></div>')
    L.append("</div>")

    # Filter bar
    L.append("""<div class="bar">
<input type="text" id="q" placeholder="Search key, family, or group id..." oninput="filt()">
<select id="sizeFilter" onchange="filt()">
  <option value="all">All family sizes</option>
  <option value="multi">Multi only (size > 1)</option>
  <option value="single">Singletons only</option>
  <option value="big">Big (size ≥ 5)</option>
  <option value="huge">Huge (size ≥ 20)</option>
</select>
<label><input type="checkbox" id="showSingles" checked onchange="toggleSingles()"> Show singleton families</label>
<div class="spacer"></div>
<span class="act" onclick="togFam(true)">Expand All Families</span>
<span class="act" onclick="togFam(false)">Collapse All</span>
</div>""")

    def _render_family(fam, is_singleton: bool):
        fam_desc = fam.get("description", {})
        groups = [g for b in fam.get("buckets", []) for g in b.get("groups", [])]
        # Inside family, sort L2 groups by size
        groups_sorted = sorted(groups, key=lambda g: (-g["size"], g["id"]))
        cls = "single" if is_singleton else "multi"
        L.append(f'<div class="fam {cls}" data-size="{fam["size"]}">')
        L.append('<div class="fam-h" onclick="this.parentElement.classList.toggle(\'open\')">')
        L.append(f'<span class="fam-title">'
                 f'<span class="fid">{fam["label"]}</span>'
                 f'<span class="fsize">{fam["size"]}</span> samples '
                 f'<span style="color:#888;font-size:.85em">&bull; {len(groups)} L2 groups</span>'
                 f'</span>')
        l2_thr = fam.get("l2_threshold")
        thr_str = (f'<span style="color:#888;font-size:.75em">L2 thresh={l2_thr:.3f}</span>'
                   if l2_thr is not None else "")
        L.append(thr_str)
        L.append("</div>")
        L.append(_family_desc_html(fam_desc))

        L.append('<div class="fam-body">')
        for grp in groups_sorted:
            g_desc = grp.get("description", {})
            L.append(f'<div class="grp" data-size="{grp["size"]}">')
            L.append('<div class="grp-h" onclick="this.parentElement.classList.toggle(\'open\')">')
            L.append(f'<span class="gid">{grp["id"]}</span>'
                     f'<span class="gsize">{grp["size"]}</span>')
            L.append(_group_ctx_html(g_desc))
            L.append(f'<span class="leader-tag">leader: {grp["leader"]}</span>')
            L.append('</div>')
            L.append('<div class="grp-body">')
            for mem in grp["members"]:
                key = mem["key"]
                url = f"{base_url}/{key}.html"
                leader_tag = '<span class="leader">LEADER</span>' if mem.get("is_leader") else ""
                L.append(f'<div class="mem"><a href="{url}" target="_blank">{key}</a>{leader_tag}</div>')
            L.append("</div></div>")
        L.append("</div></div>")

    # Multi-sample L1 families
    L.append('<h2>L1 Families (multi-sample)</h2>')
    L.append('<div id="multi-wrap">')
    for fam in multi_families:
        _render_family(fam, is_singleton=False)
    L.append("</div>")

    # Singleton L1 families
    L.append('<h2>Singleton Families '
             '<span class="toggle-singles" onclick="toggleSinglesSection()" id="singles-label">(show)</span></h2>')
    L.append('<div id="singles-wrap" class="collapsed">')
    for fam in singleton_families:
        _render_family(fam, is_singleton=True)
    L.append("</div>")

    # JS
    L.append("""<script>
function filt() {
    const q = document.getElementById('q').value.toLowerCase();
    const sz = document.getElementById('sizeFilter').value;
    document.querySelectorAll('.fam').forEach(f => {
        const size = parseInt(f.dataset.size);
        const text = f.textContent.toLowerCase();
        let pass = true;
        if (q && !text.includes(q)) pass = false;
        if (sz === 'multi' && size < 2) pass = false;
        if (sz === 'single' && size !== 1) pass = false;
        if (sz === 'big' && size < 5) pass = false;
        if (sz === 'huge' && size < 20) pass = false;
        f.classList.toggle('hidden', !pass);
    });
}
function togFam(o) {
    document.querySelectorAll('.fam').forEach(f => {
        if (o) f.classList.add('open');
        else f.classList.remove('open');
    });
}
function toggleSingles() {
    const show = document.getElementById('showSingles').checked;
    document.getElementById('singles-wrap').style.display = show ? '' : 'none';
    document.querySelectorAll('h2')[1].style.display = show ? '' : 'none';
}
function toggleSinglesSection() {
    const wrap = document.getElementById('singles-wrap');
    const label = document.getElementById('singles-label');
    wrap.classList.toggle('collapsed');
    label.textContent = wrap.classList.contains('collapsed') ? '(show)' : '(hide)';
}
</script></body></html>""")

    with open(output_path, "w") as f:
        f.write("\n".join(L))
    n_multi = len(multi_families)
    n_single = len(singleton_families)
    print(f"Done: {output_path}  (multi-families: {n_multi}, singletons: {n_single})")


def main():
    p = argparse.ArgumentParser(
        description="Render 2-level QA-kernel clustering results as HTML"
    )
    p.add_argument("--path", "--clusters", dest="clusters", required=True,
                   help="Path to clusters.json produced by qa_kernel_filter.py")
    p.add_argument("--output", default=None,
                   help="HTML output path (default: <clusters_dir>/qa_clustering_results.html)")
    p.add_argument("--base-url", default="http://20.110.191.2:8999/outputs/nuscenes_full_final",
                   help="Base URL for per-scene HTML pages (prepended to {key}.html)")
    p.add_argument("--graph-dir", default=None,
                   help="Dir with *_verified.json (for auto-annotation if descriptions missing)")
    p.add_argument("--records-dir", default=None,
                   help="Dir with scene records for nav_command lookup")
    p.add_argument("--reannotate", action="store_true",
                   help="Force re-annotation, overwriting cached descriptions in clusters.json")
    args = p.parse_args()

    clusters_path = Path(args.clusters)
    output_path = args.output or str(clusters_path.parent / "qa_clustering_results.html")
    render(str(clusters_path), output_path, args.base_url,
           graph_dir=args.graph_dir, records_dir=args.records_dir,
           reannotate=args.reannotate)


if __name__ == "__main__":
    main()
