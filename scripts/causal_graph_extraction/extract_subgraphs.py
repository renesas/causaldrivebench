import argparse
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import json
from pathlib import Path


def extract_subgraphs(graph, active_only=True, include_dormant_in_confounding=False):
    """
    Enumerate canonical causal sub-structures from an ego-centric graph.

    Priority (most specific wins at the pair level):
        Confounding > Collider ; Chain is orthogonal (requires non-ego-parent ancestor).
    Direct is extracted per edge and annotated with the node's role in any
    confounding structure, which determines which Direct QA types are valid:
        - confounder  -> R0 CaI only (CR/SC would require propagation, escalate to Confounding)
        - mediator    -> full Direct slate (CaI, CR, SC)
        - independent -> full Direct slate (CaI, CR, SC)

    Parameters
    ----------
    graph : dict
        Keys: "ego_edges" (list of {source, target:'ego', active, effect_type, ...}),
              "inter_node_edges" (list of {source, target, relation, ...}).
        Relation "AFFECTS" is a directed inter-node edge.
        Relation "CONFOUNDED_BY" marks unobserved confounding.
    active_only : bool
        If True, only edges with active=True contribute ego-parents.
    include_dormant_in_confounding : bool
        If True, dormant ego-parents still count for confounding detection
        (useful for Rung-3 "what-if-activated" analysis) but not for
        Chain/Collider extraction.

    Returns
    -------
    dict with keys: direct, chain, collider, confounding_observed, confounding_unobserved.
    """
    ego_edges   = graph.get("ego_edges", [])
    inter_edges = graph.get("inter_node_edges", [])

    active_parents = {e["source"] for e in ego_edges if e.get("active", False)}
    all_parents    = {e["source"] for e in ego_edges}
    conf_parents   = all_parents if include_dormant_in_confounding else active_parents
    chain_collider_parents = active_parents if active_only else all_parents

    # Inter-node adjacency
    inter_adj   = defaultdict(set)          # A -> {B : A -> B}
    unobs_pairs = set()                     # frozenset({A, B})
    for e in inter_edges:
        rel = e.get("relation")
        if rel == "AFFECTS":
            inter_adj[e["source"]].add(e["target"])
        elif rel == "CONFOUNDED_BY":
            unobs_pairs.add(frozenset((e["source"], e["target"])))

    def directed_edge(a, b):
        if b in inter_adj.get(a, ()):
            return (a, b)
        if a in inter_adj.get(b, ()):
            return (b, a)
        return None

    out = {"direct": [], "chain": [], "collider": [],
           "confounding_observed": [], "confounding_unobserved": []}

    # -------- (1) Pair-level classification among ego-parents --------
    confounded_pairs  = set()
    confounder_nodes  = set()   # has an inter-node edge to another ego-parent
    mediator_nodes    = set()   # receives an inter-node edge from another ego-parent
    unobs_only_nodes  = set()   # confounded only via unobserved U, no observed role

    for a, b in combinations(sorted(conf_parents), 2):
        pair = frozenset((a, b))
        d = directed_edge(a, b)
        if d is not None:
            src, tgt = d  # src confounds ego via tgt
            out["confounding_observed"].append({
                "confounder": src,
                "mediator":   tgt,
                "target":     "ego",
                "edges": [
                    {"source": src, "target": tgt,   "kind": "inter"},
                    {"source": src, "target": "ego", "kind": "ego"},
                    {"source": tgt, "target": "ego", "kind": "ego"},
                ],
            })
            confounded_pairs.add(pair)
            confounder_nodes.add(src)
            mediator_nodes.add(tgt)
        elif pair in unobs_pairs:
            out["confounding_unobserved"].append({
                "nodes":  sorted(pair),
                "target": "ego",
                "edges": [
                    {"source": "U", "target": a,     "kind": "unobserved"},
                    {"source": "U", "target": b,     "kind": "unobserved"},
                    {"source": a,   "target": "ego", "kind": "ego"},
                    {"source": b,   "target": "ego", "kind": "ego"},
                ],
            })
            confounded_pairs.add(pair)
            unobs_only_nodes.update(pair)

    # Unobserved confounding doesn't make a node a "confounder" in the
    # propagation sense — intervening on either side doesn't shift the other
    # through an observed mechanism. Treat as independent for Direct QAs.
    unobs_only_nodes -= (confounder_nodes | mediator_nodes)

    # -------- (2) Colliders: active ego-parents with no edge between and not confounded --------
    # A pair forms a collider only if neither node plays a confounder/mediator role anywhere.
    chain_mediators = {sg["mediator"] for sg in out["chain"]}
    busy_nodes = confounder_nodes | mediator_nodes | unobs_only_nodes | chain_mediators

    # (2) collider loop — replace current filter:
    for a, b in combinations(sorted(chain_collider_parents), 2):
        pair = frozenset((a, b))
        if pair in confounded_pairs:
            continue
        if a in busy_nodes or b in busy_nodes:
            continue                          # don't re-cast a confounder/mediator as a collider parent
        out["collider"].append({
            "parents": sorted(pair),
            "target":  "ego",
            "edges": [
                {"source": a, "target": "ego", "kind": "ego"},
                {"source": b, "target": "ego", "kind": "ego"},
            ],
        })

    # -------- (3) Chains: A -> B -> ego where A is NOT itself an ego-parent --------
    for a, targets in inter_adj.items():
        if a in all_parents:          # would form confounding, not a pure chain
            continue
        for b in targets:
            if b not in chain_collider_parents:
                continue
            out["chain"].append({
                "distal":   a,
                "mediator": b,
                "target":   "ego",
                "edges": [
                    {"source": a, "target": b,     "kind": "inter"},
                    {"source": b, "target": "ego", "kind": "ego"},
                ],
            })

    # -------- (4) Direct edges, with role annotation --------
    for e in ego_edges:
        if active_only and not e.get("active", False):
            continue
        src = e["source"]
        if src in confounder_nodes:
            role = "confounder"
            allowed = ["CaI"]                 # CR/SC escalate to Confounding
        elif src in mediator_nodes:
            role = "mediator"
            allowed = ["CaI", "CR", "SC"]
        else:
            role = "independent"
            allowed = ["CaI", "CR", "SC"]
        out["direct"].append({
            "source":            src,
            "target":            "ego",
            "effect_type":       e.get("effect_type"),
            "causal_status":     e.get("causal_status"),
            "confounding_role":  role,
            "allowed_direct_qas": allowed,
            "edges": [{"source": src, "target": "ego", "kind": "ego"}],
        })

    return out


# -------- Convenience: map subgraph types to the QA templates they license --------
QA_TEMPLATES = {
    "direct": {
        "R0": ["CaI"],
        "R1": [],
        "R2": [],
        "R3": ["CR", "SC"],
    },
    "chain": {
        "R0": ["CaA", "AR"],
        "R1": [],
        "R2": ["CDE"],
        "R3": ["NIE", "NDE"],
    },
    "confounding": {
        "R0": ["CaI"],
        "R1": [],
        "R2": ["CfI", "BAS", "CDE"],
        "R3": ["CR", "NIE", "NDE", "SC"],
    },
    "collider": {
        "R0": [],
        "R1": ["CB"],
        "R2": [],
        "R3": ["NC"],
    },
}


def licensed_qas(subgraphs):
    """
    Given the output of extract_subgraphs, return a flat list of
    (subgraph_type, subgraph_record, rung, qa_type) tuples the generator
    is licensed to attempt. Respects the confounder restriction on Direct.
    """
    jobs = []

    for sg in subgraphs["direct"]:
        allowed = set(sg["allowed_direct_qas"])
        for rung, qas in QA_TEMPLATES["direct"].items():
            for qa in qas:
                if qa in allowed:
                    jobs.append(("direct", sg, rung, qa))

    for sg in subgraphs["chain"]:
        for rung, qas in QA_TEMPLATES["chain"].items():
            for qa in qas:
                jobs.append(("chain", sg, rung, qa))

    for sg in (subgraphs["confounding_observed"]
               + subgraphs["confounding_unobserved"]):
        for rung, qas in QA_TEMPLATES["confounding"].items():
            for qa in qas:
                jobs.append(("confounding", sg, rung, qa))

    for sg in subgraphs["collider"]:
        for rung, qas in QA_TEMPLATES["collider"].items():
            for qa in qas:
                jobs.append(("collider", sg, rung, qa))

    return jobs


from pathlib import Path


def save_subgraph_analysis_md(graph, output_path, scene_id=None,
                                active_only=True, include_dormant_in_confounding=False):
    """
    Run extract_subgraphs and persist the identified structures as Markdown.
    """
    subgraphs = extract_subgraphs(
        graph,
        active_only=active_only,
        include_dormant_in_confounding=include_dormant_in_confounding,
    )

    lines = []
    lines.append(f"# Subgraph analysis: {scene_id or 'unnamed'}\n")

    # ---- Counts ----
    lines.append("## Counts\n")
    lines.append("| structure | n |")
    lines.append("|---|---|")
    lines.append(f"| direct | {len(subgraphs['direct'])} |")
    lines.append(f"| chain | {len(subgraphs['chain'])} |")
    lines.append(f"| collider | {len(subgraphs['collider'])} |")
    lines.append(f"| confounding (observed) | {len(subgraphs['confounding_observed'])} |")
    lines.append(f"| confounding (unobserved) | {len(subgraphs['confounding_unobserved'])} |\n")

    # ---- Subgraphs ----
    lines.append("## Subgraphs\n")

    if subgraphs["direct"]:
        lines.append("### Direct\n")
        lines.append("| id | source | effect | status | role |")
        lines.append("|---|---|---|---|---|")
        for i, sg in enumerate(subgraphs["direct"]):
            sid = f"D{i}"
            lines.append(
                f"| `{sid}` | `{sg['source']}` → ego "
                f"| {sg.get('effect_type', '')} "
                f"| {sg.get('causal_status', '')} "
                f"| {sg['confounding_role']} |"
            )
        lines.append("")

    if subgraphs["chain"]:
        lines.append("### Chain\n")
        lines.append("| id | path |")
        lines.append("|---|---|")
        for i, sg in enumerate(subgraphs["chain"]):
            sid = f"CH{i}"
            lines.append(f"| `{sid}` | `{sg['distal']}` → `{sg['mediator']}` → ego |")
        lines.append("")

    if subgraphs["collider"]:
        lines.append("### Collider\n")
        lines.append("| id | parents |")
        lines.append("|---|---|")
        for i, sg in enumerate(subgraphs["collider"]):
            sid = f"CL{i}"
            parents = ", ".join(f"`{p}`" for p in sg["parents"])
            lines.append(f"| `{sid}` | {parents} → ego |")
        lines.append("")

    if subgraphs["confounding_observed"]:
        lines.append("### Confounding (observed)\n")
        lines.append("| id | confounder | mediator |")
        lines.append("|---|---|---|")
        for i, sg in enumerate(subgraphs["confounding_observed"]):
            sid = f"CO{i}"
            lines.append(
                f"| `{sid}` | `{sg['confounder']}` | `{sg['mediator']}` |"
            )
        lines.append("")

    if subgraphs["confounding_unobserved"]:
        lines.append("### Confounding (unobserved)\n")
        lines.append("| id | nodes sharing U |")
        lines.append("|---|---|")
        for i, sg in enumerate(subgraphs["confounding_unobserved"]):
            sid = f"CU{i}"
            nodes = ", ".join(f"`{n}`" for n in sg["nodes"])
            lines.append(f"| `{sid}` | {nodes} |")
        lines.append("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))

    total = sum(len(v) for v in subgraphs.values())
    return total

# -------- CLI: batch extraction over a directory of graphs --------

_KNOWN_SUFFIXES = ("_verified", "_fixed", "_naren_edited", "_pruned", "_subgraphs")


def _resolve_graph_path(graphs_dir: Path, scene_id: str):
    """Return best available graph: verified > fixed > edited > pruned."""
    for suffix in ("_verified.json", "_fixed.json", "_naren_edited.json", "_pruned.json"):
        p = graphs_dir / f"{scene_id}{suffix}"
        if p.exists():
            return p
    return None


def _extract_scene_ids(graphs_dir: Path) -> list:
    """Discover unique scene IDs from graph files in the directory."""
    ids = set()
    for f in graphs_dir.glob("*.json"):
        name = f.stem
        for suffix in _KNOWN_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        ids.add(name)
    return sorted(ids)


def _process_single(args_tuple):
    """Process one scene: resolve graph, extract subgraphs, write Markdown."""
    graphs_dir_str, scene_id = args_tuple
    graphs_dir = Path(graphs_dir_str)
    graph_path = _resolve_graph_path(graphs_dir, scene_id)
    if graph_path is None:
        return (scene_id, -1, "no graph found")

    try:
        graph = json.loads(graph_path.read_text())
    except (json.JSONDecodeError, KeyError):
        return (scene_id, -2, "invalid JSON")

    output_path = graphs_dir / f"{scene_id}_subgraphs.md"
    n_subgraphs = save_subgraph_analysis_md(graph, output_path, scene_id=scene_id)
    return (scene_id, n_subgraphs, f"from {graph_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract causal subgraphs from all graphs in a directory"
    )
    parser.add_argument("input", nargs="+", help="JSON files or directory")
    parser.add_argument("--workers", type=int,
                        default=min(os.cpu_count() or 4, 16))
    args = parser.parse_args()

    # Collect input directory (same pattern as prune_graphs.py)
    graphs_dir = None
    for pat in args.input:
        p = Path(pat)
        if p.is_dir():
            graphs_dir = p
            break

    if graphs_dir is None:
        print("Error: provide a directory containing graph JSON files")
        return

    scene_ids = _extract_scene_ids(graphs_dir)
    tasks = [(str(graphs_dir), sid) for sid in scene_ids
             if _resolve_graph_path(graphs_dir, sid) is not None]

    print(f"Extracting subgraphs for {len(tasks)} scenes...")
    processed = 0
    total_subgraphs = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_single, t): t for t in tasks}
        for future in as_completed(futures):
            sid, n_subgraphs, msg = future.result()
            if n_subgraphs < 0:
                print(f"  SKIP {sid}: {msg}")
            else:
                processed += 1
                total_subgraphs += n_subgraphs
                print(f"  OK   {sid}: {n_subgraphs} subgraphs ({msg})")

    print(f"\nTotal: {len(tasks)} scenes | {processed} processed | {total_subgraphs} subgraphs")


if __name__ == "__main__":
    main()
