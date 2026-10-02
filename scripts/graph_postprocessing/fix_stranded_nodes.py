#!/usr/bin/env python3
"""Fix stranded nodes (not reachable from ego) in verified causal graphs.

A node is "stranded" if it is NOT reachable from ego via any edge path (ego_edges
or inter_node_edges, treated as undirected for reachability). This catches:
  - Nodes with no edges at all
  - Clusters of non-ego nodes connected to each other but with no path to ego

Rules:
  1. OTHER_AGENT stranded → move to `distractors`
  2. TRAFFIC_CONTROL / ROAD_OBSTACLE / ENVIRONMENT stranded → remove entirely
  3. Clean up inter_node_edges that reference any demoted OR removed IDs
     (keeps edges consistent with remaining nodes)

Usage:
    # Preview
    python scripts/graph_postprocessing/fix_stranded_nodes.py \
        --graph-dir /path/to/nuscenes_full_final --dry-run

    # Apply
    python scripts/graph_postprocessing/fix_stranded_nodes.py \
        --graph-dir /path/to/nuscenes_full_final
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


DEMOTE_TO_DISTRACTOR = {"OTHER_AGENT"}
REMOVE_TYPES = {"TRAFFIC_CONTROL", "ROAD_OBSTACLE", "ENVIRONMENT"}


def _reachable_from_ego(nodes: list, ego_edges: list, inter_edges: list) -> set:
    """BFS from ego over the undirected edge graph."""
    adj = {}
    for e in ego_edges + inter_edges:
        src, tgt = e.get("source"), e.get("target")
        adj.setdefault(src, set()).add(tgt)
        adj.setdefault(tgt, set()).add(src)

    reachable = set()
    if "ego" not in adj and not any(n.get("id") == "ego" for n in nodes):
        return reachable
    stack = ["ego"]
    while stack:
        n = stack.pop()
        if n in reachable:
            continue
        reachable.add(n)
        stack.extend(adj.get(n, []))
    return reachable


def process_graph(graph: dict) -> tuple[dict, dict]:
    """Return (modified_graph, stats)."""
    g = json.loads(json.dumps(graph))
    stats = Counter()

    nodes = g.get("nodes", [])
    ego_edges = g.get("ego_edges", [])
    inter_edges = g.get("inter_node_edges", [])

    reachable = _reachable_from_ego(nodes, ego_edges, inter_edges)

    kept_nodes = []
    demoted = []
    removed_ids = set()
    demoted_ids = set()

    for n in nodes:
        nid = n.get("id")
        if nid == "ego" or nid in reachable:
            kept_nodes.append(n)
            continue
        ntype = str(n.get("type", "")).upper()
        if ntype in DEMOTE_TO_DISTRACTOR:
            demoted.append(n)
            demoted_ids.add(nid)
            stats[f"demoted_{ntype}"] += 1
        elif ntype in REMOVE_TYPES:
            removed_ids.add(nid)
            stats[f"removed_{ntype}"] += 1
        else:
            removed_ids.add(nid)
            stats["removed_OTHER"] += 1

    g["nodes"] = kept_nodes

    # Demote OTHER_AGENT to distractors (skip duplicates)
    if demoted:
        existing_distractor_ids = {d.get("id") for d in g.get("distractors", [])}
        for n in demoted:
            if n.get("id") not in existing_distractor_ids:
                g.setdefault("distractors", []).append(n)

    # Clean inter_node_edges referencing removed OR demoted IDs
    dead_ids = removed_ids | demoted_ids
    if dead_ids:
        new_inter = []
        for e in inter_edges:
            if e.get("source") in dead_ids or e.get("target") in dead_ids:
                stats["cleaned_inter_edges"] += 1
                continue
            new_inter.append(e)
        g["inter_node_edges"] = new_inter

    return g, stats


def main():
    p = argparse.ArgumentParser(description="Fix stranded nodes in verified graphs")
    p.add_argument("--graph-dir", required=True, help="Dir with *_verified.json files")
    p.add_argument("--dry-run", action="store_true",
                   help="Report changes without writing files")
    p.add_argument("--suffix", default="_verified.json",
                   help="File suffix to process (default: _verified.json)")
    args = p.parse_args()

    graph_dir = Path(args.graph_dir)
    files = sorted(glob.glob(str(graph_dir / f"*{args.suffix}")))
    print(f"Found {len(files)} files matching *{args.suffix}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY'}")
    print()

    totals = Counter()
    files_changed = 0
    changed_examples = []

    for fp in files:
        with open(fp) as f:
            g = json.load(f)
        new_g, stats = process_graph(g)
        if sum(stats.values()) == 0:
            continue
        files_changed += 1
        totals.update(stats)
        if len(changed_examples) < 10:
            changed_examples.append((Path(fp).name, dict(stats)))
        if not args.dry_run:
            with open(fp, "w") as f:
                json.dump(new_g, f, indent=2)

    print(f"Files changed: {files_changed}/{len(files)}")
    print()
    print("Actions:")
    for action, n in totals.most_common():
        print(f"  {n:5d}  {action}")

    if changed_examples:
        print()
        print("Example files:")
        for name, stats in changed_examples:
            stats_str = ", ".join(f"{k}={v}" for k, v in stats.items())
            print(f"  {name}  [{stats_str}]")

    if args.dry_run:
        print("\n(dry-run — no files written)")
    else:
        print(f"\nWrote {files_changed} files in place.")


if __name__ == "__main__":
    main()
