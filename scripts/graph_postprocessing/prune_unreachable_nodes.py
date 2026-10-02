#!/usr/bin/env python3
"""
Prune nodes with no causal path to ego from `_verified.json` graphs, in place.

Same reverse-BFS algorithm as `scripts/causal_graph_extraction/prune_graphs.py`,
but operates on the post-processing file set (`*_verified.json`) and
overwrites them rather than writing a new suffix.

Usage:
    python scripts/graph_postprocessing/prune_unreachable_nodes.py \
        --input-dir /spiral_hdd_3/workspace/navvrat/Neurips/outputs/nuscenes_full_final \
        [--dry-run] [--workers 16]
"""

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from typing import Any


def build_reachable_set(ego_edges: list[dict[str, Any]],
                        inter_edges: list[dict[str, Any]]) -> set[str]:
    """BFS from ego on the reversed graph. Returns set of reachable node IDs."""
    rev: defaultdict[str, set[str]] = defaultdict(set)
    for e in ego_edges:
        src = e.get("source", e.get("from", ""))
        if src and src != "ego":
            rev["ego"].add(src)
    for e in inter_edges:
        src = e.get("source", e.get("from", ""))
        tgt = e.get("target", e.get("to", ""))
        if e.get("type", "") == "CONFOUNDED_BY":
            rev[src].add(tgt)
            rev[tgt].add(src)
        else:
            rev[tgt].add(src)
    visited = {"ego"}
    q = deque(["ego"])
    while q:
        node = q.popleft()
        for nb in rev[node]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    visited.discard("ego")
    return visited


def prune_graph(graph: dict[str, Any]) -> list[str]:
    """Mutate graph: move unreachable nodes to distractors, drop their edges.
    Returns list of pruned node IDs."""
    nodes = graph.get("nodes", [])
    ego_edges = graph.get("ego_edges", [])
    inter_edges = graph.get("inter_node_edges", [])
    distractors = graph.get("distractors", [])

    reachable = build_reachable_set(ego_edges, inter_edges)
    existing_dist_ids = {d.get("id") for d in distractors}

    keep = []
    pruned_ids: list[str] = []
    for node in nodes:
        nid = node.get("id")
        if nid == "ego" or nid in reachable:
            keep.append(node)
            continue
        pruned_ids.append(nid)
        if nid not in existing_dist_ids:
            distractors.append({
                "id": nid,
                "type": node.get("type", ""),
                "category": node.get("category", ""),
                "position_relative_to_ego": node.get("position_relative_to_ego", ""),
                "rejection_reason": "PRUNED: No causal path to ego.",
            })
            existing_dist_ids.add(nid)

    if pruned_ids:
        pset = set(pruned_ids)
        graph["inter_node_edges"] = [
            e for e in inter_edges
            if e.get("source", e.get("from", "")) not in pset
            and e.get("target", e.get("to", "")) not in pset
        ]

    graph["nodes"] = keep
    graph["distractors"] = distractors
    return pruned_ids


def process_one(fp_dry: tuple[str, bool]) -> tuple[str, list[str], str]:
    """Worker. Returns (filename, pruned_ids, status)."""
    fp, dry_run = fp_dry
    try:
        with open(fp) as fh:
            graph = json.load(fh)
    except (OSError, json.JSONDecodeError) as err:
        return (os.path.basename(fp), [], f"ERROR: {err}")

    pruned_ids = prune_graph(graph)
    if not pruned_ids:
        return (os.path.basename(fp), [], "no-op")

    if not dry_run:
        with open(fp, "w") as fh:
            json.dump(graph, fh, indent=2)
    return (os.path.basename(fp), pruned_ids, "written")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="Directory with *_verified.json graphs")
    ap.add_argument("--dry-run", action="store_true", help="Don't write changes")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 16),
                    help="Parallel workers (default: min(cpu_count, 16))")
    args = ap.parse_args()

    files = sorted(glob(os.path.join(args.input_dir, "*_verified.json")))
    if not files:
        print(f"No *_verified.json files found under {args.input_dir}")
        sys.exit(1)

    print(f"Scanning {len(files)} verified graphs with {args.workers} workers"
          f"{' (dry-run)' if args.dry_run else ''}...")

    files_modified = 0
    total_pruned = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, (fp, args.dry_run)) for fp in files]
        for fut in as_completed(futures):
            name, pruned_ids, status = fut.result()
            if status.startswith("ERROR"):
                errors += 1
                print(f"  {name}: {status}")
                continue
            if not pruned_ids:
                continue
            files_modified += 1
            total_pruned += len(pruned_ids)
            print(f"  {name}: pruned {len(pruned_ids)} → {pruned_ids}")

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Summary:")
    print(f"  Files scanned        : {len(files)}")
    print(f"  Files with prunes    : {files_modified}")
    print(f"  Total nodes pruned   : {total_pruned}")
    if errors:
        print(f"  Read errors          : {errors}")


if __name__ == "__main__":
    main()
