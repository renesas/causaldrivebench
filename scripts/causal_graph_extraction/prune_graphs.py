#!/usr/bin/env python3
"""Prune causal graphs: move nodes with no causal path to ego into distractors.

Algorithm: Reverse-BFS from ego through reversed edges — O(V + E).
Parallelized with ProcessPoolExecutor for fast batch processing.
"""

import argparse
import json
import os
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def build_reachable_set(ego_edges: list[dict[str, Any]], inter_edges: list[dict[str, Any]]) -> set[str]:
    """BFS from ego on the reversed graph. Returns set of reachable node IDs."""
    rev = defaultdict(set)
    for e in ego_edges:
        src = e.get("source", e.get("from", ""))
        if src and src != "ego":
            rev["ego"].add(src)
    for e in inter_edges:
        src = e.get("source", e.get("from", ""))
        tgt = e.get("target", e.get("to", ""))
        etype = e.get("type", "")
        if etype == "CONFOUNDED_BY":
            rev[src].add(tgt)
            rev[tgt].add(src)
        else:
            rev[tgt].add(src)
    visited = {"ego"}
    q = deque(["ego"])
    while q:
        node = q.popleft()
        for neighbor in rev[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    visited.discard("ego")
    return visited


def prune_single(fp_str: str) -> tuple[str, int, list[str]]:
    """Process one file: read, prune, write _pruned.json. Returns (filename, n_pruned, pruned_ids)."""
    fp = Path(fp_str)
    try:
        data = json.loads(fp.read_text())
    except (json.JSONDecodeError, KeyError):
        return (fp.name, -1, [])
    if "nodes" not in data:
        return (fp.name, -2, [])

    nodes = data.get("nodes", [])
    ego_edges = data.get("ego_edges", [])
    inter_edges = data.get("inter_node_edges", [])
    distractors = data.get("distractors", [])

    reachable = build_reachable_set(ego_edges, inter_edges)
    existing_dist_ids = {d.get("id") for d in distractors}

    keep = []
    pruned_ids = []
    for node in nodes:
        nid = node.get("id")
        if nid == "ego" or nid in reachable:
            keep.append(node)
        else:
            pruned_ids.append(nid)
            if nid not in existing_dist_ids:
                distractors.append({
                    "id": nid,
                    "type": node.get("type", ""),
                    "category": node.get("category", ""),
                    "position_relative_to_ego": node.get("position_relative_to_ego", ""),
                    "rejection_reason": "PRUNED: No causal path to ego."
                })
                existing_dist_ids.add(nid)

    if pruned_ids:
        pruned_set = set(pruned_ids)
        data["inter_node_edges"] = [
            e for e in inter_edges
            if e.get("source", e.get("from", "")) not in pruned_set
            and e.get("target", e.get("to", "")) not in pruned_set
        ]

    data["nodes"] = keep
    data["distractors"] = distractors

    out = fp.parent / f"{fp.stem}_pruned.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return (fp.name, len(pruned_ids), pruned_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune unreachable nodes from causal graphs")
    parser.add_argument("input", nargs="+", help="JSON files or directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pruned without modifying")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 16),
                        help="Number of parallel workers")
    args = parser.parse_args()

    files = []
    for pat in args.input:
        p = Path(pat)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            files.append(p)
        else:
            files.extend(sorted(Path(".").glob(pat)))

    files = [f for f in files if f.suffix == ".json"
             and "_changes" not in f.name
             and "_error" not in f.name
             and "_refined" not in f.name
             and "_pruned" not in f.name]

    total_pruned = 0
    files_modified = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prune_single, str(fp)): fp for fp in files}
        for future in as_completed(futures):
            name, n, ids = future.result()
            if n == -1:
                print(f"  SKIP {name}: invalid JSON")
            elif n == -2:
                print(f"  SKIP {name}: no nodes key")
            elif n > 0:
                total_pruned += n
                files_modified += 1
                print(f"  PRUNE {name}: {n} node(s) → {ids}")

    print(f"\nTotal: {len(files)} files | {files_modified} had prunable nodes | {total_pruned} nodes pruned")


if __name__ == "__main__":
    main()
