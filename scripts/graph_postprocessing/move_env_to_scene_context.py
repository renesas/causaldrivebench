#!/usr/bin/env python3
"""Move ENVIRONMENT nodes into scene_context.environment_conditions.

For each matching graph file:
  1. Find ENVIRONMENT-type nodes (wet_road, night, rain, fog, sun_glare, etc.)
  2. Aggregate their state/label descriptions into a natural-language string
  3. Add that string as scene_context.environment_conditions
  4. Remove the ENVIRONMENT nodes from graph.nodes
  5. Remove any ego_edges or inter_node_edges referencing those node IDs

Usage:
    # Preview changes
    python scripts/graph_postprocessing/move_env_to_scene_context.py \
        --graph-dir /spiral_hdd_3/workspace/navvrat/Neurips/outputs/nuscenes_full_final/ \
        --dry-run

    # Apply in-place to _verified.json (default pattern)
    python scripts/graph_postprocessing/move_env_to_scene_context.py \
        --graph-dir /spiral_hdd_3/workspace/navvrat/Neurips/outputs/nuscenes_full_final/

    # Apply to multiple variants
    python scripts/graph_postprocessing/move_env_to_scene_context.py \
        --graph-dir ... --pattern '*_verified.json'
    python scripts/graph_postprocessing/move_env_to_scene_context.py \
        --graph-dir ... --pattern '*_fixed.json'

NOTE: Does NOT update downstream QA files (active/dormant/distractor) — if any
QA references an ENVIRONMENT node ID, regenerate QAs after running this.
"""

import argparse
import copy
import glob
import json
import os
from collections import Counter


def build_environment_conditions(env_nodes: list) -> str:
    """Aggregate env node descriptions into a natural-language string.

    Prefers `state` (most descriptive), falls back to `label`, then `category`.
    Deduplicates case-insensitively while preserving order.
    """
    parts = []
    for n in env_nodes:
        state = (n.get("state") or "").strip()
        label = (n.get("label") or "").strip()
        cat = (n.get("category") or "").strip()
        desc = state or label or cat
        if desc:
            parts.append(desc)

    seen, unique = set(), []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return "; ".join(unique)


def process_graph(graph: dict) -> tuple[dict, dict]:
    """Return (new_graph, stats). Does not mutate input."""
    g = copy.deepcopy(graph)
    stats = {"env_nodes_removed": 0, "ego_edges_removed": 0, "inter_edges_removed": 0}

    env_nodes = [n for n in g.get("nodes", []) if (n.get("type") or "").upper() == "ENVIRONMENT"]
    if not env_nodes:
        return g, stats

    env_ids = {n["id"] for n in env_nodes}
    stats["env_nodes_removed"] = len(env_nodes)

    env_str = build_environment_conditions(env_nodes)
    if env_str:
        sc = g.setdefault("scene_context", {})
        existing = (sc.get("environment_conditions") or "").strip()
        if existing and existing.lower() != env_str.lower():
            sc["environment_conditions"] = f"{existing}. {env_str}"
        else:
            sc["environment_conditions"] = env_str

    g["nodes"] = [n for n in g.get("nodes", []) if n["id"] not in env_ids]

    new_ego = []
    for e in g.get("ego_edges", []):
        if e.get("source") in env_ids or e.get("target") in env_ids:
            stats["ego_edges_removed"] += 1
            continue
        new_ego.append(e)
    g["ego_edges"] = new_ego

    new_inter = []
    for e in g.get("inter_node_edges", []):
        if e.get("source") in env_ids or e.get("target") in env_ids:
            stats["inter_edges_removed"] += 1
            continue
        new_inter.append(e)
    g["inter_node_edges"] = new_inter

    return g, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph-dir", required=True,
                        help="Directory containing graph JSON files")
    parser.add_argument("--pattern", default="*_verified.json",
                        help="Glob pattern for files to process (default: *_verified.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report changes without writing files")
    parser.add_argument("--out-suffix", default=None,
                        help="If set, write to <stem><suffix>.json instead of overwriting. "
                             "e.g. --out-suffix _env_cleaned")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-file changes")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.graph_dir, args.pattern)))
    print(f"Found {len(files)} files matching '{args.pattern}' in {args.graph_dir}")
    if not files:
        return

    totals = {"modified": 0, "env_nodes_removed": 0, "ego_edges_removed": 0, "inter_edges_removed": 0}
    category_counts = Counter()
    samples = []  # (sid, env_str) for dry-run preview
    written = 0
    write_errors = 0

    for fpath in files:
        try:
            with open(fpath) as f:
                graph = json.load(f)
        except Exception as e:
            print(f"  SKIP {os.path.basename(fpath)}: {e}")
            continue

        env_nodes = [n for n in graph.get("nodes", []) if (n.get("type") or "").upper() == "ENVIRONMENT"]
        if not env_nodes:
            continue

        for n in env_nodes:
            category_counts[n.get("category", "?")] += 1

        new_graph, stats = process_graph(graph)
        totals["modified"] += 1
        for k in ("env_nodes_removed", "ego_edges_removed", "inter_edges_removed"):
            totals[k] += stats[k]

        if len(samples) < 5:
            sid = os.path.basename(fpath).replace(".json", "")
            env_str = new_graph.get("scene_context", {}).get("environment_conditions", "")
            samples.append((sid, env_str))

        if args.verbose:
            sid = os.path.basename(fpath).replace(".json", "")
            print(f"  {sid}: -{stats['env_nodes_removed']} nodes, "
                  f"-{stats['ego_edges_removed']} ego_edges, "
                  f"-{stats['inter_edges_removed']} inter_edges")

        if not args.dry_run:
            out_path = fpath
            if args.out_suffix:
                base = fpath[:-5] if fpath.endswith(".json") else fpath
                out_path = base + args.out_suffix + ".json"
            try:
                with open(out_path, "w") as f:
                    json.dump(new_graph, f, indent=2)
                written += 1
            except Exception as e:
                write_errors += 1
                print(f"  WRITE ERROR {os.path.basename(out_path)}: {e}")

    print(f"\nSummary:")
    print(f"  Files scanned:            {len(files)}")
    print(f"  Files with env nodes:     {totals['modified']}")
    print(f"  ENVIRONMENT nodes removed: {totals['env_nodes_removed']}")
    print(f"  ego_edges removed:        {totals['ego_edges_removed']}")
    print(f"  inter_edges removed:      {totals['inter_edges_removed']}")
    if args.dry_run:
        print(f"  (dry-run — no files written)")
    else:
        print(f"  Files written:            {written}")
        if write_errors:
            print(f"  Write errors:             {write_errors}")
    print(f"\n  Category breakdown:")
    for cat, n in category_counts.most_common():
        print(f"    {cat}: {n}")

    if samples:
        print(f"\n  Sample environment_conditions (first {len(samples)}):")
        for sid, env_str in samples:
            print(f"    • {sid}")
            print(f"        → {env_str}")


if __name__ == "__main__":
    main()
