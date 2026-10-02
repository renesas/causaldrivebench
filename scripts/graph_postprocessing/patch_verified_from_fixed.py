#!/usr/bin/env python3
"""
Patch `_verified.json` graphs in-place by restoring orphan nodes that the
buggy (pre-fix) `fix_causal_graphs.py` Rule 2 had demoted + Rule 3 then
deleted — using `_fixed.json` (produced by the corrected script) as the
source of truth for the missing node definitions.

An "orphan" is a node ID referenced in `ego_edges` or `inter_node_edges`
but absent from both `nodes` and `distractors` in the verified graph.

Scope:
    - Only patches scenes where BOTH `_verified.json` and `_fixed.json` exist.
    - Only restores nodes the orphan IDs point to — does NOT touch edges,
      `scene_context`, or `behavior_explanation` (so the LLM-aligned text
      from the align_scene_description pass is preserved).

Usage:
    python scripts/graph_postprocessing/patch_verified_from_fixed.py \
        --input-dir /spiral_hdd_3/workspace/navvrat/Neurips/outputs/nuscenes_full_final \
        [--dry-run]
"""
import argparse
import json
import os
from glob import glob


def _all_edge_node_ids(graph):
    ids = set()
    for e in graph.get("ego_edges", []):
        ids.add(e.get("source"))
        ids.add(e.get("target"))
    for e in graph.get("inter_node_edges", []):
        ids.add(e.get("source"))
        ids.add(e.get("target"))
    ids.discard(None)
    return ids


def _defined_ids(graph):
    ids = {n["id"] for n in graph.get("nodes", []) if "id" in n}
    ids.update(d["id"] for d in graph.get("distractors", []) if "id" in d)
    return ids


def patch_one(verified_path, fixed_path):
    with open(verified_path) as fh:
        verified = json.load(fh)
    with open(fixed_path) as fh:
        fixed = json.load(fh)

    orphan_ids = _all_edge_node_ids(verified) - _defined_ids(verified) - {"ego"}
    if not orphan_ids:
        return None

    fixed_nodes = {n["id"]: n for n in fixed.get("nodes", []) if "id" in n}
    fixed_distractors = {d["id"]: d for d in fixed.get("distractors", []) if "id" in d}

    restored = []
    still_missing = []
    for nid in sorted(orphan_ids):
        if nid in fixed_nodes:
            verified["nodes"].append(fixed_nodes[nid])
            restored.append((nid, "nodes"))
        elif nid in fixed_distractors:
            verified["nodes"].append(fixed_distractors[nid])
            restored.append((nid, "distractors"))
        else:
            still_missing.append(nid)

    return {
        "restored": restored,
        "still_missing": still_missing,
        "graph": verified,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    verified_files = sorted(glob(os.path.join(args.input_dir, "*_verified.json")))
    print(f"Scanning {len(verified_files)} verified graphs...")

    scenes_patched = 0
    nodes_restored = 0
    no_fixed_counterpart = 0
    id_counter = {}
    unresolved = []

    for vpath in verified_files:
        scene_id = os.path.basename(vpath).rsplit("_verified.json", 1)[0]
        fpath = os.path.join(args.input_dir, f"{scene_id}_fixed.json")
        if not os.path.isfile(fpath):
            # No fixed counterpart — scene was not modified by fix_causal_graphs
            # (or fix hasn't been run for it); nothing to reconcile.
            no_fixed_counterpart += 1
            continue

        result = patch_one(vpath, fpath)
        if result is None:
            continue

        scenes_patched += 1
        nodes_restored += len(result["restored"])
        for nid, _src in result["restored"]:
            id_counter[nid] = id_counter.get(nid, 0) + 1
        if result["still_missing"]:
            unresolved.append((scene_id, result["still_missing"]))

        if not args.dry_run:
            with open(vpath, "w") as fh:
                json.dump(result["graph"], fh, indent=2)

        brief = ", ".join(f"{nid}←{src}" for nid, src in result["restored"])
        print(f"  [{scene_id}] restored {len(result['restored'])} node(s): {brief}")

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Summary:")
    print(f"  Verified graphs scanned         : {len(verified_files)}")
    print(f"  No _fixed.json counterpart     : {no_fixed_counterpart}")
    print(f"  Verified graphs patched         : {scenes_patched}")
    print(f"  Node definitions restored       : {nodes_restored}")
    if id_counter:
        top = sorted(id_counter.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print(f"  Top restored IDs                : {top}")
    if unresolved:
        print(f"  Scenes with still-orphan IDs    : {len(unresolved)}")
        for sid, ids in unresolved[:10]:
            print(f"    {sid}: {ids}")


if __name__ == "__main__":
    main()
