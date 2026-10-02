#!/usr/bin/env python3
"""Single-level topology-only clustering for diverse subset selection.

One kernel, one clustering pass:

    K_topology  — joint histogram of (pattern_type × source_agent_type) over
                  extracted subgraphs (direct / chain / collider /
                  confounding_observed / confounding_unobserved).

Agglomerative complete-link + elbow-selected distance threshold picks the
cut. One max-weight leader per cluster becomes the representative.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


# ── Shared helpers ───────────────────────────────────────────────────────


def _categorize_agent(cat: str) -> str:
    cat = cat.lower()
    if "pedestrian" in cat or "bicycle" in cat or "motorcycle" in cat:
        return "vru"
    if "truck" in cat or "trailer" in cat:
        return "truck"
    if "bus" in cat:
        return "bus"
    if "car" in cat or "van" in cat or "suv" in cat:
        return "car"
    if "signal" in cat:
        return "signal"
    if "crosswalk" in cat or "crossing" in cat:
        return "crosswalk"
    if ("cone" in cat or "barrier" in cat or "construction" in cat
            or "gate" in cat or "hump" in cat or "bump" in cat):
        return "obstacle"
    if "marking" in cat:
        return "road_marking"
    if "sign" in cat:
        return "sign"
    # Dotted-namespace fallback (nuScenes sub-categories like "movable_object.x")
    if "." in cat:
        prefix = cat.split(".", 1)[0]
        if prefix in ("movable_object", "static_object", "debris"):
            return "obstacle"
        if prefix == "human":
            return "vru"
        if prefix == "vehicle":
            return "car"
        return prefix
    return cat if cat else "unknown"


def _bucket_count(n: int) -> int:
    """Log-ish bucketing: 1→1, 2→2, 3–5→3, 6+→4.
    Collapses count noise so "4 trucks" and "5 trucks" land in the same bucket."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    if n <= 5:
        return 3
    return 4


def _bucket_counter(c: Counter) -> Counter:
    return Counter({k: _bucket_count(v) for k, v in c.items()})


def _weighted_jaccard(feature_counts: list) -> np.ndarray:
    """Weighted Jaccard on count dicts: sim = Σ min(a,b) / Σ max(a,b)."""
    vocab: dict[str, int] = {}
    for c in feature_counts:
        for k in c:
            if k not in vocab:
                vocab[k] = len(vocab)
    n = len(feature_counts)
    if not vocab:
        return np.ones((n, n))

    M = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, c in enumerate(feature_counts):
        for k, v in c.items():
            M[i, vocab[k]] = float(v)

    K = np.zeros((n, n), dtype=np.float64)
    row_sums = M.sum(axis=1)
    for i in range(n):
        min_ij = np.minimum(M[i:i + 1], M)
        sum_min = min_ij.sum(axis=1)
        sum_max = row_sums[i] + row_sums - sum_min
        empty_pair = (row_sums[i] == 0) & (row_sums == 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            K[i] = np.where(sum_max > 0, sum_min / sum_max, 0.0)
        K[i][empty_pair] = 1.0
    return K


# ── Topology kernel ─────────────────────────────────────────────────────


def build_K_topology(samples, graphs, include_dormant: bool = True):
    """K_topology: joint histogram of (pattern_type × source_agent_type).

    Each subgraph extracted from extract_subgraphs contributes one joint bin
    that includes the source-agent category. "direct|car" is a different bin
    from "direct|vru"; "chain|signal>car" differs from "chain|car>signal"
    (order preserved for directed patterns; colliders / unobserved
    confounders are category-order-sorted).

    include_dormant
        If True (default), dormant ego-parents are treated as first-class
        parents — they contribute to direct / chain / collider / confounding
        subgraphs just like active parents.
        If False, only active ego-parents count (matches the stricter
        active_only=True mode of extract_subgraphs).
    """
    import sys
    _extract_dir = str(Path(__file__).resolve().parent.parent / "causal_graph_extraction")
    if _extract_dir not in sys.path:
        _sys_path_inserted = True
        sys.path.insert(0, _extract_dir)
    from extract_subgraphs import extract_subgraphs

    def _cat(node_cats: dict, nid: str) -> str:
        return _categorize_agent(node_cats.get(nid, ""))

    feature_counts: list[Counter] = []
    for s in samples:
        g = graphs.get(s.key, {})
        node_cats = {n["id"]: n.get("category", "").lower() for n in g.get("nodes", [])}
        try:
            sub = extract_subgraphs(
                g,
                active_only=not include_dormant,
                include_dormant_in_confounding=include_dormant,
            )
        except Exception:
            sub = {}

        counts: Counter = Counter()
        for d in sub.get("direct", []) or []:
            counts[f"direct|{_cat(node_cats, d.get('source'))}"] += 1
        for c in sub.get("chain", []) or []:
            counts[f"chain|{_cat(node_cats, c.get('distal'))}>{_cat(node_cats, c.get('mediator'))}"] += 1
        for c in sub.get("collider", []) or []:
            a, b = c.get("parents", ["", ""])
            pair = sorted([_cat(node_cats, a), _cat(node_cats, b)])
            counts[f"collider|{pair[0]}+{pair[1]}"] += 1
        for c in sub.get("confounding_observed", []) or []:
            counts[f"conf_obs|{_cat(node_cats, c.get('confounder'))}>{_cat(node_cats, c.get('mediator'))}"] += 1
        for c in sub.get("confounding_unobserved", []) or []:
            a, b = c.get("nodes", ["", ""])
            pair = sorted([_cat(node_cats, a), _cat(node_cats, b)])
            counts[f"conf_unobs|{pair[0]}+{pair[1]}"] += 1

        if not counts:
            counts["__no_subgraph__"] += 1
        # Log-bucket counts so small numeric differences don't snowflake graphs.
        feature_counts.append(_bucket_counter(counts))

    return _weighted_jaccard(feature_counts)


# ── Flat agglomerative clustering with elbow threshold ──────────────────


def _elbow_threshold(merge_distances: np.ndarray) -> float:
    """Kneedle-style: max perpendicular distance from the normalized diagonal."""
    n = len(merge_distances)
    if n < 3:
        return float(merge_distances[-1]) if n else 0.5
    x = np.arange(n, dtype=np.float64) / (n - 1)
    y = merge_distances - merge_distances[0]
    if y[-1] > 0:
        y = y / y[-1]
    else:
        return float(merge_distances[-1])
    diffs = y - x
    elbow_idx = int(np.argmax(np.abs(diffs)))
    return float(merge_distances[elbow_idx])


_ELBOW_FLOOR = 0.30  # minimum merge radius — prevents kneedle cutting at 0


def _flat_cluster(K: np.ndarray, linkage_method: str = "complete") -> tuple:
    """Flat agglomerative clustering with elbow-selected threshold.
    Returns (labels, threshold, merge_distances).

    The elbow threshold is floored at _ELBOW_FLOOR so that bags differing
    only by count-bucket or by one rare bin still merge when they're
    structurally the same scene class.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = K.shape[0]
    if n < 2:
        return np.ones(n, dtype=int), 0.0, np.array([0.0])

    dist = 1.0 - K
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    dist = (dist + dist.T) / 2.0

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    merge_distances = np.sort(Z[:, 2])
    elbow = _elbow_threshold(merge_distances)
    thresh = max(elbow, _ELBOW_FLOOR)
    labels = fcluster(Z, t=thresh, criterion="distance")
    return labels, thresh, merge_distances


# ── Representative selection ────────────────────────────────────────────


def _sample_weight(sample, nav_counts: Counter, effect_counts: Counter) -> float:
    """Weight = richness + rarity bonus."""
    richness = (
        sample.n_causal_nodes * 2
        + sample.n_active_edges * 3
        + sample.n_inter_edges * 4
        + sample.n_dormant_edges
    )
    rarity = 0.0
    if sample.nav_command and nav_counts[sample.nav_command]:
        rarity += 1.0 / nav_counts[sample.nav_command]
    for effect, count in sample.active_effect_counts.items():
        if count > 0 and effect_counts[effect]:
            rarity += 1.0 / effect_counts[effect]
    if sample.n_inter_edges > 0:
        rarity += 0.5
    return richness + rarity


# ── Main pipeline ───────────────────────────────────────────────────────


def run_clustering(samples: list, keys: list[str], K_topology: np.ndarray,
                   include_dormant: bool = True) -> dict:
    """Single-level flat clustering on the topology kernel."""
    n = len(samples)

    nav_counts = Counter(s.nav_command for s in samples if s.nav_command)
    effect_counts = Counter(e for s in samples for e, c in s.active_effect_counts.items() if c > 0)
    weights = [_sample_weight(samples[i], nav_counts, effect_counts) for i in range(n)]

    print(f"  Clustering {n} samples on K_topology (complete-link + elbow, floor={_ELBOW_FLOOR})...")
    labels, thresh, merge_distances = _flat_cluster(K_topology, "complete")
    print(f"  Cut threshold: {thresh:.3f}  (merge-dist range: "
          f"{merge_distances[0]:.3f} .. {merge_distances[-1]:.3f})")

    cluster_members: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        cluster_members[int(lab)].append(idx)

    groups_list = []
    for gi, (_k, members) in enumerate(
        sorted(cluster_members.items(), key=lambda kv: (-len(kv[1]), kv[0])), start=1
    ):
        leader = max(members, key=lambda i: weights[i])
        groups_list.append({
            "id": f"G{gi}",
            "leader": keys[leader],
            "size": len(members),
            "members": [
                {"key": keys[m], "weight": round(weights[m], 2), "is_leader": m == leader}
                for m in sorted(members, key=lambda i: -weights[i])
            ],
        })

    # Keep output shape compatible with render_clusters.py
    families = [{
        "label": "flat",
        "size": n,
        "buckets": [{
            "label": "all",
            "size": n,
            "groups": groups_list,
        }],
    }]

    n_multi = sum(1 for g in groups_list if g["size"] > 1)
    n_singletons = sum(1 for g in groups_list if g["size"] == 1)

    result = {
        "metadata": {
            "n_samples": n,
            "n_families": 1,
            "n_groups": len(groups_list),
            "n_multi_groups": n_multi,
            "n_singletons": n_singletons,
            "largest_group": max((g["size"] for g in groups_list), default=0),
            "elbow_threshold": float(thresh),
            "kernel": "K_topology",
            "include_dormant": bool(include_dormant),
        },
        "families": families,
    }

    print(f"  Groups: {len(groups_list)} ({n_multi} multi, {n_singletons} singleton)")
    print(f"  Largest group: {result['metadata']['largest_group']}")

    return result


def export_clusters(result: dict, output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Clusters saved: {output_path}")


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    import argparse
    import glob
    import sys
    from datetime import datetime, timezone

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from filter_diverse_graphs import extract_features

    parser = argparse.ArgumentParser(
        description="Topology-only single-level clustering — groups graphs by subgraph pattern × source-agent histogram."
    )
    parser.add_argument("--graph-dir", required=True,
                        help="Directory with graph JSON files")
    parser.add_argument("--records-dir", required=True,
                        help="Directory with scene records (meta.json)")
    parser.add_argument("--graph-suffix", default="_verified",
                        help="Graph file suffix to read (default: _verified). "
                             "Use _fixed for fix_causal_graphs outputs. "
                             "Leading underscore is optional.")
    parser.add_argument("--output", default=None,
                        help="Output manifest path or directory (default: diverse_graphs.json in graph-dir)")
    parser.add_argument("--clusters-output", default=None,
                        help="Directory (or path) to write clusters.json dump")
    parser.add_argument("--include-dormant", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Include dormant ego-parents in subgraph extraction "
                             "(default: True). Use --no-include-dormant for active-only.")
    args = parser.parse_args()

    suffix = args.graph_suffix if args.graph_suffix.startswith("_") else "_" + args.graph_suffix

    graph_dir = Path(args.graph_dir)
    records_dir = Path(args.records_dir)

    def _resolve(arg_val, default_dir: Path, default_name: str) -> Path:
        if not arg_val:
            return default_dir / default_name
        p = Path(arg_val)
        if p.is_dir() or arg_val.endswith("/") or arg_val.endswith(os.sep):
            return p / default_name
        return p

    output_path = _resolve(args.output, graph_dir, "diverse_graphs.json")
    clusters_path = _resolve(args.clusters_output, graph_dir, "clusters.json") if args.clusters_output else None

    graph_files = sorted(glob.glob(str(graph_dir / f"*{suffix}.json")))
    sample_keys = [os.path.basename(f).replace(f"{suffix}.json", "") for f in graph_files]
    print(f"Found {len(sample_keys)} {suffix} samples in {graph_dir}")
    if not sample_keys:
        print(f"No *{suffix}.json files found. Nothing to filter.")
        return

    samples, graphs, load_failures = [], {}, []
    for key in sample_keys:
        rec = extract_features(key, graph_dir, records_dir, graph_suffix=suffix)
        if rec is None:
            load_failures.append(key)
            continue
        with open(graph_dir / f"{key}{suffix}.json") as f:
            graphs[key] = json.load(f)
        samples.append(rec)

    if load_failures:
        print(f"  Warning: failed to load {len(load_failures)} samples: {load_failures[:5]}...")
    print(f"  Loaded {len(samples)} samples successfully")

    if len(samples) < 2:
        print("Too few samples to filter.")
        return

    keys = [s.key for s in samples]

    mode = "include dormant" if args.include_dormant else "active-only"
    print(f"\nBuilding K_topology (subgraph pattern × source, {mode})...", flush=True)
    k_topo = build_K_topology(samples, graphs, include_dormant=args.include_dormant)

    print("\nRunning single-level topology clustering...")
    result = run_clustering(samples, keys, k_topo, include_dormant=args.include_dormant)

    sample_by_key = {s.key: s for s in samples}
    selected_entries = []
    for fam in result["families"]:
        for bucket in fam["buckets"]:
            for group in bucket["groups"]:
                s = sample_by_key.get(group["leader"])
                if s is None:
                    continue
                selected_entries.append({
                    "key": group["leader"],
                    "scene_num": s.scene_num,
                    "sample_idx": s.sample_idx,
                    "richness": s.richness,
                    "graph_path": s.graph_path,
                    "cluster_size": group["size"],
                    "family": fam["label"],
                    "bucket": bucket["label"],
                    "group_id": group["id"],
                })

    selected_keys = {e["key"] for e in selected_entries}
    rejected_keys = sorted(k for k in keys if k not in selected_keys)

    manifest = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "graph_dir": str(graph_dir),
            "total_input": len(samples),
            "n_selected": len(selected_entries),
            "n_rejected": len(rejected_keys),
            **result["metadata"],
        },
        "selected": sorted(selected_entries, key=lambda e: (e["scene_num"], e["sample_idx"])),
        "rejected": rejected_keys,
    }

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest written to: {output_path}")

    if clusters_path:
        export_clusters(result, str(clusters_path))

    print(f"\nSummary:")
    print(f"  Input:    {len(samples)}")
    print(f"  Selected: {len(selected_entries)} (one leader per cluster)")
    print(f"  Rejected: {len(rejected_keys)}")
    print(f"  Groups:   {result['metadata']['n_groups']} "
          f"({result['metadata']['n_multi_groups']} multi, "
          f"{result['metadata']['n_singletons']} singleton)")


if __name__ == "__main__":
    main()
