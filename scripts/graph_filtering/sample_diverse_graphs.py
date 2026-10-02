#!/usr/bin/env python3
"""Sample diverse scenarios from clustering results.

Strategy:
  1. Every group contributes at least 1 sample (the leader) — guarantees
     cross-group diversity since groups are already internally similar.
  2. For large groups, pick additional samples using max-min distance
     on the combined kernel — guarantees within-group diversity.
  3. Budget controls total output size. If budget < n_groups, prioritize
     groups by rarity (smaller families first, then smaller groups).

Usage:
    # Default: 1 per group (= 1285 samples from 3009)
    python scripts/graph_filtering/sample_diverse.py

    # Budget of 1500: 1 per group + extras from large groups
    python scripts/graph_filtering/sample_diverse.py --budget 1500

    # Budget of 800: prioritize rarest groups
    python scripts/graph_filtering/sample_diverse.py --budget 800
"""

import argparse
import json
import glob
import numpy as np
from collections import Counter


def load_kernels(kernel_path: str) -> np.ndarray:
    """Load cached kernels and return combined kernel matrix."""
    data = np.load(kernel_path)
    keys = [k for k in data.files]
    print(f"  Loaded {len(keys)} kernels from {kernel_path}")
    K = sum(data[k] for k in keys) / len(keys)
    return K


def maxmin_select(indices: list, K: np.ndarray, n_pick: int, leader: int) -> list:
    """Pick n_pick diverse samples using max-min distance on kernel.

    Starts with the leader, greedily picks the sample most distant
    from all already-selected (facility location heuristic).
    """
    if n_pick >= len(indices):
        return list(indices)

    selected = [leader]
    remaining = set(indices) - {leader}

    # Track min similarity to selected set for each remaining
    min_sim = {r: K[r, leader] for r in remaining}

    for _ in range(n_pick - 1):
        if not remaining:
            break
        best_idx = min(remaining, key=lambda r: min_sim[r])
        selected.append(best_idx)
        remaining.discard(best_idx)
        for r in remaining:
            min_sim[r] = min(min_sim[r], K[r, best_idx])

    return selected


def sample_diverse(clusters_path: str, kernel_path: str = None, budget: int = None) -> dict:
    with open(clusters_path) as f:
        data = json.load(f)

    families = data["families"]
    meta = data["metadata"]
    N = meta["n_samples"]

    # Build key→index map and group inventory
    key_to_idx = {}
    all_keys = []
    groups = []  # (group_dict, family_label, bucket_label, family_size)

    idx = 0
    for fam in families:
        for bkt in fam["buckets"]:
            for grp in bkt["groups"]:
                for mem in grp["members"]:
                    if mem["key"] not in key_to_idx:
                        key_to_idx[mem["key"]] = idx
                        all_keys.append(mem["key"])
                        idx += 1
                groups.append((grp, fam["label"], bkt["label"], fam["size"]))

    n_groups = len(groups)

    if budget is None:
        budget = n_groups

    print(f"Samples: {N}, Groups: {n_groups}, Budget: {budget}")

    # Load kernels if we need within-group diversity
    K = None
    if kernel_path and budget > n_groups:
        print("Loading kernels for within-group diversity...")
        K = load_kernels(kernel_path)

    # ── Rank groups by rarity ──
    # Smaller family = rarer scenario type. Tiebreak: smaller group = less redundancy
    ranked_groups = sorted(
        groups,
        key=lambda g: (g[3], g[0]["size"]),
    )

    # ── Allocate budget ──
    if budget >= n_groups:
        # Phase 1: 1 leader per group
        selected = {}
        for grp, fam_label, bkt_label, fam_size in ranked_groups:
            leader = grp["leader"]
            selected[leader] = {
                "group_id": grp["id"],
                "family": fam_label,
                "bucket": bkt_label,
                "reason": "leader",
            }

        remaining_budget = budget - len(selected)

        if remaining_budget > 0:
            # Phase 2: extras from large groups via max-min diversity
            large_groups = [
                (grp, fam_label, bkt_label)
                for grp, fam_label, bkt_label, _ in ranked_groups
                if grp["size"] > 1
            ]
            large_groups.sort(key=lambda g: -g[0]["size"])

            total_extra_pool = sum(g[0]["size"] - 1 for g in large_groups)
            if total_extra_pool > 0:
                for grp, fam_label, bkt_label in large_groups:
                    if remaining_budget <= 0:
                        break
                    pool = grp["size"] - 1
                    n_extra = max(1, round(pool / total_extra_pool * (budget - n_groups)))
                    n_extra = min(n_extra, pool, remaining_budget)

                    if n_extra > 0 and K is not None:
                        indices = [key_to_idx[m["key"]] for m in grp["members"]]
                        leader_idx = key_to_idx[grp["leader"]]
                        diverse = maxmin_select(indices, K, 1 + n_extra, leader_idx)
                        for di in diverse:
                            key = all_keys[di]
                            if key not in selected:
                                selected[key] = {
                                    "group_id": grp["id"],
                                    "family": fam_label,
                                    "bucket": bkt_label,
                                    "reason": "within_group_diverse",
                                }
                                remaining_budget -= 1
                    elif n_extra > 0:
                        for mem in grp["members"]:
                            if remaining_budget <= 0 or n_extra <= 0:
                                break
                            if mem["key"] not in selected:
                                selected[mem["key"]] = {
                                    "group_id": grp["id"],
                                    "family": fam_label,
                                    "bucket": bkt_label,
                                    "reason": "within_group_weight",
                                }
                                remaining_budget -= 1
                                n_extra -= 1
    else:
        # Budget < n_groups: prioritize rarest groups
        selected = {}
        for grp, fam_label, bkt_label, fam_size in ranked_groups:
            if len(selected) >= budget:
                break
            leader = grp["leader"]
            selected[leader] = {
                "group_id": grp["id"],
                "family": fam_label,
                "bucket": bkt_label,
                "reason": "leader_prioritized",
            }

    # ── Coverage stats ──
    sel_families = set(info["family"] for info in selected.values())
    sel_groups = set(info["group_id"] for info in selected.values())
    all_families = set(fam["label"] for fam in families)
    reasons = Counter(info["reason"] for info in selected.values())

    family_stats = []
    for fam in sorted(families, key=lambda f: -f["size"]):
        fam_selected = [k for k, v in selected.items() if v["family"] == fam["label"]]
        family_stats.append({
            "family": fam["label"],
            "total": fam["size"],
            "selected": len(fam_selected),
        })

    result = {
        "metadata": {
            "total_samples": N,
            "total_groups": n_groups,
            "budget": budget,
            "selected": len(selected),
            "families_covered": f"{len(sel_families)}/{len(all_families)}",
            "groups_covered": f"{len(sel_groups)}/{n_groups}",
            "selection_breakdown": dict(reasons),
        },
        "selected": [
            {"key": k, **selected[k]}
            for k in sorted(selected.keys())
        ],
        "family_coverage": family_stats,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Sample diverse scenarios from clusters")
    parser.add_argument("--clusters", default="tmp/clusters_full.json")
    parser.add_argument("--kernels", default=None, help="Cached kernels .npz")
    parser.add_argument("--budget", type=int, default=None, help="Target samples (default: 1 per group)")
    parser.add_argument("--output", default="tmp/diverse_sample.json")
    args = parser.parse_args()

    if args.kernels is None:
        cache_files = sorted(glob.glob("tmp/.cache/kernels_*.npz"))
        if cache_files:
            args.kernels = cache_files[-1]
            print(f"Auto-found kernels: {args.kernels}")

    result = sample_diverse(args.clusters, args.kernels, args.budget)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    m = result["metadata"]
    print(f"\nResults:")
    print(f"  Selected: {m['selected']} / {m['total_samples']}")
    print(f"  Families covered: {m['families_covered']}")
    print(f"  Groups covered: {m['groups_covered']}")
    print(f"  Breakdown: {m['selection_breakdown']}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()