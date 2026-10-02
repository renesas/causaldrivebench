#!/usr/bin/env python3
"""t-SNE per kernel — one panel per kernel + combined, colored by topology."""

import json
import sys
from pathlib import Path

sys.path.insert(0, 'scripts/graph_filtering')
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'scripts/causal_graph_extraction')

import numpy as np
from sklearn.manifold import TSNE
from filter_diverse_graphs import extract_features
from robust_multimodal_filter import (
    ClusteringConfig,
    build_topology_kernel, build_status_effect_source_kernel,
    build_road_kernel, build_nav_kernel, build_behavior_kernel,
    build_active_sem_kernel, build_dormant_sem_kernel,
    build_interaction_kernel, build_weather_kernel,
    build_distractor_kernel,
)

TREE = Path('/spiral_hdd_3/workspace/navvrat/Neurips/outputs/nuscenes_full_final/clusters_tree.json')
D = Path('/spiral_hdd_3/workspace/navvrat/Neurips/outputs/nuscenes_full_final')
R = Path('/spiral_hdd_3/workspace/navvrat/Neurips/records/nuscenes_full')
OUT_PNG = D / "tsne_per_kernel.png"
VLLM = "http://194.68.245.163:22169"

print("Loading clusters...")
with open(TREE) as f:
    tree = json.load(f)

rows = []
for grp in tree["families"][0]["buckets"][0]["groups"]:
    desc = grp.get("description", {})
    for mem in grp["members"]:
        rows.append({
            "key": mem["key"],
            "topology": desc.get("topology", "?"),
        })

print(f"Loading {len(rows)} samples...")
samples, graphs = [], {}
good_rows = []
for r in rows:
    rec = extract_features(r["key"], D, R)
    if rec is None: continue
    samples.append(rec)
    with open(D / f"{r['key']}_verified.json") as f:
        graphs[r["key"]] = json.load(f)
    good_rows.append(r)

cfg = ClusteringConfig(embedding_endpoint=VLLM)
print("Building kernels...")
kernels = {
    "K1 topology": build_topology_kernel(samples, graphs),
    "K2 ses (status×effect×source)": build_status_effect_source_kernel(samples, graphs),
    "K3 road": build_road_kernel(samples, graphs, cfg),
    "K4 nav": build_nav_kernel(samples, graphs, cfg),
    "K5 behavior": build_behavior_kernel(samples, graphs, cfg),
    "K6a active-sem": build_active_sem_kernel(samples, graphs),
    "K6b dormant-sem": build_dormant_sem_kernel(samples, graphs),
    "K7 inter": build_interaction_kernel(samples, graphs),
    "K8 weather": build_weather_kernel(samples, graphs, cfg),
    "K9 distractor": build_distractor_kernel(samples, graphs),
}
kernels["K_avg (all 10)"] = sum(kernels.values()) / 10.0

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

topo_colors = {
    "empty": "#888888", "direct": "#3b82f6", "chain": "#f59e0b",
    "collider": "#ef4444", "confounding": "#a855f7",
    "other": "#22c55e", "?": "#ccc",
}

fig, axes = plt.subplots(3, 4, figsize=(28, 19), dpi=100)
axes = axes.flatten()
# Hide the last (12th) empty panel
axes[11].axis("off")

for ax, (name, K) in zip(axes, kernels.items()):
    print(f"  t-SNE: {name}")
    dist = 1.0 - K
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    dist = (dist + dist.T) / 2.0
    emb = TSNE(n_components=2, perplexity=30, metric="precomputed",
               init="random", learning_rate="auto", random_state=42).fit_transform(dist)
    for topo, color in topo_colors.items():
        idx = [i for i, r in enumerate(good_rows) if r["topology"] == topo]
        if idx:
            ax.scatter(emb[idx, 0], emb[idx, 1], c=color, s=5, alpha=0.55,
                       label=f"{topo} ({len(idx)})")
    ax.set_title(name, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc="best", framealpha=0.7)
plt.suptitle(f"t-SNE per kernel, colored by topology (n={len(good_rows)})", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=100, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
