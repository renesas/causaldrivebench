#!/usr/bin/env python3
"""t-SNE visualization using the actual clustering kernel distance.

Rebuilds the 8 kernels (same as robust_multimodal_filter.run_clustering),
averages them, converts to distance = 1 - K, runs t-SNE with
metric='precomputed'. This shows the exact space we're clustering in.
"""

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
    build_semantic_kernel, build_interaction_kernel, build_weather_kernel,
    build_distractor_kernel,
)

TREE = Path('outputs/nuscenes_full_final/clusters_tree.json')
D = Path('outputs/nuscenes_full_final')
R = Path('records/nuscenes_full')
OUT_PNG = D / "tsne.png"
OUT_HTML = D / "tsne.html"
VLLM = "http://194.68.245.163:22169"


# ── Load clusters ─────────────────────────────────────────────
print("Loading cluster tree...")
with open(TREE) as f:
    tree = json.load(f)

rows = []
for grp in tree["families"][0]["buckets"][0]["groups"]:
    desc = grp.get("description", {})
    for mem in grp["members"]:
        rows.append({
            "key": mem["key"],
            "group_id": grp["id"],
            "group_size": grp["size"],
            "topology": desc.get("topology", "?"),
            "weather": desc.get("weather", "?"),
            "nav": desc.get("nav", "?"),
            "road": desc.get("road", "?"),
        })

# ── Load features + graphs ────────────────────────────────────
print(f"Loading {len(rows)} samples...")
samples, graphs = [], {}
good_rows = []
for r in rows:
    rec = extract_features(r["key"], D, R)
    if rec is None:
        continue
    samples.append(rec)
    with open(D / f"{r['key']}_verified.json") as f:
        graphs[r["key"]] = json.load(f)
    good_rows.append(r)

print(f"Loaded {len(samples)}")

# ── Rebuild 8 kernels ─────────────────────────────────────────
cfg = ClusteringConfig(embedding_endpoint=VLLM)
print("Building kernels (this takes ~1 min)...")
print("  K1 topology"); K1 = build_topology_kernel(samples, graphs)
print("  K2 ses");      K2 = build_status_effect_source_kernel(samples, graphs)
print("  K3 road");     K3 = build_road_kernel(samples, graphs, cfg)
print("  K4 nav");      K4 = build_nav_kernel(samples, graphs, cfg)
print("  K5 beh");      K5 = build_behavior_kernel(samples, graphs, cfg)
print("  K6 sem");      K6 = build_semantic_kernel(samples, graphs)
print("  K7 inter");    K7 = build_interaction_kernel(samples, graphs)
print("  K8 weather");  K8 = build_weather_kernel(samples, graphs, cfg)
print("  K9 distractor"); K9 = build_distractor_kernel(samples, graphs)

K = (K1 + K2 + K3 + K4 + K5 + K6 + K7 + K8 + K9) / 9.0
dist = 1.0 - K
np.fill_diagonal(dist, 0.0)
dist = np.clip(dist, 0.0, None)
dist = (dist + dist.T) / 2.0  # enforce symmetry

# ── t-SNE on precomputed distance ─────────────────────────────
print(f"Running t-SNE on {dist.shape[0]}×{dist.shape[0]} kernel distance...")
emb = TSNE(n_components=2, perplexity=30, metric="precomputed", init="random",
           learning_rate="auto", random_state=42).fit_transform(dist)

# ── Plot ───────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

topo_colors = {
    "empty": "#888888", "direct": "#3b82f6", "chain": "#f59e0b",
    "collider": "#ef4444", "confounding": "#a855f7",
    "other": "#22c55e", "?": "#ccc",
}
weather_colors = {
    "clear-daytime": "#fbbf24", "clear": "#fbbf24",
    "night": "#1e3a8a", "night+wet": "#0f172a",
    "wet": "#0ea5e9", "fog": "#94a3b8",
    "fog+wet": "#64748b", "night+fog": "#1e293b",
    "other": "#999", "?": "#ccc",
}

nav_colors = {
    "GO STRAIGHT": "#10b981", "TURN LEFT": "#3b82f6", "TURN RIGHT": "#f59e0b", "?": "#999",
}
road_colors = {
    "intersection": "#3b82f6", "roundabout": "#a855f7",
    "parking": "#f59e0b", "highway": "#ef4444",
    "multi-lane": "#10b981", "road": "#888", "?": "#ccc",
}

fig, axes = plt.subplots(2, 2, figsize=(22, 18), dpi=120)
axes = axes.flatten()

# Panel 0: topology
for topo, color in topo_colors.items():
    idx = [i for i, r in enumerate(good_rows) if r["topology"] == topo]
    if idx:
        axes[0].scatter(emb[idx, 0], emb[idx, 1], c=color, s=8, alpha=0.6, label=f"{topo} ({len(idx)})")
axes[0].legend(fontsize=9, loc="best")
axes[0].set_title(f"t-SNE on averaged 9-kernel distance — topology (n={len(good_rows)})")

# Panel 1: weather
for w, color in weather_colors.items():
    idx = [i for i, r in enumerate(good_rows) if r["weather"] == w]
    if idx:
        axes[1].scatter(emb[idx, 0], emb[idx, 1], c=color, s=8, alpha=0.6, label=f"{w} ({len(idx)})")
axes[1].legend(fontsize=9, loc="best")
axes[1].set_title("weather")

# Panel 2: nav intent
for n, color in nav_colors.items():
    idx = [i for i, r in enumerate(good_rows) if r["nav"] == n]
    if idx:
        axes[2].scatter(emb[idx, 0], emb[idx, 1], c=color, s=8, alpha=0.6, label=f"{n} ({len(idx)})")
axes[2].legend(fontsize=9, loc="best")
axes[2].set_title("navigation intent")

# Panel 3: road geometry
for rd, color in road_colors.items():
    idx = [i for i, r in enumerate(good_rows) if r["road"] == rd]
    if idx:
        axes[3].scatter(emb[idx, 0], emb[idx, 1], c=color, s=8, alpha=0.6, label=f"{rd} ({len(idx)})")
axes[3].legend(fontsize=9, loc="best")
axes[3].set_title("road geometry")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")

# ── Interactive plotly ─────────────────────────────────────────
try:
    import plotly.graph_objs as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Topology", "Weather", "Nav Intent", "Road Geometry"))

    hover_texts = [f"{good_rows[i]['key']}<br>group={good_rows[i]['group_id']} (size {good_rows[i]['group_size']})<br>topo: {good_rows[i]['topology']}<br>weather: {good_rows[i]['weather']}<br>nav: {good_rows[i]['nav']}<br>road: {good_rows[i]['road']}" for i in range(len(good_rows))]

    for (r, c), (color_map, field) in zip([(1,1),(1,2),(2,1),(2,2)],
                                           [(topo_colors,"topology"), (weather_colors,"weather"),
                                            (nav_colors,"nav"), (road_colors,"road")]):
        for cat, color in color_map.items():
            idx = [i for i, row in enumerate(good_rows) if row[field] == cat]
            if not idx: continue
            fig.add_trace(go.Scattergl(
                x=emb[idx, 0], y=emb[idx, 1], mode="markers",
                marker=dict(size=4, color=color, opacity=0.6),
                name=f"{cat} ({len(idx)})",
                text=[hover_texts[i] for i in idx], hoverinfo="text",
                legendgroup=field, showlegend=True,
            ), row=r, col=c)

    fig.update_layout(
        title=f"t-SNE on averaged 9-kernel distance (n={len(good_rows)})",
        template="plotly_dark", height=1100, hovermode="closest",
    )
    pio.write_html(fig, str(OUT_HTML))
    print(f"Saved: {OUT_HTML}")
except Exception as e:
    print(f"Plotly failed: {e}")
