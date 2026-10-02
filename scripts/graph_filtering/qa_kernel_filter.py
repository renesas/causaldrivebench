#!/usr/bin/env python3
"""QA-kernel hierarchical clustering for diverse subset selection.

Two-level clustering, both levels kernel-based with agglomerative complete-link
+ elbow-selected distance threshold.

  Level 1 — QA + topology kernels (joint structure & QA difficulty):
      K1_combined = 0.25*K_active + 0.25*K_dormant + 0.35*K_distractor + 0.25*K_topology

  Level 2 — scene/context kernels, applied within each Level-1 cluster:
      K2_combined = mean(K_road, K_weather, K_nav, K_behavior)

One representative (max-weight leader) is picked per Level-2 leaf cluster.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import requests


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class ClusteringConfig:
    embedding_endpoint: str = ""  # vLLM endpoint, e.g. "http://194.68.245.163:22169"
    embedding_model: str = "jinaai/jina-embeddings-v5-text-small-text-matching"
    embedding_dim: int = 256
    # Level-1 kernel weights (as specified)
    w_active: float = 0.25
    w_dormant: float = 0.25
    w_distractor: float = 0.25
    w_topology: float = 0.25


# ── Embedding ────────────────────────────────────────────────────────────


def _strip_agent_ids(text: str) -> str:
    text = re.sub(
        r"\b(car|truck|bus|pedestrian|cyclist|motorcycle|trailer|vehicle|construction|bicycle|suv)_\d+\b",
        r"\1", text,
    )
    text = re.sub(r"\bn(\d+)\b", "infrastructure", text)
    return text


def embed_texts(texts: list[str], cfg: ClusteringConfig) -> np.ndarray:
    if not texts:
        return np.zeros((0, cfg.embedding_dim))
    texts = [t if t.strip() else "empty" for t in texts]

    if cfg.embedding_endpoint:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        endpoint = cfg.embedding_endpoint
        if not endpoint.startswith(("http://", "https://")):
            endpoint = "http://" + endpoint
        endpoint = endpoint.rstrip("/")

        bs = 256
        all_embs: list = [None] * len(texts)

        def _embed_batch(start: int):
            batch = texts[start:start + bs]
            resp = requests.post(
                f"{endpoint}/v1/embeddings",
                json={"model": cfg.embedding_model, "input": batch, "dimensions": cfg.embedding_dim},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            embs = [d["embedding"][:cfg.embedding_dim] for d in sorted(data, key=lambda x: x["index"])]
            return start, embs

        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(_embed_batch, i) for i in range(0, len(texts), bs)]
            for fut in as_completed(futs):
                start, embs = fut.result()
                for j, emb in enumerate(embs):
                    all_embs[start + j] = emb

        arr = np.array(all_embs, dtype=np.float64)
        nrm = np.linalg.norm(arr, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return arr / nrm
    else:
        from sentence_transformers import SentenceTransformer
        device = os.environ.get("EMBEDDING_DEVICE", "cpu")
        model = SentenceTransformer(
            cfg.embedding_model, trust_remote_code=True, device=device,
            model_kwargs={"default_task": "text-matching"},
        )
        return model.encode(texts, show_progress_bar=True, batch_size=64,
                            normalize_embeddings=True, truncate_dim=cfg.embedding_dim)


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


# ── Level-1 kernels: QA + topology ───────────────────────────────────────


def build_K_active(samples, graphs):
    """K_active: joint (effect, source_cat) for active ego_edges +
    joint (relation, src_cat, tgt_cat, src_ego_status, tgt_ego_status) for
    inter_node_edges. Weighted Jaccard on the combined histogram."""
    feature_counts = []
    for s in samples:
        g = graphs.get(s.key, {})
        node_cats = {n["id"]: n.get("category", "").lower() for n in g.get("nodes", [])}
        ego_status = {}
        counts: Counter = Counter()

        for e in g.get("ego_edges", []):
            src = e.get("source", "")
            is_active = e.get("active", False) or str(e.get("causal_status", "")).startswith("active")
            status = "A" if is_active else "D"
            ego_status[src] = status
            if is_active:
                cat = _categorize_agent(node_cats.get(src, ""))
                counts[f"ego|{e.get('effect_type','')}|{cat}"] += 1

        for e in g.get("inter_node_edges", []) or []:
            src, tgt = e.get("source", ""), e.get("target", "")
            rel = e.get("relation", "UNKNOWN")
            sc = _categorize_agent(node_cats.get(src, ""))
            tc = _categorize_agent(node_cats.get(tgt, ""))
            ss = ego_status.get(src, "X")
            ts = ego_status.get(tgt, "X")
            counts[f"inter|{rel}|{sc}({ss})>{tc}({ts})"] += 1

        if not counts:
            counts["__empty__"] += 1
        feature_counts.append(counts)
    return _weighted_jaccard(feature_counts)


def build_K_dormant(samples, graphs):
    """K_dormant: one bin per dormant ego_edge (effect, source_cat, has_inter_link).
    has_inter_link flags whether the dormant node connects to any active node."""
    feature_counts = []
    for s in samples:
        g = graphs.get(s.key, {})
        node_cats = {n["id"]: n.get("category", "").lower() for n in g.get("nodes", [])}

        active_nodes, dormant_nodes = set(), set()
        for e in g.get("ego_edges", []):
            src = e.get("source", "")
            is_active = e.get("active", False) or str(e.get("causal_status", "")).startswith("active")
            (active_nodes if is_active else dormant_nodes).add(src)

        inter_linked = set()
        for e in g.get("inter_node_edges", []) or []:
            src, tgt = e.get("source", ""), e.get("target", "")
            if src in dormant_nodes and tgt in active_nodes:
                inter_linked.add(src)
            if tgt in dormant_nodes and src in active_nodes:
                inter_linked.add(tgt)

        counts: Counter = Counter()
        for e in g.get("ego_edges", []):
            is_active = e.get("active", False) or str(e.get("causal_status", "")).startswith("active")
            if is_active:
                continue
            src = e.get("source", "")
            cat = _categorize_agent(node_cats.get(src, ""))
            linked = "linked" if src in inter_linked else "isolated"
            counts[f"{e.get('effect_type','')}|{cat}|{linked}"] += 1

        if not counts:
            counts["__no_dormant__"] += 1
        feature_counts.append(counts)
    return _weighted_jaccard(feature_counts)


def build_K_distractor(samples, graphs):
    """K_distractor: one bin per distractor coarse_category."""
    feature_counts = []
    for s in samples:
        g = graphs.get(s.key, {})
        counts: Counter = Counter()
        for d in g.get("distractors", []) or []:
            counts[_categorize_agent(d.get("category", ""))] += 1
        if not counts:
            counts["__none__"] += 1
        feature_counts.append(counts)
    return _weighted_jaccard(feature_counts)


def build_K_topology(samples, graphs):
    """K_topology: joint histogram of (pattern_type × source_agent_type) over
    extracted subgraphs (direct / chain / collider / confounding_obs /
    confounding_unobs). Order-preserving for directed patterns.
    """
    import sys
    _extract_dir = str(Path(__file__).resolve().parent.parent / "causal_graph_extraction")
    if _extract_dir not in sys.path:
        sys.path.insert(0, _extract_dir)
    from extract_subgraphs import extract_subgraphs

    def _cat(node_cats: dict, nid: str) -> str:
        return _categorize_agent(node_cats.get(nid, ""))

    feature_counts: list[Counter] = []
    for s in samples:
        g = graphs.get(s.key, {})
        node_cats = {n["id"]: n.get("category", "").lower() for n in g.get("nodes", [])}
        try:
            sub = extract_subgraphs(g, active_only=True, include_dormant_in_confounding=False)
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
        feature_counts.append(counts)

    return _weighted_jaccard(feature_counts)


# ── Level-2 kernels: scene/context embeddings ────────────────────────────


def build_K_road(samples, graphs, cfg: ClusteringConfig) -> np.ndarray:
    """Road geometry — embedding of scene_context.road_geometry."""
    texts = [f"Road geometry: {graphs.get(s.key, {}).get('scene_context', {}).get('road_geometry', '')}"
             for s in samples]
    emb = embed_texts(texts, cfg)
    return np.clip(emb @ emb.T, 0, 1)


def build_K_nav(samples, graphs, cfg: ClusteringConfig) -> np.ndarray:
    """Navigation intent — embedding of scene_context.navigation_intent."""
    texts = [f"Navigation intent: {graphs.get(s.key, {}).get('scene_context', {}).get('navigation_intent', '')}"
             for s in samples]
    emb = embed_texts(texts, cfg)
    return np.clip(emb @ emb.T, 0, 1)


def build_K_weather(samples, graphs, cfg: ClusteringConfig) -> np.ndarray:
    """Environment — embedding of scene_context.environment_conditions."""
    texts = []
    for s in samples:
        ctx = graphs.get(s.key, {}).get("scene_context", {})
        env = (ctx.get("environment_conditions") or "").strip() or "Clear daytime"
        texts.append(f"Environment: {env}")
    emb = embed_texts(texts, cfg)
    return np.clip(emb @ emb.T, 0, 1)


def build_K_behavior(samples, graphs, cfg: ClusteringConfig) -> np.ndarray:
    """Ego behavior — embedding of behavior_explanation + nav_command +
    inter_node_edge justifications."""
    texts = []
    for s in samples:
        g = graphs.get(s.key, {})
        ego = g.get("ego_state", {}) or {}
        beh = str(ego.get("behavior_explanation", "")).strip()
        inter_bits = []
        for e in g.get("inter_node_edges", []) or []:
            just = _strip_agent_ids(str(e.get("justification", "")).strip())
            if just:
                inter_bits.append(f"{e.get('relation','')}: {just}")
        t = f"Navigation: {s.nav_command}. Behavior: {beh}."
        if inter_bits:
            t += " Inter-node: " + " | ".join(inter_bits[:5])
        texts.append(t)
    emb = embed_texts(texts, cfg)
    return np.clip(emb @ emb.T, 0, 1)


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


def _flat_cluster(K: np.ndarray, linkage_method: str = "complete") -> tuple:
    """Flat agglomerative clustering with elbow-selected threshold.
    Returns (labels, threshold, merge_distances)."""
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
    thresh = _elbow_threshold(merge_distances)
    labels = fcluster(Z, t=thresh, criterion="distance")
    return labels, thresh, merge_distances


# ── Representative selection ─────────────────────────────────────────────


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


# ── Main pipeline ────────────────────────────────────────────────────────


def run_clustering(
    samples: list,
    keys: list[str],
    K_active: np.ndarray,
    K_dormant: np.ndarray,
    K_distractor: np.ndarray,
    K_topology: np.ndarray,
    K_road: np.ndarray,
    K_weather: np.ndarray,
    K_nav: np.ndarray,
    K_behavior: np.ndarray,
    cfg: ClusteringConfig,
) -> dict:
    """Two-level hierarchical clustering.

    Level 1:  K1 = w_a*K_active + w_d*K_dormant + w_dist*K_distractor + w_topo*K_topology
    Level 2:  K2 = mean(K_road, K_weather, K_nav, K_behavior) restricted to L1 members
    """
    n = len(samples)

    K1 = (cfg.w_active * K_active
          + cfg.w_dormant * K_dormant
          + cfg.w_distractor * K_distractor
          + cfg.w_topology * K_topology)
    K2_full = (K_road + K_weather + K_nav + K_behavior) / 4.0

    nav_counts = Counter(s.nav_command for s in samples if s.nav_command)
    effect_counts = Counter(e for s in samples for e, c in s.active_effect_counts.items() if c > 0)
    weights = [_sample_weight(samples[i], nav_counts, effect_counts) for i in range(n)]

    print(f"  Level 1 (QA + topology): clustering {n} samples (complete-link + elbow)...")
    l1_labels, l1_thresh, l1_merges = _flat_cluster(K1, "complete")
    l1_members: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(l1_labels):
        l1_members[int(lab)].append(idx)
    print(f"  Level 1 elbow threshold: {l1_thresh:.3f}  (merge-dist: "
          f"{l1_merges[0]:.3f} .. {l1_merges[-1]:.3f})  →  {len(l1_members)} L1 clusters")

    families = []
    total_l2_groups = 0
    for fam_rank, (l1_key, idxs) in enumerate(
        sorted(l1_members.items(), key=lambda kv: (-len(kv[1]), kv[0])), start=1
    ):
        fam_label = f"F{fam_rank}"
        fam_size = len(idxs)

        if fam_size == 1:
            only = idxs[0]
            groups_list = [{
                "id": f"{fam_label}.G1",
                "leader": keys[only],
                "size": 1,
                "members": [{"key": keys[only], "weight": round(weights[only], 2), "is_leader": True}],
            }]
            families.append({"label": fam_label, "size": 1,
                             "buckets": [{"label": "all", "size": 1, "groups": groups_list}]})
            total_l2_groups += 1
            continue

        # Restrict Level-2 kernel to this family
        sub = np.ix_(idxs, idxs)
        K2_sub = K2_full[sub]
        l2_labels, l2_thresh, _ = _flat_cluster(K2_sub, "complete")

        l2_members: dict[int, list[int]] = defaultdict(list)
        for local_i, lab in enumerate(l2_labels):
            l2_members[int(lab)].append(idxs[local_i])

        groups_list = []
        for gi, (_k, members) in enumerate(
            sorted(l2_members.items(), key=lambda kv: (-len(kv[1]), kv[0])), start=1
        ):
            leader = max(members, key=lambda i: weights[i])
            groups_list.append({
                "id": f"{fam_label}.G{gi}",
                "leader": keys[leader],
                "size": len(members),
                "members": [
                    {"key": keys[m], "weight": round(weights[m], 2), "is_leader": m == leader}
                    for m in sorted(members, key=lambda i: -weights[i])
                ],
            })
        total_l2_groups += len(groups_list)

        families.append({
            "label": fam_label,
            "size": fam_size,
            "l2_threshold": float(l2_thresh),
            "buckets": [{
                "label": "scene_context",
                "size": fam_size,
                "groups": groups_list,
            }],
        })

    n_multi = sum(1 for f in families for b in f["buckets"] for g in b["groups"] if g["size"] > 1)
    n_singletons = sum(1 for f in families for b in f["buckets"] for g in b["groups"] if g["size"] == 1)
    largest = max((g["size"] for f in families for b in f["buckets"] for g in b["groups"]), default=0)

    print(f"  Level 2 total groups: {total_l2_groups} ({n_multi} multi, {n_singletons} singleton)")
    print(f"  Largest group: {largest}")

    return {
        "metadata": {
            "n_samples": n,
            "n_families": len(families),
            "n_groups": total_l2_groups,
            "n_multi_groups": n_multi,
            "n_singletons": n_singletons,
            "largest_group": largest,
            "l1_threshold": float(l1_thresh),
            "l1_weights": {
                "K_active": cfg.w_active,
                "K_dormant": cfg.w_dormant,
                "K_distractor": cfg.w_distractor,
                "K_topology": cfg.w_topology,
            },
        },
        "families": families,
    }


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
        description="Two-level QA-kernel clustering: L1 picks groups by (active, dormant, distractor, topology); L2 refines by (road, weather, nav, behavior)."
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
                        help="Directory (or path) to write clusters.json dump of the full hierarchy")
    parser.add_argument("--embedding-endpoint", default="",
                        help="vLLM embedding endpoint URL (e.g. http://host:port). Empty = local SentenceTransformer.")
    parser.add_argument("--sampling-mode", choices=["leader", "proportional"], default="leader",
                        help="leader: one max-weight leader per L2 group (default). "
                             "proportional: take ceil(group_size * sampling_rate) top-weight members per group, with min 1.")
    parser.add_argument("--sampling-rate", type=float, default=0.3,
                        help="Fraction of each group to keep in proportional mode (min 1 per group). Ignored in leader mode.")
    parser.add_argument("--from-clusters", default=None,
                        help="Path to an existing clusters.json. Skips embedding/clustering and only re-runs sample selection.")
    args = parser.parse_args()

    if args.sampling_mode == "proportional" and not (0.0 < args.sampling_rate <= 1.0):
        parser.error("--sampling-rate must be in (0, 1]")

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

    if args.from_clusters:
        print(f"\nLoading existing clusters from: {args.from_clusters}")
        with open(args.from_clusters) as f:
            result = json.load(f)
        member_keys = sorted({m["key"]
                              for fam in result.get("families", [])
                              for bucket in fam.get("buckets", [])
                              for group in bucket.get("groups", [])
                              for m in group.get("members", [])})
        print(f"  {len(member_keys)} unique member keys across "
              f"{result.get('metadata', {}).get('n_groups', '?')} groups")

        samples, load_failures = [], []
        for key in member_keys:
            rec = extract_features(key, graph_dir, records_dir, graph_suffix=suffix)
            if rec is None:
                load_failures.append(key)
                continue
            samples.append(rec)
        if load_failures:
            print(f"  Warning: failed to load {len(load_failures)} samples: {load_failures[:5]}...")
        print(f"  Loaded {len(samples)} samples for selection")
        keys = [s.key for s in samples]
    else:
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
        cfg = ClusteringConfig()
        if args.embedding_endpoint:
            cfg.embedding_endpoint = args.embedding_endpoint

        print("\nBuilding Level-1 kernels (QA + topology)...")
        print("  K_active (active ego + inter-node joint hist)...", flush=True)
        k_active = build_K_active(samples, graphs)
        print("  K_dormant (dormant ego + inter-link flag)...", flush=True)
        k_dormant = build_K_dormant(samples, graphs)
        print("  K_distractor (category × rejection_tree)...", flush=True)
        k_distractor = build_K_distractor(samples, graphs)
        print("  K_topology (subgraph pattern × source)...", flush=True)
        k_topology = build_K_topology(samples, graphs)

        print("\nBuilding Level-2 kernels (scene/context embeddings)...")
        print("  K_road (road_geometry)...", flush=True)
        k_road = build_K_road(samples, graphs, cfg)
        print("  K_weather (environment_conditions)...", flush=True)
        k_weather = build_K_weather(samples, graphs, cfg)
        print("  K_nav (navigation_intent)...", flush=True)
        k_nav = build_K_nav(samples, graphs, cfg)
        print("  K_behavior (ego behavior + inter-node justifications)...", flush=True)
        k_behavior = build_K_behavior(samples, graphs, cfg)

        print("\nRunning 2-level clustering...")
        result = run_clustering(
            samples, keys,
            k_active, k_dormant, k_distractor, k_topology,
            k_road, k_weather, k_nav, k_behavior,
            cfg,
        )

    import math

    sample_by_key = {s.key: s for s in samples}
    selected_entries = []
    for fam in result["families"]:
        for bucket in fam["buckets"]:
            for group in bucket["groups"]:
                if args.sampling_mode == "leader":
                    chosen = [m for m in group["members"] if m.get("is_leader")] or group["members"][:1]
                else:
                    k = max(1, math.ceil(group["size"] * args.sampling_rate))
                    pool = sorted(group["members"], key=lambda m: -m["weight"])
                    chosen, chosen_scenes = [], set()
                    while len(chosen) < k and pool:
                        idx = next(
                            (i for i, m in enumerate(pool)
                             if (sample_by_key.get(m["key"]) is not None
                                 and sample_by_key[m["key"]].scene_num not in chosen_scenes)),
                            0,  # fallback: highest-weight remaining
                        )
                        m = pool.pop(idx)
                        s = sample_by_key.get(m["key"])
                        if s is not None:
                            chosen_scenes.add(s.scene_num)
                        chosen.append(m)

                for m in chosen:
                    s = sample_by_key.get(m["key"])
                    if s is None:
                        continue
                    selected_entries.append({
                        "key": m["key"],
                        "scene_num": s.scene_num,
                        "sample_idx": s.sample_idx,
                        "richness": s.richness,
                        "graph_path": s.graph_path,
                        "cluster_size": group["size"],
                        "is_leader": bool(m.get("is_leader")),
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
            "sampling_mode": args.sampling_mode,
            "sampling_rate": args.sampling_rate if args.sampling_mode == "proportional" else None,
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

    sel_desc = ("one leader per L2 cluster" if args.sampling_mode == "leader"
                else f"proportional rate={args.sampling_rate} (min 1 per group)")
    print(f"\nSummary:")
    print(f"  Input:    {len(samples)}")
    print(f"  Selected: {len(selected_entries)} ({sel_desc})")
    print(f"  Rejected: {len(rejected_keys)}")
    print(f"  Families: {result['metadata']['n_families']} (L1)")
    print(f"  Groups:   {result['metadata']['n_groups']} "
          f"({result['metadata']['n_multi_groups']} multi, "
          f"{result['metadata']['n_singletons']} singleton)")


if __name__ == "__main__":
    main()
