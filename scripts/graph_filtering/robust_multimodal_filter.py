#!/usr/bin/env python3
"""Hierarchical multi-modal graph clustering for diverse subset selection.

3-level pipeline:
  Level 1: Graph Family — deterministic classification by topology, status,
           effect, source, distractor class, edge count class
  Level 2: Scene + Intent Bucket — road type, nav command, ego heading
  Level 3: Fine Clustering — agglomerative complete-link on averaged kernels
           with data-driven distance threshold (silhouette-optimal)

No arbitrary thresholds. Level 1+2 are categorical splits. Level 3 threshold
is derived from the data.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ── Configuration (only embedding settings, no thresholds) ───────────────


@dataclass
class ClusteringConfig:
    embedding_endpoint: str = ""  # vLLM endpoint, e.g. "http://194.68.245.163:22169"
    embedding_model: str = "jinaai/jina-embeddings-v5-text-small-text-matching"
    embedding_dim: int = 256
    wl_intra_k: int = 3  # sub-clusters per node/edge group (from elbow analysis)


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
        all_embs = [None] * len(texts)

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


# ── Kernel Builders ──────────────────────────────────────────────────────


def build_road_kernel(samples: list, graphs: dict, cfg: ClusteringConfig) -> np.ndarray:
    """K3: Road geometry — embedding of scene_context.road_geometry only."""
    texts = []
    for s in samples:
        ctx = graphs.get(s.key, {}).get("scene_context", {})
        texts.append(f"Road geometry: {ctx.get('road_geometry', '')}")
    emb = embed_texts(texts, cfg)
    return np.clip(emb @ emb.T, 0, 1)


def build_nav_kernel(samples: list, graphs: dict, cfg: ClusteringConfig) -> np.ndarray:
    """K4: Navigation intent — embedding of scene_context.navigation_intent only."""
    texts = []
    for s in samples:
        ctx = graphs.get(s.key, {}).get("scene_context", {})
        texts.append(f"Navigation intent: {ctx.get('navigation_intent', '')}")
    emb = embed_texts(texts, cfg)
    return np.clip(emb @ emb.T, 0, 1)


def build_weather_kernel(samples: list, graphs: dict, cfg: ClusteringConfig) -> np.ndarray:
    """K8: Environment — embedding of scene_context.environment_conditions."""
    texts = []
    for s in samples:
        ctx = graphs.get(s.key, {}).get("scene_context", {})
        env = (ctx.get("environment_conditions") or "").strip() or "Clear daytime"
        texts.append(f"Environment: {env}")
    emb = embed_texts(texts, cfg)
    return np.clip(emb @ emb.T, 0, 1)


def build_topology_kernel(samples: list, graphs: dict) -> np.ndarray:
    """K1: Subgraph topology — joint histogram of (pattern_type × source_agent_type).

    For each subgraph extracted from extract_subgraphs, emit a joint tag that
    includes which source agent category is involved. A "direct_car" is
    a different bucket from "direct_vru"; "chain_signal_car" differs from
    "chain_car_signal" (order preserved for directed patterns).
    """
    import sys
    _extract_dir = str(Path(__file__).resolve().parent.parent / "causal_graph_extraction")
    if _extract_dir not in sys.path:
        sys.path.insert(0, _extract_dir)
    from extract_subgraphs import extract_subgraphs

    def _cat(node_cats: dict, nid: str) -> str:
        return _categorize_agent(node_cats.get(nid, ""))

    feature_counts: list[Counter] = []
    vocab: dict[str, int] = {}

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

        feature_counts.append(counts)

    return _weighted_jaccard(feature_counts)


def build_status_effect_source_kernel(samples: list, graphs: dict) -> np.ndarray:
    """K2: Joint (status × effect × source) histogram per ego edge.

    Each ego edge contributes to a single joint bin:
        {status}|{effect_type}|{source_category}
    Two graphs with the same status/effect/source marginals but different
    joint assignments end up in different buckets.
    """
    EFFECTS = [
        "CONSTRAINS_LONGITUDINAL", "CONSTRAINS_LATERAL",
        "INHIBITS_PROCEED", "GRANTS_PROCEED", "DEGRADES", "UNKNOWN",
    ]

    feature_counts: list[Counter] = []
    vocab: dict[str, int] = {}

    for s in samples:
        g = graphs.get(s.key, {})
        node_cats = {n["id"]: n.get("category", "").lower() for n in g.get("nodes", [])}
        counts: Counter = Counter()

        for e in g.get("ego_edges", []):
            st_raw = e.get("causal_status", "")
            is_active = st_raw.startswith("active") or e.get("active", False)
            st_bin = "active" if is_active else "dormant"
            et = e.get("effect_type", "")
            effect_key = next((ef for ef in EFFECTS if ef in et), "UNKNOWN")
            src_cat = _categorize_agent(node_cats.get(e.get("source", ""), ""))
            key = f"{st_bin}|{effect_key}|{src_cat}"
            counts[key] += 1

        # Add a graph-level "empty" tag when no ego edges, so empty graphs
        # cluster together instead of all appearing as zero vectors.
        if not counts:
            counts["__no_edges__"] += 1

        feature_counts.append(counts)

    return _weighted_jaccard(feature_counts)


def build_behavior_kernel(samples: list, graphs: dict, cfg: ClusteringConfig) -> np.ndarray:
    """K5: Ego behavior — embedding of behavior_explanation + nav_command +
    inter_node_edge justifications.
    """
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


def build_structure_kernel(samples: list, graphs: dict, cfg: ClusteringConfig) -> np.ndarray:
    """WL subtree kernel via explicit feature maps — O(n) + one matmul.

    Mathematically equivalent to GraKeL's WL subtree kernel (Shervashidze 2011)
    but computes φ(G) per graph then K = cosine(φ @ φ.T) instead of pairwise.
    """
    import hashlib
    from sklearn.cluster import KMeans
    from scipy.sparse import lil_matrix
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    INTRA_K = cfg.wl_intra_k
    WL_ITER = 4

    def ngroup(nd):
        t, cat = nd.get("type", "").upper(), nd.get("category", "").lower()
        if t == "EGO": return "EGO"
        if any(x in cat for x in ("pedestrian", "bicycle", "motorcycle")): return "VRU"
        if any(x in cat for x in ("truck", "bus", "trailer", "construction")): return "LARGE_VEHICLE"
        if any(x in cat for x in ("car", "van", "suv")): return "VEHICLE"
        if "signal" in cat: return "SIGNAL"
        if "sign" in cat: return "SIGN"
        if "crosswalk" in cat or "crossing" in cat: return "CROSSWALK"
        if any(x in cat for x in ("cone", "barrier", "jersey")): return "OBSTACLE"
        if "gate" in cat: return "GATE"
        return f"{t}:{cat}"

    def egroup(e):
        st = e.get("causal_status", "unknown")
        et = e.get("effect_type", e.get("relation", "UNKNOWN"))
        if st.startswith("active"): return f"active_{et}"
        if st == "dormant": return f"dormant_{et}"
        return et

    # ── Collect node/edge entries and embed for intra-cluster labels ──
    nents, eents = [], []
    for gi, s in enumerate(samples):
        g = graphs.get(s.key, {})
        nl = g.get("nodes", [])
        nm = {nd["id"]: i for i, nd in enumerate(nl)}
        for nd in nl:
            nents.append((gi, nm[nd["id"]], f"{nd.get('type','')}: {nd.get('category','')}: {_strip_agent_ids(nd.get('label',''))}", nd))
        for e in g.get("ego_edges", []):
            src, tgt = e.get("source"), e.get("target")
            if src in nm and tgt in nm:
                eents.append((gi, nm[src], nm[tgt], f"{e.get('causal_status','unknown')} {e.get('effect_type','')}: {_strip_agent_ids(e.get('justification',''))}", e))
        for e in g.get("inter_node_edges", []):
            src, tgt = e.get("source"), e.get("target")
            if src in nm and tgt in nm:
                eents.append((gi, nm[src], nm[tgt], f"{e.get('relation','')}: {_strip_agent_ids(e.get('justification',''))}", e))

    ne = embed_texts([x[2] for x in nents], cfg)
    ee = embed_texts([x[3] for x in eents], cfg)

    def intra(embs, items, gfn):
        groups: dict[str, list[int]] = {}
        for i, item in enumerate(items):
            groups.setdefault(gfn(item), []).append(i)
        labels = [""] * len(items)
        for grp, idxs in groups.items():
            if len(idxs) <= INTRA_K:
                for r, idx in enumerate(idxs):
                    labels[idx] = f"{grp}_{r}"
            else:
                ge = embs[idxs]
                ns = min(INTRA_K, len({tuple(embs[i].tolist()) for i in idxs}))
                if ns <= 1:
                    for idx in idxs:
                        labels[idx] = f"{grp}_0"
                else:
                    km = KMeans(n_clusters=ns, random_state=42, n_init=10).fit(ge)
                    for r, idx in enumerate(idxs):
                        labels[idx] = f"{grp}_{km.labels_[r]}"
        return labels

    nlabels = intra(ne, [x[3] for x in nents], ngroup)
    elabels = intra(ee, [x[4] for x in eents], egroup)

    # ── Build per-graph node labels + adjacency ──
    per_graph_nodes: list[list[str]] = [[] for _ in samples]
    per_graph_adj: list[list[tuple[int, int, str]]] = [[] for _ in samples]

    for i, (gi, ni, _, _) in enumerate(nents):
        # Extend list to fit node index
        while len(per_graph_nodes[gi]) <= ni:
            per_graph_nodes[gi].append("UNK")
        per_graph_nodes[gi][ni] = nlabels[i]
    for i, (gi, si, ti, _, _) in enumerate(eents):
        per_graph_adj[gi].append((si, ti, elabels[i]))

    # ── WL hashing: collect features across all iterations ──
    feature_vocab: dict[str, int] = {}
    graph_features: list[Counter] = [Counter() for _ in samples]

    def _fid(label: str) -> int:
        if label not in feature_vocab:
            feature_vocab[label] = len(feature_vocab)
        return feature_vocab[label]

    for gi in range(len(samples)):
        node_labels = list(per_graph_nodes[gi])
        if not node_labels:
            graph_features[gi][_fid("EMPTY")] += 1
            continue

        n_nodes = len(node_labels)
        neighbors: list[list[tuple[int, str]]] = [[] for _ in range(n_nodes)]
        for si, ti, el in per_graph_adj[gi]:
            if si < n_nodes and ti < n_nodes:
                neighbors[si].append((ti, el))
                neighbors[ti].append((si, el))

        # h=0: initial labels
        for lbl in node_labels:
            graph_features[gi][_fid(f"h0:{lbl}")] += 1

        # WL iterations
        for h in range(1, WL_ITER + 1):
            new_labels = []
            for ni in range(n_nodes):
                nbr_strs = sorted(f"{el}>{node_labels[nj]}" for nj, el in neighbors[ni])
                wl_str = f"{node_labels[ni]}|{'|'.join(nbr_strs)}"
                new_lbl = hashlib.md5(wl_str.encode()).hexdigest()[:16]
                new_labels.append(new_lbl)
                graph_features[gi][_fid(f"h{h}:{new_lbl}")] += 1
            node_labels = new_labels

    # ── Sparse feature matrix → cosine similarity ──
    n = len(samples)
    n_feats = len(feature_vocab)
    F = lil_matrix((n, n_feats), dtype=np.float32)
    for gi in range(n):
        for fid, count in graph_features[gi].items():
            F[gi, fid] = count

    K = cos_sim(F.tocsr())
    np.clip(K, 0, 1, out=K)
    return K


def _role_histograms(samples: list, graphs: dict) -> tuple:
    """Return (active_hist_list, dormant_hist_list) of agent category counts."""
    active_lists, dormant_lists = [], []
    for s in samples:
        g = graphs.get(s.key, {})
        node_cats = {n["id"]: n.get("category", "") for n in g.get("nodes", [])}
        a_counts, d_counts = Counter(), Counter()
        for e in g.get("ego_edges", []):
            st = "active" if e.get("active") or str(e.get("causal_status","")).startswith("active") else "dormant"
            cat = _categorize_agent(node_cats.get(e.get("source", ""), ""))
            if st == "active":
                a_counts[cat] += 1
            else:
                d_counts[cat] += 1
        if not a_counts: a_counts["__none__"] += 1
        if not d_counts: d_counts["__none__"] += 1
        active_lists.append(a_counts)
        dormant_lists.append(d_counts)
    return active_lists, dormant_lists


def build_active_sem_kernel(samples: list, graphs: dict) -> np.ndarray:
    """K6a: Active-source agent composition histogram (weighted Jaccard)."""
    active_lists, _ = _role_histograms(samples, graphs)
    return _weighted_jaccard(active_lists)


def build_dormant_sem_kernel(samples: list, graphs: dict) -> np.ndarray:
    """K6b: Dormant-source agent composition histogram (weighted Jaccard)."""
    _, dormant_lists = _role_histograms(samples, graphs)
    return _weighted_jaccard(dormant_lists)


# Back-compat alias: build_semantic_kernel → active kernel
def build_semantic_kernel(samples: list, graphs: dict) -> np.ndarray:
    """K6: Combined active + dormant semantic. Kept for back-compat; prefer
    build_active_sem_kernel / build_dormant_sem_kernel separately.
    """
    return (build_active_sem_kernel(samples, graphs) +
            build_dormant_sem_kernel(samples, graphs)) / 2.0


def build_interaction_kernel(samples: list, graphs: dict) -> np.ndarray:
    """K7: Inter-node interaction — joint (relation × src_type × tgt_type) hist.

    For each inter_node_edge, emit a joint bin: `{relation}|{src_cat}>{tgt_cat}`.
    For directional relations (AFFECTS), source and target order is preserved.
    For undirected relations (CONFOUNDED_BY, SHARES_UNOBS_CAUSE), categories
    are sorted so the bin is order-invariant.
    """
    DIRECTED = {"AFFECTS"}

    feature_counts: list[Counter] = []
    vocab: dict[str, int] = {}

    for s in samples:
        g = graphs.get(s.key, {})
        node_cats = {n["id"]: n.get("category", "").lower() for n in g.get("nodes", [])}
        counts: Counter = Counter()

        for e in g.get("inter_node_edges", []) or []:
            rel = e.get("relation", "UNKNOWN")
            src_cat = _categorize_agent(node_cats.get(e.get("source", ""), ""))
            tgt_cat = _categorize_agent(node_cats.get(e.get("target", ""), ""))
            if rel in DIRECTED:
                key = f"{rel}|{src_cat}>{tgt_cat}"
            else:
                a, b = sorted([src_cat, tgt_cat])
                key = f"{rel}|{a}+{b}"
            counts[key] += 1

        if not counts:
            counts["__no_inter__"] += 1

        feature_counts.append(counts)

    return _weighted_jaccard(feature_counts)


def _weighted_jaccard(feature_counts: list) -> np.ndarray:
    """Weighted Jaccard on count dicts: sim = Σ min(a,b) / Σ max(a,b).

    Preserves both set membership AND count differences. Exact match → 1.0.
    Same items with different counts → proportional penalty.
    """
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

    # Row sums for max computation: max(a,b) = a + b - min(a,b)
    # But cleaner to compute per-pair
    K = np.zeros((n, n), dtype=np.float64)
    row_sums = M.sum(axis=1)
    for i in range(n):
        if i % 500 == 0 and n > 1000:
            pass  # could print progress
        # Element-wise min/max broadcast
        min_ij = np.minimum(M[i:i+1], M)  # (n, vocab)
        sum_min = min_ij.sum(axis=1)
        sum_max = row_sums[i] + row_sums - sum_min
        # Both empty → 1.0
        empty_pair = (row_sums[i] == 0) & (row_sums == 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            K[i] = np.where(sum_max > 0, sum_min / sum_max, 0.0)
        K[i][empty_pair] = 1.0
    return K


def _jaccard_from_sets(set_list: list[set]) -> np.ndarray:
    """Vectorized Jaccard similarity from list of sets via binary matrix."""
    vocab = {}
    for s in set_list:
        for item in s:
            if item not in vocab:
                vocab[item] = len(vocab)
    n = len(set_list)
    if not vocab:
        return np.ones((n, n))

    M = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, s in enumerate(set_list):
        for item in s:
            M[i, vocab[item]] = 1.0

    intersection = M @ M.T
    row_sums = M.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection

    K = np.zeros((n, n))
    mask = union > 0
    K[mask] = intersection[mask] / union[mask]
    both_empty = (row_sums == 0)[:, None] & (row_sums == 0)[None, :]
    K[both_empty] = 1.0
    return K


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
    if "cone" in cat or "barrier" in cat or "construction" in cat or "gate" in cat:
        return "obstacle"
    if "sign" in cat:
        return "sign"
    return cat if cat else "unknown"


def build_distractor_kernel(samples: list, graphs: dict) -> np.ndarray:
    """K9: Distractor composition — weighted Jaccard on distractor category counts.

    Counts how many distractors of each category (vru, car, truck, bus, etc.)
    the scene has. {car:2, vru:1} ≠ {car:1, vru:2} — preserves proportion.
    """
    feature_counts: list[Counter] = []
    for s in samples:
        g = graphs.get(s.key, {})
        counts: Counter = Counter()
        for d in g.get("distractors", []) or []:
            cat = d.get("category", "")
            if cat:
                counts[_categorize_agent(cat)] += 1
        if not counts:
            counts["__no_distractors__"] += 1
        feature_counts.append(counts)
    return _weighted_jaccard(feature_counts)


def build_edge_profile_kernel(samples: list, graphs: dict) -> np.ndarray:
    """Jaccard similarity over (status, effect_type) edge profiles (vectorized)."""
    profiles = []
    for s in samples:
        g = graphs.get(s.key, {})
        edge_types = set()
        for e in g.get("ego_edges", []):
            st = e.get("causal_status", "unknown")
            if st.startswith("active"):
                st = "active"
            et = e.get("effect_type", "UNKNOWN")
            edge_types.add(f"{st}_{et}")
        for e in g.get("inter_node_edges", []):
            rel = e.get("relation", "UNKNOWN")
            edge_types.add(f"inter_{rel}")
        profiles.append(edge_types)
    return _jaccard_from_sets(profiles)


# ── Level 1: Graph Family (deterministic) ───────────────────────────────


def _topology_label(sample) -> str:
    sigs = {sig[0] for sig in sample.subgraph_sigs} if sample.subgraph_sigs else set()
    if "confounded" in str(sample.subgraph_sigs).lower():
        return "confounding"
    if "collider" in sigs:
        return "collider"
    if "chain" in sigs:
        return "chain"
    return "direct"


def _status_label(sample) -> str:
    if sample.n_active_edges > 0 and sample.n_dormant_edges > 0:
        return "mixed"
    if sample.n_active_edges > 0:
        return "active_only"
    if sample.n_dormant_edges > 0:
        return "dormant_only"
    return "no_edges"


def _effect_label(sample, graph: dict) -> str:
    """Sorted edge profile: each ego edge as (status, effect_type).

    Preserves graph topology — a dormant CONSTRAINS_LONGITUDINAL + active
    INHIBITS_PROCEED is a fundamentally different graph from dormant
    CONSTRAINS_LONGITUDINAL + active GRANTS_PROCEED.
    """
    profile = []
    for e in graph.get("ego_edges", []):
        st = e.get("causal_status", "unknown")
        if st.startswith("active"):
            st = "active"
        et = e.get("effect_type", "UNKNOWN")
        profile.append(f"{st}_{et}")
    if not profile:
        return "none"
    return ",".join(sorted(profile))


def _source_label(sample, graph: dict) -> str:
    """Sorted list of source types for each ego edge — preserves per-edge identity."""
    node_cats = {n["id"]: n.get("category", "").lower() for n in graph.get("nodes", [])}
    sources = []
    for e in graph.get("ego_edges", []):
        src_cat = node_cats.get(e.get("source", ""), "")
        if "pedestrian" in src_cat or "bicycle" in src_cat or "motorcycle" in src_cat:
            sources.append("vru")
        elif "truck" in src_cat or "trailer" in src_cat:
            sources.append("truck")
        elif "bus" in src_cat:
            sources.append("bus")
        elif "car" in src_cat or "van" in src_cat or "suv" in src_cat:
            sources.append("car")
        elif "signal" in src_cat:
            sources.append("signal")
        elif "crosswalk" in src_cat or "crossing" in src_cat:
            sources.append("crosswalk")
        elif "cone" in src_cat or "barrier" in src_cat or "construction" in src_cat or "gate" in src_cat:
            sources.append("obstacle")
        elif "sign" in src_cat:
            sources.append("sign")
        else:
            sources.append("other")
    if not sources:
        return "none"
    return ",".join(sorted(sources))


def _distractor_class(graph: dict) -> str:
    n = len(graph.get("distractors", []))
    if n == 0: return "no_distractors"
    if n <= 3: return "few_distractors"
    return "many_distractors"


def _edge_count_class(sample) -> str:
    total = sample.n_ego_edges
    if total <= 1: return "single_edge"
    if total <= 3: return "few_edges"
    return "many_edges"


def graph_family(sample, graph: dict) -> str:
    return " | ".join([
        _topology_label(sample),
        _status_label(sample),
        _effect_label(sample, graph),
        _source_label(sample, graph),
    ])


# ── Level 2: Scene + Intent Bucket (deterministic) ──────────────────────


def _road_type(graph: dict) -> str:
    text = graph.get("scene_context", {}).get("road_geometry", "").lower()
    if "roundabout" in text: return "roundabout"
    if "intersection" in text or "junction" in text: return "intersection"
    if "parking" in text or "service area" in text: return "parking"
    if "highway" in text or "expressway" in text: return "highway"
    if "divided road" in text or "multi-lane" in text: return "multi_lane"
    return "road"


def scene_intent_bucket(sample, graph: dict) -> str:
    road = _road_type(graph)
    nav = sample.nav_command or "UNKNOWN"
    heading = sample.ego_heading or "unknown"
    return f"{road} | {nav} | {heading}"


# ── Flat Clustering (Agglomerative + Elbow) ─────────────────────────────


def _elbow_threshold(merge_distances: np.ndarray) -> float:
    """Find the elbow point in a sorted monotonic curve (kneedle algorithm).

    Draws a line from the first to the last point, finds the point with
    the largest perpendicular distance from that line. Returns the
    corresponding merge distance.
    """
    n = len(merge_distances)
    if n < 3:
        return float(merge_distances[-1]) if n else 0.5

    # Normalize to [0, 1] on both axes
    x = np.arange(n, dtype=np.float64) / (n - 1)
    y = merge_distances - merge_distances[0]
    if y[-1] > 0:
        y = y / y[-1]
    else:
        return float(merge_distances[-1])

    # Distance from each point to the diagonal y = x
    # For a monotonic increasing curve, elbow is max(y - x) (knee below line)
    # or max(x - y) (knee above line). Take abs max.
    diffs = y - x
    elbow_idx = int(np.argmax(np.abs(diffs)))
    return float(merge_distances[elbow_idx])


def _flat_cluster(K_combined: np.ndarray, linkage_method: str = "average") -> tuple:
    """Flat agglomerative clustering with elbow-selected threshold.

    Returns (labels, threshold). Uses scipy.cluster.hierarchy.linkage for
    full dendrogram, then finds elbow in merge distances and cuts.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = K_combined.shape[0]
    dist = 1.0 - K_combined
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)

    # Enforce symmetry (floating point can drift)
    dist = (dist + dist.T) / 2.0

    # Condensed form for scipy linkage
    condensed = squareform(dist, checks=False)

    # Build dendrogram
    Z = linkage(condensed, method=linkage_method)
    merge_distances = np.sort(Z[:, 2])

    # Find elbow
    thresh = _elbow_threshold(merge_distances)

    # Cut
    labels = fcluster(Z, t=thresh, criterion="distance")
    return labels, thresh, merge_distances


# ── Representative Selection ─────────────────────────────────────────────


def _sample_weight(sample, nav_counts: Counter, effect_counts: Counter) -> float:
    """Weight = richness + rarity bonus (no arbitrary weights)."""
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


# ── Main Pipeline ────────────────────────────────────────────────────────


def run_clustering(
    samples: list,
    graphs: dict,
    keys: list[str],
    K_topology: np.ndarray,
    K_ses: np.ndarray,
    K_road: np.ndarray,
    K_nav: np.ndarray,
    K_beh: np.ndarray,
    K_active_sem: np.ndarray,
    K_dormant_sem: np.ndarray,
    K_inter: np.ndarray,
    K_weather: np.ndarray,
    K_distractor: np.ndarray,
) -> dict:
    """Run the full 3-level clustering pipeline.

    Uses 10 kernels:
      K1  K_topology     — subgraph pattern counts (direct/chain/collider/...)
      K2  K_ses          — (status × effect × source) joint histogram
      K3  K_road         — road_geometry embedding
      K4  K_nav          — navigation_intent embedding
      K5  K_beh          — ego behavior embedding
      K6a K_active_sem   — active ego-source agent composition histogram
      K6b K_dormant_sem  — dormant ego-source agent composition histogram
      K7  K_inter        — inter-node (relation × src × tgt) joint histogram
      K8  K_weather      — environment_conditions embedding
      K9  K_distractor   — distractor category count histogram
    """
    n = len(samples)

    # Combined kernel = simple average of all 10
    K_combined = (K_topology + K_ses + K_road + K_nav + K_beh
                  + K_active_sem + K_dormant_sem + K_inter + K_weather + K_distractor) / 10.0

    # Frequency counts for sample weighting
    nav_counts = Counter(s.nav_command for s in samples if s.nav_command)
    effect_counts = Counter(e for s in samples for e, c in s.active_effect_counts.items() if c > 0)
    weights = [_sample_weight(samples[i], nav_counts, effect_counts) for i in range(n)]

    # ── Flat agglomerative + elbow ──
    # Complete-linkage: all pairs in cluster must be similar (prevents chaining).
    print(f"  Clustering {n} samples (flat, agglomerative complete-link + elbow)...")
    labels, thresh, merge_distances = _flat_cluster(K_combined, linkage_method="complete")
    print(f"  Elbow threshold: {thresh:.3f}  (merge-dist range: "
          f"{merge_distances[0]:.3f} .. {merge_distances[-1]:.3f})")

    cluster_members: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        cluster_members[int(lab)].append(idx)

    groups_list = []
    for gi, (_key, members) in enumerate(sorted(cluster_members.items(),
                                                 key=lambda kv: (-len(kv[1]), kv[0])), start=1):
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

    # Keep output shape compatible with downstream (sample_diverse, renderer):
    # wrap all groups under a single virtual family + bucket.
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
            "n_families": len(families),
            "n_groups": len(groups_list),
            "n_multi_groups": n_multi,
            "n_singletons": n_singletons,
            "largest_group": max((g["size"] for g in groups_list), default=0),
            "elbow_threshold": float(thresh),
        },
        "families": families,
    }

    print(f"  Groups: {len(groups_list)} ({n_multi} multi, {n_singletons} singleton)")
    print(f"  Largest group: {result['metadata']['largest_group']}")

    return result


def export_clusters(result: dict, output_path: str) -> None:
    """Save clusters to JSON."""
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Clusters saved: {output_path}")


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    import argparse
    import glob
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from filter_diverse_graphs import extract_features

    parser = argparse.ArgumentParser(
        description="Hierarchical multi-modal diversity filter — picks one representative per cluster from *_verified.json graphs."
    )
    parser.add_argument("--graph-dir", required=True,
                        help="Directory with graph JSON files (reads *_verified.json)")
    parser.add_argument("--records-dir", required=True,
                        help="Directory with scene records (meta.json)")
    parser.add_argument("--output", default=None,
                        help="Output manifest path or directory (default: diverse_graphs.json in graph-dir)")
    parser.add_argument("--clusters-output", default=None,
                        help="Directory (or path) to write clusters.json dump of the full hierarchy")
    parser.add_argument("--embedding-endpoint", default="",
                        help="vLLM embedding endpoint URL (e.g. http://host:port). Empty = local SentenceTransformer.")
    args = parser.parse_args()

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

    verified_files = sorted(glob.glob(str(graph_dir / "*_verified.json")))
    sample_keys = [os.path.basename(f).replace("_verified.json", "") for f in verified_files]
    print(f"Found {len(sample_keys)} verified samples in {graph_dir}")
    if not sample_keys:
        print("No _verified.json files found. Nothing to filter.")
        return

    samples = []
    graphs = {}
    load_failures = []
    for key in sample_keys:
        rec = extract_features(key, graph_dir, records_dir)
        if rec is None:
            load_failures.append(key)
            continue
        with open(graph_dir / f"{key}_verified.json") as f:
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

    print("\nBuilding kernels...")
    print("  K1 topology (subgraph histogram)...", flush=True)
    k_topo = build_topology_kernel(samples, graphs)
    print("  K2 status+effect+source (histogram)...", flush=True)
    k_ses = build_status_effect_source_kernel(samples, graphs)
    print("  K3 road (geometry embedding)...", flush=True)
    k_road = build_road_kernel(samples, graphs, cfg)
    print("  K4 nav (intent embedding)...", flush=True)
    k_nav = build_nav_kernel(samples, graphs, cfg)
    print("  K5 behavior (embedding)...", flush=True)
    k_beh = build_behavior_kernel(samples, graphs, cfg)
    print("  K6a active-sem (agent histogram)...", flush=True)
    k_act_sem = build_active_sem_kernel(samples, graphs)
    print("  K6b dormant-sem (agent histogram)...", flush=True)
    k_dor_sem = build_dormant_sem_kernel(samples, graphs)
    print("  K7 inter (relation histogram)...", flush=True)
    k_inter = build_interaction_kernel(samples, graphs)
    print("  K8 weather (embedding)...", flush=True)
    k_weather = build_weather_kernel(samples, graphs, cfg)
    print("  K9 distractor (histogram)...", flush=True)
    k_dist = build_distractor_kernel(samples, graphs)

    print("\nRunning 3-level clustering...")
    result = run_clustering(samples, graphs, keys,
                            k_topo, k_ses, k_road, k_nav, k_beh,
                            k_act_sem, k_dor_sem, k_inter, k_weather, k_dist)

    sample_by_key = {s.key: s for s in samples}
    selected_entries = []
    for fam in result["families"]:
        for bucket in fam["buckets"]:
            for group in bucket["groups"]:
                leader_key = group["leader"]
                s = sample_by_key.get(leader_key)
                if s is None:
                    continue
                selected_entries.append({
                    "key": leader_key,
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
    print(f"  Families: {result['metadata']['n_families']}")
    print(f"  Groups:   {result['metadata']['n_groups']} "
          f"({result['metadata']['n_multi_groups']} multi, "
          f"{result['metadata']['n_singletons']} singleton)")


if __name__ == "__main__":
    main()
