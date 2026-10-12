#!/usr/bin/env python3
"""Filter causal graphs to a diverse subset based on scene and graph similarity.

Loads pruned/edited causal graphs from the outputs directory, extracts scene and
graph features, then applies a 3-phase greedy filtering algorithm to select a
maximally diverse subset.

Usage:
    python scripts/filter_diverse_graphs.py --verbose
    python scripts/filter_diverse_graphs.py --cross-threshold 0.80 --verbose
"""

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


# ── Inlined from scripts/old/generate_questions.py ──────────────────────────

def _build_adjacency_active(graph):
    """Build adjacency lists with metadata for active edges only."""
    children = defaultdict(set)
    parents = defaultdict(set)
    edge_meta = defaultdict(list)

    for e in graph.get("ego_edges", []):
        if not e.get("active", False):
            continue
        src, tgt = e["source"], e["target"]
        children[src].add(tgt)
        parents[tgt].add(src)
        edge_meta[(src, tgt)].append(e.get("effect_type", ""))

    for e in graph.get("inter_node_edges", []):
        src, tgt = e["source"], e["target"]
        children[src].add(tgt)
        parents[tgt].add(src)
        edge_meta[(src, tgt)].append(e.get("relation", ""))

    return children, parents, edge_meta


def _extract_active(graph, ego_id="ego"):
    """Extract active subgraph structures (direct/chain/fork/collider)."""
    children, parents, edge_meta = _build_adjacency_active(graph)
    results = {"direct": [], "chain": [], "fork": [], "collider": []}

    for a in parents.get(ego_id, []):
        effects = edge_meta.get((a, ego_id), [])
        results["direct"].append({"A": a, "effects": effects})

    for b in parents.get(ego_id, []):
        for a in parents.get(b, []):
            if a != ego_id:
                results["chain"].append({
                    "A": a, "B": b,
                    "AB_rel": edge_meta.get((a, b), []),
                    "B_ego_eff": edge_meta.get((b, ego_id), []),
                })

    for c in parents.get(ego_id, []):
        for b in children.get(c, []):
            if b != ego_id:
                results["fork"].append({
                    "C": c, "B": b,
                    "C_ego_eff": edge_meta.get((c, ego_id), []),
                    "C_B_rel": edge_meta.get((c, b), []),
                })

    ego_parents = list(parents.get(ego_id, []))
    for a, b in combinations(ego_parents, 2):
        results["collider"].append({
            "A": a, "B": b,
            "A_eff": edge_meta.get((a, ego_id), []),
            "B_eff": edge_meta.get((b, ego_id), []),
        })

    return results


def _equiv_sig(graph_type, sg):
    """Canonical signature ignoring specific node identity."""
    if graph_type == "direct":
        return ("direct", tuple(sorted(sg["effects"])))
    elif graph_type == "chain":
        return ("chain", tuple(sorted(sg["AB_rel"])), tuple(sorted(sg["B_ego_eff"])))
    elif graph_type == "fork":
        return ("fork", tuple(sorted(sg["C_ego_eff"])), tuple(sorted(sg["C_B_rel"])))
    elif graph_type == "collider":
        pair = tuple(sorted([tuple(sorted(sg["A_eff"])), tuple(sorted(sg["B_eff"]))]))
        return ("collider", pair)


# ── Constants ────────────────────────────────────────────────────────────────

EFFECT_TYPES = [
    "GRANTS_PROCEED",
    "INHIBITS_PROCEED",
    "CONSTRAINS_LONGITUDINAL",
    "CONSTRAINS_LATERAL",
    "DEGRADES",
]

NODE_TYPES = ["EGO", "OTHER_AGENT", "TRAFFIC_CONTROL", "ENVIRONMENT", "ROAD_OBSTACLE"]


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class SampleRecord:
    key: str                        # e.g. "nuscenes-scene-0001_SAMPLED_0"
    scene_num: int
    sample_idx: int
    graph_path: str                 # which file was loaded

    # Scene features
    location: str = ""
    nav_command: str = ""
    ego_speed: str = ""
    ego_heading: str = ""
    ego_accel: str = ""
    behavior_matches_nav: bool = True
    n_agents: int = 0

    # Graph features
    n_causal_nodes: int = 0
    n_distractors: int = 0
    node_type_counts: dict = field(default_factory=dict)
    node_category_counts: dict = field(default_factory=dict)
    n_ego_edges: int = 0
    n_active_edges: int = 0
    n_dormant_edges: int = 0
    edge_effect_counts: dict = field(default_factory=dict)
    active_effect_counts: dict = field(default_factory=dict)
    n_inter_edges: int = 0
    graph_depth: int = 0
    subgraph_sigs: set = field(default_factory=set)

    richness: float = 0.0

    def features_summary(self) -> str:
        effects = [e for e, c in self.edge_effect_counts.items() if c > 0]
        cats = list(self.node_category_counts.keys())
        return (
            f"{self.location}, {self.nav_command}, ego={self.ego_speed}/{self.ego_heading}, "
            f"nodes={self.n_causal_nodes}, active_edges={self.n_active_edges}, "
            f"inter_edges={self.n_inter_edges}, effects=[{', '.join(effects)}], "
            f"categories=[{', '.join(cats[:5])}]"
        )


# ── Feature extraction ──────────────────────────────────────────────────────

def load_graph(key: str, graph_dir: Path, preferred_suffix: str = "_verified") -> tuple[dict | None, str]:
    """Load graph JSON. `preferred_suffix` is tried first (e.g. "_fixed" or
    "_verified"); then falls through the standard chain _verified > _naren_edited
    > _pruned. Deduplicates so the preferred file isn't opened twice."""
    chain = [preferred_suffix, "_verified", "_naren_edited", "_pruned"]
    seen = set()
    for suffix in chain:
        if suffix in seen:
            continue
        seen.add(suffix)
        p = graph_dir / f"{key}{suffix}.json"
        if p.exists():
            with open(p) as f:
                return json.load(f), str(p)
    return None, ""


def load_meta(key: str, records_dir: Path) -> dict | None:
    """Load meta.json for a sample."""
    m = re.match(r"(nuscenes-scene-\d+)_(SAMPLED_\d+)", key)
    if not m:
        return None
    scene, sample = m.group(1), m.group(2)
    meta_path = records_dir / scene / sample / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def extract_features(key: str, graph_dir: Path, records_dir: Path,
                     graph_suffix: str = "_verified") -> SampleRecord | None:
    """Extract all features for one sample.

    `graph_suffix` (e.g. "_fixed" or "_verified") is tried first when loading
    the graph; falls back through the standard chain if absent.
    """
    m = re.match(r"nuscenes-scene-(\d+)_SAMPLED_(\d+)", key)
    if not m:
        return None

    graph, graph_path = load_graph(key, graph_dir, preferred_suffix=graph_suffix)
    if graph is None:
        return None

    meta = load_meta(key, records_dir)

    rec = SampleRecord(
        key=key,
        scene_num=int(m.group(1)),
        sample_idx=int(m.group(2)),
        graph_path=graph_path,
    )

    # Scene features from meta
    if meta:
        rec.location = meta.get("location", "")
        rec.nav_command = meta.get("nav_command", "")
        rec.n_agents = meta.get("n_agents", 0)

    # Scene features from graph ego_state
    ego_state = graph.get("ego_state", {})
    rec.ego_speed = ego_state.get("speed", "")
    rec.ego_heading = ego_state.get("inferred_heading_change", "")
    rec.ego_accel = ego_state.get("inferred_acceleration", "")
    rec.behavior_matches_nav = ego_state.get("behavior_matches_nav", True)

    # Graph node features
    nodes = [n for n in graph.get("nodes", []) if n.get("id") != "ego"]
    rec.n_causal_nodes = len(nodes)
    rec.n_distractors = len(graph.get("distractors", []))

    type_counts = Counter()
    cat_counts = Counter()
    for n in nodes:
        ntype = n.get("type", "UNKNOWN")
        ncat = n.get("category", "unknown")
        type_counts[ntype] += 1
        cat_counts[f"{ntype}:{ncat}"] += 1
    rec.node_type_counts = dict(type_counts)
    rec.node_category_counts = dict(cat_counts)

    # Graph edge features
    ego_edges = graph.get("ego_edges", [])
    rec.n_ego_edges = len(ego_edges)

    effect_counts = Counter()
    active_effect_counts = Counter()
    n_active = 0
    n_dormant = 0
    for e in ego_edges:
        etype = e.get("effect_type", "")
        effect_counts[etype] += 1
        if e.get("active", False):
            n_active += 1
            active_effect_counts[etype] += 1
        else:
            n_dormant += 1
    rec.n_active_edges = n_active
    rec.n_dormant_edges = n_dormant
    rec.edge_effect_counts = dict(effect_counts)
    rec.active_effect_counts = dict(active_effect_counts)

    # Inter-node edges
    inter_edges = graph.get("inter_node_edges", [])
    rec.n_inter_edges = len(inter_edges)
    def _safe_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0
    rec.graph_depth = max((_safe_int(e.get("layer")) for e in inter_edges), default=0)

    # Subgraph signatures
    sigs = set()
    active_sgs = _extract_active(graph)
    for sg_type, sg_list in active_sgs.items():
        for sg in sg_list:
            sig = _equiv_sig(sg_type, sg)
            if sig is not None:
                sigs.add(sig)
    rec.subgraph_sigs = sigs

    # Richness score
    rec.richness = (
        rec.n_causal_nodes * 2
        + rec.n_active_edges * 3
        + rec.n_inter_edges * 4
        + rec.n_dormant_edges * 1
    )

    return rec


# ── Similarity computation ──────────────────────────────────────────────────

def _jaccard_multiset(a: dict, b: dict) -> float:
    """Jaccard index on count dictionaries (multiset intersection / union)."""
    all_keys = set(a) | set(b)
    if not all_keys:
        return 1.0  # both empty → identical
    intersection = sum(min(a.get(k, 0), b.get(k, 0)) for k in all_keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in all_keys)
    return intersection / union if union > 0 else 1.0


def _cosine(a: dict, b: dict, keys: list[str]) -> float:
    """Cosine similarity over fixed key set."""
    va = [a.get(k, 0) for k in keys]
    vb = [b.get(k, 0) for k in keys]
    dot = sum(x * y for x, y in zip(va, vb))
    mag_a = math.sqrt(sum(x * x for x in va))
    mag_b = math.sqrt(sum(x * x for x in vb))
    if mag_a == 0 and mag_b == 0:
        return 1.0  # both zero vectors → identical
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _jaccard_sets(a: set, b: set) -> float:
    """Jaccard index on sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def scene_similarity(a: SampleRecord, b: SampleRecord) -> float:
    """Weighted categorical match on scene features."""
    weights = {
        "location": 1.0,
        "nav_command": 2.0,
        "ego_speed": 1.0,
        "ego_heading": 2.0,
        "ego_accel": 1.0,
        "behavior_matches_nav": 0.5,
    }
    score = 0.0
    total_weight = sum(weights.values())

    if a.location == b.location:
        score += weights["location"]
    if a.nav_command == b.nav_command:
        score += weights["nav_command"]
    if a.ego_speed == b.ego_speed:
        score += weights["ego_speed"]
    if a.ego_heading == b.ego_heading:
        score += weights["ego_heading"]
    if a.ego_accel == b.ego_accel:
        score += weights["ego_accel"]
    if a.behavior_matches_nav == b.behavior_matches_nav:
        score += weights["behavior_matches_nav"]

    return score / total_weight


def graph_similarity(a: SampleRecord, b: SampleRecord) -> float:
    """Composite graph structure similarity."""
    # Node composition (type:category multiset Jaccard)
    node_sim = _jaccard_multiset(a.node_category_counts, b.node_category_counts)

    # Edge effect profile (cosine over 5 effect types)
    edge_sim = _cosine(a.edge_effect_counts, b.edge_effect_counts, EFFECT_TYPES)

    # Active edge profile
    active_sim = _cosine(a.active_effect_counts, b.active_effect_counts, EFFECT_TYPES)

    # Topology similarity (normalized Euclidean on 4 features)
    topo_vec_a = [a.n_causal_nodes, a.n_ego_edges, a.n_inter_edges, a.graph_depth]
    topo_vec_b = [b.n_causal_nodes, b.n_ego_edges, b.n_inter_edges, b.graph_depth]
    # Max possible distance for normalization
    max_vals = [max(abs(x - y), 1) for x, y in zip(topo_vec_a, topo_vec_b)]
    dist = math.sqrt(sum(((x - y) / max(v, 1)) ** 2
                         for x, y, v in zip(topo_vec_a, topo_vec_b, [10, 10, 5, 3])))
    topo_sim = max(0.0, 1.0 - dist / 2.0)  # normalize so ~2.0 dist → 0 sim

    # Subgraph pattern similarity
    sig_sim = _jaccard_sets(a.subgraph_sigs, b.subgraph_sigs)

    return (
        0.25 * node_sim
        + 0.25 * edge_sim
        + 0.20 * active_sim
        + 0.15 * topo_sim
        + 0.15 * sig_sim
    )


def combined_similarity(a: SampleRecord, b: SampleRecord,
                        scene_weight: float = 0.35,
                        graph_weight: float = 0.65) -> float:
    return scene_weight * scene_similarity(a, b) + graph_weight * graph_similarity(a, b)


# ── Filtering algorithm ─────────────────────────────────────────────────────

def phase1_intra_scene_dedup(samples: list[SampleRecord],
                             threshold: float,
                             scene_weight: float,
                             graph_weight: float,
                             verbose: bool = False) -> tuple[list[SampleRecord], list[dict]]:
    """Within each NuScenes scene, deduplicate highly similar SAMPLEDs."""
    by_scene = defaultdict(list)
    for s in samples:
        by_scene[s.scene_num].append(s)

    kept = []
    rejected = []

    for scene_num in sorted(by_scene):
        scene_samples = sorted(by_scene[scene_num], key=lambda s: s.richness, reverse=True)
        if len(scene_samples) <= 1:
            kept.extend(scene_samples)
            continue

        selected_in_scene = [scene_samples[0]]
        for candidate in scene_samples[1:]:
            max_sim = max(
                combined_similarity(candidate, s, scene_weight, graph_weight)
                for s in selected_in_scene
            )
            if max_sim < threshold:
                selected_in_scene.append(candidate)
            else:
                most_similar = max(
                    selected_in_scene,
                    key=lambda s: combined_similarity(candidate, s, scene_weight, graph_weight),
                )
                rejected.append({
                    "key": candidate.key,
                    "phase": "phase1_intra_scene",
                    "most_similar_to": most_similar.key,
                    "similarity": round(max_sim, 4),
                })
                if verbose:
                    print(f"  P1 drop: {candidate.key} (sim={max_sim:.3f} to {most_similar.key})")

        kept.extend(selected_in_scene)

    return kept, rejected


def phase2_greedy_diverse_select(samples: list[SampleRecord],
                                 threshold: float,
                                 scene_weight: float,
                                 graph_weight: float,
                                 verbose: bool = False) -> tuple[list[SampleRecord], list[dict]]:
    """Cross-scene greedy diverse subset selection."""
    candidates = sorted(samples, key=lambda s: s.richness, reverse=True)
    if not candidates:
        return [], []

    selected = [candidates[0]]
    rejected = []

    for candidate in candidates[1:]:
        max_sim = max(
            combined_similarity(candidate, s, scene_weight, graph_weight)
            for s in selected
        )
        if max_sim < threshold:
            selected.append(candidate)
        else:
            most_similar = max(
                selected,
                key=lambda s: combined_similarity(candidate, s, scene_weight, graph_weight),
            )
            rejected.append({
                "key": candidate.key,
                "phase": "phase2_cross_scene",
                "most_similar_to": most_similar.key,
                "similarity": round(max_sim, 4),
            })
            if verbose:
                print(f"  P2 drop: {candidate.key} (sim={max_sim:.3f} to {most_similar.key})")

    return selected, rejected


def phase3_ensure_coverage(selected: list[SampleRecord],
                           all_samples: list[SampleRecord],
                           verbose: bool = False) -> tuple[list[SampleRecord], list[str]]:
    """Ensure coverage of important patterns; force-add from pool if missing."""
    selected_keys = {s.key for s in selected}
    remaining = [s for s in all_samples if s.key not in selected_keys]
    additions = []

    def _has(predicate, label):
        if any(predicate(s) for s in selected):
            return
        candidates = [s for s in remaining if predicate(s)]
        if candidates:
            best = max(candidates, key=lambda s: s.richness)
            selected.append(best)
            selected_keys.add(best.key)
            remaining.remove(best)
            additions.append(f"Added {best.key} for coverage: {label}")
            if verbose:
                print(f"  P3 add: {best.key} ({label})")

    # At least 1 per nav_command
    for cmd in ["GO STRAIGHT", "TURN LEFT", "TURN RIGHT", "STOPPED"]:
        _has(lambda s, c=cmd: s.nav_command == c, f"nav_command={cmd}")

    # At least 1 per location
    for loc in ["singapore-onenorth", "boston-seaport"]:
        _has(lambda s, l=loc: s.location == l, f"location={loc}")

    # At least 1 with inter-node edges
    _has(lambda s: s.n_inter_edges > 0, "has inter_node_edges")

    # At least 1 per effect type
    for eff in EFFECT_TYPES:
        _has(lambda s, e=eff: s.active_effect_counts.get(e, 0) > 0, f"active {eff}")

    # At least 1 with behavior mismatch
    _has(lambda s: not s.behavior_matches_nav, "behavior_matches_nav=false")

    return selected, additions


# ── Output ───────────────────────────────────────────────────────────────────

def build_manifest(selected: list[SampleRecord],
                   rejected: list[dict],
                   all_count: int,
                   p1_count: int,
                   args,
                   coverage_additions: list[str]) -> dict:
    """Build the output JSON manifest."""

    # Coverage stats
    loc_counts = Counter(s.location for s in selected)
    nav_counts = Counter(s.nav_command for s in selected)
    effect_present = {
        eff: any(s.active_effect_counts.get(eff, 0) > 0 for s in selected)
        for eff in EFFECT_TYPES
    }
    has_inter = any(s.n_inter_edges > 0 for s in selected)
    has_mismatch = any(not s.behavior_matches_nav for s in selected)
    heading_types = sorted(set(s.ego_heading for s in selected))

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_input": all_count,
            "after_phase1": p1_count,
            "final_selected": len(selected),
            "total_rejected": len(rejected),
            "thresholds": {
                "intra_scene": args.intra_threshold,
                "cross_scene": args.cross_threshold,
                "scene_weight": args.scene_weight,
                "graph_weight": args.graph_weight,
            },
        },
        "selected": [
            {
                "key": s.key,
                "scene_num": s.scene_num,
                "sample_idx": s.sample_idx,
                "richness": s.richness,
                "graph_path": s.graph_path,
                "features": s.features_summary(),
            }
            for s in sorted(selected, key=lambda s: (s.scene_num, s.sample_idx))
        ],
        "rejected": rejected,
        "coverage": {
            "locations": dict(loc_counts),
            "nav_commands": dict(nav_counts),
            "effect_types_present": effect_present,
            "has_inter_node_edges": has_inter,
            "has_behavior_mismatch": has_mismatch,
            "ego_heading_types": heading_types,
            "coverage_forced_additions": coverage_additions,
        },
    }


def print_summary(manifest: dict):
    """Print human-readable summary."""
    md = manifest["metadata"]
    cov = manifest["coverage"]

    print("\n" + "=" * 60)
    print("  DIVERSE GRAPH FILTERING SUMMARY")
    print("=" * 60)
    print(f"  Input samples:        {md['total_input']}")
    print(f"  After Phase 1 (intra):{md['after_phase1']}")
    print(f"  Final selected:       {md['final_selected']}")
    print(f"  Rejected:             {md['total_rejected']}")
    print(f"  Thresholds:           intra={md['thresholds']['intra_scene']}, "
          f"cross={md['thresholds']['cross_scene']}")
    print()
    print("  Coverage:")
    print(f"    Locations:      {cov['locations']}")
    print(f"    Nav commands:   {cov['nav_commands']}")
    print(f"    Effect types:   {cov['effect_types_present']}")
    print(f"    Inter-edges:    {cov['has_inter_node_edges']}")
    print(f"    Behavior mismatch: {cov['has_behavior_mismatch']}")
    print(f"    Ego headings:   {cov['ego_heading_types']}")
    if cov["coverage_forced_additions"]:
        print(f"    Forced adds:    {len(cov['coverage_forced_additions'])}")
        for a in cov["coverage_forced_additions"]:
            print(f"      - {a}")
    print("=" * 60)

    # Rejection breakdown
    phase_counts = Counter(r["phase"] for r in manifest["rejected"])
    if phase_counts:
        print("\n  Rejections by phase:")
        for phase, count in sorted(phase_counts.items()):
            print(f"    {phase}: {count}")

    # Richness distribution of selected
    richness_vals = [s["richness"] for s in manifest["selected"]]
    if richness_vals:
        print(f"\n  Selected richness: min={min(richness_vals):.0f}, "
              f"max={max(richness_vals):.0f}, "
              f"mean={sum(richness_vals)/len(richness_vals):.1f}")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_scene_range(s: str) -> tuple[int, int]:
    lo, hi = s.split("-")
    return int(lo), int(hi)


def main():
    parser = argparse.ArgumentParser(description="Filter causal graphs to a diverse subset")
    parser.add_argument("--graph-dir",
                        default="outputs/nuscenes_full_edit",
                        help="Directory with graph JSON files")
    parser.add_argument("--records-dir",
                        default="records/nuscenes_full",
                        help="Directory with scene records (meta.json)")
    parser.add_argument("--bookmarks",
                        default="edits/naren/bookmarks.json",
                        help="Path to bookmarks.json")
    parser.add_argument("--scene-range", default="1-95",
                        help="Scene number range to include (e.g. 1-95)")
    parser.add_argument("--intra-threshold", type=float, default=0.90,
                        help="Similarity threshold for intra-scene dedup (default: 0.90)")
    parser.add_argument("--cross-threshold", type=float, default=0.85,
                        help="Similarity threshold for cross-scene dedup (default: 0.85)")
    parser.add_argument("--scene-weight", type=float, default=0.35,
                        help="Weight for scene similarity in combined score (default: 0.35)")
    parser.add_argument("--graph-weight", type=float, default=0.65,
                        help="Weight for graph similarity in combined score (default: 0.65)")
    parser.add_argument("--output", default=None,
                        help="Output path for JSON manifest (default: diverse_subset.json in graph-dir)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample filtering decisions")
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    records_dir = Path(args.records_dir)
    scene_lo, scene_hi = parse_scene_range(args.scene_range)
    output_path = Path(args.output) if args.output else graph_dir / "diverse_subset.json"

    # Load bookmarks
    with open(args.bookmarks) as f:
        bookmarks = json.load(f)

    # Filter to good-flagged samples in scene range
    sample_keys = []
    for key, info in bookmarks.items():
        if info.get("flag") != "good":
            continue
        m = re.match(r"nuscenes-scene-(\d+)_SAMPLED_(\d+)", key)
        if not m:
            continue
        scene_num = int(m.group(1))
        if scene_lo <= scene_num <= scene_hi:
            sample_keys.append(key)

    print(f"Loading {len(sample_keys)} good-flagged samples from scenes {scene_lo}-{scene_hi}...")

    # Extract features
    all_samples = []
    load_failures = []
    for key in sorted(sample_keys):
        rec = extract_features(key, graph_dir, records_dir)
        if rec is None:
            load_failures.append(key)
        else:
            all_samples.append(rec)

    if load_failures:
        print(f"  Warning: failed to load {len(load_failures)} samples: {load_failures[:5]}...")
    print(f"  Loaded {len(all_samples)} samples successfully")

    total_input = len(all_samples)

    # Phase 1: Intra-scene dedup
    print("\nPhase 1: Intra-scene deduplication...")
    p1_kept, p1_rejected = phase1_intra_scene_dedup(
        all_samples, args.intra_threshold, args.scene_weight, args.graph_weight, args.verbose
    )
    print(f"  Kept {len(p1_kept)}, rejected {len(p1_rejected)}")

    # Phase 2: Cross-scene greedy diverse selection
    print("\nPhase 2: Cross-scene greedy diverse selection...")
    p2_kept, p2_rejected = phase2_greedy_diverse_select(
        p1_kept, args.cross_threshold, args.scene_weight, args.graph_weight, args.verbose
    )
    print(f"  Kept {len(p2_kept)}, rejected {len(p2_rejected)}")

    # Phase 3: Coverage guarantees
    print("\nPhase 3: Coverage guarantees...")
    p3_selected, p3_additions = phase3_ensure_coverage(p2_kept, all_samples, args.verbose)
    print(f"  Final selected: {len(p3_selected)} ({len(p3_additions)} forced additions)")

    # Build and write manifest
    all_rejected = p1_rejected + p2_rejected
    manifest = build_manifest(
        p3_selected, all_rejected, total_input, len(p1_kept), args, p3_additions
    )

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest written to: {output_path}")

    print_summary(manifest)


if __name__ == "__main__":
    main()
