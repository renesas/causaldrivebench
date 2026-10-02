#!/usr/bin/env python3
"""Compute comprehensive statistics over causal graph JSON files.

Scans a directory for all causal graph samples, loads the best-priority
variant (verified > edited > pruned), and produces a detailed markdown
report covering topology, node/edge distributions, ego behavior, graph
structure types, and more.

Usage:
    python scripts/graph_filtering/graph_statistics.py \
        --output stats_report.md

    python scripts/graph_filtering/graph_statistics.py \
        --limit 10 --output test_stats.md   # quick test
"""

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

# ── Import from sibling module ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_diverse_graphs import (
    _build_adjacency_active,
    _extract_active,
    _equiv_sig,
    EFFECT_TYPES,
    NODE_TYPES,
)


# ── Discovery ─────────────────────────────────────────────────────────────────

_SAMPLE_RE = re.compile(r"(nuscenes-scene-\d+_SAMPLED_\d+)")


def discover_samples(graph_dir: Path) -> list[str]:
    """Find all unique sample keys by scanning JSON filenames."""
    keys = set()
    for p in graph_dir.iterdir():
        if not p.suffix == ".json":
            continue
        m = _SAMPLE_RE.match(p.stem.split("_pruned")[0]
                             .split("_verified")[0]
                             .split("_naren_edited")[0])
        if m:
            keys.add(m.group(1))
    return sorted(keys)


def load_graph(key: str, graph_dir: Path) -> tuple[dict | None, str]:
    """Load graph JSON, preferring verified > naren_edited > pruned."""
    for suffix in ("_verified", "_naren_edited", "_pruned"):
        p = graph_dir / f"{key}{suffix}.json"
        if p.exists():
            with open(p) as f:
                return json.load(f), suffix.strip("_")
    return None, ""


# ── Position bucketing ────────────────────────────────────────────────────────

def bucket_position(pos: str) -> str:
    """Bucket free-text position_relative_to_ego into coarse categories."""
    low = pos.lower().strip()
    if low in ("self", "n/a", ""):
        return "self"

    # Direction
    dirs = []
    if "ahead" in low or "front" in low or "forward" in low:
        dirs.append("ahead")
    if "behind" in low or "rear" in low or "back" in low:
        dirs.append("behind")
    if "left" in low:
        dirs.append("left")
    if "right" in low:
        dirs.append("right")
    direction = "_".join(dirs) if dirs else "unknown_dir"

    # Distance
    dist_match = re.search(r"~?(\d+)\s*m", low)
    if dist_match:
        d = int(dist_match.group(1))
        dist = "close" if d < 30 else ("mid" if d < 60 else "far")
    elif "close" in low or "near" in low or "adjacent" in low:
        dist = "close"
    elif "far" in low or "distant" in low:
        dist = "far"
    else:
        dist = "unknown_dist"

    return f"{dist}_{direction}"


# ── Road geometry keyword extraction ──────────────────────────────────────────

_ROAD_KEYWORDS = [
    "intersection", "t-junction", "t-intersection", "roundabout", "highway",
    "expressway", "parking", "multi-lane", "two-lane", "one-way", "four-lane",
    "three-lane", "divided", "undivided", "crosswalk", "pedestrian crossing",
    "construction", "merging", "ramp", "bridge", "tunnel", "curve",
    "straight road", "dead end", "cul-de-sac", "bus stop", "traffic circle",
]


def extract_road_keywords(geom: str) -> list[str]:
    """Extract matching road keywords from free-text road_geometry."""
    low = geom.lower()
    return [kw for kw in _ROAD_KEYWORDS if kw in low]


# ── Extended topology detection ───────────────────────────────────────────────

def detect_upstream_forks(graph: dict, ego_id: str = "ego") -> list[dict]:
    """Detect upstream common causes: A→B→ego and A→C→ego where A is NOT
    a direct parent of ego (missed by _extract_active's fork detection)."""
    children, parents, edge_meta = _build_adjacency_active(graph)
    ego_parents = parents.get(ego_id, set())
    # For each pair of ego-parents, check if they share a common parent
    upstream_forks = []
    ego_parents_list = list(ego_parents)
    # Build reverse: for each ego-parent, who are its parents?
    for i, b1 in enumerate(ego_parents_list):
        for b2 in ego_parents_list[i + 1:]:
            common = parents.get(b1, set()) & parents.get(b2, set())
            common -= {ego_id}  # exclude ego itself
            common -= ego_parents  # exclude direct ego-parents (those are regular forks)
            for a in common:
                upstream_forks.append({"A": a, "B1": b1, "B2": b2})
    return upstream_forks


# ── Per-graph stats extraction ────────────────────────────────────────────────

def extract_graph_stats(graph: dict) -> dict:
    """Extract all per-graph raw metrics."""
    stats = {}

    # Scene context
    ctx = graph.get("scene_context", {})
    stats["road_geometry"] = ctx.get("road_geometry", "")
    stats["navigation_intent"] = ctx.get("navigation_intent", "")
    stats["road_keywords"] = extract_road_keywords(stats["road_geometry"])

    # Ego state
    ego = graph.get("ego_state", {})
    stats["speed"] = ego.get("speed", "unknown")
    stats["acceleration"] = ego.get("inferred_acceleration", "unknown")
    stats["heading"] = ego.get("inferred_heading_change", "unknown")
    stats["behavior_matches_nav"] = ego.get("behavior_matches_nav", True)

    # Nodes (excluding ego)
    nodes = [n for n in graph.get("nodes", []) if n.get("id") != "ego"]
    stats["n_nodes"] = len(nodes)
    stats["node_type_counts"] = Counter(n.get("type", "UNKNOWN") for n in nodes)
    stats["node_category_counts"] = Counter(n.get("category", "unknown") for n in nodes)

    # Position bucketing
    positions = []
    for n in nodes:
        pos = n.get("position_relative_to_ego", "")
        if pos:
            positions.append(bucket_position(pos))
    stats["position_counts"] = Counter(positions)

    # Motion
    static = sum(1 for n in nodes if n.get("motion", {}).get("is_static", False))
    stats["static_nodes"] = static
    stats["dynamic_nodes"] = len(nodes) - static

    # Camera visibility — extract camera names (cam_front, bev, etc.)
    camera_re = re.compile(r"(cam_\w+|bev|state)", re.IGNORECASE)
    visibility = []
    for n in nodes:
        vis = n.get("visible_in", "")
        if vis:
            cams = camera_re.findall(vis.lower())
            visibility.extend(cams)
    stats["camera_counts"] = Counter(visibility)

    # Ego edges
    ego_edges = graph.get("ego_edges", [])
    stats["n_ego_edges"] = len(ego_edges)

    effect_counts = Counter()
    active_effect_counts = Counter()
    causal_status_counts = Counter()
    affected_state_counts = Counter()
    n_active = 0
    n_dormant = 0
    effect_sets_per_graph = set()

    for e in ego_edges:
        etype = e.get("effect_type", "UNKNOWN")
        effect_counts[etype] += 1
        effect_sets_per_graph.add(etype)

        status = e.get("causal_status", "unknown")
        causal_status_counts[status] += 1

        if e.get("active", False):
            n_active += 1
            active_effect_counts[etype] += 1
        else:
            n_dormant += 1

        for s in e.get("affected_state", []):
            affected_state_counts[s] += 1

    stats["n_active_edges"] = n_active
    stats["n_dormant_edges"] = n_dormant
    stats["effect_counts"] = effect_counts
    stats["active_effect_counts"] = active_effect_counts
    stats["causal_status_counts"] = causal_status_counts
    stats["affected_state_counts"] = affected_state_counts
    stats["effect_set"] = frozenset(effect_sets_per_graph)

    # Distractors
    distractors = graph.get("distractors", [])
    stats["n_distractors"] = len(distractors)
    stats["distractor_type_counts"] = Counter(d.get("type", "UNKNOWN") for d in distractors)
    stats["distractor_category_counts"] = Counter(d.get("category", "unknown") for d in distractors)

    # Inter-node edges
    inter_edges = graph.get("inter_node_edges", [])
    stats["n_inter_edges"] = len(inter_edges)
    def _safe_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0
    stats["graph_depth"] = max((_safe_int(e.get("layer")) for e in inter_edges), default=0)

    # Graph structure types (from _extract_active)
    active_sgs = _extract_active(graph)
    stats["structure_counts"] = {k: len(v) for k, v in active_sgs.items()}
    stats["structure_present"] = {k: len(v) > 0 for k, v in active_sgs.items()}

    # Subgraph signatures
    sigs = []
    for sg_type, sg_list in active_sgs.items():
        for sg in sg_list:
            sig = _equiv_sig(sg_type, sg)
            if sig is not None:
                sigs.append(sig)
    stats["subgraph_sigs"] = sigs

    # Upstream forks (confounders not caught by _extract_active)
    upstream = detect_upstream_forks(graph)
    stats["n_upstream_forks"] = len(upstream)
    stats["has_upstream_fork"] = len(upstream) > 0

    # Branching factor
    children, parents, _ = _build_adjacency_active(graph)
    if children:
        stats["branching_factor"] = sum(len(v) for v in children.values()) / len(children)
    else:
        stats["branching_factor"] = 0.0

    # Richness
    stats["richness"] = (
        stats["n_nodes"] * 2
        + n_active * 3
        + stats["n_inter_edges"] * 4
        + n_dormant * 1
    )

    return stats


def check_qa_coverage(key: str, graph_dir: Path) -> dict[str, bool]:
    """Check which QA files exist in sample subdirectory."""
    subdir = graph_dir / key
    result = {}
    for qa in ("active_qa", "dormant_qa", "distractor_qa"):
        result[qa] = (subdir / f"{qa}.json").exists()
    result["has_subdir"] = subdir.is_dir()
    return result


# ── Aggregation ───────────────────────────────────────────────────────────────

class AggregateStats:
    """Accumulates statistics across all samples."""

    def __init__(self):
        self.total_discovered = 0
        self.total_loaded = 0
        self.load_failures = []

        # 1. Source distribution
        self.source_counts = Counter()

        # 2. Topology histograms
        self.node_counts = []
        self.ego_edge_counts = []
        self.inter_edge_counts = []
        self.depth_values = []

        # 3. Scene description
        self.road_keyword_counts = Counter()
        self.nav_intent_counts = Counter()
        self.road_geometry_raw = Counter()  # full text, for top-N

        # 4. Ego behavior
        self.speed_counts = Counter()
        self.accel_counts = Counter()
        self.heading_counts = Counter()
        self.behavior_match_counts = Counter()

        # 5. Active / dormant / distractor
        self.active_edge_counts = []
        self.dormant_edge_counts = []
        self.distractor_counts = []

        # 6. Graph structure types
        self.structure_instance_counts = Counter()  # total instances across all graphs
        self.structure_sample_counts = Counter()    # samples that have >= 1 of each type
        self.subgraph_sig_counts = Counter()
        self.upstream_fork_count = 0
        self.samples_with_upstream_fork = 0

        # 7. Node type distribution
        self.node_type_counts = Counter()

        # 8. Node category distribution
        self.node_category_counts = Counter()

        # 9. Edge effect type
        self.effect_counts = Counter()
        self.active_effect_counts = Counter()
        self.effect_cooccurrence = Counter()  # frozenset pairs

        # 10. Causal status
        self.causal_status_counts = Counter()

        # 11. Position
        self.position_counts = Counter()

        # 12. Motion
        self.total_static = 0
        self.total_dynamic = 0

        # 13. Affected state
        self.affected_state_counts = Counter()

        # 14. Complexity
        self.richness_values = []
        self.branching_factors = []

        # 15. Per-scene
        self.scene_sample_counts = Counter()

        # 16. QA coverage
        self.qa_counts = Counter()
        self.samples_with_subdir = 0

        # Extra: camera visibility
        self.camera_counts = Counter()

        # Extra: distractor breakdown
        self.distractor_type_counts = Counter()
        self.distractor_category_counts = Counter()

    def add(self, key: str, source: str, gs: dict, qa: dict):
        m = re.match(r"nuscenes-scene-(\d+)_SAMPLED_(\d+)", key)
        scene_num = int(m.group(1)) if m else -1

        self.total_loaded += 1
        self.source_counts[source] += 1
        self.scene_sample_counts[scene_num] += 1

        # 2. Topology
        self.node_counts.append(gs["n_nodes"])
        self.ego_edge_counts.append(gs["n_ego_edges"])
        self.inter_edge_counts.append(gs["n_inter_edges"])
        self.depth_values.append(gs["graph_depth"])

        # 3. Scene
        for kw in gs["road_keywords"]:
            self.road_keyword_counts[kw] += 1
        intent = gs["navigation_intent"].strip().lower()
        # Normalize to short form
        if intent:
            self.nav_intent_counts[intent] += 1
        geom = gs["road_geometry"].strip()
        if geom:
            self.road_geometry_raw[geom[:80]] += 1

        # 4. Ego behavior
        self.speed_counts[gs["speed"]] += 1
        self.accel_counts[gs["acceleration"]] += 1
        self.heading_counts[gs["heading"]] += 1
        self.behavior_match_counts[gs["behavior_matches_nav"]] += 1

        # 5. Active/dormant/distractor
        self.active_edge_counts.append(gs["n_active_edges"])
        self.dormant_edge_counts.append(gs["n_dormant_edges"])
        self.distractor_counts.append(gs["n_distractors"])

        # 6. Structure types
        for stype, count in gs["structure_counts"].items():
            self.structure_instance_counts[stype] += count
            if count > 0:
                self.structure_sample_counts[stype] += 1
        for sig in gs["subgraph_sigs"]:
            self.subgraph_sig_counts[sig] += 1
        self.upstream_fork_count += gs["n_upstream_forks"]
        if gs["has_upstream_fork"]:
            self.samples_with_upstream_fork += 1

        # 7. Node types
        self.node_type_counts += gs["node_type_counts"]

        # 8. Node categories
        self.node_category_counts += gs["node_category_counts"]

        # 9. Effect types
        self.effect_counts += gs["effect_counts"]
        self.active_effect_counts += gs["active_effect_counts"]
        # Co-occurrence: pairs of effect types present in same graph
        effects_present = sorted(gs["effect_set"])
        for i, e1 in enumerate(effects_present):
            for e2 in effects_present[i + 1:]:
                self.effect_cooccurrence[f"{e1} + {e2}"] += 1

        # 10. Causal status
        self.causal_status_counts += gs["causal_status_counts"]

        # 11. Position
        self.position_counts += gs["position_counts"]

        # 12. Motion
        self.total_static += gs["static_nodes"]
        self.total_dynamic += gs["dynamic_nodes"]

        # 13. Affected state
        self.affected_state_counts += gs["affected_state_counts"]

        # 14. Complexity
        self.richness_values.append(gs["richness"])
        self.branching_factors.append(gs["branching_factor"])

        # 16. QA
        if qa["has_subdir"]:
            self.samples_with_subdir += 1
        for qa_type in ("active_qa", "dormant_qa", "distractor_qa"):
            if qa[qa_type]:
                self.qa_counts[qa_type] += 1

        # Extra
        self.camera_counts += gs["camera_counts"]
        self.distractor_type_counts += gs["distractor_type_counts"]
        self.distractor_category_counts += gs["distractor_category_counts"]


# ── Markdown formatting helpers ───────────────────────────────────────────────

def _pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


def md_counter_table(counter: Counter, total: int | None = None,
                     top_n: int = 25, headers: tuple = ("Value", "Count", "%")) -> str:
    """Format a Counter as a markdown table."""
    if total is None:
        total = sum(counter.values())
    if not counter:
        return "_No data_\n"

    lines = [f"| {headers[0]} | {headers[1]} | {headers[2]} |",
             "|---|---:|---:|"]
    for val, cnt in counter.most_common(top_n):
        lines.append(f"| {val} | {cnt} | {_pct(cnt, total)} |")
    if len(counter) > top_n:
        lines.append(f"| _... {len(counter) - top_n} more_ | | |")
    return "\n".join(lines) + "\n"


def md_histogram_row(values: list, label: str) -> str:
    """One row of a histogram summary table."""
    if not values:
        return f"| {label} | - | - | - | - | - | - |"
    mn = min(values)
    mx = max(values)
    mean = statistics.mean(values)
    med = statistics.median(values)
    p75 = sorted(values)[int(len(values) * 0.75)] if len(values) > 1 else values[0]
    p90 = sorted(values)[int(len(values) * 0.90)] if len(values) > 1 else values[0]
    return f"| {label} | {mn} | {mx} | {mean:.1f} | {med:.0f} | {p75} | {p90} |"


def md_histogram_table(rows: list[tuple[str, list]]) -> str:
    """Full histogram summary table from [(label, values), ...]."""
    lines = ["| Metric | Min | Max | Mean | Median | P75 | P90 |",
             "|--------|----:|----:|-----:|-------:|----:|----:|"]
    for label, values in rows:
        lines.append(md_histogram_row(values, label))
    return "\n".join(lines) + "\n"


def md_histogram_table_float(rows: list[tuple[str, list]]) -> str:
    """Histogram summary table for float values."""
    lines = ["| Metric | Min | Max | Mean | Median | P75 | P90 |",
             "|--------|----:|----:|-----:|-------:|----:|----:|"]
    for label, values in rows:
        if not values:
            lines.append(f"| {label} | - | - | - | - | - | - |")
            continue
        sv = sorted(values)
        mn, mx = sv[0], sv[-1]
        mean = statistics.mean(values)
        med = statistics.median(values)
        p75 = sv[int(len(sv) * 0.75)]
        p90 = sv[int(len(sv) * 0.90)]
        lines.append(f"| {label} | {mn:.2f} | {mx:.2f} | {mean:.2f} | {med:.2f} | {p75:.2f} | {p90:.2f} |")
    return "\n".join(lines) + "\n"


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(agg: AggregateStats, graph_dir: str) -> str:
    """Generate full markdown report."""
    n = agg.total_loaded
    lines = []

    def section(num, title):
        lines.append(f"\n## {num}. {title}\n")

    lines.append("# Causal Graph Dataset Statistics\n")
    lines.append(f"> **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    lines.append(f"> **Source:** `{graph_dir}`  ")
    lines.append(f"> **Samples discovered:** {agg.total_discovered}  ")
    lines.append(f"> **Samples loaded:** {agg.total_loaded}  ")
    if agg.load_failures:
        lines.append(f"> **Load failures:** {len(agg.load_failures)}  ")
    lines.append("")

    # ── 1. File Source Distribution ──
    section(1, "File Source Distribution")
    lines.append("Priority order: verified > edited > pruned. Each sample counted once.\n")
    lines.append(md_counter_table(agg.source_counts, total=n,
                                  headers=("Source", "Count", "%")))

    # ── 2. Topology ──
    section(2, "Topology")
    lines.append(md_histogram_table([
        ("Causal nodes (excl. ego)", agg.node_counts),
        ("Ego edges", agg.ego_edge_counts),
        ("Inter-node edges", agg.inter_edge_counts),
        ("Graph depth (max layer)", agg.depth_values),
    ]))

    # Distribution of node counts
    node_dist = Counter(agg.node_counts)
    lines.append("\n### Node count distribution\n")
    lines.append(md_counter_table(node_dist, total=n,
                                  headers=("Nodes", "Samples", "%")))

    # ── 3. Scene Description ──
    section(3, "Scene Description")
    lines.append("### Road geometry keywords\n")
    lines.append(md_counter_table(agg.road_keyword_counts, total=n,
                                  headers=("Keyword", "Samples", "%")))

    lines.append("\n### Navigation intent (top 25)\n")
    lines.append(md_counter_table(agg.nav_intent_counts, total=n, top_n=25,
                                  headers=("Intent", "Count", "%")))

    # ── 4. Ego Behavior ──
    section(4, "Ego Behavior")
    lines.append("### Speed\n")
    lines.append(md_counter_table(agg.speed_counts, total=n,
                                  headers=("Speed", "Count", "%")))
    lines.append("\n### Acceleration\n")
    lines.append(md_counter_table(agg.accel_counts, total=n,
                                  headers=("Acceleration", "Count", "%")))
    lines.append("\n### Heading change\n")
    lines.append(md_counter_table(agg.heading_counts, total=n,
                                  headers=("Heading", "Count", "%")))
    lines.append("\n### Behavior matches navigation intent\n")
    lines.append(md_counter_table(agg.behavior_match_counts, total=n,
                                  headers=("Matches", "Count", "%")))

    # ── 5. Active / Dormant / Distractor ──
    section(5, "Active / Dormant / Distractor Counts")
    lines.append(md_histogram_table([
        ("Active ego-edges per graph", agg.active_edge_counts),
        ("Dormant ego-edges per graph", agg.dormant_edge_counts),
        ("Distractor nodes per graph", agg.distractor_counts),
    ]))

    # ── 6. Graph Structure Types ──
    section(6, "Graph Structure Types")
    lines.append("Detected from active edges only. Fork = observed common cause (confounder proxy).\n")
    struct_lines = ["| Type | Samples with type | % of samples | Total instances |",
                    "|------|------------------:|-------------:|----------------:|"]
    for stype in ["direct", "chain", "fork", "collider"]:
        sc = agg.structure_sample_counts[stype]
        ic = agg.structure_instance_counts[stype]
        struct_lines.append(f"| {stype} | {sc} | {_pct(sc, n)} | {ic} |")
    struct_lines.append(f"| upstream_fork (confounder) | {agg.samples_with_upstream_fork} "
                        f"| {_pct(agg.samples_with_upstream_fork, n)} | {agg.upstream_fork_count} |")
    lines.append("\n".join(struct_lines) + "\n")

    lines.append("\n### Top 20 subgraph signatures\n")
    lines.append("Canonical signatures (node-identity-agnostic).\n")
    top_sigs = agg.subgraph_sig_counts.most_common(20)
    sig_lines = ["| # | Signature | Count |",
                 "|--:|-----------|------:|"]
    for i, (sig, cnt) in enumerate(top_sigs, 1):
        sig_str = str(sig)
        sig_lines.append(f"| {i} | `{sig_str}` | {cnt} |")
    lines.append("\n".join(sig_lines) + "\n")

    # ── 7. Node Type Distribution ──
    section(7, "Node Type Distribution")
    total_nodes = sum(agg.node_type_counts.values())
    lines.append(f"Total causal nodes (excl. ego): **{total_nodes}**\n\n")
    lines.append(md_counter_table(agg.node_type_counts, total=total_nodes,
                                  headers=("Type", "Count", "%")))

    # ── 8. Node Category Distribution ──
    section(8, "Node Category Distribution (top 30)")
    lines.append(md_counter_table(agg.node_category_counts, total=total_nodes, top_n=30,
                                  headers=("Category", "Count", "%")))

    # ── 9. Edge Effect Type Distribution ──
    section(9, "Edge Effect Type Distribution")
    total_edges = sum(agg.effect_counts.values())
    total_active = sum(agg.active_effect_counts.values())
    lines.append(f"Total ego-edges: **{total_edges}** (active: **{total_active}**, "
                 f"dormant: **{total_edges - total_active}**)\n\n")
    eff_lines = ["| Effect Type | Total | % | Active | Active % |",
                 "|-------------|------:|--:|-------:|---------:|"]
    for eff in EFFECT_TYPES:
        tc = agg.effect_counts.get(eff, 0)
        ac = agg.active_effect_counts.get(eff, 0)
        eff_lines.append(f"| {eff} | {tc} | {_pct(tc, total_edges)} | {ac} | {_pct(ac, total_active)} |")
    # Any non-standard effects
    for eff, cnt in agg.effect_counts.most_common():
        if eff not in EFFECT_TYPES:
            ac = agg.active_effect_counts.get(eff, 0)
            eff_lines.append(f"| {eff} _(non-standard)_ | {cnt} | {_pct(cnt, total_edges)} | {ac} | {_pct(ac, total_active)} |")
    lines.append("\n".join(eff_lines) + "\n")

    lines.append("\n### Effect type co-occurrence (top 15)\n")
    lines.append("Pairs of effect types appearing in the same graph.\n\n")
    lines.append(md_counter_table(agg.effect_cooccurrence, top_n=15,
                                  headers=("Effect Pair", "Graphs", "%")))

    # ── 10. Causal Status Distribution ──
    section(10, "Causal Status Distribution")
    total_status = sum(agg.causal_status_counts.values())
    lines.append(md_counter_table(agg.causal_status_counts, total=total_status,
                                  headers=("Status", "Count", "%")))

    # ── 11. Position Relative to Ego ──
    section(11, "Position Relative to Ego")
    lines.append("Free-text positions bucketed as `distance_direction`.\n\n")
    lines.append(md_counter_table(agg.position_counts, top_n=25,
                                  headers=("Position Bucket", "Count", "%")))

    # ── 12. Motion Characteristics ──
    section(12, "Motion Characteristics")
    total_motion = agg.total_static + agg.total_dynamic
    motion_lines = [
        "| Motion | Count | % |",
        "|--------|------:|--:|",
        f"| Static | {agg.total_static} | {_pct(agg.total_static, total_motion)} |",
        f"| Dynamic | {agg.total_dynamic} | {_pct(agg.total_dynamic, total_motion)} |",
    ]
    lines.append("\n".join(motion_lines) + "\n")

    # ── 13. Affected State Distribution ──
    section(13, "Affected Ego State Distribution")
    lines.append("Which ego states are affected by causal edges (from `affected_state` field).\n\n")
    total_affected = sum(agg.affected_state_counts.values())
    lines.append(md_counter_table(agg.affected_state_counts, total=total_affected,
                                  headers=("State", "Count", "%")))

    # ── 14. Graph Complexity ──
    section(14, "Graph Complexity")
    lines.append(md_histogram_table_float([
        ("Richness score", agg.richness_values),
        ("Branching factor", agg.branching_factors),
    ]))

    # ── 15. Per-Scene Aggregates ──
    section(15, "Per-Scene Aggregates")
    scene_counts_list = list(agg.scene_sample_counts.values())
    lines.append(f"**Unique scenes:** {len(agg.scene_sample_counts)}\n\n")
    lines.append(md_histogram_table([
        ("Samples per scene", scene_counts_list),
    ]))

    scene_dist = Counter(scene_counts_list)
    lines.append("\n### Samples-per-scene distribution\n")
    lines.append(md_counter_table(scene_dist, total=len(agg.scene_sample_counts),
                                  headers=("Samples in scene", "Scenes", "%")))

    # ── 16. QA Coverage ──
    section(16, "QA Coverage")
    lines.append(f"Samples with QA subdirectory: **{agg.samples_with_subdir}** / {n}\n")
    qa_lines = ["| QA Type | Samples | % of those with subdir |",
                "|---------|--------:|-----------------------:|"]
    for qa_type in ("active_qa", "dormant_qa", "distractor_qa"):
        cnt = agg.qa_counts[qa_type]
        qa_lines.append(f"| {qa_type} | {cnt} | {_pct(cnt, agg.samples_with_subdir)} |")
    lines.append("\n".join(qa_lines) + "\n")

    # ── 17. Camera Visibility ──
    section(17, "Camera Visibility Distribution")
    lines.append("Which cameras observe causal nodes.\n\n")
    lines.append(md_counter_table(agg.camera_counts, top_n=20,
                                  headers=("Camera", "Mentions", "%")))

    # ── 18. Distractor Breakdown ──
    section(18, "Distractor Node Breakdown")
    total_dist = sum(agg.distractor_type_counts.values())
    lines.append(f"Total distractor nodes: **{total_dist}**\n\n")
    lines.append("### By type\n")
    lines.append(md_counter_table(agg.distractor_type_counts, total=total_dist,
                                  headers=("Type", "Count", "%")))
    lines.append("\n### By category (top 20)\n")
    lines.append(md_counter_table(agg.distractor_category_counts, total=total_dist, top_n=20,
                                  headers=("Category", "Count", "%")))

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute causal graph dataset statistics")
    parser.add_argument("--graph-dir",
                        default="/spiral_hdd_3/workspace/navvrat/Neurips/outputs/nuscenes_full_edit_backup",
                        help="Directory with graph JSON files")
    parser.add_argument("--output", required=True,
                        help="Output path for the markdown report")
    parser.add_argument("--output-json", default=None,
                        help="Optional output path for raw JSON stats")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample info to stdout")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N samples (for testing)")
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)

    # Discover
    print("Discovering samples...")
    all_keys = discover_samples(graph_dir)
    print(f"  Found {len(all_keys)} unique sample keys")

    if args.limit:
        all_keys = all_keys[:args.limit]
        print(f"  Limited to first {args.limit}")

    # Process
    agg = AggregateStats()
    agg.total_discovered = len(all_keys) if not args.limit else agg.total_discovered

    # Count total discovered before limit
    if not args.limit:
        agg.total_discovered = len(all_keys)
    else:
        agg.total_discovered = len(discover_samples(graph_dir))

    for i, key in enumerate(all_keys):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(all_keys)}...")

        graph, source = load_graph(key, graph_dir)
        if graph is None:
            agg.load_failures.append(key)
            if args.verbose:
                print(f"    SKIP {key}: no graph file found")
            continue

        gs = extract_graph_stats(graph)
        qa = check_qa_coverage(key, graph_dir)
        agg.add(key, source, gs, qa)

        if args.verbose:
            print(f"    {key} [{source}] nodes={gs['n_nodes']} "
                  f"edges={gs['n_ego_edges']} active={gs['n_active_edges']} "
                  f"dormant={gs['n_dormant_edges']} dist={gs['n_distractors']}")

    print(f"  Loaded {agg.total_loaded}, failures {len(agg.load_failures)}")
    if agg.load_failures:
        print(f"  Failed keys: {agg.load_failures[:10]}...")

    # Generate report
    print("Generating report...")
    report = generate_report(agg, args.graph_dir)

    output_path = Path(args.output)
    output_path.write_text(report)
    print(f"Report written to: {output_path}")

    # Optional JSON dump
    if args.output_json:
        json_data = {
            "total_discovered": agg.total_discovered,
            "total_loaded": agg.total_loaded,
            "load_failures": agg.load_failures,
            "source_counts": dict(agg.source_counts),
            "node_count_hist": agg.node_counts,
            "ego_edge_count_hist": agg.ego_edge_counts,
            "inter_edge_count_hist": agg.inter_edge_counts,
            "depth_hist": agg.depth_values,
            "road_keyword_counts": dict(agg.road_keyword_counts),
            "nav_intent_counts": dict(agg.nav_intent_counts),
            "speed_counts": dict(agg.speed_counts),
            "accel_counts": dict(agg.accel_counts),
            "heading_counts": dict(agg.heading_counts),
            "behavior_match_counts": {str(k): v for k, v in agg.behavior_match_counts.items()},
            "active_edge_hist": agg.active_edge_counts,
            "dormant_edge_hist": agg.dormant_edge_counts,
            "distractor_count_hist": agg.distractor_counts,
            "structure_instance_counts": dict(agg.structure_instance_counts),
            "structure_sample_counts": dict(agg.structure_sample_counts),
            "upstream_fork_samples": agg.samples_with_upstream_fork,
            "node_type_counts": dict(agg.node_type_counts),
            "node_category_counts": dict(agg.node_category_counts),
            "effect_counts": dict(agg.effect_counts),
            "active_effect_counts": dict(agg.active_effect_counts),
            "effect_cooccurrence": dict(agg.effect_cooccurrence),
            "causal_status_counts": dict(agg.causal_status_counts),
            "position_counts": dict(agg.position_counts),
            "motion": {"static": agg.total_static, "dynamic": agg.total_dynamic},
            "affected_state_counts": dict(agg.affected_state_counts),
            "richness_hist": agg.richness_values,
            "branching_factor_hist": agg.branching_factors,
            "scene_sample_counts": {str(k): v for k, v in agg.scene_sample_counts.items()},
            "qa_counts": dict(agg.qa_counts),
            "samples_with_subdir": agg.samples_with_subdir,
            "camera_counts": dict(agg.camera_counts),
            "distractor_type_counts": dict(agg.distractor_type_counts),
            "distractor_category_counts": dict(agg.distractor_category_counts),
        }
        json_path = Path(args.output_json)
        json_path.write_text(json.dumps(json_data, indent=2))
        print(f"JSON stats written to: {json_path}")


if __name__ == "__main__":
    main()
