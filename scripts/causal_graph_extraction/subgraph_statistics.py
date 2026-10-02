"""
Aggregate statistics over causal subgraphs extracted across a directory of scenes.

Reuses extract_subgraphs() + _resolve_graph_path() / _extract_scene_ids() from
scripts.causal_graph_extraction.extract_subgraphs so canonical definitions stay
in one place.

Produces a comprehensive markdown report covering:
  - Prevalence of each subgraph type (what fraction of scenes contain one).
  - Per-scene count distributions.
  - Structural preconditions (how the graph shape relates to which subgraphs
    get extracted — answers "when does each type occur").
  - Node-type participation (which actor categories play each role).
  - Co-occurrence of subgraph types.
  - Direct role / effect_type / causal_status breakdowns.
  - Licensed QA job counts per (structure, rung, qa_type).
  - Top / representative scenes per type.

Usage:
    python scripts/graph_filtering/subgraph_statistics.py \
        --graph-dir /spiral_hdd_3/workspace/navvrat/Neurips/outputs/nuscenes_full_final \
        --output subgraph_stats_report.md \
        [--output-json subgraph_stats.json]
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median

SAMPLE_ID_RE = re.compile(r"_SAMPLED_\d+$")
BASE_SCENE_RE = re.compile(r"_SAMPLED_\d+$")
SUFFIX_ORDER = ("_verified", "_fixed", "_naren_edited", "_pruned")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.causal_graph_extraction.extract_subgraphs import (  # noqa: E402
    QA_TEMPLATES,
    _extract_scene_ids,
    _resolve_graph_path,
    extract_subgraphs,
    licensed_qas,
)


SUBGRAPH_KEYS = [
    "direct",
    "chain",
    "collider",
    "confounding_observed",
    "confounding_unobserved",
]

NODE_TYPES = ["EGO", "OTHER_AGENT", "TRAFFIC_CONTROL", "ENVIRONMENT", "ROAD_OBSTACLE"]


def _node_type_map(graph):
    """id -> type (closed-set). Missing/unknown becomes 'UNKNOWN'."""
    return {n["id"]: n.get("type", "UNKNOWN") for n in graph.get("nodes", [])}


def ntype_lookup(nodes_list, node_id):
    for node in nodes_list:
        if node.get("id") == node_id:
            return node.get("type", "UNKNOWN")
    return "UNKNOWN"


def _process_scene(args):
    graphs_dir_str, scene_id = args
    graphs_dir = Path(graphs_dir_str)
    path = _resolve_graph_path(graphs_dir, scene_id)
    if path is None:
        return scene_id, None, "no_graph"
    try:
        graph = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return scene_id, None, "invalid_json"

    sg = extract_subgraphs(graph, active_only=True, include_dormant_in_confounding=False)

    counts = {k: len(sg[k]) for k in SUBGRAPH_KEYS}
    counts["total"] = sum(counts[k] for k in SUBGRAPH_KEYS)

    # ---- Structural preconditions measured on the graph itself ----
    ego_edges = graph.get("ego_edges", [])
    inter_edges = graph.get("inter_node_edges", [])
    active_parents = {e["source"] for e in ego_edges if e.get("active", False)}
    dormant_parents = {e["source"] for e in ego_edges if not e.get("active", False)}
    n_affects = sum(1 for e in inter_edges if e.get("relation") == "AFFECTS")
    n_unobs = sum(1 for e in inter_edges if e.get("relation") == "CONFOUNDED_BY")
    nodes = graph.get("nodes", [])
    n_causal_nodes = sum(1 for node in nodes if node.get("type") != "EGO"
                         and node.get("id") != "ego")

    # ---- Dormant direct breakdown (Stage 3b substrate) ----
    dormant_ego_edges = [e for e in ego_edges if not e.get("active", False)]
    dormant_effects = Counter(
        (e.get("effect_type") or "UNKNOWN") for e in dormant_ego_edges
    )
    dormant_source_types = Counter(
        ntype_lookup(nodes, e["source"]) for e in dormant_ego_edges
    )

    # ---- Distractor nodes (Stage 3c substrate) ----
    # Distractors are stored in a dedicated top-level field, separate from
    # `nodes` — they're candidates that were considered but *rejected* as
    # causes of ego behavior during graph construction.
    distractor_nodes = graph.get("distractors", []) or []
    distractor_types = Counter(
        (node.get("type") if isinstance(node, dict) else None) or "UNKNOWN"
        for node in distractor_nodes
    )

    # ---- Node type participation per structure ----
    ntype = _node_type_map(graph)

    def typ(nid):
        return ntype.get(nid, "UNKNOWN")

    role_counts = Counter(d["confounding_role"] for d in sg["direct"])
    effect_counts = Counter((d.get("effect_type") or "UNKNOWN") for d in sg["direct"])
    status_counts = Counter((d.get("causal_status") or "UNKNOWN") for d in sg["direct"])

    # node type per slot, per structure
    types_in_struct = defaultdict(Counter)
    for d in sg["direct"]:
        types_in_struct[f"direct|{d['confounding_role']}|source"][typ(d["source"])] += 1
    for c in sg["chain"]:
        types_in_struct["chain|distal"][typ(c["distal"])] += 1
        types_in_struct["chain|mediator"][typ(c["mediator"])] += 1
    for c in sg["collider"]:
        for p in c["parents"]:
            types_in_struct["collider|parent"][typ(p)] += 1
    for c in sg["confounding_observed"]:
        types_in_struct["confounding_observed|confounder"][typ(c["confounder"])] += 1
        types_in_struct["confounding_observed|mediator"][typ(c["mediator"])] += 1
    for c in sg["confounding_unobserved"]:
        for p in c["nodes"]:
            types_in_struct["confounding_unobserved|node"][typ(p)] += 1

    jobs = licensed_qas(sg)
    qa_counts = Counter((sg_type, rung, qa) for sg_type, _, rung, qa in jobs)

    source_suffix = _source_suffix_from_filename(path.name, scene_id)

    return scene_id, {
        "counts": counts,
        "roles": dict(role_counts),
        "effects": dict(effect_counts),
        "statuses": dict(status_counts),
        "types_in_struct": {k: dict(v) for k, v in types_in_struct.items()},
        "preconditions": {
            "n_active_parents": len(active_parents),
            "n_dormant_parents": len(dormant_parents),
            "n_affects": n_affects,
            "n_unobs": n_unobs,
            "n_nodes": len(nodes),
            "n_causal_nodes": n_causal_nodes,
            "n_distractors": len(distractor_nodes),
        },
        "dormant_effects": dict(dormant_effects),
        "dormant_source_types": dict(dormant_source_types),
        "distractor_types": dict(distractor_types),
        "qa_counts": {f"{t}|{r}|{q}": n for (t, r, q), n in qa_counts.items()},
        "graph_file": path.name,
        "source_suffix": source_suffix,
    }, None


def _source_suffix_from_filename(filename: str, scene_id: str) -> str:
    """Derive the fallback level ('_verified' / '_fixed' / ...) from the filename."""
    stem = filename[:-5] if filename.endswith(".json") else filename
    if stem.startswith(scene_id):
        tail = stem[len(scene_id):]
        if tail in SUFFIX_ORDER:
            return tail
    return "_other"


def _base_scene(scene_id: str) -> str:
    """Strip the `_SAMPLED_N` suffix to get the true scene ID."""
    return BASE_SCENE_RE.sub("", scene_id)


def aggregate(graph_dir: Path, workers: int):
    scene_ids = [sid for sid in _extract_scene_ids(graph_dir)
                 if SAMPLE_ID_RE.search(sid)]
    tasks = [(str(graph_dir), sid) for sid in scene_ids]

    per_scene = {}
    skipped = Counter()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_scene, t): t for t in tasks}
        for fut in as_completed(futures):
            sid, result, err = fut.result()
            if err:
                skipped[err] += 1
                continue
            per_scene[sid] = result

    return per_scene, skipped, len(tasks)


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _hist(values, label):
    if not values:
        return [f"_{label}: no data_", ""]
    labels = ["0", "1", "2", "3", "4", "5-9", "10-19", "20-49", "50+"]
    buckets = [0] * len(labels)
    for v in values:
        if v == 0:
            buckets[0] += 1
        elif v == 1:
            buckets[1] += 1
        elif v == 2:
            buckets[2] += 1
        elif v == 3:
            buckets[3] += 1
        elif v == 4:
            buckets[4] += 1
        elif v < 10:
            buckets[5] += 1
        elif v < 20:
            buckets[6] += 1
        elif v < 50:
            buckets[7] += 1
        else:
            buckets[8] += 1
    total = sum(buckets)
    out = [
        f"**{label}** — n={total}, mean={mean(values):.2f}, median={median(values):.1f}, max={max(values)}",
        "",
        "| bucket | count | share |",
        "|---|---|---|",
    ]
    for lab, n in zip(labels, buckets):
        share = 100 * n / total if total else 0
        out.append(f"| {lab} | {n} | {share:.1f}% |")
    out.append("")
    return out


def make_report(per_scene, skipped, total_scenes, graph_dir: Path):
    n = len(per_scene)
    lines = []
    lines.append(f"# Subgraph statistics report\n")
    lines.append(f"- Graph dir: `{graph_dir}`")
    lines.append(f"- Scenes analyzed: **{n}** / {total_scenes}")
    if skipped:
        lines.append(f"- Skipped: {dict(skipped)}")
    lines.append("")

    # =====================================================================
    # Definitions — what triggers each subgraph type
    # =====================================================================
    lines.append("## What triggers each subgraph type\n")
    lines.append(
        "Extraction rules (from `extract_subgraphs.py`), paraphrased:\n\n"
        "- **Direct** — every *active* ego-edge `X → ego`. Extracted\n"
        "  **independently of higher-order structures**: if `X` is also a\n"
        "  collider parent, a confounder, or a mediator, that same ego-edge\n"
        "  *still* produces a Direct subgraph (role-annotated as\n"
        "  `independent`/`mediator`/`confounder`). So Direct counts are a\n"
        "  superset that overlaps with Collider/Confounding.\n"
        "- **Chain** `A → B → ego` — an inter-node `AFFECTS` edge where `A` is\n"
        "  NOT itself an ego-parent and `B` IS an active ego-parent. (If `A`\n"
        "  were also an ego-parent, the pair upgrades to Confounding.)\n"
        "- **Collider** — two active ego-parents with no inter-node edge between\n"
        "  them and neither participating as confounder/mediator/chain-mediator\n"
        "  in some other triangle.\n"
        "- **Confounding (observed)** — two ego-parents `A,B` AND a directed\n"
        "  `A → B` (or `B → A`) inter-node edge: the upstream one is the\n"
        "  confounder, the downstream one is the mediator.\n"
        "- **Confounding (unobserved)** — two ego-parents with a\n"
        "  `CONFOUNDED_BY` inter-node edge between them.\n"
    )
    lines.append(
        "Subsumption / priority rules:\n\n"
        "- **Confounding subsumes Collider**: if a pair of ego-parents has a\n"
        "  directed inter-node edge (or an unobserved link), the pair is\n"
        "  classified as Confounding — it does **not** also show up as a\n"
        "  Collider.\n"
        "- **Confounding subsumes Chain** in the pair sense: a candidate\n"
        "  `A → B → ego` where `A` is *also* an ego-parent is extracted as\n"
        "  Confounding, not Chain. Chain requires `A` to be a *non-parent*\n"
        "  upstream cause.\n"
        "- **Busy-node propagation**: once a node is flagged as\n"
        "  confounder/mediator (observed or unobserved) or as a chain\n"
        "  mediator, it cannot also serve as a Collider parent in some other\n"
        "  pair. This prevents double-counting across structures.\n"
        "- **Direct overlaps with everything**: Direct subgraphs are extracted\n"
        "  per ego-edge, so every Collider parent, Confounder, Mediator, and\n"
        "  Chain mediator also produces a Direct record. Direct prevalence\n"
        "  therefore ≥ any single higher-order structure's prevalence.\n"
    )
    lines.append(
        "Key implications:\n\n"
        "- A scene can only produce confounding/collider if it has ≥ 2 active\n"
        "  ego-parents.\n"
        "- A scene can only produce a chain if there is at least one `AFFECTS`\n"
        "  edge whose source is *not* itself an ego-parent.\n"
        "- Adding inter-node edges trades Collider subgraphs for\n"
        "  Confounding/Chain subgraphs (while Direct counts stay the same).\n"
    )

    # =====================================================================
    # Prevalence
    # =====================================================================
    prev = Counter()
    totals = Counter()
    zero_total = 0
    for _, rec in per_scene.items():
        c = rec["counts"]
        if c["total"] == 0:
            zero_total += 1
        for key in SUBGRAPH_KEYS:
            totals[key] += c[key]
            if c[key] > 0:
                prev[key] += 1

    lines.append("## Prevalence: scenes containing ≥1 of each structure\n")
    lines.append("| structure | scenes with ≥1 | share | total instances | mean/scene |")
    lines.append("|---|---|---|---|---|")
    for key in SUBGRAPH_KEYS:
        share = 100 * prev[key] / n if n else 0
        per = totals[key] / n if n else 0
        lines.append(
            f"| {key} | {prev[key]} | {share:.1f}% | {totals[key]} | {per:.2f} |"
        )
    lines.append(f"\nScenes with 0 subgraphs of any kind: **{zero_total}** "
                 f"({100*zero_total/n:.1f}%)\n")

    # =====================================================================
    # Zero-subgraph decomposition
    # =====================================================================
    if zero_total:
        lines.append("### Why are scenes empty? (zero-subgraph decomposition)\n")
        lines.append(
            "Breakdown of the "
            f"**{zero_total}** scenes with no subgraphs:\n"
        )
        buckets = Counter()
        for rec in per_scene.values():
            if rec["counts"]["total"] > 0:
                continue
            p = rec["preconditions"]
            if p["n_causal_nodes"] == 0:
                buckets["ego-only graph (0 causal nodes)"] += 1
            elif p["n_active_parents"] == 0 and p["n_dormant_parents"] > 0:
                buckets["only dormant ego-parents"] += 1
            elif p["n_active_parents"] == 0:
                buckets["no ego-parents at all"] += 1
            elif p["n_active_parents"] == 1 and p["n_affects"] == 0:
                buckets["1 active parent, no inter-node edges"] += 1
            else:
                buckets["other"] += 1
        lines.append("| cause | scenes | share of empty |")
        lines.append("|---|---|---|")
        for cause, c in buckets.most_common():
            lines.append(f"| {cause} | {c} | {100*c/zero_total:.1f}% |")
        lines.append("")

    # =====================================================================
    # Source-suffix breakdown (which fallback level did each scene resolve to)
    # =====================================================================
    lines.append("## Source-suffix breakdown\n")
    lines.append(
        "Hierarchy: `_verified` > `_fixed` > `_naren_edited` > `_pruned`. "
        "Shows how many scenes resolved at each level — useful for tracking "
        "annotation progress.\n"
    )
    suffix_counts = Counter(rec["source_suffix"] for rec in per_scene.values())
    lines.append("| suffix | scenes | share |")
    lines.append("|---|---|---|")
    for suffix in SUFFIX_ORDER:
        c = suffix_counts.get(suffix, 0)
        share = 100 * c / n if n else 0
        lines.append(f"| `{suffix}` | {c} | {share:.1f}% |")
    other = suffix_counts.get("_other", 0)
    if other:
        lines.append(f"| other | {other} | {100*other/n:.1f}% |")
    lines.append("")

    # =====================================================================
    # Scene-level rollup (group SAMPLED variants under their base scene)
    # =====================================================================
    lines.append("## Scene-level rollup (grouped by base scene)\n")
    lines.append(
        "Each real scene typically has ~4 `_SAMPLED_N` variants. This view "
        "aggregates them so you can see dataset coverage at the scene level.\n"
    )
    by_base = defaultdict(list)
    for sid, rec in per_scene.items():
        by_base[_base_scene(sid)].append(rec)
    n_base = len(by_base)
    lines.append(f"- Base scenes covered: **{n_base}** "
                 f"(from {n} samples)")
    samples_per_scene = [len(v) for v in by_base.values()]
    if samples_per_scene:
        lines.append(
            f"- Samples per scene: mean={mean(samples_per_scene):.2f}, "
            f"median={median(samples_per_scene):.1f}, "
            f"min={min(samples_per_scene)}, max={max(samples_per_scene)}"
        )

    base_prev = Counter()
    for recs in by_base.values():
        has = {k: any(r["counts"][k] > 0 for r in recs) for k in SUBGRAPH_KEYS}
        for k, hv in has.items():
            if hv:
                base_prev[k] += 1
    lines.append("\n**Scenes with ≥1 of each structure (any sample):**\n")
    lines.append("| structure | scenes | share of base scenes |")
    lines.append("|---|---|---|")
    for key in SUBGRAPH_KEYS:
        share = 100 * base_prev[key] / n_base if n_base else 0
        lines.append(f"| {key} | {base_prev[key]} | {share:.1f}% |")
    lines.append("")

    # =====================================================================
    # Co-occurrence
    # =====================================================================
    lines.append("## Co-occurrence of structure types (scene-level)\n")
    lines.append("Entry `[i][j]` = scenes containing ≥1 of i AND ≥1 of j.\n")
    # Matrix
    header = "| | " + " | ".join(SUBGRAPH_KEYS) + " |"
    sep = "|---|" + "|".join(["---"] * len(SUBGRAPH_KEYS)) + "|"
    lines.append(header)
    lines.append(sep)
    for a in SUBGRAPH_KEYS:
        row = [a]
        for b in SUBGRAPH_KEYS:
            c = sum(
                1 for rec in per_scene.values()
                if rec["counts"][a] > 0 and rec["counts"][b] > 0
            )
            row.append(str(c))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Structure-presence profile: how many distinct structure types per scene
    type_presence = Counter(
        sum(1 for k in SUBGRAPH_KEYS if rec["counts"][k] > 0)
        for rec in per_scene.values()
    )
    lines.append("### Number of distinct structure types present per scene\n")
    lines.append("| distinct types | scenes | share |")
    lines.append("|---|---|---|")
    for k in range(len(SUBGRAPH_KEYS) + 1):
        c = type_presence.get(k, 0)
        lines.append(f"| {k} | {c} | {100*c/n:.1f}% |")
    lines.append("")

    # =====================================================================
    # Per-scene count distributions
    # =====================================================================
    lines.append("## Per-scene count distributions\n")
    for key in SUBGRAPH_KEYS:
        values = [rec["counts"][key] for rec in per_scene.values()]
        lines += _hist(values, key)
    lines += _hist([rec["counts"]["total"] for rec in per_scene.values()],
                   "total subgraphs per scene")

    # Scene-size context
    lines += _hist(
        [rec["preconditions"]["n_active_parents"] for rec in per_scene.values()],
        "active ego-parents per scene",
    )
    lines += _hist(
        [rec["preconditions"]["n_dormant_parents"] for rec in per_scene.values()],
        "dormant ego-parents per scene",
    )
    lines += _hist(
        [rec["preconditions"]["n_causal_nodes"] for rec in per_scene.values()],
        "causal (non-ego) nodes per scene",
    )
    lines += _hist(
        [rec["preconditions"]["n_affects"] for rec in per_scene.values()],
        "AFFECTS inter-node edges per scene",
    )
    lines += _hist(
        [rec["preconditions"]["n_unobs"] for rec in per_scene.values()],
        "CONFOUNDED_BY edges per scene",
    )

    # =====================================================================
    # Direct role / effect / status
    # =====================================================================
    lines.append("## Direct edges — role breakdown\n")
    role_totals = Counter()
    for rec in per_scene.values():
        for role, c in rec["roles"].items():
            role_totals[role] += c
    total_direct = sum(role_totals.values())
    lines.append("| role | count | share of direct edges |")
    lines.append("|---|---|---|")
    for role in ("independent", "mediator", "confounder"):
        c = role_totals.get(role, 0)
        share = 100 * c / total_direct if total_direct else 0
        lines.append(f"| {role} | {c} | {share:.1f}% |")
    lines.append("")

    lines.append("## Direct edges — effect_type distribution\n")
    eff_totals = Counter()
    for rec in per_scene.values():
        for e, c in rec["effects"].items():
            eff_totals[e] += c
    total_eff = sum(eff_totals.values())
    lines.append("| effect_type | count | share |")
    lines.append("|---|---|---|")
    for e, c in eff_totals.most_common():
        share = 100 * c / total_eff if total_eff else 0
        lines.append(f"| {e} | {c} | {share:.1f}% |")
    lines.append("")

    lines.append("## Direct edges — causal_status distribution\n")
    st_totals = Counter()
    for rec in per_scene.values():
        for s, c in rec["statuses"].items():
            st_totals[s] += c
    total_st = sum(st_totals.values())
    lines.append("| causal_status | count | share |")
    lines.append("|---|---|---|")
    for s, c in st_totals.most_common():
        share = 100 * c / total_st if total_st else 0
        lines.append(f"| {s} | {c} | {share:.1f}% |")
    lines.append("")

    # =====================================================================
    # Dormant direct edges (Stage 3b D_L QA substrate)
    # =====================================================================
    lines.append("## Dormant direct edges (Stage 3b substrate)\n")
    lines.append(
        "Dormant ego-edges `X ↛ ego` — causes present in the scene but not "
        "currently active. Each one licenses a dormant (D_L) QA: *what would "
        "happen if X activated?* For richer dormant analysis (dormant in "
        "chains, colliders, confounding), see "
        "`dormant_subgraph_statistics.py`.\n"
    )
    n_dormant_edges = sum(rec["preconditions"]["n_dormant_parents"]
                          for rec in per_scene.values())
    scenes_with_dormant = sum(
        1 for rec in per_scene.values()
        if rec["preconditions"]["n_dormant_parents"] > 0
    )
    lines.append(f"- Scenes with ≥1 dormant ego-edge: **{scenes_with_dormant}** "
                 f"({100*scenes_with_dormant/n:.1f}%)")
    lines.append(f"- Total dormant ego-edges: **{n_dormant_edges}** "
                 f"(mean {n_dormant_edges/n:.2f} per scene)\n")

    lines += _hist(
        [rec["preconditions"]["n_dormant_parents"] for rec in per_scene.values()],
        "dormant ego-edges per scene",
    )

    # effect_type of dormant edges
    de_tot = Counter()
    for rec in per_scene.values():
        for e, c in rec["dormant_effects"].items():
            de_tot[e] += c
    total_de = sum(de_tot.values())
    if total_de:
        lines.append("### Dormant direct — effect_type distribution\n")
        lines.append("| effect_type | count | share |")
        lines.append("|---|---|---|")
        for e, c in de_tot.most_common():
            share = 100 * c / total_de
            lines.append(f"| {e} | {c} | {share:.1f}% |")
        lines.append("")

    # source node type of dormant edges
    dt_tot = Counter()
    for rec in per_scene.values():
        for t, c in rec["dormant_source_types"].items():
            dt_tot[t] += c
    total_dt = sum(dt_tot.values())
    if total_dt:
        lines.append("### Dormant direct — source node type\n")
        lines.append("| node_type | count | share |")
        lines.append("|---|---|---|")
        for t in NODE_TYPES:
            c = dt_tot.get(t, 0)
            if c == 0:
                continue
            share = 100 * c / total_dt
            lines.append(f"| {t} | {c} | {share:.1f}% |")
        unk = sum(v for k, v in dt_tot.items() if k not in NODE_TYPES)
        if unk:
            lines.append(f"| UNKNOWN | {unk} | {100*unk/total_dt:.1f}% |")
        lines.append("")

    # =====================================================================
    # Distractor nodes (Stage 3c substrate)
    # =====================================================================
    lines.append("## Distractor nodes (Stage 3c substrate)\n")
    lines.append(
        "Distractors come from the graph's top-level `distractors` field — "
        "candidates that were considered and explicitly rejected as causes "
        "of ego behavior during graph construction (each carries a "
        "`rejection_reason`). Stage 3c uses them as plausible-but-wrong "
        "candidates in QAs (*is X a cause of ego's behavior?* — answer: no).\n"
    )
    n_distractors_tot = sum(rec["preconditions"]["n_distractors"]
                            for rec in per_scene.values())
    scenes_with_distractor = sum(
        1 for rec in per_scene.values()
        if rec["preconditions"]["n_distractors"] > 0
    )
    lines.append(f"- Scenes with ≥1 distractor: **{scenes_with_distractor}** "
                 f"({100*scenes_with_distractor/n:.1f}%)")
    lines.append(f"- Total distractor nodes: **{n_distractors_tot}** "
                 f"(mean {n_distractors_tot/n:.2f} per scene)\n")

    lines += _hist(
        [rec["preconditions"]["n_distractors"] for rec in per_scene.values()],
        "distractor nodes per scene",
    )

    # distractor node-type breakdown
    dist_tot = Counter()
    for rec in per_scene.values():
        for t, c in rec["distractor_types"].items():
            dist_tot[t] += c
    total_dist = sum(dist_tot.values())
    if total_dist:
        lines.append("### Distractor nodes — type distribution\n")
        lines.append("| node_type | count | share |")
        lines.append("|---|---|---|")
        for t in NODE_TYPES:
            c = dist_tot.get(t, 0)
            if c == 0:
                continue
            share = 100 * c / total_dist
            lines.append(f"| {t} | {c} | {share:.1f}% |")
        unk = sum(v for k, v in dist_tot.items() if k not in NODE_TYPES)
        if unk:
            lines.append(f"| UNKNOWN | {unk} | {100*unk/total_dist:.1f}% |")
        lines.append("")

    # =====================================================================
    # Node-type participation
    # =====================================================================
    lines.append("## Node-type participation per subgraph slot\n")
    lines.append(
        "Each row lists, for a given role-slot in a structure, which node types "
        "show up and how often. Reveals patterns like *pedestrians almost never "
        "act as confounders*, *traffic controls dominate chain-distal slots*, etc.\n"
    )
    # Aggregate across scenes
    slot_totals = defaultdict(Counter)
    for rec in per_scene.values():
        for slot, tcounts in rec["types_in_struct"].items():
            for t, c in tcounts.items():
                slot_totals[slot][t] += c

    slot_order = [
        "direct|independent|source",
        "direct|mediator|source",
        "direct|confounder|source",
        "chain|distal",
        "chain|mediator",
        "collider|parent",
        "confounding_observed|confounder",
        "confounding_observed|mediator",
        "confounding_unobserved|node",
    ]
    lines.append("| slot | total | " + " | ".join(NODE_TYPES) + " | UNKNOWN |")
    lines.append("|---|---|" + "|".join(["---"] * (len(NODE_TYPES) + 1)) + "|")
    for slot in slot_order:
        totals_slot = slot_totals.get(slot, Counter())
        total = sum(totals_slot.values())
        if total == 0:
            cells = ["0"] * (len(NODE_TYPES) + 1)
        else:
            cells = []
            for t in NODE_TYPES:
                c = totals_slot.get(t, 0)
                cells.append(f"{c} ({100*c/total:.0f}%)")
            unk = sum(v for k, v in totals_slot.items() if k not in NODE_TYPES)
            cells.append(f"{unk} ({100*unk/total:.0f}%)")
        slot_display = slot.replace("|", " / ")
        lines.append(f"| `{slot_display}` | {total} | " + " | ".join(cells) + " |")
    lines.append("")

    # =====================================================================
    # Licensed QA job counts
    # =====================================================================
    lines.append("## Licensed QA job counts\n")
    lines.append(
        "QA jobs the generator is licensed to attempt per (structure, rung, "
        "qa_type). Note: the `confounding` row collapses **both** "
        "`confounding_observed` and `confounding_unobserved` — see the "
        "prevalence table above for the split.\n"
    )
    qa_totals = Counter()
    for rec in per_scene.values():
        for key, v in rec["qa_counts"].items():
            qa_totals[key] += v

    by_struct = defaultdict(lambda: defaultdict(Counter))
    for key, v in qa_totals.items():
        struct, rung, qa = key.split("|")
        by_struct[struct][rung][qa] = v

    for struct in ("direct", "chain", "confounding", "collider"):
        if struct not in by_struct:
            continue
        lines.append(f"### {struct}\n")
        lines.append("| rung | qa_type | count |")
        lines.append("|---|---|---|")
        for rung in ("R0", "R1", "R2", "R3"):
            for qa in QA_TEMPLATES[struct].get(rung, []):
                c = by_struct[struct][rung].get(qa, 0)
                lines.append(f"| {rung} | {qa} | {c} |")
        lines.append("")

    grand_qa = sum(qa_totals.values())
    lines.append(f"**Grand total licensed QA jobs:** {grand_qa} "
                 f"({grand_qa / n:.1f} per scene on average)\n")

    # Top 20 richest scenes overall
    ranked = sorted(
        per_scene.items(), key=lambda kv: kv[1]["counts"]["total"], reverse=True
    )
    lines.append("## Top 20 richest scenes (by total subgraphs)\n")
    lines.append("| scene | total | direct | chain | collider | conf_obs | conf_unobs |")
    lines.append("|---|---|---|---|---|---|---|")
    for sid, rec in ranked[:20]:
        c = rec["counts"]
        lines.append(
            f"| {sid} | {c['total']} | {c['direct']} | {c['chain']} | "
            f"{c['collider']} | {c['confounding_observed']} | "
            f"{c['confounding_unobserved']} |"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if not args.graph_dir.is_dir():
        raise SystemExit(f"graph-dir does not exist: {args.graph_dir}")

    print(f"Scanning {args.graph_dir}...")
    per_scene, skipped, total = aggregate(args.graph_dir, args.workers)
    print(f"Processed {len(per_scene)}/{total} scenes. Skipped: {dict(skipped)}")

    report = make_report(per_scene, skipped, total, args.graph_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report -> {args.output}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {"per_scene": per_scene, "skipped": dict(skipped), "total": total},
                indent=2,
            )
        )
        print(f"JSON   -> {args.output_json}")


if __name__ == "__main__":
    main()
