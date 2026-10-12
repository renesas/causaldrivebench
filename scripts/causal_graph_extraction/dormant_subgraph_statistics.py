"""
Aggregate statistics over DORMANT causal subgraphs in three flavors:

  (1) Dormant direct only — each dormant ego-edge `X ↛ ego`. This is the
      Stage 3b D_L QA substrate ("what would happen if X activated?").
  (2) Full taxonomy over dormant parents — direct / chain / collider /
      confounding re-run with `active_only=False`, so dormant ego-edges
      participate as parents just like active ones. Compares directly to
      the active-only baseline to show the "dormant contribution".
  (3) Mixed confounding — chain/collider still require active parents but
      dormant edges are allowed to act as confounders/mediators via
      `include_dormant_in_confounding=True`. Surfaces confounding triangles
      that only materialize when dormant causes are counted.

Usage:
    python scripts/causal_graph_extraction/dormant_subgraph_statistics.py \\
        --graph-dir outputs/nuscenes_full_ice \\
        --output dormant_subgraph_stats_report.md \\
        [--output-json dormant_subgraph_stats.json]
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.causal_graph_extraction.extract_subgraphs import (  # noqa: E402
    _extract_scene_ids,
    _resolve_graph_path,
    extract_subgraphs,
)

SAMPLE_ID_RE = re.compile(r"_SAMPLED_\d+$")
SUBGRAPH_KEYS = [
    "direct",
    "chain",
    "collider",
    "confounding_observed",
    "confounding_unobserved",
]
NODE_TYPES = ["EGO", "OTHER_AGENT", "TRAFFIC_CONTROL", "ENVIRONMENT", "ROAD_OBSTACLE"]


def _node_type_map(graph):
    return {n["id"]: n.get("type", "UNKNOWN") for n in graph.get("nodes", [])}


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

    ego_edges = graph.get("ego_edges", [])
    ntype = _node_type_map(graph)

    # ---------- Flavor 1: dormant direct only ----------
    f1_dormant_direct = [e for e in ego_edges if not e.get("active", False)]
    f1_counts = {"dormant_direct": len(f1_dormant_direct)}
    f1_effects = Counter((e.get("effect_type") or "UNKNOWN") for e in f1_dormant_direct)
    f1_statuses = Counter((e.get("causal_status") or "UNKNOWN") for e in f1_dormant_direct)
    f1_types = Counter(ntype.get(e["source"], "UNKNOWN") for e in f1_dormant_direct)

    # ---------- Baseline (active only) for comparison in flavors 2 and 3 ----------
    sg_active = extract_subgraphs(graph, active_only=True,
                                  include_dormant_in_confounding=False)
    base_counts = {k: len(sg_active[k]) for k in SUBGRAPH_KEYS}

    # ---------- Flavor 2: full taxonomy over dormant parents ----------
    sg_all = extract_subgraphs(graph, active_only=False,
                               include_dormant_in_confounding=False)
    f2_counts = {k: len(sg_all[k]) for k in SUBGRAPH_KEYS}
    f2_delta = {k: f2_counts[k] - base_counts[k] for k in SUBGRAPH_KEYS}

    # Count how many subgraphs in flavor-2 involve at least one dormant ego-edge
    active_sources = {e["source"] for e in ego_edges if e.get("active", False)}

    def _involves_dormant_direct(rec):
        # any ego-edge-incident node in this record that is not an active source
        for edge in rec.get("edges", []):
            if edge.get("kind") == "ego":
                if edge["source"] not in active_sources:
                    return True
        return False

    f2_dormant_touched = {
        k: sum(1 for rec in sg_all[k] if _involves_dormant_direct(rec))
        for k in SUBGRAPH_KEYS
    }

    # Node-type participation over all-parents extraction
    f2_types_in_struct = defaultdict(Counter)
    for d in sg_all["direct"]:
        f2_types_in_struct[f"direct|{d['confounding_role']}|source"][ntype.get(d["source"], "UNKNOWN")] += 1
    for c in sg_all["chain"]:
        f2_types_in_struct["chain|distal"][ntype.get(c["distal"], "UNKNOWN")] += 1
        f2_types_in_struct["chain|mediator"][ntype.get(c["mediator"], "UNKNOWN")] += 1
    for c in sg_all["collider"]:
        for p in c["parents"]:
            f2_types_in_struct["collider|parent"][ntype.get(p, "UNKNOWN")] += 1
    for c in sg_all["confounding_observed"]:
        f2_types_in_struct["confounding_observed|confounder"][ntype.get(c["confounder"], "UNKNOWN")] += 1
        f2_types_in_struct["confounding_observed|mediator"][ntype.get(c["mediator"], "UNKNOWN")] += 1
    for c in sg_all["confounding_unobserved"]:
        for p in c["nodes"]:
            f2_types_in_struct["confounding_unobserved|node"][ntype.get(p, "UNKNOWN")] += 1

    # ---------- Flavor 3: mixed confounding (active chain/collider, dormant in confounding) ----------
    sg_mixed = extract_subgraphs(graph, active_only=True,
                                 include_dormant_in_confounding=True)
    f3_counts = {k: len(sg_mixed[k]) for k in SUBGRAPH_KEYS}
    f3_delta_conf = {
        "confounding_observed":
            f3_counts["confounding_observed"] - base_counts["confounding_observed"],
        "confounding_unobserved":
            f3_counts["confounding_unobserved"] - base_counts["confounding_unobserved"],
    }

    # Triangles that appear only when dormant participation is allowed
    def _pair_key_obs(rec):
        return frozenset((rec["confounder"], rec["mediator"]))

    def _pair_key_unobs(rec):
        return frozenset(rec["nodes"])

    base_obs_pairs = {_pair_key_obs(r) for r in sg_active["confounding_observed"]}
    base_unobs_pairs = {_pair_key_unobs(r) for r in sg_active["confounding_unobserved"]}
    new_obs_pairs = [_pair_key_obs(r) for r in sg_mixed["confounding_observed"]
                     if _pair_key_obs(r) not in base_obs_pairs]
    new_unobs_pairs = [_pair_key_unobs(r) for r in sg_mixed["confounding_unobserved"]
                       if _pair_key_unobs(r) not in base_unobs_pairs]
    f3_new_triangles = {
        "confounding_observed": len(new_obs_pairs),
        "confounding_unobserved": len(new_unobs_pairs),
    }

    return scene_id, {
        "f1_counts": f1_counts,
        "f1_effects": dict(f1_effects),
        "f1_statuses": dict(f1_statuses),
        "f1_types": dict(f1_types),
        "base_counts": base_counts,
        "f2_counts": f2_counts,
        "f2_delta": f2_delta,
        "f2_dormant_touched": f2_dormant_touched,
        "f2_types_in_struct": {k: dict(v) for k, v in f2_types_in_struct.items()},
        "f3_counts": f3_counts,
        "f3_delta_conf": f3_delta_conf,
        "f3_new_triangles": f3_new_triangles,
        "n_active_parents": len([e for e in ego_edges if e.get("active", False)]),
        "n_dormant_parents": len(f1_dormant_direct),
        "graph_file": path.name,
    }, None


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
        f"**{label}** — n={total}, mean={mean(values):.2f}, "
        f"median={median(values):.1f}, max={max(values)}",
        "",
        "| bucket | count | share |",
        "|---|---|---|",
    ]
    for lab, n in zip(labels, buckets):
        share = 100 * n / total if total else 0
        out.append(f"| {lab} | {n} | {share:.1f}% |")
    out.append("")
    return out


def _prevalence_row(per_scene, count_key, sub_key=None):
    """Count scenes where the given count is > 0."""
    n_pos = 0
    total = 0
    for rec in per_scene.values():
        v = rec[count_key] if sub_key is None else rec[count_key][sub_key]
        if v > 0:
            n_pos += 1
        total += v
    return n_pos, total


def make_report(per_scene, skipped, total_tasks, graph_dir: Path):
    n = len(per_scene)
    lines = []
    lines.append("# Dormant-subgraph statistics report\n")
    lines.append(f"- Graph dir: `{graph_dir}`")
    lines.append(f"- Scenes analyzed: **{n}** / {total_tasks}")
    if skipped:
        lines.append(f"- Skipped: {dict(skipped)}")
    lines.append("")

    # -----------------------------------------------------------------
    # Definitions
    # -----------------------------------------------------------------
    lines.append("## Three flavors of dormant extraction\n")
    lines.append(
        "- **Flavor 1 — Dormant direct only**: each dormant ego-edge\n"
        "  `X ↛ ego` is a standalone subgraph. This is the Stage 3b D_L\n"
        "  QA substrate: questions like *what would happen to ego if X\n"
        "  activated?*\n"
        "- **Flavor 2 — Full taxonomy over all parents**: re-run extraction\n"
        "  with `active_only=False`. Dormant ego-edges now count as parents\n"
        "  just like active ones, so the direct / chain / collider /\n"
        "  confounding counts can grow. Compared to the active-only\n"
        "  baseline, the *delta* reveals how many structures are\n"
        "  dormant-driven.\n"
        "- **Flavor 3 — Mixed confounding**: chain/collider still require\n"
        "  active parents, but dormant parents are allowed to participate\n"
        "  in confounding triangles (`include_dormant_in_confounding=True`).\n"
        "  Surfaces confounding triangles that only materialize when you\n"
        "  include dormant causes. Useful for Rung-3 counterfactual\n"
        "  analysis where you ask *what if the dormant parent activated?*\n"
    )

    # -----------------------------------------------------------------
    # Baseline context
    # -----------------------------------------------------------------
    lines.append("## Baseline parent counts\n")
    active_vals = [rec["n_active_parents"] for rec in per_scene.values()]
    dormant_vals = [rec["n_dormant_parents"] for rec in per_scene.values()]
    lines += _hist(active_vals, "active ego-parents per scene")
    lines += _hist(dormant_vals, "dormant ego-parents per scene")

    scenes_with_dormant = sum(1 for v in dormant_vals if v > 0)
    lines.append(f"Scenes with ≥1 dormant parent: **{scenes_with_dormant}** "
                 f"({100*scenes_with_dormant/n:.1f}%)\n")

    # =================================================================
    # Flavor 1 — Dormant direct only
    # =================================================================
    lines.append("## Flavor 1 — Dormant direct only\n")
    n_pos, total = _prevalence_row(per_scene, "f1_counts", "dormant_direct")
    lines.append(f"- Scenes with ≥1 dormant direct: **{n_pos}** "
                 f"({100*n_pos/n:.1f}%)")
    lines.append(f"- Total dormant direct subgraphs: **{total}**  "
                 f"(mean {total/n:.2f} per scene)\n")

    lines += _hist(
        [rec["f1_counts"]["dormant_direct"] for rec in per_scene.values()],
        "dormant direct per scene",
    )

    # effect_type breakdown
    eff_tot = Counter()
    for rec in per_scene.values():
        for e, c in rec["f1_effects"].items():
            eff_tot[e] += c
    total_e = sum(eff_tot.values())
    lines.append("### Dormant direct — effect_type distribution\n")
    lines.append("| effect_type | count | share |")
    lines.append("|---|---|---|")
    for e, c in eff_tot.most_common():
        share = 100 * c / total_e if total_e else 0
        lines.append(f"| {e} | {c} | {share:.1f}% |")
    lines.append("")

    # causal_status breakdown
    st_tot = Counter()
    for rec in per_scene.values():
        for s, c in rec["f1_statuses"].items():
            st_tot[s] += c
    total_s = sum(st_tot.values())
    lines.append("### Dormant direct — causal_status distribution\n")
    lines.append("| causal_status | count | share |")
    lines.append("|---|---|---|")
    for s, c in st_tot.most_common():
        share = 100 * c / total_s if total_s else 0
        lines.append(f"| {s} | {c} | {share:.1f}% |")
    lines.append("")

    # node-type breakdown
    type_tot = Counter()
    for rec in per_scene.values():
        for t, c in rec["f1_types"].items():
            type_tot[t] += c
    total_t = sum(type_tot.values())
    lines.append("### Dormant direct — source node type\n")
    lines.append("| node_type | count | share |")
    lines.append("|---|---|---|")
    for t in NODE_TYPES:
        c = type_tot.get(t, 0)
        share = 100 * c / total_t if total_t else 0
        lines.append(f"| {t} | {c} | {share:.1f}% |")
    unk = sum(v for k, v in type_tot.items() if k not in NODE_TYPES)
    if unk:
        lines.append(f"| UNKNOWN | {unk} | {100*unk/total_t:.1f}% |")
    lines.append("")

    # =================================================================
    # Flavor 2 — Full taxonomy over dormant parents
    # =================================================================
    lines.append("## Flavor 2 — Full taxonomy over all parents (active + dormant)\n")
    lines.append(
        "Re-runs extraction with dormant edges treated as parents. The "
        "**delta** column shows how many *new* subgraphs of each type "
        "appear compared to the active-only baseline.\n"
    )
    lines.append("| structure | active-only | all-parents | delta | "
                 "scenes w/ ≥1 (all) | scenes w/ ≥1 touching dormant |")
    lines.append("|---|---|---|---|---|---|")
    for key in SUBGRAPH_KEYS:
        active_tot = sum(rec["base_counts"][key] for rec in per_scene.values())
        all_tot = sum(rec["f2_counts"][key] for rec in per_scene.values())
        delta = all_tot - active_tot
        scenes_all = sum(1 for rec in per_scene.values()
                         if rec["f2_counts"][key] > 0)
        scenes_touch = sum(1 for rec in per_scene.values()
                           if rec["f2_dormant_touched"][key] > 0)
        lines.append(f"| {key} | {active_tot} | {all_tot} | +{delta} | "
                     f"{scenes_all} ({100*scenes_all/n:.1f}%) | "
                     f"{scenes_touch} ({100*scenes_touch/n:.1f}%) |")
    lines.append("")

    # Per-scene count distributions (all-parents)
    for key in SUBGRAPH_KEYS:
        lines += _hist(
            [rec["f2_counts"][key] for rec in per_scene.values()],
            f"{key} per scene (all parents)",
        )

    # Node-type participation (all-parents view)
    lines.append("### Node-type participation per subgraph slot (all parents)\n")
    slot_totals = defaultdict(Counter)
    for rec in per_scene.values():
        for slot, tcounts in rec["f2_types_in_struct"].items():
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

    # =================================================================
    # Flavor 3 — Mixed confounding
    # =================================================================
    lines.append("## Flavor 3 — Mixed confounding (active chain/collider + dormant-aware confounding)\n")
    lines.append(
        "Same active-only extraction for chain/collider, but confounding "
        "triangles may now include dormant ego-parents. The **new triangles** "
        "column counts distinct ego-parent pairs that *only* show up once "
        "dormant participation is enabled.\n"
    )
    lines.append("| structure | active-only | mixed | delta | new triangles | "
                 "scenes w/ ≥1 new triangle |")
    lines.append("|---|---|---|---|---|---|")
    for key in ("confounding_observed", "confounding_unobserved"):
        active_tot = sum(rec["base_counts"][key] for rec in per_scene.values())
        mixed_tot = sum(rec["f3_counts"][key] for rec in per_scene.values())
        delta = mixed_tot - active_tot
        new_tris = sum(rec["f3_new_triangles"][key] for rec in per_scene.values())
        scenes_new = sum(1 for rec in per_scene.values()
                         if rec["f3_new_triangles"][key] > 0)
        lines.append(f"| {key} | {active_tot} | {mixed_tot} | +{delta} | "
                     f"{new_tris} | {scenes_new} ({100*scenes_new/n:.1f}%) |")
    lines.append("")

    lines += _hist(
        [rec["f3_new_triangles"]["confounding_observed"]
         for rec in per_scene.values()],
        "new observed-confounding triangles per scene (dormant-unlocked)",
    )
    lines += _hist(
        [rec["f3_new_triangles"]["confounding_unobserved"]
         for rec in per_scene.values()],
        "new unobserved-confounding triangles per scene (dormant-unlocked)",
    )

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
