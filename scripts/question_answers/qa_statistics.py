"""
qa_statistics.py — Aggregate QA statistics across rungs and question types.

Walks per-scene subdirectories under --graph-dir, parses
{active,dormant,distractor}_qa.json, and reports counts broken down by:
  - source stage (active / dormant / distractor)
  - rung (0..3)
  - question category (CaI, CR, SC, DQ, EL, CI, CaA, AR, CDE, NIE, NDE, BAS,
    CfI, CB, NC, ...)
  - graph structure (Direct / Chain / Collider / Confounding / None)
  - answer format (mcq / binary / open)
  - difficulty (easy / medium / hard)

Only scene dirs matching ``*_SAMPLED_\\d+`` are counted (matches
subgraph_statistics.py convention).

Usage:
  python scripts/causal_graph_extraction/qa_statistics.py \
      --graph-dir /spiral_hdd_3/workspace/navvrat/Neurips/outputs/nuscenes_full_ice \
      --output qa_stats_report.md \
      --output-json qa_stats.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SAMPLED_RE = re.compile(r"_SAMPLED_\d+$")
QA_FILES = {
    "active": "active_qa.json",
    "dormant": "dormant_qa.json",
    "distractor": "distractor_qa.json",
}


def iter_scene_dirs(root: Path):
    for p in sorted(root.iterdir()):
        if p.is_dir() and SAMPLED_RE.search(p.name):
            yield p


def load_questions(path: Path):
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict) and "questions" in data:
        return data["questions"] or []
    if isinstance(data, list):
        return data
    return []


def collect(root: Path) -> dict:
    stats = {
        "totals": Counter(),                              # by source stage
        "by_rung": Counter(),                             # rung -> count
        "by_category": Counter(),                         # category -> count
        "by_source_rung": defaultdict(Counter),           # source -> rung -> count
        "by_source_category": defaultdict(Counter),       # source -> category -> count
        "by_rung_category": defaultdict(Counter),         # rung -> category -> count
        "by_structure": Counter(),                        # graph_structure -> count
        "by_structure_rung": defaultdict(Counter),        # structure -> rung -> count
        "by_format": Counter(),                           # answer_format -> count
        "by_difficulty": Counter(),                       # difficulty -> count
        "by_source_difficulty": defaultdict(Counter),
        "scenes_with_qa": defaultdict(set),               # source -> {scene_id}
        "qa_per_scene": defaultdict(Counter),             # source -> {scene_id: count}
    }
    scenes_total = 0
    scenes_with_any_qa = set()

    for scene_dir in iter_scene_dirs(root):
        scenes_total += 1
        scene_id = scene_dir.name
        for source, fname in QA_FILES.items():
            qpath = scene_dir / fname
            if not qpath.exists():
                continue
            qs = load_questions(qpath)
            if qs is None:
                continue
            if qs:
                stats["scenes_with_qa"][source].add(scene_id)
                scenes_with_any_qa.add(scene_id)
            stats["qa_per_scene"][source][scene_id] = len(qs)
            for q in qs:
                rung = q.get("rung")
                category = q.get("category") or "<unknown>"
                structure = q.get("graph_structure") or "<none>"
                fmt = q.get("answer_format") or "<unknown>"
                diff = q.get("difficulty") or "<unknown>"
                stats["totals"][source] += 1
                stats["by_rung"][rung] += 1
                stats["by_category"][category] += 1
                stats["by_source_rung"][source][rung] += 1
                stats["by_source_category"][source][category] += 1
                stats["by_rung_category"][rung][category] += 1
                stats["by_structure"][structure] += 1
                stats["by_structure_rung"][structure][rung] += 1
                stats["by_format"][fmt] += 1
                stats["by_difficulty"][diff] += 1
                stats["by_source_difficulty"][source][diff] += 1

    stats["scenes_total"] = scenes_total
    stats["scenes_with_any_qa"] = len(scenes_with_any_qa)
    return stats


# ─── Reporting ─────────────────────────────────────────────────────────────

def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.1f}%" if d else "—"


def render_markdown(stats: dict, root: Path) -> str:
    lines = []
    lines.append("# QA statistics report\n")
    lines.append(f"- Graph dir: `{root}`")
    lines.append(f"- Scene dirs scanned: **{stats['scenes_total']}**")
    lines.append(f"- Scenes with at least one QA file: **{stats['scenes_with_any_qa']}**\n")

    total_qa = sum(stats["totals"].values())
    lines.append(f"## Totals\n\n**Total QA pairs: {total_qa:,}**\n")
    lines.append("| Source stage | QA pairs | Share | Scenes covered |")
    lines.append("|---|---:|---:|---:|")
    for source in ("active", "dormant", "distractor"):
        n = stats["totals"][source]
        scenes = len(stats["scenes_with_qa"][source])
        lines.append(
            f"| {source} | {n:,} | {_pct(n, total_qa)} | {scenes:,} "
            f"({_pct(scenes, stats['scenes_total'])}) |"
        )
    lines.append("")

    # Rung breakdown
    lines.append("## By rung\n")
    lines.append("| Rung | QA pairs | Share |")
    lines.append("|---|---:|---:|")
    for rung in sorted(stats["by_rung"], key=lambda r: (r is None, r)):
        n = stats["by_rung"][rung]
        label = f"R{rung}" if rung is not None else "<none>"
        lines.append(f"| {label} | {n:,} | {_pct(n, total_qa)} |")
    lines.append("")

    # Category breakdown
    lines.append("## By question category\n")
    lines.append("| Category | QA pairs | Share |")
    lines.append("|---|---:|---:|")
    for cat, n in stats["by_category"].most_common():
        lines.append(f"| `{cat}` | {n:,} | {_pct(n, total_qa)} |")
    lines.append("")

    # Source × Rung
    lines.append("## Source stage × Rung\n")
    sources = ["active", "dormant", "distractor"]
    rungs_sorted = sorted(
        {r for src in sources for r in stats["by_source_rung"][src]},
        key=lambda r: (r is None, r),
    )
    header = "| Source | " + " | ".join(
        f"R{r}" if r is not None else "none" for r in rungs_sorted
    ) + " | Total |"
    sep = "|---|" + "---:|" * (len(rungs_sorted) + 1)
    lines.append(header)
    lines.append(sep)
    for src in sources:
        row = [src]
        total = stats["totals"][src]
        for r in rungs_sorted:
            row.append(f"{stats['by_source_rung'][src][r]:,}")
        row.append(f"{total:,}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Source × Category
    lines.append("## Source stage × Category\n")
    cats_sorted = [c for c, _ in stats["by_category"].most_common()]
    header = "| Category | " + " | ".join(sources) + " | Total |"
    sep = "|---|" + "---:|" * (len(sources) + 1)
    lines.append(header)
    lines.append(sep)
    for cat in cats_sorted:
        row = [f"`{cat}`"]
        cat_total = 0
        for src in sources:
            n = stats["by_source_category"][src][cat]
            cat_total += n
            row.append(f"{n:,}")
        row.append(f"{cat_total:,}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Rung × Category matrix
    lines.append("## Rung × Category\n")
    rungs_sorted_full = sorted(
        stats["by_rung_category"].keys(), key=lambda r: (r is None, r)
    )
    header = "| Category | " + " | ".join(
        f"R{r}" if r is not None else "none" for r in rungs_sorted_full
    ) + " | Total |"
    sep = "|---|" + "---:|" * (len(rungs_sorted_full) + 1)
    lines.append(header)
    lines.append(sep)
    for cat in cats_sorted:
        row = [f"`{cat}`"]
        ctotal = 0
        for r in rungs_sorted_full:
            n = stats["by_rung_category"][r][cat]
            ctotal += n
            row.append(f"{n:,}")
        row.append(f"{ctotal:,}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Graph structure
    lines.append("## By graph structure (active stage only — dormant/distractor have none)\n")
    lines.append("| Structure | QA pairs | Share |")
    lines.append("|---|---:|---:|")
    for struct, n in stats["by_structure"].most_common():
        lines.append(f"| {struct} | {n:,} | {_pct(n, total_qa)} |")
    lines.append("")

    # Structure × Rung
    lines.append("## Graph structure × Rung\n")
    structs_sorted = [s for s, _ in stats["by_structure"].most_common()]
    rungs_struct = sorted(
        {r for s in structs_sorted for r in stats["by_structure_rung"][s]},
        key=lambda r: (r is None, r),
    )
    header = "| Structure | " + " | ".join(
        f"R{r}" if r is not None else "none" for r in rungs_struct
    ) + " | Total |"
    sep = "|---|" + "---:|" * (len(rungs_struct) + 1)
    lines.append(header)
    lines.append(sep)
    for s in structs_sorted:
        row = [s]
        stot = 0
        for r in rungs_struct:
            n = stats["by_structure_rung"][s][r]
            stot += n
            row.append(f"{n:,}")
        row.append(f"{stot:,}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Answer format
    lines.append("## By answer format\n")
    lines.append("| Format | QA pairs | Share |")
    lines.append("|---|---:|---:|")
    for fmt, n in stats["by_format"].most_common():
        lines.append(f"| {fmt} | {n:,} | {_pct(n, total_qa)} |")
    lines.append("")

    # Difficulty
    lines.append("## By difficulty\n")
    lines.append("| Difficulty | QA pairs | Share |")
    lines.append("|---|---:|---:|")
    for diff, n in stats["by_difficulty"].most_common():
        lines.append(f"| {diff} | {n:,} | {_pct(n, total_qa)} |")
    lines.append("")

    lines.append("### Difficulty by source stage\n")
    diffs_all = [d for d, _ in stats["by_difficulty"].most_common()]
    header = "| Source | " + " | ".join(diffs_all) + " | Total |"
    sep = "|---|" + "---:|" * (len(diffs_all) + 1)
    lines.append(header)
    lines.append(sep)
    for src in sources:
        row = [src]
        stot = 0
        for d in diffs_all:
            n = stats["by_source_difficulty"][src][d]
            stot += n
            row.append(f"{n:,}")
        row.append(f"{stot:,}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-scene QA-count distribution
    lines.append("## QA pairs per scene\n")
    lines.append("| Source | Scenes covered | Mean | Median | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for src in sources:
        counts = list(stats["qa_per_scene"][src].values())
        counts = [c for c in counts if c > 0]
        if not counts:
            lines.append(f"| {src} | 0 | — | — | — | — |")
            continue
        counts.sort()
        n = len(counts)
        mean = sum(counts) / n
        median = counts[n // 2] if n % 2 else (counts[n // 2 - 1] + counts[n // 2]) / 2
        lines.append(
            f"| {src} | {n:,} | {mean:.2f} | {median:.1f} | {min(counts)} | {max(counts)} |"
        )
    lines.append("")

    return "\n".join(lines)


def stats_to_jsonable(stats: dict) -> dict:
    out = {
        "scenes_total": stats["scenes_total"],
        "scenes_with_any_qa": stats["scenes_with_any_qa"],
        "totals": dict(stats["totals"]),
        "by_rung": {str(k): v for k, v in stats["by_rung"].items()},
        "by_category": dict(stats["by_category"]),
        "by_source_rung": {
            src: {str(k): v for k, v in cnt.items()}
            for src, cnt in stats["by_source_rung"].items()
        },
        "by_source_category": {
            src: dict(cnt) for src, cnt in stats["by_source_category"].items()
        },
        "by_rung_category": {
            str(k): dict(cnt) for k, cnt in stats["by_rung_category"].items()
        },
        "by_structure": dict(stats["by_structure"]),
        "by_structure_rung": {
            s: {str(k): v for k, v in cnt.items()}
            for s, cnt in stats["by_structure_rung"].items()
        },
        "by_format": dict(stats["by_format"]),
        "by_difficulty": dict(stats["by_difficulty"]),
        "by_source_difficulty": {
            src: dict(cnt) for src, cnt in stats["by_source_difficulty"].items()
        },
        "scenes_with_qa": {
            src: sorted(scenes) for src, scenes in stats["scenes_with_qa"].items()
        },
        "qa_per_scene": {
            src: dict(cnt) for src, cnt in stats["qa_per_scene"].items()
        },
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph-dir", required=True, type=Path,
                   help="Outputs dir containing per-scene subdirectories.")
    p.add_argument("--output", required=True, type=Path,
                   help="Path to write the markdown report.")
    p.add_argument("--output-json", type=Path, default=None,
                   help="Optional JSON dump of the raw aggregates.")
    args = p.parse_args()

    if not args.graph_dir.is_dir():
        raise SystemExit(f"--graph-dir is not a directory: {args.graph_dir}")

    stats = collect(args.graph_dir)
    md = render_markdown(stats, args.graph_dir)
    args.output.write_text(md)
    print(f"Wrote {args.output} ({sum(stats['totals'].values()):,} QA pairs across "
          f"{stats['scenes_with_any_qa']:,} scenes)")

    if args.output_json:
        args.output_json.write_text(json.dumps(stats_to_jsonable(stats), indent=2))
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
