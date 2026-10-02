#!/usr/bin/env python3
"""
Package pipeline outputs into a clean per-sample release structure.

Output layout:
    <downloads_dir>/<dataset_name>/<scene>/<SAMPLED_N>/
        frames.json
        calib.json
        meta.json
        state.json
        graph.json              # verified > fixed > *_edited > pruned > raw
        graph_structure.json    # topology + subgraph counts/records
        qa/
            active_qa.json
            dormant_qa.json
            distractor_qa.json

Usage:
    python scripts/release/package_samples.py --dataset nuscenes
    python scripts/release/package_samples.py \
        --records /path --outputs /path --download /path
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config as _load_config  # noqa: E402
from common.scene_utils import get_raw_frames           # noqa: E402
from causal_graph_extraction.extract_subgraphs import (  # noqa: E402
    extract_subgraphs,
    _resolve_graph_path,
)
from graph_filtering.graph_statistics import extract_graph_stats  # noqa: E402


_TOPOLOGY_KEYS = (
    "n_nodes", "n_distractors", "n_ego_edges", "n_active_edges", "n_dormant_edges",
    "n_inter_edges", "graph_depth", "branching_factor", "richness",
    "node_type_counts", "node_category_counts", "position_counts", "camera_counts",
    "effect_counts", "active_effect_counts", "causal_status_counts",
    "affected_state_counts", "distractor_type_counts", "distractor_category_counts",
    "has_upstream_fork", "n_upstream_forks",
)


def _json_default(o):
    if isinstance(o, Counter):
        return dict(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not serializable: {type(o).__name__}")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        shutil.copy2(src, dst)
        return True
    return False


def _build_graph_structure(graph: dict, scene_id: str, source_suffix: str) -> dict:
    """Compute per-sample structural features from a loaded graph dict."""
    stats = extract_graph_stats(graph)
    topology = {k: stats[k] for k in _TOPOLOGY_KEYS if k in stats}

    active = extract_subgraphs(graph, active_only=True)
    all_parents = extract_subgraphs(graph, active_only=False)

    return {
        "scene_id":      scene_id,
        "source_suffix": source_suffix,
        "topology":      topology,
        "subgraphs_active": {
            "counts":  {k: len(v) for k, v in active.items()},
            "records": active,
        },
        "subgraphs_all": {
            "counts": {k: len(v) for k, v in all_parents.items()},
        },
    }


def package_sample(scene_name: str, sample_name: str,
                   records_dir: Path, outputs_dir: Path,
                   out_dir: Path, cfg: dict) -> dict:
    """Package one sample into out_dir/<scene_name>/<sample_name>/."""
    sample_rec = records_dir / scene_name / sample_name
    tp0  = sample_rec / "Tp0p0"
    stem = f"{scene_name}_{sample_name}"
    dest = out_dir / scene_name / sample_name
    dest.mkdir(parents=True, exist_ok=True)

    data_root = cfg.get("data_root")
    written = {
        "meta":   _copy_if_exists(sample_rec / "meta.json", dest / "meta.json"),
        "state":  _copy_if_exists(tp0 / "state.json",       dest / "state.json"),
        "calib":  _copy_if_exists(tp0 / "calib.json",       dest / "calib.json"),
    }

    # ── frames.json — relative raw sensor paths + data_root for callers ──────
    frames_data = {
        "data_root": data_root or "",
        "frames": get_raw_frames(sample_rec, cfg, data_root=data_root),
    }
    (dest / "frames.json").write_text(json.dumps(frames_data, indent=2))
    written["frames"] = bool(frames_data["frames"])

    # ── graph.json — verified > fixed > *_edited > pruned > raw ──────────────
    graph_src = _resolve_graph_path(outputs_dir, stem)
    if graph_src is None:
        raw = outputs_dir / f"{stem}.json"
        graph_src = raw if raw.exists() else None
    source_suffix = graph_src.stem[len(stem):] if graph_src else None  # "_verified", "", …
    written["graph"] = _copy_if_exists(graph_src, dest / "graph.json") if graph_src else False

    # ── graph_structure.json — per-sample structural features ────────────────
    structure_counts = None
    if graph_src is not None:
        try:
            graph = json.loads(graph_src.read_text())
            structure = _build_graph_structure(graph, stem, source_suffix)
            (dest / "graph_structure.json").write_text(
                json.dumps(structure, indent=2, default=_json_default))
            written["graph_structure"] = True
            structure_counts = structure["subgraphs_active"]["counts"]
        except Exception as e:
            written["graph_structure"] = False
            print(f"  WARN: graph_structure failed for {stem}: {e}")

    # ── qa/ ───────────────────────────────────────────────────────────────────
    qa_src = outputs_dir / stem
    qa_map = {
        "active_qa.json":     "active_qa.json",
        "dormant_qa.json":    "dormant_qa.json",
        "distractor_qa.json": "distractor_qa.json",
    }
    qa_written = {}
    qa_questions = {}
    if qa_src.is_dir():
        qa_dest = dest / "qa"
        qa_dest.mkdir(exist_ok=True)
        for src_name, dst_name in qa_map.items():
            if _copy_if_exists(qa_src / src_name, qa_dest / dst_name):
                qa_written[dst_name] = True
                try:
                    d = json.loads((qa_dest / dst_name).read_text())
                    if isinstance(d, list):
                        n = len(d)
                    elif isinstance(d, dict):
                        n = next((len(d[k]) for k in ("questions", "qa", "active_qa",
                                                      "dormant_qa", "distractor_qa")
                                  if isinstance(d.get(k), list)), 0)
                    else:
                        n = 0
                except Exception:
                    n = 0
                qa_questions[dst_name] = n
    written["qa"] = qa_written

    return {"scene": scene_name, "sample": sample_name,
            "files": written, "qa_questions": qa_questions,
            "source_suffix": source_suffix,
            "subgraph_counts": structure_counts}


def main():
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--dataset", choices=["nuscenes", "openscene", "argoverse2"],
                      default=None)
    _known, _ = _pre.parse_known_args()
    _cfg     = _load_config(_known.dataset) if _known.dataset else (_load_config() or {})
    _top_cfg = _load_config() or {}

    parser = argparse.ArgumentParser(description="Package pipeline outputs for release")
    parser.add_argument("--dataset", choices=["nuscenes", "openscene", "argoverse2"],
                        help="Dataset name — resolves records/outputs from config.yaml")
    parser.add_argument("--records",  default=_cfg.get("records"),
                        help="Records directory (default: from config.yaml)")
    parser.add_argument("--outputs",  default=_cfg.get("outputs"),
                        help="Pipeline outputs directory (default: from config.yaml)")
    parser.add_argument("--download", default=None,
                        help="Release output dir (default: downloads_dir/<outputs_name>)")
    args = parser.parse_args()

    if not args.records:
        parser.error("--records is required (or set via --dataset to use config.yaml)")
    if not args.outputs:
        parser.error("--outputs is required (or set via --dataset to use config.yaml)")

    records_dir = Path(args.records)
    outputs_dir = Path(args.outputs)

    if args.download:
        out_dir = Path(args.download)
    else:
        downloads_dir = _top_cfg.get("downloads_dir")
        if not downloads_dir:
            parser.error("--download is required (or set downloads_dir in config.yaml)")
        out_dir = Path(downloads_dir) / outputs_dir.name

    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, missing_graph, missing_qa = [], [], []

    for scene_dir in sorted(records_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for sample_dir in sorted(scene_dir.iterdir()):
            if not sample_dir.is_dir() or not sample_dir.name.startswith("SAMPLED"):
                continue

            summary = package_sample(scene_dir.name, sample_dir.name,
                                     records_dir, outputs_dir, out_dir, _top_cfg)
            summaries.append(summary)

            if not summary["files"].get("graph"):
                missing_graph.append(f"{scene_dir.name}/{sample_dir.name}")
            if not summary["files"].get("qa"):
                missing_qa.append(f"{scene_dir.name}/{sample_dir.name}")

            qa_files = len(summary["files"].get("qa", {}))
            qa_qs    = sum(summary.get("qa_questions", {}).values())
            graph_ok = "yes" if summary["files"].get("graph") else "NO"
            src_sfx  = summary.get("source_suffix") or ""
            n_sg     = sum((summary.get("subgraph_counts") or {}).values())
            print(f"  {scene_dir.name}/{sample_dir.name}: "
                  f"graph={graph_ok}{src_sfx}, subgraphs={n_sg}, "
                  f"qa_files={qa_files}/3, questions={qa_qs}")

    n_samples         = len(summaries)
    n_with_qa         = n_samples - len(missing_qa)
    n_with_structure  = sum(1 for s in summaries if s["files"].get("graph_structure"))
    q_totals          = {"active_qa.json": 0, "dormant_qa.json": 0, "distractor_qa.json": 0}
    sg_totals         = {"direct": 0, "chain": 0, "collider": 0,
                         "confounding_observed": 0, "confounding_unobserved": 0}
    source_suffix_counts = Counter()
    for s in summaries:
        for k, n in s.get("qa_questions", {}).items():
            q_totals[k] = q_totals.get(k, 0) + n
        for k, n in (s.get("subgraph_counts") or {}).items():
            sg_totals[k] = sg_totals.get(k, 0) + n
        if s.get("source_suffix") is not None:
            source_suffix_counts[s["source_suffix"] or "_raw"] += 1
    total_questions = sum(q_totals.values())
    total_subgraphs = sum(sg_totals.values())

    manifest = {
        "data_root":           _top_cfg.get("data_root", ""),
        "records":             str(records_dir),
        "outputs":             str(outputs_dir),
        "n_samples":           n_samples,
        "n_samples_with_qa":   n_with_qa,
        "n_samples_with_structure": n_with_structure,
        "n_questions":         q_totals,
        "n_subgraphs_active":  sg_totals,
        "source_suffix_counts": dict(source_suffix_counts),
        "missing_graph":       missing_graph,
        "missing_qa":          missing_qa,
        "samples":             summaries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nPackaged {n_samples} samples → {out_dir}")
    print(f"  Samples with QA:        {n_with_qa} / {n_samples}")
    print(f"  Samples without QA:     {len(missing_qa)} / {n_samples}")
    print(f"  Samples with structure: {n_with_structure} / {n_samples}")
    if missing_graph:
        print(f"  Samples without graph:  {len(missing_graph)} / {n_samples}")
    if source_suffix_counts:
        print(f"\nGraph source breakdown:")
        for sfx, n in source_suffix_counts.most_common():
            print(f"  {sfx or '_raw':<16} {n}")
    print(f"\nTotal questions: {total_questions}")
    print(f"  active_qa:     {q_totals['active_qa.json']}")
    print(f"  dormant_qa:    {q_totals['dormant_qa.json']}")
    print(f"  distractor_qa: {q_totals['distractor_qa.json']}")
    if n_with_qa:
        print(f"  Avg per QA-bearing sample: {total_questions / n_with_qa:.1f}")
    print(f"\nActive subgraph totals: {total_subgraphs}")
    for k in ("direct", "chain", "collider",
              "confounding_observed", "confounding_unobserved"):
        print(f"  {k:<22} {sg_totals[k]}")
    print(f"\nTo zip: cd {out_dir.parent} && zip -r {out_dir.name}.zip {out_dir.name}/")


if __name__ == "__main__":
    main()
