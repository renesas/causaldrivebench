"""
flag_unchanged_trajectories.py

Post-processing step: reads llm_ego_trajectory.json for every question that
went through Stage 5, and writes ego_trajectory_changed=True/False back into
the source QA JSON files (rung3_cf_trajectories.json, distractor_qa.json,
dormant_qa.json).

Questions where ego_trajectory_changed=False are false positives — the LLM
decided the counterfactual intervention didn't actually change ego behaviour.
The HTML renderer and downstream pipelines can skip these.

Usage:
    python flag_unchanged_trajectories.py --outputs-dir /path/to/outputs/openscene_50
    python flag_unchanged_trajectories.py --dataset openscene   # reads path from config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

# ── slug helpers (must match the slug logic in prepare_batches / render) ──────
def _slug(text: str, max_len: int = 80) -> str:
    return (
        text[:max_len]
        .replace(" ", "_")
        .replace("?", "")
        .replace("/", "_")
        .replace("'", "")
        .replace('"', "")
    )


def _find_traj_dir(cf_traj_base: Path, question: str) -> Path | None:
    for max_len in (80, 60, 100):
        slug = _slug(question, max_len)
        for candidate in (cf_traj_base / slug, cf_traj_base / slug.strip("_")):
            if candidate.exists():
                return candidate
    return None


def _load_llm_traj(q_dir: Path, scene_stem: str) -> dict | None:
    """Load llm_ego_trajectory.json from <q_dir>/<scene_stem>/ or <q_dir>/<first_subdir>/."""
    for candidate in (q_dir / scene_stem, ):
        p = candidate / "llm_ego_trajectory.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
    # fallback: first subdir
    subdirs = [d for d in q_dir.iterdir() if d.is_dir()]
    if subdirs:
        p = subdirs[0] / "llm_ego_trajectory.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
    return None


# ── per-scene processing ──────────────────────────────────────────────────────
def process_scene(scene_qa_dir: Path, verbose: bool = False) -> dict:
    """
    Reads all QA JSON files for one scene and stamps ego_trajectory_changed
    onto every entry that has a corresponding llm_ego_trajectory.json.

    Returns a stats dict.
    """
    cf_traj_dir = scene_qa_dir / "cf_trajectories"
    if not cf_traj_dir.exists():
        return {"stamped": 0, "missing": 0, "unchanged": 0}

    scene_stem = scene_qa_dir.name
    stats = {"stamped": 0, "missing": 0, "unchanged": 0}

    # Files to process and how to extract the question text from each entry
    QA_FILES = {
        "rung3_cf_trajectories.json": lambda e: e.get("active_question", {}).get("question", ""),
        "distractor_qa.json":         lambda e: e.get("question", ""),
        "dormant_qa.json":                 lambda e: e.get("question", ""),
    }

    for fname, get_question in QA_FILES.items():
        fpath = scene_qa_dir / fname
        if not fpath.exists():
            continue

        raw = json.loads(fpath.read_text())
        # Normalise to list
        is_dict = isinstance(raw, dict)
        entries = raw.get("questions", raw) if is_dict else raw
        if not isinstance(entries, list):
            continue

        modified = False
        for entry in entries:
            question = get_question(entry)
            if not question:
                continue

            q_dir = _find_traj_dir(cf_traj_dir, question)
            if q_dir is None:
                stats["missing"] += 1
                continue

            llm = _load_llm_traj(q_dir, scene_stem)
            if llm is None:
                stats["missing"] += 1
                continue

            changed = llm.get("trajectory_changed", True)
            entry["ego_trajectory_changed"] = changed
            if not changed:
                stats["unchanged"] += 1
            stats["stamped"] += 1
            modified = True

            if verbose:
                flag = "CHANGED" if changed else "UNCHANGED"
                print(f"  [{flag}] {question[:80]}")

        if modified:
            if is_dict:
                raw["questions"] = entries
                out = raw
            else:
                out = entries
            fpath.write_text(json.dumps(out, indent=2))

    return stats


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stamp ego_trajectory_changed into QA JSON files")
    parser.add_argument("--outputs-dir", default=None,
                        help="Path to the dataset outputs directory (e.g. outputs/openscene_50)")
    parser.add_argument("--dataset", default=None, choices=["openscene", "nuscenes", "argoverse2"],
                        help="Dataset name — resolves outputs path from config.yaml")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Resolve outputs dir
    outputs_dir = args.outputs_dir
    if not outputs_dir and args.dataset:
        cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text())
            outputs_dir = cfg.get(args.dataset, {}).get("outputs")
        except Exception as e:
            print(f"ERROR reading config.yaml: {e}", file=sys.stderr)
            sys.exit(1)

    if not outputs_dir:
        print("ERROR: provide --outputs-dir or --dataset", file=sys.stderr)
        sys.exit(1)

    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        print(f"ERROR: outputs dir not found: {outputs_path}", file=sys.stderr)
        sys.exit(1)

    # Discover scene QA dirs (subdirs that have at least one QA JSON)
    scene_dirs = sorted(
        d for d in outputs_path.iterdir()
        if d.is_dir() and any((d / f).exists() for f in (
            "rung3_cf_trajectories.json", "distractor_qa.json", "dormant_qa.json"
        ))
    )

    if not scene_dirs:
        print("No scene QA directories found.")
        sys.exit(0)

    print(f"Processing {len(scene_dirs)} scenes in {outputs_path} ...")

    total = {"stamped": 0, "missing": 0, "unchanged": 0}
    for scene_dir in scene_dirs:
        if args.verbose:
            print(f"\n{scene_dir.name}")
        s = process_scene(scene_dir, verbose=args.verbose)
        for k in total:
            total[k] += s[k]

    print(
        f"\nDone: {total['stamped']} entries stamped "
        f"({total['unchanged']} ego_trajectory_changed=False, "
        f"{total['stamped'] - total['unchanged']} True), "
        f"{total['missing']} missing llm_ego_trajectory.json"
    )


if __name__ == "__main__":
    main()
