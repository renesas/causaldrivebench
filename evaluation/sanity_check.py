"""
CausalDriveBench — Evaluation Sanity Check
==========================================
Validates all model runs in a cdb_outputs directory against the source dataset.

For each model run and dataset found, reports:
  - Expected vs actual sample/question counts
  - Missing, incomplete, or error outputs
  - Trajectory completeness
  - Overall health score

Usage:
    python evaluation/sanity_check.py --outputs-dir /datadrive/datasets/cdb_outputs_20260427
    python evaluation/sanity_check.py --outputs-dir /datadrive/datasets/cdb_outputs_20260427 \
        --bench-root /datadrive/datasets/causal_drive_bench_20260427
    python evaluation/sanity_check.py --outputs-dir /datadrive/datasets/cdb_outputs_20260427 \
        --run gemma4_e4b_20260428_044049
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_jsonl_lines(path: Path) -> int:
    """Return number of non-empty lines in a JSONL file."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    count = 0
    with open(path) as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _count_error_lines(path: Path) -> int:
    """Return number of lines whose raw_output.text starts with ERROR:."""
    if not path.exists():
        return 0
    count = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                text = rec.get("raw_output", {}).get("text", "")
                if isinstance(text, str) and text.startswith("ERROR:"):
                    count += 1
            except json.JSONDecodeError:
                pass
    return count


def _load_manifest(bench_dataset_dir: Path) -> dict | None:
    """Load and return manifest.json from the bench dataset directory.

    Checks both inside the dataset subdir and in the parent directory,
    since some dataset versions store manifest.json at the top-level root
    rather than inside causal_nuscenes/.
    """
    for candidate in (bench_dataset_dir / "manifest.json", bench_dataset_dir.parent / "manifest.json"):
        if candidate.exists():
            with open(candidate) as fh:
                return json.load(fh)
    return None


def _expected_qa_samples(bench_dataset_dir: Path, manifest: dict) -> set[str]:
    """Return set of 'scene_id/sample_id' pairs that have QA files."""
    # Use manifest missing_qa list if available
    all_pairs: set[str] = set()
    missing_qa: set[str] = set(manifest.get("missing_qa", []))
    for scene_dir in bench_dataset_dir.iterdir():
        if not scene_dir.is_dir() or scene_dir.name == "manifest.json":
            continue
        for sample_dir in scene_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            pair = f"{scene_dir.name}/{sample_dir.name}"
            if pair not in missing_qa:
                all_pairs.add(pair)
    return all_pairs


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------

def analyse_run(run_dir: Path, bench_root: Path | None) -> dict:
    """
    Analyse one model run directory.

    Returns a dict with per-dataset results plus a top-level summary.
    """
    result = {
        "run_dir": str(run_dir),
        "model_run": run_dir.name,
        "datasets": {},
    }

    # Each immediate subdirectory is a dataset (e.g. causal_nuscenes)
    dataset_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
    if not dataset_dirs:
        result["error"] = "No dataset directories found in run dir"
        return result

    for dataset_dir in sorted(dataset_dirs):
        dataset_name = dataset_dir.name
        bench_dataset_dir = (bench_root / dataset_name) if bench_root else None

        manifest = _load_manifest(bench_dataset_dir) if bench_dataset_dir else None

        # Ground-truth expected samples (those with QA in the benchmark)
        expected_qa_pairs: set[str] | None = None
        if bench_dataset_dir and manifest:
            expected_qa_pairs = _expected_qa_samples(bench_dataset_dir, manifest)

        # Scan run dir for all sample output dirs
        stats = {
            # Counts from benchmark manifest (ground truth)
            "bench_total_samples":      manifest.get("n_samples", "?") if manifest else "?",
            "bench_qa_samples":         manifest.get("n_samples_with_qa", "?") if manifest else "?",
            "bench_vqa_questions":      (sum(manifest["n_questions"].values())
                                        if manifest and manifest.get("n_questions") else "?"),

            # VQA
            "samples_with_prompts":     0,   # prompts.jsonl exists and non-empty
            "samples_with_outputs":     0,   # outputs.jsonl exists and non-empty
            "samples_complete_vqa":     0,   # outputs count == prompts count
            "samples_incomplete_vqa":   0,   # outputs < prompts (partial)
            "samples_missing_outputs":  0,   # prompts exists but outputs missing/empty
            "total_prompts_expected":   0,
            "total_outputs_written":    0,
            "total_error_outputs":      0,

            # Trajectory
            "samples_with_traj_prompts":    0,
            "samples_with_traj_outputs":    0,
            "samples_complete_traj":        0,
            "samples_incomplete_traj":      0,
            "samples_missing_traj_outputs": 0,
            "total_traj_prompts":           0,
            "total_traj_outputs":           0,
            "total_traj_errors":            0,

            # Errors / issues
            "issues": [],
        }

        # Walk scene/sample tree
        for scene_dir in sorted(dataset_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            for sample_dir in sorted(scene_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue

                pair = f"{scene_dir.name}/{sample_dir.name}"

                # ── VQA ──────────────────────────────────────────────────
                prompts_f = sample_dir / "prompts.jsonl"
                outputs_f = sample_dir / "outputs.jsonl"

                n_prompts = _count_jsonl_lines(prompts_f)
                n_outputs = _count_jsonl_lines(outputs_f)
                n_errors  = _count_error_lines(outputs_f)

                if n_prompts > 0:
                    stats["samples_with_prompts"] += 1
                    stats["total_prompts_expected"] += n_prompts

                if n_outputs > 0:
                    stats["samples_with_outputs"] += 1
                    stats["total_outputs_written"] += n_outputs
                    stats["total_error_outputs"]   += n_errors

                if n_prompts > 0 and n_outputs == n_prompts:
                    stats["samples_complete_vqa"] += 1
                elif n_prompts > 0 and 0 < n_outputs < n_prompts:
                    stats["samples_incomplete_vqa"] += 1
                    stats["issues"].append(
                        f"[VQA-PARTIAL] {pair}: {n_outputs}/{n_prompts} questions"
                    )
                elif n_prompts > 0 and n_outputs == 0:
                    stats["samples_missing_outputs"] += 1
                    if expected_qa_pairs and pair in expected_qa_pairs:
                        stats["issues"].append(
                            f"[VQA-MISSING] {pair}: prompts exist but no outputs"
                        )

                if n_errors > 0:
                    stats["issues"].append(
                        f"[VQA-ERRORS] {pair}: {n_errors}/{n_outputs} error outputs"
                    )

                # ── Trajectory ────────────────────────────────────────────
                tprompts_f = sample_dir / "trajectory_prompts.jsonl"
                toutputs_f = sample_dir / "trajectory_outputs.jsonl"

                n_tp = _count_jsonl_lines(tprompts_f)
                n_to = _count_jsonl_lines(toutputs_f)
                n_te = _count_error_lines(toutputs_f)

                if n_tp > 0:
                    stats["samples_with_traj_prompts"] += 1
                    stats["total_traj_prompts"] += n_tp

                if n_to > 0:
                    stats["samples_with_traj_outputs"] += 1
                    stats["total_traj_outputs"] += n_to
                    stats["total_traj_errors"]  += n_te

                if n_tp > 0 and n_to == n_tp:
                    stats["samples_complete_traj"] += 1
                elif n_tp > 0 and 0 < n_to < n_tp:
                    stats["samples_incomplete_traj"] += 1
                    stats["issues"].append(
                        f"[TRAJ-PARTIAL] {pair}: {n_to}/{n_tp} traj questions"
                    )
                elif n_tp > 0 and n_to == 0:
                    stats["samples_missing_traj_outputs"] += 1
                    stats["issues"].append(
                        f"[TRAJ-MISSING] {pair}: traj prompts exist but no outputs"
                    )

                if n_te > 0:
                    stats["issues"].append(
                        f"[TRAJ-ERRORS] {pair}: {n_te}/{n_to} traj error outputs"
                    )

        # ── Completion percentages ────────────────────────────────────────
        bench_qa = stats["bench_qa_samples"]
        n_complete = stats["samples_complete_vqa"]
        stats["pct_samples_complete"] = (
            f"{n_complete / bench_qa * 100:.1f}%"
            if isinstance(bench_qa, int) and bench_qa > 0 else "?"
        )
        bench_q = stats["bench_vqa_questions"]
        n_out = stats["total_outputs_written"]
        stats["pct_questions_written"] = (
            f"{n_out / bench_q * 100:.1f}%"
            if isinstance(bench_q, int) and bench_q > 0 else "?"
        )

        result["datasets"][dataset_name] = stats

    return result


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(analysis: dict, max_issues: int = 20, verbose: bool = False) -> None:
    sep = "=" * 72
    print(sep)
    print(f"  Run: {analysis['model_run']}")
    print(f"  Dir: {analysis['run_dir']}")
    print(sep)

    if "error" in analysis:
        print(f"  ERROR: {analysis['error']}")
        return

    for ds_name, s in analysis["datasets"].items():
        print(f"\n  Dataset: {ds_name}")
        print(f"  {'─'*60}")

        # Benchmark reference
        print(f"  Benchmark reference:")
        print(f"    Total samples       : {s['bench_total_samples']}")
        print(f"    Samples with QA     : {s['bench_qa_samples']}")
        print(f"    Total VQA questions : {s['bench_vqa_questions']}")

        # VQA progress
        print(f"\n  VQA outputs:")
        print(f"    Samples w/ prompts  : {s['samples_with_prompts']}")
        print(f"    Samples w/ outputs  : {s['samples_with_outputs']}")
        print(f"    Complete (✓)        : {s['samples_complete_vqa']}  ({s['pct_samples_complete']} of bench QA samples)")
        print(f"    Incomplete (partial): {s['samples_incomplete_vqa']}")
        print(f"    Missing outputs     : {s['samples_missing_outputs']}")
        print(f"    Questions expected  : {s['total_prompts_expected']}")
        print(f"    Questions written   : {s['total_outputs_written']}  ({s['pct_questions_written']} of bench total)")
        print(f"    Error outputs       : {s['total_error_outputs']}")

        # Trajectory progress
        print(f"\n  Trajectory outputs:")
        print(f"    Samples w/ traj prompts : {s['samples_with_traj_prompts']}")
        print(f"    Samples complete (✓)    : {s['samples_complete_traj']}")
        print(f"    Incomplete (partial)    : {s['samples_incomplete_traj']}")
        print(f"    Missing outputs         : {s['samples_missing_traj_outputs']}")
        print(f"    Traj questions written  : {s['total_traj_outputs']}")
        print(f"    Traj error outputs      : {s['total_traj_errors']}")

        # Issues
        issues = s["issues"]
        if issues:
            shown = issues[:max_issues] if not verbose else issues
            print(f"\n  Issues ({len(issues)} total):")
            for issue in shown:
                print(f"    {issue}")
            if not verbose and len(issues) > max_issues:
                print(f"    ... and {len(issues) - max_issues} more (use --verbose to see all)")
        else:
            print(f"\n  Issues: none")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sanity-check CausalDriveBench evaluation outputs against the source dataset."
    )
    p.add_argument(
        "--outputs-dir", required=True,
        help="Root cdb_outputs directory (e.g. /datadrive/datasets/cdb_outputs_20260427). "
             "All model run subdirectories are scanned automatically.",
    )
    p.add_argument(
        "--bench-root", default=None,
        help="Root benchmark directory (e.g. /datadrive/datasets/causal_drive_bench_20260427). "
             "Used to load manifest.json and compute expected counts. "
             "If omitted, only counts from the outputs dir are reported.",
    )
    p.add_argument(
        "--run", default=None,
        help="Limit analysis to a specific run directory name (e.g. gemma4_e4b_20260428_044049). "
             "Default: scan all runs in --outputs-dir.",
    )
    p.add_argument(
        "--max-issues", type=int, default=20,
        help="Max issues to print per dataset (default: 20). Use --verbose to show all.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print all issues (overrides --max-issues).",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of human-readable report.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"ERROR: outputs dir not found: {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    bench_root = Path(args.bench_root) if args.bench_root else None

    # Collect run directories
    if args.run:
        run_dirs = [outputs_dir / args.run]
    else:
        run_dirs = sorted(
            [d for d in outputs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

    if not run_dirs:
        print("No run directories found.", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"WARNING: run dir not found: {run_dir}", file=sys.stderr)
            continue
        result = analyse_run(run_dir, bench_root)
        all_results.append(result)

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        for result in all_results:
            print_report(result, max_issues=args.max_issues, verbose=args.verbose)


if __name__ == "__main__":
    main()
