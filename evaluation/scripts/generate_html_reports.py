"""
Generate HTML reports for a completed evaluation run.

Usage:
    python evaluation/scripts/generate_html_reports.py \
        --run-name alpamayo_1_5_20260428_081120 \
        --outputs-dir /datadrive/datasets/cdb_outputs_20260427 \
        --reports-dir /datadrive/datasets/cdb_reports_20260427 \
        --bench-dir /datadrive/datasets/causal_drive_bench_20260427/causal_nuscenes \
        --dataset causal_nuscenes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _extract_model_id(run_name: str) -> str:
    """Extract model ID from run_name (everything before last _YYYYMMDD_HHMMSS)."""
    parts = run_name.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0]
    return run_name


def build_root_aggregate(
    run_name: str,
    run_report: dict,
) -> dict:
    """Build a root_aggregate dict from a single run's report."""
    accuracy = run_report.get("metrics", {}).get("overall", {}).get("accuracy", 0.0)
    n_questions = run_report.get("n_questions", 0)
    datasets = [s["dataset"] for s in run_report.get("dataset_summaries", [])]
    generated_at = run_report.get("generated_at", datetime.now(timezone.utc).isoformat())
    model_id = _extract_model_id(run_name)

    return {
        "generated_at": generated_at,
        "runs": {
            run_name: {
                "accuracy": accuracy,
                "n_questions": n_questions,
                "datasets": datasets,
                "generated_at": generated_at,
            }
        },
        "per_model": {
            model_id: {
                "best_accuracy": accuracy,
                "avg_accuracy": accuracy,
                "n_runs": 1,
                "runs": [run_name],
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--outputs-dir", required=True, type=Path)
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--bench-dir", type=Path, default=None,
                        help="Bench dir for graph_structure.json lookup")
    parser.add_argument("--dataset", default="causal_nuscenes",
                        help="Dataset name (used as key in bench_dirs)")
    args = parser.parse_args()

    # Load run report
    run_report_path = args.outputs_dir / args.run_name / "report.json"
    if not run_report_path.exists():
        logger.error("Run report not found: %s", run_report_path)
        sys.exit(1)
    with open(run_report_path) as f:
        run_report = json.load(f)

    if run_report.get("level") != "run":
        logger.error("report.json level is %r, expected 'run'", run_report.get("level"))
        sys.exit(1)

    root_aggregate = build_root_aggregate(args.run_name, run_report)
    logger.info("Root aggregate: accuracy=%.4f, n_questions=%d",
                root_aggregate["runs"][args.run_name]["accuracy"],
                root_aggregate["runs"][args.run_name]["n_questions"])

    bench_dirs = None
    if args.bench_dir:
        bench_dirs = {args.dataset: args.bench_dir}
        logger.info("bench_dirs: %s", bench_dirs)

    from evaluation.common.html_reporter import generate_all_html
    logger.info("Starting HTML generation for run: %s", args.run_name)
    generate_all_html(
        run_name=args.run_name,
        cdb_outputs_dir=args.outputs_dir,
        cdb_reports_dir=args.reports_dir,
        root_aggregate=root_aggregate,
        bench_dirs=bench_dirs,
    )
    logger.info("HTML generation complete.")

    # Summary
    run_html = args.reports_dir / args.run_name / "index.html"
    root_html = args.reports_dir / "index.html"
    logger.info("Run HTML:  %s (exists=%s)", run_html, run_html.exists())
    logger.info("Root HTML: %s (exists=%s)", root_html, root_html.exists())

    # Count generated files
    html_files = list(args.reports_dir.rglob("*.html"))
    logger.info("Total HTML files generated: %d", len(html_files))


if __name__ == "__main__":
    main()
