"""
Evaluation orchestrator for the causal scene understanding benchmark.

Builds Docker images, runs inference and post-processing for each model,
then aggregates metrics into a comparison report.

Usage:
    python evaluation/orchestrate.py \\
        --models drivelm,drivevlm \\
        --dataset nuscenes \\
        --mode subset \\
        --subset-size 50 \\
        --output-dir evaluation/outputs \\
        --report-dir evaluation/reports

Flags:
    --skip-inference   Skip Docker inference; jump straight to post-processing
    --skip-build       Do not rebuild Docker images
    --dry-run          Print commands without executing them
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml  # requires PyYAML

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.common.metrics import aggregate_model_metrics, format_comparison_table


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Shell command helpers
# ---------------------------------------------------------------------------


def run_cmd(
    cmd: List[str],
    dry_run: bool = False,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
) -> int:
    """
    Run a shell command, printing it first.

    Returns the process return code (0 = success).
    In dry_run mode, prints the command and returns 0 without executing.
    """
    pretty = " ".join(cmd)
    print(f"\n[orchestrate] CMD: {pretty}")
    if dry_run:
        print("[orchestrate] (dry-run — not executed)")
        return 0

    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, env=merged_env, cwd=cwd)
    return result.returncode


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------


def build_image(
    compose_file: str,
    dry_run: bool = False,
    eval_dir: str = ".",
) -> bool:
    """Run docker compose build for a model. Returns True on success."""
    rc = run_cmd(
        ["docker", "compose", "-f", compose_file, "build"],  # Path to the dev.env should be custom
        dry_run=dry_run,
        cwd=eval_dir,
    )
    return rc == 0


def run_inference(
    compose_file: str,
    extra_env: dict,
    dry_run: bool = False,
    eval_dir: str = ".",  # TODO: Create a parameter to choose whether to keep the container after execution for inspection
) -> bool:
    """Run docker compose run --rm inference. Returns True on success."""
    env_args: List[str] = []
    for k, v in extra_env.items():
        if v is not None:
            env_args += ["-e", f"{k}={v}"]

    rc = run_cmd(
        ["docker", "compose", "-f", compose_file, "run", "--rm"]  # TODO: Include custom dev.env path
        + env_args
        + ["inference"],
        dry_run=dry_run,
        env=extra_env,
        cwd=eval_dir,
    )
    return rc == 0


def run_postprocess(
    compose_file: str,
    extra_env: dict,
    dry_run: bool = False,
    eval_dir: str = ".",
) -> bool:
    """Run docker compose run --rm --profile postprocess postprocess."""
    env_args: List[str] = []
    for k, v in extra_env.items():
        if v is not None:
            env_args += ["-e", f"{k}={v}"]

    rc = run_cmd(
        ["docker", "compose", "--profile", "postprocess",
         "-f", compose_file, "run", "--rm"]   # TODO: Include custom dev.env path
        + env_args
        + ["postprocess"],
        dry_run=dry_run,
        env=extra_env,
        cwd=eval_dir,
    )
    return rc == 0


def load_report(model_output_dir: Path) -> Optional[dict]:  # TODO: load latest report for the model
    """Load the report.json produced by a model's postprocess step."""
    report_path = model_output_dir / "report.json"   # Create a timestamp based path to accommodate re-runs -- <model>_%Y%M%D_%H%m%S format
    if not report_path.exists():
        print(f"[orchestrate] WARNING: report not found at {report_path}")
        return None
    with open(report_path) as fh:
        return json.load(fh)


# TODO: Parameterize and create another function to load reports based on a list of report paths

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate multi-model benchmark evaluation"
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated list of model names (e.g. drivelm,llava_next). "
             "Default: all enabled models in eval_config.yaml",
    )
    parser.add_argument(
        "--dataset",
        default="nuscenes",
        help="Dataset key from eval_config.yaml (default: nuscenes)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "subset", "single"],
        default="full",
    )
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument(
        "--output-dir",
        default="evaluation/outputs",
        help="Base directory for per-model inference outputs",
    )
    parser.add_argument(
        "--report-dir",
        default="evaluation/reports",
        help="Directory for comparison reports",
    )
    parser.add_argument(
        "--config",
        default="evaluation/eval_config.yaml",
        help="Path to eval_config.yaml",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference; re-run post-processing only",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip docker image build",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to repo root
        config_path = Path(__file__).parent / "eval_config.yaml"
    cfg = load_config(config_path)

    # Resolve paths relative to the evaluation/ directory
    eval_dir = str(Path(__file__).parent)

    # --- Dataset config --------------------------------------------------
    dataset_key = args.dataset
    if dataset_key not in cfg.get("datasets", {}):
        print(
            f"[orchestrate] ERROR: dataset '{dataset_key}' not found in config. "
            f"Available: {list(cfg.get('datasets', {}).keys())}"
        )
        sys.exit(1)
    dataset_cfg = cfg["datasets"][dataset_key]
    bench_dir = dataset_cfg["bench_dir"]
    raw_data_dir = cfg.get("raw_data_dir", "/raw_data")

    # --- Model selection -----------------------------------------------
    all_models: dict = cfg.get("models", {})
    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        requested = [k for k, v in all_models.items() if v.get("enabled", True)]

    print(f"[orchestrate] Models to evaluate: {requested}")
    print(f"[orchestrate] Dataset: {dataset_key} | Mode: {args.mode}")

    # --- Output / report dirs ------------------------------------------
    output_base = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-model loop ------------------------------------------------
    model_reports: Dict[str, dict] = {}
    model_errors: Dict[str, str] = {}

    for model_name in requested:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        if model_name not in all_models:
            print(f"[orchestrate] WARNING: '{model_name}' not in config — skipping")
            continue

        model_info = all_models[model_name]
        compose_file = model_info.get("compose_file")
        if not compose_file:
            print(f"[orchestrate] ERROR: no compose_file for '{model_name}' — skipping")
            continue

        # Compose file is relative to eval_dir
        compose_path = Path(eval_dir) / compose_file
        if not compose_path.exists() and not args.dry_run:
            print(
                f"[orchestrate] ERROR: compose file not found at {compose_path} — skipping"
            )
            model_errors[model_name] = f"compose file missing: {compose_path}"
            continue

        model_output_dir = output_base / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        extra_env = {
            "RAW_DATA_DIR": raw_data_dir,
            "CAUSAL_BENCH_DIR": bench_dir,
            "OUTPUT_DIR": str(model_output_dir),
            "MODE": args.mode,
            "SUBSET_SIZE": str(args.subset_size) if args.subset_size else "",
            "SCENE": args.scene or "",
        }

        try:
            # 1. Build image
            if not args.skip_build:
                print(f"[orchestrate] Building Docker image for {model_name} …")
                ok = build_image(str(compose_path), args.dry_run, eval_dir)
                if not ok:
                    raise RuntimeError(f"docker build failed for {model_name}")

            # 2. Run inference
            if not args.skip_inference:
                print(f"[orchestrate] Running inference for {model_name} …")
                ok = run_inference(str(compose_path), extra_env, args.dry_run, eval_dir)
                if not ok:
                    raise RuntimeError(f"inference failed for {model_name}")

            # 3. Run post-processing
            print(f"[orchestrate] Running post-processing for {model_name} …")
            ok = run_postprocess(str(compose_path), extra_env, args.dry_run, eval_dir)
            if not ok:
                raise RuntimeError(f"post-process failed for {model_name}")

            # 4. Load report
            report = load_report(model_output_dir)
            if report:
                model_reports[model_name] = report
                overall_acc = report.get("overall", {}).get("accuracy", "N/A")
                print(
                    f"[orchestrate] {model_name} overall accuracy: {overall_acc:.2%}"
                    if isinstance(overall_acc, float)
                    else f"[orchestrate] {model_name} overall accuracy: {overall_acc}"
                )
            else:
                model_errors[model_name] = "report.json not found"

        except Exception as exc:  # noqa: BLE001
            print(f"[orchestrate] ERROR for {model_name}: {exc}")
            model_errors[model_name] = str(exc)

    # --- Aggregate comparison ------------------------------------------
    print(f"\n{'='*60}")
    print("  COMPARISON TABLE")
    print(f"{'='*60}")

    if model_reports:
        aggregated = aggregate_model_metrics(model_reports)
        table = format_comparison_table(aggregated)

        # Save comparison report
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        comparison_path = report_dir / f"comparison_{timestamp}.json"
        with open(comparison_path, "w") as fh:
            json.dump(
                {
                    "timestamp": timestamp,
                    "dataset": dataset_key,
                    "mode": args.mode,
                    "aggregated": aggregated,
                    "errors": model_errors,
                    "table": table,
                },
                fh,
                indent=2,
            )
        print(f"\n[orchestrate] Comparison report saved to {comparison_path}")
    else:
        print("[orchestrate] No successful model reports to compare.")

    if model_errors:
        print("\n[orchestrate] Models with errors:")
        for m, err in model_errors.items():
            print(f"  {m}: {err}")
        sys.exit(1)

    print("\n[orchestrate] All done.")


if __name__ == "__main__":
    main()
