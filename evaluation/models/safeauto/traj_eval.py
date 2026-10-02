"""
SafeAuto Trajectory Evaluation.

End-to-end script that:
  1. Loads the SafeAuto-DriveLM model (reuses SafeAutoInference from inference.py).
  2. Iterates over selected scenes/samples in the causal_nuscenes bench dir.
  3. Prompts the model to predict 6 future ego waypoints (t=0.5…3.0 s).
  4. Parses the predicted trajectory and computes ADE / FDE against GT.
  5. Writes per-sample and aggregate reports to the run directory.

Usage:
    # First two scenes (default)
    python traj_eval.py

    # Explicit scenes
    python traj_eval.py --scenes nuscenes-scene-0001 nuscenes-scene-0002

    # Single scene
    python traj_eval.py --scenes nuscenes-scene-0003

    # Override paths
    python traj_eval.py \\
        --bench-dir /workspace/raw_data/causal_nuscenes/nuscenes_full_ice \\
        --nuscenes-root /workspace/raw_data/nuscenes \\
        --weights-path /workspace/ckpts/SafeAuto-DriveLM \\
        --output-dir /workspace/traj_eval_outputs

Environment:
    conda activate safeauto
    export WEIGHTS_PATH=/workspace/ckpts/SafeAuto-DriveLM   # optional
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Allow running from repo root or model directory
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from traj_dataloader import iter_prompts  # noqa: E402
from traj_postprocess import process_sample, aggregate_results, print_report  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_BENCH_DIR   = "/workspace/raw_data/causal_nuscenes/nuscenes_full_ice"
_DEFAULT_NS_ROOT     = "/workspace/raw_data/nuscenes"
_DEFAULT_WEIGHTS     = "/workspace/ckpts/SafeAuto-DriveLM"
_DEFAULT_OUTPUT_BASE = "/workspace/traj_eval_outputs"
_DEFAULT_SCENES      = ["nuscenes-scene-0001", "nuscenes-scene-0002"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SafeAuto ego-trajectory evaluation")
    p.add_argument("--bench-dir",     default=os.environ.get("BENCH_DIR",     _DEFAULT_BENCH_DIR))
    p.add_argument("--nuscenes-root", default=os.environ.get("NUSCENES_ROOT", _DEFAULT_NS_ROOT))
    p.add_argument("--weights-path",  default=os.environ.get("WEIGHTS_PATH",  _DEFAULT_WEIGHTS))
    p.add_argument("--output-dir",    default=os.environ.get("OUTPUT_DIR",    _DEFAULT_OUTPUT_BASE))
    p.add_argument("--device",        default=os.environ.get("DEVICE",        "cuda"))
    p.add_argument(
        "--mode",
        choices=["full", "subset", "single"],
        default=os.environ.get("MODE", "full"),
        help="full=all scenes, subset=first N, single=one scene",
    )
    p.add_argument(
        "--subset-size",
        type=int,
        default=int(os.environ.get("SUBSET_SIZE", 0)) or None,
        help="Number of scenes for --mode subset",
    )
    p.add_argument(
        "--scene",
        default=os.environ.get("SCENE") or None,
        help="Scene ID for --mode single (e.g. nuscenes-scene-0001)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume an existing run (skip samples that already have outputs)",
    )
    p.add_argument(
        "--run-dir", default=None,
        help="Existing run directory to resume (implies --resume)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a")],
    )


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def discover_scenes(bench_dir: Path, mode: str, scene: str | None, subset_size: int | None) -> list[str]:
    all_scenes = sorted(
        d.name for d in bench_dir.iterdir()
        if d.is_dir() and d.name.startswith("nuscenes-scene-")
    )
    if mode == "single":
        if not scene:
            raise ValueError("--mode single requires --scene <scene-id>")
        if scene not in all_scenes:
            raise ValueError(f"Scene '{scene}' not found in {bench_dir}")
        return [scene]
    if mode == "subset":
        n = subset_size or 2
        return all_scenes[:n]
    return all_scenes  # full


# ---------------------------------------------------------------------------
# SafeAuto model loader (lazy — reuses inference.py)
# ---------------------------------------------------------------------------

def load_safeauto(weights_path: str, device: str):
    """
    Load SafeAuto via the existing SafeAutoInference class.
    Returns a ready-to-use SafeAutoInference instance.
    """
    from inference import SafeAutoInference  # noqa: E402

    model = SafeAutoInference(model_cfg={}, device=device, cfg=None)
    model.load_model(weights_path)
    return model


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------

def run_inference(model, payload: dict) -> str:
    """
    Call SafeAutoInference.run_single() and return the raw text output.
    """
    try:
        result = model.run_single(payload)
        return result.get("text", "") if isinstance(result, dict) else str(result)
    except Exception:
        logger.error(
            "Inference error for %s/%s:\n%s",
            payload.get("scene_id"), payload.get("sample_id"),
            traceback.format_exc(),
        )
        return ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    bench_dir    = Path(args.bench_dir).resolve()
    ns_root      = Path(args.nuscenes_root).resolve()
    weights_path = str(Path(args.weights_path).resolve())

    # Run directory
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        resume = True
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir).resolve() / f"safeauto_traj_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        resume = args.resume

    setup_logging(run_dir / "traj_eval.log")

    logger.info("SafeAuto Trajectory Evaluation")
    logger.info("  bench_dir    : %s", bench_dir)
    logger.info("  nuscenes_root: %s", ns_root)
    logger.info("  weights      : %s", weights_path)
    logger.info("  run_dir      : %s", run_dir)
    logger.info("  device       : %s", args.device)

    if not bench_dir.exists():
        raise FileNotFoundError(f"bench_dir not found: {bench_dir}")
    if not ns_root.exists():
        raise FileNotFoundError(f"nuscenes_root not found: {ns_root}")

    scenes = discover_scenes(bench_dir, args.mode, args.scene, args.subset_size)
    logger.info("Mode: %s | Scenes (%d): %s", args.mode, len(scenes), scenes[:5])

    # Build all prompts first (fast, no model needed)
    prompts = list(iter_prompts(bench_dir, scenes, ns_root))
    logger.info("Total prompts: %d", len(prompts))

    if not prompts:
        logger.error("No prompts generated — check bench_dir and scene names")
        return

    # Check which samples still need inference
    def _output_path(payload: dict) -> Path:
        return run_dir / payload["scene_id"] / payload["sample_id"] / "output.json"

    pending = [p for p in prompts if not resume or not _output_path(p).exists()]
    logger.info(
        "%d/%d samples need inference (%d already done)",
        len(pending), len(prompts), len(prompts) - len(pending),
    )

    # Load model only if there's work to do
    model = None
    if pending:
        logger.info("Loading SafeAuto model …")
        model = load_safeauto(weights_path, args.device)
        logger.info("Model loaded.")

    # --- Inference loop ---
    for i, payload in enumerate(prompts, 1):
        out_path = _output_path(payload)
        sid = f"{payload['scene_id']}/{payload['sample_id']}"

        if resume and out_path.exists():
            logger.info("[%d/%d] %s — already done, skipping", i, len(prompts), sid)
            continue

        logger.info("[%d/%d] %s — running inference …", i, len(prompts), sid)
        raw_text = run_inference(model, payload)
        logger.info(
            "[%d/%d] %s — raw output (%d chars): %r…",
            i, len(prompts), sid, len(raw_text), raw_text[:120],
        )

        result = process_sample(payload, raw_text)

        # Save per-sample output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)

        logger.info(
            "[%d/%d] %s — ADE=%.4f FDE=%.4f parse=%s",
            i, len(prompts), sid,
            result["metrics"]["ade"] if result["metrics"]["ade"] is not None else float("nan"),
            result["metrics"]["fde"] if result["metrics"]["fde"] is not None else float("nan"),
            result["parse_method"],
        )

    # --- Aggregate ---
    all_results = []
    for payload in prompts:
        out_path = _output_path(payload)
        if out_path.exists():
            with out_path.open() as f:
                all_results.append(json.load(f))
        else:
            logger.warning("Missing output for %s/%s", payload["scene_id"], payload["sample_id"])

    report = aggregate_results(all_results)
    print_report(report)

    report_path = run_dir / "traj_report.json"
    with report_path.open("w") as f:
        json.dump({"aggregate": report, "samples": all_results}, f, indent=2)

    logger.info("Report saved to %s", report_path)
    print(f"\nRun directory : {run_dir}")
    print(f"Report        : {report_path}")


if __name__ == "__main__":
    main()
