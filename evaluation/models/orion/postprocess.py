"""
Orion post-processing script.

VQA path:
  Standard cascading regex extracts Yes/No or A-D from the "text" field of
  outputs.jsonl. No model-specific token stripping required.

Trajectory path:
  Trajectory inference now writes its predictions to trajectory_outputs.jsonl
  (one BTE record per sample plus per-question CTE records) as JSON-encoded
  waypoints, e.g. ``{"text": "{\"waypoints\": [{\"t\":0.5,\"x\":1.9,\"y\":0.1},...]}"}``.
  The framework's parse_waypoints_from_text() consumes that directly, so we
  inherit base behaviour and only expose a CLI flag for the trajectory pass.

Reads per sample:
  <run_dir>/<dataset>/<scene_id>/<sample_id>/outputs.jsonl
  <run_dir>/<dataset>/<scene_id>/<sample_id>/trajectory_outputs.jsonl
  <run_dir>/<dataset>/<scene_id>/<sample_id>/trajectory_prompts.jsonl

Writes per sample:
  report.json              — VQA metrics
  trajectory_report.json   — BTE/CTE ADE metrics

Usage:
    python postprocess.py --run-dir /workspace/cdb/outputs/orion_<ts>
    python postprocess.py --run-dir ... --use-llm-judge
    python postprocess.py --run-dir ... --trajectory-eval
    python postprocess.py --all-runs
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_postprocess import ModelPostprocessor

_collect_outputs = ModelPostprocessor._collect_outputs


class OrionPostprocessor(ModelPostprocessor):
    """
    Orion VQA + trajectory postprocessing.

    Inherits the base parser. Trajectory predictions are JSON-encoded into the
    "text" field at inference time, so the default parse_waypoints_from_text()
    works without an override.
    """

    def run_trajectory_postprocess(self, args, model_label: str = "Orion") -> None:
        """
        Standalone trajectory postprocessing entrypoint — mirrors Alpamayo 1.5.
        """
        from evaluation.common.utils.config_reader import CDBConfig

        cfg = CDBConfig(config_path=getattr(args, "config", None))
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")

        bench_dir = (
            Path(args.bench_dir) if getattr(args, "bench_dir", None)
            else cfg.get_bench_dir(getattr(args, "dataset", "nuscenes"))
        )
        if not bench_dir.exists():
            raise FileNotFoundError(f"bench_dir does not exist: {bench_dir}")

        causal_dataset_name = bench_dir.name
        print("=" * 60)
        print(f"[TrajPostprocess] {model_label} — Trajectory Evaluation")
        print(f"[TrajPostprocess]   run_dir   = {run_dir}")
        print(f"[TrajPostprocess]   bench_dir = {bench_dir}")
        print("=" * 60)

        ok = self.process_trajectory_outputs(
            run_dir=run_dir,
            causal_dataset_name=causal_dataset_name,
            bench_dir=bench_dir,
            schema_version=cfg.schema_version,
            mode=getattr(args, "mode", "full"),
            subset_size=getattr(args, "subset_size", None),
            scene=getattr(args, "scene", None),
            force=getattr(args, "force", False),
        )
        if ok:
            self.aggregate_trajectory_reports(run_dir, causal_dataset_name)
        else:
            print("[TrajPostprocess] No trajectory reports generated.")


def _build_arg_parser():
    import argparse, os
    parser = ModelPostprocessor.build_arg_parser(
        "Post-process Orion VQA and/or trajectory outputs"
    )
    parser.add_argument(
        "--trajectory-eval",
        action="store_true",
        default=os.environ.get("TRAJECTORY_EVAL", "").lower() == "true",
        help="Run trajectory evaluation postprocessing instead of VQA postprocessing.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    pp = OrionPostprocessor(model_cfg={})

    # VQA postprocessing always runs (also runs LLM judge when --use-llm-judge
    # is set). Trajectory postprocessing additionally runs when --trajectory-eval
    # is passed; the two are independent and write to different files.
    pp.run_postprocess(args, "Orion")

    if args.trajectory_eval:
        if not args.run_dir:
            parser.error("--run-dir is required for --trajectory-eval")
        pp.run_trajectory_postprocess(args, "Orion")


if __name__ == "__main__":
    main()
