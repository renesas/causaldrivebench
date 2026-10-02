"""
Alpamayo 1.5 post-processing script.

Parses raw VQA output from Alpamayo 1.5 to extract structured answers, generates
per-sample report.json files alongside outputs.jsonl, then aggregates dataset
and run-level reports.

DIFFERENCES FROM ALPAMAYO 1.0:
  - Output field is extra["answer"] (not extra["cot"]); text is shorter and more
    direct since the model uses the VQA interface (<|answer_start|> token).
  - The <|answer_end|> token may appear in raw output — stripped before parsing.
  - No <think>...</think> wrapper expected (generate_text() vs CoC rollout).

Reads from the per-sample output structure written by inference.py:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/outputs.jsonl

Writes:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/report.json  (Tier 1)
  <run_dir>/<causal_dataset_name>/report.json                          (Tier 2)
  <run_dir>/report.json                                                 (Tier 3)
  <cdb_reports_dir>/...                                                 (HTML, if enabled)

Usage (inside container):
    python postprocess.py --run-dir /workspace/data/cdb_outputs/alpamayo_1_5_20260421_120000
    python postprocess.py --run-dir ... --mode subset --subset-size 20
    python postprocess.py --run-dir ... --force
    python postprocess.py --run-dir ... --use-llm-judge
    python postprocess.py --all-runs
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_postprocess import ModelPostprocessor


class Alpamayo15Postprocessor(ModelPostprocessor):
    """
    Parses Alpamayo 1.5 VQA answer text.

    The model uses the native VQA interface (generate_text + <|answer_start|>)
    so outputs are typically short and direct — "Answer: Yes" or "Answer: A".
    Strips <|answer_end|> if present, then applies standard cascading regex.

    Trajectory evaluation:
      - parse_trajectory_output() strips <|answer_end|> then delegates to base parser.
      - run_trajectory_postprocess() is a standalone CLI entrypoint for trajectory eval.
    """

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r"<\|answer_end\|>.*", "", text, flags=re.DOTALL).strip()
        # The model occasionally outputs "True"/"False" instead of "Yes"/"No"
        text = re.sub(r"\bTrue\b", "Yes", text)
        text = re.sub(r"\bFalse\b", "No", text)
        return text

    def parse_trajectory_output(self, raw_output, prompt_record: dict) -> tuple:
        """
        Extract trajectory waypoints from Alpamayo 1.5 output.

        Strips the <|answer_end|> token before delegating to the base JSON parser.
        """
        from evaluation.common.trajectory_eval import parse_waypoints_from_text
        text = self._extract_text(raw_output)
        text = self._preprocess_text(text)
        return parse_waypoints_from_text(text)

    def run_trajectory_postprocess(self, args, model_label: str = "Alpamayo 1.5") -> None:
        """
        Standalone trajectory postprocessing entrypoint.

        Reads trajectory_outputs.jsonl per sample, computes ADE metrics, writes
        trajectory_report.json, and prints an aggregate summary.

        Designed to run AFTER VQA postprocessing so that original VQA answers
        (for CTE consistency checks) are available from outputs.jsonl.

        Usage (CLI):
            python postprocess.py --trajectory-eval --run-dir <run_dir>
        """
        from pathlib import Path

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


def _build_trajectory_arg_parser():
    import argparse, os
    parser = ModelPostprocessor.build_arg_parser(
        "Post-process Alpamayo 1.5 VQA and/or trajectory outputs"
    )
    parser.add_argument(
        "--trajectory-eval",
        action="store_true",
        default=os.environ.get("TRAJECTORY_EVAL", "").lower() == "true",
        help="Run trajectory evaluation postprocessing instead of VQA postprocessing.",
    )
    return parser


def main() -> None:
    parser = _build_trajectory_arg_parser()
    args = parser.parse_args()

    postprocessor = Alpamayo15Postprocessor(model_cfg={})

    if args.trajectory_eval:
        if not args.run_dir:
            parser.error("--run-dir is required for --trajectory-eval")
        postprocessor.run_trajectory_postprocess(args, "Alpamayo 1.5")
    else:
        postprocessor.run_postprocess(args, "Alpamayo 1.5")


if __name__ == "__main__":
    main()
