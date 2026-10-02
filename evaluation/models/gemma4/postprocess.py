"""
Gemma 4 post-processing script.

Parses raw Gemma 4 API responses to extract structured answers, generates
per-sample report.json files alongside outputs.jsonl, then aggregates
dataset and run-level reports.

GEMMA 4 OUTPUT FORMAT:
  With thinking mode enabled (<|think|> in system prompt), the model may
  prepend internal reasoning before the final answer.  The base class
  parse_answer() already strips <think>...</think> blocks.  This subclass
  also strips the Gemma 4 control token variant <|thinking|>...</|thinking|>
  if present, then applies the standard cascading regex to extract Yes/No or A/B/C/D.

  Typical outputs:
    <think>The vehicle stopped because...</think>\\nAnswer: Yes\\n...
    Answer: A\\nThe pedestrian is directly in the path because...

Reads from the per-sample output structure written by inference.py:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/outputs.jsonl

Writes:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/report.json  (Tier 1)
  <run_dir>/<causal_dataset_name>/report.json                          (Tier 2)
  <run_dir>/report.json                                                 (Tier 3)
  <cdb_reports_dir>/...                                                 (HTML, if enabled)

Usage:
    python postprocess.py --run-dir /workspace/data/cdb_outputs/gemma4_e2b_20260421_120000
    python postprocess.py --run-dir ... --mode subset --subset-size 20
    python postprocess.py --run-dir ... --force
    python postprocess.py --run-dir ... --use-llm-judge
    python postprocess.py --all-runs
    python postprocess.py --trajectory-eval --run-dir ...
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_postprocess import ModelPostprocessor


class Gemma4Postprocessor(ModelPostprocessor):
    """
    Parses Gemma 4 API response text.

    Thinking mode produces reasoning wrapped in <think>...</think> (handled by
    base class) or the Gemma 4 control-token variant <|thinking|>...</|thinking|>.
    Both are stripped before the standard cascading regex extracts the answer.

    Trajectory evaluation:
      - parse_trajectory_output() strips thinking tokens then delegates to base parser.
      - run_trajectory_postprocess() is a standalone CLI entrypoint for trajectory eval.
    """

    def _preprocess_text(self, text: str) -> str:
        # Strip Gemma 4 control-token thinking blocks (if not already XML-style)
        text = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", text, flags=re.DOTALL)
        # Strip any trailing generation artifacts
        text = re.sub(r"<\|[a-z_]+\|>", "", text)
        return text.strip()

    def parse_trajectory_output(self, raw_output, prompt_record: dict) -> tuple:
        """
        Extract trajectory waypoints from Gemma 4 output.

        Strips thinking tokens before delegating to the base JSON parser.
        """
        from evaluation.common.trajectory_eval import parse_waypoints_from_text
        text = self._extract_text(raw_output)
        text = self._preprocess_text(text)
        return parse_waypoints_from_text(text)

    def run_trajectory_postprocess(self, args, model_label: str = "Gemma 4") -> None:
        """
        Standalone trajectory postprocessing entrypoint.
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


def _build_trajectory_arg_parser():
    import argparse, os
    parser = ModelPostprocessor.build_arg_parser(
        "Post-process Gemma 4 VQA and/or trajectory outputs"
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

    postprocessor = Gemma4Postprocessor(model_cfg={})
    model_label = f"Gemma 4 ({_get_model_tag()})"

    if args.trajectory_eval:
        if not args.run_dir:
            parser.error("--run-dir is required for --trajectory-eval")
        postprocessor.run_trajectory_postprocess(args, model_label)
    else:
        postprocessor.run_postprocess(args, model_label)


def _get_model_tag() -> str:
    import os
    return os.environ.get("GEMMA4_MODEL_TAG", "e2b")


if __name__ == "__main__":
    main()
