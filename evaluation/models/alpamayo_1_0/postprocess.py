"""
Alpamayo 1.0 post-processing script.

Parses the raw chain-of-causation (CoC) output from AlpamayoR1 to extract
structured answers, generates per-sample report.json files alongside
outputs.jsonl, then aggregates dataset and run-level reports.

Reads from the per-sample output structure written by inference.py:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/outputs.jsonl

Writes:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/report.json  (Tier 1)
  <run_dir>/<causal_dataset_name>/report.json                          (Tier 2)
  <run_dir>/report.json                                                (Tier 3)
  <cdb_reports_dir>/...                                                (HTML, if enabled)

Usage (inside container):
    python postprocess.py --run-dir /workspace/data/cdb_outputs/alpamayo_1_0_20260410_120000
    python postprocess.py --run-dir ... --mode subset --subset-size 20
    python postprocess.py --run-dir ... --force
    python postprocess.py --run-dir ... --use-llm-judge
    python postprocess.py --all-runs

Bench directory defaults to eval_config.yaml via CDBConfig.get_bench_dir().
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_postprocess import ModelPostprocessor


class AlpamayoPostprocessor(ModelPostprocessor):
    """
    Parses AlpamayoR1's chain-of-causation (CoC) text output.

    The CoC may be wrapped in <think>...</think> tags. Standard cascading
    regex extracts Yes/No or A–D from the output. No model-specific token
    stripping required — inherits parse_answer() from ModelPostprocessor.
    """


def main() -> None:
    args = ModelPostprocessor.parse_postprocess_args(
        "Post-process Alpamayo 1.0 outputs and compute benchmark metrics"
    )
    AlpamayoPostprocessor(model_cfg={}).run_postprocess(args, "Alpamayo 1.0")


if __name__ == "__main__":
    main()
