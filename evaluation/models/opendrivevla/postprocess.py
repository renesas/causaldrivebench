"""
OpenDriveVLA post-processing script.

Parses raw Qwen2.5-VL output to extract structured answers, generates
per-sample report.json files alongside outputs.jsonl, then aggregates
dataset and run-level reports.

Reads from the per-sample output structure written by inference.py:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/outputs.jsonl

Writes:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/report.json  (Tier 1)
  <run_dir>/<causal_dataset_name>/report.json                          (Tier 2)
  <run_dir>/report.json                                                 (Tier 3)
  <cdb_reports_dir>/...                                                 (HTML, if enabled)

Usage (inside container):
    uv run python postprocess.py --run-dir /workspace/data/cdb_outputs/opendrivevla_20260421_120000
    uv run python postprocess.py --run-dir ... --mode subset --subset-size 20
    uv run python postprocess.py --run-dir ... --force
    uv run python postprocess.py --run-dir ... --use-llm-judge
    uv run python postprocess.py --all-runs
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_postprocess import ModelPostprocessor


class OpenDriveVLAPostprocessor(ModelPostprocessor):
    """
    Parses OpenDriveVLA (Qwen2.5-VL) output text.

    Qwen2.5-VL may produce <think>...</think> reasoning blocks. Standard
    cascading regex extracts Yes/No or A–D. No model-specific token stripping
    required — inherits parse_answer() from ModelPostprocessor.
    """


def main() -> None:
    args = ModelPostprocessor.parse_postprocess_args(
        "Post-process OpenDriveVLA outputs and compute benchmark metrics"
    )
    OpenDriveVLAPostprocessor(model_cfg={}).run_postprocess(args, "OpenDriveVLA")


if __name__ == "__main__":
    main()
