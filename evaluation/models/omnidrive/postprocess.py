"""
OmniDrive post-processing script.

Parses raw TinyLlama / LLaVA-Llama output to extract structured answers,
generates per-sample report.json files alongside outputs.jsonl, then aggregates
dataset and run-level reports.

Reads from the per-sample output structure written by inference.py:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/outputs.jsonl

Writes:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/report.json  (Tier 1)
  <run_dir>/<causal_dataset_name>/report.json                          (Tier 2)
  <run_dir>/report.json                                                 (Tier 3)
  <cdb_reports_dir>/...                                                 (HTML, if enabled)

Usage (inside container):
    uv run python postprocess.py --run-dir /workspace/data/cdb_outputs/omnidrive_20260421_120000
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


class OmniDrivePostprocessor(ModelPostprocessor):
    """
    Parses OmniDrive (EVA-ViT + TinyLlama LM head) output text.

    The LM head produces direct answers without reasoning wrappers. Standard
    cascading regex extracts Yes/No or A–D. No model-specific token stripping
    required — inherits parse_answer() from ModelPostprocessor.
    """


def main() -> None:
    args = ModelPostprocessor.parse_postprocess_args(
        "Post-process OmniDrive outputs and compute benchmark metrics"
    )
    OmniDrivePostprocessor(model_cfg={}).run_postprocess(args, "OmniDrive")


if __name__ == "__main__":
    main()
