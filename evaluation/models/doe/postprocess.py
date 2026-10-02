"""
Doe-1 post-processing script.

Parses raw FlexAR / Chameleon output to extract structured answers, generates
per-sample report.json files alongside outputs.jsonl, then aggregates dataset
and run-level reports.

Reads from the per-sample output structure written by inference.py:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/outputs.jsonl

Writes:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/report.json  (Tier 1)
  <run_dir>/<causal_dataset_name>/report.json                          (Tier 2)
  <run_dir>/report.json                                                 (Tier 3)
  <cdb_reports_dir>/...                                                 (HTML, if enabled)

Usage (inside container):
    python postprocess.py --run-dir /workspace/data/cdb_outputs/doe_20260421_120000
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


class DoePostprocessor(ModelPostprocessor):
    """
    Parses Doe-1 (FlexAR / Chameleon) output text.

    Chameleon may emit <|image|> tokens in its text output. These are stripped
    before standard <think> stripping and cascading regex answer extraction.
    """

    def _preprocess_text(self, text: str) -> str:
        return re.sub(r"<\|image\|>", "", text)


def main() -> None:
    args = ModelPostprocessor.parse_postprocess_args(
        "Post-process Doe-1 outputs and compute benchmark metrics"
    )
    DoePostprocessor(model_cfg={}).run_postprocess(args, "Doe-1")


if __name__ == "__main__":
    main()
