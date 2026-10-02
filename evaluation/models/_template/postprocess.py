"""
Template post-processing script — copy to your model directory and implement
``parse_answer`` if your model has a non-standard output format.

Usage
-----
    # Inside container
    python postprocess.py \
        --outputs-dir /cdb_outputs/template_20260410_120000 \
        --bench-dir /causal_drive_bench/causal_nuscenes \
        --report-path /cdb_reports/template_20260410_120000/report.json

    # On host (EVAL_RUNTIME=host resolves paths from eval_config.yaml)
    EVAL_RUNTIME=host python postprocess.py --config /path/to/eval_config.yaml
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Allow running from repo root or within container
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_postprocess import ModelPostprocessor
from evaluation.common.utils.config_reader import CDBConfig


# ---------------------------------------------------------------------------
# Postprocessor implementation
# ---------------------------------------------------------------------------


class TemplatePostprocessor(ModelPostprocessor):
    """
    Extracts MCQ letter or binary (Yes/No) answer from model raw output.

    Default heuristics handle the most common output patterns:
      - ``"Answer: B"``
      - ``"The answer is B"``
      - ``"B) The vehicle …"``  (letter at start of response)
      - Bare letter:  ``"B"``
      - Bare Yes/No
    """

    def parse_answer(self, raw_output, question: dict) -> str:
        """
        Extract the predicted answer from raw_output.

        Parameters
        ----------
        raw_output : dict (new schema: ``{"text": "…"}``) or plain str (legacy)
        question   : QA dict with ``answer_format`` and ``options`` keys

        Returns
        -------
        ``"Yes"`` / ``"No"`` for binary, or ``"A"``/``"B"``/``"C"``/``"D"`` for MCQ.
        Empty string if extraction fails.

        TODO: Add model-specific parsing if your model uses a structured format
              (e.g. JSON output: ``{"answer": "C", "reasoning": "…"}``).
        """
        # Normalise raw_output to a string
        text: str
        if isinstance(raw_output, dict):
            text = raw_output.get("text", "")
        else:
            text = str(raw_output) if raw_output else ""

        if not text:
            return ""

        text = text.strip()
        answer_format = question.get("answer_format", "binary")

        if answer_format == "binary":
            return self._parse_binary(text)
        else:
            return self._parse_mcq(text)

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_binary(text: str) -> str:
        """Extract Yes / No from a response string."""
        # "Answer: Yes" / "Answer: No"
        m = re.search(r"\bAnswer\s*[:\-]?\s*(Yes|No)\b", text, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()

        # Bare "Yes" / "No"
        m = re.match(r"^\s*(Yes|No)\b", text, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()

        # Any occurrence of Yes or No
        m = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()

        return ""

    @staticmethod
    def _parse_mcq(text: str) -> str:
        """Extract MCQ letter (A/B/C/D) from a response string."""
        # "Answer: B" / "Answer - B"
        m = re.search(r"\bAnswer\s*[:\-]?\s*([A-D])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # "The answer is B"
        m = re.search(r"answer\s+is\s+([A-D])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # Letter at start, optionally followed by ) . or space
        m = re.match(r"^\s*([A-D])[).\s]", text)
        if m:
            return m.group(1).upper()

        # Bare single letter
        m = re.match(r"^\s*([A-D])\s*$", text)
        if m:
            return m.group(1).upper()

        # First standalone MCQ letter anywhere in the text
        m = re.search(r"\b([A-D])\b", text)
        if m:
            return m.group(1).upper()

        # TODO: add model-specific JSON parsing here if needed, e.g.
        # try:
        #     import json
        #     data = json.loads(text)
        #     return str(data.get("answer", "")).strip().upper()
        # except json.JSONDecodeError:
        #     pass

        return ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process model outputs and compute benchmark metrics"
    )
    parser.add_argument(
        "--outputs-dir",
        default=os.environ.get("OUTPUT_DIR", ""),
        help=(
            "Root outputs directory for this run "
            "(e.g. /cdb_outputs/template_20260410_120000). "
            "All outputs.jsonl files are collected recursively."
        ),
    )
    parser.add_argument(
        "--bench-dir",
        default=os.environ.get("CAUSAL_BENCH_DIR", ""),
        help="Dataset-specific bench directory containing QA files",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("DATASET", "nuscenes"),
        choices=["nuscenes", "argoverse2", "openscene"],
        help="Dataset name (used when --bench-dir is not provided)",
    )
    parser.add_argument(
        "--report-path",
        default=os.environ.get("REPORT_PATH", ""),
        help="Path to write report.json (defaults to <outputs-dir>/report.json)",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        default=os.environ.get("USE_LLM_JUDGE", "false").lower() == "true",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to eval_config.yaml (auto-detected if omitted)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = CDBConfig(config_path=args.config)

    bench_dir = (
        Path(args.bench_dir) if args.bench_dir
        else cfg.get_bench_dir(args.dataset)
    )

    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else cfg.dir_cdb_outputs
    report_path = (
        Path(args.report_path) if args.report_path
        else outputs_dir / "report.json"
    )

    # Collect all outputs.jsonl files under outputs_dir
    outputs_files = sorted(outputs_dir.rglob("outputs.jsonl"))
    if not outputs_files:
        print(f"No outputs.jsonl files found under {outputs_dir}")
        return

    postprocessor = TemplatePostprocessor(model_cfg={})
    judge_cfg = None
    if args.use_llm_judge:
        judge_cfg = cfg.judge_config or {
            "provider": "openai",
            "model":    os.environ.get("JUDGE_MODEL", "gpt-4o"),
            "api_key_env": "OPENAI_API_KEY",
        }

    # If multiple outputs.jsonl files, concatenate (simple approach)
    # TODO: for per-dataset reporting, pass individual files and aggregate
    if len(outputs_files) == 1:
        outputs_path = outputs_files[0]
    else:
        import tempfile, json as _json
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        for f in outputs_files:
            with open(f) as fh:
                tmp.write(fh.read())
        tmp.flush()
        outputs_path = Path(tmp.name)
        print(f"Merged {len(outputs_files)} outputs.jsonl files for evaluation.")

    metrics = postprocessor.compute_metrics(
        outputs_path=outputs_path,
        bench_dir=bench_dir,
        use_llm_judge=args.use_llm_judge,
        judge_cfg=judge_cfg,
    )

    postprocessor.save_report(metrics, report_path)


if __name__ == "__main__":
    main()
