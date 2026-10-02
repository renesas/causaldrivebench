"""
Qwen2.5-VL post-processing for CausalDriveBench.

Parses "Answer: Yes/No" and "Answer: A/B/C/D" from model output.
Handles optional <think>...</think> reasoning blocks.

Usage (host):
    python evaluation/models/qwen25vl/postprocess.py \\
        --config evaluation/eval_config.local.yaml \\
        --run-dir /path/to/cdb_outputs/qwen25vl_YYYYMMDD_HHMMSS

Usage (Docker):
    uv run python postprocess.py \\
        --run-dir /workspace/data/cdb_outputs/qwen25vl_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_postprocess import ModelPostprocessor
from evaluation.common.utils.exceptions import EmptyResponseException


class Qwen25VLPostprocessor(ModelPostprocessor):

    def parse_answer(self, raw_output, question: dict) -> str:
        text = self._extract_text(raw_output)
        if not text or not text.strip():
            raise EmptyResponseException(f"Empty response for q={question.get('id', '?')}")

        answer_format = question.get("answer_format", "binary")
        # Strip <think>...</think> blocks (Qwen3-style, handled here for safety)
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        inner = think_match.group(1) if think_match else text
        targets = [inner, text]

        if answer_format == "binary":
            return self._parse_binary(targets)
        return self._parse_mcq(targets)

    def _parse_binary(self, targets: list) -> str:
        for text in targets:
            for pat in [
                r"\bAnswer\s*:\s*(Yes|No)\b",
                r"answer\s+is\s+(Yes|No)\b",
                r"^(Yes|No)[.,\s]",
                r"\b(Yes|No)\b",
            ]:
                m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
                if m:
                    return m.group(1).capitalize()
        return ""

    def _parse_mcq(self, targets: list) -> str:
        for text in targets:
            for pat in [
                r"\bAnswer\s*:\s*([A-D])\b",
                r"answer\s+is\s+([A-D])\b",
                r"\bOption\s+([A-D])\b",
                r"\b([A-D])[).]\s",
                r"^\s*([A-D])\s*$",
                r"\b([A-D])\b",
            ]:
                m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
                if m:
                    return m.group(1).upper()
        return ""


def main() -> None:
    args = ModelPostprocessor.parse_postprocess_args(
        "Post-process Qwen2.5-VL outputs and compute benchmark metrics"
    )
    Qwen25VLPostprocessor(model_cfg={}).run_postprocess(args, "Qwen2.5-VL")


if __name__ == "__main__":
    main()
