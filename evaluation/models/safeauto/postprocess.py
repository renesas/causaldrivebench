"""
SafeAuto post-processing script for CausalDriveBench.

SafeAuto outputs free-form text like:
  "Yes, because the traffic light ahead is red and the ego must stop."
  "The answer is A. The vehicle in front is decelerating..."
  "B) The pedestrian on the crosswalk..."

The template regex parsers handle all of these patterns correctly.

Usage:
    python postprocess.py --run-dir /workspace/cdb/outputs/safeauto_20260424_193636
    python postprocess.py --run-dir ... --bench-dir /path/to/causal_nuscenes
    python postprocess.py --run-dir ... --use-llm-judge --judge-batch-api --judge-provider anthropic
    python postprocess.py --all-runs
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_postprocess import ModelPostprocessor


class SafeAutoPostprocessor(ModelPostprocessor):

    def __init__(self, model_cfg: dict, use_batch_api: bool = False) -> None:
        super().__init__(model_cfg)
        self._use_batch_api = use_batch_api
    """
    Extracts MCQ letter or binary (Yes/No) answer from SafeAuto's free-form output.

    SafeAuto typically begins its answer with:
      "Answer: Yes/No/A/B/C/D" or starts directly with Yes/No/letter.
    The inherited regex parsers handle both patterns.
    """

    def _preprocess_text(self, text: str) -> str:
        """Extract text from dict output if needed."""
        if isinstance(text, dict):
            return text.get("text", "")
        return text

    @staticmethod
    def _parse_binary(targets: list) -> str:
        for text in targets:
            m = re.search(r"\bAnswer\s*[:\-]?\s*(Yes|No)\b", text, re.IGNORECASE)
            if m:
                return m.group(1).capitalize()
            m = re.match(r"^\s*(Yes|No)\b", text, re.IGNORECASE)
            if m:
                return m.group(1).capitalize()
            m = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
            if m:
                return m.group(1).capitalize()
        return ""

    @staticmethod
    def _parse_mcq(targets: list) -> str:
        for text in targets:
            m = re.search(r"\bAnswer\s*[:\-]?\s*([A-D])\b", text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
            m = re.search(r"answer\s+is\s+([A-D])\b", text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
            m = re.match(r"^\s*([A-D])[).\s]", text)
            if m:
                return m.group(1).upper()
            m = re.match(r"^\s*([A-D])\s*$", text)
            if m:
                return m.group(1).upper()
            m = re.search(r"\b([A-D])\b", text)
            if m:
                return m.group(1).upper()
        return ""


    def process_dataset(
        self,
        run_dir: Path,
        causal_dataset_name: str,
        bench_dir: Path,
        judge_cfg: Optional[dict],
        schema_version: str,
        enabled_metrics: List[str],
        mode: str = "full",
        subset_size: Optional[int] = None,
        scene: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """
        Override: when --use-batch-api is set and judge is enabled, collect all
        questions across every sample, submit ONE Anthropic batch, then write
        all reports.  Falls back to base (sequential) otherwise.
        """
        if not self._use_batch_api or not judge_cfg:
            return super().process_dataset(
                run_dir, causal_dataset_name, bench_dir, judge_cfg,
                schema_version, enabled_metrics, mode, subset_size, scene, force,
            )

        from evaluation.common.dataset import BenchmarkDataset
        from evaluation.common.llm_judge import LLMJudge
        from evaluation.common.report_generator import (
            generate_dataset_report,
            generate_sample_report,
            is_sample_report_fresh,
        )

        run_name    = run_dir.name
        output_base = run_dir / causal_dataset_name

        try:
            dataset_obj = BenchmarkDataset(bench_dir=bench_dir)
        except Exception:
            print(f"[Postprocess] ERROR creating BenchmarkDataset: {bench_dir}\n{traceback.format_exc()}")
            return False

        sample_ids = dataset_obj.get_sample_ids(mode=mode, subset_size=subset_size, scene=scene)
        print(f"[Postprocess] {run_name}/{causal_dataset_name}: {len(sample_ids)} sample(s) (batch-api mode)")

        # ── Pass 1: parse all outputs, collect judge inputs ────────────────
        # all_sample_data[i] = (scene_id, sample_id, report_path, qa_results, judge_slice)
        all_sample_data = []
        judge_inputs_flat: list = []   # flat list sent to batch API
        n_fresh = n_missing = 0

        for sid in sample_ids:
            scene_id  = sid["scene_id"]
            sample_id = sid["sample_id"]
            outputs_file       = output_base / scene_id / sample_id / "outputs.jsonl"
            sample_report_path = output_base / scene_id / sample_id / "report.json"

            if not force and is_sample_report_fresh(sample_report_path, schema_version):
                n_fresh += 1
                continue

            if not outputs_file.exists():
                n_missing += 1
                continue

            sample_bench_dir = bench_dir / scene_id / sample_id
            qa_index = self._load_sample_qa_index(sample_bench_dir)
            if not qa_index:
                print(f"[Postprocess] WARNING: No questions found in {sample_bench_dir}")
                n_missing += 1
                continue

            qa_results = self.parse_sample_outputs(outputs_file, qa_index)

            # Collect judge inputs for this sample
            scoreable = [
                r for r in qa_results
                if self._extract_text(r.get("raw_output", r.get("raw_output_text", ""))).strip()
            ]
            start = len(judge_inputs_flat)
            for r in scoreable:
                judge_inputs_flat.append({
                    "question":       r.get("question") or r.get("question_text", ""),
                    "ground_truth":   r.get("ground_truth", ""),
                    "model_response": self._extract_text(
                        r.get("raw_output", r.get("raw_output_text", ""))
                    ),
                })
            end = len(judge_inputs_flat)

            all_sample_data.append(
                (scene_id, sample_id, sample_report_path, qa_results, scoreable, start, end)
            )

        print(
            f"[Postprocess]   {len(all_sample_data)} samples to process, "
            f"{len(judge_inputs_flat)} judge inputs, "
            f"{n_fresh} fresh (skipped), {n_missing} missing outputs"
        )

        # ── Batch API call (one submission for everything) ─────────────────
        judge_scores_flat: list = []
        if judge_inputs_flat:
            judge = LLMJudge(
                provider=judge_cfg.get("provider", "anthropic"),
                model=judge_cfg.get("model", "claude-sonnet-4-6"),
                api_key_env=judge_cfg.get("api_key_env", "ANTHROPIC_API_KEY"),
            )
            judge_scores_flat = judge.score_batch_api(judge_inputs_flat)

        # ── Pass 2: attach scores and write per-sample reports ─────────────
        n_generated = 0
        for scene_id, sample_id, sample_report_path, qa_results, scoreable, start, end in all_sample_data:
            scores_slice = judge_scores_flat[start:end] if judge_scores_flat else []
            for r, score in zip(scoreable, scores_slice):
                r["llm_judge_score"] = score

            generate_sample_report(
                run_name=run_name, dataset=causal_dataset_name,
                scene_id=scene_id, sample_id=sample_id,
                qa_results=qa_results, report_path=sample_report_path,
                schema_version=schema_version, enabled_metrics=enabled_metrics,
            )
            n_generated += 1
            print(f"[Postprocess]   wrote report: {scene_id}/{sample_id}", flush=True)

        print(f"[Postprocess]   {n_generated} reports written")

        if n_generated + n_fresh == 0:
            print(f"[Postprocess] WARNING: No outputs found under {output_base}.")
            return False

        generate_dataset_report(
            run_name=run_name, dataset=causal_dataset_name,
            dataset_dir=output_base,
            report_path=output_base / "report.json",
            schema_version=schema_version, enabled_metrics=enabled_metrics,
        )
        return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = ModelPostprocessor.build_arg_parser(
        "Post-process SafeAuto outputs and compute benchmark metrics"
    )
    args = parser.parse_args()
    SafeAutoPostprocessor(model_cfg={}, use_batch_api=args.judge_batch_api).run_postprocess(
        args, "SafeAuto"
    )


if __name__ == "__main__":
    main()
