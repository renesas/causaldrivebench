"""
SimLingo post-processing script.

Parses raw InternVL2 output to extract structured answers, generates
per-sample report.json files alongside outputs.jsonl, then aggregates
dataset and run-level reports.

Reads from the per-sample output structure written by inference.py:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/outputs.jsonl

Writes:
  <run_dir>/<causal_dataset_name>/<scene_id>/<sample_id>/report.json  (Tier 1)
  <run_dir>/<causal_dataset_name>/report.json                          (Tier 2)
  <run_dir>/report.json                                                 (Tier 3)

Usage:
    python postprocess.py --run-dir /path/to/run
    python postprocess.py --run-dir ... --use-llm-judge
    python postprocess.py --run-dir ... --use-llm-judge --judge-batch-api
    python postprocess.py --run-dir ... --force
    python postprocess.py --all-runs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_postprocess import ModelPostprocessor


class SimLingoPostprocessor(ModelPostprocessor):
    """
    Parses SimLingo (InternVL2) output text.

    InternVL2 may produce <think>...</think> reasoning or direct answers.
    Standard cascading regex extracts Yes/No or A–D.

    With --judge-batch-api all LLM judge calls are submitted as a single
    Anthropic Message Batch (concurrent, ~50 % cheaper) instead of the
    sequential per-call path.
    """

    def __init__(self, model_cfg: dict, use_batch_api: bool = False) -> None:
        super().__init__(model_cfg)
        self._use_batch_api = use_batch_api

    # ------------------------------------------------------------------
    # Batch-API judge path
    # ------------------------------------------------------------------

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
        """Override to inject batch-API judge path when --use-batch-api is set."""
        if not (self._use_batch_api and judge_cfg):
            return super().process_dataset(
                run_dir, causal_dataset_name, bench_dir, judge_cfg,
                schema_version, enabled_metrics,
                mode=mode, subset_size=subset_size, scene=scene, force=force,
            )

        # ── Batch API path ─────────────────────────────────────────────
        from collections import defaultdict

        from evaluation.common.dataset import BenchmarkDataset
        from evaluation.common.llm_judge import LLMJudge

        judge = LLMJudge(
            provider=judge_cfg.get("provider", "anthropic"),
            model=judge_cfg.get("model", ""),
            api_key_env=judge_cfg.get("api_key_env", "ANTHROPIC_API_KEY"),
            azure_endpoint_env=judge_cfg.get("azure_endpoint_env", ""),
            azure_api_version=judge_cfg.get("azure_api_version", "2024-12-01-preview"),
        )

        # Pass 1: generate per-sample reports without judge scores.
        # force=False preserves existing reports that already have judge scores.
        ok = super().process_dataset(
            run_dir, causal_dataset_name, bench_dir, None,
            schema_version, enabled_metrics,
            mode=mode, subset_size=subset_size, scene=scene, force=force,
        )
        if not ok:
            return False

        # Collect judge inputs from all sample reports that are missing scores.
        output_base = run_dir / causal_dataset_name
        dataset_obj = BenchmarkDataset(bench_dir=bench_dir)
        sample_ids = dataset_obj.get_sample_ids(mode=mode, subset_size=subset_size, scene=scene)

        all_judge_inputs: List[Dict[str, Any]] = []
        sample_meta: List[Dict[str, Any]] = []

        for sid in sample_ids:
            scene_id  = sid["scene_id"]
            sample_id = sid["sample_id"]
            report_path = output_base / scene_id / sample_id / "report.json"

            if not report_path.exists():
                continue

            try:
                report = json.loads(report_path.read_text())
            except Exception:
                continue

            qa_list = report.get("qa_results", [])
            if qa_list and all("llm_judge_score" in q for q in qa_list):
                print(f"[Batch] Skip (already judged): {scene_id}/{sample_id}")
                continue

            for qa in qa_list:
                if "llm_judge_score" in qa:
                    continue
                judge_input = judge.build_judge_input(qa)
                if judge_input:
                    all_judge_inputs.append(judge_input)
                    sample_meta.append({
                        "scene_id":    scene_id,
                        "sample_id":   sample_id,
                        "qa_index":    qa.get("question_id"),
                        "report_path": str(report_path),
                    })

        if not all_judge_inputs:
            print("[Batch] Nothing to judge.")
            return True

        print(f"[Batch] Submitting {len(all_judge_inputs)} requests via Message Batches API …")
        batch_path = run_dir / "judge_batch_results.json"
        scored = judge.score_batch_api(all_judge_inputs, save_path=str(batch_path))

        # Pass 2: patch judge scores back into per-sample report.json files.
        by_report: Dict[str, List] = defaultdict(list)
        for meta, result in zip(sample_meta, scored):
            by_report[meta["report_path"]].append((meta["qa_index"], result))

        for rpath, items in by_report.items():
            try:
                report = json.loads(Path(rpath).read_text())
                idx_map = {
                    qa.get("question_id"): i
                    for i, qa in enumerate(report.get("qa_results", []))
                }
                for qa_id, result in items:
                    i = idx_map.get(qa_id)
                    if i is not None:
                        report["qa_results"][i]["llm_judge_score"] = result
                Path(rpath).write_text(json.dumps(report, indent=2))
            except Exception as exc:
                print(f"[Batch] WARNING: failed to update {rpath}: {exc}")

        # Pass 3: re-aggregate dataset report with judge scores now present.
        # All sample reports are fresh so they're skipped; only the dataset
        # report is regenerated, picking up the patched scores.
        return super().process_dataset(
            run_dir, causal_dataset_name, bench_dir, None,
            schema_version, enabled_metrics,
            mode=mode, subset_size=subset_size, scene=scene, force=False,
        )

    # ------------------------------------------------------------------
    # Trajectory evaluation postprocess
    # ------------------------------------------------------------------

    def run_trajectory_postprocess(self, args, model_label: str = "SimLingo") -> None:
        """
        Standalone entrypoint for trajectory evaluation postprocessing.

        Reads trajectory_prompts.jsonl + trajectory_outputs.jsonl written by
        inference.py --trajectory, computes BTE/CTE ADE metrics, writes
        trajectory_report.json per sample and trajectory_aggregate.json at the
        dataset level.

        Usage:
            python postprocess.py --trajectory-eval --run-dir <run_dir>
        """
        import os
        from evaluation.common.utils.config_reader import CDBConfig

        cfg       = CDBConfig(config_path=getattr(args, "config", None))
        run_dir   = Path(args.run_dir)
        bench_dir = (
            Path(args.bench_dir)
            if getattr(args, "bench_dir", "")
            else cfg.get_bench_dir(getattr(args, "dataset", "nuscenes"))
        )
        causal_dataset_name = bench_dir.name

        ok = self.process_trajectory_outputs(
            run_dir=run_dir,
            causal_dataset_name=causal_dataset_name,
            bench_dir=bench_dir,
            mode=getattr(args, "mode", "full"),
            subset_size=getattr(args, "subset_size", None),
            scene=getattr(args, "scene", None),
            force=getattr(args, "force", False),
        )
        if ok:
            self.aggregate_trajectory_reports(run_dir, causal_dataset_name)
            print(f"[{model_label}] Trajectory postprocess complete. Reports in {run_dir}")
        else:
            print(f"[{model_label}] No trajectory reports generated — run inference.py --trajectory first.")


def _build_trajectory_arg_parser() -> "argparse.ArgumentParser":
    import argparse
    p = argparse.ArgumentParser(
        description="Post-process SimLingo trajectory outputs (BTE + CTE ADE metrics)"
    )
    p.add_argument("--run-dir",      required=True, help="Run directory written by inference.py.")
    p.add_argument("--bench-dir",    default="",    help="Dataset bench dir (auto-detected if omitted).")
    p.add_argument("--dataset",      default="nuscenes", choices=["nuscenes", "argoverse2", "openscene"])
    p.add_argument("--mode",         default="full", choices=["full", "subset", "single"])
    p.add_argument("--subset-size",  type=int, default=None)
    p.add_argument("--scene",        default=None)
    p.add_argument("--force",        action="store_true", default=False,
                   help="Re-process samples that already have trajectory_report.json.")
    p.add_argument("--config",       default=None, help="Path to eval_config.yaml.")
    return p


def main() -> None:
    import argparse
    import sys

    # Dispatch on first positional-style flag to keep a single entry point
    if "--trajectory-eval" in sys.argv:
        sys.argv.remove("--trajectory-eval")
        args = _build_trajectory_arg_parser().parse_args()
        SimLingoPostprocessor(model_cfg={}).run_trajectory_postprocess(args, "SimLingo")
        return

    parser = ModelPostprocessor.build_arg_parser(
        "Post-process SimLingo outputs and compute benchmark metrics"
    )
    args = parser.parse_args()
    SimLingoPostprocessor(model_cfg={}, use_batch_api=args.judge_batch_api).run_postprocess(
        args, "SimLingo"
    )


if __name__ == "__main__":
    main()
