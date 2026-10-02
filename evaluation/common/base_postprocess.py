"""
Abstract base class for model output post-processing.

Each model subclasses ModelPostprocessor and may override:
  - ``_preprocess_text(text)`` — strip model-specific tokens before standard parsing
  - ``parse_answer(raw_output, question)`` — full custom extraction if needed

Concrete helpers provided by this base class
--------------------------------------------
Answer parsing (shared, no model-specific code):
  - ``parse_answer(raw_output, question)``
        Standard flow: _preprocess_text → strip <think> → _parse_binary / _parse_mcq.
        Override _preprocess_text() for model-specific tokens; override parse_answer()
        entirely for fundamentally different output formats.
  - ``_preprocess_text(text)`` → text  [no-op default]
  - ``_parse_binary(targets)`` / ``_parse_mcq(targets)``
        Cascading regex extractors shared by all models.

Tier-hierarchy postprocessing (full reusable pipeline):
  - ``parse_sample_outputs(outputs_path, qa_index)``
        Parse one sample's outputs.jsonl into a qa_results list (Tier 1 input).
  - ``process_dataset(run_dir, causal_dataset_name, bench_dir, …)``
        Write Tier 1 + Tier 2 reports for one (run, dataset) pair.
  - ``build_judge_cfg(args, cfg)``  [static]
        Build LLM judge config dict from CLI args + eval_config.yaml.
  - ``build_arg_parser(description)``  [static]
        Return a standard argparse.ArgumentParser for postprocess scripts.
  - ``parse_postprocess_args(description)``  [static]
        Parse CLI args using the standard argument set.
  - ``run_postprocess(args, model_label)``
        Full Tier 1–4 orchestration. Call from a model's main() after
        instantiating the postprocessor and parsing args.

Legacy helpers (kept for backward compatibility):
  - ``compute_metrics(outputs_path, bench_dir, …)``
  - ``save_report(metrics, report_path)``
  - ``_load_qa_index(bench_dir)``
  - ``_load_sample_qa_index(sample_bench_dir)``
  - ``_run_llm_judge(records, judge_cfg, score_all)``
  - ``_extract_text(raw_output)``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.common.metrics import (
    binary_classification_metrics,
    mcq_accuracy,
    per_metadata_accuracy,
    per_type_accuracy,
    vqa_answer_stats,
)

logger = logging.getLogger(__name__)

# Filename → internal QA type mapping (mirrors dataset.py)
_QA_FILE_TO_TYPE: Dict[str, str] = {
    "active_qa.json":     "ladder",
    "dormant_qa.json":    "dormant",
    "distractor_qa.json": "distractor",
}


class ModelPostprocessor(ABC):
    """
    Parses raw model outputs and computes benchmark metrics.

    Subclass this and either:
      - Use the inherited parse_answer() as-is (standard <think> + cascading regex).
      - Override _preprocess_text() to strip model-specific tokens first.
      - Override parse_answer() entirely for non-standard output formats.
    """

    def __init__(self, model_cfg: dict) -> None:
        self.model_cfg = model_cfg

    # ------------------------------------------------------------------
    # Answer parsing — override _preprocess_text for model-specific tokens
    # ------------------------------------------------------------------

    def _preprocess_text(self, text: str) -> str:
        """Strip model-specific tokens before standard answer parsing. No-op by default."""
        return text

    def parse_answer(self, raw_output, question: dict) -> str:
        """
        Extract the predicted answer from raw model output.

        Standard flow:
          1. Extract text string from raw_output (dict or plain string).
          2. Call _preprocess_text() for model-specific token cleanup.
          3. Strip <think>…</think> wrapper (reasoning-model style).
          4. Dispatch to _parse_binary() or _parse_mcq() by answer_format.

        Returns "Yes"/"No" for binary questions, "A"–"D" for MCQ, "" on failure.
        Raises EmptyResponseException when the extracted text is empty.
        """
        import re
        from evaluation.common.utils.exceptions import EmptyResponseException

        text = self._extract_text(raw_output)
        if not text or not text.strip():
            raise EmptyResponseException(
                f"Empty response for question id={question.get('id', '?')}"
            )

        answer_format = question.get("answer_format", "binary")
        text = self._preprocess_text(text.strip())

        think_inner = text
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        if think_match:
            think_inner = think_match.group(1)

        search_targets = [think_inner, text]
        if answer_format == "binary":
            return self._parse_binary(search_targets)
        return self._parse_mcq(search_targets)

    def _parse_binary(self, targets: list) -> str:
        """Extract Yes/No from text with cascading patterns."""
        import re
        for text in targets:
            m = re.search(r"\bAnswer\s*:\s*(Yes|No)\b", text, re.IGNORECASE)
            if m:
                return m.group(1).capitalize()
            m = re.search(r"answer\s+is\s+(Yes|No)\b", text, re.IGNORECASE)
            if m:
                return m.group(1).capitalize()
            m = re.search(r"^(Yes|No)[.,\s]", text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).capitalize()
            m = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
            if m:
                return m.group(1).capitalize()
        return ""

    def _parse_mcq(self, targets: list) -> str:
        """Extract MCQ letter (A–D) from text with cascading patterns."""
        import re
        for text in targets:
            m = re.search(r"\bAnswer\s*:\s*([A-D])\b", text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
            m = re.search(r"answer\s+is\s+([A-D])\b", text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
            m = re.search(r"\bOption\s+([A-D])\b", text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
            m = re.search(r"\b([A-D])[).]\s", text)
            if m:
                return m.group(1).upper()
            m = re.match(r"^\s*([A-D])\s*$", text, re.MULTILINE)
            if m:
                return m.group(1).upper()
            m = re.search(r"\b([A-D])\b", text)
            if m:
                return m.group(1).upper()
        return ""

    # ------------------------------------------------------------------
    # Tier-hierarchy pipeline helpers
    # ------------------------------------------------------------------

    def parse_sample_outputs(self, outputs_path: Path, qa_index: dict) -> List[dict]:
        """
        Parse outputs.jsonl for one sample and return a qa_results list.

        Each entry has the fields expected by report_generator:
            question_id, qa_type, answer_format, question_text,
            predicted, ground_truth, correct, raw_output_text,
            question_type, graph_structure, rung, causal_status
            Optional: inference_time_s
        """
        qa_results: List[dict] = []

        with open(outputs_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out = json.loads(line)
                except json.JSONDecodeError:
                    continue

                qid = out.get("question_id")
                question = qa_index.get(qid)
                if not question:
                    continue

                raw_output = out.get("raw_output", "")
                try:
                    predicted = self.parse_answer(raw_output, question)
                except Exception:
                    predicted = ""

                ground_truth = question.get("correct_answer", question.get("answer", ""))
                qa_type = question.get("_qa_type", "unknown")
                correct = predicted.strip().upper() == ground_truth.strip().upper()
                meta = question.get("meta", {}) if isinstance(question.get("meta"), dict) else {}

                # Derive question_type from question_id prefix when not explicitly set.
                # E.g. "CA1" → "CA", "NDE2" → "NDE" (strip trailing digits).
                import re as _re
                question_type = question.get("question_type", "")
                if not question_type and qid:
                    question_type = _re.sub(r"\d+$", "", qid)

                result: dict = {
                    "question_id":     qid,
                    "qa_type":         qa_type,
                    "answer_format":   question.get("answer_format", "binary"),
                    "question_text":   question.get("question", ""),
                    "options":         question.get("options", []),
                    "reasoning":       question.get("reasoning", ""),
                    "predicted":       predicted,
                    "ground_truth":    ground_truth,
                    "correct":         correct,
                    "raw_output_text": self._extract_text(raw_output),
                    "question_type":   question_type,
                    "graph_structure": question.get("graph_structure", ""),
                    "rung":            str(question.get("rung", "")) if question.get("rung") is not None else "",
                    "causal_status":   meta.get("causal_status", ""),
                }
                if "inference_time_s" in out:
                    result["inference_time_s"] = out["inference_time_s"]

                qa_results.append(result)

        return qa_results

    @staticmethod
    def build_judge_cfg(args: argparse.Namespace, cfg) -> Optional[dict]:
        """
        Build the LLM judge config dict from CLI args + CDBConfig.

        Returns None if --use-llm-judge is not set.
        """
        if not args.use_llm_judge:
            return None

        from evaluation.common.llm_judge import LLMJudge

        cfg_judge = cfg.judge_config
        provider  = args.judge_provider
        model     = args.judge_model or cfg_judge.get("model", "")
        if not model:
            model = LLMJudge._PROVIDER_DEFAULT_MODELS.get(provider, "")

        api_key_env = cfg_judge.get(
            "api_key_env",
            LLMJudge._PROVIDER_DEFAULT_KEY_ENVS.get(provider, ""),
        )
        azure_endpoint_env = cfg_judge.get(
            "azure_endpoint_env",
            LLMJudge._PROVIDER_DEFAULT_ENDPOINT_ENVS.get(provider, ""),
        )
        azure_api_version = cfg_judge.get("azure_api_version", "2024-12-01-preview")

        use_batch_api = getattr(args, "judge_batch_api", False)
        concurrency   = getattr(args, "judge_concurrency", 20)

        # Auto-configure azure_foundry as fallback when primary is not already azure_foundry
        # and the required env vars are present.  This handles expired primary API keys
        # transparently: the judge retries on 401 using the azure_foundry credentials.
        fallback_provider = ""
        fallback_model = ""
        if provider != "azure_foundry":
            foundry_key = os.environ.get("ANTHROPIC_FOUNDRY_API_KEY", "")
            foundry_endpoint = (
                os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE", "")
                or os.environ.get("FOUNDRY_BASE_URL", "")
            )
            if foundry_key and foundry_endpoint:
                fallback_provider = "azure_foundry"
                fallback_model = model  # same model name (claude-opus-4-6 works on both)
                print(f"[Postprocess] LLM judge: azure_foundry fallback enabled on auth error")

        print(f"[Postprocess] LLM judge: provider={provider} model={model} "
              f"concurrency={concurrency} batch_api={use_batch_api}"
              + (f" fallback={fallback_provider}" if fallback_provider else ""))
        return {
            "provider":           provider,
            "model":              model,
            "api_key_env":        api_key_env,
            "azure_endpoint_env": azure_endpoint_env,
            "azure_api_version":  azure_api_version,
            "concurrency":        concurrency,
            "use_batch_api":      use_batch_api,
            "fallback_provider":  fallback_provider,
            "fallback_model":     fallback_model,
        }

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
        Process all samples in one (run, dataset) pair and write Tier 1 + 2 reports.

        Uses a two-pass approach when a judge is configured:
          Pass 1 — parse all sample outputs, collect judge inputs.
          Bulk judge — score all questions in one concurrent/batch call.
          Pass 2 — attach scores and write per-sample reports.

        Returns True if any sample reports were generated or found fresh.
        Returns False (with a warning) if no outputs exist.
        """
        from evaluation.common.dataset import BenchmarkDataset
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
        print(f"[Postprocess] {run_name}/{causal_dataset_name}: {len(sample_ids)} sample(s) (mode={mode})")

        n_fresh = n_missing = 0

        # ── Pass 1: parse all samples, collect judge inputs ───────────────────
        # pending_samples: list of (scene_id, sample_id, qa_results, report_path)
        # judge_flat: flat list of (pending_idx, qa_result_idx, judge_input_dict)
        pending_samples: List[tuple] = []
        judge_flat: List[tuple] = []

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

            if judge_cfg and qa_results:
                pi = len(pending_samples)
                for qi, r in enumerate(qa_results):
                    text = self._extract_text(r.get("raw_output", r.get("raw_output_text", "")))
                    if text.strip():
                        judge_flat.append((pi, qi, {
                            "question":       r.get("question_text", ""),
                            "ground_truth":   r.get("ground_truth", ""),
                            "model_response": text,
                        }))

            pending_samples.append((scene_id, sample_id, qa_results, sample_report_path))

        # ── Bulk judge ─────────────────────────────────────────────────────────
        if judge_cfg and judge_flat:
            judge_inputs = [jf[2] for jf in judge_flat]
            print(f"[Postprocess] Running LLM judge on {len(judge_inputs)} questions "
                  f"across {len(pending_samples)} samples …")
            scored = self._run_llm_judge_bulk(judge_inputs, judge_cfg)
            for (pi, qi, _), score in zip(judge_flat, scored):
                _, _, qa_results, _ = pending_samples[pi]
                qa_results[qi]["llm_judge_score"] = score

        # ── Pass 2: write reports ─────────────────────────────────────────────
        n_generated = 0
        for scene_id, sample_id, qa_results, sample_report_path in pending_samples:
            generate_sample_report(
                run_name=run_name, dataset=causal_dataset_name,
                scene_id=scene_id, sample_id=sample_id,
                qa_results=qa_results, report_path=sample_report_path,
                schema_version=schema_version, enabled_metrics=enabled_metrics,
            )
            n_generated += 1

        print(
            f"[Postprocess]   {n_generated} generated, {n_fresh} fresh (skipped), "
            f"{n_missing} missing outputs"
        )

        if n_generated + n_fresh == 0:
            print(
                f"[Postprocess] WARNING: No outputs found under {output_base}. "
                "Run inference.py first or check --run-dir."
            )
            return False

        generate_dataset_report(
            run_name=run_name, dataset=causal_dataset_name,
            dataset_dir=output_base,
            report_path=output_base / "report.json",
            schema_version=schema_version, enabled_metrics=enabled_metrics,
        )
        return True

    @staticmethod
    def build_arg_parser(description: str) -> argparse.ArgumentParser:
        """Return a standard argparse.ArgumentParser for postprocess scripts."""
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument(
            "--mode",
            choices=["full", "subset", "single"],
            default=os.environ.get("MODE", "full"),
        )
        parser.add_argument(
            "--subset-size",
            type=int,
            default=int(os.environ.get("SUBSET_SIZE", 0)) or None,
        )
        parser.add_argument(
            "--scene",
            default=os.environ.get("SCENE") or None,
        )
        parser.add_argument(
            "--dataset",
            default=os.environ.get("DATASET", "nuscenes"),
            choices=["nuscenes", "argoverse2", "openscene"],
        )
        parser.add_argument(
            "--bench-dir",
            default="",
            help="Dataset-specific bench directory. Defaults to CDBConfig.get_bench_dir(--dataset).",
        )
        parser.add_argument(
            "--run-dir",
            default=os.environ.get("RUN_DIR") or None,
            help="Run directory written by inference.py. Mutually exclusive with --all-runs.",
        )
        parser.add_argument(
            "--all-runs",
            action="store_true",
            default=False,
            help="Process every run directory found under cdb_outputs_dir.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            default=os.environ.get("FORCE", "").lower() == "true",
            help="Regenerate all sample reports even if schema_version already matches.",
        )
        parser.add_argument(
            "--use-llm-judge",
            action="store_true",
            default=os.environ.get("USE_LLM_JUDGE", "").lower() == "true",
        )
        parser.add_argument(
            "--judge-provider",
            default=os.environ.get("JUDGE_PROVIDER", "anthropic"),
            choices=["anthropic", "openai", "azure_foundry", "azure_openai"],
        )
        parser.add_argument(
            "--judge-model",
            default=os.environ.get("JUDGE_MODEL", ""),
        )
        parser.add_argument(
            "--judge-concurrency",
            type=int,
            default=int(os.environ.get("JUDGE_CONCURRENCY", "20")),
            help="Number of concurrent LLM judge requests (default 20). "
                 "Set to 1 for serial. Ignored when --judge-batch-api is set.",
        )
        parser.add_argument(
            "--judge-batch-api",
            action="store_true",
            default=os.environ.get("JUDGE_BATCH_API", "").lower() == "true",
            help="Use Anthropic Message Batches API (50%% cheaper; requires provider=anthropic).",
        )
        parser.add_argument(
            "--config",
            default=None,
            help="Path to eval_config.yaml (auto-detected if omitted)",
        )
        return parser

    @staticmethod
    def parse_postprocess_args(description: str) -> argparse.Namespace:
        """Parse standard postprocess CLI arguments."""
        return ModelPostprocessor.build_arg_parser(description).parse_args()

    def run_postprocess(self, args: argparse.Namespace, model_label: str) -> None:
        """
        Full Tier 1–4 orchestration. Call from a model's main() after
        instantiating the postprocessor and parsing args.

        Parameters
        ----------
        args        : parsed args from parse_postprocess_args() or build_arg_parser().
        model_label : human-readable model name printed in log lines.
        """
        from evaluation.common.html_reporter import generate_all_html
        from evaluation.common.report_generator import (
            generate_run_report,
            generate_root_aggregate,
        )
        from evaluation.common.utils.config_reader import CDBConfig
        from evaluation.common.utils.exceptions import PostprocessException

        if args.all_runs and args.run_dir:
            raise ValueError("--all-runs and --run-dir are mutually exclusive.")
        if not args.all_runs and not args.run_dir:
            raise ValueError("--run-dir is required (or use --all-runs to process all runs).")

        cfg = CDBConfig(config_path=args.config)
        schema_version  = cfg.schema_version
        enabled_metrics = cfg.report_metrics
        html_enabled    = cfg.html_enabled

        judge_cfg = self.build_judge_cfg(args, cfg)

        if args.all_runs:
            run_dirs = sorted(d for d in cfg.dir_cdb_outputs.iterdir() if d.is_dir())
            if not run_dirs:
                raise PostprocessException(f"No run directories found in {cfg.dir_cdb_outputs}")
            print(f"[Postprocess] All-runs mode: {len(run_dirs)} run(s) in {cfg.dir_cdb_outputs}")
        else:
            run_dirs = [Path(args.run_dir)]

        print("=" * 60)
        print(f"[Postprocess] {model_label} Post-processing")
        print(f"[Postprocess]   schema_version={schema_version} | force={args.force}")
        print(f"[Postprocess]   use_llm_judge={args.use_llm_judge}")
        print("=" * 60)

        processed_runs: List[Path] = []
        # Maps run_dir → {dataset_name: bench_dir} for HTML generation
        run_bench_dirs: Dict[Path, Dict[str, Path]] = {}

        for run_dir in run_dirs:
            run_name     = run_dir.name
            run_had_data = False
            bench_dirs_this_run: Dict[str, Path] = {}

            print(f"\n[Postprocess] ── Run: {run_name}")

            if args.all_runs:
                causal_dirs = sorted(
                    d for d in run_dir.iterdir()
                    if d.is_dir() and d.name.startswith("causal_")
                )
                if not causal_dirs:
                    print(f"[Postprocess] WARNING: No causal_* dirs found in {run_dir}; skipping.")
                    continue

                for causal_dir in causal_dirs:
                    dataset_key = causal_dir.name[len("causal_"):]
                    bench_dir   = cfg.get_bench_dir(dataset_key)
                    if not bench_dir.exists():
                        print(f"[Postprocess] WARNING: bench_dir not found: {bench_dir}; skipping.")
                        continue
                    ok = self.process_dataset(
                        run_dir, causal_dir.name, bench_dir, judge_cfg,
                        schema_version, enabled_metrics, force=args.force,
                    )
                    if ok:
                        run_had_data = True
                        bench_dirs_this_run[causal_dir.name] = bench_dir
            else:
                bench_dir = Path(args.bench_dir) if args.bench_dir else cfg.get_bench_dir(args.dataset)
                if not bench_dir.exists():
                    raise FileNotFoundError(f"bench_dir does not exist: {bench_dir}")
                ok = self.process_dataset(
                    run_dir, bench_dir.name, bench_dir, judge_cfg,
                    schema_version, enabled_metrics,
                    mode=args.mode, subset_size=args.subset_size,
                    scene=args.scene, force=args.force,
                )
                if ok:
                    run_had_data = True
                    bench_dirs_this_run[bench_dir.name] = bench_dir

            if not run_had_data:
                continue

            run_report = generate_run_report(
                run_name=run_name, run_dir=run_dir,
                report_path=run_dir / "report.json",
                schema_version=schema_version, enabled_metrics=enabled_metrics,
            )
            processed_runs.append(run_dir)
            run_bench_dirs[run_dir] = bench_dirs_this_run

            overall  = run_report.get("metrics", {}).get("overall", {})
            per_type = run_report.get("metrics", {}).get("per_qa_type", {})
            print(f"\n[Postprocess] {run_name}: "
                  f"{overall.get('accuracy', 0.0):.2%} overall (n={overall.get('n', 0)})")
            for qt, vals in per_type.items():
                print(f"[Postprocess]   {qt:<20}: {vals.get('accuracy', 0.0):.2%} (n={vals.get('n', 0)})")

        if not processed_runs:
            raise PostprocessException("No runs were successfully processed.")

        if html_enabled:
            root_aggregate = generate_root_aggregate(cfg.dir_cdb_outputs, schema_version)
            for run_dir in processed_runs:
                generate_all_html(
                    run_dir.name, cfg.dir_cdb_outputs, cfg.dir_cdb_reports,
                    root_aggregate, bench_dirs=run_bench_dirs.get(run_dir),
                )
            print(f"\n[Postprocess] HTML reports → {cfg.dir_cdb_reports / 'index.html'}")

        print(f"\n[Postprocess] Done. {len(processed_runs)} run(s) processed.")

    # ------------------------------------------------------------------
    # Trajectory evaluation pipeline
    # ------------------------------------------------------------------

    def parse_trajectory_output(self, raw_output, prompt_record: dict) -> tuple:
        """
        Extract waypoints (and optionally a VQA answer) from raw model output.

        Override in model-specific postprocessors if the output format differs.
        Default implementation delegates to trajectory_eval.parse_waypoints_from_text().

        Parameters
        ----------
        raw_output     : raw_output field from trajectory_outputs.jsonl
        prompt_record  : matching prompt record (contains gt_waypoints, eval_type, etc.)

        Returns
        -------
        (waypoints_or_None, vqa_answer_or_None)
        """
        from evaluation.common.trajectory_eval import parse_waypoints_from_text
        text = self._extract_text(raw_output)
        return parse_waypoints_from_text(text)

    def process_trajectory_outputs(
        self,
        run_dir: "Path",
        causal_dataset_name: str,
        bench_dir: "Path",
        schema_version: str = "1.0",
        mode: str = "full",
        subset_size: "Optional[int]" = None,
        scene: "Optional[str]" = None,
        force: bool = False,
    ) -> bool:
        """
        Process trajectory_outputs.jsonl for each sample and write trajectory_report.json.

        Reads trajectory_prompts.jsonl (for GT waypoints) and
        trajectory_outputs.jsonl (for model predictions) from the per-sample
        output directory.  Writes trajectory_report.json alongside them.

        Returns True if at least one trajectory report was generated.
        """
        from evaluation.common.dataset import BenchmarkDataset
        from evaluation.common.trajectory_eval import (
            build_trajectory_report,
            evaluate_bte_output,
            evaluate_cte_output,
        )

        run_name    = run_dir.name
        output_base = run_dir / causal_dataset_name

        try:
            dataset_obj = BenchmarkDataset(bench_dir=bench_dir)
        except Exception:
            print(f"[TrajPostprocess] ERROR creating BenchmarkDataset: {bench_dir}\n"
                  f"{traceback.format_exc()}")
            return False

        sample_ids = dataset_obj.get_sample_ids(
            mode=mode, subset_size=subset_size, scene=scene
        )
        print(
            f"[TrajPostprocess] {run_name}/{causal_dataset_name}: "
            f"{len(sample_ids)} sample(s)"
        )

        n_generated = n_missing = n_fresh = 0

        for sid in sample_ids:
            scene_id  = sid["scene_id"]
            sample_id = sid["sample_id"]
            sample_dir = output_base / scene_id / sample_id

            traj_prompts_file = sample_dir / "trajectory_prompts.jsonl"
            traj_outputs_file = sample_dir / "trajectory_outputs.jsonl"
            traj_report_path  = sample_dir / "trajectory_report.json"

            if not force and traj_report_path.exists():
                n_fresh += 1
                continue

            if not traj_outputs_file.exists():
                logger.info(
                    "[TrajPostprocess] No trajectory_outputs.jsonl for %s/%s — skipping",
                    scene_id, sample_id,
                )
                n_missing += 1
                continue

            if not traj_prompts_file.exists():
                logger.warning(
                    "[TrajPostprocess] trajectory_prompts.jsonl missing for %s/%s",
                    scene_id, sample_id,
                )
                n_missing += 1
                continue

            # --- Load prompt records (GT waypoints + metadata) ---------------
            prompt_index: dict = {}
            with open(traj_prompts_file) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        prompt_index[rec["question_id"]] = rec
                    except (json.JSONDecodeError, KeyError):
                        pass

            # --- Load VQA outputs for original answers (from outputs.jsonl) --
            vqa_answers: dict = {}
            vqa_outputs_file = sample_dir / "outputs.jsonl"
            if vqa_outputs_file.exists():
                sample_bench_dir = bench_dir / scene_id / sample_id
                qa_index = self._load_sample_qa_index(sample_bench_dir)
                with open(vqa_outputs_file) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            out = json.loads(line)
                            qid = out.get("question_id", "")
                            question = qa_index.get(qid)
                            if question:
                                try:
                                    ans = self.parse_answer(out.get("raw_output", ""), question)
                                except Exception:
                                    ans = ""
                                vqa_answers[qid] = ans
                        except json.JSONDecodeError:
                            pass

            # --- Load trajectory outputs ------------------------------------
            traj_outputs: dict = {}
            with open(traj_outputs_file) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        traj_outputs[rec["question_id"]] = rec
                    except (json.JSONDecodeError, KeyError):
                        pass

            # --- Evaluate BTE -----------------------------------------------
            bte_result: Optional[dict] = None
            bte_prompt = prompt_index.get("traj_BTE")
            bte_output = traj_outputs.get("traj_BTE")
            if bte_prompt and bte_output:
                raw_text = self._extract_text(bte_output.get("raw_output", ""))
                gt_wps   = bte_prompt.get("gt_waypoints", [])
                bte_result = evaluate_bte_output(
                    raw_text=raw_text,
                    gt_waypoints=gt_wps,
                    question_id="traj_BTE",
                )
                logger.info(
                    "[TrajPostprocess] %s/%s BTE: parse=%s ade=%s grade=%s",
                    scene_id, sample_id,
                    bte_result.get("parse_success"),
                    bte_result.get("ade"),
                    bte_result.get("analysis", {}).get("grade"),
                )
            else:
                logger.info(
                    "[TrajPostprocess] %s/%s BTE prompt/output missing", scene_id, sample_id
                )

            # --- Evaluate CTE -----------------------------------------------
            cte_results: list = []
            for question_id, prompt_rec in prompt_index.items():
                if prompt_rec.get("eval_type", "") == "BTE":
                    continue  # already handled above
                traj_out = traj_outputs.get(question_id)
                if not traj_out:
                    logger.info(
                        "[TrajPostprocess] No output for %s/%s q=%s — skipping",
                        scene_id, sample_id, question_id,
                    )
                    continue

                raw_text = self._extract_text(traj_out.get("raw_output", ""))
                gt_wps   = prompt_rec.get("gt_waypoints", [])
                eval_type = prompt_rec.get("eval_type", "CTE")
                parent_qid = prompt_rec.get("parent_question_id", "")
                original_vqa = vqa_answers.get(parent_qid)

                cte_result = evaluate_cte_output(
                    raw_text=raw_text,
                    gt_waypoints=gt_wps,
                    eval_type=eval_type,
                    question_id=question_id,
                    parent_question_id=parent_qid,
                    original_vqa_answer=original_vqa,
                    bte_result=bte_result,
                )
                cte_results.append(cte_result)
                logger.info(
                    "[TrajPostprocess] %s/%s %s q=%s: parse=%s ade=%s flags=%s",
                    scene_id, sample_id, eval_type, parent_qid,
                    cte_result.get("parse_success"),
                    cte_result.get("ade"),
                    cte_result.get("analysis", {}).get("flags", []),
                )

            # --- Build and save trajectory_report.json ----------------------
            report = build_trajectory_report(
                scene_id=scene_id,
                sample_id=sample_id,
                bte_result=bte_result,
                cte_results=cte_results,
                schema_version=schema_version,
            )
            traj_report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(traj_report_path, "w") as fh:
                json.dump(report, fh, indent=2)
            n_generated += 1
            logger.info(
                "[TrajPostprocess] Saved trajectory_report.json → %s", traj_report_path
            )

        print(
            f"[TrajPostprocess] Done: {n_generated} generated, "
            f"{n_fresh} fresh (skipped), {n_missing} missing outputs"
        )
        return n_generated > 0

    def aggregate_trajectory_reports(
        self,
        run_dir: "Path",
        causal_dataset_name: str,
    ) -> dict:
        """
        Aggregate all per-sample trajectory_report.json files into a dataset-level summary.

        Returns a dict with aggregate BTE/CTE ADE statistics and flag counts.
        Written to <run_dir>/<causal_dataset_name>/trajectory_aggregate.json.
        """
        from collections import Counter

        output_base = run_dir / causal_dataset_name
        all_reports = list(output_base.rglob("*/*/trajectory_report.json"))
        if not all_reports:
            logger.warning(
                "[TrajPostprocess] No trajectory_report.json found under %s", output_base
            )
            return {}

        bte_ades: list = []
        cte_ades_by_type: dict = {}
        flag_counter: Counter = Counter()
        grade_counter_bte: Counter = Counter()
        grade_counter_cte: Counter = Counter()

        for rpath in sorted(all_reports):
            try:
                with open(rpath) as fh:
                    report = json.load(fh)
            except Exception as exc:
                logger.warning("Cannot read %s: %s", rpath, exc)
                continue

            bte = report.get("bte")
            if bte and bte.get("parse_success") and bte.get("ade") is not None:
                bte_ades.append(bte["ade"])
                grade_counter_bte[bte.get("analysis", {}).get("grade", "?")] += 1

            for cte in report.get("cte", []):
                if not cte.get("parse_success"):
                    continue
                ade_val = cte.get("ade")
                et = cte.get("eval_type", "CTE_?")
                if ade_val is not None:
                    cte_ades_by_type.setdefault(et, []).append(ade_val)
                for flag in cte.get("analysis", {}).get("flags", []):
                    flag_counter[flag] += 1
                grade = cte.get("analysis", {}).get("grade", "?")
                grade_counter_cte[grade] += 1

        def _stats(vals: list) -> dict:
            if not vals:
                return {}
            return {
                "n":    len(vals),
                "mean": round(sum(vals) / len(vals), 4),
                "min":  round(min(vals), 4),
                "max":  round(max(vals), 4),
            }

        cte_stats = {et: _stats(v) for et, v in sorted(cte_ades_by_type.items())}
        aggregate = {
            "n_samples":          len(all_reports),
            "bte_ade_stats":      _stats(bte_ades),
            "bte_grade_counts":   dict(grade_counter_bte),
            "cte_ade_stats":      cte_stats,
            "cte_grade_counts":   dict(grade_counter_cte),
            "cte_flag_counts":    dict(flag_counter),
        }

        agg_path = output_base / "trajectory_aggregate.json"
        with open(agg_path, "w") as fh:
            json.dump(aggregate, fh, indent=2)
        print(f"[TrajPostprocess] Aggregate → {agg_path}")
        print(f"  BTE ADE: {aggregate.get('bte_ade_stats')}")
        for et, stats in cte_stats.items():
            print(f"  {et}: {stats}")
        print(f"  CTE flags: {aggregate.get('cte_flag_counts')}")
        return aggregate

    # ------------------------------------------------------------------
    # Output collection utility
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_outputs(sample_ids: List[dict], output_base: Path) -> tuple:
        """
        Collect raw outputs.jsonl records from the nested scene/sample directory structure.

        Returns (all_outputs, missing) where missing is a list of
        "<scene_id>/<sample_id>" strings for samples with no outputs.jsonl.

        Used by the legacy compute_metrics path and tests.
        """
        all_outputs: List[dict] = []
        missing: List[str] = []

        for sid in sample_ids:
            outputs_file = output_base / sid["scene_id"] / sid["sample_id"] / "outputs.jsonl"
            if not outputs_file.exists():
                missing.append(f"{sid['scene_id']}/{sid['sample_id']}")
                continue
            with open(outputs_file) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_outputs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        return all_outputs, missing

    # ------------------------------------------------------------------
    # QA index loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load_qa_index(bench_dir: str | Path) -> Dict[str, Dict]:
        """
        Build a flat ``{question_id → question_dict}`` index from all QA files
        found recursively under *bench_dir*.

        Walks the nested ``<scene>/<SAMPLED_N>/qa/`` structure and reads
        ``active_qa.json``, ``dormant_qa.json``, and ``distractor_qa.json``.
        """
        bench_dir = Path(bench_dir)
        qa_index: Dict[str, Dict] = {}

        def _ingest_file(qa_file: Path, qa_type: str) -> None:
            try:
                with open(qa_file) as fh:
                    loaded = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Cannot load QA file %s: %s", qa_file, exc)
                return

            if isinstance(loaded, dict) and "questions" in loaded:
                questions = loaded["questions"]
            elif isinstance(loaded, list):
                questions = loaded
            else:
                logger.warning("Unexpected QA format in %s; skipping", qa_file)
                return

            for q in questions:
                if not isinstance(q, dict):
                    continue
                qid = q.get("id")
                if not qid:
                    continue
                q.setdefault("_qa_type", qa_type)
                q.setdefault("_source_file", qa_file.name)
                if qid in qa_index:
                    logger.debug("Duplicate question_id '%s' in %s; overwriting", qid, qa_file)
                qa_index[qid] = q

        for qa_file in bench_dir.rglob("*/qa/*.json"):
            qa_type = _QA_FILE_TO_TYPE.get(qa_file.name)
            if qa_type is None:
                continue
            _ingest_file(qa_file, qa_type)

        logger.info(
            "_load_qa_index: %d questions indexed from %s", len(qa_index), bench_dir
        )
        return qa_index

    @staticmethod
    def _load_sample_qa_index(sample_bench_dir: str | Path) -> Dict[str, Dict]:
        """
        Build a QA index scoped to a single sample directory.

        Reads only the qa/ subfolder directly under *sample_bench_dir*
        (e.g. causal_nuscenes/nuscenes-scene-0007/SAMPLED_0).

        Question IDs like CI1, NI1 are reused across every scene, so the
        global ``_load_qa_index`` produces wrong lookups when IDs collide.
        This method avoids that by scoping the index to exactly the sample
        being evaluated.
        """
        sample_bench_dir = Path(sample_bench_dir)
        qa_index: Dict[str, Dict] = {}
        qa_dir = sample_bench_dir / "qa"
        if not qa_dir.exists():
            logger.warning("qa/ folder not found under %s", sample_bench_dir)
            return qa_index

        for filename, qa_type in _QA_FILE_TO_TYPE.items():
            qa_file = qa_dir / filename
            if not qa_file.exists():
                continue
            try:
                with open(qa_file) as fh:
                    loaded = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Cannot load %s: %s", qa_file, exc)
                continue

            if isinstance(loaded, dict) and "questions" in loaded:
                questions = loaded["questions"]
            elif isinstance(loaded, list):
                questions = loaded
            else:
                continue

            for q in questions:
                if not isinstance(q, dict):
                    continue
                qid = q.get("id")
                if not qid:
                    continue
                q.setdefault("_qa_type", qa_type)
                q.setdefault("_source_file", qa_file.name)
                qa_index[qid] = q

        logger.debug(
            "_load_sample_qa_index: %d questions from %s", len(qa_index), sample_bench_dir
        )
        return qa_index

    # ------------------------------------------------------------------
    # Legacy: flat-merge metrics pipeline (kept for backward compatibility)
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        outputs_path: str | Path,
        bench_dir: str | Path,
        use_llm_judge: bool = False,
        judge_cfg: Optional[dict] = None,
    ) -> dict:
        """
        Load outputs.jsonl, compare to ground truth, and return metrics dict.

        Legacy method used by models that write a single merged outputs.jsonl.
        Prefer the Tier 1–4 pipeline (process_dataset / run_postprocess) for
        new or updated model integrations.
        """
        outputs_path = Path(outputs_path)
        if not outputs_path.exists():
            raise FileNotFoundError(f"outputs.jsonl not found: {outputs_path}")

        qa_index = self._load_qa_index(bench_dir)

        outputs: List[dict] = []
        with open(outputs_path) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    outputs.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed line %d in %s: %s", lineno, outputs_path, exc
                    )

        logger.info("compute_metrics: loaded %d outputs from %s", len(outputs), outputs_path)

        records: List[dict] = []
        skipped = 0

        for out in outputs:
            qid = out.get("question_id")
            if not qid or qid not in qa_index:
                logger.debug("question_id '%s' not found in QA index; skipping", qid)
                skipped += 1
                continue

            question   = qa_index[qid]
            raw_output = out.get("raw_output", "")

            try:
                predicted = self.parse_answer(raw_output, question)
            except Exception as exc:
                logger.warning("parse_answer failed for question_id=%s: %s", qid, exc)
                predicted = ""

            ground_truth = question.get("correct_answer", question.get("answer", ""))
            qa_type = question.get("_qa_type", "unknown")

            records.append(
                {
                    "question_id":     qid,
                    "type":            qa_type,
                    "predicted":       predicted,
                    "ground_truth":    ground_truth,
                    "answer_format":   question.get("answer_format", ""),
                    "raw_output":      raw_output,
                    "question":        question.get("question", ""),
                    "question_type":   question.get("question_type", ""),
                    "graph_structure": question.get("graph_structure", ""),
                    "rung":            str(question.get("rung", "")) if question.get("rung") is not None else "",
                    "causal_status":   question.get("meta", {}).get("causal_status", "")
                                       if isinstance(question.get("meta"), dict) else "",
                }
            )

        if skipped:
            logger.warning(
                "%d output(s) skipped — question_id not found in QA index", skipped
            )

        if use_llm_judge and judge_cfg and records:
            self._run_llm_judge(records, judge_cfg)

        all_pred = [r["predicted"] for r in records]
        all_gt   = [r["ground_truth"] for r in records]
        overall_acc = mcq_accuracy(all_pred, all_gt) if records else 0.0

        return {
            "overall":             {"accuracy": overall_acc, "n": len(records)},
            "per_qa_type":         per_type_accuracy(records),
            "per_question_type":   per_metadata_accuracy(records, "question_type"),
            "per_graph_structure": per_metadata_accuracy(records, "graph_structure"),
            "per_rung":            per_metadata_accuracy(records, "rung"),
            "answer_stats":        vqa_answer_stats(records),
            "binary_metrics":      binary_classification_metrics(records),
        }

    # ------------------------------------------------------------------
    # LLM judge (internal helpers)
    # ------------------------------------------------------------------

    def _make_judge(self, judge_cfg: dict):
        """Instantiate an LLMJudge from a judge_cfg dict."""
        from evaluation.common.llm_judge import LLMJudge
        return LLMJudge(
            provider=judge_cfg.get("provider", "anthropic"),
            model=judge_cfg.get("model", "claude-opus-4-6"),
            api_key_env=judge_cfg.get("api_key_env", "ANTHROPIC_API_KEY"),
            azure_endpoint_env=judge_cfg.get("azure_endpoint_env", ""),
            azure_api_version=judge_cfg.get("azure_api_version", "2024-12-01-preview"),
            fallback_provider=judge_cfg.get("fallback_provider", ""),
            fallback_model=judge_cfg.get("fallback_model", ""),
        )

    def _run_llm_judge_bulk(
        self,
        judge_inputs: List[dict],
        judge_cfg: dict,
    ) -> List[dict]:
        """
        Score a flat list of judge input dicts and return scores in the same order.

        Dispatches to:
          - score_batch_api()       if use_batch_api=True  (anthropic provider only)
          - score_batch_concurrent() if concurrency > 1     (all providers)
          - score_batch()           otherwise               (serial fallback)

        Each input dict must have keys: "question", "ground_truth", "model_response".
        Returns a list of score dicts (same length as judge_inputs).
        """
        try:
            judge = self._make_judge(judge_cfg)
        except ImportError as exc:
            logger.error("LLMJudge unavailable: %s", exc)
            return [{"scores": {}, "composite_score": 0.0, "reasoning": "", "error": str(exc)}] * len(judge_inputs)

        use_batch_api = judge_cfg.get("use_batch_api", False)
        concurrency   = judge_cfg.get("concurrency", 1)

        if use_batch_api:
            return judge.score_batch_api(judge_inputs)
        if concurrency > 1:
            return judge.score_batch_concurrent(judge_inputs, concurrency=concurrency)
        return judge.score_batch(judge_inputs)

    def _run_llm_judge(
        self,
        records: List[dict],
        judge_cfg: dict,
        score_all: bool = False,
    ) -> None:
        """
        Run LLM judge on responses and attach scores to records in-place.

        Kept for backward compatibility (legacy per-sample call path).
        Prefer _run_llm_judge_bulk() for bulk operations.
        """
        try:
            judge = self._make_judge(judge_cfg)
        except ImportError as exc:
            logger.error("LLMJudge unavailable: %s", exc)
            return

        if score_all:
            scoreable = [
                r for r in records
                if self._extract_text(
                    r.get("raw_output", r.get("raw_output_text", ""))
                ).strip()
            ]
        else:
            scoreable = [
                r for r in records
                if r.get("type") not in ("ladder", "dormant", "distractor")
            ]

        if not scoreable:
            logger.info("_run_llm_judge: no scoreable records (score_all=%s)", score_all)
            return

        judge_inputs = [
            {
                "question":       r.get("question") or r.get("question_text", ""),
                "ground_truth":   r.get("ground_truth", ""),
                "model_response": self._extract_text(
                    r.get("raw_output", r.get("raw_output_text", ""))
                ),
            }
            for r in scoreable
        ]

        concurrency = judge_cfg.get("concurrency", 1)
        if concurrency > 1:
            scored = judge.score_batch_concurrent(judge_inputs, concurrency=concurrency)
        else:
            scored = judge.score_batch(judge_inputs)

        for r, score in zip(scoreable, scored):
            r["llm_judge_score"] = score

    # ------------------------------------------------------------------
    # Report saving (legacy)
    # ------------------------------------------------------------------

    def save_report(self, metrics: dict, report_path: str | Path) -> None:
        """Save metrics to a JSON file and print a human-readable summary."""
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as fh:
            json.dump(metrics, fh, indent=2)

        print("\n" + "=" * 60)
        print("  EVALUATION REPORT")
        print("=" * 60)
        overall = metrics.get("overall", {})
        print(
            f"  Overall accuracy : {overall.get('accuracy', 0.0):.2%}  "
            f"(n={overall.get('n', 0)})"
        )
        print()

        per_type = metrics.get("per_qa_type", {})
        if per_type:
            print(f"  {'QA Type':<20} {'Accuracy':>10} {'N':>6}")
            print(f"  {'-' * 20} {'-' * 10} {'-' * 6}")
            for qt, vals in per_type.items():
                print(
                    f"  {qt:<20} {vals.get('accuracy', 0.0):>10.2%} "
                    f"{vals.get('n', 0):>6}"
                )

        print()
        print(f"  Full report saved to: {report_path}")
        print("=" * 60 + "\n")
        logger.info("Report saved to %s", report_path)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(raw_output) -> str:
        """Extract the text string from raw_output (dict or legacy plain string)."""
        if isinstance(raw_output, dict):
            return raw_output.get("text", "")
        return str(raw_output) if raw_output else ""
