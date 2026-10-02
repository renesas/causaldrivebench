"""
Multi-level report generation for the CausalDriveBench evaluation pipeline.

Four report tiers, all stored as JSON inside cdb_outputs/:

  Tier 1 — sample   : cdb_outputs/<run>/<dataset>/<scene>/<sample>/report.json
  Tier 2 — dataset  : cdb_outputs/<run>/<dataset>/report.json
  Tier 3 — run      : cdb_outputs/<run>/report.json
  Tier 4 — root     : root aggregate dict (written by html_reporter as index.html)

Incremental logic
-----------------
Sample reports (Tier 1) are the only expensive tier — they require running the
model-specific postprocessor to parse answers.  They are skipped if their stored
``schema_version`` matches the value from eval_config.yaml.

Tier 2, 3, and 4 always regenerate (fast: they only read pre-computed JSONs).

Extending metrics
-----------------
To add a new metric:
  1. Implement a function in evaluation/common/metrics.py.
  2. Add a key to ``reporting.metrics`` in eval_config.yaml.
  3. Handle it in ``_compute_metrics()`` below.
  4. Bump ``reporting.schema_version`` in eval_config.yaml to force
     regeneration of all existing Tier 1 reports.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.common.metrics import (
    binary_classification_metrics,
    mcq_accuracy,
    per_metadata_accuracy,
    per_type_accuracy,
    reasoning_score_summary,
    vqa_answer_stats,
)

logger = logging.getLogger(__name__)

# Bump this constant when the sample report JSON schema changes.
# It acts as the fallback when eval_config.yaml has no reporting section.
SCHEMA_VERSION = "1.0"

# Maps question_json_file value → internal qa_type label
_FILE_TO_QA_TYPE: Dict[str, str] = {
    "active_qa.json":     "ladder",
    "dormant_qa.json":    "dormant",
    "distractor_qa.json": "distractor",
}


# ---------------------------------------------------------------------------
# Incremental-check helpers
# ---------------------------------------------------------------------------

def is_sample_report_fresh(path: Path, schema_version: str) -> bool:
    """
    Return True if the sample report at *path* exists and was generated with
    *schema_version*.  Reads only the top-level ``schema_version`` key.
    """
    if not path.exists():
        return False
    try:
        with open(path) as fh:
            stored = json.load(fh).get("schema_version")
        return stored == schema_version
    except (json.JSONDecodeError, OSError, AttributeError):
        return False


def load_report(path: Path) -> Optional[dict]:
    """Load a JSON report file, returning None on any error."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        logger.debug("Could not load report: %s", path)
        return None


def write_report(report: dict, path: Path) -> None:
    """Write *report* as formatted JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    logger.debug("Wrote report: %s", path)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(qa_results: List[dict], enabled: List[str]) -> dict:
    """
    Compute all enabled metrics from a flat list of QA result dicts.

    Each entry in *qa_results* must have:
        qa_type       : str
        predicted     : str
        ground_truth  : str
        correct       : bool
    Optional metadata fields: question_type, graph_structure, rung, causal_status,
        answer_format

    Returns a metrics dict keyed by metric name.
    """
    if not qa_results:
        return {}

    metrics: dict = {}

    # Normalised record list used by per_type_accuracy and per_metadata_accuracy
    typed = [
        {
            "type":           r.get("qa_type", "unknown"),
            "predicted":      r.get("predicted", ""),
            "ground_truth":   r.get("ground_truth", ""),
            "answer_format":  r.get("answer_format", ""),
            "question_type":  r.get("question_type", ""),
            "graph_structure": r.get("graph_structure", ""),
            "rung":           str(r.get("rung", "")) if r.get("rung") is not None else "",
            "causal_status":  r.get("causal_status", ""),
        }
        for r in qa_results
    ]

    if "overall" in enabled:
        correct = sum(1 for r in qa_results if r.get("correct", False))
        metrics["overall"] = {
            "accuracy": correct / len(qa_results),
            "n":        len(qa_results),
            "correct":  correct,
        }

    if "per_qa_type" in enabled:
        metrics["per_qa_type"] = per_type_accuracy(typed)

    if "per_question_type" in enabled:
        metrics["per_question_type"] = per_metadata_accuracy(typed, "question_type")

    if "per_graph_structure" in enabled:
        metrics["per_graph_structure"] = per_metadata_accuracy(typed, "graph_structure")

    if "per_rung" in enabled:
        metrics["per_rung"] = per_metadata_accuracy(typed, "rung")

    if "per_causal_status" in enabled:
        metrics["per_causal_status"] = per_metadata_accuracy(typed, "causal_status")

    if "answer_stats" in enabled:
        metrics["answer_stats"] = vqa_answer_stats(typed)

    if "binary_metrics" in enabled:
        metrics["binary_metrics"] = binary_classification_metrics(typed)

    if "reasoning_score" in enabled:
        scored = [r for r in qa_results if "llm_judge_score" in r]
        if scored:
            metrics["reasoning_score"] = reasoning_score_summary(scored)

    return metrics


# ---------------------------------------------------------------------------
# Tier 1 — sample reports
# ---------------------------------------------------------------------------

def build_sample_report(
    run_name: str,
    dataset: str,
    scene_id: str,
    sample_id: str,
    qa_results: List[dict],
    schema_version: str,
    enabled_metrics: List[str],
) -> dict:
    """
    Build and return a sample-level report dict.

    Parameters
    ----------
    qa_results : list of dicts; each must have:
        question_id, qa_type, answer_format, question_text,
        predicted, ground_truth, correct, raw_output_text
        Optional: inference_time_s, llm_judge_score
    """
    return {
        "schema_version": schema_version,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "level":          "sample",
        "run_name":       run_name,
        "dataset":        dataset,
        "scene_id":       scene_id,
        "sample_id":      sample_id,
        "n_questions":    len(qa_results),
        "metrics":        _compute_metrics(qa_results, enabled_metrics),
        "qa_results":     qa_results,
    }


def generate_sample_report(
    run_name: str,
    dataset: str,
    scene_id: str,
    sample_id: str,
    qa_results: List[dict],
    report_path: Path,
    schema_version: str,
    enabled_metrics: List[str],
) -> dict:
    """
    Build and write a Tier 1 sample report.  Returns the report dict.
    """
    report = build_sample_report(
        run_name, dataset, scene_id, sample_id,
        qa_results, schema_version, enabled_metrics,
    )
    write_report(report, report_path)
    logger.info(
        "Sample report: %s/%s | %d Q | acc=%.2f",
        scene_id, sample_id,
        report["n_questions"],
        report["metrics"].get("overall", {}).get("accuracy", 0.0),
    )
    return report


# ---------------------------------------------------------------------------
# Tier 2 — dataset aggregate
# ---------------------------------------------------------------------------

def generate_dataset_report(
    run_name: str,
    dataset: str,
    dataset_dir: Path,
    report_path: Path,
    schema_version: str,
    enabled_metrics: List[str],
) -> dict:
    """
    Aggregate all Tier 1 sample reports found under *dataset_dir*.

    Scans ``<dataset_dir>/<scene>/<sample>/report.json`` recursively.
    Always regenerates (fast: reads pre-computed sample reports).
    """
    all_qa_results: List[dict] = []
    sample_summaries: List[dict] = []
    scene_set: set = set()

    for sample_report_path in sorted(dataset_dir.rglob("*/*/report.json")):
        sample_report = load_report(sample_report_path)
        if not sample_report or sample_report.get("level") != "sample":
            continue

        qa = sample_report.get("qa_results", [])
        all_qa_results.extend(qa)
        scene_set.add(sample_report.get("scene_id", ""))

        overall = sample_report.get("metrics", {}).get("overall", {})
        sample_summaries.append({
            "scene_id":   sample_report.get("scene_id", ""),
            "sample_id":  sample_report.get("sample_id", ""),
            "n_questions": sample_report.get("n_questions", 0),
            "accuracy":   overall.get("accuracy", 0.0),
            "correct":    overall.get("correct", 0),
            "report_path": str(sample_report_path.relative_to(dataset_dir)),
        })

    report = {
        "schema_version":   schema_version,
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "level":            "dataset",
        "run_name":         run_name,
        "dataset":          dataset,
        "n_scenes":         len(scene_set),
        "n_samples":        len(sample_summaries),
        "n_questions":      len(all_qa_results),
        "metrics":          _compute_metrics(all_qa_results, enabled_metrics),
        "sample_summaries": sample_summaries,
    }
    write_report(report, report_path)
    logger.info(
        "Dataset report: %s | %d samples | %d Q | acc=%.2f",
        dataset, len(sample_summaries), len(all_qa_results),
        report["metrics"].get("overall", {}).get("accuracy", 0.0),
    )
    return report


# ---------------------------------------------------------------------------
# Tier 3 — run aggregate
# ---------------------------------------------------------------------------

def generate_run_report(
    run_name: str,
    run_dir: Path,
    report_path: Path,
    schema_version: str,
    enabled_metrics: List[str],
) -> dict:
    """
    Aggregate all Tier 2 dataset reports found directly under *run_dir*.

    Scans ``<run_dir>/<dataset>/report.json`` (depth 1 only).
    Always regenerates.
    """
    all_qa_results: List[dict] = []
    dataset_summaries: List[dict] = []

    for dataset_dir in sorted(run_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_report_path = dataset_dir / "report.json"
        if not dataset_report_path.exists():
            continue
        dataset_report = load_report(dataset_report_path)
        if not dataset_report or dataset_report.get("level") != "dataset":
            continue

        # Re-load all sample qa_results for this dataset for metric computation
        for sample_report_path in sorted(dataset_dir.rglob("*/*/report.json")):
            sr = load_report(sample_report_path)
            if sr and sr.get("level") == "sample":
                all_qa_results.extend(sr.get("qa_results", []))

        overall = dataset_report.get("metrics", {}).get("overall", {})
        dataset_summaries.append({
            "dataset":     dataset_report.get("dataset", dataset_dir.name),
            "n_samples":   dataset_report.get("n_samples", 0),
            "n_questions": dataset_report.get("n_questions", 0),
            "accuracy":    overall.get("accuracy", 0.0),
            "correct":     overall.get("correct", 0),
        })

    report = {
        "schema_version":    schema_version,
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "level":             "run",
        "run_name":          run_name,
        "n_questions":       len(all_qa_results),
        "metrics":           _compute_metrics(all_qa_results, enabled_metrics),
        "dataset_summaries": dataset_summaries,
    }
    write_report(report, report_path)
    logger.info(
        "Run report: %s | %d datasets | %d Q | acc=%.2f",
        run_name, len(dataset_summaries), len(all_qa_results),
        report["metrics"].get("overall", {}).get("accuracy", 0.0),
    )
    return report


# ---------------------------------------------------------------------------
# Tier 4 — root aggregate (across all runs in cdb_outputs/)
# ---------------------------------------------------------------------------

def generate_root_aggregate(
    cdb_outputs_dir: Path,
    schema_version: str,
) -> dict:
    """
    Scan all run directories in *cdb_outputs_dir*, read their Tier 3 run
    reports, and return a root aggregate dict.

    Only directories that contain a ``report.json`` with ``"level": "run"``
    are included.
    """
    runs: Dict[str, dict] = {}
    per_model: Dict[str, list] = defaultdict(list)

    for run_dir in sorted(cdb_outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_report_path = run_dir / "report.json"
        if not run_report_path.exists():
            continue
        run_report = load_report(run_report_path)
        if not run_report or run_report.get("level") != "run":
            continue

        run_name = run_dir.name
        overall = run_report.get("metrics", {}).get("overall", {})
        datasets = [s["dataset"] for s in run_report.get("dataset_summaries", [])]

        # Infer model_id as everything before the last timestamp segment
        # e.g. "alpamayo_1_0_20260416_215836" → "alpamayo_1_0"
        parts = run_name.rsplit("_", 2)
        model_id = parts[0] if len(parts) == 3 else run_name

        run_metrics = run_report.get("metrics", {})
        run_summary = {
            "run_name":           run_name,
            "model_id":           model_id,
            "generated_at":       run_report.get("generated_at", ""),
            "n_questions":        run_report.get("n_questions", 0),
            "accuracy":           overall.get("accuracy", 0.0),
            "correct":            overall.get("correct", 0),
            "datasets":           datasets,
            "per_qa_type":        run_metrics.get("per_qa_type", {}),
            "per_question_type":  run_metrics.get("per_question_type", {}),
            "per_graph_structure": run_metrics.get("per_graph_structure", {}),
            "per_rung":           run_metrics.get("per_rung", {}),
            "binary_metrics":     run_metrics.get("binary_metrics", {}),
        }
        runs[run_name] = run_summary
        per_model[model_id].append(run_summary)

    # Aggregate best/average per model
    model_summaries: Dict[str, dict] = {}
    for model_id, model_runs in sorted(per_model.items()):
        accs = [r["accuracy"] for r in model_runs]
        model_summaries[model_id] = {
            "n_runs":        len(model_runs),
            "best_accuracy": max(accs) if accs else 0.0,
            "avg_accuracy":  sum(accs) / len(accs) if accs else 0.0,
            "runs":          [r["run_name"] for r in model_runs],
        }

    return {
        "schema_version": schema_version,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "level":          "root",
        "n_runs":         len(runs),
        "runs":           runs,
        "per_model":      model_summaries,
    }
