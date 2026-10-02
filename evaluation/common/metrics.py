"""
Evaluation metrics for the CausalDriveBench benchmark.

All functions operate on plain Python lists/dicts and have no heavy
dependencies — they can be imported without GPU/torch.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Core accuracy helpers
# ---------------------------------------------------------------------------


def mcq_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """
    Compute exact-match accuracy for multiple-choice or binary questions.

    Comparison is case-insensitive and strips surrounding whitespace.
    Returns 0.0 when both lists are empty.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truth)} ground-truth labels"
        )
    if not predictions:
        return 0.0
    correct = sum(
        p.strip().upper() == g.strip().upper()
        for p, g in zip(predictions, ground_truth)
    )
    return correct / len(predictions)


def per_type_accuracy(records: List[dict]) -> Dict[str, dict]:
    """
    Compute accuracy broken down by question type (qa_type field).

    Parameters
    ----------
    records : list of dicts with keys: "type", "predicted", "ground_truth"

    Returns
    -------
    Dict keyed by question type: {"accuracy": float, "n": int, "correct": int}
    """
    grouped: Dict[str, List[tuple]] = defaultdict(list)
    for rec in records:
        qtype = rec.get("type", "unknown")
        grouped[qtype].append((rec.get("predicted", ""), rec.get("ground_truth", "")))

    result: Dict[str, dict] = {}
    for qtype, pairs in sorted(grouped.items()):
        preds = [p for p, _ in pairs]
        gts = [g for _, g in pairs]
        acc = mcq_accuracy(preds, gts)
        result[qtype] = {
            "accuracy": acc,
            "n":        len(pairs),
            "correct":  sum(
                p.strip().upper() == g.strip().upper() for p, g in pairs
            ),
        }
    return result


def per_metadata_accuracy(records: List[dict], meta_key: str) -> Dict[str, dict]:
    """
    Generic accuracy breakdown by any metadata field present in each record.

    Parameters
    ----------
    records  : list of dicts with keys: meta_key, "predicted", "ground_truth"
    meta_key : field name to group by (e.g. "question_type", "rung", "graph_structure")

    Returns
    -------
    Dict keyed by meta_key value: {"accuracy": float, "n": int, "correct": int}
    """
    grouped: Dict[str, List[tuple]] = defaultdict(list)
    for rec in records:
        val = rec.get(meta_key)
        if val is None:
            continue  # skip records that don't have this metadata
        key = str(val)
        grouped[key].append((rec.get("predicted", ""), rec.get("ground_truth", "")))

    result: Dict[str, dict] = {}
    for key, pairs in sorted(grouped.items()):
        preds = [p for p, _ in pairs]
        gts   = [g for _, g in pairs]
        acc = mcq_accuracy(preds, gts)
        result[key] = {
            "accuracy": round(acc, 4),
            "n":        len(pairs),
            "correct":  sum(
                p.strip().upper() == g.strip().upper() for p, g in pairs
            ),
        }
    return result


# ---------------------------------------------------------------------------
# VQA-style metrics (replace confusion matrix)
# ---------------------------------------------------------------------------


def vqa_answer_stats(records: List[dict]) -> dict:
    """
    VQA-style answer distribution and bias statistics.

    Replaces the confusion matrix with more meaningful metrics for a benchmark
    where each QA pair is unique and the label space is small (Yes/No/A/B/C/D).

    Returns
    -------
    dict with:
        "gt_distribution"        : {answer: {"count": int, "pct": float}}
        "predicted_distribution" : {answer: {"count": int, "pct": float}}
        "per_answer_accuracy"    : {gt_answer: {"n": int, "correct": int, "accuracy": float}}
        "answer_bias"            : float — max(predicted %) − (100 / n_classes)
                                   Positive means over-prediction of one answer.
        "n_answer_classes"       : int
    """
    if not records:
        return {}

    gt_counts: Dict[str, int]   = defaultdict(int)
    pred_counts: Dict[str, int] = defaultdict(int)
    per_answer: Dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})

    for rec in records:
        gt   = rec.get("ground_truth", "?").strip().upper()
        pred = rec.get("predicted",    "?").strip().upper()
        gt_counts[gt]   += 1
        pred_counts[pred] += 1
        per_answer[gt]["n"] += 1
        if pred == gt:
            per_answer[gt]["correct"] += 1

    n = len(records)

    per_answer_accuracy: Dict[str, dict] = {}
    for ans, counts in per_answer.items():
        per_answer_accuracy[ans] = {
            "n":        counts["n"],
            "correct":  counts["correct"],
            "accuracy": round(counts["correct"] / counts["n"], 4) if counts["n"] else 0.0,
        }

    gt_dist   = {k: {"count": v, "pct": round(v / n * 100, 1)}
                 for k, v in sorted(gt_counts.items())}
    pred_dist = {k: {"count": v, "pct": round(v / n * 100, 1)}
                 for k, v in sorted(pred_counts.items())}

    n_classes = len(gt_counts)
    expected_pct = 100.0 / n_classes if n_classes else 100.0
    max_pred_pct = max((v / n * 100 for v in pred_counts.values()), default=0.0)
    bias = round(max_pred_pct - expected_pct, 2)

    return {
        "gt_distribution":        gt_dist,
        "predicted_distribution": pred_dist,
        "per_answer_accuracy":    per_answer_accuracy,
        "answer_bias":            bias,
        "n_answer_classes":       n_classes,
    }


def binary_classification_metrics(records: List[dict]) -> dict:
    """
    Precision, recall, and F1 for binary Yes/No questions.

    Treats 'Yes' as the positive class.  Only processes records whose
    ground_truth is Yes or No (or where answer_format == 'binary').

    Returns empty dict when no binary records are found.
    """
    binary = [
        r for r in records
        if r.get("answer_format") == "binary"
        or r.get("ground_truth", "").strip().upper() in ("YES", "NO")
    ]
    if not binary:
        return {}

    tp = fp = fn = tn = 0
    for rec in binary:
        gt   = rec.get("ground_truth", "").strip().upper()
        pred = rec.get("predicted",    "").strip().upper()
        if gt == "YES" and pred == "YES":
            tp += 1
        elif gt == "NO" and pred == "YES":
            fp += 1
        elif gt == "YES" and pred == "NO":
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp)     if (tp + fp) else 0.0
    recall    = tp / (tp + fn)     if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    accuracy  = (tp + tn) / len(binary)

    return {
        "n":         len(binary),
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Trajectory metric
# ---------------------------------------------------------------------------


def ade(
    actual: Sequence[Tuple[float, float]],
    predicted: Sequence[Tuple[float, float]],
) -> float:
    """
    Average Displacement Error (ADE) for a single predicted trajectory.

    Mean L2 distance between predicted and actual waypoints across all timesteps.
    Both sequences must have the same length (pre-aligned by the caller).

    Parameters
    ----------
    actual    : actual waypoints  [(x0,y0), (x1,y1), …]
    predicted : predicted waypoints, same length as *actual*

    Returns
    -------
    ADE in the same spatial units as the input coordinates.
    Returns 0.0 if either argument is empty.

    Raises
    ------
    ValueError
        When the two sequences have different lengths.
    """
    if not actual or not predicted:
        return 0.0
    if len(actual) != len(predicted):
        raise ValueError(
            f"Sequence length mismatch: actual={len(actual)}, predicted={len(predicted)}"
        )
    total = sum(
        math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        for (px, py), (ax, ay) in zip(predicted, actual)
    )
    return total / len(actual)


def min_ade(
    actual: Sequence[Tuple[float, float]],
    predicted_list: Sequence[Sequence[Tuple[float, float]]],
) -> float:
    """
    Minimum Average Displacement Error (minADE) over K predicted trajectories.

    For each candidate trajectory, compute the mean L2 distance to the actual
    waypoints across all timesteps.  Returns the minimum over all K candidates.

    Parameters
    ----------
    actual         : actual waypoint sequence  [(x0,y0), (x1,y1), …]
    predicted_list : K predicted trajectories, each [(x0,y0), …]
                     Every trajectory must have the same length as *actual*.

    Returns
    -------
    minADE in the same spatial units as the input coordinates.
    Returns 0.0 if either argument is empty.

    Raises
    ------
    ValueError
        When any predicted trajectory length differs from *actual*.
    """
    if not actual or not predicted_list:
        return 0.0

    T = len(actual)

    def _ade(pred: Sequence[Tuple[float, float]]) -> float:
        if len(pred) != T:
            raise ValueError(
                f"Predicted trajectory length {len(pred)} != actual length {T}"
            )
        total = sum(
            math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
            for (px, py), (ax, ay) in zip(pred, actual)
        )
        return total / T

    return min(_ade(pred) for pred in predicted_list)


# ---------------------------------------------------------------------------
# Cross-model aggregation
# ---------------------------------------------------------------------------


def aggregate_model_metrics(model_results: Dict[str, dict]) -> dict:
    """
    Aggregate metrics from multiple models into a comparison structure.

    Parameters
    ----------
    model_results : dict mapping model_name → metrics_dict

    Returns
    -------
    Aggregated dict suitable for ``format_comparison_table()``.
    """
    aggregated: dict = {
        "models":              {},
        "qa_types":            set(),
        "question_types":      set(),
        "graph_structures":    set(),
        "rungs":               set(),
    }

    for model_name, metrics in model_results.items():
        overall       = metrics.get("overall", {})
        per_type      = metrics.get("per_qa_type", {})
        per_qtype     = metrics.get("per_question_type", {})
        per_graph     = metrics.get("per_graph_structure", {})
        per_rung      = metrics.get("per_rung", {})
        binary_m      = metrics.get("binary_metrics", {})

        aggregated["models"][model_name] = {
            "overall_accuracy":     overall.get("accuracy", 0.0),
            "overall_n":            overall.get("n", 0),
            "per_qa_type":          per_type,
            "per_question_type":    per_qtype,
            "per_graph_structure":  per_graph,
            "per_rung":             per_rung,
            "binary_f1":            binary_m.get("f1", None),
        }
        for qt in per_type:
            aggregated["qa_types"].add(qt)
        for qt in per_qtype:
            aggregated["question_types"].add(qt)
        for gs in per_graph:
            aggregated["graph_structures"].add(gs)
        for rung in per_rung:
            aggregated["rungs"].add(rung)

    aggregated["qa_types"]         = sorted(aggregated["qa_types"])
    aggregated["question_types"]   = sorted(aggregated["question_types"])
    aggregated["graph_structures"] = sorted(aggregated["graph_structures"])
    aggregated["rungs"]            = sorted(aggregated["rungs"], key=lambda x: int(x) if str(x).isdigit() else 999)
    return aggregated


def reasoning_score_summary(records: List[dict]) -> dict:
    """
    Aggregate LLM judge reasoning scores from a list of scored output records.

    Each record may have a ``"llm_judge_score"`` key containing the output of
    ``LLMJudge.score_response()``.

    Returns
    -------
    dict with:
        "n"              : number of successfully scored records
        "composite_mean" : float (0–2 scale)
        "per_dimension"  : {dim → {"mean": float, "distribution": {0: int, 1: int, 2: int}}}
    """
    _DIMENSIONS = [
        "spatial_grounding",
        "mechanism_identification",
        "conditionality_awareness",
        "answer_consistency",
    ]

    scored = [
        r["llm_judge_score"]
        for r in records
        if isinstance(r.get("llm_judge_score"), dict)
        and "scores" in r["llm_judge_score"]
        and "error" not in r["llm_judge_score"]
    ]

    if not scored:
        return {
            "n": 0,
            "composite_mean": 0.0,
            "per_dimension": {
                dim: {"mean": 0.0, "distribution": {0: 0, 1: 0, 2: 0}}
                for dim in _DIMENSIONS
            },
        }

    composite_mean = sum(s.get("composite_score", 0.0) for s in scored) / len(scored)

    per_dimension: Dict[str, dict] = {}
    for dim in _DIMENSIONS:
        dim_scores = [s["scores"].get(dim, 0) for s in scored]
        dim_mean = sum(dim_scores) / len(dim_scores)
        distribution: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        for val in dim_scores:
            clamped = max(0, min(2, int(val)))
            distribution[clamped] = distribution.get(clamped, 0) + 1
        per_dimension[dim] = {
            "mean":         round(dim_mean, 4),
            "distribution": distribution,
        }

    return {
        "n":              len(scored),
        "composite_mean": round(composite_mean, 4),
        "per_dimension":  per_dimension,
    }


def format_comparison_table(aggregated: dict) -> str:
    """
    Render an ASCII comparison table of model metrics.

    Parameters
    ----------
    aggregated : dict as returned by ``aggregate_model_metrics()``

    Returns
    -------
    Multi-line ASCII table string.
    """
    models    = list(aggregated.get("models", {}).keys())
    qa_types  = list(aggregated.get("qa_types", []))

    if not models:
        return "(no models to compare)"

    col_names  = ["Model", "Overall"] + qa_types
    col_widths = [max(len(c), 12) for c in col_names]

    rows = []
    for model_name in models:
        info = aggregated["models"][model_name]
        row  = [model_name, f"{info['overall_accuracy']:.2%}"]
        for qt in qa_types:
            acc = info["per_qa_type"].get(qt, {}).get("accuracy", None)
            row.append(f"{acc:.2%}" if acc is not None else "—")
        rows.append(row)

    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(val) + 2)

    def _fmt_row(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    sep   = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    lines = [sep, _fmt_row(col_names), sep]
    for row in rows:
        lines.append(_fmt_row(row))
    lines.append(sep)

    table = "\n".join(lines)
    print(table)
    return table
