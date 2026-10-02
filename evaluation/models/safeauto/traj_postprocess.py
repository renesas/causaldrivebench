"""
SafeAuto Trajectory Postprocessor.

Parses SafeAuto's free-form text output for trajectory waypoints and
computes displacement metrics against ground-truth from state.json.

Expected model output (ideal):
    [{"t": 0.5, "x": 2.1, "y": 0.2}, {"t": 1.0, "x": 4.8, "y": 1.0}, ...]

The parser tries multiple extraction strategies:
  1. JSON list directly in the output
  2. JSON list inside a markdown code block
  3. Regex-extracted (x, y) pairs in order

Metrics computed per sample:
  - ADE  : mean L2 displacement across all 6 future timesteps
  - FDE  : L2 displacement at the final timestep (t=3.0 s)
  - per_t: per-timestep displacement errors
  - parse_ok: whether waypoint extraction succeeded

Aggregate report:
  - mean_ade, mean_fde across all samples
  - per-timestep mean error
  - parse_success_rate
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FUTURE_TIMES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

def _parse_json_waypoints(text: str) -> Optional[List[Tuple[float, float]]]:
    """
    Extract waypoints from a JSON list in the model output.

    Tries the raw text and any ```json``` code block.
    Returns a list of (x, y) tuples in time order, or None on failure.
    """
    candidates = [text]

    # Extract from markdown code block
    block = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if block:
        candidates.insert(0, block.group(1))

    # Also try the first [...] span in the raw text
    bracket = re.search(r"(\[.*?\])", text, re.DOTALL)
    if bracket:
        candidates.insert(0, bracket.group(1))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, list) and len(data) >= 1:
                waypoints = []
                for item in data:
                    if isinstance(item, dict):
                        x = float(item.get("x", item.get("X", 0.0)))
                        y = float(item.get("y", item.get("Y", 0.0)))
                        waypoints.append((x, y))
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        waypoints.append((float(item[0]), float(item[1])))
                if waypoints:
                    return waypoints
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    return None


def _parse_regex_waypoints(text: str) -> Optional[List[Tuple[float, float]]]:
    """
    Fallback: extract (x, y) pairs using regex.

    Matches patterns like:
      (2.17, 0.21)  |  x=2.17, y=0.21  |  2.17 0.21
    """
    # Pattern: two floats separated by comma, optionally in parens
    pairs = re.findall(
        r"[-+]?\d+\.?\d*\s*,\s*[-+]?\d+\.?\d*",
        text,
    )
    if len(pairs) >= 3:
        waypoints = []
        for pair in pairs:
            nums = [float(n.strip()) for n in pair.split(",")]
            waypoints.append((nums[0], nums[1]))
        return waypoints if len(waypoints) >= 3 else None

    # Pattern: "x: 2.17  y: 0.21" or "x=2.17  y=0.21"
    xy_pairs = re.findall(
        r"x\s*[=:]\s*([-+]?\d+\.?\d*)\s+y\s*[=:]\s*([-+]?\d+\.?\d*)",
        text, re.IGNORECASE,
    )
    if len(xy_pairs) >= 3:
        return [(float(x), float(y)) for x, y in xy_pairs]

    return None


def parse_trajectory(text: str) -> Tuple[Optional[List[Tuple[float, float]]], str]:
    """
    Parse predicted waypoints from model output text.

    Returns (waypoints, method) where method is one of:
      "json", "regex", "failed"
    """
    if not text or not text.strip():
        return None, "failed"

    wps = _parse_json_waypoints(text)
    if wps is not None:
        return wps, "json"

    wps = _parse_regex_waypoints(text)
    if wps is not None:
        return wps, "regex"

    return None, "failed"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _l2(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    return math.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)


def compute_sample_metrics(
    pred_waypoints: List[Tuple[float, float]],
    gt_future: List[Dict],
) -> dict:
    """
    Compute ADE, FDE, and per-timestep errors for one sample.

    Parameters
    ----------
    pred_waypoints : list of (x, y) tuples, length ≥ 1
    gt_future      : list of {"t": float, "x": float, "y": float} from state.json

    Returns
    -------
    dict with keys: ade, fde, per_t (list of dicts), n_matched
    """
    # Align predicted to GT time steps (take the first N matching GT points)
    gt_points = [(float(pt["x"]), float(pt["y"])) for pt in gt_future]
    n = min(len(pred_waypoints), len(gt_points))

    if n == 0:
        return {"ade": None, "fde": None, "per_t": [], "n_matched": 0}

    errors = [_l2(pred_waypoints[i], gt_points[i]) for i in range(n)]
    ade = sum(errors) / n
    fde = errors[-1]

    per_t = []
    for i in range(n):
        t_val = gt_future[i]["t"] if i < len(gt_future) else FUTURE_TIMES[i]
        per_t.append({"t": t_val, "error": round(errors[i], 4)})

    return {
        "ade":       round(ade, 4),
        "fde":       round(fde, 4),
        "per_t":     per_t,
        "n_matched": n,
    }


# ---------------------------------------------------------------------------
# Per-sample result builder
# ---------------------------------------------------------------------------

def process_sample(payload: dict, raw_output: str) -> dict:
    """
    Parse the model output and compute metrics for one trajectory prompt.

    Parameters
    ----------
    payload    : prompt payload from traj_dataloader.build_traj_prompts()
    raw_output : raw text string returned by SafeAutoInference.run_single()

    Returns
    -------
    Sample result dict suitable for JSON serialisation.
    """
    scene_id    = payload["scene_id"]
    sample_id   = payload["sample_id"]
    eval_type   = payload.get("type", "BTE")
    question_id = payload.get("question_id")
    gt_waypoints = payload.get("gt_waypoints", payload.get("gt_future", []))

    pred_wps, parse_method = parse_trajectory(raw_output)

    if pred_wps is None:
        logger.warning("[%s/%s] [%s] Failed to parse trajectory", scene_id, sample_id, eval_type)
        metrics = {"ade": None, "fde": None, "per_t": [], "n_matched": 0}
    else:
        metrics = compute_sample_metrics(pred_wps, gt_waypoints)

    return {
        "type":           eval_type,
        "scene_id":       scene_id,
        "sample_id":      sample_id,
        "question_id":    question_id,
        "parse_method":   parse_method,
        "parse_ok":       pred_wps is not None,
        "pred_waypoints": [{"x": x, "y": y} for x, y in pred_wps] if pred_wps else [],
        "gt_waypoints":   gt_waypoints,
        "raw_output":     raw_output,
        "metrics":        metrics,
    }


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def _aggregate_group(results: List[dict]) -> dict:
    """Aggregate metrics for a list of results of a single type."""
    valid = [r for r in results if r["metrics"]["ade"] is not None]
    n_total = len(results)
    n_valid = len(valid)

    if not valid:
        return {
            "n_total":            n_total,
            "n_valid":            0,
            "parse_success_rate": 0.0,
            "mean_ade":           None,
            "mean_fde":           None,
            "per_t_mean_error":   {},
        }

    mean_ade = sum(r["metrics"]["ade"] for r in valid) / n_valid
    mean_fde = sum(r["metrics"]["fde"] for r in valid) / n_valid

    per_t_accum: Dict[float, List[float]] = {}
    for r in valid:
        for pt in r["metrics"]["per_t"]:
            per_t_accum.setdefault(pt["t"], []).append(pt["error"])

    per_t_means = {
        str(t): round(sum(errs) / len(errs), 4)
        for t, errs in sorted(per_t_accum.items())
    }

    return {
        "n_total":            n_total,
        "n_valid":            n_valid,
        "parse_success_rate": round(n_valid / n_total, 4) if n_total else 0.0,
        "mean_ade":           round(mean_ade, 4),
        "mean_fde":           round(mean_fde, 4),
        "per_t_mean_error":   per_t_means,
    }


def aggregate_results(results: List[dict]) -> dict:
    """
    Compute aggregate trajectory metrics across all samples, broken down by
    eval type (BTE, CTE_R2, CTE_R3) and overall.
    """
    by_type: Dict[str, List[dict]] = {}
    for r in results:
        t = r.get("type", "BTE")
        by_type.setdefault(t, []).append(r)

    breakdown = {t: _aggregate_group(rs) for t, rs in by_type.items()}
    overall   = _aggregate_group(results)
    overall["by_type"] = breakdown
    return overall


def print_report(report: dict) -> None:
    """Print a human-readable summary of the aggregate report."""
    def _print_group(label: str, g: dict) -> None:
        print(f"\n  [{label}]")
        print(f"    Total : {g['n_total']}  |  Parsed: {g['n_valid']}"
              f"  ({g['parse_success_rate']:.1%})")
        if g["mean_ade"] is not None:
            print(f"    Mean ADE : {g['mean_ade']:.4f} m")
            print(f"    Mean FDE : {g['mean_fde']:.4f} m")
            for t, err in g.get("per_t_mean_error", {}).items():
                print(f"      t={t}s : {err:.4f} m")

    print("\n" + "=" * 60)
    print("  SafeAuto Trajectory Evaluation Results")
    print("=" * 60)
    _print_group("Overall", report)
    for type_name, group in report.get("by_type", {}).items():
        _print_group(type_name, group)
    print("=" * 60 + "\n")
