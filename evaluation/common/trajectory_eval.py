"""
Trajectory evaluation for CausalDriveBench.

Three evaluation modes
-----------------------
BTE   Baseline Trajectory Evaluation
      One question per SAMPLED directory.
      Ground truth: state.json → ego.future waypoints.

CTE_R2  Counterfactual Trajectory Evaluation (Rung 2)
CTE_R3  Counterfactual Trajectory Evaluation (Rung 3)
      One question per active_qa question that carries an alternate_trajectory.
      Ground truth: question["alternate_trajectory"]["waypoints"].
      The model re-answers the VQA question AND predicts the counterfactual trajectory.

Metric
------
ADE  Average Displacement Error — mean L2 distance (metres) over N matched timesteps.
minADE is the min over K sampled trajectories (metrics.py already has min_ade()).

Ground-truth alignment rules
-----------------------------
- Only interpolate GT to match prediction timesteps; NEVER extrapolate.
- If the model predicts beyond the GT horizon, trim the prediction to the GT max t.
- If the model predicts at a finer grid within the GT range, interpolate GT to those times.

CTE analysis flags
------------------
ANSWER_MISMATCH        VQA answer in CTE prompt ≠ original VQA answer.
TRAJECTORY_SAME_AS_BTE Mean displacement between CTE and BTE predictions < threshold.
GOOD_COUNTERFACTUAL    CTE trajectory is meaningfully different from BTE AND CTE ADE is below threshold.
FULL_MATCH             Answers match + trajectory distinct + CTE ADE within threshold.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation.common.utils.exceptions import EvaluationException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

BTE_THRESHOLDS: Dict[str, float] = {
    "excellent": 0.5,   # ADE < 0.5 m
    "good":      1.0,   # ADE < 1.0 m
    "fair":      2.0,   # ADE < 2.0 m
    # else: "poor"
}

CTE_THRESHOLDS: Dict[str, float] = {
    "same_as_bte_mean_disp":  0.30,  # mean CTE–BTE displacement below this → flag
    "good_ade":               1.50,  # CTE ADE below this → trajectory is plausible
    "excellent_ade":          0.75,
}

# Timesteps requested in BTE prompts (seconds)
BTE_TIMESTEPS: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Waypoint = Dict  # {"t": float, "x": float, "y": float}


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------

def extract_bte_ground_truth(state_json: dict) -> List[Waypoint]:
    """
    Extract BTE ground-truth waypoints from state.json.

    Returns ego.future entries as [{t, x, y}, ...].  The current position
    at t=0 is (0,0) by convention and is NOT included (prediction starts at t=0.5).

    Raises EvaluationException if ego.future is missing or empty.
    """
    ego = state_json.get("ego", {})
    future = ego.get("future", [])
    if not future:
        raise EvaluationException(
            "state.json missing ego.future — cannot build BTE ground truth"
        )
    waypoints: List[Waypoint] = []
    for wp in future:
        if not isinstance(wp, dict):
            continue
        t = wp.get("t")
        x = wp.get("x")
        y = wp.get("y")
        if t is None or x is None or y is None:
            logger.warning("Skipping malformed BTE waypoint: %s", wp)
            continue
        waypoints.append({"t": float(t), "x": float(x), "y": float(y)})

    if not waypoints:
        raise EvaluationException("No valid BTE waypoints found in ego.future")

    waypoints.sort(key=lambda w: w["t"])
    logger.debug(
        "extract_bte_ground_truth: %d waypoints t=[%.1f … %.1f]",
        len(waypoints), waypoints[0]["t"], waypoints[-1]["t"],
    )
    return waypoints


def _get_alternate_trajectory(question: dict) -> Optional[dict]:
    """Return alternate_trajectory dict from either meta.alternate_trajectory or top-level."""
    meta = question.get("meta") or {}
    at = meta.get("alternate_trajectory") if isinstance(meta, dict) else None
    if not at:
        at = question.get("alternate_trajectory")
    return at if isinstance(at, dict) else None


def extract_cte_ground_truth(question: dict) -> Tuple[List[Waypoint], int, str]:
    """
    Extract CTE ground-truth waypoints from alternate_trajectory in a question dict.

    Returns (waypoints, rung, eval_type_str).
    waypoints excludes t=0.0 (current position is always (0,0) and is not predicted).

    Raises EvaluationException if alternate_trajectory is absent or malformed.
    """
    at = _get_alternate_trajectory(question)
    if not at:
        raise EvaluationException(
            f"Question {question.get('id','?')} has no alternate_trajectory"
        )

    rung: int = int(at.get("rung", question.get("rung", 0)))
    raw_wps = at.get("waypoints", [])
    if not raw_wps:
        raise EvaluationException(
            f"alternate_trajectory.waypoints is empty for question {question.get('id','?')}"
        )

    waypoints: List[Waypoint] = []
    for wp in raw_wps:
        if not isinstance(wp, dict):
            continue
        t = wp.get("t")
        x = wp.get("x")
        y = wp.get("y")
        if t is None or x is None or y is None:
            logger.warning("Skipping malformed CTE waypoint: %s", wp)
            continue
        # Exclude t=0.0 — current position is always origin, not a prediction target
        if abs(float(t)) < 1e-6:
            continue
        waypoints.append({"t": float(t), "x": float(x), "y": float(y)})

    if not waypoints:
        raise EvaluationException(
            f"No future waypoints (t>0) in alternate_trajectory for question {question.get('id','?')}"
        )

    waypoints.sort(key=lambda w: w["t"])

    q_type = _get_question_type(question)
    eval_type = f"CTE_R{rung}_{q_type}"

    logger.debug(
        "extract_cte_ground_truth: q=%s rung=%d %d waypoints t=[%.1f…%.1f]",
        question.get("id", "?"), rung, len(waypoints),
        waypoints[0]["t"], waypoints[-1]["t"],
    )
    return waypoints, rung, eval_type


def _get_question_type(question: dict) -> str:
    """Derive short question-type tag from question dict (e.g. 'CDE', 'CR', 'SC')."""
    qtype = question.get("category", "")
    if not qtype:
        qid = question.get("id", "")
        qtype = re.sub(r"\d+$", "", qid)
    return qtype.upper() or "UNK"


def find_trajectory_questions(qa_data: dict) -> List[dict]:
    """
    Return every question in qa_data['questions'] that has a non-empty alternate_trajectory.

    Parameters
    ----------
    qa_data : parsed JSON from active_qa.json (dict with "questions" list)

    Returns
    -------
    List of question dicts, each with alternate_trajectory guaranteed present.
    """
    if isinstance(qa_data, list):
        questions = qa_data
    else:
        questions = qa_data.get("questions", [])
    result = []
    for q in questions:
        at = _get_alternate_trajectory(q)
        if at and at.get("waypoints"):
            result.append(q)
    logger.debug("find_trajectory_questions: %d CTE question(s) found", len(result))
    return result


def get_eval_type(question: dict) -> str:
    """Return eval_type string for a CTE question, e.g. 'CTE_R2_CDE' or 'CTE_R3_CR'."""
    at = _get_alternate_trajectory(question) or {}
    rung = int(at.get("rung", question.get("rung", 0)))
    q_type = _get_question_type(question)
    return f"CTE_R{rung}_{q_type}"


# ---------------------------------------------------------------------------
# Trajectory alignment
# ---------------------------------------------------------------------------

def interpolate_waypoints(
    gt_waypoints: List[Waypoint],
    target_times: Sequence[float],
) -> List[Waypoint]:
    """
    Interpolate ground-truth waypoints to a target set of timesteps.

    Only interpolates within the GT time range — target times outside the range
    are excluded (callers must trim predictions first).

    Parameters
    ----------
    gt_waypoints   : sorted list of {t, x, y} ground-truth waypoints
    target_times   : requested output timesteps (seconds)

    Returns
    -------
    List of {t, x, y} at every target_time that falls within [gt_t_min, gt_t_max].
    Returns empty list if gt_waypoints is empty.
    """
    if not gt_waypoints:
        return []

    gt_t = np.array([w["t"] for w in gt_waypoints], dtype=np.float64)
    gt_x = np.array([w["x"] for w in gt_waypoints], dtype=np.float64)
    gt_y = np.array([w["y"] for w in gt_waypoints], dtype=np.float64)

    t_min, t_max = gt_t.min(), gt_t.max()

    result: List[Waypoint] = []
    for t in target_times:
        t = float(t)
        if t < t_min - 1e-6 or t > t_max + 1e-6:
            continue
        x = float(np.interp(t, gt_t, gt_x))
        y = float(np.interp(t, gt_t, gt_y))
        result.append({"t": round(t, 6), "x": round(x, 6), "y": round(y, 6)})

    return result


def align_trajectories(
    gt_waypoints: List[Waypoint],
    pred_waypoints: List[Waypoint],
) -> Tuple[List[Waypoint], List[Waypoint], int]:
    """
    Align GT and predicted trajectories for ADE computation.

    Rules (per spec):
    1. Trim prediction to GT time range (no extrapolation beyond GT max t).
    2. Interpolate GT to the prediction's timesteps (within range).
    3. Pair remaining GT and pred waypoints.

    Parameters
    ----------
    gt_waypoints   : ground-truth waypoints [{t, x, y}, ...]
    pred_waypoints : predicted waypoints [{t, x, y}, ...]

    Returns
    -------
    (aligned_gt, aligned_pred, n_eval_points)
    Both lists have the same length. n_eval_points may be 0 if no overlap.
    """
    if not gt_waypoints or not pred_waypoints:
        return [], [], 0

    gt_t_min = min(w["t"] for w in gt_waypoints)
    gt_t_max = max(w["t"] for w in gt_waypoints)

    # Step 1: trim prediction to GT range
    pred_in_range = [
        w for w in pred_waypoints
        if gt_t_min - 1e-6 <= w["t"] <= gt_t_max + 1e-6
    ]
    if not pred_in_range:
        logger.warning(
            "align_trajectories: no prediction points within GT range [%.1f, %.1f]; "
            "pred range is [%.1f, %.1f]",
            gt_t_min, gt_t_max,
            min(w["t"] for w in pred_waypoints),
            max(w["t"] for w in pred_waypoints),
        )
        return [], [], 0

    # Step 2: interpolate GT to prediction timesteps
    pred_times = [w["t"] for w in pred_in_range]
    aligned_gt = interpolate_waypoints(gt_waypoints, pred_times)

    if len(aligned_gt) != len(pred_in_range):
        # Defensive: keep only times present in both
        gt_t_set = {round(w["t"], 4) for w in aligned_gt}
        pred_in_range = [w for w in pred_in_range if round(w["t"], 4) in gt_t_set]
        if not pred_in_range:
            return [], [], 0

    n = min(len(aligned_gt), len(pred_in_range))
    return aligned_gt[:n], pred_in_range[:n], n


# ---------------------------------------------------------------------------
# ADE
# ---------------------------------------------------------------------------

def compute_ade(
    gt_waypoints: List[Waypoint],
    pred_waypoints: List[Waypoint],
) -> Tuple[float, int]:
    """
    Compute Average Displacement Error (ADE) between GT and prediction.

    Both lists must be pre-aligned (same length, same timesteps).

    Returns
    -------
    (ade_metres, n_eval_points)
    ade_metres is 0.0 and n_eval_points is 0 if either list is empty.
    """
    if not gt_waypoints or not pred_waypoints:
        return 0.0, 0

    n = min(len(gt_waypoints), len(pred_waypoints))
    if n == 0:
        return 0.0, 0

    total = 0.0
    for gt, pred in zip(gt_waypoints[:n], pred_waypoints[:n]):
        dx = pred["x"] - gt["x"]
        dy = pred["y"] - gt["y"]
        total += math.sqrt(dx * dx + dy * dy)

    return round(total / n, 4), n


def compute_mean_displacement(
    waypoints_a: List[Waypoint],
    waypoints_b: List[Waypoint],
) -> float:
    """Mean L2 distance between two equally-long aligned trajectory lists."""
    if not waypoints_a or not waypoints_b:
        return 0.0
    n = min(len(waypoints_a), len(waypoints_b))
    total = 0.0
    for a, b in zip(waypoints_a[:n], waypoints_b[:n]):
        dx = a["x"] - b["x"]
        dy = a["y"] - b["y"]
        total += math.sqrt(dx * dx + dy * dy)
    return round(total / n, 4)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_bte(ade: float) -> dict:
    """
    Grade a BTE ADE score with threshold labels.

    Returns
    -------
    {"ade": float, "grade": str, "thresholds": dict}
    """
    if ade < BTE_THRESHOLDS["excellent"]:
        grade = "EXCELLENT"
    elif ade < BTE_THRESHOLDS["good"]:
        grade = "GOOD"
    elif ade < BTE_THRESHOLDS["fair"]:
        grade = "FAIR"
    else:
        grade = "POOR"

    return {
        "ade":        ade,
        "grade":      grade,
        "thresholds": BTE_THRESHOLDS,
    }


def analyze_cte(
    cte_ade: float,
    bte_ade: Optional[float],
    aligned_bte_pred: Optional[List[Waypoint]],
    aligned_cte_pred: Optional[List[Waypoint]],
    vqa_original_answer: Optional[str],
    vqa_cte_answer: Optional[str],
) -> dict:
    """
    Analyse a CTE result and assign flags + an overall grade.

    Flags (may be multiple):
      ANSWER_MISMATCH        — cte VQA answer differs from the original VQA answer
      TRAJECTORY_SAME_AS_BTE — CTE trajectory is indistinguishable from BTE
      GOOD_COUNTERFACTUAL    — CTE ≠ BTE beyond threshold, and CTE ADE is plausible
      FULL_MATCH             — answers agree + trajectory distinct + CTE ADE excellent

    Returns dict with: flags, grade, cte_ade, bte_ade, cte_bte_mean_disp, answer_match.
    """
    flags: List[str] = []

    # --- Answer comparison ---
    answer_match: Optional[bool] = None
    if vqa_original_answer is not None and vqa_cte_answer is not None:
        answer_match = (
            vqa_original_answer.strip().upper() == vqa_cte_answer.strip().upper()
        )
        if not answer_match:
            flags.append("ANSWER_MISMATCH")
            logger.info(
                "CTE ANSWER_MISMATCH: original=%r cte=%r",
                vqa_original_answer, vqa_cte_answer,
            )

    # --- Trajectory comparison with BTE ---
    cte_bte_mean_disp: Optional[float] = None
    if aligned_bte_pred and aligned_cte_pred:
        n = min(len(aligned_bte_pred), len(aligned_cte_pred))
        cte_bte_mean_disp = compute_mean_displacement(
            aligned_bte_pred[:n], aligned_cte_pred[:n]
        )
        if cte_bte_mean_disp < CTE_THRESHOLDS["same_as_bte_mean_disp"]:
            flags.append("TRAJECTORY_SAME_AS_BTE")
            logger.info(
                "CTE TRAJECTORY_SAME_AS_BTE: mean_disp=%.3f < threshold=%.2f",
                cte_bte_mean_disp, CTE_THRESHOLDS["same_as_bte_mean_disp"],
            )

    # --- CTE validity assessment ---
    traj_is_distinct = (
        cte_bte_mean_disp is not None
        and cte_bte_mean_disp >= CTE_THRESHOLDS["same_as_bte_mean_disp"]
    )
    cte_ade_good = cte_ade < CTE_THRESHOLDS["good_ade"]
    cte_ade_excellent = cte_ade < CTE_THRESHOLDS["excellent_ade"]

    if traj_is_distinct and cte_ade_good:
        flags.append("GOOD_COUNTERFACTUAL")

    if answer_match and traj_is_distinct and cte_ade_excellent:
        flags.append("FULL_MATCH")

    # --- Overall grade ---
    if "FULL_MATCH" in flags:
        grade = "EXCELLENT"
    elif "GOOD_COUNTERFACTUAL" in flags and not "ANSWER_MISMATCH" in flags:
        grade = "GOOD"
    elif "ANSWER_MISMATCH" in flags or "TRAJECTORY_SAME_AS_BTE" in flags:
        grade = "FLAGGED"
    else:
        grade = "INCONCLUSIVE"

    return {
        "flags":                flags,
        "grade":                grade,
        "answer_match":         answer_match,
        "cte_ade":              cte_ade,
        "bte_ade":              bte_ade,
        "cte_bte_mean_disp":    cte_bte_mean_disp,
        "thresholds":           CTE_THRESHOLDS,
    }


# ---------------------------------------------------------------------------
# Waypoint text parser
# ---------------------------------------------------------------------------

_WP_JSON_RE = re.compile(
    r'\{\s*"waypoints"\s*:\s*(\[.*?\])\s*\}',
    re.DOTALL | re.IGNORECASE,
)
_ANSWER_JSON_RE = re.compile(
    r'\{\s*"answer"\s*:\s*"([^"]+)"\s*,\s*"waypoints"\s*:\s*(\[.*?\])\s*\}',
    re.DOTALL | re.IGNORECASE,
)
_WP_ARRAY_RE = re.compile(
    r'\[\s*\{[^]]*"t"\s*:[^]]*\}\s*\]',
    re.DOTALL,
)


def parse_waypoints_from_text(
    text: str,
) -> Tuple[Optional[List[Waypoint]], Optional[str]]:
    """
    Extract waypoints (and optionally a VQA answer) from model output text.

    Tries in order:
      1. Full JSON {"answer": "...", "waypoints": [...]}  (CTE format)
      2. JSON {"waypoints": [...]}                         (BTE format)
      3. JSON array [...] containing dicts with "t"        (bare array)
      4. Heuristic: "t=X x=Y y=Z" patterns                (fallback)

    Parameters
    ----------
    text : raw model output string

    Returns
    -------
    (waypoints_or_None, vqa_answer_or_None)
    waypoints is a list of {t, x, y} dicts sorted by t.
    vqa_answer is the string answer for CTE questions (None for BTE).
    """
    if not text:
        return None, None

    # Attempt 1: {"answer": "...", "waypoints": [...]}
    m = _ANSWER_JSON_RE.search(text)
    if m:
        vqa_answer = m.group(1).strip()
        wps = _parse_wp_array(m.group(2))
        if wps:
            logger.debug("parse_waypoints_from_text: matched ANSWER+WAYPOINTS JSON")
            return wps, vqa_answer

    # Attempt 2: {"waypoints": [...]}
    m = _WP_JSON_RE.search(text)
    if m:
        wps = _parse_wp_array(m.group(1))
        if wps:
            logger.debug("parse_waypoints_from_text: matched WAYPOINTS JSON")
            return wps, None

    # Attempt 3: bare JSON array with "t" keys
    m = _WP_ARRAY_RE.search(text)
    if m:
        wps = _parse_wp_array(m.group(0))
        if wps:
            logger.debug("parse_waypoints_from_text: matched bare waypoint array")
            return wps, None

    # Attempt 4: heuristic "t=X x=Y y=Z" or "t:X, x:Y, y:Z"
    wps = _parse_heuristic(text)
    if wps:
        logger.debug("parse_waypoints_from_text: matched heuristic t/x/y pattern")
        return wps, None

    logger.warning("parse_waypoints_from_text: no waypoints found in output")
    return None, None


def _parse_wp_array(array_str: str) -> Optional[List[Waypoint]]:
    """Try to parse a JSON array string into waypoint dicts."""
    try:
        arr = json.loads(array_str)
    except json.JSONDecodeError:
        # Permissive: try to fix common issues (trailing commas, single quotes)
        fixed = array_str.replace("'", '"')
        fixed = re.sub(r",\s*}", "}", fixed)
        fixed = re.sub(r",\s*\]", "]", fixed)
        try:
            arr = json.loads(fixed)
        except json.JSONDecodeError:
            return None

    if not isinstance(arr, list):
        return None

    result: List[Waypoint] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        t = item.get("t")
        x = item.get("x")
        y = item.get("y")
        if t is None or x is None or y is None:
            continue
        try:
            result.append({"t": float(t), "x": float(x), "y": float(y)})
        except (TypeError, ValueError):
            continue

    if not result:
        return None
    result.sort(key=lambda w: w["t"])
    return result


_HEURISTIC_RE = re.compile(
    r"t\s*[=:]\s*(-?[\d.]+)\s*[,;]?\s*x\s*[=:]\s*(-?[\d.]+)\s*[,;]?\s*y\s*[=:]\s*(-?[\d.]+)",
    re.IGNORECASE,
)


def _parse_heuristic(text: str) -> Optional[List[Waypoint]]:
    """Fallback: extract t/x/y from free-text patterns."""
    matches = _HEURISTIC_RE.findall(text)
    if not matches:
        return None
    result = []
    for t_s, x_s, y_s in matches:
        try:
            result.append({"t": float(t_s), "x": float(x_s), "y": float(y_s)})
        except ValueError:
            continue
    if not result:
        return None
    result.sort(key=lambda w: w["t"])
    return result


# ---------------------------------------------------------------------------
# Trajectory prompt text builders
# ---------------------------------------------------------------------------

def build_bte_prompt_text(state_json: dict, gt_timesteps: List[float]) -> str:
    """
    Build the BTE trajectory prompt string from state.json and target timesteps.

    This text goes into the prompt record's ``qa_text`` field and is passed
    directly to the model's VQA interface.
    """
    ego = state_json.get("ego", {})
    speed = ego.get("speed", 0.0)
    nav_cmd = state_json.get("nav_command", "UNKNOWN")

    # Build the waypoint template the model should fill
    template_wps = ", ".join(
        f'{{"t": {t}, "x": 0.0, "y": 0.0}}' for t in gt_timesteps
    )

    return (
        "You are an autonomous driving system with access to multi-camera images "
        "from the past 1.5 seconds up to now (T=0).\n\n"
        f"Navigation command: {nav_cmd}\n"
        f"Current ego speed:  {speed:.2f} m/s\n"
        "Ego-centric frame:  (0, 0) = current position, +x = forward, +y = left\n\n"
        "Task: Predict the ego vehicle's future trajectory waypoints.\n"
        f"Output timesteps (seconds): {gt_timesteps}\n\n"
        'Output ONLY the following JSON (replace x/y with your predictions):\n'
        f'{{"waypoints": [{template_wps}]}}'
    )


def build_cte_prompt_text(
    question: dict,
    gt_timesteps: List[float],
    state_json: dict,
) -> str:
    """
    Build the CTE trajectory prompt string for a counterfactual question.

    Combines the VQA question with a request to predict the counterfactual trajectory.
    The model must output both its counterfactual VQA answer and waypoints as JSON.
    """
    q_text = question.get("question", "")
    answer_format = question.get("answer_format", "binary")
    answer_hint = "Yes or No" if answer_format == "binary" else "A, B, C, or D"

    ego = state_json.get("ego", {})
    speed = ego.get("speed", 0.0)
    nav_cmd = state_json.get("nav_command", "UNKNOWN")

    # Counterfactual intervention description
    meta = question.get("meta", {})
    ci = meta.get("counterfactual_intervention", {}) if isinstance(meta, dict) else {}
    intervention_detail = ci.get("operation_detail", "")

    template_wps = ", ".join(
        f'{{"t": {t}, "x": 0.0, "y": 0.0}}' for t in gt_timesteps
    )

    prompt = (
        "You are an autonomous driving system with access to multi-camera images "
        "from the past 1.5 seconds up to now (T=0).\n\n"
        f"Navigation command: {nav_cmd}\n"
        f"Current ego speed:  {speed:.2f} m/s\n"
        "Ego-centric frame:  (0, 0) = current position, +x = forward, +y = left\n\n"
        "=== Causal Reasoning Question ===\n"
        f"{q_text}\n"
    )
    if intervention_detail:
        prompt += f"\nCounterfactual scenario: {intervention_detail}\n"
    prompt += (
        f"\nTask 1 — Answer the question ({answer_hint}).\n"
        "Task 2 — Given your counterfactual reasoning, predict the ego vehicle's "
        f"trajectory under this scenario at timesteps: {gt_timesteps} seconds.\n\n"
        "Output ONLY the following JSON (fill in answer and x/y values):\n"
        f'{{"answer": "<{answer_hint}>", "waypoints": [{template_wps}]}}'
    )
    return prompt


# ---------------------------------------------------------------------------
# Trajectory report builder
# ---------------------------------------------------------------------------

def build_trajectory_report(
    scene_id: str,
    sample_id: str,
    bte_result: Optional[dict],
    cte_results: List[dict],
    schema_version: str = "1.0",
) -> dict:
    """
    Assemble the trajectory_report.json structure for one sample.

    Parameters
    ----------
    bte_result   : dict from evaluate_bte_output() or None if unavailable
    cte_results  : list of dicts from evaluate_cte_output()
    """
    return {
        "schema_version":  schema_version,
        "level":           "sample",
        "scene_id":        scene_id,
        "sample_id":       sample_id,
        "bte":             bte_result,
        "cte":             cte_results,
        "n_cte":           len(cte_results),
        "summary": _summarize(bte_result, cte_results),
    }


def _summarize(bte: Optional[dict], cte_list: List[dict]) -> dict:
    summary: dict = {}
    if bte:
        summary["bte_ade"]   = bte.get("ade")
        summary["bte_grade"] = bte.get("analysis", {}).get("grade")
    if cte_list:
        ades = [c["ade"] for c in cte_list if c.get("ade") is not None]
        summary["cte_mean_ade"] = round(sum(ades) / len(ades), 4) if ades else None
        grades = [c.get("analysis", {}).get("grade", "") for c in cte_list]
        summary["cte_grades"]   = grades
        flags = []
        for c in cte_list:
            flags.extend(c.get("analysis", {}).get("flags", []))
        summary["all_flags"] = sorted(set(flags))
    return summary


# ---------------------------------------------------------------------------
# Single-output evaluators
# ---------------------------------------------------------------------------

def evaluate_bte_output(
    raw_text: str,
    gt_waypoints: List[Waypoint],
    question_id: str = "traj_BTE",
) -> dict:
    """
    Parse model output and compute BTE ADE.

    Returns a result dict suitable for trajectory_report.json["bte"].
    Never raises — errors are captured in the result with parse_success=False.
    """
    pred_wps, _ = parse_waypoints_from_text(raw_text)
    if pred_wps is None:
        logger.warning("[evaluate_bte] Could not parse waypoints for %s", question_id)
        return {
            "question_id":   question_id,
            "eval_type":     "BTE",
            "parse_success": False,
            "error":         "waypoints not found in model output",
            "raw_text_excerpt": raw_text[:300],
        }

    aligned_gt, aligned_pred, n_eval = align_trajectories(gt_waypoints, pred_wps)
    if n_eval == 0:
        return {
            "question_id":   question_id,
            "eval_type":     "BTE",
            "parse_success": True,
            "n_eval_points": 0,
            "error":         "no timestep overlap between GT and prediction",
            "gt_t_range":    [gt_waypoints[0]["t"], gt_waypoints[-1]["t"]],
            "pred_t_range":  [pred_wps[0]["t"], pred_wps[-1]["t"]],
        }

    ade, _ = compute_ade(aligned_gt, aligned_pred)
    analysis = analyze_bte(ade)

    return {
        "question_id":        question_id,
        "eval_type":          "BTE",
        "parse_success":      True,
        "ade":                ade,
        "n_waypoints_gt":     len(gt_waypoints),
        "n_waypoints_pred":   len(pred_wps),
        "n_eval_points":      n_eval,
        "gt_t_range":         [gt_waypoints[0]["t"], gt_waypoints[-1]["t"]],
        "analysis":           analysis,
        "ground_truth_waypoints":  aligned_gt,
        "predicted_waypoints":     aligned_pred,
    }


def evaluate_cte_output(
    raw_text: str,
    gt_waypoints: List[Waypoint],
    eval_type: str,
    question_id: str,
    parent_question_id: str,
    original_vqa_answer: Optional[str],
    bte_result: Optional[dict],
) -> dict:
    """
    Parse model output and compute CTE ADE + analysis flags.

    Returns a result dict for trajectory_report.json["cte"][i].
    """
    pred_wps, cte_vqa_answer = parse_waypoints_from_text(raw_text)
    if pred_wps is None:
        logger.warning(
            "[evaluate_cte] Could not parse waypoints for %s (q=%s)",
            eval_type, parent_question_id,
        )
        return {
            "question_id":        question_id,
            "parent_question_id": parent_question_id,
            "eval_type":          eval_type,
            "parse_success":      False,
            "vqa_answer_cte":     cte_vqa_answer,
            "vqa_answer_original": original_vqa_answer,
            "error":              "waypoints not found in model output",
            "raw_text_excerpt":   raw_text[:300],
        }

    aligned_gt, aligned_pred, n_eval = align_trajectories(gt_waypoints, pred_wps)
    if n_eval == 0:
        return {
            "question_id":        question_id,
            "parent_question_id": parent_question_id,
            "eval_type":          eval_type,
            "parse_success":      True,
            "n_eval_points":      0,
            "vqa_answer_cte":     cte_vqa_answer,
            "vqa_answer_original": original_vqa_answer,
            "error":              "no timestep overlap between GT and prediction",
        }

    ade, _ = compute_ade(aligned_gt, aligned_pred)

    # BTE predicted waypoints for same timesteps (for same-as-BTE check)
    bte_pred_aligned: Optional[List[Waypoint]] = None
    bte_ade: Optional[float] = None
    if bte_result and bte_result.get("parse_success") and bte_result.get("predicted_waypoints"):
        bte_preds = bte_result["predicted_waypoints"]
        pred_times = [w["t"] for w in aligned_pred]
        bte_pred_aligned = interpolate_waypoints(bte_preds, pred_times)
        bte_ade = bte_result.get("ade")

    analysis = analyze_cte(
        cte_ade=ade,
        bte_ade=bte_ade,
        aligned_bte_pred=bte_pred_aligned,
        aligned_cte_pred=aligned_pred,
        vqa_original_answer=original_vqa_answer,
        vqa_cte_answer=cte_vqa_answer,
    )

    return {
        "question_id":            question_id,
        "parent_question_id":     parent_question_id,
        "eval_type":              eval_type,
        "parse_success":          True,
        "ade":                    ade,
        "n_waypoints_gt":         len(gt_waypoints),
        "n_waypoints_pred":       len(pred_wps),
        "n_eval_points":          n_eval,
        "gt_t_range":             [gt_waypoints[0]["t"], gt_waypoints[-1]["t"]],
        "vqa_answer_cte":         cte_vqa_answer,
        "vqa_answer_original":    original_vqa_answer,
        "analysis":               analysis,
        "ground_truth_waypoints": aligned_gt,
        "predicted_waypoints":    aligned_pred,
    }
