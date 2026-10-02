"""
Unit tests for evaluation/common/trajectory_eval.py

Tests cover:
  - Ground truth extraction (BTE from state.json, CTE from alternate_trajectory)
  - find_trajectory_questions() filtering
  - Trajectory alignment and interpolation
  - ADE computation
  - BTE analysis grading
  - CTE analysis flags
  - Waypoint text parsing (JSON and heuristic)
  - Full evaluate_bte_output() and evaluate_cte_output() pipelines
  - build_trajectory_report() structure

Integration tests use sample_data/causal_nuscenes/nuscenes-scene-0072/SAMPLED_1
(rung=3 alternate_trajectory, CR1).  For scene-0074 (rung=2, CDE1), tests are
skipped unless the full benchmark dataset is available.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common.trajectory_eval import (
    BTE_THRESHOLDS,
    CTE_THRESHOLDS,
    BTE_TIMESTEPS,
    align_trajectories,
    analyze_bte,
    analyze_cte,
    build_trajectory_report,
    compute_ade,
    compute_mean_displacement,
    evaluate_bte_output,
    evaluate_cte_output,
    extract_bte_ground_truth,
    extract_cte_ground_truth,
    find_trajectory_questions,
    get_eval_type,
    interpolate_waypoints,
    parse_waypoints_from_text,
)
from evaluation.common.utils.exceptions import EvaluationException

# ---------------------------------------------------------------------------
# Sample data paths
# ---------------------------------------------------------------------------

# scene-0072/SAMPLED_1 is included in sample_data (always available)
_SAMPLE_DATA = Path(__file__).resolve().parents[2] / "evaluation" / "sample_data"
_S0072 = _SAMPLE_DATA / "causal_nuscenes" / "nuscenes-scene-0072" / "SAMPLED_1"

# scene-0074 is only in the full benchmark dataset
_BENCH = Path("/datadrive/datasets/causal_drive_bench_20260427/causal_nuscenes")
_S0074 = _BENCH / "nuscenes-scene-0074" / "SAMPLED_1"

_REAL_DATA = _BENCH.exists()


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def state_json_simple():
    """Minimal state.json with 6 future waypoints."""
    return {
        "t_rel_s": 0.0,
        "nav_command": "GO STRAIGHT",
        "ego": {
            "speed": 0.528,
            "history": [
                {"t": -1.5, "x": -1.31, "y": -0.02, "speed": 1.514},
                {"t": -1.0, "x": -0.71, "y":  0.00, "speed": 1.196},
                {"t": -0.5, "x": -0.26, "y":  0.01, "speed": 0.902},
            ],
            "future": [
                {"t": 0.5, "x": 0.14, "y": -0.01, "speed": 0.28},
                {"t": 1.0, "x": 0.20, "y": -0.02, "speed": 0.11},
                {"t": 1.5, "x": 0.19, "y": -0.02, "speed": 0.02},
                {"t": 2.0, "x": 0.18, "y": -0.03, "speed": 0.01},
                {"t": 2.5, "x": 0.18, "y": -0.03, "speed": 0.01},
                {"t": 3.0, "x": 0.18, "y": -0.04, "speed": 0.01},
            ],
        },
        "agents": {},
    }


@pytest.fixture
def question_with_alt_traj_r3():
    """Minimal question dict with rung-3 alternate_trajectory."""
    return {
        "id": "CR1",
        "graph_structure": "Direct",
        "rung": 3,
        "category": "CR",
        "question": "Would you still decelerate if the crosswalk were removed?",
        "answer_format": "binary",
        "correct_answer": "No",
        "meta": {
            "causal_status": "active_sustaining",
            "counterfactual_intervention": {
                "operation_detail": "Crosswalk removed",
            },
        },
        "alternate_trajectory": {
            "frame": "T=-1.5",
            "rung": 3,
            "waypoints": [
                {"t": 0.0, "x": 0.0,   "y": 0.0,   "heading": 0.0,    "speed": 0.528},
                {"t": 0.5, "x": 0.582, "y": -0.19, "heading": -0.316, "speed": 1.8},
                {"t": 1.0, "x": 1.782, "y": -0.16, "heading":  0.025, "speed": 3.0},
                {"t": 1.5, "x": 3.582, "y":  0.0,  "heading":  0.089, "speed": 4.2},
            ],
        },
    }


@pytest.fixture
def question_with_alt_traj_r2():
    """Minimal question dict with rung-2 alternate_trajectory."""
    return {
        "id": "CDE1",
        "graph_structure": "Confounding",
        "rung": 2,
        "category": "CDE",
        "question": "Would the stop sign still have played a role?",
        "answer_format": "binary",
        "correct_answer": "Yes",
        "meta": {
            "causal_status": "active_triggering",
            "counterfactual_intervention": {
                "operation_detail": "Sedan frozen at current position",
            },
        },
        "alternate_trajectory": {
            "frame": "T=0",
            "rung": 2,
            "waypoints": [
                {"t": 0.5, "x": 0.632, "y": 0.399, "heading": 0.563, "speed": 1.0},
                {"t": 1.0, "x": 0.957, "y": 0.625, "heading": 0.608, "speed": 0.3},
                {"t": 1.5, "x": 1.032, "y": 0.623, "heading": -0.027, "speed": 0.0},
                {"t": 2.0, "x": 1.032, "y": 0.623, "heading": -0.027, "speed": 0.0},
                {"t": 2.5, "x": 1.107, "y": 0.618, "heading": -0.067, "speed": 0.3},
                {"t": 3.0, "x": 1.382, "y": 0.836, "heading": 0.670, "speed": 0.8},
                {"t": 3.5, "x": 1.882, "y": 1.110, "heading": 0.501, "speed": 1.2},
                {"t": 4.0, "x": 2.557, "y": 1.412, "heading": 0.421, "speed": 1.5},
            ],
        },
    }


@pytest.fixture
def qa_data_with_traj(question_with_alt_traj_r3, question_with_alt_traj_r2):
    """QA data dict with both trajectory questions and one without."""
    return {
        "questions": [
            {
                "id": "CaI1",
                "rung": 0,
                "category": "CaI",
                "question": "What is blocking you?",
                "answer_format": "mcq",
                "correct_answer": "A",
                "meta": {},
                # no alternate_trajectory
            },
            question_with_alt_traj_r3,
            question_with_alt_traj_r2,
        ]
    }


# ===========================================================================
# Tests: extract_bte_ground_truth
# ===========================================================================

class TestExtractBteGroundTruth:

    def test_basic_extraction(self, state_json_simple):
        wps = extract_bte_ground_truth(state_json_simple)
        assert len(wps) == 6
        assert wps[0]["t"] == pytest.approx(0.5)
        assert wps[-1]["t"] == pytest.approx(3.0)
        assert all("x" in w and "y" in w for w in wps)

    def test_sorted_by_time(self, state_json_simple):
        # Shuffle the future list
        state_json_simple["ego"]["future"].reverse()
        wps = extract_bte_ground_truth(state_json_simple)
        times = [w["t"] for w in wps]
        assert times == sorted(times)

    def test_t0_not_included(self, state_json_simple):
        state_json_simple["ego"]["future"].insert(0, {"t": 0.0, "x": 0.0, "y": 0.0, "speed": 0.5})
        wps = extract_bte_ground_truth(state_json_simple)
        times = [w["t"] for w in wps]
        assert 0.0 in times  # t=0 is NOT filtered here (BTE GT includes all future points)

    def test_raises_on_missing_future(self):
        with pytest.raises(EvaluationException, match="missing ego.future"):
            extract_bte_ground_truth({"ego": {}, "nav_command": "GO"})

    def test_raises_on_empty_future(self):
        with pytest.raises(EvaluationException):
            extract_bte_ground_truth({"ego": {"future": []}})

    def test_sample_scene_0072(self):
        with open(_S0072 / "state.json") as f:
            state = json.load(f)
        wps = extract_bte_ground_truth(state)
        assert len(wps) == 6
        assert wps[0]["t"] == pytest.approx(0.5, abs=0.01)
        assert wps[0]["x"] == pytest.approx(0.1406, abs=0.01)

    @pytest.mark.skipif(not _REAL_DATA, reason="full benchmark dataset not available")
    def test_real_scene_0074(self):
        with open(_S0074 / "state.json") as f:
            state = json.load(f)
        wps = extract_bte_ground_truth(state)
        assert len(wps) == 6
        assert wps[-1]["t"] == pytest.approx(3.0, abs=0.01)


# ===========================================================================
# Tests: extract_cte_ground_truth
# ===========================================================================

class TestExtractCteGroundTruth:

    def test_rung3_filters_t0(self, question_with_alt_traj_r3):
        wps, rung, eval_type = extract_cte_ground_truth(question_with_alt_traj_r3)
        times = [w["t"] for w in wps]
        assert 0.0 not in times          # t=0 excluded
        assert 0.5 in times
        assert rung == 3
        assert eval_type == "CTE_R3_CR"

    def test_rung2_timesteps(self, question_with_alt_traj_r2):
        wps, rung, eval_type = extract_cte_ground_truth(question_with_alt_traj_r2)
        assert rung == 2
        assert eval_type == "CTE_R2_CDE"
        assert len(wps) == 8
        assert wps[0]["t"] == pytest.approx(0.5)
        assert wps[-1]["t"] == pytest.approx(4.0)

    def test_missing_alternate_trajectory(self):
        with pytest.raises(EvaluationException, match="no alternate_trajectory"):
            extract_cte_ground_truth({"id": "CR1"})

    def test_empty_waypoints(self):
        q = {"id": "X1", "alternate_trajectory": {"rung": 2, "waypoints": []}}
        with pytest.raises(EvaluationException):
            extract_cte_ground_truth(q)

    def test_sample_scene_0072_cr1(self):
        with open(_S0072 / "qa" / "active_qa.json") as f:
            qa = json.load(f)
        traj_qs = find_trajectory_questions(qa)
        assert len(traj_qs) >= 1
        q = next(q for q in traj_qs if q["id"] == "CR1")
        wps, rung, eval_type = extract_cte_ground_truth(q)
        assert rung == 3
        assert eval_type == "CTE_R3_CR"
        assert all(w["t"] > 0 for w in wps)

    @pytest.mark.skipif(not _REAL_DATA, reason="full benchmark dataset not available")
    def test_real_scene_0074_cde1(self):
        with open(_S0074 / "qa" / "active_qa.json") as f:
            qa = json.load(f)
        traj_qs = find_trajectory_questions(qa)
        q = next(q for q in traj_qs if q["id"] == "CDE1")
        wps, rung, eval_type = extract_cte_ground_truth(q)
        assert rung == 2
        assert len(wps) == 8
        assert wps[0]["x"] == pytest.approx(0.632, abs=0.01)


# ===========================================================================
# Tests: find_trajectory_questions
# ===========================================================================

class TestFindTrajectoryQuestions:

    def test_finds_correct_count(self, qa_data_with_traj):
        result = find_trajectory_questions(qa_data_with_traj)
        assert len(result) == 2

    def test_returns_only_with_alt_traj(self, qa_data_with_traj):
        result = find_trajectory_questions(qa_data_with_traj)
        ids = {q["id"] for q in result}
        assert "CaI1" not in ids
        assert "CR1" in ids
        assert "CDE1" in ids

    def test_empty_qa_data(self):
        assert find_trajectory_questions({"questions": []}) == []

    def test_list_format(self, question_with_alt_traj_r3):
        result = find_trajectory_questions([question_with_alt_traj_r3])
        assert len(result) == 1


# ===========================================================================
# Tests: interpolate_waypoints
# ===========================================================================

class TestInterpolateWaypoints:

    @pytest.fixture
    def gt_wps(self):
        return [
            {"t": 0.5, "x": 0.0, "y": 0.0},
            {"t": 1.0, "x": 1.0, "y": 0.5},
            {"t": 1.5, "x": 2.0, "y": 1.0},
            {"t": 2.0, "x": 3.0, "y": 1.5},
            {"t": 2.5, "x": 4.0, "y": 2.0},
            {"t": 3.0, "x": 5.0, "y": 2.5},
        ]

    def test_same_timesteps(self, gt_wps):
        result = interpolate_waypoints(gt_wps, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        assert len(result) == 6
        assert result[0]["x"] == pytest.approx(0.0)
        assert result[-1]["x"] == pytest.approx(5.0)

    def test_midpoint_interpolation(self, gt_wps):
        result = interpolate_waypoints(gt_wps, [0.75])
        assert len(result) == 1
        assert result[0]["x"] == pytest.approx(0.5, abs=0.01)
        assert result[0]["y"] == pytest.approx(0.25, abs=0.01)

    def test_excludes_outside_range(self, gt_wps):
        result = interpolate_waypoints(gt_wps, [0.1, 0.5, 3.5, 4.0])
        ts = [w["t"] for w in result]
        assert 0.1 not in ts
        assert 3.5 not in ts

    def test_empty_gt(self):
        assert interpolate_waypoints([], [1.0, 2.0]) == []

    def test_subset_of_gt_times(self, gt_wps):
        result = interpolate_waypoints(gt_wps, [0.5, 1.5, 3.0])
        assert len(result) == 3
        assert result[1]["x"] == pytest.approx(2.0)


# ===========================================================================
# Tests: align_trajectories
# ===========================================================================

class TestAlignTrajectories:

    @pytest.fixture
    def gt_wps(self):
        return [
            {"t": 0.5, "x": 0.0, "y": 0.0},
            {"t": 1.0, "x": 1.0, "y": 0.5},
            {"t": 1.5, "x": 2.0, "y": 1.0},
            {"t": 2.0, "x": 3.0, "y": 1.5},
            {"t": 2.5, "x": 4.0, "y": 2.0},
            {"t": 3.0, "x": 5.0, "y": 2.5},
        ]

    def test_same_length(self, gt_wps):
        pred = [{"t": t, "x": 0.0, "y": 0.0} for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]
        aligned_gt, aligned_pred, n = align_trajectories(gt_wps, pred)
        assert n == 6
        assert len(aligned_gt) == len(aligned_pred) == 6

    def test_pred_shorter(self, gt_wps):
        pred = [{"t": t, "x": 0.0, "y": 0.0} for t in [0.5, 1.0, 1.5]]
        _, _, n = align_trajectories(gt_wps, pred)
        assert n == 3  # GT trimmed/interpolated to pred length

    def test_pred_longer_trimmed(self, gt_wps):
        pred = [{"t": t, "x": 0.0, "y": 0.0}
                for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]]
        _, aligned_pred, n = align_trajectories(gt_wps, pred)
        # Prediction is trimmed to GT max (3.0)
        assert n == 6
        max_t = max(w["t"] for w in aligned_pred)
        assert max_t <= 3.0 + 1e-6

    def test_no_overlap(self, gt_wps):
        pred = [{"t": 4.0, "x": 0.0, "y": 0.0}, {"t": 5.0, "x": 1.0, "y": 0.0}]
        _, _, n = align_trajectories(gt_wps, pred)
        assert n == 0

    def test_empty_inputs(self, gt_wps):
        assert align_trajectories([], gt_wps)[2] == 0
        assert align_trajectories(gt_wps, [])[2] == 0


# ===========================================================================
# Tests: compute_ade
# ===========================================================================

class TestComputeAde:

    def test_perfect_prediction(self):
        gt   = [{"t": 0.5, "x": 1.0, "y": 0.0}, {"t": 1.0, "x": 2.0, "y": 0.0}]
        pred = [{"t": 0.5, "x": 1.0, "y": 0.0}, {"t": 1.0, "x": 2.0, "y": 0.0}]
        ade_val, n = compute_ade(gt, pred)
        assert ade_val == pytest.approx(0.0)
        assert n == 2

    def test_unit_offset(self):
        gt   = [{"t": 0.5, "x": 0.0, "y": 0.0}, {"t": 1.0, "x": 1.0, "y": 0.0}]
        pred = [{"t": 0.5, "x": 1.0, "y": 0.0}, {"t": 1.0, "x": 2.0, "y": 0.0}]
        ade_val, n = compute_ade(gt, pred)
        assert ade_val == pytest.approx(1.0)
        assert n == 2

    def test_diagonal_offset(self):
        gt   = [{"t": 0.5, "x": 0.0, "y": 0.0}]
        pred = [{"t": 0.5, "x": 3.0, "y": 4.0}]
        ade_val, n = compute_ade(gt, pred)
        assert ade_val == pytest.approx(5.0)

    def test_empty(self):
        ade_val, n = compute_ade([], [])
        assert ade_val == 0.0
        assert n == 0


# ===========================================================================
# Tests: analyze_bte
# ===========================================================================

class TestAnalyzeBte:

    def test_excellent(self):
        result = analyze_bte(0.3)
        assert result["grade"] == "EXCELLENT"

    def test_good(self):
        result = analyze_bte(0.8)
        assert result["grade"] == "GOOD"

    def test_fair(self):
        result = analyze_bte(1.5)
        assert result["grade"] == "FAIR"

    def test_poor(self):
        result = analyze_bte(3.0)
        assert result["grade"] == "POOR"

    def test_threshold_boundary(self):
        assert analyze_bte(BTE_THRESHOLDS["excellent"])["grade"] == "GOOD"
        assert analyze_bte(BTE_THRESHOLDS["good"])["grade"] == "FAIR"


# ===========================================================================
# Tests: analyze_cte
# ===========================================================================

class TestAnalyzeCte:

    def _make_wps(self, xs):
        return [{"t": 0.5 * (i + 1), "x": x, "y": 0.0} for i, x in enumerate(xs)]

    def test_full_match(self):
        bte_pred = self._make_wps([0.1, 0.2, 0.3])
        cte_pred = self._make_wps([1.0, 2.0, 3.0])  # clearly different from BTE
        result = analyze_cte(
            cte_ade=0.5,
            bte_ade=1.5,
            aligned_bte_pred=bte_pred,
            aligned_cte_pred=cte_pred,
            vqa_original_answer="No",
            vqa_cte_answer="No",
        )
        assert "FULL_MATCH" in result["flags"]
        assert result["grade"] == "EXCELLENT"

    def test_answer_mismatch(self):
        result = analyze_cte(
            cte_ade=0.4,
            bte_ade=1.5,
            aligned_bte_pred=None,
            aligned_cte_pred=None,
            vqa_original_answer="No",
            vqa_cte_answer="Yes",
        )
        assert "ANSWER_MISMATCH" in result["flags"]
        assert result["grade"] == "FLAGGED"

    def test_trajectory_same_as_bte(self):
        wps = self._make_wps([0.1, 0.2, 0.3])
        result = analyze_cte(
            cte_ade=0.5,
            bte_ade=0.5,
            aligned_bte_pred=wps,
            aligned_cte_pred=wps,  # identical → SAME_AS_BTE
            vqa_original_answer="Yes",
            vqa_cte_answer="Yes",
        )
        assert "TRAJECTORY_SAME_AS_BTE" in result["flags"]
        assert result["grade"] == "FLAGGED"

    def test_no_vqa_answer_provided(self):
        result = analyze_cte(
            cte_ade=0.5,
            bte_ade=None,
            aligned_bte_pred=None,
            aligned_cte_pred=None,
            vqa_original_answer=None,
            vqa_cte_answer=None,
        )
        assert result["answer_match"] is None
        assert "ANSWER_MISMATCH" not in result["flags"]


# ===========================================================================
# Tests: parse_waypoints_from_text
# ===========================================================================

class TestParseWaypointsFromText:

    def test_waypoints_only_json(self):
        text = '{"waypoints": [{"t": 0.5, "x": 1.2, "y": 0.3}, {"t": 1.0, "x": 2.4, "y": 0.6}]}'
        wps, ans = parse_waypoints_from_text(text)
        assert wps is not None
        assert len(wps) == 2
        assert wps[0]["x"] == pytest.approx(1.2)
        assert ans is None

    def test_answer_and_waypoints_json(self):
        text = '{"answer": "No", "waypoints": [{"t": 0.5, "x": 0.58, "y": -0.19}]}'
        wps, ans = parse_waypoints_from_text(text)
        assert wps is not None
        assert ans == "No"
        assert wps[0]["x"] == pytest.approx(0.58)

    def test_embedded_in_prose(self):
        text = (
            "The vehicle would accelerate through the intersection. "
            'Based on my analysis: {"waypoints": [{"t": 0.5, "x": 0.5, "y": 0.1}]}'
        )
        wps, _ = parse_waypoints_from_text(text)
        assert wps is not None
        assert len(wps) == 1

    def test_heuristic_fallback(self):
        text = "t=0.5 x=1.2 y=-0.3, t=1.0 x=2.4 y=-0.6"
        wps, _ = parse_waypoints_from_text(text)
        assert wps is not None
        assert len(wps) == 2
        assert wps[0]["t"] == pytest.approx(0.5)
        assert wps[0]["x"] == pytest.approx(1.2)

    def test_empty_text(self):
        wps, ans = parse_waypoints_from_text("")
        assert wps is None
        assert ans is None

    def test_no_waypoints(self):
        text = "Answer: Yes. The vehicle would decelerate."
        wps, _ = parse_waypoints_from_text(text)
        assert wps is None

    def test_sorted_by_time(self):
        text = '{"waypoints": [{"t": 1.0, "x": 2.0, "y": 0.0}, {"t": 0.5, "x": 1.0, "y": 0.0}]}'
        wps, _ = parse_waypoints_from_text(text)
        assert wps[0]["t"] == pytest.approx(0.5)
        assert wps[1]["t"] == pytest.approx(1.0)

    def test_trailing_comma_tolerant(self):
        text = '{"waypoints": [{"t": 0.5, "x": 1.0, "y": 0.0,}]}'
        wps, _ = parse_waypoints_from_text(text)
        assert wps is not None


# ===========================================================================
# Tests: evaluate_bte_output (end-to-end)
# ===========================================================================

class TestEvaluateBteOutput:

    @pytest.fixture
    def gt_wps(self):
        return [
            {"t": 0.5, "x": 0.14, "y": -0.01},
            {"t": 1.0, "x": 0.20, "y": -0.02},
            {"t": 1.5, "x": 0.19, "y": -0.02},
            {"t": 2.0, "x": 0.18, "y": -0.03},
            {"t": 2.5, "x": 0.18, "y": -0.03},
            {"t": 3.0, "x": 0.18, "y": -0.04},
        ]

    def test_perfect_output(self, gt_wps):
        text = json.dumps({
            "waypoints": [{"t": w["t"], "x": w["x"], "y": w["y"]} for w in gt_wps]
        })
        result = evaluate_bte_output(text, gt_wps)
        assert result["parse_success"] is True
        assert result["ade"] == pytest.approx(0.0, abs=0.001)
        assert result["analysis"]["grade"] == "EXCELLENT"

    def test_offset_prediction(self, gt_wps):
        shifted = json.dumps({
            "waypoints": [{"t": w["t"], "x": w["x"] + 1.0, "y": w["y"]} for w in gt_wps]
        })
        result = evaluate_bte_output(shifted, gt_wps)
        assert result["parse_success"] is True
        assert result["ade"] == pytest.approx(1.0, abs=0.01)

    def test_no_waypoints_in_output(self, gt_wps):
        result = evaluate_bte_output("The vehicle will stop.", gt_wps)
        assert result["parse_success"] is False
        assert "error" in result

    def test_shorter_prediction_trimmed(self, gt_wps):
        partial = json.dumps({
            "waypoints": [{"t": 0.5, "x": 0.14, "y": -0.01},
                          {"t": 1.0, "x": 0.20, "y": -0.02}]
        })
        result = evaluate_bte_output(partial, gt_wps)
        assert result["parse_success"] is True
        assert result["n_eval_points"] == 2

    def test_longer_prediction_trimmed(self, gt_wps):
        extended = [{"t": 0.5 * (i + 1), "x": 0.1, "y": 0.0} for i in range(10)]
        text = json.dumps({"waypoints": extended})
        result = evaluate_bte_output(text, gt_wps)
        assert result["parse_success"] is True
        assert result["n_eval_points"] <= len(gt_wps)


# ===========================================================================
# Tests: evaluate_cte_output (end-to-end)
# ===========================================================================

class TestEvaluateCteOutput:

    @pytest.fixture
    def gt_wps(self):
        return [
            {"t": 0.5, "x": 0.582, "y": -0.19},
            {"t": 1.0, "x": 1.782, "y": -0.16},
            {"t": 1.5, "x": 3.582, "y":  0.00},
        ]

    @pytest.fixture
    def bte_result_slow(self):
        return {
            "parse_success": True,
            "ade": 1.5,
            "predicted_waypoints": [
                {"t": 0.5, "x": 0.14, "y": -0.01},
                {"t": 1.0, "x": 0.20, "y": -0.02},
                {"t": 1.5, "x": 0.19, "y": -0.02},
            ],
        }

    def test_good_counterfactual(self, gt_wps, bte_result_slow):
        text = json.dumps({
            "answer": "No",
            "waypoints": [{"t": w["t"], "x": w["x"], "y": w["y"]} for w in gt_wps]
        })
        result = evaluate_cte_output(
            raw_text=text,
            gt_waypoints=gt_wps,
            eval_type="CTE_R3_CR",
            question_id="traj_CTE_CR1",
            parent_question_id="CR1",
            original_vqa_answer="No",
            bte_result=bte_result_slow,
        )
        assert result["parse_success"] is True
        assert result["ade"] == pytest.approx(0.0, abs=0.001)
        assert result["vqa_answer_cte"] == "No"
        assert "GOOD_COUNTERFACTUAL" in result["analysis"]["flags"]

    def test_answer_mismatch_flagged(self, gt_wps, bte_result_slow):
        text = json.dumps({
            "answer": "Yes",  # wrong answer
            "waypoints": [{"t": w["t"], "x": w["x"], "y": w["y"]} for w in gt_wps]
        })
        result = evaluate_cte_output(
            raw_text=text,
            gt_waypoints=gt_wps,
            eval_type="CTE_R3_CR",
            question_id="traj_CTE_CR1",
            parent_question_id="CR1",
            original_vqa_answer="No",
            bte_result=bte_result_slow,
        )
        assert "ANSWER_MISMATCH" in result["analysis"]["flags"]

    def test_parse_failure(self, gt_wps, bte_result_slow):
        result = evaluate_cte_output(
            raw_text="I would accelerate through the intersection.",
            gt_waypoints=gt_wps,
            eval_type="CTE_R3_CR",
            question_id="traj_CTE_CR1",
            parent_question_id="CR1",
            original_vqa_answer="No",
            bte_result=bte_result_slow,
        )
        assert result["parse_success"] is False


# ===========================================================================
# Tests: build_trajectory_report
# ===========================================================================

class TestBuildTrajectoryReport:

    def test_structure(self):
        bte = {"parse_success": True, "ade": 0.4, "analysis": {"grade": "EXCELLENT"}}
        cte = [{"parse_success": True, "ade": 0.3, "analysis": {"grade": "EXCELLENT", "flags": []}}]
        report = build_trajectory_report("scene-0072", "SAMPLED_1", bte, cte)
        assert report["scene_id"] == "scene-0072"
        assert report["sample_id"] == "SAMPLED_1"
        assert report["bte"] is bte
        assert report["n_cte"] == 1
        assert "summary" in report

    def test_summary_bte_grade(self):
        bte = {"parse_success": True, "ade": 2.5, "analysis": {"grade": "POOR"}}
        report = build_trajectory_report("s", "S", bte, [])
        assert report["summary"]["bte_grade"] == "POOR"

    def test_no_bte(self):
        report = build_trajectory_report("s", "S", None, [])
        assert report["bte"] is None
        assert "bte_ade" not in report["summary"]


# ===========================================================================
# Sample data round-trip (always runs — uses sample_data/nuscenes-scene-0072)
# ===========================================================================

class TestSampleDataRoundTrip:
    """End-to-end BTE/CTE tests using sample_data (no external dataset needed)."""

    def test_scene_0072_bte_gt_extraction(self):
        with open(_S0072 / "state.json") as f:
            state = json.load(f)
        wps = extract_bte_ground_truth(state)
        assert len(wps) == 6
        assert abs(wps[0]["x"]) < 0.5  # vehicle near stop at t=0.5

    def test_scene_0072_cte_gt_extraction(self):
        with open(_S0072 / "qa" / "active_qa.json") as f:
            qa = json.load(f)
        traj_qs = find_trajectory_questions(qa)
        assert len(traj_qs) == 1
        wps, rung, et = extract_cte_ground_truth(traj_qs[0])
        assert rung == 3
        assert wps[0]["x"] == pytest.approx(0.582, abs=0.01)  # accelerating in counterfactual

    def test_scene_0072_perfect_cte_scores_zero_ade(self):
        with open(_S0072 / "qa" / "active_qa.json") as f:
            qa = json.load(f)
        with open(_S0072 / "state.json") as f:
            state = json.load(f)

        traj_qs = find_trajectory_questions(qa)
        gt_wps, rung, et = extract_cte_ground_truth(traj_qs[0])
        bte_gt = extract_bte_ground_truth(state)

        bte_mock_result = {
            "parse_success": True,
            "ade": 0.05,
            "predicted_waypoints": bte_gt,
        }

        perfect_output = json.dumps({
            "answer": "No",
            "waypoints": [{"t": w["t"], "x": w["x"], "y": w["y"]} for w in gt_wps],
        })
        result = evaluate_cte_output(
            raw_text=perfect_output,
            gt_waypoints=gt_wps,
            eval_type=et,
            question_id="traj_CTE_CR1",
            parent_question_id="CR1",
            original_vqa_answer="No",
            bte_result=bte_mock_result,
        )
        assert result["parse_success"] is True
        assert result["ade"] == pytest.approx(0.0, abs=0.001)
        assert "GOOD_COUNTERFACTUAL" in result["analysis"]["flags"]
        assert "TRAJECTORY_SAME_AS_BTE" not in result["analysis"]["flags"]


# ===========================================================================
# Real benchmark round-trip (skipped unless full dataset is present)
# ===========================================================================

@pytest.mark.skipif(not _REAL_DATA, reason="full benchmark dataset not available")
class TestRealDataRoundTrip:

    def test_scene_0074_bte_and_cte_gt(self):
        with open(_S0074 / "state.json") as f:
            state = json.load(f)
        with open(_S0074 / "qa" / "active_qa.json") as f:
            qa = json.load(f)

        bte_wps = extract_bte_ground_truth(state)
        assert len(bte_wps) == 6

        traj_qs = find_trajectory_questions(qa)
        assert len(traj_qs) == 1
        cte_wps, rung, et = extract_cte_ground_truth(traj_qs[0])
        assert rung == 2
        assert et == "CTE_R2_CDE"
        assert len(cte_wps) == 8
