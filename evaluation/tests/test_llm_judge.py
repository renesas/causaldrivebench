"""
Sanity tests for evaluation/common/llm_judge.py.

Unit tests (always run, no API key required):
  - _extract_json parsing variants
  - _clamp boundary behaviour
  - LLMJudge initialisation and provider validation
  - LLMJudge.aggregate_scores with valid/error/empty inputs

Integration tests (auto-skipped when provider env vars are absent):
  - test_score_response_valid_* — live scoring of a well-written VLA response
  - test_score_response_invalid_* — live scoring of a poor VLA response
  - test_score_batch_with_sample_data — batch scoring using STATIC_RESPONSES
    against QA pairs loaded from evaluation/sample_data/causal_nuscenes/
  - test_score_batch_save_path — verifies JSONL is written correctly
  - test_valid_scores_higher_than_invalid — aggregate check over batch

STATIC_RESPONSES contains pre-written model responses for three questions from
causal_nuscenes/nuscenes-scene-0072/SAMPLED_1/qa/active_qa.json:
  CaI1 (MCQ, correct=A), CR1 (binary, correct=No), SC1 (binary, correct=Yes).
Each question has a "valid" response (spatially grounded, mechanistically correct,
conditionality-aware, answer-consistent) and an "invalid" response (vague or wrong).

Run all tests:
    pytest evaluation/tests/test_llm_judge.py -v

Run only unit tests (no API key needed):
    pytest evaluation/tests/test_llm_judge.py -v -m "not integration"

Run integration tests only:
    pytest evaluation/tests/test_llm_judge.py -v -m integration
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — import without installing the package
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.common.llm_judge import (  # noqa: E402
    LLMJudge,
    _DIMENSIONS,
    _clamp,
    _extract_json,
)

# ---------------------------------------------------------------------------
# Sample data paths
# ---------------------------------------------------------------------------

_SAMPLE_DATA = _REPO_ROOT / "evaluation" / "sample_data"
_NUSCENES_QA = (
    _SAMPLE_DATA
    / "causal_nuscenes"
    / "nuscenes-scene-0072"
    / "SAMPLED_1"
    / "qa"
    / "active_qa.json"
)

# ---------------------------------------------------------------------------
# Static VLA model responses
#
# Each entry maps to a question in the nuscenes active_qa.json.
# "valid"   — spatially grounded, causally correct, condition-aware, answer-consistent
# "invalid" — vague, wrong answer, or missing causal explanation
# ---------------------------------------------------------------------------

STATIC_RESPONSES: dict[str, dict] = {
    # CaI1: MCQ, correct = A (crosswalk markings ahead are the primary reason for decelerating)
    "CaI1_valid": {
        "question_id": "CaI1",
        "quality": "valid",
        "model_response": (
            "The crosswalk markings spanning the forward path through the intersection are the "
            "primary cause of deceleration. The ego vehicle is approaching a marked pedestrian "
            "crossing that spans the lane directly ahead, requiring the vehicle to slow and check "
            "for pedestrians before proceeding. The pedestrian on the far sidewalk is walking "
            "parallel and poses no immediate crossing hazard. The warning sign is an informational "
            "cue but not the direct cause of braking — the markings themselves trigger the "
            "yielding obligation. The parked truck and car are behind and off the travel path. "
            "Answer: A"
        ),
    },
    "CaI1_invalid": {
        "question_id": "CaI1",
        "quality": "invalid",
        "model_response": "B) The pedestrian on the sidewalk looks dangerous so I am slowing down.",
    },

    # CR1: Binary, correct = No (crosswalk markings alone don't cause deceleration without context)
    "CR1_valid": {
        "question_id": "CR1",
        "quality": "valid",
        "model_response": (
            "No. The pedestrian is on the far sidewalk walking parallel to the road and not "
            "attempting to cross. The primary reason for decelerating is the crosswalk marking "
            "spanning the forward path, not the presence of the sidewalk pedestrian. If the "
            "crosswalk markings were not there, the pedestrian's position — safely on the "
            "sidewalk — would not independently compel the ego vehicle to brake so significantly "
            "at this point. The deceleration is triggered by the marked crossing obligation, not "
            "by a pedestrian who is not yet crossing. Answer: No"
        ),
    },
    "CR1_invalid": {
        "question_id": "CR1",
        "quality": "invalid",
        "model_response": "Yes, the pedestrian might cross at any moment so we must stop.",
    },

    # SC1: Binary, correct = Yes (crosswalk alone sufficient cause for slowing)
    "SC1_valid": {
        "question_id": "SC1",
        "quality": "valid",
        "model_response": (
            "Yes. The crosswalk marking ahead creates a legal and physical obligation to slow "
            "down and yield before the pedestrian crossing zone. Even with no pedestrians and no "
            "other visible hazards, a driver approaching a crosswalk must reduce speed in "
            "anticipation of potential pedestrian use and to comply with yield obligations. The "
            "marking alone — spanning the ego lane at the intersection — is sufficient to cause "
            "the observed deceleration. Answer: Yes"
        ),
    },
    "SC1_invalid": {
        "question_id": "SC1",
        "quality": "invalid",
        "model_response": "No, an empty crosswalk with nothing else visible means you can drive through normally.",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_nuscenes_questions() -> dict[str, dict]:
    """Load QA questions from the nuscenes sample data, keyed by question id."""
    if not _NUSCENES_QA.exists():
        pytest.skip(f"Sample data not found: {_NUSCENES_QA}")
    data = json.loads(_NUSCENES_QA.read_text())
    return {q["id"]: q for q in data["questions"]}


def _build_batch_records() -> list[dict]:
    """
    Build judge input records by pairing STATIC_RESPONSES with sample QA.
    Each record includes question text (with preamble), ground_truth, model_response,
    plus metadata fields question_id and quality for assertion purposes.
    """
    questions = _load_nuscenes_questions()
    records = []
    for key, resp in STATIC_RESPONSES.items():
        qid = resp["question_id"]
        if qid not in questions:
            continue
        q = questions[qid]
        preamble = q.get("preamble", "")
        full_question = f"{preamble}\n\n{q['question']}" if preamble else q["question"]
        records.append({
            "question": full_question,
            "ground_truth": q["correct_answer"],
            "model_response": resp["model_response"],
            "question_id": qid,
            "quality": resp["quality"],
        })
    return records


def _anthropic_judge() -> LLMJudge:
    return LLMJudge(provider="anthropic")


def _azure_foundry_judge() -> LLMJudge:
    return LLMJudge(provider="azure_foundry")


# ---------------------------------------------------------------------------
# pytest marks and skip conditions
# ---------------------------------------------------------------------------

integration = pytest.mark.integration

_has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
_has_foundry_keys = bool(
    os.environ.get("ANTHROPIC_FOUNDRY_API_KEY")
    and os.environ.get("FOUNDRY_BASE_URL")
)

skip_no_anthropic = pytest.mark.skipif(
    not _has_anthropic_key,
    reason="ANTHROPIC_API_KEY not set",
)
skip_no_foundry = pytest.mark.skipif(
    not _has_foundry_keys,
    reason="ANTHROPIC_FOUNDRY_API_KEY or FOUNDRY_BASE_URL not set",
)


# ===========================================================================
# Unit tests — no API key required
# ===========================================================================

class TestExtractJson:
    def test_plain_json(self):
        raw = '{"scores": {"spatial_grounding": 2}, "reasoning": "ok"}'
        assert _extract_json(raw) == raw

    def test_json_fence(self):
        raw = '```json\n{"scores": {"spatial_grounding": 1}}\n```'
        result = _extract_json(raw)
        assert json.loads(result) == {"scores": {"spatial_grounding": 1}}

    def test_bare_fence(self):
        raw = '```\n{"a": 1}\n```'
        result = _extract_json(raw)
        assert json.loads(result) == {"a": 1}

    def test_brace_extraction_from_text(self):
        raw = 'Here is the result: {"scores": {}} more text'
        result = _extract_json(raw)
        assert json.loads(result) == {"scores": {}}

    def test_invalid_returns_stripped(self):
        raw = "  not json at all  "
        result = _extract_json(raw)
        assert result == "not json at all"


class TestClamp:
    def test_within_range(self):
        assert _clamp(1) == 1

    def test_at_max(self):
        assert _clamp(2) == 2

    def test_above_max(self):
        assert _clamp(5) == 2

    def test_at_min(self):
        assert _clamp(0) == 0

    def test_below_min(self):
        assert _clamp(-3) == 0

    def test_custom_bounds(self):
        assert _clamp(10, lo=0, hi=5) == 5


class TestLLMJudgeInit:
    def test_default_provider_is_anthropic(self):
        judge = LLMJudge()
        assert judge.provider == "anthropic"
        assert judge.model == "claude-opus-4-6"
        assert judge.api_key_env == "ANTHROPIC_API_KEY"

    def test_openai_provider_defaults(self):
        judge = LLMJudge(provider="openai")
        assert judge.model == "gpt-4o"
        assert judge.api_key_env == "OPENAI_API_KEY"

    def test_azure_foundry_defaults(self):
        judge = LLMJudge(provider="azure_foundry")
        assert judge.api_key_env == "ANTHROPIC_FOUNDRY_API_KEY"
        assert judge.azure_endpoint_env == "FOUNDRY_BASE_URL"

    def test_azure_openai_defaults(self):
        judge = LLMJudge(provider="azure_openai")
        assert judge.api_key_env == "AZURE_OPENAI_API_KEY"
        assert judge.azure_endpoint_env == "AZURE_OPENAI_ENDPOINT"

    def test_custom_model_override(self):
        judge = LLMJudge(provider="anthropic", model="claude-haiku-4-5-20251001")
        assert judge.model == "claude-haiku-4-5-20251001"

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMJudge(provider="unknown_provider")

    def test_client_not_initialized_before_first_call(self):
        judge = LLMJudge(provider="anthropic")
        assert judge._client is None

    def test_missing_api_key_raises_environment_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        judge = LLMJudge(provider="anthropic")
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            judge._get_client()

    def test_missing_foundry_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_FOUNDRY_API_KEY", raising=False)
        judge = LLMJudge(provider="azure_foundry")
        with pytest.raises(EnvironmentError, match="ANTHROPIC_FOUNDRY_API_KEY"):
            judge._get_client()

    def test_missing_foundry_key_and_resource_raises(self, monkeypatch):
        """AnthropicFoundry raises when neither api_key nor resource is available."""
        monkeypatch.delenv("ANTHROPIC_FOUNDRY_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
        monkeypatch.delenv("FOUNDRY_BASE_URL", raising=False)
        judge = LLMJudge(provider="azure_foundry")
        # api_key env var is absent → LLMJudge raises EnvironmentError before AnthropicFoundry
        with pytest.raises(EnvironmentError, match="ANTHROPIC_FOUNDRY_API_KEY"):
            judge._get_client()


class TestAggregateScores:
    def _make_score(self, values: dict, error: str | None = None) -> dict:
        rec = {
            "scores": {dim: values.get(dim, 0) for dim in _DIMENSIONS},
            "composite_score": sum(values.get(d, 0) for d in _DIMENSIONS) / len(_DIMENSIONS),
            "reasoning": "test reasoning",
        }
        if error:
            rec["error"] = error
        return rec

    def test_empty_list_returns_zeros(self):
        result = LLMJudge.aggregate_scores([])
        assert result["composite_mean"] == 0.0
        assert result["n"] == 0
        assert result["n_errors"] == 0

    def test_all_errors_returns_zeros(self):
        records = [
            self._make_score({}, error="API timeout"),
            self._make_score({}, error="JSON decode error"),
        ]
        result = LLMJudge.aggregate_scores(records)
        assert result["n"] == 0
        assert result["n_errors"] == 2
        assert result["composite_mean"] == 0.0

    def test_single_perfect_score(self):
        records = [self._make_score({d: 2 for d in _DIMENSIONS})]
        result = LLMJudge.aggregate_scores(records)
        assert result["n"] == 1
        assert result["n_errors"] == 0
        assert result["composite_mean"] == 2.0
        for dim in _DIMENSIONS:
            assert result["per_dimension"][dim]["mean"] == 2.0

    def test_mixed_valid_and_error(self):
        records = [
            self._make_score({d: 2 for d in _DIMENSIONS}),
            self._make_score({}, error="timeout"),
        ]
        result = LLMJudge.aggregate_scores(records)
        assert result["n"] == 1
        assert result["n_errors"] == 1

    def test_distribution_counts_are_correct(self):
        records = [
            self._make_score({d: 0 for d in _DIMENSIONS}),
            self._make_score({d: 1 for d in _DIMENSIONS}),
            self._make_score({d: 2 for d in _DIMENSIONS}),
        ]
        result = LLMJudge.aggregate_scores(records)
        for dim in _DIMENSIONS:
            dist = result["per_dimension"][dim]["distribution"]
            assert dist[0] == 1
            assert dist[1] == 1
            assert dist[2] == 1

    def test_mean_is_rounded_to_4_decimals(self):
        records = [
            self._make_score({d: 1 for d in _DIMENSIONS}),
            self._make_score({d: 2 for d in _DIMENSIONS}),
        ]
        result = LLMJudge.aggregate_scores(records)
        assert result["composite_mean"] == round(result["composite_mean"], 4)


class TestStaticResponsesStructure:
    """Verify static responses are well-formed before using them in integration tests."""

    def test_all_responses_have_required_keys(self):
        for key, resp in STATIC_RESPONSES.items():
            assert "question_id" in resp, f"{key}: missing question_id"
            assert "quality" in resp, f"{key}: missing quality"
            assert "model_response" in resp, f"{key}: missing model_response"

    def test_quality_values_are_valid(self):
        valid_qualities = {"valid", "invalid"}
        for key, resp in STATIC_RESPONSES.items():
            assert resp["quality"] in valid_qualities, (
                f"{key}: unexpected quality {resp['quality']!r}"
            )

    def test_valid_responses_are_substantive(self):
        for key, resp in STATIC_RESPONSES.items():
            if resp["quality"] == "valid":
                assert len(resp["model_response"]) > 100, (
                    f"{key}: valid response is too short to be a meaningful VLA response"
                )

    def test_equal_count_valid_and_invalid(self):
        valid_count = sum(1 for r in STATIC_RESPONSES.values() if r["quality"] == "valid")
        invalid_count = sum(1 for r in STATIC_RESPONSES.values() if r["quality"] == "invalid")
        assert valid_count == invalid_count, (
            f"Expected equal valid/invalid pairs, got {valid_count} vs {invalid_count}"
        )

    def test_question_ids_reference_nuscenes_qa(self):
        expected_qids = {"CaI1", "CR1", "SC1"}
        actual_qids = {r["question_id"] for r in STATIC_RESPONSES.values()}
        assert actual_qids == expected_qids


# ===========================================================================
# Integration tests — require live API access
# ===========================================================================

class TestScoreResponseAnthropic:
    """Live Anthropic API tests — skipped when ANTHROPIC_API_KEY is not set."""

    @integration
    @skip_no_anthropic
    def test_score_response_returns_required_keys(self):
        judge = _anthropic_judge()
        result = judge.score_response(
            question="Is there a pedestrian in the crosswalk ahead?",
            ground_truth="Yes",
            model_response=(
                "Yes. A pedestrian is walking across the crosswalk directly ahead of the "
                "ego vehicle, positioned in the travel lane. The vehicle must stop to yield."
            ),
        )
        assert "scores" in result
        assert "composite_score" in result
        assert "reasoning" in result
        assert set(result["scores"].keys()) == set(_DIMENSIONS)
        assert all(isinstance(v, int) for v in result["scores"].values())
        assert 0.0 <= result["composite_score"] <= 2.0

    @integration
    @skip_no_anthropic
    def test_score_response_valid_answer_scores_nonzero(self):
        """A well-written, spatially-grounded, causally-correct response should score > 0."""
        judge = _anthropic_judge()
        questions = _load_nuscenes_questions()
        q = questions["CaI1"]
        result = judge.score_response(
            question=q["question"],
            ground_truth=q["correct_answer"],
            model_response=STATIC_RESPONSES["CaI1_valid"]["model_response"],
        )
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert result["composite_score"] > 0.0, (
            "Valid response should score above 0 on at least one dimension"
        )

    @integration
    @skip_no_anthropic
    def test_score_response_invalid_answer_scores_lower(self):
        """A vague/wrong response should score at or below a well-written one."""
        judge = _anthropic_judge()
        questions = _load_nuscenes_questions()
        q = questions["CaI1"]
        valid_result = judge.score_response(
            question=q["question"],
            ground_truth=q["correct_answer"],
            model_response=STATIC_RESPONSES["CaI1_valid"]["model_response"],
        )
        invalid_result = judge.score_response(
            question=q["question"],
            ground_truth=q["correct_answer"],
            model_response=STATIC_RESPONSES["CaI1_invalid"]["model_response"],
        )
        assert valid_result["composite_score"] >= invalid_result["composite_score"], (
            f"Valid ({valid_result['composite_score']}) should >= invalid ({invalid_result['composite_score']})"
        )

    @integration
    @skip_no_anthropic
    def test_score_response_with_criteria_extra(self):
        """Additional criteria parameter should not break scoring."""
        judge = _anthropic_judge()
        result = judge.score_response(
            question="Is the vehicle stopped?",
            ground_truth="Yes",
            model_response="Yes, the vehicle is stopped because pedestrians are in the crosswalk ahead.",
            criteria="Focus on whether the model identifies the causal agent.",
        )
        assert "scores" in result
        assert "error" not in result


class TestScoreBatchAnthropic:
    """Batch scoring integration tests using the nuscenes sample data."""

    @integration
    @skip_no_anthropic
    def test_score_batch_returns_correct_length(self):
        judge = _anthropic_judge()
        records = _build_batch_records()
        results = judge.score_batch(records, requests_per_minute=30)
        assert len(results) == len(records)

    @integration
    @skip_no_anthropic
    def test_score_batch_all_have_required_structure(self):
        judge = _anthropic_judge()
        records = _build_batch_records()
        results = judge.score_batch(records, requests_per_minute=30)
        for i, res in enumerate(results):
            if "error" in res:
                continue  # errors are acceptable; check structure of valid ones
            assert "scores" in res, f"Record {i}: missing 'scores'"
            assert "composite_score" in res, f"Record {i}: missing 'composite_score'"
            assert "reasoning" in res, f"Record {i}: missing 'reasoning'"
            assert set(res["scores"].keys()) == set(_DIMENSIONS), (
                f"Record {i}: unexpected dimensions {set(res['scores'].keys())}"
            )

    @integration
    @skip_no_anthropic
    def test_score_batch_save_path_writes_jsonl(self):
        judge = _anthropic_judge()
        records = _build_batch_records()[:2]  # use only 2 to keep test fast
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "judge_out" / "results.jsonl"
            judge.score_batch(records, requests_per_minute=30, save_path=save_path)
            assert save_path.exists(), "JSONL file was not created"
            lines = save_path.read_text().strip().splitlines()
            assert len(lines) == 2
            for line in lines:
                obj = json.loads(line)
                assert "question" in obj
                assert "scores" in obj or "error" in obj

    @integration
    @skip_no_anthropic
    def test_valid_responses_score_higher_on_average(self):
        """Aggregate composite score for valid responses should exceed invalid ones."""
        judge = _anthropic_judge()
        records = _build_batch_records()
        results = judge.score_batch(records, requests_per_minute=30)

        valid_scores = [
            res["composite_score"]
            for rec, res in zip(records, results)
            if rec["quality"] == "valid" and "error" not in res
        ]
        invalid_scores = [
            res["composite_score"]
            for rec, res in zip(records, results)
            if rec["quality"] == "invalid" and "error" not in res
        ]

        assert valid_scores, "No valid responses scored without error"
        assert invalid_scores, "No invalid responses scored without error"

        avg_valid = sum(valid_scores) / len(valid_scores)
        avg_invalid = sum(invalid_scores) / len(invalid_scores)

        assert avg_valid > avg_invalid, (
            f"Expected valid avg ({avg_valid:.3f}) > invalid avg ({avg_invalid:.3f})"
        )

    @integration
    @skip_no_anthropic
    def test_aggregate_scores_after_batch(self):
        judge = _anthropic_judge()
        records = _build_batch_records()
        results = judge.score_batch(records, requests_per_minute=30)
        agg = LLMJudge.aggregate_scores(results)

        assert "composite_mean" in agg
        assert "n" in agg
        assert "n_errors" in agg
        assert "per_dimension" in agg
        assert agg["n"] + agg["n_errors"] == len(results)
        assert set(agg["per_dimension"].keys()) == set(_DIMENSIONS)


class TestScoreResponseAzureFoundry:
    """Azure AI Foundry integration tests — skipped when foundry keys are not set."""

    @integration
    @skip_no_foundry
    def test_azure_foundry_score_response_returns_structure(self):
        judge = _azure_foundry_judge()
        questions = _load_nuscenes_questions()
        q = questions["CR1"]
        result = judge.score_response(
            question=q["question"],
            ground_truth=q["correct_answer"],
            model_response=STATIC_RESPONSES["CR1_valid"]["model_response"],
        )
        assert "scores" in result, f"Expected 'scores' in result, got: {result}"
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert result["composite_score"] >= 0.0

    @integration
    @skip_no_foundry
    def test_azure_foundry_client_is_anthropic_foundry(self):
        """Verify the azure_foundry client is an AnthropicFoundry instance."""
        from anthropic import AnthropicFoundry
        judge = LLMJudge(provider="azure_foundry")
        client = judge._get_client()
        assert isinstance(client, AnthropicFoundry)


# ===========================================================================
# End-to-end sample data smoke test
# ===========================================================================

class TestSampleDataIntegrity:
    """Verify sample QA data is readable and has the expected question IDs."""

    def test_nuscenes_qa_file_exists(self):
        assert _NUSCENES_QA.exists(), f"Sample QA not found at {_NUSCENES_QA}"

    def test_nuscenes_qa_loads_correctly(self):
        data = json.loads(_NUSCENES_QA.read_text())
        assert "questions" in data
        assert len(data["questions"]) >= 3

    def test_static_response_question_ids_in_qa(self):
        questions = _load_nuscenes_questions()
        for key, resp in STATIC_RESPONSES.items():
            qid = resp["question_id"]
            assert qid in questions, (
                f"STATIC_RESPONSES[{key!r}] references {qid!r} "
                f"which is not in the sample QA file"
            )

    def test_qa_questions_have_correct_answer(self):
        questions = _load_nuscenes_questions()
        for qid, q in questions.items():
            assert "correct_answer" in q, f"Question {qid} missing 'correct_answer'"
            assert q["correct_answer"] in ("A", "B", "C", "D", "Yes", "No"), (
                f"Question {qid} has unexpected answer {q['correct_answer']!r}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
