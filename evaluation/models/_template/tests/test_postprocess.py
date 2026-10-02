"""
Template unit tests for the Postprocessor.

Covers:
  - parse_answer() with dict raw_output (new format) and legacy plain string
  - _load_qa_index() using correct QA file naming (active_qa.json, dormant_qa.json,
    distractor_qa.json) and nested bench_dir structure
  - _extract_text() utility for both dict and string raw_output
  - compute_metrics() end-to-end
  - save_report() file creation

Uses the factory pattern — ``make_postprocessor()`` is the single customisation
point when adapting this file for a new model.

Run from repo root:
    pytest evaluation/models/_template/tests/test_postprocess.py -v

Or inside the container:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postprocess import TemplatePostprocessor  # noqa: E402
from evaluation.common.base_postprocess import ModelPostprocessor

_SAMPLE_DATA_ROOT = Path(__file__).parents[4] / "evaluation" / "sample_data"
NUSCENES_SAMPLE_DATA = _SAMPLE_DATA_ROOT / "causal_nuscenes"
NUSCENES_SCENE = "nuscenes-scene-0159"


# ---------------------------------------------------------------------------
# Factory function — update ONLY this when customising for a new model
# ---------------------------------------------------------------------------


def make_postprocessor() -> TemplatePostprocessor:
    """Factory: returns a TemplatePostprocessor ready for testing."""
    return TemplatePostprocessor(model_cfg={})


# ---------------------------------------------------------------------------
# Binary answer parsing — dict raw_output (new format)
# ---------------------------------------------------------------------------

BINARY_Q = {"answer_format": "binary", "id": "test_q"}


@pytest.mark.parametrize("raw_output, expected", [
    ({"text": "Answer: Yes"},                           "Yes"),
    ({"text": "Answer: No"},                            "No"),
    ({"text": "answer: yes"},                           "Yes"),
    ({"text": "The answer is No."},                     "No"),
    ({"text": "<think>Some reasoning</think>\nAnswer: Yes"}, "Yes"),
    ({"text": "Yes"},                                   "Yes"),
    ({"text": "No"},                                    "No"),
])
def test_parse_binary_dict_format(raw_output, expected):
    """parse_answer with dict raw_output must extract binary answer correctly."""
    pp = make_postprocessor()
    assert pp.parse_answer(raw_output, BINARY_Q) == expected


# ---------------------------------------------------------------------------
# Binary answer parsing — legacy plain string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw_str, expected", [
    ("Answer: Yes",        "Yes"),
    ("Answer: No",         "No"),
    ("answer: yes",        "Yes"),
    ("Yes",                "Yes"),
    ("No",                 "No"),
])
def test_parse_binary_legacy_string(raw_str, expected):
    """parse_answer with legacy plain string must still work."""
    pp = make_postprocessor()
    assert pp.parse_answer(raw_str, BINARY_Q) == expected


def test_parse_binary_empty_dict_returns_empty():
    pp = make_postprocessor()
    assert pp.parse_answer({"text": ""}, BINARY_Q) == ""


def test_parse_binary_empty_string_returns_empty():
    pp = make_postprocessor()
    assert pp.parse_answer("", BINARY_Q) == ""


def test_parse_binary_whitespace_returns_empty():
    pp = make_postprocessor()
    assert pp.parse_answer({"text": "   \n\t  "}, BINARY_Q) == ""


# ---------------------------------------------------------------------------
# MCQ answer parsing — dict raw_output
# ---------------------------------------------------------------------------

MCQ_Q = {"answer_format": "mcq", "id": "test_q"}


@pytest.mark.parametrize("raw_output, expected", [
    ({"text": "Answer: A"},            "A"),
    ({"text": "Answer: B"},            "B"),
    ({"text": "answer: c"},            "C"),
    ({"text": "The answer is D."},     "D"),
    ({"text": "A. The truck is …"},    "A"),
    ({"text": "B) Because …"},         "B"),
    ({"text": "<think>...</think>\nAnswer: C"}, "C"),
    ({"text": "A"},                    "A"),
])
def test_parse_mcq_dict_format(raw_output, expected):
    """parse_answer with dict raw_output must extract MCQ letter correctly."""
    pp = make_postprocessor()
    assert pp.parse_answer(raw_output, MCQ_Q) == expected


def test_parse_mcq_empty_returns_empty():
    pp = make_postprocessor()
    assert pp.parse_answer({"text": ""}, MCQ_Q) == ""


# ---------------------------------------------------------------------------
# _extract_text utility
# ---------------------------------------------------------------------------


def test_extract_text_from_dict():
    """_extract_text must return the 'text' value from a dict."""
    result = ModelPostprocessor._extract_text({"text": "Answer: A\nReasoning: …"})
    assert result == "Answer: A\nReasoning: …"


def test_extract_text_from_dict_missing_key():
    """_extract_text must return '' when 'text' key is absent."""
    result = ModelPostprocessor._extract_text({"score": 0})
    assert result == ""


def test_extract_text_from_string():
    """_extract_text must return the string as-is for legacy format."""
    result = ModelPostprocessor._extract_text("Answer: B")
    assert result == "Answer: B"


def test_extract_text_from_none():
    """_extract_text must return '' for None."""
    result = ModelPostprocessor._extract_text(None)
    assert result == ""


def test_extract_text_empty_dict():
    result = ModelPostprocessor._extract_text({})
    assert result == ""


# ---------------------------------------------------------------------------
# _load_qa_index — correct file naming and nested structure
# ---------------------------------------------------------------------------


def _make_bench_dir(tmpdir: Path) -> Path:
    """Create a minimal bench_dir with correct QA file naming conventions."""
    scene_id = "nuscenes-scene-test"
    sample_id = "SAMPLED_0"
    qa_dir = tmpdir / scene_id / sample_id / "qa"
    qa_dir.mkdir(parents=True)

    # active_qa.json → "ladder"
    ladder_qa = {"questions": [
        {"id": "L1", "_qa_type": "ladder", "answer_format": "mcq",
         "question": "Which element?", "correct_answer": "A",
         "options": ["A) truck", "B) car", "C) ped", "D) sign"],
         "reasoning": "The truck is blocking."}
    ]}
    # dormant_qa.json → "dormant"
    dormant_qa = {"questions": [
        {"id": "D1", "_qa_type": "dormant", "answer_format": "binary",
         "question": "Would a pedestrian change driving?", "correct_answer": "No",
         "options": None, "reasoning": "Pedestrian is not on path."}
    ]}
    # distractor_qa.json → "distractor"
    distractor_qa = {"questions": [
        {"id": "X1", "_qa_type": "distractor", "answer_format": "binary",
         "question": "Is the parked car relevant?", "correct_answer": "No",
         "options": None, "reasoning": "Parked car is not in path."}
    ]}

    (qa_dir / "active_qa.json").write_text(json.dumps(ladder_qa))
    (qa_dir / "dormant_qa.json").write_text(json.dumps(dormant_qa))
    (qa_dir / "distractor_qa.json").write_text(json.dumps(distractor_qa))

    return tmpdir


def test_load_qa_index_finds_all_types():
    """_load_qa_index must find ladder/dormant/distractor questions."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bench_dir = _make_bench_dir(tmpdir)
        qa_index = ModelPostprocessor._load_qa_index(bench_dir)

        assert len(qa_index) == 3, f"Expected 3 questions, got {len(qa_index)}"
        assert "L1" in qa_index, "ladder question L1 not indexed"
        assert "D1" in qa_index, "dormant question D1 not indexed"
        assert "X1" in qa_index, "distractor question X1 not indexed"


def test_load_qa_index_correct_qa_types():
    """_load_qa_index must assign correct _qa_type to each question."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bench_dir = _make_bench_dir(tmpdir)
        qa_index = ModelPostprocessor._load_qa_index(bench_dir)

        assert qa_index["L1"]["_qa_type"] == "ladder"
        assert qa_index["D1"]["_qa_type"] == "dormant"
        assert qa_index["X1"]["_qa_type"] == "distractor"


def test_load_qa_index_ignores_wrong_filenames():
    """_load_qa_index must NOT index questions from wrongly named files (e.g. qa.json)."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        qa_dir = tmpdir / "scene" / "SAMPLED_0" / "qa"
        qa_dir.mkdir(parents=True)

        # Wrong filename: qa.json (old format)
        wrong_qa = {"questions": [
            {"id": "OLD1", "answer_format": "binary", "correct_answer": "Yes",
             "question": "test", "reasoning": "test"}
        ]}
        (qa_dir / "qa.json").write_text(json.dumps(wrong_qa))

        qa_index = ModelPostprocessor._load_qa_index(tmpdir)
        assert "OLD1" not in qa_index, "Old qa.json format must not be indexed"


def test_load_qa_index_with_sample_data():
    """_load_qa_index must index questions from the bundled sample data."""
    if not NUSCENES_SAMPLE_DATA.exists():
        pytest.skip("nuscenes sample data not found")

    qa_index = ModelPostprocessor._load_qa_index(NUSCENES_SAMPLE_DATA)
    assert len(qa_index) > 0, "No questions indexed from sample data"

    # All questions should have _qa_type set
    for qid, q in qa_index.items():
        assert "_qa_type" in q, f"Question {qid} missing _qa_type"
        assert q["_qa_type"] in ("ladder", "dormant", "distractor"), (
            f"Unexpected _qa_type: {q['_qa_type']!r}"
        )


# ---------------------------------------------------------------------------
# compute_metrics end-to-end
# ---------------------------------------------------------------------------


def _write_outputs_jsonl(path: Path, records: list) -> None:
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_compute_metrics_correct_prediction():
    """compute_metrics must score correct prediction as accuracy=1.0."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bench_dir = _make_bench_dir(tmpdir)

        outputs_path = tmpdir / "outputs.jsonl"
        _write_outputs_jsonl(outputs_path, [
            {
                "scene_id": "nuscenes-scene-test",
                "sample_id": "SAMPLED_0",
                "question_id": "D1",
                "prompt_id": "000001",
                "raw_output": {"text": "Answer: No\nReasoning: Pedestrian off path."},
                "inference_time_s": 1.0,
            }
        ])

        pp = make_postprocessor()
        metrics = pp.compute_metrics(
            outputs_path=outputs_path,
            bench_dir=bench_dir,
        )
        assert metrics["overall"]["n"] == 1
        assert metrics["overall"]["accuracy"] == 1.0


def test_compute_metrics_wrong_prediction():
    """compute_metrics must score wrong prediction as accuracy=0.0."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bench_dir = _make_bench_dir(tmpdir)

        outputs_path = tmpdir / "outputs.jsonl"
        _write_outputs_jsonl(outputs_path, [
            {
                "scene_id": "nuscenes-scene-test",
                "sample_id": "SAMPLED_0",
                "question_id": "L1",
                "prompt_id": "000001",
                "raw_output": {"text": "Answer: B"},  # wrong, correct is A
                "inference_time_s": 1.0,
            }
        ])

        pp = make_postprocessor()
        metrics = pp.compute_metrics(
            outputs_path=outputs_path,
            bench_dir=bench_dir,
        )
        assert metrics["overall"]["accuracy"] == 0.0


def test_compute_metrics_accepts_legacy_string_raw_output():
    """compute_metrics must handle plain string raw_output (backward compat)."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bench_dir = _make_bench_dir(tmpdir)

        outputs_path = tmpdir / "outputs.jsonl"
        _write_outputs_jsonl(outputs_path, [
            {
                "question_id": "D1",
                "raw_output": "Answer: No",  # legacy plain string
                "inference_time_s": 1.0,
            }
        ])

        pp = make_postprocessor()
        metrics = pp.compute_metrics(
            outputs_path=outputs_path,
            bench_dir=bench_dir,
        )
        assert metrics["overall"]["accuracy"] == 1.0


def test_compute_metrics_per_qa_type_breakdown():
    """compute_metrics must report accuracy broken down by qa_type."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bench_dir = _make_bench_dir(tmpdir)

        outputs_path = tmpdir / "outputs.jsonl"
        _write_outputs_jsonl(outputs_path, [
            {"question_id": "L1", "raw_output": {"text": "Answer: A"}},  # correct
            {"question_id": "D1", "raw_output": {"text": "Answer: Yes"}},  # wrong (expected No)
            {"question_id": "X1", "raw_output": {"text": "Answer: No"}},  # correct
        ])

        pp = make_postprocessor()
        metrics = pp.compute_metrics(
            outputs_path=outputs_path,
            bench_dir=bench_dir,
        )
        assert "per_qa_type" in metrics
        per_type = metrics["per_qa_type"]
        assert per_type.get("ladder", {}).get("accuracy") == 1.0
        assert per_type.get("dormant", {}).get("accuracy") == 0.0
        assert per_type.get("distractor", {}).get("accuracy") == 1.0


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------


def test_save_report_creates_file():
    """save_report must write valid JSON at the specified path."""
    pp = make_postprocessor()
    metrics = {
        "overall": {"accuracy": 0.75, "n": 4},
        "per_qa_type": {"dormant": {"accuracy": 0.75, "n": 4, "correct": 3}},
        "confusion": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "report.json"
        pp.save_report(metrics, report_path)
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["overall"]["accuracy"] == 0.75


def test_save_report_creates_parent_dirs():
    """save_report must create parent directories if they don't exist."""
    pp = make_postprocessor()
    metrics = {"overall": {"accuracy": 1.0, "n": 1}, "per_qa_type": {}, "confusion": {}}
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "nested" / "deep" / "report.json"
        pp.save_report(metrics, report_path)
        assert report_path.exists()


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
