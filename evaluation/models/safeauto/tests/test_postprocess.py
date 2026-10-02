"""
Unit tests for SafeAutoPostprocessor.

SafeAuto outputs free-form text like:
  "Yes, because the light is red."
  "Answer: A. The vehicle is braking."
  "B) The pedestrian is crossing."

Run from repo root:
    pytest evaluation/models/safeauto/tests/test_postprocess.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postprocess import SafeAutoPostprocessor  # noqa: E402
from evaluation.common.base_postprocess import ModelPostprocessor

_SAMPLE_DATA_ROOT = Path(__file__).parents[4] / "evaluation" / "sample_data"
NUSCENES_SAMPLE_DATA = _SAMPLE_DATA_ROOT / "causal_nuscenes"
NUSCENES_SCENE = "nuscenes-scene-0159"


def make_postprocessor() -> SafeAutoPostprocessor:
    return SafeAutoPostprocessor(model_cfg={})


BINARY_Q = {"answer_format": "binary", "id": "test_q"}
MCQ_Q = {"answer_format": "mcq", "id": "test_q"}


# ---------------------------------------------------------------------------
# Binary parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw_output, expected", [
    ({"text": "Answer: Yes"},                                    "Yes"),
    ({"text": "Answer: No"},                                     "No"),
    ({"text": "answer: yes"},                                    "Yes"),
    ({"text": "Yes, because the traffic light is red."},         "Yes"),
    ({"text": "No, the vehicle is not relevant."},               "No"),
    ({"text": "<think>Let me think...</think>\nAnswer: Yes"},    "Yes"),
    ({"text": "Yes"},                                            "Yes"),
    ({"text": "No"},                                             "No"),
])
def test_parse_binary_dict(raw_output, expected):
    pp = make_postprocessor()
    assert pp.parse_answer(raw_output, BINARY_Q) == expected


@pytest.mark.parametrize("raw_str, expected", [
    ("Answer: Yes",        "Yes"),
    ("Answer: No",         "No"),
    ("Yes",                "Yes"),
    ("No",                 "No"),
])
def test_parse_binary_string(raw_str, expected):
    pp = make_postprocessor()
    assert pp.parse_answer(raw_str, BINARY_Q) == expected


def test_parse_binary_empty_returns_empty():
    pp = make_postprocessor()
    assert pp.parse_answer({"text": ""}, BINARY_Q) == ""


def test_parse_binary_none_text_returns_empty():
    pp = make_postprocessor()
    assert pp.parse_answer({"text": None}, BINARY_Q) == ""


# ---------------------------------------------------------------------------
# MCQ parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw_output, expected", [
    ({"text": "Answer: A"},                          "A"),
    ({"text": "Answer: B"},                          "B"),
    ({"text": "answer: c"},                          "C"),
    ({"text": "The answer is D."},                   "D"),
    ({"text": "A. The truck is blocking the road."}, "A"),
    ({"text": "B) The vehicle is decelerating."},    "B"),
    ({"text": "A"},                                  "A"),
    ({"text": "<think>...</think>\nAnswer: C"},      "C"),
])
def test_parse_mcq_dict(raw_output, expected):
    pp = make_postprocessor()
    assert pp.parse_answer(raw_output, MCQ_Q) == expected


def test_parse_mcq_empty_returns_empty():
    pp = make_postprocessor()
    assert pp.parse_answer({"text": ""}, MCQ_Q) == ""


# ---------------------------------------------------------------------------
# _extract_text utility (from base class)
# ---------------------------------------------------------------------------

def test_extract_text_from_dict():
    assert ModelPostprocessor._extract_text({"text": "Answer: A"}) == "Answer: A"


def test_extract_text_from_string():
    assert ModelPostprocessor._extract_text("Answer: B") == "Answer: B"


def test_extract_text_none():
    assert ModelPostprocessor._extract_text(None) == ""


def test_extract_text_empty_dict():
    assert ModelPostprocessor._extract_text({}) == ""


# ---------------------------------------------------------------------------
# _load_qa_index
# ---------------------------------------------------------------------------

def _make_bench_dir(tmpdir: Path) -> Path:
    """Minimal bench_dir with correct QA naming."""
    qa_dir = tmpdir / "nuscenes-scene-test" / "SAMPLED_0" / "qa"
    qa_dir.mkdir(parents=True)

    (qa_dir / "active_qa.json").write_text(json.dumps({"questions": [
        {"id": "L1", "_qa_type": "ladder", "answer_format": "mcq",
         "question": "Which?", "correct_answer": "A",
         "options": ["A) truck", "B) car", "C) ped", "D) sign"],
         "reasoning": "Truck is blocking."}
    ]}))
    (qa_dir / "dormant_qa.json").write_text(json.dumps({"questions": [
        {"id": "D1", "_qa_type": "dormant", "answer_format": "binary",
         "question": "Would pedestrian affect?", "correct_answer": "No",
         "options": None, "reasoning": "Not on path."}
    ]}))
    (qa_dir / "distractor_qa.json").write_text(json.dumps({"questions": [
        {"id": "X1", "_qa_type": "distractor", "answer_format": "binary",
         "question": "Is parked car relevant?", "correct_answer": "No",
         "options": None, "reasoning": "Not in path."}
    ]}))
    return tmpdir


def test_load_qa_index_all_types():
    with tempfile.TemporaryDirectory() as tmp:
        bench_dir = _make_bench_dir(Path(tmp))
        qa_index = ModelPostprocessor._load_qa_index(bench_dir)
        assert "L1" in qa_index
        assert "D1" in qa_index
        assert "X1" in qa_index


def test_load_qa_index_qa_types_correct():
    with tempfile.TemporaryDirectory() as tmp:
        bench_dir = _make_bench_dir(Path(tmp))
        qa_index = ModelPostprocessor._load_qa_index(bench_dir)
        assert qa_index["L1"]["_qa_type"] == "ladder"
        assert qa_index["D1"]["_qa_type"] == "dormant"
        assert qa_index["X1"]["_qa_type"] == "distractor"


# ---------------------------------------------------------------------------
# compute_metrics end-to-end
# ---------------------------------------------------------------------------

def _write_outputs(path: Path, records: list) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_compute_metrics_correct():
    with tempfile.TemporaryDirectory() as tmp:
        bench_dir = _make_bench_dir(Path(tmp))
        out = Path(tmp) / "outputs.jsonl"
        _write_outputs(out, [
            {"question_id": "D1", "raw_output": {"text": "Answer: No"}},
        ])
        pp = make_postprocessor()
        metrics = pp.compute_metrics(out, bench_dir)
        assert metrics["overall"]["accuracy"] == 1.0


def test_compute_metrics_wrong():
    with tempfile.TemporaryDirectory() as tmp:
        bench_dir = _make_bench_dir(Path(tmp))
        out = Path(tmp) / "outputs.jsonl"
        _write_outputs(out, [
            {"question_id": "L1", "raw_output": {"text": "Answer: B"}},  # correct is A
        ])
        pp = make_postprocessor()
        metrics = pp.compute_metrics(out, bench_dir)
        assert metrics["overall"]["accuracy"] == 0.0


def test_compute_metrics_per_qa_type():
    with tempfile.TemporaryDirectory() as tmp:
        bench_dir = _make_bench_dir(Path(tmp))
        out = Path(tmp) / "outputs.jsonl"
        _write_outputs(out, [
            {"question_id": "L1", "raw_output": {"text": "A"}},   # correct
            {"question_id": "D1", "raw_output": {"text": "Yes"}},  # wrong
            {"question_id": "X1", "raw_output": {"text": "No"}},   # correct
        ])
        pp = make_postprocessor()
        metrics = pp.compute_metrics(out, bench_dir)
        assert metrics["per_qa_type"]["ladder"]["accuracy"] == 1.0
        assert metrics["per_qa_type"]["dormant"]["accuracy"] == 0.0
        assert metrics["per_qa_type"]["distractor"]["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

def test_save_report_creates_valid_json():
    pp = make_postprocessor()
    metrics = {"overall": {"accuracy": 0.5, "n": 2}, "per_qa_type": {}, "confusion": {}}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.json"
        pp.save_report(metrics, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["overall"]["accuracy"] == 0.5


def test_save_report_creates_parent_dirs():
    pp = make_postprocessor()
    metrics = {"overall": {"accuracy": 1.0, "n": 1}, "per_qa_type": {}, "confusion": {}}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "deep" / "nested" / "report.json"
        pp.save_report(metrics, path)
        assert path.exists()


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
