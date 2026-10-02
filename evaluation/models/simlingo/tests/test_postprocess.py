"""
Unit tests for SimLingoPostprocessor.

Covers:
  - parse_answer() with new dict raw_output format ({"text": "..."})
  - parse_answer() with legacy plain string raw_output (backward compat)
  - binary and MCQ extraction patterns
  - _collect_outputs() with nested scene_id/sample_id directory structure
  - compute_metrics() end-to-end with correct QA file naming convention
  - save_report() file creation

Uses the factory pattern — ``make_postprocessor()`` is the single customisation
point when adapting this file for a new model.

Run from the repo root:
    pytest evaluation/models/simlingo/tests/test_postprocess.py -v

Run all simlingo tests:
    pytest evaluation/models/simlingo/tests/ -v

Inside the container:
    uv run python -m pytest /workspace/repo/evaluation/models/simlingo/tests/ -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Repo root (4 levels up from tests/ folder)
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
# Model dir (for local imports: dataloader, postprocess, inference)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common.base_postprocess import ModelPostprocessor  # noqa: E402
from postprocess import SimLingoPostprocessor, _collect_outputs  # noqa: E402

_SAMPLE_DATA_ROOT = Path(__file__).parents[4] / "evaluation" / "sample_data"
NUSCENES_SAMPLE_DATA = _SAMPLE_DATA_ROOT / "causal_nuscenes"
NUSCENES_SCENE = "nuscenes-scene-0159"


# ---------------------------------------------------------------------------
# Factory function — update ONLY this when customising for a new model
# ---------------------------------------------------------------------------


def make_postprocessor() -> SimLingoPostprocessor:
    """Factory: returns a SimLingoPostprocessor ready for testing."""
    return SimLingoPostprocessor(model_cfg={})


# ---------------------------------------------------------------------------
# Binary answer parsing — dict raw_output (new format)
# ---------------------------------------------------------------------------

BINARY_Q = {"answer_format": "binary", "id": "test_q"}


@pytest.mark.parametrize("raw_output, expected", [
    ({"text": "Answer: Yes\nReasoning: The crosswalk is empty."}, "Yes"),
    ({"text": "Answer: No\nReasoning: The truck is parked far away."}, "No"),
    ({"text": "answer: yes"}, "Yes"),
    ({"text": "ANSWER: NO"}, "No"),
    ({"text": "<think>\nAnswer: No\nReasoning: Truck is stationary.\n</think>"}, "No"),
    ({"text": "<think>The pedestrian is not on the path.\nAnswer: Yes</think>"}, "Yes"),
    ({"text": "The answer is Yes because the crosswalk is clear."}, "Yes"),
    ({"text": "Yes, because the pedestrian stepped off the curb."}, "Yes"),
    ({"text": "No"}, "No"),
])
def test_parse_binary_dict_format(raw_output, expected):
    """parse_answer with dict raw_output must extract correct binary answer."""
    pp = make_postprocessor()
    assert pp.parse_answer(raw_output, BINARY_Q) == expected


# ---------------------------------------------------------------------------
# Binary answer parsing — legacy plain string raw_output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw_str, expected", [
    ("Answer: Yes\nReasoning: The crosswalk is empty.", "Yes"),
    ("Answer: No\nReasoning: The truck is parked far away.", "No"),
    ("answer: yes", "Yes"),
    ("<think>\nAnswer: No\n</think>", "No"),
    ("Yes", "Yes"),
    ("No", "No"),
])
def test_parse_binary_legacy_string(raw_str, expected):
    """parse_answer with legacy plain string must still work (backward compat)."""
    pp = make_postprocessor()
    assert pp.parse_answer(raw_str, BINARY_Q) == expected


# ---------------------------------------------------------------------------
# MCQ answer parsing — dict raw_output
# ---------------------------------------------------------------------------

MCQ_Q = {"answer_format": "mcq", "id": "test_q"}


@pytest.mark.parametrize("raw_output, expected", [
    ({"text": "Answer: A\nReasoning: The truck limits the lane."}, "A"),
    ({"text": "Answer: B"}, "B"),
    ({"text": "Answer: C\nReasoning: ..."}, "C"),
    ({"text": "Answer: D"}, "D"),
    ({"text": "answer: b"}, "B"),
    ({"text": "The answer is C because the sedan is irrelevant."}, "C"),
    ({"text": "Option B is the correct choice here."}, "B"),
    ({"text": "<think>\nAfter analysis, Answer: A\n</think>"}, "A"),
    ({"text": "B. The parked truck is visible but not in the path."}, "B"),
])
def test_parse_mcq_dict_format(raw_output, expected):
    """parse_answer with dict raw_output must extract correct MCQ letter."""
    pp = make_postprocessor()
    assert pp.parse_answer(raw_output, MCQ_Q) == expected


# ---------------------------------------------------------------------------
# Empty response handling
# ---------------------------------------------------------------------------


def test_parse_empty_dict_raises():
    """Empty dict raw_output must raise EmptyResponseException."""
    from evaluation.common.utils.exceptions import EmptyResponseException
    pp = make_postprocessor()
    with pytest.raises(EmptyResponseException):
        pp.parse_answer({"text": ""}, BINARY_Q)


def test_parse_whitespace_dict_raises():
    """Whitespace-only text in dict must raise EmptyResponseException."""
    from evaluation.common.utils.exceptions import EmptyResponseException
    pp = make_postprocessor()
    with pytest.raises(EmptyResponseException):
        pp.parse_answer({"text": "   "}, BINARY_Q)


def test_parse_empty_string_raises():
    """Empty string legacy raw_output must raise EmptyResponseException."""
    from evaluation.common.utils.exceptions import EmptyResponseException
    pp = make_postprocessor()
    with pytest.raises(EmptyResponseException):
        pp.parse_answer("", BINARY_Q)


# ---------------------------------------------------------------------------
# _extract_text utility (base class)
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
# _collect_outputs — nested scene_id/sample_id structure
# ---------------------------------------------------------------------------


def _write_outputs_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_collect_outputs_finds_nested_outputs():
    """_collect_outputs must find outputs in <output_base>/<scene_id>/<sample_id>/ dirs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_base = Path(tmpdir)
        scene_id = "nuscenes-scene-0001"
        sample_id = "SAMPLED_0"

        out_file = output_base / scene_id / sample_id / "outputs.jsonl"
        _write_outputs_jsonl(out_file, [
            {"question_id": "Q1", "raw_output": {"text": "Answer: Yes"}, "scene_id": scene_id},
            {"question_id": "Q2", "raw_output": {"text": "Answer: No"}, "scene_id": scene_id},
        ])

        sample_ids = [{"scene_id": scene_id, "sample_id": sample_id}]
        all_outputs, missing = _collect_outputs(sample_ids, output_base)

        assert len(all_outputs) == 2, f"Expected 2 outputs, got {len(all_outputs)}"
        assert len(missing) == 0, f"Unexpected missing samples: {missing}"


def test_collect_outputs_missing_sample():
    """_collect_outputs must report missing samples when outputs.jsonl absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_base = Path(tmpdir)
        sample_ids = [{"scene_id": "nuscenes-scene-0001", "sample_id": "SAMPLED_0"}]
        all_outputs, missing = _collect_outputs(sample_ids, output_base)

        assert len(all_outputs) == 0
        assert len(missing) == 1
        assert "nuscenes-scene-0001/SAMPLED_0" in missing[0]


def test_collect_outputs_multiple_samples():
    """_collect_outputs must collect from multiple nested directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_base = Path(tmpdir)
        samples = [
            ("nuscenes-scene-0001", "SAMPLED_0"),
            ("nuscenes-scene-0001", "SAMPLED_1"),
            ("nuscenes-scene-0002", "SAMPLED_0"),
        ]
        for scene_id, sample_id in samples:
            out_file = output_base / scene_id / sample_id / "outputs.jsonl"
            _write_outputs_jsonl(out_file, [
                {"question_id": f"{scene_id}_{sample_id}_Q1", "raw_output": {"text": "Answer: A"}}
            ])

        sample_ids = [{"scene_id": s, "sample_id": p} for s, p in samples]
        all_outputs, missing = _collect_outputs(sample_ids, output_base)

        assert len(all_outputs) == 3, f"Expected 3 outputs, got {len(all_outputs)}"
        assert len(missing) == 0


def test_collect_outputs_rejects_flat_structure():
    """_collect_outputs must NOT find outputs in old flat <scene>_<sample>/ dirs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_base = Path(tmpdir)
        # Old flat format: scene_sample/outputs.jsonl
        old_flat_dir = output_base / "nuscenes-scene-0001_SAMPLED_0"
        _write_outputs_jsonl(old_flat_dir / "outputs.jsonl", [
            {"question_id": "Q1", "raw_output": {"text": "Answer: Yes"}}
        ])

        # New nested format expected: scene_id/sample_id/outputs.jsonl
        sample_ids = [{"scene_id": "nuscenes-scene-0001", "sample_id": "SAMPLED_0"}]
        all_outputs, missing = _collect_outputs(sample_ids, output_base)

        # Should NOT find the flat file; should report missing
        assert len(all_outputs) == 0
        assert len(missing) == 1


# ---------------------------------------------------------------------------
# _load_qa_index — correct file naming and nested structure
# ---------------------------------------------------------------------------


def _make_bench_dir(tmpdir: Path) -> tuple[Path, list]:
    """Create a minimal bench_dir with correct nested QA file structure."""
    scene_id = "nuscenes-scene-test"
    sample_id = "SAMPLED_0"
    qa_dir = tmpdir / "bench" / scene_id / sample_id / "qa"
    qa_dir.mkdir(parents=True)

    ladder_qa = {"questions": [
        {"id": "L1", "_qa_type": "ladder", "answer_format": "mcq",
         "question": "Which element?", "correct_answer": "A",
         "options": ["A) truck", "B) car", "C) ped", "D) sign"],
         "reasoning": "The truck is blocking."}
    ]}
    dormant_qa = {"questions": [
        {"id": "D1", "_qa_type": "dormant", "answer_format": "binary",
         "question": "Would a pedestrian change driving?", "correct_answer": "No",
         "options": None, "reasoning": "Pedestrian is not on path."}
    ]}
    distractor_qa = {"questions": [
        {"id": "X1", "_qa_type": "distractor", "answer_format": "binary",
         "question": "Is the parked car relevant?", "correct_answer": "No",
         "options": None, "reasoning": "Parked car is not in path."}
    ]}

    bench_dir = tmpdir / "bench"
    (qa_dir / "active_qa.json").write_text(json.dumps(ladder_qa))
    (qa_dir / "dormant_qa.json").write_text(json.dumps(dormant_qa))
    (qa_dir / "distractor_qa.json").write_text(json.dumps(distractor_qa))

    sample_ids = [{"scene_id": scene_id, "sample_id": sample_id}]
    return bench_dir, sample_ids


def test_load_qa_index_finds_all_types():
    """_load_qa_index must find ladder/dormant/distractor questions."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bench_dir, _ = _make_bench_dir(tmpdir)
        qa_index = ModelPostprocessor._load_qa_index(bench_dir)

        assert len(qa_index) == 3, f"Expected 3 questions, got {len(qa_index)}"
        assert "L1" in qa_index, "ladder question L1 not indexed"
        assert "D1" in qa_index, "dormant question D1 not indexed"
        assert "X1" in qa_index, "distractor question X1 not indexed"


def test_load_qa_index_correct_qa_types():
    """_load_qa_index must assign correct _qa_type to each question."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bench_dir, _ = _make_bench_dir(tmpdir)
        qa_index = ModelPostprocessor._load_qa_index(bench_dir)

        assert qa_index["L1"]["_qa_type"] == "ladder"
        assert qa_index["D1"]["_qa_type"] == "dormant"
        assert qa_index["X1"]["_qa_type"] == "distractor"


def test_load_qa_index_with_sample_data():
    """_load_qa_index must index questions from the bundled sample data."""
    if not NUSCENES_SAMPLE_DATA.exists():
        pytest.skip("nuscenes sample data not found")

    qa_index = ModelPostprocessor._load_qa_index(NUSCENES_SAMPLE_DATA)
    assert len(qa_index) > 0, "No questions indexed from sample data"

    for qid, q in qa_index.items():
        assert "_qa_type" in q, f"Question {qid} missing _qa_type"
        assert q["_qa_type"] in ("ladder", "dormant", "distractor"), (
            f"Unexpected _qa_type: {q['_qa_type']!r}"
        )


# ---------------------------------------------------------------------------
# compute_metrics end-to-end
# ---------------------------------------------------------------------------


def _write_outputs_flat(path: Path, records: list) -> None:
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_compute_metrics_correct_prediction():
    """compute_metrics must score correct prediction as accuracy=1.0."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bench_dir, _ = _make_bench_dir(tmpdir)

        outputs_path = tmpdir / "outputs.jsonl"
        _write_outputs_flat(outputs_path, [
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
            use_llm_judge=False,
        )
        assert metrics["overall"]["n"] == 1
        assert metrics["overall"]["accuracy"] == 1.0


def test_compute_metrics_wrong_prediction():
    """compute_metrics must score wrong prediction as accuracy=0.0."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bench_dir, _ = _make_bench_dir(tmpdir)

        outputs_path = tmpdir / "outputs.jsonl"
        _write_outputs_flat(outputs_path, [
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
            use_llm_judge=False,
        )
        assert metrics["overall"]["accuracy"] == 0.0


def test_compute_metrics_accepts_legacy_string_raw_output():
    """compute_metrics must handle plain string raw_output (backward compat)."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bench_dir, _ = _make_bench_dir(tmpdir)

        outputs_path = tmpdir / "outputs.jsonl"
        _write_outputs_flat(outputs_path, [
            {
                "scene_id": "nuscenes-scene-test",
                "sample_id": "SAMPLED_0",
                "question_id": "D1",
                "raw_output": "Answer: No",  # legacy plain string
                "inference_time_s": 1.0,
            }
        ])

        pp = make_postprocessor()
        metrics = pp.compute_metrics(
            outputs_path=outputs_path,
            bench_dir=bench_dir,
            use_llm_judge=False,
        )
        assert metrics["overall"]["accuracy"] == 1.0


def test_compute_metrics_per_qa_type_breakdown():
    """compute_metrics must report accuracy broken down by qa_type."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bench_dir, _ = _make_bench_dir(tmpdir)

        outputs_path = tmpdir / "outputs.jsonl"
        _write_outputs_flat(outputs_path, [
            {"question_id": "L1", "raw_output": {"text": "Answer: A"}},   # correct
            {"question_id": "D1", "raw_output": {"text": "Answer: Yes"}}, # wrong (expected No)
            {"question_id": "X1", "raw_output": {"text": "Answer: No"}},  # correct
        ])

        pp = make_postprocessor()
        metrics = pp.compute_metrics(
            outputs_path=outputs_path,
            bench_dir=bench_dir,
            use_llm_judge=False,
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
        "overall": {"accuracy": 0.75, "n": 8},
        "per_qa_type": {
            "dormant":    {"accuracy": 0.8, "n": 5, "correct": 4},
            "distractor": {"accuracy": 0.67, "n": 3, "correct": 2},
        },
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
