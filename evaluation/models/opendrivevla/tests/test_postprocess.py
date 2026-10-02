"""
Unit tests for OpenDriveVLAPostprocessor.

Covers:
  - parse_answer() with new dict raw_output format ({"text": "..."})
  - parse_answer() with legacy plain string raw_output (backward compat)
  - binary and MCQ extraction patterns
  - _collect_outputs() with nested scene_id/sample_id directory structure
  - compute_metrics() end-to-end
  - save_report() file creation

Run from the repo root:
    pytest evaluation/models/opendrivevla/tests/test_postprocess.py -v

Run all tests:
    pytest evaluation/models/opendrivevla/tests/ -v

Inside the container:
    uv run python -m pytest /workspace/repo/evaluation/models/opendrivevla/tests/ -v
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
from postprocess import OpenDriveVLAPostprocessor, _collect_outputs  # noqa: E402

_SAMPLE_DATA_ROOT = Path(__file__).parents[4] / "evaluation" / "sample_data"
NUSCENES_SAMPLE_DATA = _SAMPLE_DATA_ROOT / "causal_nuscenes"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_postprocessor() -> OpenDriveVLAPostprocessor:
    return OpenDriveVLAPostprocessor(model_cfg={})


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
    pp = make_postprocessor()
    assert pp.parse_answer(raw_output, BINARY_Q) == expected


# ---------------------------------------------------------------------------
# Binary answer parsing — legacy plain string
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
    pp = make_postprocessor()
    assert pp.parse_answer(raw_output, MCQ_Q) == expected


# ---------------------------------------------------------------------------
# Empty response handling
# ---------------------------------------------------------------------------

def test_parse_empty_dict_raises():
    from evaluation.common.utils.exceptions import EmptyResponseException
    pp = make_postprocessor()
    with pytest.raises(EmptyResponseException):
        pp.parse_answer({"text": ""}, BINARY_Q)


def test_parse_empty_string_raises():
    from evaluation.common.utils.exceptions import EmptyResponseException
    pp = make_postprocessor()
    with pytest.raises(EmptyResponseException):
        pp.parse_answer("", BINARY_Q)


# ---------------------------------------------------------------------------
# _extract_text utility (base class)
# ---------------------------------------------------------------------------

def test_extract_text_from_dict():
    result = ModelPostprocessor._extract_text({"text": "Answer: A\nReasoning: …"})
    assert result == "Answer: A\nReasoning: …"


def test_extract_text_from_string():
    result = ModelPostprocessor._extract_text("Answer: B")
    assert result == "Answer: B"


# ---------------------------------------------------------------------------
# _collect_outputs — nested structure
# ---------------------------------------------------------------------------

def _write_outputs_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_collect_outputs_finds_nested_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_base = Path(tmpdir)
        scene_id = "nuscenes-scene-0001"
        sample_id = "SAMPLED_0"

        out_file = output_base / scene_id / sample_id / "outputs.jsonl"
        _write_outputs_jsonl(out_file, [
            {"question_id": "Q1", "raw_output": {"text": "Answer: Yes"}},
            {"question_id": "Q2", "raw_output": {"text": "Answer: No"}},
        ])

        sample_ids = [{"scene_id": scene_id, "sample_id": sample_id}]
        all_outputs, missing = _collect_outputs(sample_ids, output_base)

        assert len(all_outputs) == 2
        assert len(missing) == 0


def test_collect_outputs_missing_sample():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_base = Path(tmpdir)
        sample_ids = [{"scene_id": "nuscenes-scene-0001", "sample_id": "SAMPLED_0"}]
        all_outputs, missing = _collect_outputs(sample_ids, output_base)

        assert len(all_outputs) == 0
        assert len(missing) == 1
        assert "nuscenes-scene-0001/SAMPLED_0" in missing[0]


def test_collect_outputs_rejects_flat_structure():
    """Old flat scene_sample/ format must NOT be found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_base = Path(tmpdir)
        old_flat_dir = output_base / "nuscenes-scene-0001_SAMPLED_0"
        _write_outputs_jsonl(old_flat_dir / "outputs.jsonl", [
            {"question_id": "Q1", "raw_output": {"text": "Answer: Yes"}}
        ])

        sample_ids = [{"scene_id": "nuscenes-scene-0001", "sample_id": "SAMPLED_0"}]
        all_outputs, missing = _collect_outputs(sample_ids, output_base)

        assert len(all_outputs) == 0
        assert len(missing) == 1


# ---------------------------------------------------------------------------
# compute_metrics end-to-end
# ---------------------------------------------------------------------------

def _make_bench_dir(tmpdir: Path) -> tuple[Path, list]:
    scene_id = "nuscenes-scene-test"
    sample_id = "SAMPLED_0"
    qa_dir = tmpdir / "bench" / scene_id / sample_id / "qa"
    qa_dir.mkdir(parents=True)

    dormant_qa = {"questions": [
        {"id": "D1", "_qa_type": "dormant", "answer_format": "binary",
         "question": "Would a pedestrian change driving?", "correct_answer": "No",
         "options": None, "reasoning": "Pedestrian is not on path."}
    ]}
    ladder_qa = {"questions": [
        {"id": "L1", "_qa_type": "ladder", "answer_format": "mcq",
         "question": "Which element?", "correct_answer": "A",
         "options": ["A) truck", "B) car", "C) ped", "D) sign"],
         "reasoning": "The truck is blocking."}
    ]}

    bench_dir = tmpdir / "bench"
    (qa_dir / "dormant_qa.json").write_text(json.dumps(dormant_qa))
    (qa_dir / "active_qa.json").write_text(json.dumps(ladder_qa))

    sample_ids = [{"scene_id": scene_id, "sample_id": sample_id}]
    return bench_dir, sample_ids


def _write_outputs_flat(path: Path, records: list) -> None:
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_compute_metrics_correct_prediction():
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


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

def test_save_report_creates_file():
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


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
