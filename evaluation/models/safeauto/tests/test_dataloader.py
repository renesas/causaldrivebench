"""
Unit tests for SafeAutoDataLoader.

Tests validate the interface contract and SafeAuto-specific behaviour:
  - Only T=0 (Tp0p0) images are included in image_paths
  - Only 4 nuscenes cameras: front, front_left, front_right, back
  - conversation_text contains 4 <image> tokens
  - conversation_text contains the question text
  - All framework-required fields are present in iter_questions output

Run from repo root:
    pytest evaluation/models/safeauto/tests/test_dataloader.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common.base_dataloader import ModelDataLoader
from evaluation.common.dataset import BenchmarkDataset
from dataloader import (  # noqa: E402
    SafeAutoDataLoader,
    make_dataloader,
    dataset_name_from_bench_dir,
    _CAMERAS,
)

_SAMPLE_DATA_ROOT = Path(__file__).parents[4] / "evaluation" / "sample_data"
NUSCENES_SAMPLE_DATA = _SAMPLE_DATA_ROOT / "causal_nuscenes"
NUSCENES_SCENE = "nuscenes-scene-0159"


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_dataset() -> BenchmarkDataset:
    return BenchmarkDataset(bench_dir=NUSCENES_SAMPLE_DATA)


def make_loader() -> tuple[SafeAutoDataLoader, BenchmarkDataset]:
    ds = make_dataset()
    loader = make_dataloader(
        "nuscenes", ds, {"qa_types": ["ladder", "dormant", "distractor"]}
    )
    return loader, ds


def _first_question(record: dict) -> dict:
    for qa_type in ("ladder", "dormant", "distractor"):
        qs = record["qa"].get(qa_type, [])
        if qs:
            return qs[0]
    pytest.skip("No questions in sample data")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_record() -> dict:
    _, ds = make_loader()
    return ds.load_record(NUSCENES_SCENE, sample_idx=0)


@pytest.fixture
def loader_and_ds():
    return make_loader()


# ---------------------------------------------------------------------------
# SafeAuto-specific: build_image_paths
# ---------------------------------------------------------------------------

def test_build_image_paths_only_t0(loader_and_ds, sample_record):
    """SafeAuto must only include T=0 (Tp0p0) images, not temporal frames."""
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    for item in paths:
        assert item["time_key"] == "Tp0p0", (
            f"Got non-T0 frame: {item['time_key']}"
        )


def test_build_image_paths_four_cameras(loader_and_ds, sample_record):
    """SafeAuto should include up to 4 camera paths."""
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    assert len(paths) <= 4
    assert len(paths) > 0


def test_build_image_paths_correct_camera_keys(loader_and_ds, sample_record):
    """Camera keys must be the 4 nuScenes cameras."""
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    for item in paths:
        assert item["camera_key"] in _CAMERAS, (
            f"Unexpected camera key: {item['camera_key']}"
        )


def test_build_image_paths_no_back_left(loader_and_ds, sample_record):
    """SafeAuto must NOT include cam_back_left or cam_back_right."""
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    cam_keys = {p["camera_key"] for p in paths}
    assert "cam_back_left" not in cam_keys
    assert "cam_back_right" not in cam_keys


def test_build_image_paths_dict_structure(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    for item in paths:
        assert "path" in item
        assert "time_key" in item
        assert "camera_key" in item
        assert isinstance(item["path"], str)


def test_build_image_paths_serialisable(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    json.dumps(paths)


# ---------------------------------------------------------------------------
# SafeAuto-specific: get_prompt
# ---------------------------------------------------------------------------

def test_get_prompt_has_conversation_text(loader_and_ds, sample_record):
    """get_prompt must return 'conversation_text' key."""
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    q = _first_question(sample_record)
    payload = loader.get_prompt(fmt, q)
    assert "conversation_text" in payload


def test_get_prompt_four_image_tokens(loader_and_ds, sample_record):
    """conversation_text must contain exactly 4 <image> tokens (one per camera)."""
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    q = _first_question(sample_record)
    payload = loader.get_prompt(fmt, q)
    text = payload["conversation_text"]
    assert text.count("<image>") == 4, (
        f"Expected 4 <image> tokens, got {text.count('<image>')}"
    )


def test_get_prompt_contains_question_text(loader_and_ds, sample_record):
    """conversation_text must include the question text."""
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    q = _first_question(sample_record)
    payload = loader.get_prompt(fmt, q)
    text = payload["conversation_text"]
    question_snippet = q["question"][:30]
    assert question_snippet in text, (
        f"Question text not found in prompt: {question_snippet!r}"
    )


def test_get_prompt_binary_format_hint(loader_and_ds, sample_record):
    """Binary questions must include Yes/No instruction."""
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    q = next(
        (qq for qa_type in ("dormant", "distractor", "ladder")
         for qq in sample_record["qa"].get(qa_type, [])
         if qq.get("answer_format") == "binary"),
        None,
    )
    if q is None:
        pytest.skip("No binary question in sample data")
    payload = loader.get_prompt(fmt, q)
    text = payload["conversation_text"].lower()
    assert "yes" in text or "no" in text


def test_get_prompt_mcq_format_hint(loader_and_ds, sample_record):
    """MCQ questions must include A/B/C/D instruction."""
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    q = next(
        (qq for qa_type in ("ladder", "dormant", "distractor")
         for qq in sample_record["qa"].get(qa_type, [])
         if qq.get("answer_format") == "mcq"),
        None,
    )
    if q is None:
        pytest.skip("No MCQ question in sample data")
    payload = loader.get_prompt(fmt, q)
    text = payload["conversation_text"]
    assert any(letter in text for letter in ("A", "B", "C", "D"))


def test_get_prompt_serialisable(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    q = _first_question(sample_record)
    payload = loader.get_prompt(fmt, q)
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Interface contract: format_sample
# ---------------------------------------------------------------------------

def test_format_sample_required_keys(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    assert "scene_id" in fmt
    assert "sample_id" in fmt


def test_format_sample_ids_match(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    assert fmt["scene_id"] == sample_record["scene_id"]
    assert fmt["sample_id"] == sample_record["sample_id"]


# ---------------------------------------------------------------------------
# Interface contract: iter_questions
# ---------------------------------------------------------------------------

def test_iter_questions_yields_records(loader_and_ds):
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    records = list(loader.iter_questions(sample_ids))
    assert len(records) > 0


def test_iter_questions_framework_fields(loader_and_ds):
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    required = {
        "scene_id", "sample_id", "question_id", "prompt_id",
        "is_evaluated", "question_json_file", "answer_format",
        "question_text", "qa_text", "image_paths",
    }
    for record in loader.iter_questions(sample_ids):
        missing = required - record.keys()
        assert not missing, f"Missing framework fields: {missing}"


def test_iter_questions_has_conversation_text(loader_and_ds):
    """All SafeAuto prompt records must have conversation_text."""
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    for record in loader.iter_questions(sample_ids):
        assert "conversation_text" in record, "Missing conversation_text in prompt record"


def test_iter_questions_prompt_ids_unique(loader_and_ds):
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    prompt_ids = [r["prompt_id"] for r in loader.iter_questions(sample_ids)]
    assert len(prompt_ids) == len(set(prompt_ids))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_returns_safeauto_loader():
    ds = make_dataset()
    loader = make_dataloader("nuscenes", ds, {})
    assert isinstance(loader, SafeAutoDataLoader)


def test_factory_unknown_dataset_raises():
    ds = make_dataset()
    with pytest.raises(ValueError, match="SafeAuto currently supports"):
        make_dataloader("argoverse2", ds, {})


def test_dataset_name_from_bench_dir():
    assert dataset_name_from_bench_dir("causal_nuscenes") == "nuscenes"
    assert dataset_name_from_bench_dir("causal_openscene") == "openscene"
    assert dataset_name_from_bench_dir("nuscenes") == "nuscenes"


# ---------------------------------------------------------------------------
# save_prompts
# ---------------------------------------------------------------------------

def test_save_prompts_creates_valid_jsonl(tmp_path, loader_and_ds):
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    out = tmp_path / "prompts.jsonl"
    n = loader.save_prompts(sample_ids, out)
    assert out.exists()
    assert n > 0
    for line in out.read_text().splitlines():
        if line.strip():
            json.loads(line)


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
