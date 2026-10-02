"""
Unit tests for OmniDriveDataLoader.

Validates the new API:
  - build_image_paths() returns list of {path, time_key, camera_key} dicts (time-major order)
  - format_sample() uses record["scene_id"], record["sample_id"], record["state"]
  - get_prompt() returns only model-specific fields (framework fields injected by base class)
  - iter_questions() yields complete prompt record dicts (not (sid, qid, payload) tuples)

Run from repo root:
    pytest evaluation/models/omnidrive/tests/test_dataloader.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Repo root (4 levels up from tests/ folder)
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
# Model dir (for local imports: dataloader, postprocess, inference)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common.base_dataloader import ModelDataLoader
from evaluation.common.dataset import BenchmarkDataset
from dataloader import OmniDriveDataLoader, FRAME_ORDER, CAMERA_ORDER  # noqa: E402

SAMPLE_DATA = Path(__file__).parents[4] / "evaluation/sample_data/causal_nuscenes"
NUSCENES_SCENE = "nuscenes-scene-0159"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_loader():
    if not SAMPLE_DATA.exists():
        pytest.skip("sample data not found")
    ds = BenchmarkDataset(SAMPLE_DATA)
    return OmniDriveDataLoader(dataset=ds, model_cfg={})


@pytest.fixture
def loader_and_record():
    loader = make_loader()
    ds = loader.dataset
    scenes = ds.list_scenes()
    scene = next((s for s in scenes if NUSCENES_SCENE in s), scenes[0])
    samples = ds.list_samples(scene)
    record = ds.load_record_by_name(scene, samples[0])
    return loader, record


# ---------------------------------------------------------------------------
# build_image_paths
# ---------------------------------------------------------------------------


def test_build_image_paths_returns_list(loader_and_record):
    loader, record = loader_and_record
    result = loader.build_image_paths(record)
    assert isinstance(result, list)


def test_build_image_paths_entry_structure(loader_and_record):
    loader, record = loader_and_record
    result = loader.build_image_paths(record)
    for item in result:
        assert "path" in item
        assert "time_key" in item
        assert "camera_key" in item


def test_build_image_paths_time_major_order(loader_and_record):
    """TIME-MAJOR: first 4 entries all share the same time_key (Tm1p5)."""
    loader, record = loader_and_record
    result = loader.build_image_paths(record)
    if len(result) < 4:
        pytest.skip("fewer than 4 images in sample data")
    # All cameras at the first timestep come before any camera at the next timestep
    assert result[0]["time_key"] == FRAME_ORDER[0]
    assert result[1]["time_key"] == FRAME_ORDER[0]
    assert result[2]["time_key"] == FRAME_ORDER[0]
    assert result[3]["time_key"] == FRAME_ORDER[0]


def test_build_image_paths_json_serialisable(loader_and_record):
    loader, record = loader_and_record
    result = loader.build_image_paths(record)
    json.dumps(result)  # must not raise


def test_build_image_paths_count(loader_and_record):
    loader, record = loader_and_record
    result = loader.build_image_paths(record)
    expected = len(FRAME_ORDER) * len(CAMERA_ORDER)
    assert len(result) == expected, f"Expected {expected} image paths, got {len(result)}"


# ---------------------------------------------------------------------------
# format_sample
# ---------------------------------------------------------------------------


def test_format_sample_uses_scene_id(loader_and_record):
    loader, record = loader_and_record
    fmt = loader.format_sample(record)
    assert "scene_id" in fmt
    assert fmt["scene_id"] == record["scene_id"]


def test_format_sample_uses_sample_id(loader_and_record):
    loader, record = loader_and_record
    fmt = loader.format_sample(record)
    assert "sample_id" in fmt
    assert fmt["sample_id"] == record["sample_id"]


def test_format_sample_no_old_api_keys(loader_and_record):
    """format_sample must NOT use old 'scene' or 'sample' keys."""
    loader, record = loader_and_record
    fmt = loader.format_sample(record)
    assert "scene" not in fmt, "Old 'scene' key found; use 'scene_id'"
    assert "sample" not in fmt, "Old 'sample' key found; use 'sample_id'"


def test_format_sample_no_image_paths_dict(loader_and_record):
    """format_sample must NOT return a dict-keyed image_paths."""
    loader, record = loader_and_record
    fmt = loader.format_sample(record)
    if "image_paths" in fmt:
        assert not isinstance(fmt["image_paths"], dict), (
            "Old dict-keyed image_paths found; use build_image_paths() instead"
        )


# ---------------------------------------------------------------------------
# get_prompt
# ---------------------------------------------------------------------------


def _first_question(record: dict) -> dict:
    for qa_type in ("ladder", "dormant", "distractor"):
        qs = record.get("qa", {}).get(qa_type, [])
        if qs:
            return qs[0]
    pytest.skip("No questions in sample data")


def test_get_prompt_no_framework_fields(loader_and_record):
    """get_prompt must NOT return any framework fields (set by base class)."""
    loader, record = loader_and_record
    fmt = loader.format_sample(record)
    payload = loader.get_prompt(fmt, _first_question(record))
    framework = {
        "scene_id", "sample_id", "question_id", "prompt_id",
        "is_evaluated", "question_json_file", "answer_format",
        "question_text", "qa_text", "image_paths",
    }
    overlap = framework & payload.keys()
    assert not overlap, f"get_prompt leaked framework fields: {overlap}"


def test_get_prompt_has_prompt_context(loader_and_record):
    loader, record = loader_and_record
    fmt = loader.format_sample(record)
    payload = loader.get_prompt(fmt, _first_question(record))
    assert "prompt_context" in payload
    assert isinstance(payload["prompt_context"], str)
    assert len(payload["prompt_context"]) > 0


def test_get_prompt_json_serialisable(loader_and_record):
    loader, record = loader_and_record
    fmt = loader.format_sample(record)
    payload = loader.get_prompt(fmt, _first_question(record))
    json.dumps(payload)  # must not raise


# ---------------------------------------------------------------------------
# iter_questions
# ---------------------------------------------------------------------------


def test_iter_questions_yields_dicts():
    loader = make_loader()
    ds = loader.dataset
    sample_ids = ds.get_sample_ids(mode="full")
    for record in loader.iter_questions(sample_ids):
        assert isinstance(record, dict), "iter_questions must yield dicts, not tuples"
        break


def test_iter_questions_framework_fields():
    loader = make_loader()
    ds = loader.dataset
    sample_ids = ds.get_sample_ids(mode="full")
    required = {
        "scene_id", "sample_id", "question_id", "prompt_id",
        "is_evaluated", "question_json_file", "answer_format",
        "question_text", "qa_text", "image_paths",
    }
    for record in loader.iter_questions(sample_ids):
        missing = required - record.keys()
        assert not missing, f"Record missing framework fields: {missing}"
        break


def test_iter_questions_image_paths_list_format():
    loader = make_loader()
    ds = loader.dataset
    sample_ids = ds.get_sample_ids(mode="full")
    for record in loader.iter_questions(sample_ids):
        ip = record["image_paths"]
        assert isinstance(ip, list), f"image_paths must be list, got {type(ip).__name__}"
        for item in ip:
            assert "path" in item
            assert "time_key" in item
            assert "camera_key" in item
        break


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
