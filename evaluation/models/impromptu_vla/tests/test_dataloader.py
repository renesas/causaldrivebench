"""
Unit tests for ImpromptuVLADataLoader.

Validates the new API:
  - build_image_paths() returns list of {path, time_key, camera_key} dicts (time-major order)
  - format_sample() uses record["scene_id"], record["sample_id"], record["state"]
  - get_prompt() returns only model-specific fields (framework fields injected by base class)
  - iter_questions() yields complete prompt record dicts (not (sid, qid, payload) tuples)

Run from the repo root:
    pytest evaluation/models/impromptu_vla/tests/test_dataloader.py -v

Run all tests:
    pytest evaluation/models/impromptu_vla/tests/ -v

Inside the container:
    uv run python -m pytest /workspace/repo/evaluation/models/impromptu_vla/tests/ -v
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
from dataloader import ImpromptuVLADataLoader, FRAME_ORDER, CAMERA_ORDER  # noqa: E402

SAMPLE_DATA = Path(__file__).parents[4] / "evaluation/sample_data/causal_nuscenes"
NUSCENES_SCENE = "nuscenes-scene-0159"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_loader():
    if not SAMPLE_DATA.exists():
        pytest.skip("sample data not found")
    ds = BenchmarkDataset(SAMPLE_DATA)
    return ImpromptuVLADataLoader(dataset=ds, model_cfg={})


@pytest.fixture
def loader():
    return make_loader()


@pytest.fixture
def dataset():
    if not SAMPLE_DATA.exists():
        pytest.skip("sample data not found")
    return BenchmarkDataset(SAMPLE_DATA)


@pytest.fixture
def sample_record(dataset):
    return dataset.load_record(NUSCENES_SCENE, sample_idx=0)


# ---------------------------------------------------------------------------
# build_image_paths
# ---------------------------------------------------------------------------

def test_build_image_paths_returns_list(loader, sample_record):
    paths = loader.build_image_paths(sample_record)
    assert isinstance(paths, list)


def test_build_image_paths_dict_structure(loader, sample_record):
    paths = loader.build_image_paths(sample_record)
    for item in paths:
        assert "path" in item
        assert "time_key" in item
        assert "camera_key" in item
        assert isinstance(item["path"], str)


def test_build_image_paths_json_serialisable(loader, sample_record):
    paths = loader.build_image_paths(sample_record)
    json.dumps(paths)  # must not raise


def test_build_image_paths_time_major_order(loader, sample_record):
    """Paths must be time-major: all cameras for frame0 before frame1."""
    paths = loader.build_image_paths(sample_record)
    if not paths:
        pytest.skip("No image paths — sample data may have no frames")
    # In time-major order, time keys appear in blocks
    time_keys_seen: list[str] = []
    current_time = None
    for item in paths:
        tk = item["time_key"]
        if tk != current_time:
            assert tk not in time_keys_seen, (
                f"Time key {tk!r} is non-consecutive — not time-major order"
            )
            time_keys_seen.append(tk)
            current_time = tk


# ---------------------------------------------------------------------------
# format_sample
# ---------------------------------------------------------------------------

def test_format_sample_uses_scene_id(loader, sample_record):
    fmt = loader.format_sample(sample_record)
    assert "scene_id" in fmt
    assert fmt["scene_id"] == sample_record["scene_id"]


def test_format_sample_uses_sample_id(loader, sample_record):
    fmt = loader.format_sample(sample_record)
    assert "sample_id" in fmt
    assert fmt["sample_id"] == sample_record["sample_id"]


def test_format_sample_no_old_keys(loader, sample_record):
    """format_sample must NOT return old 'scene' or 'sample' keys."""
    fmt = loader.format_sample(sample_record)
    assert "scene" not in fmt, "Old 'scene' key found; use 'scene_id'"
    assert "sample" not in fmt, "Old 'sample' key found; use 'sample_id'"


def test_format_sample_no_image_paths_dict(loader, sample_record):
    """format_sample must NOT return a dict-keyed image_paths (old format)."""
    fmt = loader.format_sample(sample_record)
    if "image_paths" in fmt:
        assert not isinstance(fmt["image_paths"], dict), (
            "Old dict-keyed image_paths found; removed from format_sample"
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


def test_get_prompt_no_framework_fields(loader, sample_record):
    """get_prompt must NOT return any framework fields (set by base class)."""
    fmt = loader.format_sample(sample_record)
    payload = loader.get_prompt(fmt, _first_question(sample_record))
    framework = {
        "scene_id", "sample_id", "question_id", "image_paths",
        "qa_text", "answer_format", "question_text",
    }
    overlap = framework & payload.keys()
    assert not overlap, f"get_prompt leaked framework fields: {overlap}"


def test_get_prompt_has_prompt_context(loader, sample_record):
    """get_prompt must return a 'prompt_context' key."""
    fmt = loader.format_sample(sample_record)
    payload = loader.get_prompt(fmt, _first_question(sample_record))
    assert "prompt_context" in payload
    assert isinstance(payload["prompt_context"], str)
    assert len(payload["prompt_context"]) > 0


def test_get_prompt_json_serialisable(loader, sample_record):
    fmt = loader.format_sample(sample_record)
    payload = loader.get_prompt(fmt, _first_question(sample_record))
    json.dumps(payload)  # must not raise


# ---------------------------------------------------------------------------
# iter_questions
# ---------------------------------------------------------------------------

def test_iter_questions_yields_dicts(loader, dataset):
    sample_ids = dataset.get_sample_ids(mode="full")
    for record in loader.iter_questions(sample_ids):
        assert isinstance(record, dict), "iter_questions must yield dicts, not tuples"
        break


def test_iter_questions_framework_fields(loader, dataset):
    sample_ids = dataset.get_sample_ids(mode="full")
    required = {"scene_id", "sample_id", "question_id", "image_paths"}
    for record in loader.iter_questions(sample_ids):
        missing = required - record.keys()
        assert not missing, f"Record missing framework fields: {missing}"
        break


def test_iter_questions_image_paths_list_format(loader, dataset):
    sample_ids = dataset.get_sample_ids(mode="full")
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
