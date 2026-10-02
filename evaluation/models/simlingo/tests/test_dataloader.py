"""
Unit tests for SimLingoDataLoader.

Validates the new API:
  - build_image_paths() returns list of {path, time_key, camera_key} dicts
  - format_sample() uses record["scene_id"], record["sample_id"], record["state"]
  - get_prompt() returns only model-specific fields (no framework fields)
  - iter_questions() yields complete prompt record dicts

Run from the repo root:
    pytest evaluation/models/simlingo/tests/test_dataloader.py -v

Run all simlingo tests:
    pytest evaluation/models/simlingo/tests/ -v

Inside the container:
    uv run python -m pytest /workspace/repo/evaluation/models/simlingo/tests/ -v
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
from dataloader import (  # noqa: E402
    SimLingoDataLoader,
    FRAME_ORDER,
    CAMERA_ORDER,
    _INCLUDE_SIDE,
    _SIDE_CAMS,
)

SAMPLE_DATA = Path(__file__).parents[4] / "evaluation/sample_data/causal_nuscenes"
NUSCENES_SCENE = "nuscenes-scene-0159"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_loader() -> tuple[SimLingoDataLoader, BenchmarkDataset]:
    ds = BenchmarkDataset(bench_dir=SAMPLE_DATA)
    loader = SimLingoDataLoader(
        dataset=ds,
        model_cfg={"qa_types": ["ladder", "dormant", "distractor"]},
    )
    return loader, ds


@pytest.fixture
def sample_record() -> dict:
    _, ds = make_loader()
    return ds.load_record(NUSCENES_SCENE, sample_idx=0)


@pytest.fixture
def loader_and_ds() -> tuple[SimLingoDataLoader, BenchmarkDataset]:
    return make_loader()


# ---------------------------------------------------------------------------
# format_question_text (base class)
# ---------------------------------------------------------------------------


def test_format_question_text_binary_no_options():
    q = {"question": "Is the crosswalk affecting driving?", "options": None}
    text = ModelDataLoader.format_question_text(q)
    assert "Question:" in text
    assert "A)" not in text


def test_format_question_text_mcq_list():
    q = {
        "question": "Which element?",
        "options": ["A) truck", "B) car", "C) ped", "D) light"],
    }
    text = ModelDataLoader.format_question_text(q)
    for letter in "ABCD":
        assert f"{letter})" in text


def test_format_question_text_mcq_dict():
    q = {
        "question": "Which element?",
        "options": {"A": "truck", "B": "car", "C": "ped", "D": "light"},
    }
    text = ModelDataLoader.format_question_text(q)
    for letter in "ABCD":
        assert f"{letter})" in text


# ---------------------------------------------------------------------------
# build_image_paths — abstract method contract
# ---------------------------------------------------------------------------


def test_build_image_paths_returns_list(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    assert isinstance(paths, list)


def test_build_image_paths_dict_structure(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    for item in paths:
        assert "path" in item
        assert "time_key" in item
        assert "camera_key" in item
        assert isinstance(item["path"], str)


def test_build_image_paths_json_serialisable(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    json.dumps(paths)  # must not raise


def test_build_image_paths_count(loader_and_ds, sample_record):
    """Should return 4 entries (front camera × 4 timesteps) when _INCLUDE_SIDE=False."""
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    expected = len(FRAME_ORDER) * len(CAMERA_ORDER) + (len(_SIDE_CAMS) if _INCLUDE_SIDE else 0)
    assert len(paths) == expected, f"Expected {expected} paths, got {len(paths)}"


def test_build_image_paths_valid_time_keys(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    valid_time_keys = set(FRAME_ORDER)
    for item in paths:
        assert item["time_key"] in valid_time_keys, (
            f"Unexpected time_key: {item['time_key']!r}"
        )


def test_build_image_paths_valid_camera_keys(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    valid_cams = set(CAMERA_ORDER) | (set(_SIDE_CAMS) if _INCLUDE_SIDE else set())
    for item in paths:
        assert item["camera_key"] in valid_cams, (
            f"Unexpected camera_key: {item['camera_key']!r}"
        )


def test_build_image_paths_front_camera_only(loader_and_ds, sample_record):
    """When _INCLUDE_SIDE=False, all paths must use CAMERA_ORDER cameras only."""
    loader, _ = loader_and_ds
    if _INCLUDE_SIDE:
        pytest.skip("_INCLUDE_SIDE=True; this test is for front-only mode")
    paths = loader.build_image_paths(sample_record)
    for item in paths:
        assert item["camera_key"] in CAMERA_ORDER


def test_build_image_paths_temporal_order(loader_and_ds, sample_record):
    """Primary paths must appear in FRAME_ORDER order for each camera."""
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    # Front camera paths (all if no side cameras)
    front_paths = [p for p in paths if p["camera_key"] == CAMERA_ORDER[0]]
    time_keys = [p["time_key"] for p in front_paths]
    assert time_keys == FRAME_ORDER[:len(time_keys)], (
        f"Expected FRAME_ORDER order, got {time_keys}"
    )


# ---------------------------------------------------------------------------
# format_sample — new API (scene_id / sample_id / state at record top level)
# ---------------------------------------------------------------------------


def test_format_sample_uses_scene_id(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    assert "scene_id" in fmt
    assert fmt["scene_id"] == sample_record["scene_id"]


def test_format_sample_uses_sample_id(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    assert "sample_id" in fmt
    assert fmt["sample_id"] == sample_record["sample_id"]


def test_format_sample_has_images(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    assert "images" in fmt
    expected = len(FRAME_ORDER) * len(CAMERA_ORDER) + (len(_SIDE_CAMS) if _INCLUDE_SIDE else 0)
    assert len(fmt["images"]) == expected, (
        f"Expected {expected} images, got {len(fmt['images'])}"
    )


def test_format_sample_no_old_image_paths_dict(loader_and_ds, sample_record):
    """format_sample must NOT return a dict-keyed image_paths (old format)."""
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    if "image_paths" in fmt:
        assert not isinstance(fmt["image_paths"], dict), (
            "Old dict-keyed image_paths found; use build_image_paths() instead"
        )


def test_format_sample_state_summary(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    assert isinstance(fmt.get("state_summary"), str)


def test_format_sample_ego_text(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    assert isinstance(fmt.get("ego_text"), str)


# ---------------------------------------------------------------------------
# get_prompt — model-specific fields only
# ---------------------------------------------------------------------------


def _first_question(record: dict) -> dict:
    for qa_type in ("ladder", "dormant", "distractor"):
        qs = record["qa"].get(qa_type, [])
        if qs:
            return qs[0]
    pytest.skip("No questions in sample data")


def test_get_prompt_has_prompt_context(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    payload = loader.get_prompt(fmt, _first_question(sample_record))
    assert "prompt_context" in payload


def test_get_prompt_no_framework_fields(loader_and_ds, sample_record):
    """get_prompt must NOT return any framework fields (set by base class)."""
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    payload = loader.get_prompt(fmt, _first_question(sample_record))
    framework = {
        "scene_id", "sample_id", "question_id", "prompt_id",
        "is_evaluated", "question_json_file", "answer_format",
        "question_text", "qa_text", "image_paths",
    }
    overlap = framework & payload.keys()
    assert not overlap, f"get_prompt leaked framework fields: {overlap}"


def test_get_prompt_json_serialisable(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    payload = loader.get_prompt(fmt, _first_question(sample_record))
    json.dumps(payload)  # must not raise


def test_get_prompt_prompt_context_is_str(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    payload = loader.get_prompt(fmt, _first_question(sample_record))
    assert isinstance(payload["prompt_context"], str)


# ---------------------------------------------------------------------------
# iter_questions — new contract (yields dicts)
# ---------------------------------------------------------------------------


def test_iter_questions_yields_dicts(loader_and_ds):
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    for record in loader.iter_questions(sample_ids):
        assert isinstance(record, dict), "iter_questions must yield dicts, not tuples"
        break


def test_iter_questions_framework_fields_present(loader_and_ds):
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    required = {
        "scene_id", "sample_id", "question_id", "prompt_id",
        "is_evaluated", "question_json_file", "answer_format",
        "question_text", "qa_text", "image_paths",
    }
    for record in loader.iter_questions(sample_ids):
        missing = required - record.keys()
        assert not missing, f"Record missing framework fields: {missing}"


def test_iter_questions_model_fields_present(loader_and_ds):
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    for record in loader.iter_questions(sample_ids):
        assert "prompt_context" in record
        break


def test_iter_questions_image_paths_list_format(loader_and_ds):
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    for record in loader.iter_questions(sample_ids):
        ip = record["image_paths"]
        assert isinstance(ip, list), f"image_paths must be list, got {type(ip).__name__}"
        for item in ip:
            assert "path" in item
            assert "time_key" in item
            assert "camera_key" in item
        break


def test_iter_questions_prompt_ids_unique(loader_and_ds):
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    prompt_ids = [r["prompt_id"] for r in loader.iter_questions(sample_ids)]
    assert len(prompt_ids) == len(set(prompt_ids)), "Duplicate prompt_ids found"


def test_iter_questions_is_evaluated_false(loader_and_ds):
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    for record in loader.iter_questions(sample_ids):
        assert record["is_evaluated"] is False


# ---------------------------------------------------------------------------
# save_prompts
# ---------------------------------------------------------------------------


def test_save_prompts_creates_file(tmp_path, loader_and_ds):
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    out = tmp_path / "prompts.jsonl"
    loader.save_prompts(sample_ids, out)
    assert out.exists()


def test_save_prompts_valid_jsonl(tmp_path, loader_and_ds):
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    out = tmp_path / "prompts.jsonl"
    n = loader.save_prompts(sample_ids, out)
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == n
    for line in lines:
        json.loads(line)


# ---------------------------------------------------------------------------
# Dataset helpers — new record schema validation
# ---------------------------------------------------------------------------


def test_dataset_lists_scenes():
    ds = BenchmarkDataset(bench_dir=SAMPLE_DATA)
    assert len(ds.list_scenes()) > 0


def test_dataset_record_uses_scene_id():
    ds = BenchmarkDataset(bench_dir=SAMPLE_DATA)
    record = ds.load_record(NUSCENES_SCENE, sample_idx=0)
    assert "scene_id" in record
    assert "sample_id" in record
    assert record["scene_id"] == NUSCENES_SCENE


def test_dataset_record_state_at_top_level():
    """state must be at record top level (not inside frames['Tp0p0'])."""
    ds = BenchmarkDataset(bench_dir=SAMPLE_DATA)
    record = ds.load_record(NUSCENES_SCENE, sample_idx=0)
    assert "state" in record
    frames_tp0 = record.get("frames", {}).get("Tp0p0", {})
    assert "state_json" not in frames_tp0, (
        "state_json inside frames['Tp0p0'] is the old format; use record['state']"
    )


def test_dataset_frames_new_format():
    """frames must use {cam_key: path}, not {cam_front_path: path}."""
    ds = BenchmarkDataset(bench_dir=SAMPLE_DATA)
    record = ds.load_record(NUSCENES_SCENE, sample_idx=0)
    for time_key, cam_dict in record["frames"].items():
        for cam_key in cam_dict:
            assert not cam_key.endswith("_path"), (
                f"Old-style cam key '{cam_key}'. Expected plain name like 'cam_front'."
            )


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
