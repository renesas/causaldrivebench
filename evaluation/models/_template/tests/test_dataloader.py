"""
Unit tests for the TemplateDataLoader.

Uses the factory pattern — ``make_loader()`` is the single customisation point
when adapting this file for a new model.  All tests validate the ModelDataLoader
*interface contract*, so they work for any conforming implementation.

Run from the repo root:
    pytest evaluation/models/_template/tests/test_dataloader.py -v

Or inside the container:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Resolve repo root whether running from container or locally
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from evaluation.common.base_dataloader import ModelDataLoader
from evaluation.common.dataset import BenchmarkDataset

# TODO: update this import when you rename your DataLoader class
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader import (  # noqa: E402
    TemplateDataLoader,
    NuScenesDataLoader,
    Argoverse2DataLoader,
    OpenSceneDataLoader,
    make_dataloader,
    dataset_name_from_bench_dir,
    DATASET_CAMERAS,
)

# ---------------------------------------------------------------------------
# Sample data paths
# ---------------------------------------------------------------------------

# The bundled sample data lives under evaluation/sample_data/
_SAMPLE_DATA_ROOT = Path(__file__).parents[4] / "evaluation" / "sample_data"

# nuscenes sample data — used as default for tests
NUSCENES_SAMPLE_DATA = _SAMPLE_DATA_ROOT / "causal_nuscenes"

# The scene available in sample data
NUSCENES_SCENE = "nuscenes-scene-0159"


# ---------------------------------------------------------------------------
# Factory functions — update ONLY these when customising for a new model
# ---------------------------------------------------------------------------


def make_dataset(bench_dir: Path = NUSCENES_SAMPLE_DATA) -> BenchmarkDataset:
    """Factory: benchmark dataset pointing at the bundled sample data."""
    return BenchmarkDataset(bench_dir=bench_dir)


def make_loader(
    bench_dir: Path = NUSCENES_SAMPLE_DATA,
    dataset_name: str = "nuscenes",
) -> tuple[TemplateDataLoader, BenchmarkDataset]:
    """
    Factory: returns (DataLoader, Dataset) using the per-dataset factory.

    TODO: Update dataset_name when customising for a different dataset.
    """
    ds = make_dataset(bench_dir)
    loader = make_dataloader(
        dataset_name=dataset_name,
        dataset=ds,
        model_cfg={"qa_types": ["ladder", "dormant", "distractor"]},
    )
    return loader, ds


def _first_question(record: dict) -> dict:
    """Return the first available question from any QA type."""
    for qa_type in ("ladder", "dormant", "distractor"):
        qs = record["qa"].get(qa_type, [])
        if qs:
            return qs[0]
    pytest.skip("No questions found in this sample — check sample data")


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_record() -> dict:
    """Load the first sample from the bundled nuscenes data."""
    _, ds = make_loader()
    return ds.load_record(NUSCENES_SCENE, sample_idx=0)


@pytest.fixture
def loader_and_ds() -> tuple[TemplateDataLoader, BenchmarkDataset]:
    return make_loader()


# ---------------------------------------------------------------------------
# Static helpers — format_question_text / format_qa_text
# ---------------------------------------------------------------------------


def test_format_question_text_binary_no_options():
    """Binary questions with options=None must not raise."""
    q = {"question": "Is the crosswalk affecting driving?", "options": None}
    text = ModelDataLoader.format_question_text(q)
    assert "Question:" in text
    assert "A)" not in text


def test_format_question_text_mcq_list_options():
    """MCQ with list-style options (the actual format in QA files)."""
    q = {
        "question": "Which element is the primary cause?",
        "options": [
            "A) The truck",
            "B) The car",
            "C) The pedestrian",
            "D) The traffic light",
        ],
    }
    text = ModelDataLoader.format_question_text(q)
    for letter in "ABCD":
        assert f"{letter})" in text, f"Option {letter} missing from text"


def test_format_question_text_mcq_dict_options():
    """MCQ with legacy dict-style options must also work."""
    q = {
        "question": "What is blocking you?",
        "options": {"A": "truck", "B": "car", "C": "pedestrian", "D": "signal"},
    }
    text = ModelDataLoader.format_question_text(q)
    for letter in "ABCD":
        assert f"{letter})" in text


def test_format_qa_text_includes_answer(sample_record):
    """format_qa_text must include the ground-truth answer."""
    q = _first_question(sample_record)
    text = ModelDataLoader.format_qa_text(q)
    assert "Answer:" in text
    assert q["correct_answer"] in text


def test_format_qa_text_includes_reasoning(sample_record):
    """format_qa_text must include the reasoning text."""
    q = _first_question(sample_record)
    text = ModelDataLoader.format_qa_text(q)
    reasoning_snippet = q.get("reasoning", "")[:30]
    if reasoning_snippet:
        assert reasoning_snippet in text


# ---------------------------------------------------------------------------
# build_image_paths — interface contract
# ---------------------------------------------------------------------------


def test_build_image_paths_returns_list(loader_and_ds, sample_record):
    """build_image_paths must return a list (possibly empty if images absent)."""
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    assert isinstance(paths, list)


def test_build_image_paths_dict_structure(loader_and_ds, sample_record):
    """Each element must have 'path', 'time_key', 'camera_key' keys."""
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    for item in paths:
        assert "path" in item, "Missing 'path' key"
        assert "time_key" in item, "Missing 'time_key' key"
        assert "camera_key" in item, "Missing 'camera_key' key"
        assert isinstance(item["path"], str), "'path' must be a string"


def test_build_image_paths_all_serialisable(loader_and_ds, sample_record):
    """build_image_paths output must round-trip through JSON."""
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    json.dumps(paths)  # must not raise


# ---------------------------------------------------------------------------
# format_sample — interface contract
# ---------------------------------------------------------------------------


def test_format_sample_required_keys(loader_and_ds, sample_record):
    """format_sample must return at minimum scene_id and sample_id."""
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    for key in ("scene_id", "sample_id"):
        assert key in fmt, f"Missing key '{key}' in format_sample output"


def test_format_sample_scene_propagated(loader_and_ds, sample_record):
    """scene_id must match the original record."""
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    assert fmt["scene_id"] == sample_record["scene_id"]


def test_format_sample_sample_propagated(loader_and_ds, sample_record):
    """sample_id must match the original record."""
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    assert fmt["sample_id"] == sample_record["sample_id"]


# ---------------------------------------------------------------------------
# get_prompt — interface contract
# ---------------------------------------------------------------------------


def test_get_prompt_json_serialisable(loader_and_ds, sample_record):
    """get_prompt payload must round-trip through JSON."""
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    q = _first_question(sample_record)
    payload = loader.get_prompt(fmt, q)
    json.dumps(payload)  # must not raise TypeError / ValueError


# ---------------------------------------------------------------------------
# iter_questions — full prompt record contract
# ---------------------------------------------------------------------------


def test_iter_questions_yields_records(loader_and_ds):
    """iter_questions must yield at least one prompt record."""
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    records = list(loader.iter_questions(sample_ids))
    assert len(records) > 0, "iter_questions yielded nothing"


def test_iter_questions_record_schema(loader_and_ds):
    """Every yielded record must contain the required framework fields."""
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


def test_iter_questions_prompt_ids_unique(loader_and_ds):
    """prompt_id values must be unique within a single iter_questions call."""
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    prompt_ids = [r["prompt_id"] for r in loader.iter_questions(sample_ids)]
    assert len(prompt_ids) == len(set(prompt_ids)), "Duplicate prompt_id values found"


def test_iter_questions_is_evaluated_false(loader_and_ds):
    """All newly generated prompt records must have is_evaluated=False."""
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    for record in loader.iter_questions(sample_ids):
        assert record["is_evaluated"] is False


def test_iter_questions_image_paths_structure(loader_and_ds):
    """image_paths must be a list of dicts with path/time_key/camera_key."""
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    for record in loader.iter_questions(sample_ids):
        for item in record["image_paths"]:
            assert "path" in item
            assert "time_key" in item
            assert "camera_key" in item
        break  # check first record only


# ---------------------------------------------------------------------------
# save_prompts — output file contract
# ---------------------------------------------------------------------------


def test_save_prompts_creates_file(tmp_path, loader_and_ds):
    """save_prompts must create the output file."""
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    out = tmp_path / "prompts.jsonl"
    loader.save_prompts(sample_ids, out)
    assert out.exists()


def test_save_prompts_valid_jsonl(tmp_path, loader_and_ds):
    """Every line in the output file must be valid JSON."""
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    out = tmp_path / "prompts.jsonl"
    n = loader.save_prompts(sample_ids, out)
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == n
    for line in lines:
        json.loads(line)  # must not raise


def test_save_prompts_returns_count(tmp_path, loader_and_ds):
    """save_prompts must return the number of records written."""
    loader, ds = loader_and_ds
    sample_ids = ds.get_sample_ids(mode="full")
    out = tmp_path / "prompts.jsonl"
    n = loader.save_prompts(sample_ids, out)
    assert n > 0
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == n


# ---------------------------------------------------------------------------
# Dataset helpers (sanity checks for the sample data)
# ---------------------------------------------------------------------------


def test_dataset_lists_scenes():
    """list_scenes must return at least one scene from sample data."""
    ds = make_dataset()
    scenes = ds.list_scenes()
    assert len(scenes) > 0


def test_dataset_list_samples():
    """list_samples must return SAMPLED_N entries for the test scene."""
    ds = make_dataset()
    samples = ds.list_samples(NUSCENES_SCENE)
    assert len(samples) > 0
    for s in samples:
        assert s.startswith("SAMPLED_")


def test_dataset_load_record_keys():
    """load_record must return dict with required top-level keys."""
    ds = make_dataset()
    record = ds.load_record(NUSCENES_SCENE, sample_idx=0)
    for key in ("scene_id", "sample_id", "meta", "frames", "state", "calib", "qa"):
        assert key in record, f"Missing key '{key}' in record"


def test_dataset_qa_types():
    """Record qa dict must have ladder/dormant/distractor keys."""
    ds = make_dataset()
    record = ds.load_record(NUSCENES_SCENE, sample_idx=0)
    for qa_type in ("ladder", "dormant", "distractor"):
        assert qa_type in record["qa"], f"Missing qa_type '{qa_type}'"


def test_dataset_frames_structure():
    """Frames dict must have time_key → cam_key → str structure."""
    ds = make_dataset()
    record = ds.load_record(NUSCENES_SCENE, sample_idx=0)
    frames = record["frames"]
    assert isinstance(frames, dict)
    for time_key, cam_dict in frames.items():
        assert isinstance(cam_dict, dict)
        for cam_key, path in cam_dict.items():
            assert isinstance(path, str), f"Path for {time_key}/{cam_key} is not a string"


# ---------------------------------------------------------------------------
# Per-dataset factory and camera configuration tests
# ---------------------------------------------------------------------------

_ARGOVERSE2_SAMPLE_DATA = _SAMPLE_DATA_ROOT / "causal_argoverse2"
_OPENSCENE_SAMPLE_DATA  = _SAMPLE_DATA_ROOT / "causal_openscene"


def test_factory_nuscenes_type():
    """make_dataloader('nuscenes') must return NuScenesDataLoader."""
    loader, _ = make_loader(NUSCENES_SAMPLE_DATA, "nuscenes")
    assert isinstance(loader, NuScenesDataLoader)


def test_factory_argoverse2_type():
    """make_dataloader('argoverse2') must return Argoverse2DataLoader."""
    if not _ARGOVERSE2_SAMPLE_DATA.exists():
        pytest.skip("argoverse2 sample data not found")
    ds = make_dataset(_ARGOVERSE2_SAMPLE_DATA)
    loader = make_dataloader("argoverse2", ds, {"qa_types": ["ladder"]})
    assert isinstance(loader, Argoverse2DataLoader)


def test_factory_openscene_type():
    """make_dataloader('openscene') must return OpenSceneDataLoader."""
    if not _OPENSCENE_SAMPLE_DATA.exists():
        pytest.skip("openscene sample data not found")
    ds = make_dataset(_OPENSCENE_SAMPLE_DATA)
    loader = make_dataloader("openscene", ds, {"qa_types": ["ladder"]})
    assert isinstance(loader, OpenSceneDataLoader)


def test_factory_unknown_dataset_raises():
    """make_dataloader with unknown name must raise ValueError."""
    ds = make_dataset()
    with pytest.raises(ValueError, match="Unknown dataset_name"):
        make_dataloader("unknown_dataset", ds, {})


def test_nuscenes_camera_config():
    """NuScenesDataLoader must use the correct camera keys."""
    assert "cam_back" in DATASET_CAMERAS["nuscenes"]
    assert "cam_back_left" not in DATASET_CAMERAS["nuscenes"]


def test_argoverse2_camera_config():
    """Argoverse2DataLoader must use back-left and back-right (no cam_back)."""
    assert "cam_back_left" in DATASET_CAMERAS["argoverse2"]
    assert "cam_back_right" in DATASET_CAMERAS["argoverse2"]
    assert "cam_back" not in DATASET_CAMERAS["argoverse2"]


def test_dataset_name_from_bench_dir():
    """dataset_name_from_bench_dir must strip the 'causal_' prefix."""
    assert dataset_name_from_bench_dir("causal_nuscenes") == "nuscenes"
    assert dataset_name_from_bench_dir("causal_argoverse2") == "argoverse2"
    assert dataset_name_from_bench_dir("causal_openscene") == "openscene"
    assert dataset_name_from_bench_dir("nuscenes") == "nuscenes"  # no prefix


def test_argoverse2_image_paths():
    """Argoverse2 sample must return paths with argoverse2-specific camera keys."""
    if not _ARGOVERSE2_SAMPLE_DATA.exists():
        pytest.skip("argoverse2 sample data not found")
    ds = make_dataset(_ARGOVERSE2_SAMPLE_DATA)
    loader = make_dataloader("argoverse2", ds, {"qa_types": ["ladder"]})
    scene = ds.list_scenes()[0]
    record = ds.load_record(scene, sample_idx=0)
    paths = loader.build_image_paths(record)
    camera_keys_in_paths = {p["camera_key"] for p in paths}
    # Argoverse2 should have back_left and/or back_right but not cam_back
    assert "cam_back" not in camera_keys_in_paths
    expected = {"cam_back_left", "cam_back_right"}
    assert expected & camera_keys_in_paths, (
        f"No argoverse2-specific cameras found in: {camera_keys_in_paths}"
    )


def test_openscene_image_paths():
    """OpenScene sample must return paths with openscene camera keys."""
    if not _OPENSCENE_SAMPLE_DATA.exists():
        pytest.skip("openscene sample data not found")
    ds = make_dataset(_OPENSCENE_SAMPLE_DATA)
    loader = make_dataloader("openscene", ds, {"qa_types": ["ladder"]})
    scene = ds.list_scenes()[0]
    record = ds.load_record(scene, sample_idx=0)
    paths = loader.build_image_paths(record)
    # openscene should have some paths
    assert len(paths) > 0


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
