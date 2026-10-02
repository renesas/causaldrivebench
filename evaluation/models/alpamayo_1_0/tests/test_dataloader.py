"""
Unit tests for AlpamayoDataLoader.

Validates the new API:
  - build_image_paths() returns list of {path, time_key, camera_key} dicts (camera-major order)
  - format_sample() uses record["scene_id"], record["sample_id"], record["state"]
  - get_prompt() returns only model-specific fields (framework fields injected by base class)
  - iter_questions() yields complete prompt record dicts (not (sid, qid, payload) tuples)

Run from the repo root:
    pytest evaluation/models/alpamayo_1_0/tests/test_dataloader.py -v

Run all alpamayo tests:
    pytest evaluation/models/alpamayo_1_0/tests/ -v

Inside the container:
    uv run python -m pytest /workspace/repo/evaluation/models/alpamayo_1_0/tests/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Repo root (4 levels up from tests/ folder)
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
# Model dir (for local imports: dataloader, postprocess, inference)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common.base_dataloader import ModelDataLoader
from evaluation.common.dataset import BenchmarkDataset
from dataloader import (  # noqa: E402
    AlpamayoDataLoader,
    FRAME_ORDER,
    CAMERA_ORDER,
    _compute_rotations,
    _heading_to_rotation_matrix,
    interpolate_ego_history,
    _interpolate_ego_history,
)

SAMPLE_DATA = Path(__file__).parents[4] / "evaluation/sample_data/causal_nuscenes"
NUSCENES_SCENE = "nuscenes-scene-0072"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_loader() -> tuple[AlpamayoDataLoader, BenchmarkDataset]:
    ds = BenchmarkDataset(bench_dir=SAMPLE_DATA)
    loader = AlpamayoDataLoader(
        dataset=ds,
        model_cfg={"qa_types": ["ladder", "dormant", "distractor"]},
    )
    return loader, ds


@pytest.fixture
def sample_record() -> dict:
    _, ds = _make_loader()
    return ds.load_record(NUSCENES_SCENE, sample_idx=0)


@pytest.fixture
def loader_and_ds() -> tuple[AlpamayoDataLoader, BenchmarkDataset]:
    return _make_loader()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def test_heading_to_rotation_matrix_identity():
    """theta=0 should give identity-like matrix (pointing forward along +X)."""
    R = _heading_to_rotation_matrix(0.0)
    assert R.shape == (3, 3)
    assert abs(R[0, 0] - 1.0) < 1e-6
    assert abs(R[1, 1] - 1.0) < 1e-6
    assert abs(R[0, 1]) < 1e-6


def test_compute_rotations_shape():
    positions = np.array(
        [[-5.7, 1.7, 0], [-3.9, 0.8, 0], [-2.0, 0.2, 0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    rots = _compute_rotations(positions)
    assert rots.shape == (4, 3, 3), f"Expected (4, 3, 3), got {rots.shape}"


def test_compute_rotations_so3():
    """Each rotation matrix must have det ≈ 1 (valid SO(3))."""
    positions = np.array(
        [[-5.7, 1.7, 0], [-3.9, 0.8, 0], [-2.0, 0.2, 0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    rots = _compute_rotations(positions)
    for i, R in enumerate(rots):
        det = float(np.linalg.det(R))
        assert abs(det - 1.0) < 1e-5, f"Rotation[{i}] det={det:.6f}, expected 1.0"


def test_compute_rotations_stationary():
    """All-zero displacements must not raise and return valid matrices."""
    positions = np.zeros((4, 3), dtype=np.float32)
    rots = _compute_rotations(positions)
    assert rots.shape == (4, 3, 3)
    np.testing.assert_allclose(rots[0], _heading_to_rotation_matrix(0.0), atol=1e-6)


# ---------------------------------------------------------------------------
# interpolate_ego_history
# ---------------------------------------------------------------------------


def test_interpolate_ego_history_output_shapes():
    """interpolate_ego_history must return (16, 3) and (16, 3, 3) arrays."""
    xyz = np.zeros((4, 3), dtype=np.float32)
    rot = np.stack([np.eye(3)] * 4, axis=0).astype(np.float32)
    xyz_dense, rot_dense = interpolate_ego_history(xyz, rot, num_steps=16)
    assert xyz_dense.shape == (16, 3), f"Expected (16, 3), got {xyz_dense.shape}"
    assert rot_dense.shape == (16, 3, 3), f"Expected (16, 3, 3), got {rot_dense.shape}"


def test_interpolate_ego_history_endpoints():
    """Sparse endpoints must map exactly to dense endpoints."""
    xyz = np.array([[-5.0, 1.0, 0.0], [-3.0, 0.5, 0.0], [-1.0, 0.1, 0.0], [0.0, 0.0, 0.0]],
                   dtype=np.float32)
    rot = np.stack([np.eye(3)] * 4).astype(np.float32)
    xyz_dense, _ = interpolate_ego_history(xyz, rot, num_steps=16)
    np.testing.assert_allclose(xyz_dense[0], xyz[0], atol=1e-5, err_msg="t=-1.5 mismatch")
    np.testing.assert_allclose(xyz_dense[-1], xyz[-1], atol=1e-5, err_msg="t=0 mismatch")


def test_interpolate_ego_history_backward_compat():
    """_interpolate_ego_history alias must return 16 steps."""
    xyz = np.zeros((4, 3), dtype=np.float32)
    rot = np.stack([np.eye(3)] * 4).astype(np.float32)
    xyz_dense, rot_dense = _interpolate_ego_history(xyz, rot)
    assert xyz_dense.shape == (16, 3)
    assert rot_dense.shape == (16, 3, 3)


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


def test_build_image_paths_camera_major_order(loader_and_ds, sample_record):
    """Paths must be camera-major: all frames for cam0 before cam1."""
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    if not paths:
        pytest.skip("No image paths — sample data may have no frames")
    cameras_seen: list[str] = []
    current_cam = None
    for item in paths:
        cam = item["camera_key"]
        if cam != current_cam:
            assert cam not in cameras_seen, (
                f"Camera {cam!r} is non-consecutive — not camera-major order"
            )
            cameras_seen.append(cam)
            current_cam = cam


def test_build_image_paths_valid_time_keys(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    valid = set(FRAME_ORDER)
    for item in paths:
        assert item["time_key"] in valid, f"Unexpected time_key: {item['time_key']!r}"


def test_build_image_paths_valid_camera_keys(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    paths = loader.build_image_paths(sample_record)
    valid = set(CAMERA_ORDER)
    for item in paths:
        assert item["camera_key"] in valid, f"Unexpected camera_key: {item['camera_key']!r}"


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


def test_format_sample_image_count(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    expected = len(FRAME_ORDER) * len(CAMERA_ORDER)
    assert len(fmt["images"]) == expected, (
        f"Expected {expected} images, got {len(fmt['images'])}"
    )


def test_format_sample_ego_shapes(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    assert fmt["ego_history_xyz"].shape == (4, 3)
    assert fmt["ego_history_rot"].shape == (4, 3, 3)


def test_format_sample_t0_is_origin(loader_and_ds, sample_record):
    """T=0 position must be [0, 0, 0] (ego-centric frame)."""
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    np.testing.assert_array_equal(fmt["ego_history_xyz"][-1], [0.0, 0.0, 0.0])


def test_format_sample_state_summary(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    assert isinstance(fmt.get("state_summary"), str)
    assert len(fmt["state_summary"]) > 0


def test_format_sample_no_old_image_paths_dict(loader_and_ds, sample_record):
    """format_sample must NOT return a dict-keyed image_paths (old format)."""
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    if "image_paths" in fmt:
        assert not isinstance(fmt["image_paths"], dict), (
            "Old dict-keyed image_paths found; use build_image_paths() instead"
        )


# ---------------------------------------------------------------------------
# get_prompt — model-specific fields only
# ---------------------------------------------------------------------------


def _first_question(record: dict) -> dict:
    for qa_type in ("ladder", "dormant", "distractor"):
        qs = record["qa"].get(qa_type, [])
        if qs:
            return qs[0]
    pytest.skip("No questions in sample data")


def test_get_prompt_model_specific_fields(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    payload = loader.get_prompt(fmt, _first_question(sample_record))
    assert "ego_history_xyz" in payload
    assert "ego_history_rot" in payload


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


def test_get_prompt_ego_xyz_is_list(loader_and_ds, sample_record):
    loader, _ = loader_and_ds
    fmt = loader.format_sample(sample_record)
    payload = loader.get_prompt(fmt, _first_question(sample_record))
    assert isinstance(payload["ego_history_xyz"], list)
    assert isinstance(payload["ego_history_rot"], list)


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
        assert "ego_history_xyz" in record
        assert "ego_history_rot" in record
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
    lines = [l for l in out.read_text().splitlines() if l.strip()]
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
    """state must be at record top level (not inside frames["Tp0p0"])."""
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
