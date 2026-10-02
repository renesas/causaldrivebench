"""
Unit tests for _build_img_metas calibration logic in orion/inference.py.

Tests that real calib.json data is correctly converted to lidar2img,
intrinsics, and extrinsics matrices with proper resolution scaling.

Run from repo root:
    pytest evaluation/models/orion/tests/test_inference_calib.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import _build_img_metas, CAMERA_ORDER  # noqa: E402

# Representative cam_front entry matching real calib.json layout
_CALIB_FRONT = {
    "intrinsic": [
        [1266.417203046554, 0.0, 816.2670197447984],
        [0.0, 1266.417203046554, 491.50706579294757],
        [0.0, 0.0, 1.0],
    ],
    "R_ego2cam": [
        [0.005684778682715579, -0.9999835174398974, 0.0008050713376760664],
        [-0.0056366677335982955, -0.0008371152724919503, -0.9999837634756282],
        [0.9999679551206575, 0.005680148462035488, -0.005641333641939506],
    ],
    "t_ego2cam": [0.005060319404301624, 1.52053324958156, -1.6923034452898127],
    "image_size": [1600, 900],
}

_CALIB_ONE_CAM = {"cam_front": _CALIB_FRONT}
_IMG_SHAPE = (640, 640, 3)
_CAMERA_ORDER_ONE = ["cam_front"]


def _meta(calib=None, camera_order=None):
    co = camera_order or _CAMERA_ORDER_ONE
    return _build_img_metas(
        sample_id="test_sample",
        n_cameras=len(co),
        img_shape=_IMG_SHAPE,
        calib=calib or {},
        camera_order=co,
    )


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_returns_list_of_one():
    result = _meta(_CALIB_ONE_CAM)
    assert isinstance(result, list) and len(result) == 1


def test_meta_has_required_keys():
    meta = _meta(_CALIB_ONE_CAM)[0]
    for key in ("lidar2img", "intrinsics", "extrinsics", "ego_pose", "can_bus"):
        assert key in meta, f"Missing key: {key}"


def test_lidar2img_count_matches_cameras():
    meta = _meta(_CALIB_ONE_CAM)[0]
    assert len(meta["lidar2img"]) == 1


def test_intrinsics_count_matches_cameras():
    meta = _meta(_CALIB_ONE_CAM)[0]
    assert len(meta["intrinsics"]) == 1


def test_extrinsics_count_matches_cameras():
    meta = _meta(_CALIB_ONE_CAM)[0]
    assert len(meta["extrinsics"]) == 1


# ---------------------------------------------------------------------------
# Intrinsic scaling (1600×900 → 640×640)
# ---------------------------------------------------------------------------


def test_intrinsics_fx_scaled():
    """fx must be scaled by 640/1600 = 0.4."""
    meta = _meta(_CALIB_ONE_CAM)[0]
    K = meta["intrinsics"][0]
    expected_fx = 1266.417203046554 * (640 / 1600)
    assert abs(K[0, 0] - expected_fx) < 1e-6


def test_intrinsics_fy_scaled():
    """fy must be scaled by 640/900."""
    meta = _meta(_CALIB_ONE_CAM)[0]
    K = meta["intrinsics"][0]
    expected_fy = 1266.417203046554 * (640 / 900)
    assert abs(K[1, 1] - expected_fy) < 1e-6


def test_intrinsics_cx_scaled():
    """cx must be scaled by 640/1600 = 0.4."""
    meta = _meta(_CALIB_ONE_CAM)[0]
    K = meta["intrinsics"][0]
    expected_cx = 816.2670197447984 * (640 / 1600)
    assert abs(K[0, 2] - expected_cx) < 1e-6


def test_intrinsics_cy_scaled():
    """cy must be scaled by 640/900."""
    meta = _meta(_CALIB_ONE_CAM)[0]
    K = meta["intrinsics"][0]
    expected_cy = 491.50706579294757 * (640 / 900)
    assert abs(K[1, 2] - expected_cy) < 1e-6


def test_intrinsics_bottom_row():
    """Bottom row of intrinsic must be [0, 0, 1]."""
    meta = _meta(_CALIB_ONE_CAM)[0]
    K = meta["intrinsics"][0]
    np.testing.assert_array_equal(K[2], [0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Extrinsic (ego→cam)
# ---------------------------------------------------------------------------


def test_extrinsic_rotation_block():
    """Top-left 3×3 of extrinsic must match R_ego2cam."""
    meta = _meta(_CALIB_ONE_CAM)[0]
    E = meta["extrinsics"][0]
    R_expected = np.array(_CALIB_FRONT["R_ego2cam"])
    np.testing.assert_allclose(E[:3, :3], R_expected, atol=1e-10)


def test_extrinsic_translation():
    """Top-right column of extrinsic must match t_ego2cam."""
    meta = _meta(_CALIB_ONE_CAM)[0]
    E = meta["extrinsics"][0]
    t_expected = np.array(_CALIB_FRONT["t_ego2cam"])
    np.testing.assert_allclose(E[:3, 3], t_expected, atol=1e-10)


def test_extrinsic_bottom_row():
    """Bottom row of extrinsic must be [0, 0, 0, 1]."""
    meta = _meta(_CALIB_ONE_CAM)[0]
    E = meta["extrinsics"][0]
    np.testing.assert_array_equal(E[3], [0.0, 0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# lidar2img consistency
# ---------------------------------------------------------------------------


def test_lidar2img_is_4x4():
    meta = _meta(_CALIB_ONE_CAM)[0]
    L = meta["lidar2img"][0]
    assert L.shape == (4, 4)


def test_lidar2img_equals_k4x4_times_extrinsic():
    """lidar2img must equal K_4×4 @ extrinsic."""
    meta = _meta(_CALIB_ONE_CAM)[0]
    K = meta["intrinsics"][0]
    E = meta["extrinsics"][0]
    L = meta["lidar2img"][0]

    K_4x4 = np.eye(4)
    K_4x4[:3, :3] = K
    expected = K_4x4 @ E
    np.testing.assert_allclose(L, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Fallback to identity for missing cameras
# ---------------------------------------------------------------------------


def test_missing_camera_gets_identity_lidar2img():
    """Cameras absent from calib must get identity lidar2img."""
    calib = {}  # no cameras present
    meta = _meta(calib, camera_order=["cam_front"])[0]
    np.testing.assert_array_equal(meta["lidar2img"][0], np.eye(4))


def test_missing_camera_gets_identity_intrinsic():
    calib = {}
    meta = _meta(calib, camera_order=["cam_front"])[0]
    np.testing.assert_array_equal(meta["intrinsics"][0], np.eye(3))


def test_missing_camera_gets_identity_extrinsic():
    calib = {}
    meta = _meta(calib, camera_order=["cam_front"])[0]
    np.testing.assert_array_equal(meta["extrinsics"][0], np.eye(4))


# ---------------------------------------------------------------------------
# Multi-camera ordering
# ---------------------------------------------------------------------------


def test_four_camera_count():
    """All four benchmark cameras should produce four entries."""
    calib = {cam: _CALIB_FRONT for cam in CAMERA_ORDER}
    meta = _build_img_metas(
        sample_id="s",
        n_cameras=len(CAMERA_ORDER),
        img_shape=_IMG_SHAPE,
        calib=calib,
        camera_order=CAMERA_ORDER,
    )[0]
    assert len(meta["lidar2img"]) == len(CAMERA_ORDER)
    assert len(meta["intrinsics"]) == len(CAMERA_ORDER)
    assert len(meta["extrinsics"]) == len(CAMERA_ORDER)


def test_partial_calib_mixed_identity_and_real():
    """Only cameras present in calib get real matrices; others get identity."""
    calib = {"cam_front": _CALIB_FRONT}  # only front camera
    meta = _build_img_metas(
        sample_id="s",
        n_cameras=len(CAMERA_ORDER),
        img_shape=_IMG_SHAPE,
        calib=calib,
        camera_order=CAMERA_ORDER,
    )[0]

    # cam_front (index 0) — should have non-identity intrinsics
    K_front = meta["intrinsics"][0]
    assert K_front[0, 0] != 1.0, "cam_front should have real fx, not 1.0"

    # cam_front_left (index 1) — should be identity
    np.testing.assert_array_equal(meta["intrinsics"][1], np.eye(3))


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
