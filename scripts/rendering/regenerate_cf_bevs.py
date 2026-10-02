#!/usr/bin/env python3
"""Regenerate BEV + camera overlay images for existing CF trajectory results.

Reads result.json from each cf_trajectories/ subfolder and draws:
- bev_original.png: original trajectory (green) on BEV
- bev_counterfactual.png: counterfactual trajectory (red) on BEV
- bev_overlay.png: both trajectories overlaid
- cam_original.png: original trajectory (green) on front camera
- cam_counterfactual.png: counterfactual trajectory (red) on front camera
- cam_proj.png: both trajectories on front camera

For observation_window CFs, uses T=-1.5 BEV and camera instead of T=0.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config as _load_config

import cv2
import numpy as np

BEV_SIZE = 1536
BEV_RANGE = 30.0
ORIG_COLOR_BGR = (0, 255, 0)      # green
TRAJ_COLOR_BGR = (0, 0, 255)      # red


def ego_local_to_bev(x, y):
    cx, cy = BEV_SIZE // 2, BEV_SIZE // 2
    scale = BEV_SIZE / (2 * BEV_RANGE)
    px = int(cx + y * scale)
    py = int(cy - x * scale)
    return px, py


def draw_trajectory_on_bev(bev_img, waypoints, color, thickness=2, radius=4):
    pts = []
    for wp in waypoints:
        if isinstance(wp, list) and len(wp) >= 2:
            px, py = ego_local_to_bev(wp[0], wp[1])
        elif isinstance(wp, dict):
            px, py = ego_local_to_bev(wp["x"], wp["y"])
        else:
            continue
        pts.append((px, py))
        cv2.circle(bev_img, (px, py), radius, color, -1)
    for i in range(len(pts) - 1):
        cv2.line(bev_img, pts[i], pts[i + 1], color, thickness)


def project_to_cam(x, y, z, calib_cam):
    """Project ego-local 3D point to camera pixel coords."""
    R = np.array(calib_cam["R_ego2cam"])
    t = np.array(calib_cam["t_ego2cam"])
    K = np.array(calib_cam["intrinsic"])
    pt_ego = np.array([x, y, z])
    pt_cam = R @ pt_ego + t
    if pt_cam[2] <= 0.1:
        return None
    u = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
    v = K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]
    return int(u), int(v)


def draw_trajectory_on_cam(cam_img, waypoints, color, calib_cam, thickness=2, radius=4):
    """Draw waypoints on camera image with ground corridor projection."""
    h, w = cam_img.shape[:2]
    pts = []
    for wp in waypoints:
        if isinstance(wp, list) and len(wp) >= 4:
            x, y = wp[0], wp[1]
        elif isinstance(wp, dict):
            x, y = wp["x"], wp["y"]
        else:
            continue
        for dy in [-0.9, 0, 0.9]:
            uv = project_to_cam(x, y + dy, 0.0, calib_cam)
            if uv and 0 <= uv[0] < w and 0 <= uv[1] < h:
                cv2.circle(cam_img, uv, radius, color, -1)
        uv_center = project_to_cam(x, y, 0.0, calib_cam)
        if uv_center and 0 <= uv_center[0] < w and 0 <= uv_center[1] < h:
            pts.append(uv_center)
    for i in range(len(pts) - 1):
        cv2.line(cam_img, pts[i], pts[i + 1], color, thickness)


def process_result(result_path: Path, records_dir: Path):
    result = json.loads(result_path.read_text())
    gen = result.get("generator", {})
    if not gen:
        return False

    scene_id = result.get("scene", "")
    parts = scene_id.rsplit("_SAMPLED_", 1)
    if len(parts) == 2:
        scene_name, sample_idx = parts[0], f"SAMPLED_{parts[1]}"
    else:
        return False

    scene_dir = records_dir / scene_name / sample_idx
    if not scene_dir.exists():
        print(f"  SKIP {scene_id}: scene_dir not found at {scene_dir}")
        return False

    temporal_anchor = gen.get("temporal_anchor", "t0")
    orig_traj = gen.get("original_trajectory", [])
    cf_traj = gen.get("counterfactual_trajectory", [])

    if not orig_traj and not cf_traj:
        return False

    # Pick timestep dir based on temporal anchor
    if temporal_anchor == "observation_window":
        ts_dir = scene_dir / "Tm1p5"
    else:
        ts_dir = scene_dir / "Tp0p0"

    bev_path = ts_dir / "bev.png"
    bev_base = cv2.imread(str(bev_path))
    if bev_base is None:
        bev_base = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.uint8)

    out_dir = result_path.parent

    # ── BEV overlays ──
    bev_orig = bev_base.copy()
    draw_trajectory_on_bev(bev_orig, orig_traj, ORIG_COLOR_BGR, thickness=3, radius=5)
    cv2.imwrite(str(out_dir / "bev_original.png"), bev_orig)

    if cf_traj:
        bev_cf = bev_base.copy()
        draw_trajectory_on_bev(bev_cf, cf_traj, TRAJ_COLOR_BGR, thickness=3, radius=5)
        cv2.imwrite(str(out_dir / "bev_counterfactual.png"), bev_cf)

    bev_combined = bev_base.copy()
    draw_trajectory_on_bev(bev_combined, orig_traj, ORIG_COLOR_BGR, thickness=3, radius=5)
    if cf_traj:
        draw_trajectory_on_bev(bev_combined, cf_traj, TRAJ_COLOR_BGR, thickness=2, radius=4)
    cv2.imwrite(str(out_dir / "bev_overlay.png"), bev_combined)

    # ── Camera overlays ──
    cam_path = ts_dir / "cam_front.jpg"
    cam_base = cv2.imread(str(cam_path))
    calib_path = ts_dir / "calib.json"
    calib_cam = None
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
        calib_cam = calib.get("cam_front")

    if cam_base is not None and calib_cam:
        cam_orig = cam_base.copy()
        draw_trajectory_on_cam(cam_orig, orig_traj, ORIG_COLOR_BGR, calib_cam, thickness=6, radius=12)
        cv2.imwrite(str(out_dir / "cam_original.png"), cam_orig)

        if cf_traj:
            cam_cf = cam_base.copy()
            draw_trajectory_on_cam(cam_cf, cf_traj, TRAJ_COLOR_BGR, calib_cam, thickness=5, radius=10)
            cv2.imwrite(str(out_dir / "cam_counterfactual.png"), cam_cf)

        cam_combined = cam_base.copy()
        draw_trajectory_on_cam(cam_combined, orig_traj, ORIG_COLOR_BGR, calib_cam, thickness=6, radius=12)
        if cf_traj:
            draw_trajectory_on_cam(cam_combined, cf_traj, TRAJ_COLOR_BGR, calib_cam, thickness=5, radius=10)
        cv2.imwrite(str(out_dir / "cam_proj.png"), cam_combined)

    return True


def main():
    # Pre-parse --dataset so config defaults are available
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--dataset", choices=["nuscenes", "openscene", "argoverse2"], default=None)
    _known, _ = _pre.parse_known_args()
    _cfg = _load_config(_known.dataset) if _known.dataset else (_load_config() or {})

    parser = argparse.ArgumentParser(description="Regenerate BEV + camera overlays for CF trajectory results")
    parser.add_argument("--dataset", choices=["nuscenes", "openscene", "argoverse2"],
                        default=None, help="Dataset name (used to resolve config defaults)")
    parser.add_argument("--outputs", default=_cfg.get("outputs"), help="Outputs directory")
    parser.add_argument("--records", default=_cfg.get("records"), help="Records directory")
    args = parser.parse_args()
    if not args.outputs:
        parser.error("--outputs is required (or set via --dataset to use config.yaml)")
    if not args.records:
        parser.error("--records is required (or set via --dataset to use config.yaml)")

    outputs_dir = Path(args.outputs)
    records_dir = Path(args.records)

    results = list(set(
        list(outputs_dir.rglob("cf_trajectories/*/result.json")) +
        list(outputs_dir.rglob("cf_trajectories/*/*/result.json"))
    ))

    print(f"Found {len(results)} CF trajectory results in {outputs_dir}")
    generated = 0
    for rp in sorted(results):
        if process_result(rp, records_dir):
            generated += 1

    print(f"\nDone. Generated overlays for {generated}/{len(results)} results.")


if __name__ == "__main__":
    main()
