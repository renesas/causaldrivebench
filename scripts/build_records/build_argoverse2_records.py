"""
build_argoverse2_records.py

Convert Argoverse V2 sensor scenes into the standard scenario_records format
used by generate_causal_graph.py.

Output matches the format produced by STSBench/build_scenario_records.py:
  scenario_records/<scene>/<maneuver>/
      Tm1p5/ Tm1p0/ Tm0p5/ Tp0p0/   (camera images per timestep)
      Tp0p0/bev.png  state.json      (BEV + ego-local state at T=0)
      meta.json

Usage:
  python scripts/build_records/build_argoverse2_records.py \
      --data data/argoverse2/sensor \
      --split val \
      --records argoverse2_records \
      --num-scenes 3
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# ── Constants ──────────────────────────────────────────────────────────────
RADIUS = 30          # metres — vehicle filtering radius
RADIUS_PED = 25      # metres — pedestrian filtering radius
BEV_SIZE = 1536
BEV_SUPERSAMPLE = 2
BEV_RANGE = 50.0     # metres each side of ego
# Camera images saved at native resolution (no resize)

# Camera mapping: Argoverse V2 → our convention
CAM_MAP = {
    "ring_front_center": "cam_front",
    "ring_front_left":   "cam_front_left",
    "ring_front_right":  "cam_front_right",
    "ring_rear_right":   "cam_back_right",
    "ring_rear_left":    "cam_back_left",
}

# Categories to SKIP — static objects / clutter, not road users
SKIP_CATEGORIES = {
    "bollard", "construction_barrel", "construction_cone",
    "sign", "stop_sign", "dog", "stroller", "wheeled_device",
}

# Timestep slugs (2Hz: 0.5s apart)
TEMPORAL_OFFSETS = [(-3, "Tm1p5"), (-2, "Tm1p0"), (-1, "Tm0p5"), (0, "Tp0p0")]

# BEV / camera overlay colours (BGR)
COLOR_EGO     = (255, 200, 0)
COLOR_VEHICLE = (180, 200, 255)  # peach (BGR) — distinct from all map/ego colors
COLOR_PED     = (0, 255, 128)
COLOR_CYCLE   = (0, 200, 255)

# Map colours (BGR) — matching STSBench style
MAP_COLORS = {
    "drivable":      (50,  50,  50),
    "lane":          (70,  70,  65),
    "lane_boundary": (180, 180, 180),  # white lane lines
    "ped_crossing":  (50, 100, 180),
    "intersection":  (80,  55, 100),
}


# ── Lane direction arrow helper ───────────────────────────────────────────
def _draw_arrows_along(canvas, bev_pts, spacing, arrow_len, color, thickness=1):
    """Draw evenly-spaced triangle arrows along a polyline of BEV pixel coords."""
    if len(bev_pts) < 2:
        return
    # Build cumulative distance along path
    dists = [0.0]
    for k in range(1, len(bev_pts)):
        dx = bev_pts[k][0] - bev_pts[k-1][0]
        dy = bev_pts[k][1] - bev_pts[k-1][1]
        dists.append(dists[-1] + math.sqrt(dx*dx + dy*dy))
    total = dists[-1]
    if total < 3:
        return
    # Place arrows at exact spacing intervals; at least one per path
    actual_spacing = min(spacing, total * 0.5)
    seg_idx = 0
    for d in np.arange(actual_spacing * 0.5, total, actual_spacing):
        # Find which segment this distance falls in
        while seg_idx < len(dists) - 2 and dists[seg_idx + 1] < d:
            seg_idx += 1
        # Interpolate position
        d0, d1 = dists[seg_idx], dists[seg_idx + 1]
        if d1 - d0 < 0.5:
            continue
        t = (d - d0) / (d1 - d0)
        x0, y0 = bev_pts[seg_idx]
        x1, y1 = bev_pts[seg_idx + 1]
        tx = x0 + t * (x1 - x0)
        ty = y0 + t * (y1 - y0)
        # Direction from segment
        dx, dy = x1 - x0, y1 - y0
        seg = math.sqrt(dx*dx + dy*dy)
        ux, uy = dx / seg, dy / seg
        # Triangle arrow
        p0 = (int(tx), int(ty))
        p1 = (int(tx - arrow_len * (ux - uy * 0.5)),
              int(ty - arrow_len * (uy + ux * 0.5)))
        p2 = (int(tx - arrow_len * (ux + uy * 0.5)),
              int(ty - arrow_len * (uy - ux * 0.5)))
        cv2.fillPoly(canvas, [np.array([p0, p1, p2], dtype=np.int32)], color)


# ── Coordinate utilities ──────────────────────────────────────────────────
def quat_to_yaw(qw, qx, qy, qz):
    """Convert quaternion to yaw angle."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])


def make_se3(R, t):
    """Build 4x4 SE3 matrix from 3x3 rotation and 3-vector translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def make_ego_local_transform(ego_x, ego_y, ego_yaw):
    """Build functions that map global → ego-local at T=0."""
    cos_e, sin_e = math.cos(-ego_yaw), math.sin(-ego_yaw)

    def to_ego(wx, wy):
        dx, dy = wx - ego_x, wy - ego_y
        return round(cos_e * dx - sin_e * dy, 4), round(sin_e * dx + cos_e * dy, 4)

    def norm_yaw(yaw):
        y = yaw - ego_yaw
        return round((y + math.pi) % (2 * math.pi) - math.pi, 4)

    return to_ego, norm_yaw


def get_3d_box_corners(x, y, z, l, w, h, yaw):
    """Get 8 corners of a 3D box."""
    cos_a, sin_a = math.cos(yaw), math.sin(yaw)
    l2, w2, h2 = l / 2, w / 2, h / 2
    corners = []
    for dz in [-h2, h2]:
        for dl, dw in [(l2, w2), (l2, -w2), (-l2, -w2), (-l2, w2)]:
            cx = x + cos_a * dl - sin_a * dw
            cy = y + sin_a * dl + cos_a * dw
            corners.append([cx, cy, z + dz])
    return np.array(corners)


def infer_nav_command_from_trajectory(ego_poses_df, t0_ns, n_future_s=4.0):
    """Infer nav command from final heading in ego-local frame.
    Uses T=-1.5 to T=+4.0 window. Dead zone of 0.15 rad (~8.5°)."""
    t_start = t0_ns - int(1.5 * 1e9)
    t_end = t0_ns + int(n_future_s * 1e9)
    traj = ego_poses_df[(ego_poses_df.timestamp_ns >= t_start) &
                        (ego_poses_df.timestamp_ns <= t_end)]
    if len(traj) < 5:
        return "GO STRAIGHT"
    yaw_start = quat_to_yaw(traj.iloc[0].qw, traj.iloc[0].qx,
                            traj.iloc[0].qy, traj.iloc[0].qz)
    yaw_end = quat_to_yaw(traj.iloc[-1].qw, traj.iloc[-1].qx,
                          traj.iloc[-1].qy, traj.iloc[-1].qz)
    heading_change = (yaw_end - yaw_start + math.pi) % (2 * math.pi) - math.pi
    if abs(heading_change) < 0.15:
        return "GO STRAIGHT"
    elif heading_change > 0:
        return "TURN LEFT"
    else:
        return "TURN RIGHT"


def category_short(cat):
    """Argoverse categories → short form."""
    cat = cat.lower()
    if "pedestrian" in cat or "wheelchair" in cat:
        return "pedestrian"
    elif "bicycle" in cat or "bicyclist" in cat or "cyclist" in cat or "motorcycle" in cat:
        return "cyclist"
    elif "vehicle" in cat or "truck" in cat or "bus" in cat:
        return "vehicle"
    return cat


def _agent_color(cat):
    cat = cat.lower()
    if "pedestrian" in cat or "wheelchair" in cat:
        return COLOR_PED
    elif "cycle" in cat or "bicycle" in cat or "motorcycle" in cat:
        return COLOR_CYCLE
    return COLOR_VEHICLE


def _agent_in_any_cam(corners, calib):
    """Return True if any of the 8 box corners projects into any camera."""
    for cam_name in CAM_MAP:
        if cam_name not in calib:
            continue
        cal = calib[cam_name]
        R, t = cal["R_ego2cam"], cal["t_ego2cam"]
        pts_cam = (R @ corners.T).T + t  # (8, 3)
        mask = pts_cam[:, 2] > 0.5
        if not np.any(mask):
            continue
        pts_front = pts_cam[mask]
        u = cal["fx"] * pts_front[:, 0] / pts_front[:, 2] + cal["cx"]
        v = cal["fy"] * pts_front[:, 1] / pts_front[:, 2] + cal["cy"]
        if np.any((u >= 0) & (u < cal["width"]) & (v >= 0) & (v < cal["height"])):
            return True
    return False


def agents_visible_in_any_fov(agents, calib):
    """Return subset of agents visible in at least one camera FOV.
    Projects all 8 box corners; agent is visible if ANY corner
    projects into ANY camera image.
    """
    visible = []
    for ag in agents:
        corners = get_3d_box_corners(
            ag["x"], ag["y"], ag["z"],
            ag["l"], ag["w"], ag["h"], ag["yaw"]
        )
        if _agent_in_any_cam(corners, calib):
            visible.append(ag)
    return visible


def agents_visible_multitemporal(t0_agents, ann_df, ann_2hz, t0_idx, calib,
                                  track_key="track_uuid"):
    """Check FOV across all 4 temporal frames. Return t0_agents visible in
    ANY frame (t-1.5, t-1.0, t-0.5, t0)."""
    t0_tracks = {a[track_key] for a in t0_agents}
    visible_tracks = set()

    for offset, _ in TEMPORAL_OFFSETS:
        ts_idx = t0_idx + offset
        if ts_idx < 0 or ts_idx >= len(ann_2hz):
            continue
        ts_ns = ann_2hz[ts_idx]
        ann_frame = ann_df[ann_df.timestamp_ns == ts_ns]

        for _, row in ann_frame.iterrows():
            if row.track_uuid not in t0_tracks or row.track_uuid in visible_tracks:
                continue
            corners = get_3d_box_corners(
                float(row.tx_m), float(row.ty_m), float(row.tz_m),
                float(row.length_m), float(row.width_m), float(row.height_m),
                quat_to_yaw(row.qw, row.qx, row.qy, row.qz),
            )
            if _agent_in_any_cam(corners, calib):
                visible_tracks.add(row.track_uuid)

    return [a for a in t0_agents if a[track_key] in visible_tracks]


# ── IOU-based occlusion filter ──────────────────────────────────────────
IOU_OCCL_THRESH = 0.80   # drop if projected-2D IOU > this with a closer agent


def _projected_2d_box(corners_3d, cal):
    """Project 8 corners → axis-aligned 2D bbox [u_min, v_min, u_max, v_max]
    in camera image. Returns None if nothing is in front of camera."""
    R, t = cal["R_ego2cam"], cal["t_ego2cam"]
    pts_cam = (R @ corners_3d.T).T + t
    mask = pts_cam[:, 2] > 0.5
    if not np.any(mask):
        return None
    pts = pts_cam[mask]
    u = cal["fx"] * pts[:, 0] / pts[:, 2] + cal["cx"]
    v = cal["fy"] * pts[:, 1] / pts[:, 2] + cal["cy"]
    u = np.clip(u, 0, cal["width"])
    v = np.clip(v, 0, cal["height"])
    if (u.max() - u.min()) < 1 or (v.max() - v.min()) < 1:
        return None
    return [float(u.min()), float(v.min()), float(u.max()), float(v.max())]


def _iou_2d(a, b):
    """IOU of two [x1,y1,x2,y2] boxes."""
    xi = max(a[0], b[0]); yi = max(a[1], b[1])
    xa = min(a[2], b[2]); ya = min(a[3], b[3])
    inter = max(0, xa - xi) * max(0, ya - yi)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0.0


def filter_occluded_agents(agents, min_pts=100):
    """Drop agents with fewer than min_pts LiDAR points — likely occluded."""
    return [a for a in agents if a.get("num_pts", min_pts) >= min_pts]


def _ray_intersects_aabb(ray_dir, aabb_min, aabb_max):
    """Test if ray from origin (0,0,0) with direction ray_dir hits an AABB (slab method)."""
    t_min = -1e30
    t_max = 1e30
    for i in range(3):
        if abs(ray_dir[i]) < 1e-9:
            if aabb_min[i] > 0 or aabb_max[i] < 0:
                return False
        else:
            t1 = aabb_min[i] / ray_dir[i]
            t2 = aabb_max[i] / ray_dir[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
    return t_max > 0


def filter_occluded_agents_raycast(agents, calib, ray_thresh=0.80):
    """Drop agents where >=80% of rays through their 3D box are blocked
    by a closer VEHICLE in ALL cameras they appear in.

    Multi-ray: 8 corners + center = 9 rays per agent per camera.
    Only vehicles can block. Depth uses camera z.
    """
    if len(agents) <= 1:
        return agents

    n = len(agents)
    is_vehicle = [a.get("category", "").upper() in ("REGULAR_VEHICLE", "LARGE_VEHICLE",
                  "BUS", "BOX_TRUCK", "TRUCK", "VEHICULAR_TRAILER", "TRUCK_CAB",
                  "SCHOOL_BUS", "ARTICULATED_BUS", "MESSAGE_BOARD_TRAILER")
                  or "vehicle" in a.get("category", "").lower()
                  for a in agents]
    corners_list = [
        get_3d_box_corners(a["x"], a["y"], a["z"],
                           a["l"], a["w"], a["h"], a["yaw"])
        for a in agents
    ]

    visible_in = [set() for _ in range(n)]
    occluded_in = [set() for _ in range(n)]

    for av_cam, our_cam in CAM_MAP.items():
        if av_cam not in calib:
            continue
        cal = calib[av_cam]
        R = cal["R_ego2cam"]
        t = cal["t_ego2cam"]
        fx, fy = cal["fx"], cal["fy"]
        cx, cy = cal["cx"], cal["cy"]
        img_w, img_h = cal["width"], cal["height"]

        all_pts_cam = []
        centers_cam = []
        aabbs = []
        in_front = []
        for i in range(n):
            pts_cam = (R @ corners_list[i].T).T + t
            center = pts_cam.mean(axis=0)
            all_pts_cam.append(pts_cam)
            centers_cam.append(center)
            in_front.append(center[2] > 0.5)
            aabbs.append((pts_cam.min(axis=0), pts_cam.max(axis=0)))

        for i in range(n):
            if not in_front[i]:
                continue
            c = centers_cam[i]
            u = fx * c[0] / c[2] + cx
            v = fy * c[1] / c[2] + cy
            if not (0 <= u <= img_w and 0 <= v <= img_h):
                continue
            visible_in[i].add(av_cam)

            ray_points = np.vstack([all_pts_cam[i], centers_cam[i].reshape(1, 3)])
            blocked = 0
            total = len(ray_points)

            for ray_dir in ray_points:
                if np.linalg.norm(ray_dir) < 0.1:
                    continue
                target_z = ray_dir[2]
                for j in range(n):
                    if j == i or not in_front[j]:
                        continue
                    if not is_vehicle[j]:
                        continue
                    if centers_cam[j][2] >= target_z:
                        continue
                    if aabbs[j][1][2] < 0.5:
                        continue
                    if _ray_intersects_aabb(ray_dir, aabbs[j][0], aabbs[j][1]):
                        blocked += 1
                        break

            if total > 0 and blocked / total >= ray_thresh:
                occluded_in[i].add(av_cam)

    keep = []
    for i, ag in enumerate(agents):
        if not visible_in[i]:
            keep.append(ag)
            continue
        if occluded_in[i] and occluded_in[i] >= visible_in[i]:
            continue
        keep.append(ag)
    return keep


def merge_colocated_agents(agents):
    """
    Merge annotations that describe the same physical entity:
      - BICYCLIST + BICYCLE at the same position → keep BICYCLIST as "cyclist"
      - PEDESTRIAN + WHEELCHAIR at the same position → keep PEDESTRIAN
    Uses 1.5m distance threshold.
    """
    MERGE_DIST = 1.5  # metres
    # Priority: higher value = preferred annotation to keep
    PRIORITY = {"bicyclist": 3, "bicycle": 1, "pedestrian": 3,
                "wheelchair": 1, "motorcycle": 2, "motorcyclist": 3}

    merged = list(agents)
    drop_indices = set()

    for i in range(len(merged)):
        if i in drop_indices:
            continue
        ci = merged[i]["category"].lower()
        for j in range(i + 1, len(merged)):
            if j in drop_indices:
                continue
            cj = merged[j]["category"].lower()
            # Check if they are a known merge pair
            pair = {ci, cj}
            is_bike_pair = pair <= {"bicyclist", "bicycle"}
            is_ped_pair = pair <= {"pedestrian", "wheelchair"}
            is_moto_pair = pair <= {"motorcyclist", "motorcycle"}
            if not (is_bike_pair or is_ped_pair or is_moto_pair):
                continue
            # Check spatial proximity
            dx = merged[i]["x"] - merged[j]["x"]
            dy = merged[i]["y"] - merged[j]["y"]
            if math.sqrt(dx**2 + dy**2) < MERGE_DIST:
                # Drop the lower-priority one
                pi = PRIORITY.get(ci, 0)
                pj = PRIORITY.get(cj, 0)
                drop_indices.add(j if pi >= pj else i)

    return [a for idx, a in enumerate(merged) if idx not in drop_indices]


def filter_and_label_agents(agents_raw):
    """
    Filter and relabel agents:
    1. Drop SKIP_CATEGORIES (bollards, cones, signs, etc.)
    2. Merge co-located annotations (bicycle+bicyclist, pedestrian+wheelchair)
    3. Sort: moving agents first, then by distance
    4. Assign clean sequential labels: vehicle_1, pedestrian_2, cyclist_3, ...
    """
    agents_raw = [a for a in agents_raw if a["category"].lower() not in SKIP_CATEGORIES]
    agents_raw = merge_colocated_agents(agents_raw)

    for a in agents_raw:
        a["_dist"] = math.sqrt(a["x"]**2 + a["y"]**2)
    agents_raw.sort(key=lambda a: (a["speed"] < 0.5, a["_dist"]))

    counters = {}
    for a in agents_raw:
        short = category_short(a["category"])
        counters[short] = counters.get(short, 0) + 1
        a["label"] = f"{short}_{counters[short]}"
        del a["_dist"]

    return agents_raw


# ── Camera timestamp matching ─────────────────────────────────────────────
def get_camera_timestamps(scene_dir, cam_name):
    """Get sorted list of camera timestamps from image filenames."""
    cam_dir = scene_dir / "sensors" / "cameras" / cam_name
    if not cam_dir.exists():
        return []
    return sorted([int(f.stem) for f in cam_dir.glob("*.jpg")])


def find_closest_image(cam_timestamps, target_ns, max_diff_ms=100):
    """Find the closest camera image timestamp to the target."""
    if not cam_timestamps:
        return None
    idx = np.searchsorted(cam_timestamps, target_ns)
    candidates = []
    if idx > 0:
        candidates.append(cam_timestamps[idx - 1])
    if idx < len(cam_timestamps):
        candidates.append(cam_timestamps[idx])
    best = min(candidates, key=lambda t: abs(t - target_ns))
    if abs(best - target_ns) > max_diff_ms * 1e6:
        return None
    return best


# ── Camera calibration loading ────────────────────────────────────────────
def load_calibration(scene_dir):
    """
    Load camera intrinsics and extrinsics.
    Returns dict: cam_name → {fx, fy, cx, cy, width, height, R_ego2cam, t_ego2cam}
    """
    cal_dir = scene_dir / "calibration"
    intr_df = pd.read_feather(cal_dir / "intrinsics.feather")
    extr_df = pd.read_feather(cal_dir / "egovehicle_SE3_sensor.feather")

    calib = {}
    for _, row in intr_df.iterrows():
        name = row.sensor_name
        if name not in CAM_MAP:
            continue
        # Get extrinsics for this sensor
        extr_row = extr_df[extr_df.sensor_name == name].iloc[0]
        R_ego2cam = quat_to_rotation_matrix(
            extr_row.qw, extr_row.qx, extr_row.qy, extr_row.qz
        )
        t_ego2cam = np.array([extr_row.tx_m, extr_row.ty_m, extr_row.tz_m])

        # egovehicle_SE3_sensor means: sensor_pos_in_ego = R @ sensor_origin + t
        # So ego → sensor: sensor_pt = R^T @ (ego_pt - t)
        R_inv = R_ego2cam.T
        t_inv = -R_inv @ t_ego2cam

        calib[name] = {
            "fx": row.fx_px, "fy": row.fy_px,
            "cx": row.cx_px, "cy": row.cy_px,
            "width": int(row.width_px), "height": int(row.height_px),
            "R_ego2cam": R_inv,   # ego → camera rotation
            "t_ego2cam": t_inv,   # ego → camera translation
        }
    return calib


# ── Camera projection & bbox drawing ─────────────────────────────────────
def project_boxes_to_cam(agents_ego, cam_calib):
    """
    Project ego-frame 3D boxes → 2D bboxes in a camera image.
    Returns list of (x1, y1, x2, y2, label, color) for visible agents.
    """
    R = cam_calib["R_ego2cam"]
    t = cam_calib["t_ego2cam"]
    fx, fy = cam_calib["fx"], cam_calib["fy"]
    cx, cy = cam_calib["cx"], cam_calib["cy"]
    img_w, img_h = cam_calib["width"], cam_calib["height"]

    results = []
    for ag in agents_ego:
        corners = get_3d_box_corners(
            ag["x"], ag["y"], ag["z"],
            ag["l"], ag["w"], ag["h"], ag["yaw"]
        )
        # Transform corners: ego → camera frame
        pts_cam = (R @ corners.T).T + t  # (8, 3)

        # Need some points in front of camera
        mask = pts_cam[:, 2] > 0.1
        if not np.any(mask):
            continue

        pts_front = pts_cam[mask]
        u = fx * pts_front[:, 0] / pts_front[:, 2] + cx
        v = fy * pts_front[:, 1] / pts_front[:, 2] + cy

        # Raw (unclipped) bbox
        x1_raw, y1_raw = float(u.min()), float(v.min())
        x2_raw, y2_raw = float(u.max()), float(v.max())
        raw_area = (x2_raw - x1_raw) * (y2_raw - y1_raw)
        if raw_area < 1:
            continue

        # Any corner on image?
        any_corner_in = np.any(
            (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h))
        if not any_corner_in:
            continue

        # Clip to image bounds
        x1 = int(np.clip(x1_raw, 0, img_w - 1))
        y1 = int(np.clip(y1_raw, 0, img_h - 1))
        x2 = int(np.clip(x2_raw, 0, img_w - 1))
        y2 = int(np.clip(y2_raw, 0, img_h - 1))
        # Skip if too small on screen
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue

        results.append((x1, y1, x2, y2, ag.get("label", ""),
                         _agent_color(ag["category"])))

    return results


def draw_cam_bboxes(img, bboxes, scale_x=1.0, scale_y=1.0):
    """Draw 2D bboxes on a camera image."""
    for x1, y1, x2, y2, label, color in bboxes:
        x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
        x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            ty = max(y1 - 4, th + 4)
            cv2.rectangle(img, (x1 - 1, ty - th - 3), (x1 + tw + 3, ty + 2), (0, 0, 0), -1)
            cv2.putText(img, label, (x1, ty), font, 0.5, color, 1)
    return img


# ── Lane ID assignment ────────────────────────────────────────────────────
def _assign_lane_ids(map_data, ego_x0, ego_y0, ego_yaw0, ego_local_pts):
    """Assign lane IDs to positions given in ego-local frame.

    Unlike OpenScene/nuScenes, AV2 intersection lane_segments are individual
    lane-width paths with their own boundaries — so they work as-is for lane
    comparison. All segments get L{i} tags directly.

    ego_local_pts: list of (x, y) in ego-local coords (ego at origin, +x fwd)
    Returns: list of lane_id strings (or None).
    """
    if map_data is None:
        return [None] * len(ego_local_pts)

    cos_e, sin_e = math.cos(ego_yaw0), math.sin(ego_yaw0)

    # Convert ego-local → city coords
    city_pts = []
    for ex, ey in ego_local_pts:
        cx = ego_x0 + cos_e * ex - sin_e * ey
        cy = ego_y0 + sin_e * ex + cos_e * ey
        city_pts.append((float(cx), float(cy)))

    # Build lane polygon contours once (city coords)
    lane_polys = []  # (index, contour)
    for i, (ls_id, ls) in enumerate(map_data.get("lane_segments", {}).items()):
        left_pts = [(p["x"], p["y"]) for p in ls["left_lane_boundary"]]
        right_pts = [(p["x"], p["y"]) for p in ls["right_lane_boundary"]]
        poly_pts = left_pts + list(reversed(right_pts))
        if len(poly_pts) < 3:
            continue
        contour = np.array(poly_pts, dtype=np.float32)
        lane_polys.append((i, contour))

    # Test each point
    result = []
    for pt in city_pts:
        lane_id = None
        for idx, contour in lane_polys:
            if cv2.pointPolygonTest(contour, pt, False) >= 0:
                lane_id = f"L{idx}"
                break
        result.append(lane_id)
    return result


# ── BEV Rendering ─────────────────────────────────────────────────────────
def render_bev(agents_ego, map_data=None, to_ego_fn=None,
               agent_radius=RADIUS, bev_range=BEV_RANGE, ego_speed=0.0,
               ego_pos=(0, 0), timestamp_label="T=0"):
    """Render ego-centric BEV with map layers and agents."""
    S = BEV_SUPERSAMPLE
    SZ = BEV_SIZE * S
    canvas = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    cx, cy = SZ // 2, SZ // 2

    def ego_to_bev(ex, ey):
        bx = int(cx - ey * SZ / (2 * bev_range))
        by = int(cy - ex * SZ / (2 * bev_range))
        return bx, by

    # Draw map layers
    if map_data is not None and to_ego_fn is not None:
        _draw_map_layers(canvas, map_data, to_ego_fn, ego_to_bev, SZ, S, bev_range)

    # ── Grid rings and crosshair removed for cleaner BEV

    def draw_box(x, y, l, w, yaw, color, thickness=1, fill=False, label_text=None, speed=0.0):
        cos_a, sin_a = math.cos(yaw), math.sin(yaw)
        l2, w2 = l / 2, w / 2
        corners = [
            (x + cos_a*l2 - sin_a*w2, y + sin_a*l2 + cos_a*w2),
            (x + cos_a*l2 + sin_a*w2, y + sin_a*l2 - cos_a*w2),
            (x - cos_a*l2 + sin_a*w2, y - sin_a*l2 - cos_a*w2),
            (x - cos_a*l2 - sin_a*w2, y - sin_a*l2 + cos_a*w2),
        ]
        bev_corners = np.array([ego_to_bev(cx_, cy_) for cx_, cy_ in corners], dtype=np.int32)
        th = thickness * S
        if fill:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [bev_corners.reshape(-1, 1, 2)], color)
            cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
        cv2.polylines(canvas, [bev_corners.reshape(-1, 1, 2)], True, color, th)
        front_pts = np.array([ego_to_bev(*corners[0]), ego_to_bev(*corners[1])], dtype=np.int32)
        cv2.polylines(canvas, [front_pts.reshape(-1, 1, 2)], False,
                      tuple(min(255, c + 80) for c in color), th + S)
        # Heading arrow from front edge outward (only for moving agents)
        if speed >= 0.5:
            front_cx = int((bev_corners[0][0] + bev_corners[1][0]) / 2)
            front_cy = int((bev_corners[0][1] + bev_corners[1][1]) / 2)
            back_cx = int((bev_corners[2][0] + bev_corners[3][0]) / 2)
            back_cy = int((bev_corners[2][1] + bev_corners[3][1]) / 2)
            dx_h = front_cx - back_cx
            dy_h = front_cy - back_cy
            h_len = max(1, math.sqrt(dx_h**2 + dy_h**2))
            arrow_len = 18 * S
            ax = int(front_cx + dx_h / h_len * arrow_len)
            ay = int(front_cy + dy_h / h_len * arrow_len)
            arrow_color = tuple(min(255, c + 60) for c in color)
            cv2.arrowedLine(canvas, (front_cx, front_cy), (ax, ay),
                            arrow_color, 2 * S, tipLength=0.35)
        if label_text:
            bcx = int(np.mean(bev_corners[:, 0]))
            bcy = int(np.min(bev_corners[:, 1])) - 4 * S
            cv2.putText(canvas, label_text, (bcx - 4*S, bcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35 * S, color, S)

    # Save map-only canvas before agents
    map_only_canvas = canvas.copy()

    # Ego box (at ego_pos — (0,0) for T=0, past position for temporal frames)
    draw_box(ego_pos[0], ego_pos[1], 4.5, 2.0, 0, COLOR_EGO, 2, fill=True, label_text="EGO", speed=ego_speed)

    # Agent boxes
    for ag in agents_ego:
        dist = math.sqrt(ag["x"]**2 + ag["y"]**2)
        if dist > agent_radius:
            continue
        color = _agent_color(ag["category"])
        draw_box(ag["x"], ag["y"], ag["l"], ag["w"], ag["yaw"],
                 color, 1, label_text=ag.get("label", ""), speed=ag.get("speed", 0.0))

    # Labels
    cv2.putText(canvas, timestamp_label, (8*S, 22*S),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5*S, (220, 220, 220), S)
    # Agent legend (bottom-left)
    for i, (c, txt) in enumerate([(COLOR_EGO, "Ego"), (COLOR_VEHICLE, "Vehicles"),
                                   (COLOR_PED, "Peds"), (COLOR_CYCLE, "Cyclist")]):
        y0 = SZ - 12*S - i*14*S
        cv2.rectangle(canvas, (6*S, y0-8*S), (16*S, y0+2*S), c, -1)
        cv2.putText(canvas, txt, (20*S, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35*S, (200, 200, 200), S)

    # Map legend (bottom-right)
    map_legend = [
        (MAP_COLORS["ped_crossing"],  "Ped Crossing"),
        (MAP_COLORS["lane_boundary"], "Lane Boundary"),
        ((40, 90, 40),                "Walkway"),
        (MAP_COLORS["lane"],          "Lane"),
        (MAP_COLORS["drivable"],      "Drivable"),
    ]
    for i, (c, txt) in enumerate(map_legend):
        y0 = SZ - 12*S - i*14*S
        xo = SZ - 100*S
        cv2.rectangle(canvas, (xo, y0-8*S), (xo+10*S, y0+2*S), c, -1)
        cv2.putText(canvas, txt, (xo+14*S, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30*S, (200, 200, 200), S)

    canvas = cv2.resize(canvas, (BEV_SIZE, BEV_SIZE), interpolation=cv2.INTER_AREA)
    map_only = cv2.resize(map_only_canvas, (BEV_SIZE, BEV_SIZE), interpolation=cv2.INTER_AREA)
    return canvas, map_only


def _draw_map_layers(canvas, map_data, to_ego_fn, ego_to_bev, SZ, S, bev_range):
    """Draw drivable area, lane boundaries, and crosswalks on BEV canvas."""
    clip = bev_range * 1.2

    # 1. Drivable areas (filled polygons) — include all vertices, let OpenCV clip
    for da_id, da in map_data.get("drivable_areas", {}).items():
        pts = []
        for p in da["area_boundary"]:
            ex, ey = to_ego_fn(p["x"], p["y"])
            pts.append(ego_to_bev(ex, ey))
        if len(pts) > 2:
            arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(canvas, [arr.reshape(-1, 1, 2)], MAP_COLORS["drivable"])

    # 2. Lane segments — fill lane polygons + draw boundaries
    for ls_id, ls in map_data.get("lane_segments", {}).items():
        left_pts = [(p["x"], p["y"]) for p in ls["left_lane_boundary"]]
        right_pts = [(p["x"], p["y"]) for p in ls["right_lane_boundary"]]
        poly_pts = left_pts + list(reversed(right_pts))
        bev_poly = [ego_to_bev(*to_ego_fn(px, py)) for px, py in poly_pts]
        if len(bev_poly) > 2:
            arr = np.array(bev_poly, dtype=np.int32)
            is_intersection = ls.get("is_intersection", False)
            color = MAP_COLORS["lane"]  # intersections use same color as lanes
            cv2.fillPoly(canvas, [arr.reshape(-1, 1, 2)], color)

        # Draw lane boundaries as lines
        for side in ["left_lane_boundary", "right_lane_boundary"]:
            pts_bev = [ego_to_bev(*to_ego_fn(p["x"], p["y"])) for p in ls[side]]
            if len(pts_bev) > 1:
                arr = np.array(pts_bev, dtype=np.int32)
                cv2.polylines(canvas, [arr.reshape(-1, 1, 2)], False,
                              MAP_COLORS["lane_boundary"], 2 * S, lineType=cv2.LINE_AA)

    # 2b. Lane direction arrows along centerline (including intersections)
    _ARR_SPACE = 24 * S
    _ARR_LEN = 7 * S
    _ARR_CLR = (200, 200, 200)
    for ls_id, ls in map_data.get("lane_segments", {}).items():
        left_b = ls["left_lane_boundary"]
        right_b = ls["right_lane_boundary"]
        if len(left_b) < 2 or len(right_b) < 2:
            continue
        # Build centerline by interpolating both boundaries to equal lengths
        from numpy import linspace
        nl, nr = len(left_b), len(right_b)
        n_pts = max(nl, nr)
        lt = linspace(0, 1, nl)
        rt = linspace(0, 1, nr)
        ct = linspace(0, 1, n_pts)
        lx = np.interp(ct, lt, [p["x"] for p in left_b])
        ly = np.interp(ct, lt, [p["y"] for p in left_b])
        rx = np.interp(ct, rt, [p["x"] for p in right_b])
        ry = np.interp(ct, rt, [p["y"] for p in right_b])
        ctr_world = [((lx[i]+rx[i])/2, (ly[i]+ry[i])/2) for i in range(n_pts)]
        dense = []
        for i in range(len(ctr_world) - 1):
            x0, y0 = ctr_world[i]
            x1, y1 = ctr_world[i + 1]
            seg = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            n_sub = max(1, int(seg / 1.0))
            for j in range(n_sub):
                t = j / n_sub
                dense.append((x0 + t*(x1-x0), y0 + t*(y1-y0)))
        dense.append(ctr_world[-1])
        ctr = [ego_to_bev(*to_ego_fn(wx, wy)) for wx, wy in dense]
        _draw_arrows_along(canvas, ctr, _ARR_SPACE, _ARR_LEN, _ARR_CLR, S)

    # 3. Pedestrian crossings (filled polygon from edge1 + edge2)
    for pc_id, pc in map_data.get("pedestrian_crossings", {}).items():
        e1 = [ego_to_bev(*to_ego_fn(p["x"], p["y"])) for p in pc["edge1"]]
        e2 = [ego_to_bev(*to_ego_fn(p["x"], p["y"])) for p in pc["edge2"]]
        bev_pts = e1 + list(reversed(e2))
        if len(bev_pts) > 2:
            arr = np.array(bev_pts, dtype=np.int32)
            cv2.fillPoly(canvas, [arr.reshape(-1, 1, 2)], MAP_COLORS["ped_crossing"])
            cv2.polylines(canvas, [arr.reshape(-1, 1, 2)], True, (0, 100, 200), S)


# ── Main builder ──────────────────────────────────────────────────────────
def build_scene(scene_dir, out_dir, t0_idx=None, sample_idx=0,
                history_frames=3, future_frames=8,
                _cache=None):
    """Build scenario records for one Argoverse V2 scene."""
    scene_uuid = scene_dir.name

    # ── Load data (use cache if provided to avoid reloading per sample) ──
    if _cache is not None:
        ann_df, ego_df, map_data, calib, ann_2hz = _cache
    else:
        ann_df = pd.read_feather(scene_dir / "annotations.feather")
        ego_df = pd.read_feather(scene_dir / "city_SE3_egovehicle.feather")
        ego_df = ego_df.sort_values("timestamp_ns").reset_index(drop=True)

        map_data = None
        for f in (scene_dir / "map").glob("log_map_archive_*.json"):
            with open(f) as fp:
                map_data = json.load(fp)
            break

        calib = load_calibration(scene_dir)

        ann_timestamps = sorted(ann_df.timestamp_ns.unique())
        ann_2hz = ann_timestamps[::5]  # 10Hz → 2Hz

    min_frames = history_frames + future_frames + 1
    if len(ann_2hz) < min_frames:
        return False

    if t0_idx is None:
        t0_idx = min(len(ann_2hz) - future_frames - 1, max(history_frames, len(ann_2hz) // 2))
    if t0_idx < history_frames or t0_idx >= len(ann_2hz) - future_frames:
        return False
    t0_ns = ann_2hz[t0_idx]

    # ── Camera timestamps ──
    cam_timestamps = {av_cam: get_camera_timestamps(scene_dir, av_cam) for av_cam in CAM_MAP}
    missing_cams = [c for c in CAM_MAP if not cam_timestamps[c]]
    if missing_cams:
        return False

    # ── Ego pose at T=0 ──
    ego_t0_row = ego_df.iloc[(ego_df.timestamp_ns - t0_ns).abs().argsort().iloc[0]]
    ego_x0, ego_y0 = ego_t0_row.tx_m, ego_t0_row.ty_m
    ego_yaw0 = quat_to_yaw(ego_t0_row.qw, ego_t0_row.qx,
                            ego_t0_row.qy, ego_t0_row.qz)
    to_ego, norm_yaw = make_ego_local_transform(ego_x0, ego_y0, ego_yaw0)

    # ── Scene output ──
    scene_name = f"argoverse2-{scene_uuid[:16]}"
    scene_out_dir = out_dir / scene_name / f"SAMPLED_{sample_idx}"
    scene_out_dir.mkdir(parents=True, exist_ok=True)
    nav_command = infer_nav_command_from_trajectory(ego_df, t0_ns)

    # ── Build agents at T=0 first (needed for consistent labels) ──
    # Get previous frame for speed estimation
    prev_positions = {}
    if t0_idx > 0:
        prev_ts = ann_2hz[t0_idx - 1]
        ann_prev = ann_df[ann_df.timestamp_ns == prev_ts]
        dt_prev = (t0_ns - prev_ts) / 1e9
        for _, prow in ann_prev.iterrows():
            prev_positions[prow.track_uuid] = (float(prow.tx_m), float(prow.ty_m), dt_prev)

    # Scan all 4 temporal frames: agent eligible if passes radius + num_pts in ANY frame
    eligible_tracks = set()
    cam_eligible = {}  # (track_uuid, ts_ns) → bool — for camera bbox gating
    for offset_e, _ in TEMPORAL_OFFSETS:
        fidx = t0_idx + offset_e
        if fidx < 0 or fidx >= len(ann_2hz):
            continue
        fts = ann_2hz[fidx]
        ann_frame = ann_df[ann_df.timestamp_ns == fts]
        for _, row in ann_frame.iterrows():
            x, y = float(row.tx_m), float(row.ty_m)
            is_ped = "pedestrian" in row.category.lower()
            r = RADIUS_PED if is_ped else RADIUS
            in_radius = math.sqrt(x**2 + y**2) <= r
            has_pts = int(row.num_interior_pts) >= 100
            passes = in_radius and has_pts
            cam_eligible[(row.track_uuid, fts)] = passes
            if passes:
                eligible_tracks.add(row.track_uuid)

    # Build T=0 agents from eligible set
    ann_t0 = ann_df[ann_df.timestamp_ns == t0_ns]
    agents_t0 = []
    for _, row in ann_t0.iterrows():
        if row.track_uuid not in eligible_tracks:
            continue
        x, y, z = float(row.tx_m), float(row.ty_m), float(row.tz_m)
        l, w, h = float(row.length_m), float(row.width_m), float(row.height_m)
        yaw = quat_to_yaw(row.qw, row.qx, row.qy, row.qz)
        speed = 0.0
        if row.track_uuid in prev_positions:
            px, py, dt = prev_positions[row.track_uuid]
            if dt > 0.01:
                speed = round(math.sqrt((x - px)**2 + (y - py)**2) / dt, 3)
        agents_t0.append({
            "x": round(x, 4), "y": round(y, 4), "z": round(z, 4),
            "l": round(l, 4), "w": round(w, 4), "h": round(h, 4),
            "yaw": round(yaw, 4), "category": row.category,
            "track_uuid": row.track_uuid, "num_pts": int(row.num_interior_pts),
            "speed": speed,
            "label": "",
        })

    # Keep only agents visible in at least one camera FOV
    agents_t0 = agents_visible_in_any_fov(agents_t0, calib)

    # Filter 1: Drop agents with too few LiDAR points
    agents_t0 = filter_occluded_agents(agents_t0)

    # Raycast disabled for AV2 — num_interior_pts already handles occlusion

    # Sort and relabel
    agents_t0 = filter_and_label_agents(agents_t0)
    label_by_track = {a["track_uuid"]: a["label"] for a in agents_t0}

    # ── Save camera images WITH bounding boxes for 4 temporal frames ──
    for offset, slug in TEMPORAL_OFFSETS:
        ts_idx = t0_idx + offset
        if ts_idx < 0 or ts_idx >= len(ann_2hz):
            continue
        ts_ns = ann_2hz[ts_idx]
        slug_dir = scene_out_dir / slug
        slug_dir.mkdir(parents=True, exist_ok=True)

        # Build agents at this frame — camera bbox only if eligible in THIS frame
        ann_frame = ann_df[ann_df.timestamp_ns == ts_ns]
        f_agents = []
        for _, row in ann_frame.iterrows():
            tk = row.track_uuid
            if tk not in label_by_track:
                continue
            # Camera bbox: only if passes radius + pts in this specific frame
            if not cam_eligible.get((tk, ts_ns), False):
                continue
            x, y, z = float(row.tx_m), float(row.ty_m), float(row.tz_m)
            if math.sqrt(x**2 + y**2) > RADIUS * 1.5:
                continue
            f_agents.append({
                "x": x, "y": y, "z": z,
                "l": float(row.length_m), "w": float(row.width_m), "h": float(row.height_m),
                "yaw": quat_to_yaw(row.qw, row.qx, row.qy, row.qz),
                "category": row.category,
                "label": label_by_track[tk],
            })

        for av_cam, our_cam in CAM_MAP.items():
            img_ts = find_closest_image(cam_timestamps[av_cam], ts_ns)
            if img_ts is None:
                continue
            src_path = scene_dir / "sensors" / "cameras" / av_cam / f"{img_ts}.jpg"
            if not src_path.exists():
                continue

            img = cv2.imread(str(src_path))
            if img is None:
                continue

            # Draw bboxes only for agents visible in THIS frame's camera
            if av_cam in calib:
                bboxes = project_boxes_to_cam(f_agents, calib[av_cam])
                draw_cam_bboxes(img, bboxes)

            cv2.imwrite(str(slug_dir / f"{our_cam}.jpg"), img)

    # ── Render BEV (T=0 + temporal past frames) ──
    tp0_dir = scene_out_dir / "Tp0p0"
    tp0_dir.mkdir(parents=True, exist_ok=True)
    ego_speed = _compute_ego_speed(ego_df, t0_ns)
    bev, bev_map = render_bev(agents_t0, map_data=map_data, to_ego_fn=to_ego, agent_radius=RADIUS, ego_speed=ego_speed)
    cv2.imwrite(str(tp0_dir / "bev.png"), bev)
    cv2.imwrite(str(tp0_dir / "bev_map.png"), bev_map)

    # Temporal BEVs: agents + ego at past positions, T=0 ego frame
    for offset, slug, ts_label in [(-3, "Tm1p5", "T=-1.5s"), (-2, "Tm1p0", "T=-1.0s"), (-1, "Tm0p5", "T=-0.5s")]:
        ts_idx = t0_idx + offset
        if ts_idx < 0:
            continue
        ts_ns = ann_2hz[ts_idx]
        # Ego past position in T=0 ego-local frame
        ego_row = ego_df.iloc[(ego_df.timestamp_ns - ts_ns).abs().argsort().iloc[0]]
        ego_ex, ego_ey = to_ego(ego_row.tx_m, ego_row.ty_m)
        past_ego_speed = _compute_ego_speed(ego_df, ts_ns)
        # Past agents — convert past-ego-local → world → T=0-ego-local
        ann_frame = ann_df[ann_df.timestamp_ns == ts_ns]
        # Past ego pose for coordinate transform: past-ego-local → world → T=0-ego-local
        past_ego_row = ego_df.iloc[(ego_df.timestamp_ns - ts_ns).abs().argsort().iloc[0]]
        past_R = Rotation.from_quat([past_ego_row.qx, past_ego_row.qy,
                                     past_ego_row.qz, past_ego_row.qw]).as_matrix()
        past_t = np.array([past_ego_row.tx_m, past_ego_row.ty_m, past_ego_row.tz_m])
        past_ego_yaw = quat_to_yaw(past_ego_row.qw, past_ego_row.qx,
                                    past_ego_row.qy, past_ego_row.qz)
        def past_to_world(lx, ly):
            """Convert past-frame ego-local → world."""
            pt = past_R @ np.array([lx, ly, 0]) + past_t
            return float(pt[0]), float(pt[1])
        def past_yaw_to_t0(local_yaw):
            """Convert yaw from past-ego frame → T=0 ego frame."""
            world_yaw = local_yaw + past_ego_yaw
            return (world_yaw - ego_yaw0 + math.pi) % (2 * math.pi) - math.pi

        # Speed from consecutive frames (prev frame also needs its own ego→world transform)
        prev_pos_map = {}
        prev_idx = t0_idx + offset - 1
        if prev_idx >= 0:
            prev_ns = ann_2hz[prev_idx]
            prev_ego_row = ego_df.iloc[(ego_df.timestamp_ns - prev_ns).abs().argsort().iloc[0]]
            prev_R = Rotation.from_quat([prev_ego_row.qx, prev_ego_row.qy,
                                         prev_ego_row.qz, prev_ego_row.qw]).as_matrix()
            prev_t = np.array([prev_ego_row.tx_m, prev_ego_row.ty_m, prev_ego_row.tz_m])
            ann_prev_frame = ann_df[ann_df.timestamp_ns == prev_ns]
            dt_speed = (ts_ns - prev_ns) / 1e9
            for _, prow in ann_prev_frame.iterrows():
                pt = prev_R @ np.array([float(prow.tx_m), float(prow.ty_m), 0]) + prev_t
                prev_pos_map[prow.track_uuid] = (float(pt[0]), float(pt[1]), dt_speed)

        past_agents = []
        for _, row in ann_frame.iterrows():
            tk = row.track_uuid
            if tk not in label_by_track:
                continue
            # Past-ego-local → world → T=0-ego-local
            wx, wy = past_to_world(float(row.tx_m), float(row.ty_m))
            ex, ey = to_ego(wx, wy)
            dist = math.sqrt(ex**2 + ey**2)
            if dist > RADIUS * 1.5:
                continue
            yaw_local = past_yaw_to_t0(quat_to_yaw(row.qw, row.qx, row.qy, row.qz))
            # Speed from world coords
            spd = 0.0
            if tk in prev_pos_map:
                px, py, dt_s = prev_pos_map[tk]
                if dt_s > 0.01:
                    spd = round(math.sqrt((wx - px)**2 + (wy - py)**2) / dt_s, 3)
            past_agents.append({
                "label": label_by_track[tk], "category": row.category,
                "x": ex, "y": ey, "yaw": yaw_local,
                "l": float(row.length_m), "w": float(row.width_m),
                "speed": spd,
            })
        slug_dir = scene_out_dir / slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        past_bev, past_bev_map = render_bev(past_agents, map_data=map_data, to_ego_fn=to_ego,
                              agent_radius=RADIUS * 1.5, ego_speed=past_ego_speed,
                              ego_pos=(ego_ex, ego_ey), timestamp_label=ts_label)
        cv2.imwrite(str(slug_dir / "bev.png"), past_bev)
        cv2.imwrite(str(slug_dir / "bev_map.png"), past_bev_map)

    ego_history = []
    for offset in [-3, -2, -1]:
        ts_idx = t0_idx + offset
        if ts_idx < 0:
            continue
        ts_ns = ann_2hz[ts_idx]
        ego_row = ego_df.iloc[(ego_df.timestamp_ns - ts_ns).abs().argsort().iloc[0]]
        ex, ey = to_ego(ego_row.tx_m, ego_row.ty_m)
        ego_history.append({"t": offset*0.5, "x": ex, "y": ey,
                            "speed": _compute_ego_speed(ego_df, ts_ns)})

    ego_future = []
    for offset in range(1, future_frames + 1):
        ts_idx = t0_idx + offset
        if ts_idx >= len(ann_2hz):
            continue
        ts_ns = ann_2hz[ts_idx]
        ego_row = ego_df.iloc[(ego_df.timestamp_ns - ts_ns).abs().argsort().iloc[0]]
        ex, ey = to_ego(ego_row.tx_m, ego_row.ty_m)
        ego_future.append({"t": offset*0.5, "x": ex, "y": ey,
                           "speed": _compute_ego_speed(ego_df, ts_ns)})

    # Agent history
    t0_track_set = {ag["track_uuid"] for ag in agents_t0}
    agent_hist = defaultdict(list)
    for offset in [-3, -2, -1]:
        ts_idx = t0_idx + offset
        if ts_idx < 0:
            continue
        ts_ns = ann_2hz[ts_idx]
        ann_hist = ann_df[ann_df.timestamp_ns == ts_ns]
        ego_row_h = ego_df.iloc[(ego_df.timestamp_ns - ts_ns).abs().argsort().iloc[0]]
        ego_R_h = quat_to_rotation_matrix(ego_row_h.qw, ego_row_h.qx,
                                           ego_row_h.qy, ego_row_h.qz)
        ego_t_h = np.array([ego_row_h.tx_m, ego_row_h.ty_m, ego_row_h.tz_m])

        for _, row in ann_hist.iterrows():
            tk = row.track_uuid
            if tk not in t0_track_set:
                continue
            agent_ego_h = np.array([row.tx_m, row.ty_m, row.tz_m])
            agent_city = ego_R_h @ agent_ego_h + ego_t_h
            ex, ey = to_ego(agent_city[0], agent_city[1])
            agent_hist[tk].append({"t": offset*0.5, "x": ex, "y": ey, "speed": 0.0})

    # Compute agent speeds from history
    for tk in agent_hist:
        hist = sorted(agent_hist[tk], key=lambda h: h["t"])
        for i in range(1, len(hist)):
            dt = hist[i]["t"] - hist[i-1]["t"]
            if dt > 0:
                dx = hist[i]["x"] - hist[i-1]["x"]
                dy = hist[i]["y"] - hist[i-1]["y"]
                hist[i]["speed"] = round(math.sqrt(dx**2 + dy**2) / dt, 3)
        agent_hist[tk] = hist

    # Assign lane IDs from map (ego-local → city → polygon test)
    agents_out = {}
    for idx, ag in enumerate(agents_t0):
        tk = ag["track_uuid"]
        a_yaw_norm = (ag["yaw"] + math.pi) % (2 * math.pi) - math.pi
        entry = {
            "category": ag["category"],
            "position": {"x": ag["x"], "y": ag["y"], "yaw": round(a_yaw_norm, 4)},
            "speed": _estimate_agent_speed(ag, agent_hist.get(tk, [])),
        }
        hist = sorted(agent_hist.get(tk, []), key=lambda h: h["t"])
        if hist:
            entry["history"] = hist
        agents_out[ag["label"]] = entry

    state = {
        "t_rel_s": 0.0, "nav_command": nav_command,
        "ego": {"speed": ego_speed,
                "history": ego_history, "future": ego_future},
        "agents": agents_out,
    }
    with open(tp0_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2)

    # ── state_desc.json (natural language version of state for LLM prompts) ──
    from common.scene_utils import state_to_natural_language
    with open(tp0_dir / "state_desc.json", "w") as f:
        json.dump({"description": state_to_natural_language(state)}, f, indent=2)

    # ── uuid_map.json (label → track_uuid) ──
    uuid_map = {label: track_uuid for track_uuid, label in label_by_track.items()}
    with open(tp0_dir / "uuid_map.json", "w") as f:
        json.dump(uuid_map, f, indent=2)

    # ── Save camera calibration for downstream projection (counterfactual pipeline)
    # Intrinsics at native resolution (no scaling)
    calib_out = {}
    for av2_cam, std_cam in CAM_MAP.items():
        if av2_cam in calib:
            cc = calib[av2_cam]
            K = np.array([[cc["fx"], 0, cc["cx"]],
                          [0, cc["fy"], cc["cy"]],
                          [0, 0, 1]])
            calib_out[std_cam] = {
                "intrinsic": K.tolist(),
                "R_ego2cam": cc["R_ego2cam"].tolist(),
                "t_ego2cam": cc["t_ego2cam"].tolist(),
                "image_size": [cc["width"], cc["height"]],
            }
    with open(tp0_dir / "calib.json", "w") as f:
        json.dump(calib_out, f, indent=2)

    meta = {
        "scene": scene_name, "maneuver_type": f"SAMPLED_{sample_idx}", "nav_command": nav_command,
        "dataset": "argoverse2", "scene_uuid": scene_uuid, "t0_frame_idx": t0_idx, "sample_idx": sample_idx,
        "t0_timestamp_ns": int(t0_ns), "n_agents": len(agents_out),
        "raw_data_dir": str(scene_dir),
    }
    with open(scene_out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return True


def _compute_ego_speed(ego_df, t_ns, window_ms=100):
    """Compute ego speed from finite differences."""
    t_start = t_ns - int(window_ms * 1e6)
    t_end = t_ns + int(window_ms * 1e6)
    window = ego_df[(ego_df.timestamp_ns >= t_start) & (ego_df.timestamp_ns <= t_end)]
    if len(window) < 2:
        return 0.0
    first, last = window.iloc[0], window.iloc[-1]
    dt = (last.timestamp_ns - first.timestamp_ns) / 1e9
    if dt < 0.01:
        return 0.0
    return round(math.sqrt((last.tx_m - first.tx_m)**2 + (last.ty_m - first.ty_m)**2) / dt, 3)


def _estimate_agent_speed(agent_t0, hist):
    """Estimate agent speed at T=0 from history."""
    if not hist:
        return 0.0
    last_h = max(hist, key=lambda h: h["t"])
    dt = 0.0 - last_h["t"]
    if dt < 0.1:
        return last_h.get("speed", 0.0)
    dx = agent_t0["x"] - last_h["x"]
    dy = agent_t0["y"] - last_h["y"]
    return round(math.sqrt(dx**2 + dy**2) / dt, 3)


def _process_one_scene(scene_dir, out_dir, history_frames, future_frames,
                       sampling_step, max_samples, samples_filter, verbose=False):
    """Process all samples for one scene. Returns (n_samples, scene_name)."""
    scene_name = f"argoverse2-{scene_dir.name[:16]}"
    if verbose:
        print(f"Processing {scene_name}...")

    ann_df = pd.read_feather(scene_dir / "annotations.feather")
    ego_df = pd.read_feather(scene_dir / "city_SE3_egovehicle.feather")
    ego_df = ego_df.sort_values("timestamp_ns").reset_index(drop=True)
    map_data = None
    for f in (scene_dir / "map").glob("log_map_archive_*.json"):
        with open(f) as fp:
            map_data = json.load(fp)
        break
    calib = load_calibration(scene_dir)
    ann_timestamps = sorted(ann_df.timestamp_ns.unique())
    ann_2hz = ann_timestamps[::5]

    min_frames = history_frames + future_frames + 1
    if len(ann_2hz) < min_frames:
        return 0, scene_name

    t0_start = history_frames
    t0_end = len(ann_2hz) - future_frames
    if max_samples and max_samples > 0:
        valid_range = t0_end - t0_start
        if valid_range <= 0:
            return 0, scene_name
        n_samp = min(max_samples, valid_range)
        t0_indices = [t0_start + int(valid_range * (i + 1) / (n_samp + 1))
                      for i in range(n_samp)]
    else:
        t0_indices = list(range(t0_start, t0_end, sampling_step))

    cache = (ann_df, ego_df, map_data, calib, ann_2hz)
    wanted_indices = samples_filter.get(scene_name) if samples_filter else None
    n_samples = 0
    for sample_idx, t0_idx in enumerate(t0_indices):
        if wanted_indices is not None and sample_idx not in wanted_indices:
            continue
        if build_scene(scene_dir, out_dir, t0_idx=t0_idx, sample_idx=sample_idx,
                      history_frames=history_frames,
                      future_frames=future_frames, _cache=cache):
            n_samples += 1
    if verbose and n_samples:
        print(f"    → {n_samples} samples from {scene_name}")
    return n_samples, scene_name


def main():
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config_loader import load_config
    cfg = load_config("argoverse2")

    parser = argparse.ArgumentParser(description="Build scenario records from Argoverse V2 data")
    parser.add_argument("--data", default=cfg.get("data"), help="Path to argoverse2/sensor/")
    parser.add_argument("--split", default=cfg.get("split", "val"),
                        choices=["train", "val", "both"],
                        help="Dataset split to use (default: val)")
    parser.add_argument("--records", default=cfg.get("records", "argoverse2_records"), help="Output records directory")
    parser.add_argument("--num-scenes", type=int, default=cfg.get("num_scenes", 10))
    parser.add_argument("--sampling-step", type=int, default=cfg.get("sampling_step", 6))
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per scene (overrides sampling-step with quartile sampling)")
    parser.add_argument("--history-frames", type=int, default=cfg.get("history_frames", 3))
    parser.add_argument("--future-frames", type=int, default=cfg.get("future_frames", 8))
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Specific scene dir names to build (e.g. argoverse2-0aa4e8f5-2f9a-39)")
    parser.add_argument("--samples-file", default=None,
                        help="File with scene/SAMPLED_N lines to build only those exact samples")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Parse samples file
    samples_filter = None
    if args.samples_file:
        samples_filter = {}
        with open(args.samples_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                scene_dir, sampled = line.split("/")
                idx = int(sampled.replace("SAMPLED_", ""))
                samples_filter.setdefault(scene_dir, set()).add(idx)
        if not args.scenes:
            args.scenes = list(samples_filter.keys())
        print(f"  Samples file: {sum(len(v) for v in samples_filter.values())} samples across {len(samples_filter)} scenes")

    splits = ["train", "val"] if args.split == "both" else [args.split]
    data_root, out_dir = Path(args.data), Path(args.records)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = []
    for split in splits:
        split_dir = data_root / split
        split_scenes = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        scenes.extend(split_scenes)
        print(f"Found {len(split_scenes)} scenes in {split_dir}")
    print(f"Total: {len(scenes)} scenes across {len(splits)} split(s)")
    print(f"  Sampling: step={args.sampling_step}, history={args.history_frames}, future={args.future_frames}")

    # Filter to specific scenes if requested
    if args.scenes:
        scene_filter = set(args.scenes)
        scenes = [d for d in scenes if f"argoverse2-{d.name[:16]}" in scene_filter]
        print(f"  Filtering to {len(scenes)} scenes")

    # Build job list: filter scenes and compute which to process
    jobs = []
    skipped = 0
    for scene_dir in scenes:
        if not args.scenes and (len(jobs) + skipped) >= args.num_scenes:
            break
        scene_name = f"argoverse2-{scene_dir.name[:16]}"
        if (out_dir / scene_name).exists() and not args.force:
            print(f"  SKIP {scene_name}: already exists (use --force)")
            skipped += 1
            continue
        if not (scene_dir / "annotations.feather").exists():
            continue
        if not (scene_dir / "city_SE3_egovehicle.feather").exists():
            continue
        jobs.append(scene_dir)

    print(f"  {len(jobs)} scenes to process, {skipped} skipped")

    scene_args = dict(out_dir=out_dir, history_frames=args.history_frames,
                      future_frames=args.future_frames, sampling_step=args.sampling_step,
                      max_samples=args.max_samples, samples_filter=samples_filter)

    from tqdm import tqdm

    if args.workers > 1:
        from multiprocessing import Pool
        from functools import partial
        worker = partial(_process_one_scene, **scene_args)
        with Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(worker, jobs), total=len(jobs), desc="Building scenes"))
    else:
        results = []
        for sd in jobs:
            results.append(_process_one_scene(sd, verbose=True, **scene_args))

    built = sum(1 for n, _ in results if n > 0)
    total_samples = sum(n for n, _ in results)
    print(f"\nDone. {built} scenes, {total_samples} samples built → {out_dir}")


if __name__ == "__main__":
    main()
