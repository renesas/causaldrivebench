"""
build_openscene_records.py

Convert OpenScene/NAVSIM .pkl logs + sensor blobs + nuPlan maps
into the standard scenario_records format used by generate_causal_graph.py.

Output matches the format produced by STSBench/build_scenario_records.py:
  scenario_records/<scene>/<maneuver>/
      Tm1p5/ Tm1p0/ Tm0p5/ Tp0p0/   (camera images per timestep)
      Tp0p0/bev.png  state.json      (BEV + ego-local state at T=0)
      meta.json

Usage:
  python scripts/build_records/build_openscene_records.py \
      --logs /spiral_hdd_1/workspace/navvrat/data/navsim_logs/trainval \
      --blobs /spiral_hdd_1/workspace/navvrat/data/sensor_blobs/trainval \
      --maps /spiral_hdd_2/workspace/navvrat/Neurips/navsim/download/maps \
      --records openscene_records \
      --num-scenes 3
"""

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────
RADIUS = 30          # metres — vehicle filtering radius
RADIUS_PED = 25      # metres — pedestrian filtering radius
BEV_SIZE = 1536
BEV_SUPERSAMPLE = 2
BEV_RANGE = 50.0     # metres each side of ego
ORIG_CAM_W, ORIG_CAM_H = 1920, 1080  # OpenScene native camera resolution

# Categories to SKIP — static clutter, not drivable agents
SKIP_CATEGORIES = {"generic_object", "traffic_cone", "barrier"}

# Camera mapping: OpenScene → our convention
CAM_MAP = {
    "CAM_F0": "cam_front",
    "CAM_L0": "cam_front_left",
    "CAM_R0": "cam_front_right",
    "CAM_B0": "cam_back",
}

# Timestep slugs for history + T=0  (2Hz: 0.5s apart)
TEMPORAL_OFFSETS = [(-3, "Tm1p5"), (-2, "Tm1p0"), (-1, "Tm0p5"), (0, "Tp0p0")]

# Driving command one-hot → string
DRIVING_CMD = {
    (1, 0, 0, 0): "TURN LEFT",
    (0, 1, 0, 0): "GO STRAIGHT",
    (0, 0, 1, 0): "TURN RIGHT",
    (0, 0, 0, 1): "STOP",
}

# BEV / camera overlay colours (BGR)
COLOR_EGO     = (255, 200, 0)
COLOR_VEHICLE = (180, 200, 255)  # peach (BGR) — distinct from all map/ego colors
COLOR_PED     = (0, 255, 128)
COLOR_CYCLE   = (0, 200, 200)

# Map colours (BGR) — matching STSBench
MAP_COLORS = {
    "drivable":       (50,  50,  50),
    "lane":           (70,  70,  65),
    "lane_connector": (75,  65,  55),
    "lane_div":       (200, 200, 200),  # bright white for VLM visibility
    "road_div":       (0,   200, 255),  # yellow (BGR) for VLM visibility
    "intersection":   (80,  55, 100),
    "ped_crossing":   (50, 100, 180),
    "walkway":        (40,  90,  40),
    "stop_line":      (40,  40, 220),
    "carpark":        (55,  55,  75),
}

# nuPlan map location → gpkg subdirectory
LOCATION_TO_DIR = {
    "us-nv-las-vegas-strip": "us-nv-las-vegas-strip",
    "us-ma-boston":           "us-ma-boston",
    "us-pa-pittsburgh-hazelwood": "us-pa-pittsburgh-hazelwood",
    "sg-one-north":          "sg-one-north",
}


# ── Coordinate utilities ──────────────────────────────────────────────────
def se3_to_xy_yaw(T):
    """Extract (x, y, yaw) from a 4×4 SE3 matrix."""
    return T[0, 3], T[1, 3], math.atan2(T[1, 0], T[0, 0])


def make_ego_local_transform(ego_T):
    """Build functions that map global → ego-local at T=0."""
    x0, y0, yaw0 = se3_to_xy_yaw(ego_T)
    cos_e, sin_e = math.cos(-yaw0), math.sin(-yaw0)

    def to_ego(wx, wy):
        dx, dy = wx - x0, wy - y0
        return round(cos_e * dx - sin_e * dy, 4), round(sin_e * dx + cos_e * dy, 4)

    def norm_yaw(yaw):
        y = yaw - yaw0
        return round((y + math.pi) % (2 * math.pi) - math.pi, 4)

    return to_ego, norm_yaw


def box_to_global(box, ego_T):
    """Convert agent box from ego frame to global. Returns (gx, gy, gz, yaw_global)."""
    pt = ego_T @ np.array([box[0], box[1], box[2], 1.0])
    _, _, ego_yaw = se3_to_xy_yaw(ego_T)
    return pt[0], pt[1], pt[2], box[6] + ego_yaw


def _draw_arrows_along(canvas, bev_pts, spacing, arrow_len, color, thickness):
    """Draw small triangle arrows along a polyline of BEV pixel coords."""
    if len(bev_pts) < 2:
        return
    accum = 0.0
    for k in range(1, len(bev_pts)):
        x0, y0 = bev_pts[k - 1]
        x1, y1 = bev_pts[k]
        dx, dy = x1 - x0, y1 - y0
        seg = math.sqrt(dx * dx + dy * dy)
        if seg < 1:
            continue
        accum += seg
        if accum >= spacing:
            accum = 0.0
            ux, uy = dx / seg, dy / seg
            tx, ty = int((x0 + x1) / 2), int((y0 + y1) / 2)
            p1 = (int(tx - arrow_len * (ux - uy * 0.5)),
                  int(ty - arrow_len * (uy + ux * 0.5)))
            p2 = (int(tx - arrow_len * (ux + uy * 0.5)),
                  int(ty - arrow_len * (uy - ux * 0.5)))
            cv2.fillPoly(canvas,
                         [np.array([(tx, ty), p1, p2], dtype=np.int32)],
                         color)


def get_3d_box_corners(x, y, z, l, w, h, yaw):
    """Get 8 corners of a 3D box in the frame where (x,y,z,yaw) are defined."""
    cos_a, sin_a = math.cos(yaw), math.sin(yaw)
    l2, w2, h2 = l / 2, w / 2, h / 2
    corners = []
    for dz in [-h2, h2]:
        for dl, dw in [(l2, w2), (l2, -w2), (-l2, -w2), (-l2, w2)]:
            cx = x + cos_a * dl - sin_a * dw
            cy = y + sin_a * dl + cos_a * dw
            cz = z + dz
            corners.append([cx, cy, cz])
    return np.array(corners)


def infer_nav_command_from_trajectory(frames, t0_idx, n_future=8):
    """Infer nav command from heading change over T=-1.5 to T=+4.0.
    Dead zone of 0.15 rad (~8.5°). No STOPPED — only STRAIGHT/LEFT/RIGHT."""
    start_idx = max(0, t0_idx - 3)  # T=-1.5
    end_idx = min(t0_idx + n_future, len(frames) - 1)
    if end_idx <= start_idx:
        return "GO STRAIGHT"
    _, _, yaw_start = se3_to_xy_yaw(frames[start_idx]["ego2global"])
    _, _, yaw_end = se3_to_xy_yaw(frames[end_idx]["ego2global"])
    heading_change = (yaw_end - yaw_start + math.pi) % (2 * math.pi) - math.pi
    if abs(heading_change) < 0.15:
        return "GO STRAIGHT"
    elif heading_change > 0:
        return "TURN LEFT"
    else:
        return "TURN RIGHT"


def category_short(cat):
    """OpenScene uses: vehicle, pedestrian, generic_object."""
    cat = cat.lower()
    if "pedestrian" in cat:
        return "pedestrian"
    elif "vehicle" in cat:
        return "vehicle"
    elif "bicycle" in cat or "cycle" in cat:
        return "cyclist"
    return cat


def _agent_color(cat):
    cat = cat.lower()
    if "pedestrian" in cat:
        return COLOR_PED
    elif "cycle" in cat or "bicycle" in cat:
        return COLOR_CYCLE
    return COLOR_VEHICLE


def _cam_transform(cam_info):
    """Return (R_l2s, t_l2s, K) from OpenScene cam_info dict."""
    R_s2l = cam_info["sensor2lidar_rotation"]
    t_s2l = cam_info["sensor2lidar_translation"]
    K = cam_info["cam_intrinsic"]
    R_l2s = R_s2l.T
    t_l2s = -R_l2s @ t_s2l
    return R_l2s, t_l2s, K


def _agent_in_any_cam_os(corners, cams_dict):
    """Return True if any of the 8 box corners projects into any camera (OpenScene)."""
    for os_cam in CAM_MAP:
        if os_cam not in cams_dict:
            continue
        R_l2s, t_l2s, K = _cam_transform(cams_dict[os_cam])
        pts_cam = (R_l2s @ corners.T).T + t_l2s
        mask = pts_cam[:, 2] > 0.5
        if not np.any(mask):
            continue
        pts_front = pts_cam[mask]
        u = K[0, 0] * pts_front[:, 0] / pts_front[:, 2] + K[0, 2]
        v = K[1, 1] * pts_front[:, 1] / pts_front[:, 2] + K[1, 2]
        if np.any((u >= 0) & (u < 1920) & (v >= 0) & (v < 1080)):
            return True
    return False


def agents_visible_in_any_fov(agents, cams_dict):
    """
    Return subset of agents visible in at least one camera FOV.
    Projects all 8 box corners; agent is visible if ANY corner
    projects into ANY camera image.
    """
    visible = []
    for ag in agents:
        corners = get_3d_box_corners(
            ag["x"], ag["y"], ag["z"],
            ag["l"], ag["w"], ag["h"], ag["yaw"]
        )
        if _agent_in_any_cam_os(corners, cams_dict):
            visible.append(ag)
    return visible


def agents_visible_multitemporal_os(t0_agents, frames, t0_idx, n_frames):
    """Check FOV across all 4 temporal frames. Return t0_agents visible in
    ANY frame (t-1.5, t-1.0, t-0.5, t0)."""
    t0_tracks = {a["track_token"] for a in t0_agents}
    visible_tracks = set()

    for offset, _ in TEMPORAL_OFFSETS:
        fidx = t0_idx + offset
        if fidx < 0 or fidx >= n_frames:
            continue
        frame = frames[fidx]
        boxes = frame["anns"]["gt_boxes"]
        tracks = frame["anns"]["track_tokens"]
        cams = frame["cams"]

        for i in range(len(tracks)):
            tk = tracks[i]
            if tk not in t0_tracks or tk in visible_tracks:
                continue
            corners = get_3d_box_corners(
                float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]),
                float(boxes[i, 3]), float(boxes[i, 4]), float(boxes[i, 5]),
                float(boxes[i, 6]),
            )
            if _agent_in_any_cam_os(corners, cams):
                visible_tracks.add(tk)

    return [a for a in t0_agents if a["track_token"] in visible_tracks]


# ── IOU-based occlusion filter (OpenScene cameras) ─────────────────────
IOU_OCCL_THRESH = 0.80


def _projected_2d_box_os(corners_3d, cam_info):
    """Project 8 corners → axis-aligned 2D bbox in OpenScene camera image.
    Returns [u_min, v_min, u_max, v_max] or None."""
    R_l2s, t_l2s, K = _cam_transform(cam_info)
    pts_cam = (R_l2s @ corners_3d.T).T + t_l2s
    mask = pts_cam[:, 2] > 0.5
    if not np.any(mask):
        return None
    pts = pts_cam[mask]
    u = K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2]
    v = K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]
    u = np.clip(u, 0, 1920)
    v = np.clip(v, 0, 1080)
    if (u.max() - u.min()) < 1 or (v.max() - v.min()) < 1:
        return None
    return [float(u.min()), float(v.min()), float(u.max()), float(v.max())]


def _iou_2d(a, b):
    xi = max(a[0], b[0]); yi = max(a[1], b[1])
    xa = min(a[2], b[2]); ya = min(a[3], b[3])
    inter = max(0, xa - xi) * max(0, ya - yi)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0.0


def _ray_intersects_aabb(ray_dir, aabb_min, aabb_max):
    """Test if ray from origin (0,0,0) with direction ray_dir hits an AABB.
    Uses slab method. Returns True if ray intersects the box."""
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


def filter_occluded_agents_os(agents, cams_dict, ray_thresh=0.80):
    """Drop agents where >=80% of rays through their 3D box are blocked
    by a closer VEHICLE in ALL cameras they appear in.

    Multi-ray: 8 corners + center = 9 rays per agent per camera.
    Only vehicles can block. Depth uses camera z (not ego distance).
    """
    if len(agents) <= 1:
        return agents

    n = len(agents)
    is_vehicle = ["vehicle" in a.get("category", "").lower() for a in agents]
    corners_list = [
        get_3d_box_corners(a["x"], a["y"], a["z"],
                           a["l"], a["w"], a["h"], a["yaw"])
        for a in agents
    ]

    visible_in = [set() for _ in range(n)]
    occluded_in = [set() for _ in range(n)]

    for os_cam in CAM_MAP:
        if os_cam not in cams_dict:
            continue
        R_l2s, t_l2s, K = _cam_transform(cams_dict[os_cam])

        all_pts_cam = []
        centers_cam = []
        aabbs = []
        in_front = []
        for i in range(n):
            pts_cam = (R_l2s @ corners_list[i].T).T + t_l2s
            center = pts_cam.mean(axis=0)
            all_pts_cam.append(pts_cam)
            centers_cam.append(center)
            in_front.append(center[2] > 0.5)
            aabbs.append((pts_cam.min(axis=0), pts_cam.max(axis=0)))

        for i in range(n):
            if not in_front[i]:
                continue
            c = centers_cam[i]
            u = K[0, 0] * c[0] / c[2] + K[0, 2]
            v = K[1, 1] * c[1] / c[2] + K[1, 2]
            if not (0 <= u <= 1920 and 0 <= v <= 1080):
                continue
            visible_in[i].add(os_cam)

            # 9 rays: 8 corners + center
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
                occluded_in[i].add(os_cam)

    keep = []
    for i, ag in enumerate(agents):
        if not visible_in[i]:
            keep.append(ag)
            continue
        if occluded_in[i] and occluded_in[i] >= visible_in[i]:
            continue
        keep.append(ag)
    return keep


def filter_and_label_agents(agents_raw):
    """
    Filter and relabel agents:
    1. Drop SKIP_CATEGORIES (generic_object etc.)
    2. Sort: moving agents first, then by distance
    3. Assign clean sequential labels: vehicle_1, pedestrian_2, cyclist_3, ...
    """
    agents = [a for a in agents_raw if a["category"].lower() not in SKIP_CATEGORIES]

    for a in agents:
        a["_dist"] = math.sqrt(a["x"]**2 + a["y"]**2)
    agents.sort(key=lambda a: (a["speed"] < 0.5, a["_dist"]))

    counters = {}
    for a in agents:
        short = category_short(a["category"])
        counters[short] = counters.get(short, 0) + 1
        a["label"] = f"{short}_{counters[short]}"
        del a["_dist"]

    return agents


# ── Camera projection & bbox drawing ─────────────────────────────────────
def project_boxes_to_cam(agents_ego, cam_info):
    """
    Project ego-frame 3D boxes → 2D bboxes in a camera image.

    cam_info: dict with sensor2lidar_rotation (3x3), sensor2lidar_translation (3,),
              cam_intrinsic (3x3).
    Returns list of (x1, y1, x2, y2, label, color) for visible agents.
    """
    R_s2l = cam_info["sensor2lidar_rotation"]     # sensor → lidar/ego
    t_s2l = cam_info["sensor2lidar_translation"]   # sensor → lidar/ego
    K = cam_info["cam_intrinsic"]

    # lidar/ego → sensor (inverse)
    R_l2s = R_s2l.T
    t_l2s = -R_l2s @ t_s2l

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # OpenScene images are 1920×1080 (nuPlan standard)
    img_w, img_h = 1920, 1080

    results = []
    for ag in agents_ego:
        corners = get_3d_box_corners(
            ag["x"], ag["y"], ag["z"],
            ag["l"], ag["w"], ag["h"], ag["yaw"]
        )
        # Transform corners: ego → camera sensor frame
        pts_cam = (R_l2s @ corners.T).T + t_l2s  # (8, 3)

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
    """Draw 2D bboxes on a camera image. bboxes: list of (x1,y1,x2,y2,label,color)."""
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


# ── nuPlan Map Renderer ──────────────────────────────────────────────────
class NuPlanMapBEV:
    """Load nuPlan .gpkg map and render layers into BEV canvas."""

    def __init__(self, maps_dir):
        self.maps_dir = Path(maps_dir)
        self._cache = {}

    def _load(self, location):
        if location in self._cache:
            return self._cache[location]

        dir_name = LOCATION_TO_DIR.get(location, location)
        map_root = self.maps_dir / dir_name
        if not map_root.exists():
            self._cache[location] = None
            return None

        # Find the version subdirectory
        versions = [d for d in map_root.iterdir() if d.is_dir()]
        if not versions:
            self._cache[location] = None
            return None
        gpkg_path = versions[0] / "map.gpkg"
        if not gpkg_path.exists():
            self._cache[location] = None
            return None

        import geopandas as gpd
        from pyproj import Transformer

        # Determine UTM zone from location
        utm_crs = {
            "us-nv-las-vegas-strip": "EPSG:32611",
            "us-ma-boston":           "EPSG:32619",
            "us-pa-pittsburgh-hazelwood": "EPSG:32617",
            "sg-one-north":          "EPSG:32648",
        }
        target_crs = utm_crs.get(location, "EPSG:32611")

        data = {}
        layer_names = {
            "generic_drivable_areas": "drivable",
            "lanes_polygons": "lanes",
            "lane_groups_polygons": "lane_groups",
            "lane_group_connectors": "lgc_polygons",
            "lane_connectors": "lane_connector_lines",
            "intersections": "intersections",
            "crosswalks": "crosswalks",
            "walkways": "walkways",
            "stop_polygons": "stops",
            "carpark_areas": "carparks",
            "boundaries": "boundaries",
            "road_segments": "road_segments",
            "baseline_paths": "baseline_paths",
        }

        for gpkg_layer, key in layer_names.items():
            try:
                gdf = gpd.read_file(gpkg_path, layer=gpkg_layer)
                gdf = gdf.to_crs(target_crs)
                data[key] = gdf
            except Exception:
                data[key] = None

        # Build lane_fid → L-index lookup from lanes_polygons
        lane_fid_to_idx = {}
        if data.get("lanes") is not None:
            for i, (_, row) in enumerate(data["lanes"].iterrows()):
                lane_fid = row.get("lane_fid")
                if lane_fid is not None:
                    lane_fid_to_idx[int(lane_fid)] = i
        data["_lane_fid_to_idx"] = lane_fid_to_idx

        # Keep backward-compat key for BEV rendering (uses "lane_connectors")
        data["lane_connectors"] = data.get("lgc_polygons")

        self._cache[location] = data
        return data

    def get_lane_id(self, location, utm_x, utm_y):
        """Return an L-type lane_id for a UTM (x, y) position, or None.

        1. If the point is inside a regular lane polygon → return L{i}.
        2. If inside a lane_group_connector (intersection), find the nearest
           lane_connector LineString, read its entry_lane_fid, and map that
           back to the L-type lane the vehicle came from.
        """
        data = self._load(location)
        if data is None:
            return None
        from shapely.geometry import Point
        pt = Point(float(utm_x), float(utm_y))

        # 1. Check regular lanes first
        if data.get("lanes") is not None:
            for i, (_, row) in enumerate(data["lanes"].iterrows()):
                if row.geometry is not None and not row.geometry.is_empty:
                    if row.geometry.contains(pt):
                        return f"L{i}"

        # 2. Check lane_group_connector polygons (intersection areas)
        lgc = data.get("lgc_polygons")
        lc_lines = data.get("lane_connector_lines")
        fid_map = data.get("_lane_fid_to_idx", {})
        if lgc is not None and lc_lines is not None and fid_map:
            for _, lgc_row in lgc.iterrows():
                if lgc_row.geometry is None or lgc_row.geometry.is_empty:
                    continue
                if not lgc_row.geometry.contains(pt):
                    continue
                # Point is inside this lane_group_connector — find nearest
                # lane_connector LineString to resolve the actual lane.
                lgc_fid = lgc_row.get("fid", None)
                best_dist, best_entry_fid = float("inf"), None
                for _, lc_row in lc_lines.iterrows():
                    # Filter to connectors in this group if fid available
                    if lgc_fid is not None:
                        lc_lgc = lc_row.get("lane_group_connector_fid")
                        if lc_lgc is not None and int(lc_lgc) != int(lgc_fid):
                            continue
                    if lc_row.geometry is None or lc_row.geometry.is_empty:
                        continue
                    d = lc_row.geometry.distance(pt)
                    if d < best_dist:
                        best_dist = d
                        best_entry_fid = lc_row.get("entry_lane_fid")
                if best_entry_fid is not None:
                    idx = fid_map.get(int(best_entry_fid))
                    if idx is not None:
                        return f"L{idx}"
        return None

    def render_into(self, canvas, location, ego_x, ego_y, ego_to_bev, bev_range, S):
        """Draw map layers onto canvas. ego_x, ego_y in UTM coords."""
        data = self._load(location)
        if data is None:
            return

        clip = bev_range * 1.2

        def geom_to_bev_pts(geom, ego_to_bev_fn):
            """Convert shapely geometry exterior to BEV pixel coords."""
            if geom.is_empty:
                return None
            try:
                coords = list(geom.exterior.coords)
            except AttributeError:
                return None
            pts = []
            for x, y in coords:
                ex, ey = x - ego_x, y - ego_y
                if abs(ex) > clip and abs(ey) > clip:
                    continue
                pts.append(ego_to_bev_fn(ex, ey))
            return np.array(pts, dtype=np.int32) if len(pts) > 2 else None

        def line_to_bev_pts(geom, ego_to_bev_fn):
            """Convert shapely linestring to BEV pixel coords."""
            if geom.is_empty:
                return None
            try:
                coords = list(geom.coords)
            except Exception:
                return None
            pts = []
            for x, y, *_ in coords:
                ex, ey = x - ego_x, y - ego_y
                if abs(ex) > clip and abs(ey) > clip:
                    continue
                pts.append(ego_to_bev_fn(ex, ey))
            return np.array(pts, dtype=np.int32) if len(pts) > 1 else None

        # 1. Drivable areas
        if data.get("drivable") is not None:
            for _, row in data["drivable"].iterrows():
                px = geom_to_bev_pts(row.geometry, ego_to_bev)
                if px is not None:
                    cv2.fillPoly(canvas, [px.reshape(-1, 1, 2)], MAP_COLORS["drivable"])

        # 2. Road segments
        if data.get("road_segments") is not None:
            for _, row in data["road_segments"].iterrows():
                px = geom_to_bev_pts(row.geometry, ego_to_bev)
                if px is not None:
                    cv2.fillPoly(canvas, [px.reshape(-1, 1, 2)], MAP_COLORS["lane"])
                    cv2.polylines(canvas, [px.reshape(-1, 1, 2)], True, (70, 70, 70), S)

        # 3. Lanes
        if data.get("lanes") is not None:
            for _, row in data["lanes"].iterrows():
                px = geom_to_bev_pts(row.geometry, ego_to_bev)
                if px is not None:
                    cv2.fillPoly(canvas, [px.reshape(-1, 1, 2)], MAP_COLORS["lane"])
                    cv2.polylines(canvas, [px.reshape(-1, 1, 2)], True, (85, 85, 80), S)

        # 4. Intersections
        if data.get("intersections") is not None:
            for _, row in data["intersections"].iterrows():
                px = geom_to_bev_pts(row.geometry, ego_to_bev)
                if px is not None:
                    cv2.fillPoly(canvas, [px.reshape(-1, 1, 2)], MAP_COLORS["lane"])  # same as lanes

        # 5. Lane connectors (disabled — clutters intersection areas)
        # if data.get("lane_connectors") is not None:
        #     for _, row in data["lane_connectors"].iterrows():
        #         px = geom_to_bev_pts(row.geometry, ego_to_bev)
        #         if px is not None:
        #             cv2.fillPoly(canvas, [px.reshape(-1, 1, 2)], MAP_COLORS["lane_connector"])

        # 6. Walkways
        if data.get("walkways") is not None:
            for _, row in data["walkways"].iterrows():
                px = geom_to_bev_pts(row.geometry, ego_to_bev)
                if px is not None:
                    cv2.fillPoly(canvas, [px.reshape(-1, 1, 2)], MAP_COLORS["walkway"])
                    cv2.polylines(canvas, [px.reshape(-1, 1, 2)], True, (30, 80, 30), S)

        # 7. Crosswalks
        if data.get("crosswalks") is not None:
            for _, row in data["crosswalks"].iterrows():
                px = geom_to_bev_pts(row.geometry, ego_to_bev)
                if px is not None:
                    cv2.fillPoly(canvas, [px.reshape(-1, 1, 2)], MAP_COLORS["ped_crossing"])
                    cv2.polylines(canvas, [px.reshape(-1, 1, 2)], True, (0, 100, 200), S)

        # 8. (Stop lines removed — not useful for VLM)

        # 9. Carpark areas
        if data.get("carparks") is not None:
            for _, row in data["carparks"].iterrows():
                px = geom_to_bev_pts(row.geometry, ego_to_bev)
                if px is not None:
                    cv2.fillPoly(canvas, [px.reshape(-1, 1, 2)], MAP_COLORS["carpark"])

        # 10. Boundaries (lines — lane dividers / road dividers)
        if data.get("boundaries") is not None:
            for _, row in data["boundaries"].iterrows():
                px = line_to_bev_pts(row.geometry, ego_to_bev)
                if px is not None:
                    btype = row.get("boundary_type_fid", 0)
                    color = MAP_COLORS["road_div"] if btype == 1 else MAP_COLORS["lane_div"]
                    cv2.polylines(canvas, [px.reshape(-1, 1, 2)], False, color, 2 * S)

        # 11. Lane direction arrows from baseline_paths
        if data.get("baseline_paths") is not None:
            _ARR_SPACE = 36 * S
            _ARR_LEN = 8 * S
            _ARR_CLR = (180, 180, 180)
            for _, row in data["baseline_paths"].iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                try:
                    coords = list(geom.coords)
                except Exception:
                    continue
                bev_pts = []
                for c in coords:
                    ex, ey = c[0] - ego_x, c[1] - ego_y
                    if abs(ex) > clip and abs(ey) > clip:
                        continue
                    bev_pts.append(ego_to_bev(ex, ey))
                _draw_arrows_along(canvas, bev_pts, _ARR_SPACE, _ARR_LEN,
                                   _ARR_CLR, S)


# ── BEV Rendering ─────────────────────────────────────────────────────────
def render_bev(agents_ego, map_renderer=None, map_location=None,
               ego_x_utm=0, ego_y_utm=0, ego_yaw=0,
               agent_radius=RADIUS, bev_range=BEV_RANGE, ego_speed=0.0,
               ego_pos=(0, 0), timestamp_label="T=0"):
    """
    Render ego-centric BEV with map layers and agents.
    agents_ego: list of dicts with x, y, z, l, w, h, yaw (in ego frame).
    """
    S = BEV_SUPERSAMPLE
    SZ = BEV_SIZE * S
    canvas = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    bev_cx, bev_cy = SZ // 2, SZ // 2

    cos_e, sin_e = math.cos(-ego_yaw), math.sin(-ego_yaw)

    def ego_to_bev(ex, ey):
        """ego-local (x=fwd, y=left) → BEV pixel."""
        # Rotate UTM offset to ego-local
        lx = cos_e * ex - sin_e * ey
        ly = sin_e * ex + cos_e * ey
        bx = int(bev_cx - ly * SZ / (2 * bev_range))
        by = int(bev_cy - lx * SZ / (2 * bev_range))
        return bx, by

    def ego_local_to_bev(ex, ey):
        """Already ego-local → BEV pixel."""
        bx = int(bev_cx - ey * SZ / (2 * bev_range))
        by = int(bev_cy - ex * SZ / (2 * bev_range))
        return bx, by

    # Draw map layers
    if map_renderer is not None and map_location is not None:
        map_renderer.render_into(canvas, map_location, ego_x_utm, ego_y_utm,
                                 ego_to_bev, bev_range, S)

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
        bev_corners = np.array([ego_local_to_bev(cx_, cy_) for cx_, cy_ in corners],
                               dtype=np.int32)
        th = thickness * S
        if fill:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [bev_corners.reshape(-1, 1, 2)], color)
            cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
        cv2.polylines(canvas, [bev_corners.reshape(-1, 1, 2)], True, color, th)
        front_pts = np.array([ego_local_to_bev(*corners[0]),
                              ego_local_to_bev(*corners[1])], dtype=np.int32)
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
        (MAP_COLORS["ped_crossing"],   "Ped Crossing"),
        (MAP_COLORS["lane_div"],       "Lane Boundary"),
        (MAP_COLORS["walkway"],        "Walkway"),
        (MAP_COLORS["lane"],           "Lane"),
        ((50, 50, 50),                 "Drivable"),
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


# ── Main builder ──────────────────────────────────────────────────────────
def build_scene(pkl_path, blobs_dir, out_dir, map_renderer=None, t0_idx=None,
                sample_idx=0, history_frames=3, future_frames=8):
    """Build scenario records for one OpenScene log."""
    log_name = pkl_path.stem

    with open(pkl_path, "rb") as f:
        frames = pickle.load(f)

    n_frames = len(frames)
    if t0_idx is None:
        t0_idx = min(n_frames - future_frames - 1, max(history_frames, n_frames // 2))
    if t0_idx < history_frames or t0_idx >= n_frames - future_frames:
        return False

    frame_t0 = frames[t0_idx]

    # Check sensor blobs
    log_blob_dir = blobs_dir / log_name
    missing_cams = [c for c in CAM_MAP if not (log_blob_dir / c).exists()]
    if missing_cams:
        return False

    # ── Scene name and output dir ──
    scene_name = f"openscene-{log_name[:30]}"
    scene_dir = out_dir / scene_name / f"SAMPLED_{sample_idx}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # ── T=0 ego transform ──
    ego_T0 = frame_t0["ego2global"]
    to_ego, norm_yaw = make_ego_local_transform(ego_T0)
    ego_x0, ego_y0, ego_yaw0 = se3_to_xy_yaw(ego_T0)

    # ── Decode driving command ──
    cmd_onehot = tuple(int(x) for x in frame_t0["driving_command"])
    nav_command = DRIVING_CMD.get(cmd_onehot, "GO STRAIGHT")
    inferred = infer_nav_command_from_trajectory(frames, t0_idx)
    if nav_command == "GO STRAIGHT" and inferred != "GO STRAIGHT":
        nav_command = inferred

    # ── Scan all 4 frames: agent eligible if in radius in ANY frame ──
    eligible_tracks = set()
    for offset_e, _ in TEMPORAL_OFFSETS:
        fidx = t0_idx + offset_e
        if fidx < 0 or fidx >= len(frames):
            continue
        f = frames[fidx]
        f_boxes = f["anns"]["gt_boxes"]
        f_tracks = f["anns"]["track_tokens"]
        f_names = f["anns"]["gt_names"]
        for i in range(len(f_tracks)):
            x, y = float(f_boxes[i, 0]), float(f_boxes[i, 1])
            dist = math.sqrt(x**2 + y**2)
            r = RADIUS_PED if "pedestrian" in f_names[i].lower() else RADIUS
            if dist <= r:
                eligible_tracks.add(f_tracks[i])

    # ── Build agents at T=0 from eligible set ──
    boxes = frame_t0["anns"]["gt_boxes"]
    names = frame_t0["anns"]["gt_names"]
    vels = frame_t0["anns"]["gt_velocity_3d"]
    tracks = frame_t0["anns"]["track_tokens"]

    agents_t0 = []
    for i in range(len(names)):
        if tracks[i] not in eligible_tracks:
            continue
        x, y, z = float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2])
        l, w, h = float(boxes[i, 3]), float(boxes[i, 4]), float(boxes[i, 5])
        yaw = float(boxes[i, 6])
        speed = math.sqrt(float(vels[i, 0])**2 + float(vels[i, 1])**2)
        agents_t0.append({
            "x": x, "y": y, "z": z, "l": l, "w": w, "h": h, "yaw": yaw,
            "category": names[i], "speed": round(speed, 3),
            "track_token": tracks[i],
            "label": "",
        })

    # Keep only agents visible in at least one camera FOV at T=0
    agents_t0 = agents_visible_in_any_fov(agents_t0, frame_t0["cams"])

    # Drop agents occluded by closer agents via 3D raycast in all cameras
    agents_t0 = filter_occluded_agents_os(agents_t0, frame_t0["cams"])

    # Filter, prioritize, and relabel
    agents_t0 = filter_and_label_agents(agents_t0)

    # Build label→track_token map for history matching
    label_by_track = {a["track_token"]: a["label"] for a in agents_t0}

    # ── Save camera images WITH bounding boxes for 4 temporal frames ──
    for offset, slug in TEMPORAL_OFFSETS:
        fidx = t0_idx + offset
        if fidx < 0 or fidx >= n_frames:
            continue
        frame = frames[fidx]
        slug_dir = scene_dir / slug
        slug_dir.mkdir(parents=True, exist_ok=True)

        # Build agents at this frame for bbox drawing (use T=0 filtered set
        # matched by track_token so labels are consistent across frames)
        f_boxes = frame["anns"]["gt_boxes"]
        f_names = frame["anns"]["gt_names"]
        f_tracks = frame["anns"]["track_tokens"]
        f_agents = []
        for i in range(len(f_names)):
            tk = f_tracks[i]
            if tk not in label_by_track:
                continue  # not in our filtered set
            dist = math.sqrt(float(f_boxes[i, 0])**2 + float(f_boxes[i, 1])**2)
            if dist > RADIUS * 1.5:  # slightly larger for bbox drawing
                continue
            f_agents.append({
                "x": float(f_boxes[i, 0]), "y": float(f_boxes[i, 1]),
                "z": float(f_boxes[i, 2]), "l": float(f_boxes[i, 3]),
                "w": float(f_boxes[i, 4]), "h": float(f_boxes[i, 5]),
                "yaw": float(f_boxes[i, 6]), "category": f_names[i],
                "label": label_by_track[tk],
            })

        for os_cam, our_cam in CAM_MAP.items():
            if os_cam not in frame["cams"]:
                continue
            src_path = blobs_dir / frame["cams"][os_cam]["data_path"]
            dst_path = slug_dir / f"{our_cam}.jpg"
            if not src_path.exists():
                print(f"    WARN: missing image {src_path}")
                continue

            img = cv2.imread(str(src_path))
            if img is None:
                continue

            # Project and draw bounding boxes
            cam_info = frame["cams"][os_cam]
            bboxes = project_boxes_to_cam(f_agents, cam_info)
            draw_cam_bboxes(img, bboxes)
            cv2.imwrite(str(dst_path), img)

    # ── Render BEV (T=0 + temporal past frames) ──
    tp0_dir = scene_dir / "Tp0p0"
    tp0_dir.mkdir(parents=True, exist_ok=True)
    map_location = frame_t0.get("map_location", None)
    eds = frame_t0["ego_dynamic_state"]
    ego_speed = round(math.sqrt(eds[0]**2 + eds[1]**2), 3)
    bev, bev_map = render_bev(
        agents_t0, map_renderer=map_renderer, map_location=map_location,
        ego_x_utm=ego_x0, ego_y_utm=ego_y0, ego_yaw=ego_yaw0,
        agent_radius=RADIUS, ego_speed=ego_speed,
    )
    cv2.imwrite(str(tp0_dir / "bev.png"), bev)
    cv2.imwrite(str(tp0_dir / "bev_map.png"), bev_map)

    # Temporal BEVs: agents + ego at past positions, T=0 ego frame
    for offset, slug, ts_label in [(-3, "Tm1p5", "T=-1.5s"), (-2, "Tm1p0", "T=-1.0s"), (-1, "Tm0p5", "T=-0.5s")]:
        fidx = t0_idx + offset
        if fidx < 0 or fidx >= n_frames:
            continue
        frame = frames[fidx]
        # Ego past position in T=0 ego-local frame
        fx, fy, f_yaw = se3_to_xy_yaw(frame["ego2global"])
        ego_ex, ego_ey = to_ego(fx, fy)
        f_eds = frame["ego_dynamic_state"]
        past_ego_speed = round(math.sqrt(f_eds[0]**2 + f_eds[1]**2), 3)
        # Past agents: past-ego-local → world → T=0-ego-local
        f_boxes = frame["anns"]["gt_boxes"]
        f_names = frame["anns"]["gt_names"]
        f_tracks = frame["anns"]["track_tokens"]
        f_vels = frame["anns"]["gt_velocity_3d"]
        past_T = np.array(frame["ego2global"]).reshape(4, 4)
        past_R = past_T[:3, :3]
        past_t = past_T[:3, 3]
        past_agents = []
        for i in range(len(f_names)):
            tk = f_tracks[i]
            if tk not in label_by_track:
                continue
            lx, ly, lz = float(f_boxes[i, 0]), float(f_boxes[i, 1]), float(f_boxes[i, 2])
            wp = past_R @ np.array([lx, ly, lz]) + past_t
            ex_a, ey_a = to_ego(float(wp[0]), float(wp[1]))
            dist = math.sqrt(ex_a**2 + ey_a**2)
            if dist > RADIUS * 1.5:
                continue
            local_yaw = float(f_boxes[i, 6])
            t0_yaw = (local_yaw + f_yaw - ego_yaw0 + math.pi) % (2 * math.pi) - math.pi
            spd = round(math.sqrt(float(f_vels[i, 0])**2 + float(f_vels[i, 1])**2), 3)
            past_agents.append({
                "x": ex_a, "y": ey_a,
                "z": float(f_boxes[i, 2]), "l": float(f_boxes[i, 3]),
                "w": float(f_boxes[i, 4]), "h": float(f_boxes[i, 5]),
                "yaw": t0_yaw, "category": f_names[i],
                "label": label_by_track[tk], "speed": spd,
            })
        slug_dir = scene_dir / slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        past_bev, past_bev_map = render_bev(
            past_agents, map_renderer=map_renderer, map_location=map_location,
            ego_x_utm=ego_x0, ego_y_utm=ego_y0, ego_yaw=ego_yaw0,
            agent_radius=RADIUS * 1.5, ego_speed=past_ego_speed,
            ego_pos=(ego_ex, ego_ey), timestamp_label=ts_label,
        )
        cv2.imwrite(str(slug_dir / "bev.png"), past_bev)
        cv2.imwrite(str(slug_dir / "bev_map.png"), past_bev_map)

    ego_history = []
    for offset in [-3, -2, -1]:
        fidx = t0_idx + offset
        if fidx < 0:
            continue
        f = frames[fidx]
        fx, fy = se3_to_xy_yaw(f["ego2global"])[:2]
        ex, ey = to_ego(fx, fy)
        f_eds = f["ego_dynamic_state"]
        ego_history.append({"t": offset*0.5, "x": ex, "y": ey,
                            "speed": round(math.sqrt(f_eds[0]**2+f_eds[1]**2), 3)})

    ego_future = []
    for offset in range(1, future_frames + 1):
        fidx = t0_idx + offset
        if fidx >= n_frames:
            continue
        f = frames[fidx]
        fx, fy = se3_to_xy_yaw(f["ego2global"])[:2]
        ex, ey = to_ego(fx, fy)
        f_eds = f["ego_dynamic_state"]
        ego_future.append({"t": offset*0.5, "x": ex, "y": ey,
                           "speed": round(math.sqrt(f_eds[0]**2+f_eds[1]**2), 3)})

    # Agent history
    t0_track_set = {ag["track_token"] for ag in agents_t0}
    agent_hist = defaultdict(list)
    for offset in [-3, -2, -1]:
        fidx = t0_idx + offset
        if fidx < 0:
            continue
        f = frames[fidx]
        f_ego_T = f["ego2global"]
        h_boxes, h_vels, h_tracks = f["anns"]["gt_boxes"], f["anns"]["gt_velocity_3d"], f["anns"]["track_tokens"]
        for j in range(len(h_tracks)):
            tk = h_tracks[j]
            if tk not in t0_track_set:
                continue
            gx, gy, _, _ = box_to_global(h_boxes[j], f_ego_T)
            ex, ey = to_ego(gx, gy)
            spd = round(math.sqrt(float(h_vels[j,0])**2 + float(h_vels[j,1])**2), 3)
            agent_hist[tk].append({"t": offset*0.5, "x": ex, "y": ey, "speed": spd})

    # Assign lane IDs from map (ego-local → UTM → polygon test)
    def _ego_to_utm(ex, ey):
        cos_e, sin_e = math.cos(ego_yaw0), math.sin(ego_yaw0)
        return ego_x0 + cos_e * ex - sin_e * ey, ego_y0 + sin_e * ex + cos_e * ey

    agents_out = {}
    for ag in agents_t0:
        tk = ag["track_token"]
        a_yaw_norm = (ag["yaw"] + math.pi) % (2 * math.pi) - math.pi
        entry = {
            "category": ag["category"],
            "position": {"x": round(ag["x"], 4), "y": round(ag["y"], 4),
                         "yaw": round(a_yaw_norm, 4)},
            "speed": ag["speed"],
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

    # ── uuid_map.json (label → track_token) ──
    uuid_map = {label: track_token for track_token, label in label_by_track.items()}
    with open(tp0_dir / "uuid_map.json", "w") as f:
        json.dump(uuid_map, f, indent=2)

    # ── Build calib.json (camera intrinsics + extrinsics at native resolution) ──
    calib_out = {}
    for os_cam, std_cam in CAM_MAP.items():
        if os_cam in frame_t0["cams"]:
            ci = frame_t0["cams"][os_cam]
            R_s2l = np.array(ci["sensor2lidar_rotation"])
            t_s2l = np.array(ci["sensor2lidar_translation"])
            K = np.array(ci["cam_intrinsic"]).copy()
            # No scaling — images saved at native resolution
            R_ego2cam = R_s2l.T
            t_ego2cam = (-R_ego2cam @ t_s2l).tolist()
            calib_out[std_cam] = {
                "intrinsic": K.tolist(),
                "R_ego2cam": R_ego2cam.tolist(),
                "t_ego2cam": t_ego2cam,
                "image_size": [ORIG_CAM_W, ORIG_CAM_H],
            }
    with open(tp0_dir / "calib.json", "w") as f:
        json.dump(calib_out, f, indent=2)

    meta = {
        "scene": scene_name, "maneuver_type": f"SAMPLED_{sample_idx}", "nav_command": nav_command,
        "dataset": "openscene", "log_name": log_name, "t0_frame_idx": t0_idx, "sample_idx": sample_idx,
        "timestamp": frame_t0["timestamp"], "map_location": map_location,
        "n_agents": len(agents_out),
        "pkl_path": str(pkl_path),
    }
    with open(scene_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return True


def _process_one_scene(pkl, blobs_dir, out_dir, maps_dir, history_frames,
                       future_frames, sampling_step, max_samples, samples_filter,
                       verbose=False):
    """Process all samples for one OpenScene log. Returns (n_samples, scene_name)."""
    import pickle as _pickle
    scene_name = f"openscene-{pkl.stem[:30]}"
    if verbose:
        print(f"Processing {scene_name}...")

    with open(pkl, "rb") as f:
        frames = _pickle.load(f)
    n_frames = len(frames)
    min_frames = history_frames + future_frames + 1
    if n_frames < min_frames:
        return 0, scene_name

    t0_start = history_frames
    t0_end = n_frames - future_frames
    if max_samples and max_samples > 0:
        valid_range = t0_end - t0_start
        if valid_range <= 0:
            return 0, scene_name
        n_samp = min(max_samples, valid_range)
        t0_indices = [t0_start + int(valid_range * (i + 1) / (n_samp + 1))
                      for i in range(n_samp)]
    else:
        t0_indices = list(range(t0_start, t0_end, sampling_step))

    # Each worker creates its own map renderer (has internal cache)
    map_renderer = NuPlanMapBEV(maps_dir) if maps_dir else None

    wanted_indices = samples_filter.get(scene_name) if samples_filter else None
    n_samples = 0
    for sample_idx, t0_idx in enumerate(t0_indices):
        if wanted_indices is not None and sample_idx not in wanted_indices:
            continue
        if build_scene(pkl, blobs_dir, out_dir, map_renderer,
                      t0_idx=t0_idx, sample_idx=sample_idx,
                      history_frames=history_frames,
                      future_frames=future_frames):
            n_samples += 1
    if verbose and n_samples:
        print(f"    → {n_samples} samples from {scene_name}")
    return n_samples, scene_name


def main():
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config_loader import load_config
    cfg = load_config("openscene")

    parser = argparse.ArgumentParser(description="Build scenario records from OpenScene/NAVSIM data")
    parser.add_argument("--logs", default=cfg.get("logs"), help="Path to navsim_logs/trainval/")
    parser.add_argument("--blobs", default=cfg.get("blobs"), help="Path to sensor_blobs/trainval/")
    parser.add_argument("--maps", default=cfg.get("maps"), help="Path to nuPlan maps directory")
    parser.add_argument("--records", default=cfg.get("records", "openscene_records"), help="Output records directory")
    parser.add_argument("--num-scenes", type=int, default=cfg.get("num_scenes", 10))
    parser.add_argument("--sampling-step", type=int, default=cfg.get("sampling_step", 6))
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per scene (overrides sampling-step with dynamic step)")
    parser.add_argument("--history-frames", type=int, default=cfg.get("history_frames", 3))
    parser.add_argument("--future-frames", type=int, default=cfg.get("future_frames", 8))
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Specific scene dir names to build (e.g. openscene-2021.05.12.19.36.12_veh-35_002)")
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

    logs_dir, blobs_dir, out_dir = Path(args.logs), Path(args.blobs), Path(args.records)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map renderer created per-worker (each needs its own cache)

    valid_logs = [pkl for pkl in sorted(logs_dir.glob("*.pkl"))
                  if all((blobs_dir / pkl.stem / cam).exists() for cam in CAM_MAP)]
    print(f"Found {len(valid_logs)} logs with all 4 cameras (of {len(list(logs_dir.glob('*.pkl')))} total)")
    print(f"  Sampling: step={args.sampling_step}, history={args.history_frames}, future={args.future_frames}")
    if not valid_logs:
        print("ERROR: No logs with all 4 cameras found.")
        return

    # Filter to specific scenes if requested
    if args.scenes:
        scene_filter = set(args.scenes)
        valid_logs = [pkl for pkl in valid_logs if f"openscene-{pkl.stem[:30]}" in scene_filter]
        print(f"  Filtering to {len(valid_logs)} scenes")

    # Build job list
    jobs = []
    skipped = 0
    for pkl in valid_logs:
        if not args.scenes and (len(jobs) + skipped) >= args.num_scenes:
            break
        scene_name = f"openscene-{pkl.stem[:30]}"
        if (out_dir / scene_name).exists() and not args.force:
            print(f"  SKIP {scene_name}: already exists (use --force)")
            skipped += 1
            continue
        jobs.append(pkl)

    print(f"  {len(jobs)} scenes to process, {skipped} skipped")

    scene_args = dict(blobs_dir=blobs_dir, out_dir=out_dir, maps_dir=args.maps,
                      history_frames=args.history_frames, future_frames=args.future_frames,
                      sampling_step=args.sampling_step, max_samples=args.max_samples,
                      samples_filter=samples_filter)

    from tqdm import tqdm

    if args.workers > 1:
        from multiprocessing import Pool
        from functools import partial
        worker = partial(_process_one_scene, **scene_args)
        with Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(worker, jobs), total=len(jobs), desc="Building scenes"))
    else:
        results = []
        for pkl in jobs:
            results.append(_process_one_scene(pkl, verbose=True, **scene_args))

    built = sum(1 for n, _ in results if n > 0)
    total_samples = sum(n for n, _ in results)
    print(f"\nDone. {built} scenes, {total_samples} samples built → {out_dir}")


if __name__ == "__main__":
    main()
