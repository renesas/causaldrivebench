"""
build_nuscenes_records.py

Convert nuScenes v1.0-trainval data into the standard scenario_records format.

Output:
  <out>/<scene_name>/SAMPLED/
      Tm1p5/ Tm1p0/ Tm0p5/ Tp0p0/   (camera images per timestep)
      Tm1p5/bev.png  Tm1p0/bev.png  Tm0p5/bev.png  Tp0p0/bev.png
      Tp0p0/state.json  Tp0p0/state_desc.json  calib.json
      meta.json

Usage:
  python scripts/build_records/build_nuscenes_records.py \
      --data nuscenes \
      --version v1.0-trainval \
      --maps nuscenes_full/maps/expansion \
      --records nuscenes_records \
      --num-scenes 10
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    import lanelet2
    from lanelet2.io import Origin, load as lanelet2_load
    from lanelet2.core import BasicPoint2d
    from lanelet2.geometry import findNearest as lanelet2_findNearest
    from lanelet2.projection import UtmProjector
    HAS_LANELET2 = True
except ImportError:
    HAS_LANELET2 = False

# ── Constants ──────────────────────────────────────────────────────────────
RADIUS = 30          # metres — vehicle filtering radius
RADIUS_PED = 25      # metres — pedestrian filtering radius
BEV_SIZE = 1536
BEV_SUPERSAMPLE = 2
BEV_RANGE = 50.0     # metres each side of ego (matches OpenScene/STSBench default)

# Minimum nuScenes visibility token — "1"=0-40%, "2"=40-60%, "3"=60-80%, "4"=80-100%
MIN_VISIBILITY_VEHICLE = "2"   # 40-60% — vehicles are large, still identifiable
MIN_VISIBILITY_PEDESTRIAN = "3"  # 60-100% — pedestrians need higher visibility

# Categories to SKIP (static clutter)
SKIP_CATEGORIES = {
    "movable_object.barrier", "movable_object.trafficcone",
    "movable_object.pushable_pullable", "movable_object.debris",
    "static_object.bicycle_rack", "animal",
}


# Camera mapping: nuScenes channel → our convention
CAM_MAP = {
    "CAM_FRONT": "cam_front",
    "CAM_FRONT_LEFT": "cam_front_left",
    "CAM_FRONT_RIGHT": "cam_front_right",
    "CAM_BACK": "cam_back",
}

# Timestep slugs for history + T=0 (2Hz keyframes)
TIMESTEP_OFFSETS = {
    -3: "Tm1p5",   # T = -1.5 s
    -2: "Tm1p0",   # T = -1.0 s
    -1: "Tm0p5",   # T = -0.5 s
     0: "Tp0p0",   # T =  0.0 s
}

# Agent category → short prefix
CAT_PREFIX = {
    "human.pedestrian": "pedestrian",
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.construction": "construction",
    "vehicle.trailer": "trailer",
    "vehicle.emergency": "emergency",
}

# BEV colours per category family
BEV_COLORS = {
    "vehicle":    (180, 220, 255),   # light blue
    "human":      (180, 255, 180),   # light green
    "motorcycle": (255, 200, 180),   # peach
    "bicycle":    (255, 255, 180),   # light yellow
    "default":    (200, 200, 200),   # grey
}

CAM_BBOX_COLORS = {
    "vehicle":    (180, 200, 255),   # peach (BGR) — matches STSBench COLOR_VEHICLE
    "human":      (0, 255, 128),     # green — matches STSBench COLOR_PED
    "pedestrian": (0, 255, 128),     # green
    "motorcycle": (0, 200, 200),     # yellow-ish — matches STSBench COLOR_CYCLE
    "bicycle":    (0, 200, 200),     # yellow-ish
    "cycle":      (0, 200, 200),     # yellow-ish
    "default":    (180, 200, 255),   # default to vehicle color
}


# ── Quaternion / Transform Helpers ─────────────────────────────────────────

def quat_to_yaw(qw, qx, qy, qz):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def quat_to_rotation_matrix(qw, qx, qy, qz):
    R = np.eye(3)
    R[0, 0] = 1 - 2*(qy**2 + qz**2)
    R[0, 1] = 2*(qx*qy - qw*qz)
    R[0, 2] = 2*(qx*qz + qw*qy)
    R[1, 0] = 2*(qx*qy + qw*qz)
    R[1, 1] = 1 - 2*(qx**2 + qz**2)
    R[1, 2] = 2*(qy*qz - qw*qx)
    R[2, 0] = 2*(qx*qz - qw*qy)
    R[2, 1] = 2*(qy*qz + qw*qx)
    R[2, 2] = 1 - 2*(qx**2 + qy**2)
    return R


def make_T(translation, rotation):
    """Build 4x4 transform from [x,y,z] translation and [qw,qx,qy,qz] rotation."""
    T = np.eye(4)
    T[:3, :3] = quat_to_rotation_matrix(*rotation)
    T[:3, 3] = translation
    return T


# ── Lightweight nuScenes JSON Loader ───────────────────────────────────────

class NuScenesData:
    """Load nuScenes metadata from JSON files (no devkit dependency)."""

    def __init__(self, dataroot, version="v1.0-trainval"):
        root = Path(dataroot)
        meta = root / version

        def load(name):
            data = json.loads((meta / name).read_text())
            return {d["token"]: d for d in data}

        self.root = root
        self.scenes = load("scene.json")
        self.samples = load("sample.json")
        self.sample_data = load("sample_data.json")
        self.annotations = load("sample_annotation.json")
        self.ego_poses = load("ego_pose.json")
        self.cal_sensors = load("calibrated_sensor.json")
        self.instances = load("instance.json")
        self.categories = load("category.json")
        self.attributes = load("attribute.json")
        self.logs = load("log.json")

        # Attribute token → name lookup
        self.attr_name = {tok: a["name"] for tok, a in self.attributes.items()}
        # Static attributes (speed should be 0)
        self.STATIC_ATTRS = {"vehicle.parked", "vehicle.stopped",
                             "pedestrian.standing", "pedestrian.sitting_lying_down"}

        sensors_raw = json.loads((meta / "sensor.json").read_text())
        self.sensors = {s["token"]: s for s in sensors_raw}

        # Build calibrated_sensor → channel lookup
        self.cs_channel = {}
        for cs_tok, cs in self.cal_sensors.items():
            s = self.sensors.get(cs["sensor_token"], {})
            self.cs_channel[cs_tok] = s.get("channel", "UNKNOWN")

        # Build sample → {channel: sample_data_token} for keyframes
        self.sample_sd = defaultdict(dict)  # sample_token → {channel: sd_token}
        for sd_tok, sd in self.sample_data.items():
            if sd["is_key_frame"]:
                ch = self.cs_channel.get(sd["calibrated_sensor_token"], "")
                self.sample_sd[sd["sample_token"]][ch] = sd_tok

        # Category token → name
        self.cat_name = {tok: c["name"] for tok, c in self.categories.items()}
        # Instance token → category name
        self.inst_cat = {}
        for tok, inst in self.instances.items():
            self.inst_cat[tok] = self.cat_name.get(inst["category_token"], "unknown")

    def scene_samples(self, scene_token):
        """Return ordered list of sample tokens for a scene."""
        scene = self.scenes[scene_token]
        tokens = []
        tok = scene["first_sample_token"]
        while tok:
            tokens.append(tok)
            nxt = self.samples[tok].get("next", "")
            tok = nxt if nxt else None
        return tokens

    def get_ego_pose(self, sample_token):
        """Get ego pose for a sample (via LIDAR_TOP sample_data)."""
        sd_map = self.sample_sd.get(sample_token, {})
        lidar_sd_tok = sd_map.get("LIDAR_TOP")
        if lidar_sd_tok:
            sd = self.sample_data[lidar_sd_tok]
            return self.ego_poses[sd["ego_pose_token"]]
        # Fallback: use any camera's ego_pose
        for ch in CAM_MAP:
            sd_tok = sd_map.get(ch)
            if sd_tok:
                sd = self.sample_data[sd_tok]
                return self.ego_poses[sd["ego_pose_token"]]
        return None

    def get_cam_sd(self, sample_token, channel):
        """Get sample_data dict for a camera channel at a sample."""
        sd_tok = self.sample_sd.get(sample_token, {}).get(channel)
        if sd_tok:
            return self.sample_data[sd_tok]
        return None

    def get_cam_image_path(self, sample_token, channel):
        """Get full image path for a camera at a sample."""
        sd = self.get_cam_sd(sample_token, channel)
        if sd:
            return self.root / sd["filename"]
        return None

    def get_cam_calibration(self, sample_token, channel):
        """Get calibrated_sensor for a camera at a sample."""
        sd = self.get_cam_sd(sample_token, channel)
        if sd:
            return self.cal_sensors[sd["calibrated_sensor_token"]]
        return None

    def get_annotations(self, sample_token):
        """Get all annotations for a sample."""
        sample = self.samples[sample_token]
        # Walk annotation linked list from sample
        # Actually, annotations are indexed by token; we need sample → annotations
        # Use brute force filter (pre-index would be better for large datasets)
        return [a for a in self.annotations.values() if a["sample_token"] == sample_token]

    def get_location(self, scene_token):
        """Get map location string for a scene."""
        scene = self.scenes[scene_token]
        log = self.logs[scene["log_token"]]
        return log["location"]


# Pre-index annotations by sample_token for speed
def build_ann_index(nsd):
    """Build sample_token → list of annotations."""
    idx = defaultdict(list)
    for ann in nsd.annotations.values():
        idx[ann["sample_token"]].append(ann)
    return idx


# ── 3D Box Helpers ─────────────────────────────────────────────────────────

def get_3d_box_corners_global(agent):
    """Return 8x3 array of 3D box corners in global frame (using full quaternion)."""
    l, w, h = agent["l"], agent["w"], agent["h"]
    corners = np.array([
        [ l/2,  w/2, -h/2], [ l/2, -w/2, -h/2], [-l/2, -w/2, -h/2], [-l/2,  w/2, -h/2],
        [ l/2,  w/2,  h/2], [ l/2, -w/2,  h/2], [-l/2, -w/2,  h/2], [-l/2,  w/2,  h/2],
    ])  # (8, 3)
    T_global = make_T(
        [agent["gx"], agent["gy"], agent["gz"]],
        [agent["qw"], agent["qx"], agent["qy"], agent["qz"]],
    )
    pts_h = np.hstack([corners, np.ones((8, 1))])  # (8, 4)
    pts_global = (T_global @ pts_h.T).T[:, :3]     # (8, 3)
    return pts_global


def _agent_color(category, palette):
    for prefix, color in palette.items():
        if prefix in category:
            return color
    return palette.get("default", (200, 200, 200))


def _cat_prefix(category):
    for prefix, short in CAT_PREFIX.items():
        if category.startswith(prefix):
            return short
    if "human.pedestrian" in category:
        return "pedestrian"
    if "vehicle" in category:
        return "vehicle"
    return "object"


# ── Camera Projection & BBox Drawing ──────────────────────────────────────

def project_boxes_to_cam(agents_ego, cam_calib, ego_pose):
    """Project agents to 2D camera bboxes using global→ego→camera chain.

    agents_ego: list of dicts with global coords (gx,gy,gz,qw,qx,qy,qz) + l,w,h
    cam_calib: calibrated_sensor dict with translation, rotation, camera_intrinsic
    ego_pose: dict with 'translation' [x,y,z] and 'rotation' [qw,qx,qy,qz]
    """
    # Global → ego (invert ego pose)
    T_ego = make_T(ego_pose["translation"], ego_pose["rotation"])
    T_ego_inv = np.linalg.inv(T_ego)

    # Ego → camera (invert sensor-to-ego)
    T_sensor_ego = make_T(cam_calib["translation"], cam_calib["rotation"])
    T_cam_inv = np.linalg.inv(T_sensor_ego)

    # Combined: global → camera
    T_global_cam = T_cam_inv @ T_ego_inv

    K = np.array(cam_calib["camera_intrinsic"])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    img_w, img_h = 1600, 900

    results = []
    for ag in agents_ego:
        corners_global = get_3d_box_corners_global(ag)
        pts_h = np.hstack([corners_global, np.ones((8, 1))])
        pts_cam = (T_global_cam @ pts_h.T).T[:, :3]

        # Filter: must be in front of camera
        mask = pts_cam[:, 2] > 0.1
        if not np.any(mask):
            continue

        pts_front = pts_cam[mask]
        u = fx * pts_front[:, 0] / pts_front[:, 2] + cx
        v = fy * pts_front[:, 1] / pts_front[:, 2] + cy

        # Generous ±20% margin (like STSBench) for partially visible objects
        in_img = ((u >= -img_w * 0.2) & (u < img_w * 1.2) &
                  (v >= -img_h * 0.2) & (v < img_h * 1.2))
        if not np.any(in_img):
            continue

        x1 = int(np.clip(u.min(), 0, img_w - 1))
        y1 = int(np.clip(v.min(), 0, img_h - 1))
        x2 = int(np.clip(u.max(), 0, img_w - 1))
        y2 = int(np.clip(v.max(), 0, img_h - 1))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue

        results.append((x1, y1, x2, y2, ag.get("label", ""),
                        _agent_color(ag["category"], CAM_BBOX_COLORS)))
    return results


def draw_cam_bboxes(img, bboxes):
    """Draw 2D bboxes on a camera image (matches STSBench style)."""
    for x1, y1, x2, y2, label, color in bboxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            ty = max(y1 - 6, th + 4)
            cv2.rectangle(img, (x1 - 2, ty - th - 4), (x1 + tw + 4, ty + baseline + 2),
                          (0, 0, 0), -1)
            cv2.putText(img, label, (x1, ty), font, font_scale, color, thickness)


def _draw_arrows_along(canvas, bev_pts, spacing, arrow_len, color):
    """Draw small triangle arrows along a polyline of BEV pixel coords (STSBench)."""
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
        ux, uy = dx / seg, dy / seg
        # Walk along this segment placing arrows every `spacing` pixels
        pos = 0.0
        while pos < seg:
            remaining = spacing - accum
            if pos + remaining > seg:
                accum += seg - pos
                break
            pos += remaining
            accum = 0.0
            tx = int(x0 + ux * pos)
            ty = int(y0 + uy * pos)
            p1 = (int(tx - arrow_len * (ux - uy * 0.5)),
                  int(ty - arrow_len * (uy + ux * 0.5)))
            p2 = (int(tx - arrow_len * (ux + uy * 0.5)),
                  int(ty - arrow_len * (uy - ux * 0.5)))
            cv2.fillPoly(canvas,
                         [np.array([(tx, ty), p1, p2], dtype=np.int32)],
                         color)


# ── nuScenes HD Map BEV Renderer ──────────────────────────────────────────

class NuScenesMapBEV:
    """Load and render nuScenes HD maps for BEV."""

    def __init__(self, maps_dir):
        self.maps_dir = Path(maps_dir)
        self._cache = {}

    def _load(self, location):
        if location in self._cache:
            return self._cache[location]
        path = self.maps_dir / f"{location}.json"
        if not path.exists():
            print(f"  WARNING: Map not found: {path}")
            self._cache[location] = None
            return None
        raw = json.loads(path.read_text())

        # Index nodes
        nodes = {n["token"]: (n["x"], n["y"]) for n in raw.get("node", [])}
        # Index polygons
        polys = {}
        for p in raw.get("polygon", []):
            pts = [nodes[t] for t in p["exterior_node_tokens"] if t in nodes]
            if len(pts) >= 3:
                polys[p["token"]] = np.array(pts, dtype=np.float64)
        # Index lines
        lines = {}
        for ln in raw.get("line", []):
            pts = [nodes[t] for t in ln["node_tokens"] if t in nodes]
            if pts:
                lines[ln["token"]] = np.array(pts, dtype=np.float64)

        # Index arcline paths (lane centerlines)
        arclines = raw.get("arcline_path_3", {})

        data = {"raw": raw, "nodes": nodes, "polys": polys, "lines": lines,
                "arclines": arclines}
        self._cache[location] = data
        return data

    def render_into(self, canvas, location, ego_x, ego_y, world_to_bev, bev_range, S=1):
        """Draw map layers onto BEV canvas (matches STSBench style)."""
        data = self._load(location)
        if data is None:
            return
        raw, polys, lines = data["raw"], data["polys"], data["lines"]
        arclines = data.get("arclines", {})

        # Colors (BGR, 1:1 from STSBench MAP_COLORS)
        C_DRIVABLE    = (50,  50,  50)
        C_LANE        = (70,  70,  65)
        C_LANE_CONN   = (75,  65,  55)
        C_INTERSECTION = (80,  55, 100)
        C_WALKWAY     = (40,  90,  40)
        C_PED_CROSS   = (50, 100, 180)
        C_STOP_LINE   = (40,  40, 220)
        C_CARPARK     = (55,  55,  75)
        C_LANE_DIV    = (200, 200, 200)
        C_ROAD_DIV    = (0,   200, 255)
        C_BORDER      = (200, 200, 200)

        def poly_px(poly_token):
            pts = polys.get(poly_token)
            if pts is None or len(pts) < 3:
                return None
            return np.array([world_to_bev(x, y) for x, y in pts], dtype=np.int32)

        def line_px(line_token):
            pts = lines.get(line_token)
            if pts is None or len(pts) < 2:
                return None
            return np.array([world_to_bev(x, y) for x, y in pts], dtype=np.int32)

        # 1. Drivable area (base layer) — uses polygon_tokens (plural!)
        for da in raw.get("drivable_area", []):
            for pt in da.get("polygon_tokens", []):
                px = poly_px(pt)
                if px is not None:
                    cv2.fillPoly(canvas, [px], C_DRIVABLE)

        # 2. Road segments (intersection vs plain)
        _B = max(1, S)  # border thickness scale
        for rs in raw.get("road_segment", []):
            px = poly_px(rs.get("polygon_token", ""))
            if px is None:
                continue
            c = C_LANE  # intersections use same color as lanes
            cv2.fillPoly(canvas, [px], c)
            cv2.polylines(canvas, [px], True, C_BORDER, 2 * _B)

        # 3. Lanes (with border)
        for ln in raw.get("lane", []):
            px = poly_px(ln.get("polygon_token", ""))
            if px is not None:
                cv2.fillPoly(canvas, [px], C_LANE)
                cv2.polylines(canvas, [px], True, C_BORDER, 2 * _B)

        # 4. Lane connectors — removed (no fill, no boundaries, arrows from nuScenes arclines only)

        # 5. Carpark
        for cp in raw.get("carpark_area", []):
            px = poly_px(cp.get("polygon_token", ""))
            if px is not None:
                cv2.fillPoly(canvas, [px], C_CARPARK)

        # 6. Walkways
        for wk in raw.get("walkway", []):
            px = poly_px(wk.get("polygon_token", ""))
            if px is not None:
                cv2.fillPoly(canvas, [px], C_WALKWAY)
                cv2.polylines(canvas, [px], True, (20, 60, 20), 2 * _B)

        # 7. Pedestrian crossings
        for pc in raw.get("ped_crossing", []):
            px = poly_px(pc.get("polygon_token", ""))
            if px is not None:
                cv2.fillPoly(canvas, [px], C_PED_CROSS)
                cv2.polylines(canvas, [px], True, (0, 70, 150), 2 * _B)

        # 8. (Stop lines removed — not useful for VLM)

        # 9. Lane dividers (bright white for VLM visibility)
        _DIV_S = max(1, canvas.shape[0] // BEV_SIZE)  # supersample factor
        for ld in raw.get("lane_divider", []):
            lt = ld.get("line_token")
            if lt:
                px = line_px(lt)
                if px is not None:
                    cv2.polylines(canvas, [px.reshape(-1, 1, 2)], False,
                                  C_LANE_DIV, 2 * _DIV_S)

        # 10. Road dividers (yellow for VLM visibility)
        for rd in raw.get("road_divider", []):
            lt = rd.get("line_token")
            if lt:
                px = line_px(lt)
                if px is not None:
                    cv2.polylines(canvas, [px.reshape(-1, 1, 2)], False,
                                  C_ROAD_DIV, 2 * _DIV_S)

        # 11. Lane direction arrows — from arcline_path_3 centerlines
        _S = max(1, canvas.shape[0] // BEV_SIZE)
        _ARR_CLR = (200, 200, 200)
        _ARR_SPACE = 36 * _S
        _ARR_LEN = 8 * _S

        def _arcline_to_bev_path(token):
            """Convert arcline_path_3 to BEV path using Dubins path geometry.
            shape: 3 chars — L=left arc, S=straight, R=right arc.
            radius: shared turning radius for L/R segments.
            segment_length: [len0, len1, len2] for the 3 sub-segments."""
            arcs = arclines.get(token)
            if not arcs:
                return []
            dense = []
            for arc in arcs:
                sx, sy, syaw = arc["start_pose"]
                shape = arc.get("shape", "SSS")
                radius = arc.get("radius", 1e9)
                seg_lens = arc.get("segment_length", [0, 0, 0])
                cx, cy, cyaw = sx, sy, syaw
                for j, ch in enumerate(shape):
                    slen = seg_lens[j] if j < len(seg_lens) else 0
                    if slen < 0.01:
                        continue
                    n_steps = max(2, int(slen / 0.5))
                    if ch == 'S':
                        # Straight
                        for i in range(n_steps + 1):
                            s = slen * i / n_steps
                            dense.append((cx + s * math.cos(cyaw),
                                          cy + s * math.sin(cyaw)))
                        cx += slen * math.cos(cyaw)
                        cy += slen * math.sin(cyaw)
                    else:
                        # L = left arc (+1/r), R = right arc (-1/r)
                        curv = (1.0 / radius) if ch == 'L' else (-1.0 / radius)
                        r_signed = 1.0 / curv
                        for i in range(n_steps + 1):
                            s = slen * i / n_steps
                            theta = s * curv
                            x = cx + r_signed * (math.sin(cyaw + theta) - math.sin(cyaw))
                            y = cy - r_signed * (math.cos(cyaw + theta) - math.cos(cyaw))
                            dense.append((x, y))
                        total_theta = slen * curv
                        cx += r_signed * (math.sin(cyaw + total_theta) - math.sin(cyaw))
                        cy -= r_signed * (math.cos(cyaw + total_theta) - math.cos(cyaw))
                        cyaw += total_theta
            # Filter to BEV range
            bev_path = []
            for wx, wy in dense:
                d = math.sqrt((wx - ego_x) ** 2 + (wy - ego_y) ** 2)
                if d <= bev_range * 1.4:
                    bev_path.append(world_to_bev(wx, wy))
            return bev_path

        for ln in raw.get("lane", []):
            bev_path = _arcline_to_bev_path(ln["token"])
            if len(bev_path) >= 2:
                _draw_arrows_along(canvas, bev_path, _ARR_SPACE,
                                   _ARR_LEN, _ARR_CLR)

        # Lane connector arrows (through intersections)
        for lc in raw.get("lane_connector", []):
            bev_path = _arcline_to_bev_path(lc["token"])
            if len(bev_path) >= 2:
                _draw_arrows_along(canvas, bev_path, _ARR_SPACE,
                                   _ARR_LEN, _ARR_CLR)

    def get_lane_token(self, location, wx, wy, yaw=None, is_pedestrian=False):
        """Robust lane assignment using Lanelet2 + nuScenes walkway check.

        1. Walkway check (nuScenes polygons) — pedestrians on sidewalks get None
        2. Lanelet2 findNearest + heading filter — correct lane even at polygon boundaries
        3. Chain grouping (following/previous, skip connectors, merge left/right) — same road = same ID
        """
        ll_data = self._get_lanelet2_data(location)
        if ll_data is None:
            return None, None

        # 1. Walkway check for pedestrians
        if is_pedestrian:
            for wp in ll_data["walkway_polys"]:
                if cv2.pointPolygonTest(wp.astype(np.float32),
                                        (float(wx), float(wy)), False) >= 0:
                    return None, None

        # 2. Lanelet2 findNearest + heading filter
        from lanelet2.core import BasicPoint2d
        from lanelet2.geometry import findNearest

        candidates = findNearest(ll_data["lanelet_map"].laneletLayer,
                                 BasicPoint2d(wx, wy), 5)
        best_ll = None
        best_score = float('inf')
        for dist, ll in candidates:
            if dist > 15.0:
                continue
            if yaw is not None:
                ll_heading = self._lanelet_heading(ll)
                hdiff = abs(((yaw - ll_heading + math.pi) % (2 * math.pi)) - math.pi)
                if hdiff > math.pi / 2:
                    continue  # skip opposing direction
                score = dist + hdiff * 5.0
            else:
                score = dist
            if score < best_score:
                best_score = score
                best_ll = ll

        if best_ll is None:
            return None, None

        # Pedestrians far from any lane — not on road
        if is_pedestrian and best_score > 3.0:
            return None, None

        # 3. Chain-based grouping
        chain = ll_data["chain_of"].get(best_ll.id)
        if chain is not None:
            return "lane", f"chain_{chain}"
        return None, None

    @staticmethod
    def _lanelet_heading(ll):
        """Get overall heading of a lanelet from its centerline."""
        cl = ll.centerline
        if len(cl) < 2:
            return 0.0
        return math.atan2(cl[-1].y - cl[0].y, cl[-1].x - cl[0].x)

    # Lanelet2 origin coordinates per nuScenes location
    _LANELET2_ORIGINS = {
        "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
        "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
        "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
        "boston-seaport": [42.336849169438615, -71.05785369873047],
    }

    def _get_lanelet2_data(self, location):
        """Load Lanelet2 map and build chain grouping. Cached per location."""
        if not hasattr(self, "_ll2_cache"):
            self._ll2_cache = {}
        if location in self._ll2_cache:
            return self._ll2_cache[location]

        try:
            import lanelet2
            from lanelet2.io import Origin, load
            from lanelet2.projection import UtmProjector

            if location not in self._LANELET2_ORIGINS:
                print(f"  WARNING: No Lanelet2 origin for {location}")
                self._ll2_cache[location] = None
                return None

            # Find .osm file
            osm_path = None
            for search_dir in [
                self.maps_dir.parent / "lanelet2",
                self.maps_dir / "lanelet2",
            ]:
                candidate = search_dir / f"{location}.osm"
                if candidate.exists():
                    osm_path = candidate
                    break

            if osm_path is None:
                print(f"  WARNING: Lanelet2 .osm not found for {location}")
                self._ll2_cache[location] = None
                return None

            lat, lon = self._LANELET2_ORIGINS[location]
            proj = UtmProjector(Origin(lat, lon))
            lanelet_map = load(str(osm_path), proj)
            tr = lanelet2.traffic_rules.create(
                lanelet2.traffic_rules.Locations.Germany,
                lanelet2.traffic_rules.Participants.Vehicle,
            )
            routing_graph = lanelet2.routing.RoutingGraph(lanelet_map, tr)

            # Identify road lanelets (not lane connectors)
            road_ids = set()
            for ll in lanelet_map.laneletLayer:
                attrs = dict(ll.attributes)
                if attrs.get("subtype") == "road":
                    road_ids.add(ll.id)

            # Chain road lanelets via following/previous (stop at connectors)
            chain_of = {}
            cid = 0
            for ll in lanelet_map.laneletLayer:
                if ll.id not in road_ids or ll.id in chain_of:
                    continue
                chain = set()
                stack = [ll]
                while stack:
                    cur = stack.pop()
                    if cur.id in chain or cur.id not in road_ids:
                        continue
                    chain.add(cur.id)
                    for n in list(routing_graph.following(cur)) + \
                             list(routing_graph.previous(cur)):
                        if n.id in road_ids and n.id not in chain:
                            stack.append(n)
                cid += 1
                for lid in chain:
                    chain_of[lid] = cid

            # Merge left/right neighbors into same chain
            for ll in lanelet_map.laneletLayer:
                if ll.id not in road_ids:
                    continue
                my_c = chain_of.get(ll.id)
                if not my_c:
                    continue
                for nb in [routing_graph.left(ll), routing_graph.right(ll)]:
                    if nb and nb.id in road_ids:
                        nb_c = chain_of.get(nb.id)
                        if nb_c and nb_c != my_c:
                            for lid in list(chain_of.keys()):
                                if chain_of[lid] == nb_c:
                                    chain_of[lid] = my_c

            # Also assign lane connectors to a chain via their neighbors
            for ll in lanelet_map.laneletLayer:
                if ll.id in road_ids or ll.id in chain_of:
                    continue
                # Find a connected road lanelet
                for n in list(routing_graph.following(ll)) + \
                         list(routing_graph.previous(ll)):
                    if n.id in chain_of:
                        chain_of[ll.id] = chain_of[n.id]
                        break

            n_chains = len(set(chain_of.values()))
            print(f"  Lanelet2 {location}: {len(lanelet_map.laneletLayer)} lanelets, "
                  f"{len(road_ids)} road, {n_chains} chains")

            # Build walkway polygons from nuScenes map
            data = self._load(location)
            walkway_polys = []
            if data:
                raw, polys_map = data["raw"], data["polys"]
                for wk in raw.get("walkway", []):
                    wp = polys_map.get(wk.get("polygon_token"))
                    if wp is not None and len(wp) >= 3:
                        walkway_polys.append(wp)

            result = {
                "lanelet_map": lanelet_map,
                "routing_graph": routing_graph,
                "chain_of": chain_of,
                "walkway_polys": walkway_polys,
            }
            self._ll2_cache[location] = result
            return result

        except ImportError:
            print("  WARNING: lanelet2 package not installed, falling back to polygon containment")
            self._ll2_cache[location] = None
            return None
        except Exception as e:
            print(f"  WARNING: Lanelet2 load failed for {location}: {e}")
            self._ll2_cache[location] = None
            return None


# ── BEV Rendering ─────────────────────────────────────────────────────────

def render_bev(agents_ego, ego_pose, map_renderer=None, location=None,
               agent_radius=RADIUS, ego_speed=0.0, ref_ego_pose=None):
    """Render ego-centric BEV with map + agents (STSBench style).

    agents_ego: list of agent dicts with global coords (gx,gy,gz,qw,qx,qy,qz)
    ego_pose: dict with 'translation' [x,y,z] and 'rotation' [qw,qx,qy,qz]
    ref_ego_pose: if provided, use this as the coordinate frame origin (for temporal BEVs).
                  ego_pose is used for drawing the ego box at its actual position.
    """
    S = BEV_SUPERSAMPLE
    SZ = BEV_SIZE * S
    canvas = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    cx, cy = SZ // 2, SZ // 2

    # Coordinate frame: use ref_ego_pose if provided (temporal BEV), else ego_pose
    frame_pose = ref_ego_pose if ref_ego_pose is not None else ego_pose
    T_wc = make_T(frame_pose["translation"], frame_pose["rotation"])
    T_cw = np.linalg.inv(T_wc)

    # Map center uses frame pose (T=0 ego position for all temporal frames)
    frame_x, frame_y = frame_pose["translation"][0], frame_pose["translation"][1]
    # Ego box uses actual ego_pose position
    ego_x, ego_y = ego_pose["translation"][0], ego_pose["translation"][1]

    def world_to_bev(wx, wy):
        pt = T_cw @ np.array([wx, wy, 0, 1])
        bx = int(cx - pt[1] * SZ / (2 * BEV_RANGE))
        by = int(cy - pt[0] * SZ / (2 * BEV_RANGE))
        return bx, by

    # Draw map (centered on frame pose = T=0 ego)
    if map_renderer and location:
        map_renderer.render_into(canvas, location, frame_x, frame_y,
                                 world_to_bev, BEV_RANGE, S)

    # Colors (1:1 from STSBench)
    COLOR_EGO     = (255, 200, 0)    # blue (matches OpenScene)
    COLOR_VEHICLE = (180, 200, 255)  # peach (BGR)
    COLOR_PED     = (0, 255, 128)    # green
    COLOR_CYCLE   = (0, 200, 200)    # yellow-ish

    def draw_box(gx, gy, gz, qw, qx, qy, qz, l, w, color,
                 thickness=1, fill=False, label_text=None, speed=0.0):
        T_ann = make_T([gx, gy, gz], [qw, qx, qy, qz])
        l2, w2 = l / 2, w / 2
        corners_local = np.array([
            [l2, w2, 0, 1], [l2, -w2, 0, 1],
            [-l2, -w2, 0, 1], [-l2, w2, 0, 1],
        ]).T
        corners_world = (T_ann @ corners_local).T
        corners_bev = np.array([world_to_bev(c[0], c[1]) for c in corners_world],
                               dtype=np.int32)
        th = thickness * S
        if fill:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [corners_bev.reshape(-1, 1, 2)], color)
            cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
        cv2.polylines(canvas, [corners_bev.reshape(-1, 1, 2)], True, color, th)
        # Front edge highlight
        front_pts = np.array([corners_bev[0], corners_bev[1]], dtype=np.int32)
        cv2.polylines(canvas, [front_pts.reshape(-1, 1, 2)], False,
                      tuple(min(255, c + 80) for c in color), th + S)
        # Heading arrow from front edge outward (only for moving agents)
        front_cx = int((corners_bev[0][0] + corners_bev[1][0]) / 2)
        front_cy = int((corners_bev[0][1] + corners_bev[1][1]) / 2)
        back_cx = int((corners_bev[2][0] + corners_bev[3][0]) / 2)
        back_cy = int((corners_bev[2][1] + corners_bev[3][1]) / 2)
        if speed >= 0.5:  # only draw heading for moving agents
            dx_h = front_cx - back_cx
            dy_h = front_cy - back_cy
            h_len = max(1, math.sqrt(dx_h**2 + dy_h**2))
            arrow_len = 18 * S
            ax = int(front_cx + dx_h / h_len * arrow_len)
            ay = int(front_cy + dy_h / h_len * arrow_len)
            arrow_color = tuple(min(255, c + 60) for c in color)
            cv2.arrowedLine(canvas, (front_cx, front_cy), (ax, ay),
                            arrow_color, 2 * S, tipLength=0.35)
        # Label — small text above the box top
        if label_text:
            font_scale = 0.30 * S
            font_th = max(1, S)
            (tw, th_txt), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_th)
            bcx = int(np.mean(corners_bev[:, 0]))
            bcy = int(np.min(corners_bev[:, 1])) - 4 * S
            tx = bcx - tw // 2
            ty = bcy
            cv2.putText(canvas, label_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_th)

    # Save map-only canvas before agents
    map_only_canvas = canvas.copy()

    # Draw ego box
    ego_yaw = quat_to_yaw(*ego_pose["rotation"])
    draw_box(ego_x, ego_y, ego_pose["translation"][2],
             *ego_pose["rotation"], 4.5, 2.0,
             COLOR_EGO, 2, fill=True, label_text="EGO", speed=ego_speed)

    # Draw agent boxes (using global coords + quaternion)
    for ag in agents_ego:
        dist = math.sqrt(ag["x"]**2 + ag["y"]**2)
        if dist > agent_radius:
            continue
        cat = ag["category"]
        if "pedestrian" in cat or "human" in cat:
            color = COLOR_PED
        elif "cycle" in cat:
            color = COLOR_CYCLE
        else:
            color = COLOR_VEHICLE
        draw_box(ag["gx"], ag["gy"], ag["gz"],
                 ag["qw"], ag["qx"], ag["qy"], ag["qz"],
                 ag["l"], ag["w"], color,
                 label_text=ag.get("label", ""), speed=ag.get("speed", 0.0))

    # Legend (bottom-left)
    legend = [
        (COLOR_EGO,     "Ego"),
        (COLOR_VEHICLE, "Vehicles"),
        (COLOR_PED,     "Peds"),
        (COLOR_CYCLE,   "Cyclist"),
    ]
    for i, (c, txt) in enumerate(legend):
        y0 = SZ - 12 * S - i * 14 * S
        cv2.rectangle(canvas, (6 * S, y0 - 8 * S), (16 * S, y0 + 2 * S), c, -1)
        cv2.putText(canvas, txt, (20 * S, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * S, (200, 200, 200), S)

    # Map legend (bottom-right)
    map_legend = [
        ((50, 100, 180),  "Ped Crossing"),
        ((200, 200, 200), "Lane Boundary"),
        ((40, 90, 40),    "Walkway"),
        ((70, 70, 65),    "Lane"),
        ((50, 50, 50),    "Drivable"),
    ]
    for i, (c, txt) in enumerate(map_legend):
        y0 = SZ - 12 * S - i * 14 * S
        xo = SZ - 100 * S
        cv2.rectangle(canvas, (xo, y0 - 8 * S), (xo + 10 * S, y0 + 2 * S), c, -1)
        cv2.putText(canvas, txt, (xo + 14 * S, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30 * S, (200, 200, 200), S)

    # Downsample
    canvas = cv2.resize(canvas, (BEV_SIZE, BEV_SIZE), interpolation=cv2.INTER_AREA)
    map_only = cv2.resize(map_only_canvas, (BEV_SIZE, BEV_SIZE), interpolation=cv2.INTER_AREA)
    return canvas, map_only


# ── Nav Command Inference ─────────────────────────────────────────────────

def infer_nav_command(ego_poses_local):
    """Infer navigation command from ego trajectory heading in ego-local coords.
    Uses final heading direction. Dead zone 0.15 rad (~8.5°). No STOPPED."""
    if len(ego_poses_local) < 3:
        return "GO STRAIGHT"

    dx = ego_poses_local[-1]["x"] - ego_poses_local[-3]["x"]
    dy = ego_poses_local[-1]["y"] - ego_poses_local[-3]["y"]

    if dx**2 + dy**2 < 0.01:
        return "GO STRAIGHT"

    heading = math.atan2(dy, dx)
    if abs(heading) < 0.15:
        return "GO STRAIGHT"
    elif heading > 0:
        return "TURN LEFT"
    else:
        return "TURN RIGHT"


# ── Visibility Filtering ──────────────────────────────────────────────────

def filter_visible_agents(agents_ego, nsd, sample_tokens, t0_idx, ann_index=None):
    """Keep agents visible in at least one camera in ANY of the 4 temporal frames.
    Checks each agent's actual position in each frame (not just T=0 position)."""
    visible_labels = set()
    inst_to_label = {ag["instance_token"]: ag["label"] for ag in agents_ego}
    if ann_index is None:
        ann_index = {}

    for offset in [-3, -2, -1, 0]:
        idx = t0_idx + offset
        if idx < 0 or idx >= len(sample_tokens):
            continue
        st = sample_tokens[idx]
        frame_ego_pose = nsd.get_ego_pose(st)
        if frame_ego_pose is None:
            continue
        T_ego = make_T(frame_ego_pose["translation"], frame_ego_pose["rotation"])
        T_ego_inv = np.linalg.inv(T_ego)

        # Get agents at their actual positions in this frame
        if offset == 0:
            frame_agents = [(ag["gx"], ag["gy"], ag["gz"], ag["label"]) for ag in agents_ego]
        else:
            frame_agents = []
            for ann in ann_index.get(st, []):
                label = inst_to_label.get(ann["instance_token"])
                if label and label not in visible_labels:
                    gx, gy, gz = ann["translation"]
                    frame_agents.append((gx, gy, gz, label))

        for ns_cam in CAM_MAP:
            cal = nsd.get_cam_calibration(st, ns_cam)
            if cal is None:
                continue
            T_sensor_ego = make_T(cal["translation"], cal["rotation"])
            T_cam_inv = np.linalg.inv(T_sensor_ego)
            T_global_cam = T_cam_inv @ T_ego_inv
            K = np.array(cal["camera_intrinsic"])
            fx, fy, cxc, cyc = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            for gx, gy, gz, label in frame_agents:
                if label in visible_labels:
                    continue
                pt_cam = T_global_cam @ np.array([gx, gy, gz, 1.0])
                if pt_cam[2] < 0.5:
                    continue
                u = fx * pt_cam[0] / pt_cam[2] + cxc
                v = fy * pt_cam[1] / pt_cam[2] + cyc
                if 0 <= u < 1600 and 0 <= v < 900:
                    visible_labels.add(label)

    return [ag for ag in agents_ego if ag["label"] in visible_labels]


# ── Main Build Function ───────────────────────────────────────────────────

def build_scene(nsd, ann_index, scene_token, out_dir, map_renderer=None, nusc=None,
                t0_idx=None, sample_idx=0, history_frames=3, future_frames=8):
    """Build scenario record for one nuScenes scene."""
    scene = nsd.scenes[scene_token]
    scene_name = f"nuscenes-{scene['name']}"
    sample_tokens = nsd.scene_samples(scene_token)
    n_samples = len(sample_tokens)

    min_frames = history_frames + future_frames + 1
    if n_samples < min_frames:
        return False

    # T=0 frame selection
    if t0_idx is None:
        t0_idx = min(n_samples - future_frames - 1, max(history_frames, n_samples // 2))
    if t0_idx < history_frames or t0_idx >= n_samples - future_frames:
        return False
    t0_token = sample_tokens[t0_idx]

    # Check all 4 cameras exist on disk at T=0
    for ns_cam in CAM_MAP:
        img_path = nsd.get_cam_image_path(t0_token, ns_cam)
        if img_path is None or not img_path.exists():
            return False

    # ── Ego pose at T=0 ──
    ego_pose_t0 = nsd.get_ego_pose(t0_token)
    if ego_pose_t0 is None:
        return False

    ego_x0, ego_y0 = ego_pose_t0["translation"][0], ego_pose_t0["translation"][1]
    ego_z0 = ego_pose_t0["translation"][2]
    ego_yaw0 = quat_to_yaw(*ego_pose_t0["rotation"])
    cos_e, sin_e = math.cos(-ego_yaw0), math.sin(-ego_yaw0)

    def to_ego(wx, wy):
        dx, dy = wx - ego_x0, wy - ego_y0
        return round(cos_e * dx - sin_e * dy, 4), round(sin_e * dx + cos_e * dy, 4)

    def norm_yaw(yaw):
        y = yaw - ego_yaw0
        return round((y + math.pi) % (2 * math.pi) - math.pi, 4)

    location = nsd.get_location(scene_token)

    # ── Agents at T=0 ──
    anns_t0 = ann_index.get(t0_token, [])
    agents_ego = []
    track_labels = {}  # instance_token → label
    cat_counters = defaultdict(int)

    # First pass: scan ALL 4 temporal frames for eligible agents.
    # An agent is eligible if it passes visibility + radius in ANY frame.
    # This ensures temporal stability — agents don't pop in/out between frames.
    eligible_instances = set()
    cam_visible = {}  # (inst_tok, sample_tok) → bool — for camera bbox gating
    for offset in [-3, -2, -1, 0]:
        fidx = t0_idx + offset
        if fidx < 0 or fidx >= n_samples:
            continue
        fst = sample_tokens[fidx]
        for ann in ann_index.get(fst, []):
            inst_tok = ann["instance_token"]
            cat = nsd.inst_cat.get(inst_tok, "unknown")
            if cat in SKIP_CATEGORIES:
                continue
            vis_tok = ann.get("visibility_token", "4")
            is_pedestrian = "pedestrian" in cat
            min_vis = MIN_VISIBILITY_PEDESTRIAN if is_pedestrian else MIN_VISIBILITY_VEHICLE
            passes_vis = vis_tok >= min_vis
            gx, gy, _ = ann["translation"]
            ex, ey = to_ego(gx, gy)
            r = RADIUS_PED if is_pedestrian else RADIUS
            passes_radius = math.sqrt(ex**2 + ey**2) <= r
            cam_visible[(inst_tok, fst)] = passes_vis and passes_radius
            if passes_vis and passes_radius:
                eligible_instances.add(inst_tok)

    # Collect T=0 data for eligible agents
    eligible = []
    for ann in anns_t0:
        inst_tok = ann["instance_token"]
        cat = nsd.inst_cat.get(inst_tok, "unknown")
        if cat in SKIP_CATEGORIES or inst_tok not in eligible_instances:
            continue
        gx, gy, gz = ann["translation"]
        ex, ey = to_ego(gx, gy)
        eligible.append((ann, cat, gx, gy, gz, ex, ey))

    # Second pass: assign sequential labels (no gaps)
    for ann, cat, gx, gy, gz, ex, ey in eligible:
        ann_yaw = quat_to_yaw(*ann["rotation"])
        yaw_ego = norm_yaw(ann_yaw)

        # Size: [width, length, height] in nuScenes
        w, l, h = ann["size"]

        # Assign label
        prefix = _cat_prefix(cat)
        cat_counters[prefix] += 1
        label = f"{prefix}_{cat_counters[prefix]}"
        track_labels[ann["instance_token"]] = label

        # Speed: check attribute first (parked/stopped → 0), else use box_velocity
        ann_attrs = {nsd.attr_name.get(t, "") for t in ann.get("attribute_tokens", [])}
        if ann_attrs & nsd.STATIC_ATTRS:
            speed = 0.0
        else:
            vel = nusc.box_velocity(ann["token"])  # [vx, vy, vz] in global frame
            if any(np.isnan(vel)):
                speed = 0.0
            else:
                speed = round(math.sqrt(vel[0]**2 + vel[1]**2), 3)

        qw, qx, qy, qz = ann["rotation"]
        agents_ego.append({
            "label": label, "category": cat,
            "x": ex, "y": ey, "z": round(gz - ego_z0, 2),
            "yaw": yaw_ego, "l": l, "w": w, "h": h,
            "speed": speed,
            "instance_token": ann["instance_token"],
            # Global coords for accurate camera projection
            "gx": gx, "gy": gy, "gz": gz,
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
        })

    # Visibility filtering
    agents_ego = filter_visible_agents(agents_ego, nsd, sample_tokens, t0_idx, ann_index)
    if not agents_ego:
        return False

    # ── Output directories ──
    scene_dir = out_dir / scene_name / f"SAMPLED_{sample_idx}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # ── Save camera images for each timestep ──
    for offset, slug in TIMESTEP_OFFSETS.items():
        idx = t0_idx + offset
        if idx < 0 or idx >= n_samples:
            continue
        st = sample_tokens[idx]
        slug_dir = scene_dir / slug
        slug_dir.mkdir(parents=True, exist_ok=True)

        # Build per-frame agents for camera bbox — only if visible in this frame
        frame_anns = ann_index.get(st, []) if offset != 0 else anns_t0
        frame_agents = []
        for ann in frame_anns:
            inst_tok = ann["instance_token"]
            if inst_tok not in track_labels:
                continue
            # Camera bbox: only draw if passes visibility + radius in THIS frame
            if not cam_visible.get((inst_tok, st), False):
                continue
            gx, gy, gz = ann["translation"]
            qw, qx, qy, qz = ann["rotation"]
            w_ann, l_ann, h_ann = ann["size"]
            frame_agents.append({
                "label": track_labels[inst_tok],
                "category": nsd.inst_cat.get(inst_tok, "unknown"),
                "gx": gx, "gy": gy, "gz": gz,
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "l": l_ann, "w": w_ann, "h": h_ann,
            })

        for ns_cam, our_cam in CAM_MAP.items():
            img_path = nsd.get_cam_image_path(st, ns_cam)
            dst_path = slug_dir / f"{our_cam}.jpg"
            if img_path is None or not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Draw bboxes for agents visible in THIS frame's camera
            cam_calib = nsd.get_cam_calibration(st, ns_cam)
            frame_ego_pose = nsd.get_ego_pose(st)
            if cam_calib and frame_ego_pose and frame_agents:
                bboxes = project_boxes_to_cam(frame_agents, cam_calib, frame_ego_pose)
                draw_cam_bboxes(img, bboxes)

            cv2.imwrite(str(dst_path), img)

    # ── Ego speed (computed before BEV so we can draw heading arrow) ──
    ego_speed_t0 = 0.0
    ego_pose_prev = nsd.get_ego_pose(sample_tokens[max(0, t0_idx - 1)])
    if ego_pose_prev:
        dt = (nsd.samples[t0_token]["timestamp"] -
              nsd.samples[sample_tokens[max(0, t0_idx - 1)]]["timestamp"]) / 1e6
        if dt > 0:
            dx = ego_pose_t0["translation"][0] - ego_pose_prev["translation"][0]
            dy = ego_pose_t0["translation"][1] - ego_pose_prev["translation"][1]
            ego_speed_t0 = round(math.sqrt(dx**2 + dy**2) / dt, 3)

    # ── Render BEV (T=0 + temporal past frames) ──
    # All frames use T=0 ego as the coordinate frame origin (map stays fixed).
    # Ego and agents appear at their actual positions at each timestep.
    tp0_dir = scene_dir / "Tp0p0"
    tp0_dir.mkdir(parents=True, exist_ok=True)
    bev, bev_map = render_bev(
        agents_ego, ego_pose=ego_pose_t0,
        map_renderer=map_renderer, location=location,
        agent_radius=RADIUS, ego_speed=ego_speed_t0,
    )
    cv2.imwrite(str(tp0_dir / "bev.png"), bev)
    cv2.imwrite(str(tp0_dir / "bev_map.png"), bev_map)

    # Temporal BEVs: ego + agents at their actual past positions, T=0 ego frame
    for offset, slug in [(-3, "Tm1p5"), (-2, "Tm1p0"), (-1, "Tm0p5")]:
        idx = t0_idx + offset
        if idx < 0 or idx >= n_samples:
            continue
        st = sample_tokens[idx]
        past_ego_pose = nsd.get_ego_pose(st)
        if past_ego_pose is None:
            continue

        # Compute ego speed at this past frame
        past_ego_speed = 0.0
        if idx > 0:
            prev_ep = nsd.get_ego_pose(sample_tokens[idx - 1])
            if prev_ep:
                dt = (nsd.samples[st]["timestamp"] -
                      nsd.samples[sample_tokens[idx - 1]]["timestamp"]) / 1e6
                if dt > 0:
                    dx = past_ego_pose["translation"][0] - prev_ep["translation"][0]
                    dy = past_ego_pose["translation"][1] - prev_ep["translation"][1]
                    past_ego_speed = round(math.sqrt(dx**2 + dy**2) / dt, 3)

        # Build past agents (only those visible at T=0)
        frame_anns = ann_index.get(st, [])
        past_agents = []
        for ann in frame_anns:
            inst_tok = ann["instance_token"]
            if inst_tok not in track_labels:
                continue
            gx, gy, gz = ann["translation"]
            ex, ey = to_ego(gx, gy)
            dist = math.sqrt(ex**2 + ey**2)
            if dist > RADIUS * 1.5:  # slightly larger radius for past positions
                continue
            qw, qx, qy, qz = ann["rotation"]
            w_ann, l_ann, h_ann = ann["size"]
            spd = 0.0
            if ann.get("next"):
                next_ann = nsd.annotations.get(ann["next"])
                if next_ann:
                    dt_s = (nsd.samples[next_ann["sample_token"]]["timestamp"] -
                            nsd.samples[ann["sample_token"]]["timestamp"]) / 1e6
                    if dt_s > 0:
                        dx = next_ann["translation"][0] - ann["translation"][0]
                        dy = next_ann["translation"][1] - ann["translation"][1]
                        spd = round(math.sqrt(dx**2 + dy**2) / dt_s, 3)
            past_agents.append({
                "label": track_labels[inst_tok],
                "category": nsd.inst_cat.get(inst_tok, "unknown"),
                "x": ex, "y": ey, "z": round(gz - ego_z0, 2),
                "gx": gx, "gy": gy, "gz": gz,
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "l": l_ann, "w": w_ann, "h": h_ann,
                "speed": spd,
            })

        slug_dir = scene_dir / slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        # Use past ego pose for ego box position, but T=0 ego pose for the
        # coordinate frame (world_to_bev transform) so map stays fixed
        past_bev, past_bev_map = render_bev(
            past_agents, ego_pose=past_ego_pose,
            map_renderer=map_renderer, location=location,
            agent_radius=RADIUS * 1.5, ego_speed=past_ego_speed,
            ref_ego_pose=ego_pose_t0,
        )
        cv2.imwrite(str(slug_dir / "bev.png"), past_bev)
        cv2.imwrite(str(slug_dir / "bev_map.png"), past_bev_map)

    ego_history = []
    for offset in [-3, -2, -1]:
        idx = t0_idx + offset
        if idx < 0:
            continue
        ep = nsd.get_ego_pose(sample_tokens[idx])
        if ep is None:
            continue
        ex, ey = to_ego(ep["translation"][0], ep["translation"][1])
        # Speed from displacement
        spd = 0.0
        if idx > 0:
            ep_prev = nsd.get_ego_pose(sample_tokens[idx - 1])
            if ep_prev:
                dt = (nsd.samples[sample_tokens[idx]]["timestamp"] -
                      nsd.samples[sample_tokens[idx - 1]]["timestamp"]) / 1e6
                if dt > 0:
                    dx = ep["translation"][0] - ep_prev["translation"][0]
                    dy = ep["translation"][1] - ep_prev["translation"][1]
                    spd = round(math.sqrt(dx**2 + dy**2) / dt, 3)
        ego_history.append({"t": offset * 0.5, "x": ex, "y": ey, "speed": spd})

    ego_future = []
    for offset in range(1, future_frames + 1):
        idx = t0_idx + offset
        if idx >= n_samples:
            continue
        ep = nsd.get_ego_pose(sample_tokens[idx])
        if ep is None:
            continue
        ex, ey = to_ego(ep["translation"][0], ep["translation"][1])
        spd = 0.0
        if idx > 0:
            ep_prev = nsd.get_ego_pose(sample_tokens[idx - 1])
            if ep_prev:
                dt = (nsd.samples[sample_tokens[idx]]["timestamp"] -
                      nsd.samples[sample_tokens[idx - 1]]["timestamp"]) / 1e6
                if dt > 0:
                    dx = ep["translation"][0] - ep_prev["translation"][0]
                    dy = ep["translation"][1] - ep_prev["translation"][1]
                    spd = round(math.sqrt(dx**2 + dy**2) / dt, 3)
        ego_future.append({"t": offset * 0.5, "x": ex, "y": ey, "speed": spd})

    # Nav command from future trajectory
    nav_command = infer_nav_command(ego_future) if ego_future else "GO STRAIGHT"

    # Lane IDs removed — LLM infers spatial relationships from BEV + state.json positions

    # ── Agent history ──
    t0_instance_set = {ag["instance_token"] for ag in agents_ego}
    agent_hist = defaultdict(list)
    for offset in [-3, -2, -1]:
        idx = t0_idx + offset
        if idx < 0:
            continue
        st = sample_tokens[idx]
        for ann in ann_index.get(st, []):
            if ann["instance_token"] not in t0_instance_set:
                continue
            gx, gy = ann["translation"][0], ann["translation"][1]
            ex, ey = to_ego(gx, gy)
            # Speed
            spd = 0.0
            if ann.get("next"):
                next_ann = nsd.annotations.get(ann["next"])
                if next_ann:
                    dt_s = (nsd.samples[next_ann["sample_token"]]["timestamp"] -
                            nsd.samples[ann["sample_token"]]["timestamp"]) / 1e6
                    if dt_s > 0:
                        dx = next_ann["translation"][0] - ann["translation"][0]
                        dy = next_ann["translation"][1] - ann["translation"][1]
                        spd = round(math.sqrt(dx**2 + dy**2) / dt_s, 3)
            agent_hist[ann["instance_token"]].append(
                {"t": offset * 0.5, "x": ex, "y": ey, "speed": spd})

    # ── Build agents dict for state.json ──
    agents_out = {}
    for ag in agents_ego:
        entry = {
            "category": ag["category"],
            "position": {"x": ag["x"], "y": ag["y"], "yaw": ag["yaw"]},
            "speed": ag["speed"],
        }
        hist = sorted(agent_hist.get(ag["instance_token"], []), key=lambda h: h["t"])
        if hist:
            entry["history"] = hist
        agents_out[ag["label"]] = entry

    # ── state.json ──
    state = {
        "t_rel_s": 0.0, "nav_command": nav_command,
        "ego": {"speed": ego_speed_t0,
                "history": ego_history, "future": ego_future},
        "agents": agents_out,
    }
    with open(tp0_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2)

    # ── state_desc.json (natural language version of state for LLM prompts) ──
    from common.scene_utils import state_to_natural_language
    with open(tp0_dir / "state_desc.json", "w") as f:
        json.dump({"description": state_to_natural_language(state)}, f, indent=2)

    # ── uuid_map.json (label → instance_token) ──
    uuid_map = {label: inst_tok for inst_tok, label in track_labels.items()}
    with open(tp0_dir / "uuid_map.json", "w") as f:
        json.dump(uuid_map, f, indent=2)

    # ── calib.json ──
    calib_out = {}
    for ns_cam, std_cam in CAM_MAP.items():
        cal = nsd.get_cam_calibration(t0_token, ns_cam)
        if cal is None:
            continue
        K = np.array(cal["camera_intrinsic"])
        # Extrinsics: sensor → ego transform
        T_sensor_ego = make_T(cal["translation"], cal["rotation"])
        T_ego_sensor = np.linalg.inv(T_sensor_ego)
        R_ego2cam = T_ego_sensor[:3, :3]
        t_ego2cam = T_ego_sensor[:3, 3]
        calib_out[std_cam] = {
            "intrinsic": K.tolist(),
            "R_ego2cam": R_ego2cam.tolist(),
            "t_ego2cam": t_ego2cam.tolist(),
            "image_size": [1600, 900],
        }
    with open(tp0_dir / "calib.json", "w") as f:
        json.dump(calib_out, f, indent=2)

    # ── meta.json ──
    meta = {
        "scene": scene_name, "maneuver_type": f"SAMPLED_{sample_idx}", "nav_command": nav_command,
        "dataset": "nuscenes", "nuscenes_scene": scene["name"],
        "location": location, "t0_sample_token": t0_token,
        "t0_frame_idx": t0_idx, "sample_idx": sample_idx,
        "timestamp": nsd.samples[t0_token]["timestamp"],
        "n_agents": len(agents_out),
    }
    with open(scene_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return True


# ── Parallel worker ───────────────────────────────────────────────────────

# Module-level globals set by _init_worker (shared via fork)
_g_nsd = _g_ann_index = _g_nusc = _g_map_renderer = None


def _init_worker(nsd, ann_index, nusc, map_renderer):
    global _g_nsd, _g_ann_index, _g_nusc, _g_map_renderer
    _g_nsd, _g_ann_index, _g_nusc, _g_map_renderer = nsd, ann_index, nusc, map_renderer


def _process_one_scene(job):
    """Process all samples for one nuScenes scene. Returns (n_samples, scene_name)."""
    scene, out_dir, history_frames, future_frames, sampling_step, samples_filter, verbose = job
    nsd, ann_index, nusc, map_renderer = _g_nsd, _g_ann_index, _g_nusc, _g_map_renderer

    scene_name = f"nuscenes-{scene['name']}"
    if verbose:
        print(f"Processing {scene_name}...")
    sample_tokens = nsd.scene_samples(scene["token"])
    n_samples_total = len(sample_tokens)
    min_frames = history_frames + future_frames + 1
    if n_samples_total < min_frames:
        return 0, scene_name

    t0_start = history_frames
    t0_end = n_samples_total - future_frames
    t0_indices = list(range(t0_start, t0_end, sampling_step))

    wanted_indices = samples_filter.get(scene_name) if samples_filter else None
    n_samples = 0
    for sample_idx, t0_idx in enumerate(t0_indices):
        if wanted_indices is not None and sample_idx not in wanted_indices:
            continue
        if build_scene(nsd, ann_index, scene["token"], out_dir, map_renderer,
                      nusc=nusc, t0_idx=t0_idx, sample_idx=sample_idx,
                      history_frames=history_frames,
                      future_frames=future_frames):
            n_samples += 1
    if verbose and n_samples:
        print(f"    → {n_samples} samples from {scene_name}")
    return n_samples, scene_name


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config_loader import load_config
    cfg = load_config("nuscenes")

    parser = argparse.ArgumentParser(description="Build scenario records from nuScenes data")
    parser.add_argument("--data", default=cfg.get("data"), help="Path to nuScenes root")
    parser.add_argument("--version", default=cfg.get("version", "v1.0-trainval"), help="Metadata version folder")
    parser.add_argument("--maps", default=cfg.get("maps"), help="Path to HD map expansion/ directory")
    parser.add_argument("--records", default=cfg.get("records", "nuscenes_records"), help="Output records directory")
    parser.add_argument("--num-scenes", type=int, default=cfg.get("num_scenes", 10))
    parser.add_argument("--sampling-step", type=int, default=cfg.get("sampling_step", 6))
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per scene (overrides sampling-step with dynamic step)")
    parser.add_argument("--history-frames", type=int, default=cfg.get("history_frames", 3))
    parser.add_argument("--future-frames", type=int, default=cfg.get("future_frames", 8))
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Specific scene names to build (e.g. scene-0001 scene-0023)")
    parser.add_argument("--samples-file", default=None,
                        help="File with scene/SAMPLED_N lines to build only those exact samples")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Parse samples file: scene_dir/SAMPLED_N → {scene_dir: {sample_idx, ...}}
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
        # Auto-derive --scenes from samples file
        if not args.scenes:
            args.scenes = [s.replace("nuscenes-", "") for s in samples_filter]
        print(f"  Samples file: {sum(len(v) for v in samples_filter.values())} samples across {len(samples_filter)} scenes")

    out_dir = Path(args.records)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading nuScenes metadata ...")
    nsd = NuScenesData(args.data, args.version)
    ann_index = build_ann_index(nsd)
    print(f"  {len(nsd.scenes)} scenes, {len(nsd.annotations)} annotations")
    print(f"  Sampling: step={args.sampling_step}, history={args.history_frames}, future={args.future_frames}")

    # Official NuScenes devkit for box_velocity
    from nuscenes.nuscenes import NuScenes as NuScenesDevkit
    print("Loading NuScenes devkit (for velocity) ...")
    nusc = NuScenesDevkit(version=args.version, dataroot=args.data, verbose=False)

    map_renderer = NuScenesMapBEV(args.maps) if args.maps else None

    # Sort scenes by name for reproducibility
    scene_list = sorted(nsd.scenes.values(), key=lambda s: s["name"])

    # Filter to specific scenes if requested
    if args.scenes:
        scene_filter = set(args.scenes)
        scene_list = [s for s in scene_list if s["name"] in scene_filter]
        print(f"  Filtering to {len(scene_list)} scenes: {sorted(scene_filter)}")

    # Build job list
    jobs = []
    skipped = 0
    for scene in scene_list:
        if not args.scenes and (len(jobs) + skipped) >= args.num_scenes:
            break
        scene_name = f"nuscenes-{scene['name']}"
        if (out_dir / scene_name).exists() and not args.force:
            print(f"  SKIP {scene_name}: already exists (use --force)")
            skipped += 1
            continue
        jobs.append((scene, out_dir, args.history_frames, args.future_frames,
                     args.sampling_step, samples_filter))

    print(f"  {len(jobs)} scenes to process, {skipped} skipped")

    # Set globals for the main process (used by workers via fork, or directly if serial)
    _init_worker(nsd, ann_index, nusc, map_renderer)

    from tqdm import tqdm

    if args.workers > 1:
        parallel_jobs = [j + (False,) for j in jobs]
        from multiprocessing import Pool
        with Pool(args.workers, initializer=_init_worker,
                  initargs=(nsd, ann_index, nusc, map_renderer)) as pool:
            results = list(tqdm(pool.imap(_process_one_scene, parallel_jobs), total=len(jobs), desc="Building scenes"))
    else:
        verbose_jobs = [j + (True,) for j in jobs]
        results = [_process_one_scene(job) for job in verbose_jobs]

    built = sum(1 for n, _ in results if n > 0)
    total_samples = sum(n for n, _ in results)
    print(f"\nDone. {built} scenes, {total_samples} samples built → {out_dir}")


if __name__ == "__main__":
    main()
