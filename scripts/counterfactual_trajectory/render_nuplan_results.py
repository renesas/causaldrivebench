#!/usr/bin/env python3
"""
render_nuplan_results.py — Parse nuPlan sim results, draw trajectories on BEV, copy videos.

Reads results.json from nuPlan counterfactual simulation, matches each entry to
our cf_trajectories/ output dirs, and:
1. Draws original (green) + CF (red) ego trajectories on BEV → bev_nuplan_overlay.png
2. Copies simulation videos (original + CF) → sim_original.mp4, sim_cf.mp4

Usage:
    python render_nuplan_results.py \
        --results /path/to/nuplan/results.json \
        --nuplan-base /path/to/nuplan-devkit \
        --records-base records \
        --outputs-base outputs
"""

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

# ── BEV Constants ────────────────────────────────────────────────────────
BEV_SIZE = 1536
BEV_RANGE = 50.0
BEV_SCALE = BEV_SIZE / (2 * BEV_RANGE)
BEV_CENTER = BEV_SIZE // 2

COLOR_ORIG = (0, 255, 0)     # green
COLOR_CF = (0, 0, 255)       # red
COLOR_NUPLAN = (255, 165, 0) # orange — nuplan CF (to distinguish from LLM CF)


def ego_to_bev(x, y):
    px = int(BEV_CENTER - y * BEV_SCALE)
    py = int(BEV_CENTER - x * BEV_SCALE)
    return px, py


def parse_trajectory_txt(path):
    """Parse nuPlan trajectory.txt → list of [t, x, y, yaw, speed]."""
    waypoints = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                waypoints.append({
                    "t": float(parts[0]),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "yaw": float(parts[3]),
                    "speed": float(parts[4]),
                })
    return waypoints


def draw_trajectory(bev, waypoints, color, radius=5, thickness=2, label=None):
    """Draw trajectory dots + line on BEV."""
    pts = []
    for wp in waypoints:
        px, py = ego_to_bev(wp["x"], wp["y"])
        pts.append((px, py))
        cv2.circle(bev, (px, py), radius, color, -1)
    for i in range(len(pts) - 1):
        cv2.line(bev, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)
    if label and pts:
        cv2.putText(bev, label, (pts[0][0] + 10, pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def question_to_slug(question, max_len=80):
    """Convert question text to directory slug (matching prepare_batches.py)."""
    slug = question[:max_len]
    slug = re.sub(r'[^a-zA-Z0-9_\- ]', '', slug)
    slug = slug.replace(' ', '_')
    return slug


_key_to_slug = {}

def find_cf_dir(cf_base, question, key=None):
    """Find the cf_trajectories subdirectory matching a question."""
    # Try manifest key→slug first (most reliable)
    if key and key in _key_to_slug:
        slug = _key_to_slug[key]
        candidate = cf_base / slug
        if candidate.exists():
            return candidate

    # Fallback: derive slug from question
    slug = question_to_slug(question)
    candidate = cf_base / slug
    if candidate.exists():
        return candidate
    # Fuzzy match
    slug_prefix = slug[:40]
    for d in cf_base.iterdir():
        if d.is_dir() and d.name.startswith(slug_prefix):
            return d
    return None


def get_dataset_suffix(dataset):
    """Map dataset name to our _50 suffix."""
    return f"{dataset}_50"


def process_entry(entry, nuplan_base, records_base, outputs_base):
    """Process a single nuplan result entry."""
    if entry["status"] != "ok":
        return False

    dataset = entry["dataset"]
    scene_id = entry["scene"]
    question = entry["question"]
    outputs = entry.get("outputs", {})

    ds_suffix = get_dataset_suffix(dataset)
    outputs_dir = outputs_base / ds_suffix
    records_dir = records_base / ds_suffix

    # Find our cf_trajectories dir for this scene + question
    cf_base = outputs_dir / scene_id / "cf_trajectories"
    if not cf_base.exists():
        print(f"  SKIP {scene_id}: no cf_trajectories dir")
        return False

    q_dir = find_cf_dir(cf_base, question, key=entry.get("key"))
    if not q_dir:
        print(f"  SKIP: can't match question slug for: {question[:60]}...")
        return False

    # Find the scene subdir (may be q_dir/<scene_id>/ or q_dir/)
    scene_subdir = q_dir / scene_id if (q_dir / scene_id).exists() else q_dir

    # Parse scene_id → records path for BEV
    parts = scene_id.rsplit("_SAMPLED_", 1)
    if len(parts) != 2:
        return False
    rec_scene_dir = records_dir / parts[0] / f"SAMPLED_{parts[1]}"

    # Load BEV
    bev_path = rec_scene_dir / "Tp0p0" / "bev.png"
    if not bev_path.exists():
        print(f"  SKIP: no BEV at {bev_path}")
        return False
    bev = cv2.imread(str(bev_path))

    # Parse trajectories
    cf_outputs = outputs.get("counterfactual", {})
    orig_outputs = outputs.get("original", {})

    orig_traj_path = nuplan_base / cf_outputs.get("trajectory_original", "")
    cf_traj_path = nuplan_base / cf_outputs.get("trajectory", "")

    orig_wps = parse_trajectory_txt(orig_traj_path) if orig_traj_path.exists() else []
    cf_wps = parse_trajectory_txt(cf_traj_path) if cf_traj_path.exists() else []

    # Draw on BEV
    overlay = bev.copy()
    if orig_wps:
        draw_trajectory(overlay, orig_wps, COLOR_ORIG, radius=5, thickness=3, label="Original")
    if cf_wps:
        draw_trajectory(overlay, cf_wps, COLOR_NUPLAN, radius=4, thickness=2, label="nuPlan CF")

    # Add legend
    h = overlay.shape[0]
    cv2.putText(overlay, f"nuPlan: {entry['planner']} | {entry['intervention']}",
                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"Green=original ({entry['orig_dist']:.1f}m) | Orange=CF ({entry['cf_dist']:.1f}m) | delta={entry['delta']:.1f}m",
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(str(scene_subdir / "bev_nuplan_trajectory.png"), overlay)

    # Copy videos
    cf_video = nuplan_base / cf_outputs.get("video", "")
    orig_video = nuplan_base / orig_outputs.get("video", "")

    if cf_video.exists():
        shutil.copy2(str(cf_video), str(scene_subdir / f"sim_{entry['planner']}_cf.mp4"))
    if orig_video.exists():
        shutil.copy2(str(orig_video), str(scene_subdir / f"sim_{entry['planner']}_original.mp4"))

    # Save nuplan result metadata
    nuplan_meta = {
        "planner": entry["planner"],
        "intervention": entry["intervention"],
        "orig_dist": entry["orig_dist"],
        "cf_dist": entry["cf_dist"],
        "delta": entry["delta"],
        "original_trajectory": orig_wps,
        "counterfactual_trajectory": cf_wps,
    }
    with open(scene_subdir / "nuplan_result.json", "w") as f:
        json.dump(nuplan_meta, f, indent=2)

    return True


def load_manifests(outputs_base):
    """Load all stage4cf manifests to build key→slug lookup."""
    manifest_base = Path("nuscences_batchjobs")
    key_to_slug = {}
    for ds in ["nuscenes_50", "openscene_50", "argoverse2_50"]:
        ds_short = ds.split("_")[0]
        mf = manifest_base / ds / f"{ds_short}_stage4cf_manifest.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            for k, v in m.items():
                key_to_slug[k] = v.get("slug", "")
    return key_to_slug


def main():
    parser = argparse.ArgumentParser(description="Render nuPlan counterfactual results")
    parser.add_argument("--results", required=True, help="Path to nuplan results.json")
    parser.add_argument("--nuplan-base", required=True, help="Path to nuplan-devkit root")
    parser.add_argument("--records-base", default="records")
    parser.add_argument("--outputs-base", default="outputs")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    nuplan_base = Path(args.nuplan_base)
    records_base = Path(args.records_base)
    outputs_base = Path(args.outputs_base)

    # Load manifests for key→slug mapping
    global _key_to_slug
    _key_to_slug = load_manifests(outputs_base)
    print(f"Loaded {len(_key_to_slug)} manifest entries for slug matching")

    print(f"Processing {len(results)} nuPlan results")

    ok = 0
    for entry in results:
        if process_entry(entry, nuplan_base, records_base, outputs_base):
            ok += 1

    print(f"\nDone. Rendered {ok}/{len(results)} entries.")
    print(f"  (Skipped {len(results) - ok} — failed status or missing data)")


if __name__ == "__main__":
    main()
