# Causal Scene Graph Benchmark Pipeline

End-to-end pipeline for generating **causal scene graphs**, **visual question-answering (VQA)**, and **counterfactual/activation trajectories** from autonomous driving datasets.

**Supported Datasets**: nuScenes, OpenScene (NAVSIM), Argoverse V2

**Pipeline**: Raw Data → Record Building → Causal Graph (Stage 1) → Pruning → QA Generation → CF/AC Trajectories → HTML Visualization

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Configuration](#2-configuration)
3. [Data Acquisition](#3-data-acquisition)
4. [Record Building](#4-record-building)
5. [Pipeline Overview](#5-pipeline-overview)
6. [Running the Batch Pipeline](#6-running-the-batch-pipeline)
7. [Graph Pruning](#7-graph-pruning)
8. [HTML Visualization](#8-html-visualization)
9. [Directory Structure](#9-directory-structure)
10. [Data Formats](#10-data-formats)
11. [Cost Estimates](#11-cost-estimates)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

### Conda Environment

```bash
conda env create -f environment.yaml
conda activate causal
```

Key packages: `anthropic`, `nuscenes-devkit`, `opencv-python`, `pillow`, `shapely`, `numpy`, `networkx`, `pyyaml`.

### System Dependencies

```bash
sudo apt install graphviz    # Required for graph PNG rendering
```

---

## 2. Configuration

All paths and settings are in `config.yaml` at the repo root:

```bash
cp config.yaml.example config.yaml
# Edit with your paths and API keys
```

Key settings:

```yaml
provider: claude                    # openai | gemini | claude
model: claude-opus-4-6
reasoning: none                     # none | low | medium | high

sampling_step: 8                    # frames between samples (at 2Hz: 8 = 4s)

batch:
  batch_dir: /path/to/batch_jobs    # where batch .jsonl files go
  results_dir: /path/to/batch_results

nuscenes:
  data: /path/to/nuscenes
  maps: /path/to/maps/expansion
  records: /path/to/records/nuscenes
  outputs: /path/to/outputs/nuscenes
```

CLI arguments override config values when provided.

---

## 3. Data Acquisition

### 3.1 nuScenes

Download from https://www.nuscenes.org/:
- **Full dataset (v1.0-trainval)**: metadata JSONs + camera images
- **Map expansion pack**: HD map JSONs with lane geometry

### 3.2 OpenScene (NAVSIM)

Three components — use download scripts in `scripts/download/`:

```bash
bash scripts/download/download_openscene_metadata.sh <LOGS_DIR>
bash scripts/download/download_openscene_sensor.sh
bash scripts/download/download_nuplan_maps.sh
```

### 3.3 Argoverse V2

Download from https://www.argoverse.org/av2.html (sensor split, train and/or val set).

---

## 4. Record Building

Record building converts raw data into a **unified record structure** consumed by all downstream steps.

### Record Format

```
<RECORDS>/<scene_name>/
├── SAMPLED_0/                        # First sample in scene
│   ├── meta.json
│   ├── Tm1p5/                        # T = -1.5s
│   │   ├── bev.png                   # Temporal BEV with heading arrows
│   │   ├── cam_front.jpg
│   │   ├── cam_front_left.jpg
│   │   ├── cam_front_right.jpg
│   │   └── cam_back.jpg
│   ├── Tm1p0/                        # T = -1.0s
│   │   └── [bev.png + 4 cameras]
│   ├── Tm0p5/                        # T = -0.5s
│   │   └── [bev.png + 4 cameras]
│   └── Tp0p0/                        # T = 0.0s (present)
│       ├── bev.png                   # BEV (1536x1536) with heading arrows
│       ├── cam_front.jpg             # Annotated with 2D bboxes + labels
│       ├── cam_front_left.jpg
│       ├── cam_front_right.jpg
│       ├── cam_back.jpg
│       ├── state.json                # Ego + agent states (ego-local coords)
│       ├── calib.json                # Camera intrinsics + extrinsics
│       └── uuid_map.json             # Short labels → dataset UUIDs
├── SAMPLED_1/
│   └── ...
└── SAMPLED_N/
```

### Key Features

- **BEV**: 1536x1536, rendered at 3072 then downsampled (supersample=2)
- **Heading arrows**: Drawn on moving agents (speed > 0.1) and ego in all BEV frames
- **Temporal BEVs**: 4 frames (T=-1.5, T=-1.0, T=-0.5, T=0) — all in T=0 ego coordinate frame
- **Camera bboxes**: Drawn per-frame using that frame's camera FOV
- **uuid_map.json**: Maps short labels (car_1) → original dataset UUIDs for planner integration
- **No lane_id**: Removed from state.json — LLM infers spatial relationships from BEV + positions
- **Visibility filter** (nuScenes): Vehicles >=40% visible, pedestrians >=60% visible
- **Radius**: Vehicles 30m, Pedestrians 25m (all datasets)
- **Occlusion filter**: Multi-ray 3D raycast (9 rays, 80% threshold, vehicles-only blockers, camera z-depth)
- **AV2**: 5 cameras (front, front_left, front_right, back_right, back_left)
- **Images > 2000px**: Auto-resized at batch prep time (Claude multi-image limit)

### Build Commands

```bash
conda activate causal

# nuScenes (120 scenes, ~4 samples each)
python scripts/build_records/build_nuscenes_records.py \
  --data /path/to/nuscenes \
  --maps /path/to/maps/expansion \
  --records /path/to/records/nuscenes \
  --num-scenes 120

# OpenScene (100 scenes, 4 samples each via quartile sampling)
python scripts/build_records/build_openscene_records.py \
  --logs /path/to/navsim_logs \
  --blobs /path/to/openscene \
  --maps /path/to/openscene/maps \
  --records /path/to/records/openscene \
  --num-scenes 100 --max-samples 4

# Argoverse V2 (133 scenes, ~3 samples each, 5 cameras)
python scripts/build_records/build_argoverse2_records.py \
  --data /path/to/argoverse2/sensor \
  --split val \          # train, val, or both
  --records /path/to/records/argoverse2 \
  --num-scenes 133
```

All builders skip existing scenes by default (use `--force` to overwrite).

Additional flags:
- `--scenes scene-0001 scene-0023` (nuScenes) or `--scenes openscene-... argoverse2-...` — build only specific scenes
- `--samples-file /path/to/samples.txt` — build only specific scene/SAMPLED_N pairs (one per line)
- `--max-samples 4` — quartile sampling (OpenScene only)
- `--workers N` — parallel scene processing (default: 4)

### Regenerate CF BEV Overlays

After CF trajectories are generated, create BEV + camera overlay images:

```bash
python scripts/rendering/regenerate_cf_bevs.py \
  --outputs /path/to/outputs/nuscenes \
  --records /path/to/records/nuscenes
```

For observation_window (alternate history) CFs, uses T=-1.5 BEV and camera instead of T=0.

### Dry-Run Validation

Before committing to a paid batch run, validate the full pipeline:

```bash
bash scripts/dry_run_validate.sh
```

Checks: records exist, pruned graphs valid, prompt files present, batch JSONL generation works, image counts correct, cost estimate.

---

## 5. Pipeline Overview

The pipeline has been simplified to remove Stage 2 (discriminator/refinement). The new flow:

```
Stage 1 (Batch API)  →  Pruning (local)  →  QA Stages  →  Trajectories
     ↓                       ↓                  ↓              ↓
  raw .json            _pruned.json         QA .json      trajectory .json
```

### Stages

| Stage | Name | Description |
|-------|------|-------------|
| **1** | Generator | Images + state.json → raw causal graph |
| **prune** | Pruning | Reverse-BFS reachability: nodes with no causal path to ego → distractors |
| **3a** | Causal Ladder QA | Active edges → CaI, CA, AR, CB, CoI, BAS, CDE, CR, SC, NC, NIE, NDE |
| **3b** | Dormant Nodes QA | Dormant edges → DQ, EL, AB, AC question types |
| **3c** | Distractor Nodes QA | Distractors → NI, CI, WC, DR question types |
| **4cf** | CF Trajectories | Questions with trajectory_changes=true from all QA sources |
| **5** | LLM Ego Trajectory | Agent interventions (remove/reposition/speed_change) → LLM-predicted ego response |

### Stage 4cf — CF Trajectory Routing

Stage 4cf scans `ladder_qa.json`, `dl_qa.json`, and `distractor_qa.json` for questions with `trajectory_changes=true`. Three routes:

| Route | Condition | Prompt | Output |
|-------|-----------|--------|--------|
| **nuplan_config** | `entity_type=agent` | `nuplan_config/` | Agent intervention config JSON → `nuplan_configs.json` |
| **llm_t0** | `entity_type=infrastructure, anchor=t0` | `alternate_trajectories/Intervention_*_Prompt.md` | Ego CF trajectory (future WP1-8) |
| **llm_obs_window** | `entity_type=infrastructure, anchor=observation_window` | `alternate_trajectories/CounterFactual_*_Prompt.md` | Ego CF trajectory (past WP1-4) |

### Stage 5 — LLM Ego Trajectory (Agent Interventions)

Stage 5 processes the agent intervention configs produced by Stage 4cf (`nuplan_configs.json`) and asks an LLM to predict how the ego vehicle would respond. Use `--nuplan` to run the nuPlan simulator instead (Stage 4np).

| Route | Condition | Prompt | Output |
|-------|-----------|--------|--------|
| **T0** | `apply_at_t == 0.0` | `agent_intervention/CF_Ego_Trajectory_T0_*` | 8 future waypoints → `llm_ego_trajectory.json` |
| **Past** | `apply_at_t < 0` | `agent_intervention/CF_Ego_Trajectory_Past_*` | 4 past waypoints → `llm_ego_trajectory.json` |

Camera ordering (`IMAGE_ORDER_BLOCK`) is built dynamically — no separate av2 prompt files needed.

```bash
# LLM ego trajectory (default)
bash scripts/batch_processing/run_batch_pipeline.sh argoverse2 --provider claude

# nuPlan simulator instead
bash scripts/batch_processing/run_batch_pipeline.sh argoverse2 --provider claude --nuplan
```

### Image Order (v5 prompts)

**nuScenes/OpenScene** — 4 cams per timestep:
```
T=-1.5: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, BEV
T=-1.0 → T=-0.5 → T=0: [same 5 per timestep]
```

**AV2** — 5 cams per timestep:
```
T=-1.5: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, CAM_BACK_LEFT, BEV
T=-1.0 → T=-0.5 → T=0: [same 6 per timestep]
```

### Prompt Routing

All datasets share the same prompt templates. AV2-specific differences (camera count, names, image ordering) are injected at load time via variables — no separate `av2_prompts/` directories.

| Dataset | Causal Graph | QA | CF Trajectory | nuPlan Config | LLM Ego Trajectory |
|---------|-------------|-----|---------------|---------------|--------------------|
| nuScenes | `causal_scene_graph/v5/` | `causal_qas/v5/` | `alternate_trajectories/` | `nuplan_config/` | `agent_intervention/` |
| OpenScene | `causal_scene_graph/v5/` | `causal_qas/v5/` | `alternate_trajectories/` | `nuplan_config/` | `agent_intervention/` |
| AV2 | `causal_scene_graph/v5/` (AV2 vars) | `causal_qas/v5/` (AV2 vars) | `alternate_trajectories/` (AV2 vars) | `nuplan_config/` (AV2 vars) | `agent_intervention/` (dynamic cams) |

### Causal Graph Structure

- **5 node types**: TRAFFIC_CONTROL, ROAD_OBSTACLE, OTHER_AGENT, ENVIRONMENT, EGO
- **5 ego-edge types**: CONSTRAINS_LONGITUDINAL, CONSTRAINS_LATERAL, INHIBITS_PROCEED, GRANTS_PROCEED, DEGRADES
- **2 inter-node edge types**: AFFECTS (directed), CONFOUNDED_BY (bidirected)
- **3 causal statuses**: active_sustaining (L2), active_triggering (L3), dormant
- **Distractors array**: Excluded entities with rejection reasons

---

## 6. Running the Batch Pipeline

The batch pipeline uses the Claude Batch API for cost-effective large-scale processing.

### Full Pipeline

```bash
cd neurips-dataset

bash scripts/batch_processing/run_batch_pipeline.sh nuscenes \
  --provider claude \
  --records-dir /path/to/records/nuscenes \
  --output-dir /path/to/outputs/nuscenes \
  --batch-dir /path/to/batch_jobs
```

### Key Flags

| Flag | Description |
|------|-------------|
| `--provider claude` | LLM provider (claude, openai, gemini) |
| `--reasoning medium` | Claude adaptive thinking (none/low/medium/high) |
| `--records-dir DIR` | Override records directory |
| `--output-dir DIR` | Override output directory |
| `--batch-dir DIR` | Where batch .jsonl and results are stored (**use separate dir per run**) |
| `--stop-after 1` | Stop after Stage 1 (graph generation only) |
| `--stop-after prune` | Stop after pruning |
| `--stop-after 3c` | Stop after all QA stages |
| `--stop-after 4cf` | Stop after CF trajectories |
| `--stop-after 5` | Stop after LLM ego trajectory stage |
| `--nuplan` | Use nuPlan simulator for agent interventions (Stage 4np) instead of LLM (Stage 5) |
| `--resume 3a` | Resume from Stage 3a (skip graph generation) |
| `--resume 4cf` | Resume from CF trajectory generation |
| `--max-samples N` | Process only N samples |

### Before Re-running a Stage

**Always nuke the batch dir** to avoid reprocessing old results:
```bash
rm -rf /path/to/batch_jobs
mkdir -p /path/to/batch_jobs
```

### Examples

```bash
# Stage 1 only (generate raw graphs)
bash scripts/batch_processing/run_batch_pipeline.sh nuscenes \
  --provider claude \
  --records-dir /path/to/records/nuscenes \
  --output-dir /path/to/outputs/nuscenes \
  --batch-dir /path/to/batch_jobs \
  --stop-after 1

# Full pipeline through QA
bash scripts/batch_processing/run_batch_pipeline.sh nuscenes \
  --provider claude \
  --records-dir /path/to/records/nuscenes \
  --output-dir /path/to/outputs/nuscenes \
  --batch-dir /path/to/batch_jobs \
  --stop-after 3c

# Resume from QA (graphs already exist)
bash scripts/batch_processing/run_batch_pipeline.sh nuscenes \
  --provider claude \
  --records-dir /path/to/records/nuscenes \
  --output-dir /path/to/outputs/nuscenes \
  --batch-dir /path/to/batch_jobs \
  --resume 3a
```

---

## 7. Graph Pruning

After Stage 1, the pruning script removes nodes that have no causal path to ego using reverse-BFS on the edge graph.

### Algorithm

1. Reverse all edges (ego_edges: node→ego becomes ego→node; AFFECTS: A→B becomes B→A; CONFOUNDED_BY: bidirectional)
2. BFS from ego through the reversed graph
3. Any node not visited has no causal influence on ego → moved to distractors

Complexity: O(V + E), parallelized across files with ProcessPoolExecutor.

### Standalone Usage

```bash
# Dry run (show what would be pruned)
python scripts/causal_graph_extraction/prune_graphs.py /path/to/outputs/nuscenes --dry-run

# Run pruning (creates *_pruned.json alongside each raw .json)
python scripts/causal_graph_extraction/prune_graphs.py /path/to/outputs/nuscenes

# Custom worker count
python scripts/causal_graph_extraction/prune_graphs.py /path/to/outputs/nuscenes --workers 8
```

Pruning is automatically run as part of the batch pipeline between Stage 1 and Stage 3a.

---

## 8. HTML Visualization

Renders interactive HTML pages with D3.js causal graphs, camera images, BEV, and QA cards.

Pass `--dataset` and the script reads output/records paths directly from `config.yaml`:

```bash
# Render, generate index, and serve (canonical dataset dirs from config.yaml)
python scripts/rendering/render_causal_graph.py \
  --dataset nuscenes \
  --html --index --root-index \
  --serve 9888

# Override dirs for a non-canonical output folder
python scripts/rendering/render_causal_graph.py \
  --dataset nuscenes \
  --output-dir /path/to/outputs/nuscenes_50 \
  --records-dir /path/to/records/nuscenes_50 \
  --html --index --force
```

Companion files (`_pruned.json`, `_changes.json`, `_verification.json`, `nuplan_intervention_manifest.json`) are skipped automatically.

Each HTML page has an **Original / Pruned toggle** — Pruned is active by default. The JSON viewer at the bottom shows the causal graph JSON and, if present, the verification JSON in separate tabs.

The server serves from the parent of the outputs directory so that `records/` (image source) is reachable at the same host. Image paths in the HTML are relative — no files are copied to an `assets/` directory.

---

## 9. Directory Structure

### Outputs Directory

```
<OUTPUTS>/<dataset>/
├── <scene>_SAMPLED_0.json              # Raw causal graph (Stage 1)
├── <scene>_SAMPLED_0_pruned.json       # Pruned graph (used by QA stages)
├── <scene>_SAMPLED_0.html              # Interactive HTML viewer
├── <scene>_SAMPLED_0.png               # Static graph PNG
├── <scene>_SAMPLED_0_verification.json # Graph verification results (if run)
│
├── <scene>_SAMPLED_0/                  # QA results directory
│   ├── ladder_qa.json                  # Active causal questions (Causal Ladder)
│   ├── dl_qa.json                      # Dormant node questions
│   ├── distractor_qa.json              # Distractor node questions
│   ├── cf_trajectories/                # CF trajectory outputs
│   └── ac_trajectories/                # AC trajectory outputs
│
├── verification_report.json            # Aggregate verification report
└── index.html                          # Landing page linking all scenes
```

### Release / Downloads Directory

Packaged with `scripts/release/package_samples.py` (see section 9a below).

```
<downloads_dir>/<dataset>/
└── <scene>/
    └── <SAMPLED_N>/
        ├── meta.json           # Scene metadata (token, log, location, timestamp)
        ├── state.json          # Agent states at T=0 (ego + all agents)
        ├── calib.json          # Camera calibration (intrinsics + extrinsics)
        ├── frames.json         # Original raw sensor image paths per timestep (Tm1p5 … Tp0p0)
        ├── graph.json          # Pruned causal graph (falls back to raw if pruned missing)
        └── qa/
            ├── active_qa.json      # Causal Ladder questions
            ├── dormant_qa.json     # Dormant node questions
            └── distractor_qa.json  # Distractor node questions
```

### Batch Directory

```
<BATCH_DIR>/
├── nuscenes_stage1_generator.jsonl     # Batch request files
├── nuscenes_stage3a_ladder.jsonl
├── ...
└── batch_results/                      # Downloaded batch results
    ├── <batch_id>_results.jsonl
    └── ...
```

---

## 9a. Release Packaging

`scripts/release/package_samples.py` assembles the final release structure from records and pipeline outputs.

```bash
python scripts/release/package_samples.py --dataset nuscenes
```

Reads `records` and `outputs` from `config.yaml` for the given dataset. Output lands in `downloads_dir/<outputs_basename>/` (e.g. `downloads/nuscenes_full/`), also from `config.yaml`.

**Options:**
```
--dataset   nuscenes | openscene | argoverse2   (resolves paths from config.yaml)
--records   Override records directory
--outputs   Override pipeline outputs directory
--download  Override release download directory
```

**`frames.json`** contains paths to the **original raw sensor images** (not the annotated copies in records). Paths are relative to `data_root` (set in `config.yaml`). Source resolution per dataset:

| Dataset | Raw Image Source |
|---------|-----------------|
| nuScenes | `nuscenes.data` + sample_data filenames from nuScenes metadata JSONs |
| OpenScene | `openscene.blobs` + `data_path` from the scene pkl file |
| Argoverse2 | `meta.json.raw_data_dir` + closest camera timestamp from `annotations.feather` |

The script writes a `manifest.json` at the root listing all samples and flagging any with missing graph or QA files.

---

## 10. Data Formats

### state.json

```json
{
  "t_rel_s": 0.0,
  "nav_command": "GO STRAIGHT",
  "ego": {
    "speed": 4.058,
    "history": [{"t": -1.5, "x": -6.82, "y": -0.27, "speed": 4.53}, ...],
    "future": [{"t": 0.5, "x": 2.07, "y": -0.03, "speed": 4.14}, ...]
  },
  "agents": {
    "car_1": {
      "category": "vehicle.car",
      "position": {"x": 12.5, "y": -11.3, "yaw": 1.65},
      "speed": 0.0,
      "history": [{"t": -1.5, "x": 12.66, "y": -11.33, "speed": 0.11}, ...]
    }
  }
}
```

**Coordinates**: Ego-local at T=0. `+x = forward`, `+y = left`. Ego at origin.
History: 3 steps (t=-1.5, -1.0, -0.5). Future: 6 steps (t=0.5 to 3.0).

### Causal Graph JSON (Stage 1 output)

```json
{
  "scene_id": "...",
  "scene_context": {"road_geometry": "...", "navigation_intent": "..."},
  "ego_state": {"speed_mps": "4.0", "inferred_acceleration": "...", ...},
  "nodes": [
    {"id": "ego", "type": "EGO", ...},
    {"id": "n1", "type": "TRAFFIC_CONTROL", "label": "Green signal", ...}
  ],
  "ego_edges": [
    {"source": "n1", "target": "ego", "effect_type": "GRANTS_PROCEED",
     "active": true, "causal_status": "active_sustaining", "justification": "..."}
  ],
  "inter_node_edges": [...],
  "distractors": [
    {"id": "car_2", "rejection_reason": "TREE 5 Q8: ..."}
  ]
}
```

Full schema: `prompts/causal_scene_graph/v5/system_prompt.md`

### Prompt Versions

| Version | Location | Images | Description |
|---------|----------|--------|-------------|
| v4 | `prompts/causal_scene_graph/v4/` | 16 cameras + 1 BEV | Current stable |
| v5 | `prompts/causal_scene_graph/v5/` | 16 cameras + 4 temporal BEVs | Latest with temporal BEVs |

QA prompts: `prompts/causal_qas/v5/` (3 prompt pairs: Causal_Ladder, Dormant_Nodes, Distractor_Nodes)
CF trajectory prompts: `prompts/alternate_trajectories/` (Intervention + CounterFactual prompt pairs)
nuPlan config prompts: `prompts/nuplan_config/` (system + user prompt pair)
LLM ego trajectory prompts: `prompts/agent_intervention/` (T0 + Past, system + user prompt pairs)

---

## 11. Cost Estimates

Using Claude Opus 4.6 Batch API ($2.50/MTok input, $12.50/MTok output):

| Setting | Avg Input | Avg Output | Cost/Sample |
|---------|-----------|------------|-------------|
| No thinking (4 cams) | ~46K | ~3.5K | ~$0.16 |
| Medium thinking (4 cams) | ~50K | ~9K | ~$0.24 |
| Medium thinking (AV2, 5 cams) | ~55K | ~9K | ~$0.25 |

| Scale | No Thinking | Medium Thinking |
|-------|-------------|-----------------|
| 400 samples | ~$64 | ~$96 |
| 1,000 samples | ~$160 | ~$240 |
| 12,000 samples | ~$1,920 | ~$2,880 |

---

## 12. Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `max_tokens` hit (truncated JSON) | Response too long | Currently set to 65536 in `llm_provider.py` |
| Image dimensions exceed 2000px | AV2 cameras are 2048x1550 | Auto-resized at batch prep time in `scene_utils.py` |
| Old batch results reprocessed | Batch dir not cleaned | Nuke entire `--batch-dir` before re-running |
| No Causal Ladder questions | All ego_edges `active: false` | Check graph quality; Ladder needs active edges |
| No Dormant Node questions | All ego_edges `active: true` | Normal — dormant QA needs dormant edges |
| Missing heading arrows in temporal BEV | `speed: 0.0` for past agents | Rebuild records (fixed: computes actual speed) |
| `_pruned.json` not found by QA | Pruning not run | Run `python scripts/causal_graph_extraction/prune_graphs.py <output_dir>` |
| Duplicate HTML files | `_pruned.json` rendered as standalone | Render script auto-filters `_pruned` files |
| `argparse: invalid choice 'v5'` | Old prepare_batches.py | Pull latest — v5 prompt version added to choices |
| OpenScene scene name dots | Claude batch API rejects dots in custom_id | Handled: dots replaced with `-` in sanitizer |
| Empty BEV (no lanes/roads) | `--maps` not provided | Re-run record builder with map path |

### Notes

- `--reasoning none` is the default and recommended setting
- The LLM outputs analysis text before JSON — `extract_json()` handles markdown fences + fallback brace search
- Batch API chunks large requests (>20) into multiple batches submitted in parallel
- Claude returns `active: "false"` (string) — handled in render + QA scripts

---

## Graph Post-Processing

Scripts in `scripts/graph_postprocessing/` for cleaning up annotated causal graphs after human review. Produces `{scene_id}_verified.json` — the canonical graph for downstream QA, trajectories, and release.

**Pipeline:** Backup → Filter Edges → Align Descriptions → Verified JSON

### Directory Arguments

| Argument | Path | Contents |
|----------|------|----------|
| `--input-dir` | `outputs/nuscenes_full_edit/` | Graph JSONs: `*.json` (original), `*_pruned.json`, `*_edited.json`, `index.html` |
| `--edits-dir` | `edits/` | Per-user bookmarks: `{username}/bookmarks.json` with good/needs_review flags |
| `--records-dir` | `records/nuscenes_full/` | Scene records: `{scene}/{sample}/Tp0p0/state.json`, `state_desc.json`, images |

### Full Pipeline

```bash
# 1. Backup (never modify originals)
cp -r outputs/nuscenes_full_edit/ outputs/nuscenes_full_edit_backup/

# 2. Filter edges (modifies backup in place)
#    - Removes dormant edges from vehicles behind ego
#    - Removes crosswalk ego-edges when pedestrian already has active INHIBITS_PROCEED
#    - Skips needs_review, uses assigned annotator's _edited.json or _pruned.json if good
python scripts/graph_postprocessing/fix_causal_graphs.py \
  --input-dir outputs/nuscenes_full_edit_backup/ \
  --edits-dir edits/ \
  --dry-run                 # remove --dry-run to apply

# 3. Align scene descriptions (batch API, only _edited.json from active annotators)
#    Sends original graph + edited graph + state_desc.json to Claude Sonnet
#    Rewrites scene_context and behavior_explanation using only active edges
python scripts/graph_postprocessing/align_scene_description.py submit \
  --input-dir outputs/nuscenes_full_edit_backup/ \
  --edits-dir edits/ \
  --records-dir records/nuscenes_full/

# 4. Check batch status
python scripts/graph_postprocessing/align_scene_description.py status \
  --input-dir outputs/nuscenes_full_edit_backup/

# 5. Apply results → writes {scene_id}_verified.json in same dir
#    - Edited scenes: filtered graph + aligned NL → verified.json
#    - Good-pruned scenes: filtered graph copied as verified.json
python scripts/graph_postprocessing/align_scene_description.py apply \
  --input-dir outputs/nuscenes_full_edit_backup/ \
  --edits-dir edits/
```

### Test Before Full Run

```bash
# Test alignment on 5 complex scenes (real-time API, no batch)
# Writes per-sample dirs with comparison.json + BEV + camera images
python scripts/graph_postprocessing/align_scene_description.py test \
  --input-dir outputs/nuscenes_full_edit_backup/ \
  --edits-dir edits/ \
  --records-dir records/nuscenes_full/ \
  --out-dir /path/to/comparison_output/
```

### Scene Assignment

Each scene is assigned to one active annotator. Only that annotator's `_edited.json` is used. Assignments are defined in `fix_causal_graphs.py:ACTIVE_ASSIGNMENTS` (mirrored from `editor_server.py`).

| Annotator | Scene Range (0-indexed) |
|-----------|------------------------|
| naren | 0–95, 803–850 |
| navvrat | 95–190, 756–803 |
| aman | 190–285 |
| siddharth | 285–380 |
| abhinav | 380–474 |
| rajarshee | 474–568, 707–756 |
| umesh | 568–707 |

### Output

`{scene_id}_verified.json` — same schema as edited graphs, with:
- Filtered edges (dormant behind-vehicle removed, redundant crosswalk removed)
- Aligned `scene_context.road_geometry` and `navigation_intent`
- Aligned `ego_state.behavior_explanation` (active edges only)
- No username in filename — ready for downstream scripts
