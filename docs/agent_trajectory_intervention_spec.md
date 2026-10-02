# Agent-to-Ego Trajectory Intervention Specification

## Overview

When the causal QA pipeline generates counterfactual questions about **agents** (not infrastructure), we need a motion planner (nuPlan) to re-simulate ego's trajectory under the modified scene. This document specifies every intervention type, what data is available, and what the planner needs to execute.

---

## Two Sources of Agent Interventions

### 1. CR (Counterfactual Reasoning) — from `ladder_qa.json`

**What it asks:** "If this active agent were removed/changed, would ego drive differently?"

**Metadata location:** `meta.cr_intervention`
```json
{
  "entity_type": "agent",
  "entity_id": "truck_2",
  "intervention": "remove",
  "trajectory_changes": true,
  "triggering_window": null | {"effect_timestamp": "-1.0", "committed_maneuver": "lane_offset_right"}
}
```

**Additional metadata:**
- `meta.behavioral_impact`: `trajectory_changed | speed_changed | no_change`
- `meta.effect_types`: `["CONSTRAINS_LATERAL"]` or `["CONSTRAINS_LONGITUDINAL"]`
- `meta.causal_status`: `active_sustaining | active_triggering`
- `meta.redundancy_check`: `unique | redundant_with:<element_id>`
- `correct_answer`: `Yes` (no change) or `No` (trajectory changes)

### 2. AC (Activation Counterfactual) — from `dl_qa.json`

**What it asks:** "If this dormant agent activated, would ego's trajectory change?"

**Metadata location:** `meta.ac_intervention`
```json
{
  "entity_type": "agent",
  "entity_id": "trailer_1",
  "trajectory_changes": true
}
```

**Additional metadata:**
- `meta.effect_types`: `["CONSTRAINS_LATERAL"]` or `["CONSTRAINS_LONGITUDINAL"]`
- `meta.causal_status`: `dormant`
- `meta.latency_mechanism`: `shielded | decoupled | correlated | N/A`
- `activation_plausibility`: `High | Medium | Low`
- `correct_answer`: `Yes` (trajectory changes) or `No`

---

## Intervention Categories

### Category A: Remove Active Agent (CR, intervention="remove")

**Ego is currently responding to this agent. Remove it and re-plan.**

| Field | Value |
|-------|-------|
| Source | `ladder_qa.json` → CR question |
| `cr_intervention.intervention` | `"remove"` |
| `cr_intervention.entity_type` | `"agent"` |
| Planner action | Delete agent UUID from scene at T=0, re-run ego planner |
| Apply at | T=0 (unless `triggering_window` specifies earlier timestamp) |

#### Subtypes by causal_status:

**A1: Active-Sustaining + trajectory_changes=true (LIVE CONSTRAINT)**
Agent is actively constraining ego right now. Removing it changes ego's behavior.

```
Example: scene-0001_SAMPLED_3
  Question: "If the flatbed truck ahead were removed, would you still be maintaining your rightward offset?"
  Answer: No
  Agent: truck_2 at (8.3, 4.6), yaw=-0.01, speed=0.0, category=vehicle.truck
  Effect: CONSTRAINS_LATERAL — truck encroaches into ego's lane, ego steers right
  behavioral_impact: trajectory_changed

  → PLANNER: Remove truck_2 (UUID from uuid_map.json) at T=0
  → EXPECTED: Ego centers in lane instead of maintaining rightward offset
  → COMPARE: original ego.future vs re-planned trajectory
```

```
Example: scene-0171_SAMPLED_2
  Question: "If the parked semi-trailer on your right were no longer there, would you still be decelerating?"
  Answer: No
  Agent: trailer_1 at (3.5, -3.7), yaw=-3.14, speed=0.0, category=vehicle.trailer
  Effect: CONSTRAINS_LONGITUDINAL — trailer narrows road, ego slows to pass
  behavioral_impact: speed_changed

  → PLANNER: Remove trailer_1 at T=0
  → EXPECTED: Ego maintains speed instead of decelerating
```

```
Example: scene-0038_SAMPLED_0
  Question: "If the parked flatbed truck ahead on the left were no longer there, would you still be holding your rightward position?"
  Answer: No
  Agent: truck_1 at (19.1, 3.5), yaw=0.12, speed=0.0, category=vehicle.truck
  Effect: CONSTRAINS_LATERAL
  behavioral_impact: trajectory_changed

  → PLANNER: Remove truck_1 at T=0
  → EXPECTED: Ego centers in lane
```

**A2: Active-Sustaining + trajectory_changes=false (REDUNDANT)**
Agent constrains ego, but another element provides the same constraint. Removing this agent alone doesn't change behavior.

```
Example: scene-0001_SAMPLED_2
  Question: "If the flatbed truck ahead were removed, would you still drive at reduced speed?"
  Answer: Yes (no change)
  Agent: truck_1 at (14.1, 4.2), speed=0.0
  Effect: CONSTRAINS_LONGITUDINAL
  redundancy: redundant_with:pedestrian_2
  behavioral_impact: no_change

  → PLANNER: Remove truck_1 at T=0
  → EXPECTED: Ego trajectory identical — pedestrian_2 still constrains speed
  → VALIDATE: Compare trajectories, confirm no meaningful deviation
```

**A3: Active-Triggering (RETROSPECTIVE)**
Agent caused ego's current maneuver earlier in the observation window, but removing it NOW doesn't change the committed maneuver.

```
Example: scene-0001_SAMPLED_3
  Question: "If the construction truck behind you had not been parked there as you approached, would you still have the same rightward offset?"
  Answer: Yes (no change)
  Agent: truck_1 at (-6.1, 4.1), speed=0.0 (now behind ego)
  Effect: CONSTRAINS_LATERAL
  triggering_window: {"effect_timestamp": "-1.0", "committed_maneuver": "lane_offset_right"}
  redundancy: redundant_with:truck_2
  behavioral_impact: no_change

  → PLANNER: Two options:
    Option 1: Remove truck_1 at T=0 → trajectory unchanged (maneuver already committed)
    Option 2: Remove truck_1 at t=-1.0 (effect_timestamp) → ego may not have offset right
  → For validation: Option 1 confirms "Yes" answer
  → For counterfactual generation: Option 2 shows what WOULD have been different
```

```
Example: scene-0038_SAMPLED_2
  Question: "If the parked flatbed truck you just passed had not been there as you approached?"
  Answer: Yes (no change)
  Agent: truck_2 at (-3.8, 3.5), speed=0.0 (now behind ego)
  triggering_window: {"effect_timestamp": "-1.0", "committed_maneuver": "lane_offset_right"}
  redundancy: redundant_with:truck_1

  → PLANNER: Remove at T=0 → no change (truck_1 still ahead constrains)
```

---

### Category B: Activate Dormant Agent (AC, entity_type="agent")

**Agent is NOT currently affecting ego. Simulate it becoming active and re-plan.**

This is harder than removal — the planner needs to know HOW the agent activates. The activation scenario is described in the question text.

#### Activation Types:

**B1: Tailgating / Gap Closing (follower activates)**
Dormant follower behind ego suddenly closes gap.

```
Example: scene-0177_SAMPLED_1
  Question: "If the white sedan behind you had suddenly accelerated and closed to within a couple of meters, would your stopping trajectory have changed?"
  Answer: Yes
  Agent: car_1 at (-16.0, -0.2), yaw=0.11, speed=6.251 (behind ego, same lane)
  Effect: CONSTRAINS_LONGITUDINAL (dormant — not yet constraining)

  → PLANNER ACTION:
    1. Modify car_1's trajectory: accelerate from 6.25 m/s, close gap to ~2m behind ego
    2. Re-plan ego with aggressive follower
  → EXPECTED: Ego brakes less aggressively (can't hard-brake with tailgater)

  → INTERVENTION SPECIFICATION:
    {
      "target_agent_uuid": uuid_map["car_1"],
      "modification": "accelerate_and_close",
      "target_gap_m": 2.0,
      "new_speed_mps": 8.0
    }
```

```
Example: scene-0204_SAMPLED_0
  Question: "If the dark sedan behind you had suddenly accelerated to match your speed and closed to within one car-length?"
  Answer: Yes
  Agent: car_10 at (-15.0, 0.2), yaw=0.01, speed=4.782
  Effect: CONSTRAINS_LONGITUDINAL (dormant)

  → PLANNER: Close car_10 to 1 car-length (~5m) behind ego, match ego speed
  → EXPECTED: Ego moderates acceleration to maintain safe gap
```

**B2: Lane Intrusion (parked/adjacent agent enters ego lane)**

```
Example: scene-0171_SAMPLED_2
  Question: "If you had been drifting right instead of holding straight, would your trajectory have been different because of the parked trailer alongside you?"
  Answer: Yes
  Agent: trailer_1 at (3.5, -3.7), yaw=-3.14, speed=0.0 (parked right of ego)
  Effect: CONSTRAINS_LATERAL (dormant — ego not drifting toward it)

  → PLANNER ACTION:
    1. Modify EGO trajectory to drift right
    2. Check if trailer_1 constrains that drift
  → NOTE: This is more of an ego-modification than agent-modification

  → ALTERNATIVE: Modify trailer_1 to encroach into ego's lane
    {
      "target_agent_uuid": uuid_map["trailer_1"],
      "modification": "shift_lateral",
      "shift_y_m": 2.0,  // move 2m toward ego's lane (leftward)
    }
```

**B3: Speed Constraint Activation (dormant speed influence becomes binding)**

```
Example: scene-0171_SAMPLED_0
  Question: "If the parked trailer had started limiting your safe approach speed, would your acceleration trajectory have been different?"
  Answer: Yes
  Agent: trailer_1 at (9.3, -2.7), speed=0.082
  Effect: CONSTRAINS_LONGITUDINAL (dormant — currently only CONSTRAINS_LATERAL is active)

  → PLANNER ACTION:
    1. This is a conceptual activation — the trailer's lateral presence
       should also create a speed constraint
    2. Add speed limit of ~15 km/h when passing the trailer position
  → EXPECTED: Ego decelerates more before passing trailer
```

---

## Data Available Per Intervention

### From Records (`records/{dataset}/{scene}/SAMPLED_X/`)

```
Tp0p0/
├── state.json       ← Ego + all agents at T=0 (position, speed, yaw, history, future)
├── uuid_map.json    ← {"car_1": "original-dataset-uuid-..."}
├── bev.png          ← BEV at T=0 (1536×1536)
├── calib.json       ← Camera calibration
├── cam_front.jpg    ← Camera images at T=0
├── cam_front_left.jpg
├── cam_front_right.jpg
├── cam_back.jpg
Tm1p5/bev.png        ← Temporal BEV at t=-1.5s
Tm1p0/bev.png        ← Temporal BEV at t=-1.0s
Tm0p5/bev.png        ← Temporal BEV at t=-0.5s
meta.json            ← Scene token, timestamp, location, dataset
```

### From Outputs (`outputs/{dataset}/{scene}_SAMPLED_X/`)

```
ladder_qa.json       ← CR questions with cr_intervention metadata
dl_qa.json           ← AC questions with ac_intervention metadata
{scene}_SAMPLED_X_refined.json  ← Full causal graph
```

### Agent State (from state.json)

```json
{
  "car_1": {
    "category": "vehicle.car",
    "position": {"x": -16.0, "y": -0.2, "yaw": 0.11},
    "speed": 6.251,
    "history": [
      {"t": -1.5, "x": -22.5, "y": -0.3, "speed": 5.8},
      {"t": -1.0, "x": -19.8, "y": -0.3, "speed": 6.0},
      {"t": -0.5, "x": -17.9, "y": -0.2, "speed": 6.1}
    ]
  }
}
```

### UUID Mapping (from uuid_map.json)

```json
{
  "car_1": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
  "truck_1": "b2c3d4e5-6789-01bc-defg-2345678901bc"
}
```

---

## Planner Input/Output Contract

### Input to Planner

```json
{
  "dataset": "nuscenes | argoverse2 | openscene",
  "scene_token": "from meta.json → t0_sample_token",
  "timestamp_ns": "from meta.json → timestamp",
  "location": "from meta.json → location",

  "intervention": {
    "type": "remove | modify_trajectory | modify_state",
    "target_agent_uuid": "original-dataset-uuid",
    "target_agent_label": "car_1",
    "apply_at_s": 0.0,

    "remove": {},

    "modify_trajectory": {
      "new_waypoints": [
        {"t": 0.0, "x": -16.0, "y": -0.2, "yaw": 0.11, "speed": 6.25},
        {"t": 0.5, "x": -12.0, "y": -0.2, "yaw": 0.11, "speed": 8.0},
        {"t": 1.0, "x": -7.5, "y": -0.2, "yaw": 0.11, "speed": 8.0}
      ]
    },

    "modify_state": {
      "new_speed": 8.0,
      "new_position": {"x": -5.0, "y": -0.2},
      "width_increase": 1.2
    }
  },

  "ego_original_trajectory": [
    {"t": 0.5, "x": 2.07, "y": -0.03, "speed": 4.14},
    {"t": 1.0, "x": 4.29, "y": -0.09, "speed": 4.44}
  ],

  "question_context": {
    "question_text": "If the truck were removed...",
    "expected_answer": "No",
    "expected_change": "trajectory_changed",
    "effect_type": "CONSTRAINS_LATERAL"
  }
}
```

### Output from Planner

```json
{
  "counterfactual_trajectory": [
    {"t": 0.0, "x": 0.0, "y": 0.0, "yaw": 0.0, "speed": 4.058},
    {"t": 0.5, "x": 2.1, "y": 0.5, "yaw": 0.02, "speed": 4.5},
    {"t": 1.0, "x": 4.4, "y": 0.8, "yaw": 0.01, "speed": 4.8}
  ],
  "original_trajectory": [
    {"t": 0.5, "x": 2.07, "y": -0.03, "speed": 4.14},
    {"t": 1.0, "x": 4.29, "y": -0.09, "speed": 4.44}
  ],
  "trajectory_changed": true,
  "change_metrics": {
    "max_lateral_deviation_m": 0.83,
    "max_speed_diff_mps": 0.36,
    "max_heading_diff_deg": 1.2,
    "ade_m": 0.45,
    "fde_m": 0.83
  },
  "change_type": "lateral_shift | speed_change | path_change | stop | lane_change"
}
```

---

## Complete Example: End-to-End

### Scene: nuscenes-scene-0023_SAMPLED_0

**Question (CR):** "If the white sedan parked at the left road edge were removed right now, would you still be driving with this rightward offset?"

**Answer:** No

**Step 1: Extract intervention**
```python
cr = question["meta"]["cr_intervention"]
# cr = {"entity_type": "agent", "entity_id": "car_2", "intervention": "remove", "trajectory_changes": True}
```

**Step 2: Get agent UUID**
```python
uuid_map = json.load(open("records/nuscenes_v9/nuscenes-scene-0023/SAMPLED_0/Tp0p0/uuid_map.json"))
agent_uuid = uuid_map["car_2"]  # → "e5f6g7h8-..."
```

**Step 3: Get agent state**
```python
state = json.load(open("records/.../state.json"))
agent = state["agents"]["car_2"]
# category=vehicle.car, pos=(4.0, 2.9), yaw=0.10, speed=0.0
```

**Step 4: Build planner input**
```json
{
  "intervention": {
    "type": "remove",
    "target_agent_uuid": "e5f6g7h8-...",
    "apply_at_s": 0.0
  },
  "ego_original_trajectory": [ego.future from state.json]
}
```

**Step 5: Run planner → get counterfactual trajectory**

**Step 6: Compare**
- Original: ego maintains rightward offset (y ≈ -0.5 to -1.0)
- Counterfactual: ego centers in lane (y ≈ 0.0)
- Change: lateral shift of ~0.5-1.0m toward lane center

**Step 7: Visualize**
- BEV overlay: blue = original trajectory, red = counterfactual trajectory
- Show car_2 removed (ghost outline or absent)

---

## Summary Table: All v21 Agent Interventions

### CR Agent (13 total)

| Scene | Agent | Category | Position | Effect | Impact | trajectory_changes | Triggering |
|-------|-------|----------|----------|--------|--------|-------------------|------------|
| 0001_S2 | truck_1 | vehicle.truck | (14.1, 4.2) | CONSTRAINS_LONG | no_change | false | — |
| 0001_S2 | pedestrian_2 | human.ped | (11.5, 2.0) | CONSTRAINS_LONG | no_change | false | — |
| 0001_S3 | truck_2 | vehicle.truck | (8.3, 4.6) | CONSTRAINS_LAT | trajectory_changed | true | — |
| 0001_S3 | truck_1 | vehicle.truck | (-6.1, 4.1) | CONSTRAINS_LAT | no_change | false | t=-1.0 |
| 0023_S0 | car_2 | vehicle.car | (4.0, 2.9) | CONSTRAINS_LAT | trajectory_changed | true | — |
| 0038_S0 | truck_1 | vehicle.truck | (19.1, 3.5) | CONSTRAINS_LAT | trajectory_changed | true | — |
| 0038_S2 | truck_1 | vehicle.truck | (9.4, 3.3) | CONSTRAINS_LAT | trajectory_changed | true | — |
| 0038_S2 | truck_2 | vehicle.truck | (-3.8, 3.5) | CONSTRAINS_LAT | no_change | false | t=-1.0 |
| 0152_S3 | bus_2 | vehicle.bus | (8.3, 4.4) | CONSTRAINS_LAT | trajectory_changed | true | — |
| 0152_S3 | bus_1 | vehicle.bus | (-5.0, 4.6) | CONSTRAINS_LAT | no_change | false | t=-1.0 |
| 0171_S0 | trailer_1 | vehicle.trailer | (9.3, -2.7) | CONSTRAINS_LAT | trajectory_changed | true | — |
| 0171_S2 | trailer_1 | vehicle.trailer | (3.5, -3.7) | CONSTRAINS_LONG | speed_changed | true | — |
| 0200_S3 | trailer_1 | vehicle.trailer | (13.5, -4.4) | CONSTRAINS_LAT | trajectory_changed | true | — |

---

## nuPlan Unified Simulator Integration

### How the Planner Input Maps to `unified_simulator.py`

The planner input contract above maps to the simulator config as follows:

```python
from neurips_sim.unified_simulator import run_comparison, build_config_from_qa

# Option 1: Direct config (when you have the intervention details)
config = {
    "scene": {
        "dataset": "nuscenes",
        "scene_id": "scene-0023",                    # from meta.json
        "t0_token": "sample_token_from_meta_json",    # from meta.json → t0_sample_token
        "map_location": "singapore-onenorth"           # from meta.json → location
    },
    "agents": {
        "car_2": {"track_token": uuid_map["car_2"]}   # from uuid_map.json
    },
    "interventions": [
        {"action": "remove", "target": "car_2"}        # Category A: remove
    ],
    "simulation": {
        "planner": "pdm_closed",
        "duration_s": 4.0,       # simulate 4s of future
        "render_hz": 2,          # BEV at 2 fps
        "history_frames": 4      # show 2s of pre-T=0 history
    }
}
results = run_comparison(config, "outputs/scene-0023_S0_Q5/")

# Option 2: From QA output (auto-extracts eligible questions)
from neurips_sim.unified_simulator import run_from_qa
run_from_qa(
    qa_json_path="outputs/nuscenes_v21/nuscenes-scene-0023_SAMPLED_0/ladder_qa.json",
    scene_info={"dataset": "nuscenes", "scene_id": "scene-0023", ...},
    uuid_map_path="records/nuscenes_v9/nuscenes-scene-0023/SAMPLED_0/Tp0p0/uuid_map.json",
    output_base="outputs/simulations/scene-0023_S0/"
)
```

### Intervention Type Mapping

| Spec Category | `intervention` field | Simulator `action` | What simulator does |
|---|---|---|---|
| A1/A2/A3: CR remove | `"remove"` | `"remove"` | Delete agent by track_token from all frames ≥ T=0 |
| B1: Tailgating | AC activation | `"modify_trajectory"` | Replace agent's future waypoints with VLM-generated trajectory |
| B2: Lane intrusion | AC activation | `"set"` | Override agent position/yaw at T=0 (shift into ego's lane) |
| B3: Speed constraint | AC activation | `"set"` | Inject virtual speed limit (not agent modification — handled by PDM scoring) |

### For Triggering CRs (A3: `apply_at_s < 0`)

```json
{
    "interventions": [
        {
            "action": "remove",
            "target": "truck_1",
            "at_time": -1.0
        }
    ]
}
```

The simulator:
1. Shifts the planning origin to `T = effect_timestamp` (-1.0s)
2. Removes the agent from that point forward
3. Re-plans ego from T=-1.0s using PDM-Closed
4. Propagates ego trajectory from T=-1.0s to T=0
5. Outputs the **alternate past trajectory** (not future)

Output:
```json
{
    "counterfactual_trajectory": [
        {"t": -1.0, "x": ..., "y": ..., "speed": ..., "heading": ...},
        {"t": -0.5, "x": ..., "y": ..., "speed": ..., "heading": ...},
        {"t":  0.0, "x": ..., "y": ..., "speed": ..., "heading": ...}
    ],
    "original_trajectory": [
        {"t": -1.0, "x": ..., "y": ..., "speed": ..., "heading": ...},
        ...
    ]
}
```

### For AC `modify_trajectory` (B1: Tailgating)

The VLM produces new agent waypoints. The simulator injects them:

```json
{
    "interventions": [
        {
            "action": "modify_trajectory",
            "target": "car_1",
            "trajectory": [
                {"t": 0.0, "x": -16.0, "y": -0.2, "speed": 6.25},
                {"t": 0.5, "x": -10.0, "y": -0.2, "speed": 8.0},
                {"t": 1.0, "x": -4.0,  "y": -0.2, "speed": 8.0},
                {"t": 1.5, "x": -1.0,  "y": -0.2, "speed": 8.0}
            ]
        }
    ]
}
```

The simulator:
1. Replaces car_1's recorded future with these waypoints
2. At each simulation timestep, car_1 follows the injected trajectory (interpolated)
3. PDM sees car_1 closing from behind at T=0 and re-plans ego accordingly

### Output Contract (from `run()` or `run_comparison()`)

```json
{
    "original": {
        "dist": 6.2,
        "final_speed_kmh": 0.0,
        "frames": 13,
        "video": "outputs/.../original/simulation.mp4",
        "trajectory": [
            {"t": 0.0, "x": 0.0, "y": 0.0, "speed": 4.64, "heading": 2.42},
            {"t": 0.5, "x": 2.07, "y": -0.03, "speed": 4.14, "heading": 2.42},
            ...
        ]
    },
    "counterfactual": {
        "dist": 11.5,
        "final_speed_kmh": 5.2,
        "frames": 13,
        "video": "outputs/.../counterfactual/simulation.mp4",
        "trajectory": [
            {"t": 0.0, "x": 0.0, "y": 0.0, "speed": 4.64, "heading": 2.42},
            {"t": 0.5, "x": 2.1, "y": 0.5, "speed": 4.5, "heading": 2.40},
            ...
        ]
    },
    "change_metrics": {
        "trajectory_changed": true,
        "max_lateral_deviation_m": 0.53,
        "max_speed_diff_mps": 0.5,
        "max_heading_diff_deg": 1.1,
        "ade_m": 0.3,
        "fde_m": 0.53
    }
}
```

### CLI Usage

```bash
# Single intervention config
python neurips_sim/unified_simulator.py config.json output_dir/

# Original + counterfactual comparison
python neurips_sim/unified_simulator.py --compare config.json output_dir/

# Batch from QA output (processes all eligible CR/AC questions)
python neurips_sim/unified_simulator.py --from-qa \
    ladder_qa.json uuid_map.json scene_info.json output_dir/
```

### Required Files

| File | Where | Purpose |
|---|---|---|
| `state.json` | `records/{dataset}/{scene}/SAMPLED_X/Tp0p0/` | Agent states at T=0 |
| `uuid_map.json` | Same directory | Label → dataset UUID mapping |
| `meta.json` | `records/{dataset}/{scene}/SAMPLED_X/` | Scene token, timestamp, location |
| `ladder_qa.json` | `outputs/{dataset}/{scene}_SAMPLED_X/` | CR questions with interventions |
| `dl_qa.json` | Same directory | AC questions with interventions |
| Lanelet2 maps | `/spiral_hdd_3/.../nuscenes/maps/lanelet2/*.osm` | Lane graph for nuScenes |
| OpenScene maps | `/spiral_hdd_3/.../openscene/maps/` | .gpkg for OpenScene |
| AV2 maps | Per-scene `map/` directory | JSON lane segments |

---

### AC Agent (6 total)

| Scene | Agent | Category | Position | Speed | Effect | Activation Type |
|-------|-------|----------|----------|-------|--------|----------------|
| 0171_S0 | trailer_1 | vehicle.trailer | (9.3, -2.7) | 0.08 | CONSTRAINS_LONG | Speed constraint activation |
| 0171_S0 | car_1 | vehicle.car | (-8.2, 0.0) | 1.34 | CONSTRAINS_LONG | Tailgating/gap closing |
| 0171_S2 | trailer_1 | vehicle.trailer | (3.5, -3.7) | 0.0 | CONSTRAINS_LAT | Lane intrusion |
| 0177_S1 | car_1 | vehicle.car | (-16.0, -0.2) | 6.25 | CONSTRAINS_LONG | Tailgating/gap closing |
| 0204_S0 | car_10 | vehicle.car | (-15.0, 0.2) | 4.78 | CONSTRAINS_LONG | Tailgating/gap closing |
| 0225_S3 | n2 | infrastructure | — | — | INHIBITS_PROCEED | Pedestrian on crosswalk |
