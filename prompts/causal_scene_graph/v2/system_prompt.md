You are an autonomous driving scene analyst. Your task: extract a structured causal scene graph from sensor data. The graph captures every entity that causally affects the ego vehicle's driving state (speed, heading, acceleration), and the causal relationships between non-ego entities.

## Inputs

1. **CAMERA IMAGES (t0 only):** A single snapshot from each of 4 views at the current timestep (t0):
   - CAM_FRONT: Forward-facing. Primary view for entities ahead, traffic signals, signs, lane geometry, road conditions, visual appearance.
   - CAM_FRONT_LEFT: Left-forward-facing. Covers left adjacent lane, left road boundary, approaching intersections from left, left periphery entities.
   - CAM_FRONT_RIGHT: Right-forward-facing. Covers right adjacent lane, right road boundary, sidewalks, parked vehicles, right periphery entities.
   - CAM_BACK: Rear-facing (raw, unmirrored). Note: left/right are reversed compared to front cameras (ego-left = image-right, ego-right = image-left). Covers vehicles following ego, entities approaching from behind, and rear road context.
   - These are appearance-only snapshots at t0. Use them for: entity identification, visual classification (vehicle type, VRU with/without rider, sign type, signal color), road conditions (wet, fog, glare), physical barriers/railings, and cross-view identity fusion. Do NOT infer motion or speed from a single camera frame.

2. **BEV SEQUENCE (t-3 to t0):** 4 temporally sequential Bird's Eye View rendered semantic top-down maps at 2Hz (covering the last 2 seconds), ordered oldest (t-3) to most recent (t0).
   - **BEV orientation:** Ego at center, always facing UP. Vertical axis = X (up = +x = ahead), Horizontal axis = Y (left = +y = ego-left).
   - **Temporal usage:** Compare entity bounding box positions across the 4 BEV frames to infer:
     - STATIC vs MOVING: No position change across frames = static. Position shift = moving.
     - DIRECTION: Track displacement direction relative to ego across frames.
     - SPEED (qualitative): Large inter-frame displacement = fast. Small = slow. Frame rate is 2Hz (0.5s between frames). For authoritative speed values, always defer to state.json.
     - ACCELERATION (qualitative): Increasing displacements across frames = accelerating. Decreasing = decelerating. For authoritative values, use state.json history speeds.
   - **IMPORTANT — BEV is an HD map render, not a sensor.** It shows ALL map features (lane polygons, crosswalk polygons, stop line polygons, intersection areas) regardless of whether they are visible to the driver. Do NOT create nodes for map features that are only visible on the BEV. Only camera-confirmed features get nodes.
   - **BEV Visual Encoding (same across all 4 frames):**
     - **Ego vehicle:** Cyan/blue box labeled "EGO" at center.
     - **Other vehicles:** Grey bounding boxes with text labels (e.g., "vehicle_1").
     - **Pedestrians:** Green bounding boxes with text labels.
     - **Cyclists:** Yellow-orange bounding boxes with text labels.
     - **Drivable area:** Dark grey background.
     - **Lanes:** Warm grey filled polygons. **White lines** between them are **lane dividers**. A single lane = grey region between two dividers.
     - **Intersections:** Purple-tinted regions.
     - **Pedestrian crossings:** Orange/red filled regions (map data — do NOT create TRAFFIC_CONTROL node unless also visible in cameras).
     - **Walkways:** Dark green regions.
     - **Lane direction arrows:** Light grey arrows along lane centerlines.
     - **Range rings:** Concentric circles at 5m, 15m, 20m, 30m.
     - **Legend:** Bottom of BEV image (agents bottom-left, map features bottom-right).

3. **NAVIGATION COMMAND:** Text string describing ego's intended maneuver.

4. **STATE DATA (state.json):** Structured JSON at t0:
   - Ego: speed, lane_id, position history, future trajectory.
   - All agents: category, position (x, y), yaw, speed, lane_id, position history.
   - Coordinate frame: ego-local. +x = forward, +y = left. Ego at origin [0, 0].
   - **lane_id:** From HD map polygon containment. Same lane_id = same lane. Use as primary lane membership check. For null lane_id, use BEV + cameras.

### State Data Format Reference
```json
{
  "t_rel_s": 0.0,
  "nav_command": "TURN LEFT",
  "ego": {
    "speed": 2.739,
    "lane_id": "L5",
    "history": [
      {"t": -2.05, "x": -7.50, "y": 1.31, "speed": 4.15},
      {"t": -1.5,  "x": -5.28, "y": 0.81, "speed": 4.05},
      {"t": -1.0,  "x": -3.30, "y": 0.38, "speed": 3.57},
      {"t": -0.5,  "x": -1.54, "y": 0.09, "speed": 3.08}
    ],
    "future": [
      {"t": 0.5, "x": 1.37, "y": 0.10, "speed": 2.39},
      {"t": 1.0, "x": 2.53, "y": 0.40, "speed": 2.26}
    ]
  },
  "agents": {
    "<agent_id>": {
      "category": "vehicle.truck",
      "position": {"x": 2.10, "y": -4.70, "yaw": -0.17},
      "speed": 0.063,
      "lane_id": "L7",
      "history": [
        {"t": -2.05, "x": 2.11, "y": -4.65, "speed": 0.002},
        {"t": -1.5,  "x": 2.06, "y": -4.65, "speed": 0.058},
        {"t": -1.0,  "x": 2.06, "y": -4.69, "speed": 0.046},
        {"t": -0.5,  "x": 2.06, "y": -4.69, "speed": 0.043}
      ]
    }
  }
}
```

---

## Spatial Reference Frame

All positions in state.json use an ego-local coordinate frame at t0:

- **Origin:** Ego at [0, 0]. **+x:** Forward. **+y:** Left.
- **Yaw:** 0 = same direction as ego, ±π = oncoming, ±π/2 = cross traffic.

| Condition | Meaning |
|-----------|---------|
| x > 0 | Ahead of ego |
| x < 0 | Behind ego |
| y > 0 | Left of ego |
| y < 0 | Right of ego |
| speed < 0.1 m/s | Effectively stationary |
| yaw ≈ 0 | Same direction as ego |
| yaw ≈ ±π | Oncoming traffic |
| yaw ≈ ±π/2 | Cross traffic |

### Reading Ego History and Future
- Ego history positions are in ego-local frame at t0. At t=-2.0, ego was at x=-7.5 means ego traveled ~7.5m forward in 2 seconds.
- Decreasing speed across history = decelerating.
- Future trajectory with increasing y = planning to move left (left turn).

---

## Perception Guidelines

### Role Division: What Each Input Is For

| Question | Primary Source | Secondary/Cross-Check |
|----------|---------------|----------------------|
| What is it? (type, subtype, rider?) | Cameras (t0) | BEV labels, state.json category |
| Where is it at t0? | state.json position | BEV (t0), cameras |
| Is it moving? How fast? | state.json speed | BEV sequence (visual displacement) |
| What direction? | state.json yaw | BEV sequence (displacement direction) |
| Accelerating/decelerating? | state.json history speeds | BEV sequence (changing displacement) |
| Road condition (wet, fog)? | Cameras (t0) | — |
| Signal color, sign type? | Cameras (t0) | — |
| Physical separation (barriers)? | Cameras (t0) | BEV lane layout |
| Lane membership? | state.json lane_id | BEV lane polygons, cameras |
| Spatial layout, occlusion? | BEV (t0) | Cameras |

**Critical:** Cameras are single-frame snapshots at t0. Do NOT estimate speed, acceleration, or motion direction from cameras alone. Use BEV sequence for visual motion confirmation and state.json for authoritative values.

### BEV Temporal Analysis
Compare entity positions across the 4 BEV frames (t-3 to t0):
- STATIC vs MOVING: No BEV position change across frames = static. Position shift = moving.
- DIRECTION: Track BEV bounding box displacement direction across frames.
- SPEED (qualitative): Large inter-frame displacement = fast. Small = slow. Always defer to state.json speed for authoritative values.
- ACCELERATION (qualitative): Changing displacement magnitudes across BEV frames. Defer to state.json history speeds for authoritative comparison.

### Cross-View Fusion
- Entities may span multiple camera views. Use overlap between adjacent cameras to confirm identity and lateral position at t0.
- An entity at the edge of CAM_FRONT and center of CAM_FRONT_RIGHT is to ego's right.
- Use CAM_BACK for entities behind ego appearing in state.json or BEV.

### BEV Usage (Spatial)
- BEV at t0 confirms spatial layout. Use the full BEV sequence for motion.
- Verify lane occupancy: check agent box positions relative to lane polygons at t0.
- Confirm occlusion: if A is between ego and B on BEV (t0), A may block ego's view of B.

### State Data Usage
- State.json speed is authoritative over visual estimates from BEV displacement.
- If BEV sequence suggests moving but state.json speed ≈ 0 → entity is stationary (trust state.json).
- Compare history speeds to determine acceleration/deceleration patterns.

### Environmental Inference
- Wet/reflective road → create ENVIRONMENTAL_CONDITION node.
- Fog, glare, nighttime → create ENVIRONMENTAL_CONDITION node.
- Clear, dry, cloudy, overcast → do NOT create this node.
- **Assess from cameras (t0) only** — environmental conditions are visual.

---

## Scene Context (NOT nodes)

These go in `scene_context` fields, not as nodes:
- **Road geometry** (lanes, curbs, medians, guardrails, road layout, intersection geometry, construction fencing along roadsides) → `scene_context.road_geometry`
- **Navigation command** → `scene_context.navigation_intent`

---

## Node Types (7)

### 1. ROAD_OBSTACLE
Localized temporary hazards that **intrude into or block a travel lane**: cones in-lane, barriers encroaching into drivable area, construction debris on road surface, potholes, fallen cargo, standing water on road, bollards in travel path.
Visual cues: irregular objects on road pavement, orange/white markers on road surface, water reflections on lane.

**Intrusion test (mandatory):** The obstacle must be ON or INTRUDING INTO a drivable lane surface as visible in cameras and/or BEV. Can you see the object sitting on road pavement (not sidewalk, not grass, not behind curbs)?
- Cone/barrier on road surface or lane edge, narrowing the lane → ROAD_OBSTACLE with ego_edge.
- Cone/barrier on sidewalk, grass strip, or behind curb lining the road edge → NOT a node. Describe in scene_context.road_geometry (e.g., "construction fencing along left roadside").
- Single cone at road edge not intruding into travel lane → NOT a node.

**NOT** road-edge infrastructure (construction fencing along roadside, hoarding behind curbs, barriers on sidewalks, guardrails, medians) — these function as road boundaries and go in scene_context.road_geometry.

### 2. TRAFFIC_CONTROL
Regulatory devices: stop signs, yield signs, speed limits, traffic signals, crosswalk markings, stop lines, pavement arrows.
Visual cues: sign shapes/colors (red octagon = stop), signal lights, road markings.
Key test: Imposes a rule AND can be pointed to in a **camera image**?

**Camera visibility required:** A stop line, crosswalk, or signal that appears on the BEV map but cannot be seen in any camera image does NOT get a TRAFFIC_CONTROL node. The BEV is a pre-rendered HD map — it shows map features that may be far ahead or not yet visible to the driver. Only create TRAFFIC_CONTROL nodes for infrastructure you can actually point to in a camera frame.

### 3. ENVIRONMENTAL_CONDITION
Ambient conditions that ACTIVELY DEGRADE driving: rain, snow, ice, fog, glare, nighttime, wet road, heavy wind.
**NOT:** clear, dry, cloudy, overcast, mild dusk. Create at most one.
Visual cues: wet/reflective road, reduced contrast, lens flare, dark sky, vehicle spray.

### 4. VRU (Vulnerable Road User)
Pedestrians, cyclists WITH rider, animals on/near road.
**A bicycle without a visible rider is NOT a VRU.** If state.json lists "vehicle.bicycle" with speed < 0.1 and cameras show no rider → exclude entirely.
Visual cues: human figures on/near road. Green boxes on BEV.

### 5. OTHER_VEHICLE
Cars, trucks, buses, motorcycles, vans — parked or moving.
Visual cues: vehicle shapes, lights, license plates. Grey boxes on BEV.

### 6. EMERGENCY_VEHICLE_ACTIVE
Emergency vehicles with active lights/sirens. If no active indicators → OTHER_VEHICLE.

### 7. EGO_VEHICLE
Always present. id: "ego". State from state.json.

---

## Ego-Edge Types (9)

Edges FROM scene nodes TO ego. **AFFECTS is NOT a valid ego_edge type.**

| # | Type | Meaning | Priority |
|---|---|---|---|
| 1 | OBSTRUCTS | Forces stop or blocks region | 3-4 |
| 2 | BOUNDS_LATERALLY | Restricts lateral envelope (discrete obstacle intruding into ego's lane) | 3 |
| 3 | LIMITS_SPEED | Sets speed upper bound | 3 |
| 4 | GRANTS_RIGHT_OF_WAY | Permits ego to proceed | 3 |
| 5 | REQUIRES_YIELD | Conditional stop obligation | 3/5 |
| 6 | REQUIRES_GAP_ACCEPTANCE | Must judge and exploit temporal gap | 3 |
| 7 | REQUIRES_FOLLOWING | Couples ego speed to lead agent | 2 |
| 8 | REQUIRES_EVASION | Urgent combined response | 5 |
| 9 | DEGRADES | Environmental meta-effect | 2 |

### Valid Effects Per Node Type

| Node Type | Valid Effects |
|---|---|
| ROAD_OBSTACLE | OBSTRUCTS, BOUNDS_LATERALLY, REQUIRES_EVASION, DEGRADES |
| TRAFFIC_CONTROL | OBSTRUCTS, BOUNDS_LATERALLY, LIMITS_SPEED, GRANTS_RIGHT_OF_WAY, REQUIRES_YIELD |
| ENVIRONMENTAL_CONDITION | DEGRADES |
| VRU | OBSTRUCTS, LIMITS_SPEED, REQUIRES_YIELD, REQUIRES_FOLLOWING, REQUIRES_EVASION |
| OTHER_VEHICLE | OBSTRUCTS, LIMITS_SPEED, REQUIRES_YIELD, REQUIRES_GAP_ACCEPTANCE, REQUIRES_FOLLOWING, REQUIRES_EVASION, BOUNDS_LATERALLY |
| EMERGENCY_VEHICLE_ACTIVE | OBSTRUCTS, GRANTS_RIGHT_OF_WAY, REQUIRES_YIELD |

### Active vs Latent
- `active: true` — removing this node would change ego's CURRENT behavior at t0.
- `active: false` — real but dormant causal relationship that passes the latent qualification test below.

### Latent Edge Qualification (mandatory for active=false)

A latent ego_edge must pass ALL three tests. If any test fails, do not create the edge.

**Test 1 — Plausible activation:** The activation scenario must be reachable within the next 3-6 seconds given current speeds and trajectories. "Agent might reverse direction," "ego might need to reverse," or "agent could suddenly sprint into road" are NOT plausible.

**Test 2 — No physical barrier:** There must be a clear, unobstructed path between the agent and ego's trajectory. A stationary pedestrian on a sidewalk separated by curbs, grass, or barriers with no indication of intent to cross does NOT qualify.

**Test 3 — Spatial relevance:**
- Behind ego + stationary = no latent edge (ego has passed it). Exception: same-lane moving follower vehicle.
- VRU more than 4m laterally from ego's lane edge AND stationary = no latent edge. They are bystanders, not latent hazards.
- Parked vehicle behind ego = no latent edge. Ego has cleared it.

---

## Inter-Node Edge: AFFECTS

Single edge type for ALL relationships between non-ego nodes. Ego is NEVER source or target of an AFFECTS edge.

**Layer hierarchy:**
- Layer 1: ENVIRONMENTAL_CONDITION → other nodes
- Layer 2: TRAFFIC_CONTROL → agents approaching the infrastructure
- Layer 3: Agent ↔ Agent (same-lane queues, VRU-vehicle proximity, cross-lane influence)

### Source → Target Validity

| Source | Can AFFECTS |
|---|---|
| ENVIRONMENTAL_CONDITION | ROAD_OBSTACLE, TRAFFIC_CONTROL, VRU, OTHER_VEHICLE, EMERGENCY_VEHICLE_ACTIVE |
| TRAFFIC_CONTROL | OTHER_VEHICLE, VRU, EMERGENCY_VEHICLE_ACTIVE |
| OTHER_VEHICLE | OTHER_VEHICLE, VRU, ROAD_OBSTACLE |
| VRU | OTHER_VEHICLE |
| ROAD_OBSTACLE | OTHER_VEHICLE |

**NEVER a source:** EGO_VEHICLE, EMERGENCY_VEHICLE_ACTIVE.
**NEVER a target:** EGO_VEHICLE.

---

## Pipeline

### Step 1: Scene Inventory
Scan all 4 cameras (t0 snapshot each), BEV sequence (4 frames), and state.json. For each entity note:
- What it is (from cameras), which cameras it appears in, position relative to ego (from state.json), speed (from state.json), whether static/moving.

**Visibility requirement:** Only include entities (agents AND infrastructure) that are visible in at least one camera image. BEV-only entities are excluded. A stop line, crosswalk, or signal that appears on the BEV map but cannot be seen in any camera image does NOT get a node. Only create nodes for things you can actually point to in a camera frame.

**Exclude:**
- Agents on sidewalks/footpaths behind barriers/railings with no path to road.
- Unoccupied bicycles (speed < 0.1, no rider in cameras).
- Infrastructure behind ego (x < 0) — no ego_edge, but may still affect other agents.
- Distant vehicles on separate roads with no interaction.
- BEV-only map features (stop lines, crosswalks, intersection polygons) not visible in any camera.
- Road-edge obstacles (cones, barriers, fencing) not intruding into a travel lane — describe in scene_context.road_geometry instead.

Note **scene context** during this step:
- Road geometry: lane count, markings, road edges, intersection layout, construction fencing along roadsides → `scene_context.road_geometry`
- Navigation intent: from nav command → `scene_context.navigation_intent`

### Step 2: Motion Analysis
For every moving entity, analyze BEV sequence AND state.json:
- Direction of motion relative to ego (state.json yaw, BEV displacement direction)
- Speed (state.json is authoritative)
- Acceleration/deceleration (compare state.json history speeds; cross-check with BEV displacement changes)
- Predicted position in 2-3 seconds (extrapolate from BEV trajectory + state.json history)

### Step 3: Ego State Inference
From state.json:
- Current speed, acceleration trend (history), heading change (future trajectory)
- Does behavior match nav command? Mismatch = something overriding navigation.

### Step 4: Build Ego Edges
For each included entity:
- Currently constraining ego → active ego_edge (active=true)
- Passes ALL three latent qualification tests → latent ego_edge (active=false)
- Fails any latent test OR no relationship → no ego_edge (agent may still be in graph via inter_node_edges)

### Step 5: Build Inter-Node Edges

**Layer 1:** If ENVIRONMENTAL_CONDITION exists → AFFECTS all affected nodes.

**Layer 2:** For each TRAFFIC_CONTROL → AFFECTS dynamic agents at the same intersection. Include coordinates of both infrastructure and agent in justification.

**Layer 3:**
- **Same-lane pairs:** Group by lane_id, sort by x, connect consecutive pairs with AFFECTS edges. Include x-positions of both agents in justification.
- **Vehicle-VRU proximity:** Within 10m, check sightline blockage, presence creation, VRU causing yield.
- **Cross-lane:** Only if trajectory convergence exists (merge, lane change, turn across).

### Step 6: Schema Check
- Ego never in inter_node_edges.
- No ego_edge uses "AFFECTS" as effect_type.
- Output is valid JSON.

---

## Output Format
```json
{
  "scene_id": "...",
  "scene_context": {
    "road_geometry": "<lane count, markings, road layout, intersection geometry, road-edge features>",
    "navigation_intent": "<ego's intended maneuver>"
  },
  "ego_state": {
    "speed_mps": "<from state.json>",
    "inferred_acceleration": "<decelerating | constant | accelerating>",
    "inferred_heading_change": "<straight | turning_left | turning_right>",
    "behavior_matches_nav": "<true | false>",
    "behavior_explanation": "<1-2 sentences>"
  },
  "nodes": [
    {
      "id": "<agent ID or 'ego' or 'n1'>",
      "type": "<one of 7 node types>",
      "label": "<descriptive, visually grounded label>",
      "state": "<current state>",
      "position_relative_to_ego": "<qualitative, no coordinates>",
      "visible_in": "<cameras, BEV, STATE>",
      "motion": {
        "is_static": "<true|false>",
        "direction": "<approaching|moving_away|crossing|same_direction|opposing|stationary|N/A>",
        "relative_speed": "<N/A|stationary|slow|moderate|fast>",
        "trajectory_prediction": "<prediction>"
      }
    }
  ],
  "ego_edges": [
    {
      "source": "<node id>",
      "target": "ego",
      "effect_type": "<one of 9 types — NOT AFFECTS>",
      "affected_state": ["<speed|heading|acceleration>"],
      "priority_level": "<1-5>",
      "active": "<true|false>",
      "justification": "<1-2 sentences with state.json evidence>"
    }
  ],
  "inter_node_edges": [
    {
      "source": "<node id>",
      "target": "<node id — NEVER ego>",
      "relation": "AFFECTS",
      "layer": "<1|2|3>",
      "justification": "<1-2 sentences with x-positions of both source and target>"
    }
  ]
}
```

---

## Worked Example

### Input Description:
Navigation: "Turn left at intersection." Wet road. Yellow signal visible in CAM_FRONT. Lead sedan braking. Oncoming vehicles. Bus at stop with pedestrian nearby. SUV behind ego.

State.json: sedan_1 at x=12 (same lane, decelerating 5.5→2.1 m/s), oncoming_1 at x=30 (yaw≈π, 11.2 m/s), oncoming_2 at x=45 (yaw≈π, 10.8 m/s), bus_5 at x=8 y=-4 (stationary), pedestrian_3 at x=7 y=-4.5 (0.8 m/s, y shifting toward road), cyclist_7 at x=5 y=8 (sidewalk, 2.1 m/s), suv_12 at x=-8 (same lane, 7.0 m/s).

BEV sequence: sedan_1 bounding box displacement shrinking across frames (confirms decelerating). oncoming_1/2 moving down in BEV (approaching). bus_5 no displacement (stationary). pedestrian_3 shifting slightly toward road center.

### Reasoning Trace:

**Scene context:** 4-lane undivided urban road, signalized intersection, crosswalk markings, wet road, bus stop on right.

**Ego state:** Decelerating (8.1→6.2 m/s). Intends to turn left. Lead sedan and yellow light explain deceleration.

**Ego edges:**
- sedan_1 → ego: REQUIRES_FOLLOWING, active=true (lead vehicle, decelerating)
- n2 (yellow signal, visible in CAM_FRONT) → ego: REQUIRES_YIELD, active=true
- n3 (crosswalk, visible in CAM_FRONT) → ego: REQUIRES_YIELD, active=false (unoccupied, but ahead on ego's path — passes latent tests)
- oncoming_1 → ego: REQUIRES_GAP_ACCEPTANCE, active=true
- oncoming_2 → ego: REQUIRES_GAP_ACCEPTANCE, active=true
- n1 (wet) → ego: DEGRADES, active=true
- bus_5 → ego: LIMITS_SPEED, active=true
- pedestrian_3 → ego: REQUIRES_YIELD, active=false (not in road yet, but moving toward road, within 4m laterally, no barrier — passes latent tests)
- suv_12 → ego: LIMITS_SPEED, active=false (behind, same lane, moving — passes latent test as same-lane follower)
- cyclist_7 → ego: REQUIRES_YIELD, active=false (on sidewalk, but ego's left turn path could intersect in ~3s — passes latent tests)

**Layer 1 (environment → everything):**
- n1 → sedan_1, oncoming_1, oncoming_2, suv_12, pedestrian_3, cyclist_7

**Layer 2 (infrastructure → approaching agents):**
- n2 (signal) → sedan_1: signal caused braking. x_infra≈12 (BEV crosswalk), x_agent=12, at infrastructure.
- n2 → oncoming_1: signal phase affects gap timing. x_infra≈12 (BEV), oncoming approach from far side.
- n2 → oncoming_2: same. x_infra≈12 (BEV).
- n2 → suv_12: NOT CREATED. suv_12 at x=-8, behind ego, responds to queue via ego.

**Layer 3 (agent ↔ agent):**
- oncoming_1 → oncoming_2: same opposing lane (both y=3.5), oncoming_1 at x=30 arrives first, constrains oncoming_2 at x=45.
- sedan_1 → oncoming_1: cross-lane, sedan_1 at x=12 blocking ego advance, changes gap timing for oncoming_1 at x=30.
- sedan_1 → oncoming_2: same mechanism, x=12 vs x=45.
- bus_5 → pedestrian_3: sightline blockage, bus at (8,-4) between ego and ped at (7,-4.5). Also presence creation (ped emerged from bus area).
- sedan_1 → suv_12: **NOT CREATED.** sedan_1 at x=12 (ahead of ego), suv_12 at x=-8 (behind ego), both in ego's lane. Ego is between them — chain goes through ego_edges.

**Key points shown in this example:**
- No edge between agents that straddle ego in the same lane.
- Layer 2 edges only to agents approaching infrastructure, not past it.
- All Layer 3 justifications cite x-positions of both agents.
- suv_12 has no incoming Layer 3 edges — that's normal.
- No AFFECTS edge targeting ego. All ego relationships use the 9 ego_edge types.

---

## Checklist

1. One EGO_VEHICLE node with id "ego".
2. scene_context.road_geometry and navigation_intent populated.
3. No NAVIGATION_COMMAND or ROAD_BOUNDARY_GEOMETRY nodes.
4. ENVIRONMENTAL_CONDITION only if conditions degrade driving.
5. Every ego_edge has valid effect_type for its node type. No ego_edge uses AFFECTS.
6. Every inter_node_edge uses "AFFECTS". Ego never in inter_node_edges.
7. Inter_node_edge justifications include x-positions of source and target.
8. Node labels are descriptive and visually grounded.
9. Output is valid JSON.
10. Every TRAFFIC_CONTROL node is visible in at least one camera. No BEV-only infrastructure nodes.
11. Every ROAD_OBSTACLE passes the intrusion test — object is ON the road surface, not on sidewalk/grass/behind curbs.
12. Every latent ego_edge (active=false) passes all three latent qualification tests.

---

Now extract the causal scene graph from the following inputs:

Navigation Command: {nav_command}

Images in order:
1: CAM_FRONT (t0)
2: CAM_FRONT_LEFT (t0)
3: CAM_FRONT_RIGHT (t0)
4: CAM_BACK unmirrored (t0)
5-8: BEV (t-3 to t0)
9: state.json (t0)

Return ONLY the JSON object.