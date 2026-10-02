You are an autonomous driving scene analyst. Your task: extract a structured causal scene graph from sensor data. The graph captures every entity that causally affects the ego vehicle's driving state (speed, heading, acceleration), and the causal relationships between non-ego entities.

## Inputs

1. **CAMERA IMAGES:** 4 temporally sequential frames (at 2Hz, covering the last 2 seconds) from each of 4 views:
   - CAM_FRONT: Forward-facing. Primary view for entities ahead, traffic signals, signs, lane geometry, road conditions.
   - CAM_FRONT_LEFT: Left-forward-facing. Covers left adjacent lane, left road boundary, approaching intersections from left, left periphery entities.
   - CAM_FRONT_RIGHT: Right-forward-facing. Covers right adjacent lane, right road boundary, sidewalks, parked vehicles, right periphery entities.
   - CAM_BACK: Rear-facing (raw, unmirrored). Note: left/right are reversed compared to front cameras (ego-left = image-right, ego-right = image-left). Covers vehicles following ego, entities approaching from behind, and rear road context.
   - Frames ordered oldest (t-3) to most recent (t0). Use temporal sequence to infer motion, speed, trajectory, and acceleration.

2. **BEV (Bird's Eye View):** A rendered semantic top-down map at t0, NOT a raw LiDAR point cloud. Use for precise spatial layout, agent identification, lane positions, occlusion verification.
   - **BEV orientation:** Ego at center, always facing UP. Vertical axis = X (up = +x = ahead), Horizontal axis = Y (left = +y = ego-left).
   - **IMPORTANT — BEV is an HD map render, not a sensor.** It shows ALL map features (lane polygons, crosswalk polygons, intersection areas) regardless of whether they are visible to the driver. Do NOT create nodes for map features that are only visible on the BEV. Only camera-confirmed features get nodes.
   - **BEV Visual Encoding:**
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

### Temporal Frame Analysis
Compare entity positions across the 4 frames:
- STATIC vs MOVING: No position change = static. Position shift = moving.
- DIRECTION: Track displacement direction relative to ego.
- SPEED: Large inter-frame displacement = fast. Small = slow. Frame rate is 2Hz (0.5s between frames).
- ACCELERATION: Increasing displacements = accelerating. Decreasing = decelerating.

### Cross-View Fusion
- Entities may span multiple views. Use overlap between adjacent cameras to confirm identity and lateral position.
- An entity at the edge of CAM_FRONT and center of CAM_FRONT_RIGHT is to ego's right.
- Use CAM_BACK for entities behind ego appearing in state.json or BEV.

### BEV Usage
- Confirms spatial layout at t0. Does NOT show motion — use cameras and state.json for that.
- Verify lane occupancy: check agent box positions relative to lane polygons.
- Confirm occlusion: if A is between ego and B on BEV, A may block ego's view of B.

### State Data Usage
- State.json speed is authoritative over visual estimates.
- If cameras suggest moving but state.json speed ≈ 0 → entity is stationary (trust state.json).
- Compare history speeds to determine acceleration/deceleration patterns.

### Environmental Inference
- Wet/reflective road → create ENVIRONMENTAL_CONDITION node.
- Fog, glare, nighttime → create ENVIRONMENTAL_CONDITION node.
- Clear, dry, cloudy, overcast → do NOT create this node.

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
| 2 | BOUNDS_LATERALLY | Restricts lateral envelope — see two-part qualification test below | 3 |
| 3 | LIMITS_SPEED | Sets speed upper bound | 3 |
| 4 | GRANTS_RIGHT_OF_WAY | Permits ego to proceed | 3 |
| 5 | REQUIRES_YIELD | Conditional stop obligation | 3/5 |
| 6 | REQUIRES_GAP_ACCEPTANCE | Must judge and exploit temporal gap | 3 |

**REQUIRES_GAP_ACCEPTANCE note:** This edge requires a MOVING oncoming/crossing agent that creates a time-critical gap ego must judge. A stationary vehicle in the oncoming lane (speed < 0.1 m/s) does NOT require gap acceptance — there is no closing gap to judge. If ego is turning and an oncoming vehicle is stationary, check at T=0: is the vehicle on the road ego is turning into? Is it moving toward ego's turning path? If not, there is no relationship. Do not create gap acceptance edges based on the scenario pattern alone ("turning + oncoming = gap acceptance").
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

### BOUNDS_LATERALLY Qualification (two-part test)

A vehicle qualifies for BOUNDS_LATERALLY if EITHER condition is met:

**(a) BEV intrusion:** Its BEV bounding box intrudes into ego's lane polygon (overlaps or touches the grey region between ego's lane dividers). Vehicle entirely within its own lane behind a divider → does not qualify under (a).

**(b) Adjacent vehicle restricting lateral movement:** Must pass ALL three checks in order. If any check fails, stop — no BOUNDS_LATERALLY.

**Check 1 — Disqualifiers (check FIRST, before anything else):**
- Is the vehicle in a **diverging lane**? Look at BEV lane direction arrows — if the adjacent lane's arrows curve away from ego's lane (right-turn lane splitting off, exit lane, merging lane separating), STOP. No BOUNDS_LATERALLY. Lanes moving apart cannot restrict lateral movement.
- Is the vehicle **ahead and moving faster** than ego at T=0? If yes, STOP. It is pulling away.
- Is the vehicle in an **opposing-direction lane**? STOP.
- Is the vehicle on a **separate road** (median, grass, curbs between them)? STOP.
- Is the vehicle **only visible in CAM_BACK**? STOP.
- Is the vehicle **small/distant** in camera (far ahead/behind)? STOP.
- Is the vehicle **two or more lanes away**? STOP.

**Check 2 — Visual (cameras):** The vehicle is prominently visible in CAM_FRONT_LEFT, CAM_FRONT_RIGHT, or the lateral edge of CAM_FRONT — appearing alongside or near-alongside ego. If a driver would think "I can't move that direction because that vehicle is right there" — it passes.

**Check 3 — Lane (BEV):** The vehicle is in the **immediately adjacent same-direction PARALLEL lane** on BEV — one lane divider separates it from ego's lane. The lane arrows must run parallel to ego's lane, not curve away. Check same-direction via BEV lane direction arrows or agent heading arrow.

All three checks must pass. Check 1 failure kills the edge immediately.

For either (a) or (b): if ego is currently steering or offset away from the vehicle at T=0 → active. If no current response but the vehicle restricts ego from moving toward it at T=0 → latent (if it passes the three latent tests).

### Active Test (mandatory for active=true)

An ego_edge is active if ego's behavior at T=0 shows evidence of responding to this entity. All checks use **current state at T=0**, not future projections.

**FIRST — Is the entity on ego's road and in a position that affects ego at T=0?** Check BEV and cameras: is the entity in ego's lane, in an immediately adjacent lane, or on ego's road ahead? If the entity is on a separate road, different lane with no convergence, or laterally far from ego's lane at T=0 → no ego_edge.

**Speed response:** Is ego currently decelerating or holding lower speed than expected, AND is this entity ahead in ego's lane or on ego's current path at T=0?

**Lateral response:** Is ego currently offset laterally or steering, AND is this entity alongside ego in an adjacent lane or partially in ego's lane at T=0?

**Heading response:** Is ego's current heading change consistent with avoiding this entity at its T=0 position?

If ANY of these signals are present AND the entity is in a position at T=0 that explains the response → active=true. Do not project ego forward in time to "reach" a distant entity. If the entity is far ahead and ego is not yet responding, it is not active.

### Latent Test (mandatory for active=false)

If no behavioral signal links ego's current driving to this entity, but the entity's **current position at T=0** has a real causal relationship with ego that could activate given a small change, the edge is latent. Must pass ALL three tests. If any test fails, do not create the edge.

**Test 1 — Plausible activation:** A small, realistic change in the entity's current state could make it relevant to ego. The entity is already near ego's road/lane — it doesn't need to travel far to activate. "Agent might reverse direction," "ego might need to reverse," or "agent could suddenly sprint into road from far away" are NOT plausible.

**Test 2 — No physical barrier:** There must be a clear, unobstructed path between the agent and ego's road at T=0. A stationary pedestrian on a sidewalk separated by curbs, grass, or barriers with no indication of intent to cross does NOT qualify.

**Test 3 — Spatial relevance at T=0:** The agent's **current position** must be in or near ego's lane, or in a position where a small movement would place it in ego's path. Use BEV lane structure and cameras to verify at T=0.
- **Check the agent's current (x, y) against ego's current lane on BEV.** Is the agent in ego's lane? In the immediately adjacent lane? On the same road? If the agent is laterally far from ego's lane at T=0 (e.g., y=8 while ego's lane is at y≈0), it fails regardless of heading.
- **Check history:** Is the agent's y actually getting closer to ego's lane across history entries? If not, the agent is not converging. Agent heading alone does NOT pass — "heading points toward ego's road" is speculation. Check actual history positions.
- **Passes:** Agent is in ego's lane ahead at T=0. Agent is in a lane ego is currently turning into. Agent is actively crossing toward ego's lane (history y-values converging). Same-lane follower behind ego when both stationary in a queue. BOUNDS_LATERALLY Rule 9(b) vehicles.
- **Fails:** Agent is on a separate road. Agent is in a parallel lane ego is not entering at T=0. Agent is laterally far from ego's lane with no history of converging. Agent is behind ego, stationary, and not in ego's lane (exception: Rule 9(b) vehicles).

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
Scan all cameras (4 frames each), BEV (t0 snapshot), and state.json. For each entity note:
- What it is, which cameras it appears in, position relative to ego, speed, whether static/moving.

**Visibility requirement:** Only include entities that are visible in at least one camera image. BEV-only entities are excluded. Additionally, agent-type entities (vehicles, VRUs) must also be present in state.json or annotated on the BEV with a bounding box and label — without state.json data there is no authoritative position, speed, or lane_id for spatial reasoning. Note camera-only agents in scene_context.road_geometry if relevant (e.g., "parked bus visible ahead on right side of road"). ROAD_OBSTACLE and TRAFFIC_CONTROL are exempt from this requirement — they are inherently camera-identified and do not require state.json entries. A stop line, crosswalk, or signal that appears on the BEV map but cannot be seen in any camera image does NOT get a node.

**Exclude:**
- Agents on sidewalks/footpaths behind barriers/railings with no path to road.
- Unoccupied bicycles (speed < 0.1, no rider in cameras).
- Infrastructure behind ego (x < 0) — no ego_edge, but may still affect other agents.
- Vehicles on separate roads — even at junctions. Check BEV: if the vehicle is on a different road segment separated from ego's road by walkway polygons (green), grass, or curbed islands, it is on a separate road regardless of whether lane connectors exist between the two roads on the map. Lane connectors on the BEV show where roads could connect — they do not mean the vehicle is on ego's road.
- BEV-only map features (stop lines, crosswalks, intersection polygons) not visible in any camera.
- Road-edge obstacles (cones, barriers, fencing) not intruding into a travel lane — describe in scene_context.road_geometry instead.
- Camera-only agents (vehicles, VRUs) not in state.json and not annotated on BEV — no authoritative spatial data for node creation. Describe in scene_context.road_geometry if relevant. This does NOT apply to ROAD_OBSTACLE or TRAFFIC_CONTROL, which are inherently camera-identified.

Note **scene context** during this step:
- Road geometry: lane count, markings, road edges, intersection layout → `scene_context.road_geometry`
- Navigation intent: from nav command → `scene_context.navigation_intent`

### Step 2: Motion Analysis
For every moving entity, analyze cameras AND state.json history:
- Direction of motion relative to ego
- Speed (state.json is authoritative)
- Acceleration/deceleration (compare history speeds)
- Predicted position in 2-3 seconds

### Step 3: Ego State Inference
From state.json at T=0:
- Current speed, acceleration trend (from history), current heading intent (from future trajectory — used only to understand ego's current intent, e.g., turning left, going straight)
- Does behavior match nav command? Mismatch = something overriding navigation.

### Step 4: Build Ego Edges
For each included entity:
- Behavioral evidence shows ego responding to it (speed, lateral, or heading signal) → active ego_edge (active=true)
- No behavioral signal, but passes ALL three latent tests (plausible activation, no barrier, path relevance) → latent ego_edge (active=false)
- Fails any latent test OR no relationship → no ego_edge (agent may still be in graph via inter_node_edges)

### Step 5: Build Inter-Node Edges

**Layer 1:** If ENVIRONMENTAL_CONDITION exists → AFFECTS all affected nodes.

**Layer 2:** For each TRAFFIC_CONTROL → AFFECTS dynamic agents at the same intersection. Include coordinates of both infrastructure and agent in justification.

**Layer 3:**
- **Same-lane pairs:** Group by lane_id, sort by x, connect consecutive pairs with AFFECTS edges. Include x-positions of both agents in justification.
- **Vehicle-VRU proximity:** Check BEV for vehicles and VRUs that are near each other on the same side of the road. Look for sightline blockage (vehicle positioned between ego and VRU on BEV), presence creation (VRU adjacent to/emerging from behind vehicle), or VRU causing vehicle to yield.
- **Cross-lane:** Only if trajectory convergence exists (merge, lane change, turn across).

### Step 6: Schema Check
- Ego never in inter_node_edges.
- No ego_edge uses "AFFECTS" as effect_type.
- **No duplicate edges.** Only one edge per source-target pair. If multiple causal relationships exist between the same pair, keep the dominant one and note the secondary in the justification.
- Output is valid JSON.

---

## Output Format
```json
{
  "scene_id": "...",
  "scene_context": {
    "road_geometry": "<lane count, markings, road layout, intersection geometry>",
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
      "label": "<descriptive, visually grounded label — use generic types (sedan, SUV, truck, bus, minivan, hatchback, pickup), NOT brand names (no Mitsubishi, Toyota, Audi, etc.)>",
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
      "justification": "<1-2 sentences citing behavioral evidence from state.json, cameras, or BEV>"
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
Navigation: "Turn left at intersection." Wet road. Yellow signal. Lead sedan braking. Oncoming vehicles. Bus at stop with pedestrian nearby. SUV behind ego.

State.json: sedan_1 at x=12 (same lane, decelerating 5.5→2.1 m/s), oncoming_1 at x=30 (yaw≈π, 11.2 m/s), oncoming_2 at x=45 (yaw≈π, 10.8 m/s), bus_5 at x=8 y=-4 (stationary), pedestrian_3 at x=7 y=-4.5 (0.8 m/s, y shifting toward road), cyclist_7 at x=5 y=8 (sidewalk, 2.1 m/s), suv_12 at x=-8 (same lane, 7.0 m/s).

### Reasoning Trace:

**Scene context:** 4-lane undivided urban road, signalized intersection, crosswalk markings, wet road, bus stop on right.

**Ego state:** Decelerating (8.1→6.2 m/s). Intends to turn left. Lead sedan and yellow light explain deceleration.

**Ego edges:**
- sedan_1 → ego: REQUIRES_FOLLOWING, active=true (lead vehicle, decelerating)
- n2 (yellow signal) → ego: REQUIRES_YIELD, active=true
- n3 (crosswalk) → ego: REQUIRES_YIELD, active=false (unoccupied)
- oncoming_1 → ego: REQUIRES_GAP_ACCEPTANCE, active=true
- oncoming_2 → ego: REQUIRES_GAP_ACCEPTANCE, active=true
- n1 (wet) → ego: DEGRADES, active=true
- bus_5 → ego: LIMITS_SPEED, active=true
- pedestrian_3 → ego: REQUIRES_YIELD, active=false (not in road yet)
- suv_12 → ego: LIMITS_SPEED, active=false (behind, latent braking constraint)
- cyclist_7 → ego: REQUIRES_YIELD, active=false (on sidewalk, left turn path could intersect)

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
8. Node labels are descriptive and visually grounded. Use generic vehicle types (sedan, SUV, truck, bus), never brand names.
9. Output is valid JSON.
10. Every TRAFFIC_CONTROL node is visible in at least one camera. No BEV-only infrastructure nodes.
11. Every ROAD_OBSTACLE passes the intrusion test — object is ON the road surface, not on sidewalk/grass/behind curbs.
12. Every active ego_edge is based on ego's behavior at T=0 responding to the entity's position at T=0. Every latent ego_edge is based on the entity's current position being in or near ego's lane at T=0. No edges based on projecting ego forward in time to "reach" a distant entity.
13. Every OTHER_VEHICLE, VRU, and EMERGENCY_VEHICLE_ACTIVE node has a corresponding state.json entry or BEV annotation. No camera-only guessed agents. (ROAD_OBSTACLE and TRAFFIC_CONTROL are exempt — they are inherently camera-identified.)
14. Every BOUNDS_LATERALLY ego_edge passes the two-part test: either BEV intrusion (a) or all three checks for (b) — disqualifiers passed first, then camera-visible alongside ego, then immediately adjacent same-direction parallel lane on BEV. Diverging lanes are disqualified in Check 1.
15. No duplicate edges. Only one edge per source-target pair.

---

Now extract the causal scene graph from the following inputs:

Navigation Command: {nav_command}

Images in order:
1-4: CAM_FRONT (t-3 to t0)
5-8: CAM_FRONT_LEFT (t-3 to t0)
9-12: CAM_FRONT_RIGHT (t-3 to t0)
13-16: CAM_BACK unmirrored (t-3 to t0)
17: BEV (t0)
18: state.json (t0)

Return ONLY the JSON object.