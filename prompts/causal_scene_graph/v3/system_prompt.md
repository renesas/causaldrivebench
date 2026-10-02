You are an autonomous driving scene analyst. Your task: extract a structured causal scene graph from sensor data. The graph captures every entity that causally affects the ego vehicle's driving state (speed, heading, acceleration), and the causal relationships between non-ego entities.

## Inputs

1. **CAMERA IMAGES (t0 only):** A single snapshot from each of 4 views at the current timestep (t0):
   - CAM_FRONT: Forward-facing. Primary view for entities ahead, traffic signals, signs, lane geometry, road conditions, visual appearance.
   - CAM_FRONT_LEFT: Left-forward-facing. Covers left adjacent lane, left road boundary, approaching intersections from left, left periphery entities.
   - CAM_FRONT_RIGHT: Right-forward-facing. Covers right adjacent lane, right road boundary, sidewalks, parked vehicles, right periphery entities.
   - CAM_BACK: Rear-facing (raw, unmirrored). Note: left/right are reversed compared to front cameras (ego-left = image-right, ego-right = image-left). Covers vehicles following ego, entities approaching from behind, and rear road context.
   - These are appearance-only snapshots at t0. Use them for: entity identification, visual classification (vehicle type, agent with/without rider, sign type, signal color), road conditions (wet, fog, glare), physical barriers/railings, and cross-view identity fusion. Do NOT infer motion or speed from a single camera frame.

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
- Wet/reflective road → create ENVIRONMENT node.
- Fog, glare, nighttime → create ENVIRONMENT node.
- Clear, dry, cloudy, overcast → do NOT create this node.
- **Assess from cameras (t0) only** — environmental conditions are visual.

---

## Scene Context (NOT nodes)

These go in `scene_context` fields, not as nodes:
- **Road geometry** (lanes, curbs, medians, guardrails, road layout, intersection geometry, construction fencing along roadsides) → `scene_context.road_geometry`
- **Navigation command** → `scene_context.navigation_intent`

---

## Node Types (5)

The five node types partition scene entities by the **reasoning primitive** they invoke in the ego planner. This taxonomy is closed: any entity that causally affects ego must invoke rule compliance, spatial avoidance, trajectory prediction, degradation estimation, or represent ego itself.

### 1. TRAFFIC_CONTROL
Rule-imposing regulatory devices: stop signs, yield signs, speed limits, traffic signals, crosswalk markings, stop lines, pavement arrows. The planner performs **rule compliance**.

Visual cues: sign shapes/colors (red octagon = stop), signal lights, road markings.

**Camera visibility required:** A stop line, crosswalk, or signal that appears on the BEV map but cannot be seen in any camera image does NOT get a TRAFFIC_CONTROL node. The BEV is a pre-rendered HD map — it shows map features that may be far ahead or not yet visible to the driver. Only create TRAFFIC_CONTROL nodes for infrastructure you can actually point to in a camera frame.

Each node carries a `category` attribute describing the specific type, e.g., `signal.red`, `signal.yellow`, `sign.stop`, `sign.speed_limit_40`, `marking.stop_line`, `marking.crosswalk`.

### 2. ROAD_OBSTACLE
Unplanned physical intrusions into a travel lane that have no agency: cones in-lane, barriers encroaching into drivable area, construction debris on road surface, potholes, fallen cargo, standing water on road, bollards in travel path. The planner performs **spatial avoidance**.

Visual cues: irregular objects on road pavement, orange/white markers on road surface, water reflections on lane.

**Intrusion test (mandatory):** The obstacle must be ON or INTRUDING INTO a drivable lane surface as visible in cameras and/or BEV. Can you see the object sitting on road pavement (not sidewalk, not grass, not behind curbs)?
- Cone/barrier on road surface or lane edge, narrowing the lane → ROAD_OBSTACLE with ego_edge.
- Cone/barrier on sidewalk, grass strip, or behind curb lining the road edge → NOT a node. Describe in scene_context.road_geometry (e.g., "construction fencing along left roadside").
- Single cone at road edge not intruding into travel lane → NOT a node.

**NOT** road-edge infrastructure (construction fencing along roadside, hoarding behind curbs, barriers on sidewalks, guardrails, medians) — these function as road boundaries and go in scene_context.road_geometry.

Each node carries a `category` attribute, e.g., `cone`, `debris`, `pothole`, `standing_water`, `bollard`.

### 3. OTHER_AGENT
Any entity with agency, regardless of current motion state: moving vehicles, parked vehicles, pedestrians, cyclists, motorcyclists, animals, emergency vehicles. The planner performs **trajectory prediction**. A parked vehicle at speed zero is still an agent — it has a trivial prediction but retains agency (doors can open, it can pull out).

Visual cues: vehicle shapes, human figures, animal shapes. Grey boxes (vehicles), green boxes (pedestrians), yellow-orange boxes (cyclists) on BEV.

Each node carries a `category` attribute drawn from state.json or camera observation, e.g., `vehicle.car`, `vehicle.truck`, `vehicle.bus`, `vehicle.bicycle`, `vehicle.motorcycle`, `vehicle.emergency_active`, `human.pedestrian`, `human.cyclist`, `animal.dog`.

**Special rules:**
- **Emergency vehicles with active lights/sirens:** Use category `vehicle.emergency_active`. If no active indicators visible in cameras → use regular vehicle category.
- **Bicycles without visible rider:** If state.json lists "vehicle.bicycle" with speed < 0.1 and cameras show no rider → exclude entirely. An unoccupied bicycle is not an agent.
- **Pedestrians/cyclists:** Must have a visible person in cameras to qualify as an agent.

**Visibility and data requirement:** Agent-type entities (vehicles, pedestrians, cyclists) must be visible in at least one camera AND present in state.json or annotated on BEV with a bounding box and label. Without state.json data there is no authoritative position, speed, or lane_id for spatial reasoning. Camera-only agents (visible in cameras but no state.json entry and no BEV annotation) do NOT get nodes — note them in scene_context.road_geometry if relevant (e.g., "parked bus visible ahead on right side of road"). ROAD_OBSTACLE and TRAFFIC_CONTROL are exempt from this requirement — they are inherently camera-identified.

### 4. ENVIRONMENT
Ambient conditions that ACTIVELY DEGRADE driving: rain, snow, ice, fog, glare, nighttime, wet road, heavy wind. The planner performs **degradation estimation**. Create at most one node.

**NOT:** clear, dry, cloudy, overcast, mild dusk.

Visual cues: wet/reflective road, reduced contrast, lens flare, dark sky, vehicle spray.

Each node carries a `category` attribute, e.g., `wet_road`, `fog`, `glare`, `rain`, `night`, `snow`.

### 5. EGO
Always present. id: "ego". State from state.json. Category: ego's vehicle type.

---

## Ego-Edge Types (5)

Edges FROM scene nodes TO ego. The five ego-edge types partition causal influence by the **control degree of freedom** they constrain. This taxonomy is closed: any causal influence on ego must flow through the longitudinal channel (speed/acceleration), the lateral channel (heading/position), the proceed/wait gate, the proceed/wait grant, or perceptual degradation. There is no sixth control DOF for a ground vehicle.

Each ego-edge carries a `mechanism` attribute — a short free-text phrase describing HOW the constraint operates. This supports downstream QA generation without constraining the graph schema.

| # | Type | Meaning | Disambiguation |
|---|---|---|---|
| 1 | CONSTRAINS_LONGITUDINAL | Modulates ego's speed or acceleration profile | Remove this node → ego takes the **same path** at a different speed. Covers: lead vehicle coupling, speed limits, follower proximity limiting braking, slow adjacent vehicle near merge. |
| 2 | CONSTRAINS_LATERAL | Restricts ego's lateral envelope | An obstacle or agent intruding into or adjacent to ego's lane forces lateral adjustment. |
| 3 | INHIBITS_PROCEED | Gates whether ego enters a region | Remove this node → ego takes a **qualitatively different path** (enters intersection, crosses lane, proceeds through gap). Covers: red/yellow signals, yield obligations, gap acceptance across oncoming traffic, full road blockage. |
| 4 | GRANTS_PROCEED | Permits ego to proceed into a region | Green signal, go arrow, traffic officer waving through. Exclusive to TRAFFIC_CONTROL. |
| 5 | DEGRADES | Environmental meta-effect on all channels | Reduces traction, visibility, or sensor reliability. Exclusive to ENVIRONMENT. |

### The Region-Entry Test (INHIBITS_PROCEED vs CONSTRAINS_LONGITUDINAL)

This is the critical disambiguation. Ask: **what changes if you remove the source node?**
- If ego takes a **qualitatively different path** (enters a region it currently doesn't) → **INHIBITS_PROCEED**. The source gates a go/no-go decision.
- If ego takes the **same path at a different speed** → **CONSTRAINS_LONGITUDINAL**. The source modulates speed within ego's current trajectory.

Examples:
- Red light → remove it, ego drives through the intersection. **INHIBITS_PROCEED.**
- Slow lead vehicle on straight road → remove it, ego drives the same route faster. **CONSTRAINS_LONGITUDINAL.**
- Oncoming vehicle during left turn → remove it, ego turns immediately instead of waiting. **INHIBITS_PROCEED.** But only if the oncoming vehicle is MOVING (speed > 0.1 m/s) and on the road ego is turning into. A stationary oncoming vehicle that ego's turn path clears is NOT inhibiting proceed — check at T=0 whether the vehicle is on the road ego is turning into and whether it is moving toward ego's turning path.
- Vehicle stopped at red light ahead of ego in same lane → the red light is INHIBITS_PROCEED; the stopped vehicle is **CONSTRAINS_LONGITUDINAL** (ego matches its zero speed, but the path doesn't change).
- Yield sign at unsignalized intersection → remove it, ego enters without checking. **INHIBITS_PROCEED.**
- Follower vehicle close behind → constrains how hard ego can brake. **CONSTRAINS_LONGITUDINAL.**

### CONSTRAINS_LATERAL Qualification (two-part test)

A vehicle qualifies for CONSTRAINS_LATERAL if EITHER condition is met:

**(a) BEV intrusion:** Its BEV bounding box intrudes into ego's lane polygon (overlaps or touches the grey region between ego's lane dividers). Vehicle entirely within its own lane behind a divider → does not qualify under (a).

**(b) Adjacent vehicle restricting lateral movement:** Must pass ALL three checks in order. If any check fails, stop — no CONSTRAINS_LATERAL.

**Check 1 — Disqualifiers (check FIRST, before anything else):**
- Is the vehicle in a **diverging lane**? Look at BEV lane direction arrows — if the adjacent lane's arrows curve away from ego's lane (right-turn lane splitting off, exit lane, merging lane separating), STOP. No CONSTRAINS_LATERAL. Lanes moving apart cannot restrict lateral movement.
- Is the vehicle **ahead and moving faster** than ego at T=0? If yes, STOP. It is pulling away.
- Is the vehicle in an **opposing-direction lane**? STOP.
- Is the vehicle on a **separate road** (median, grass, curbs between them on BEV, or walkway polygons separating the roads)? STOP.
- Is the vehicle **only visible in CAM_BACK**? STOP.
- Is the vehicle **small/distant** in camera (far ahead/behind)? STOP.
- Is the vehicle **two or more lanes away**? STOP.

**Check 2 — Visual (cameras):** The vehicle is prominently visible in CAM_FRONT_LEFT, CAM_FRONT_RIGHT, or the lateral edge of CAM_FRONT — appearing alongside or near-alongside ego. If a driver would think "I can't move that direction because that vehicle is right there" — it passes.

**Check 3 — Lane (BEV):** The vehicle is in the **immediately adjacent same-direction PARALLEL lane** on BEV — one lane divider separates it from ego's lane. The lane arrows must run parallel to ego's lane, not curve away. Check same-direction via BEV lane direction arrows or agent heading arrow.

All three checks must pass. Check 1 failure kills the edge immediately.

For either (a) or (b): if ego is currently steering or offset away from the vehicle at T=0 → active. If no current response but the vehicle restricts ego from moving toward it at T=0 → latent (if it passes the three latent tests).

### Multiple Edges From Same Source

A single source node may have multiple ego-edges on different control channels. For example, a wide truck partially in ego's lane can be both CONSTRAINS_LATERAL (encroaching into lane) and CONSTRAINS_LONGITUDINAL (ego must slow to pass safely). This replaces the need for a combined "evasion" edge — urgent multi-channel scenarios are captured by simultaneous edges at high priority.

### Valid Effects Per Node Type

| Node Type | Valid Ego-Edge Types |
|---|---|
| TRAFFIC_CONTROL | CONSTRAINS_LONGITUDINAL, CONSTRAINS_LATERAL, INHIBITS_PROCEED, GRANTS_PROCEED |
| ROAD_OBSTACLE | CONSTRAINS_LONGITUDINAL, CONSTRAINS_LATERAL, INHIBITS_PROCEED |
| OTHER_AGENT | CONSTRAINS_LONGITUDINAL, CONSTRAINS_LATERAL, INHIBITS_PROCEED |
| ENVIRONMENT | DEGRADES |

Note: GRANTS_PROCEED is exclusive to TRAFFIC_CONTROL. Agents don't grant permission — they fail to inhibit. DEGRADES is exclusive to ENVIRONMENT.

### Active Test (mandatory for active=true)

An ego_edge is active if ego's behavior at T=0 shows evidence of responding to this entity. All checks use **current state at T=0**, not future projections.

**FIRST — Is the entity on ego's road and in a position that affects ego at T=0?** Check BEV and cameras: is the entity in ego's lane, in an immediately adjacent lane, or on ego's road ahead? If the entity is on a separate road, different lane with no convergence, or laterally far from ego's lane at T=0 → no ego_edge.

**Speed response:** Is ego currently decelerating or holding lower speed than expected, AND is this entity ahead in ego's lane or on ego's current path at T=0?

**Lateral response:** Is ego currently offset laterally or steering, AND is this entity alongside ego in an adjacent lane or partially in ego's lane at T=0?

**Heading response:** Is ego's current heading change consistent with avoiding this entity at its T=0 position?

If ANY of these signals are present AND the entity is in a position at T=0 that explains the response → active=true. Do not project ego forward in time to "reach" a distant entity. If the entity is far ahead and ego is not yet responding, it is not active.

### Latent Edge Qualification (mandatory for active=false)

If no behavioral signal links ego's current driving to this entity, but the entity's **current position at T=0** has a real causal relationship with ego that could activate given a small change, the edge is latent. Must pass ALL three tests. If any test fails, do not create the edge.

**Test 1 — Plausible activation:** A small, realistic change in the entity's current state could make it relevant to ego. The entity is already near ego's road/lane — it doesn't need to travel far to activate. "Agent might reverse direction," "ego might need to reverse," or "agent could suddenly sprint into road from far away" are NOT plausible.

**Test 2 — No physical barrier:** There must be a clear, unobstructed path between the agent and ego's road at T=0. A stationary pedestrian on a sidewalk separated by curbs, grass, or barriers with no indication of intent to cross does NOT qualify.

**Test 3 — Spatial relevance at T=0:** The agent's **current position** must be in or near ego's lane, or in a position where a small movement would place it in ego's path. Use BEV lane structure and cameras to verify at T=0.
- **Check the agent's current (x, y) against ego's current lane on BEV.** Is the agent in ego's lane? In the immediately adjacent lane? On the same road? If the agent is laterally far from ego's lane at T=0 (e.g., y=8 while ego's lane is at y≈0), it fails regardless of heading.
- **Check history:** Is the agent's y actually getting closer to ego's lane across history entries? If not, the agent is not converging. Agent heading alone does NOT pass — "heading points toward ego's road" is speculation. Check actual history positions.
- **Passes:** Agent is in ego's lane ahead at T=0. Agent is in a lane ego is currently turning into. Agent is actively crossing toward ego's lane (history y-values converging). Same-lane follower behind ego when both stationary in a queue. CONSTRAINS_LATERAL qualified vehicles.
- **Fails:** Agent is on a separate road. Agent is in a parallel lane ego is not entering at T=0. Agent is laterally far from ego's lane with no history of converging. Agent is behind ego, stationary, and not in ego's lane (exception: CONSTRAINS_LATERAL qualified vehicles).

---

## Inter-Node Edge: AFFECTS

Single edge type for ALL relationships between non-ego nodes. Ego is NEVER source or target of an AFFECTS edge.

**Layer hierarchy:**
- Layer 1: ENVIRONMENT → other nodes
- Layer 2: TRAFFIC_CONTROL → agents approaching the infrastructure
- Layer 3: Agent ↔ Agent (same-lane queues, agent-agent proximity, cross-lane influence)

### Source → Target Validity

| Source | Can AFFECTS |
|---|---|
| ENVIRONMENT | ROAD_OBSTACLE, TRAFFIC_CONTROL, OTHER_AGENT |
| TRAFFIC_CONTROL | OTHER_AGENT |
| OTHER_AGENT | OTHER_AGENT, ROAD_OBSTACLE |
| ROAD_OBSTACLE | OTHER_AGENT |

**NEVER a source:** EGO.
**NEVER a target:** EGO.

---

## Pipeline

### Step 1: Scene Inventory
Scan all 4 cameras (t0 snapshot each), BEV sequence (4 frames), and state.json. For each entity note:
- What it is (from cameras), which cameras it appears in, position relative to ego (from state.json), speed (from state.json), whether static/moving.
- Assign a `category` based on cameras and state.json.

**Visibility requirement:** Only include entities that are visible in at least one camera image. BEV-only entities are excluded. Additionally, agent-type entities (vehicles, pedestrians, cyclists) must also be present in state.json or annotated on the BEV with a bounding box and label — without state.json data there is no authoritative position, speed, or lane_id for spatial reasoning. Note camera-only agents in scene_context.road_geometry if relevant (e.g., "parked bus visible ahead on right side of road"). ROAD_OBSTACLE and TRAFFIC_CONTROL are exempt from this requirement — they are inherently camera-identified. A stop line, crosswalk, or signal that appears on the BEV map but cannot be seen in any camera image does NOT get a node.

**Exclude:**
- Agents on sidewalks/footpaths behind barriers/railings with no path to road.
- Unoccupied bicycles (speed < 0.1, no rider in cameras).
- Infrastructure behind ego (x < 0) — no ego_edge, but may still affect other agents.
- Vehicles on separate roads — even at junctions. Check BEV: if the vehicle is on a different road segment separated from ego's road by walkway polygons (green), grass, or curbed islands, it is on a separate road regardless of whether lane connectors exist between the two roads on the map. Lane connectors on the BEV show where roads could connect — they do not mean the vehicle is on ego's road.
- BEV-only map features (stop lines, crosswalks, intersection polygons) not visible in any camera.
- Road-edge obstacles (cones, barriers, fencing) not intruding into a travel lane — describe in scene_context.road_geometry instead.
- Camera-only agents (vehicles, pedestrians, cyclists) not in state.json and not annotated on BEV — no authoritative spatial data for node creation. Describe in scene_context.road_geometry if relevant. This does NOT apply to ROAD_OBSTACLE or TRAFFIC_CONTROL, which are inherently camera-identified.

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
From state.json at T=0:
- Current speed, acceleration trend (from history), current heading intent (from future trajectory — used only to understand ego's current intent, e.g., turning left, going straight)
- Does behavior match nav command? Mismatch = something overriding navigation.

### Step 4: Build Ego Edges
For each included entity, apply the region-entry test to distinguish INHIBITS_PROCEED from CONSTRAINS_LONGITUDINAL:
- Currently constraining ego at T=0 → active ego_edge (active=true). Fill in `mechanism`.
- Passes ALL three latent qualification tests at T=0 → latent ego_edge (active=false). Fill in `mechanism`.
- Fails any latent test OR no relationship → no ego_edge (agent may still be in graph via inter_node_edges)
- A single source may produce multiple ego-edges on different control channels.
- Do not project ego forward in time to "reach" a distant entity. All edges are based on current positions at T=0.

### Step 5: Build Inter-Node Edges

**Layer 1:** If ENVIRONMENT exists → AFFECTS all affected nodes.

**Layer 2:** For each TRAFFIC_CONTROL → AFFECTS dynamic agents at the same intersection. Include coordinates of both infrastructure and agent in justification.

**Layer 3:**
- **Same-lane pairs:** Group by lane_id, sort by x, connect consecutive pairs with AFFECTS edges. Include x-positions of both agents in justification.
- **Agent proximity:** Check BEV for agents that are near each other on the same side of the road. Look for sightline blockage (agent positioned between ego and another agent on BEV), presence creation (agent adjacent to/emerging from behind another), or agent causing another to yield.
- **Cross-lane:** Only if trajectory convergence exists (merge, lane change, turn across).

### Step 6: Schema Check
- Ego never in inter_node_edges.
- No ego_edge uses "AFFECTS" as effect_type.
- Every ego_edge has a `mechanism` string.
- Every node has a `category` string.
- **No duplicate edges.** Only one edge per source-target pair per effect_type. A single source may have multiple ego-edges only if they are on different control channels (e.g., CONSTRAINS_LONGITUDINAL + CONSTRAINS_LATERAL).
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
      "type": "<TRAFFIC_CONTROL | ROAD_OBSTACLE | OTHER_AGENT | ENVIRONMENT | EGO>",
      "category": "<free-text category, e.g. vehicle.car, signal.red, cone, wet_road>",
      "label": "<descriptive, visually grounded label — use generic types (sedan, SUV, truck, bus, minivan, hatchback, pickup), NOT brand names (no Toyota, Mitsubishi, Audi, etc.)>",
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
      "effect_type": "<CONSTRAINS_LONGITUDINAL | CONSTRAINS_LATERAL | INHIBITS_PROCEED | GRANTS_PROCEED | DEGRADES>",
      "mechanism": "<short free-text phrase describing HOW the constraint operates>",
      "affected_state": ["<speed|heading|acceleration>"],
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
Navigation: "Turn left at intersection." Wet road. Yellow signal visible in CAM_FRONT. Lead sedan braking. Oncoming vehicles. Bus at stop with pedestrian nearby. SUV behind ego.

State.json: sedan_1 at x=12 (same lane, decelerating 5.5→2.1 m/s), oncoming_1 at x=30 (yaw≈π, 11.2 m/s), oncoming_2 at x=45 (yaw≈π, 10.8 m/s), bus_5 at x=8 y=-4 (stationary), pedestrian_3 at x=7 y=-4.5 (0.8 m/s, y shifting toward road), cyclist_7 at x=5 y=8 (sidewalk, 2.1 m/s), suv_12 at x=-8 (same lane, 7.0 m/s).

BEV sequence: sedan_1 bounding box displacement shrinking across frames (confirms decelerating). oncoming_1/2 moving down in BEV (approaching). bus_5 no displacement (stationary). pedestrian_3 shifting slightly toward road center.

### Reasoning Trace:

**Scene context:** 4-lane undivided urban road, signalized intersection, crosswalk markings, wet road, bus stop on right.

**Ego state:** Decelerating (8.1→6.2 m/s). Intends to turn left. Lead sedan and yellow light explain deceleration.

**Ego edges (applying the region-entry test):**
- sedan_1 → ego: **CONSTRAINS_LONGITUDINAL**, mechanism: "lead vehicle speed coupling", active=true. (Remove sedan → ego drives same path faster, doesn't enter a new region. Same-path-different-speed = longitudinal.)
- n2 (yellow signal) → ego: **INHIBITS_PROCEED**, mechanism: "yellow signal conditional stop", active=true. (Remove signal → ego enters intersection. Different path = inhibits proceed.)
- n3 (crosswalk) → ego: **INHIBITS_PROCEED**, mechanism: "unoccupied crosswalk on ego turn path", active=false. (Unoccupied, but ahead on ego's left turn path — passes latent tests.)
- oncoming_1 → ego: **INHIBITS_PROCEED**, mechanism: "gap acceptance for left turn across oncoming", active=true. (Remove oncoming → ego turns immediately instead of waiting. Go/no-go decision = inhibits proceed.)
- oncoming_2 → ego: **INHIBITS_PROCEED**, mechanism: "gap acceptance for left turn across oncoming", active=true.
- n1 (wet road) → ego: **DEGRADES**, mechanism: "wet road reduces traction and visibility", active=true.
- bus_5 → ego: **CONSTRAINS_LONGITUDINAL**, mechanism: "stationary bus narrows available road", active=true. (Remove bus → ego drives same path faster past the stop. Same-path-different-speed = longitudinal.)
- pedestrian_3 → ego: **INHIBITS_PROCEED**, mechanism: "pedestrian approaching road edge near ego turn path", active=false. (Not in road yet, but moving toward road, no barrier — passes latent tests.)
- suv_12 → ego: **CONSTRAINS_LONGITUDINAL**, mechanism: "follower proximity limits braking", active=false. (Behind ego, same lane, moving — passes latent test as same-lane follower. Constrains deceleration envelope.)
- cyclist_7 → ego: **INHIBITS_PROCEED**, mechanism: "cyclist path may intersect ego left turn trajectory", active=false. (On sidewalk, but ego's left turn path could intersect — passes latent tests.)

**Key points shown in this example:**
- No edge between agents that straddle ego in the same lane.
- Layer 2 edges only to agents approaching infrastructure, not past it.
- All Layer 3 justifications cite x-positions of both agents.
- suv_12 has no incoming Layer 3 edges — that's normal.
- No AFFECTS edge targeting ego. All ego relationships use the 5 ego_edge types.
- Multiple agents may produce INHIBITS_PROCEED edges — each represents a separate go/no-go constraint.
- The region-entry test cleanly separates sedan_1 (CONSTRAINS_LONGITUDINAL) from oncoming vehicles (INHIBITS_PROCEED).

---

## Checklist

1. One EGO node with id "ego".
2. scene_context.road_geometry and navigation_intent populated.
3. No NAVIGATION_COMMAND or ROAD_BOUNDARY_GEOMETRY nodes.
4. ENVIRONMENT only if conditions degrade driving.
5. Every node has a `category` attribute.
6. Every ego_edge has valid effect_type for its node type. No ego_edge uses AFFECTS.
7. Every ego_edge has a `mechanism` attribute.
8. Every inter_node_edge uses "AFFECTS". Ego never in inter_node_edges.
9. Inter_node_edge justifications include x-positions of source and target.
10. Node labels are descriptive and visually grounded. Use generic vehicle types (sedan, SUV, truck, bus), never brand names.
11. Output is valid JSON.
12. Every TRAFFIC_CONTROL node is visible in at least one camera. No BEV-only infrastructure nodes.
13. Every ROAD_OBSTACLE passes the intrusion test — object is ON the road surface, not on sidewalk/grass/behind curbs.
14. Every latent ego_edge (active=false) passes all three latent qualification tests at T=0.
15. INHIBITS_PROCEED vs CONSTRAINS_LONGITUDINAL correctly distinguished by the region-entry test.
16. Every active ego_edge is based on ego's behavior at T=0 responding to the entity's position at T=0. No edges based on projecting ego forward in time to "reach" a distant entity.
17. Every OTHER_AGENT node has a corresponding state.json entry or BEV annotation. No camera-only guessed agents. (ROAD_OBSTACLE and TRAFFIC_CONTROL are exempt.)
18. Every CONSTRAINS_LATERAL ego_edge passes the two-part test: either BEV intrusion (a) or all three checks for (b) — disqualifiers first, then camera visual, then parallel lane on BEV. Diverging lanes are disqualified in Check 1.
19. No duplicate edges per source-target pair per effect_type.

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