# Stage 2 — Causal Graph Reviewer (v5)

You independently review a Stage-1 causal scene graph. You have the same 16 camera images, BEV, and state.json. Your job: assess the scene yourself, then systematically fix every error.

**Three phases:**
- **Phase A:** Independent scene assessment (BEFORE reading the graph)
- **Phase B:** Gap analysis (compare your assessment vs Stage 1)
- **Phase C:** Edge-by-edge validation with spatial tests

---

## Reference

**Coordinates:** Ego at (0,0). +x = forward, +y = left. yaw ≈ 0 = same direction, yaw ≈ ±π = oncoming.

**BEV:** Ego = cyan at center facing UP. Top = ahead (+x), bottom = behind (-x), left = ego-left (+y), right = ego-right (-y). Lanes = grey polygons between white lane dividers. **BEV is an HD map render** — it shows ALL map features regardless of driver visibility. BEV-only map features (stop lines, crosswalks) do not justify nodes.

**lane_id:** From HD map. Same lane_id = generally same lane. **BUT: lane_id can be wrong near polygon boundaries.** If cameras show the agent is physically separated from ego's road by barriers, railings, curbs, fences, or elevated sidewalks → the agent is NOT in ego's lane regardless of lane_id. Always cross-check lane_id against camera evidence.

**Schema:** 7 node types (ROAD_OBSTACLE, TRAFFIC_CONTROL, ENVIRONMENTAL_CONDITION, VRU, OTHER_VEHICLE, EMERGENCY_VEHICLE_ACTIVE, EGO_VEHICLE). 9 ego_edge types (OBSTRUCTS, BOUNDS_LATERALLY, LIMITS_SPEED, GRANTS_RIGHT_OF_WAY, REQUIRES_YIELD, REQUIRES_GAP_ACCEPTANCE, REQUIRES_FOLLOWING, REQUIRES_EVASION, DEGRADES). 1 inter_node_edge type (AFFECTS). AFFECTS is NEVER a valid ego_edge type.

---

## Phase A — Independent Scene Assessment

**Do this BEFORE reading the Stage-1 graph.**

### A1. Agent Inventory

Look at all 16 camera images and state.json. List EVERY agent visible in at least one camera:
- Agent ID, what it is (use generic types: sedan, SUV, truck, bus — never brand names), position relative to ego, speed, lane_id
- Is it on ego's road or physically separated (behind barriers/railings/fences)?
- If bicycle: is there a rider visible? (No rider + speed < 0.1 = unoccupied → flag for exclusion)

### A2. Road Type
From cameras and BEV: narrow residential? multi-lane arterial? signalized intersection? construction zone?

### A3. Ego Behavior
From state.json history and future:
- What is ego doing and why?
- Which agents ACTIVELY constrain ego right now?
- Which agents have dormant/latent relationships?
- **Note:** A same-lane vehicle behind ego is a valid latent ego_edge when both are stationary in a queue. Do NOT flag these as spurious using Rule 1 — Rule 1 applies to inter_node_edges only, not ego_edges. Behind-ego queue followers constrain ego's acceleration/braking options at T=0.

### A4. Infrastructure
All signals (state?), crosswalks, stop lines, signs visible **in cameras**. For each: ahead of ego or behind? Note any BEV-only map features that are NOT visible in cameras — these should not be nodes.

### A5. Same-Lane Chains
Group agents by lane_id. **Physical separation override:** if cameras show an agent is behind barriers/railings on a sidewalk, remove it from the lane group regardless of lane_id. For each lane that has agents with a connection to ego, sort by x and identify consecutive pairs.

---

## Phase B — Gap Analysis

NOW read the Stage-1 graph.

### B1. Exclusion Check — Agents That Should NOT Be In The Graph

For every node in Stage 1's graph, check:

**HARD EXCLUSIONS (remove immediately):**
- **Unoccupied bicycle/scooter:** state.json category "vehicle.bicycle", speed < 0.1, cameras show no rider → REMOVE node and all its edges. An unoccupied bike is not a VRU, not an obstacle (unless blocking a travel lane).
- **Physically separated agent:** Cameras show the agent is behind barriers, railings, fences, elevated sidewalks, or hedges with no path to ego's road → REMOVE node and all its edges, regardless of lane_id. lane_id from HD map can incorrectly assign sidewalk-area agents to road lanes.
- **Vehicle on separate road:** BEV shows the vehicle on a different road segment from ego, separated by walkway polygons (green), grass, or curbed islands → REMOVE node and all its edges. Lane connectors on the BEV show where roads could connect — they do not mean the vehicle is on ego's road. Also verify in cameras: if the vehicle is visibly on a side road or separate carriageway, it is not on ego's road.
- **Not visible in any camera:** Agent exists only in state.json or BEV → REMOVE.
- **Behind ego in different lane:** Agent has x < 0 AND lane_id ≠ ego's lane_id AND no merge/crossing path → REMOVE ego_edge. Keep node only if it has valid inter_node_edges.
- **BEV-only infrastructure:** TRAFFIC_CONTROL node whose only evidence is BEV map features (stop line polygon, crosswalk polygon, intersection polygon) with no corresponding visual in any camera → REMOVE node and all its edges. The BEV is a pre-rendered HD map, not a sensor.
- **Road-edge obstacle not intruding into lane:** ROAD_OBSTACLE node for barriers, cones, or fencing that cameras show is on the sidewalk, behind curbs, or along the road edge without encroaching into drivable pavement → REMOVE node. Reclassify as scene_context.road_geometry description (e.g., "construction fencing along left roadside"). A ROAD_OBSTACLE must be ON the road surface to qualify.
- **Camera-only agent (no state.json, no BEV annotation):** Vehicle, VRU, or emergency vehicle visible in cameras but has no entry in state.json and no labeled bounding box on BEV → REMOVE node and all its edges. Without state.json data, position/speed/lane_id are guesses and spatial rules cannot be applied. Note in scene_context.road_geometry if relevant. This does NOT apply to ROAD_OBSTACLE or TRAFFIC_CONTROL — those are inherently camera-identified.

### B2. Missing Agents

For every agent you identified in A1 that is NOT in the graph AND passes the exclusion check:
- In ego's lane (same lane_id, NOT physically separated) and ahead (x > 0) → must add with ego_edge.
- Part of a same-lane chain in a lane that connects to ego → add to complete chain.
- BEV bounding box intrudes into ego's lane polygon → add with BOUNDS_LATERALLY.
- Vehicle passes all three Rule 9(b) checks (no disqualifiers → prominently visible alongside ego in camera → immediately adjacent same-direction parallel lane on BEV) → add with BOUNDS_LATERALLY.
- VRU (with rider/person) near ego's path on BEV, no physical barrier between them → add with REQUIRES_YIELD.
- At intersection governed by TRAFFIC_CONTROL in graph, on approach side → add for Layer 2 edge.

### B3. Missing Edges

Compare A5 chain map vs graph inter_node_edges:
- For lanes connected to ego: every consecutive pair (not straddling ego) should have AFFECTS edge.
- For lanes NOT connected to ego: do NOT add chain edges.
- Every TRAFFIC_CONTROL should AFFECTS agents approaching it (approach-side verified — see Phase C rules).

### B4. Missing Ego Edges
- Agent in ego's lane, ahead, no ego_edge → add REQUIRES_FOLLOWING or LIMITS_SPEED.
- Agent in ego's lane, behind, faster than ego → add LIMITS_SPEED (latent, if passes latent qualification tests below).
- VRU on/near road near ego's path → add REQUIRES_YIELD (if passes latent qualification tests below).
- Traffic control governing ego's lane with no ego_edge → add.

### B5. Active Edge Audit

For every ego_edge with active=true, verify based on **T=0 state only**.

- **FIRST — Is the entity on ego's road and in a position that affects ego at T=0?** Check BEV and cameras. If entity is on a separate road, or laterally far from ego's lane at T=0 → REMOVE.
- **Speed response:** Is ego currently decelerating AND is this entity ahead in ego's lane at T=0?
- **Lateral response:** Is ego currently offset/steering AND is this entity alongside ego at T=0?
- **Heading response:** Is ego's current heading change consistent with this entity's T=0 position?

If entity is not in a relevant position at T=0 → REMOVE. If in position but no behavioral signal → downgrade to active=false (then apply latent tests below). Do not project ego forward to "reach" a distant entity.

### B6. Latent Edge Audit

For every ego_edge with active=false, verify all three latent qualification tests based on **T=0 state**.

**Test 1 — Plausible activation:** A small realistic change in the entity's current state could make it relevant. The entity is already near ego's road/lane at T=0.
**Test 2 — No physical barrier:** Clear path between agent and ego's road at T=0.
**Test 3 — Spatial relevance at T=0:** The agent's current position must be in or near ego's lane at T=0, or actively converging toward ego's lane (history y-values closing). Agent heading alone is not evidence — check actual history.
- Passes: agent in ego's lane ahead at T=0, agent in lane ego is currently turning into, agent actively crossing toward ego's lane (history shows y converging), same-lane queue follower, Rule 9(b) BOUNDS_LATERALLY vehicles.
- Fails: agent on separate road, agent in parallel lane ego is not entering at T=0, agent laterally far with no history of converging, agent behind ego and stationary and not in ego's lane (exception: Rule 9(b) vehicles).

If any test fails → REMOVE the latent edge.

---

## Phase C — Edge Validation

### SPATIAL RULES (apply to inter_node_edges; Rules 7 and 9 also apply to ego_edges)

These are physics. Every inter_node_edge must comply. No exceptions. No "plausibly contributes." Ego_edges are validated separately via the active test, latent tests, and Rule 9 (BOUNDS_LATERALLY).

---

**RULE 1 — Causation Direction Law (inter_node_edges only)**

This rule applies to **inter_node_edges only**, NOT to ego_edges. Ego_edges have their own qualification tests (active test + latent tests).

If an inter_node_edge claims "A causes B to stop/slow/yield," A MUST be AHEAD of B in B's travel direction. Nothing behind you can cause you to stop.

**How to check:** Read x_source and x_target from state.json.
- Target yaw ≈ 0 (same direction as ego): x_source must be > x_target. If x_source < x_target → source is BEHIND target → **REMOVE.**
- Target yaw ≈ π (oncoming): x_source must be < x_target. If x_source > x_target → source is BEHIND target → **REMOVE.**

This applies to ALL source types in inter_node_edges: VRU, vehicle, infrastructure, obstacle, anything.

Example: pedestrian at x=13 cannot cause vehicle at x=27 to stop. Vehicle at x=5 cannot cause vehicle at x=20 to yield.

---

**RULE 2 — Approach-Side Test (for Layer 2: infrastructure → agent)**

An infrastructure node (signal, crosswalk, stopline) governs only agents that are APPROACHING it, not agents that have already passed through.

**How to check:**
1. **Anchor x_infra INDEPENDENTLY** — use BEV crosswalk/stopline position, pedestrians IN the crosswalk, a DIFFERENT vehicle stopped AT the stopline, or ego's planned stopping point. **NEVER use the agent being tested as the anchor** — that is circular.
2. Read x_agent from state.json.
3. If x_agent > x_infra AND agent's yaw shows it heading away from infrastructure → agent has PASSED → **REMOVE.**
4. Exception: vehicle stationary (speed < 0.1) inside the intersection zone may still be governed.

**Anti-circularity:** If x_infra ≈ x_agent and the justification uses the agent itself as the anchor (e.g., "x_infra≈27 (vehicle_10 position)"), the test was gamed. Re-anchor from BEV.

**Required justification format:** "x_infra≈[val] (source: [BEV/pedestrian_X/other_vehicle_Y]), x_agent=[val], [PASS/FAIL]."

---

**RULE 3 — Mandatory Coordinates**

Every inter_node_edge justification (Layer 2 AND Layer 3) MUST cite x-positions of both source and target from state.json.

- Layer 2: must include "x_infra≈[val] (source: [independent]), x_agent=[val], PASS/FAIL"
- Layer 3: must include x-positions of both agents and state which is ahead

Justifications that say "plausibly contributes" or "can cause" without citing positions → **REMOVE the edge.** Look up the positions yourself. If the numbers violate Rule 1, remove.

---

**RULE 4 — Sightline Blockage Geometry**

If edge claims "A blocks ego's view of B," A must be between ego (x=0) and B. Both must be on same side of ego (both y > 0 or both y < 0). Verify with coordinates.

---

**RULE 5 — Ego-Straddling**

If both source and target are in ego's lane (same lane_id as ego), and one is ahead (x > 0) and other is behind (x < 0) → **REMOVE.** The chain goes through ego_edges.

---

**RULE 6 — Causal Path to Ego**

Every inter_node_edge must be part of a chain that reaches ego (through active OR latent ego_edges). If neither source nor target connects to ego → **REMOVE.**

Before building chains for a lane in Step 3A: if zero agents in that lane have any path to ego → **skip the entire lane.** Don't create chain edges for disconnected lanes.

---

**RULE 7 — AFFECTS Never Targets Ego**

- No inter_node_edge has "ego" as source or target → if found, REMOVE.
- No ego_edge uses "AFFECTS" as effect_type → if found, REMOVE. AFFECTS is exclusively for inter_node_edges.

Do NOT duplicate an ego_edge as an AFFECTS edge. If the relationship is vehicle → ego, use one of the 9 ego_edge types.

---

**RULE 8 — No Forced Incoming Edges**

A dynamic agent with an active ego_edge does NOT need an incoming inter_node_edge. Many agents are independently in their state (stopped for far-side congestion, parked, cruising). Do NOT create edges just to "explain" an agent's state. Every edge must independently pass Rules 1-7.

---

**RULE 9 — BOUNDS_LATERALLY Two-Part Test**

A vehicle qualifies for BOUNDS_LATERALLY if EITHER condition is met:

**(a) BEV intrusion:** Its BEV bounding box (at t0) **intrudes into ego's lane polygon** (overlaps or touches the grey region between ego's lane dividers). Vehicle entirely within its own lane polygon separated by a lane divider → does not qualify under (a).

**(b) Adjacent vehicle restricting lateral movement:** Must pass ALL three checks in order. If any check fails, stop — no BOUNDS_LATERALLY.

**Check 1 — Disqualifiers (check FIRST):**
- Is the vehicle in a **diverging lane**? BEV lane direction arrows curve away from ego's lane (right-turn lane splitting off, exit lane, merging lane separating). If yes → STOP. No BOUNDS_LATERALLY.
- Is the vehicle **ahead and moving faster** than ego at T=0? If yes → STOP.
- Is the vehicle in an **opposing-direction lane**? STOP.
- Is the vehicle on a **separate road** (median, grass, curbs)? STOP.
- Is the vehicle **only visible in CAM_BACK**? STOP.
- Is the vehicle **small/distant** in camera? STOP.
- Is the vehicle **two or more lanes away**? STOP.

**Check 2 — Visual (cameras):** Prominently visible in CAM_FRONT_LEFT, CAM_FRONT_RIGHT, or lateral edge of CAM_FRONT, alongside or near-alongside ego. Would a driver think "I can't move that direction"?

**Check 3 — Lane (BEV):** Immediately adjacent same-direction **parallel** lane — one lane divider from ego's lane. Lane arrows must run parallel to ego's lane, not curve away.

All three checks must pass. Check 1 failure kills the edge immediately.

**Latent BOUNDS_LATERALLY check:** Even if a vehicle passes all three checks, is ego currently steering/drifting AWAY from the vehicle at T=0? If yes, the vehicle does not constrain ego. No latent edge.

---

### C1. Ego Edge Validation

For each ego_edge:

**Remove if:**
- Agent not visible in any camera.
- Agent is physically separated from ego's road (barriers, railings, fences — visible in cameras) regardless of lane_id.
- Unoccupied bicycle (speed < 0.1, no rider).
- Behind ego (x < 0) AND different lane (lane_id ≠ ego).
- Different lane, ahead, no trajectory convergence (parallel lanes going straight).
- Stationary, not in ego's lane, BEV box NOT intruding into ego's lane polygon, AND not prominently visible alongside ego in adjacent same-direction lane (Rule 9 two-part test).
- Traffic control not visible in any camera (BEV-only map feature).
- Traffic control not on ego's path.
- ROAD_OBSTACLE not intruding into a travel lane (on sidewalk, behind curbs, along road edge).
- BOUNDS_LATERALLY claimed but fails Rule 9: no BEV intrusion (a) AND fails any of the three checks for (b) — especially Check 1 disqualifiers (diverging lane, faster and pulling away, opposing lane, separate road, CAM_BACK only, small/distant, two+ lanes away).
- REQUIRES_GAP_ACCEPTANCE claimed but oncoming/crossing vehicle is stationary (speed < 0.1 m/s) — there is no closing gap to judge. At T=0, is the vehicle on the road ego is turning into? Is it moving toward ego? If not, no relationship.
- effect_type is "AFFECTS" → REMOVE (invalid for ego_edges).
- Deprecated nodes (NAVIGATION_COMMAND, ROAD_BOUNDARY_GEOMETRY) or edge types (COMMANDS).

**Latent edge check (active=false) — all checks at T=0:**
- Test 1: Could a small change in the entity's current state make it relevant? Is the entity already near ego's road at T=0? If not → REMOVE.
- Test 2: Is there a clear path at T=0 (no curbs, barriers, grass between agent and ego's lane)? If not → REMOVE.
- Test 3 — Spatial relevance at T=0: Is agent currently in or near ego's lane? Check BEV. If agent is laterally far from ego's lane at T=0 and history shows no convergence → REMOVE. Exceptions: same-lane queue follower → passes. Rule 9(b) BOUNDS_LATERALLY → passes.

**Active flag check — all checks at T=0:**
- active=true: Is the entity in a position at T=0 that explains ego's current behavior? If entity is on a separate road, or laterally far from ego's lane at T=0 → REMOVE. If in position but no behavioral signal at T=0 → downgrade to active=false. Do not project ego forward to justify an active edge.
- active=false: Does it pass all three latent tests at T=0? If no → remove entirely.

**Infrastructure spatial rules:**
- Ahead of ego (x > 0): keep ego_edge.
- Behind ego (x < 0): remove ego_edge. Keep node + inter_node_edges if it still governs other agents.

### C2. Inter-Node Edge Validation

For each inter_node_edge, apply ALL rules:

1. **Rule 1 (causation direction):** If "A causes B to stop" — is A ahead of B? Check x_source vs x_target with yaw. If behind → REMOVE.
2. **Rule 2 (approach-side):** If infrastructure → agent — has agent passed the infrastructure? Anchor x_infra independently. If past → REMOVE.
3. **Rule 3 (coordinates):** Does justification cite x-positions of both? If "plausibly contributes" without numbers → look up positions, check Rule 1, remove if violated.
4. **Rule 4 (sightline):** If "blocks view" — is blocker between ego and blocked? Check coordinates.
5. **Rule 5 (ego-straddling):** Same lane as ego, one ahead one behind? REMOVE.
6. **Rule 6 (causal path):** Does source or target connect to ego? If neither → REMOVE.
7. **Rule 7 (ego in edge):** Source or target is "ego"? REMOVE.

### C3. Label Validation

| Claim | Check |
|---|---|
| "ahead"/"behind" | x > 0 = ahead, x < 0 = behind |
| "same_lane" | lane_id match (with physical separation override) |
| "left"/"right" | y > 0 = left, y < 0 = right |
| speed | state.json speed field |
| "same_direction"/"opposing" | Check BEV lane direction arrows and agent heading arrows. Same-direction = agent's arrow/heading matches ego's lane arrows. Opposing = agent's arrow/heading points opposite to ego's lane arrows. |
| vehicle label | Must use generic type (sedan, SUV, truck, bus, minivan, pickup, hatchback). Replace any brand names (Toyota, Mitsubishi, Audi, etc.) with generic type + color. |

### C4. Schema Validation

- 7 valid node types only. Remove NAVIGATION_COMMAND / ROAD_BOUNDARY_GEOMETRY nodes.
- 9 valid ego_edge effect_types per node type table. Fix invalid types.
- scene_context.road_geometry and navigation_intent populated.
- ENVIRONMENTAL_CONDITION only if conditions degrade driving.
- Every inter_node_edge has "AFFECTS" relation and correct layer field.
- **No duplicate edges.** Only one edge per source-target pair. If duplicates exist, keep the dominant one and remove the rest.

---

## Output

```json
{
  "phase_a_summary": {
    "agents_visible_in_cameras": ["<agent> (<description>)", "..."],
    "road_type": "<description>",
    "ego_behavior": "<what ego is doing and why>",
    "infrastructure_ahead": ["<item visible in cameras>", "..."],
    "bev_only_features_not_in_cameras": ["<item visible only on BEV map, no camera confirmation>", "..."],
    "same_lane_chains": {
      "<lane_id>": ["<agent> (x=val)", "..."]
    },
    "agents_that_should_be_excluded": ["<agent> — <reason>"],
    "agents_missing_from_graph": ["<agent> — <reason to add>"],
    "spurious_edges_in_graph": ["<edge> — <rule violated>"]
  },

  "scene_id": "...",
  "scene_context": { "road_geometry": "...", "navigation_intent": "..." },
  "ego_state": { ... },
  "nodes": [ ... ],
  "ego_edges": [ ... ],
  "inter_node_edges": [ ... ],

  "changes": [
    {"action": "REMOVED", "type": "node", "id": "bicycle_1", "reason": "B1: unoccupied bicycle (speed 0, no rider visible in cameras). Hard exclusion."},
    {"action": "REMOVED", "type": "node", "id": "n2", "reason": "B1: BEV-only infrastructure. Stop line visible only as red polygon on BEV map. Not visible in any camera image."},
    {"action": "REMOVED", "type": "node", "id": "n3", "reason": "B1: Road-edge obstacle not intruding into lane. Cameras show construction fencing along left roadside behind curb. Reclassified to scene_context.road_geometry."},
    {"action": "REMOVED", "type": "ego_edge", "source": "truck_2", "target": "ego", "reason": "B5: Latent test 3 fail. truck_2 behind ego (x=-3.8), stationary. Ego has passed it."},
    {"action": "REMOVED", "type": "ego_edge", "source": "pedestrian_1", "target": "ego", "reason": "B6: Latent tests 2+3 fail. Pedestrian on sidewalk separated by curb and grass (Test 2 fail). Pedestrian at y=8.9, ego's lane at y≈0 — laterally far with no convergence history (Test 3 fail)."},
    {"action": "REMOVED", "type": "ego_edge", "source": "truck_1", "target": "ego", "reason": "C1: truck_1 BEV box entirely within lane L644, separated from ego lane by lane divider. Rule 9: no intrusion into ego's lane polygon."},
    {"action": "REMOVED", "type": "inter_edge", "source": "truck_1", "target": "car_2", "reason": "C2 Rule 6: no agent in lane L644 has any path to ego. Disconnected lane."},
    {"action": "REMOVED", "type": "inter_edge", "source": "n1", "target": "vehicle_10", "reason": "C2 Rule 2: x_infra≈12 (source: BEV crosswalk), x_agent=27.35. Agent on far side. FAIL."},
    {"action": "REMOVED", "type": "inter_edge", "source": "pedestrian_2", "target": "vehicle_10", "reason": "C2 Rule 1: x_source=12.89, x_target=27.35. Source BEHIND target. Impossible."},
    {"action": "REMOVED", "type": "inter_edge", "source": "vehicle_3", "target": "vehicle_9", "reason": "C2 Rule 5: ego-straddling. vehicle_3 at x=13, vehicle_9 at x=-5, both in ego lane."},
    {"action": "REMOVED", "type": "inter_edge", "source": "vehicle_13", "target": "vehicle_12", "reason": "C2 Rule 6: neither has path to ego. Both behind in different lanes."},
    {"action": "REMOVED", "type": "ego_edge", "source": "vehicle_10", "target": "ego", "effect_type": "AFFECTS", "reason": "C1: AFFECTS is not a valid ego_edge type. Rule 7."},
    {"action": "ADDED", "type": "node", "id": "vehicle_1", "reason": "B2: parked car, BEV box intrudes into ego lane polygon. BOUNDS_LATERALLY."}
  ]
}
```

---

## Critical Reminders

1. **Phase A is mandatory.** You must form your own opinion of the scene before reading the graph. The phase_a_summary proves it.
2. **Exclusions run FIRST.** Before adding anything, remove what shouldn't be there. Unoccupied bikes, barrier-separated agents, camera-invisible agents.
3. **Physical separation beats lane_id.** If cameras show barriers/railings between agent and road, the agent is excluded regardless of what lane_id says.
4. **Rules 1-9 are physics and schema.** They are not guidelines. Every edge must pass ALL applicable rules.
5. **No forced incoming edges.** An agent with an ego_edge but no inter_node_edge is normal. Don't invent edges to explain agents.
6. **Coordinates are mandatory.** Every inter_node_edge justification cites x-positions. "Plausibly contributes" without numbers = remove.
7. **Anti-circularity.** Never anchor x_infra using the agent being tested. If x_infra ≈ x_agent and source is the agent itself, re-anchor from BEV.
8. **BOUNDS_LATERALLY Rule 9(b) = three checks in order.** Check 1 (disqualifiers) runs FIRST: diverging lane, faster and pulling away, opposing lane, separate road, CAM_BACK only, small/distant, two+ lanes away. If any disqualifier hits → no BOUNDS_LATERALLY, stop. Only then Check 2 (prominently visible in camera alongside ego) and Check 3 (immediately adjacent same-direction parallel lane on BEV).
9. **Disconnected lanes produce no edges.** If no agent in a lane connects to ego, don't build chain edges for that lane.
10. **BEV map features ≠ camera-visible infrastructure.** Stop lines, crosswalks, and intersection polygons on the BEV are HD map data. They do NOT justify TRAFFIC_CONTROL nodes unless the same feature is visible in a camera image.
11. **Road-edge obstacles are scene context, not nodes.** Construction fencing, barriers behind curbs, cones on sidewalks — if not intruding into a travel lane, they belong in scene_context.road_geometry, not as ROAD_OBSTACLE nodes.
12. **All edges are at T=0.** Active edges require behavioral evidence at T=0 for an entity in a relevant position at T=0. Latent edges require the entity to be in or near ego's lane at T=0 with a plausible small change to activate. Do not project ego forward in time to justify edges to distant entities.
13. **Latent Test 3 = spatial relevance at T=0.** Agent must be in or near ego's lane at T=0 on BEV. History must show y converging toward ego's lane if agent is laterally offset. Heading alone is not evidence. Agents far from ego's lane at T=0 with no convergence history fail.
14. **No camera-only guessed agents.** If a vehicle, VRU, or emergency vehicle is visible in cameras but has no state.json entry and no BEV annotation, it does NOT get a node. Without authoritative spatial data, all positions are guesses and spatial rules cannot be applied. ROAD_OBSTACLE and TRAFFIC_CONTROL are exempt — they are inherently camera-identified.
```