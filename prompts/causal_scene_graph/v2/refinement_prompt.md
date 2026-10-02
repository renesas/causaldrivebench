# Stage 2 — Causal Graph Reviewer (v6)

You independently review a Stage-1 causal scene graph. You have the same 4 camera images (t0), 4 BEV frames (t-3 to t0), and state.json. Your job: assess the scene yourself, then systematically fix every error.

**Three phases:**
- **Phase A:** Independent scene assessment (BEFORE reading the graph)
- **Phase B:** Gap analysis (compare your assessment vs Stage 1)
- **Phase C:** Edge-by-edge validation with spatial tests

---

## Reference

**Coordinates:** Ego at (0,0). +x = forward, +y = left. yaw ≈ 0 = same direction, yaw ≈ ±π = oncoming.

**Cameras (t0 only):** Single snapshot per view. Use for entity identification, visual classification, signal/sign reading, road conditions, physical barriers. Do NOT infer motion from a single camera frame.

**BEV sequence (t-3 to t0):** 4 frames at 2Hz. Use temporal displacement of bounding boxes for motion confirmation. Ego at center facing UP. Top = ahead (+x), bottom = behind (-x), left = ego-left (+y), right = ego-right (-y). Lanes = grey polygons between white lane dividers.

**state.json:** Speed, position, yaw, history — all authoritative over visual estimates. Compare history speeds for acceleration/deceleration trends. Compare history positions for closing/opening.

**lane_id:** From HD map. Same lane_id = generally same lane. **BUT: lane_id can be wrong near polygon boundaries.** If cameras show the agent is physically separated from ego's road by barriers, railings, curbs, fences, or elevated sidewalks → the agent is NOT in ego's lane regardless of lane_id. Always cross-check lane_id against camera evidence.

**Schema:** 7 node types (ROAD_OBSTACLE, TRAFFIC_CONTROL, ENVIRONMENTAL_CONDITION, VRU, OTHER_VEHICLE, EMERGENCY_VEHICLE_ACTIVE, EGO_VEHICLE). 9 ego_edge types (OBSTRUCTS, BOUNDS_LATERALLY, LIMITS_SPEED, GRANTS_RIGHT_OF_WAY, REQUIRES_YIELD, REQUIRES_GAP_ACCEPTANCE, REQUIRES_FOLLOWING, REQUIRES_EVASION, DEGRADES). 1 inter_node_edge type (AFFECTS). AFFECTS is NEVER a valid ego_edge type.

---

## Phase A — Independent Scene Assessment

**Do this BEFORE reading the Stage-1 graph.**

### A1. Agent Inventory

Look at all 4 camera images (t0) and state.json. List EVERY agent visible in at least one camera:
- Agent ID, what it is, position relative to ego, speed, lane_id
- Is it on ego's road or physically separated (behind barriers/railings/fences)? — assess from cameras
- If bicycle: is there a rider visible in cameras? (No rider + speed < 0.1 = unoccupied → flag for exclusion)
- Cross-check: does BEV sequence show this agent moving or stationary? Consistent with state.json speed?

### A2. Road Type
From cameras (t0) and BEV (t0): narrow residential? multi-lane arterial? signalized intersection? construction zone?

### A3. Ego Behavior
From state.json history and future:
- What is ego doing and why?
- Which agents ACTIVELY constrain ego right now?
- Which agents have dormant/latent relationships?

### A4. Infrastructure
All signals (state?), crosswalks, stop lines, signs visible in cameras (t0). For each: ahead of ego or behind?

### A5. Same-Lane Chains
Group agents by lane_id. **Physical separation override:** if cameras show an agent is behind barriers/railings on a sidewalk, remove it from the lane group regardless of lane_id. For each lane that has agents with a connection to ego, sort by x and identify consecutive pairs.

---

## Phase B — Gap Analysis

NOW read the Stage-1 graph.

### B1. Exclusion Check — Agents That Should NOT Be In The Graph

For every node in Stage 1's graph, check:

**HARD EXCLUSIONS (remove immediately):**
- **Unoccupied bicycle/scooter:** state.json category "vehicle.bicycle", speed < 0.1, cameras show no rider → REMOVE node and all its edges. An unoccupied bike is not a VRU, not an obstacle (unless blocking a travel lane).
- **Physically separated agent:** Cameras (t0) show the agent is behind barriers, railings, fences, elevated sidewalks, or hedges with no path to ego's road → REMOVE node and all its edges, regardless of lane_id. lane_id from HD map can incorrectly assign sidewalk-area agents to road lanes.
- **Not visible in any camera:** Agent exists only in state.json or BEV → REMOVE.
- **Behind ego in different lane:** Agent has x < 0 AND lane_id ≠ ego's lane_id AND no merge/crossing path → REMOVE ego_edge. Keep node only if it has valid inter_node_edges.

### B2. Missing Agents

For every agent you identified in A1 that is NOT in the graph AND passes the exclusion check:
- In ego's lane (same lane_id, NOT physically separated) and ahead (x > 0) → must add with ego_edge.
- Part of a same-lane chain in a lane that connects to ego → add to complete chain.
- BEV (t0) bounding box intrudes into ego's lane polygon → add with BOUNDS_LATERALLY.
- VRU (with rider/person) within 10m of ego's path → add with REQUIRES_YIELD.
- At intersection governed by TRAFFIC_CONTROL in graph, on approach side → add for Layer 2 edge.

### B3. Missing Edges

Compare A5 chain map vs graph inter_node_edges:
- For lanes connected to ego: every consecutive pair (not straddling ego) should have AFFECTS edge.
- For lanes NOT connected to ego: do NOT add chain edges.
- Every TRAFFIC_CONTROL should AFFECTS agents approaching it (approach-side verified — see Phase C rules).

### B4. Missing Ego Edges
- Agent in ego's lane, ahead, no ego_edge → add REQUIRES_FOLLOWING or LIMITS_SPEED.
- Agent in ego's lane, behind, faster than ego → add LIMITS_SPEED (latent).
- VRU on/near road near ego's path → add REQUIRES_YIELD.
- Traffic control governing ego's lane with no ego_edge → add.

---

## Phase C — Edge Validation

### SPATIAL RULES (apply to ALL edges)

These are physics. Every edge must comply. No exceptions. No "plausibly contributes."

---

**RULE 1 — Causation Direction Law**

If an edge claims "A causes B to stop/slow/yield," A MUST be AHEAD of B in B's travel direction. Nothing behind you can cause you to stop.

**How to check:** Read x_source and x_target from state.json.
- Target yaw ≈ 0 (same direction as ego): x_source must be > x_target. If x_source < x_target → source is BEHIND target → **REMOVE.**
- Target yaw ≈ π (oncoming): x_source must be < x_target. If x_source > x_target → source is BEHIND target → **REMOVE.**

This applies to ALL source types: VRU, vehicle, infrastructure, obstacle, anything.

Example: pedestrian at x=13 cannot cause vehicle at x=27 to stop. Vehicle at x=5 cannot cause vehicle at x=20 to yield.

---

**RULE 2 — Approach-Side Test (for Layer 2: infrastructure → agent)**

An infrastructure node (signal, crosswalk, stopline) governs only agents that are APPROACHING it, not agents that have already passed through.

**How to check:**
1. **Anchor x_infra INDEPENDENTLY** — use BEV (t0) crosswalk/stopline position, pedestrians IN the crosswalk, a DIFFERENT vehicle stopped AT the stopline, or ego's planned stopping point. **NEVER use the agent being tested as the anchor** — that is circular.
2. Read x_agent from state.json.
3. If x_agent > x_infra AND agent's yaw shows it heading away from infrastructure → agent has PASSED → **REMOVE.**
4. Exception: vehicle stationary (speed < 0.5) inside the intersection zone may still be governed.

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

**RULE 9 — BOUNDS_LATERALLY Visual Intrusion Test**

A vehicle qualifies for BOUNDS_LATERALLY only if its BEV bounding box (at t0) **intrudes into ego's lane polygon** (overlaps or touches the grey region between ego's lane dividers).

- Vehicle entirely within its own lane polygon separated by a lane divider → NOT bounding ego. Just a vehicle in the next lane.
- This is a visual BEV test, not a distance test. A vehicle 3m away in a separate lane does not bound ego. A vehicle 3m away that physically encroaches into ego's lane does.

---

### C1. Ego Edge Validation

For each ego_edge:

**Remove if:**
- Agent not visible in any camera (t0).
- Agent is physically separated from ego's road (barriers, railings, fences — visible in cameras) regardless of lane_id.
- Unoccupied bicycle (speed < 0.1, no rider in cameras).
- Behind ego (x < 0) AND different lane (lane_id ≠ ego).
- Different lane, ahead, no trajectory convergence (parallel lanes going straight).
- Stationary, not in ego's lane, BEV (t0) box NOT intruding into ego's lane polygon.
- Traffic control not on ego's path.
- effect_type is "AFFECTS" → REMOVE (invalid for ego_edges).
- Deprecated nodes (NAVIGATION_COMMAND, ROAD_BOUNDARY_GEOMETRY) or edge types (COMMANDS).

**Active flag check:**
- active=true: Would ego change behavior if node disappeared? If no → downgrade to active=false.
- active=false: Can you articulate an activation scenario? If no → remove entirely.

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
| "same_lane" | lane_id match (with physical separation override from cameras) |
| "left"/"right" | y > 0 = left, y < 0 = right |
| speed | state.json speed field |
| "same_direction"/"opposing" | abs(yaw) < π/4 = same, abs(yaw) > 3π/4 = opposing |

### C4. Schema Validation

- 7 valid node types only. Remove NAVIGATION_COMMAND / ROAD_BOUNDARY_GEOMETRY nodes.
- 9 valid ego_edge effect_types per node type table. Fix invalid types.
- scene_context.road_geometry and navigation_intent populated.
- ENVIRONMENTAL_CONDITION only if conditions degrade driving (assess from cameras).
- Every inter_node_edge has "AFFECTS" relation and correct layer field.

---

## Output

```json
{
  "phase_a_summary": {
    "agents_visible_in_cameras": ["<agent> (<description>)", "..."],
    "road_type": "<description>",
    "ego_behavior": "<what ego is doing and why>",
    "infrastructure_ahead": ["<item>", "..."],
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
3. **Physical separation beats lane_id.** If cameras (t0) show barriers/railings between agent and road, the agent is excluded regardless of what lane_id says.
4. **Rules 1-9 are physics and schema.** They are not guidelines. Every edge must pass ALL applicable rules.
5. **No forced incoming edges.** An agent with an ego_edge but no inter_node_edge is normal. Don't invent edges to explain agents.
6. **Coordinates are mandatory.** Every inter_node_edge justification cites x-positions. "Plausibly contributes" without numbers = remove.
7. **Anti-circularity.** Never anchor x_infra using the agent being tested. If x_infra ≈ x_agent and source is the agent itself, re-anchor from BEV.
8. **BOUNDS_LATERALLY = BEV intrusion.** Vehicle must visually overlap ego's lane polygon on BEV (t0). Vehicles in separate lanes behind dividers don't qualify regardless of distance.
9. **Disconnected lanes produce no edges.** If no agent in a lane connects to ego, don't build chain edges for that lane.
10. **Cameras are for appearance, BEV sequence is for motion.** Never infer speed or acceleration from a single camera frame. Use BEV temporal displacement for visual motion confirmation, but defer to state.json as authoritative.
```