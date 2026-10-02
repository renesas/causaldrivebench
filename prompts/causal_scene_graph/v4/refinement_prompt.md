# Stage 2 — Causal Graph Reviewer

You independently review a Stage-1 causal scene graph. You receive: (1) the same 16 camera images (4 views × 4 timesteps), BEV (T=0), state.json (T=0), and navigation command as Stage 1; and (2) the Stage-1 JSON output.

**Three phases:**
- **Phase A:** Independent scene assessment (BEFORE reading the graph)
- **Phase B:** Gap analysis (compare your assessment vs Stage 1)
- **Phase C:** Edge-by-edge validation with spatial tests

---

## 1. Reference

**Coordinates:** Ego at (0,0). +x = forward, +y = left. yaw ≈ 0 = same direction, ≈ ±π = oncoming.

**Cameras:** 16 images — 4 timesteps (T=-1.5, T=-1.0, T=-0.5, T=0) × 4 views (FRONT, FRONT_LEFT, FRONT_RIGHT, BACK). Compare same view across timesteps for motion. State.json speed is authoritative.

**BEV (T=0):** HD map render, NOT sensor. Ego at center facing UP. Vehicles = peach boxes (moving have heading arrows; stationary have none — use white lane direction arrows). Pedestrians = green. Cyclists = yellow. Lanes = grey polygons between boundaries. Intersections = purple. Crosswalks = orange.

**Lane membership:** Trace lane boundaries on BEV (dividers, center lines, road edges). NOT state.json y-offset. On undivided roads, same polygon may contain both directions — check lane direction arrows. Physical separation in cameras overrides BEV position.

**Traffic side and safe-default position:** Infer from BEV lane direction arrows. Ego's lane LEFT of center line → left-hand traffic. RIGHT of center line → right-hand traffic. Ego's **safe-default position** = side AWAY from center line. Entity at safe-default side pushes ego toward center line / oncoming traffic.

**Temporal observation:** Use ALL timesteps (T=-1.5 to T=0) for entity assessment. An entity ahead at T=-1.5 but behind at T=0 means ego passed it — ego's current state may reflect that interaction. Check front/side cameras at earlier timesteps, not just T=0.

**Causal status:**
- **Active-sustaining:** do(remove) at T=0 changes ego's next action. Evidence: speed/y/heading history across T=-1.5 to T=0.
- **Active-triggering:** Entity initiated ego's committed maneuver during window, but removal at T=0 alone wouldn't change trajectory.
- **Dormant:** No effect in window, but single-step transition (0.5s) at T=0 would create one.
- **Distractor:** No causal path AND fails reachability.

**Schema summary:**
- **5 node types:** TRAFFIC_CONTROL, ROAD_OBSTACLE, OTHER_AGENT, ENVIRONMENT, EGO.
- **5 ego-edge types:** CONSTRAINS_LONGITUDINAL, CONSTRAINS_LATERAL, INHIBITS_PROCEED, GRANTS_PROCEED, DEGRADES.
- **2 inter_node_edge types:** AFFECTS (directed, layer 1/2/3), CONFOUNDED_BY (bidirected, layer N/A, inferred_cause required).
- **Valid effects:** TC→{INHIBITS_PROCEED, GRANTS_PROCEED}. OBSTACLE→{CL_LONG, CL_LAT, INHIBITS}. AGENT→{CL_LONG, CL_LAT, INHIBITS}. ENV→{DEGRADES}.
- **affected_state:** CL_LONG→["speed","acceleration"]. CL_LAT→["heading"]. INHIBITS/GRANTS/DEGRADES→["speed","heading","acceleration"].
- **Co-occurrence:** Partial blockage always produces BOTH CL_LAT + CL_LONG. Full blockage → INHIBITS_PROCEED.
- **Deprecated node types:** VRU, OTHER_VEHICLE, EMERGENCY_VEHICLE_ACTIVE, ENVIRONMENTAL_CONDITION, EGO_VEHICLE, NAVIGATION_COMMAND, ROAD_BOUNDARY_GEOMETRY.
- **Deprecated ego-edge types:** OBSTRUCTS, BOUNDS_LATERALLY, LIMITS_SPEED, GRANTS_RIGHT_OF_WAY, REQUIRES_YIELD, REQUIRES_GAP_ACCEPTANCE, REQUIRES_FOLLOWING, REQUIRES_EVASION, COMMANDS, AFFECTS, CONFOUNDED_BY.

---

## 2. Phase A — Independent Scene Assessment

**Do this BEFORE reading the Stage-1 graph.**

### A1. Agent Inventory
Look at all 16 camera images and state.json. List EVERY agent visible in at least one camera: ID, category (generic types — sedan, SUV, truck, bus, never brand names), position, speed, which lane on BEV, on ego's road or separated. Run each through TREE 1 mentally — flag any that should be excluded.

### A2. Road Type
From cameras and BEV: narrow residential? multi-lane arterial? signalized intersection?

### A3. Ego Behavior
From state.json history and future: what is ego doing and why? Which agents actively constrain ego? Which are dormant? Which are distractors? For queued vehicles ahead, which is immediately adjacent to ego (only that one gets a direct ego_edge)?

### A4. Infrastructure
All signals, crosswalks, signs visible in cameras. Ahead or behind ego? Signal color from cameras (authoritative). BEV-only features not in cameras → should not be nodes. Crosswalks with nearby pedestrians → note for Layer 2 AFFECTS.

### A5. Same-Lane Chains
Group agents by lane on BEV. Physical separation in cameras overrides BEV. For each lane connected to ego, sort by x, identify immediately adjacent pair only (one hop). The closest vehicle ahead in ego's lane gets the direct ego_edge; further vehicles connect through the chain.

### A6. Unobserved Common Causes
Pairs with correlated behavior without visible shared cause → CONFOUNDED_BY candidates. If shared cause has a node → AFFECTS from that node instead.

---

## 3. Phase B — Gap Analysis

NOW read the Stage-1 graph.

### B1. Exclusion Check
Run every Stage-1 node through TREE 1 and TREE 2. Remove any that fail:
- Unoccupied bicycle (speed < 0.1, no rider)
- Physically separated agent (cameras show barriers/fences)
- Vehicle on separate road (BEV shows walkway/median between)
- Not visible in any camera at any timestep
- BEV-only infrastructure (not in any camera)
- Road-edge obstacle not intruding into lane
- Camera-only agent (no state.json, no BEV annotation)

### B2. Missing Agents
For every agent in A1 NOT in the graph that passes TREE 1: add with appropriate ego_edge per TREES 3→5/6/7→8.

### B3. Missing Edges
Compare A5 chains vs graph inter_node_edges. Run through TREE 9:
- Same-lane adjacent pairs → AFFECTS (Layer 3)
- TRAFFIC_CONTROL → approaching agents (Layer 2)
- Crosswalk → nearby pedestrian (Layer 2)
- Obstacle → approaching agent
- ENVIRONMENT → every OTHER_AGENT (mandatory)
- CONFOUNDED_BY for A6 pairs

### B4. Missing Ego Edges
Run each node through the applicable edge-type tree (4a/4b/5/6/7). If Stage 1 has no ego_edge but the tree produces one → add it. Apply queue mediation: only the immediately adjacent vehicle ahead qualifies. If ego's nav requires entering an adjacent lane occupied by a vehicle → should be INHIBITS_PROCEED.

### B5. Active Edge Audit
For every ego_edge with active=true:

**Queue mediation check:** If target is ahead in ego's lane, verify no closer same-lane vehicle sits between ego and target on BEV. If one does → REMOVE.

**Active-sustaining verification:** Use the entire observation window (T=-1.5 to T=0). Check TREE 8 Q1-Q2:
- Entity on ego's road? If separate road → REMOVE.
- Ego responded (speed, lateral, heading) across the window? "Finished responding by T=0" still counts.
- Entity still a live constraint at T=0?
- Multiple entities can independently qualify. Evaluate each independently.

**Active-triggering verification:** TREE 8 Q3. Entity influenced ego during window, ego mid-maneuver, removal at T=0 wouldn't change trajectory. Infrastructure behind ego (x < 0) NEVER qualifies. If entity still constrains at T=0 → reclassify as sustaining.

**Partial-block co-occurrence:** If entity has CL_LAT from intrusion, verify CL_LONG also exists (and vice versa). Add if missing.

If no behavioral evidence → downgrade to dormant (apply TREE 8 Q4-Q6). If dormant tests fail → REMOVE and move to distractors.

### B6. Dormant Edge Audit
For every ego_edge with active=false, verify TREE 8 Q4-Q6:
- **Q4 (single-step reachability, 0.5s):** Auto-pass exceptions: same-lane follower, CL_LAT qualified vehicle, regulatory sign, signal state change. Auto-fail: queue mediation (closer vehicle between).
- **Q5 (no barrier):** Clear path at T=0.
- **Q6 (spatial relevance):** In or near ego's lane/planned path on BEV (trace boundaries, NOT y-value).

If any test fails → REMOVE and move to distractors.

### B7. Edge Type Validation
Run every ego_edge through the applicable tree (4a/4b/5/6/7). If the tree produces a different effect_type than Stage 1 assigned → RECLASSIFY. Key checks:
- Full blockage should be INHIBITS_PROCEED, not CL_LONG or CL_LAT.
- Adjacent vehicle + nav-required lane change should be INHIBITS_PROCEED, not CL_LAT.
- CL_LAT path (b) requires behavioral evidence (ego offset). No offset + no nav requirement = no CL_LAT.
- Lead vehicle at red light: vehicle = CL_LONG, light = INHIBITS_PROCEED.

### B8. Distractors Audit
For every entity in distractors array: verify rejection_reason cites the correct tree question and specific evidence. If entity actually passes all dormant tests → move to nodes. For excluded entities not in distractors that are visible in cameras with state.json data → add with rejection reason. If array missing → create it.

### B9. CONFOUNDED_BY Audit
For every CONFOUNDED_BY edge: inferred_cause populated? Common cause genuinely unobserved (no node)? If observed → replace with two AFFECTS. Clear causal direction → reclassify as AFFECTS. Layer = N/A? One direction only? Ego not involved? Both entities connect to ego?

---

## 4. Phase C — Inter-Node Edge Validation

The decision trees (TREE 9, TREE 10) handle most validation. These additional spatial rules require coordinate lookups that trees cannot encode. Apply ALL rules to every inter_node_edge — if ANY fails, REMOVE.

### RULE 1 — Causation Direction (AFFECTS only, NOT ego_edges)

If "A causes B to stop/slow": A must be AHEAD of B in B's travel direction.
- Target yaw ≈ 0: x_source > x_target required. If not → REMOVE.
- Target yaw ≈ π: x_source < x_target required. If not → REMOVE.

Does not apply to CONFOUNDED_BY (no causal direction) or ego_edges.

### RULE 2 — Approach-Side (Layer 2: infrastructure → agent)

Agent must be APPROACHING infrastructure, not past it.
- Anchor x_infra INDEPENDENTLY (BEV position, different vehicle, pedestrian in crosswalk). NEVER use the agent being tested.
- If x_agent > x_infra AND agent heading away → REMOVE.
- Required format: "x_infra≈[val] (source: [independent]), x_agent=[val], PASS/FAIL."

### RULE 3 — Mandatory Coordinates

Every inter_node_edge justification must cite x-positions of both source and target. "Plausibly contributes" without numbers → look up positions, check Rule 1, REMOVE if violated.

### RULE 4 — Sightline Blockage

If "A blocks ego's view of B": A must be between ego (x=0) and B. Both must be same side (both y > 0 or both y < 0). Verify with coordinates.

### RULE 5 — Ego-Straddling

Both source and target in ego's lane, one ahead (x > 0) one behind (x < 0):
- **Infrastructure → agent:** Independence test: would agent obey without ego? YES → KEEP. NO → REMOVE.
- **Agent → agent:** Always REMOVE. Chain flows through ego.

### RULE 6 — Causal Path to Ego

Both source and target must connect to ego (via ego_edges or chain). If neither → REMOVE. If no agent in a lane connects to ego → skip that lane entirely.

### Schema Check

Run every edge through TREE 10. Fix: wrong effect_type for node type, missing mechanism/causal_status, wrong affected_state, duplicates, ego in inter_node_edge, wrong CONFOUNDED_BY layer/direction.

---

## 5. Decision Trees

These trees are identical to Stage 1 and are the authoritative procedural rules. When a Stage-1 edge disagrees with the tree output, the tree is authoritative.

### TREE 1: ENTITY INCLUSION

```
Q1: Visible in at least one camera at any timestep (T=-1.5 to T=0)?
├─ NO → EXCLUDE (not observable)
└─ YES → Q2

Q2: Is it an agent (vehicle/pedestrian/cyclist)?
├─ NO → Is it TRAFFIC_CONTROL or ROAD_OBSTACLE? → TREE 2
└─ YES → Q3

Q3: Has state.json entry or BEV annotation?
├─ NO → EXCLUDE (camera-only agent, note in scene_context)
└─ YES → Q4

Q4: Bicycle with speed < 0.1 and no rider in cameras?
├─ YES → EXCLUDE (unoccupied)
└─ NO → Q5

Q5: Physically separated from ego's road in cameras
    (barriers/fences/railings/elevated sidewalk)?
├─ YES → EXCLUDE (d-separated by physical barrier)
└─ NO → Q6

Q6: On a separate road on BEV (walkway/grass/median/curbed
    island between entity's road segment and ego's)?
├─ YES → EXCLUDE (separate road)
└─ NO → ENTITY INCLUDED → TREE 3
```

### TREE 2: INFRASTRUCTURE / OBSTACLE GATE

```
Q1: Traffic signal, sign, crosswalk marking, or road marking?
├─ YES → Q2a
└─ NO → Q2b (obstacle candidate)

Q2a: Visible in any camera at any timestep (not BEV-only)?
├─ NO → EXCLUDE (BEV-only map feature)
└─ YES → TRAFFIC_CONTROL node → TREE 4a

Q2b: Object ON drivable road surface (not sidewalk/grass/
     behind curbs/along road edge without intruding)?
├─ NO → EXCLUDE (road-edge feature, describe in scene_context)
└─ YES → ROAD_OBSTACLE node → TREE 4b
```

### TREE 3: LANE RELATIONSHIP (per included agent)

```
Q1: Is there a lane boundary (divider line, center line, or
    road edge) between ego and entity on BEV?
├─ NO → SAME LANE as ego → Q1a
│
│  Q1a: Is entity moving TOWARD ego (opposing direction)?
│       Check: entity's BEV heading arrow opposes lane
│       direction arrows (for moving entities), OR entity's
│       state.json yaw ≈ ±π AND speed > 0.1.
│  ├─ YES → OPPOSING IN SHARED LANE (undivided road)
│  │        → TREE 6
│  └─ NO → TREE 5 (same-lane, same-direction)
│
└─ YES → Q2: Lane direction arrows on both sides of the
              boundary — do they point the same direction?
    ├─ NO → OPPOSING LANE → TREE 6
    └─ YES → ADJACENT SAME-DIRECTION LANE → TREE 7
```

### TREE 4a: TRAFFIC_CONTROL → EGO EDGE

```
Q0: Is the infrastructure behind ego (x < 0)?
├─ YES → NO EGO_EDGE. Keep node only for inter_node_edges.
│        Never active_triggering for infrastructure behind ego.
└─ NO → Q1

Q1: Signal or regulatory sign/crosswalk?
├─ Signal → Q2
└─ Sign/crosswalk → Q3

Q2: What color in cameras?
├─ Red    → INHIBITS_PROCEED → TREE 8
├─ Yellow → INHIBITS_PROCEED → TREE 8
│          (active if sole cause of braking, dormant if
│           another element explains ego's stop.
│           NEVER GRANTS_PROCEED.)
├─ Green  → GRANTS_PROCEED → TREE 8
└─ Flashing yellow/red → INHIBITS_PROCEED → TREE 8

Q3: Crosswalk visible in cameras on ego's traversal path?
│   (Ego's future trajectory will physically cross the
│    crosswalk area — check BEV. NOT perpendicular streets
│    or lanes ego won't enter.)
├─ YES → INHIBITS_PROCEED (dormant if unoccupied) → TREE 8
└─ NO → Q4

Q4: Regulatory sign at ego's intersection constraining a
    maneuver ego could physically take?
├─ YES → INHIBITS_PROCEED, dormant → DONE
└─ NO → EXCLUDE
```

### TREE 4b: ROAD_OBSTACLE → EGO EDGE

```
Q1: Does the obstacle FULLY block ego's lane
    (no room to pass, even with lateral offset)?
├─ YES → INHIBITS_PROCEED → TREE 8
└─ NO → Q2

Q2: Is the obstacle partially in ego's lane
    (room to pass with offset)?
├─ YES → CONSTRAINS_LATERAL + CONSTRAINS_LONGITUDINAL
│        (both mandatory — ego offsets AND slows)
│        → TREE 8
└─ NO → Q3

Q3: Is the obstacle at the lane edge, narrowing the
    available lane width?
├─ YES → CONSTRAINS_LATERAL → TREE 8
└─ NO → EXCLUDE
```

### TREE 5: SAME-LANE, SAME-DIRECTION AGENT

```
Q1: Ahead of ego (x > 0)?
├─ YES → Q2
└─ NO → Q8 (behind ego)

--- AHEAD PATH ---

Q2: Does entity FULLY block ego's lane at T=0
    (no room to pass, even with offset)?
├─ YES → INHIBITS_PROCEED → TREE 8
└─ NO → Q3

Q3: Does entity PARTIALLY occupy ego's lane such that
    ego must offset laterally to pass?
├─ YES → CONSTRAINS_LATERAL + CONSTRAINS_LONGITUDINAL
│        (both mandatory for partial-block)
│        Check Q3a: closer same-lane vehicle between ego
│        and this entity? If YES → NO EGO_EDGE (mediated).
│        If NO → TREE 8
└─ NO → Q3b

Q3b: Is this a two-lane undivided/opposing road (one lane
     each direction, no median — only center line separates,
     check BEV lane direction arrows) AND is the entity at
     the SAFE-DEFAULT side of ego's lane (the side AWAY from
     the center line — i.e., the road-edge side)?

     How to check: On BEV, identify the center line (boundary
     where lane direction arrows point opposite directions).
     Ego's safe-default side is AWAY from this line. If entity
     is between ego's travel path and the road edge on the
     safe-default side, it pushes ego toward center line /
     oncoming traffic.
├─ YES → CONSTRAINS_LATERAL
│        (entity at road edge pushes ego toward center line
│         and oncoming traffic, restricting lateral positioning)
│        Check: did ego slow during observation window AND
│        this entity explains it?
│        ├─ YES → also add CONSTRAINS_LONGITUDINAL
│        └─ NO → CONSTRAINS_LATERAL only
│        → TREE 8
└─ NO → Q3c

Q3c: Is this a two-lane undivided/opposing road AND is the
     entity at the CENTER-LINE side of ego's lane (between
     ego's travel path and the center line)?
├─ YES → CONSTRAINS_LATERAL
│        (entity pushes ego toward the road edge, narrowing
│         ego's lane usage from the center side)
│        Same CL_LONG co-occurrence check as Q3b.
│        → TREE 8
└─ NO → Q4

Q4: Entity fully in ego's lane, not blocking — just ahead.
    Is there a closer same-lane vehicle between ego
    and this entity on BEV?
├─ YES → NO EGO_EDGE (queue mediation).
│        Connect via inter_node_edge (Layer 3 AFFECTS).
└─ NO → Q5

Q5: Would ego travel faster on this path if entity
    were absent? (Entity stationary, slower, decelerating,
    OR ego decelerated to match during observation window.)
├─ YES → CONSTRAINS_LONGITUDINAL → TREE 8
└─ NO → Q6

Q6: Is entity faster or pulling away?
├─ YES → No CONSTRAINS_LONGITUDINAL.
│        Check: does ego's nav/future path require entering
│        a region this entity currently occupies?
│  ├─ YES → INHIBITS_PROCEED → TREE 8
│  └─ NO → DISTRACTOR
└─ NO → DISTRACTOR

--- BEHIND PATH ---

Q8: Behind ego (x < 0), same lane, same direction.
    Is there a closer same-lane vehicle between ego
    and this entity (behind ego) on BEV?
├─ YES → NO EGO_EDGE (mediated — follower chain)
└─ NO → Q8a

Q8a: Was this entity AHEAD of ego at any earlier timestep
     during the observation window (T=-1.5 to T=0)?
     Check ALL of: entity visible in front/side cameras at
     earlier timesteps but now in CAM_BACK; entity's
     state.json history x-positions were ahead of ego's
     history positions at earlier timesteps; entity moves
     from front to rear cameras across the temporal frames.
├─ YES → Ego passed this entity during the window.
│        Does ego's current lateral position or speed
│        reflect having passed it? (Ego offset from lane
│        center toward either side, or ego slowed during
│        the approach phase visible in earlier timesteps.)
│  ├─ YES → Assign based on constraint:
│  │        • Ego offset away from entity →
│  │          CONSTRAINS_LATERAL, active_triggering
│  │          (ego's position is result of passing;
│  │           do(remove at T=0) won't change trajectory)
│  │        • Ego slowed during approach →
│  │          CONSTRAINS_LONGITUDINAL, active_triggering
│  │          (if ego has resumed speed) or active_sustaining
│  │          (if still constrained)
│  │        Both edges can apply simultaneously.
│  │        → DONE
│  └─ NO → Q9 (normal behind-ego follower path)
└─ NO → Q9

Q9: Is entity's heading arrow consistent with following
    ego's path?
├─ NO → DISTRACTOR (different trajectory)
└─ YES → Q10

Q10: Are ego AND entity both inside an intersection polygon
     (purple region on BEV) rather than a proper lane polygon?
├─ YES → DISTRACTOR (intersection area ≠ lane)
└─ NO → CONSTRAINS_LONGITUDINAL, dormant
         (follower constrains ego's braking freedom) → DONE
```

### TREE 6: OPPOSING-DIRECTION AGENT

```
Q1: Does entity's BEV bounding box intrude into
    ego's lane polygon?
├─ YES → Q1a: Is entity moving toward ego (speed > 0.1)?
│  ├─ YES → INHIBITS_PROCEED (head-on conflict) → TREE 8
│  └─ NO → Q1b: Does intrusion block ego's lane partially
│                or fully?
│     ├─ Fully blocks → INHIBITS_PROCEED → TREE 8
│     ├─ Partially blocks → CONSTRAINS_LATERAL
│     │  + CONSTRAINS_LONGITUDINAL → TREE 8
│     └─ No significant intrusion → DISTRACTOR
│
└─ NO → Q1c

Q1c: Is ALL of the following true?
     • Two-lane undivided/opposing road (one lane each
       direction, no median — check BEV lane arrows)
     • Entity is ahead of ego (x > 0)
     • Entity is stationary or slow (speed < 2 m/s)
     • Entity is large (truck, bus, trailer, wide vehicle
       — not bicycle or pedestrian)
├─ YES → CONSTRAINS_LATERAL
│        (ego will pass alongside this opposing-lane
│         entity with only a center line between them;
│         ego's comfort zone on the center-line side is
│         restricted, pushing ego toward safe-default side)
│        Check: is ego slowing or did ego slow in
│        approach? → also add CONSTRAINS_LONGITUDINAL
│        → TREE 8
│
└─ NO → Q2: Is ego turning across this entity's path?
    ├─ NO → DISTRACTOR
    └─ YES → Q3: Entity moving (speed > 0.1) toward ego's
                  turn path?
        ├─ NO → DISTRACTOR (stationary opposing, ego clears it)
        └─ YES → INHIBITS_PROCEED (gap acceptance) → TREE 8
```

### TREE 7: ADJACENT SAME-DIRECTION LANE

```
--- PATH (a): BEV INTRUSION ---
Q1: Does entity's BEV bounding box intrude into ego's
    lane polygon?
├─ YES → Q1a: Does intrusion fully block ego's lane?
│  ├─ YES → INHIBITS_PROCEED → TREE 8
│  └─ NO → CONSTRAINS_LATERAL
│          + CONSTRAINS_LONGITUDINAL (if entity at x ≥ 0)
│          → TREE 8
│
└─ NO → PATH (b): ADJACENT LANE OCCUPANCY

--- PATH (b): NO INTRUSION ---
Q2: Does ANY disqualifier apply?
    • Diverging lane (lane arrows curve away)?
    • Entity ahead AND moving faster than ego?
    • Opposing-direction lane (check LANE DIRECTION
      arrows, NOT vehicle heading/yaw)?
    • On separate road (median/barrier/walkway)?
    • Small/distant in camera across all timesteps?
    • Two or more lanes away?
├─ ANY YES → No CONSTRAINS_LATERAL → Q7
└─ ALL NO → Q3

Q3: Is entity prominently visible in CAM_FRONT, FRONT_LEFT,
    or FRONT_RIGHT at any timestep (T=-1.5 to T=0)?
    Heuristic: within ~20m on BEV, substantial in camera.
├─ NO → No CONSTRAINS_LATERAL → Q7
└─ YES → Q4

Q4: Is ego holding a lateral offset AWAY from this entity?
    (y-history T=-1.5 to T=0 shows shift away, AND this
     entity's position explains the offset.)
├─ YES → CONSTRAINS_LATERAL, active_sustaining.
│        Did ego also decelerate AND entity explains it?
│        ├─ YES → Also add CONSTRAINS_LONGITUDINAL
│        └─ NO → CONSTRAINS_LATERAL only
│        → DONE
│
└─ NO → Q5: Does ego's nav/planned path require entering
             the entity's lane (lane change, merge)?
    ├─ YES → INHIBITS_PROCEED → TREE 8
    │
    └─ NO → No CONSTRAINS_LATERAL → Q7

Q7: Does entity qualify for any other ego_edge type?
    • Merging into ego's lane (history convergence)?
      → INHIBITS_PROCEED
    • Other convergence? → evaluate
    If none → DISTRACTOR (may still have inter_node_edges)
```

### TREE 8: ACTIVE / DORMANT / DISTRACTOR

```
Q0: Is this a TRAFFIC_CONTROL node behind ego (x < 0)?
├─ YES → NO EGO_EDGE. Keep node only for inter_node_edges.
└─ NO → Q1

Q1: Did ego show behavioral response across observation
    window (T=-1.5 to T=0)?

    Use the ENTIRE window — all 16 images, state.json
    history, BEV. "Finished responding" ≠ "not responding."

    Evidence (match to edge type):
    • Speed: decelerated or holds lower speed
    • Lateral: shifted or holds offset
    • Heading: changed to avoid
    • Proceed: stopped/waiting at gate

    Multiple entities can independently qualify.

├─ YES → Q2
└─ NO → Q4 (dormant candidate)

Q2: Entity still a live constraint at T=0?
    (do(remove) changes ego's next action?)
├─ YES → ACTIVE_SUSTAINING → DONE
└─ NO → Q3

Q3: Entity initiated ego's committed maneuver during window,
    but removal at T=0 wouldn't change trajectory?
    (Infrastructure behind ego NEVER qualifies.)
├─ YES → ACTIVE_TRIGGERING → DONE
└─ NO → Q4

--- DORMANT TESTS ---

Q4: Single-step reachability (0.5s, current kinematics)?
    AUTO-PASS: same-lane follower, CL_LAT qualified,
    regulatory sign, signal state change.
    AUTO-FAIL: queue mediation (closer vehicle between).
├─ FAIL → DISTRACTOR
└─ PASS → Q5

Q5: Clear path (no curbs/barriers/fences between)?
├─ NO → DISTRACTOR
└─ YES → Q6

Q6: In or near ego's lane/planned path at T=0?
    (BEV boundaries, NOT state.json y-value.)
├─ FAIL → DISTRACTOR
└─ PASS → DORMANT → DONE
```

### TREE 9: INTER-NODE EDGES

```
--- LAYER 1: ENVIRONMENT ---
Q1: ENVIRONMENT node exists?
├─ YES → AFFECTS → every OTHER_AGENT (mandatory)
└─ continue

--- LAYER 2: TRAFFIC_CONTROL → OTHER_AGENT ---
Q2: Agent APPROACHING infrastructure (not past)?
    (Anchor x_infra independently.)
├─ NO → NO EDGE
└─ YES → Q2a: Ego-straddling?
    ├─ NO → AFFECTS (Layer 2)
    └─ YES → Independence test: agent obeys without ego?
        ├─ YES → AFFECTS (keep)
        └─ NO → NO EDGE

Q2b: Crosswalk + pedestrian on/approaching it?
├─ YES → AFFECTS crosswalk → pedestrian (Layer 2)
└─ skip

--- LAYER 3 ---
Q3: Same-lane agents?
├─ YES → Adjacent pair only, one hop.
│   Straddling ego's lane? → NO EDGE (agent→agent always blocked)
│   Otherwise → AFFECTS (Layer 3) + x-positions
└─ Q4

Q4: Agent proximity on BEV?
    Sightline blockage, presence creation, yielding.
    (Source ahead of target. Include x-positions.)

Q5: ROAD_OBSTACLE ↔ OTHER_AGENT?
    Obstacle ahead of approaching agent → AFFECTS
    Agent created obstacle → AFFECTS

Q6: Cross-lane convergence?
├─ YES → AFFECTS (Layer 3)
└─ Q7

--- CONFOUNDED_BY ---
Q7: Check for unobserved common causes between entity
    pairs. Scan for these patterns:

    PATTERN A — Co-located context agents:
    Two+ agents co-present in same area whose presence
    is explained by shared context NOT captured as a node:
    construction vehicles + workers = "active construction
    zone"; delivery truck + dolly = "active delivery";
    emergency vehicles at same location = "emergency scene."

    PATTERN B — Correlated stopping/yielding:
    Two+ agents both stopped or decelerating at positions
    consistent with shared cause not visible in cameras:
    both stopped at signal location = "inferred signal
    state"; both yielding = "shared yielding context."

    PATTERN C — Correlated behavior shift:
    Multiple agents with simultaneous behavior change
    (all braking, all lane-changing) without visible
    shared trigger = "unseen hazard ahead" or
    "unseen emergency vehicle."

    For each pattern match:

Q7a: Clear causal direction between the two?
├─ YES → AFFECTS (not CONFOUNDED_BY)
└─ NO → Q7b

Q7b: Common cause observed and has a node in graph?
├─ YES → Two AFFECTS from observed cause to each entity.
│        (Apply straddling rules.) No CONFOUNDED_BY.
└─ NO → Q7c

Q7c: Both entities connect to ego?
├─ NO → NO EDGE (disconnected)
└─ YES → CONFOUNDED_BY (layer N/A, lower→higher ID,
         inferred_cause = specific pattern above)
```

### TREE 10: SCHEMA VALIDATION

```
Q1: Ego_edge?
├─ YES → Check:
│   • Source type → valid effect_type?
│     TC→{INHIBITS,GRANTS}, OBSTACLE→{CL_LONG,CL_LAT,INHIBITS},
│     AGENT→{CL_LONG,CL_LAT,INHIBITS}, ENV→{DEGRADES}
│   • mechanism present?
│   • causal_status consistent with active flag?
│   • affected_state matches mapping?
│   • No duplicates per source-target-effect_type
│
└─ NO → Inter_node_edge: Check:
    • AFFECTS: ego not involved, layer ∈ {1,2,3},
      x-positions in justification, valid source→target
    • CONFOUNDED_BY: layer=N/A, inferred_cause populated,
      one direction only, ego not involved, both connect to ego
```

---

## 6. Output

```json
{
  "phase_a_summary": {
    "agents_visible_in_cameras": ["<agent> (<category>, <description>)", "..."],
    "road_type": "<description>",
    "ego_behavior": "<what ego is doing and why>",
    "infrastructure_ahead": ["<item> (<category>)", "..."],
    "bev_only_features_not_in_cameras": ["<item>", "..."],
    "same_lane_chains": { "<lane>": ["<agent> (x=val)", "..."] },
    "unobserved_common_causes": ["<A> + <B> — <cause>", "..."],
    "agents_that_should_be_excluded": ["<agent> — <reason>"],
    "agents_missing_from_graph": ["<agent> — <reason>"],
    "spurious_edges_in_graph": ["<edge> — <tree/rule violated>"]
  },
  "scene_id": "...",
  "scene_context": { "road_geometry": "...", "navigation_intent": "..." },
  "ego_state": { ... },
  "nodes": [ ... ],
  "ego_edges": [
    {
      "source": "<node id>", "target": "ego",
      "effect_type": "<CL_LONG | CL_LAT | INHIBITS_PROCEED | GRANTS_PROCEED | DEGRADES>",
      "mechanism": "<short phrase>",
      "affected_state": ["<speed|heading|acceleration>"],
      "active": "<true|false>",
      "causal_status": "<active_sustaining|active_triggering|dormant>",
      "justification": "<1-2 sentences>"
    }
  ],
  "inter_node_edges": [
    {
      "source": "<node id>", "target": "<node id — NEVER ego>",
      "relation": "<AFFECTS | CONFOUNDED_BY>",
      "layer": "<1|2|3|N/A>",
      "inferred_cause": "<only for CONFOUNDED_BY>",
      "justification": "<1-2 sentences with x-positions>"
    }
  ],
  "distractors": [
    {
      "id": "<agent ID>", "type": "<OTHER_AGENT | TRAFFIC_CONTROL | ROAD_OBSTACLE>",
      "category": "<free-text>",
      "position_relative_to_ego": "<qualitative>",
      "rejection_reason": "<which tree question failed — cite evidence>"
    }
  ],
  "changes": [
    {"action": "REMOVED|ADDED|RECLASSIFIED", "type": "node|ego_edge|inter_edge|distractor",
     "id/source/target": "...", "reason": "B1/B5/C-Rule2: <explanation with tree reference>"}
  ]
}
```

---

## 7. Critical Reminders

1. **Phase A is mandatory.** Form your own opinion before reading the graph.
2. **Trees are authoritative.** When Stage-1 disagrees with a tree, the tree wins.
3. **Physical separation beats BEV.** Cameras show barriers → excluded regardless of BEV.
4. **Inter_node_edge Rules 1-6 are coordinate checks.** Every edge must pass ALL. "Plausibly contributes" without x-positions = REMOVE.
5. **Partial blockage = both CL_LAT + CL_LONG.** Full blockage = INHIBITS_PROCEED.
6. **CL_LAT path (b) requires ego offset.** Adjacent-lane occupancy alone without behavioral evidence = no edge. Nav-required lane entry = INHIBITS_PROCEED.
7. **Queue mediation.** Only the immediately adjacent vehicle ahead gets a direct ego_edge.
8. **ENVIRONMENT → every OTHER_AGENT.** No exceptions.
9. **Active-sustaining uses entire observation window.** "Finished responding" ≠ "not responding."
10. **Infrastructure behind ego (x < 0) gets no ego_edge.** Not even active_triggering.
11. **Distractors array must be populated.** Every excluded entity with camera visibility + state.json data.
12. **CONFOUNDED_BY: layer N/A, inferred_cause required, one direction only.**