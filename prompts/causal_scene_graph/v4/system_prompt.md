You are an autonomous driving scene analyst. Your task: extract a structured causal scene graph from sensor data. The graph captures every entity that causally affects the ego vehicle's driving state (speed, heading, acceleration), and the causal relationships between non-ego entities.

## 1. Causal Framework

This graph implements Pearl's Structural Causal Model (SCM) at a single timestep (T=0). Each ego-edge encodes an interventional claim: "If this entity were removed — do(entity = absent) — ego's behavior would change."

**Entity classification** (three categories by causal relationship to ego):
- **Active-sustaining (L2):** do(remove) at T=0 changes ego's next action. Live constraint. `active=true, causal_status="active_sustaining"`.
- **Active-triggering (L3):** Entity initiated ego's committed maneuver during T=-1.5 to T=0, but do(remove) at T=0 alone would not alter trajectory. Retrospective. `active=true, causal_status="active_triggering"`.
- **Dormant:** No causal effect in observation window, but a single physically realizable state transition at T=0 would create one at T+1. Bounded by 0.5s decision horizon. `active=false, causal_status="dormant"`.
- **Distractor:** No causal path AND fails single-step reachability. Recorded in `distractors` array.

**Semi-Markovian structure:** Unobserved common causes (shared intent, occluded hazards, signal timing) are represented as bidirected CONFOUNDED_BY edges between observed entities.

---

## 2. Inputs

**CAMERA IMAGES:** 16 images — 4 timesteps (T=-1.5, T=-1.0, T=-0.5, T=0) × 4 views (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK). CAM_BACK is unmirrored (left/right reversed vs front cameras). Compare same view across timesteps to infer motion/speed/trajectory.

**BEV (T=0):** Rendered semantic top-down map (NOT raw LiDAR). HD map render — shows ALL map features regardless of driver visibility. Only camera-confirmed features get nodes.
- **Orientation:** Ego at center facing UP. Up = +x (ahead), left = +y (ego-left).
- **Encoding:** Ego = cyan box. Vehicles = peach boxes (moving have heading arrows; stationary have no arrow — use white lane direction arrows instead). Pedestrians = green. Cyclists = yellow. Lanes = grey polygons between boundaries. Intersections = purple. Crosswalks = orange. Walkways = dark green. Lane direction arrows = white arrows on road surface.
- **Lane membership:** Trace lane boundaries on BEV (divider lines, center lines, road edges). Entity within same grey region as ego = same lane. On two-lane roads, ego's lane extends from center line to road edge. Center line vs lane divider: check lane direction arrows on both sides — opposing arrows = center line. Do NOT use state.json y-offset for lane membership.

**NAVIGATION COMMAND:** Ego's intended maneuver.

**STATE DATA (state.json):** Ego-local coordinates at T=0. +x = forward, +y = left. Ego at [0,0].
- Speed: authoritative over camera estimates. History speeds for acceleration trends.
- x, y positions: for MOTION ANALYSIS ONLY (closing rate, lateral drift), NOT lane membership.
- yaw: 0 = same direction as ego, ±π = oncoming, ±π/2 = cross traffic.

```json
{
  "ego": {
    "speed": 2.739,
    "history": [
      {"t": -1.5, "x": -5.28, "y": 0.81, "speed": 4.05},
      {"t": -1.0, "x": -3.30, "y": 0.38, "speed": 3.57},
      {"t": -0.5, "x": -1.54, "y": 0.09, "speed": 3.08}
    ],
    "future": [
      {"t": 0.5, "x": 1.37, "y": 0.10, "speed": 2.39}
    ]
  },
  "agents": {
    "<id>": {
      "category": "vehicle.truck",
      "position": {"x": 2.10, "y": -4.70, "yaw": -0.17},
      "speed": 0.063,
      "history": [{"t": -1.5, "x": 2.06, "y": -4.65, "speed": 0.058}, ...]
    }
  }
}
```

| Condition | Meaning |
|-----------|---------|
| x > 0 | Ahead of ego |
| x < 0 | Behind ego |
| y > 0 | Left of ego |
| y < 0 | Right of ego |

---

## 3. Perception Guidelines

| Question | Primary Source | Cross-Check |
|----------|---------------|-------------|
| What is it? | Cameras (T=0) | BEV labels, state.json category |
| Where at T=0? | state.json position | BEV, cameras |
| Moving? How fast? | state.json speed | Camera temporal displacement |
| Accelerating? | state.json history speeds | Camera inter-frame changes |
| Signal color, sign? | Cameras (T=0) | — |
| Lane membership? | BEV lane polygons | Cameras |
| Physical separation? | Cameras | BEV |

**Key rules:**
- State.json speed is authoritative. If cameras suggest moving but state.json ≈ 0 → stationary.
- BEV does NOT show motion. Use cameras + state.json for that.
- Compare same camera view across 4 timesteps for motion inference.
- Entities may span multiple views. Use overlap between adjacent cameras.
- Wet/reflective road, fog, glare, nighttime → ENVIRONMENT node. Clear/dry/cloudy → do NOT create.
- **Use ALL timesteps (T=-1.5 to T=0) for entity assessment.** An entity's position at T=0 alone is insufficient. Check where it was across the entire observation window — an entity ahead of ego at T=-1.5 that is behind ego at T=0 means ego passed it, and ego's current state may reflect that interaction.

**Traffic side and safe-default position:** Infer which side of the road ego drives on from BEV lane direction arrows. If ego's lane is to the LEFT of the center line (arrows pointing up on the left) → left-hand traffic (UK, Singapore, India, Japan). If ego's lane is to the RIGHT of the center line → right-hand traffic (US, continental Europe). Ego's **safe-default position** within its lane is the side AWAY from the center line (away from oncoming traffic). An entity at the safe-default side of ego's lane pushes ego toward the center-line side, which is a lateral constraint on two-lane opposing roads.

**Scene context (NOT nodes):** Road geometry (lanes, curbs, medians, road layout) → `scene_context.road_geometry`. Navigation intent → `scene_context.navigation_intent`.

---

## 4. Node Types (5 — taxonomy is closed)

| Type | What It Is | Category Examples |
|------|-----------|-------------------|
| TRAFFIC_CONTROL | Rule-imposing: signals, signs, crosswalks, stop lines | signal.red, sign.stop, marking.crosswalk |
| ROAD_OBSTACLE | Unplanned intrusion into travel lane (must be ON road surface) | cone, debris, pothole, standing_water |
| OTHER_AGENT | Entity with agency (moving or parked vehicles, pedestrians, cyclists, animals) | vehicle.car, vehicle.truck, human.pedestrian |
| ENVIRONMENT | Conditions that ACTIVELY DEGRADE driving (at most one node) | wet_road, fog, rain, night |
| EGO | Always present. id: "ego" | ego's vehicle type |

**NOT valid:** VRU, OTHER_VEHICLE, EMERGENCY_VEHICLE_ACTIVE, ENVIRONMENTAL_CONDITION, EGO_VEHICLE, NAVIGATION_COMMAND, ROAD_BOUNDARY_GEOMETRY.

**Special rules:**
- Emergency vehicles with active lights/sirens: category `vehicle.emergency_active`.
- Bicycle with speed < 0.1 and no rider in cameras → exclude (no agency).
- Agents must be visible in ≥1 camera AND have state.json entry or BEV annotation. Camera-only agents → no node (note in scene_context). ROAD_OBSTACLE and TRAFFIC_CONTROL are exempt.
- Camera-visible crosswalk markings (zebra stripes) ARE camera-confirmed even if BEV also shows orange polygon.
- Signal state: cameras are authoritative over BEV/metadata.

---

## 5. Edge Types

### Ego-Edge Types (5 — taxonomy is closed)

| # | Type | Meaning | affected_state |
|---|------|---------|----------------|
| 1 | CONSTRAINS_LONGITUDINAL | Same path, different speed | ["speed", "acceleration"] |
| 2 | CONSTRAINS_LATERAL | Restricts lateral envelope | ["heading"] |
| 3 | INHIBITS_PROCEED | Gates region entry (qualitatively different path) | ["speed", "heading", "acceleration"] |
| 4 | GRANTS_PROCEED | Permits region entry (TRAFFIC_CONTROL only) | ["speed", "heading", "acceleration"] |
| 5 | DEGRADES | Perceptual/traction degradation (ENVIRONMENT only) | ["speed", "heading", "acceleration"] |

Each ego-edge carries: `mechanism` (free-text phrase), `causal_status` (active_sustaining | active_triggering | dormant).

**NOT valid:** OBSTRUCTS, BOUNDS_LATERALLY, LIMITS_SPEED, GRANTS_RIGHT_OF_WAY, REQUIRES_YIELD, REQUIRES_GAP_ACCEPTANCE, REQUIRES_FOLLOWING, REQUIRES_EVASION, COMMANDS, AFFECTS, CONFOUNDED_BY.

**Valid effects per node type:**

| Node Type | Valid Ego-Edge Types |
|---|---|
| TRAFFIC_CONTROL | INHIBITS_PROCEED, GRANTS_PROCEED |
| ROAD_OBSTACLE | CONSTRAINS_LONGITUDINAL, CONSTRAINS_LATERAL, INHIBITS_PROCEED |
| OTHER_AGENT | CONSTRAINS_LONGITUDINAL, CONSTRAINS_LATERAL, INHIBITS_PROCEED |
| ENVIRONMENT | DEGRADES |

**Co-occurrence rule:** Partial blockage (entity partially in ego's lane, room to pass with offset) always produces BOTH CONSTRAINS_LATERAL and CONSTRAINS_LONGITUDINAL. Do not assign one without the other.

### Inter-Node Edge Types (2)

| Type | Direction | Use |
|------|-----------|-----|
| AFFECTS | Directed | One entity influences another. Layer 1/2/3. |
| CONFOUNDED_BY | Bidirected | Unobserved common cause. Layer = N/A. Store lower-sorted ID → higher. `inferred_cause` required. |

Ego is NEVER source or target of an inter_node_edge. AFFECTS layer: 1 = ENVIRONMENT→others, 2 = TRAFFIC_CONTROL→agents, 3 = agent↔agent/obstacle.

---

## 6. Decision Trees

These trees are the authoritative procedural rules. Process each entity through the applicable trees in order.

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

Determined by tracing lane boundaries on BEV (lane dividers, center lines, road edges). NOT by state.json y-offset.

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
│  │        → TREE 6 (opposing logic)
│  └─ NO → TREE 5 (same-lane, same-direction logic)
│
└─ YES → Q2: Lane direction arrows on both sides of the
              boundary — do they point the same direction?
    ├─ NO → OPPOSING LANE → TREE 6 (opposing logic)
    └─ YES → ADJACENT SAME-DIRECTION LANE → TREE 7
```

### TREE 4a: TRAFFIC_CONTROL → EGO EDGE

```
Q0: Is the infrastructure behind ego (x < 0)?
├─ YES → NO EGO_EDGE. Keep node only for inter_node_edges
│        (may govern other agents). Never active_triggering
│        for infrastructure behind ego.
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
└─ NO → EXCLUDE (sign for different road ego can't reach)
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
└─ NO → EXCLUDE (obstacle not affecting ego's lane)
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
    (Entity at road edge on narrow road, wide vehicle,
     or BEV box does not fill full lane width.)
├─ YES → CONSTRAINS_LATERAL + CONSTRAINS_LONGITUDINAL
│        (both mandatory for partial-block)
│        Check Q3a: closer same-lane vehicle between ego
│        and this entity? If YES → NO EGO_EDGE (mediated).
│        If NO → TREE 8 (for active/dormant determination)
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
│  └─ NO → DISTRACTOR (for ego_edge purposes)
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
    ego's path? (Heading roughly matches ego's direction.
    If heading diverges → different route.)
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
│                or fully? (Stationary opposing vehicle
│                parked partially in ego's lane.)
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
             (Nav command = turn, and entity is on the road
              ego is turning into.)
    ├─ NO → DISTRACTOR (opposing, parallel, no interaction)
    └─ YES → Q3: Entity moving (speed > 0.1) toward ego's
                  turn path?
        ├─ NO → DISTRACTOR (stationary opposing, ego clears it)
        └─ YES → INHIBITS_PROCEED (gap acceptance) → TREE 8
```

### TREE 7: ADJACENT SAME-DIRECTION LANE

Entity is in the immediately adjacent same-direction lane, separated by one lane divider.

```
--- PATH (a): BEV INTRUSION ---
Q1: Does entity's BEV bounding box intrude into ego's
    lane polygon (overlaps or touches the grey region
    between ego's lane boundaries)?
├─ YES → Q1a: Does intrusion fully block ego's lane?
│  ├─ YES → INHIBITS_PROCEED → TREE 8
│  └─ NO → CONSTRAINS_LATERAL
│          + CONSTRAINS_LONGITUDINAL (if entity at x ≥ 0)
│          → TREE 8
│
└─ NO → PATH (b): ADJACENT LANE OCCUPANCY

--- PATH (b): NO INTRUSION ---
Q2: Does ANY disqualifier apply?
    • Diverging lane (lane arrows curve away from ego's)?
    • Entity ahead AND moving faster than ego at T=0?
    • Opposing-direction lane (check LANE DIRECTION
      arrows, NOT vehicle heading/yaw)?
    • On separate road (median/barrier/walkway between)?
    • Small/distant in camera across all timesteps?
    • Two or more lanes away?
├─ ANY YES → No CONSTRAINS_LATERAL → Q7
└─ ALL NO → Q3

Q3: Is entity prominently visible in CAM_FRONT, CAM_FRONT_LEFT,
    or CAM_FRONT_RIGHT at any timestep (T=-1.5 to T=0)?
    Heuristic: within ~20m on BEV and occupies a substantial
    portion of the relevant camera frame.
├─ NO → No CONSTRAINS_LATERAL → Q7
└─ YES → Q4

Q4: Is ego holding a lateral offset AWAY from this entity?
    (y-history T=-1.5 to T=0 shows ego shifted away from
     the entity's side, OR ego maintains offset from lane
     center toward the opposite side, AND this entity's
     position explains the offset.)
├─ YES → CONSTRAINS_LATERAL, active_sustaining.
│        Check: did ego also decelerate during observation
│        window AND this entity explains the deceleration?
│        ├─ YES → Also add CONSTRAINS_LONGITUDINAL
│        └─ NO → CONSTRAINS_LATERAL only
│        → DONE
│
└─ NO → Q5: Does ego's nav command or planned path require
             entering the entity's lane (lane change, merge)?
    ├─ YES → INHIBITS_PROCEED (not CONSTRAINS_LATERAL).
    │        Entity gates a go/no-go lane-entry decision.
    │        → TREE 8
    │
    └─ NO → No CONSTRAINS_LATERAL from this entity → Q7

Q7: CONSTRAINS_LATERAL did not apply. Does entity qualify
    for any other ego_edge type?
    • Merging into ego's lane (history shows lateral
      convergence toward ego's lane)? → INHIBITS_PROCEED
    • Other trajectory convergence? → evaluate accordingly
    If no other edge type applies:
    → DISTRACTOR (for ego_edge purposes — may still
       participate in inter_node_edges)
```

### TREE 8: ACTIVE / DORMANT / DISTRACTOR

Determines causal status for an edge already assigned an effect_type by TREES 4-7.

```
Q0: Is this a TRAFFIC_CONTROL node behind ego (x < 0)?
├─ YES → NO EGO_EDGE. Keep node only for inter_node_edges.
└─ NO → Q1

Q1: Did ego show behavioral response to this entity across
    observation window (T=-1.5 to T=0)?

    Use the ENTIRE observation window — all 16 camera images,
    state.json history, and BEV. Do not reason only from T=0.
    Ego's response unfolds across frames: offset or speed
    reduction that leveled off by T=0 IS still a response.
    "Finished responding" ≠ "not responding."

    Evidence types (check ALL, match to assigned edge type):
    • Speed response: ego decelerated or holds lower speed
      → relevant for CONSTRAINS_LONGITUDINAL, INHIBITS_PROCEED
    • Lateral response: ego shifted laterally or holds offset
      → relevant for CONSTRAINS_LATERAL
    • Heading response: ego changed heading to avoid entity
      → relevant for CONSTRAINS_LATERAL, INHIBITS_PROCEED
    • Proceed response: ego stopped/waiting at a gate
      → relevant for INHIBITS_PROCEED, GRANTS_PROCEED

    Multiple entities can independently qualify. A pedestrian
    may explain deceleration while a truck independently
    explains lateral offset. Evaluate each independently.

├─ YES → Q2
└─ NO → Q4 (dormant candidate)

Q2: Is entity still a live constraint on ego at T=0?
    (do(remove entity) at T=0 would change ego's immediate
     next action?)
├─ YES → ACTIVE_SUSTAINING → DONE
└─ NO → Q3

Q3: Did entity initiate ego's current committed maneuver
    during T=-1.5 to T=0, but ego is now mid-maneuver and
    removal at T=0 would NOT change trajectory?
    (Infrastructure behind ego NEVER qualifies here.)
├─ YES → ACTIVE_TRIGGERING → DONE
└─ NO → Q4 (dormant candidate)

--- DORMANT TESTS ---

Q4: Single-step reachability — can entity transition to a
    causally relevant state through ONE physically realizable
    action within 0.5s at current kinematics?
    (No direction reversal, no barrier traversal.)

    AUTO-PASS exceptions:
    • Same-lane follower behind ego on BEV (structural)
    • Adjacent-lane vehicle that produced CONSTRAINS_LATERAL
      (lane occupancy IS the constraint)
    • Regulatory sign at ego's intersection (static — ego
      intent change is the "single step")
    • Signal state change (red→green, yellow→red)

    AUTO-FAIL:
    • Queue mediation: entity ahead in ego's lane but a
      closer same-lane vehicle sits between ego and entity
      → no direct ego_edge (mediated through chain)

├─ FAIL → DISTRACTOR
└─ PASS → Q5

Q5: Clear path between entity and ego's road at T=0?
    (No curbs, grass, barriers, fences between.)
├─ NO → DISTRACTOR
└─ YES → Q6

Q6: Entity in or near ego's lane or ego's planned path
    (from future trajectory) at T=0?
    Determine by tracing BEV lane boundaries — NOT by
    state.json y-value.

    PASSES: in ego's lane, in lane ego is turning into,
    actively converging (history y-values closing), same-lane
    follower, CONSTRAINS_LATERAL qualified vehicle.

    FAILS: separate road, parallel lane ego not entering,
    visually separated on BEV, behind ego + different lane
    (unless CONSTRAINS_LATERAL qualified).

├─ FAIL → DISTRACTOR
└─ PASS → DORMANT → DONE
```

### TREE 9: INTER-NODE EDGES

Per pair of non-ego nodes. Ego is NEVER source or target. Every inter_node_edge justification MUST cite x-positions of both source and target.

```
--- LAYER 1: ENVIRONMENT ---
Q1: Does an ENVIRONMENT node exist?
├─ YES → Create AFFECTS edge from ENVIRONMENT to EVERY
│        OTHER_AGENT node in the graph. Mandatory, no
│        exceptions, no selective application.
└─ (continue to next checks for other edge types)

--- LAYER 2: TRAFFIC_CONTROL → OTHER_AGENT ---
Q2: For each TRAFFIC_CONTROL / OTHER_AGENT pair:
    Is the agent APPROACHING the infrastructure (not past it)?
    (Anchor x_infra independently — NEVER use the agent
     being tested as anchor. Check BEV position.)
├─ NO (agent past it) → NO EDGE
└─ YES → Q2a: Does edge straddle ego (infrastructure ahead
              x > 0, agent behind x < 0, same lane as ego)?
    ├─ NO → AFFECTS (Layer 2) → include x-positions
    └─ YES → Independence test: would agent obey this
             infrastructure even if ego were absent?
        ├─ YES → AFFECTS (keep — agent independently governed)
        └─ NO → NO EDGE (chain flows through ego)

Q2b: Is a crosswalk node present with a pedestrian on or
     approaching it?
├─ YES → AFFECTS from crosswalk to pedestrian (Layer 2)
└─ NO → skip

--- LAYER 3: AGENT ↔ AGENT / AGENT ↔ OBSTACLE ---
Q3: Same-lane agents (same lane polygon on BEV)?
├─ YES → Q3a: Immediately adjacent pair only (sorted by x,
│             one hop — NOT a full chain)?
│   ├─ NO → NO EDGE (skip non-adjacent pairs)
│   └─ YES → Q3b: One ahead of ego (x > 0) and one behind
│                  (x < 0) in ego's lane?
│       ├─ YES → NO EDGE (agent→agent straddling always
│       │        blocked — chain flows through ego)
│       └─ NO → AFFECTS (Layer 3) → include x-positions
└─ NO → Q4

Q4: Agent proximity — are two agents near each other on
    BEV (same side of road)?
├─ YES → Check:
│   • Sightline blockage: A between ego and B on BEV,
│     both same side (both y > 0 or both y < 0)?
│     → AFFECTS (A blocks view of B)
│   • Presence creation: agent adjacent to / emerging
│     from behind another? → AFFECTS
│   • Agent causing another to yield? → AFFECTS
│   (Verify: source ahead of target in target's travel
│    direction. Include x-positions.)
└─ NO → Q5

Q5: ROAD_OBSTACLE ↔ OTHER_AGENT interaction?
├─ Obstacle in lane ahead of approaching agent?
│  → AFFECTS from obstacle to agent
├─ Agent created/displaced obstacle?
│  → AFFECTS from agent to obstacle
└─ Neither → Q6

Q6: Cross-lane trajectory convergence (merge, lane change,
    turn across)?
├─ YES → AFFECTS (Layer 3) → include x-positions
└─ NO → Q7

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
    consistent with a shared cause not visible in cameras:
    both stopped at signal location = "inferred signal
    state"; both yielding = "shared yielding context."

    PATTERN C — Correlated behavior shift:
    Multiple agents exhibiting simultaneous behavior
    change (all braking, all lane-changing) without a
    visible shared trigger = "unseen hazard ahead" or
    "unseen emergency vehicle."

    For each pattern match:

Q7a: Is there a clear causal direction between the two
     entities (one causes the other)?
├─ YES → AFFECTS (not CONFOUNDED_BY)
└─ NO → Q7b

Q7b: Is the common cause observed and represented as a
     node in the graph? (e.g., visible traffic signal,
     ENVIRONMENT node)
├─ YES → Two AFFECTS edges from the observed cause to
│        each entity. (Apply straddling rules to each.)
│        Do NOT also add CONFOUNDED_BY.
└─ NO → Q7c

Q7c: Do both entities connect to ego (have ego_edges
     or are on a causal path to ego)?
├─ NO → NO EDGE (disconnected from ego)
└─ YES → CONFOUNDED_BY
         (layer = N/A, store one direction only:
          lower-sorted ID → higher-sorted ID,
          populate inferred_cause with the specific
          pattern identified above)
```

### TREE 10: SCHEMA VALIDATION (per edge)

```
Q1: Is this an ego_edge?
├─ YES → Q2: Source node type?
│   ├─ TRAFFIC_CONTROL → effect_type ∈ {INHIBITS_PROCEED,
│   │                    GRANTS_PROCEED}? If not → RECLASSIFY
│   ├─ ROAD_OBSTACLE → effect_type ∈ {CONSTRAINS_LONGITUDINAL,
│   │                  CONSTRAINS_LATERAL, INHIBITS_PROCEED}?
│   ├─ OTHER_AGENT → effect_type ∈ {CONSTRAINS_LONGITUDINAL,
│   │                CONSTRAINS_LATERAL, INHIBITS_PROCEED}?
│   ├─ ENVIRONMENT → effect_type = DEGRADES? If not → FIX
│   └─ Any mismatch → RECLASSIFY
│
│   Then check:
│   • mechanism field present?
│   • causal_status matches active flag?
│     (active_sustaining/triggering → active=true,
│      dormant → active=false)
│   • affected_state matches mapping?
│     CL_LONG → ["speed","acceleration"]
│     CL_LAT → ["heading"]
│     INHIBITS/GRANTS/DEGRADES → ["speed","heading","acceleration"]
│   • No duplicate: one edge per source-target per effect_type
│
└─ NO (inter_node_edge) → Q3
    Q3: relation = AFFECTS or CONFOUNDED_BY?
    ├─ AFFECTS → Check:
    │   • Ego is NOT source or target
    │   • layer ∈ {1, 2, 3}
    │   • Justification cites x-positions of both
    │   • Source→target validity: ENVIRONMENT→{OBSTACLE,TC,AGENT},
    │     TC→{AGENT}, AGENT→{AGENT,OBSTACLE}, OBSTACLE→{AGENT}
    │
    └─ CONFOUNDED_BY → Check:
        • layer = N/A
        • inferred_cause populated
        • Stored one direction: lower-sorted ID → higher
        • Ego not source or target
        • Both entities connect to ego
```

---

## 7. Pipeline Summary

The decision trees encode the full pipeline. For reference, the processing order is:

1. **Scene inventory:** Run each entity through TREE 1 (inclusion) and TREE 2 (infrastructure/obstacle gate). Record scene_context.
2. **Lane classification:** Run each included agent through TREE 3 (lane relationship).
3. **Ego state:** Infer speed, acceleration, heading from state.json history and future. Check if behavior matches nav command.
4. **Ego edges:** Run each entity through TREES 4a/4b/5/6/7 (edge type assignment) → TREE 8 (active/dormant/distractor).
5. **Inter-node edges:** Run each non-ego pair through TREE 9.
6. **Validation:** Run all edges through TREE 10 (schema check).

Entities that fail all ego_edge trees → record in `distractors` array with rejection reason citing which tree question failed and specific spatial/kinematic evidence.

---

## 8. Output Format

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
      "category": "<free-text, e.g. vehicle.car, signal.red, cone, wet_road>",
      "label": "<descriptive — use generic types (sedan, SUV, truck, bus), NOT brand names>",
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
      "mechanism": "<short free-text phrase>",
      "affected_state": ["<speed|heading|acceleration>"],
      "active": "<true|false>",
      "causal_status": "<active_sustaining|active_triggering|dormant>",
      "justification": "<1-2 sentences citing behavioral evidence>"
    }
  ],
  "inter_node_edges": [
    {
      "source": "<node id>",
      "target": "<node id — NEVER ego>",
      "relation": "<AFFECTS | CONFOUNDED_BY>",
      "layer": "<1|2|3|N/A>",
      "inferred_cause": "<only for CONFOUNDED_BY>",
      "justification": "<1-2 sentences with x-positions of both source and target>"
    }
  ],
  "distractors": [
    {
      "id": "<agent ID>",
      "type": "<OTHER_AGENT | TRAFFIC_CONTROL | ROAD_OBSTACLE>",
      "category": "<free-text>",
      "position_relative_to_ego": "<qualitative>",
      "rejection_reason": "<which tree question failed and why — cite spatial/kinematic evidence>"
    }
  ]
}
```

---

## 9. Worked Example

**Input:** Nav: "Turn left at intersection." Wet road. Yellow signal in CAM_FRONT. Lead sedan braking. Two oncoming vehicles. Bus at stop with nearby pedestrian. SUV behind ego. Cyclist on distant sidewalk.

**State.json:** sedan_1 x=12 (same lane, decelerating 5.5→2.1), oncoming_1 x=30 (yaw≈π, 11.2 m/s), oncoming_2 x=45 (yaw≈π, 10.8), bus_5 x=8 y=-4 (stationary), pedestrian_3 x=7 y=-4.5 (0.8 m/s toward road), cyclist_7 x=5 y=8 (sidewalk, 2.1), suv_12 x=-8 (same lane, 7.0).

**Ego:** Decelerating 8.1→6.2 m/s. Intends left turn.

**Tree traces:**

sedan_1: TREE 1→included. TREE 3→same lane, same direction. TREE 5: Q1=ahead→Q2=no full block→Q3=no partial→Q4=no mediation→Q5=ego would go faster→CONSTRAINS_LONGITUDINAL. TREE 8: Q1=yes (ego decelerating)→Q2=yes (live constraint)→ACTIVE_SUSTAINING.

n2 (yellow signal): TREE 2→camera-visible→TRAFFIC_CONTROL. TREE 4a: Q0=ahead→Q1=signal→Q2=yellow→INHIBITS_PROCEED. TREE 8: Q1=ego decelerating, but sedan_1 and oncoming vehicles explain it→Q4=signal can turn red→Q5=clear→Q6=ahead on ego's path→DORMANT.

oncoming_1: TREE 1→included. TREE 3→opposing lane. TREE 6: Q1=no intrusion→Q2=ego turning across path→Q3=moving→INHIBITS_PROCEED. TREE 8: Q1=ego waiting→Q2=yes→ACTIVE_SUSTAINING.

oncoming_2: Same trace as oncoming_1 → INHIBITS_PROCEED, ACTIVE_SUSTAINING.

n3 (crosswalk): TREE 2→camera-visible→TRAFFIC_CONTROL. TREE 4a: Q0=ahead→Q3=on ego's turn path, unoccupied→INHIBITS_PROCEED, dormant.

n1 (wet road): ENVIRONMENT node→DEGRADES ego, ACTIVE_SUSTAINING.

bus_5: TREE 5: Q3=partially in ego's lane on narrow road→CONSTRAINS_LATERAL + CONSTRAINS_LONGITUDINAL. TREE 8: Q1=ego responding→ACTIVE_SUSTAINING.

pedestrian_3: TREE 1→included. Near ego's turn path, moving toward road at 0.8 m/s from ~0.3m→INHIBITS_PROCEED. TREE 8: Q1=no response yet→Q4=0.3m at 0.8m/s, one step into lane→Q5=no barrier→Q6=near ego's turn path→DORMANT.

suv_12: TREE 5: Q8=behind→Q9=heading consistent→Q10=not intersection→CONSTRAINS_LONGITUDINAL, DORMANT.

cyclist_7: TREE 8: Q4=8m at 2.1 m/s=1.05m in 0.5s, cannot reach ego's lane→DISTRACTOR.

**Inter-node edges (TREE 9):**
- n1 AFFECTS sedan_1, oncoming_1, oncoming_2, bus_5, pedestrian_3, suv_12 (Layer 1 — mandatory for all agents)
- n2 AFFECTS sedan_1, oncoming_1, oncoming_2 (Layer 2 — approaching signal)
- n3 AFFECTS pedestrian_3 (Layer 2 — crosswalk governs pedestrian)
- bus_5 AFFECTS pedestrian_3 (Layer 3 — sightline blockage)
- sedan_1 CONFOUNDED_BY oncoming_1 (Layer N/A — inferred_cause: "both responding to yellow signal timing cycle")

---

## 10. Checklist (items NOT already encoded in trees)

1. One EGO node with id "ego".
2. scene_context.road_geometry and navigation_intent populated.
3. Node labels use generic vehicle types (sedan, SUV, truck, bus), never brand names.
4. Output is valid JSON.
5. Inter_node_edge justifications include x-positions of both source and target.
6. No duplicate edges per source-target pair per effect_type.
7. Distractors array populated for all excluded entities visible in cameras with state.json data.
8. If ENVIRONMENT exists, it AFFECTS every OTHER_AGENT. No exceptions.
9. Active-sustaining uses the entire observation window (T=-1.5 to T=0). "Finished responding by T=0" ≠ "not responding."
10. Do not project ego forward in time to "reach" distant entities for active edges.

---

Now extract the causal scene graph from the following inputs:

Navigation Command: {nav_command}

Images in order:
1-4: CAM_FRONT (T=-1.5 to T=0)
5-8: CAM_FRONT_LEFT (T=-1.5 to T=0)
9-12: CAM_FRONT_RIGHT (T=-1.5 to T=0)
13-16: CAM_BACK unmirrored (T=-1.5 to T=0)
17: BEV (T=0)
18: state.json (T=0)

Return ONLY the JSON object.

This JSON output, together with the identical sensor inputs (16 camera images, BEV, state.json) and the navigation command, will be passed to an independent reviewer for validation.