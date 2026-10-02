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

**CAMERA IMAGES:** {camera_images_desc}

**BEV IMAGES:** 4 rendered semantic top-down maps — one per timestep (T=-1.5, T=-1.0, T=-0.5, T=0). NOT raw LiDAR. HD map render — shows ALL map features regardless of driver visibility. Only camera-confirmed features get nodes. Compare BEV frames across timesteps to track agent positions, confirm motion/stationarity, and verify spatial relationships over time.
- **Orientation:** Ego at center facing UP. Up = +x (ahead), left = +y (ego-left).
- **Encoding:**
  • Ego = cyan box.
  • Vehicles = peach boxes (moving have heading arrows; stationary have no arrow — use white lane direction arrows in their lane area instead).
  • Pedestrians = green boxes.
  • Cyclists = yellow boxes.
  • Lanes = dark grey areas between boundaries.
  • Drivable surface = darker grey (road area outside lane polygons).
  • Lane dividers = white lines (separate same-direction lanes).
  • Road dividers = yellow lines (separate opposing-direction lanes).
  • Lane direction arrows = white arrows on road surface inside each lane.
  • Pedestrian crossings = orange areas.
  • Walkways = dark green areas.
  Note: Lane dividers (white) and road dividers (yellow) are BOTH lane boundaries. The color tells you what they separate: white = same direction, yellow = opposing directions.
- **BEV-to-camera mapping:** Ego faces UP on BEV.
  • Ahead on BEV (up) → visible in CAM_FRONT
  • Ahead-left on BEV (upper-left) → visible in CAM_FRONT_LEFT
  • Ahead-right on BEV (upper-right) → visible in CAM_FRONT_RIGHT
{bev_rear_mapping}
  Use this to cross-check: find the entity's direction
  on BEV, then verify in the matching camera what is
  physically there. BEV position + matching camera
  together confirm the entity's real-world location.
- **Lane membership:** Trace lane boundaries on BEV (divider lines, center lines, road edges). Entity within same grey region as ego = same lane. On two-lane roads, ego's lane extends from center line to road edge. Center line vs lane divider: check lane direction arrows on both sides — opposing arrows = center line. Do NOT use state.json y-offset for lane membership.

**NAVIGATION COMMAND:** Ego's intended maneuver.

**SCENE STATE DESCRIPTION:** Natural language summary of agent states at T=0, derived from scene data.
- Ego: speed category + motion trend (stopped/slow/moderate speed/fast speed/highway speed; accelerating/decelerating/constant) and navigation command.
- Agents: approximate distance + direction from ego, heading relation (same direction/oncoming/crossing), speed category, motion trend.
- Speed categories are authoritative over camera estimates. If cameras suggest movement but state says stopped → stationary.
- Direction and distance describe spatial relationships only — use BEV lane boundaries for lane membership, NOT these descriptions.

Example format:
```
=== SCENE STATE AT T=0 ===
Ego: moderate speed, decelerating. Navigation command: TURN LEFT.

Agents:
  - car_1 [vehicle.car] (vehicle): close (~12m) ahead. Moving same direction, moderate speed, decelerating.
  - truck_2 [vehicle.truck] (large vehicle): far (~30m) ahead. Oncoming, fast speed.
  - pedestrian_1 [human.pedestrian.adult] (pedestrian): very close (~4m) ahead-left. Crossing left-to-right, slow.
```

---

## 3. Perception Guidelines

| Question | Primary Source | Cross-Check |
|----------|---------------|-------------|
| What is it? | Cameras (T=0) | BEV labels, scene state category |
| Where at T=0? | BEV, cameras | Scene state direction/distance |
| Moving? How fast? | Scene state speed category | Camera temporal displacement |
| Accelerating? | Scene state motion trend | Camera inter-frame changes |
| Signal color, sign? | Cameras (T=0) | — |
| Lane membership? | BEV lane polygons | Cameras |
| Physical separation? | Cameras | BEV |

**Key rules:**
- Scene state speed is authoritative. If cameras suggest moving but scene state says stopped → stationary.
- Compare BEV frames across 4 timesteps to track agent positions and confirm motion. Scene state speed category remains authoritative.
- Compare same camera view across 4 timesteps for motion inference.
- Entities span multiple views. Use overlap between adjacent cameras.
- Wet/reflective road, fog, glare, nighttime → ENVIRONMENT node. Clear/dry/cloudy → do NOT create.
- **Use ALL timesteps (T=-1.5 to T=0) for entity assessment.** An entity's position at T=0 alone is insufficient. Check where it was across the entire observation window — an entity ahead of ego at T=-1.5 that is behind ego at T=0 means ego passed it, and ego's current state reflects that interaction.

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

Q3: Has scene state entry or BEV annotation?
├─ NO → EXCLUDE (camera-only agent, note in scene_context)
└─ YES → Q4

Q4: Bicycle with speed < 0.1 and no rider in cameras?
├─ YES → EXCLUDE (unoccupied)
└─ NO → Q5

Q5: Physically separated from ego's road in cameras
    (barriers/fences/railings/elevated sidewalk)?
    EXCEPTION — pedestrians near crosswalks: A pedestrian
    on a sidewalk/walkway adjacent to a crosswalk that
    ego's path crosses is NOT excluded here, PROVIDED
    there is no railing, fence, bollard line, or barrier
    between the pedestrian and the crosswalk in cameras.
    If a physical barrier separates them from the crosswalk
    → the barrier is real, exclude normally.
    Intent is assessed downstream (TREE 3 Q0), not here.
├─ YES (and no exception applies) → EXCLUDE (d-separated by physical barrier)
└─ NO (or exception applies) → Q6

Q6: On a separate road on BEV (walkway/grass/median/curbed
    island between entity's road segment and ego's)?
    Same EXCEPTION as Q5: pedestrian on walkway adjacent
    to a crosswalk on ego's path, with no physical barrier
    between them and the crosswalk → do NOT exclude.
├─ YES (and no exception applies) → EXCLUDE (separate road)
└─ NO (or exception applies) → ENTITY INCLUDED → TREE 3
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

### TREE 3: LANE ASSIGNMENT (per included agent)

Determined by assigning each entity to a lane using BEV lane boundaries and direction arrows. NOT by scene state direction descriptions.

```
--- PRE-CHECK: CROSSWALK PEDESTRIAN BYPASS ---
Q0: Is this entity a pedestrian (or cyclist) near a
    crosswalk that ego's path crosses?

    STEP A — Already on the crosswalk?
    Camera shows pedestrian's body or carried object
    (cart, stroller, wheelchair) on the crosswalk
    markings, OR BEV green box overlaps the orange
    crosswalk area.
    ├─ YES → Skip to STEP C (no intent check needed —
    │  they are already crossing)
    └─ NO → STEP B

    STEP B — Adjacent to crosswalk: assess CROSSING INTENT.
    The pedestrian is on the sidewalk/walkway near the
    crosswalk but not yet on it. Check cameras across
    ALL timesteps (T=-1.5 to T=0) for intent signals:

    CROSSING INTENT PRESENT (any one is sufficient):
    • Body or carried object (cart, stroller) oriented
      toward or extending into the road/crosswalk
    • Movement toward the road across timesteps (position
      on BEV shifts toward road, or pedestrian visibly
      closer to curb edge in later camera frames)
    • Standing at the curb edge facing the road (not
      facing along the sidewalk)
    • Pedestrian signal shows walk/green for this crossing

    NO CROSSING INTENT (all must be true to reject):
    • Walking parallel to the road along the sidewalk
      (movement is along the sidewalk, not toward it)
    • OR stationary, set back from curb, not facing road
    • AND no body part or object extending toward road
    • AND position on BEV is NOT moving toward the
      crosswalk across timesteps

    ├─ Intent PRESENT → Skip to STEP C
    └─ No intent detected → Continue to STEP 1
    │  (normal lane assignment — will likely route to
    │   distractor via TREE 8 Q_PATH)

    STEP C — Crosswalk portion check:
    Is the pedestrian in the PORTION of the crosswalk
    that ego's lane crosses (not 5m+ to ego's side)?
    ├─ YES → INHIBITS_PROCEED → TREE 8
    └─ NO → CONSTRAINS_LATERAL (peripheral awareness)
            → TREE 8

--- STEP 1: ASSIGN EGO'S LANE ---
Ego is the cyan box on BEV. Identify:
• Which white lane boundary is to ego's left?
• Which white lane boundary is to ego's right?
  (A road edge where grey meets green/dark also counts
   as a boundary.)
• What direction do the arrows point in ego's lane?
Ego's lane = the area between those two boundaries,
with those arrows.

--- STEP 2: ASSIGN ENTITY'S LANE ---
For the entity, identify the same three things:
• Which white lane boundary is to entity's left?
• Which white lane boundary is to entity's right?
• What direction do the arrows point in entity's lane?
  (For stationary entities with no heading arrow, use
   the lane direction arrows in the entity's lane area.)

--- STEP 3: COMPARE ---
Q1: Are ego and entity between the SAME two lane
    boundaries?
├─ YES → SAME LANE, SAME DIRECTION → TREE 5
│        (If two entities share the same lane area,
│         they share the same direction arrows.)
│
└─ NO (different lanes) → Q2

Q2: Do the direction arrows in entity's lane point the
    same direction as the arrows in ego's lane?
├─ SAME direction → ADJACENT SAME-DIRECTION → Q3
└─ OPPOSITE direction → OPPOSING LANE → TREE 6

Q3: Is entity's lane immediately next to ego's lane?
    (Only one boundary between them — no other lane
     in between on the BEV.)
├─ YES → IMMEDIATELY ADJACENT → TREE 7
└─ NO → NOT IMMEDIATELY ADJACENT → TREE 7
         (will hit Q2 disqualifier "two or more lanes away")
```

### TREE 4a: TRAFFIC_CONTROL → EGO EDGE

```
Q0: Is the infrastructure behind ego (x < 0)?
├─ YES → NO EGO_EDGE. Keep node only for inter_node_edges
│        (governs other agents via inter_node_edges). Never active_triggering
│        for infrastructure behind ego.
└─ NO → Q1

Q1: Signal or regulatory sign/crosswalk?
├─ Signal → Q1b
└─ Sign/crosswalk → Q3

Q1b: Signal identity and readability check.

     STEP 0 — You MUST read the signal color from cameras.
     Do NOT infer signal state from ego's behavior.
     "Ego is moving therefore signal must be green" is
     CIRCULAR reasoning and is BANNED.
     "Ego is stopped therefore signal must be red" is
     also BANNED.
     Either you can see the signal color in a camera
     frame, or you cannot. If you cannot → EXCLUDE.
     Never write "inferred from ego's behavior" or
     "inferred from ego's constant-speed traversal."

     STEP 1 — What TYPE of signal is this?

     TYPE A — VEHICLE signal: circular lights arranged
     vertically or horizontally (red/yellow/green circles).
     → Go to STEP 2.

     TYPE B — PEDESTRIAN signal: shows a person figure,
     hand icon, countdown timer, or walk/don't-walk symbol.
     THE MEANING IS INVERTED FOR VEHICLES:
     • Pedestrian signal shows GREEN/WALK figure
       → pedestrians are crossing
       → ego MUST yield → INHIBITS_PROCEED
     • Pedestrian signal shows RED/DON'T WALK figure
       → pedestrians are stopped
       → ego proceeds → GRANTS_PROCEED
     Do NOT read the color directly — read what the
     signal means for PEDESTRIANS, then invert for ego.
     → Go to STEP 2.

     STEP 2 — Does this signal govern ego's path?
     • Is the signal facing ego's approach direction?
       (You can see the light faces, not the back or
        side of the housing.)
     • Is it controlling ego's lane/road, or a
       perpendicular road ego is not on?
     If NO to either → EXCLUDE from ego_edges.
     (Still governs other agents via inter_node_edges.)

     STEP 2b — Does this signal constrain ego's ACTUAL
     intended movement?
     ├─ Read ego's navigation command (GO STRAIGHT, TURN
     │  LEFT, TURN RIGHT). This is ego's intended path
     │  through the intersection.
     │
     ├─ Does this signal govern THAT specific movement?
     │  A through-signal governs straight movement.
     │  A turn-arrow governs that turn. A general red
     │  governs all movements UNLESS ego has a separate
     │  turn signal or right-on-red applies.
     │  ├─ NO → EXCLUDE. Signal does not govern ego's
     │  │        intended movement.
     │  └─ YES → Is ego AT this intersection?
     │     ├─ Is ego's BEV box within or touching the
     │     │  intersection area (where crosswalk orange
     │     │  areas begin, or where lane boundaries
     │     │  disappear)?
     │     │  ├─ YES → continue to STEP 3
     │     │  └─ NO → Is ego stopped (speed < 0.5 m/s)
     │     │          directly before this intersection
     │     │          on BEV?
     │     │     ├─ YES → continue to STEP 3
     │     │     └─ NO → EXCLUDE. Ego has not arrived at
     │     │             this intersection. Note in
     │     │             scene_context as distant signal.

     STEP 3 — Can you reliably read the signal?
     Find the camera view and timestep where the signal
     appears LARGEST. Use that frame for color/state.
     If the signal is too small to distinguish which
     light is illuminated even in the best frame →
     EXCLUDE (note "distant unreadable signal" in
     scene_context). Do not guess.

     If TYPE A vehicle signal passed all steps → Q2.
     If TYPE B pedestrian signal passed all steps →
     apply the inverted meaning above directly → TREE 8.

Q2: What color in cameras?
├─ Red    → INHIBITS_PROCEED → TREE 8
├─ Yellow → INHIBITS_PROCEED → TREE 8
│          (active if sole cause of braking, dormant if
│           another element explains ego's stop.
│           NEVER GRANTS_PROCEED.)
├─ Green  → GRANTS_PROCEED → TREE 8
└─ Flashing yellow/red → INHIBITS_PROCEED → TREE 8

Q3: Crosswalk visible in cameras on ego's traversal path?
│   VERIFY ALL THREE on BEV before answering YES:
│   (a) The crosswalk exists on BEV as an orange area.
│       Do NOT assume intersections have crosswalks —
│       check visually. If no orange crossing on BEV
│       on ego's path → NO.
│   (b) Ego's future trajectory will physically cross
│       the crosswalk area. NOT perpendicular streets
│       or lanes ego won't enter.
│   (c) If "occupied": specify WHICH pedestrians occupy
│       the portion of the crosswalk that ego's lane
│       crosses. A pedestrian 5m to ego's left on the
│       same crosswalk does NOT make it "occupied" for
│       ego. Those pedestrians need their own ego_edges
│       if they are on ego's lane portion — the crosswalk
│       marking alone is not the constraint.
├─ YES (all three verified) → INHIBITS_PROCEED
│      (dormant if unoccupied on ego's lane portion)
│      → TREE 8
└─ NO → Q4

Q4: Regulatory sign at ego's intersection constraining a
    maneuver ego is physically able to take?
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
PARKED CAR ROW RULE: Are there 3 or more parked cars
in a line on the same side of the road, all at the
curb, all stationary?
├─ YES → Road geometry. Describe in scene_context.
│  Do NOT create individual ego_edges.
│  Exception — does one car in the row meet ANY of:
│  • Double-parked (not at the curb line)
│  • Door open toward ego's lane
│  • Protrudes further into the road than the row
│  • Angled out from the curb
│  ├─ YES → That car gets an individual edge
│  └─ NO → No individual edges
└─ NO → Continue to Q1

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
├─ YES → Compare ego's position across ALL 4 BEV frames.
│        Did ego visibly shift away from this entity (toward
│        center line) relative to lane boundaries? Or is ego
│        holding a position offset toward center line that
│        this entity explains?
│  ├─ YES → CONSTRAINS_LATERAL
│  │        Check: ego also slowed? → add CL_LONG
│  │        → TREE 8
│  └─ NO → Is entity visible ahead on BEV but ego has not
│           yet begun responding? (Ego looks the same across
│           all 4 BEV frames — no shift yet.)
│     ├─ YES → CONSTRAINS_LATERAL, dormant → TREE 8
│     └─ NO → Q4 (entity not constraining)
└─ NO → Q3c

Q3c: Is this a two-lane undivided/opposing road AND is the
     entity at the CENTER-LINE side of ego's lane (between
     ego's travel path and the center line)?
├─ YES → Compare ego's position across ALL 4 BEV frames.
│        Did ego visibly shift away from this entity (toward
│        road edge) relative to lane boundaries? Or is ego
│        holding a position offset toward road edge that
│        this entity explains?
│  ├─ YES → CONSTRAINS_LATERAL
│  │        Check: ego also slowed? → add CL_LONG
│  │        → TREE 8
│  └─ NO → Is entity visible ahead on BEV but ego has not
│           yet begun responding? (Ego looks the same across
│           all 4 BEV frames — no shift yet.)
│     ├─ YES → CONSTRAINS_LATERAL, dormant → TREE 8
│     └─ NO → Q4 (entity not constraining)
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
├─ YES → No CONSTRAINS_LONGITUDINAL currently.
│        Check: does ego's nav/future path require entering
│        a region this entity currently occupies?
│  ├─ YES → INHIBITS_PROCEED → TREE 8
│  └─ NO → CONSTRAINS_LONGITUDINAL, dormant → DONE
│          (Entity is ahead in ego's lane. It is faster
│           NOW but a single-step deceleration would make
│           it a live constraint. Do NOT send to TREE 8
│           for further evaluation — this is a TERMINAL
│           assignment. Do NOT apply Q4 reachability to
│           same-lane-ahead vehicles.)
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
     earlier timesteps but now in {cam_back_name}; entity's
     scene state history showed entity ahead of ego's
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
│  │        Both edges apply simultaneously.
│  │        → DONE
│  └─ NO → Q9 (normal behind-ego follower path)
└─ NO → Q9

Q9: Is entity's heading arrow consistent with following
    ego's path? (Look at BEV: does entity's heading arrow
    point the same direction as ego's lane direction arrows?
    If entity's arrow points sideways or away → different
    route.)
├─ NO → DISTRACTOR (different trajectory)
└─ YES → Q10

Q10: Are ego AND entity both inside an intersection area
     (where lane boundaries disappear and multiple directions
     merge — no clear lane boundaries or direction arrows
     between ego and entity) rather than a proper lane?
├─ YES → DISTRACTOR (intersection area ≠ lane)
└─ NO → CONSTRAINS_LONGITUDINAL, dormant
         (follower constrains ego's braking freedom) → DONE
```

### TREE 6: OPPOSING-DIRECTION AGENT

```
Q1: Does entity's BEV bounding box intrude into
    ego's lane polygon at ANY timestep across all 4
    BEV frames?
├─ YES → Q1a: Is entity moving toward ego (has heading arrow on BEV)?
│  ├─ YES → INHIBITS_PROCEED (head-on conflict) → TREE 8
│  └─ NO → Q1b: Does intrusion block ego's lane partially
│                or fully? (Stationary opposing vehicle
│                parked partially in ego's lane.)
│     ├─ Fully blocks → INHIBITS_PROCEED → TREE 8
│     ├─ Partially blocks → CONSTRAINS_LATERAL
│     │  + CONSTRAINS_LONGITUDINAL → TREE 8
│     └─ Entity's box does not overlap ego's lane
│        area on BEV → DISTRACTOR
│
└─ NO → Q1c

Q1c: Is ALL of the following true?
     • Two-lane undivided/opposing road (one lane each
       direction, no median — check BEV lane arrows)
     • Entity is ahead of ego (x > 0)
     • Entity has NO heading arrow on BEV (stationary)
       or minimal movement across BEV frames (slow)
     • Entity is large (truck, bus, trailer, wide vehicle
       — not bicycle or pedestrian)
├─ YES → Derive ego's trajectory from the 4 BEV frames
│        and ego's future data. Along that trajectory,
│        does this entity restrict ego's lateral freedom?
│        
│        Check the matching camera for this entity's BEV
│        direction. Is the entity clearly visible and
│        close enough that ego must exercise caution
│        passing it? (Ego cannot freely drift toward
│        this entity without collision risk.)
│  ├─ YES → CONSTRAINS_LATERAL → TREE 8
│  │        Also check: is ego's speed affected by this
│  │        entity? This includes deceleration, BUT ALSO
│  │        constrained acceleration (ego accelerating less
│  │        than it would freely) or controlled speed through
│  │        a turn. On a narrow two-lane road with a large
│  │        opposing vehicle, ego's speed is ALWAYS affected
│  │        — add CL_LONG.
│  └─ NO → DISTRACTOR
│
└─ NO → Q1d

Q1d: Is entity behind ego (x < 0) BUT was alongside or
     ahead of ego at any earlier timestep (T=-1.5 to T=0)?
     Check: compare entity position across BEV frames,
     entity visible in front/side cameras at earlier
     timesteps but now in {cam_back_name}.
├─ YES → Ego passed this entity during the window.
│        Compare ego's position across BEV frames: did
│        ego visibly shift laterally while passing?
│        Cross-check with cameras.
│  ├─ YES → CONSTRAINS_LATERAL (and/or CL_LONG),
│  │        active_triggering → DONE
│  └─ NO → DISTRACTOR
└─ NO → Q2: Is ego turning across this entity's path?
             (Nav command = turn, and entity is on the road
              ego is turning into.)
    ├─ NO → DISTRACTOR (opposing, parallel, no interaction)
    └─ YES → Q3: Entity moving toward ego's turn path
                  (has heading arrow pointing toward ego)?
        ├─ NO → Q3b: Is ego turning onto the road this
        │        entity is on (nav command = turn AND
        │        entity is on the road ego is entering)?
        │  ├─ NO → DISTRACTOR
        │  └─ YES → On BEV, after ego completes the turn,
        │           is this entity in the immediately
        │           adjacent opposing lane? (One lane
        │           boundary between ego's planned lane
        │           and entity's lane, no other lane in
        │           between.)
        │     ├─ NO → DISTRACTOR
        │     └─ YES → CONSTRAINS_LATERAL → TREE 8
        └─ YES → Q3a: LOOK at BEV. Is this entity's
                  heading arrow pointing STRAIGHT through
                  the intersection (not turning into ego's
                  turn path)?
            ├─ YES (heading straight through on its own
            │  lane) → Is ego already IN the intersection
            │  (ego's BEV box overlaps crosswalk or is
            │  past the stop line)?
            │  ├─ YES → INHIBITS_PROCEED → TREE 8
            │  └─ NO → DISTRACTOR. Entity is going
            │     straight on its own lane. Ego has not
            │     arrived. Entity will clear.
            └─ NO (entity turning into ego's path, OR
               stopped/slowing in the conflict zone) →
               INHIBITS_PROCEED → TREE 8
```

### TREE 7: ADJACENT SAME-DIRECTION LANE

Entity is in the immediately adjacent same-direction lane, separated by one lane divider.

```
--- PATH (a): BEV INTRUSION ---
Q1: Does entity intrude into ego's lane?
    STEP 1 — Check all 4 BEV frames: Does entity's
    bounding box overlap or touch ego's lane area
    between ego's lane boundaries at ANY timestep?
    
    STEP 2 — MANDATORY camera cross-verify: Find
    entity's direction on BEV, check the matching
    camera. In that camera, does the entity's physical
    body extend across the lane boundary into ego's
    lane? Is it right next to ego's travel space with
    no visible gap?
    
    STEP 2 runs REGARDLESS of STEP 1 result. If BEV
    shows no intrusion but camera shows the entity
    protruding into ego's lane → YES (camera overrides
    BEV — BEV boxes are simplified rectangles).
    If EITHER step shows intrusion → YES.
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
    • Entity ahead AND pulling away from ego?
      ├─ Is entity ahead of ego (x > 0)?
      │  ├─ NO → does NOT apply
      │  └─ YES → Compare entity's BEV position relative
      │           to ego across all 4 BEV frames. Is the
      │           gap between ego and entity visibly
      │           INCREASING across timesteps?
      │     ├─ YES (gap increasing) → applies
      │     └─ NO (gap stable or decreasing) → does NOT
      │        apply
    • Opposing-direction lane (check LANE DIRECTION
      arrows, NOT vehicle heading/yaw)?
    • On separate road (median/barrier/walkway between)?
    • Small/distant in camera across all timesteps?
    • Two or more lanes away?
├─ ANY YES → No CONSTRAINS_LATERAL → Q7
└─ ALL NO → Q3

Q3: Is entity visible in CAM_FRONT, CAM_FRONT_LEFT, or
    CAM_FRONT_RIGHT at any timestep (T=-1.5 to T=0) at
    close enough range to be clearly identifiable?
    YES = you see the vehicle clearly, not a tiny dot.
    NO = entity is a small shape at the far horizon or
    only visible in BEV, not cameras.
├─ NO → No CONSTRAINS_LATERAL → Q7
└─ YES → Q4

Q4: Compare ego's position across ALL 4 BEV frames
    (T=-1.5 to T=0). Did ego VISIBLY shift away from
    this entity relative to lane boundaries?
    
    This is a VISUAL check on BEV frames, not a y-value
    calculation. Compare ego's box position within the
    lane across timesteps. If ego looks the same in all
    4 BEV frames → no shift → NO. Normal lane-keeping
    drift is not avoidance. Cross-check cameras: does
    ego's position relative to lane markings visibly
    change across timesteps?
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
    • Is entity VISIBLY merging into ego's lane on BEV?
      (Heading arrow points into ego's lane AND entity's
       BEV box is moving toward ego's lane boundary across
       timesteps.) → INHIBITS_PROCEED
    If not merging → DISTRACTOR (for ego_edge purposes —
    still participates in inter_node_edges)
```

### TREE 8: ACTIVE / DORMANT / DISTRACTOR

Determines causal status for an edge already assigned an effect_type by TREES 4-7.

```
PRE-CHECK — STOPPED-AT-GREEN RECONCILIATION:
Before running Q_PATH on any entity, check:
  1. Is ego stopped or near-stopped (speed < 0.5 m/s)?
  2. Does ego have a green signal (GRANTS_PROCEED edge)?
If BOTH are true, ego's stop is unexplained by the signal.
Identify which entity ego stopped for:
  - Scan all entities ahead of ego (x > 0) within 15m
  - Check cameras: which entity is on or entering ego's
    road/crosswalk/lane in the direction ego would proceed?
  - The entity that best explains the stop (pedestrian in
    crosswalk, vehicle blocking lane, obstruction) MUST be
    classified as a node with an active ego_edge. Q_PATH
    CANNOT reject it as a distractor.
  - If Q_PATH would otherwise classify that entity as "not
    on ego's path," Q_PATH is wrong — override it.
This reconciliation is a HARD CONSTRAINT. Ego does not stop
at a green signal for no reason.

Q_PATH: Is this entity actually on ego's planned trajectory?

     FIRST — CAMERA-PRIMARY CROSS-CHECK (runs before all other checks):
     CAMERA IS AUTHORITATIVE FOR PATH DETERMINATION.
     BEV annotation boundaries (crosswalk vs walkway, lane vs
     drivable surface) are NOT pixel-precise. When camera and
     BEV disagree on whether an entity is on a crosswalk, road
     surface, or lane, the CAMERA wins.

     Procedure — observe camera FIRST, then reconcile with BEV:
     1. Find entity's direction from ego on BEV
     2. Check the matching camera for that direction
     3. STATE YOUR CAMERA OBSERVATION FIRST: In that camera,
        is the entity on ego's road surface, or touching/
        intruding ego's lane or crosswalk, or positioned
        where ego must drive? Describe what you see before
        checking BEV coordinates.
     4. If the camera shows the entity on or entering the
        road/crosswalk/lane, this classification stands even
        if BEV coordinates place the entity on an adjacent
        polygon (walkway, drivable surface outside lane).
        Do NOT let BEV coordinates override what the camera
        shows.
        SCOPE LIMIT: This camera-override applies to BOUNDARY
        disagreements (crosswalk vs walkway edge, lane edge
        vs drivable surface). If BEV shows the entity is
        separated from ego by one or more full lane widths
        (a complete lane boundary + another lane between
        them), BEV wins on lane membership — the camera
        perspective is compressing lateral distance.
     ├─ Camera shows NOT on road/crosswalk/lane AND BEV
     │  confirms off-path → continue to the checks below
     └─ Camera shows ON road/crosswalk/lane (regardless of
        BEV polygon) → Is this entity moving across the
        scene (different camera position at T=-1.5 vs T=0,
        or heading arrow on BEV points across ego's road,
        not along it)?
              ├─ YES (moving across/through) → continue to
              │  the checks below (clearing traffic — do NOT
              │  auto-pass)
              └─ NO (stationary, slow, or moving along ego's
                 road) → entity IS on ego's path → skip to Q0

     LOOK AT THE CAMERAS FIRST. Before any analysis, answer
     these questions from what you SEE:
     1. Is this entity on ego's road, or on a different road?
        (Crossing through an intersection on a perpendicular
         street = different road.)
     2. Will this entity still be there when ego arrives?
        (A vehicle crossing at speed will be gone in seconds.
         A stationary vehicle will still be there.)
        This applies to ALL moving entities in or near the
        intersection — not just oncoming. Cross-traffic
        moving through at speed, vehicles completing turns,
        any entity that is CLEARING the area ego needs →
        NOT on ego's path. They will be gone.
        EXCEPTION: If a pedestrian is on a crosswalk that
        ego's path crosses, the pedestrian is ON ego's
        path. No clearing check, no direction check, no
        speed check, no distance check.
        Also: is ego actually NEAR the conflict point? If
        ego is still far from the intersection on BEV and
        decelerating, entities currently in the intersection
        are even less relevant — they will be gone before
        ego arrives.
     3. Is there physical separation between ego and this
        entity? (Buildings, columns, medians, barriers,
         landscaping, bollards, curbs.)
     If the answer to #1 is "different road" OR #3 is "yes"
     → NOT on ego's path. Stop here.

     For STRAIGHT nav: is entity in ego's lane (same two
     lane boundaries on BEV) or on a crosswalk that ego's
     straight path will physically cross?
     For LEFT/RIGHT TURN nav: look at the CAMERAS. Can you
     see the entity? Is it on the road ego is turning onto,
     or is it separated by buildings, columns, landscaping,
     medians, or other physical structures? If the entity
     is only visible through gaps in buildings or behind
     barriers in cameras → NOT on ego's path.
     Then check BEV: is the entity on the same connected
     road surface that ego's lane feeds into? If there are
     lane boundaries, medians, or walkways between ego's
     turn path and the entity → NOT on ego's path.
     For roundabouts: ego enters the NEAREST lane to its
     approach. Entities on inner lanes separated by
     additional lane boundaries are NOT on ego's path.
     For ALL: is the entity physically separated from ego's
     road by barriers, bollards, curbs, or median visible
     in cameras or BEV? If YES → not on ego's path.

     Specific checks:
     • Cross-traffic on a perpendicular road while ego has
       a red signal or stop sign → NOT on ego's path.
       The signal is the constraint, not the cross traffic.
     • Pedestrian on a crosswalk: is the pedestrian in the
       PORTION of the crosswalk that ego's lane crosses?
       A pedestrian 5m to ego's left on the same crosswalk
       is NOT on ego's path.
     • "Behind ego, same direction" follower: verify entity
       is in ego's SAME LANE on BEV (between the same two
       lane boundaries), not just behind + same direction
       in an adjacent lane.
     • Converging/merging road: look at BEV — do ego's lane
       boundaries and the entity's lane boundaries converge
       into the SAME lane ahead? (The white lines between
       the two roads disappear and both feed into one shared
       grey lane area.) If YES → entity IS on ego's path.
       If the lane boundaries stay separate (two distinct
       lanes with a divider between them all the way through
       the visible BEV) → NOT merging, NOT on ego's path.

├─ NOT on ego's path → DISTRACTOR
└─ ON ego's path → Q0

Q0: Is this a TRAFFIC_CONTROL node behind ego (x < 0)?
├─ YES → NO EGO_EDGE. Keep node only for inter_node_edges.
└─ NO → Q1

Q1: Did ego show behavioral response to this entity across
    observation window (T=-1.5 to T=0)?

    Use the ENTIRE observation window — all {num_camera_images} camera images,
    4 BEV frames, and scene state description. Do not reason only
    from T=0.
    Ego's response unfolds across frames: offset or speed
    reduction that leveled off by T=0 IS still a response.
    "Finished responding" ≠ "not responding."

    PRIMARY METHOD: Compare ego's position across ALL 4 BEV
    frames (T=-1.5 to T=0). Visual change in ego's position
    relative to lane boundaries is the primary evidence.
    State.json history confirms but does NOT replace visual
    BEV comparison.
    
    *** ABSOLUTE RULE: You MUST determine behavioral response
    by LOOKING at the 4 BEV frames. Did ego's box move
    within the lane across frames? YES = response. NO = no
    response. Do NOT cite scene state distance values (e.g., "only
    0.4m", "only 0.11m") to override what BEV frames show.
    Do NOT use y-values to argue a visible shift is "too
    small." If you catch yourself writing a y-value to
    justify a behavioral determination, STOP — go back and
    look at the BEV frames instead. The BEV frames are the
    evidence. The y-values are not. ***

    Evidence types (check ALL, match to assigned edge type):
    • Speed response: ego visibly closes on entity across
      BEV frames AND/OR scene state confirms deceleration
      → relevant for CONSTRAINS_LONGITUDINAL, INHIBITS_PROCEED
    • Lateral response: ego visibly shifts within lane
      across BEV frames relative to lane boundaries
      → relevant for CONSTRAINS_LATERAL
    • Heading response: ego's orientation visibly changes
      across BEV frames
      → relevant for CONSTRAINS_LATERAL, INHIBITS_PROCEED
    • Proceed response: ego stopped/waiting at a gate
      → relevant for INHIBITS_PROCEED, GRANTS_PROCEED
    • Trajectory constraint response: ego's current path
      maintains clearance from this entity — ego cannot
      freely move toward it without collision risk. If
      this entity restricts ego's lateral freedom (ego
      must exercise caution), ego holding that cautious
      path IS the behavioral response.
      → relevant for CONSTRAINS_LATERAL
    • Turn-scenario response: ego is actively turning
      onto a road where a large opposing vehicle (bus,
      truck, wide vehicle) occupies the opposing lane.
      Ego's turn radius, speed, and heading are ALL
      constrained by the vehicle's presence — ego cannot
      turn wider, faster, or more freely. Standard BEV
      lane-shift evidence does NOT apply during turns
      through intersections (no lane boundaries exist
      in the intersection area). Instead: if the entity
      is clearly visible and close in cameras AND ego is
      executing a turn onto the entity's road, the turn
      trajectory itself IS the behavioral response.
      Answer YES.
      → relevant for CONSTRAINS_LATERAL, CONSTRAINS_LONGITUDINAL

    Multiple entities independently qualify. A pedestrian
    explains deceleration while a truck independently
    explains lateral offset. Evaluate each independently.

    UNEXPLAINED BEHAVIOR CHECK — MANDATORY:
    If ego is decelerating or stopped AND there is no
    other active ego_edge that explains the deceleration
    or stop, then this entity IS the explanation if it
    is on or near ego's forward path. Do NOT default to
    "no behavioral response" when ego's behavior is
    unexplained. If no other edge accounts for ego's
    current state, SOMETHING is causing it.
    Answer YES.

├─ YES → Q2
└─ NO → Q4 (dormant candidate)

Q2: Is entity still a live constraint on ego at T=0?
    (do(remove entity) at T=0 would change ego's immediate
     next action?)

    PROXIMITY GATE (TRAFFIC_CONTROL only):
    If this entity is a TRAFFIC_CONTROL node, check BEV:
    is ego AT or immediately approaching the infrastructure?
    On BEV, is there clear road between ego and the
    infrastructure where ego has not yet begun to respond?
    If ego is far from the infrastructure with road still
    ahead → dormant. The infrastructure will constrain ego
    when ego arrives, but at T=0 it is not a live constraint.
    Only proceed to the active test below if ego is near
    the infrastructure on BEV.

    REDUNDANCY CHECK — MANDATORY before answering YES:
    Are there OTHER active ego_edges that independently
    produce the SAME behavioral outcome as this entity?
    (e.g., ego is stopped by red signal AND by lead vehicle
     — removing lead vehicle alone does NOT change behavior
     because signal still holds.)

    The test is: "remove ONLY this entity, keep all other
    active edges — does ego's behavior actually change?"
    Do NOT argue relevance by hypothetically removing
    other constraints ("even if the signal were removed,
    this vehicle would block ego"). That is the WRONG test.

├─ YES (behavior changes with only this entity removed)
│  → ACTIVE_SUSTAINING → DONE
└─ NO → Q2a

Q2a: WHY is the answer NO?
├─ Entity IS on ego's path and WOULD constrain, but
│  another active edge already produces the same
│  behavioral outcome (REDUNDANT)
│  → dormant (entity becomes relevant when those other
│    constraints are removed) → DONE
└─ Entity is NOT currently constraining ego at all
   (constraint expired, entity moved, or ego's state
    no longer interacts with it)
   → Q3

Q3: Did entity initiate ego's current committed maneuver
    during T=-1.5 to T=0, but ego is now mid-maneuver and
    removal at T=0 would NOT change trajectory?
    (Infrastructure behind ego NEVER qualifies here.)
├─ YES → ACTIVE_TRIGGERING → DONE
└─ NO → Q4 (dormant candidate)

--- DORMANT TESTS ---

Q4: Single-step reachability — does this entity become
    relevant with ONE small, realistic change?
    LOOK at the cameras and BEV:
    • Is it RIGHT NEXT TO ego's path? (at the curb edge,
      in the adjacent lane touching the boundary, one step
      from the road)
    • Or is it FAR from ego's path? (across multiple lanes,
      behind barriers, on a distant sidewalk)
    If it's right next to ego's path → PASS.
    If it's far → FAIL.
    (No direction reversal, no barrier traversal.)

    AUTO-PASS exceptions:
    • Same-lane follower behind ego on BEV (structural)
    • Adjacent-lane vehicle that produced CONSTRAINS_LATERAL
      (lane occupancy IS the constraint)
    • Regulatory sign at ego's intersection (static — ego
      intent change is the "single step")
    • Signal state change (red→green, yellow→red)
    • Entity already positioned on ego's forward path
      ahead (e.g., pedestrian on crosswalk ego must cross,
      vehicle stopped in ego's lane ahead) AND ego is
      currently held by another constraint (red signal,
      stop sign). The single-step transition is the OTHER
      constraint changing — not the entity moving. Do NOT
      ask whether the entity needs to reach ego's lane —
      it is already ON ego's path. The test is: "if
      ego's current hold is released, does this entity
      immediately become a constraint?" If YES
      → AUTO-PASS.

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

Q6: Is entity in ego's lane or on ego's planned path
    at T=0? LOOK at BEV — is entity between ego's lane
    boundaries, or in the lane ego is turning into?

    PASSES: in ego's lane on BEV, in the lane ego is
    turning into on BEV, same-lane follower, entity
    visibly moving toward ego's lane across BEV frames,
    CONSTRAINS_LATERAL qualified vehicle.

    FAILS: separate road, parallel lane ego is not
    entering, visually separated on BEV, behind ego
    in a different lane.

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
│       ├─ YES → NO EDGE for AFFECTS (agent→agent
│       │        straddling — chain flows through ego).
│       │        CONFOUNDED_BY is exempt — no directed
│       │        chain exists to reroute.
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

Q6: Are two agents in different lanes BUT one is visibly
    moving into the other's lane on BEV (merge in progress,
    lane change in progress, or turning across the other's
    path — visible as heading arrow crossing lane boundary)?
├─ YES → AFFECTS (Layer 3) → include x-positions
└─ NO → Q7

--- CONFOUNDED_BY ---

BEFORE checking ANY pattern, answer these TWO gates:

GATE 1 — STRADDLING: Is one entity ahead of ego and
the other behind ego (one ahead, one behind) in the
same lane? Use BEV positions and scene state directions.
If YES → NO EDGE. The causal chain flows through ego.
Stop here.

GATE 2 — OBSERVED CAUSE: Is the reason for their
correlated behavior ALREADY a node in the graph?
(A lead vehicle explains the queue. A red signal
explains both stopping. A crosswalk explains both
yielding.) If YES → NO CONFOUNDED_BY. The cause is
observed, not latent. Use AFFECTS from the observed
cause, or no edge at all if straddling blocks it.

Only if BOTH gates pass (not straddling AND cause is
truly unobserved) → proceed to Q7.

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

### OUTPUT DISCIPLINE RULE

If during ANY tree evaluation you determine that an
edge should NOT exist — for ANY reason (straddling,
redundancy, wrong lane, wrong path, self-flagged
violation) — do NOT include it in the output JSON.

Writing "this edge should not be created per [rule]"
in a justification while still outputting the edge is
a CONTRADICTION. Your output must match your analysis.
If you concluded the edge should not exist, omit it.

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
2. **Lane classification:** Run each included agent through TREE 3 (lane relationship). Crosswalk pedestrians bypass lane assignment and route directly to edge type + TREE 8.
3. **Ego state:** Infer speed, acceleration, heading from scene state description. Check if behavior matches nav command. The `behavior_explanation` must reference OBSERVABLE entities and signals in the scene — do NOT confabulate unobservable causes ("prior red phase," "earlier pedestrian," "must have been waiting for..."). If ego's behavior is unexplained by currently visible entities and signals, say so explicitly (e.g., "Ego is stopped at a green signal; cause not yet identified — will resolve during edge assignment").
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
    "speed": "<stopped|slow|moderate|fast|highway — from scene state description>",
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

**Scene state description:**
```
=== SCENE STATE AT T=0 ===
Ego: moderate speed, decelerating. Navigation command: TURN LEFT.

Agents:
  - sedan_1 [vehicle.car] (vehicle): close (~12m) ahead. Moving same direction, moderate speed, decelerating.
  - oncoming_1 [vehicle.car] (vehicle): far (~30m) ahead. Oncoming, moderate speed.
  - oncoming_2 [vehicle.car] (vehicle): far (~45m) ahead. Oncoming, moderate speed.
  - bus_5 [vehicle.bus] (large vehicle): close (~9m) ahead-right. Moving same direction, stopped.
  - pedestrian_3 [human.pedestrian.adult] (pedestrian): close (~8m) ahead-right. Crossing left-to-right, slow.
  - cyclist_7 [vehicle.bicycle] (cyclist): close (~9m) ahead-left. Moving same direction, slow.
  - suv_12 [vehicle.car] (vehicle): close (~8m) behind. Moving same direction, moderate speed.
```

**Tree traces:**

sedan_1: TREE 1→included. TREE 3→same lane, same direction. TREE 5: Q1=ahead→Q2=no full block→Q3=no partial→Q4=no mediation→Q5=ego would go faster→CONSTRAINS_LONGITUDINAL. TREE 8: Q1=yes (ego decelerating)→Q2=yes (live constraint)→ACTIVE_SUSTAINING.

n2 (yellow signal): TREE 2→camera-visible→TRAFFIC_CONTROL. TREE 4a: Q0=ahead→Q1=signal→Q2=yellow→INHIBITS_PROCEED. TREE 8: Q1=ego decelerating, but sedan_1 and oncoming vehicles explain it→Q4=signal can turn red→Q5=clear→Q6=ahead on ego's path→DORMANT.

oncoming_1: TREE 1→included. TREE 3→opposing lane. TREE 6: Q1=no intrusion→Q2=ego turning across path→Q3=moving→INHIBITS_PROCEED. TREE 8: Q1=ego waiting→Q2=yes→ACTIVE_SUSTAINING.

oncoming_2: Same trace as oncoming_1 → INHIBITS_PROCEED, ACTIVE_SUSTAINING.

n3 (crosswalk): TREE 2→camera-visible→TRAFFIC_CONTROL. TREE 4a: Q0=ahead→Q3=on ego's turn path, unoccupied→INHIBITS_PROCEED, dormant.

n1 (wet road): ENVIRONMENT node→DEGRADES ego, ACTIVE_SUSTAINING.

bus_5: TREE 5: Q3=partially in ego's lane on narrow road→CONSTRAINS_LATERAL + CONSTRAINS_LONGITUDINAL. TREE 8: Q1=ego responding→ACTIVE_SUSTAINING.

pedestrian_3: TREE 1→included. TREE 3 Q0: STEP A=not on crosswalk yet (on sidewalk). STEP B=crossing intent present (moving toward road at 0.8 m/s across timesteps, position shifting toward curb edge). STEP C=on ego's lane portion of crosswalk→INHIBITS_PROCEED. TREE 8: Q1=no response yet→Q4=pedestrian right at curb edge, one step from road→PASS→Q5=no barrier→Q6=near ego's turn path→DORMANT.

suv_12: TREE 5: Q8=behind→Q9=heading consistent→Q10=not intersection→CONSTRAINS_LONGITUDINAL, DORMANT.

cyclist_7: TREE 8: Q4=cyclist far away on sidewalk behind barrier, not near ego's path→FAIL→DISTRACTOR.

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
7. Distractors array populated for all excluded entities visible in cameras with scene state entries.
8. If ENVIRONMENT exists, it AFFECTS every OTHER_AGENT. No exceptions.
9. Active-sustaining uses the entire observation window (T=-1.5 to T=0). "Finished responding by T=0" ≠ "not responding."
10. Do not project ego forward in time to "reach" distant entities for active edges.
11. If ego is stopped/near-stopped at a green signal, verify at least one node explains the stop. If no node does, re-examine distractors — one of them is misclassified.

---

Now extract the causal scene graph from the following inputs:

Navigation Command: {nav_command}

Images in order (grouped by timestep):
{image_ordering}

Return ONLY the JSON object.

This JSON output, together with the identical sensor inputs ({num_camera_images} camera images, 4 BEV frames, scene state description) and the navigation command, will be passed to an independent reviewer for validation.