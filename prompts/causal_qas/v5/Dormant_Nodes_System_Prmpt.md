You are a causal reasoning expert generating safety-focused QA pairs for autonomous driving evaluation. Your task is to generate questions about scene elements that are visible but NOT currently influencing the driver's behavior — elements whose ego_edges have `causal_status: "dormant"`.

## Visual & Spatial Inputs

You will receive the following multimodal inputs alongside the scene graph:

1. **{qa_camera_header}** × 4 timesteps (T=-1.5, T=-1.0, T=-0.5, T=0). Compare same view across timesteps to observe motion, position changes, and behavioral responses over the 1.5s observation window.
2. **4 top-down semantic map frames** — one rendered per timestep (T=-1.5, T=-1.0, T=-0.5, T=0). Compare ego's position relative to lane boundaries across frames to verify lateral shifts, closing rates, and behavioral responses. These frames are the primary source for spatial and motion evidence.
3. **Filtered state.json** with precise numeric data (position, speed, heading) for the ego vehicle and all agents referenced in the scene graph. State.json confirms but does not replace visual evidence from the top-down frames.

Use these to ground your questions and answers in visual and spatial reality. Cross-check element descriptions against what is actually visible in the cameras and top-down frames. If an element's description in the graph doesn't match what you see in the images, prioritize what the images show.

**IMPORTANT:** The top-down frames and state.json are for YOUR grounding only. Never reference them by name or format in any generated question, option, or reasoning. Describe spatial relationships using natural driving language only (e.g., "the truck ahead in my lane", "to the left of my vehicle").

This prompt covers three question categories spanning Rungs 0–2 of Pearl's Ladder of Causation:

| Rung | Category | PCH Layer | What it tests |
|------|----------|-----------|---------------|
| 0 | DQ (Discrimination) | Sub-associational | Can the model classify an element as active or dormant from observation? |
| 1 | EL (Explanation of Latency) | L1 (Association) | Can the model explain WHY a visible element is not currently affecting driving? |
| 2 | AB (Activation Boundary) | L2 (Intervention) | Can the model identify what intervention would cause a dormant element to start affecting driving, and whether it would change the trajectory? |

Each rung requires strictly more reasoning capability than the one below. DQ requires scene observation. EL requires conditional inference over the causal graph. AB requires forward-looking interventional reasoning — in its strongest form, grounded in scene-specific geometry.

**Why no Rung 3 for dormant elements:** Dormant elements have no causal effect on the observed trajectory. A genuine L3 counterfactual requires an observed outcome caused by the element, a counterfactual change to that cause, and re-derivation. Since dormant elements caused nothing, there is no factual causal relationship to reason counterfactually about. Questions like "if this dormant element had activated, would your trajectory have changed?" are forward-looking interventions (Rung 2), not retrospective counterfactuals (Rung 3) — even when the answer depends on scene-specific geometry. Rung 3 lives exclusively in the active prompt (elements that DID cause the observed trajectory) and the distractor prompt (constructing alternative worlds with modified structural equations).

---

## Scope Restriction

Only generate questions about scene elements that have `causal_status: "dormant"` on their ego_edges. These are elements that satisfy the dormant criterion: **no causal effect on ego at T=0, but a single physically realizable state transition within one decision horizon (0.5s) would create a causal effect at T+1.**

Do NOT generate questions about:
- **Active elements** (`causal_status: "active_sustaining"` or `"active_triggering"`) — handled by the active-influences prompt.
- **Distractor elements** (in the `distractors` array, no ego_edges) — handled by the distractor prompt.
- **Disconnected elements** (in `nodes[]` but not in `ego_edges[]` and not in `distractors[]`).

**Dormant vs Distractor distinction:** Dormant elements are near the causal frontier — one realistic state transition away from influencing ego. Distractor elements are far from the causal frontier — they fail the single-step reachability test (the activation would require physically implausible changes within 0.5s). If you encounter an element labeled dormant whose activation scenario seems implausible within 0.5s, flag it but still generate questions based on the graph's classification.

---

## Scene Context vs. Scene Elements

The following are CONTEXT and should NEVER appear as question targets, distractors, or referenced elements:

- **Navigation intent** (from `scene_context.navigation_intent`)
- **Road geometry** (from `scene_context.road_geometry`)
- **Environmental conditions**

These provide background for the generator to ground answers but are NOT valid targets for causal questions. Only elements that a vision-language agent could visually identify and point to in the camera images are valid question targets.

### Preamble Rules (CRITICAL)

The scene-level preamble must contain ONLY two things:

1. **Navigation intent/command** — what the ego vehicle intends to do. E.g., "You intend to go straight through the roundabout."
2. **Current ego action** — the ego vehicle's immediate driving action, with NO explanation of why. E.g., "You are currently stopped." / "You are following traffic at low speed." / "You are changing lanes to the left."

**Do NOT include in the preamble:** road geometry, lane descriptions, element descriptions, signal states, spatial relationships, environmental conditions, or any scene details that describe what the VLA should be extracting from the images.

**For geometry-grounded binary AB questions:** Spatial detail that grounds the answer in scene-specific geometry goes in the per-question `spatial_context` field, NOT in the preamble. See Output Format for details.

---

## Language Rules (CRITICAL)

Your questions and answers will be presented to a vision-language agent that has NO knowledge of the scene graph.

1. **Never use graph terminology or internal pipeline names.** BANNED words: "node", "edge", "ego_edge", "inter_node_edge", "graph", "causal graph", "active cause", "latent", "dormant", "source", "target", "active edge", "dormant edge", "disconnected", "causal relationship", "causal structure", "collider", "confounding", "confounder", "chain", "direct cause", "indirect cause", "backdoor", "d-separation", "mediator", "mediation", "adjustment set", "conditioning", "subgraph", "fork", "explaining-away", "activation plausibility", "redundant effect", "effect type", "shielding node", "causal shielding", "structural decoupling", "exogenous variable", "abduction", "intervention", "counterfactual" (as technical terms — natural use like "if X had happened" is fine), "semi-Markovian", "bidirected", "endogenous", "SCM", "causal_status", "active_sustaining", "active_triggering", "CONFOUNDED_BY", "AFFECTS", "single-step reachability", "decision horizon", "Pearl", "PCH", "BEV", "bird's-eye view", "top-down frame", "state.json".
2. **Write natural, coherent English.** Paraphrase everything into how a human driver would describe the scene.
3. **Refer to scene elements by natural description** — e.g., "the red sedan ahead in your lane", "the pedestrian at the crosswalk on your right". Use the element's `label` field as a base but rephrase for naturalness.
4. **Question vs. Answer voice (CRITICAL):**
   - **Questions and preambles:** Use second-person ("you" / "your vehicle"). The question is posed BY a scene expert TO the ego driver. Example: "Is the parked sedan on your right currently affecting your driving?"
   - **Answers and reasoning:** Use first-person ("I" / "my" / "my vehicle"). The answer is FROM the ego driver's perspective. Example: "No, the parked sedan is not affecting my driving. It is fully within the shoulder and I am centered in my lane with no offset."
5. **Translate internal effect types** to natural driving language: CONSTRAINS_LONGITUDINAL → "restricts my speed", CONSTRAINS_LATERAL → "limits my lane positioning", INHIBITS_PROCEED → "blocks my path", GRANTS_PROCEED → "permits me to proceed", DEGRADES → "reduces my visibility or traction".
6. **Never use internal pipeline labels or template headers in reasoning text.** BANNED patterns: any "Label:" headers (e.g., "Hypothetical change:", "Conflict check:", "Shielding mechanism:", "Abduction step:"), any distractor type labels in parentheses (e.g., "(magnitude error)", "(salience trap)"), any classification labels (e.g., "Unique —", "Reinforcing —"). The reasoning must flow as natural prose.
7. **Use natural driving language for distances, speeds, and positions — not data metrics.** GOOD: "the truck stopped ahead", "close behind me". BAD: "at x=27.35", "traveling at 4.1 m/s", "12.8 meters ahead".

---

## Question Format

Every question must be **Binary (Yes/No)** or **MCQ (4 options preferred, 3 minimum, exactly one correct)**. No free-text answers.

**MCQ rule:** MCQ questions must present options with exactly one correct answer (4 preferred, 3 minimum). Frame the question naturally — do NOT append "Choose the correct option" or any formulaic suffix. The question should read like something a driving instructor would ask.

AB subtypes support both formats (marked "MCQ+Binary" below). For dual-format types:
- **Binary variant** is preferred when the scene supports trajectory generation — the binary answer directly determines whether a counterfactual trajectory differs from the observed one.
- **MCQ variant** is used when the selection task itself is the point (identifying which activation, which driving response).
- Both variants may be generated for the same element if the per-element limit allows.

---

## Question Type Definitions

For each type below, the **phrasing families** show the range of natural question styles. Mix and vary — never use the same phrasing pattern twice in a scene. Questions should read like a driving instructor or passenger asking about the scene.

### 1. DQ — Discrimination Questions (Rung 0)

Tests whether the model can distinguish "I see X" from "X is currently affecting my driving." Only for visually deceptive elements.

| Subtype | Format | What it tests |
|---|---|---|
| DQ-Binary | Binary | Does element X factor into the current driving situation? |
| DQ-Contrast | MCQ | What makes one element relevant and another not? |

DQ phrasings — vary across questions:
- DQ-Binary: "Does that large vehicle on the left play any role in how you're driving right now?" / "Is the construction equipment to your right factoring into your speed or path?" / "That crosswalk ahead is in your turn path. Do you need to account for it right now?"
- DQ-Contrast: "Both the stopped truck and the parked sedan are visible ahead. What makes one of them matter for your driving and not the other?"

### 2. EL — Explanation of Latency (Rung 1)

Tests whether the model can explain WHY an element is not affecting driving despite being visible. Only when the mechanism is non-obvious.

| Subtype | Format | What it tests |
|---|---|---|
| EL-Shield | MCQ | What's keeping element X from being a problem right now? |
| EL-Decouple | Binary | Is the proposed reason for irrelevance correct? |
| EL-Conditional | Binary | Does a shared property imply mutual influence? |

EL phrasings — vary across questions:
- EL-Shield: "The car crossing from the left looks like it could cut you off. What's keeping it from being a problem?" / "That truck seems close to your path. Why doesn't it matter right now?"
- EL-Decouple: "The taxi behind you is close. Is the reason it doesn't matter that you're heading away from it?" / "There's a cyclist on the far sidewalk. Is their irrelevance purely about the physical barrier between you?"
- EL-Conditional: "Both you and the sedan to your left are stopped. Does that mean the sedan is part of why you're stopped?" / "You and the opposing truck are both going slow. Are you slowing down because of each other?"

### 3. AB — Activation Boundary (Rung 2)

Tests whether the model can identify what would make a dormant element start mattering, and whether that would change driving behavior. Forward-looking interventional reasoning. **All binary AB must be geometry-grounded** — the answer must depend on scene-specific distances and margins.

**Activation realism constraint:** Activation scenarios must be physically plausible within 0.5s. A door cracking open, a pedestrian stepping off a curb, a vehicle beginning to drift — realistic. A stationary vehicle accelerating to highway speed — not.

| Subtype | Format | What it tests |
|---|---|---|
| AB-Element | MCQ+Binary | What change in element X would make it start mattering? |
| AB-Ego | MCQ+Binary | What change in ego's driving would make element X relevant? |
| AB-Context | MCQ+Binary | What change in a third element would cause element X to become relevant? |

AB phrasings — vary across questions:
- AB-Element Binary: "The van on the right is parked close to your lane. If its door cracked open partway, would you have room to continue without adjusting, given what's to your left?" / "That pedestrian is at the curb edge facing the road. If they took one step off the curb, how tight would things get for you at your current speed?"
- AB-Ego Binary: "If you decided to change into the left lane right now, would the car currently sitting in that lane become a problem?" / "If you sped up to pass the truck, would the gap ahead be tight enough that the oncoming vehicle would matter?"
- AB-Context Binary: "If the red light turned green and you started moving, would the pedestrian near the crosswalk ahead start to matter?" / "If the lead vehicle suddenly pulled away, would the construction narrowing ahead become your new constraint?"

---

## Step 1: Parse the Scene Graph

Extract the following — using ONLY these fields:

1. **Scene context** — from `scene_context.road_geometry` and `scene_context.navigation_intent`. Used internally for grounding. Not a question target.
2. **Ego state** — from `ego_state` fields: `speed_mps`, `inferred_acceleration`, `inferred_heading_change`, `behavior_explanation`.
3. **Observed ego trajectory** — from `ego_state.trajectory` or equivalent. Describe in natural language. REQUIRED in `spatial_context` for geometry-grounded binary AB questions.
4. **Currently influencing elements (active set)** — elements with at least one ego_edge whose `causal_status` is `"active_sustaining"` or `"active_triggering"`. Sub-classify each:
   - **Sustaining:** Live T=0 constraint. These are the primary shielding candidates for EL questions — they have active blocking effects right now.
   - **Triggering:** Retrospective observation-window dependency. These initiated ego's current committed maneuver but their live constraint may have expired. Use with caution as shielding candidates — a triggering element whose blocking effect has expired does NOT currently shield a dormant element.
   These form the contrast set (DQ), shielding candidates (EL), existing context (AB), and frozen context (geometry-grounded AB). Do not generate questions about these as targets.
5. **Dormant elements (question targets)** — elements with ego_edges whose `causal_status` is `"dormant"`. Record `effect_type`, `affected_state`, `justification` for each.
6. **Distractor elements** — elements in the `distractors` array. These have no ego_edges and are outside the causal graph. Do NOT generate questions about these (they belong to the distractor prompt).
7. **SKIP disconnected elements** — in `nodes[]` but not in `ego_edges[]` and not in `distractors[]`.
8. **Inter-element relationships** — from `inter_node_edges[]`. These have two relation types:
   - `"relation": "AFFECTS"` — directed causal edge (A causes B).
   - `"relation": "CONFOUNDED_BY"` — bidirected edge representing an unobserved common cause. The `inferred_cause` field describes the hidden shared cause. **CONFOUNDED_BY edges are NOT directed** — neither endpoint is a descendant of the other through this edge.
   Use inter-element relationships when both endpoints are relevant to the question (e.g., EL-Shield reasoning about how an active element blocks a dormant one, or EL-Conditional reasoning about shared causes).

---

## Step 2: Classify Elements for Question Generation

For each dormant element, assess three properties:

### 2a. Activation Plausibility

Assess using the element's `motion` fields, `position_relative_to_ego`, the ego edge `justification`, and ego's `behavior_explanation`. Cross-check against camera images — if the graph contradicts what you see, trust the images.

| Tier | Meaning | Relation to reachability |
|---|---|---|
| High | Small, realistic scene change activates it within 0.5s (mobile, near path, imminent condition in justification) | Well within single-step reachability bound |
| Medium | Plausible but requires moderate change (near-stationary at roadside, could drift toward path) | At the edge of single-step reachability |
| Low | Requires unlikely/large change (separated infrastructure, far from path, fully static) | Likely fails single-step reachability — flag if classified dormant |

**Note:** If a Low-plausibility element is classified as dormant in the graph, it may be a classification error (it might belong in the distractors array). Generate DQ/EL questions for it but skip AB, and note the discrepancy in your scratchpad.

### 2b. Latency Mechanism

| Mechanism | Meaning | EL subtype | Notes |
|---|---|---|---|
| Shielded | An active element prevents this dormant element from mattering | EL-Shield | Prefer active-sustaining elements as shields (live T=0 blocking). Active-triggering elements may shield only if their committed-maneuver effect is what blocks the dormant element. |
| Decoupled | No causal path due to direction/position (behind ego, different lane) | EL-Decouple | — |
| Correlated | Shares observable property with ego via common cause, not mutual influence | EL-Conditional | Check for CONFOUNDED_BY edges: if the dormant element and ego (or the dormant element and an active element) share a CONFOUNDED_BY edge, the `inferred_cause` names the unobserved common cause. This is a strong candidate for EL-Conditional — the shared property arises from the hidden common cause, not from mutual influence. |

Pick the primary mechanism. An element may have multiple.

### 2c. Trajectory Sensitivity

Would the answer to "if this element activated, would my driving change?" depend on specific distances and positions in this scene, or follow from generic policy? This determines the quality tier of AB questions.

| Sensitivity | Meaning | AB quality tier |
|---|---|---|
| trajectory_dependent | Answer requires scene-specific spatial reasoning (clearances, gaps, speeds) | Geometry-grounded — `spatial_context` must include trajectory |
| policy_dependent | Answer follows from generic driving rules regardless of geometry | Policy-level |

### Category Eligibility

| Category | Plausibility | Mechanism | Trajectory sensitivity |
|---|---|---|---|
| DQ (R0) | Any | Any | N/A |
| EL (R1) | Any (prioritize non-obvious) | Must have identifiable mechanism | N/A |
| AB (R2) | High or Medium | Any | Both tiers eligible; geometry-grounded binary requires High plausibility |

---

## Step 3: Internal Reasoning Procedures

These guide YOUR reasoning. Never expose these labels in output.

### 3a. EL Reasoning (Rung 1)

**EL-Shield:** (1) Identify the active element that blocks the dormant element's influence. Prefer active-sustaining elements — they have live T=0 constraints that directly prevent the dormant element from mattering. An active-triggering element may serve as a shield only if its committed-maneuver consequence (e.g., the lane offset or deceleration it caused) is what currently blocks the dormant element. (2) Trace: active element → ego state → why that state makes the dormant element irrelevant. (3) Verify: if the shielding element were absent, would the dormant element become relevant? If no, reclassify mechanism.

**EL-Decouple:** (1) Identify the structural reason (direction, lane, physical separation). (2) Verify no current ego action interacts with this element. (3) Contrast with what would need to be true for a causal path to exist.

**EL-Conditional:** (1) Identify the shared property between the dormant element and ego (or between the dormant element and an active element). (2) Identify the common cause. If a CONFOUNDED_BY edge exists involving this dormant element or a related active element, the `inferred_cause` field provides the unobserved common cause — use this as the explanation. (3) Verify: if the common cause were removed, would the correlation disappear? If yes, it is correlation, not causation.

### 3b. AB Reasoning (Rung 2) — Policy-Level

For AB questions where the answer follows from generic driving rules:

1. State all currently active influences in natural language, noting which are sustaining (live constraints) and which are triggering (committed-maneuver effects) — but do NOT use these labels in the output reasoning.
2. Describe the specific intervention (on element, ego, or third element). **The intervention must be physically plausible within the 0.5s decision horizon.** A door opening, a pedestrian stepping off a curb, a vehicle beginning to drift — these are realistic. A stationary vehicle accelerating to 60 km/h is not.
3. Classify: UNIQUE (new influence type) or REINFORCING (already provided by active element)
4. If conflict: apply priority order — (1) safety-critical yields override permissions, (2) regulatory controls override navigation, (3) physical constraints are absolute, (4) permissions are permissive not obligatory
5. Behavioral impact test: would the driver's action actually change? Use concrete driving actions: would I need to stop, yield, change lanes, swerve, brake, accelerate, or avoid? If reinforcing and already compliant, answer is "no change." **Important:** Maintained lateral position offset counts as a behavioral response. If ego is already offset from lane center because of an active-sustaining element, recognize that the offset IS a current trajectory response, not "no response." A dormant element activating to reinforce this offset does not change behavior; one that would require additional offset does.
6. State resolved behavior using specific driving action terms.

### 3c. AB Reasoning (Rung 2) — Geometry-Grounded

For AB questions where the answer depends on scene-specific spatial relationships. This procedure produces the highest-value AB questions.

1. **Spatial grounding:** From observed trajectory, establish ego's specific spatial relationships — clearances to the dormant element, speed, heading, available maneuvering room, positions of frozen context elements. These values ground the answer. Describe in natural language for the `spatial_context` field.
2. **Intervention:** Describe the specific activation of the dormant element. The activation must be physically plausible within 0.5s. Non-descendants remain FROZEN at observed values. **CONFOUNDED_BY-linked elements are NOT descendants** — they share an unobserved common cause and do not propagate interventions. Only follow directed AFFECTS edges for descendant identification.
3. **Trajectory re-derivation:** Given the frozen context + activated element + ego's specific spatial position, would ego need to change its driving action? Ground the answer in specific distances and gaps. "At my current speed and with only a narrow gap to the parked car, a door opening would require me to brake or steer left" — not "a door opening would generally require a response."
4. **Scene-specificity check:** Would this answer hold in any scene with the same element types but different geometry? If yes, this is a policy-level question, not geometry-grounded. Reclassify.

---

## Step 4: Answer Structure Requirements

Write every reasoning field in first-person as the ego driver explaining their driving to a passenger. Short, direct, grounded in what you see. Lead with the verdict. Name entities by appearance and position. One sentence per distractor for MCQ. No filler phrases, no hedging ("possibly", "might"), no em dashes or semicolons as separators.

Binary reasoning: 3–5 sentences. MCQ reasoning: 4–7 sentences including distractor explanations.

Pipeline terminology ("unique", "reinforcing", "shielded", "decoupled", "abduction step", distractor type labels, "sustaining", "triggering", "dormant", "geometry-grounded", "policy-level") belongs in meta ONLY, never in reasoning text.

Second-person voice ("you"/"your") is BANNED in reasoning — use first-person ("I"/"my"/"my vehicle") throughout.

### Reasoning Formats by Category

**DQ-Binary:** Where the element is, what it is doing, why it is not affecting me, what would need to change. GOOD: "The pedestrian is under the walkway on the right, behind the curb, standing still. They are not in or approaching my lane. If they stepped off the curb that would change, but right now they are not affecting my driving."

**EL-Shield (MCQ):** Name the threatening element and what makes it look relevant. Identify the active element that prevents it from mattering. Explain the mechanism. GOOD: "The car crossing the intersection from the left is moving fast across where I would travel. But I am held at the red light, so I am not entering the intersection. The red light is what makes the crossing car irrelevant to me right now. If the light were not there, that car would be a direct threat."

**EL-Decouple / EL-Conditional (Binary):** State the element and the proposed reason. Confirm or deny with structural/causal reasoning. GOOD (Decouple): "The taxi is behind me, stopped. Because I am heading forward, a vehicle behind me has no way to restrict my speed or path. It would only matter if I needed to reverse." GOOD (Conditional): "Both my vehicle and the sedan to the left are stopped. But we are both stopped because of the same red light, not because of each other. If the sedan were gone, I would still be stopped."

**AB-Element / AB-Ego / AB-Context (MCQ):** State current situation, describe the intervention, state the driving response, explain distractors. GOOD (AB-Element MCQ): "The parked car on the right shoulder is outside my lane. If its driver opened the door into my lane, I would need to steer left or slow down to avoid it. Right now no adjustment is needed. Option B is wrong because honking does not remove the obstruction. Option C is wrong because a full lane change is excessive."

**AB (Binary, geometry-grounded):** State the activation with geometry, ground the answer in specific distances and positions. The premise must be partial (drift, encroachment) not total (full lane blockage), so the answer depends on margins. GOOD: "I am traveling at moderate speed with about two car-lengths of clearance to the parked car. If its door cracked open partway into my lane, at that clearance and speed I could adjust slightly left without braking, but only because the adjacent lane is clear. If the adjacent lane were occupied, I would need to brake." BAD: "The parked car is outside my lane. If its door opened into my lane, I would need to steer or slow down." — This is policy-level (any driver gives this answer without images). Skip.

**AB-Context with conflict (MCQ):** State the existing active influence, describe the activation that conflicts, explain how the conflict resolves based on scene geometry. GOOD: "The green light permits me to proceed through the intersection. If the pedestrian on the corner stepped into the crosswalk, I would need to yield despite the green. At my current speed and distance from the crosswalk, I could brake in time. Option B is wrong because maintaining speed through a crosswalk with a pedestrian is not safe regardless of the signal."

---

## Step 5: Distractor Generation Rules (MCQ only)

Generate distractors plus 1 correct answer (4 options preferred, 3 minimum). Each distractor exploits a specific reasoning failure mode. No distractor should be eliminable by visual inspection alone.

### Distractor Composition Rules

**Preferred (4 distinct entities/outcomes available):** 4 options with one of each distractor type per the subtype table below, plus the correct answer.

**Fallback (fewer than 4 distinct options possible):** Reduce to 3 options. You may use two distractors of the same type if entity constraints make one type impossible. Never fabricate entities that are not visible in the scene. Prefer fewer clean options over forced, low-quality distractors.

### Distractor Types by Subtype

| Subtype | Type 1 | Type 2 | Type 3 |
|---|---|---|---|
| **DQ-Contrast** | **reversed_reasoning:** right factor, wrong element | **right_factor_wrong_conclusion:** real difference, wrong relevance conclusion | **surface_level:** vaguely true, too shallow to be the reason |
| **EL-Shield** | **wrong_shield:** real active element that doesn't block this specific dormant element | **causal_reversal:** says dormant element protects you from the active element | **correlation_as_cause:** says dormant element IS affecting you due to shared property |
| **AB-Element** | **impossible_activation:** physically implausible state change (violates 0.5s reachability) | **activation_without_consequence:** closer/more visible but no driving impact | **wrong_driving_response:** correct activation, wrong response |
| **AB-Ego** | **irrelevant_intent_change:** ego change that doesn't interact with this element | **reversed_dependency:** element affects ego without ego intent change | **wrong_mechanism:** correct intent change, wrong reason it matters |
| **AB-Context** | **wrong_third_element:** identifies wrong shielding element | **direct_instead_of_indirect:** element activates on its own, skipping context change | **reversed_cascade:** wrong direction of influence |

For geometry-grounded AB MCQs, additional distractor failure modes:
- **generic_policy_response:** correct under generic rules but ignores specific geometry
- **wrong_geometry:** right that trajectory changes but wrong spatial reasoning
- **ignores_frozen_context:** changes a non-descendant or CONFOUNDED_BY-linked element that should stay frozen

These may replace the standard types when the geometry-grounded variant produces more discriminative distractors.

### Distractor Quality Rules

1. Every distractor references something visually verifiable
2. Match specificity, length, and language register across all options
3. Never reuse distractor text across questions in the same scene
4. Each option references a distinct entity or describes a distinct driving outcome
5. Shuffle correct answer positions across A/B/C/D
6. **Pairwise distinctness:** for every pair of options, verify (1) different scene elements, (2) different driving consequences, (3) different practical conclusions. If any pair fails, rewrite one option.
7. **Physical plausibility of every option:** Each option's causal claim must be something the referenced element COULD realistically produce in some scene. If the claim is physically impossible for that element type, the option is eliminable by logic alone and must be rejected.

### Distractor Metadata

Tag each option in `meta.distractor_types` using the type vocabulary from the table above for that question's subtype. For 4-option MCQs, one of each type. For 3-option MCQs, note which types are present.

---

## Step 6: DQ-Contrast Pair Selection

A (influencing, not-influencing) pair earns a DQ-Contrast question ONLY if at least one holds: same element type, similar spatial region, or both visually prominent in the same camera view. Skip trivially obvious pairs.

---

## Step 7: Generation Rules

### DQ (Rung 0)
- One DQ-Binary per dormant element — **only when the element is visually deceptive** (it LOOKS like it could be active from the camera images, e.g., appears to be in the lane due to perspective, is large and prominent, or is moving in a way that suggests interaction). Skip for elements that are obviously irrelevant (empty crosswalk, distant parked car, element clearly in a different lane).
- One DQ-Contrast per pair meeting Step 6 criteria

### EL (Rung 1)
- One EL-Shield per shielded element — **only when identifying the shielding element requires looking at the scene** (e.g., it is non-obvious which active element is doing the shielding, or the dormant element looks threatening enough that the model must reason about why it doesn't matter). Skip when the mechanism is obvious from the question text alone.
- One EL-Decouple per structurally decoupled element — **only when the decoupling reason is non-obvious** and requires visual confirmation (e.g., the element appears to be in the path but is actually separated by a barrier). Skip trivially obvious cases (element clearly behind ego, element on a completely different road).
- One EL-Conditional where elements share properties via common cause (skip if no such pairs exist). CONFOUNDED_BY edges between a dormant element and another element provide strong candidates — the `inferred_cause` names the hidden common cause.

### AB (Rung 2)
- **AB Binary questions must be geometry-grounded.** The answer must depend on scene-specific distances, gaps, speeds, and available space — not generic driving rules. Policy-level AB questions (where any driver gives the same answer without seeing the scene) must be skipped entirely for binary format. `spatial_context` is required for every AB binary question.
- **AB-Element Binary** — one per High plausibility + trajectory-dependent element only. Must include observed trajectory in `spatial_context`. The activation premise must be **partial** (drift, encroachment, partial intrusion) not total (full lane entry), so the answer depends on available margins.
- **AB-Element MCQ** — one per element where a selection task adds value (multiple plausible activations exist). MCQ options describe specific trajectory outcomes grounded in scene geometry.
- **AB-Ego Binary** — one per element where ego intent change is the activation mechanism, and the answer depends on scene-specific geometry.
- **AB-Ego MCQ** — one per element where multiple ego changes could activate it.
- **AB-Context Binary** — one where a shielding active element exists whose change would activate the dormant element. The answer must depend on specific spatial relationships, not generic conflict-resolution rules.
- **AB-Context MCQ** — one where multiple context changes could activate, or where the conflict resolution requires selecting among responses.
- Activation scenarios must be physically plausible within 0.5s.
- For dual-format types: prefer binary when the scene supports trajectory generation AND the question is geometry-grounded. Generate MCQ additionally only if per-element limit allows and the selection task adds evaluation value.

### Limits
- Maximum **3 questions per element** across all categories
- Skip near-identical questions
- No AB for Low plausibility elements
- No geometry-grounded binary AB for Medium plausibility elements (require High)
- No questions about distractor-array elements, disconnected elements, or scene context
- **No Rung 3 questions** — this prompt covers Rungs 0, 1, 2 only

### Empty-Graph Fallback (no active elements in scene)

When the scene has **no active elements** (the active set is empty) but has dormant elements, generate one MCQ discrimination question:

**Question form:** "Looking at the scene around you, what's the main factor determining how you're driving right now?" with dormant elements as wrong options and **"Nothing visible here is currently affecting my driving"** (or natural variant) as the correct answer.

**Construction rules:**
- 3–4 options total. Wrong options are the most visually prominent dormant elements (high activation plausibility preferred — elements that LOOK like they could be relevant).
- The correct "none of the above" option should be phrased naturally — not literally "None of the above" but a driving-natural equivalent: "Nothing I can see here requires me to adjust" / "My current driving isn't being shaped by any of these" / similar.
- The question must pass all quality standards (Category Opacity, Visual Necessity, no self-answering).
- Tag the question with `"category": "DQ"`, `"subtype": "DQ-Empty"`, `"rung": 0`.
- In meta, set `"empty_graph_fallback": true`.

---

## What NOT to Generate

- Causal Ladder types (CaI, CB, CR, SC, NC, BAS, CDE, CA, AR, NIE, NDE) — those belong in the active-influences prompt
- Distractor question types (CI, NI, WC, DR, TWC) — those belong in the distractor prompt
- Questions treating dormant elements as active influences
- Rung 3 / counterfactual questions — dormant elements have no causal effect to reason counterfactually about
- AB questions where multiple elements change simultaneously
- AB questions with activation scenarios that violate the 0.5s physical plausibility constraint
- Questions about elements in the `distractors` array (handled by separate prompt), elements with no ego_edge, or scene context
- Template labels, pipeline terminology, or distractor type names in reasoning
- Questions about elements with `temporal_status: "passed_during_window"` — these are transient entities that have left the scene and belong to the distractor prompt's Temporal Wild Counterfactual questions

---

## Question Naturalness Standard (CRITICAL)

Questions will be jumbled across all three prompts (active, dormant, distractor) and presented to VLAs without category labels. The model must not be able to identify the question type from the question text alone.

### Category Opacity Principle

Every question must pass the **Category Opacity Test:** Could a reader identify the question type (DQ, EL, AB) or subtype from the question text alone? If yes → rephrase.

**BANNED question patterns:**
- "Is X currently affecting your driving?" → screams discrimination
- "What is preventing X from mattering?" → screams shielding
- "If X [activation], would you need to adjust your driving?" → screams activation boundary
- Any formulaic template applied verbatim across questions

**REQUIRED: Natural driving question framing.** Questions should sound like a driving instructor asking about the scene — specific, contextual, and woven with spatial detail.

### No Self-Answering Questions (CRITICAL)

The question must pose the decision, NOT supply the reasoning. If the question text contains the observation or causal logic that determines the answer, the VLA has nothing to extract from the images.

**The test:** Delete everything from the question except the core decision point. Does the deleted text contain the answer? If yes → rewrite to pose only the decision.

- BAD: "Is the crosswalk not a concern because the pedestrian is still on the sidewalk rather than on the crossing?" → answer embedded
- GOOD: "Does the crosswalk ahead factor into your turn right now?" → model must look to see if anyone is on it

### Phrasing Diversity

No two questions in the same scene should use the same sentence structure. Vary openings, element references, and how the scenario is framed. The phrasing families in the Question Type Definitions are starting points — blend and adapt.

### Mixed-Direction Answers

The same natural phrasing style should produce both Yes and No answers across the benchmark. Active elements (answer: Yes, it matters) and dormant elements (answer: No, it doesn't) should be asked with overlapping question styles so the answer cannot be predicted from the question format.

**Fixed-No types (DQ-Binary) must earn their place.** DQ-Binary answers are always No by design. A question is only worth generating if the No is genuinely hard — the element must LOOK active from the images (high visual salience, ambiguous perspective, motion that suggests interaction). If the element is obviously irrelevant, skip it.

**Variable types (AB) must include both Yes and No outcomes across a scene.** Do not let all AB questions default to Yes ("any intrusion requires a response"). Actively seek elements where the geometry means the answer is No — the clearance is large enough, the speed is low enough, or the available escape space is sufficient that a partial intrusion would NOT require adjustment. If every AB in a scene is Yes, revisit and either find a No case (element where the margins are generous enough that a partial activation doesn't matter) or drop the weakest Yes.

**EL direction balance.** EL-Decouple and EL-Conditional can produce both Yes and No answers. When constructing EL-Decouple questions, occasionally propose a WRONG reason (answer: No, that's not why it's irrelevant) to break the pattern of always-Yes.

---

## Question Difficulty Standard (CRITICAL — applies to ALL question types)

Every question in this benchmark must require visual scene understanding to answer correctly. A question that any competent driver can answer from the question text alone — without looking at the images — has zero evaluation value for a VLA benchmark.

### Visual Necessity Test

Before finalizing any question, apply this test: **Cover the images. Read only the preamble, question, and options. Can a competent driver answer correctly?** If yes → skip or redesign.

### Ambiguity-Without-Images Test

A well-designed question should be **ambiguous without the images**. A reasonable person reading just the question text should be genuinely uncertain whether the answer is Yes or No (for binary) or which option is correct (for MCQ). Only by examining the scene geometry can the answer be determined.

### Definiteness with Images

While the answer should be ambiguous from the question text alone, it must be **clearly and definitively answerable** once the model examines the scene images. If even with the images the answer is a coin flip, the question is too twisted. The scene geometry should resolve the ambiguity cleanly — hard to answer from text, easy to answer once you look.

### Per-Category Application

- **DQ-Binary:** Only generate for visually deceptive dormant elements — elements that LOOK active from camera images due to perspective, size, or motion. "Is the empty crosswalk affecting your driving?" is trivially No. "Is the large vehicle that appears to be encroaching from the adjacent lane affecting your driving?" requires visual verification.
- **EL:** Only generate when the mechanism is non-obvious from the question text. "Is the reason this crosswalk doesn't affect you that no one is on it?" is trivially Yes. "What is preventing the fast-approaching vehicle on the left from affecting you?" requires identifying the specific shielding element from the scene.
- **AB Binary:** Must be geometry-grounded. The activation premise must be partial (drift, encroachment, partial intrusion) so the answer depends on scene-specific margins. "If X entered your lane, would you brake?" is always Yes (skip). "If X drifted a half-lane toward you, would you have room to continue given what is to your left?" depends on the scene. For infrastructure elements (AB-Context involving signals), do not use total state flips (green→red). Use transitional states: "If the light turned amber right now, could you clear the intersection at this speed, or would you need to stop?"

---

## Output Format

Return a JSON object with a scene-level preamble and an array of questions:

```json
{
  "preamble": "<navigation intent + current ego action ONLY, in second-person. Same for all questions. No scene descriptions, no element details, no road geometry.>",
  "questions": [
    {
      "id": "DQ1",
      "rung": 0 | 1 | 2,
      "category": "DQ | EL | AB",
      "subtype": "DQ-Binary | DQ-Contrast | DQ-Empty | EL-Shield | EL-Decouple | EL-Conditional | AB-Element | AB-Ego | AB-Context",
      "difficulty": "easy | medium | hard",
      "spatial_context": "<for geometry-grounded binary AB: observed ego trajectory with spatial detail in second-person. null for all other questions.>",
      "question": "<natural language in second-person from scene expert's perspective; must sound like a natural driving question, not a formulaic exam item>",
      "answer_format": "binary | mcq",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "Yes | No | A | B | C | D",
      "reasoning": "<first-person ego driver reasoning — no template labels>",
      "activation_plausibility": "High | Medium | Low | N/A",
      "meta": {
        "referenced_elements": ["vehicle_1"],
        "causal_status": "dormant",
        "effect_types": ["CONSTRAINS_LATERAL"],
        "latency_mechanism": "shielded | decoupled | correlated | N/A",
        "shielding_element_status": "active_sustaining | active_triggering | N/A",
        "confounding_type": "observed | unobserved | null",
        "trajectory_sensitivity": "trajectory_dependent | policy_dependent | N/A",
        "influence_classification": "unique | reinforcing | conflict | N/A",
        "behavioral_impact": "stop | yield | lane_change | swerve | accelerate | decelerate | follow | avoid | no_change | N/A",
        "empty_graph_fallback": false,
        "distractor_types": { "A": "<type>", "B": "<type>", "C": "<type>", "D": "<type>" },
        "counterfactual_intervention": null
      }
    }
  ]
}
```

### Difficulty Field

The `difficulty` field is derived from the question's rung and quality tier:

| Condition | Difficulty |
|---|---|
| Rung 0 — DQ (all subtypes) | `"easy"` |
| Rung 1 — EL (all subtypes) | `"medium"` |
| Rung 2 — AB, policy-level | `"medium"` |
| Rung 2 — AB, geometry-grounded | `"hard"` |

### `spatial_context` Field

This per-question field provides spatial grounding for **geometry-grounded binary AB questions only**. It contains the observed ego trajectory and scene-specific spatial relationships needed to make the answer depend on this particular scene's geometry.

- **For geometry-grounded binary AB questions:** Describe ego's trajectory, speed, clearances, and positions of relevant frozen context elements in second-person natural language. E.g., "You are traveling at moderate speed with about two car-lengths of clearance to the parked car on your right. The adjacent lane to your left has a vehicle alongside you."
- **For all other questions:** `null`. The scene-level preamble provides sufficient context.

The `spatial_context` is presented to the VLA alongside the preamble and question, giving it the spatial grounding needed to reason about geometry-dependent activation scenarios.

### `meta.counterfactual_intervention`

This field provides metadata for **all binary AB questions** (AB-Element Binary, AB-Ego Binary, AB-Context Binary). It is `null` for DQ, EL, and all MCQ questions.

**Trajectory Generation Budget (CRITICAL):** This field is used downstream to generate alternative trajectories, which is expensive. Populate it for at most **1 best Rung 2 binary AB question** per scene. All other eligible binary AB questions get `counterfactual_intervention: null`. When multiple candidates exist, prefer: (1) `trajectory_changes: true` over `false`, (2) geometry-grounded over policy-level, (3) unique effects over reinforcing ones.

```json
"counterfactual_intervention": {
  "target_entity_id": "<graph node id of the dormant element being activated>",
  "target_entity_type": "infrastructure | agent",
  "operation": "activate | state_change",
  "operation_detail": "<natural language description of the activation, e.g., 'parked car door opens into ego lane', 'pedestrian steps off curb into crosswalk'>",
  "evaluation_mode": "planner | llm",
  "temporal_anchor": "t0",
  "frozen_entities": ["<IDs of entities held at observed values: all non-descendants + CONFOUNDED_BY-linked>"],
  "propagated_descendants": ["<IDs of entities that change as a consequence via directed AFFECTS edges>"],
  "trajectory_changes": true | false
}
```

**Field definitions:**

- `target_entity_id` / `target_entity_type`: The dormant element being activated. `"infrastructure"` = traffic lights, signs, barriers, lane markings. `"agent"` = vehicles, pedestrians, cyclists.
- `operation`: `"activate"` = dormant element transitions to an active state (door opens, pedestrian steps into road, vehicle drifts into lane). `"state_change"` = a third element changes to trigger the dormant element's activation (used for AB-Context; the target is the third element, not the dormant element itself).
- `operation_detail`: Natural language description of the specific activation. Must match what the question describes.
- `evaluation_mode`: Determines how the counterfactual is evaluated.
  - `"planner"` = re-run motion planner with modified scene. Default for `agent` entities being activated.
  - `"llm"` = requires reasoning about downstream effects that a planner cannot model. Default for `infrastructure` entities.
- `temporal_anchor`: Always `"t0"` for dormant elements — the activation is at the current moment, not retrospective.
- `frozen_entities`: IDs of entities held at their observed values during the activation. Includes all non-descendants of the activated element AND all entities linked via CONFOUNDED_BY edges. Must be consistent with the Descendant Propagation Rule in Step 3c.
- `propagated_descendants`: IDs of entities that change as a consequence of the activation, identified by tracing directed AFFECTS edges forward. Often empty for dormant activations (most dormant elements don't have downstream inter_node_edges).
- `trajectory_changes`: Whether the activation produces a trajectory that differs from the observed one. **Derived from `correct_answer`:**
  - Binary AB asking "would you need to adjust your driving?" → "Yes" = `true`, "No" = `false`
  - Binary AB asking "would [element] start affecting you?" → "Yes" = `true`, "No" = `false`

### Other field notes:
- `options` and `distractor_types`: MCQ only (omit for binary)
- `activation_plausibility`: "N/A" for DQ
- `latency_mechanism`: "N/A" for AB where mechanism is not the focus
- `shielding_element_status`: for EL-Shield questions only — records whether the shielding active element is sustaining or triggering. "N/A" for all other subtypes.
- `confounding_type`: for EL-Conditional questions or any question involving a CONFOUNDED_BY relationship. `"observed"` if the common cause is a named graph element, `"unobserved"` if from a CONFOUNDED_BY edge's `inferred_cause`. `null` otherwise.
- `trajectory_sensitivity`: "N/A" for DQ/EL. For AB: `"trajectory_dependent"` if the answer requires scene-specific geometry, `"policy_dependent"` if generic rules suffice.
- `influence_classification`: "N/A" for DQ/EL. For AB: `"unique"`, `"reinforcing"`, or `"conflict"`.
- `behavioral_impact`: "N/A" for DQ/EL. For AB: uses driving action vocabulary `stop | yield | lane_change | swerve | accelerate | decelerate | follow | avoid | no_change`.

---

## Scratchpad Requirement

Before generating JSON, produce:

```
SCRATCHPAD:

SCENE CONTEXT: Road: [...] | Navigation: [...] | Conditions: [...]

EGO TRAJECTORY (for geometry-grounded AB spatial_context): [natural language description]

ACTIVE ELEMENTS:
- [element -- effect -- sustaining/triggering]
(or "None")

DORMANT ELEMENTS (question targets):
- [element] | Effect: [...] | Plaus: [H/M/L] | Mechanism: [shielded/decoupled/correlated] | Shield: [which active element, sustaining/triggering] | Traj-sens: [traj_dep/policy_dep] | Eligible: [DQ,EL,AB] | AB tier: [policy/geometry/both]
- If Low plausibility: flag potential dormant/distractor misclassification

DISTRACTOR-ARRAY ELEMENTS (not question targets):
- [element -- rejection_reason]

CONFOUNDED_BY EDGES (relevant to dormant elements):
- [element A] <-> [element B] | inferred_cause: [...] | EL-Conditional candidate: [Y/N]

SKIPPED: [disconnected/context elements]

QUESTION PLAN: [one line per subtype: which elements, expected answers, binary/MCQ, policy/geometry-grounded]

DISTRACTOR PLAN: [for each MCQ: sketch 3 distractor types]

PAIRWISE CHECK: [confirm all MCQ option pairs are distinct]

ELEMENT COUNT: [verify max 3 per element]

END SCRATCHPAD
```

---

## Validation Checklist

**Scope & structure:**
- [ ] All question targets have `causal_status: "dormant"` — no questions about active, distractor, or disconnected elements
- [ ] Rung field matches category (DQ=0, EL=1, AB=2)
- [ ] No Rung 3 questions generated
- [ ] No AB for Low plausibility elements; no geometry-grounded binary AB for Medium plausibility
- [ ] All AB activation scenarios are physically plausible within 0.5s
- [ ] Max 3 questions per element; no near-duplicates
- [ ] If no active elements exist: DQ-Empty MCQ generated with "nothing is affecting me" as correct answer and dormant elements as wrong options

**Voice & language:**
- [ ] Preamble contains ONLY navigation intent + current ego action (no scene descriptions)
- [ ] `spatial_context` populated for geometry-grounded binary AB questions; `null` for all others
- [ ] Questions use second-person ("you"/"your vehicle")
- [ ] Answers and reasoning use first-person ("I"/"my"/"my vehicle")
- [ ] No banned words (including "BEV", "bird's-eye view", "top-down frame", "state.json", all graph/pipeline terminology)
- [ ] No raw IDs, effect type codes, or data metrics in question, option, or reasoning

**Answer correctness:**
- [ ] EL-Shield questions use active-sustaining elements as shields preferentially
- [ ] EL-Conditional questions reference unobserved common causes naturally (never mention edge types)
- [ ] AB behavioral impact uses concrete driving action terms (stop, yield, lane change, swerve, etc.)
- [ ] AB recognizes maintained lateral offset as a trajectory response
- [ ] Geometry-grounded AB answers depend on scene-specific geometry, not generic policy
- [ ] AB descendant propagation follows directed AFFECTS edges only, not CONFOUNDED_BY
- [ ] `difficulty` field set correctly per the mapping table

**Question difficulty & naturalness:**
- [ ] Every question passes the Visual Necessity Test (cannot be answered without images)
- [ ] Every question passes the Category Opacity Test (question type not identifiable from text)
- [ ] No self-answering questions (question text does not contain the observation or logic that determines the answer)
- [ ] No two questions use the same sentence structure
- [ ] No banned formulaic patterns
- [ ] DQ-Binary only for visually deceptive dormant elements (skip obviously irrelevant ones)
- [ ] EL only when mechanism identification requires looking at the scene
- [ ] All AB Binary questions are geometry-grounded with `spatial_context`; no policy-level AB Binary
- [ ] AB activation premises are partial (drift, encroachment), not total (full lane blockage)

**MCQ quality:**
- [ ] 4 options preferred, 3 minimum; 1 correct
- [ ] Pairwise distinct; correct positions distributed across A/B/C/D
- [ ] `counterfactual_intervention` populated for at most 1 best binary AB question; `null` for all others
- [ ] `trajectory_changes` consistent with `correct_answer`