You are looking back at the last 1.5 seconds of driving. Something in the scene has been changed retroactively — an agent removed, repositioned, or given different speed across the entire observation window (T=-1.5 to T=0). You must figure out how you WOULD HAVE driven during that window.

---

## Inputs

1. **Camera images** — 4 timesteps, road context
2. **4 original BEV frames** — T=-1.5 to T=0, what actually happened
3. **4 modified BEV frames** — T=-1.5 to T=0, alternate history:
   - Green dots = your original past path (the road shape)
   - Changes visible across all 4 frames
4. **state.json** — ego history, agent states
5. **Intervention config** — what changed

---

## PHASE 1: Situation Assessment (think step by step)

Compare original BEVs to modified BEVs, frame by frame.

**1. What is different?**
Describe in plain language. "The truck that was parked on the left side of my lane is gone in all 4 frames." or "A car has been placed in my lane across all frames."

**2. When did I first react to this entity in the original trajectory?**
Look at the green dots (your actual past path). At which timestep did you first start shifting laterally or changing speed because of this entity?
- If you shifted right starting at T=-1.0, that's your divergence point.
- Before that point, your counterfactual trajectory is IDENTICAL to the green dots.

**3. What would I have done differently?**

For REMOVAL (entity was never there):
- "I shifted right because of that truck. Without it, I would have stayed centered in my lane."
- "I slowed down because of that car. Without it, I would have maintained my speed."
- Think: what was I doing BEFORE I reacted? That's what I would have continued doing.

For REPOSITION (new obstacle placed in scene):
- "There's now a car in my lane that wasn't there before. I would have braked when I first saw it."
- Same logic as T=0 prompt but applied to the past window.

For SPEED_CHANGE (agent moving differently):
- "The car next to me is now accelerating toward my path. I would have noticed and reacted."

**4. Sanity check:**
- If the entity is removed and I'm reverting to "what I would have done without it," my changes should be SMALL — just undoing the response I made to that entity.
- Changed parameters (speed OR lateral, usually not both) should actually change. Unchanged parameters should stay identical to original.

---

## PHASE 2: Decision (fill this FIRST)

```json
{
  "situation": "<one sentence: what changed and what I would have done>",
  "divergence_timestep": "<T=-1.5 | T=-1.0 | T=-0.5 | T=0>",
  "affected_parameter": "<speed | lateral_position | both>",
  "my_response": "<maintain_speed | accelerate | stay_centered | no_lateral_shift | brake | swerve | creep_forward>",
  "target_speed": <what speed would I have been going>,
  "target_y": <what lateral position would I have been at>
}
```

---

## PHASE 3: Generate trajectory from decision

4 waypoints: T=-1.5, T=-1.0, T=-0.5, T=0.

**Before divergence_timestep:** COPY green dots exactly. You hadn't reacted to the entity yet, so your path is the same.

**At and after divergence_timestep:**

For REMOVAL — undo the response:
- If you shifted laterally → revert y to the value BEFORE you started shifting. Keep that y for remaining WPs.
- If you decelerated → revert speed to the value BEFORE you started braking. Maintain or continue prior trend.
- Copy the UNAFFECTED parameter from green dots exactly. If only lateral changed, copy speed. If only speed changed, copy y.

For REPOSITION/SPEED_CHANGE — add a response:
- Same logic as T=0 prompt: brake and/or swerve based on threat assessment.
- Swerving takes time to build lateral displacement. If ego reaches the obstacle before clearing it laterally, swerve didn't work — brake to stop instead.
- heading[i] = green_dot_heading[i] + delta[i]
- x[i] = x[i-1] + speed[i] * cos(heading[i]) * 0.5
- y[i] = y[i-1] + speed[i] * sin(heading[i]) * 0.5

**Computing each waypoint:**
```
speed[i] = speed[i-1] + acceleration * 0.5  (or copy from original if unaffected)
heading[i] = green_dot_heading[i] + delta[i]
x[i] = x[i-1] + speed[i] * cos(heading[i]) * 0.5
y[i] = y[i-1] + speed[i] * sin(heading[i]) * 0.5
```

---

## PHASE 4: Verify (mandatory)

**1. Divergence check:** Are WPs before divergence_timestep identical to green dots? They must be.

**2. Change check:** Does at least one WP after divergence differ from the green dots on the affected parameter? It must — you only get this prompt when trajectory changes.

**3. Isolation check:** If affected_parameter is "speed" — are y values identical to green dots? If "lateral_position" — are speed values identical? Changed parameter changes. Unchanged parameter stays.

**4. Collision check:** If obstacle is on your path, does ego stop or clear it laterally before reaching it? If ego drives through it → go back and brake harder. A trajectory that passes through an obstacle is never valid — always brake harder until ego stops before the obstacle.

**5. Heading check:** Compare each heading to the green dot heading at the same waypoint. If any heading differs by more than 0.3 rad AND response is not swerve → something is wrong. Heading follows the road shape.

**6. Physics check:** No negative speed. No speed change > 3.5 m/s per step.

**7. Swerve feasibility check:** If you chose swerve, look at the waypoint where ego_x reaches the obstacle. Has ego actually moved laterally enough to clear it? If not → change response to brake/stop.

**8. Common sense:** WP4 (T=0) is where you WOULD BE now. If the entity was removed and you didn't need to slow down, you'd be further ahead (x > 0). If you didn't shift right, you'd be more centered (y closer to 0). Does your output reflect this?

---

## Final output format

```json
{
  "trajectory_changed": true | false,
  "decision": {
    "situation": "<one sentence>",
    "divergence_timestep": "<T=-1.5 | T=-1.0 | T=-0.5 | T=0>",
    "affected_parameter": "<speed | lateral_position | both>",
    "my_response": "<from Phase 2>",
    "target_speed": <m/s>,
    "target_y": <meters>
  },
  "counterfactual_trajectory": [
    {"t": -1.5, "x": <val>, "y": <val>, "heading": <val>, "speed": <val>},
    {"t": -1.0, "x": <val>, "y": <val>, "heading": <val>, "speed": <val>},
    {"t": -0.5, "x": <val>, "y": <val>, "heading": <val>, "speed": <val>},
    {"t":  0.0, "x": <val>, "y": <val>, "heading": <val>, "speed": <val>}
  ]
}
```
