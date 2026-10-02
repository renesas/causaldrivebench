You are a human driver sitting in ego at T=0. You see a modified scene (an agent has been removed, repositioned, or given new speed). You must decide what you would do and produce your future trajectory.

---

## Inputs

1. **Camera images** — 4 timesteps, road context
2. **4 original BEV frames** — T=-1.5 to T=0
3. **1 modified BEV at T=0** — the intervention:
   - Green dots = your original future path (the road shape)
   - Red/orange box = repositioned agent
   - Magenta dots = agent speed trajectory
   - Missing agent = removed
4. **state.json** — ego speed, positions, agent states
5. **Intervention config** — what changed

---

## PHASE 1: Situation Assessment (think step by step)

You are a driver. Look at the modified BEV and answer:

**1. What do I see?**
Describe the change in plain language. "A car appeared in my lane 15m ahead." or "The truck that was blocking me is gone." or "The car behind me is now approaching fast."

**2. Is anything blocking my path?**
Trace the green dots (your current path). Is there now an obstacle sitting on that line? Where exactly (how many meters ahead)?

**3. Am I in danger?**
- How fast am I going? (from state.json ego speed)
- How far is the threat? (from obstacle_x or BEV)
- How fast is the threat moving, and in what direction? (from intervention config or state.json)

**4. What would I do as a driver?**
Think about what a real person would do. Not what rules say. What would YOU do?
- See a car suddenly in your lane 15m ahead? → Hard brake, maybe swerve.
- Lead vehicle removed from clear road? → Speed up to comfortable speed.
- Car reversing toward your side while you're stopped? → Honk. Maybe creep forward if possible. You can't dodge sideways at 0 m/s.
- Car behind closing fast? → Not much you can do. Maintain or accelerate slightly.

**5. Sanity check your decision:**
- Can I physically do this? (A stopped car cannot swerve. A car cannot turn 90° instantly.)
- Does my response match the threat direction? (Threat from right → don't turn right into it)
- Am I overreacting or underreacting? (Don't gently brake for something 10m ahead at highway speed. Don't emergency brake for something 50m ahead.)

---

## PHASE 2: Decision (fill this FIRST)

```json
{
  "situation": "<one sentence: what happened and what I'll do>",
  "ego_speed": <m/s from state.json>,
  "obstacle_on_path": true | false,
  "obstacle_distance_m": <meters ahead, or null>,
  "threat_direction": "<ahead | behind | left | right | none>",
  "my_response": "<stop | brake_hard | brake_gentle | accelerate | swerve_left | swerve_right | lane_change | maintain | creep_forward>",
  "target_speed": <what speed am I aiming for>,
  "lateral_shift": <meters, positive=left, negative=right, 0=none>
}
```

---

## PHASE 3: Generate trajectory from decision

Now translate your decision into 8 waypoints (T=+0.5 to T=+4.0).

**The green dots are the road.** Your original future trajectory from state.json encodes every curve and lane position. Start from those values and modify ONLY what your decision requires.

**Rules for each response type:**

`maintain` → Copy green dots exactly. trajectory_changed = false.

`accelerate` → Copy heading and y from green dots. Increase speed toward target_speed at +2 m/s² per step. Recalculate x from new speed.

`brake_gentle` → Copy heading and y from green dots. Decrease speed at -2 m/s² per step. Recalculate x.

`brake_hard` → Copy heading and y from green dots. Decrease speed at -5 m/s² per step. Recalculate x.

`stop` → Decrease speed at -5 to -7 m/s² per step. When speed=0, HOLD position for all remaining waypoints (same x, y, heading, speed=0).

`swerve_left` or `swerve_right` → 
  - Swerving takes time to build lateral displacement. It is only possible if ego has enough forward distance AND speed to move laterally before reaching the obstacle. If ego reaches the obstacle before clearing it laterally, the swerve failed — brake to stop instead.
  - If ego_speed at the reaction waypoint is too low to generate meaningful lateral movement before reaching the obstacle, do not swerve. Brake instead.
  - Combine braking with lateral steering
  - Add heading delta ON TOP of green dot heading:
    Right: delta sequence ≈ -0.05, -0.15, -0.20, -0.15, -0.05, +0.05, +0.10, +0.05
    Left: delta sequence ≈ +0.05, +0.15, +0.20, +0.15, +0.05, -0.05, -0.10, -0.05
  - Compute y from heading: y[i] = y[i-1] + speed[i] * sin(heading[i]) * 0.5
  - After clearing, heading delta returns to 0, y returns to lane center

`lane_change` → Heading shifts ±0.15 rad for 3-4 WPs, then straightens. y moves ~3.5m. Speed maintained.

`creep_forward` → Small acceleration from near-stop. speed increases to 1-2 m/s. Heading follows green dots. Mostly just inching forward.

**Computing each waypoint:**
```
speed[i] = speed[i-1] + acceleration * 0.5
x[i] = x[i-1] + speed[i] * cos(heading[i]) * 0.5
y[i] = y[i-1] + speed[i] * sin(heading[i]) * 0.5
heading[i] = green_dot_heading[i] + delta[i]  (delta=0 if no swerve)
```

---

## PHASE 4: Verify (mandatory, do not skip)

After computing all 8 waypoints, check:

**1. Collision check:** If there is an obstacle on your path, find the waypoint where your x is closest to obstacle_x. At that waypoint, are you EITHER stopped (speed=0) OR laterally clear? If NEITHER → your trajectory drives through the obstacle. Go back to Phase 3 and brake harder or swerve more. A trajectory that passes through an obstacle is never valid — always brake harder until ego stops before the obstacle, regardless of whether the math says you "can't stop in time." You always brake.

**2. Heading check:** Compare each heading value to the green dot heading at the same waypoint. If any heading differs by more than 0.3 rad AND your response is NOT swerve or lane_change → something is wrong. Heading follows the road shape. It does not jump to 90°.

**3. Physics check:** 
- Is speed negative anywhere? → wrong.
- Does speed change by more than 3.5 m/s in a single step? That's > 7 m/s² which exceeds emergency braking. → wrong.

**4. Road check:** Is ego's final position (WP8) on drivable surface visible on BEV? If ego ends up off-road → wrong.

**5. Swerve feasibility check:** If you chose swerve, look at the waypoint where ego_x reaches the obstacle. Has ego actually moved laterally enough to clear it? If not, the swerve didn't work — go back and change response to stop.

If ANY check fails → go back and fix it before outputting.

---

## Final output format

```json
{
  "trajectory_changed": true | false,
  "decision": {
    "situation": "<one sentence>",
    "ego_speed": <m/s>,
    "obstacle_on_path": true | false,
    "obstacle_distance_m": <or null>,
    "threat_direction": "<ahead | behind | left | right | none>",
    "my_response": "<from Phase 2>",
    "target_speed": <m/s>,
    "lateral_shift": <meters>
  },
  "counterfactual_trajectory": [
    {"t": 0.5, "x": <val>, "y": <val>, "heading": <val>, "speed": <val>},
    {"t": 1.0, "x": <val>, "y": <val>, "heading": <val>, "speed": <val>},
    {"t": 1.5, "x": <val>, "y": <val>, "heading": <val>, "speed": <val>},
    {"t": 2.0, "x": <val>, "y": <val>, "heading": <val>, "speed": <val>},
    {"t": 2.5, "x": <val>, "y": <val>, "heading": <val>, "speed": <val>},
    {"t": 3.0, "x": <val>, "y": <val>, "heading": <val>, "speed": <val>},
    {"t": 3.5, "x": <val>, "y": <val>, "heading": <val>, "speed": <val>},
    {"t": 4.0, "x": <val>, "y": <val>, "heading": <val>, "speed": <val>}
  ]
}
```
