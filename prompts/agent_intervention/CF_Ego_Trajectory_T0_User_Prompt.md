## Image Order

**Original scene images:**
{{IMAGE_ORDER_BLOCK}}

**Last image: MODIFIED BEV at T=0**
- Green dots/line = ego's original future trajectory (the road baseline)
- Red/magenta = intervention (agent removed / repositioned / speed trajectory dots)

---

## Scene Data

**Navigation command:** {{NAV_COMMAND}}

**state.json:**
```json
{{STATE_JSON}}
```

## Counterfactual Question

{{QUESTION}}

## Intervention Config

```json
{{INTERVENTION_CONFIG}}
```

## Intervention Type: {{ACTION_TYPE}}

{{#if REMOVE}}
The agent has been REMOVED from the scene. LOOK at the modified BEV — the agent is gone. The green dots show where ego originally drove. How does ego drive now that this agent is absent?
{{/if}}

{{#if REPOSITION}}
The agent has been REPOSITIONED. LOOK at the modified BEV — the red/orange box shows where the agent IS now at T=0. The green dots show ego's original path. Is the agent on the green dot path? If yes, how does ego respond?
{{/if}}

{{#if SPEED_CHANGE}}
The agent's SPEED has changed. LOOK at the modified BEV — the magenta dots show where the agent WILL BE at T=+0.5, T=+1.0, ... at its new speed. The green dots show ego's original path. Do the magenta dots intersect or approach the green dots? If yes, at which waypoint? How does ego respond?
{{/if}}

Generate ego's counterfactual trajectory — 8 waypoints from T=+0.5 to T=+4.0. Return ONLY the JSON output specified in your instructions.
