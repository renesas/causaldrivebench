## Image Order

**Original scene images:**
{{IMAGE_ORDER_BLOCK}}

**Last 4 images: MODIFIED BEVs showing alternate history**
- Modified BEV at T=-1.5
- Modified BEV at T=-1.0
- Modified BEV at T=-0.5
- Modified BEV at T=0

On each modified BEV:
- Green dots/line = ego's original past trajectory (the road baseline)
- Red/magenta = intervention (agent removed / repositioned / altered speed positions)

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
The agent has been REMOVED from ALL 4 BEV frames. It was never there during the observation window. The green dots show where ego originally drove. How WOULD ego have driven from T=-1.5 to T=0 without this agent?
{{/if}}

{{#if REPOSITION}}
The agent has been REPOSITIONED in ALL 4 BEV frames. It was at its new location from T=-1.5 onward. The green dots show ego's original path. Scan all 4 frames — when does the repositioned agent first affect ego's path? How would ego have responded?
{{/if}}

{{#if SPEED_CHANGE}}
The agent was moving at a DIFFERENT SPEED from T=-1.5. LOOK at all 4 modified BEV frames — the agent is at different positions than the original because it was faster/slower. The green dots show ego's original path. Scan the 4 frames — at which frame does the agent's altered position first reach ego's vicinity? How would ego have responded?
{{/if}}

Generate ego's counterfactual past trajectory — 4 waypoints at T=-1.5, T=-1.0, T=-0.5, T=0. Return ONLY the JSON output specified in your instructions.
