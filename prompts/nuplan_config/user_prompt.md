Given the following counterfactual question and scene data, generate the agent intervention config for the simulator.

<COUNTERFACTUAL_QA>
Question: {cf_question}
Answer: {cf_answer}
Reasoning: {cf_reasoning}
Metadata: {cf_metadata_json}
</COUNTERFACTUAL_QA>

<STATE_JSON>
{state_json}
</STATE_JSON>

<CAUSAL_GRAPH>
{scene_graph_json}
</CAUSAL_GRAPH>

Images provided in order (grouped by timestep):
{image_ordering}

Generate the intervention config. For removals, determine `apply_at_t` from the temporal anchor. For trajectory changes, compute waypoints from the agent's current state in state.json to the target state described in the question. Cross-check against BEV frames and cameras for spatial plausibility. Return ONLY valid JSON.