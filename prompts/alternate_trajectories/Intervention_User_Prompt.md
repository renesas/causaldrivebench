=== STATE DATA ===
{state_json}

=== CAUSAL GRAPH ===
{causal_graph_json}

=== ORIGINAL EGO TRAJECTORY (2Hz, ego-frame, 8 waypoints) ===
WP1: [x, y, heading, speed]  (T+0.5s)
WP2: [x, y, heading, speed]  (T+1.0s)
WP3: [x, y, heading, speed]  (T+1.5s)
WP4: [x, y, heading, speed]  (T+2.0s)
WP5: [x, y, heading, speed]  (T+2.5s)
WP6: [x, y, heading, speed]  (T+3.0s)
WP7: [x, y, heading, speed]  (T+3.5s)
WP8: [x, y, heading, speed]  (T+4.0s)

=== NAVIGATION INTENT ===
{nav_command}

=== COUNTERFACTUAL Q&A ===
Question: {cf_question}
Answer: {cf_answer}
Reasoning: {cf_reasoning}

{intervention_images_section}

{intervention_generation_instruction}