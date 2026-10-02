#!/usr/bin/env python3
"""
CELLO-style Causal Question Generator with Deduplication
========================================================
Generates causal reasoning questions from scene graphs using:
1. Subgraph extraction (from subgraph_extraction.py)
2. Equivalence class deduplication
3. Importance scoring
4. Template-based question generation
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import itertools

# Import the subgraph extraction module
import subgraph_extraction


# ─── Readable names mapping ───
NODE_NAMES = {
    "ego": "the ego vehicle",
    "n_rb1": "the road boundary",
    "n_tc1": "the traffic control boundary",
    "n_tc2": "the VRU warning sign",
    "n_env": "the environmental conditions",
    "nav_cmd": "the navigation command",
}

EDGE_DESCRIPTIONS = {
    "BOUNDS_LATERALLY": "constrains lateral movement of",
    "LIMITS_SPEED": "imposes a speed limit on",
    "REQUIRES_YIELD": "requires a yield response from",
    "DEGRADES": "degrades the perception of",
    "COMMANDS": "commands the heading of",
    "REQUIRES_GAP_ACCEPTANCE": "requires gap acceptance from",
    "CAUSES_STATE_OF": "causes the presence/state of",
    "CONSTRAINS_RESPONSE": "constrains the movement response of",
}

# Effect types ranked by planning relevance
EFFECT_IMPORTANCE = {
    "REQUIRES_YIELD": 5,         # Direct safety constraint
    "COMMANDS": 4,                # Primary planning directive
    "REQUIRES_GAP_ACCEPTANCE": 4, # Key planning subtask
    "LIMITS_SPEED": 3,           # Speed constraint
    "BOUNDS_LATERALLY": 2,       # Geometric constraint
    "CONSTRAINS_RESPONSE": 3,    # Mediating constraint
    "CAUSES_STATE_OF": 3,        # Causal mechanism
    "DEGRADES": 1,               # Usually inactive
}


def n(node_id):
    """Get readable name for node."""
    # Handle pedestrian nodes specially
    if node_id.startswith("pedestrian_"):
        # Try to get label from node data if available
        return node_id.replace("_", " ")
    return NODE_NAMES.get(node_id, node_id)


def edge_desc(effect):
    """Get readable description for edge effect."""
    return EDGE_DESCRIPTIONS.get(effect, effect)


def build_adjacency_active(graph):
    """Build adjacency lists with metadata for active edges only."""
    children = defaultdict(set)
    parents = defaultdict(set)
    edge_meta = defaultdict(list)

    for e in graph.get("ego_edges", []):
        if not e.get("active", False):
            continue
        src, tgt = e["source"], e["target"]
        children[src].add(tgt)
        parents[tgt].add(src)
        edge_meta[(src, tgt)].append(e.get("effect_type", ""))

    for e in graph.get("inter_node_edges", []):
        # Inter-node edges don't have active flag, include all
        src, tgt = e["source"], e["target"]
        children[src].add(tgt)
        parents[tgt].add(src)
        edge_meta[(src, tgt)].append(e.get("relation", ""))

    return children, parents, edge_meta


def extract_active(graph, ego_id="ego"):
    """Extract active subgraphs with metadata."""
    children, parents, edge_meta = build_adjacency_active(graph)
    results = {"direct": [], "chain": [], "fork": [], "collider": []}

    # Direct: A -> ego
    for a in parents.get(ego_id, []):
        effects = edge_meta.get((a, ego_id), [])
        results["direct"].append({"A": a, "effects": effects})

    # Chain: A -> B -> ego
    for b in parents.get(ego_id, []):
        for a in parents.get(b, []):
            if a != ego_id:
                results["chain"].append({
                    "A": a, "B": b,
                    "AB_rel": edge_meta.get((a, b), []),
                    "B_ego_eff": edge_meta.get((b, ego_id), [])
                })

    # Fork: ego <- C -> B
    for c in parents.get(ego_id, []):
        for b in children.get(c, []):
            if b != ego_id:
                results["fork"].append({
                    "C": c, "B": b,
                    "C_ego_eff": edge_meta.get((c, ego_id), []),
                    "C_B_rel": edge_meta.get((c, b), [])
                })

    # Collider: A -> ego <- B
    ego_parents = list(parents.get(ego_id, []))
    for a, b in itertools.combinations(ego_parents, 2):
        results["collider"].append({
            "A": a, "B": b,
            "A_eff": edge_meta.get((a, ego_id), []),
            "B_eff": edge_meta.get((b, ego_id), [])
        })

    return results


def equiv_sig(graph_type, sg):
    """Produce a canonical signature that ignores specific node identity."""
    if graph_type == "direct":
        return ("direct", tuple(sorted(sg["effects"])))
    elif graph_type == "chain":
        return ("chain", tuple(sorted(sg["AB_rel"])), tuple(sorted(sg["B_ego_eff"])))
    elif graph_type == "fork":
        return ("fork", tuple(sorted(sg["C_ego_eff"])), tuple(sorted(sg["C_B_rel"])))
    elif graph_type == "collider":
        # Canonical: sort the two effect lists
        pair = tuple(sorted([tuple(sorted(sg["A_eff"])), tuple(sorted(sg["B_eff"]))]))
        return ("collider", pair)


def importance_score(graph_type, sg):
    """Calculate importance score for a subgraph."""
    score = 0
    if graph_type == "direct":
        for eff in sg["effects"]:
            score += EFFECT_IMPORTANCE.get(eff, 0)
    elif graph_type == "chain":
        for eff in sg["AB_rel"] + sg["B_ego_eff"]:
            score += EFFECT_IMPORTANCE.get(eff, 0)
        # Bonus for chains involving primary planning conflict
        if "REQUIRES_YIELD" in sg["B_ego_eff"]:
            score += 2
    elif graph_type == "fork":
        for eff in sg["C_ego_eff"] + sg["C_B_rel"]:
            score += EFFECT_IMPORTANCE.get(eff, 0)
    elif graph_type == "collider":
        for eff in sg["A_eff"] + sg["B_eff"]:
            score += EFFECT_IMPORTANCE.get(eff, 0)
        # Bonus for colliders capturing yield-vs-command tension
        all_effs = set(sg["A_eff"] + sg["B_eff"])
        if "REQUIRES_YIELD" in all_effs and ("COMMANDS" in all_effs or "REQUIRES_GAP_ACCEPTANCE" in all_effs):
            score += 5
        # Bonus for heterogeneous effect types
        if len(set(sg["A_eff"])) != len(set(sg["B_eff"])) or set(sg["A_eff"]) != set(sg["B_eff"]):
            score += 2
    return score


def node_set(graph_type, sg):
    """Return the set of non-ego nodes in this subgraph."""
    if graph_type == "direct":
        return frozenset([sg["A"]])
    elif graph_type == "chain":
        return frozenset([sg["A"], sg["B"]])
    elif graph_type == "fork":
        return frozenset([sg["C"], sg["B"]])
    elif graph_type == "collider":
        return frozenset([sg["A"], sg["B"]])


# ─── Question generation functions ───

def generate_direct_questions(sg):
    """Generate questions for direct subgraphs."""
    A = sg["A"]
    eff = sg["effects"][0] if sg["effects"] else "affects"
    questions = []

    # Rung 0 — Causality Identification
    questions.append({
        "task": "Causality Identification (CaI)",
        "rung": "0 - Discovery",
        "question": f"Which of the following is a direct cause of {n('ego')}'s current behavior?",
        "correct_answer": f"{n(A)}, because it {edge_desc(eff)} {n('ego')}.",
        "distractors": [
            "A vehicle in the adjacent lane (image distractor)",
            "The traffic signal at the next intersection (graph distractor)",
            "Road construction ahead (text distractor)"
        ],
        "format": "multiple_choice"
    })

    # Rung 3 — Counterfactual based on effect type
    if eff == "REQUIRES_YIELD":
        questions.append({
            "task": "Counterfactual Reasoning (CR)",
            "rung": "3 - Counterfactual",
            "question": f"If {n(A)} were not present near the road, would {n('ego')} still be decelerating?",
            "correct_answer": "Not necessarily — the yield requirement would be removed. Ego might still decelerate due to other active constraints (speed limit, other pedestrians).",
            "format": "binary+explanation"
        })
    elif eff == "LIMITS_SPEED":
        questions.append({
            "task": "Sufficient Cause (SC)",
            "rung": "3 - Counterfactual",
            "question": f"Is {n(A)} alone sufficient to cause {n('ego')} to reduce speed in this zone?",
            "correct_answer": "Yes — the speed limit is sufficient for deceleration independent of whether any pedestrians are actually present.",
            "format": "binary+explanation"
        })
    elif eff == "BOUNDS_LATERALLY":
        questions.append({
            "task": "Counterfactual Reasoning (CR)",
            "rung": "3 - Counterfactual",
            "question": f"If {n(A)} did not exist, would {n('ego')} still be constrained to its current lateral position?",
            "correct_answer": f"Ego would lose this lateral bound, potentially allowing a different turn trajectory. Other lateral constraints would still apply.",
            "format": "binary+explanation"
        })
    elif eff == "COMMANDS":
        questions.append({
            "task": "Counterfactual Reasoning (CR)",
            "rung": "3 - Counterfactual",
            "question": f"If {n(A)} had issued a 'continue straight' command instead of 'turn left', would {n('ego')} still need to decelerate?",
            "correct_answer": "Ego would not need gap acceptance for a left turn, but might still decelerate due to the speed limit and pedestrian yield requirements.",
            "format": "binary+explanation"
        })
    else:
        # Generic counterfactual
        questions.append({
            "task": "Counterfactual Reasoning (CR)",
            "rung": "3 - Counterfactual",
            "question": f"If {n(A)} were removed from the scene, would {n('ego')}'s behavior change?",
            "correct_answer": f"Yes — removing {n(A)} would eliminate the {eff} constraint on ego.",
            "format": "binary+explanation"
        })

    return questions


def generate_chain_questions(sg):
    """Generate questions for chain subgraphs."""
    A, B = sg["A"], sg["B"]
    ab_rel = sg["AB_rel"][0] if sg["AB_rel"] else "affects"
    b_ego_eff = sg["B_ego_eff"][0] if sg["B_ego_eff"] else "affects"
    questions = []

    # Rung 0 — Causal Attribution
    questions.append({
        "task": "Causal Attribution (CA)",
        "rung": "0 - Discovery",
        "question": f"What indirectly causes {n('ego')}'s response to {n(B)}?",
        "correct_answer": f"{n(A)}, because it {edge_desc(ab_rel)} {n(B)}, which in turn {edge_desc(b_ego_eff)} {n('ego')}.",
        "distractors": [
            f"The navigation command (graph distractor — affects ego but not through {n(B)})",
            "Another vehicle behind ego (image distractor)",
            "The weather conditions (text distractor — inactive edge)"
        ],
        "format": "multiple_choice"
    })

    # Rung 2 — Controlled Direct Effect
    questions.append({
        "task": "Controlled Direct Effect (CDE)",
        "rung": "2 - Intervention",
        "question": f"If {n(B)}'s position were held fixed and we changed {n(A)}, would {n('ego')}'s behavior change?",
        "correct_answer": f"No — the mediator ({n(B)}) is held constant, so changes to {n(A)} would not propagate to ego through this chain.",
        "format": "binary+explanation"
    })

    # Rung 3 — Natural Indirect Effect
    questions.append({
        "task": "Natural Indirect Effect (NIE)",
        "rung": "3 - Counterfactual",
        "question": f"If {n(A)} had a different configuration, would {n('ego')}'s response to {n(B)} change?",
        "correct_answer": f"Potentially — changing {n(A)} would alter {n(B)}'s state via {ab_rel}, which could change ego's required response.",
        "format": "binary+explanation"
    })

    return questions


def generate_fork_questions(sg, has_chain_overlap=False):
    """Generate questions for fork subgraphs."""
    C, B = sg["C"], sg["B"]
    c_ego_eff = sg["C_ego_eff"][0] if sg["C_ego_eff"] else "affects"
    c_b_rel = sg["C_B_rel"][0] if sg["C_B_rel"] else "affects"
    questions = []

    # Rung 0 — Causality Identification (confounder framing)
    questions.append({
        "task": "Causality Identification (CaI)",
        "rung": "0 - Discovery",
        "question": f"What common factor explains both {n('ego')}'s behavior and {n(B)}'s positioning?",
        "correct_answer": f"{n(C)} — it {edge_desc(c_ego_eff)} {n('ego')}, and simultaneously {edge_desc(c_b_rel)} {n(B)}.",
        "distractors": [
            f"{n(B)} directly causing ego's constraint (graph distractor — reverses causality)",
            "Other pedestrians in the scene (image distractor)",
            "Ego's own speed setting (text distractor)"
        ],
        "format": "multiple_choice"
    })

    # Rung 2 — Backdoor Adjustment Set
    questions.append({
        "task": "Backdoor Adjustment Set (BAS)",
        "rung": "2 - Intervention",
        "question": f"To assess the direct causal effect of {n(B)} on {n('ego')}, which variable must be controlled for?",
        "correct_answer": f"{n(C)} — controlling for it blocks the backdoor path ego ← {C} → {B} and isolates {n(B)}'s direct effect.",
        "distractors": [
            f"{n(B)} itself (incorrect — cannot adjust for the treatment)",
            "The navigation command (graph distractor — independent of this fork)",
            "No adjustment needed (incorrect — confounder bias)"
        ],
        "format": "multiple_choice"
    })

    # Rung 3 — Counterfactual (skip if overlaps with chain)
    if not has_chain_overlap:
        questions.append({
            "task": "Counterfactual Reasoning (CR)",
            "rung": "3 - Counterfactual",
            "question": f"If {n(C)} had a different configuration, would both {n('ego')}'s behavior and {n(B)}'s position change?",
            "correct_answer": f"Yes — as the common cause, altering {n(C)} affects both branches of the fork.",
            "format": "binary+explanation"
        })

    return questions


def generate_collider_questions(sg):
    """Generate questions for collider subgraphs."""
    A, B = sg["A"], sg["B"]
    a_eff = sg["A_eff"][0] if sg["A_eff"] else "affects"
    b_eff = sg["B_eff"][0] if sg["B_eff"] else "affects"
    questions = []

    # Rung 1 — Collider Bias
    questions.append({
        "task": "Collider Bias (CB)",
        "rung": "1 - Association",
        "question": f"Given that {n('ego')} is affected, does observing {n(A)}'s state tell us anything about {n(B)}'s state?",
        "correct_answer": f"Yes — conditioning on the collider (ego) induces a spurious association between {n(A)} and {n(B)} via explaining-away.",
        "format": "binary+explanation"
    })

    # Rung 3 — Necessary Cause
    # Pick the node with lower importance
    a_imp = sum(EFFECT_IMPORTANCE.get(e, 0) for e in sg["A_eff"])
    b_imp = sum(EFFECT_IMPORTANCE.get(e, 0) for e in sg["B_eff"])
    if a_imp <= b_imp:
        nc_node, nc_eff, other_node, other_eff = A, a_eff, B, b_eff
    else:
        nc_node, nc_eff, other_node, other_eff = B, b_eff, A, a_eff

    questions.append({
        "task": "Necessary Cause (NC)",
        "rung": "3 - Counterfactual",
        "question": f"If {n(nc_node)} were removed, would {n('ego')} still exhibit the same behavior?",
        "correct_answer": f"Partially — ego would lose the {nc_eff} constraint from {n(nc_node)}, but the {other_eff} constraint from {n(other_node)} remains.",
        "format": "binary+explanation"
    })

    return questions


def deduplicate_and_generate(raw_subgraphs):
    """Apply deduplication and generate questions."""
    # Group into equivalence classes
    equiv_classes = defaultdict(list)
    for gtype, subgraphs in raw_subgraphs.items():
        for sg in subgraphs:
            sig = equiv_sig(gtype, sg)
            score = importance_score(gtype, sg)
            equiv_classes[sig].append((score, gtype, sg))

    # Pick best representative from each class
    representatives = []
    for sig, members in equiv_classes.items():
        members.sort(key=lambda x: x[0], reverse=True)
        best_score, gtype, best_sg = members[0]
        representatives.append({
            "gtype": gtype,
            "sg": best_sg,
            "sig": sig,
            "score": best_score,
            "class_size": len(members)
        })

    # Sort by importance
    representatives.sort(key=lambda x: x["score"], reverse=True)

    # Detect chain/fork overlap
    chain_node_sets = set()
    for rep in representatives:
        if rep["gtype"] == "chain":
            chain_node_sets.add(node_set("chain", rep["sg"]))

    # Generate questions
    all_questions = []
    for rep in representatives:
        gtype = rep["gtype"]
        sg = rep["sg"]

        if gtype == "direct":
            qs = generate_direct_questions(sg)
        elif gtype == "chain":
            qs = generate_chain_questions(sg)
        elif gtype == "fork":
            overlap = node_set("fork", sg) in chain_node_sets
            qs = generate_fork_questions(sg, has_chain_overlap=overlap)
        elif gtype == "collider":
            qs = generate_collider_questions(sg)
        else:
            qs = []

        for q in qs:
            q["graph_type"] = gtype
            q["importance"] = rep["score"]
            q["equivalence_class_size"] = rep["class_size"]
            all_questions.append(q)

    return all_questions, representatives, equiv_classes


def main():
    parser = argparse.ArgumentParser(
        description="Generate deduplicated causal reasoning questions from scene graphs"
    )
    parser.add_argument(
        "input_file",
        help="Input JSON file with scene graph (e.g., scene_graphs/scene_080.json)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file for questions (default: input_name_questions.json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information about generation"
    )

    args = parser.parse_args()

    # Load scene graph
    try:
        with open(args.input_file, 'r') as f:
            scene_graph = json.load(f)
        print(f"✓ Loaded scene graph from: {args.input_file}")
    except Exception as e:
        print(f"✗ Error loading scene graph: {e}")
        return 1

    # Extract scene ID
    scene_id = scene_graph.get("scene_id", Path(args.input_file).stem)

    # Extract active subgraphs
    raw_subgraphs = extract_active(scene_graph)
    total_raw = sum(len(v) for v in raw_subgraphs.values())
    print(f"✓ Extracted {total_raw} raw subgraphs")

    # Generate deduplicated questions
    questions, representatives, equiv_classes = deduplicate_and_generate(raw_subgraphs)
    print(f"✓ Generated {len(questions)} questions from {len(equiv_classes)} equivalence classes")

    # Add metadata to questions
    for i, q in enumerate(questions):
        q["id"] = f"{scene_id}_q{i+1:03d}"
        q["scene_id"] = scene_id

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        base_name = Path(args.input_file).stem
        output_dir = Path(args.input_file).parent
        output_file = output_dir / f"{base_name}_questions.json"

    # Save questions
    try:
        with open(output_file, 'w') as f:
            json.dump(questions, f, indent=2)
        print(f"✓ Saved {len(questions)} questions to: {output_file}")
    except Exception as e:
        print(f"✗ Error saving questions: {e}")
        return 1

    # Print summary
    if args.verbose:
        print("\n" + "="*60)
        print(f"  QUESTION GENERATION SUMMARY")
        print("="*60)
        print(f"\n  Scene ID: {scene_id}")
        print(f"  Raw subgraphs: {total_raw}")
        print(f"  Equivalence classes: {len(equiv_classes)}")
        print(f"  Questions generated: {len(questions)}")
        print(f"  Reduction: {100 - round(100*len(questions)/max(total_raw*2, 1)):.0f}%")

        # Count by type
        rung_counts = defaultdict(int)
        task_counts = defaultdict(int)
        graph_counts = defaultdict(int)
        for q in questions:
            rung_counts[q["rung"]] += 1
            task_counts[q["task"].split("(")[0].strip()] += 1
            graph_counts[q["graph_type"]] += 1

        print(f"\n  By graph type:")
        for g, c in sorted(graph_counts.items()):
            print(f"    {g:<12} {c:>3}")
        print(f"\n  By rung:")
        for r, c in sorted(rung_counts.items()):
            print(f"    {r:<25} {c:>3}")
        print(f"\n  Top tasks:")
        for t, c in sorted(task_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {t:<35} {c:>3}")

        # Show top importance representatives
        print(f"\n  Top 5 important subgraphs:")
        for i, rep in enumerate(representatives[:5], 1):
            gtype = rep["gtype"]
            sg = rep["sg"]
            if gtype == "direct":
                sg_str = f"{sg['A']} → ego"
            elif gtype == "chain":
                sg_str = f"{sg['A']} → {sg['B']} → ego"
            elif gtype == "fork":
                sg_str = f"ego ← {sg['C']} → {sg['B']}"
            elif gtype == "collider":
                sg_str = f"{sg['A']} → ego ← {sg['B']}"
            else:
                sg_str = str(sg)
            print(f"    {i}. [{gtype}] {sg_str} (score={rep['score']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
