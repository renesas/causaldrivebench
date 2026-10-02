# CausalDriveBench — Scene Graph Analysis Report
**Generated:** 2026-04-14 02:44 UTC  
**Data Root:** `/neurips-dataset/evaluation/sample_data`  
**Total scene graphs analysed:** 4  


## Table of Contents
1. [Dataset Overview](#1-dataset-overview)
2. [Unique Graph Topology Types](#2-unique-graph-topology-types)
3. [Causal Status Analysis](#3-causal-status-analysis)
4. [Node Composition & Similarity](#4-node-composition--similarity)
5. [Edge & Effect-Type Statistics](#5-edge--effect-type-statistics)
6. [Directional & Spatial Statistics](#6-directional--spatial-statistics)
7. [Ego State & Navigation Patterns](#7-ego-state--navigation-patterns)
8. [Distractor Statistics](#8-distractor-statistics)
9. [Scenario Complexity](#9-scenario-complexity)
10. [Inter-Node Relationship Patterns](#10-inter-node-relationship-patterns)
11. [Dataset-Level Token Usage](#11-dataset-level-token-usage)
12. [Key Autonomous Driving Insights](#12-key-autonomous-driving-insights)

## 1. Dataset Overview
The CausalDriveBench (CDB) dataset provides pruned causal scene graphs for autonomous driving scenarios sourced from three open-source datasets. Each SAMPLED instance represents one annotated moment in a scene, containing nodes (agents, traffic controls), directed ego-edges (causal links to ego behavior), inter-node edges (lateral relationships), and distractor nodes.

| Dataset | # Graphs | % of Total |
|---|---|---|
| `causal_argoverse2` | 1 | 25.0% |
| `causal_nuscenes` | 2 | 50.0% |
| `causal_openscene` | 1 | 25.0% |
| **Total** | **4** | 100% |

## 2. Unique Graph Topology Types
A graph's **topology signature** is defined solely by:
- The multiset of **node categories** present (excluding ego, normalised to 2-level hierarchy),
- The multiset of **(source category, effect_type)** pairs for ego-edges.

This makes the signature ID-independent — two graphs sharing the same set of agent types and causal edge patterns are considered the **same topology**, regardless of which specific pedestrian or vehicle IDs appear.

**Unique topologies found:** 4  
**Singleton topologies (appear once):** 4 (100.0% of types)  
**Recurring topologies:** 0  


### Top 15 Most Frequent Graph Topologies
| Rank | Hash | Frequency | % Graphs | Bar | Representative scene |
|---|---|---|---|---|---|
| 1 | `2886492d` | 1 | 25.0% | ████░░░░░░░░░░░ | `fusionopolis_crosswalk_stop` |
| 2 | `c4d16982` | 1 | 25.0% | ████░░░░░░░░░░░ | `fusionopolis_construction_zone` |
| 3 | `e2110c4c` | 1 | 25.0% | ████░░░░░░░░░░░ | `las_vegas_strip_straight` |
| 4 | `9c569169` | 1 | 25.0% | ████░░░░░░░░░░░ | `scene_001` |


### Topology Signature Details (Top 10)
The signature encodes which categories of nodes are in the graph and what causal effects they exert on ego.


#### Topology #1 — `2886492d` (1 graphs)
**Nodes (non-ego categories):** human.pedestrian, human.pedestrian, marking.crosswalk, sign.give_way, vehicle.car
**Ego-edge patterns:** human.pedestrian → INHIBITS_PROCEED; human.pedestrian → INHIBITS_PROCEED; marking.crosswalk → INHIBITS_PROCEED; sign.give_way → INHIBITS_PROCEED; vehicle.car → CONSTRAINS_LONGITUDINAL
**Example scenes:** `fusionopolis_crosswalk_stop`

#### Topology #2 — `c4d16982` (1 graphs)
**Nodes (non-ego categories):** human.pedestrian, human.pedestrian, marking.crosswalk, signal.green, vehicle.car
**Ego-edge patterns:** human.pedestrian → INHIBITS_PROCEED; human.pedestrian → INHIBITS_PROCEED; marking.crosswalk → INHIBITS_PROCEED; signal.green → GRANTS_PROCEED; vehicle.car → CONSTRAINS_LONGITUDINAL
**Example scenes:** `fusionopolis_construction_zone`

#### Topology #3 — `e2110c4c` (1 graphs)
**Nodes (non-ego categories):** vehicle.car
**Ego-edge patterns:** vehicle.car → CONSTRAINS_LONGITUDINAL
**Example scenes:** `las_vegas_strip_straight`

#### Topology #4 — `9c569169` (1 graphs)
**Nodes (non-ego categories):** marking.crosswalk, signal.yellow, vehicle.car
**Ego-edge patterns:** marking.crosswalk → INHIBITS_PROCEED; signal.yellow → INHIBITS_PROCEED; vehicle.car → CONSTRAINS_LONGITUDINAL
**Example scenes:** `scene_001`

## 3. Causal Status Analysis
Each ego-edge carries a `causal_status` that classifies **when and how** the causal link operates:

| Status | Meaning |
|---|---|
| `active_sustaining` | Actively constraining ego's current behaviour at T=0 |
| `active_triggering` | Previously triggered ego's current state but is now clearing |
| `dormant` | Structurally present but not currently constraining ego |


### Overall Causal Status Distribution
| Status | # Edges | % of All Ego-Edges | Bar |
|---|---|---|---|
| `active_sustaining` | 7 | 50.0% | ██████████░░░░░░░░░░ |
| `active_triggering` | 1 | 7.1% | █░░░░░░░░░░░░░░░░░░░ |
| `dormant` | 6 | 42.9% | █████████░░░░░░░░░░░ |
| **Total** | **14** | — | — |


### Graph-Level Active vs. Dormant Summary
| Metric | Count | % Graphs |
|---|---|---|
| Graphs with ≥1 active inhibiting edge | 3 | 75.0% |
| Graphs with >1 active inhibiting edge (multi-constraint) | 2 | 50.0% |
| Graphs with only dormant edges (no current constraint) | 1 | 25.0% |
| Graphs where ego is stopped (speed < 0.5 m/s) | 1 | 25.0% |

**Average active ego-edges per graph:** 2.0 (σ=1.581)
**Average dormant ego-edges per graph:** 1.5 (σ=0.5)


### Causal Status by Effect Type
This cross-tabulation reveals which effect types tend to be active vs. dormant.
> _Note: Joint (status × effect_type) statistics require per-edge detail. See per-graph cache for full edge-level data._

## 4. Node Composition & Similarity
Node composition describes **what agents and infrastructure** populate the causal graph. Understanding which node categories co-occur reveals recurring scenario archetypes.


### 4.1 Node Type Distribution
| Node Type | Total Instances | % of All Nodes |
|---|---|---|
| `OTHER_AGENT` | 8 | 44.4% |
| `TRAFFIC_CONTROL` | 6 | 33.3% |
| `EGO` | 4 | 22.2% |


### 4.2 Node Category Distribution (Top 20)
| Category | Total | % Instances | Bar |
|---|---|---|---|
| `human.pedestrian` | 4 | 28.6% | ██████░░░░░░░░░░░░░░ |
| `vehicle.car` | 4 | 28.6% | ██████░░░░░░░░░░░░░░ |
| `marking.crosswalk` | 3 | 21.4% | ████░░░░░░░░░░░░░░░░ |
| `sign.give_way` | 1 | 7.1% | █░░░░░░░░░░░░░░░░░░░ |
| `signal.green` | 1 | 7.1% | █░░░░░░░░░░░░░░░░░░░ |
| `signal.yellow` | 1 | 7.1% | █░░░░░░░░░░░░░░░░░░░ |


### 4.3 Average Graph Composition
| Metric | Min | Max | Mean | Median | Std Dev |
|---|---|---|---|---|---|
| Non-ego nodes per graph | 1 | 5 | 3.5 | 4.0 | 1.658 |
| Ego-edges per graph | 1 | 5 | 3.5 | 4.0 | 1.658 |
| Inter-node edges per graph | 0 | 5 | 2.25 | 2.0 | 1.92 |
| Distractor nodes per graph | 1 | 11 | 4.5 | 3.0 | 3.905 |


### 4.4 Presence Flags — Common Scene Archetypes
Percentage of graphs containing specific node types or patterns:

| Scene Archetype | # Graphs | % Graphs | Bar |
|---|---|---|---|
| Has at least one pedestrian agent | 2 | 50.0% | ██████████░░░░░░░░░░ |
| Has at least one vehicle agent (non-ego) | 4 | 100.0% | ████████████████████ |
| Has at least one traffic signal | 2 | 50.0% | ██████████░░░░░░░░░░ |
| Has at least one crosswalk marking | 3 | 75.0% | ███████████████░░░░░ |
| Has at least one road sign | 1 | 25.0% | █████░░░░░░░░░░░░░░░ |
| Construction zone detected (any node or distractor) | 2 | 50.0% | ██████████░░░░░░░░░░ |
| Has CONFOUNDED_BY inter-node edge | 2 | 50.0% | ██████████░░░░░░░░░░ |
| Multiple simultaneous active INHIBITS_PROCEED edges | 2 | 50.0% | ██████████░░░░░░░░░░ |

## 5. Edge & Effect-Type Statistics
The **effect type** on an ego-edge defines the causal mechanism: whether a node prevents ego from proceeding, restricts its longitudinal freedom, or grants permission.


### 5.1 Ego-Edge Effect Type Distribution
| Effect Type | # Edges | % of Ego-Edges | Bar |
|---|---|---|---|
| `INHIBITS_PROCEED` | 9 | 64.3% | █████████████░░░░░░░ |
| `CONSTRAINS_LONGITUDINAL` | 4 | 28.6% | ██████░░░░░░░░░░░░░░ |
| `GRANTS_PROCEED` | 1 | 7.1% | █░░░░░░░░░░░░░░░░░░░ |


### 5.2 Affected State Variables
What aspects of ego's motion are affected by causal edges?
| State Variable | # Edge References | % of References |
|---|---|---|
| `speed` | 14 | 36.8% |
| `acceleration` | 14 | 36.8% |
| `heading` | 10 | 26.3% |


### 5.3 Active vs. Dormant Edges by Effect Type
Understanding which effect types remain dormant helps us identify latent risks in the driving scene — constraints that could become active with small state changes.

> _Per-effect active/dormant cross-tabulation requires per-edge data stored in the cache._

## 6. Directional & Spatial Statistics
These statistics characterise **where agents are** relative to ego and **how they move**.


### 6.1 Agent Positions Relative to Ego
| Position | # Agent Mentions | % of Mentions | Bar |
|---|---|---|---|
| `ahead` | 11 | 50.0% | ██████████░░░░░░░░░░ |
| `right` | 4 | 18.2% | ████░░░░░░░░░░░░░░░░ |
| `behind` | 3 | 13.6% | ███░░░░░░░░░░░░░░░░░ |
| `left` | 2 | 9.1% | ██░░░░░░░░░░░░░░░░░░ |
| `center` | 2 | 9.1% | ██░░░░░░░░░░░░░░░░░░ |


### 6.2 Agent Motion Direction
| Motion Direction | # Agents | % of Agents | Bar |
|---|---|---|---|
| `n/a` | 6 | 42.9% | █████████░░░░░░░░░░░ |
| `same_direction` | 4 | 28.6% | ██████░░░░░░░░░░░░░░ |
| `crossing` | 3 | 21.4% | ████░░░░░░░░░░░░░░░░ |
| `moving_away` | 1 | 7.1% | █░░░░░░░░░░░░░░░░░░░ |


### 6.3 Static vs. Moving Agents
| Motion State | # Agents | % of Agents | Bar |
|---|---|---|---|
| `static` | 8 | 57.1% | ███████████░░░░░░░░░ |
| `moving` | 6 | 42.9% | █████████░░░░░░░░░░░ |


### 6.4 Agent Relative Speed
| Relative Speed | # Agents | % of Agents | Bar |
|---|---|---|---|
| `n/a` | 6 | 42.9% | █████████░░░░░░░░░░░ |
| `slow` | 5 | 35.7% | ███████░░░░░░░░░░░░░ |
| `stationary` | 2 | 14.3% | ███░░░░░░░░░░░░░░░░░ |
| `moderate` | 1 | 7.1% | █░░░░░░░░░░░░░░░░░░░ |


### 6.5 Agent Visibility — Camera Coverage
| Camera / Sensor | # Agent Mentions | % of Mentions | Bar |
|---|---|---|---|
| `cam_front` | 11 | 29.7% | ██████░░░░░░░░░░░░░░ |
| `bev` | 11 | 29.7% | ██████░░░░░░░░░░░░░░ |
| `state` | 8 | 21.6% | ████░░░░░░░░░░░░░░░░ |
| `cam_back` | 3 | 8.1% | ██░░░░░░░░░░░░░░░░░░ |
| `cam_front_right` | 2 | 5.4% | █░░░░░░░░░░░░░░░░░░░ |
| `cam_back_left` | 1 | 2.7% | █░░░░░░░░░░░░░░░░░░░ |
| `cam_back_right` | 1 | 2.7% | █░░░░░░░░░░░░░░░░░░░ |

## 7. Ego State & Navigation Patterns
The ego state captures the autonomous vehicle's kinematic situation at T=0.


### 7.1 Ego Speed Distribution
| Statistic | Value (m/s) |
|---|---|
| Min | 0.002 |
| Max | 10.824 |
| Mean | 4.734 |
| Median | 4.054 |
| Std | 4.621 |


#### Ego Speed Histogram (m/s)
| Speed Range (m/s) | Count | % Graphs | Bar |
|---|---|---|---|
| 0–1 | 2 | 50.0% | ██████████░░░░░░░░░░ |
| 1–3 | 0 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| 3–7 | 0 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| 7–12 | 2 | 50.0% | ██████████░░░░░░░░░░ |
| 12–18 | 0 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| 18–30 | 0 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| 30+ | 0 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |


### 7.2 Ego Acceleration Pattern
| Acceleration | # Graphs | % Graphs | Bar |
|---|---|---|---|
| `constant` | 2 | 50.0% | ██████████░░░░░░░░░░ |
| `accelerating` | 1 | 25.0% | █████░░░░░░░░░░░░░░░ |
| `decelerating` | 1 | 25.0% | █████░░░░░░░░░░░░░░░ |


### 7.3 Ego Heading Change
| Heading Change | # Graphs | % Graphs | Bar |
|---|---|---|---|
| `straight` | 4 | 100.0% | ████████████████████ |


### 7.4 Navigation Intent
| Navigation Intent | # Graphs | % Graphs | Bar |
|---|---|---|---|
| `straight` | 3 | 75.0% | ███████████████░░░░░ |
| `stopped` | 1 | 25.0% | █████░░░░░░░░░░░░░░░ |

**Behavior matches navigation intent:** 4 / 4 graphs (100.0%) — high agreement indicates the graphs are well-annotated and the ego behavior is self-consistent.

## 8. Distractor Statistics
**Distractors** are scene elements present in the raw data that the causal graph algorithm explicitly excluded because they do not causally affect ego's behavior. Their analysis reveals the range of 'irrelevant' agents in real driving and the reasoning patterns of the causal pruning algorithm.

**Total distractor instances across all graphs:** 18
**Average distractors per graph:** 4.5 (σ=3.905)
**Max distractors in a single graph:** 11


### 8.1 Distractor Category Distribution
| Category | # Distractors | % of All Distractors | Bar |
|---|---|---|---|
| `human.pedestrian` | 12 | 66.7% | █████████████░░░░░░░ |
| `vehicle.car` | 4 | 22.2% | ████░░░░░░░░░░░░░░░░ |
| `vehicle.truck` | 1 | 5.6% | █░░░░░░░░░░░░░░░░░░░ |
| `vehicle.trailer` | 1 | 5.6% | █░░░░░░░░░░░░░░░░░░░ |


### 8.2 Distractor Rejection Reasons
Why were these nodes excluded from the causal graph?
| Rejection Reason | # Distractors | % of Distractors | Bar |
|---|---|---|---|
| `sidewalk_separated` | 11 | 32.4% | ██████░░░░░░░░░░░░░░ |
| `behind_ego` | 7 | 20.6% | ████░░░░░░░░░░░░░░░░ |
| `adjacent_lane` | 6 | 17.6% | ████░░░░░░░░░░░░░░░░ |
| `not_on_path` | 6 | 17.6% | ████░░░░░░░░░░░░░░░░ |
| `opposing_direction` | 2 | 5.9% | █░░░░░░░░░░░░░░░░░░░ |
| `far_distance` | 2 | 5.9% | █░░░░░░░░░░░░░░░░░░░ |


### 8.3 Distractor-to-Active Ratio
**Total active (non-ego) graph nodes:** 14  
**Total distractor nodes:** 18  
**Distractor ratio:** 1.29 distractors per active node  

A high ratio indicates scenes with many scene elements that the causal graph correctly prunes as irrelevant — typical in crowded urban environments like Las Vegas Strip.

## 9. Scenario Complexity
The **complexity score** is a composite heuristic:

```
complexity = 3×(active INHIBITS_PROCEED edges)
           + 2×(active CONSTRAINS edges)
           + (non-ego node count)
           + 0.5×(distractor count)
           + 0.5×(inter-node edge count)
```

Higher scores indicate more challenging scenarios with multiple simultaneous constraints.

| Statistic | Value |
|---|---|
| Min | 6.5 |
| Max | 17.5 |
| Mean | 12.125 |
| Median | 12.25 |
| Std | 4.709 |


### Complexity Distribution
| Complexity Band | # Graphs | % Graphs | Bar |
|---|---|---|---|
| 0–3 (simple) | 0 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| 3–6 (low) | 0 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| 6–10 (medium) | 2 | 50.0% | ██████████░░░░░░░░░░ |
| 10–15 (high) | 0 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |
| 15–20 (very high) | 2 | 50.0% | ██████████░░░░░░░░░░ |
| 20+ (extreme) | 0 | 0.0% | ░░░░░░░░░░░░░░░░░░░░ |


### Top 10 Most Complex Graphs
| Rank | Scene | Sample | Dataset | Score | Active Inhibit | Nodes |
|---|---|---|---|---|---|---|
| 1 | `fusionopolis_construction_zone` | SAMPLED_1 | causal_nuscenes | 17.5 | 3 | 5 |
| 2 | `fusionopolis_crosswalk_stop` | SAMPLED_0 | causal_nuscenes | 16.0 | 3 | 5 |
| 3 | `scene_001` | SAMPLED_2 | causal_argoverse2 | 8.5 | 1 | 3 |
| 4 | `las_vegas_strip_straight` | SAMPLED_2 | causal_openscene | 6.5 | 0 | 1 |

## 10. Inter-Node Relationship Patterns
Inter-node edges encode lateral causal relationships between non-ego agents — for instance, a traffic signal governing a vehicle, or two pedestrians appearing simultaneously due to shared confounding (e.g., a pedestrian signal phase).


### 10.1 Inter-Node Relation Type Distribution
| Relation Type | # Edges | % of Inter-Node Edges | Bar |
|---|---|---|---|
| `AFFECTS` | 7 | 77.8% | ████████████████░░░░ |
| `CONFOUNDED_BY` | 2 | 22.2% | ████░░░░░░░░░░░░░░░░ |


### 10.2 Graph Layer Distribution
The `layer` field in inter-node edges encodes depth in the causal chain (Layer 2 = direct influence on ego-relevant agent, Layer 3 = two hops, etc.).
| Layer | # Edges | % of Inter-Node Edges |
|---|---|---|
| 2 | 6 | 66.7% |
| 3 | 1 | 11.1% |
| N/A | 2 | 22.2% |


### 10.3 CONFOUNDED_BY Patterns
**2** graphs (50.0%) contain at least one `CONFOUNDED_BY` edge, indicating co-present agents explained by a common unobserved cause (e.g., shared pedestrian signal phase). These are important for causal inference accuracy — confounded links must not be mistaken for direct causation.

## 11. Dataset-Level Token Usage
Each graph.json records the LLM token usage from the causal graph generation step. This reveals the annotation cost and the correlation between scene complexity and LLM compute.

| Token Type | Total | Mean/graph | Median/graph | Max/graph |
|---|---|---|---|---|
| Prompt tokens | 205179 | 51294.75 | 50295.5 | 58622 |
| Completion tokens | 20527 | 5131.75 | 4038.0 | 8829 |
| Total tokens | 225706 | 56426.5 | 54225.5 | 67451 |

## 12. Key Autonomous Driving Insights
This section synthesises the statistical findings into actionable insights for autonomous driving research and benchmark design.

### 12.1 Pedestrian-Dominated Constraint Scenarios
Pedestrians appear in **50.0%** of graphs and are the primary source of `INHIBITS_PROCEED` edges. Combined with crosswalk presence (75.0% of graphs), this confirms that pedestrian-at-crosswalk scenarios are the dominant 'hard stop' archetype in the dataset. The ego vehicle is fully stopped (< 0.5 m/s) in 25.0% of scenes, and most of these are pedestrian-driven stops.

### 12.2 Multi-Constraint Scenarios as Benchmark Difficulty Signals
**50.0%** of graphs have more than one simultaneously active `INHIBITS_PROCEED` edge. These represent **compound constraints** — e.g., a pedestrian on a crosswalk that is itself governed by a traffic signal. Such scenarios are significantly harder for VLMs because the model must reason about multiple independent causal chains simultaneously. The QA ladder structure targets exactly this difficulty.

### 12.3 Dormant Edges as Latent Risk Indicators
**25.0%** of graphs contain only dormant (non-active) causal constraints. These scenes look simple on the surface but contain latent risks: a following vehicle that constrains braking freedom, or an unoccupied crosswalk that could be activated by a pedestrian. The `dormant_qa` questions in the benchmark explicitly probe whether models can reason about these counterfactual risks.

### 12.4 Traffic Signal Complexity
Traffic signals appear in **50.0%** of graphs. Yellow signals (transition state) are particularly challenging because they require the model to reason about whether to stop or proceed — a key decision that depends on speed, distance, and reaction time. Green signal + occupied crosswalk (as seen in the construction zone samples) creates a conflict scenario that tests priority reasoning.

### 12.5 Construction Zones as High-Distractor Scenarios
Construction zone indicators appear in **50.0%** of graphs. These scenes consistently show the highest distractor counts because construction environments have many workers, barriers, and equipment that are spatially close but causally irrelevant to ego. The distractor pruning algorithm correctly identifies these, but VLMs may incorrectly attribute causal relevance to them.

### 12.6 CONFOUNDED_BY as a Causal Reasoning Challenge
**50.0%** of graphs contain `CONFOUNDED_BY` inter-node edges. These represent cases where two agents appear to affect each other but are actually co-caused by an unobserved variable (shared signal phase, construction activity). This is a foundational challenge in causal inference: models must distinguish correlation from causation, which is exactly what the CDB benchmark tests.

### 12.7 Graph Topology Diversity as Dataset Coverage Metric
With **4** unique graph topologies across 4 graphs, the dataset has a topology diversity ratio of **1.00**. A ratio close to 1.0 indicates high scenario diversity (every scene is unique), while lower ratios indicate recurring scenario patterns. The top recurring topologies reveal the most common real-world driving scenario templates and can guide targeted data collection.

### 12.8 Directional Bias in Causal Agents
**50.0%** of agents are positioned _ahead_ of ego vs. **13.6%** _behind_. This strong forward bias is expected: the causal pruning algorithm specifically filters agents that cannot affect ego's forward motion. Agents behind ego are typically retained only when they constrain braking (CONSTRAINS_LONGITUDINAL) or represent a structural follower risk.

### 12.9 Recommended Benchmark Splits Based on Graph Properties
Based on the analysis, we recommend the following stratification axes for train/val/test splits:

| Split Criterion | Rationale |
|---|---|
| Ego speed range | Tests whether models reason differently for stopped vs. moving ego |
| Active inhibit count (1 vs. 2+) | Isolates single-constraint vs. compound-constraint difficulty |
| Presence of CONFOUNDED_BY | Tests causal vs. correlational reasoning |
| Graph topology hash | Ensures topology diversity across splits |
| Dataset source | Ensures cross-dataset generalisation |
| Navigation intent | Tests intent-behavior alignment reasoning |