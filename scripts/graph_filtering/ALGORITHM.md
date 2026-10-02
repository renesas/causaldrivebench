# Scenario Grouping: Hierarchical Multi-Modal Graph Clustering

## What This Does

Given ~3000 causal scene graphs from autonomous driving, group similar scenes together so we can:
- See how many unique scenario types exist in our dataset
- Pick representative samples from each group
- Identify over-represented patterns (e.g. 500 copies of "car ahead, ego slows down")
- Ensure the final dataset has balanced coverage across all scenario types

## How It Works (3 Levels)

The grouping is **coarse-to-fine**:

```
Level 1: Graph Family     → What kind of causal pattern is this?
Level 2: Scene + Intent   → What road type and what is ego doing?
Level 3: Fine Clustering  → Within a family+scene bucket, which samples are truly redundant?
```

Each level narrows down. By the time we reach Level 3, we're only comparing scenes that already share the same causal pattern AND the same road/intent type.

---

## Level 1: Graph Family

Every causal graph gets classified into a **family** based on four properties:

### 1a. Topology — What shape is the graph?

Looks at the structural patterns (from active subgraph extraction). These are the four causal graph topologies from Pearl's SCM framework:

| Topology | Meaning | Example |
|----------|---------|---------|
| `direct` | A → ego (single direct cause) | car → ego (CONSTRAINS_LONG) |
| `chain` | A → B → ego (mediated cause) | signal stops car, car stops ego |
| `collider` | A → ego ← B (independent causes converging) | truck blocks + pedestrian crosses, independently |
| `confounding` | A ← C → B, both affect ego (CONFOUNDED_BY edge) | shared signal timing causes both ego stop and lead car stop |

### 1b. Status — Active, dormant, or mixed?

| Status | Meaning |
|--------|---------|
| `active_only` | All ego-edges are active (sustaining or triggering) |
| `dormant_only` | All ego-edges are dormant |
| `mixed_status` | Some active, some dormant |
| `no_ego_edges` | No ego-edges at all |

### 1c. Effect — What kind of causal influence?

Based on the `effect_type` of ego-edges:

| Effect | Meaning |
|--------|---------|
| `longitudinal` | Only CONSTRAINS_LONGITUDINAL |
| `lateral` | Only CONSTRAINS_LATERAL |
| `inhibits` | Only INHIBITS_PROCEED |
| `grants` | Only GRANTS_PROCEED |
| `degrades` | Only DEGRADES |
| `longitudinal_mixed` | Primarily longitudinal + others |
| `inhibits_mixed` | Primarily inhibits + others |

### 1d. Source — What type of agent is causing the constraint?

Looks at which node categories are sources of ego-edges:

| Source | Meaning |
|--------|---------|
| `car` | Ego-edges come from cars |
| `truck` | From trucks/trailers |
| `bus` | From buses |
| `pedestrian` | From pedestrians |
| `cyclist` | From cyclists/motorcycles |
| `signal` | From traffic signals |
| `crosswalk` | From crosswalk markings |
| `obstacle` | From cones/barriers/construction |
| `mixed_vehicle` | Multiple vehicle types |
| `mixed_context` | Mix of different source types |

### Family Label

These four combine into a single family label:

```
"parallel | mixed_status | longitudinal_mixed | truck"
"single_edge | active_only | inhibits | pedestrian"
"inter_fork | active_only | grants | signal"
```

Scenes in the same family have the same **causal pattern type**. A scene with "car following ego" and a scene with "pedestrian blocking ego" are in different families.

---

## Level 2: Scene + Intent Bucket

Within each graph family, scenes are split further by:

### 2a. Road Type

Extracted from `scene_context.road_geometry` text:

| Road | Keywords matched |
|------|-----------------|
| `intersection` | "intersection", "junction" |
| `roundabout` | "roundabout" |
| `parking_service` | "parking", "service area" |
| `highway` | "highway", "expressway" |
| `multi_lane_road` | "divided road", "multi-lane" |
| `road` | Everything else |

### 2b. Navigation Command

Directly from `nav_command`: `GO STRAIGHT`, `TURN LEFT`, `TURN RIGHT`

### 2c. Ego Heading

From `inferred_heading_change`: `straight`, `turning_left`, `turning_right`

### Bucket Label

```
"intersection | TURN LEFT | turning_left"
"multi_lane_road | GO STRAIGHT | straight"
```

So within a family like "parallel | active_only | longitudinal | car", we might have buckets for "intersection turning left" and "highway going straight" — these are clearly different driving scenarios even though the causal pattern is the same.

---

## Level 3: Fine Clustering (Complete-Link Agglomerative)

Within each family+bucket, some samples may still be near-duplicates. Level 3 uses the **robust pairwise score** to cluster them.

### The Robust Pairwise Score

For every pair of samples that reach Level 3, we compute a score that says "how redundant are these two?"

#### Step 1: Hard Guards

Binary vetoes — if ANY is true, score = 0 (never redundant):

- Node count differs by more than `max_node_diff` (default: 2)
- Active edge count differs by more than `max_active_edge_diff` (default: 2)
- Dormant edge count differs by more than `max_dormant_edge_diff` (default: 2)
- Inter-node edge count differs by more than `max_inter_edge_diff` (default: 2)
- Active effect Jaccard < threshold (when `active_profile_mode = "hard"`)
- Active source Jaccard < threshold (when `active_profile_mode = "hard"`)
- Status profile Jaccard < threshold
- Distractor count differs by more than threshold (when `use_distractor_guards = True`)
- Distractor type Jaccard < threshold (when `use_distractor_guards = True`)

#### Step 2: Conjunctive Score (Geometric Mean)

For pairs that pass the guards, compute how similar they are across 5 (or 7) modality kernels:

```
margin_road      = max(0, (K_road[i,j]      - threshold_road)      / (1 - threshold_road))
margin_behavior  = max(0, (K_behavior[i,j]  - threshold_behavior)  / (1 - threshold_behavior))
margin_structure = max(0, (K_structure[i,j]  - threshold_structure) / (1 - threshold_structure))
margin_semantic  = max(0, (K_semantic[i,j]   - threshold_semantic)  / (1 - threshold_semantic))
margin_inter     = max(0, (K_inter[i,j]      - threshold_inter)    / (1 - threshold_inter))
```

If `active_profile_mode = "soft"`, also add:
```
margin_active_effect = max(0, (active_effect_sim - threshold) / (1 - threshold))
margin_active_source = max(0, (active_source_sim - threshold) / (1 - threshold))
```

Then: `score = geometric_mean(all_margins)`

Properties:
- If ANY margin is 0, score is 0 (one bad dimension kills it)
- If all margins are barely positive, score is tiny
- Only strong agreement across ALL dimensions gives a high score

#### Step 3: Mutual Top-k

Even with a positive score, only call it a real redundancy if both sides agree. For each sample, find its top-k most similar neighbors. A pair (i, j) becomes an edge only if:
- `score(i, j) >= min_redundancy_score`
- j is in i's top-k
- i is in j's top-k

#### Step 4: Complete-Link Clustering

Using the score matrix from above, apply `AgglomerativeClustering` with complete linkage:

```python
AgglomerativeClustering(
    n_clusters=None,
    metric="precomputed",
    linkage="complete",
    distance_threshold=1 - cluster_score_threshold,
)
```

Complete linkage means: two groups only merge if ALL pairs between them are similar enough. This prevents the chaining problem where A≈B and B≈C causes A and C to merge even though they're different.

### Result

Each family+bucket gets split into fine groups. Each group contains samples that are truly redundant — same graph family, same road+intent, AND high pairwise similarity across all modalities.

---

## The 5 Kernels

### K_road — Road Layout Similarity

Embeds `scene_context.road_geometry` + `navigation_intent` using a sentence transformer model. Cosine similarity between embeddings.

**Captures:** Road type, lane configuration, intersection geometry, traffic direction.

### K_behavior — Ego Behavior Similarity

Embeds `ego_state.behavior_explanation` + ego speed/heading/nav + active edge justifications (with agent IDs stripped). Cosine similarity.

**Captures:** Why ego is doing what it's doing. "Stopped at red light" vs "slowing for truck ahead."

### K_structure — Graph Topology Similarity (WL Kernel)

Weisfeiler-Lehman subtree kernel with intra-cluster semantic labels.

**Node labels:** Grouped by type (EGO, VEHICLE, LARGE_VEHICLE, VRU, SIGNAL, SIGN, CROSSWALK, OBSTACLE, GATE), then sub-clustered within each group by embedding the `type:category:label` text.

**Edge labels:** Grouped by `causal_status + effect_type` (e.g. `active_CONSTRAINS_LONGITUDINAL`, `dormant_INHIBITS_PROCEED`), then sub-clustered within each group by embedding the justification text.

**Captures:** Graph topology + semantic meaning of nodes and edges. Two graphs with different numbers of nodes, different edge types, or different neighborhood patterns get low similarity.

### K_semantic — Agent Composition Similarity

Feature vector with binary flags (has_pedestrian, has_cyclist, has_truck, has_bus, has_signal, has_crosswalk, has_construction) + normalized counts (active_edges, dormant_edges, inter_edges). Cosine similarity.

**Captures:** Quick coarse check — is this a pedestrian scene or a vehicle scene?

### K_inter — Inter-Node Interaction Similarity

Embeds the inter-node edge structure (AFFECTS/CONFOUNDED_BY relationships between non-ego nodes). Cosine similarity.

**Captures:** Multi-agent interaction patterns. A scene where "signal controls pedestrian" has different inter-node structure than "truck blocks cyclist."

---

## Configuration

From `scenario_grouping_report.py`, all parameters are environment-variable overridable:

```python
GROUP_ROAD_THRESHOLD = 0.84          # K_road margin threshold
GROUP_BEHAVIOR_THRESHOLD = 0.80      # K_behavior margin threshold  
GROUP_STRUCTURE_THRESHOLD = 0.50     # K_structure margin threshold
GROUP_SEMANTIC_THRESHOLD = 0.55      # K_semantic margin threshold
GROUP_INTER_THRESHOLD = 0.45         # K_inter margin threshold
GROUP_ACTIVE_EFFECT_THRESHOLD = 0.45 # Active effect Jaccard threshold
GROUP_ACTIVE_SOURCE_THRESHOLD = 0.45 # Active source Jaccard threshold
GROUP_TOP_K = 14                     # Mutual top-k neighbors
GROUP_MIN_SCORE = 0.06               # Minimum conjunctive score for edge
GROUP_CLUSTER_SCORE_THRESHOLD = 0.36 # Complete-link clustering threshold
GROUP_ACTIVE_PROFILE_MODE = "soft"   # "hard" (guard) or "soft" (margin)
GROUP_MAX_NODE_DIFF = 3              # Hard guard: max node count difference
GROUP_MAX_ACTIVE_EDGE_DIFF = 3       # Hard guard: max active edge count difference
GROUP_MAX_DORMANT_EDGE_DIFF = 3      # Hard guard: max dormant edge count difference
GROUP_MAX_INTER_EDGE_DIFF = 3        # Hard guard: max inter-edge count difference
GROUP_DISTRACTOR_COUNT_THRESHOLD = 5 # Hard guard: max distractor count difference
GROUP_DISTRACTOR_JACCARD_THRESHOLD = 0.30  # Hard guard: min distractor type Jaccard
GROUP_STATUS_PROFILE_JACCARD_THRESHOLD = 0.30  # Hard guard: min status profile Jaccard
```

Note: for grouping, the thresholds are **looser** than for filtering (we want to find groups, not eliminate). The `scenario_grouping_report.py` uses `active_profile_mode = "soft"` and `use_distractor_guards = False` by default — more permissive grouping.

---

## Output

The HTML report shows:

1. **Summary:** Total families, groups, multi-sample groups, singletons, largest group
2. **Family table:** Each family with its graph type label, sample count, bucket count, group count
3. **Per-family detail:** Expandable sections showing:
   - Each scene+intent bucket
   - Each fine cluster within that bucket
   - Leader sample (highest weight/rarity)
   - All member samples with clickable links to scene viewer
   - Group affinity (density and mean score)
4. **Singletons:** Samples with no match — unique scenarios

---

## Files

| File | Purpose |
|------|---------|
| `scripts/graph_filtering/scenario_grouping_report.py` | Main script: loads data, builds hierarchy, renders HTML |
| `scripts/graph_filtering/robust_multimodal_filter.py` | Core algorithm: hard guards, conjunctive scoring, MWIS |
| `scripts/graph_filtering/filter_diverse_graphs.py` | Feature extraction (`extract_features`, `SampleRecord`) |
| `scripts/graph_filtering/ALGORITHM.md` | This document |
| `tmp/render_comparison_v4.py` | Kernel builders (road, behavior, structure, semantic, interaction) |
| `tmp/.cache/scenario_grouping_kernels_*.npz` | Cached kernel matrices (auto-generated) |

## How to Run

```bash
# With cached kernels (fast, ~0.2s)
python scripts/graph_filtering/scenario_grouping_report.py

# Force rebuild kernels (needs GPU or vLLM endpoint for embeddings)
rm tmp/.cache/scenario_grouping_kernels_*.npz
EMBEDDING_DEVICE=cuda python scripts/graph_filtering/scenario_grouping_report.py

# Override thresholds
GROUP_STRUCTURE_THRESHOLD=0.60 GROUP_TOP_K=10 python scripts/graph_filtering/scenario_grouping_report.py
```

Output: `scenario_grouping_report.html` in the outputs directory.

---

## Why This Works

### Why hierarchical (not flat)?

Flat clustering (k-means, facility location) tries to group everything at once. A "truck blocking ego at intersection" and a "pedestrian crossing at intersection" might end up in the same cluster because they share "intersection." 

Hierarchical grouping first separates by **causal pattern** (truck constraint ≠ pedestrian blocking), then by **road+intent**, then by fine similarity. By the time we cluster, we're only comparing truly comparable scenes.

### Why complete linkage (not single linkage)?

Single linkage creates chains: A≈B and B≈C → A, B, C merge even if A and C are different. Complete linkage requires ALL pairs in a cluster to be similar. This prevents the "everything merges into one mega-cluster" problem.

### Why geometric mean (not average)?

Average similarity can mask one bad dimension with good ones. Geometric mean ensures ALL dimensions must be above threshold. One zero kills the score.

### Why mutual top-k?

On single-location data (all Singapore), road similarity is high for everything. Without mutual top-k, every scene looks redundant with dozens of others. Mutual top-k says: "only your closest matches count."

---

## References

- Shervashidze et al., *Weisfeiler-Lehman Graph Kernels*, JMLR 2011
- Zelnik-Manor and Perona, *Self-Tuning Spectral Clustering*, NeurIPS 2004
- Bansal, Blum, Chawla, *Correlation Clustering*, FOCS 2002
- Lin and Bilmes, *A Class of Submodular Functions for Document Summarization*, ACL 2011




Kernel 1: Subgraph topology, use the subgraph extractions script
Kernel 2: Combine 1b, 1c, 1d
Kernel 3: Road geometry
Kernel 4: Navigation intent
Kernel 5: Ego Behavior
Kernel 6: K_semantic
Kernel 7: K_inter
