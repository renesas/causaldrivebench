import itertools
from collections import defaultdict

def build_adjacency(graph):
    children = defaultdict(set)
    parents = defaultdict(set)

    for e in graph["ego_edges"]:
        children[e["source"]].add(e["target"])
        parents[e["target"]].add(e["source"])

    for e in graph["inter_node_edges"]:
        children[e["source"]].add(e["target"])
        parents[e["target"]].add(e["source"])

    return children, parents

def extract_ego_structures(graph, ego_id="ego"):
    children, parents = build_adjacency(graph)
    all_nodes = {n["id"] for n in graph["nodes"]}

    results = {"direct": [], "chain": [], "fork": [], "collider": []}

    # Direct: A -> ego
    for a in parents[ego_id]:
        results["direct"].append((a, ego_id))

    # Chain: A -> B -> ego
    for b in parents[ego_id]:          # B -> ego exists
        for a in parents[b]:           # A -> B exists
            if a != ego_id:            # avoid self-loops
                results["chain"].append((a, b, ego_id))

    # Fork: ego <- C -> B  (C is common cause)
    for c in parents[ego_id]:          # C -> ego exists
        for b in children[c]:          # C -> B exists
            if b != ego_id:            # B is some other node
                results["fork"].append((ego_id, c, b))

    # Collider: A -> ego <- B  (ego is the collider)
    ego_parents = list(parents[ego_id])
    for a, b in itertools.combinations(ego_parents, 2):
        results["collider"].append((a, ego_id, b))

    return results
