#!/usr/bin/env python3
"""
Post-process causal graph JSONs. Applies these rules in order:

1.  Remove edges from traffic-light nodes to pedestrian nodes (inter_node_edges).
1b. If a pedestrian has an active INHIBITS_PROCEED ego_edge AND a crosswalk
    also has an ego_edge, drop the crosswalk's ego_edge (crosswalk causality
    to ego is redundant once the pedestrian is already gating ego).
2.  Remove ENVIRONMENT nodes and fold their state into
    `scene_context.environment_conditions`; drop all edges referencing them.
3.  Move OTHER_AGENT nodes positioned behind ego to `distractors` and remove
    all incoming/outgoing edges (ego + inter) for those nodes.
4.  Prune nodes with no directed causal path to ego (reverse BFS). Pruned nodes
    are appended to `distractors`; their inter_node_edges are dropped.
5.  Remove stranded nodes — nodes with no undirected connectivity to ego.
    OTHER_AGENT stranded nodes move to `distractors`; others are removed.
6.  Filter `distractors` to keep only OTHER_AGENT entries.
"""

import json, glob, os, sys, argparse, copy, hashlib, datetime
from collections import Counter, defaultdict, deque


def _canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _load_canonical_hash(path: str):
    """SHA256 of the JSON at `path` after key-sorting. Returns None if the
    file is missing or unparseable."""
    try:
        with open(path) as fh:
            return _canonical_hash(json.load(fh))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _resolve_fixed_path(fpath: str) -> str:
    """Strip known suffixes from source path and append `_fixed.json`."""
    base = fpath.rsplit(".json", 1)[0]
    for suffix in ("_naren_edited", "_navvrat_edited", "_aman_edited",
                   "_siddharth_edited", "_abhinav_edited", "_rajarshee_edited",
                   "_umesh_edited", "_edited", "_pruned", "_verified"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base + "_fixed.json"


# ── node-type helpers ─────────────────────────────────────────

def _upper(x): return (x or "").upper()
def _lower(x): return (x or "").lower()


def is_other_agent_node(node):
    return _upper(node.get("type")) == "OTHER_AGENT"


def is_environment_node(node):
    return _upper(node.get("type")) == "ENVIRONMENT"


def is_behind_ego(node):
    return "behind" in _lower(node.get("position_relative_to_ego"))


def is_traffic_light_node(node):
    if _upper(node.get("type")) != "TRAFFIC_CONTROL":
        return False
    cat = _lower(node.get("category"))
    label = _lower(node.get("label"))
    return cat.startswith("signal.") or "traffic_light" in label or "traffic light" in label


def is_pedestrian_node(node):
    cat = _lower(node.get("category"))
    label = _lower(node.get("label"))
    nid = _lower(node.get("id", ""))
    return (cat.startswith("human.pedestrian") or "pedestrian" in cat
            or "pedestrian" in label or nid.startswith("pedestrian"))


def is_crosswalk_node(node):
    if _upper(node.get("type")) != "TRAFFIC_CONTROL":
        return "crosswalk" in _lower(node.get("category"))
    cat = _lower(node.get("category"))
    label = _lower(node.get("label"))
    return "crosswalk" in cat or "crosswalk" in label


def _is_active_edge(edge):
    return edge.get("active") is True or _lower(edge.get("causal_status")).startswith("active")


# ── step 1: traffic_light → pedestrian inter edges ────────────

def _step1_remove_tl_to_ped_edges(g, stats):
    nodes_by_id = {n["id"]: n for n in g.get("nodes", [])}
    kept = []
    for e in g.get("inter_node_edges", []):
        src = nodes_by_id.get(e.get("source"), {})
        tgt = nodes_by_id.get(e.get("target"), {})
        if is_traffic_light_node(src) and is_pedestrian_node(tgt):
            stats["step1_tl_to_ped_edges_removed"] += 1
            continue
        kept.append(e)
    g["inter_node_edges"] = kept


# ── step 1b: crosswalk ego_edge when pedestrian INHIBITS ego ──

def _step1b_remove_crosswalk_when_ped_inhibits(g, stats):
    """When a pedestrian has an active INHIBITS_PROCEED ego_edge, any crosswalk
    ego_edges are redundant — the pedestrian already gates ego. The crosswalk
    node itself stays in `nodes`, since inter_node_edges like
    crosswalk→pedestrian AFFECTS may still reference it."""
    nodes_by_id = {n["id"]: n for n in g.get("nodes", [])}
    ego_edges = g.get("ego_edges", [])
    has_active_ped_inhibit = any(
        is_pedestrian_node(nodes_by_id.get(e.get("source"), {}))
        and e.get("effect_type") == "INHIBITS_PROCEED"
        and _is_active_edge(e)
        for e in ego_edges
    )
    if not has_active_ped_inhibit:
        return
    kept = []
    for e in ego_edges:
        if is_crosswalk_node(nodes_by_id.get(e.get("source"), {})):
            stats["step1b_crosswalk_ego_edges_removed"] += 1
            continue
        kept.append(e)
    g["ego_edges"] = kept


# ── step 2: ENVIRONMENT → scene_context ───────────────────────

def _build_environment_str(env_nodes):
    parts = []
    for n in env_nodes:
        desc = ((n.get("state") or "").strip()
                or (n.get("label") or "").strip()
                or (n.get("category") or "").strip())
        if desc:
            parts.append(desc)
    seen, unique = set(), []
    for p in parts:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            unique.append(p)
    return "; ".join(unique)


def _step2_move_environment_to_scene_context(g, stats):
    env_nodes = [n for n in g.get("nodes", []) if is_environment_node(n)]
    if not env_nodes:
        return
    env_ids = {n["id"] for n in env_nodes}
    stats["step2_env_nodes_removed"] += len(env_nodes)

    env_str = _build_environment_str(env_nodes)
    if env_str:
        sc = g.setdefault("scene_context", {})
        existing = (sc.get("environment_conditions") or "").strip()
        if existing and existing.lower() != env_str.lower():
            sc["environment_conditions"] = f"{existing}. {env_str}"
        else:
            sc["environment_conditions"] = env_str

    g["nodes"] = [n for n in g.get("nodes", []) if n["id"] not in env_ids]

    before = len(g.get("ego_edges", []))
    g["ego_edges"] = [e for e in g.get("ego_edges", [])
                      if e.get("source") not in env_ids and e.get("target") not in env_ids]
    stats["step2_env_ego_edges_removed"] += before - len(g["ego_edges"])

    before = len(g.get("inter_node_edges", []))
    g["inter_node_edges"] = [e for e in g.get("inter_node_edges", [])
                             if e.get("source") not in env_ids and e.get("target") not in env_ids]
    stats["step2_env_inter_edges_removed"] += before - len(g["inter_node_edges"])


# ── step 3: OTHER_AGENT behind ego → distractors ──────────────

def _step3_move_behind_agents_to_distractors(g, stats):
    behind_ids = {
        n["id"] for n in g.get("nodes", [])
        if n.get("id") != "ego" and is_other_agent_node(n) and is_behind_ego(n)
    }
    if not behind_ids:
        return

    distractors = g.setdefault("distractors", [])
    existing_ids = {d.get("id") for d in distractors}
    kept = []
    for n in g.get("nodes", []):
        if n.get("id") in behind_ids:
            if n["id"] not in existing_ids:
                distractors.append(n)
                existing_ids.add(n["id"])
            stats["step3_behind_agents_demoted"] += 1
        else:
            kept.append(n)
    g["nodes"] = kept

    before = len(g.get("ego_edges", []))
    g["ego_edges"] = [e for e in g.get("ego_edges", [])
                      if e.get("source") not in behind_ids and e.get("target") not in behind_ids]
    stats["step3_behind_ego_edges_removed"] += before - len(g["ego_edges"])

    before = len(g.get("inter_node_edges", []))
    g["inter_node_edges"] = [e for e in g.get("inter_node_edges", [])
                             if e.get("source") not in behind_ids and e.get("target") not in behind_ids]
    stats["step3_behind_inter_edges_removed"] += before - len(g["inter_node_edges"])


# ── step 4: prune nodes with no directed path to ego ──────────

def _reverse_reachable_from_ego(ego_edges, inter_edges):
    rev = defaultdict(set)
    for e in ego_edges:
        src = e.get("source")
        tgt = e.get("target")
        if src and src != "ego" and tgt == "ego":
            rev["ego"].add(src)
        elif tgt and tgt != "ego" and src == "ego":
            rev["ego"].add(tgt)
    for e in inter_edges:
        src = e.get("source")
        tgt = e.get("target")
        if not src or not tgt:
            continue
        if e.get("type") == "CONFOUNDED_BY" or e.get("relation") == "CONFOUNDED_BY":
            rev[src].add(tgt)
            rev[tgt].add(src)
        else:
            rev[tgt].add(src)
    visited = {"ego"}
    q = deque(["ego"])
    while q:
        n = q.popleft()
        for nb in rev[n]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    visited.discard("ego")
    return visited


def _step4_prune_unreachable(g, stats):
    ego_edges = g.get("ego_edges", [])
    inter_edges = g.get("inter_node_edges", [])
    reachable = _reverse_reachable_from_ego(ego_edges, inter_edges)

    distractors = g.setdefault("distractors", [])
    existing_ids = {d.get("id") for d in distractors}
    kept = []
    pruned = set()
    for n in g.get("nodes", []):
        nid = n.get("id")
        if nid == "ego" or nid in reachable:
            kept.append(n)
            continue
        pruned.add(nid)
        if nid not in existing_ids:
            distractors.append({
                "id": nid,
                "type": n.get("type", ""),
                "category": n.get("category", ""),
                "position_relative_to_ego": n.get("position_relative_to_ego", ""),
                "rejection_reason": "PRUNED: No causal path to ego.",
            })
            existing_ids.add(nid)

    if pruned:
        g["inter_node_edges"] = [
            e for e in inter_edges
            if e.get("source") not in pruned and e.get("target") not in pruned
        ]
        stats["step4_unreachable_nodes_pruned"] += len(pruned)
    g["nodes"] = kept


# ── step 5: remove stranded nodes (undirected BFS from ego) ───

def _undirected_reachable_from_ego(nodes, ego_edges, inter_edges):
    adj = defaultdict(set)
    for e in ego_edges + inter_edges:
        src, tgt = e.get("source"), e.get("target")
        if src and tgt:
            adj[src].add(tgt)
            adj[tgt].add(src)
    has_ego = "ego" in adj or any(n.get("id") == "ego" for n in nodes)
    if not has_ego:
        return set()
    visited, stack = set(), ["ego"]
    while stack:
        n = stack.pop()
        if n in visited:
            continue
        visited.add(n)
        stack.extend(adj[n])
    return visited


def _step5_remove_stranded(g, stats):
    nodes = g.get("nodes", [])
    ego_edges = g.get("ego_edges", [])
    inter_edges = g.get("inter_node_edges", [])
    reachable = _undirected_reachable_from_ego(nodes, ego_edges, inter_edges)

    distractors = g.setdefault("distractors", [])
    existing_ids = {d.get("id") for d in distractors}
    kept = []
    dead_ids = set()
    for n in nodes:
        nid = n.get("id")
        if nid == "ego" or nid in reachable:
            kept.append(n)
            continue
        dead_ids.add(nid)
        if is_other_agent_node(n):
            if nid not in existing_ids:
                distractors.append(n)
                existing_ids.add(nid)
            stats["step5_stranded_demoted"] += 1
        else:
            stats["step5_stranded_removed"] += 1

    if dead_ids:
        g["inter_node_edges"] = [
            e for e in inter_edges
            if e.get("source") not in dead_ids and e.get("target") not in dead_ids
        ]
    g["nodes"] = kept


# ── step 6: only OTHER_AGENT distractors ──────────────────────

def _step6_filter_distractors(g, stats):
    distractors = g.get("distractors", [])
    kept = [d for d in distractors if _upper(d.get("type")) == "OTHER_AGENT"]
    stats["step6_non_other_agent_distractors_removed"] += len(distractors) - len(kept)
    g["distractors"] = kept


def process_graph(graph):
    """Apply the six rules in order. Returns (modified_graph, stats_dict)."""
    g = copy.deepcopy(graph)
    stats = Counter()
    _step1_remove_tl_to_ped_edges(g, stats)
    _step1b_remove_crosswalk_when_ped_inhibits(g, stats)
    _step2_move_environment_to_scene_context(g, stats)
    _step3_move_behind_agents_to_distractors(g, stats)
    _step4_prune_unreachable(g, stats)
    _step5_remove_stranded(g, stats)
    _step6_filter_distractors(g, stats)
    return g, dict(stats)


# ── Scene Assignments (from editor_server.py admin dashboard) ────────────

# Active annotators — 0-based index into sorted scene list
ACTIVE_ASSIGNMENTS = {
    "naren": [(0, 95), (803, 850)],
    "navvrat": [(95, 190), (756, 803)],
    "aman": [(190, 285)],
    "siddharth": [(285, 380)],
    "abhinav": [(380, 474)],
    "rajarshee": [(474, 568), (707, 756)],
    "umesh": [(568, 707)],
}

INACTIVE_ASSIGNMENTS = {
    "sahil": [(662, 756)],
    "shreedhar": [(756, 850)],
}


def load_scene_flags(edits_dir):
    """Load scene flags from active annotators' bookmarks only.
    Returns (good, needs_review) sets."""
    good, needs_review = set(), set()
    if not os.path.isdir(edits_dir):
        return good, needs_review
    for user_dir in ACTIVE_ASSIGNMENTS:
        bpath = os.path.join(edits_dir, user_dir, "bookmarks.json")
        if not os.path.isfile(bpath):
            continue
        with open(bpath) as f:
            for scene_id, entry in json.load(f).items():
                flag = entry.get("flag", "")
                if flag == "good":
                    good.add(scene_id)
                elif flag == "needs_review":
                    needs_review.add(scene_id)
    return good, needs_review


def build_scene_list(outputs_dir):
    """Build sorted scene list from outputs dir (same logic as editor_server._build_scene_list)."""
    import re
    index_path = os.path.join(outputs_dir, "index.html")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            html = f.read()
        hrefs = re.findall(r'href="([^"]+\.html)"', html)
        return [h.replace(".html", "") for h in hrefs
                if h != "index.html" and not h.startswith("verification")]
    # Fallback: derive from JSON files
    sids = set()
    for f in glob.glob(os.path.join(outputs_dir, "*.json")):
        base = os.path.basename(f).replace(".json", "")
        if "_pruned" in base or "_edited" in base or "_aligned" in base or base.startswith("."):
            continue
        sids.add(base)
    return sorted(sids)


def build_assignment_map(outputs_dir):
    """Build scene_id → assigned_username map for active annotators.

    Returns dict: {scene_id: username} for all scenes assigned to active annotators.
    """
    scenes = build_scene_list(outputs_dir)

    # Group samples by scene name
    scene_map = {}
    for sid in scenes:
        scene = sid.rsplit("_SAMPLED_", 1)[0] if "_SAMPLED_" in sid else sid
        scene_map.setdefault(scene, []).append(sid)
    scene_names = sorted(scene_map.keys())

    # Map each sample to its assigned active annotator
    sid_to_user = {}
    for username, ranges in ACTIVE_ASSIGNMENTS.items():
        for start, end in ranges:
            for sc in scene_names[start:end]:
                for sid in scene_map.get(sc, []):
                    sid_to_user[sid] = username
    return sid_to_user


def username_from_edited_path(fpath):
    """nuscenes-scene-0029_SAMPLED_0_naren_edited.json → naren"""
    base = os.path.basename(fpath).rsplit("_edited.json", 1)[0]
    return base.rsplit("_", 1)[1]


def collect_files(outputs_dir, edits_dir):
    """
    File selection logic (excludes needs_review; processes good + unflagged):
      1. needs_review → skip entirely
      2. Only use _edited.json from the assigned active annotator for that scene
      3. has assigned _edited.json (good or unflagged) → use _edited.json
      4. no assigned _edited.json (good or unflagged) → use _pruned.json
    """
    good_scenes, review_scenes = load_scene_flags(edits_dir)
    assignment_map = build_assignment_map(outputs_dir)

    # Index edited files — only keep the one from the assigned annotator
    edited = {}
    for f in glob.glob(os.path.join(outputs_dir, "*_edited.json")):
        base = os.path.basename(f).rsplit("_edited.json", 1)[0]
        scene_id = base.rsplit("_", 1)[0]
        user = base.rsplit("_", 1)[1]
        assigned_user = assignment_map.get(scene_id)
        if assigned_user and user == assigned_user:
            edited[scene_id] = f

    pruned_map = {}
    for f in glob.glob(os.path.join(outputs_dir, "*_pruned.json")):
        scene_id = os.path.basename(f).replace("_pruned.json", "")
        pruned_map[scene_id] = f

    files = []
    # Edited files (good or unflagged; NOT needs_review)
    for sid, fp in sorted(edited.items()):
        if sid in review_scenes:
            continue
        tag = "edited+good" if sid in good_scenes else "edited"
        files.append((sid, fp, tag))

    # Pruned fallback — good OR unflagged, but NOT needs_review, NOT already edited
    for sid, fp in sorted(pruned_map.items()):
        if sid in review_scenes or sid in edited:
            continue
        tag = "good" if sid in good_scenes else "unflagged"
        files.append((sid, fp, tag))

    return files


def main():
    parser = argparse.ArgumentParser(description="Post-process causal graphs")
    parser.add_argument("--input-dir", required=True, help="Outputs directory with graph JSONs")
    parser.add_argument("--edits-dir", required=True, help="Edits directory with user bookmarks")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    files = collect_files(args.input_dir, args.edits_dir)
    if not files:
        print("No files to process.")
        sys.exit(1)

    n_edited = sum(1 for _, _, t in files if "edited" in t)
    n_good = sum(1 for _, _, t in files if t == "good")
    n_unflagged = sum(1 for _, _, t in files if t == "unflagged")
    print(f"Processing {len(files)} graphs "
          f"({n_edited} edited, {n_good} good-pruned, {n_unflagged} unflagged-pruned)...")

    step_label = {
        "step1_tl_to_ped_edges_removed":           "Step 1 — traffic_light→pedestrian inter edges removed",
        "step1b_crosswalk_ego_edges_removed":      "Step 1b — crosswalk ego_edges removed (ped active INHIBITS)",
        "step2_env_nodes_removed":                 "Step 2 — ENVIRONMENT nodes folded into scene_context",
        "step2_env_ego_edges_removed":             "Step 2 — env ego_edges removed",
        "step2_env_inter_edges_removed":           "Step 2 — env inter_edges removed",
        "step3_behind_agents_demoted":             "Step 3 — OTHER_AGENT behind-ego → distractors",
        "step3_behind_ego_edges_removed":          "Step 3 — behind-agent ego_edges removed",
        "step3_behind_inter_edges_removed":        "Step 3 — behind-agent inter_edges removed",
        "step4_unreachable_nodes_pruned":          "Step 4 — nodes with no causal path to ego pruned",
        "step5_stranded_demoted":                  "Step 5 — stranded OTHER_AGENT → distractors",
        "step5_stranded_removed":                  "Step 5 — stranded non-OTHER_AGENT removed",
        "step6_non_other_agent_distractors_removed": "Step 6 — non-OTHER_AGENT distractors removed",
    }
    totals = Counter()
    fix_applied_ct = 0       # fix rules fired
    content_changed_ct = 0   # _fixed.json content differs from prior (or didn't exist)
    manifest_entries = []

    for sid, fpath, ftype in files:
        with open(fpath) as fh:
            graph = json.load(fh)

        new_graph, stats = process_graph(graph)
        fix_applied = any(stats.values())
        was_edited = "edited" in ftype
        fixed_path = _resolve_fixed_path(fpath)

        # content_changed drives align --use-manifest: True iff the fixed graph
        # differs from the _pruned.json baseline (i.e., Sonnet alignment is
        # warranted). Good-pruned scenes with no fix rule fired end up
        # byte-identical to _pruned.json and should be copied straight to
        # _verified.json instead of paying for a Sonnet call.
        new_hash = _canonical_hash(new_graph)
        if "edited" in ftype:
            pruned_hash = _load_canonical_hash(
                os.path.join(args.input_dir, f"{sid}_pruned.json"))
        else:
            # fpath is already the _pruned.json — reuse the graph we loaded.
            pruned_hash = _canonical_hash(graph)
        content_changed = pruned_hash is None or new_hash != pruned_hash

        if fix_applied:
            fix_applied_ct += 1
            totals.update(stats)
        if content_changed:
            content_changed_ct += 1

        if not args.dry_run:
            # Always write _fixed.json, even if content unchanged (keeps the
            # set complete for downstream). Manifest records whether content
            # actually changed.
            with open(fixed_path, "w") as fh:
                json.dump(new_graph, fh, indent=2)

        manifest_entries.append({
            "scene_id": sid,
            "source_type": ftype,
            "source_path": os.path.basename(fpath),
            "fixed_path": os.path.basename(fixed_path),
            "was_edited": was_edited,
            "fix_applied": fix_applied,
            "content_changed": content_changed,
            "content_hash": new_hash,
            "fix_stats": {k: v for k, v in stats.items() if v},
        })

        if fix_applied:
            parts = [f"{k.split('_', 1)[0]}:{v}" for k, v in stats.items() if v]
            print(f"  [{ftype}] {os.path.basename(fpath)}: {', '.join(parts)}")

    if not args.dry_run:
        manifest_path = os.path.join(args.input_dir, "fix_manifest.json")
        manifest = {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "input_dir": args.input_dir,
            "edits_dir": args.edits_dir,
            "summary": {
                "processed": len(files),
                "fix_applied": fix_applied_ct,
                "content_changed": content_changed_ct,
                "edited_sources": sum(1 for e in manifest_entries if e["was_edited"]),
            },
            "entries": manifest_entries,
        }
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\n  Manifest: {manifest_path}")

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Summary:")
    print(f"  Files processed      : {len(files)}  ({n_edited} edited, "
          f"{n_good} good-pruned, {n_unflagged} unflagged-pruned)")
    print(f"  Fix rules applied    : {fix_applied_ct}")
    print(f"  Content changed      : {content_changed_ct}  "
          f"(fixed ≠ _pruned.json — needs Sonnet align; the other "
          f"{len(files) - content_changed_ct} can be copied from _pruned.json)")
    for key, label in step_label.items():
        if totals[key]:
            print(f"  {label}: {totals[key]}")


if __name__ == "__main__":
    main()
