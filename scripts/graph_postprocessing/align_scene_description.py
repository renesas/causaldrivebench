#!/usr/bin/env python3
"""
Align scene_context and ego_state in edited causal graphs using Claude Sonnet.

Passes: original graph + edited graph + state.json + v6 output schema context
→ Sonnet rewrites scene_context and ego_state.behavior_explanation using active edges only.

Uses batch API for full runs. Real-time for small tests.

Usage:
  # Test 5 samples (real-time, writes per-sample comparison dirs)
  python align_scene_description.py test \
    --input-dir ... --edits-dir ... --records-dir ... --out-dir ...

  # Submit batch, poll until done, and apply results
  python align_scene_description.py run \
    --input-dir ... --edits-dir ... --records-dir ...

  # Dry-run to estimate cost
  python align_scene_description.py run \
    --input-dir ... --edits-dir ... --records-dir ... --dry-run
"""

import json, glob, os, sys, argparse, shutil, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import load_config

def _extract_json(text):
    """Extract JSON object from text that may contain markdown fences or analysis."""
    import re
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try markdown fence
    m = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try finding first { ... last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("No valid JSON found", text, 0)


def _load_v6_system_prompt():
    """Load the v6 causal graph system prompt from prompts/ directory."""
    prompt_dir = Path(__file__).resolve().parent.parent.parent / "prompts" / "causal_scene_graph" / "v6"
    prompt_path = prompt_dir / "system_prompt.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"v6 system prompt not found at {prompt_path}")
    return prompt_path.read_text()


ALIGNMENT_TASK = """\
You are given the original causal graph generation system prompt above as context.

Now your task is different: you must correct the natural language fields in a human-edited causal graph.

You receive:
1. ORIGINAL graph — the LLM-generated graph before human review
2. EDITED graph — the human-reviewed version (nodes may be added/removed/modified, edges changed)
3. State.json — precise position, heading and velocity of all agents from sensor data

Assumptiosn about state.json:
- +x = forward, +y = left. Ego at [0,0].
- x, y positions: for MOTION ANALYSIS ONLY (closing rate, lateral drift), NOT lane membership.
- yaw: 0 = same direction as ego, ±π = oncoming, ±π/2 = cross traffic.


Rewrite TWO fields to be consistent with the EDITED graph:

=== scene_context ===
- road_geometry: Describe road layout, lanes, markings, and relevant static features.
  Only reference agents/nodes present in EDITED graph's "nodes" list.
  Do NOT mention agents in "distractors" or removed from the graph.
- navigation_intent: 1 sentence describing ego's intent.

=== ego_state.behavior_explanation ===
- Explain ego's CURRENT behavior using ONLY active edges (causal_status = "active_sustaining" or "active_triggering", active=true).
- For each active edge, explain how that source constrains/affects ego.
- Do NOT reference dormant edges or distractors in the behavior explanation.
- 2-3 sentences max.

Output ONLY valid JSON:
{"scene_context": {"road_geometry": "...", "navigation_intent": "..."}, "behavior_explanation": "..."}

No markdown, no explanation, just the JSON object."""


def get_system_prompt():
    """Build full system prompt: v6 generation prompt + alignment task."""
    v6_prompt = _load_v6_system_prompt()
    return v6_prompt + "\n\n---\n\n" + ALIGNMENT_TASK


def get_client():
    cfg = load_config()
    api_key = cfg.get("anthropic_api_key", "")
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def sid_from_edited_path(fpath):
    """nuscenes-scene-0029_SAMPLED_0_naren_edited.json → nuscenes-scene-0029_SAMPLED_0"""
    base = os.path.basename(fpath).rsplit("_edited.json", 1)[0]
    return base.rsplit("_", 1)[0]


def sid_to_records_path(sid, records_dir):
    parts = sid.rsplit("_SAMPLED_", 1)
    if len(parts) != 2:
        return None
    return os.path.join(records_dir, parts[0], f"SAMPLED_{parts[1]}", "Tp0p0")


def get_pruned_path(sid, outputs_dir):
    """Get the pruned (original LLM output) path."""
    return os.path.join(outputs_dir, sid + "_pruned.json")


def get_changed_from_manifest(input_dir, edits_dir=None, manifest_path=None):
    """Read `fix_manifest.json` and return (sid, fixed_path) for content_changed
    entries only. This is the recommended flow when fix_causal_graphs has been
    run — it aligns ONLY scenes whose `_fixed.json` content differs from the
    prior run, skipping anything unchanged since the last alignment.

    `manifest_path` overrides the default `{input_dir}/fix_manifest.json`.
    needs_review is still filtered (belt-and-suspenders; fix_causal_graphs
    already skips those, but mtime drift could leave them in an old manifest)."""
    path = manifest_path or os.path.join(input_dir, "fix_manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path) as fh:
        manifest = json.load(fh)

    review_scenes = set()
    if edits_dir:
        from fix_causal_graphs import load_scene_flags
        _, review_scenes = load_scene_flags(edits_dir)

    files = []
    missing = 0
    skipped_review = 0
    for entry in manifest.get("entries", []):
        if not entry.get("content_changed"):
            continue
        sid = entry["scene_id"]
        if sid in review_scenes:
            skipped_review += 1
            continue
        fixed_path = os.path.join(input_dir, entry["fixed_path"])
        if not os.path.isfile(fixed_path):
            missing += 1
            continue
        files.append((sid, fixed_path))

    print(f"Manifest: {path}")
    print(f"  content_changed entries: {sum(1 for e in manifest.get('entries', []) if e.get('content_changed'))}")
    print(f"  eligible after filters:  {len(files)}  "
          f"(skipped {skipped_review} needs_review, {missing} missing _fixed.json)")
    return files


def copy_unchanged_from_manifest(input_dir, manifest_path=None, dry_run=False):
    """For manifest entries with `content_changed=False`, `_fixed.json` is
    byte-identical to `_pruned.json` per the fix_causal_graphs contract — so
    `_verified.json` is just a copy of `_pruned.json`, no Sonnet call needed.
    Returns (copied, missing_pruned). In dry_run mode, no files are written."""
    path = manifest_path or os.path.join(input_dir, "fix_manifest.json")
    if not os.path.isfile(path):
        return 0, 0
    with open(path) as fh:
        manifest = json.load(fh)

    copied = 0
    missing = 0
    for entry in manifest.get("entries", []):
        if entry.get("content_changed"):
            continue
        sid = entry["scene_id"]
        pruned_path = os.path.join(input_dir, f"{sid}_pruned.json")
        if not os.path.isfile(pruned_path):
            missing += 1
            continue
        if not dry_run:
            verified_path = os.path.join(input_dir, f"{sid}_verified.json")
            shutil.copy2(pruned_path, verified_path)
        copied += 1
    return copied, missing


def get_valid_edited_files(input_dir, edits_dir, fix_only=False):
    """Return files to align via Sonnet: any scene with _fixed.json OR an assigned
    _<user>_edited.json, excluding needs_review.

    Prefers _fixed.json over _edited.json when both exist.

    If fix_only is True: return only scenes that have _fixed.json AND NO _edited.json
    (useful to re-align fix-only scenes that were missed in a previous run).
    """
    from fix_causal_graphs import load_scene_flags, build_assignment_map, username_from_edited_path
    _, review_scenes = load_scene_flags(edits_dir)
    assignment_map = build_assignment_map(input_dir)

    # Index fixed files by scene_id
    fixed_map = {}
    for f in sorted(glob.glob(os.path.join(input_dir, "*_fixed.json"))):
        sid = os.path.basename(f).replace("_fixed.json", "")
        fixed_map[sid] = f

    # Index edited files (only from assigned annotator)
    edited_map = {}
    any_edited = set()  # scenes with an _edited.json from anyone
    for f in sorted(glob.glob(os.path.join(input_dir, "*_edited.json"))):
        sid = sid_from_edited_path(f)
        any_edited.add(sid)
        user = username_from_edited_path(f)
        assigned_user = assignment_map.get(sid)
        if assigned_user and user == assigned_user:
            edited_map[sid] = f

    files = []
    if fix_only:
        candidate_sids = set(fixed_map) - any_edited
    else:
        candidate_sids = set(fixed_map) | set(edited_map)

    for sid in sorted(candidate_sids):
        if sid in review_scenes:
            continue
        fpath = fixed_map.get(sid) or edited_map[sid]
        files.append((sid, fpath))
    return files


def build_prompt(original, edited, state):
    return (
        f"ORIGINAL GRAPH:\n{json.dumps(original, indent=2)}\n\n"
        f"EDITED GRAPH:\n{json.dumps(edited, indent=2)}\n\n"
        f"STATE.JSON:\n{json.dumps(state, indent=2)}\n\n"
        f"Rewrite scene_context and behavior_explanation for the EDITED graph. Output only JSON."
    )


def build_comparison(sid, fpath, original, edited, new_result):
    """Build per-sample comparison dict."""
    # Graph changes: original → edited
    orig_node_ids = {n["id"] for n in original.get("nodes", [])}
    edit_node_ids = {n["id"] for n in edited.get("nodes", [])}
    orig_edges = {(e["source"], e.get("effect_type", "")) for e in original.get("ego_edges", [])}
    edit_edges = {(e["source"], e.get("effect_type", "")) for e in edited.get("ego_edges", [])}
    orig_distractors = {d["id"] for d in original.get("distractors", [])}
    edit_distractors = {d["id"] for d in edited.get("distractors", [])}

    return {
        "scene_id": sid,
        "file": os.path.basename(fpath),
        "graph_changes": {
            "nodes_added": sorted(edit_node_ids - orig_node_ids),
            "nodes_removed": sorted(orig_node_ids - edit_node_ids),
            "edges_added": [{"source": s, "effect_type": e} for s, e in sorted(edit_edges - orig_edges)],
            "edges_removed": [{"source": s, "effect_type": e} for s, e in sorted(orig_edges - edit_edges)],
            "promoted_from_distractors": sorted(orig_distractors - edit_distractors),
            "demoted_to_distractors": sorted(edit_distractors - orig_distractors),
        },
        "scene_context": {
            "old": edited.get("scene_context", {}),
            "new": new_result["scene_context"],
        },
        "behavior_explanation": {
            "old": edited.get("ego_state", {}).get("behavior_explanation", ""),
            "new": new_result["behavior_explanation"],
        },
        "active_edges": [
            {"source": e["source"], "effect_type": e["effect_type"], "mechanism": e.get("mechanism", "")}
            for e in edited.get("ego_edges", [])
            if e.get("active") is True or e.get("causal_status", "").startswith("active")
        ],
    }


def write_sample_dir(out_dir, sid, comparison, records_dir):
    """Write comparison.json + copy images into per-sample directory."""
    sample_dir = os.path.join(out_dir, sid)
    os.makedirs(sample_dir, exist_ok=True)

    with open(os.path.join(sample_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    rec_path = sid_to_records_path(sid, records_dir)
    if rec_path and os.path.isdir(rec_path):
        for img in ["bev.png", "cam_front.jpg", "cam_front_left.jpg", "cam_front_right.jpg", "cam_back.jpg"]:
            src = os.path.join(rec_path, img)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(sample_dir, img))


# ── Test (5 samples, real-time) ──────────────────────────────────────────

def _load_scene_ids(args):
    """Resolve explicit scene IDs from --scene-ids or --scene-ids-file.

    Accepts:
      - comma-separated list
      - .txt / plain file: one sid per line
      - .json file: list[str], or list[{'scene_id': str, ...}]
    """
    ids = []
    raw = getattr(args, "scene_ids", None)
    if raw:
        ids.extend(s.strip() for s in raw.split(",") if s.strip())
    path = getattr(args, "scene_ids_file", None)
    if path:
        if path.endswith(".json"):
            with open(path) as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        ids.append(item.strip())
                    elif isinstance(item, dict) and item.get("scene_id"):
                        ids.append(item["scene_id"].strip())
        else:
            with open(path) as fh:
                ids.extend(line.strip() for line in fh if line.strip() and not line.startswith("#"))
    # Dedupe preserve order
    seen, out = set(), []
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def cmd_test(args):
    if getattr(args, "use_manifest", False):
        valid_files = get_changed_from_manifest(
            args.input_dir, args.edits_dir,
            manifest_path=getattr(args, "manifest_path", None))
    else:
        valid_files = get_valid_edited_files(args.input_dir, args.edits_dir)

    # Explicit scene-id list (overrides complexity scoring) — for cluster-driven sampling.
    explicit_ids = _load_scene_ids(args)
    if explicit_ids:
        valid_by_sid = {sid: fpath for sid, fpath in valid_files}
        picks_all = []
        for sid in explicit_ids:
            fp = valid_by_sid.get(sid)
            if fp is None:
                print(f"  WARNING: {sid} not in valid_files — skipping")
                continue
            picks_all.append((0, sid, fp))
        print(f"Testing {len(picks_all)} scenes from explicit list (real-time)...\n")
        candidates = picks_all  # already ordered; sort below is a no-op
        # Fall through to picks = candidates[:N] with N = len(candidates)
    else:
        # Score and pick 5 most complex
        candidates = []
        for sid, fpath in valid_files:
            with open(fpath) as fh:
                g = json.load(fh)
            nodes = g.get("nodes", [])
            edges = g.get("ego_edges", [])
            n_active = sum(
                1 for e in edges
                if e.get("active") is True or e.get("causal_status", "").startswith("active")
            )
            text = json.dumps(g).lower()
            score = len(nodes) + n_active
            for w in ("intersection", "crosswalk", "turning", "turn left", "turn right", "signal"):
                if w in text:
                    score += 5
                    break
            for w in ("pedestrian", "cyclist", "bicycle"):
                if w in text:
                    score += 3
                    break
            candidates.append((score, sid, fpath))

    candidates.sort(key=lambda x: -x[0])
    picks = candidates if explicit_ids else candidates[:5]

    if not explicit_ids:
        print(f"Testing {len(picks)} complex scenes (real-time)...\n")
    out_dir = args.out_dir
    apply_verified = getattr(args, "apply", False)

    client = None  # lazy init — only needed if we actually call the API
    n_cached = n_called = n_applied = n_skipped = 0

    for score, sid, fpath in picks:
        rec_path = sid_to_records_path(sid, args.records_dir)
        if not rec_path or not os.path.isdir(rec_path):
            print(f"  SKIP {sid}: no records")
            continue

        state_path = os.path.join(rec_path, "state.json")
        if not os.path.isfile(state_path):
            print(f"  SKIP {sid}: no state.json")
            continue

        # Load pruned (original LLM output) + edited + state.json
        pruned_path = get_pruned_path(sid, args.input_dir)
        if not os.path.isfile(pruned_path):
            print(f"  SKIP {sid}: no pruned graph")
            continue

        with open(pruned_path) as f:
            original = json.load(f)
        with open(fpath) as f:
            edited = json.load(f)

        # Cached-result check — skip Sonnet call if comparison.json already has fields.
        cached_comparison_path = os.path.join(out_dir, sid, "comparison.json")
        new = None
        if os.path.isfile(cached_comparison_path):
            try:
                cached = json.load(open(cached_comparison_path))
                sc_new = cached.get("scene_context", {}).get("new")
                be_new = cached.get("behavior_explanation", {}).get("new")
                if sc_new and be_new:
                    new = {"scene_context": sc_new, "behavior_explanation": be_new}
                    print(f"  CACHED {sid} (score={score})  — using {cached_comparison_path}")
                    n_cached += 1
            except (json.JSONDecodeError, OSError):
                pass

        if new is None:
            with open(state_path) as f:
                state = json.load(f)
            prompt = build_prompt(original, edited, state)
            print(f"  Processing {sid} (score={score})...", end=" ", flush=True)

            if client is None:
                client = get_client()
            try:
                response = client.messages.create(
                    model=args.model,
                    max_tokens=1024,
                    system=get_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                )
                text = ""
                for block in response.content:
                    if block.type == "text":
                        text = block.text.strip()
                        break

                new = _extract_json(text)
                assert "scene_context" in new and "behavior_explanation" in new
                u = response.usage
                cost = u.input_tokens * 3 / 1e6 + u.output_tokens * 15 / 1e6
                print(f"OK (in={u.input_tokens}, out={u.output_tokens}, ${cost:.4f})")
                n_called += 1
            except Exception as exc:
                print(f"ERROR {sid}: {exc}")
                if text:
                    print(f"    Raw response: {text[:300]}...")
                continue

        # At this point `new` is populated (from cache or fresh call). Write
        # comparison.json + copy BEV/camera images for eyeballing.
        comparison = build_comparison(sid, fpath, original, edited, new)
        write_sample_dir(out_dir, sid, comparison, args.records_dir)

        # Optionally apply to {sid}_verified.json in input-dir
        if apply_verified:
            verified_path = os.path.join(args.input_dir, f"{sid}_verified.json")
            if os.path.isfile(verified_path) and not getattr(args, "force_apply", False):
                print(f"    SKIP apply {sid}: {verified_path} exists (use --force-apply to overwrite)")
                n_skipped += 1
                continue
            graph = json.loads(json.dumps(edited))  # deep copy
            graph["scene_context"] = new["scene_context"]
            graph.setdefault("ego_state", {})["behavior_explanation"] = new["behavior_explanation"]
            with open(verified_path, "w") as f:
                json.dump(graph, f, indent=2)
            print(f"    APPLIED → {verified_path}")
            n_applied += 1

    print(f"\nSummary: {n_cached} cached, {n_called} API-called, "
          f"{n_applied} _verified.json written, {n_skipped} apply-skipped")
    print(f"Comparison dirs under: {out_dir}")


# ── Run (submit + poll + apply) ──────────────────────────────────────────

def _build_requests(args):
    """Build batch requests and file map. Returns (requests, file_map)."""
    if getattr(args, "use_manifest", False):
        valid_files = get_changed_from_manifest(
            args.input_dir, args.edits_dir,
            manifest_path=getattr(args, "manifest_path", None))
    else:
        valid_files = get_valid_edited_files(args.input_dir, args.edits_dir,
                                             fix_only=getattr(args, "fix_only", False))
    if args.limit > 0:
        valid_files = valid_files[:args.limit]

    if not valid_files:
        print("No valid edited files found.")
        return [], {}

    print(f"Building batch: {len(valid_files)} edited graphs...")
    system_prompt = get_system_prompt()
    requests = []
    file_map = {}

    for sid, fpath in valid_files:
        rec_path = sid_to_records_path(sid, args.records_dir)
        if not rec_path or not os.path.isdir(rec_path):
            continue

        state_path = os.path.join(rec_path, "state.json")
        if not os.path.isfile(state_path):
            continue

        pruned_path = get_pruned_path(sid, args.input_dir)
        if not os.path.isfile(pruned_path):
            continue

        with open(pruned_path) as f:
            original = json.load(f)
        with open(fpath) as f:
            edited = json.load(f)
        with open(state_path) as f:
            state = json.load(f)

        prompt = build_prompt(original, edited, state)
        requests.append({
            "custom_id": sid,
            "params": {
                "model": args.model,
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
            }
        })
        file_map[sid] = fpath

    return requests, file_map


def _chunk_requests(requests):
    """Split requests into chunks under Anthropic batch API size limit."""
    MAX_BYTES = 95 * 1024 * 1024  # 95MB safety margin
    chunks = []
    current_chunk = []
    current_size = 0

    for req in requests:
        req_size = len(json.dumps(req).encode("utf-8"))
        if current_chunk and current_size + req_size > MAX_BYTES:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append(req)
        current_size += req_size

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _poll_batches(client, batch_ids, poll_interval=30):
    """Poll until all batches have ended. Prints progress."""
    while True:
        all_done = True
        for bid in batch_ids:
            batch = client.messages.batches.retrieve(bid)
            c = batch.request_counts
            status = batch.processing_status
            print(f"  {bid}: {status} — succeeded={c.succeeded}, errored={c.errored}, processing={c.processing}")
            if status != "ended":
                all_done = False
        if all_done:
            print("All batches completed.")
            return
        print(f"  Waiting {poll_interval}s...")
        time.sleep(poll_interval)


def _apply_results(client, batch_ids, file_map, input_dir, edits_dir, skip_copy_fallback=False):
    """Stream batch results and write _verified.json files. Returns (updated, errors, good_copied)."""
    updated = 0
    errors = 0
    aligned_sids = set()

    for bid in batch_ids:
        for result in client.messages.batches.results(bid):
            fpath = file_map.get(result.custom_id)
            if not fpath:
                continue

            if result.result.type != "succeeded":
                errors += 1
                print(f"  ERROR {result.custom_id}: {result.result.type}")
                continue

            text = ""
            for block in result.result.message.content:
                if block.type == "text":
                    text = block.text.strip()
                    break

            try:
                new = _extract_json(text)
                assert "scene_context" in new and "behavior_explanation" in new
            except (json.JSONDecodeError, AssertionError) as e:
                errors += 1
                print(f"  ERROR {result.custom_id}: bad JSON — {e}")
                continue

            with open(fpath) as f:
                graph = json.load(f)

            graph["scene_context"] = new["scene_context"]
            graph["ego_state"]["behavior_explanation"] = new["behavior_explanation"]

            sid = result.custom_id
            verified_path = os.path.join(input_dir, f"{sid}_verified.json")
            with open(verified_path, "w") as f:
                json.dump(graph, f, indent=2)

            aligned_sids.add(sid)
            updated += 1

    # Also produce verified.json for good scenes that had NO fix and NO edit
    # (these are scenes where the pruned graph was already correct)
    good_copied = 0
    if edits_dir and not skip_copy_fallback:
        from fix_causal_graphs import load_scene_flags
        good_scenes, review_scenes = load_scene_flags(edits_dir)
        for sid in sorted(good_scenes):
            if sid in aligned_sids or sid in review_scenes:
                continue
            # Skip if _fixed.json or any _edited.json exists — those should have
            # gone through Sonnet (if they didn't, it's a config error, not a fallback case)
            if os.path.isfile(os.path.join(input_dir, f"{sid}_fixed.json")):
                continue
            if glob.glob(os.path.join(input_dir, f"{sid}_*_edited.json")):
                continue
            pruned_path = os.path.join(input_dir, f"{sid}_pruned.json")
            if not os.path.isfile(pruned_path):
                continue
            verified_path = os.path.join(input_dir, f"{sid}_verified.json")
            with open(pruned_path) as f:
                graph = json.load(f)
            with open(verified_path, "w") as f:
                json.dump(graph, f, indent=2)
            good_copied += 1

    return updated, errors, good_copied


def _print_file_summary(input_dir):
    """Print counts of pruned, edited, fixed, and verified samples."""
    pruned = set()
    edited = set()
    fixed = set()
    verified = set()

    for f in glob.glob(os.path.join(input_dir, "*.json")):
        base = os.path.basename(f)
        if base.startswith("."):
            continue
        if base.endswith("_pruned.json"):
            pruned.add(base.replace("_pruned.json", ""))
        elif base.endswith("_edited.json"):
            # e.g. scene-0029_SAMPLED_0_naren_edited.json → scene-0029_SAMPLED_0
            sid = sid_from_edited_path(f)
            edited.add(sid)
        elif base.endswith("_fixed.json"):
            fixed.add(base.replace("_fixed.json", ""))
        elif base.endswith("_verified.json"):
            verified.add(base.replace("_verified.json", ""))

    both = edited & fixed
    print(f"Sample summary ({input_dir}):")
    print(f"  Pruned:              {len(pruned)}")
    print(f"  Edited:              {len(edited)}")
    print(f"  Fixed:               {len(fixed)}")
    print(f"  Edited + Fixed:      {len(both)}")
    print(f"  Verified (existing): {len(verified)}")
    print()


def cmd_run(args):
    requests, file_map = _build_requests(args)
    use_manifest = getattr(args, "use_manifest", False)

    # Under --use-manifest, manifest entries with content_changed=False get
    # `_verified.json` via a straight copy from `_pruned.json` (no Sonnet call).
    n_manifest_copy = 0
    n_manifest_missing = 0
    if use_manifest:
        n_manifest_copy, n_manifest_missing = copy_unchanged_from_manifest(
            args.input_dir, getattr(args, "manifest_path", None), dry_run=True)

    if not requests and n_manifest_copy == 0:
        print("Nothing to do.")
        return

    chunks = _chunk_requests(requests) if requests else []
    total_size_mb = sum(len(json.dumps(r).encode("utf-8")) for r in requests) / 1e6

    # Estimate tokens: ~4 chars per token for JSON/English
    total_input_chars = sum(
        len(r["params"]["system"]) + len(r["params"]["messages"][0]["content"])
        for r in requests
    )
    est_input_tokens = total_input_chars // 4
    est_output_tokens = len(requests) * 300  # ~300 tokens per response (short JSON)

    # Batch API pricing (50% of real-time): Sonnet = $1.5/M input, $7.5/M output
    est_cost = est_input_tokens * 1.5 / 1e6 + est_output_tokens * 7.5 / 1e6

    # Count good scenes that will be copied as-is from _pruned.json
    # (good-flagged, no _fixed.json, no assigned _edited.json).
    # Disabled under --use-manifest: manifest-driven copy is used instead.
    n_good_pruned = 0
    if not getattr(args, "fix_only", False) and not use_manifest:
        from fix_causal_graphs import load_scene_flags
        good_scenes, review_scenes = load_scene_flags(args.edits_dir)
        aligned_sids = {r["custom_id"] for r in requests}
        for sid in good_scenes:
            if sid in aligned_sids or sid in review_scenes:
                continue
            if os.path.isfile(os.path.join(args.input_dir, f"{sid}_fixed.json")):
                continue
            if glob.glob(os.path.join(args.input_dir, f"{sid}_*_edited.json")):
                continue
            if os.path.isfile(os.path.join(args.input_dir, f"{sid}_pruned.json")):
                n_good_pruned += 1

    if args.dry_run:
        print(f"\nDRY RUN — Estimate:")
        print(f"  Requests: {len(requests)}")
        if requests:
            print(f"  Payload: {total_size_mb:.1f}MB in {len(chunks)} chunk(s)")
            print(f"  Est. input tokens: ~{est_input_tokens:,}")
            print(f"  Est. output tokens: ~{est_output_tokens:,}")
            print(f"  Est. batch cost: ~${est_cost:.2f} ({args.model})")
        print(f"\n  Verified samples to be written:")
        print(f"    Aligned (from batch):           {len(requests)}")
        if use_manifest:
            miss = f"  (missing _pruned.json: {n_manifest_missing})" if n_manifest_missing else ""
            print(f"    Copied (manifest unchanged):    {n_manifest_copy}{miss}")
            print(f"    Total:                          {len(requests) + n_manifest_copy}")
        else:
            print(f"    Copied (good-pruned fallback):  {n_good_pruned}")
            print(f"    Total:                          {len(requests) + n_good_pruned}")
        _print_file_summary(args.input_dir)
        return

    # Fast path: no Sonnet needed, just copy manifest-unchanged entries.
    if not requests and use_manifest:
        print(f"No Sonnet requests. Copying {n_manifest_copy} unchanged scenes "
              f"from _pruned.json...")
        copied, missing = copy_unchanged_from_manifest(
            args.input_dir, getattr(args, "manifest_path", None))
        print(f"  Copied: {copied}  (missing _pruned.json: {missing})")
        _print_file_summary(args.input_dir)
        return

    # Submit
    client = get_client()
    print(f"Submitting {len(requests)} requests ({total_size_mb:.1f}MB) in {len(chunks)} chunk(s)...")
    print(f"  Est. batch cost: ~${est_cost:.2f}")

    batch_ids = []
    for idx, chunk in enumerate(chunks):
        chunk_size_mb = sum(len(json.dumps(r).encode("utf-8")) for r in chunk) / 1e6
        batch = client.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        print(f"  Chunk {idx + 1}/{len(chunks)}: {batch.id} ({batch.processing_status}, {len(chunk)} requests, {chunk_size_mb:.1f}MB)")

    # Save manifest (in case of interruption)
    manifest_path = os.path.join(args.input_dir, "align_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"batch_ids": batch_ids, "files": file_map}, f, indent=2)
    print(f"Manifest: {manifest_path}")

    # Poll
    print("\nPolling for completion...")
    _poll_batches(client, batch_ids, poll_interval=args.poll_interval)

    # Apply
    print("\nApplying results...")
    updated, errors, good_copied = _apply_results(
        client, batch_ids, file_map, args.input_dir, args.edits_dir,
        skip_copy_fallback=getattr(args, "fix_only", False) or use_manifest,
    )

    # Manifest-driven copy for content_changed=False entries.
    manifest_copied = 0
    manifest_missing = 0
    if use_manifest:
        print(f"\nCopying {n_manifest_copy} unchanged scenes from _pruned.json...")
        manifest_copied, manifest_missing = copy_unchanged_from_manifest(
            args.input_dir, getattr(args, "manifest_path", None))
        if manifest_missing:
            print(f"  WARNING: {manifest_missing} scenes had no _pruned.json")

    print(f"\nSummary:")
    print(f"  Aligned (edited): {updated}, Errors: {errors}, Batches: {len(batch_ids)}")
    if use_manifest:
        print(f"  Copied (manifest unchanged): {manifest_copied}")
        total = updated + manifest_copied
    else:
        print(f"  Copied (good-pruned): {good_copied}")
        total = updated + good_copied
    print(f"  Total verified: {total}")
    print(f"  Output: {args.input_dir}/*_verified.json")
    _print_file_summary(args.input_dir)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Align scene descriptions via Claude Sonnet")
    sub = parser.add_subparsers(dest="command", required=True)

    p_test = sub.add_parser("test", help="Test on 5 complex scenes")
    p_test.add_argument("--input-dir", required=True)
    p_test.add_argument("--edits-dir", required=True)
    p_test.add_argument("--records-dir", required=True)
    p_test.add_argument("--model", default="claude-sonnet-4-6")
    p_test.add_argument("--out-dir", required=True, help="Output dir for per-sample comparisons")
    p_test.add_argument("--use-manifest", action="store_true",
                        help="Read fix_manifest.json and only process content_changed samples.")
    p_test.add_argument("--manifest-path", default=None,
                        help="Override manifest path (default: <input-dir>/fix_manifest.json).")
    p_test.add_argument("--scene-ids", default=None,
                        help="Comma-separated scene IDs to test (overrides complexity scoring).")
    p_test.add_argument("--scene-ids-file", default=None,
                        help="File with scene IDs: .txt (one per line) or .json (list of strings or "
                             "list of objects with 'scene_id' key — e.g. alignment_test_samples.json).")
    p_test.add_argument("--apply", action="store_true",
                        help="Also write {sid}_verified.json to --input-dir after each sample. "
                             "Skips scenes that already have _verified.json unless --force-apply.")
    p_test.add_argument("--force-apply", action="store_true",
                        help="With --apply, overwrite existing _verified.json files.")

    p_run = sub.add_parser("run", help="Submit batch, poll until done, and apply results")
    p_run.add_argument("--input-dir", required=True)
    p_run.add_argument("--edits-dir", required=True)
    p_run.add_argument("--records-dir", required=True)
    p_run.add_argument("--model", default="claude-sonnet-4-6")
    p_run.add_argument("--limit", type=int, default=0)
    p_run.add_argument("--dry-run", action="store_true", help="Estimate cost without submitting")
    p_run.add_argument("--poll-interval", type=int, default=30, help="Seconds between status polls (default: 30)")
    p_run.add_argument("--fix-only", action="store_true",
                       help="Align only scenes that have _fixed.json AND no _edited.json "
                            "(skips already-aligned scenes and skips good-pruned copy fallback)")
    p_run.add_argument("--use-manifest", action="store_true",
                       help="Read fix_manifest.json and align ONLY content_changed samples "
                            "(skips good-pruned copy fallback; incompatible with --fix-only).")
    p_run.add_argument("--manifest-path", default=None,
                       help="Override manifest path (default: <input-dir>/fix_manifest.json).")

    args = parser.parse_args()
    {"test": cmd_test, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
