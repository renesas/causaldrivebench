#!/usr/bin/env python3
"""
Process batch results from OpenAI Batch API and save to appropriate locations.

Usage:
    python process_results.py --file batch_results/batch_xxx_results.jsonl --stage 1  --output /path/to/graphs
    python process_results.py --file batch_results/batch_xxx_results.jsonl --stage 2  --output /path/to/graphs
    python process_results.py --file batch_results/batch_xxx_results.jsonl --stage 3a --output /path/to/qa_results
    python process_results.py --file batch_results/batch_xxx_results.jsonl --stage 3b --output /path/to/qa_results
    python process_results.py --file batch_results/batch_xxx_results.jsonl --stage 4cf --output /path/to/qa_results --records /path/to/records --manifest batch_jobs/dataset_stage4cf_manifest.json
    python process_results.py --file batch_results/batch_xxx_results.jsonl --stage 4ac --output /path/to/qa_results --records /path/to/records --manifest batch_jobs/dataset_stage4ac_manifest.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_id_map(batch_dir: Path) -> Dict[str, str]:
    """Load all sanitized→original ID maps from batch_jobs/."""
    id_map = {}
    for f in batch_dir.glob("*_id_map.json"):
        with open(f) as fp:
            id_map.update(json.load(fp))
    return id_map


def _resolve_scene_id(custom_id: str, stage_suffix: str, id_map: Dict[str, str]) -> str:
    """Resolve custom_id to original scene_id using id_map or suffix stripping."""
    # Try id_map first (handles sanitized IDs)
    original = id_map.get(custom_id)
    if original:
        return original.removesuffix(f"_{stage_suffix}")
    # Fallback: direct suffix removal
    return custom_id.removesuffix(f"_{stage_suffix}")

sys.path.append(str(Path(__file__).parent.parent))
from common.scene_utils import parse_batch_response, extract_json
from config_loader import load_config as _load_config


def parse_claude_response(line: str):
    """Parse a Claude Message Batches response line."""
    data = json.loads(line)
    custom_id = data.get("custom_id", "")
    result = data.get("result", {})
    if result.get("type") != "succeeded":
        err = result.get("error", {})
        return custom_id, {}, err.get("message", result.get("type", "failed"))
    msg = result.get("message", {})
    raw_text = ""
    for block in msg.get("content", []):
        if block.get("type") == "text":
            raw_text = block.get("text", "").strip()
            break
    # Extract JSON from raw text
    parsed = extract_json(raw_text)
    # Wrap lists so we can attach metadata
    if isinstance(parsed, list):
        parsed = {"_items": parsed}
    usage = msg.get("usage", {})
    if usage:
        parsed["_usage"] = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        }
    return custom_id, parsed, None


# ── Stage 1 ───────────────────────────────────────────────────────────────

def process_stage1_results(results_file: Path, output_dir: Path,
                           parse_fn=None, id_map: Dict[str, str] = None) -> Tuple[int, int]:
    """Process Stage 1 (generator) results.
    custom_id format: {scene_id}_stage1  e.g. nuscenes-scene-0001_SAMPLED_0_stage1
    """
    if parse_fn is None:
        parse_fn = parse_batch_response
    print("Processing Stage 1 results...")
    success_count = error_count = 0
    errors = defaultdict(list)

    with open(results_file) as f:
        for line in f:
            custom_id, result, error = parse_fn(line)
            if error:
                errors[error].append(custom_id)
                error_count += 1
                continue

            scene_id = _resolve_scene_id(custom_id, "stage1", id_map)
            output_file = output_dir / f"{scene_id}.json"
            with open(output_file, "w") as out:
                json.dump(result, out, indent=2)
            success_count += 1

    _print_summary(success_count, error_count, errors)
    return success_count, error_count


# ── Stage 2 ───────────────────────────────────────────────────────────────

def process_stage2_results(results_file: Path, output_dir: Path,
                           parse_fn=None, id_map: Dict[str, str] = None) -> Tuple[int, int]:
    """Process Stage 2 (discriminator) results."""
    if parse_fn is None:
        parse_fn = parse_batch_response
    print("Processing Stage 2 results...")
    success_count = error_count = 0
    errors = defaultdict(list)

    with open(results_file) as f:
        for line in f:
            custom_id, result, error = parse_fn(line)
            if error:
                errors[error].append(custom_id)
                error_count += 1
                continue

            scene_id = _resolve_scene_id(custom_id, "stage2", id_map)
            graph = result.get("graph", result)

            output_file = output_dir / f"{scene_id}_refined.json"
            with open(output_file, "w") as out:
                json.dump(graph, out, indent=2)

            if "changes" in result or "reasoning_trace" in result:
                changes_file = output_dir / f"{scene_id}_changes.json"
                with open(changes_file, "w") as out:
                    json.dump({
                        "changes": result.get("changes", []),
                        "reasoning_trace": result.get("reasoning_trace", ""),
                    }, out, indent=2)

            success_count += 1

    _print_summary(success_count, error_count, errors)
    return success_count, error_count


# ── Stage 3a ──────────────────────────────────────────────────────────────

def process_stage3a_results(results_file: Path, qa_dir: Path,
                            parse_fn=None, id_map: Dict[str, str] = None) -> Tuple[int, int]:
    """Process Stage 3a (QA Active) results."""
    if parse_fn is None:
        parse_fn = parse_batch_response
    print("Processing Stage 3a results...")
    success_count = error_count = 0

    with open(results_file) as f:
        for line in f:
            custom_id, result, error = parse_fn(line)
            if error:
                error_count += 1
                continue

            scene_id = _resolve_scene_id(custom_id, "stage3a", id_map)
            scene_qa_dir = qa_dir / scene_id
            scene_qa_dir.mkdir(parents=True, exist_ok=True)

            questions = result.get("_items", result.get("questions", []))
            active_data = {"questions": questions, "usage": result.get("_usage", {})}
            with open(scene_qa_dir / "active_qa.json", "w") as out:
                json.dump(active_data, out, indent=2)

            success_count += 1

    _print_summary(success_count, error_count)
    return success_count, error_count


# ── Stage 3b ──────────────────────────────────────────────────────────────

def process_stage3b_results(results_file: Path, qa_dir: Path,
                            parse_fn=None, id_map: Dict[str, str] = None) -> Tuple[int, int]:
    """Process Stage 3b (QA Dormant) results."""
    if parse_fn is None:
        parse_fn = parse_batch_response
    print("Processing Stage 3b results...")
    success_count = error_count = 0

    with open(results_file) as f:
        for line in f:
            custom_id, result, error = parse_fn(line)
            if error:
                error_count += 1
                continue

            scene_id = _resolve_scene_id(custom_id, "stage3b", id_map)
            scene_qa_dir = qa_dir / scene_id
            scene_qa_dir.mkdir(parents=True, exist_ok=True)

            questions = result.get("_items", result.get("questions", []))
            dl_data = {"questions": questions, "usage": result.get("_usage", {})}
            with open(scene_qa_dir / "dormant_qa.json", "w") as out:
                json.dump(dl_data, out, indent=2)

            success_count += 1

    _print_summary(success_count, error_count)
    merge_qa_results(qa_dir)
    return success_count, error_count


def process_stage3c_results(results_file: Path, qa_dir: Path,
                            parse_fn=None, id_map: Dict[str, str] = None) -> Tuple[int, int]:
    """Process Stage 3c (Distractor Nodes QA) results."""
    if parse_fn is None:
        parse_fn = parse_batch_response
    print("Processing Stage 3c results...")
    success_count = error_count = 0

    with open(results_file) as f:
        for line in f:
            custom_id, result, error = parse_fn(line)
            if error:
                error_count += 1
                continue

            scene_id = _resolve_scene_id(custom_id, "stage3c", id_map)
            scene_qa_dir = qa_dir / scene_id
            scene_qa_dir.mkdir(parents=True, exist_ok=True)

            questions = result.get("_items", result.get("questions", []))
            dist_data = {"questions": questions, "usage": result.get("_usage", {})}
            with open(scene_qa_dir / "distractor_qa.json", "w") as out:
                json.dump(dist_data, out, indent=2)

            success_count += 1

    _print_summary(success_count, error_count)
    merge_qa_results(qa_dir)
    return success_count, error_count


def merge_qa_results(qa_dir: Path):
    """Merge active, dormant, and distractor results into final result.json for each scene."""
    print("\nMerging QA results...")
    merged_count = 0

    for scene_dir in qa_dir.iterdir():
        if not scene_dir.is_dir():
            continue

        active_file = scene_dir / "active_qa.json"
        dl_file     = scene_dir / "dormant_qa.json"

        distractor_file = scene_dir / "distractor_qa.json"

        active_data     = json.loads(active_file.read_text())     if active_file.exists()     else {}
        dl_data         = json.loads(dl_file.read_text())         if dl_file.exists()         else {}
        distractor_data = json.loads(distractor_file.read_text()) if distractor_file.exists() else {}

        if not active_data and not dl_data and not distractor_data:
            continue

        n_total = (len(active_data.get("questions", [])) +
                   len(dl_data.get("questions", [])) +
                   len(distractor_data.get("questions", [])))

        result = {
            "active_questions":     active_data.get("questions", []),
            "dl_questions":         dl_data.get("questions", []),
            "distractor_questions": distractor_data.get("questions", []),
            "total_questions":      n_total,
            "usage": {
                "active":     active_data.get("usage", {}),
                "dl":         dl_data.get("usage", {}),
                "distractor": distractor_data.get("usage", {}),
            },
        }
        with open(scene_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        merged_count += 1

    print(f"  Merged {merged_count} scene results")


# ── Stage 4cf / 4ac ──────────────────────────────────────────────────────

def process_stage4_results(results_file: Path, qa_dir: Path, records_dir: Path,
                           manifest_file: Path, prompt_type: str = "cf",
                           parse_fn=None) -> Tuple[int, int]:
    """Process Stage 4 CF or AC trajectory batch results.

    Reads manifest for question metadata, saves generator.json + result.json,
    and runs local BEV + camera visualization.
    """
    label = "4cf" if prompt_type == "cf" else "4ac"
    traj_subdir = "cf_trajectories" if prompt_type == "cf" else "ac_trajectories"
    if parse_fn is None:
        parse_fn = parse_batch_response
    print(f"Processing Stage {label} results...")

    if not manifest_file.exists():
        print(f"ERROR: manifest not found: {manifest_file}")
        return 0, 0

    manifest: Dict = json.loads(manifest_file.read_text())

    # Lazy import of visualize() from counterfactual_pipeline to avoid hard dep at module level
    sys.path.insert(0, str(Path(__file__).parent.parent / "counterfactual_trajectory"))
    from counterfactual_pipeline import visualize  # type: ignore

    success_count = error_count = 0
    errors = defaultdict(list)

    with open(results_file) as f:
        for line in f:
            custom_id, result, error = parse_fn(line)
            if error:
                errors[error].append(custom_id)
                error_count += 1
                continue

            meta = manifest.get(custom_id)
            if not meta:
                # Try matching without q_idx for backwards compat with old batches
                # Old format: scene__type__slug  New format: scene__type__idx__slug
                parts = custom_id.split("__")
                if len(parts) >= 3:
                    for mkey, mval in manifest.items():
                        if mval["scene_id"] == parts[0] and mval["slug"] == parts[-1]:
                            meta = mval
                            break
                if not meta:
                    print(f"  WARNING: no manifest entry for '{custom_id}' — skipping")
                    error_count += 1
                    continue

            scene_id   = meta["scene_id"]
            scene_dir  = Path(meta["scene_dir"])
            question   = meta["question"]
            cf_answer  = meta["cf_answer"]
            slug       = meta["slug"]

            # Output directory mirrors generate_causal_qa.py structure:
            # qa_results/{scene_id}/{traj_subdir}/{slug}/{scene_id}/
            out_dir = qa_dir / scene_id / traj_subdir / slug / scene_id
            out_dir.mkdir(parents=True, exist_ok=True)

            # Load original trajectory for visualization
            tp0_dir    = scene_dir / "Tp0p0"
            state_json = json.loads((tp0_dir / "state.json").read_text())
            original_trajectory = state_json.get("ego", {}).get("future", [])

            # Load calib for camera projection
            calib = None
            calib_path = tp0_dir / "calib.json"
            if calib_path.exists():
                calib = json.loads(calib_path.read_text())

            # Save generator output
            with open(out_dir / "generator.json", "w") as f_out:
                json.dump(result, f_out, indent=2)

            # Run local visualization (BEV + camera overlays) — only for LLM trajectory routes
            route = meta.get("route", "llm_t0")
            if route != "nuplan_config":
                visualize(scene_dir, out_dir, result, original_trajectory, calib)

            # Save result.json
            result_data = {
                "scene":     scene_id,
                "question":  question,
                "answer":    cf_answer,
                "generator": result,
                "usage":     result.get("_usage", {}),
                "error":     None,
            }
            with open(out_dir / "result.json", "w") as f_out:
                json.dump(result_data, f_out, indent=2)

            success_count += 1
            print(f"  Saved {scene_id} / {slug}")

    _print_summary(success_count, error_count, errors)

    # ── Write consolidated JSON for render_causal_graph.py ─────────────────
    # CF: rung3_cf_trajectories.json  → {"active_question", "cf_pipeline_result"}
    # AC: ac_binary_trajectories.json → {"dormant_question",     "ac_pipeline_result"}
    is_cf       = prompt_type == "cf"
    out_filename = "rung3_cf_trajectories.json" if is_cf else "ac_binary_trajectories.json"
    qa_filename  = "active_qa.json"             if is_cf else "dormant_qa.json"
    q_key        = "active_question"            if is_cf else "dormant_question"
    r_key        = "cf_pipeline_result"         if is_cf else "ac_pipeline_result"

    consolidated: Dict[str, List] = defaultdict(list)

    for custom_id, meta in manifest.items():
        scene_id  = meta["scene_id"]
        slug      = meta["slug"]
        question  = meta["question"]
        out_dir   = qa_dir / scene_id / traj_subdir / slug / scene_id
        result_path = out_dir / "result.json"
        if not result_path.exists():
            continue

        result_data = json.loads(result_path.read_text())

        # Match to the source question in active_qa.json or dormant_qa.json
        matched_question = {}
        qa_path = qa_dir / scene_id / qa_filename
        if qa_path.exists():
            qa_data = json.loads(qa_path.read_text())
            for q in qa_data.get("questions", []):
                if q.get("question", "").strip() == question.strip():
                    matched_question = q
                    break

        consolidated[scene_id].append({
            q_key: matched_question,
            r_key: result_data,
        })

    for scene_id, entries in consolidated.items():
        out_path = qa_dir / scene_id / out_filename
        with open(out_path, "w") as f_out:
            json.dump(entries, f_out, indent=2)
        print(f"  Wrote {out_path}")

    # ── Write consolidated nuplan_intervention_configs.json per scene ─────
    np_configs: Dict[str, List] = defaultdict(list)
    for custom_id, meta in manifest.items():
        if meta.get("route") != "nuplan_config":
            continue
        scene_id = meta["scene_id"]
        slug = meta["slug"]
        out_dir = qa_dir / scene_id / traj_subdir / slug / scene_id
        gen_path = out_dir / "generator.json"
        if not gen_path.exists():
            continue
        gen_data = json.loads(gen_path.read_text())

        # Load uuid_map and meta.json for this scene
        scene_dir = Path(meta["scene_dir"])
        tp0_dir = scene_dir / "Tp0p0"
        uuid_map = {}
        uuid_path = tp0_dir / "uuid_map.json"
        if uuid_path.exists():
            uuid_map = json.loads(uuid_path.read_text())

        scene_meta = {}
        meta_path = scene_dir / "meta.json"
        if meta_path.exists():
            scene_meta = json.loads(meta_path.read_text())

        entity_id = meta.get("entity_id", "")
        dataset = scene_meta.get("dataset", "")
        # Build scene identifier based on dataset
        scene_ref = {}
        if dataset == "nuscenes":
            scene_ref = {
                "scene_token": scene_meta.get("t0_sample_token", ""),
                "location": scene_meta.get("location", ""),
                "nuscenes_scene": scene_meta.get("nuscenes_scene", ""),
            }
        elif dataset == "argoverse2":
            scene_ref = {
                "scene_uuid": scene_meta.get("scene_uuid", ""),
                "t0_timestamp_ns": scene_meta.get("t0_timestamp_ns", ""),
            }
        elif dataset == "openscene":
            scene_ref = {
                "log_name": scene_meta.get("log_name", ""),
                "t0_frame_idx": scene_meta.get("t0_frame_idx", ""),
                "map_location": scene_meta.get("map_location", ""),
            }

        np_configs[scene_id].append({
            "question": meta["question"],
            "entity_id": entity_id,
            "entity_uuid": uuid_map.get(entity_id, ""),
            "temporal_anchor": meta.get("temporal_anchor", "t0"),
            "config": gen_data,
            "dataset": dataset,
            "scene_ref": scene_ref,
            "timestamp": scene_meta.get("timestamp", scene_meta.get("t0_timestamp_ns", "")),
            "uuid_map": uuid_map,
        })

    if np_configs:
        # Per-scene
        for scene_id, configs in np_configs.items():
            out_path = qa_dir / scene_id / "nuplan_configs.json"
            with open(out_path, "w") as f_out:
                json.dump(configs, f_out, indent=2)
            print(f"  nuPlan configs: {out_path} ({len(configs)} configs)")

        # Global consolidated
        all_configs = []
        for scene_id, configs in np_configs.items():
            for c in configs:
                c["scene_id"] = scene_id
                all_configs.append(c)
        global_path = qa_dir / "nuplan_config.json"
        with open(global_path, "w") as f_out:
            json.dump(all_configs, f_out, indent=2)
        print(f"  Global nuPlan manifest: {global_path} ({len(all_configs)} configs)")

    return success_count, error_count


# ── Stage 4np: nuPlan Config Results ──────────────────────────────────────

def process_stage4np_results(results_file: Path, qa_dir: Path, records_dir: Path,
                              manifest_file: Path, parse_fn=None) -> Tuple[int, int]:
    """Process Stage 4np nuPlan config generation results.

    Saves intervention configs as nuplan_config.json per scene.
    Also builds a consolidated manifest with uuid_map data.
    """
    if parse_fn is None:
        parse_fn = parse_batch_response
    print("Processing Stage 4np results...")

    if not manifest_file.exists():
        print(f"ERROR: manifest not found: {manifest_file}")
        return 0, 0

    manifest = json.loads(manifest_file.read_text())
    success_count = error_count = 0
    errors = defaultdict(list)
    all_configs = []

    with open(results_file) as f:
        for line in f:
            custom_id, result, error = parse_fn(line)
            if error:
                errors[error].append(custom_id)
                error_count += 1
                continue

            meta = manifest.get(custom_id)
            if not meta:
                print(f"  WARNING: no manifest entry for '{custom_id}' — skipping")
                error_count += 1
                continue

            scene_id = meta["scene_id"]
            scene_dir = Path(meta["scene_dir"])
            entity_id = meta.get("entity_id", "")

            # Load uuid_map and meta.json for scene context
            tp0_dir = scene_dir / "Tp0p0"
            uuid_map = {}
            uuid_f = tp0_dir / "uuid_map.json"
            if uuid_f.exists():
                uuid_map = json.loads(uuid_f.read_text())

            scene_meta = {}
            meta_f = scene_dir / "meta.json"
            if meta_f.exists():
                scene_meta = json.loads(meta_f.read_text())

            # Build full config with scene context
            config = {
                "scene": {
                    "dataset": scene_meta.get("dataset", ""),
                    "scene_id": scene_meta.get("nuscenes_scene", scene_meta.get("scene", scene_id)),
                    "sample_idx": scene_meta.get("sample_idx", 0),
                    "t0_token": scene_meta.get("t0_sample_token", ""),
                    "map_location": scene_meta.get("location", ""),
                    "timestamp": scene_meta.get("timestamp", 0),
                },
                "uuid_map": uuid_map,
                "intervention": result,  # LLM-generated intervention config
                "source": {
                    "type": meta.get("q_type", ""),
                    "entity_id": entity_id,
                    "entity_uuid": uuid_map.get(entity_id, ""),
                    "question": meta.get("question", ""),
                    "answer": meta.get("cf_answer", ""),
                },
            }

            # Save per-scene config
            out_dir = qa_dir / scene_id / "nuplan_configs"
            out_dir.mkdir(parents=True, exist_ok=True)
            config_file = out_dir / f"{entity_id}_{meta.get('q_type', 'intervention')}.json"
            with open(config_file, "w") as f_out:
                json.dump(config, f_out, indent=2)

            all_configs.append(config)
            success_count += 1
            print(f"  Saved {scene_id} / {entity_id} ({meta.get('q_type', '')})")

    _print_summary(success_count, error_count, errors)

    # Save consolidated manifest
    consolidated_path = qa_dir / "nuplan_config.json"
    with open(consolidated_path, "w") as f_out:
        json.dump(all_configs, f_out, indent=2)
    print(f"  Consolidated manifest: {consolidated_path} ({len(all_configs)} configs)")

    return success_count, error_count


# ── Stage 5: Agent Intervention LLM Ego Trajectory ─────────────────────────

def process_stage5_results(results_file: Path, qa_dir: Path,
                           manifest_file: Path, parse_fn=None) -> Tuple[int, int]:
    """Process Stage 5 LLM ego trajectory results.

    Saves llm_ego_trajectory.json in the cf_trajectories subdirectory
    identified by the manifest's cf_subdir field.
    """
    if parse_fn is None:
        parse_fn = parse_batch_response

    print("Processing Stage 5 results...")

    if not manifest_file.exists():
        print(f"ERROR: manifest not found: {manifest_file}")
        return 0, 0
    manifest = json.loads(manifest_file.read_text())

    success_count = error_count = 0
    errors = defaultdict(list)

    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            custom_id, result, error = parse_fn(line)
            if error:
                errors[error].append(custom_id)
                error_count += 1
                continue

            meta = manifest.get(custom_id)
            if not meta:
                print(f"  WARNING: no manifest entry for '{custom_id}' — skipping")
                error_count += 1
                continue

            cf_subdir = Path(meta["cf_subdir"])
            cf_subdir.mkdir(parents=True, exist_ok=True)

            # Attach metadata from manifest
            if isinstance(result, dict):
                result["_apply_at_t"] = meta.get("apply_at_t", 0.0)
                result["_action_type"] = meta.get("action_type", "")
                result["_question"] = meta.get("question", "")
            out_file = cf_subdir / "llm_ego_trajectory.json"
            with open(out_file, "w") as f_out:
                json.dump(result, f_out, indent=2)

            scene_id = meta.get("scene_id", custom_id)
            print(f"  {scene_id}: saved llm_ego_trajectory.json")
            success_count += 1

    _print_summary(success_count, error_count, errors)
    return success_count, error_count


# ── Stage verify: Graph Verification ─────────────────────────────────────

def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def process_verify_results(
    results_file: Path,
    output_dir: Path,
    parse_fn=None,
    id_map: Dict[str, str] = None,
) -> Tuple[int, int]:
    """Process Stage verify (graph verification) results.

    custom_id format: {scene_id}_verify

    Output per scene: {output_dir}/verification/{scene_id}_verification.json

    After all scenes are written, calls generate_verification_report() to
    produce {output_dir}/verification_report.json with flagged scenes.
    """
    if parse_fn is None:
        parse_fn = parse_batch_response
    if id_map is None:
        id_map = {}

    print("Processing Stage verify results...")
    verification_dir = output_dir
    verification_dir.mkdir(parents=True, exist_ok=True)

    success_count = error_count = 0
    errors: Dict[str, List[str]] = defaultdict(list)
    parse_failures: List[str] = []

    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            custom_id, result, error = parse_fn(line)
            if error:
                errors[str(error)].append(custom_id)
                error_count += 1
                continue

            scene_id = _resolve_scene_id(custom_id, "verify", id_map)

            # Validate Haiku produced a parseable verification JSON
            if not result.get("overall_verdict"):
                fallback = {
                    "_parse_failure": True,
                    "overall_verdict": "review_needed",
                    "flags": ["Haiku response could not be parsed as valid verification JSON"],
                    "node_verdicts": [],
                    "edge_verdicts": [],
                    "distractor_verdicts": [],
                    "confidence": "low",
                    "_meta": {
                        "scene_id": scene_id,
                        "processed_at": _utc_now(),
                        "usage": result.get("_usage", {}),
                    },
                }
                out = verification_dir / f"{scene_id}_verification.json"
                with open(out, "w") as f_out:
                    json.dump(fallback, f_out, indent=2)
                parse_failures.append(scene_id)
                error_count += 1
                continue

            result["_meta"] = {
                "scene_id": scene_id,
                "processed_at": _utc_now(),
                "usage": result.pop("_usage", {}),
            }

            out = verification_dir / f"{scene_id}_verification.json"
            with open(out, "w") as f_out:
                json.dump(result, f_out, indent=2)

            verdict = result["overall_verdict"]
            n_flags = len(result.get("flags", []))
            print(f"  {scene_id}: {verdict}  ({n_flags} flag(s))")
            success_count += 1

    _print_summary(success_count, error_count, errors)
    if parse_failures:
        print(f"\n  Parse failures ({len(parse_failures)}) — marked review_needed:")
        for sid in parse_failures[:5]:
            print(f"    {sid}")
        if len(parse_failures) > 5:
            print(f"    ... and {len(parse_failures) - 5} more")

    generate_verification_report(output_dir)
    return success_count, error_count


def generate_verification_report(output_dir: Path) -> Path:
    """Scan all *_verification.json files in output_dir and produce verification_report.json.

    Reads:  {output_dir}/*_verification.json
    Writes: {output_dir}/verification_report.json

    Report structure:
        summary:              total/passed/review_needed/parse_failures/pass_rate/generated_at
        verdict_breakdown:    node/edge/distractor agree/disagree/uncertain counts
        confidence_breakdown: high/medium/low counts
        flagged_scenes:       sorted list of scenes needing review with flag details
    """
    print("\nGenerating verification report...")

    total = passed = review_needed = parse_failures = 0
    node_v: Dict[str, int] = {"agree": 0, "disagree": 0, "uncertain": 0}
    edge_v: Dict[str, int] = {"agree": 0, "disagree": 0, "uncertain": 0}
    dist_v: Dict[str, int] = {"agree": 0, "disagree": 0, "uncertain": 0}
    conf_v: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    flagged_scenes = []

    for vfile in sorted(output_dir.glob("*_verification.json")):
        data = json.loads(vfile.read_text())
        total += 1
        is_parse_fail = data.get("_parse_failure", False)
        if is_parse_fail:
            parse_failures += 1

        verdict = data.get("overall_verdict", "review_needed")
        if verdict == "pass":
            passed += 1
        else:
            review_needed += 1
            meta = data.get("_meta", {})
            scene_id = meta.get("scene_id") or vfile.stem.removesuffix("_verification")
            n_nd = sum(1 for v in data.get("node_verdicts",      []) if v.get("verdict") == "disagree")
            n_ed = sum(1 for v in data.get("edge_verdicts",      []) if v.get("verdict") == "disagree")
            n_dd = sum(1 for v in data.get("distractor_verdicts",[]) if v.get("verdict") == "disagree")
            flagged_scenes.append({
                "scene_id":        scene_id,
                "overall_verdict": "review_needed",
                "flags":           data.get("flags", []),
                "n_node_disagree": n_nd,
                "n_edge_disagree": n_ed,
                "n_dist_disagree": n_dd,
                "confidence":      data.get("confidence", "unknown"),
                "parse_failure":   is_parse_fail,
            })

        for v in data.get("node_verdicts", []):
            k = v.get("verdict", "uncertain")
            node_v[k] = node_v.get(k, 0) + 1
        for v in data.get("edge_verdicts", []):
            k = v.get("verdict", "uncertain")
            edge_v[k] = edge_v.get(k, 0) + 1
        for v in data.get("distractor_verdicts", []):
            k = v.get("verdict", "uncertain")
            dist_v[k] = dist_v.get(k, 0) + 1

        conf = data.get("confidence", "low")
        conf_v[conf] = conf_v.get(conf, 0) + 1

    # Sort: parse failures first, then by total disagree count descending
    flagged_scenes.sort(key=lambda x: (
        not x["parse_failure"],
        -(x["n_node_disagree"] + x["n_edge_disagree"] + x["n_dist_disagree"]),
    ))

    pass_rate = passed / total if total > 0 else 0.0
    report = {
        "summary": {
            "total_scenes":   total,
            "passed":         passed,
            "review_needed":  review_needed,
            "parse_failures": parse_failures,
            "pass_rate":      round(pass_rate, 4),
            "generated_at":   _utc_now(),
        },
        "verdict_breakdown": {
            "node_agree":     node_v["agree"],   "node_disagree":  node_v["disagree"],  "node_uncertain":  node_v["uncertain"],
            "edge_agree":     edge_v["agree"],   "edge_disagree":  edge_v["disagree"],  "edge_uncertain":  edge_v["uncertain"],
            "dist_agree":     dist_v["agree"],   "dist_disagree":  dist_v["disagree"],  "dist_uncertain":  dist_v["uncertain"],
        },
        "confidence_breakdown": conf_v,
        "flagged_scenes": flagged_scenes,
    }

    report_path = output_dir / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Total scenes verified:  {total}")
    print(f"  Passed:                 {passed}  ({pass_rate*100:.1f}%)")
    print(f"  Review needed:          {review_needed}")
    print(f"  Parse failures:         {parse_failures}")
    print(f"  Report → {report_path}")
    return report_path


# ── Helpers ───────────────────────────────────────────────────────────────

def _print_summary(success: int, errors: int, error_map: Dict = None):
    print(f"  Success: {success}")
    print(f"  Errors:  {errors}")
    if error_map:
        print("\n  Error breakdown:")
        for msg, ids in error_map.items():
            print(f"    {msg}: {len(ids)} scene(s)")
            if len(ids) <= 3:
                print(f"      {', '.join(ids)}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    _cfg = _load_config() or {}
    parser = argparse.ArgumentParser(description="Process batch API results")
    parser.add_argument("--file",     required=True, help="Results JSONL file")
    parser.add_argument("--stage",    required=True,
                        choices=["1", "2", "3a", "3b", "3c", "4cf", "4ac", "4np", "5", "verify"])
    parser.add_argument("--output",   required=True,
                        help="Output dir (graphs for 1-2, qa_results for 3a-3b-4cf-4ac)")
    parser.add_argument("--records",  help="Records dir (required for stage 4cf/4ac)")
    parser.add_argument("--manifest", help="Manifest JSON written by prepare_batches (required for stage 4cf/4ac)")
    parser.add_argument("--provider", choices=["openai", "claude", "gemini"],
                        default=_cfg.get("provider"), help="LLM provider")
    args = parser.parse_args()

    results_file = Path(args.file)
    output_dir   = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        sys.exit(1)

    # Select parse function based on provider
    parse_fn = parse_claude_response if args.provider == "claude" else parse_batch_response

    # Load ID map for sanitized custom_id resolution
    # id_map files are in the batch dir (parent of batch_results/)
    batch_jobs_dir = results_file.parent.parent
    id_map = _load_id_map(batch_jobs_dir) if batch_jobs_dir.exists() else {}

    if args.stage == "1":
        process_stage1_results(results_file, output_dir, parse_fn=parse_fn, id_map=id_map)
    elif args.stage == "2":
        process_stage2_results(results_file, output_dir, parse_fn=parse_fn, id_map=id_map)
    elif args.stage == "3a":
        process_stage3a_results(results_file, output_dir, parse_fn=parse_fn, id_map=id_map)
    elif args.stage == "3b":
        process_stage3b_results(results_file, output_dir, parse_fn=parse_fn, id_map=id_map)
    elif args.stage == "3c":
        process_stage3c_results(results_file, output_dir, parse_fn=parse_fn, id_map=id_map)
    elif args.stage in ("4cf", "4ac"):
        if not args.records:
            print(f"ERROR: --records required for stage {args.stage}"); sys.exit(1)
        if not args.manifest:
            print(f"ERROR: --manifest required for stage {args.stage}"); sys.exit(1)
        prompt_type = args.stage[1:]  # "cf" or "ac"
        process_stage4_results(
            results_file, output_dir, Path(args.records),
            Path(args.manifest), prompt_type, parse_fn=parse_fn,
        )
    elif args.stage == "4np":
        if not args.records:
            print("ERROR: --records required for stage 4np"); sys.exit(1)
        if not args.manifest:
            print("ERROR: --manifest required for stage 4np"); sys.exit(1)
        process_stage4np_results(
            results_file, output_dir, Path(args.records),
            Path(args.manifest), parse_fn=parse_fn,
        )

    elif args.stage == "5":
        if not args.manifest:
            print("ERROR: --manifest required for stage 5"); sys.exit(1)
        process_stage5_results(
            results_file, output_dir,
            Path(args.manifest), parse_fn=parse_fn,
        )

    elif args.stage == "verify":
        process_verify_results(
            results_file, output_dir, parse_fn=parse_fn, id_map=id_map
        )


if __name__ == "__main__":
    main()
