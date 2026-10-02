#!/usr/bin/env python3
"""
CausalDriveBench — Evaluation Output Reorganizer, Verifier & Report Generator
================================================================================

Collects evaluation outputs submitted by different teams via a Google Drive zip,
normalises directory structure, runs sanity checks, regenerates reports, and
emits markdown + LaTeX tables ready for the paper.

Pipeline phases
---------------
1. Extract    Recursively unzip the Google Drive archive and all nested zips.
2. Discover   Scan extracted tree; identify model run directories.
3. Reorganize Copy to CDB_OUTPUTS_DIR/<model_id>_<YYYYMMDD_HHMMSS>/causal_nuscenes/
4. Sanity     Validate every reorganised run against the benchmark manifest.
5. Reports    Regenerate Tier 2 + Tier 3 JSON reports from existing Tier 1 samples.
6. HTML       Generate HTML reports for every run with a valid run-level report.
7. Markdown   Write per-model sanity summaries + combined statistics to docs/.
8. LaTeX      Emit Table 1 (VQA accuracy) and Table 2 (trajectory ADE) values.

Usage
-----
    cd /datadrive/code/ssk/neurips-dataset
    python evaluation/scripts/reorganize_and_report.py            # full pipeline
    python evaluation/scripts/reorganize_and_report.py --dry-run  # discover only
    python evaluation/scripts/reorganize_and_report.py --skip-extract
    python evaluation/scripts/reorganize_and_report.py --skip-extract --skip-reorganize

Parameters (edit defaults below or pass via CLI)
-------------------------------------------------
    --zip-path       Path to the top-level Google Drive zip file
    --extract-dir    Scratch directory for extracted content
    --outputs-dir    CDB_OUTPUTS_DIR target (reorganised outputs land here)
    --reports-dir    CDB_REPORTS_DIR target (HTML reports land here)
    --bench-root     Benchmark root (causal_drive_bench_*)
    --docs-dir       Directory for markdown/LaTeX summaries
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import traceback
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("reorganize")

# ---------------------------------------------------------------------------
# Default paths  (override via CLI)
# ---------------------------------------------------------------------------

DEFAULT_ZIP_PATH    = Path("/datadrive/datasets/cdb_all_models_20260429/Evaluations-20260429T154157Z-3-001.zip")
DEFAULT_EXTRACT_DIR = Path("/datadrive/datasets/cdb_all_models_20260429/extracted")
DEFAULT_OUTPUTS_DIR = Path("/datadrive/datasets/cdb_outputs_20260429")
DEFAULT_REPORTS_DIR = Path("/datadrive/datasets/cdb_reports_20260429")
DEFAULT_BENCH_ROOT  = Path("/datadrive/datasets/causal_drive_bench_20260429")
DEFAULT_DOCS_DIR    = Path("/datadrive/code/ssk/neurips-dataset/evaluation/docs/eval_20260429")

EVAL_REPO_ROOT = Path("/datadrive/code/ssk/neurips-dataset")

# Canonical causal dataset names (standard output structure)
CAUSAL_DATASET_NAMES = ["causal_nuscenes", "causal_argoverse2", "causal_openscene"]

# Non-standard dataset dir names used by external teams → canonical name
# These are renamed to the canonical form during reorganisation.
DATASET_DIR_ALIASES: Dict[str, str] = {
    "nuscenes_full_ice":   "causal_nuscenes",
    "nuscenes_qa_test_1":  "causal_nuscenes",
    "nuscenes_full":       "causal_nuscenes",
}

# All recognised dataset dir names (canonical + aliases)
ALL_DATASET_DIR_NAMES = CAUSAL_DATASET_NAMES + list(DATASET_DIR_ALIASES.keys())

# ---------------------------------------------------------------------------
# Model display-name registry
# Maps internal model_id → (display_name, category)
# Category: "driving_vla" | "general_vlm" | "finetuned"
# Order determines row order in LaTeX tables.
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    # Driving VLAs (in paper order)
    "omni_q":          ("OmniQ (OmniDrive)", "driving_vla"),  # omni_q_merged + filtered variants
    "wisead":          ("WiseAD",       "driving_vla"),
    "wisead_8img":     ("WiseAD",       "driving_vla"),
    "safeauto":        ("SafeAuto",     "driving_vla"),
    "simlingo":        ("SimLingo",     "driving_vla"),
    "orion":           ("Orion",        "driving_vla"),
    "drivemoe":        ("DriveMoE",     "driving_vla"),
    "impromptu_vla":   ("ImpromptuVLA", "driving_vla"),
    "recogdrive":      ("RecogDrive",   "driving_vla"),
    "opendrivevla":    ("OpenREAD",     "driving_vla"),
    "openread":        ("OpenREAD",     "driving_vla"),
    "alpamayo_1_5":    ("AlpaMayo-1.5", "driving_vla"),
    # General VLMs
    "cosmos_reason":   ("Cosmos-Reason",   "general_vlm"),
    "qwen25vl":        ("Qwen2.5-VL",      "general_vlm"),
    "intern_vl3_5_8b": ("InternVL-3.5-8B", "general_vlm"),
    "qwen3_vl_8b":     ("Qwen3-VL-8B",     "general_vlm"),
    "gemma4_e4b":      ("Gemma-VL (E4B)",   "general_vlm"),
    "gemma4_31b":      ("Gemma-VL (31B)",   "general_vlm"),
    "gemma4_e2b":      ("Gemma-VL (E2B)",   "general_vlm"),
    "gemma4_26b_a4b":  ("Gemma-VL (26B-A4B)", "general_vlm"),
    "gemma4":          ("Gemma-VL",         "general_vlm"),
    "doe":             ("Doe-1",            "general_vlm"),
    # Fine-tuned
    "qwen3vl_ft":      ("Qwen3-VL (fine-tuned)", "finetuned"),
}

# Known model ID substrings for fuzzy matching (longest first wins)
KNOWN_MODEL_IDS = sorted(MODEL_REGISTRY.keys(), key=len, reverse=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class DiscoveredRun(NamedTuple):
    """A model run directory discovered in the extracted archive."""
    source_dir: Path               # absolute path to the run dir inside extracted/
    model_id: str                  # inferred model ID (e.g. "alpamayo_1_5")
    run_name: str                  # full run name (e.g. "alpamayo_1_5_20260428_081120")
    timestamp: str                 # YYYYMMDD_HHMMSS string
    datasets: List[str]            # canonical dataset names (e.g. ["causal_nuscenes"])
    dataset_src_names: List[str]   # original dataset dir names before aliasing
    has_sample_reports: bool       # any Tier-1 report.json found?
    has_run_report: bool           # run-level report.json found?
    has_jsonl: bool                # any outputs.jsonl found?
    notes: List[str]               # any warnings / issues during discovery


class ReorganizedRun(NamedTuple):
    """A model run that has been copied to CDB_OUTPUTS_DIR."""
    dest_dir: Path            # path in CDB_OUTPUTS_DIR
    model_id: str
    run_name: str
    datasets: List[str]
    had_sample_reports: bool
    had_run_report: bool


# ---------------------------------------------------------------------------
# Phase 1 — Recursive extraction
# ---------------------------------------------------------------------------

def extract_recursive(zip_path: Path, extract_dir: Path) -> None:
    """
    Recursively extract *zip_path* and every nested zip found inside it.
    All content lands under *extract_dir*.
    """
    logger.info("=" * 72)
    logger.info("PHASE 1: EXTRACT")
    logger.info("  Source zip : %s", zip_path)
    logger.info("  Extract dir: %s", extract_dir)

    if not zip_path.exists():
        logger.error("Zip file not found: %s", zip_path)
        sys.exit(1)

    extract_dir.mkdir(parents=True, exist_ok=True)

    # Queue: (zip_file_path, target_directory)
    queue: List[Tuple[Path, Path]] = [(zip_path, extract_dir)]
    processed_zips: List[Path] = []
    total_extracted = 0

    while queue:
        current_zip, target = queue.pop(0)
        logger.info("  Extracting: %s  →  %s", current_zip, target)
        target.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(current_zip, "r") as zf:
                names = zf.namelist()
                logger.info("    %d entries in archive", len(names))
                zf.extractall(target)
                total_extracted += len(names)
        except zipfile.BadZipFile as exc:
            logger.warning("  BAD ZIP (skipped): %s — %s", current_zip, exc)
            continue

        processed_zips.append(current_zip)

        # Discover any nested zips (but not the one we just extracted from, if it
        # was itself inside extract_dir already)
        for nested in sorted(target.rglob("*.zip")):
            if nested == current_zip:
                continue
            if any(str(nested).startswith(str(p.parent)) and nested.name == p.name
                   for p in processed_zips):
                continue
            logger.info("  Found nested zip: %s", nested)
            # Extract into a sibling directory named after the zip stem
            nested_target = nested.parent / nested.stem
            queue.append((nested, nested_target))

    logger.info("PHASE 1 DONE: extracted %d total entries from %d zip(s)",
                total_extracted, len(processed_zips))


# ---------------------------------------------------------------------------
# Phase 2 — Discovery helpers
# ---------------------------------------------------------------------------

def _infer_model_id(name: str) -> str:
    """
    Infer a canonical model_id from a directory name by stripping timestamps
    and matching against the known model registry.

    Examples
    --------
    "alpamayo_1_5_20260428_081120" → "alpamayo_1_5"
    "gemma4_e4b_run1"              → "gemma4_e4b"
    "OmniDrive_v2_output"          → "omnidrive"
    "unknown_team_model"           → "unknown_team_model"
    """
    # Normalise to lowercase for matching
    normalised = name.lower().strip()

    # Strip trailing timestamps like _20260428_081120 or _20260428
    normalised = re.sub(r"_\d{8}_\d{6}$", "", normalised)
    normalised = re.sub(r"_\d{8}$", "", normalised)
    # Strip embedded timestamps anywhere (e.g. omni_q_20260423_211724_filtered)
    normalised = re.sub(r"_\d{8}_\d{6}", "", normalised)
    # Strip common suffixes
    normalised = re.sub(r"[_\-](run|output|eval|inference|result|v\d+|final|sample|merged|filtered)$",
                        "", normalised)

    # Try longest-match against known IDs
    for known_id in KNOWN_MODEL_IDS:
        if normalised == known_id:
            return known_id
        if normalised.startswith(known_id + "_") or normalised.endswith("_" + known_id):
            return known_id
        if known_id in normalised:
            return known_id

    return normalised  # return cleaned name even if not in registry


def _extract_timestamp_from_name(name: str) -> Optional[str]:
    """Return 'YYYYMMDD_HHMMSS' if present in *name*, else None."""
    m = re.search(r"(\d{8})_(\d{6})", name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m = re.search(r"(\d{8})", name)
    if m:
        return f"{m.group(1)}_000000"
    return None


def _extract_timestamp_from_report(report_path: Path) -> Optional[str]:
    """Parse 'generated_at' from a report.json → 'YYYYMMDD_HHMMSS'."""
    try:
        with open(report_path) as fh:
            data = json.load(fh)
        ts_str = data.get("generated_at", "")
        if ts_str:
            # ISO format: 2026-04-28T08:11:20.123456+00:00
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return dt.strftime("%Y%m%d_%H%M%S")
    except Exception:
        pass
    return None


def _extract_timestamp_from_outputs(dataset_dir: Path) -> Optional[str]:
    """Sample the first outputs.jsonl or trajectory_outputs.jsonl and read its 'timestamp' field."""
    for pattern in ("outputs.jsonl", "trajectory_outputs.jsonl"):
        for f in dataset_dir.rglob(pattern):
            try:
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rec = json.loads(line)
                            ts_str = rec.get("timestamp", "")
                            if ts_str:
                                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                return dt.strftime("%Y%m%d_%H%M%S")
            except Exception:
                continue
    return None


def _find_run_dirs_in_tree(root: Path) -> List[Path]:
    """
    Heuristic scan to find 'run directories' inside *root*.

    Strategy (in order of confidence):
    1. Any directory containing a 'report.json' with level='run'.
    2. The parent of any directory named 'causal_nuscenes', 'causal_argoverse2',
       or 'causal_openscene'.
    3. If neither found, any directory that has 'outputs.jsonl' files at depth ≥ 3.

    Returns deduplicated list of run_dirs, sorted.
    """
    candidates: set = set()

    # Strategy 1 — run-level report.json
    for rp in root.rglob("report.json"):
        try:
            with open(rp) as fh:
                data = json.load(fh)
            if data.get("level") == "run":
                candidates.add(rp.parent)
                logger.debug("  [discover] run-level report.json → %s", rp.parent)
        except Exception:
            pass

    # Strategy 2 — causal dataset directories (canonical + aliased names)
    for ds_name in ALL_DATASET_DIR_NAMES:
        for ds_dir in root.rglob(ds_name):
            if ds_dir.is_dir():
                run_dir = ds_dir.parent
                candidates.add(run_dir)
                logger.debug("  [discover] dataset dir '%s' parent → %s", ds_name, run_dir)

    # Strategy 3 — outputs.jsonl at depth ≥ 3
    if not candidates:
        logger.info("  No run dirs found via strategies 1+2; falling back to outputs.jsonl scan")
        for op in root.rglob("outputs.jsonl"):
            # sample_dir / scene_dir / dataset_dir / run_dir
            parts = op.parts
            if len(parts) >= len(root.parts) + 4:
                run_dir = Path(*parts[: len(root.parts) + 1])
                candidates.add(run_dir)

    # Deduplicate: if A is an ancestor of B, keep only A
    sorted_candidates = sorted(candidates, key=lambda p: len(p.parts))
    deduplicated: List[Path] = []
    for cand in sorted_candidates:
        if not any(cand != kept and str(cand).startswith(str(kept) + os.sep)
                   for kept in deduplicated):
            deduplicated.append(cand)

    return deduplicated


def discover_run_dirs(extract_dir: Path) -> List[DiscoveredRun]:
    """
    Scan *extract_dir* and return one DiscoveredRun per model run found.
    """
    logger.info("=" * 72)
    logger.info("PHASE 2: DISCOVER")
    logger.info("  Scanning: %s", extract_dir)

    run_dirs = _find_run_dirs_in_tree(extract_dir)
    logger.info("  Found %d candidate run directories", len(run_dirs))

    discovered: List[DiscoveredRun] = []
    run_name_counts: Dict[str, dict] = {}  # base_run_name → {has_jsonl}

    for run_dir in run_dirs:
        notes: List[str] = []

        # --- Model ID inference ---
        raw_name = run_dir.name
        model_id = _infer_model_id(raw_name)

        # Check if there's a run-level report.json that overrides inference
        run_report_path = run_dir / "report.json"
        has_run_report = False
        if run_report_path.exists():
            try:
                with open(run_report_path) as fh:
                    rr = json.load(fh)
                if rr.get("level") == "run":
                    has_run_report = True
                    if rr.get("run_name"):
                        # run_name in report may be more accurate
                        rpt_model_id = _infer_model_id(rr["run_name"])
                        if rpt_model_id != model_id:
                            notes.append(
                                f"Model ID from report ({rpt_model_id}) differs from "
                                f"folder name inference ({model_id}); using report value"
                            )
                            model_id = rpt_model_id
            except Exception as exc:
                notes.append(f"Could not parse run report.json: {exc}")

        # --- Datasets (canonical + aliased names) ---
        # Map: canonical_name → source_dir_name
        ds_canonical: List[str] = []
        ds_src_names: List[str] = []
        for ds_name in ALL_DATASET_DIR_NAMES:
            if (run_dir / ds_name).is_dir():
                canonical = DATASET_DIR_ALIASES.get(ds_name, ds_name)
                if canonical not in ds_canonical:
                    ds_canonical.append(canonical)
                    ds_src_names.append(ds_name)

        # --- Check for JSONL artifacts (filter out HTML-only directories) ---
        # Accept VQA outputs.jsonl OR trajectory_outputs.jsonl (trajectory-only models)
        has_jsonl = (
            any(True for _ in run_dir.rglob("outputs.jsonl"))
            or any(True for _ in run_dir.rglob("trajectory_outputs.jsonl"))
        )

        # Skip HTML-only dirs BEFORE incrementing the name counter so that a
        # report-only mirror never steals the base run name from the real JSONL dir.
        if not has_jsonl and not has_run_report:
            logger.info(
                "  SKIP   %-40s  (no JSONL and no run-level report — HTML-only)",
                run_dir.relative_to(extract_dir),
            )
            continue

        # --- Timestamp ---
        timestamp = (
            _extract_timestamp_from_name(raw_name)
            or (has_run_report and _extract_timestamp_from_report(run_report_path))
            or None
        )
        if timestamp is None:
            # Try outputs.jsonl timestamps across all dataset dirs
            for src_ds_name in ds_src_names:
                ds_dir = run_dir / src_ds_name
                if ds_dir.exists():
                    timestamp = _extract_timestamp_from_outputs(ds_dir)
                    if timestamp:
                        break
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            notes.append("Timestamp not found in folder/report; using script execution time")

        # --- Sample reports ---
        has_sample_reports = any(
            True for rp in run_dir.rglob("report.json")
            if rp != run_dir / "report.json"  # exclude run-level
        )

        if model_id not in MODEL_REGISTRY:
            notes.append(f"Model ID '{model_id}' not in registry — add to MODEL_REGISTRY if needed")

        # --- Dedup: if same base_run_name seen before, keep only the richer source ---
        base_run_name = f"{model_id}_{timestamp}"
        if base_run_name in run_name_counts:
            # Already seen — keep whichever has more data (JSONL > report-only)
            prev_has_jsonl = run_name_counts[base_run_name]["has_jsonl"]
            if prev_has_jsonl and not has_jsonl:
                # Previous is richer — skip this one
                logger.info(
                    "  SKIP   %-40s  (duplicate of %s, previous source has JSONL)",
                    run_dir.relative_to(extract_dir), base_run_name,
                )
                continue
            if has_jsonl and not prev_has_jsonl:
                # This one is richer — replace previous entry
                logger.info(
                    "  REPLACING previous %s with richer source: %s",
                    base_run_name, run_dir.relative_to(extract_dir),
                )
                discovered[:] = [dr for dr in discovered if dr.run_name != base_run_name]
            else:
                # Both equal (both have JSONL or both don't) — skip duplicate
                logger.info(
                    "  SKIP   %-40s  (duplicate of %s, same data quality)",
                    run_dir.relative_to(extract_dir), base_run_name,
                )
                continue

        run_name_counts[base_run_name] = {"has_jsonl": has_jsonl}
        run_name = base_run_name

        dr = DiscoveredRun(
            source_dir=run_dir,
            model_id=model_id,
            run_name=run_name,
            timestamp=timestamp,
            datasets=ds_canonical,
            dataset_src_names=ds_src_names,
            has_sample_reports=has_sample_reports,
            has_run_report=has_run_report,
            has_jsonl=has_jsonl,
            notes=notes,
        )
        discovered.append(dr)

        ds_display = ",".join(
            f"{s}→{c}" if s != c else c
            for s, c in zip(ds_src_names, ds_canonical)
        ) or "?"
        logger.info(
            "  FOUND  %-40s  model=%-22s  ts=%s  datasets=%s  jsonl=%s  reports=%s/%s",
            run_dir.relative_to(extract_dir),
            model_id,
            timestamp,
            ds_display,
            "✓" if has_jsonl else "✗",
            "T1✓" if has_sample_reports else "T1✗",
            "T3✓" if has_run_report else "T3✗",
        )
        for note in notes:
            logger.info("         NOTE: %s", note)

    logger.info("PHASE 2 DONE: %d runs discovered", len(discovered))
    return discovered


# ---------------------------------------------------------------------------
# Phase 3 — Reorganise
# ---------------------------------------------------------------------------

def _copy_sample_tree(
    src_dataset_dir: Path,
    dst_dataset_dir: Path,
    dataset_name: str,
) -> int:
    """
    Copy <scene_id>/<sample_id>/ trees from *src* to *dst*, returning file count.
    Only copies files matching known artifact names.
    """
    ARTIFACT_NAMES = {
        "prompts.jsonl",
        "outputs.jsonl",
        "report.json",
        "trajectory_prompts.jsonl",
        "trajectory_outputs.jsonl",
        "trajectory_report.json",
        "trajectory_aggregate.json",
    }
    copied = 0
    for scene_dir in sorted(src_dataset_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for sample_dir in sorted(scene_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            dst_sample = dst_dataset_dir / scene_dir.name / sample_dir.name
            dst_sample.mkdir(parents=True, exist_ok=True)
            for artifact in ARTIFACT_NAMES:
                src_f = sample_dir / artifact
                if src_f.exists():
                    shutil.copy2(src_f, dst_sample / artifact)
                    copied += 1
    return copied


def reorganize_runs(
    discovered: List[DiscoveredRun],
    outputs_dir: Path,
    dry_run: bool,
) -> List[ReorganizedRun]:
    """
    Copy each discovered run to ``outputs_dir/<run_name>/`` in the standard layout.
    """
    logger.info("=" * 72)
    logger.info("PHASE 3: REORGANIZE  (dry_run=%s)", dry_run)
    logger.info("  Target outputs dir: %s", outputs_dir)

    if not dry_run:
        outputs_dir.mkdir(parents=True, exist_ok=True)

    reorganized: List[ReorganizedRun] = []

    for dr in discovered:
        dest_dir = outputs_dir / dr.run_name
        logger.info("")
        logger.info("  ── %s ──", dr.run_name)
        logger.info("    Source : %s", dr.source_dir)
        logger.info("    Dest   : %s", dest_dir)
        ds_display = ", ".join(
            f"{s}→{c}" if s != c else c
            for s, c in zip(dr.dataset_src_names, dr.datasets)
        ) or "(none)"
        logger.info("    Datasets: %s", ds_display)
        logger.info("    Has JSONL: %s", dr.has_jsonl)

        if dry_run:
            logger.info("    [DRY-RUN] Skipping actual copy")
            reorganized.append(ReorganizedRun(
                dest_dir=dest_dir,
                model_id=dr.model_id,
                run_name=dr.run_name,
                datasets=dr.datasets,
                had_sample_reports=dr.has_sample_reports,
                had_run_report=dr.has_run_report,
            ))
            continue

        if dest_dir.exists():
            logger.warning("    Destination already exists — skipping (delete manually to redo)")
            reorganized.append(ReorganizedRun(
                dest_dir=dest_dir,
                model_id=dr.model_id,
                run_name=dr.run_name,
                datasets=dr.datasets,
                had_sample_reports=dr.has_sample_reports,
                had_run_report=dr.has_run_report,
            ))
            continue

        copied_files = 0
        effective_datasets: List[str] = []

        if dr.datasets:
            # Copy each dataset dir (renaming aliases to canonical names)
            for src_ds_name, canonical_ds_name in zip(dr.dataset_src_names, dr.datasets):
                src_ds = dr.source_dir / src_ds_name
                dst_ds = dest_dir / canonical_ds_name
                n = _copy_sample_tree(src_ds, dst_ds, canonical_ds_name)
                copied_files += n
                logger.info("    Copied %d artifact files → %s", n, dst_ds)
                effective_datasets.append(canonical_ds_name)

                # Copy dataset-level report.json if it exists
                src_ds_report = src_ds / "report.json"
                if src_ds_report.exists():
                    try:
                        with open(src_ds_report) as fh:
                            ds_rpt = json.load(fh)
                        if ds_rpt.get("level") == "dataset":
                            shutil.copy2(src_ds_report, dst_ds / "report.json")
                            logger.info("    Copied dataset report.json")
                    except Exception:
                        pass

                # Copy dataset-level trajectory_aggregate.json if it exists.
                # Also look in a sibling cdb_results dir (some teams submit raw
                # outputs in cdb_outputs and processed aggregates in cdb_results).
                traj_agg_candidates = [src_ds / "trajectory_aggregate.json"]
                sibling_results = dr.source_dir.parent.parent / "cdb_results" / dr.source_dir.name
                if sibling_results.is_dir():
                    for ds_alias in (src_ds_name, canonical_ds_name):
                        traj_agg_candidates.append(
                            sibling_results / ds_alias / "trajectory_aggregate.json"
                        )
                for traj_agg_src in traj_agg_candidates:
                    if traj_agg_src.exists():
                        dst_ds.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(traj_agg_src, dst_ds / "trajectory_aggregate.json")
                        logger.info("    Copied trajectory_aggregate.json (from %s)", traj_agg_src.parent)
                        break
        else:
            # Fallback: treat run dir itself as scene/sample tree under causal_nuscenes
            logger.warning("    No dataset dir found; attempting flat copy under causal_nuscenes/")
            dst_ds = dest_dir / "causal_nuscenes"
            n = _copy_sample_tree(dr.source_dir, dst_ds, "causal_nuscenes")
            copied_files += n
            effective_datasets = ["causal_nuscenes"]
            logger.info("    Flat-copy: %d artifact files → %s", n, dst_ds)

        # Copy run-level report.json if it exists
        src_run_report = dr.source_dir / "report.json"
        if src_run_report.exists():
            try:
                with open(src_run_report) as fh:
                    rr = json.load(fh)
                if rr.get("level") == "run":
                    shutil.copy2(src_run_report, dest_dir / "report.json")
                    logger.info("    Copied run-level report.json")
            except Exception:
                pass

        logger.info("    Total copied: %d files", copied_files)

        reorganized.append(ReorganizedRun(
            dest_dir=dest_dir,
            model_id=dr.model_id,
            run_name=dr.run_name,
            datasets=effective_datasets or ["causal_nuscenes"],
            had_sample_reports=dr.has_sample_reports,
            had_run_report=dr.has_run_report,
        ))

    logger.info("")
    logger.info("PHASE 3 DONE: %d runs reorganized to %s", len(reorganized), outputs_dir)
    return reorganized


# ---------------------------------------------------------------------------
# Phase 4 — Sanity check
# ---------------------------------------------------------------------------

def run_sanity_checks(
    outputs_dir: Path,
    bench_root: Optional[Path],
) -> Dict[str, dict]:
    """
    Run sanity check analysis for every run directory in *outputs_dir*.
    Returns mapping run_name → analysis dict (same format as sanity_check.analyse_run).
    """
    logger.info("=" * 72)
    logger.info("PHASE 4: SANITY CHECK")
    logger.info("  Outputs dir : %s", outputs_dir)
    logger.info("  Bench root  : %s", bench_root or "(none — counts only)")

    # Import from repo — add repo root to sys.path
    _ensure_repo_on_path()

    from evaluation.sanity_check import analyse_run  # type: ignore

    results: Dict[str, dict] = {}

    run_dirs = sorted(d for d in outputs_dir.iterdir() if d.is_dir())
    logger.info("  Analysing %d run directories…", len(run_dirs))

    for run_dir in run_dirs:
        logger.info("  └─ %s", run_dir.name)
        try:
            analysis = analyse_run(run_dir, bench_root)
            results[run_dir.name] = analysis

            # Log brief summary per dataset
            for ds_name, s in analysis.get("datasets", {}).items():
                n_complete = s["samples_complete_vqa"]
                bench_qa   = s["bench_qa_samples"]
                pct        = s["pct_samples_complete"]
                n_issues   = len(s["issues"])
                logger.info(
                    "     %-30s  complete=%d/%s (%s)  issues=%d",
                    ds_name, n_complete, bench_qa, pct, n_issues,
                )
        except Exception as exc:
            logger.error("  ERROR analysing %s: %s", run_dir.name, exc)
            logger.debug(traceback.format_exc())
            results[run_dir.name] = {"run_dir": str(run_dir), "model_run": run_dir.name,
                                      "error": str(exc), "datasets": {}}

    logger.info("PHASE 4 DONE: %d runs analysed", len(results))
    return results


# ---------------------------------------------------------------------------
# Phase 5 — Report regeneration
# ---------------------------------------------------------------------------

def regenerate_reports(
    reorganized: List[ReorganizedRun],
    schema_version: str = "1.0",
) -> None:
    """
    For each reorganised run, regenerate Tier 2 (dataset) and Tier 3 (run)
    reports from existing Tier 1 sample reports.

    Runs that have no Tier-1 sample reports are skipped with a warning.
    """
    logger.info("=" * 72)
    logger.info("PHASE 5: REPORT REGENERATION")

    _ensure_repo_on_path()

    from evaluation.common.report_generator import (  # type: ignore
        generate_dataset_report,
        generate_run_report,
    )

    ENABLED_METRICS = [
        "overall", "per_qa_type", "per_question_type",
        "per_graph_structure", "per_rung", "answer_stats",
        "binary_metrics", "reasoning_score",
    ]

    for rr in reorganized:
        logger.info("")
        logger.info("  ── %s ──", rr.run_name)

        if not rr.dest_dir.exists():
            logger.warning("    Destination dir missing — skipping: %s", rr.dest_dir)
            continue

        # Check for any Tier-1 sample reports
        sample_reports = list(rr.dest_dir.rglob("*/*/report.json"))
        sample_reports = [
            p for p in sample_reports
            if _is_sample_report(p)
        ]

        if not sample_reports:
            logger.warning(
                "    No Tier-1 sample reports found — cannot regenerate Tier 2/3. "
                "Run the model-specific postprocessor first."
            )
            continue

        logger.info("    Found %d sample reports", len(sample_reports))

        # Tier 2: dataset-level reports
        dataset_dirs = [d for d in rr.dest_dir.iterdir() if d.is_dir()]
        for ds_dir in sorted(dataset_dirs):
            ds_name = ds_dir.name
            ds_report_path = ds_dir / "report.json"
            logger.info("    Regenerating dataset report: %s", ds_name)
            try:
                generate_dataset_report(
                    run_name=rr.run_name,
                    dataset=ds_name,
                    dataset_dir=ds_dir,
                    report_path=ds_report_path,
                    schema_version=schema_version,
                    enabled_metrics=ENABLED_METRICS,
                )
            except Exception as exc:
                logger.error("    ERROR generating dataset report for %s: %s", ds_name, exc)
                logger.debug(traceback.format_exc())

        # Tier 3: run-level report
        run_report_path = rr.dest_dir / "report.json"
        logger.info("    Regenerating run report")
        try:
            generate_run_report(
                run_name=rr.run_name,
                run_dir=rr.dest_dir,
                report_path=run_report_path,
                schema_version=schema_version,
                enabled_metrics=ENABLED_METRICS,
            )
            logger.info("    Run report written: %s", run_report_path)
        except Exception as exc:
            logger.error("    ERROR generating run report: %s", exc)
            logger.debug(traceback.format_exc())

    logger.info("")
    logger.info("PHASE 5 DONE")


def _is_sample_report(path: Path) -> bool:
    """Return True if *path* is a Tier-1 (sample-level) report.json."""
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data.get("level") == "sample"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Phase 6 — HTML report generation
# ---------------------------------------------------------------------------

def generate_html_reports(
    reorganized: List[ReorganizedRun],
    outputs_dir: Path,
    reports_dir: Path,
    bench_root: Optional[Path],
    dataset: str = "causal_nuscenes",
) -> None:
    """
    Generate HTML reports for every run that has a valid run-level report.json.
    """
    logger.info("=" * 72)
    logger.info("PHASE 6: HTML REPORT GENERATION")
    logger.info("  Reports dir: %s", reports_dir)

    _ensure_repo_on_path()

    from evaluation.common.html_reporter import generate_all_html  # type: ignore
    from evaluation.scripts.generate_html_reports import build_root_aggregate  # type: ignore

    reports_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0

    for rr in reorganized:
        run_report_path = rr.dest_dir / "report.json"
        if not run_report_path.exists():
            logger.warning("  SKIP (no run report): %s", rr.run_name)
            skipped += 1
            continue

        try:
            with open(run_report_path) as fh:
                run_report = json.load(fh)
            if run_report.get("level") != "run":
                logger.warning("  SKIP (wrong level): %s", rr.run_name)
                skipped += 1
                continue
        except Exception as exc:
            logger.error("  ERROR reading run report for %s: %s", rr.run_name, exc)
            skipped += 1
            continue

        root_aggregate = build_root_aggregate(rr.run_name, run_report)

        bench_dirs = None
        if bench_root:
            bench_dir = bench_root / dataset
            if bench_dir.exists():
                bench_dirs = {dataset: bench_dir}

        logger.info("  Generating HTML: %s", rr.run_name)
        try:
            generate_all_html(
                run_name=rr.run_name,
                cdb_outputs_dir=outputs_dir,
                cdb_reports_dir=reports_dir,
                root_aggregate=root_aggregate,
                bench_dirs=bench_dirs,
            )
            run_html = reports_dir / rr.run_name / "index.html"
            logger.info("    → %s (exists=%s)", run_html, run_html.exists())
            success += 1
        except Exception as exc:
            logger.error("    ERROR: %s", exc)
            logger.debug(traceback.format_exc())
            skipped += 1

    logger.info("PHASE 6 DONE: %d succeeded, %d skipped/failed", success, skipped)


# ---------------------------------------------------------------------------
# Phase 7 — Markdown summaries
# ---------------------------------------------------------------------------

def _pct(val: float) -> str:
    return f"{val * 100:.1f}"


def _load_run_report(run_dir: Path) -> Optional[dict]:
    rp = run_dir / "report.json"
    if not rp.exists():
        return None
    try:
        with open(rp) as fh:
            data = json.load(fh)
        return data if data.get("level") == "run" else None
    except Exception:
        return None


def write_markdown_summaries(
    sanity_results: Dict[str, dict],
    outputs_dir: Path,
    reports_dir: Path,
    docs_dir: Path,
    bench_root: Optional[Path],
) -> None:
    """
    Write per-model sanity summaries and a combined statistics table.
    """
    logger.info("=" * 72)
    logger.info("PHASE 7: MARKDOWN SUMMARIES")
    logger.info("  Docs dir: %s", docs_dir)

    docs_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-model sanity markdown ──────────────────────────────────────────
    for run_name, analysis in sanity_results.items():
        model_id = _infer_model_id(run_name)
        display = MODEL_REGISTRY.get(model_id, (model_id, "unknown"))[0]
        md_path = docs_dir / f"sanity_{run_name}.md"
        _write_model_sanity_md(md_path, run_name, display, analysis, outputs_dir)
        logger.info("  Wrote %s", md_path)

    # ── Combined statistics table ──────────────────────────────────────────
    combined_path = docs_dir / "combined_statistics.md"
    _write_combined_statistics(combined_path, sanity_results, outputs_dir, reports_dir)
    logger.info("  Wrote %s", combined_path)

    # ── Model readiness checklist ──────────────────────────────────────────
    checklist_path = docs_dir / "model_readiness.md"
    _write_readiness_checklist(checklist_path, sanity_results, outputs_dir)
    logger.info("  Wrote %s", checklist_path)

    logger.info("PHASE 7 DONE")


def _write_model_sanity_md(
    path: Path,
    run_name: str,
    display_name: str,
    analysis: dict,
    outputs_dir: Path,
) -> None:
    lines: List[str] = []
    lines.append(f"# Sanity Check Report: {display_name}")
    lines.append(f"")
    lines.append(f"**Run name:** `{run_name}`  ")
    lines.append(f"**Run dir:** `{analysis.get('run_dir', '?')}`  ")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ")
    lines.append("")

    if "error" in analysis:
        lines.append(f"## Error")
        lines.append(f"```")
        lines.append(analysis["error"])
        lines.append(f"```")
        path.write_text("\n".join(lines))
        return

    # Per-dataset tables
    for ds_name, s in analysis.get("datasets", {}).items():
        lines.append(f"## Dataset: `{ds_name}`")
        lines.append("")
        lines.append("### Benchmark Reference")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total samples | {s['bench_total_samples']} |")
        lines.append(f"| Samples with QA | {s['bench_qa_samples']} |")
        lines.append(f"| Total VQA questions | {s['bench_vqa_questions']} |")
        lines.append("")

        lines.append("### VQA Output Status")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Samples with prompts | {s['samples_with_prompts']} |")
        lines.append(f"| Samples with outputs | {s['samples_with_outputs']} |")
        lines.append(f"| **Complete (✓)** | **{s['samples_complete_vqa']}** ({s['pct_samples_complete']}) |")
        lines.append(f"| Incomplete (partial) | {s['samples_incomplete_vqa']} |")
        lines.append(f"| Missing outputs | {s['samples_missing_outputs']} |")
        lines.append(f"| Questions expected | {s['total_prompts_expected']} |")
        lines.append(f"| Questions written | {s['total_outputs_written']} ({s['pct_questions_written']}) |")
        lines.append(f"| Error outputs | {s['total_error_outputs']} |")
        lines.append("")

        lines.append("### Trajectory Output Status")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Samples with traj prompts | {s['samples_with_traj_prompts']} |")
        lines.append(f"| **Complete (✓)** | **{s['samples_complete_traj']}** |")
        lines.append(f"| Incomplete | {s['samples_incomplete_traj']} |")
        lines.append(f"| Missing | {s['samples_missing_traj_outputs']} |")
        lines.append(f"| Traj questions written | {s['total_traj_outputs']} |")
        lines.append(f"| Traj error outputs | {s['total_traj_errors']} |")
        lines.append("")

        # Issues
        issues = s.get("issues", [])
        if issues:
            lines.append(f"### Issues ({len(issues)} total)")
            lines.append("")
            for issue in issues:
                lines.append(f"- `{issue}`")
            lines.append("")
        else:
            lines.append("### Issues")
            lines.append("")
            lines.append("_No issues found._")
            lines.append("")

    # Run-level metrics (if report exists)
    run_report = _load_run_report(outputs_dir / run_name)
    if run_report:
        lines.append("## Evaluation Metrics (from run report)")
        lines.append("")
        m = run_report.get("metrics", {})

        overall = m.get("overall", {})
        lines.append(f"**Overall accuracy:** {_pct(overall.get('accuracy', 0))}%"
                     f"  ({overall.get('correct', '?')}/{overall.get('n', '?')} correct)")
        lines.append("")

        per_qa = m.get("per_qa_type", {})
        if per_qa:
            lines.append("### Accuracy by QA Type (Source)")
            lines.append("")
            lines.append("| QA Type | Accuracy | Correct | N |")
            lines.append("|---------|----------|---------|---|")
            for qa_type in ["ladder", "dormant", "distractor"]:
                v = per_qa.get(qa_type, {})
                lines.append(
                    f"| {qa_type.capitalize()} | {_pct(v.get('accuracy', 0))}% | "
                    f"{v.get('correct', '?')} | {v.get('n', '?')} |"
                )
            lines.append("")

        per_rung = m.get("per_rung", {})
        if per_rung:
            lines.append("### Accuracy by Rung (PCH Layer)")
            lines.append("")
            lines.append("| Rung | Accuracy | Correct | N |")
            lines.append("|------|----------|---------|---|")
            for rung in sorted(per_rung.keys(), key=lambda r: int(r) if r.isdigit() else 99):
                v = per_rung[rung]
                lines.append(
                    f"| R{rung} | {_pct(v.get('accuracy', 0))}% | "
                    f"{v.get('correct', '?')} | {v.get('n', '?')} |"
                )
            lines.append("")

        per_qt = m.get("per_question_type", {})
        if per_qt:
            lines.append("### Accuracy by Question Type")
            lines.append("")
            lines.append("| Question Type | Accuracy | Correct | N |")
            lines.append("|--------------|----------|---------|---|")
            for qt, v in sorted(per_qt.items()):
                lines.append(
                    f"| {qt} | {_pct(v.get('accuracy', 0))}% | "
                    f"{v.get('correct', '?')} | {v.get('n', '?')} |"
                )
            lines.append("")

        ans_stats = m.get("answer_stats", {})
        if ans_stats:
            lines.append("### Answer Distribution")
            lines.append("")
            lines.append("| Answer | GT Count | GT % | Pred Count | Pred % | Accuracy |")
            lines.append("|--------|----------|------|------------|--------|----------|")
            gt_dist   = ans_stats.get("gt_distribution", {})
            pred_dist = ans_stats.get("predicted_distribution", {})
            per_ans   = ans_stats.get("per_answer_accuracy", {})
            all_ans   = sorted(set(list(gt_dist) + list(pred_dist)))
            for ans in all_ans:
                gt_v    = gt_dist.get(ans, {})
                pred_v  = pred_dist.get(ans, {})
                acc_v   = per_ans.get(ans, {})
                lines.append(
                    f"| {ans} | {gt_v.get('count', 0)} | {gt_v.get('pct', 0):.1f}% | "
                    f"{pred_v.get('count', 0)} | {pred_v.get('pct', 0):.1f}% | "
                    f"{_pct(acc_v.get('accuracy', 0))}% |"
                )
            lines.append("")
            lines.append(f"**Answer bias:** {ans_stats.get('answer_bias', '?')}%  ")
            lines.append(f"**Answer classes:** {ans_stats.get('n_answer_classes', '?')}  ")
            lines.append("")

        bin_m = m.get("binary_metrics", {})
        if bin_m:
            lines.append("### Binary Classification Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Accuracy | {_pct(bin_m.get('accuracy', 0))}% |")
            lines.append(f"| Precision | {_pct(bin_m.get('precision', 0))}% |")
            lines.append(f"| Recall | {_pct(bin_m.get('recall', 0))}% |")
            lines.append(f"| F1 | {_pct(bin_m.get('f1', 0))}% |")
            lines.append(f"| TP/FP/FN/TN | {bin_m.get('tp','?')}/{bin_m.get('fp','?')}/{bin_m.get('fn','?')}/{bin_m.get('tn','?')} |")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_combined_statistics(
    path: Path,
    sanity_results: Dict[str, dict],
    outputs_dir: Path,
    reports_dir: Path,
) -> None:
    lines: List[str] = []
    lines.append("# CausalDriveBench Evaluation — Combined Statistics")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")
    lines.append("## Overview Table")
    lines.append("")

    # Build rows
    header = (
        "| Model | Run Name | VQA Complete | VQA% | Traj Complete | "
        "Overall Acc | Active (Ladder) | Dormant | Distractor | R0 | R1 | R2 | R3 | "
        "Issues | T1 Reports | T3 Report |"
    )
    sep = "|".join(["---"] * (header.count("|") - 1))
    sep = f"|{sep}|"
    lines.append(header)
    lines.append(sep)

    for run_name, analysis in sorted(sanity_results.items()):
        model_id = _infer_model_id(run_name)
        display  = MODEL_REGISTRY.get(model_id, (model_id, "?"))[0]

        # Sanity numbers (first dataset only for brevity)
        ds_data = analysis.get("datasets", {})
        first_ds = next(iter(ds_data.values()), {}) if ds_data else {}
        n_complete = first_ds.get("samples_complete_vqa", "?")
        bench_qa   = first_ds.get("bench_qa_samples", "?")
        pct_vqa    = first_ds.get("pct_samples_complete", "?")
        n_traj     = first_ds.get("samples_complete_traj", 0)
        n_issues   = len(first_ds.get("issues", []))

        # Accuracy metrics from run report
        run_report = _load_run_report(outputs_dir / run_name)
        overall_acc = "—"
        ladder_acc  = "—"
        dormant_acc = "—"
        dist_acc    = "—"
        r0 = r1 = r2 = r3 = "—"
        has_t1 = has_t3 = "✗"

        if run_report:
            has_t3 = "✓"
            m = run_report.get("metrics", {})
            ov = m.get("overall", {})
            if ov.get("accuracy") is not None:
                overall_acc = f"{_pct(ov['accuracy'])}%"

            pqa = m.get("per_qa_type", {})
            if pqa.get("ladder"):
                ladder_acc = f"{_pct(pqa['ladder']['accuracy'])}%"
            if pqa.get("dormant"):
                dormant_acc = f"{_pct(pqa['dormant']['accuracy'])}%"
            if pqa.get("distractor"):
                dist_acc = f"{_pct(pqa['distractor']['accuracy'])}%"

            pr = m.get("per_rung", {})
            if pr.get("0"):
                r0 = f"{_pct(pr['0']['accuracy'])}%"
            if pr.get("1"):
                r1 = f"{_pct(pr['1']['accuracy'])}%"
            if pr.get("2"):
                r2 = f"{_pct(pr['2']['accuracy'])}%"
            if pr.get("3"):
                r3 = f"{_pct(pr['3']['accuracy'])}%"

        # Check for any sample reports
        run_dir = outputs_dir / run_name
        if run_dir.exists() and any(run_dir.rglob("*/*/report.json")):
            has_t1 = "✓"

        lines.append(
            f"| {display} | `{run_name}` | {n_complete}/{bench_qa} | {pct_vqa} | {n_traj} | "
            f"{overall_acc} | {ladder_acc} | {dormant_acc} | {dist_acc} | "
            f"{r0} | {r1} | {r2} | {r3} | {n_issues} | {has_t1} | {has_t3} |"
        )

    lines.append("")
    lines.append("## VQA Sanity Details (per model, causal_nuscenes)")
    lines.append("")
    lines.append("| Model | Run | Expected Q | Written Q | Errors | Partial | Missing |")
    lines.append("|-------|-----|-----------|----------|--------|---------|---------|")

    for run_name, analysis in sorted(sanity_results.items()):
        model_id = _infer_model_id(run_name)
        display  = MODEL_REGISTRY.get(model_id, (model_id, "?"))[0]
        ds_data  = analysis.get("datasets", {})
        for ds_name, s in ds_data.items():
            lines.append(
                f"| {display} | `{run_name}` | {s['total_prompts_expected']} | "
                f"{s['total_outputs_written']} | {s['total_error_outputs']} | "
                f"{s['samples_incomplete_vqa']} | {s['samples_missing_outputs']} |"
            )

    lines.append("")
    lines.append("## Trajectory Sanity Details")
    lines.append("")
    lines.append("| Model | Run | Traj Prompts | Traj Outputs | Traj Errors | Complete |")
    lines.append("|-------|-----|-------------|-------------|-------------|---------|")

    for run_name, analysis in sorted(sanity_results.items()):
        model_id = _infer_model_id(run_name)
        display  = MODEL_REGISTRY.get(model_id, (model_id, "?"))[0]
        ds_data  = analysis.get("datasets", {})
        for ds_name, s in ds_data.items():
            if s["samples_with_traj_prompts"] > 0:
                lines.append(
                    f"| {display} | `{run_name}` | {s['total_traj_prompts']} | "
                    f"{s['total_traj_outputs']} | {s['total_traj_errors']} | "
                    f"{s['samples_complete_traj']} |"
                )

    lines.append("")
    lines.append("## HTML Reports")
    lines.append("")
    html_index = reports_dir / "index.html"
    if html_index.exists():
        lines.append(f"Root index: `{html_index}`")
    else:
        lines.append("_Root index.html not generated yet._")
    lines.append("")
    lines.append("| Model | HTML Report |")
    lines.append("|-------|------------|")
    for run_name in sorted(sanity_results.keys()):
        model_id = _infer_model_id(run_name)
        display  = MODEL_REGISTRY.get(model_id, (model_id, "?"))[0]
        run_html = reports_dir / run_name / "index.html"
        status   = f"`{run_html}`" if run_html.exists() else "_not generated_"
        lines.append(f"| {display} | {status} |")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_readiness_checklist(
    path: Path,
    sanity_results: Dict[str, dict],
    outputs_dir: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Model Readiness Checklist")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")
    lines.append("Tick the boxes that need action before report generation is complete.")
    lines.append("")
    lines.append("| Model | Run | VQA OK? | Traj OK? | T1 Reports? | T3 Report? | Action Required |")
    lines.append("|-------|-----|---------|----------|------------|------------|----------------|")

    for run_name, analysis in sorted(sanity_results.items()):
        model_id = _infer_model_id(run_name)
        display  = MODEL_REGISTRY.get(model_id, (model_id, "?"))[0]
        ds_data  = analysis.get("datasets", {})
        first_s  = next(iter(ds_data.values()), {}) if ds_data else {}

        bench_qa   = first_s.get("bench_qa_samples", 0)
        n_complete = first_s.get("samples_complete_vqa", 0)
        n_traj     = first_s.get("samples_with_traj_prompts", 0)
        n_traj_ok  = first_s.get("samples_complete_traj", 0)

        vqa_ok  = "✅" if (isinstance(bench_qa, int) and bench_qa > 0
                           and n_complete >= bench_qa * 0.9) else "❌"
        traj_ok = "✅" if (n_traj > 0 and n_traj_ok >= n_traj * 0.9) else (
                  "N/A" if n_traj == 0 else "❌")

        run_dir = outputs_dir / run_name
        has_t1  = "✅" if (run_dir.exists() and any(run_dir.rglob("*/*/report.json"))) else "❌"
        has_t3  = "✅" if (run_dir / "report.json").exists() else "❌"

        actions: List[str] = []
        if vqa_ok == "❌":
            actions.append("VQA incomplete")
        if traj_ok == "❌":
            actions.append("Traj incomplete")
        if has_t1 == "❌":
            actions.append("Run postprocessor for T1 reports")
        if has_t3 == "❌":
            actions.append("Regenerate T2/T3 reports")

        action_str = "; ".join(actions) if actions else "_Ready_"
        lines.append(
            f"| {display} | `{run_name}` | {vqa_ok} | {traj_ok} | "
            f"{has_t1} | {has_t3} | {action_str} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 8 — LaTeX tables
# ---------------------------------------------------------------------------

def _fmt(val: Optional[float], decimals: int = 1) -> str:
    """Format a float for LaTeX, or return \\texttt{[X]} placeholder."""
    if val is None:
        return r"\texttt{[X]}"
    return f"{val * 100:.{decimals}f}"


def _fmt_ade(val: Optional[float]) -> str:
    """Format an ADE value in metres for LaTeX."""
    if val is None:
        return r"\texttt{[X]}"
    return f"{val:.2f}"


def _load_trajectory_metrics(run_dir: Path) -> Dict[str, Optional[float]]:
    """
    Try to extract BTE-ADE, CTE-ADE-R2, CTE-ADE-R3 from the run.

    Looks in (in priority order):
      1. <run_dir>/causal_nuscenes/report.json → metrics.trajectory_eval
      2. <run_dir>/report.json → metrics.trajectory_eval
      3. <run_dir>/causal_nuscenes/trajectory_aggregate.json (team-submitted aggregate)
    """
    result: Dict[str, Optional[float]] = {"bte_ade": None, "cte_ade_r2": None, "cte_ade_r3": None}

    # Check run/dataset-level report.json for trajectory_eval section
    for candidate in [run_dir / "causal_nuscenes" / "report.json", run_dir / "report.json"]:
        if not candidate.exists():
            continue
        try:
            with open(candidate) as fh:
                data = json.load(fh)
            traj = data.get("metrics", {}).get("trajectory_eval", {})
            if traj:
                result["bte_ade"]    = traj.get("bte", {}).get("mean_ade")
                result["cte_ade_r2"] = traj.get("cte_r2", {}).get("mean_ade")
                result["cte_ade_r3"] = traj.get("cte_r3", {}).get("mean_ade")
                return result
        except Exception:
            pass

    # Fall back to trajectory_aggregate.json (team-submitted, different schema)
    agg_path = run_dir / "causal_nuscenes" / "trajectory_aggregate.json"
    if agg_path.exists():
        try:
            with open(agg_path) as fh:
                agg = json.load(fh)
            result["bte_ade"] = agg.get("bte_ade_stats", {}).get("mean")
            cte_stats = agg.get("cte_ade_stats", {})
            for rung_prefix, key in [("CTE_R2_", "cte_ade_r2"), ("CTE_R3_", "cte_ade_r3")]:
                entries = [(v["n"], v["mean"]) for k, v in cte_stats.items()
                           if k.startswith(rung_prefix) and v.get("n", 0) > 0]
                if entries:
                    total_n = sum(n for n, _ in entries)
                    result[key] = sum(n * m for n, m in entries) / total_n
        except Exception:
            pass

    return result


def generate_latex_tables(
    outputs_dir: Path,
    docs_dir: Path,
) -> None:
    """
    Scan all run directories in *outputs_dir*, read their run-level reports,
    and emit:
      - table1_vqa.tex   — VQA accuracy table (Table 1 in the paper)
      - table2_traj.tex  — Trajectory ADE table (Table 2 in the paper)
      - paper_tables.md  — Same values formatted as readable Markdown
    """
    logger.info("=" * 72)
    logger.info("PHASE 8: LATEX TABLE GENERATION")

    docs_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect metrics per model ──────────────────────────────────────────
    # model_id → {accuracy, per_qa_type, per_rung, traj, n_sample_reports}
    model_metrics: Dict[str, dict] = {}

    def _count_sample_reports(run_dir: Path) -> int:
        """Count sample-level report.json files (level=sample) in a run dir."""
        count = 0
        for rp in run_dir.rglob("report.json"):
            try:
                with open(rp) as fh:
                    d = json.load(fh)
                if d.get("level") == "sample":
                    count += 1
            except Exception:
                pass
        return count

    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_report = _load_run_report(run_dir)
        if not run_report:
            continue

        run_name = run_dir.name
        model_id = _infer_model_id(run_name)
        m = run_report.get("metrics", {})
        n_questions = run_report.get("n_questions", 0)
        n_sample_reports = _count_sample_reports(run_dir)

        # If we've already seen this model, prefer the run with actual sample-level
        # reports. When sample counts are equal, prefer the run with more questions.
        if model_id in model_metrics:
            existing = model_metrics[model_id]
            existing_sample_reports = existing.get("n_sample_reports", 0)
            existing_n_questions = existing.get("n_questions", 0)

            # Prefer runs with actual sample reports over report-aggregate-only copies
            if n_sample_reports < existing_sample_reports:
                logger.info(
                    "  Skipping %s (fewer sample reports %d < %d for %s)",
                    run_name, n_sample_reports, existing_sample_reports, model_id,
                )
                continue
            if n_sample_reports == existing_sample_reports and n_questions <= existing_n_questions:
                logger.info(
                    "  Skipping %s (same sample reports, fewer/equal questions %d <= %d for %s)",
                    run_name, n_questions, existing_n_questions, model_id,
                )
                continue
            logger.info(
                "  Replacing %s with %s (sample_reports: %d→%d, questions: %d→%d)",
                model_id, run_name, existing_sample_reports, n_sample_reports,
                existing_n_questions, n_questions,
            )

        traj = _load_trajectory_metrics(run_dir)

        model_metrics[model_id] = {
            "run_name":        run_name,
            "n_questions":     n_questions,
            "n_sample_reports": n_sample_reports,
            "overall":         m.get("overall", {}).get("accuracy"),
            "per_qa_type":     m.get("per_qa_type", {}),
            "per_rung":        m.get("per_rung", {}),
            "traj":            traj,
        }
        logger.info(
            "  Loaded metrics for: %s (%s) [%d sample reports, %d questions]",
            model_id, run_name, n_sample_reports, n_questions,
        )

    # ── Second pass: pick up trajectory-only runs (no run report but have traj data) ──
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        model_id = _infer_model_id(run_name)
        # Only process if not already loaded via VQA path
        if model_id in model_metrics:
            existing_traj = model_metrics[model_id].get("traj", {})
            if any(v is not None for v in existing_traj.values()):
                continue  # already has traj data
        traj = _load_trajectory_metrics(run_dir)
        if not any(v is not None for v in traj.values()):
            continue  # no trajectory data to add
        if model_id in model_metrics:
            # Patch trajectory into existing VQA entry
            model_metrics[model_id]["traj"] = traj
            logger.info("  Patched traj metrics into %s (%s)", model_id, run_name)
        else:
            # Trajectory-only model: create a metrics entry with no VQA data
            model_metrics[model_id] = {
                "run_name": run_name, "n_questions": 0, "n_sample_reports": 0,
                "overall": None, "per_qa_type": {}, "per_rung": {}, "traj": traj,
            }
            logger.info("  Loaded traj-only metrics for: %s (%s)", model_id, run_name)

    # ── Build rows per paper category ──────────────────────────────────────
    driving_vla_ids = [k for k, (_, cat) in MODEL_REGISTRY.items() if cat == "driving_vla"]
    general_vlm_ids = [k for k, (_, cat) in MODEL_REGISTRY.items() if cat == "general_vlm"]
    finetuned_ids   = [k for k, (_, cat) in MODEL_REGISTRY.items() if cat == "finetuned"]

    # Also include any models found in outputs that aren't in our registry
    found_ids = set(model_metrics.keys())
    known_ids = set(MODEL_REGISTRY.keys())
    unknown_ids = found_ids - known_ids
    if unknown_ids:
        logger.warning("  Unknown model IDs found in outputs: %s", unknown_ids)

    def _row_vqa(model_id: str, display: str) -> str:
        mm = model_metrics.get(model_id, {})
        pqa = mm.get("per_qa_type", {})
        pr  = mm.get("per_rung", {})
        r0 = _fmt(pr.get("0", {}).get("accuracy") if pr.get("0") else None)
        r1 = _fmt(pr.get("1", {}).get("accuracy") if pr.get("1") else None)
        r2 = _fmt(pr.get("2", {}).get("accuracy") if pr.get("2") else None)
        r3 = _fmt(pr.get("3", {}).get("accuracy") if pr.get("3") else None)
        active  = _fmt(pqa.get("ladder",     {}).get("accuracy") if pqa.get("ladder")     else None)
        dormant = _fmt(pqa.get("dormant",    {}).get("accuracy") if pqa.get("dormant")    else None)
        dist    = _fmt(pqa.get("distractor", {}).get("accuracy") if pqa.get("distractor") else None)
        overall = _fmt(mm.get("overall"))
        return (
            f"    {display:<32} & {r0} & {r1} & {r2} & {r3} "
            f"& {active} & {dormant} & {dist} & {overall} \\\\"
        )

    def _row_traj(model_id: str, display: str) -> str:
        mm   = model_metrics.get(model_id, {})
        traj = mm.get("traj", {})
        bte   = _fmt_ade(traj.get("bte_ade"))
        cte_r2 = _fmt_ade(traj.get("cte_ade_r2"))
        cte_r3 = _fmt_ade(traj.get("cte_ade_r3"))
        return f"    {display:<32} & {bte} & {cte_r2} & {cte_r3} \\\\"

    # ── Table 1: VQA ──────────────────────────────────────────────────────
    t1_lines: List[str] = []
    t1_lines.append(r"\begin{table}[h!]")
    t1_lines.append(r"  \caption{Causal QA accuracy (\%) by PCH layer and question source. "
                    r"Higher is better. Active = entities currently constraining ego; "
                    r"Dormant = entities one realisable transition from constraining ego; "
                    r"Distractor = visible but causally irrelevant entities. "
                    r"The Active--Distractor gap is the discrimination diagnostic; "
                    r"a model relying on perceptual salience scores high on Active and low on Distractor.}")
    t1_lines.append(r"  \label{tab:qa_results}")
    t1_lines.append(r"  \centering")
    t1_lines.append(r"  \footnotesize")
    t1_lines.append(r"  \begin{tabular}{l c c c c c c c c}")
    t1_lines.append(r"    \toprule")
    t1_lines.append(r"    & \multicolumn{4}{c}{\textbf{By PCH layer}} & \multicolumn{3}{c}{\textbf{By source}} & \\")
    t1_lines.append(r"    \cmidrule(lr){2-5} \cmidrule(lr){6-8}")
    t1_lines.append(r"    Model & R0 & R1 & R2 & R3 & Active & Dormant & Distractor & Overall \\")
    t1_lines.append(r"    \midrule")
    def _has_vqa_data(model_id: str) -> bool:
        mm = model_metrics.get(model_id, {})
        return mm.get("overall") is not None

    def _has_traj_data(model_id: str) -> bool:
        traj = model_metrics.get(model_id, {}).get("traj", {})
        return any(v is not None for v in traj.values())

    t1_lines.append(r"    \multicolumn{9}{l}{\textit{Driving VLAs}} \\")
    for mid in driving_vla_ids:
        if not _has_vqa_data(mid):
            continue
        display = MODEL_REGISTRY[mid][0]
        t1_lines.append(_row_vqa(mid, display))
    t1_lines.append(r"    \midrule")
    t1_lines.append(r"    \multicolumn{9}{l}{\textit{General VLMs}} \\")
    for mid in general_vlm_ids:
        if not _has_vqa_data(mid):
            continue
        display = MODEL_REGISTRY[mid][0]
        t1_lines.append(_row_vqa(mid, display))
    if finetuned_ids:
        finetuned_rows = [(mid, MODEL_REGISTRY[mid][0]) for mid in finetuned_ids if _has_vqa_data(mid)]
        if finetuned_rows:
            t1_lines.append(r"    \midrule")
            for mid, display in finetuned_rows:
                t1_lines.append(_row_vqa(mid, display))
    t1_lines.append(r"    \bottomrule")
    t1_lines.append(r"  \end{tabular}")
    t1_lines.append(r"\end{table}")

    # ── Table 2: Trajectory ───────────────────────────────────────────────
    traj_models = driving_vla_ids + finetuned_ids  # only trajectory-producing models
    t2_lines: List[str] = []
    t2_lines.append(r"\begin{table}[h!]")
    t2_lines.append(r"  \caption{Trajectory-level evaluation. BTE-ADE measures sensitivity "
                    r"of the planned trajectory to a factually incorrect rationale; "
                    r"near-zero BTE-ADE indicates the rationale is decorative. "
                    r"CTE-ADE-R2 covers Rung 2 binary interventions (CDE, NIE, NDE, AB) and "
                    r"CTE-ADE-R3 covers Rung 3 counterfactuals (CR, NC, WC, TWC). "
                    r"Trajectory-producing models only.}")
    t2_lines.append(r"  \label{tab:trajectory}")
    t2_lines.append(r"  \centering")
    t2_lines.append(r"  \footnotesize")
    t2_lines.append(r"  \begin{tabular}{l c c c}")
    t2_lines.append(r"    \toprule")
    t2_lines.append(r"    Model & BTE-ADE (m) $\uparrow$ & CTE-ADE-R2 (m) $\downarrow$ & CTE-ADE-R3 (m) $\downarrow$ \\")
    t2_lines.append(r"    \midrule")
    for mid in driving_vla_ids:
        if not _has_traj_data(mid):
            continue
        display = MODEL_REGISTRY[mid][0]
        t2_lines.append(_row_traj(mid, display))
    if finetuned_ids:
        finetuned_traj_rows = [(mid, MODEL_REGISTRY[mid][0]) for mid in finetuned_ids if _has_traj_data(mid)]
        if finetuned_traj_rows:
            t2_lines.append(r"    \midrule")
            for mid, display in finetuned_traj_rows:
                t2_lines.append(_row_traj(mid, display))
    t2_lines.append(r"    \bottomrule")
    t2_lines.append(r"  \end{tabular}")
    t2_lines.append(r"\end{table}")

    # ── Write files ───────────────────────────────────────────────────────
    t1_path = docs_dir / "table1_vqa.tex"
    t1_path.write_text("\n".join(t1_lines), encoding="utf-8")
    logger.info("  Wrote %s", t1_path)

    t2_path = docs_dir / "table2_trajectory.tex"
    t2_path.write_text("\n".join(t2_lines), encoding="utf-8")
    logger.info("  Wrote %s", t2_path)

    # ── Markdown version of the tables ────────────────────────────────────
    md_lines: List[str] = []
    md_lines.append("# Paper Tables — Extracted Values")
    md_lines.append("")
    md_lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    md_lines.append("")
    md_lines.append("## Table 1: Causal QA Accuracy (%)")
    md_lines.append("")
    md_lines.append("| Model | Category | R0 | R1 | R2 | R3 | Active | Dormant | Distractor | Overall | N Questions | Run |")
    md_lines.append("|-------|----------|----|----|----|----|--------|---------|------------|---------|------------|-----|")

    for mid in driving_vla_ids + general_vlm_ids + finetuned_ids + sorted(unknown_ids):
        if mid in MODEL_REGISTRY:
            display, cat = MODEL_REGISTRY[mid]
        else:
            display, cat = mid, "unknown"
        mm  = model_metrics.get(mid, {})
        pqa = mm.get("per_qa_type", {})
        pr  = mm.get("per_rung", {})
        r0 = f"{_pct(pr['0']['accuracy'])}%" if pr.get("0") else "—"
        r1 = f"{_pct(pr['1']['accuracy'])}%" if pr.get("1") else "—"
        r2 = f"{_pct(pr['2']['accuracy'])}%" if pr.get("2") else "—"
        r3 = f"{_pct(pr['3']['accuracy'])}%" if pr.get("3") else "—"
        active  = f"{_pct(pqa['ladder']['accuracy'])}%"     if pqa.get("ladder")     else "—"
        dormant = f"{_pct(pqa['dormant']['accuracy'])}%"    if pqa.get("dormant")    else "—"
        dist    = f"{_pct(pqa['distractor']['accuracy'])}%" if pqa.get("distractor") else "—"
        overall = f"{_pct(mm['overall'])}%"                 if mm.get("overall") is not None else "—"
        nq      = mm.get("n_questions", "—")
        run     = f"`{mm.get('run_name', '—')}`"
        md_lines.append(
            f"| {display} | {cat} | {r0} | {r1} | {r2} | {r3} | "
            f"{active} | {dormant} | {dist} | {overall} | {nq} | {run} |"
        )

    md_lines.append("")
    md_lines.append("## Table 2: Trajectory Evaluation (ADE in metres)")
    md_lines.append("")
    md_lines.append("| Model | BTE-ADE ↑ | CTE-ADE-R2 ↓ | CTE-ADE-R3 ↓ | Run |")
    md_lines.append("|-------|-----------|-------------|-------------|-----|")

    for mid in driving_vla_ids + finetuned_ids:
        display = MODEL_REGISTRY[mid][0]
        mm      = model_metrics.get(mid, {})
        traj    = mm.get("traj", {})
        bte     = f"{traj['bte_ade']:.2f} m"   if traj.get("bte_ade")    is not None else "—"
        cte_r2  = f"{traj['cte_ade_r2']:.2f} m" if traj.get("cte_ade_r2") is not None else "—"
        cte_r3  = f"{traj['cte_ade_r3']:.2f} m" if traj.get("cte_ade_r3") is not None else "—"
        run     = f"`{mm.get('run_name', '—')}`"
        md_lines.append(f"| {display} | {bte} | {cte_r2} | {cte_r3} | {run} |")

    md_lines.append("")
    md_lines.append("## Model → Run Mapping")
    md_lines.append("")
    md_lines.append("| Model ID (code) | Display Name | Category | Run Name |")
    md_lines.append("|----------------|-------------|----------|---------|")
    for mid, mm in sorted(model_metrics.items()):
        display, cat = MODEL_REGISTRY.get(mid, (mid, "unknown"))
        md_lines.append(f"| `{mid}` | {display} | {cat} | `{mm['run_name']}` |")

    if unknown_ids:
        md_lines.append("")
        md_lines.append("## Unknown Model IDs")
        md_lines.append("")
        md_lines.append("These model IDs were found in outputs but are not in MODEL_REGISTRY.")
        md_lines.append("Add them to MODEL_REGISTRY in `reorganize_and_report.py`.")
        md_lines.append("")
        for mid in sorted(unknown_ids):
            md_lines.append(f"- `{mid}` → run: `{model_metrics[mid]['run_name']}`")

    md_path = docs_dir / "paper_tables.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("  Wrote %s", md_path)

    logger.info("PHASE 8 DONE")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_repo_on_path() -> None:
    """Add EVAL_REPO_ROOT to sys.path so evaluation.* imports work."""
    repo_str = str(EVAL_REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _print_summary(
    discovered: List[DiscoveredRun],
    reorganized: List[ReorganizedRun],
    sanity: Dict[str, dict],
    outputs_dir: Path,
    docs_dir: Path,
) -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 72)
    logger.info("  Runs discovered      : %d", len(discovered))
    logger.info("  Runs reorganized     : %d", len(reorganized))
    logger.info("  Runs sanity-checked  : %d", len(sanity))

    def _is_90pct_complete(s: dict) -> bool:
        bench_qa = s.get("bench_qa_samples", 0)
        n_complete = s.get("samples_complete_vqa", 0)
        if not isinstance(bench_qa, int) or bench_qa == 0:
            return False
        return n_complete >= bench_qa * 0.9

    runs_with_bench_count = [
        a for a in sanity.values()
        if any(isinstance(s.get("bench_qa_samples"), int) and s["bench_qa_samples"] > 0
               for s in a.get("datasets", {}).values())
    ]
    complete = sum(
        1 for a in runs_with_bench_count
        if any(_is_90pct_complete(s) for s in a.get("datasets", {}).values())
    )
    if runs_with_bench_count:
        logger.info("  Runs ≥90%% VQA complete: %d / %d (bench count known)", complete, len(runs_with_bench_count))
    else:
        logger.info("  Runs ≥90%% VQA complete: N/A (bench manifest lacks n_samples_with_qa)")

    no_t1 = [rr.run_name for rr in reorganized if not rr.had_sample_reports]
    if no_t1:
        logger.warning(
            "  Runs WITHOUT Tier-1 sample reports (%d) — postprocessor required:", len(no_t1)
        )
        for name in no_t1:
            logger.warning("    • %s", name)

    no_t3 = [rr.run_name for rr in reorganized if not (outputs_dir / rr.run_name / "report.json").exists()]
    if no_t3:
        logger.warning(
            "  Runs WITHOUT Tier-3 run reports (%d) — re-run phase 5:", len(no_t3)
        )
        for name in no_t3:
            logger.warning("    • %s", name)

    logger.info("")
    logger.info("  Docs written to: %s", docs_dir)
    logger.info("  Key files:")
    for f in [
        docs_dir / "combined_statistics.md",
        docs_dir / "model_readiness.md",
        docs_dir / "paper_tables.md",
        docs_dir / "table1_vqa.tex",
        docs_dir / "table2_trajectory.tex",
    ]:
        logger.info("    %s  (%s)", f.name, "✓" if f.exists() else "✗ MISSING")
    logger.info("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CausalDriveBench evaluation output reorganizer and report generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--zip-path",     type=Path, default=DEFAULT_ZIP_PATH,
                   help="Path to the top-level Google Drive zip file")
    p.add_argument("--extract-dir",  type=Path, default=DEFAULT_EXTRACT_DIR,
                   help="Scratch directory for extracted zip content")
    p.add_argument("--outputs-dir",  type=Path, default=DEFAULT_OUTPUTS_DIR,
                   help="CDB_OUTPUTS_DIR: reorganised outputs land here")
    p.add_argument("--reports-dir",  type=Path, default=DEFAULT_REPORTS_DIR,
                   help="CDB_REPORTS_DIR: HTML reports land here")
    p.add_argument("--bench-root",   type=Path, default=DEFAULT_BENCH_ROOT,
                   help="Benchmark root (causal_drive_bench_*)")
    p.add_argument("--docs-dir",     type=Path, default=DEFAULT_DOCS_DIR,
                   help="Directory for markdown/LaTeX summaries")
    p.add_argument("--dry-run",      action="store_true",
                   help="Discover and log only; do not copy files or generate reports")
    p.add_argument("--skip-extract", action="store_true",
                   help="Skip Phase 1 (assume zip already extracted to --extract-dir)")
    p.add_argument("--skip-reorganize", action="store_true",
                   help="Skip Phases 2+3 (assume outputs already in --outputs-dir)")
    p.add_argument("--skip-html",    action="store_true",
                   help="Skip Phase 6 (HTML generation)")
    p.add_argument("--dataset",      default="causal_nuscenes",
                   help="Primary dataset name for HTML generation (default: causal_nuscenes)")
    p.add_argument("--log-level",    default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default: INFO)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=" * 72)
    logger.info("CausalDriveBench — Evaluation Reorganizer & Report Generator")
    logger.info("=" * 72)
    logger.info("  zip-path     : %s", args.zip_path)
    logger.info("  extract-dir  : %s", args.extract_dir)
    logger.info("  outputs-dir  : %s", args.outputs_dir)
    logger.info("  reports-dir  : %s", args.reports_dir)
    logger.info("  bench-root   : %s", args.bench_root)
    logger.info("  docs-dir     : %s", args.docs_dir)
    logger.info("  dry-run      : %s", args.dry_run)
    logger.info("  skip-extract : %s", args.skip_extract)
    logger.info("  skip-reorganize: %s", args.skip_reorganize)
    logger.info("=" * 72)

    # ── Phase 1: Extract ──────────────────────────────────────────────────
    if not args.skip_extract and not args.skip_reorganize:
        extract_recursive(args.zip_path, args.extract_dir)
    else:
        logger.info("PHASE 1: SKIP (--skip-extract or --skip-reorganize set)")

    # ── Phase 2: Discover ─────────────────────────────────────────────────
    discovered: List[DiscoveredRun] = []
    if not args.skip_reorganize:
        discovered = discover_run_dirs(args.extract_dir)
    else:
        logger.info("PHASE 2: SKIP (--skip-reorganize set)")

    # ── Phase 3: Reorganize ───────────────────────────────────────────────
    reorganized: List[ReorganizedRun] = []
    if not args.skip_reorganize:
        reorganized = reorganize_runs(discovered, args.outputs_dir, args.dry_run)
    else:
        # Build reorganized list from existing outputs dir
        logger.info("PHASE 3: SKIP — loading existing runs from %s", args.outputs_dir)
        if args.outputs_dir.exists():
            for run_dir in sorted(args.outputs_dir.iterdir()):
                if run_dir.is_dir():
                    model_id = _infer_model_id(run_dir.name)
                    datasets_found = [
                        d.name for d in run_dir.iterdir()
                        if d.is_dir() and d.name.startswith("causal_")
                    ]
                    has_t1 = any(True for _ in run_dir.rglob("*/*/report.json"))
                    has_t3 = (run_dir / "report.json").exists()
                    reorganized.append(ReorganizedRun(
                        dest_dir=run_dir,
                        model_id=model_id,
                        run_name=run_dir.name,
                        datasets=datasets_found,
                        had_sample_reports=has_t1,
                        had_run_report=has_t3,
                    ))
            logger.info("  Found %d existing run dirs", len(reorganized))

    if args.dry_run:
        logger.info("DRY-RUN mode: stopping after discovery")
        _print_summary(discovered, reorganized, {}, args.outputs_dir, args.docs_dir)
        return

    # ── Phase 4: Sanity check ─────────────────────────────────────────────
    bench_root = args.bench_root if args.bench_root.exists() else None
    if bench_root is None:
        logger.warning("Bench root not found: %s — sanity check will omit expected counts",
                       args.bench_root)

    sanity_results = run_sanity_checks(args.outputs_dir, bench_root)

    # ── Phase 5: Report regeneration ──────────────────────────────────────
    regenerate_reports(reorganized)

    # ── Phase 6: HTML ─────────────────────────────────────────────────────
    if not args.skip_html:
        generate_html_reports(
            reorganized, args.outputs_dir, args.reports_dir,
            bench_root, args.dataset,
        )
    else:
        logger.info("PHASE 6: SKIP (--skip-html set)")

    # ── Phase 7: Markdown summaries ───────────────────────────────────────
    write_markdown_summaries(
        sanity_results, args.outputs_dir, args.reports_dir,
        args.docs_dir, bench_root,
    )

    # ── Phase 8: LaTeX tables ─────────────────────────────────────────────
    generate_latex_tables(args.outputs_dir, args.docs_dir)

    # ── Final summary ─────────────────────────────────────────────────────
    _print_summary(discovered, reorganized, sanity_results, args.outputs_dir, args.docs_dir)


if __name__ == "__main__":
    main()
