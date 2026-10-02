#!/usr/bin/env python3
"""
Retry LLM judge scoring for questions that failed (error) or were not scored.

Workflow:
  1. Scan all sample report.json files in a run-dir for failed/unscored questions.
  2. Submit them as a single Anthropic Batch API request (50% cheaper, high throughput).
  3. Poll until the batch completes.
  4. Write scores back into each sample report.
  5. Re-aggregate dataset + run-level reports.

Usage:
  python evaluation/scripts/judge_retry_batch.py \\
    --run-dir /datadrive/datasets/cdb_outputs_20260427/alpamayo_1_5_20260428_081120 \\
    --dataset causal_nuscenes \\
    [--api-key-env ANTHROPIC_API_KEY] \\
    [--model claude-sonnet-4-6] \\
    [--concurrency 50] \\
    [--batch-api] \\
    [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Dimensions scored by the judge (must match llm_judge.py)
_DIMENSIONS = [
    "spatial_grounding",
    "mechanism_identification",
    "conditionality_awareness",
    "answer_consistency",
]

JUDGE_SYSTEM_PROMPT: str = ""
JUDGE_USER_TEMPLATE: str = ""


def _load_prompts(eval_root: Path) -> None:
    global JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE
    prompts_dir = eval_root / "evaluation" / "common" / "prompts"
    JUDGE_SYSTEM_PROMPT = (prompts_dir / "llm_judge_system.txt").read_text()
    JUDGE_USER_TEMPLATE = (prompts_dir / "llm_judge_user_template.txt").read_text()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v, lo=0, hi=2):
    return max(lo, min(hi, int(v)))


def _fix_json(text: str) -> str:
    """Apply light repairs to common LLM JSON formatting issues."""
    # Missing comma between closing } of "scores" block and next key ("reasoning")
    # e.g.  }\n  "reasoning":  →  },\n  "reasoning":
    text = re.sub(r'(\})\s*\n(\s*"(?:reasoning|scores)\b)', r'\1,\n\2', text)
    # Trailing comma before closing brace: {"a":1,}  →  {"a":1}
    text = re.sub(r',\s*(\})', r'\1', text)
    return text


def _extract_json(text: str) -> str:
    text = text.strip()
    # Try direct parse first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    # Try after light repair
    try:
        fixed = _fix_json(text)
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        candidate = m.group(1)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            try:
                fixed = _fix_json(candidate)
                json.loads(fixed)
                return fixed
            except json.JSONDecodeError:
                pass
    # Last resort: find first {...} block and repair
    m2 = re.search(r"\{.*\}", text, re.DOTALL)
    if m2:
        candidate = m2.group(0)
        try:
            fixed = _fix_json(candidate)
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            return candidate
    return text


def _parse_score(raw_content: str) -> dict:
    # Primary path: full JSON extraction + parse
    try:
        content = _extract_json(raw_content)
        parsed = json.loads(content)
        raw_scores = parsed.get("scores", {})
        scores = {dim: _clamp(raw_scores.get(dim, 0)) for dim in _DIMENSIONS}
        composite = sum(scores.values()) / len(scores)
        return {
            "scores": scores,
            "composite_score": round(composite, 4),
            "reasoning": parsed.get("reasoning", ""),
        }
    except json.JSONDecodeError:
        pass

    # Fallback: response truncated (max_tokens hit). The "scores" block is always
    # emitted before "reasoning", so it is present even in truncated responses.
    # Extract scores from the sub-dict and accept an empty reasoning string.
    scores_match = re.search(r'"scores"\s*:\s*\{([^}]*)\}', raw_content, re.DOTALL)
    if scores_match:
        try:
            raw_scores = json.loads("{" + scores_match.group(1) + "}")
            scores = {dim: _clamp(raw_scores.get(dim, 0)) for dim in _DIMENSIONS}
            composite = sum(scores.values()) / len(scores)
            return {
                "scores": scores,
                "composite_score": round(composite, 4),
                "reasoning": "[truncated — max_tokens exceeded]",
            }
        except json.JSONDecodeError:
            pass

    # Unable to extract anything useful — return zero with error flag
    return {
        "scores": {dim: 0 for dim in _DIMENSIONS},
        "composite_score": 0.0,
        "reasoning": "",
        "error": "JSON parse failed: could not extract scores from response",
    }


def _zero_score(reason: str = "") -> dict:
    """Zero score with an error note (excluded from aggregate n)."""
    d = {"scores": {dim: 0 for dim in _DIMENSIONS}, "composite_score": 0.0, "reasoning": ""}
    if reason:
        d["error"] = reason
    return d


def _zero_score_no_response() -> dict:
    """Zero score for empty model response — no error key, so it counts in aggregate n."""
    return {"scores": {dim: 0 for dim in _DIMENSIONS}, "composite_score": 0.0,
            "reasoning": "No model response provided."}


def _make_client(api_key: str):
    """Build an Anthropic client bypassing Azure Foundry routing."""
    import anthropic
    _foundry_vars = (
        "ANTHROPIC_FOUNDRY_RESOURCE",
        "ANTHROPIC_FOUNDRY_API_KEY",
        "FOUNDRY_BASE_URL",
        "CLAUDE_CODE_USE_FOUNDRY",
    )
    _saved = {k: os.environ.pop(k, None) for k in _foundry_vars}
    try:
        client = anthropic.Anthropic(api_key=api_key)
    finally:
        for k, v in _saved.items():
            if v is not None:
                os.environ[k] = v
    return client


# ---------------------------------------------------------------------------
# Step 1 — collect failing questions
# ---------------------------------------------------------------------------

def collect_failures(dataset_dir: Path) -> list[dict]:
    """
    Scan all sample report.json files and return a list of questions that
    have an error or are missing an llm_judge_score.

    Returns a list of dicts:
      {
        "report_path": Path,
        "qa_idx":      int,
        "question_text": str,
        "ground_truth":  str,
        "raw_output_text": str,
      }
    """
    failures = []
    report_files = sorted(dataset_dir.rglob("*/*/report.json"))
    log.info("Scanning %d sample report files …", len(report_files))

    for rp in report_files:
        with open(rp) as f:
            rep = json.load(f)
        if rep.get("level") != "sample":
            continue
        for qi, q in enumerate(rep.get("qa_results", [])):
            js = q.get("llm_judge_score", None)
            needs_score = (
                js is None
                or "error" in js
            )
            if needs_score:
                raw_text = q.get("raw_output_text", "") or ""
                raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                if not raw_text:
                    continue  # no model response — skip
                failures.append({
                    "report_path":    rp,
                    "qa_idx":         qi,
                    "question_text":  q.get("question_text", ""),
                    "ground_truth":   q.get("ground_truth", ""),
                    "raw_output_text": raw_text,
                })

    log.info("Found %d questions needing (re-)scoring.", len(failures))
    return failures


def zero_score_empty_responses(dataset_dir: Path) -> int:
    """
    Set llm_judge_score to explicit zero (no error key) for questions where
    the model gave no output. These count toward aggregate n as 0-scored, which
    is the correct treatment: the model failed to respond.

    Returns count of questions updated.
    """
    import re
    report_files = sorted(dataset_dir.rglob("*/*/report.json"))
    updated = 0
    by_report: dict = {}

    for rp in report_files:
        with open(rp) as f:
            rep = json.load(f)
        if rep.get("level") != "sample":
            continue
        changed = False
        for qi, q in enumerate(rep.get("qa_results", [])):
            if q.get("llm_judge_score") is not None:
                continue  # already scored (including errored)
            raw_text = q.get("raw_output_text", "") or ""
            raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            if raw_text:
                continue  # has content — will be retried via batch
            rep["qa_results"][qi]["llm_judge_score"] = _zero_score_no_response()
            changed = True
            updated += 1
        if changed:
            rep["generated_at"] = datetime.now(timezone.utc).isoformat()
            tmp = rp.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(rep, f, indent=2)
            tmp.replace(rp)

    log.info("Zero-scored %d empty-response questions across sample reports.", updated)
    return updated


# ---------------------------------------------------------------------------
# Step 2a — Anthropic Batch API path
# ---------------------------------------------------------------------------

def _build_batch_requests(failures: list[dict]) -> list[dict]:
    requests = []
    for i, item in enumerate(failures):
        user_content = JUDGE_USER_TEMPLATE.format(
            question=item["question_text"],
            ground_truth=item["ground_truth"],
            model_response=item["raw_output_text"],
            criteria_section="",
        )
        requests.append({
            "custom_id": str(i),
            "params": {
                "model": "",  # filled in caller
                "max_tokens": 1024,
                "temperature": 0.0,
                "system": JUDGE_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_content}],
            },
        })
    return requests


def submit_batch(client, model: str, failures: list[dict], checkpoint_path: Path) -> str:
    """Submit batch and return batch_id. Saves batch_id to checkpoint."""
    batch_requests = _build_batch_requests(failures)
    for r in batch_requests:
        r["params"]["model"] = model

    log.info("Submitting batch of %d requests …", len(batch_requests))
    for attempt in range(5):
        try:
            batch = client.messages.batches.create(requests=batch_requests)
            break
        except Exception as exc:
            wait = 30 * (attempt + 1)
            log.warning("Batch submit attempt %d failed (%s); retrying in %ds", attempt + 1, exc, wait)
            time.sleep(wait)
    else:
        raise RuntimeError("All batch submit attempts failed")

    batch_id = batch.id
    log.info("Batch submitted: %s (%d requests)", batch_id, len(batch_requests))
    checkpoint_path.write_text(json.dumps({"batch_id": batch_id, "n": len(failures)}))
    return batch_id


def poll_batch(client, batch_id: str, poll_interval: int = 30) -> None:
    """Poll until batch is complete."""
    log.info("Polling batch %s …", batch_id)
    while True:
        status = client.messages.batches.retrieve(batch_id)
        counts = status.request_counts
        print(
            f"  [{datetime.now().strftime('%H:%M:%S')}] {status.processing_status} | "
            f"succeeded={counts.succeeded} errored={counts.errored} "
            f"processing={counts.processing} canceled={counts.canceled}",
            flush=True,
        )
        if status.processing_status == "ended":
            break
        time.sleep(poll_interval)


def retrieve_batch_results(client, batch_id: str, n: int) -> list[dict]:
    """Retrieve and parse batch results. Returns list of score dicts indexed by custom_id."""
    raw: dict[str, dict] = {}
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            raw[cid] = _parse_score(result.result.message.content[0].text)
        else:
            raw[cid] = _zero_score(str(result.result))

    missing = _zero_score("missing from batch results")
    return [raw.get(str(i), missing) for i in range(n)]


# ---------------------------------------------------------------------------
# Step 2b — Concurrent API path (fallback)
# ---------------------------------------------------------------------------

def score_concurrent(client, model: str, failures: list[dict], concurrency: int = 50) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[dict] = [{}] * len(failures)

    def _score_one(args):
        idx, item = args
        user_content = JUDGE_USER_TEMPLATE.format(
            question=item["question_text"],
            ground_truth=item["ground_truth"],
            model_response=item["raw_output_text"],
            criteria_section="",
        )
        for attempt in range(4):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    temperature=0.0,
                    system=JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )
                return idx, _parse_score(resp.content[0].text)
            except Exception as exc:
                err = str(exc)
                if "429" in err or "rate" in err.lower():
                    wait = 10 * (attempt + 1)
                    log.warning("Rate limit (idx=%d, attempt=%d); sleeping %ds", idx, attempt, wait)
                    time.sleep(wait)
                else:
                    log.warning("Error scoring idx=%d: %s", idx, err[:120])
                    return idx, _zero_score(err)
        return idx, _zero_score("rate limit: max retries exceeded")

    log.info("Concurrent scoring: %d questions, concurrency=%d", len(failures), concurrency)
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_score_one, (i, r)): i for i, r in enumerate(failures)}
        for future in as_completed(futures):
            idx, score = future.result()
            results[idx] = score
            done += 1
            if done % 100 == 0 or done == len(failures):
                log.info("Scored %d / %d", done, len(failures))

    return results


# ---------------------------------------------------------------------------
# Step 3 — write scores back to sample reports
# ---------------------------------------------------------------------------

def write_scores_back(failures: list[dict], scores: list[dict]) -> dict[Path, int]:
    """Update each sample report.json with the new llm_judge_score. Returns {path: n_updated}."""
    # Group by report_path
    by_report: dict[Path, list[tuple[int, dict]]] = {}
    for item, score in zip(failures, scores):
        rp = item["report_path"]
        by_report.setdefault(rp, []).append((item["qa_idx"], score))

    updated_counts: dict[Path, int] = {}
    for rp, updates in by_report.items():
        with open(rp) as f:
            rep = json.load(f)
        for qi, score in updates:
            rep["qa_results"][qi]["llm_judge_score"] = score
        # Update report timestamp
        rep["generated_at"] = datetime.now(timezone.utc).isoformat()
        # Write atomically via tmp file
        tmp = rp.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(rep, f, indent=2)
        tmp.replace(rp)
        updated_counts[rp] = len(updates)

    log.info("Updated %d sample reports.", len(updated_counts))
    return updated_counts


# ---------------------------------------------------------------------------
# Step 4 — re-aggregate reports
# ---------------------------------------------------------------------------

def _merge_trajectory_into_report(report_path: Path, traj_agg_path: Path) -> None:
    """Read trajectory_aggregate.json and inject it into report.json under metrics.trajectory."""
    if not traj_agg_path.exists():
        log.warning("trajectory_aggregate.json not found at %s — skipping merge", traj_agg_path)
        return
    with open(traj_agg_path) as f:
        traj = json.load(f)
    with open(report_path) as f:
        report = json.load(f)
    report["metrics"]["trajectory"] = traj
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = report_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    tmp.replace(report_path)
    log.info("Merged trajectory metrics into %s", report_path)


def reaggregate(run_dir: Path, dataset_name: str, eval_root: Path) -> None:
    """Re-generate dataset + run-level aggregate reports and merge trajectory data."""
    sys.path.insert(0, str(eval_root))
    from evaluation.common.report_generator import generate_dataset_report, generate_run_report

    dataset_dir = run_dir / dataset_name
    sample_reports = sorted(dataset_dir.rglob("*/*/report.json"))
    schema_version = "1.0"
    enabled_metrics = [
        "overall", "per_qa_type", "per_question_type", "per_graph_structure",
        "per_rung", "answer_stats", "binary_metrics", "reasoning_score",
    ]
    if sample_reports:
        try:
            rep = json.loads(sample_reports[0].read_text())
            schema_version = rep.get("schema_version", schema_version)
        except Exception:
            pass

    log.info("Re-generating dataset report …")
    generate_dataset_report(
        run_name=run_dir.name,
        dataset=dataset_name,
        dataset_dir=dataset_dir,
        report_path=dataset_dir / "report.json",
        schema_version=schema_version,
        enabled_metrics=enabled_metrics,
    )
    # Merge trajectory into dataset-level report
    traj_agg = dataset_dir / "trajectory_aggregate.json"
    _merge_trajectory_into_report(dataset_dir / "report.json", traj_agg)

    log.info("Re-generating run report …")
    generate_run_report(
        run_name=run_dir.name,
        run_dir=run_dir,
        report_path=run_dir / "report.json",
        schema_version=schema_version,
        enabled_metrics=enabled_metrics,
    )
    # Merge trajectory into run-level report too
    _merge_trajectory_into_report(run_dir / "report.json", traj_agg)

    log.info("Aggregation complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run-dir",  required=True, type=Path)
    p.add_argument("--dataset",  default="causal_nuscenes")
    p.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    p.add_argument("--model",    default="claude-sonnet-4-6")
    p.add_argument("--batch-api", action="store_true",
                   help="Use Anthropic Batch API (50%% cheaper; async; recommended)")
    p.add_argument("--concurrency", type=int, default=50,
                   help="Threads for concurrent (non-batch) mode")
    p.add_argument("--poll-interval", type=int, default=30,
                   help="Seconds between batch status polls")
    p.add_argument("--resume-batch-id", default="",
                   help="Skip submit; resume an existing batch by ID")
    p.add_argument("--dry-run", action="store_true",
                   help="Collect failures and report count without scoring")
    p.add_argument("--eval-root", type=Path,
                   default=Path(__file__).resolve().parent.parent.parent,
                   help="Root of neurips-dataset repo (default: auto-detect)")
    args = p.parse_args()

    eval_root = args.eval_root
    _load_prompts(eval_root)

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and not args.dry_run:
        sys.exit(f"ERROR: env var {args.api_key_env!r} is not set.")

    dataset_dir = args.run_dir / args.dataset
    if not dataset_dir.exists():
        sys.exit(f"ERROR: dataset dir not found: {dataset_dir}")

    # ── Step 1: collect failures ──────────────────────────────────────────
    failures = collect_failures(dataset_dir)
    print(f"\n{'='*60}")
    print(f"  Run dir : {args.run_dir}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Model   : {args.model}")
    print(f"  Mode    : {'batch API' if args.batch_api else f'concurrent (n={args.concurrency})'}")
    print(f"  To score: {len(failures)} questions")
    print(f"{'='*60}\n")

    if args.dry_run:
        # Also count empty-response questions
        import re
        n_empty = sum(
            1 for rp in dataset_dir.rglob("*/*/report.json")
            for q in (json.loads(rp.read_text()).get("qa_results", []) if json.loads(rp.read_text()).get("level") == "sample" else [])
            if q.get("llm_judge_score") is None
            and not re.sub(r"<think>.*?</think>", "", q.get("raw_output_text", "") or "", flags=re.DOTALL).strip()
        )
        print(f"[dry-run] Would score {len(failures)} questions ({n_empty} empty-response zero-scores). Exiting.")
        return

    # Zero-score questions with no model response (counts as 0, not excluded)
    n_empty = zero_score_empty_responses(dataset_dir)
    print(f"Zero-scored {n_empty} empty-response questions.")

    if not failures:
        print("Nothing to retry. All scoreable questions done.")
        reaggregate(args.run_dir, args.dataset, eval_root)
        return

    # ── Step 2: score ──────────────────────────────────────────────────────
    client = _make_client(api_key)
    checkpoint = args.run_dir / f".judge_batch_checkpoint_{args.dataset}.json"

    if args.batch_api:
        if args.resume_batch_id:
            batch_id = args.resume_batch_id
            log.info("Resuming batch %s", batch_id)
        else:
            batch_id = submit_batch(client, args.model, failures, checkpoint)

        poll_batch(client, batch_id, args.poll_interval)
        scores = retrieve_batch_results(client, batch_id, len(failures))
        # Remove checkpoint on success
        checkpoint.unlink(missing_ok=True)
    else:
        scores = score_concurrent(client, args.model, failures, args.concurrency)

    # ── Step 3: write back ─────────────────────────────────────────────────
    updated = write_scores_back(failures, scores)
    n_ok    = sum(1 for s in scores if "error" not in s)
    n_err   = sum(1 for s in scores if "error" in s)
    print(f"\nScored: {n_ok} OK, {n_err} errors (out of {len(scores)} attempted)")

    # ── Step 4: re-aggregate ───────────────────────────────────────────────
    reaggregate(args.run_dir, args.dataset, eval_root)

    print(f"\nDone. Check {args.run_dir}/report.json for updated aggregate metrics.")


if __name__ == "__main__":
    main()
