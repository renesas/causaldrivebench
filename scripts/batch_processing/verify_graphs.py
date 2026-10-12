#!/usr/bin/env python3
"""
Standalone graph verification pipeline using Claude Haiku.

For each scene that has a *_pruned.json in the graphs/outputs directory,
sends T=0 images + condensed causal graph to Haiku and asks it to verify
node classifications (active/dormant/distractor) and edge justifications.

Scenes where Haiku disagrees are flagged in verification_report.json.

Output layout (flat, alongside *_pruned.json):
    {graphs_dir}/
        nuscenes-scene-XXXX_SAMPLED_N_verification.json   ← per scene
        verification_report.json                           ← aggregate summary

Usage:
    # Use paths from config.yaml
    python verify_graphs.py --dataset nuscenes

    # Override graphs dir (e.g. nuscenes_full instead of nuscenes)
    python verify_graphs.py --dataset nuscenes \\
        --graphs outputs/nuscenes_full

    # Limit scenes for a quick test
    python verify_graphs.py --dataset nuscenes --max-samples 5

    # Skip submission (just prepare the batch JSONL)
    python verify_graphs.py --dataset nuscenes --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config_loader import load_config
from common.llm_provider import get_provider_from_config

from prepare_batches import prepare_stage_verify_batch
from process_results import process_verify_results, parse_claude_response, _load_id_map

# ── Constants ─────────────────────────────────────────────────────────────────
VERIFY_MODEL    = "claude-haiku-4-5-20251001"
VERIFY_PROVIDER = "claude"
POLL_INTERVAL   = 30  # seconds between status checks

# Haiku batch pricing (claude-haiku-4-5, batch API rates).
_INPUT_PRICE_PER_TOK  = 0.50 / 1_000_000   # $0.50 per MTok
_OUTPUT_PRICE_PER_TOK = 2.50 / 1_000_000   # $2.50 per MTok
_AVG_OUTPUT_TOK       = 800   # verification responses are short (node/edge verdicts)


def _estimate_cost(batch_file: Path) -> None:
    """Print a token and cost estimate for the prepared batch JSONL.

    Uses the same heuristic as dry_run_validate.sh:
      input_tokens ≈ text_chars / 4 + n_images * 1000

    Image token estimate of 1000 is conservative for ~600×600 images.
    Output is estimated at AVG_OUTPUT_TOK per request.
    """
    total_input = total_output = n_req = 0

    with open(batch_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            req  = json.loads(line)
            body = req.get("body", req.get("params", {}))
            msgs = body.get("messages", [])

            n_images = text_chars = 0
            for msg in msgs:
                for block in msg.get("content", []):
                    if not isinstance(block, dict):
                        continue
                    t = block.get("type", "")
                    if t in ("image", "image_url"):
                        n_images += 1
                    elif t == "text":
                        text_chars += len(block.get("text", ""))
                    elif t == "input_image":
                        n_images += 1
                    elif t == "input_text":
                        text_chars += len(block.get("text", ""))

            sys_prompt = body.get("system", "")
            if isinstance(sys_prompt, list):
                text_chars += sum(len(s.get("text", "")) for s in sys_prompt)
            else:
                text_chars += len(sys_prompt)

            total_input  += text_chars // 4 + n_images * 1_000
            total_output += _AVG_OUTPUT_TOK
            n_req += 1

    if n_req == 0:
        return

    cost = total_input * _INPUT_PRICE_PER_TOK + total_output * _OUTPUT_PRICE_PER_TOK
    avg_in = total_input // n_req

    print(f"\n  Cost estimate ({VERIFY_MODEL}, batch API rates):")
    print(f"    Requests:      {n_req}")
    print(f"    Input tokens:  ~{total_input/1_000:.0f}K  (~{avg_in/1_000:.1f}K / request)")
    print(f"    Output tokens: ~{total_output/1_000:.0f}K  (~{_AVG_OUTPUT_TOK} / request, estimated)")
    print(f"    Estimated cost: ${cost:.2f}")
    print(f"    Pricing: ${_INPUT_PRICE_PER_TOK*1e6:.2f}/MTok in, ${_OUTPUT_PRICE_PER_TOK*1e6:.2f}/MTok out")


def _poll_until_done(provider, batch_id: str) -> bool:
    """Poll batch status until completed/failed. Returns True on success."""
    print(f"  Waiting for batch {batch_id} to complete (polling every {POLL_INTERVAL}s)...")
    start = time.time()
    while True:
        status = provider.get_batch_status(batch_id)
        state  = status["status"]
        total  = status["total"]
        done   = status["succeeded"] + status["errored"]
        elapsed = int(time.time() - start)
        print(f"\r  Status: {state}  |  {done}/{total}  |  {elapsed//60}m{elapsed%60:02d}s", end="", flush=True)

        if state == "completed":
            print()
            return True
        if state in ("failed", "cancelled", "expired"):
            print(f"\nBatch {state}.")
            return False

        time.sleep(POLL_INTERVAL)


def run_verification(
    records_dir: Path,
    graphs_dir: Path,
    dataset: str,
    batch_dir: Path,
    results_dir: Path,
    max_samples: int = None,
    dry_run: bool = False,
    force: bool = False,
):
    """End-to-end verification: prepare → submit → poll → download → process."""

    cfg = load_config() or {}
    provider = get_provider_from_config({
        **cfg,
        "provider": VERIFY_PROVIDER,
        "model":    VERIFY_MODEL,
        "reasoning": "none",
    })

    # ── 1. Prepare ────────────────────────────────────────────────────────────
    print(f"\n[1/4] Preparing batch requests...")
    batch_file = prepare_stage_verify_batch(
        records_dir=records_dir,
        graphs_dir=graphs_dir,
        dataset=dataset,
        output_dir=batch_dir,
        max_samples=max_samples,
        provider=provider,
        force=force,
    )

    # Count requests
    n_requests = sum(1 for line in open(batch_file) if line.strip())
    if n_requests == 0:
        print("  No scenes to verify — all already done or no pruned graphs found.")
        return

    print(f"  Queued {n_requests} scene(s) → {batch_file}")
    _estimate_cost(batch_file)

    if dry_run:
        print("\n  [dry-run] Stopping before submission.")
        return

    # ── 2. Split into chunks if > 95 MB ───────────────────────────────────────
    MAX_BYTES = 95 * 1024 * 1024
    file_size = batch_file.stat().st_size
    id_map = _load_id_map(batch_dir)

    if file_size <= MAX_BYTES:
        chunk_files = [batch_file]
    else:
        chunk_dir = batch_dir / f"chunks_{batch_file.stem}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_files = []
        chunk, size, idx = [], 0, 0
        with open(batch_file) as f:
            for line in f:
                lb = len(line.encode())
                if size + lb > MAX_BYTES and chunk:
                    p = chunk_dir / f"chunk_{idx:03d}.jsonl"
                    p.write_text("".join(chunk))
                    chunk_files.append(p)
                    chunk, size, idx = [], 0, idx + 1
                chunk.append(line)
                size += lb
        if chunk:
            p = chunk_dir / f"chunk_{idx:03d}.jsonl"
            p.write_text("".join(chunk))
            chunk_files.append(p)
        print(f"  File size {file_size/1e6:.1f}MB > 95MB — split into {len(chunk_files)} chunks")

    # ── 3. Submit all chunks ───────────────────────────────────────────────────
    print(f"\n[2/4] Submitting {len(chunk_files)} batch chunk(s) to {VERIFY_PROVIDER} ({VERIFY_MODEL})...")
    batch_ids = []
    for i, cf in enumerate(chunk_files):
        requests = [json.loads(l) for l in cf.open() if l.strip()]
        label = f"{dataset} graph verification" if len(chunk_files) == 1 else f"{dataset} graph verification chunk {i+1}/{len(chunk_files)}"
        bid = provider.submit_batch(requests, description=label)
        print(f"  Chunk {i+1}: {bid}")
        batch_ids.append(bid)

    # ── 4. Poll all chunks in parallel ────────────────────────────────────────
    print(f"\n[3/4] Waiting for {len(batch_ids)} batch(es) to complete...")
    import threading
    results = {}
    def _poll(bid):
        results[bid] = _poll_until_done(provider, bid)
    threads = [threading.Thread(target=_poll, args=(bid,)) for bid in batch_ids]
    for t in threads: t.start()
    for t in threads: t.join()

    failed = [bid for bid, ok in results.items() if not ok]
    if failed:
        print(f"WARNING: {len(failed)} chunk(s) failed: {failed}")

    # ── 5. Download + Process each chunk ──────────────────────────────────────
    print(f"\n[4/4] Downloading and processing results...")
    results_dir.mkdir(parents=True, exist_ok=True)
    total_success = total_errors = 0
    for bid in batch_ids:
        if not results.get(bid):
            print(f"  Skipping failed batch {bid}")
            continue
        results_path = str(results_dir / f"{bid}_results.jsonl")
        provider.download_batch_results(bid, results_path)
        print(f"  Results → {results_path}")
        s, e = process_verify_results(
            results_file=Path(results_path),
            output_dir=graphs_dir,
            parse_fn=parse_claude_response,
            id_map=id_map,
        )
        total_success += s
        total_errors += e

    print(f"\nDone.  {total_success} verified,  {total_errors} errors.")
    print(f"Verification files: {graphs_dir}/*_verification.json")
    print(f"Report:             {graphs_dir}/verification_report.json")


def main():
    parser = argparse.ArgumentParser(
        description="Run Haiku graph verification independently of the main pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True,
                        choices=["nuscenes", "openscene", "argoverse2"],
                        help="Dataset to verify")
    parser.add_argument("--graphs",
                        help="Path to outputs/graphs directory (default: from config.yaml). "
                             "Verification JSONs are written here alongside *_pruned.json.")
    parser.add_argument("--records",
                        help="Path to records directory (default: from config.yaml)")
    parser.add_argument("--batch-dir",
                        help="Directory for batch JSONL files (default: from config.yaml). "
                             "When set, results are stored in <batch-dir>/batch_results/")
    parser.add_argument("--max-samples", type=int,
                        help="Limit to first N samples (useful for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare the batch JSONL but do not submit")
    parser.add_argument("--force", action="store_true",
                        help="Re-verify scenes that already have a _verification.json")
    parser.add_argument("--config",
                        help="Path to config.yaml (default: auto-discover)")

    args = parser.parse_args()

    # Load config
    cfg_path = args.config or None
    cfg      = load_config(cfg_path) or {}
    ds_cfg   = cfg.get(args.dataset, {})
    batch_cfg = cfg.get("batch", {})

    records_dir = Path(args.records or ds_cfg.get("records") or "")
    graphs_dir  = Path(args.graphs  or ds_cfg.get("outputs") or "")
    batch_dir = Path(args.batch_dir or batch_cfg.get("batch_dir", "batch_jobs"))
    # Mirror run_batch_pipeline.sh: if --batch-dir is explicit, results live inside it;
    # otherwise use the config's dedicated results_dir.
    if args.batch_dir:
        results_dir = batch_dir / "batch_results"
    else:
        results_dir = Path(batch_cfg.get("results_dir", "batch_results"))

    if not records_dir or not records_dir.exists():
        print(f"ERROR: records directory not found: {records_dir}")
        print("  Set via --records or config.yaml nuscenes.records")
        sys.exit(1)
    if not graphs_dir or not graphs_dir.exists():
        print(f"ERROR: graphs/outputs directory not found: {graphs_dir}")
        print("  Set via --graphs or config.yaml nuscenes.outputs")
        sys.exit(1)

    print(f"Dataset:     {args.dataset}")
    print(f"Records:     {records_dir}")
    print(f"Graphs dir:  {graphs_dir}")
    print(f"Batch dir:   {batch_dir}")
    print(f"Results dir: {results_dir}")
    if args.max_samples:
        print(f"Max samples:  {args.max_samples}")
    if args.dry_run:
        print(f"Mode:        dry-run (no submission)")

    batch_dir.mkdir(parents=True, exist_ok=True)

    run_verification(
        records_dir=records_dir,
        graphs_dir=graphs_dir,
        dataset=args.dataset,
        batch_dir=batch_dir,
        results_dir=results_dir,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
