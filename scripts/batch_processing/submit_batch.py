#!/usr/bin/env python3
"""
Submit and manage batch jobs via any supported LLM provider.

Uses llm_provider.py for provider-agnostic batch operations.
Supports: OpenAI, Claude (Anthropic). Reads provider from config.yaml.

Usage:
    python submit_batch.py submit   --file batch_jobs/nuscenes_stage1.jsonl
    python submit_batch.py status   --batch-id <id>
    python submit_batch.py list
    python submit_batch.py download --batch-id <id> --output batch_results/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config as _load_config
from common.llm_provider import get_provider_from_config


def _get_provider(args_provider=None):
    """Get provider from args or config."""
    cfg = _load_config() or {}
    if args_provider:
        cfg["provider"] = args_provider
    return get_provider_from_config(cfg)


def submit_batch(batch_file: Path, provider, description: str = None) -> str:
    """Submit a batch file and return the batch ID."""
    requests = []
    with open(batch_file) as f:
        for line in f:
            line = line.strip()
            if line:
                requests.append(json.loads(line))

    if not requests:
        print("ERROR: No requests in batch file")
        sys.exit(1)

    print(f"Submitting {len(requests)} requests via {provider.provider_name}...")
    batch_id = provider.submit_batch(requests, description or f"Batch from {batch_file.name}")

    print(f"  Batch created: {batch_id}")

    # Save batch info
    batch_info = {
        "batch_id": batch_id,
        "provider": provider.provider_name,
        "source_file": str(batch_file),
        "submitted_at": datetime.now().isoformat(),
        "description": description or f"Batch from {batch_file.name}",
        "n_requests": len(requests),
    }
    info_file = batch_file.with_suffix(".batch_info.json")
    with open(info_file, "w") as f:
        json.dump(batch_info, f, indent=2)

    print(f"  Batch info saved: {info_file}")
    return batch_id


def check_batch_status(batch_id: str, provider):
    """Check status of a batch."""
    status = provider.get_batch_status(batch_id)
    total = status["total"]
    done = status["succeeded"] + status["errored"]

    print(f"\nBatch {batch_id} ({provider.provider_name}):")
    print(f"  Status: {status['status']}")
    print(f"  Progress: {done}/{total}")
    print(f"  Succeeded: {status['succeeded']}  Errored: {status['errored']}")

    return status["status"]


def list_batches(provider, limit: int = 10):
    """List recent batches."""
    batches = provider.list_batches(limit)
    print(f"\nRecent {provider.provider_name} batches:")
    for b in batches:
        total = b.get("total", 0)
        succ = b.get("succeeded", 0)
        print(f"  {b['id']} | {b['status']:12} | {succ}/{total} | {b.get('description','')}")


def download_results(batch_id: str, provider, output_dir: Path = None):
    """Download results for a completed batch."""
    if output_dir is None:
        output_dir = Path("batch_results")
    output_dir.mkdir(exist_ok=True)

    output_path = str(output_dir / f"{batch_id}_results.jsonl")
    print(f"Downloading results for {batch_id}...")

    result = provider.download_batch_results(batch_id, output_path)
    if result is None:
        print(f"  Batch not ready or failed")
        return None

    # Count results
    n_success = n_error = 0
    with open(output_path) as f:
        for line in f:
            _, raw_text, usage = provider.parse_batch_response(line)
            if "error" in usage:
                n_error += 1
            else:
                n_success += 1

    print(f"  Results saved: {output_path}")
    print(f"  Succeeded: {n_success}  Errors: {n_error}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Submit and manage batch jobs")
    parser.add_argument("--provider", help="Override provider (openai/claude)")
    subparsers = parser.add_subparsers(dest="command")

    sub = subparsers.add_parser("submit")
    sub.add_argument("--file", required=True)
    sub.add_argument("--description")

    sub = subparsers.add_parser("status")
    sub.add_argument("--batch-id", required=True)

    sub = subparsers.add_parser("list")
    sub.add_argument("--limit", type=int, default=10)

    sub = subparsers.add_parser("download")
    sub.add_argument("--batch-id", required=True)
    sub.add_argument("--output")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    provider = _get_provider(args.provider)

    if args.command == "submit":
        batch_file = Path(args.file)
        if not batch_file.exists():
            print(f"ERROR: File not found: {batch_file}")
            sys.exit(1)
        batch_id = submit_batch(batch_file, provider, args.description)
        print(f"\nBatch ID: {batch_id}")
        print(f"Check status: python {Path(__file__).name} status --batch-id {batch_id}")

    elif args.command == "status":
        check_batch_status(args.batch_id, provider)

    elif args.command == "list":
        list_batches(provider, args.limit)

    elif args.command == "download":
        output_dir = Path(args.output) if args.output else None
        download_results(args.batch_id, provider, output_dir)


if __name__ == "__main__":
    main()
