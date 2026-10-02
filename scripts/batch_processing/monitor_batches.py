#!/usr/bin/env python3
"""
Monitor active OpenAI batch jobs and display progress.

Usage:
    python monitor_batches.py                    # Monitor all active batches
    python monitor_batches.py --batch-id xxx     # Monitor specific batch
    python monitor_batches.py --watch            # Auto-refresh every 30 seconds
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

import os

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config as _load_config

os.environ["OPENAI_API_KEY"] = (_load_config() or {})["api_key"]


def _parse_created_at(created_at) -> datetime:
    """Parse created_at which may be a Unix timestamp int or ISO string."""
    if isinstance(created_at, (int, float)):
        return datetime.utcfromtimestamp(created_at)
    return datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).replace(tzinfo=None)


def format_time_elapsed(created_at) -> str:
    """Format time elapsed since batch creation."""
    created = _parse_created_at(created_at)
    elapsed = datetime.now() - created

    if elapsed.days > 0:
        return f"{elapsed.days}d {elapsed.seconds // 3600}h"
    elif elapsed.seconds > 3600:
        return f"{elapsed.seconds // 3600}h {(elapsed.seconds % 3600) // 60}m"
    else:
        return f"{elapsed.seconds // 60}m"


def get_batch_info(batch_id: str) -> Dict:
    """Get detailed info for a single batch."""
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    info = {
        "id": batch.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "metadata": batch.metadata or {},
        "request_counts": {
            "total": batch.request_counts.total,
            "completed": batch.request_counts.completed,
            "failed": batch.request_counts.failed,
        },
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "errors": batch.errors,
    }

    # Calculate progress percentage
    if info["request_counts"]["total"] > 0:
        info["progress"] = (info["request_counts"]["completed"] /
                          info["request_counts"]["total"] * 100)
    else:
        info["progress"] = 0

    return info


def list_active_batches(limit: int = 20) -> List[Dict]:
    """List all active (non-completed) batches."""
    client = OpenAI()
    batches = client.batches.list(limit=limit)

    active = []
    for batch in batches.data:
        if batch.status not in ["completed", "failed", "cancelled"]:
            info = {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "metadata": batch.metadata or {},
                "request_counts": {
                    "total": batch.request_counts.total,
                    "completed": batch.request_counts.completed,
                    "failed": batch.request_counts.failed,
                }
            }

            if info["request_counts"]["total"] > 0:
                info["progress"] = (info["request_counts"]["completed"] /
                                  info["request_counts"]["total"] * 100)
            else:
                info["progress"] = 0

            active.append(info)

    return active


def display_batch_status(batch_info: Dict, verbose: bool = False):
    """Display formatted batch status."""
    status_emoji = {
        "validating": "⏳",
        "in_progress": "🔄",
        "finalizing": "📝",
        "completed": "✅",
        "failed": "❌",
        "cancelled": "🚫",
    }.get(batch_info["status"], "❓")

    elapsed = format_time_elapsed(batch_info["created_at"])
    progress = batch_info["progress"]
    counts = batch_info["request_counts"]

    # Basic info
    print(f"\n{status_emoji} Batch: {batch_info['id']}")
    print(f"   Status: {batch_info['status']}")
    print(f"   Time: {elapsed}")

    # Description from metadata
    if "description" in batch_info["metadata"]:
        print(f"   Description: {batch_info['metadata']['description']}")

    # Progress bar
    if batch_info["status"] == "in_progress":
        bar_width = 40
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"   Progress: [{bar}] {progress:.1f}%")
        print(f"   Completed: {counts['completed']}/{counts['total']}")

        if counts["failed"] > 0:
            print(f"   ⚠️  Failed: {counts['failed']}")

        # Estimate time remaining
        if progress > 10:  # Only estimate after 10% completion
            elapsed_sec = (datetime.now() - _parse_created_at(batch_info["created_at"])).total_seconds()

            estimated_total = elapsed_sec / (progress / 100)
            remaining_sec = estimated_total - elapsed_sec

            if remaining_sec > 3600:
                remaining = f"{int(remaining_sec // 3600)}h {int((remaining_sec % 3600) // 60)}m"
            else:
                remaining = f"{int(remaining_sec // 60)}m"

            print(f"   Est. remaining: {remaining}")

    elif batch_info["status"] == "completed":
        print(f"   Completed: {counts['completed']}/{counts['total']}")
        if counts["failed"] > 0:
            print(f"   ⚠️  Failed: {counts['failed']}")
        if batch_info.get("output_file_id"):
            print(f"   Output: {batch_info['output_file_id']}")

    elif batch_info["status"] == "failed":
        if batch_info.get("errors"):
            print(f"   Error: {batch_info['errors']}")

    # Verbose mode shows more details
    if verbose:
        print(f"   Input file: {batch_info['metadata'].get('source_file', 'N/A')}")
        if batch_info.get("error_file_id"):
            print(f"   Error file: {batch_info['error_file_id']}")


def monitor_loop(batch_id: Optional[str] = None, interval: int = 30):
    """Monitor batches in a loop with auto-refresh."""
    print("Monitoring batches... (Press Ctrl+C to stop)")

    try:
        while True:
            # Clear screen (works on Unix-like systems)
            print("\033[2J\033[H")
            print(f"=== Batch Monitor === {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            if batch_id:
                # Monitor specific batch
                try:
                    info = get_batch_info(batch_id)
                    display_batch_status(info, verbose=True)

                    if info["status"] in ["completed", "failed", "cancelled"]:
                        print(f"\n✓ Batch {info['status']}. Monitoring stopped.")
                        break
                except Exception as e:
                    print(f"Error fetching batch {batch_id}: {e}")
            else:
                # Monitor all active batches
                active = list_active_batches()

                if not active:
                    print("\nNo active batches found.")
                else:
                    print(f"\nFound {len(active)} active batch(es):")
                    for info in active:
                        display_batch_status(info)

            print(f"\nRefreshing in {interval} seconds...")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def check_batch_files(batch_dir: Path = Path("batch_jobs")):
    """Check for local batch info files to identify recent batches."""
    print(f"\n=== Recent Batch Files in {batch_dir} ===")

    info_files = sorted(batch_dir.glob("*.batch_info.json"),
                       key=lambda x: x.stat().st_mtime, reverse=True)

    if not info_files:
        print("No batch info files found.")
        return

    for info_file in info_files[:10]:  # Show last 10
        with open(info_file) as f:
            info = json.load(f)

        batch_id = info.get("batch_id", "unknown")
        source = Path(info.get("source_file", "")).name
        submitted = info.get("submitted_at", "")

        # Try to get current status
        try:
            current = get_batch_info(batch_id)
            status_emoji = {
                "validating": "⏳",
                "in_progress": "🔄",
                "finalizing": "📝",
                "completed": "✅",
                "failed": "❌",
                "cancelled": "🚫",
            }.get(current["status"], "❓")

            print(f"{status_emoji} {batch_id} | {source} | {current['status']}")

        except:
            print(f"❓ {batch_id} | {source} | (unable to fetch status)")


def main():
    parser = argparse.ArgumentParser(description="Monitor OpenAI batch jobs")
    parser.add_argument("--batch-id", help="Specific batch ID to monitor")
    parser.add_argument("--watch", action="store_true",
                       help="Auto-refresh every N seconds")
    parser.add_argument("--interval", type=int, default=30,
                       help="Refresh interval in seconds (default: 30)")
    parser.add_argument("--check-files", action="store_true",
                       help="Check local batch info files")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed information")

    args = parser.parse_args()

    if args.check_files:
        check_batch_files()
        return

    if args.watch:
        monitor_loop(args.batch_id, args.interval)
    elif args.batch_id:
        try:
            info = get_batch_info(args.batch_id)
            display_batch_status(info, verbose=args.verbose)

            # Show download command if completed
            if info["status"] == "completed":
                print(f"\n💡 Download results with:")
                print(f"   python submit_batch.py download --batch-id {args.batch_id}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Show all active batches once
        active = list_active_batches()

        if not active:
            print("No active batches found.")
            print("\nRecent batch files:")
            check_batch_files()
        else:
            print(f"Found {len(active)} active batch(es):")
            for info in active:
                display_batch_status(info, verbose=args.verbose)

            print(f"\n💡 To monitor continuously, use --watch")
            print(f"💡 To monitor a specific batch: --batch-id <id>")


if __name__ == "__main__":
    main()