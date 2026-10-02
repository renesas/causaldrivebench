#!/usr/bin/env python3
"""Backfill scene_context.environment_conditions = "Clear daytime" for verified
graphs missing the field.

Usage:
    # Preview
    python scripts/graph_postprocessing/backfill_environment.py \
        --graph-dir /path/to/nuscenes_full_final --dry-run

    # Apply
    python scripts/graph_postprocessing/backfill_environment.py \
        --graph-dir /path/to/nuscenes_full_final
"""

import argparse
import glob
import json
from pathlib import Path

DEFAULT_VALUE = "Clear daytime"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--suffix", default="_verified.json")
    p.add_argument("--default", default=DEFAULT_VALUE)
    args = p.parse_args()

    files = sorted(glob.glob(str(Path(args.graph_dir) / f"*{args.suffix}")))
    print(f"Found {len(files)} files")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY'}")
    print(f"Default value: '{args.default}'")
    print()

    changed = 0
    for fp in files:
        with open(fp) as f:
            g = json.load(f)
        ctx = g.setdefault("scene_context", {})
        if ctx.get("environment_conditions", "").strip():
            continue  # already has a value
        ctx["environment_conditions"] = args.default
        changed += 1
        if not args.dry_run:
            with open(fp, "w") as f:
                json.dump(g, f, indent=2)

    print(f"Files needing backfill: {changed}/{len(files)}")
    if args.dry_run:
        print("(dry-run — no files written)")
    else:
        print(f"Wrote {changed} files.")


if __name__ == "__main__":
    main()
