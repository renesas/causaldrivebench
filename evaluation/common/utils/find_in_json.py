#!/usr/bin/env python3
"""
Two-condition JSON file search.

Condition 1: find all JSON files containing a top-level or nested key (fast text pre-filter).
Condition 2: within dicts that carry that key, check a dotted sub-path against allowed values.

Usage examples
--------------
# Find files with alternate_trajectory where rung is 2 or 3
python find_in_json.py \
    --dir /path/to/causal_nuscenes \
    --cond1 alternate_trajectory \
    --cond2 alternate_trajectory.rung=2,3

# Same but only look in *_qa.json files, save output
python find_in_json.py \
    --dir /path/to/causal_nuscenes \
    --cond1 alternate_trajectory \
    --cond2 alternate_trajectory.rung=2,3 \
    --pattern "*_qa.json" \
    --output results.txt

# Show full matching dict instead of a summary line
python find_in_json.py \
    --dir /path/to/causal_nuscenes \
    --cond1 alternate_trajectory \
    --cond2 alternate_trajectory.rung=2,3 \
    --full-match
"""

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path
from typing import Any, Generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_nested(obj: Any, dotted_path: str) -> Any:
    """Resolve a dot-separated key path into obj; return _MISSING on failure."""
    _MISSING = object()
    parts = dotted_path.split(".")
    cur = obj
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _iter_dicts(obj: Any) -> Generator[dict, None, None]:
    """Yield every dict reachable from obj (depth-first, including obj itself)."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_dicts(item)


def _find_files(root: str, pattern: str) -> Generator[Path, None, None]:
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fnmatch.fnmatch(fname, pattern):
                yield Path(dirpath) / fname


def _text_matches(path: Path, text: str) -> bool:
    """Fast text pre-filter — avoids parsing files that can't match."""
    try:
        return text in path.read_text(errors="replace")
    except OSError:
        return False


def _coerce(raw: str) -> Any:
    """Try to convert a string to int/float/bool before comparing."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def search(
    root: str,
    cond1_key: str,
    cond2_spec: str,           # e.g. "alternate_trajectory.rung=2,3"
    pattern: str = "*.json",
    full_match: bool = False,
    output_path: str | None = None,
):
    # Parse cond2
    if "=" not in cond2_spec:
        sys.exit(f"--cond2 must be in the form KEY.PATH=VAL1,VAL2  Got: {cond2_spec!r}")
    cond2_path, raw_vals = cond2_spec.split("=", 1)
    allowed = {_coerce(v.strip()) for v in raw_vals.split(",")}

    lines_out: list[str] = []

    def emit(line: str):
        print(line)
        lines_out.append(line)

    file_count = match_count = 0

    for fpath in sorted(_find_files(root, pattern)):
        # Condition 1: text pre-filter
        if not _text_matches(fpath, cond1_key):
            continue
        file_count += 1

        try:
            data = json.loads(fpath.read_text())
        except json.JSONDecodeError as exc:
            emit(f"[WARN] Cannot parse {fpath}: {exc}")
            continue

        file_hits: list[str] = []

        for d in _iter_dicts(data):
            if cond1_key not in d:
                continue
            # Condition 2: check the sub-path value
            val = _get_nested(d, cond2_path)
            if val is not object and val in allowed:
                match_count += 1
                if full_match:
                    snippet = json.dumps(d, indent=2)
                else:
                    # Compact summary: show the cond2 field plus "frame" sibling if present
                    at = d[cond1_key]
                    frame = at.get("frame", "(no frame)") if isinstance(at, dict) else "?"
                    rung  = val
                    # Try to get a question id if parent has one
                    q_id = d.get("id", "")
                    snippet = f"  id={q_id!r:8s}  {cond2_path}={rung}  frame={frame}"
                file_hits.append(snippet)

        if file_hits:
            emit(f"\n=== {fpath}")
            for h in file_hits:
                emit(h)

    # Summary
    summary = f"\nSummary: scanned {file_count} files matching cond1={cond1_key!r}; found {match_count} dict(s) where {cond2_spec}"
    emit(summary)

    if output_path:
        Path(output_path).write_text("\n".join(lines_out) + "\n")
        print(f"[saved → {output_path}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Find JSON files by two conditions and report matching entries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dir",     required=True,  help="Root directory to search")
    ap.add_argument("--cond1",   required=True,  help="Key that must exist in a dict (also used as text pre-filter)")
    ap.add_argument("--cond2",   required=True,  help="Dotted path + allowed values: PATH.TO.KEY=VAL1,VAL2")
    ap.add_argument("--pattern", default="*.json", help="Glob filename pattern (default: *.json)")
    ap.add_argument("--output",  default=None,   help="Optional file to write output to")
    ap.add_argument("--full-match", action="store_true", help="Print entire matching dict instead of summary line")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        sys.exit(f"Directory not found: {args.dir}")

    search(
        root=args.dir,
        cond1_key=args.cond1,
        cond2_spec=args.cond2,
        pattern=args.pattern,
        full_match=args.full_match,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
