#!/usr/bin/env python3
"""Shuffle MCQ option order in QA result JSONs so the correct answer
is uniformly distributed across A/B/C/D.

Updates:
  - options list order
  - correct_answer letter
  - reasoning text (replaces letter references)
  - meta.distractor_types keys

Usage:
  python shuffle_mcq_options.py --qa-dir /path/to/outputs
  python shuffle_mcq_options.py --qa-dir /path/to/outputs --dry-run
"""

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

LETTERS = ["A", "B", "C", "D"]


def shuffle_mcq(question: dict, rng: random.Random) -> dict:
    """Shuffle a single MCQ question in-place and return it."""
    options = question.get("options", [])
    if len(options) != 4:
        return question

    old_correct = question.get("correct_answer", "")
    if old_correct not in LETTERS:
        return question

    # Build old letter -> option text mapping
    old_map = {}  # letter -> option text (without "X) " prefix)
    for i, opt in enumerate(options):
        letter = LETTERS[i]
        # Strip "A) " prefix
        text = re.sub(r"^[A-D]\)\s*", "", opt)
        old_map[letter] = text

    # Shuffle indices
    indices = list(range(4))
    rng.shuffle(indices)

    # Build new ordering: indices[new_pos] = old_pos
    # new_pos 0 gets old index indices[0], etc.
    new_options = []
    old_to_new = {}  # old letter -> new letter
    new_distractor_types = {}

    for new_pos, old_pos in enumerate(indices):
        old_letter = LETTERS[old_pos]
        new_letter = LETTERS[new_pos]
        old_to_new[old_letter] = new_letter
        new_options.append(f"{new_letter}) {old_map[old_letter]}")

        # Remap distractor_types
        dt = question.get("meta", {}).get("distractor_types", {})
        if old_letter in dt:
            new_distractor_types[new_letter] = dt[old_letter]

    question["options"] = new_options
    question["correct_answer"] = old_to_new[old_correct]

    # Update reasoning — replace letter references like "A is correct", "B is wrong"
    reasoning = question.get("reasoning", "")
    if reasoning:
        question["reasoning"] = remap_letters_in_text(reasoning, old_to_new)

    # Update distractor_types
    if question.get("meta", {}).get("distractor_types"):
        question["meta"]["distractor_types"] = new_distractor_types

    return question


def remap_letters_in_text(text: str, old_to_new: dict) -> str:
    """Replace standalone letter references (A, B, C, D) in reasoning text
    with their new letters, using a two-pass approach to avoid collisions."""
    # Use placeholders first to avoid A->C then C->D issues
    placeholders = {}
    for old_letter, new_letter in old_to_new.items():
        ph = f"__PLACEHOLDER_{old_letter}__"
        placeholders[ph] = new_letter
        # Match "A " at start, " A ", "A)" patterns — standalone letter refs
        # Pattern: letter followed by space, ), is, was — or preceded by space/(
        text = re.sub(
            rf'(?<![a-zA-Z]){old_letter}(?=\s|[).,;:]|$| is | was )',
            ph,
            text,
        )

    # Replace placeholders with new letters
    for ph, new_letter in placeholders.items():
        text = text.replace(ph, new_letter)

    return text


def process_file(filepath: str, rng: random.Random, dry_run: bool) -> int:
    """Process a single QA JSON file. Returns count of shuffled MCQs."""
    with open(filepath) as f:
        data = json.load(f)

    questions = data.get("questions", []) if isinstance(data, dict) else []
    if not questions:
        return 0

    count = 0
    for q in questions:
        if not isinstance(q, dict):
            continue
        if q.get("answer_format") != "mcq":
            continue
        shuffle_mcq(q, rng)
        count += 1

    if count > 0 and not dry_run:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return count


def main():
    parser = argparse.ArgumentParser(description="Shuffle MCQ options in QA JSONs")
    parser.add_argument(
        "--qa-dir",
        required=True,
        help="Root outputs directory (searches recursively for *_qa.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    files = sorted(glob.glob(f"{args.qa_dir}/**/*_qa.json", recursive=True))
    files = [f for f in files if "_changes" not in f]

    if not files:
        print(f"No *_qa.json files found under {args.qa_dir}")
        sys.exit(1)

    total_files = 0
    total_mcqs = 0
    for fp in files:
        n = process_file(fp, rng, args.dry_run)
        if n > 0:
            total_files += 1
            total_mcqs += n
            rel = Path(fp).relative_to(args.qa_dir)
            print(f"  {rel}: {n} MCQs shuffled")

    action = "would shuffle" if args.dry_run else "shuffled"
    print(f"\n{action} {total_mcqs} MCQs across {total_files} files")

    # Show new distribution
    if not args.dry_run and total_mcqs > 0:
        dist = {l: 0 for l in LETTERS}
        for fp in files:
            with open(fp) as f:
                data = json.load(f)
            for q in data.get("questions", []) if isinstance(data, dict) else []:
                if isinstance(q, dict) and q.get("answer_format") == "mcq":
                    dist[q["correct_answer"]] = dist.get(q["correct_answer"], 0) + 1
        print("New correct_answer distribution:")
        for l in LETTERS:
            pct = dist[l] / total_mcqs * 100 if total_mcqs else 0
            print(f"  {l}: {dist[l]} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
