#!/usr/bin/env python3
"""Estimate precise input token counts for a batch JSONL file using Anthropic's count_tokens API.

Usage:
    python estimate_tokens.py <batch.jsonl> [--samples N]

Samples N requests from the JSONL (default 5), calls count_tokens for each,
and extrapolates to the full file.
"""
import argparse
import json
import random
import sys

import anthropic


def main():
    parser = argparse.ArgumentParser(description="Estimate tokens for a batch JSONL")
    parser.add_argument("jsonl", help="Path to batch JSONL file")
    parser.add_argument("--samples", type=int, default=5, help="Number of requests to sample (default 5)")
    args = parser.parse_args()

    # Read all lines
    with open(args.jsonl) as f:
        lines = f.readlines()

    total_requests = len(lines)
    if total_requests == 0:
        print("Empty JSONL file.")
        sys.exit(1)

    n_samples = min(args.samples, total_requests)
    sample_indices = sorted(random.sample(range(total_requests), n_samples))

    client = anthropic.Anthropic()
    token_counts = []

    print(f"File: {args.jsonl}")
    print(f"Total requests: {total_requests}")
    print(f"Sampling {n_samples} requests for token counting...\n")

    for i, idx in enumerate(sample_indices):
        req = json.loads(lines[idx])
        body = req.get("body", req.get("params", {}))

        model = body.get("model", "claude-opus-4-6")
        messages = body.get("messages", [])
        system = body.get("system", "")

        kwargs = dict(model=model, messages=messages)
        if system:
            if isinstance(system, str):
                kwargs["system"] = system
            else:
                kwargs["system"] = system

        resp = client.messages.count_tokens(**kwargs)
        token_counts.append(resp.input_tokens)
        print(f"  [{i+1}/{n_samples}] Request #{idx}: {resp.input_tokens:,} input tokens")

    avg_input = sum(token_counts) / len(token_counts)
    min_input = min(token_counts)
    max_input = max(token_counts)
    total_input_est = avg_input * total_requests

    # Pricing (Anthropic batch API: 50% discount on standard)
    INPUT_PRICE = 2.50 / 1_000_000   # Claude Opus batch input: $2.50/MTok
    OUTPUT_PRICE = 12.50 / 1_000_000  # Claude Opus batch output: $12.50/MTok
    AVG_OUTPUT_TOK = 2600  # measured from 5-sample test batch

    total_output_est = AVG_OUTPUT_TOK * total_requests
    input_cost = total_input_est * INPUT_PRICE
    output_cost = total_output_est * OUTPUT_PRICE

    print(f"\n{'='*55}")
    print(f"  Sampled input tokens:")
    print(f"    avg: {avg_input:,.0f}   min: {min_input:,}   max: {max_input:,}")
    print(f"\n  Extrapolated to {total_requests} requests:")
    print(f"    Total input:  ~{total_input_est/1_000_000:.2f}M tokens")
    print(f"    Total output: ~{total_output_est/1_000_000:.2f}M tokens (est {AVG_OUTPUT_TOK}/req)")
    print(f"\n  Estimated cost (Opus batch pricing):")
    print(f"    Input:  ${input_cost:.2f}")
    print(f"    Output: ${output_cost:.2f} (assumes {AVG_OUTPUT_TOK} output tok/req)")
    print(f"    TOTAL:  ${input_cost + output_cost:.2f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
