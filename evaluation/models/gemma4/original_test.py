"""
Gemma 4 API connectivity test.

Sends a single text+image request to the model server using a real benchmark
image (from sample_data/) to verify the model is correctly responding.

Used by the `original` docker-compose service (--profile original) to confirm
the selected model variant is healthy before running the full benchmark.

Usage:
    python original_test.py \
        --api-base-url http://gemma4-server:8080/v1 \
        --model-name ai/gemma4:e2b \
        --bench-dir /workspace/data/causal_drive_bench/causal_nuscenes
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _encode_image(path: Path, size: int = 224) -> str:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 API connectivity test")
    parser.add_argument("--api-base-url", default="http://gemma4-server:8080/v1")
    parser.add_argument("--model-name", default="ai/gemma4:e2b")
    parser.add_argument(
        "--bench-dir",
        default="/workspace/data/causal_drive_bench/causal_nuscenes",
        help="Path to a benchmark directory (used to find a sample image).",
    )
    args = parser.parse_args()

    import openai
    client = openai.OpenAI(base_url=args.api_base_url, api_key="none")

    # ── List available models ──────────────────────────────────────────────
    print(f"\n[Test] Connecting to {args.api_base_url} …")
    try:
        models = client.models.list()
        print(f"[Test] Available models: {[m.id for m in models.data]}")
    except Exception as exc:
        print(f"[Test] WARNING: Could not list models: {exc}")

    # ── Find a sample image from the benchmark dataset ─────────────────────
    sample_image: Path | None = None
    bench_dir = Path(args.bench_dir)
    if bench_dir.exists():
        for jpg in bench_dir.rglob("*.jpg"):
            sample_image = jpg
            break

    if sample_image is None:
        print("[Test] No sample image found — falling back to text-only test.")
        messages = [
            {"role": "system", "content": "You are a driving assistant."},
            {"role": "user",   "content": "Is a red traffic light a reason to stop? Answer: Yes or No."},
        ]
    else:
        print(f"[Test] Using image: {sample_image}")
        b64 = _encode_image(sample_image)
        messages = [
            {"role": "system", "content": "You are an autonomous driving assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {
                        "type": "text",
                        "text": (
                            "This is a camera image from a self-driving vehicle. "
                            "Is there any object that could require the vehicle to slow down or stop? "
                            "Answer: Yes or No, then give a brief reason."
                        ),
                    },
                ],
            },
        ]

    # ── Send test request ─────────────────────────────────────────────────
    print(f"[Test] Sending request to model '{args.model_name}' …")
    try:
        response = client.chat.completions.create(
            model=args.model_name,
            messages=messages,
            temperature=1.0,
            top_p=0.95,
            max_tokens=256,
        )
        answer = response.choices[0].message.content
        print(f"\n[Test] SUCCESS — Model response:\n{answer}")
        print("\n[Test] Gemma 4 API is working correctly.")
    except Exception as exc:
        print(f"\n[Test] FAILED — {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
