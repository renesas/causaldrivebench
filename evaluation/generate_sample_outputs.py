"""
Generate sample_outputs and sample_reports for developer documentation.

This script simulates the full alpamayo_1_0 inference → postprocess pipeline
on the sample_data directory WITHOUT loading the GPU model.  Instead it creates
synthetic outputs.jsonl entries with realistic chain-of-causation (CoC) text
and mostly correct answers, so that postprocess.py produces meaningful reports.

Run from the repo root:
    python evaluation/generate_sample_outputs.py

Outputs
-------
evaluation/sample_outputs/<run_name>/
    causal_nuscenes/
        nuscenes-scene-0159/
            SAMPLED_0/
                prompts.jsonl   ← real prompts from AlpamayoDataLoader
                outputs.jsonl   ← synthetic model responses
                report.json     ← Tier 1 (sample-level)
            SAMPLED_1/
                ...
        report.json             ← Tier 2 (dataset-level)
    report.json                 ← Tier 3 (run-level)

evaluation/sample_reports/
    index.html                  ← Tier 4 (all runs root)
    <run_name>/
        index.html              ← Tier 3 HTML
        causal_nuscenes/
            index.html          ← Tier 2 HTML
            nuscenes-scene-0159/
                index.html      ← Scene HTML
                SAMPLED_0/
                    index.html  ← Tier 1 HTML
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EVAL_DIR    = REPO_ROOT / "evaluation"
SAMPLE_DATA = EVAL_DIR / "sample_data"
OUTPUT_DIR  = EVAL_DIR / "sample_outputs"
REPORTS_DIR = EVAL_DIR / "sample_reports"

MODEL_ID      = "alpamayo_1_0"
DATASET_KEY   = "nuscenes"
BENCH_NAME    = "causal_nuscenes"
SCHEMA_VERSION = "1.0"

ENABLED_METRICS = [
    "overall",
    "per_qa_type",
    "per_question_type",
    "per_graph_structure",
    "per_rung",
    "answer_stats",
    "binary_metrics",
    "reasoning_score",
]

# ---------------------------------------------------------------------------
# Synthetic CoC templates
# ---------------------------------------------------------------------------

_COC_MCQ = """<think>
Let me analyse the causal structure in the scene.

Looking at the spatial configuration, I can identify the relevant actors and
their relationships. The ego vehicle trajectory and surrounding context provide
strong cues for causal attribution.

Based on my analysis:
- The primary mechanism involves direct interaction between the agents
- The counterfactual reasoning supports option {answer}
- Spatial grounding confirms this interpretation

Answer: {answer}
</think>"""

_COC_BINARY = """<think>
Examining the scene carefully.

The causal chain involves: ego vehicle → environmental trigger → response.
The graph structure and isolation conditions are satisfied.
My reasoning about the counterfactual scenario leads to a clear conclusion.

Answer: {answer}
</think>"""


def _make_coc(answer: str, fmt: str) -> str:
    """Generate a synthetic CoC response for the given answer and format."""
    template = _COC_MCQ if fmt == "mcq" else _COC_BINARY
    return template.format(answer=answer).strip()


def _pick_answer(correct: str, fmt: str, error_rate: float = 0.25) -> str:
    """Return the correct answer with (1-error_rate) probability, else a wrong one."""
    if random.random() > error_rate:
        return correct
    if fmt == "binary":
        return "No" if correct.capitalize() == "Yes" else "Yes"
    options = ["A", "B", "C", "D"]
    wrong = [o for o in options if o != correct.upper()]
    return random.choice(wrong)


# ---------------------------------------------------------------------------
# Prompt generation using AlpamayoDataLoader
# ---------------------------------------------------------------------------

def generate_prompts(bench_dir: Path, sample_id_dict: dict, out_dir: Path) -> list:
    """
    Use AlpamayoDataLoader to generate and save prompts.jsonl.
    Returns the list of prompt dicts.
    """
    sys.path.insert(0, str(EVAL_DIR / "models" / "alpamayo_1_0"))
    from dataloader import AlpamayoDataLoader
    from evaluation.common.dataset import BenchmarkDataset

    dataset = BenchmarkDataset(bench_dir=bench_dir)
    loader  = AlpamayoDataLoader(
        dataset=dataset,
        model_cfg={"qa_types": ["ladder", "dormant", "distractor"]},
        cfg=None,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_file = out_dir / "prompts.jsonl"
    loader.save_prompts([sample_id_dict], prompts_file)

    prompts = []
    with open(prompts_file) as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


# ---------------------------------------------------------------------------
# Synthetic output generation
# ---------------------------------------------------------------------------

def generate_outputs(prompts: list, bench_dir: Path, sample_id: str, out_dir: Path) -> None:
    """
    Build synthetic outputs.jsonl that simulate realistic CoC model responses.

    Uses the QA index to look up correct answers, then applies an error_rate
    so ~75% of answers are correct.
    """
    # Build per-sample QA index
    from evaluation.common.base_postprocess import ModelPostprocessor
    scene_id = prompts[0]["scene_id"] if prompts else ""
    sample_bench_dir = bench_dir / scene_id / sample_id
    qa_index = ModelPostprocessor._load_sample_qa_index(sample_bench_dir)

    outputs_file = out_dir / "outputs.jsonl"
    with open(outputs_file, "w") as fh:
        for i, prompt in enumerate(prompts):
            qid = prompt.get("question_id", "")
            question = qa_index.get(qid, {})
            correct  = question.get("correct_answer", "A")
            fmt      = question.get("answer_format", "binary")

            answer   = _pick_answer(correct, fmt, error_rate=0.25)
            coc_text = _make_coc(answer, fmt)

            out = {
                "scene_id":       prompt.get("scene_id", ""),
                "sample_id":      prompt.get("sample_id", ""),
                "question_id":    qid,
                "prompt_id":      f"{i+1:06d}",
                "raw_output":     {"text": coc_text},
                "inference_time_s": round(1.2 + random.uniform(0, 2.0), 3),
                "timestamp":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            fh.write(json.dumps(out) + "\n")
    print(f"  [generate_outputs] Wrote {len(prompts)} outputs → {outputs_file}")


# ---------------------------------------------------------------------------
# Run postprocess tier reports
# ---------------------------------------------------------------------------

def run_postprocess(run_dir: Path, bench_dir: Path) -> None:
    """
    Invoke postprocess.py logic directly (without subprocess) to generate
    Tier 1/2/3 report.json files.
    """
    from evaluation.models.alpamayo_1_0.postprocess import AlpamayoPostprocessor
    from evaluation.common.base_postprocess import ModelPostprocessor
    from evaluation.common.report_generator import (
        generate_sample_report,
        generate_dataset_report,
        generate_run_report,
        is_sample_report_fresh,
    )
    from evaluation.common.dataset import BenchmarkDataset

    dataset_dir = run_dir / BENCH_NAME
    postprocessor = AlpamayoPostprocessor(model_cfg={})
    dataset_obj   = BenchmarkDataset(bench_dir=bench_dir)

    for sid in dataset_obj.get_sample_ids(mode="full"):
        scene_id  = sid["scene_id"]
        sample_id = sid["sample_id"]
        outputs_file      = dataset_dir / scene_id / sample_id / "outputs.jsonl"
        sample_report_path = dataset_dir / scene_id / sample_id / "report.json"

        if not outputs_file.exists():
            print(f"  [postprocess] Skipping {scene_id}/{sample_id}: no outputs.jsonl")
            continue

        sample_bench_dir = bench_dir / scene_id / sample_id
        qa_index   = ModelPostprocessor._load_sample_qa_index(sample_bench_dir)
        qa_results = postprocessor.parse_sample_outputs(outputs_file, qa_index)

        generate_sample_report(
            run_name=run_dir.name, dataset=BENCH_NAME,
            scene_id=scene_id, sample_id=sample_id,
            qa_results=qa_results, report_path=sample_report_path,
            schema_version=SCHEMA_VERSION, enabled_metrics=ENABLED_METRICS,
        )
        print(
            f"  [postprocess] {scene_id}/{sample_id}: "
            f"{sum(r['correct'] for r in qa_results)}/{len(qa_results)} correct"
        )

    # Tier 2 — dataset report
    generate_dataset_report(
        run_name=run_dir.name, dataset=BENCH_NAME,
        dataset_dir=dataset_dir,
        report_path=dataset_dir / "report.json",
        schema_version=SCHEMA_VERSION, enabled_metrics=ENABLED_METRICS,
    )

    # Tier 3 — run report
    generate_run_report(
        run_name=run_dir.name, run_dir=run_dir,
        report_path=run_dir / "report.json",
        schema_version=SCHEMA_VERSION, enabled_metrics=ENABLED_METRICS,
    )


# ---------------------------------------------------------------------------
# HTML reports
# ---------------------------------------------------------------------------

def run_html_reports(run_dir: Path) -> None:
    """Generate HTML reports in sample_reports/."""
    from evaluation.common.report_generator import generate_root_aggregate
    from evaluation.common.html_reporter import generate_all_html, generate_root_index

    root_aggregate = generate_root_aggregate(OUTPUT_DIR, SCHEMA_VERSION)
    generate_all_html(run_dir.name, OUTPUT_DIR, REPORTS_DIR, root_aggregate)
    print(f"  [html] Reports → {REPORTS_DIR / 'index.html'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(42)

    bench_dir = SAMPLE_DATA / BENCH_NAME
    if not bench_dir.exists():
        raise FileNotFoundError(f"sample_data bench dir not found: {bench_dir}")

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # static timestamp for consistency
    timestamp = "20260421_064223"
    run_name  = f"{MODEL_ID}_sample_{timestamp}"
    run_dir   = OUTPUT_DIR / run_name

    print("=" * 60)
    print("  CausalDriveBench — Sample Output Generation")
    print(f"  bench_dir  : {bench_dir}")
    print(f"  run_dir    : {run_dir}")
    print("=" * 60)

    # Discover samples from sample_data manifest / directory walk
    from evaluation.common.dataset import BenchmarkDataset
    dataset = BenchmarkDataset(bench_dir=bench_dir)
    sample_ids = dataset.get_sample_ids(mode="full")
    print(f"\nFound {len(sample_ids)} sample(s) in sample_data\n")

    # --- Step 1 & 2: Generate prompts + synthetic outputs per sample --------
    for sid in sample_ids:
        scene_id  = sid["scene_id"]
        sample_id = sid["sample_id"]
        print(f"[{scene_id} / {sample_id}]")

        out_dir = run_dir / BENCH_NAME / scene_id / sample_id
        try:
            prompts = generate_prompts(bench_dir, sid, out_dir)
            print(f"  [dataloader] Generated {len(prompts)} prompts")
        except Exception as exc:
            print(f"  [dataloader] ERROR: {exc}")
            # Fall back: create minimal prompts from QA index
            prompts = _fallback_prompts(bench_dir, sid, out_dir)

        generate_outputs(prompts, bench_dir, sample_id, out_dir)

    # --- Step 3: Postprocess (Tier 1/2/3 reports) --------------------------
    print("\n--- Postprocessing ---")
    run_postprocess(run_dir, bench_dir)

    # --- Step 4: HTML reports ----------------------------------------------
    print("\n--- HTML report generation ---")
    run_html_reports(run_dir)

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Done!")
    print(f"  sample_outputs : {run_dir}")
    print(f"  sample_reports : {REPORTS_DIR / 'index.html'}")
    print("=" * 60)


def _fallback_prompts(bench_dir: Path, sid: dict, out_dir: Path) -> list:
    """Minimal prompt stubs when the dataloader fails (e.g. missing frames)."""
    from evaluation.common.base_postprocess import ModelPostprocessor
    scene_id  = sid["scene_id"]
    sample_id = sid["sample_id"]
    qa_index  = ModelPostprocessor._load_sample_qa_index(bench_dir / scene_id / sample_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = [
        {
            "scene_id":   scene_id,
            "sample_id":  sample_id,
            "question_id": qid,
            "qa_text":    q.get("question", ""),
            "image_paths": [],
            "ego_history_xyz": [[0.0, 0.0, 0.0]] * 4,
            "ego_history_rot": [[[1,0,0],[0,1,0],[0,0,1]]] * 4,
        }
        for qid, q in qa_index.items()
    ]
    prompts_file = out_dir / "prompts.jsonl"
    with open(prompts_file, "w") as fh:
        for i, p in enumerate(prompts):
            p["prompt_id"] = f"{i+1:06d}"
            fh.write(json.dumps(p) + "\n")
    return prompts


if __name__ == "__main__":
    main()
