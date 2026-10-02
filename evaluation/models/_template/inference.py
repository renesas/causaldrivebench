"""
Template inference script — copy to your model directory and fill in the TODO sections.

Run modes
---------
    # All samples in a bench directory
    python inference.py --bench-dir /causal_drive_bench/causal_nuscenes

    # Random subset of N scenes
    python inference.py --bench-dir ... --mode subset --subset-size 20

    # Single scene
    python inference.py --bench-dir ... --mode single --scene nuscenes-scene-0159

    # Resume an existing run (skips already-processed questions)
    python inference.py --bench-dir ... --run-dir /cdb_outputs/template_20260410_120000

Environment variables (set in dev.env or docker-compose)
---------------------------------------------------------
    EVAL_RUNTIME      — "host" (default) or "docker"; selects path roots
    CAUSAL_BENCH_DIR  — dataset-specific bench directory
    OUTPUT_DIR        — override for dir_cdb_outputs from config (optional)
    WEIGHTS_PATH      — path to model weights
    DEVICE            — "cuda" or "cpu"
    BATCH_SIZE        — inference batch size
    MODE              — full | subset | single
    SUBSET_SIZE       — number of scenes for mode=subset
    SCENE             — scene name for mode=single

Output directory layout (created automatically)
-----------------------------------------------
    <dir_cdb_outputs>/<MODEL_ID>_<timestamp>/   ← run_dir
    ├── inference.log
    └── causal_nuscenes/                         ← mirrors bench_dir name
        └── nuscenes-scene-0159/
            └── SAMPLED_0/
                ├── prompts.jsonl
                └── outputs.jsonl
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow running from repo root or within container
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_inference import ModelInference
from evaluation.common.dataset import BenchmarkDataset
from evaluation.common.utils.config_reader import CDBConfig

# Local import: dataloader lives in the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader import make_dataloader, dataset_name_from_bench_dir  # noqa: E402

# ---------------------------------------------------------------------------
# Model identity — update when copying to a new model directory
# ---------------------------------------------------------------------------

MODEL_ID = "_template"

# ---------------------------------------------------------------------------
# Inference implementation
# ---------------------------------------------------------------------------


class TemplateInference(ModelInference):
    """
    Concrete inference class for <YOUR MODEL NAME>.

    TODO: Replace load_model and run_single with real implementations.
    """

    def __init__(self, model_cfg: dict, device: str = "cuda") -> None:
        super().__init__(model_cfg, device)
        self.tokenizer = None
        self.processor = None

    def load_model(self, weights_path: str) -> None:
        """
        Load the model from *weights_path*.

        TODO: Replace with your model's actual loading code.

        Example (HuggingFace):
        ::

            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(weights_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                weights_path,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
        """
        logger.info("Loading model from %s on device %s", weights_path, self.device)
        logger.warning(
            "Stub load_model() — replace with real model-loading code."
        )
        self.model = None  # TODO: remove this placeholder

    def run_single(self, prompt_payload: dict) -> dict:
        """
        Run inference for one prompt payload and return the raw output dict.

        The returned dict must contain at least a ``"text"`` key with the
        model's response string.  Add other model-specific keys as needed.

        TODO: Replace with your model's generate / forward pass.

        Example:
        ::

            conversation = prompt_payload.get("conversation", [])
            image_paths  = prompt_payload.get("image_paths", [])
            images = [
                Image.open(cfg.resolve_frame_path(p["path"])).convert("RGB")
                for p in image_paths
                if Path(cfg.resolve_frame_path(p["path"])).exists()
            ]
            inputs = self.processor(
                text=conversation, images=images, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out_ids = self.model.generate(**inputs, max_new_tokens=256)
            text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            return {"text": text}
        """
        if self.model is None:
            # Stub: return a plausible placeholder for pipeline testing
            return {"text": "Answer: A\nReasoning: Stub response — model not loaded."}

        # TODO: implement real forward pass
        return {"text": "Answer: A\nReasoning: Replace this placeholder."}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run {MODEL_ID} inference on the CausalDriveBench dataset"
    )
    parser.add_argument(
        "--bench-dir",
        default=os.environ.get("CAUSAL_BENCH_DIR", ""),
        help=(
            "Dataset-specific bench directory (e.g. /causal_drive_bench/causal_nuscenes). "
            "Falls back to CDBConfig.get_bench_dir('nuscenes') if empty."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("DATASET", "nuscenes"),
        choices=["nuscenes", "argoverse2", "openscene"],
        help="Dataset name — used when --bench-dir is not specified",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "subset", "single"],
        default=os.environ.get("MODE", "full"),
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=int(os.environ.get("SUBSET_SIZE", 0)) or None,
        help="Number of scenes for mode=subset",
    )
    parser.add_argument(
        "--scene",
        default=os.environ.get("SCENE") or None,
        help="Scene name for mode=single",
    )
    parser.add_argument(
        "--run-dir",
        default=os.environ.get("RUN_DIR") or None,
        help=(
            "Existing run directory to resume (e.g. /cdb_outputs/template_20260410_120000). "
            "If omitted, a new timestamped directory is created."
        ),
    )
    parser.add_argument(
        "--output-base",
        default=os.environ.get("OUTPUT_DIR") or None,
        help="Override for dir_cdb_outputs from eval_config.yaml",
    )
    parser.add_argument(
        "--weights-path",
        default=os.environ.get("WEIGHTS_PATH", ""),
        help="Path to model weights (overrides CDBConfig.dir_model_weights)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", 4)),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to eval_config.yaml (auto-detected if omitted)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(log_file: Path) -> None:
    """Configure root logger to write to both stderr and *log_file*."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    logger.info("Logging to %s", log_file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # --- Config and path resolution ---------------------------------------
    cfg = CDBConfig(config_path=args.config)
    print(cfg.summary())

    bench_dir = Path(args.bench_dir) if args.bench_dir else cfg.get_bench_dir(args.dataset)
    causal_dataset_name = bench_dir.name                               # e.g. "causal_nuscenes"
    dataset_name = dataset_name_from_bench_dir(causal_dataset_name)    # e.g. "nuscenes"

    weights_path = args.weights_path or str(cfg.dir_model_weights)

    # --- Run directory setup ---------------------------------------------
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(
                f"--run-dir does not exist: {run_dir}"
            )
        logger.info("Resuming existing run: %s", run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path(args.output_base) if args.output_base else cfg.dir_cdb_outputs
        run_dir = output_base / f"{MODEL_ID}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    # --- Logging (write to run_dir/inference.log) -------------------------
    setup_logging(run_dir / "inference.log")
    logger.info("Run directory: %s", run_dir)
    logger.info("Bench directory: %s", bench_dir)
    logger.info("Mode: %s | batch_size: %d | device: %s", args.mode, args.batch_size, args.device)

    # --- Dataset and DataLoader ------------------------------------------
    dataset = BenchmarkDataset(bench_dir=bench_dir)
    model_cfg: dict = {
        "qa_types": cfg.qa_types,
        "batch_size": args.batch_size,
    }
    # Factory: select per-dataset camera configuration automatically
    loader = make_dataloader(dataset_name, dataset, model_cfg)
    logger.info("Using %s for dataset '%s'", type(loader).__name__, dataset_name)

    sample_ids = dataset.get_sample_ids(
        mode=args.mode,
        subset_size=args.subset_size,
        scene=args.scene,
    )
    logger.info("%d sample(s) selected (mode=%s)", len(sample_ids), args.mode)

    # --- Load model (only if there is work to do) ------------------------
    inference = TemplateInference(model_cfg=model_cfg, device=args.device)

    # Quick check: count how many samples need processing to avoid loading
    # the model when everything is already done (resume case).
    has_pending = _has_pending_work(sample_ids, run_dir, causal_dataset_name)
    if not has_pending:
        logger.info("All samples already processed — nothing to do.")
        return

    inference.load_model(weights_path)

    # --- Per-sample inference loop ----------------------------------------
    for sid in sample_ids:
        scene_id = sid["scene_id"]
        sample_id = sid["sample_id"]

        sample_out_dir = run_dir / causal_dataset_name / scene_id / sample_id
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        prompts_file = sample_out_dir / "prompts.jsonl"
        outputs_file = sample_out_dir / "outputs.jsonl"

        # Always regenerate prompts.jsonl (deterministic + cheap relative to inference)
        n_prompts = loader.save_prompts([sid], prompts_file)
        if n_prompts == 0:
            logger.warning(
                "No prompts generated for %s/%s — skipping", scene_id, sample_id
            )
            continue

        logger.info(
            "Running inference for %s/%s (%d prompts)", scene_id, sample_id, n_prompts
        )
        n_written = inference.run_from_jsonl(
            input_path=prompts_file,
            output_path=outputs_file,
            batch_size=args.batch_size,
            resume=True,
        )
        logger.info(
            "Completed %s/%s: %d new output(s)", scene_id, sample_id, n_written
        )

    # --- Summary ----------------------------------------------------------
    logger.info("Inference complete. Timing: %s", inference.timing_stats)
    logger.info("Outputs in: %s", run_dir)
    print(f"\nRun directory: {run_dir}")


def _has_pending_work(
    sample_ids: list[dict],
    run_dir: Path,
    causal_dataset_name: str,
) -> bool:
    """
    Return True if any sample has unprocessed questions.

    A sample is considered complete if its outputs.jsonl exists and is non-empty.
    This is a lightweight check to avoid loading the model unnecessarily.
    """
    for sid in sample_ids:
        outputs_file = (
            run_dir / causal_dataset_name / sid["scene_id"] / sid["sample_id"] / "outputs.jsonl"
        )
        if not outputs_file.exists() or outputs_file.stat().st_size == 0:
            return True
    return False


if __name__ == "__main__":
    main()
