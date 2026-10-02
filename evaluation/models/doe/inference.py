"""
Doe-1 inference script.

Loads Doe-1 (FlexAR / Chameleon-based multimodal model) and runs inference
on the benchmark's causal reasoning questions.

Doe-1 uses FlexARInferenceSolver from model.lumina_mgpt.inference_solver.
The model processes images and text jointly through a Chameleon-style
(discrete token) architecture.

IMAGE INPUT:
  Images are passed as a list of PIL Images to inferencer.generate().
  The FlexARItemProcessor handles tokenization and image discretization.
  We pass up to 6 images (front camera × 4 timesteps + side cameras at t=0).

WEIGHTS:
  Doe-1 weights are hosted on Tsinghua Cloud (not HuggingFace).
  Download and set WEIGHTS_PATH=/workspace/weights in dev.env.

Usage (inside container):
    python inference.py
    python inference.py --mode subset --subset-size 50
    python inference.py --mode single --scene nuscenes-scene-0001
    python inference.py --bench-dir /causal_drive_bench/causal_nuscenes --output-base /cdb_outputs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_inference import ModelInference
from evaluation.common.dataset import BenchmarkDataset
from evaluation.common.utils.config_reader import CDBConfig
from evaluation.common.utils.exceptions import MissingFrameException, ModelNotReadyException
from dataloader import DoeDataLoader, FRAME_ORDER, _TEMPORAL_CAM, _SPATIAL_CAMS  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "doe"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    "precision": "bf16",           # FlexARInferenceSolver precision
    "max_gen_len": 512,
    "temperature": 0.7,
    "do_sample": True,
    "target_size": 512,            # FlexARItemProcessor target_size
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert autonomous driving assistant. "
    "You are shown camera images from a self-driving vehicle and asked causal "
    "reasoning questions about the driving scene. "
    "For Binary (Yes/No) questions respond with 'Answer: Yes' or 'Answer: No' "
    "followed by a brief reasoning explanation. "
    "For Multiple Choice (A/B/C/D) questions respond with 'Answer: <letter>' "
    "followed by reasoning."
)


def _load_system_prompt() -> str:
    if _SYSTEM_PROMPT_PATH.exists():
        prompt = _SYSTEM_PROMPT_PATH.read_text().strip()
        print(f"[Inference] Loaded system prompt from {_SYSTEM_PROMPT_PATH}")
        return prompt
    print("[Inference] system_prompt.txt not found, using default system prompt")
    return _DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class DoeInference(ModelInference):
    """
    Inference class for Doe-1 (FlexAR / Chameleon) on the causal benchmark.

    Uses FlexARInferenceSolver from model.lumina_mgpt.inference_solver.
    Images and text are jointly tokenized by FlexARItemProcessor.
    """

    def __init__(self, model_cfg: dict, device: str = "cuda") -> None:
        super().__init__(model_cfg, device)
        self.system_prompt = _load_system_prompt()
        self._device_type = "cuda" if "cuda" in device else "cpu"
        print(f"[DoeInference] Initialized | device={device} | precision={CONFIG['precision']}")

    def load_model(self, weights_path: str) -> None:
        """Load Doe-1 via FlexARInferenceSolver from the model package."""
        print(f"[load_model] Loading Doe-1 from {weights_path} …")

        # Doe model package is installed under /workspace/model
        _doe_src = Path("/workspace/model")
        if _doe_src.exists() and str(_doe_src) not in sys.path:
            sys.path.insert(0, str(_doe_src))
            print(f"[load_model] Added {_doe_src} to sys.path")

        try:
            from model.lumina_mgpt.inference_solver import FlexARInferenceSolver

            self.model = FlexARInferenceSolver(
                model_path=weights_path,
                precision=CONFIG["precision"],
                target_size=CONFIG["target_size"],
                device=self.device,
            )
            print(f"[load_model] Doe-1 FlexARInferenceSolver loaded")
        except Exception:
            print(f"[load_model] ERROR loading model:\n{traceback.format_exc()}")
            raise

    def run_single(self, prompt_payload: dict) -> dict:
        """
        Run inference for one QA prompt and return a dict with the raw response.

        Pipeline:
          1. Reload PIL images from image_paths list (list of dicts)
          2. Build FlexAR conversation with system context + images + text
          3. Call inferencer.generate() and extract text response
        """
        if self.model is None:
            raise ModelNotReadyException("Call load_model() before run_single()")

        from PIL import Image as PILImage

        sample_id = prompt_payload.get("sample_id", "?")
        question_id = prompt_payload.get("question_id", "?")
        print(f"\n[run_single] {sample_id} | q={question_id}")

        image_paths: list = prompt_payload.get("image_paths", [])
        prompt_context: str = prompt_payload.get("prompt_context", "")
        question_text: str = prompt_payload.get("question_text", "")

        # Build lookup: (time_key, camera_key) → path
        lookup = {
            (item["time_key"], item["camera_key"]): item["path"]
            for item in image_paths
        }

        # ------------------------------------------------------------------
        # Step 1: Reload PIL images
        # ------------------------------------------------------------------
        pil_images: List[PILImage.Image] = []
        missing = 0

        # Temporal front camera
        for frame_key in FRAME_ORDER:
            path = lookup.get((frame_key, _TEMPORAL_CAM))
            if path and Path(path).exists():
                try:
                    pil_images.append(PILImage.open(path).convert("RGB"))
                except Exception:
                    print(f"[run_single] WARN: failed to load {path}")
                    pil_images.append(PILImage.new("RGB", (1600, 900), color=(128, 128, 128)))
                    missing += 1
            else:
                if path:
                    try:
                        raise MissingFrameException(f"Image file not found: {path}")
                    except MissingFrameException as exc:
                        print(f"[run_single] WARN: {exc}")
                pil_images.append(PILImage.new("RGB", (1600, 900), color=(128, 128, 128)))
                missing += 1

        # Spatial side cameras at t=0
        for cam in _SPATIAL_CAMS:
            path = lookup.get(("Tp0p0", cam))
            if path and Path(path).exists():
                try:
                    pil_images.append(PILImage.open(path).convert("RGB"))
                except Exception:
                    pil_images.append(PILImage.new("RGB", (1600, 900), color=(128, 128, 128)))
                    missing += 1
            else:
                if path:
                    try:
                        raise MissingFrameException(f"Image file not found: {path}")
                    except MissingFrameException as exc:
                        print(f"[run_single] WARN: {exc}")
                pil_images.append(PILImage.new("RGB", (1600, 900), color=(128, 128, 128)))
                missing += 1

        n_total = len(pil_images)
        print(f"[run_single] {n_total - missing}/{n_total} images ({missing} placeholder(s))")

        # ------------------------------------------------------------------
        # Step 2: Build conversation (system context prepended to question)
        # ------------------------------------------------------------------
        full_text = f"{prompt_context}\n\n{question_text}".strip() if prompt_context else question_text
        full_question = f"{self.system_prompt}\n\n{full_text}".strip()
        qas = [[full_question, None]]

        # ------------------------------------------------------------------
        # Step 3: Generate
        # ------------------------------------------------------------------
        print(
            f"[run_single] Calling inferencer.generate() | n_images={n_total} "
            f"max_gen_len={CONFIG['max_gen_len']} temperature={CONFIG['temperature']}"
        )
        try:
            result = self.model.generate(
                qas=qas,
                images=pil_images,
                max_gen_len=CONFIG["max_gen_len"],
                temperature=CONFIG["temperature"],
                do_sample=CONFIG["do_sample"],
            )
        except Exception:
            print(f"[run_single] ERROR in generate():\n{traceback.format_exc()}")
            raise

        # result is (generated_text, generated_images, past_key_values)
        # generated_text from decode_ids() is itself (text_str, images_list)
        if isinstance(result, tuple) and len(result) >= 1:
            generated = result[0]
        else:
            generated = result

        # decode_ids returns (str, list) — extract the text string
        if isinstance(generated, tuple) and len(generated) >= 1:
            response = str(generated[0])
        else:
            response = str(generated)

        print(f"[run_single] Response ({len(response)} chars): {response[:120]!r}…")
        return {"text": response}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    logger.info("Logging to %s", log_file)


# ---------------------------------------------------------------------------
# Pending work check
# ---------------------------------------------------------------------------

def _has_pending_work(sample_ids: list, run_dir: Path, causal_dataset_name: str) -> bool:
    for sid in sample_ids:
        outputs_file = (
            run_dir / causal_dataset_name / sid["scene_id"] / sid["sample_id"] / "outputs.jsonl"
        )
        if not outputs_file.exists() or outputs_file.stat().st_size == 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run {MODEL_ID} inference on the CausalDriveBench dataset"
    )
    parser.add_argument(
        "--bench-dir",
        default="",
        help="Dataset-specific bench directory. Defaults to CDBConfig.get_bench_dir(--dataset).",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("DATASET", "nuscenes"),
        choices=["nuscenes", "argoverse2", "openscene"],
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
    )
    parser.add_argument(
        "--scene",
        default=os.environ.get("SCENE") or None,
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Existing run directory to resume.",
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help="Override for dir_cdb_outputs from eval_config.yaml.",
    )
    parser.add_argument(
        "--weights-path",
        default="",
        help="Path to Doe-1 weights directory.",
    )
    parser.add_argument(
        "--model-checkpoint",
        default=None,
        help="Path to Doe-1 model checkpoint (overrides --weights-path).",
    )
    parser.add_argument(
        "--vqvae-path",
        default=None,
        help="Path to VQ-VAE checkpoint for image tokenization.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to eval_config.yaml (auto-detected if omitted)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = CDBConfig(config_path=args.config)
    print(cfg.summary())

    bench_dir = Path(args.bench_dir) if args.bench_dir else cfg.get_bench_dir(args.dataset)
    causal_dataset_name = bench_dir.name

    weights_path = args.model_checkpoint or args.weights_path or str(cfg.dir_model_weights)
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
        logger.info("Resuming existing run: %s", run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path(args.output_base) if args.output_base else cfg.dir_cdb_outputs
        run_dir = output_base / f"{MODEL_ID}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir / "inference.log")
    logger.info("Run directory: %s", run_dir)
    logger.info("Bench directory: %s", bench_dir)
    logger.info("Mode: %s | batch_size: %d | device: %s", args.mode, batch_size, args.device)

    dataset = BenchmarkDataset(bench_dir=bench_dir)
    model_cfg: dict = {"qa_types": cfg.qa_types, "batch_size": batch_size}
    loader = DoeDataLoader(dataset=dataset, model_cfg=model_cfg)

    sample_ids = dataset.get_sample_ids(
        mode=args.mode,
        subset_size=args.subset_size,
        scene=args.scene,
    )
    logger.info("%d sample(s) selected (mode=%s)", len(sample_ids), args.mode)

    inference = DoeInference(model_cfg=model_cfg, device=args.device)

    if not _has_pending_work(sample_ids, run_dir, causal_dataset_name):
        logger.info("All samples already processed — nothing to do.")
        return

    try:
        inference.load_model(weights_path)
    except Exception:
        logger.error("FATAL: model load failed:\n%s", traceback.format_exc())
        raise

    total_prompts = 0
    total_outputs = 0

    for sid in sample_ids:
        scene_id = sid["scene_id"]
        sample_id = sid["sample_id"]

        sample_out_dir = run_dir / causal_dataset_name / scene_id / sample_id
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        prompts_file = sample_out_dir / "prompts.jsonl"
        outputs_file = sample_out_dir / "outputs.jsonl"

        logger.info("[main] ── %s/%s ──", scene_id, sample_id)

        try:
            n_prompts = loader.save_prompts([sid], prompts_file)
        except Exception:
            logger.error(
                "ERROR saving prompts for %s/%s:\n%s",
                scene_id, sample_id, traceback.format_exc(),
            )
            continue

        if n_prompts == 0:
            logger.warning("No prompts generated for %s/%s — skipping", scene_id, sample_id)
            continue

        total_prompts += n_prompts
        logger.info("[main] Running inference for %s/%s (%d prompts)", scene_id, sample_id, n_prompts)

        try:
            n_written = inference.run_from_jsonl(
                input_path=prompts_file,
                output_path=outputs_file,
                batch_size=batch_size,
                resume=True,
            )
            logger.info("[main] Completed %s/%s: %d new output(s)", scene_id, sample_id, n_written)
            total_outputs += n_written
        except Exception:
            logger.error(
                "ERROR during inference for %s/%s:\n%s",
                scene_id, sample_id, traceback.format_exc(),
            )
            continue

    logger.info("Inference complete.")
    logger.info("  Total prompts : %d", total_prompts)
    logger.info("  Total outputs : %d", total_outputs)
    logger.info("  Timing stats  : %s", inference.timing_stats)
    logger.info("  Outputs under : %s", run_dir)
    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    main()
