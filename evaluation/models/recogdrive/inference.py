"""
ReCogDrive inference script.

Loads ReCogDrive (InternVL3 backbone) and runs inference on the benchmark's
causal reasoning questions.

ReCogDrive (owl10/ReCogDrive-8B-RL) uses InternVL3 as its VLM with a diffusion
planner head for NAVSIM planning. For VQA evaluation we use the InternVL3
backbone directly via AutoModel + model.chat().

IMAGE PREPROCESSING:
  InternVL3 processes images at 448×448 via IMAGENET-normalised tensors.
  Each image: T.Resize(448) → T.ToTensor() → T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

GPU REQUIREMENT:
  Requires SM >= 8 (Ampere or newer) for flash-attn.
  Use 8B variant for best QA performance; 2B for memory-constrained setups.

MULTI-IMAGE:
  InternVL3 uses '<image>\\n' tokens in the question (one per patch).
  pixel_values: (N, 3, 448, 448), num_patches_list: [1] * N

Usage (inside container):
    uv run python /workspace/repo/evaluation/models/recogdrive/inference.py \\
        --bench-dir /causal_drive_bench/causal_nuscenes \\
        --output-base /cdb_outputs \\
        --weights-path /workspace/weights
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
from dataloader import ReCogDriveDataLoader, FRAME_ORDER, CAMERA_ORDER  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------

MODEL_ID = "recogdrive"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    "model_dtype": torch.bfloat16,
    "max_new_tokens": 512,
    "do_sample": False,
    "image_size": 448,
    "imagenet_mean": (0.485, 0.456, 0.406),
    "imagenet_std": (0.229, 0.224, 0.225),
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert autonomous driving assistant. "
    "You are shown camera images from a self-driving vehicle across multiple "
    "timesteps and camera views (past to present) and asked causal reasoning "
    "questions about the driving scene. "
    "For Binary (Yes/No) questions respond with 'Answer: Yes' or 'Answer: No' "
    "followed by a brief reasoning explanation. "
    "For Multiple Choice (A/B/C/D) questions respond with 'Answer: <letter>' "
    "followed by reasoning. "
    "Ground your reasoning in what you observe in the camera images."
)


def _load_system_prompt() -> str:
    if _SYSTEM_PROMPT_PATH.exists():
        prompt = _SYSTEM_PROMPT_PATH.read_text().strip()
        print(f"[Inference] Loaded system prompt from {_SYSTEM_PROMPT_PATH}")
        return prompt
    print("[Inference] system_prompt.txt not found, using default system prompt")
    return _DEFAULT_SYSTEM_PROMPT


def _build_internvl_transform():
    """Build InternVL3's standard image transform pipeline."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Resize(
            (CONFIG["image_size"], CONFIG["image_size"]),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.ToTensor(),
        T.Normalize(mean=CONFIG["imagenet_mean"], std=CONFIG["imagenet_std"]),
    ])


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class ReCogDriveInference(ModelInference):
    """
    Inference class for ReCogDrive (InternVL3) on the causal benchmark.

    Provides 2 frames × 4 cameras = 8 images in InternVL3's expected format.
    Uses model.chat() for single-turn multi-image QA.
    """

    def __init__(self, model_cfg: dict, device: str = "cuda") -> None:
        super().__init__(model_cfg, device)
        self.tokenizer = None
        self.system_prompt = _load_system_prompt()
        self.transform = _build_internvl_transform()
        self._device_type = "cuda" if "cuda" in device else "cpu"
        print(f"[ReCogDriveInference] Initialized | device={device} | dtype={CONFIG['model_dtype']}")

    def load_model(self, weights_path: str) -> None:
        """Load ReCogDrive (InternVL3) from a local HuggingFace weights directory."""
        print(f"[load_model] Loading ReCogDrive from {weights_path} …")

        # ReCogDrive internvl_chat package may need to be on sys.path
        internvl_src = Path("/workspace/model/internvl_chat")
        if internvl_src.exists() and str(internvl_src) not in sys.path:
            sys.path.insert(0, str(internvl_src))
            print(f"[load_model] Added {internvl_src} to sys.path")

        try:
            from transformers import AutoTokenizer, AutoModel

            self.tokenizer = AutoTokenizer.from_pretrained(
                weights_path, trust_remote_code=True, use_fast=False
            )
            self.model = AutoModel.from_pretrained(
                weights_path,
                torch_dtype=CONFIG["model_dtype"],
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
            ).eval().to(self.device)

            # Override system message with QA-focused prompt
            if hasattr(self.model, "system_message"):
                self.model.system_message = self.system_prompt

            print(f"[load_model] ReCogDrive (InternVL3) loaded on {self.device}")
        except Exception:
            print(f"[load_model] ERROR loading model:\n{traceback.format_exc()}")
            raise

    def run_single(self, prompt_payload: dict) -> dict:
        """
        Run inference for one QA prompt and return the raw response dict.

        Pipeline:
          1. Reload PIL images → apply InternVL3 transform → stack pixel_values
          2. Prepend '<image>\\n' × N_images to full_text
          3. Call model.chat() for generation
        """
        if self.model is None:
            raise ModelNotReadyException("Call load_model() before run_single()")

        from PIL import Image as PILImage

        question_id = prompt_payload.get("question_id", "?")
        print(f"\n[run_single] {prompt_payload.get('scene_id', '?')}/{prompt_payload.get('sample_id', '?')} | q={question_id}")

        # image_paths is a list of {path, time_key, camera_key}
        image_paths: list = prompt_payload.get("image_paths", [])
        question_text: str = prompt_payload.get("question_text", "")
        prompt_context: str = prompt_payload.get("prompt_context", "")
        full_text = f"{prompt_context}\n\n{question_text}".strip() if prompt_context else question_text

        # Build lookup from (time_key, camera_key) -> path
        lookup = {(item["time_key"], item["camera_key"]): item["path"] for item in image_paths}

        # ------------------------------------------------------------------
        # Step 1: Load and transform images in TIME-MAJOR order
        # ------------------------------------------------------------------
        pixel_tensors: List[torch.Tensor] = []
        num_patches_list: List[int] = []
        missing = 0

        for frame_key in FRAME_ORDER:
            for cam in CAMERA_ORDER:
                path = lookup.get((frame_key, cam))
                if path and Path(path).exists():
                    try:
                        img = PILImage.open(path).convert("RGB")
                    except Exception:
                        print(f"[run_single] WARN: failed to load {path}")
                        img = PILImage.new("RGB", (1600, 900), color=(128, 128, 128))
                        missing += 1
                else:
                    if path:
                        try:
                            raise MissingFrameException(f"Image file not found: {path}")
                        except MissingFrameException as exc:
                            print(f"[run_single] WARN: {exc}")
                    img = PILImage.new("RGB", (1600, 900), color=(128, 128, 128))
                    missing += 1
                pixel_tensors.append(self.transform(img))
                num_patches_list.append(1)

        n_total = len(pixel_tensors)
        print(f"[run_single] {n_total - missing}/{n_total} images ({missing} placeholder(s))")

        pixel_values = torch.stack(pixel_tensors, dim=0).to(
            device=self.device, dtype=CONFIG["model_dtype"]
        )

        # ------------------------------------------------------------------
        # Step 2: Build question string with InternVL image tokens
        # ------------------------------------------------------------------
        # Image tokens are already embedded inline in prompt_context with
        # per-view labels (e.g. "[t=0.0s | FRONT]:\n<image>") so we use
        # full_text directly. Fall back to prepending only if tokens are absent.
        if "<image>" in full_text:
            question = full_text
        else:
            question = "<image>\n" * n_total + full_text

        generation_config = dict(
            max_new_tokens=CONFIG["max_new_tokens"],
            do_sample=CONFIG["do_sample"],
        )

        # ------------------------------------------------------------------
        # Step 3: Generate via model.chat()
        # ------------------------------------------------------------------
        print(
            f"[run_single] Calling model.chat() | n_images={n_total} "
            f"max_new_tokens={CONFIG['max_new_tokens']}"
        )
        with torch.no_grad():
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )

        if isinstance(response, tuple):
            response = response[0]
        response = str(response)
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
        help="Path to ReCogDrive weights (local HuggingFace directory).",
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

    weights_path = args.weights_path or str(cfg.dir_model_weights)
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
    model_cfg: dict = {
        "qa_types":    cfg.qa_types,
        "batch_size":  batch_size,
        "bench_dir":   str(bench_dir),
        "raw_data_dir": str(cfg.dir_raw_data) if hasattr(cfg, "dir_raw_data") else "",
    }
    loader = ReCogDriveDataLoader(dataset=dataset, model_cfg=model_cfg)

    sample_ids = dataset.get_sample_ids(
        mode=args.mode,
        subset_size=args.subset_size,
        scene=args.scene,
    )
    logger.info("%d sample(s) selected (mode=%s)", len(sample_ids), args.mode)

    inference = ReCogDriveInference(model_cfg=model_cfg, device=args.device)

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
