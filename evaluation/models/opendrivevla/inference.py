"""
OpenDriveVLA inference script.

Loads OpenDriveVLA (Qwen2.5-VL based) and runs inference on the benchmark's
causal reasoning questions.

OpenDriveVLA is based on Qwen2.5-VL fine-tuned with UniAD feature extraction.
For VQA evaluation we use the raw Qwen2.5-VL backbone directly with images,
bypassing the UniAD preprocessing (which requires full NuScenes pipeline).

Usage (inside container):
    uv run python /workspace/repo/evaluation/models/opendrivevla/inference.py \\
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
from dataloader import OpenDriveVLADataLoader, FRAME_ORDER, CAMERA_ORDER  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------

MODEL_ID = "opendrivevla"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    "model_dtype": torch.bfloat16,
    "max_new_tokens": 512,
    "temperature": 0.01,
    "do_sample": False,
    "min_pixels": 256 * 28 * 28,
    "max_pixels": 1280 * 28 * 28,
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert autonomous driving assistant. "
    "You are shown camera images from a self-driving vehicle across multiple "
    "timesteps (past to present) and asked causal reasoning questions about "
    "the driving scene. "
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


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class OpenDriveVLAInference(ModelInference):
    """
    Inference class for OpenDriveVLA (Qwen2.5-VL backbone).

    Uses HuggingFace Qwen2_5_VLForConditionalGeneration directly.
    UniAD feature extraction is bypassed for the VQA benchmark evaluation.
    """

    def __init__(self, model_cfg: dict, device: str = "cuda") -> None:
        super().__init__(model_cfg, device)
        self.processor = None
        self.system_prompt = _load_system_prompt()
        self._device_type = "cuda" if "cuda" in device else "cpu"
        print(f"[OpenDriveVLAInference] Initialized | device={device} | dtype={CONFIG['model_dtype']}")

    def load_model(self, weights_path: str) -> None:
        """Load OpenDriveVLA from a local HuggingFace weights directory."""
        print(f"[load_model] Loading OpenDriveVLA from {weights_path} …")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                weights_path,
                min_pixels=CONFIG["min_pixels"],
                max_pixels=CONFIG["max_pixels"],
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                weights_path,
                torch_dtype=CONFIG["model_dtype"],
                device_map=self.device,
            )
            self.model.eval()
            print(f"[load_model] OpenDriveVLA loaded on {self.device}")
        except Exception:
            print(f"[load_model] ERROR loading model:\n{traceback.format_exc()}")
            raise

    def run_single(self, prompt_payload: dict) -> dict:
        """
        Run inference for one QA prompt and return the raw response dict.

        Pipeline:
          1. Reload PIL images from image_paths list
          2. Build Qwen2.5-VL chat messages with images + full_text
          3. Apply chat template and process images
          4. Generate and decode response
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
        # Step 1: Reload images in TIME-MAJOR order
        # ------------------------------------------------------------------
        pil_images: List[PILImage.Image] = []
        missing = 0

        for frame_key in FRAME_ORDER:
            for cam in CAMERA_ORDER:
                path = lookup.get((frame_key, cam))
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

        n_total = len(pil_images)
        print(f"[run_single] {n_total - missing}/{n_total} images ({missing} placeholder(s))")

        # ------------------------------------------------------------------
        # Step 2: Build Qwen2.5-VL chat messages
        # ------------------------------------------------------------------
        image_content = [{"type": "image", "image": img} for img in pil_images]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": full_text}],
            },
        ]

        # ------------------------------------------------------------------
        # Step 3: Apply chat template and process inputs
        # ------------------------------------------------------------------
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            print("[run_single] WARN: qwen_vl_utils not found, using processor directly")
            process_vision_info = None

        print("[run_single] Applying chat template …")
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if process_vision_info is not None:
                image_inputs, _ = process_vision_info(messages)
            else:
                image_inputs = pil_images
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        except Exception:
            print(f"[run_single] ERROR in chat template processing:\n{traceback.format_exc()}")
            raise

        # ------------------------------------------------------------------
        # Step 4: Generate
        # ------------------------------------------------------------------
        print(
            f"[run_single] Generating | max_new_tokens={CONFIG['max_new_tokens']} "
            f"do_sample={CONFIG['do_sample']}"
        )
        with torch.no_grad():
            with torch.autocast(self._device_type, dtype=CONFIG["model_dtype"]):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=CONFIG["max_new_tokens"],
                    do_sample=CONFIG["do_sample"],
                    temperature=CONFIG["temperature"] if CONFIG["do_sample"] else None,
                )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
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
        help="Path to OpenDriveVLA weights (local HuggingFace directory).",
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
    model_cfg: dict = {"qa_types": cfg.qa_types, "batch_size": batch_size}
    loader = OpenDriveVLADataLoader(dataset=dataset, model_cfg=model_cfg)

    sample_ids = dataset.get_sample_ids(
        mode=args.mode,
        subset_size=args.subset_size,
        scene=args.scene,
    )
    logger.info("%d sample(s) selected (mode=%s)", len(sample_ids), args.mode)

    inference = OpenDriveVLAInference(model_cfg=model_cfg, device=args.device)

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
