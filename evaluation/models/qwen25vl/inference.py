"""
Qwen2.5-VL inference script for CausalDriveBench.

Supports: Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct (HuggingFace IDs or local paths).

Usage (host):
    python evaluation/models/qwen25vl/inference.py \\
        --config evaluation/eval_config.local.yaml \\
        --model-id Qwen/Qwen2.5-VL-3B-Instruct \\
        --mode subset --subset-size 10

    python evaluation/models/qwen25vl/inference.py \\
        --model-id /path/to/local/weights \\
        --bench-dir /path/to/causal_nuscenes \\
        --output-base /path/to/cdb_outputs

Usage (Docker):
    uv run python inference.py --mode full
    uv run python inference.py --mode subset --subset-size 10
    uv run python inference.py --mode single --scene nuscenes-scene-0159
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_inference import ModelInference
from evaluation.common.dataset import BenchmarkDataset
from evaluation.common.utils.config_reader import CDBConfig
from evaluation.common.utils.exceptions import ModelNotReadyException
from dataloader import Qwen25VLDataLoader  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_ID = "qwen25vl"

CONFIG = {
    "torch_dtype": torch.bfloat16,
    "max_new_tokens": 512,
    "temperature": 0.1,
    "do_sample": False,
    "min_pixels": 256 * 28 * 28,
    "max_pixels": 1280 * 28 * 28,
}


class Qwen25VLInference(ModelInference):

    def __init__(self, model_cfg: dict, device: str = "cuda", cfg: "CDBConfig | None" = None) -> None:
        super().__init__(model_cfg, device)
        self.cfg = cfg
        self.processor = None
        logger.info("[Qwen25VLInference] Initialized | device=%s | dtype=%s", device, CONFIG["torch_dtype"])

    def load_model(self, weights_path: str) -> None:
        logger.info("[load_model] Loading Qwen2.5-VL from %s ...", weights_path)
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                weights_path,
                torch_dtype=CONFIG["torch_dtype"],
                device_map="auto",
            )
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(
                weights_path,
                min_pixels=CONFIG["min_pixels"],
                max_pixels=CONFIG["max_pixels"],
            )
            logger.info("[load_model] Qwen2.5-VL loaded on device=%s", next(self.model.parameters()).device)
        except Exception:
            logger.error("[load_model] ERROR:\n%s", traceback.format_exc())
            raise

    def run_single(self, prompt_payload: dict) -> dict:
        if self.model is None:
            raise ModelNotReadyException("Call load_model() first")

        from qwen_vl_utils import process_vision_info
        from PIL import Image as PILImage

        user_text = prompt_payload.get("user_text", "")
        image_paths: list = prompt_payload.get("image_paths", [])

        # Resolve relative paths via CDBConfig; fall back to treating as absolute
        pil_images = []
        missing = 0
        for item in image_paths:
            rel_path = item.get("path", "")
            if rel_path and self.cfg is not None:
                abs_path = self.cfg.resolve_frame_path(rel_path)
            elif rel_path:
                abs_path = Path(rel_path)
            else:
                abs_path = None

            if abs_path and abs_path.exists():
                try:
                    pil_images.append(PILImage.open(abs_path).convert("RGB"))
                except Exception:
                    pil_images.append(PILImage.new("RGB", (1600, 900), color=(128, 128, 128)))
                    missing += 1
            else:
                pil_images.append(PILImage.new("RGB", (1600, 900), color=(128, 128, 128)))
                missing += 1

        n_total = len(pil_images)
        logger.info("[run_single] %d/%d images (%d placeholder(s))", n_total - missing, n_total, missing)

        from dataloader import SYSTEM_PROMPT
        image_content = [{"type": "image", "image": img} for img in pil_images]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": image_content + [{"type": "text", "text": user_text}]},
        ]

        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=CONFIG["max_new_tokens"],
                    temperature=CONFIG["temperature"],
                    do_sample=CONFIG["do_sample"],
                )

            generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            logger.info("[run_single] Response (%d chars): %r...", len(response), response[:80])
            return {"text": response}

        except Exception:
            err = traceback.format_exc()
            logger.error("[run_single] ERROR:\n%s", err)
            return {"text": f"ERROR: {err[:200]}"}


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a")],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL CausalDriveBench inference")
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct"),
                        help="HuggingFace model ID or local path to weights")
    parser.add_argument("--config", default=None, help="Path to eval_config.yaml (auto-detected if omitted)")
    parser.add_argument("--bench-dir", default="", help="Override bench dir from eval_config.yaml")
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "nuscenes"),
                        choices=["nuscenes", "argoverse2", "openscene"])
    parser.add_argument("--mode", choices=["full", "subset", "single"], default=os.environ.get("MODE", "full"))
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--scene", default=os.environ.get("SCENE") or None)
    parser.add_argument("--output-base", default=None, help="Override cdb_outputs dir from eval_config.yaml")
    parser.add_argument("--run-dir", default=None, help="Existing run directory to resume")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = CDBConfig(config_path=args.config)
    logger.info(cfg.summary())

    bench_dir = Path(args.bench_dir) if args.bench_dir else cfg.get_bench_dir(args.dataset)
    causal_dataset_name = bench_dir.name
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path(args.output_base) if args.output_base else cfg.dir_cdb_outputs
        run_dir = output_base / f"{MODEL_ID}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(run_dir / "inference.log")
    logger.info("Run directory: %s", run_dir)
    logger.info("Bench directory: %s", bench_dir)
    logger.info("Model: %s | mode: %s | batch_size: %d | device: %s",
                args.model_id, args.mode, batch_size, args.device)

    dataset = BenchmarkDataset(bench_dir=bench_dir)
    model_cfg = {"qa_types": cfg.qa_types, "batch_size": batch_size}
    loader = Qwen25VLDataLoader(dataset=dataset, model_cfg=model_cfg)
    sample_ids = dataset.get_sample_ids(mode=args.mode, subset_size=args.subset_size, scene=args.scene)
    logger.info("%d sample(s) selected (mode=%s)", len(sample_ids), args.mode)

    inference = Qwen25VLInference(model_cfg=model_cfg, device=args.device, cfg=cfg)
    inference.load_model(args.model_id)

    total_prompts = total_outputs = 0

    for sample in sample_ids:
        scene_id, sample_id = sample["scene_id"], sample["sample_id"]
        sample_out_dir = run_dir / causal_dataset_name / scene_id / sample_id
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        prompts_jsonl = sample_out_dir / "prompts.jsonl"
        outputs_jsonl = sample_out_dir / "outputs.jsonl"

        logger.info("[main] ── %s/%s ──", scene_id, sample_id)
        try:
            n_prompts = loader.save_prompts([sample], prompts_jsonl)
        except Exception:
            logger.error("ERROR saving prompts for %s/%s:\n%s", scene_id, sample_id, traceback.format_exc())
            continue

        if n_prompts == 0:
            logger.warning("No prompts for %s/%s — skipping", scene_id, sample_id)
            continue
        total_prompts += n_prompts

        try:
            n_written = inference.run_from_jsonl(
                input_path=prompts_jsonl,
                output_path=outputs_jsonl,
                batch_size=batch_size,
                resume=True,
            )
            logger.info("[main] Completed %s/%s: %d output(s)", scene_id, sample_id, n_written)
            total_outputs += n_written
        except Exception:
            logger.error("ERROR during inference for %s/%s:\n%s", scene_id, sample_id, traceback.format_exc())
            continue

    logger.info("Inference complete. prompts=%d outputs=%d run_dir=%s", total_prompts, total_outputs, run_dir)
    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    main()
