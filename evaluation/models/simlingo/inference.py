"""
SimLingo inference script.

Loads SimLingo (InternVL2 backbone) and runs inference on the benchmark's
causal reasoning questions.

SimLingo is trained on CARLA front-camera data and uses InternVL2 as its VLM.
We provide front-camera images across 4 timesteps matching its training distribution.

IMAGE PREPROCESSING:
  InternVL2 processes images at 448×448 via IMAGENET-normalised tensors.
  Each image is: T.Resize(448) → T.ToTensor() → T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

MULTI-IMAGE:
  pixel_values: (N, 3, 448, 448) torch.Tensor
  question:     '<image>\\n' repeated N times prepended to the prompt
  model.chat():  handles multi-image via num_patches_list

Usage (inside container):
    uv run python inference.py
    uv run python inference.py --mode subset --subset-size 50
    uv run python inference.py --mode single --scene nuscenes-scene-0001
    uv run python inference.py --bench-dir /causal_drive_bench/causal_nuscenes --output-base /cdb_outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_inference import ModelInference
from evaluation.common.dataset import BenchmarkDataset
from evaluation.common.utils.config_reader import CDBConfig
from evaluation.common.utils.exceptions import MissingFrameException, ModelNotReadyException
from dataloader import SimLingoDataLoader, FRAME_ORDER, CAMERA_ORDER, _INCLUDE_SIDE, _SIDE_CAMS  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "simlingo"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    "model_dtype": torch.bfloat16,
    "max_new_tokens": 512,
    "do_sample": False,
    # InternVL2 image size (CLIPImageProcessor)
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
    "You are shown front-camera images from a self-driving vehicle across multiple "
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


def _build_internvl_transform():
    """Build InternVL2's standard image transform pipeline."""
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

class SimLingoInference(ModelInference):
    """
    Inference class for SimLingo (InternVL2 backbone) on the causal benchmark.

    Provides front-camera temporal images in InternVL2's expected format.
    Uses model.chat() for single-turn multi-image QA.
    """

    def __init__(self, model_cfg: dict, device: str = "cuda") -> None:
        super().__init__(model_cfg, device)
        self.tokenizer = None
        self.system_prompt = _load_system_prompt()
        self.transform = _build_internvl_transform()
        self._device_type = "cuda" if "cuda" in device else "cpu"
        print(f"[SimLingoInference] Initialized | device={device} | dtype={CONFIG['model_dtype']}")

    def load_model(self, weights_path: str) -> None:
        print(f"[load_model] Loading SimLingo from {weights_path} …")
        try:
            # Load tokenizer from base model
            self.tokenizer = AutoTokenizer.from_pretrained(
                "OpenGVLab/InternVL2-1B",
                trust_remote_code=True,
                use_fast=False
            )

            # Load base InternVL2 architecture
            self.model = AutoModel.from_pretrained(
                "OpenGVLab/InternVL2-1B",
                torch_dtype=CONFIG["model_dtype"],
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            

            # Overlay SimLingo fine-tuned weights
            state_dict = torch.load(weights_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only = False)
            self.model.load_state_dict(state_dict, strict=False)

            self.model.eval().to(self.device)

            if hasattr(self.model, "system_message"):
                self.model.system_message = self.system_prompt

            print(f"[load_model] SimLingo (InternVL2) loaded on {self.device}")
        except Exception:
            print(f"[load_model] ERROR loading model:\n{traceback.format_exc()}")
            raise

    def run_single(self, prompt_payload: dict) -> dict:
        """
        Run inference for one QA prompt and return a dict with the raw response.

        Pipeline:
          1. Reload PIL images from image_paths list → apply InternVL2 transform
          2. Build question with <image>\\n tokens prepended
          3. Call model.chat() for generation
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
        # Step 1: Load and transform images
        # ------------------------------------------------------------------
        pixel_tensors: List[torch.Tensor] = []
        num_patches_list: List[int] = []
        missing = 0

        # Front camera temporal sequence
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

        # Optional side cameras at t=0
        if _INCLUDE_SIDE:
            for cam in _SIDE_CAMS:
                path = lookup.get(("Tp0p0", cam))
                if path and Path(path).exists():
                    try:
                        img = PILImage.open(path).convert("RGB")
                    except Exception:
                        img = PILImage.new("RGB", (1600, 900), color=(128, 128, 128))
                        missing += 1
                else:
                    img = PILImage.new("RGB", (1600, 900), color=(128, 128, 128))
                    missing += 1
                pixel_tensors.append(self.transform(img))
                num_patches_list.append(1)

        n_total = len(pixel_tensors)
        print(f"[run_single] {n_total - missing}/{n_total} images ({missing} placeholder(s))")

        # Stack to (N, 3, 448, 448) and move to device
        pixel_values = torch.stack(pixel_tensors, dim=0).to(
            device=self.device, dtype=CONFIG["model_dtype"]
        )

        # ------------------------------------------------------------------
        # Step 2: Build question with image tokens
        # ------------------------------------------------------------------
        full_text = f"{prompt_context}\n\n{question_text}".strip() if prompt_context else question_text
        # InternVL2 requires one '<image>\n' per image patch in the question
        image_prefix = "<image>\n" * n_total
        question = image_prefix + full_text

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

        # response is a string (or tuple if return_history=True)
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


def _has_pending_traj_work(sample_ids: list, run_dir: Path, causal_dataset_name: str, loader=None) -> bool:
    """
    Returns True if any sample is missing a trajectory output for any question in
    the *freshly-rebuilt* expected prompt set (BTE + per-question CTE).

    The previous on-disk trajectory_prompts.jsonl can be stale (e.g. predates
    CTE enumeration), so we recompute the expected question_id set per sample
    via iter_trajectory_questions().
    """
    if loader is None:
        # Without a loader we can only fall back to checking output presence.
        for sid in sample_ids:
            traj_file = (
                run_dir / causal_dataset_name / sid["scene_id"] / sid["sample_id"] / "trajectory_outputs.jsonl"
            )
            if not traj_file.exists() or traj_file.stat().st_size == 0:
                return True
        return False

    for sid in sample_ids:
        sample_dir = run_dir / causal_dataset_name / sid["scene_id"] / sid["sample_id"]
        traj_file = sample_dir / "trajectory_outputs.jsonl"
        try:
            expected = {
                rec.get("question_id", "")
                for rec in loader.iter_trajectory_questions([sid])
                if rec.get("question_id")
            }
        except Exception:
            return True
        if not expected:
            continue
        if not traj_file.exists() or traj_file.stat().st_size == 0:
            return True
        done = set()
        with open(traj_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line).get("question_id", ""))
                except json.JSONDecodeError:
                    pass
        if expected - done:
            return True
    return False


def _run_trajectory_inference(
    sample_ids: list,
    run_dir: Path,
    causal_dataset_name: str,
    loader,
    weights_path: str,
    device: str,
) -> None:
    """
    Second-pass trajectory inference using SimLingo's driving head.

    For each sample, writes:
      trajectory_prompts.jsonl  — one record per BTE/CTE question (GT + prompt text)
      trajectory_outputs.jsonl  — one record per question with raw_output JSON waypoints

    BTE uses build_bte_prompt_text; CTE uses build_cte_prompt_text (question-
    dependent). Both condition the LM features that the driving head operates on.
    The VQA answer for CTE is left to the postprocessor to look up from outputs.jsonl.
    """
    # Import SimLingoTrajModel from traj_eval.py (same directory)
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    from traj_eval import SimLingoTrajModel, GT_MATCH_IDX, GT_MATCH_TIMES, SIMLINGO_DT, N_WPS

    logger.info("[Traj] Loading SimLingoTrajModel for %d sample(s)…", len(sample_ids))
    model = SimLingoTrajModel(device=device)
    model.load(weights_path)

    import numpy as np

    n_ok = n_err = n_skip = 0

    for sid in sample_ids:
        scene_id  = sid["scene_id"]
        sample_id = sid["sample_id"]
        sample_out_dir = run_dir / causal_dataset_name / scene_id / sample_id

        traj_prompts_file = sample_out_dir / "trajectory_prompts.jsonl"
        traj_outputs_file = sample_out_dir / "trajectory_outputs.jsonl"

        sample_out_dir.mkdir(parents=True, exist_ok=True)

        # Always (re)build trajectory_prompts.jsonl so newly-added CTE entries
        # are picked up on resume. This is cheap (file IO only).
        try:
            n_prompts = loader.save_trajectory_prompts([sid], traj_prompts_file)
        except Exception:
            logger.error(
                "[Traj] save_trajectory_prompts failed for %s/%s:\n%s",
                scene_id, sample_id, traceback.format_exc(),
            )
            n_err += 1
            continue

        if n_prompts == 0:
            logger.warning("[Traj] No trajectory prompts for %s/%s — skipping", scene_id, sample_id)
            n_err += 1
            continue

        # Per-question resume: read existing outputs and skip done question_ids.
        existing_outputs: list = []
        done_qids: set = set()
        if traj_outputs_file.exists():
            with open(traj_outputs_file) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    qid = rec.get("question_id", "")
                    if qid:
                        done_qids.add(qid)
                        existing_outputs.append(line)

        # Find front-camera image path at T=0 (Tp0p0) from the first prompt record
        img_path = None
        try:
            with open(traj_prompts_file) as fh:
                first_line = fh.readline().strip()
            if first_line:
                first_rec = json.loads(first_line)
                for ip in first_rec.get("image_paths", []):
                    if ip.get("time_key") == "Tp0p0" and ip.get("camera_key") == "cam_front":
                        img_path = ip.get("path")
                        break
        except Exception:
            logger.error(
                "[Traj] Failed reading image path from prompts for %s/%s:\n%s",
                scene_id, sample_id, traceback.format_exc(),
            )

        if not img_path or not Path(img_path).exists():
            logger.warning(
                "[Traj] Front cam image not found for %s/%s — looked at %r (exists=%s) — skipping",
                scene_id, sample_id, img_path, bool(img_path) and Path(img_path).exists(),
            )
            n_err += 1
            continue

        # Run driving head for every prompt record (BTE + each CTE question)
        output_lines = list(existing_outputs)
        new_lines = 0
        with open(traj_prompts_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    prompt_rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                question_id = prompt_rec.get("question_id", "")
                qa_text     = prompt_rec.get("qa_text", "")

                if question_id in done_qids:
                    continue

                if not qa_text:
                    logger.warning("[Traj] Empty qa_text for %s/%s q=%s", scene_id, sample_id, question_id)
                    continue

                try:
                    pred_wps = model.predict(img_path, qa_text)  # [10, 2]
                except Exception:
                    logger.error(
                        "[Traj] predict() failed for %s/%s q=%s:\n%s",
                        scene_id, sample_id, question_id, traceback.format_exc(),
                    )
                    continue

                # Native predictions: t=0.25,0.5,...,2.5s (indices 0-9 at 0.25s steps)
                # Build a dense prediction array covering all GT timesteps.
                # For t <= 2.5s we interpolate from native points; beyond 2.5s we
                # linearly extrapolate using the last velocity step (pred[9]-pred[8]).
                gt_timesteps = prompt_rec.get("gt_timesteps", list(GT_MATCH_TIMES) + [3.0])
                native_times = [(i + 1) * SIMLINGO_DT for i in range(N_WPS)]  # 0.25..2.5s
                delta = pred_wps[9] - pred_wps[8]  # velocity over last 0.25s step

                formatted_wps = []
                for t in gt_timesteps:
                    if t <= native_times[-1] + 1e-6:
                        # interpolate within native range
                        idx_f = t / SIMLINGO_DT - 1          # 0-based float index
                        idx_lo = max(0, min(int(idx_f), N_WPS - 2))
                        idx_hi = idx_lo + 1
                        alpha  = idx_f - idx_lo
                        wp = (1 - alpha) * pred_wps[idx_lo] + alpha * pred_wps[idx_hi]
                    else:
                        # linear extrapolation beyond t=2.5s
                        steps_beyond = (t - native_times[-1]) / SIMLINGO_DT
                        wp = pred_wps[9] + steps_beyond * delta
                    formatted_wps.append({"t": float(t), "x": float(wp[0]), "y": float(wp[1])})
                raw_output = json.dumps({"waypoints": formatted_wps})
                output_lines.append(json.dumps({"question_id": question_id, "raw_output": raw_output}))
                new_lines += 1

                logger.info("[Traj] %s/%s q=%s | %d waypoints written", scene_id, sample_id, question_id, len(formatted_wps))

        if output_lines:
            traj_outputs_file.write_text("\n".join(output_lines) + "\n")
            if new_lines:
                n_ok += 1
            else:
                n_skip += 1
        else:
            logger.warning("[Traj] No outputs produced for %s/%s", scene_id, sample_id)
            n_err += 1

    logger.info("[Traj] Done: %d ok | %d errors | %d skipped (already done)", n_ok, n_err, n_skip)


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
        help="Path to SimLingo weights (local HuggingFace directory).",
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
    parser.add_argument(
        "--trajectory",
        action="store_true",
        default=False,
        help="Also run trajectory inference (BTE + CTE) using SimLingo's driving head.",
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
        "qa_types": cfg.qa_types,
        "batch_size": batch_size,
        "ns_root": str(cfg.dir_raw_data / "nuscenes"),
    }
    loader = SimLingoDataLoader(dataset=dataset, model_cfg=model_cfg)

    sample_ids = dataset.get_sample_ids(
        mode=args.mode,
        subset_size=args.subset_size,
        scene=args.scene,
    )
    logger.info("%d sample(s) selected (mode=%s)", len(sample_ids), args.mode)

    inference = SimLingoInference(model_cfg=model_cfg, device=args.device)

    vqa_pending  = _has_pending_work(sample_ids, run_dir, causal_dataset_name)
    traj_pending = args.trajectory and _has_pending_traj_work(sample_ids, run_dir, causal_dataset_name, loader=loader)

    if not vqa_pending and not traj_pending:
        logger.info("All samples already processed — nothing to do.")
        return

    if vqa_pending:
        try:
            inference.load_model(weights_path)
        except Exception:
            logger.error("FATAL: model load failed:\n%s", traceback.format_exc())
            raise

    total_prompts = 0
    total_outputs = 0

    if not vqa_pending:
        logger.info("All VQA outputs already present — skipping VQA inference.")

    for sid in sample_ids if vqa_pending else []:
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

    if vqa_pending:
        logger.info("VQA inference complete.")
        logger.info("  Total prompts : %d", total_prompts)
        logger.info("  Total outputs : %d", total_outputs)
        logger.info("  Timing stats  : %s", inference.timing_stats)

    if traj_pending:
        # Free VQA model from GPU before loading the trajectory driving head
        logger.info("Freeing VQA model before trajectory inference…")
        del inference.model
        inference.model = None
        import gc; gc.collect()
        torch.cuda.empty_cache()

        _run_trajectory_inference(
            sample_ids=sample_ids,
            run_dir=run_dir,
            causal_dataset_name=causal_dataset_name,
            loader=loader,
            weights_path=weights_path,
            device=args.device,
        )

    logger.info("  Outputs under : %s", run_dir)
    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    main()
