"""
Alpamayo 1.0 inference script.

Loads the AlpamayoR1-10B model and runs inference on the benchmark's causal
reasoning questions, producing a chain-of-causation (CoC) response for each.

ADAPTATIONS from PhysicalAI-AV dataset → CausalDriveBench:

  1. IMAGE FORMAT
     Original:  torch.Tensor (N_cameras, num_frames, 3, H, W)  then .flatten(0,1)
                → (16, 3, H, W) torch.Tensor passed to helper.create_message()
     This file: Load JPEGs via PIL from image_paths list (camera-major order),
                convert each to (3, H, W) uint8 tensor, stack → (16, 3, H, W).
     Critical:  helper.create_message() checks `frames.ndim == 4` and expects
                torch.Tensor(N, C, H, W).  numpy arrays or PIL Images will crash.

  2. EGO HISTORY SHAPE
     Original:  (1, 1, 16, 3)  —  batch=1, traj_sets=1, history_steps=16, xyz=3
     NuScenes:  state.json provides only 4 timesteps at [-1.5, -1.0, -0.5, 0.0]s
     Fix:       Linearly interpolate (4, 3) → (16, 3) via interpolate_ego_history(),
                then unsqueeze twice → (1, 1, 16, 3).

  3. QA INJECTION
     The Alpamayo system prompt and trajectory generation instruction are kept but
     the system role is augmented with our driving-QA system prompt.  The QA
     question text is appended as an extra user content block after trajectory tokens.
     With continue_final_message=True, the model continues from <|cot_start|> and
     produces chain-of-causation text that includes the structured answer.

  4. OUTPUT
     We extract extra["cot"] from sample_trajectories_from_data_with_vlm_rollout.
     The CoC text is returned as {"text": cot_text}; postprocess.py parses it.

  5. OUTPUT DIRECTORY STRUCTURE
     Outputs are written under a timestamped run directory:
       <dir_cdb_outputs>/<MODEL_ID>_<timestamp>/   ← run_dir
       ├── inference.log
       └── causal_nuscenes/
           └── nuscenes-scene-0159/
               └── SAMPLED_0/
                   ├── prompts.jsonl
                   └── outputs.jsonl

Run modes:
    python inference.py --bench-dir /workspace/data/causal_drive_bench/causal_nuscenes
    python inference.py --bench-dir ... --mode subset --subset-size 20
    python inference.py --bench-dir ... --mode single --scene nuscenes-scene-0159
    python inference.py --run-dir /workspace/data/cdb_outputs/alpamayo_1_0_20260410_120000

Paths and batch size default to eval_config.yaml via CDBConfig.
CLI flags are explicit overrides only.

Environment variables (runtime flags only — paths come from eval_config.yaml):
    EVAL_RUNTIME — "host" (default) or "docker"
    RUN_DIR      — existing run dir to resume
    DEVICE       — "cuda" or "cpu"
    MODE         — full | subset | single
    SUBSET_SIZE  — number of scenes for mode=subset
    SCENE        — scene name for mode=single
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

import numpy as np
import torch

# Allow running from repo root or within container
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_inference import ModelInference
from evaluation.common.dataset import BenchmarkDataset
from evaluation.common.utils.config_reader import CDBConfig
from evaluation.common.utils.errors import TensorShapeError
from evaluation.common.utils.exceptions import MissingFrameException, ModelNotReadyException

# Local imports: dataloader lives in the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader import (  # noqa: E402
    AlpamayoDataLoader,
    FRAME_ORDER,
    CAMERA_ORDER,
    _PLACEHOLDER_SIZE,
    interpolate_ego_history,
    inject_qa_into_messages,
    build_fallback_messages,
    CONFIG as DATALOADER_CONFIG,
)

logger = logging.getLogger(__name__)

MODEL_ID = "alpamayo_1_0"

# ---------------------------------------------------------------------------
# Config — update inference constants here
# ---------------------------------------------------------------------------
CONFIG = {
    "model_dtype": torch.bfloat16,
    "top_p": 0.95,
    "temperature": 0.6,
    "num_traj_samples": 1,
    "max_generation_length": 512,
    "num_history_steps": DATALOADER_CONFIG["num_history_steps"],
    "history_t_start_s": DATALOADER_CONFIG["history_t_start_s"],
    "history_t_end_s": DATALOADER_CONFIG["history_t_end_s"],
    "nuscenes_frame_times_s": DATALOADER_CONFIG["nuscenes_frame_times_s"],
    "placeholder_image_size_wh": _PLACEHOLDER_SIZE,
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
        logger.info("Loaded system prompt from %s", _SYSTEM_PROMPT_PATH)
        return prompt
    logger.info("system_prompt.txt not found, using default system prompt")
    return _DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class AlpamayoInference(ModelInference):
    """
    Inference class for the Alpamayo R1-10B model.

    Loads AlpamayoR1 from a local HuggingFace weights directory, formats the
    benchmark data into the model's input format, injects the causal reasoning
    question, and returns the chain-of-causation (CoC) text.

    Key differences from the original test_inference.py:
      - Images come from the image_paths list in prompts.jsonl (list of dicts)
      - Ego history is 4 sparse steps → interpolated to 16 dense steps
      - System prompt is augmented with the QA-focused system_prompt.txt
      - QA question is appended to the user message before generation
      - run_single() returns {"text": cot_text} (dict, not plain str)
    """

    def __init__(
        self,
        model_cfg: dict,
        device: str = "cuda",
        cfg: CDBConfig = None,
    ) -> None:
        super().__init__(model_cfg, device)
        self.cfg = cfg                       # CDBConfig for path resolution
        self.processor = None
        self.helper = None
        self.system_prompt = _load_system_prompt()
        self._device_type = "cuda" if "cuda" in device else "cpu"
        logger.info(
            "AlpamayoInference initialized | device=%s | dtype=%s",
            device, CONFIG["model_dtype"],
        )

    def load_model(self, weights_path: str) -> None:
        """
        Load AlpamayoR1-10B from a local HuggingFace weights directory.

        Sets self.model, self.helper, self.processor.
        """
        logger.info("Loading AlpamayoR1 from %s …", weights_path)

        _alpamayo_src = Path("/workspace/model/src")
        if _alpamayo_src.exists() and str(_alpamayo_src) not in sys.path:
            sys.path.insert(0, str(_alpamayo_src))
            logger.info("Added %s to sys.path", _alpamayo_src)

        try:
            from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
            from alpamayo_r1 import helper as alpamayo_helper

            self.helper = alpamayo_helper
            logger.info("alpamayo_r1 imported successfully")

            self.model = AlpamayoR1.from_pretrained(
                weights_path, dtype=CONFIG["model_dtype"]
            ).to(self.device)
            self.model.eval()
            logger.info("AlpamayoR1 moved to %s, set to eval mode", self.device)

            self.processor = alpamayo_helper.get_processor(self.model.tokenizer)
            logger.info("Qwen3VL processor loaded: %s", type(self.processor).__name__)

        except Exception:
            logger.error("ERROR loading model:\n%s", traceback.format_exc())
            raise

    def run_single(self, prompt_payload: dict) -> dict:
        """
        Run inference for one QA prompt and return the raw output dict.

        Pipeline:
          1. Load images from image_paths list → (16, 3, H, W) torch.Tensor
             in CAMERA-MAJOR order using CDBConfig for path resolution.
          2. Load ego_history_xyz (4, 3) and ego_history_rot (4, 3, 3) from payload
             → interpolate to (16, 3) and (16, 3, 3)
             → unsqueeze to (1, 1, 16, 3) and (1, 1, 16, 3, 3) for model.
          3. Build Qwen3VL messages using helper.create_message(image_tensor)
             → inject QA question (from framework field qa_text) into user message.
          4. Tokenise with processor.apply_chat_template().
          5. Call model.sample_trajectories_from_data_with_vlm_rollout().
          6. Extract extra["cot"] and return {"text": cot_text}.

        Args:
            prompt_payload: prompt record from prompts.jsonl (framework + model fields)

        Returns:
            dict with "text" key containing the raw CoC string.
        """
        if self.model is None:
            raise ModelNotReadyException("Call load_model() before run_single()")

        scene_id = prompt_payload.get("scene_id", "?")
        sample_id = prompt_payload.get("sample_id", "?")
        question_id = prompt_payload.get("question_id", "?")
        logger.info("[run_single] %s/%s | q=%s", scene_id, sample_id, question_id)

        # framework fields
        image_paths: List[dict] = prompt_payload.get("image_paths", [])
        qa_text: str = prompt_payload.get("qa_text", "")

        # model-specific fields
        ego_xyz_list = prompt_payload.get("ego_history_xyz", [])   # (4, 3) nested list
        ego_rot_list = prompt_payload.get("ego_history_rot", [])   # (4, 3, 3) nested list

        # ------------------------------------------------------------------
        # Step 1: Load images → torch.Tensor (16, 3, H, W), CAMERA-MAJOR
        # ------------------------------------------------------------------
        from PIL import Image as PILImage

        # Build a (time_key, camera_key) → path lookup for missing-frame handling
        path_lookup: dict = {
            (item["time_key"], item["camera_key"]): item["path"]
            for item in image_paths
            if isinstance(item, dict)
        }

        frame_tensors: List[torch.Tensor] = []
        missing_images = 0

        for cam in CAMERA_ORDER:
            for frame_key in FRAME_ORDER:
                rel_path = path_lookup.get((frame_key, cam))

                # Resolve relative path via CDBConfig if available
                if rel_path and self.cfg is not None:
                    abs_path = self.cfg.resolve_frame_path(rel_path)
                elif rel_path:
                    abs_path = Path(rel_path)
                else:
                    abs_path = None

                if abs_path and abs_path.exists():
                    try:
                        img_np = np.array(PILImage.open(abs_path).convert("RGB"))
                    except Exception:
                        logger.warning(
                            "[run_single] Failed to load %s:\n%s",
                            abs_path, traceback.format_exc(),
                        )
                        img_np = np.full(
                            (_PLACEHOLDER_SIZE[1], _PLACEHOLDER_SIZE[0], 3),
                            128, dtype=np.uint8,
                        )
                        missing_images += 1
                else:
                    if rel_path:
                        logger.warning(
                            "[run_single] Image file not found: %s", rel_path
                        )
                    img_np = np.full(
                        (_PLACEHOLDER_SIZE[1], _PLACEHOLDER_SIZE[0], 3),
                        128, dtype=np.uint8,
                    )
                    missing_images += 1

                # (H, W, 3) numpy uint8 → (3, H, W) torch uint8
                tensor = torch.from_numpy(img_np).permute(2, 0, 1)
                frame_tensors.append(tensor)

        image_frames = torch.stack(frame_tensors, dim=0)  # (16, 3, H, W)
        if image_frames.ndim != 4:
            raise TensorShapeError(
                f"Expected 4-D image tensor (N,C,H,W), got {tuple(image_frames.shape)}",
                obj=image_frames,
            )
        n_total = len(CAMERA_ORDER) * len(FRAME_ORDER)
        logger.info(
            "[run_single] Images: shape=%s (%d/%d real, %d placeholder)",
            tuple(image_frames.shape), n_total - missing_images, n_total, missing_images,
        )

        # ------------------------------------------------------------------
        # Step 2: Ego trajectory → (1, 1, 16, 3) and (1, 1, 16, 3, 3)
        # ------------------------------------------------------------------
        xyz_sparse = np.array(ego_xyz_list, dtype=np.float32)   # (4, 3)
        rot_sparse = np.array(ego_rot_list, dtype=np.float32)   # (4, 3, 3)

        xyz_dense, rot_dense = interpolate_ego_history(
            xyz_sparse, rot_sparse,
            num_steps=CONFIG["num_history_steps"],
            t_start=CONFIG["history_t_start_s"],
            t_end=CONFIG["history_t_end_s"],
        )

        # IMPORTANT: Must use float32 — original code uses torch.from_numpy(...).float()
        ego_history_xyz = (
            torch.tensor(xyz_dense, dtype=torch.float32, device=self.device)
            .unsqueeze(0)   # (1, 16, 3)
            .unsqueeze(0)   # (1, 1, 16, 3)
        )
        ego_history_rot = (
            torch.tensor(rot_dense, dtype=torch.float32, device=self.device)
            .unsqueeze(0)   # (1, 16, 3, 3)
            .unsqueeze(0)   # (1, 1, 16, 3, 3)
        )
        logger.info(
            "[run_single] Ego history: xyz=%s rot=%s",
            tuple(ego_history_xyz.shape), tuple(ego_history_rot.shape),
        )

        # ------------------------------------------------------------------
        # Step 3: Build Qwen3VL messages with QA injection
        # ------------------------------------------------------------------
        try:
            if self.helper is None:
                raise RuntimeError("helper module not loaded")
            base_messages = self.helper.create_message(image_frames)
            messages = inject_qa_into_messages(
                base_messages, qa_text, self.system_prompt,
                override_system_prompt=True,
            )
            logger.info("[run_single] Messages built via helper.create_message() + QA injection")
        except Exception:
            logger.warning(
                "[run_single] helper.create_message() failed, using fallback:\n%s",
                traceback.format_exc(),
            )
            messages = build_fallback_messages(image_frames, qa_text, self.system_prompt)

        # ------------------------------------------------------------------
        # Step 4: Tokenise with processor
        # ------------------------------------------------------------------
        logger.info("[run_single] Applying chat template and tokenising …")
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,  # continue from <|cot_start|>
            return_dict=True,
            return_tensors="pt",
        )

        # ------------------------------------------------------------------
        # Step 5: Assemble model inputs and run VLM rollout
        # ------------------------------------------------------------------
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        model_inputs = self.helper.to_device(model_inputs, "cuda")

        logger.info(
            "[run_single] Running VLM rollout | "
            "top_p=%s temperature=%s num_traj=%s max_gen=%s",
            CONFIG["top_p"], CONFIG["temperature"],
            CONFIG["num_traj_samples"], CONFIG["max_generation_length"],
        )
        with torch.no_grad():
            with torch.autocast(self._device_type, dtype=CONFIG["model_dtype"]):
                _, _, extra = (
                    self.model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=CONFIG["top_p"],
                        temperature=CONFIG["temperature"],
                        num_traj_samples=CONFIG["num_traj_samples"],
                        max_generation_length=CONFIG["max_generation_length"],
                        return_extra=True,
                    )
                )

        # ------------------------------------------------------------------
        # Step 6: Extract CoC text from extra["cot"]
        # ------------------------------------------------------------------
        cot = extra.get("cot", [""])
        logger.info(
            "[run_single] CoC raw type=%s shape=%s",
            type(cot), getattr(cot, "shape", len(cot)),
        )

        # Unwrap nested list/array structure (batch=1, traj_sets=1, num_samples=1)
        for _ in range(3):
            if isinstance(cot, (list, tuple, np.ndarray)) and len(cot) > 0:
                cot = cot[0]

        result = str(cot)
        logger.info("[run_single] CoC text (%d chars): %r…", len(result), result[:120])
        return {"text": result}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run {MODEL_ID} inference on the CausalDriveBench dataset"
    )
    parser.add_argument(
        "--bench-dir",
        default="",
        help=(
            "Dataset-specific bench directory "
            "(e.g. /workspace/data/causal_drive_bench/causal_nuscenes). "
            "Defaults to CDBConfig.get_bench_dir(--dataset) from eval_config.yaml."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("DATASET", "nuscenes"),
        choices=["nuscenes", "argoverse2", "openscene"],
        help="Dataset name — used to resolve bench-dir from eval_config.yaml",
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
        help="Scene name for mode=single (e.g. nuscenes-scene-0159)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Existing run directory to resume "
            "(e.g. /workspace/data/cdb_outputs/alpamayo_1_0_20260410_120000). "
            "If omitted, a new timestamped directory is created under dir_cdb_outputs."
        ),
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help=(
            "Override for dir_cdb_outputs from eval_config.yaml. "
            "Defaults to CDBConfig.dir_cdb_outputs."
        ),
    )
    parser.add_argument(
        "--weights-path",
        default="",
        help=(
            "Path to Alpamayo model weights (local HuggingFace directory). "
            "Defaults to CDBConfig.dir_model_weights from eval_config.yaml."
        ),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Inference batch size (1 recommended for Alpamayo-R1-10B). "
            "Defaults to evaluation.batch_size from eval_config.yaml."
        ),
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
    """Configure root logger to write to both stdout and *log_file*."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    logger.info("Logging to %s", log_file)


# ---------------------------------------------------------------------------
# Pending work check (GPU-lazy loading)
# ---------------------------------------------------------------------------

def _has_pending_work(
    sample_ids: list,
    run_dir: Path,
    causal_dataset_name: str,
) -> bool:
    """
    Return True if any sample has unprocessed questions.

    A sample is considered complete if its outputs.jsonl exists and is non-empty.
    This lightweight check prevents loading the GPU model when all work is done.
    """
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

def main() -> None:
    args = parse_args()

    # --- Config and path resolution ----------------------------------------
    cfg = CDBConfig(config_path=args.config)
    print(cfg.summary())

    bench_dir = Path(args.bench_dir) if args.bench_dir else cfg.get_bench_dir(args.dataset)
    causal_dataset_name = bench_dir.name                    # e.g. "causal_nuscenes"

    weights_path = args.weights_path or str(cfg.dir_model_weights)
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size

    # --- Run directory setup -----------------------------------------------
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

    # --- Logging (write to run_dir/inference.log) --------------------------
    setup_logging(run_dir / "inference.log")
    logger.info("Run directory: %s", run_dir)
    logger.info("Bench directory: %s", bench_dir)
    logger.info(
        "Mode: %s | batch_size: %d | device: %s",
        args.mode, batch_size, args.device,
    )

    # --- Dataset and DataLoader --------------------------------------------
    dataset = BenchmarkDataset(bench_dir=bench_dir)
    model_cfg: dict = {
        "qa_types": cfg.qa_types,
        "batch_size": batch_size,
    }
    loader = AlpamayoDataLoader(dataset=dataset, model_cfg=model_cfg, cfg=cfg)
    logger.info("Bench dir: %s | dataset: %s", bench_dir, causal_dataset_name)

    sample_ids = dataset.get_sample_ids(
        mode=args.mode,
        subset_size=args.subset_size,
        scene=args.scene,
    )
    logger.info("%d sample(s) selected (mode=%s)", len(sample_ids), args.mode)

    # --- GPU-lazy load: skip if everything already processed ---------------
    inference = AlpamayoInference(model_cfg=model_cfg, device=args.device, cfg=cfg)

    has_pending = _has_pending_work(sample_ids, run_dir, causal_dataset_name)
    if not has_pending:
        logger.info("All samples already processed — nothing to do.")
        return

    try:
        inference.load_model(weights_path)
    except Exception:
        logger.error("FATAL: model load failed:\n%s", traceback.format_exc())
        raise

    # --- Per-sample inference loop ----------------------------------------
    total_prompts = 0
    total_outputs = 0

    for sid in sample_ids:
        scene_id = sid["scene_id"]
        sample_id = sid["sample_id"]

        # Nested output dir: run_dir/<causal_dataset>/<scene_id>/<sample_id>/
        sample_out_dir = run_dir / causal_dataset_name / scene_id / sample_id
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        prompts_file = sample_out_dir / "prompts.jsonl"
        outputs_file = sample_out_dir / "outputs.jsonl"

        logger.info("[main] ── %s/%s ──", scene_id, sample_id)

        # Always regenerate prompts.jsonl (deterministic + cheap vs inference)
        try:
            n_prompts = loader.save_prompts([sid], prompts_file)
        except Exception:
            logger.error(
                "ERROR saving prompts for %s/%s:\n%s",
                scene_id, sample_id, traceback.format_exc(),
            )
            continue

        if n_prompts == 0:
            logger.warning(
                "No prompts generated for %s/%s — skipping", scene_id, sample_id
            )
            continue

        total_prompts += n_prompts
        logger.info(
            "[main] Running inference for %s/%s (%d prompts)", scene_id, sample_id, n_prompts
        )

        try:
            n_written = inference.run_from_jsonl(
                input_path=prompts_file,
                output_path=outputs_file,
                batch_size=batch_size,
                resume=True,
            )
            logger.info(
                "[main] Completed %s/%s: %d new output(s)", scene_id, sample_id, n_written
            )
            total_outputs += n_written
        except Exception:
            logger.error(
                "ERROR during inference for %s/%s:\n%s",
                scene_id, sample_id, traceback.format_exc(),
            )
            continue

    # --- Summary -----------------------------------------------------------
    logger.info("Inference complete.")
    logger.info("  Total prompts : %d", total_prompts)
    logger.info("  Total outputs : %d", total_outputs)
    logger.info("  Timing stats  : %s", inference.timing_stats)
    logger.info("  Outputs under : %s", run_dir)
    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    main()
