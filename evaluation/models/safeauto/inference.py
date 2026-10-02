"""
SafeAuto inference script for CausalDriveBench.

SafeAuto is a LLaVA-based model (LanguageBind image encoder + LLaMA backbone)
fine-tuned on DriveLM (nuScenes multi-view QA) with PDCE loss and optional
RAG/MLN safety verification.  We use the SafeAuto-DriveLM checkpoint with
the image modality only (matching DriveLM evaluation setup).

Key adaptations from the original eval_custom_predsig_drivinglm.py:
  1. WORKING DIRECTORY: Must chdir to the SafeAuto repo before loading because
     builder.py hardcodes cache_dir='./cache_dir' for LanguageBind tower loading.
  2. SINGLE-TURN: Each CausalDriveBench question is one independent turn;
     conversation is reset per question (not multi-turn like DriveLM).
  3. 4 CAMERAS: DriveLM uses 6; we have front/front-left/front-right/back.
  4. IMAGE TENSOR: processor['image'] is called with abs path strings, producing
     (1, C, H, W) tensors; these are wrapped as [[t1, t2, t3, t4]], key=['image'].
  5. TOKEN INDEX: tokenizer_X_token uses X_TOKEN_INDEX['VIDEO'] (-201) to replace
     <image> tokens — this matches the DriveLM training tokenization quirk.

Run modes:
    python inference.py --bench-dir /path/to/causal_nuscenes --mode single --scene nuscenes-scene-0159
    python inference.py --bench-dir /path/to/causal_nuscenes --mode full
    python inference.py --bench-dir ... --run-dir /path/to/existing/run  # resume

Environment:
    conda activate safeauto
    export WEIGHTS_PATH=/workspace/ckpts/SafeAuto-DriveLM
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

import torch

SAFEAUTO_REPO = Path(__file__).resolve().parent / "SafeAuto"

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_inference import ModelInference
from evaluation.common.dataset import BenchmarkDataset
from evaluation.common.utils.config_reader import CDBConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader import make_dataloader, dataset_name_from_bench_dir  # noqa: E402
from traj_dataloader import iter_traj_prompts                         # noqa: E402
from traj_postprocess import process_sample, aggregate_results, print_report  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_ID = "safeauto"


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class SafeAutoInference(ModelInference):
    """
    Inference class for SafeAuto-DriveLM on the CausalDriveBench benchmark.

    Wraps the SafeAuto LLaVA inference pipeline, adapted for single-turn
    causal QA with 4 nuScenes camera images.
    """

    def __init__(
        self,
        model_cfg: dict,
        device: str = "cuda",
        cfg: CDBConfig = None,
    ) -> None:
        super().__init__(model_cfg, device)
        self.cfg = cfg
        self.tokenizer = None
        self.image_processor = None

    def _ensure_languagebind_encoders(self) -> None:
        """
        Download LanguageBind vision encoders to SafeAuto/cache_dir/ if not present.

        Both Image and Video_merge towers are declared in the checkpoint config
        ("X": ["Video", "Image"]) and are initialised even with delay_load=True,
        so both directories must exist before load_pretrained_model() is called.
        """
        from huggingface_hub import snapshot_download

        cache_dir = SAFEAUTO_REPO / "cache_dir"
        cache_dir.mkdir(exist_ok=True)

        encoders = [
            ("LanguageBind/LanguageBind_Image",       cache_dir / "LanguageBind_Image"),
            ("LanguageBind/LanguageBind_Video_merge", cache_dir / "LanguageBind_Video_merge"),
        ]

        for repo_id, local_dir in encoders:
            # A downloaded repo always has a config.json at its root
            if (local_dir / "config.json").exists():
                logger.info("LanguageBind encoder already present: %s", local_dir)
                continue
            logger.info(
                "Downloading %s → %s (first-time setup, ~3 GB) …", repo_id, local_dir
            )
            snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
            logger.info("Downloaded %s", local_dir.name)

    def load_model(self, weights_path: str) -> None:
        """
        Load SafeAuto-DriveLM from a local HuggingFace weights directory.

        Must chdir to the SafeAuto repo so that builder.py's hardcoded
        cache_dir='./cache_dir' resolves to the LanguageBind models.

        LanguageBind encoders are downloaded automatically on first run if
        not already present at SafeAuto/cache_dir/LanguageBind_{Image,Video_merge}.
        """
        logger.info("Changing cwd to SafeAuto repo: %s", SAFEAUTO_REPO)
        os.chdir(SAFEAUTO_REPO)

        if str(SAFEAUTO_REPO) not in sys.path:
            sys.path.insert(0, str(SAFEAUTO_REPO))

        # LanguageBind encoders must exist at ./cache_dir/ (relative to SafeAuto repo)
        # before loading the model — the builder hardcodes cache_dir='./cache_dir'.
        self._ensure_languagebind_encoders()

        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()

        # builder.py routes on 'llava' in model_name.lower() to load visual towers.
        # The HF checkpoint dir is named "SafeAuto-DriveLM" (no "llava"), so we
        # inject a name that passes the check without changing the weights path.
        model_name = "llava-SafeAuto-DriveLM"
        logger.info("Loading SafeAuto model '%s' from %s", model_name, weights_path)

        self.tokenizer, self.model, processor, self.context_len = load_pretrained_model(
            weights_path,
            model_base=None,
            model_name=model_name,
            device=self.device,
        )
        self.model.eval()
        self.image_processor = processor["image"]
        logger.info(
            "SafeAuto loaded | device=%s | context_len=%d", self.device, self.context_len
        )

    def run_single(self, prompt_payload: dict) -> dict:
        """
        Run inference for one CausalDriveBench QA question.

        Pipeline:
          1. Load 4 T=0 camera images via the LanguageBind image processor.
          2. Build a single-turn DriveLM-style conversation (reset per question).
          3. Tokenise with tokenizer_X_token (splits on <image>, inserts VIDEO index).
          4. Generate with model.generate(images=[tensor_list, ['image']]).
          5. Decode, strip stop tokens, return {"text": output}.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before run_single()")

        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_X_token, KeywordsStoppingCriteria
        from llava.constants import X_TOKEN_INDEX

        scene_id = prompt_payload.get("scene_id", "?")
        sample_id = prompt_payload.get("sample_id", "?")
        question_id = prompt_payload.get("question_id", "?")
        logger.debug("[run_single] %s/%s | q=%s", scene_id, sample_id, question_id)

        image_paths: list = prompt_payload.get("image_paths", [])
        conversation_text: str = prompt_payload.get("conversation_text", "")

        # ------------------------------------------------------------------
        # Step 1: Load images via LanguageBind image processor
        # processor['image'] accepts a file path string and returns a dict
        # with 'pixel_values' of shape (1, C, H, W).
        # ------------------------------------------------------------------
        image_tensors = []
        missing = 0

        for item in image_paths:
            rel_path = item.get("path", "")
            if rel_path and self.cfg is not None:
                abs_path = self.cfg.resolve_frame_path(rel_path)
                # frames.json contains a 'samples/' level that may be absent locally
                # (e.g. raw_data/nuscenes/samples/CAM_FRONT/ → raw_data/nuscenes/CAM_FRONT/)
                # if not abs_path.exists() and "/samples/" in str(abs_path):
                #     abs_path = Path(str(abs_path).replace("/samples/", "/"))
            elif rel_path:
                abs_path = Path(rel_path)
            else:
                abs_path = None

            if abs_path and abs_path.exists():
                try:
                    t = self.image_processor(str(abs_path), return_tensors="pt")[
                        "pixel_values"
                    ]
                    image_tensors.append(t)
                except Exception:
                    logger.warning(
                        "[run_single] Failed to process %s:\n%s",
                        abs_path, traceback.format_exc(),
                    )
                    missing += 1
                    continue
            else:
                logger.warning("[run_single] Image not found: %s", abs_path or rel_path)
                missing += 1

        if not image_tensors:
            logger.error(
                "[run_single] No images loaded for %s/%s — returning empty",
                scene_id, sample_id,
            )
            return {"text": ""}

        if missing:
            logger.warning(
                "[run_single] %d/%d images missing for %s/%s",
                missing, len(image_paths), scene_id, sample_id,
            )

        # tensor: [[t1, t2, t3, t4]]  each t_i: (1, C, H, W) float16
        tensor = [[t.to(self.model.device, dtype=torch.float16) for t in image_tensors]]
        key = ["image"]

        # ------------------------------------------------------------------
        # Step 2: Build single-turn "driving" conversation
        # ------------------------------------------------------------------
        conv = conv_templates["driving"].copy()
        conv.messages.clear()
        conv.append_message(conv.roles[0], conversation_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # ------------------------------------------------------------------
        # Step 3: Tokenise — replace <image> tokens with VIDEO index (-201)
        # This matches DriveLM training tokenization: images are embedded as
        # X_TOKEN_INDEX['VIDEO'] even when passed with key=['image'].
        # ------------------------------------------------------------------
        input_ids = (
            tokenizer_X_token(
                prompt, self.tokenizer, X_TOKEN_INDEX["VIDEO"], return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        stop_str = (
            conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        )
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        # ------------------------------------------------------------------
        # Step 4: Generate
        # ------------------------------------------------------------------
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[tensor, key],
                do_sample=False,
                temperature=0.0,
                max_new_tokens=256,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        # ------------------------------------------------------------------
        # Step 5: Decode
        # ------------------------------------------------------------------
        output = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        output = output.replace("</s>", "").strip()

        logger.debug(
            "[run_single] output (%d chars): %r…", len(output), output[:120]
        )
        return {"text": output}


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
            "Dataset-specific bench directory "
            "(e.g. /path/to/causal_nuscenes). "
            "Falls back to CDBConfig.get_bench_dir('nuscenes') if empty."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("DATASET", "nuscenes"),
        choices=["nuscenes", "openscene"],
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
    )
    parser.add_argument(
        "--scene",
        default=os.environ.get("SCENE") or None,
        help="Scene name for mode=single (e.g. nuscenes-scene-0159)",
    )
    parser.add_argument(
        "--run-dir",
        default=os.environ.get("RUN_DIR") or None,
        help="Existing run directory to resume.",
    )
    parser.add_argument(
        "--output-base",
        default=os.environ.get("OUTPUT_DIR") or None,
    )
    parser.add_argument(
        "--weights-path",
        default=os.environ.get(
            "WEIGHTS_PATH", "/workspace/ckpts/SafeAuto-DriveLM"
        ),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", 2)),
        help="Inference batch size (1 recommended for SafeAuto)",
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
        help=(
            "Run trajectory evaluation (BTE + CTE_R2 + CTE_R3) after QA inference. "
            "Writes trajectory_outputs.jsonl and trajectory_report.json per sample."
        ),
    )
    parser.add_argument(
        "--force-trajectory",
        action="store_true",
        default=False,
        help="Force regeneration of trajectory outputs even if already present. Implies --trajectory.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
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
# Pending-work check (avoids loading the model when nothing to do)
# ---------------------------------------------------------------------------

def _has_pending_work(
    sample_ids: list,
    run_dir: Path,
    causal_dataset_name: str,
    check_trajectory: bool = False,
) -> bool:
    for sid in sample_ids:
        sample_dir = run_dir / causal_dataset_name / sid["scene_id"] / sid["sample_id"]
        outputs_file = sample_dir / "outputs.jsonl"
        if not outputs_file.exists() or outputs_file.stat().st_size == 0:
            return True
        if check_trajectory:
            traj_file = sample_dir / "trajectory_outputs.jsonl"
            if not traj_file.exists() or traj_file.stat().st_size == 0:
                return True
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg = CDBConfig(config_path=args.config)
    print(cfg.summary())

    bench_dir = (
        Path(args.bench_dir) if args.bench_dir else cfg.get_bench_dir(args.dataset)
    ).resolve()  # absolute — load_model() calls os.chdir() which breaks relative paths
    causal_dataset_name = bench_dir.name           # e.g. "nuscenes_full_ice"
    dataset_name = args.dataset or dataset_name_from_bench_dir(causal_dataset_name)

    weights_path = str(Path(args.weights_path or str(cfg.dir_model_weights)).resolve())
    batch_size = args.batch_size

    # Run directory — also resolved to absolute for the same reason
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
        logger.info("Resuming existing run: %s", run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = (
            Path(args.output_base) if args.output_base else cfg.dir_cdb_outputs
        )
        run_dir = (output_base / f"{MODEL_ID}_{timestamp}").resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir / "inference.log")
    logger.info("Run directory: %s", run_dir)
    logger.info("Bench directory: %s", bench_dir)
    logger.info(
        "Mode: %s | batch_size: %d | device: %s",
        args.mode, batch_size, args.device,
    )

    # Dataset and dataloader
    dataset = BenchmarkDataset(bench_dir=bench_dir)
    model_cfg: dict = {"qa_types": cfg.qa_types, "batch_size": batch_size}
    loader = make_dataloader(dataset_name, dataset, model_cfg)

    sample_ids = dataset.get_sample_ids(
        mode=args.mode,
        subset_size=args.subset_size,
        scene=args.scene,
    )
    logger.info("%d sample(s) selected (mode=%s)", len(sample_ids), args.mode)

    # GPU-lazy load
    inference = SafeAutoInference(model_cfg=model_cfg, device=args.device, cfg=cfg)

    run_trajectory = args.trajectory or args.force_trajectory
    if not _has_pending_work(sample_ids, run_dir, causal_dataset_name, check_trajectory=run_trajectory):
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
            logger.warning(
                "No prompts for %s/%s — skipping", scene_id, sample_id
            )
            continue

        total_prompts += n_prompts
        logger.info(
            "[main] Inference for %s/%s (%d prompts)", scene_id, sample_id, n_prompts
        )

        try:
            n_written = inference.run_from_jsonl(
                input_path=prompts_file,
                output_path=outputs_file,
                batch_size=batch_size,
                resume=True,
            )
            total_outputs += n_written
            logger.info(
                "[main] Done %s/%s: %d output(s)", scene_id, sample_id, n_written
            )
        except Exception:
            logger.error(
                "ERROR during inference for %s/%s:\n%s",
                scene_id, sample_id, traceback.format_exc(),
            )
            continue

    logger.info(
        "Inference complete | prompts=%d outputs=%d | run=%s",
        total_prompts, total_outputs, run_dir,
    )

    # ------------------------------------------------------------------
    # Trajectory evaluation (BTE + CTE_R2 + CTE_R3)
    # ------------------------------------------------------------------
    if run_trajectory:
        logger.info("── Trajectory evaluation ──")

        all_traj_results = []

        for sid in sample_ids:
            scene_id  = sid["scene_id"]
            sample_id = sid["sample_id"]

            sample_out_dir = run_dir / causal_dataset_name / scene_id / sample_id
            traj_out_file  = sample_out_dir / "trajectory_outputs.jsonl"

            # Build BTE + CTE prompts for this sample
            traj_prompts = list(iter_traj_prompts(bench_dir, [sid]))
            if not traj_prompts:
                logger.warning("[traj] No prompts for %s/%s", scene_id, sample_id)
                continue

            # Resume: load already-done results (skipped when --force-trajectory)
            done_keys: set = set()
            existing_results: list = []
            if traj_out_file.exists() and not args.force_trajectory:
                for line in traj_out_file.read_text().splitlines():
                    try:
                        r = json.loads(line)
                        done_keys.add((r.get("type"), r.get("question_id")))
                        existing_results.append(r)
                    except Exception:
                        pass

            pending = [
                p for p in traj_prompts
                if (p["type"], p.get("question_id")) not in done_keys
            ]
            logger.info(
                "[traj] %s/%s: %d prompts (%d pending, %d already done)",
                scene_id, sample_id, len(traj_prompts), len(pending), len(existing_results),
            )

            sample_out_dir.mkdir(parents=True, exist_ok=True)
            new_results: list = []

            for payload in pending:
                try:
                    result_raw = inference.run_single(payload)
                    raw_text   = result_raw.get("text", "") if isinstance(result_raw, dict) else str(result_raw)
                except Exception:
                    logger.error(
                        "[traj] Error on %s/%s [%s q=%s]:\n%s",
                        scene_id, sample_id, payload["type"], payload.get("question_id"),
                        traceback.format_exc(),
                    )
                    raw_text = ""

                result = process_sample(payload, raw_text)
                new_results.append(result)
                logger.info(
                    "[traj] %s/%s [%s q=%s] ADE=%s parse=%s",
                    scene_id, sample_id, result["type"], result["question_id"],
                    f"{result['metrics']['ade']:.4f}" if result["metrics"]["ade"] is not None else "None",
                    result["parse_method"],
                )

            # Write new results (overwrite on force, append on resume)
            if new_results:
                mode = "w" if args.force_trajectory else "a"
                with traj_out_file.open(mode) as f:
                    for r in new_results:
                        f.write(json.dumps(r) + "\n")

            sample_results = existing_results + new_results
            all_traj_results.extend(sample_results)

            # Per-sample trajectory report
            traj_report = aggregate_results(sample_results)
            traj_report_path = sample_out_dir / "trajectory_report.json"
            traj_report_path.write_text(
                json.dumps({"aggregate": traj_report, "samples": sample_results}, indent=2)
            )

        # Run-level trajectory aggregate
        if all_traj_results:
            run_traj_report = aggregate_results(all_traj_results)
            print_report(run_traj_report)
            run_traj_path = run_dir / "trajectory_report.json"
            run_traj_path.write_text(
                json.dumps({"aggregate": run_traj_report, "samples": all_traj_results}, indent=2)
            )
            logger.info("[traj] Report saved: %s", run_traj_path)
        else:
            logger.warning("[traj] No trajectory results produced.")

    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    main()
