"""
Gemma 4 inference script.

Calls the Gemma 4 model via an OpenAI-compatible vision API (served by the
ai/gemma4 Docker image) and writes benchmark outputs.jsonl.

KEY DIFFERENCES FROM ALPAMAYO-BASED MODELS:

  1. API-BASED INFERENCE
     Gemma 4 runs as a separate Docker service (ai/gemma4:<tag>). This
     container only calls its HTTP API — no GPU or model weights needed here.
     load_model() initialises an openai.OpenAI client rather than loading
     weights into memory.

  2. VISION API FORMAT
     Images are loaded, resized (default 448×448), base64-encoded, and sent
     as image_url entries in the OpenAI chat completion request.  Text labels
     ("Front camera (T-1.5s to T=0):") precede each camera group so the
     model knows spatial context.

  3. MULTI-VARIANT SUPPORT
     MODEL_ID is derived from GEMMA4_MODEL_TAG at startup so each of the
     four model sizes (e2b, e4b, 26b-a4b, 31b) writes to a separate output
     directory.  Switch variants by changing GEMMA4_MODEL_TAG in dev.env.

  4. THINKING MODE
     system_prompt.txt ends with <|think|> to enable Gemma 4's chain-of-
     thought reasoning.  postprocess.py strips <think>…</think> blocks via
     the base-class parse_answer() before extracting the final answer.

  5. WEIGHTS PATH CONVENTION
     --weights-path is re-purposed as an API endpoint override for interface
     compatibility with the orchestrator.  Use --api-base-url for clarity.

Run modes:
    python inference.py --bench-dir /workspace/data/causal_drive_bench/causal_nuscenes
    python inference.py --bench-dir ... --mode subset --subset-size 20
    python inference.py --bench-dir ... --mode single --scene nuscenes-scene-0159
    python inference.py --run-dir /workspace/data/cdb_outputs/gemma4_e2b_20260421_120000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_inference import ModelInference
from evaluation.common.dataset import BenchmarkDataset
from evaluation.common.utils.config_reader import CDBConfig
from evaluation.common.utils.exceptions import ModelNotReadyException

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader import (  # noqa: E402
    Gemma4DataLoader,
    FRAME_ORDER,
    CAMERA_ORDER,
    CAMERA_LABELS,
    _PLACEHOLDER_SIZE,
    CONFIG as DATALOADER_CONFIG,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model identity (dynamic — one variant per run directory)
# ---------------------------------------------------------------------------

_MODEL_TAG: str = os.environ.get("GEMMA4_MODEL_TAG", "e2b")
MODEL_ID: str = f"gemma4_{_MODEL_TAG.replace('-', '_')}"

# ---------------------------------------------------------------------------
# Inference config
# ---------------------------------------------------------------------------

CONFIG = {
    # Default API endpoint (from env or docker-compose service name)
    "default_api_base_url": os.environ.get(
        "GEMMA4_API_BASE_URL", "http://gemma4-server:8080/v1"
    ),
    # Model name used in chat completion requests
    "default_model_name": os.environ.get(
        "GEMMA4_API_MODEL_NAME", f"ai/gemma4:{_MODEL_TAG}"
    ),
    # Generation parameters (per Docker Hub recommendation)
    "temperature": 1.0,
    "top_p": 0.95,
    # Image resize before base64 encoding (pixels on the shorter side)
    "default_image_size": int(os.environ.get("IMAGE_SIZE", "448")),
    # Max tokens per answer
    "default_max_tokens": int(os.environ.get("MAX_TOKENS", "512")),
    # JPEG quality for base64 encoding (tradeoff: quality vs request size)
    "jpeg_quality": 85,
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert autonomous driving assistant analyzing causal reasoning in driving scenes.\n"
    "You are shown multi-camera images (front, front-left, front-right, rear) across 4 timesteps "
    "(T-1.5s, T-1.0s, T-0.5s, T=0 current) in camera-major order.\n"
    "For Binary questions respond: 'Answer: Yes' or 'Answer: No' then explain.\n"
    "For Multiple Choice questions respond: 'Answer: A' (or B/C/D) then explain.\n"
    "<|think|>"
)


def _load_system_prompt() -> str:
    if _SYSTEM_PROMPT_PATH.exists():
        text = _SYSTEM_PROMPT_PATH.read_text().strip()
        logger.info("Loaded system prompt from %s", _SYSTEM_PROMPT_PATH)
        return text
    logger.info("system_prompt.txt not found — using built-in default")
    return _DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resize_and_encode(abs_path: Path, size: int, quality: int) -> str:
    """
    Load an image, resize to square side=size, return base64-encoded JPEG string.
    Falls back to a grey placeholder if loading fails.
    """
    from PIL import Image

    try:
        img = Image.open(abs_path).convert("RGB")
    except Exception:
        logger.warning("[encode] Failed to open %s — using placeholder", abs_path)
        img = Image.new("RGB", (size, size), color=(128, 128, 128))

    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _placeholder_b64(size: int, quality: int) -> str:
    """Return a grey placeholder image as a base64 JPEG string."""
    from PIL import Image

    img = Image.new("RGB", (size, size), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def preload_image_cache(
    sample_ids: list,
    dataset,
    cfg: Optional[CDBConfig],
    image_size: int,
    jpeg_quality: int,
    num_workers: int = 64,
) -> dict:
    """
    Pre-encode all dataset images into RAM before inference starts.

    Each unique image is encoded once (PIL open → LANCZOS resize → JPEG → base64)
    using a thread pool, then cached in a dict keyed by (str(abs_path), image_size).
    Subsequent run_single() calls hit the cache (dict lookup) instead of doing disk I/O.

    RAM cost: ~40 KB per image × N unique images (≈2 GB for 53K NuScenes images).
    Time cost: ~2 min with 16 threads for the full dataset.
    """
    from tqdm import tqdm

    # Collect unique image paths across all samples
    unique_paths: set = set()
    for sid in sample_ids:
        try:
            record = dataset.load_record_by_name(sid["scene_id"], sid["sample_id"])
        except Exception:
            continue
        frames = record.get("frames", {})
        for cam in CAMERA_ORDER:
            for frame_key in FRAME_ORDER:
                rel_path = frames.get(frame_key, {}).get(cam)
                if rel_path:
                    abs_path = (
                        cfg.resolve_frame_path(rel_path)
                        if cfg is not None
                        else Path(rel_path)
                    )
                    unique_paths.add((str(abs_path), image_size))

    logger.info("Preloading %d unique images into RAM cache (%d workers)…", len(unique_paths), num_workers)

    def _encode_one(path_and_size):
        abs_path_str, size = path_and_size
        try:
            b64 = _resize_and_encode(Path(abs_path_str), size, jpeg_quality)
        except Exception as exc:
            logger.warning("Preload failed for %s: %s", abs_path_str, exc)
            b64 = _placeholder_b64(size, jpeg_quality)
        return path_and_size, b64

    cache: dict = {}
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        for key, b64 in tqdm(
            exe.map(_encode_one, unique_paths),
            total=len(unique_paths),
            desc="Preloading images",
            unit="img",
        ):
            cache[key] = b64

    logger.info("Image cache ready: %d entries", len(cache))
    return cache


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _build_vision_messages(
    image_paths: List[dict],
    state_summary: str,
    qa_text: str,
    system_prompt: str,
    cfg: Optional[CDBConfig],
    image_size: int,
    jpeg_quality: int,
    image_cache: Optional[dict] = None,
) -> list:
    """
    Build an OpenAI-format message list with camera-grouped images and text context.

    Layout:
      system: <system_prompt>
      user:
        [text] "Front camera (T-1.5s to T=0):"
        [image] cam_front@Tm1p5
        [image] cam_front@Tm1p0
        [image] cam_front@Tm0p5
        [image] cam_front@Tp0p0
        ... (repeat for each camera) ...
        [text] "Scene state:\n{state_summary}\n\n{qa_text}"

    image_cache: shared dict keyed by (str(abs_path), image_size) → b64 string.
    All 12 questions in a sample share the same 16 images; the cache avoids
    re-encoding them from disk on every call.
    """
    # Index image_paths by (time_key, camera_key) for fast lookup
    path_index: dict = {
        (item["time_key"], item["camera_key"]): item["path"]
        for item in image_paths
        if isinstance(item, dict)
    }

    user_content = []

    for cam in CAMERA_ORDER:
        cam_label = CAMERA_LABELS.get(cam, cam)
        user_content.append({
            "type": "text",
            "text": f"{cam_label} views (T-1.5s → T-1.0s → T-0.5s → T=0 current):",
        })

        for frame_key in FRAME_ORDER:
            rel_path = path_index.get((frame_key, cam))

            if rel_path:
                abs_path = (
                    cfg.resolve_frame_path(rel_path)
                    if cfg is not None
                    else Path(rel_path)
                )
                cache_key = (str(abs_path), image_size)
                if image_cache is not None:
                    b64 = image_cache.get(cache_key)
                    if b64 is None:
                        b64 = _resize_and_encode(abs_path, image_size, jpeg_quality)
                        image_cache[cache_key] = b64
                else:
                    b64 = _resize_and_encode(abs_path, image_size, jpeg_quality)
            else:
                logger.warning("[build_messages] Missing %s/%s — inserting placeholder", frame_key, cam)
                b64 = _placeholder_b64(image_size, jpeg_quality)

            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

    # Append structured state context and the question
    user_content.append({
        "type": "text",
        "text": f"Scene state:\n{state_summary}\n\n{qa_text}",
    })

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class Gemma4Inference(ModelInference):
    """
    Inference class for Gemma 4 models via OpenAI-compatible HTTP API.

    load_model() initialises an openai.OpenAI client rather than loading
    weights, because Gemma 4 runs as a separate Docker service.

    run_single() encodes images as base64 JPEG, builds a camera-grouped
    vision message, and calls the chat completions endpoint.
    """

    def __init__(
        self,
        model_cfg: dict,
        device: str = "cpu",  # unused; kept for interface compatibility
        cfg: Optional[CDBConfig] = None,
        model_name: str = "",
        image_size: int = 0,
        max_tokens: int = 0,
    ) -> None:
        super().__init__(model_cfg, device)
        self.cfg = cfg
        self.client = None
        self._model_name = model_name or CONFIG["default_model_name"]
        self._image_size = image_size or CONFIG["default_image_size"]
        self._max_tokens = max_tokens or CONFIG["default_max_tokens"]
        self.system_prompt = _load_system_prompt()
        self._concurrent_requests = max(1, int(os.environ.get("CONCURRENT_REQUESTS", "4")))
        self._stats_lock = threading.Lock()
        # Shared image cache: (str(abs_path), image_size) → base64 string.
        # Populated lazily by run_single(); pre-populated by preload_image_cache().
        self._image_cache: dict = {}
        logger.info(
            "Gemma4Inference | model=%s | image_size=%d | max_tokens=%d | concurrent=%d",
            self._model_name, self._image_size, self._max_tokens, self._concurrent_requests,
        )

    def load_model(self, weights_path: str) -> None:
        """
        Initialise OpenAI client pointing at the Gemma 4 model server.

        weights_path is re-purposed as an optional API base URL override for
        compatibility with the framework's --weights-path argument convention.
        Prefer --api-base-url; this argument is a legacy fallback only.
        """
        import openai

        api_base = weights_path.strip() or CONFIG["default_api_base_url"]
        logger.info("Connecting to Gemma 4 API at %s …", api_base)

        self.client = openai.OpenAI(
            base_url=api_base,
            api_key="none",  # model runner does not require authentication
        )

        # Probe /v1/models to verify server is responding
        try:
            models = self.client.models.list()
            available = [m.id for m in models.data]
            logger.info("Model server reachable. Available models: %s", available)
        except Exception:
            logger.warning(
                "Could not enumerate models from server (may still work):\n%s",
                traceback.format_exc(),
            )

    def run_single(self, prompt_payload: dict) -> dict:
        """
        Run Gemma 4 vision inference for one question.

        Pipeline:
          1. Extract image_paths, state_summary, and qa_text from payload.
          2. Load each image, resize to self._image_size, base64-encode as JPEG.
          3. Build camera-grouped vision messages with text labels.
          4. Call client.chat.completions.create() with temperature=1.0, top_p=0.95.
          5. Return {"text": <raw response string>}.

        Args:
            prompt_payload: one row from prompts.jsonl (framework + model fields)

        Returns:
            {"text": <raw answer string (may include <think>...</think> block)>}
        """
        if self.client is None:
            raise ModelNotReadyException("Call load_model() before run_single()")

        scene_id    = prompt_payload.get("scene_id", "?")
        sample_id   = prompt_payload.get("sample_id", "?")
        question_id = prompt_payload.get("question_id", "?")
        logger.info("[run_single] %s/%s | q=%s", scene_id, sample_id, question_id)

        image_paths   = prompt_payload.get("image_paths", [])
        state_summary = prompt_payload.get("state_summary", "")
        qa_text       = prompt_payload.get("qa_text", prompt_payload.get("question_text", ""))

        messages = _build_vision_messages(
            image_paths=image_paths,
            state_summary=state_summary,
            qa_text=qa_text,
            system_prompt=self.system_prompt,
            cfg=self.cfg,
            image_size=self._image_size,
            jpeg_quality=CONFIG["jpeg_quality"],
            image_cache=self._image_cache,
        )

        logger.info(
            "[run_single] Calling API | model=%s | images=%d",
            self._model_name, sum(1 for c in messages[1]["content"] if c["type"] == "image_url"),
        )

        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            max_tokens=self._max_tokens,
        )

        text = response.choices[0].message.content or ""
        logger.info("[run_single] Response (%d chars): %r…", len(text), text[:120])
        return {"text": text}


    def run_from_jsonl(
        self,
        input_path,
        output_path,
        batch_size: int = 1,
        resume: bool = True,
    ) -> int:
        """
        Override base class to process prompts with concurrent API requests.

        Sends up to CONCURRENT_REQUESTS API calls in parallel via
        ThreadPoolExecutor, overlapping image encoding with model inference
        and improving GPU utilisation through the model runner's continuous
        batching.  Output ordering in the JSONL file follows completion order
        (resume logic is question_id-based, so order doesn't matter).
        """
        from tqdm import tqdm

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            raise FileNotFoundError(f"prompts.jsonl not found: {input_path}")

        # Resume: collect already-processed question_ids
        done: set = set()
        if resume and output_path.exists():
            with open(output_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        qid = rec.get("question_id")
                        if qid:
                            done.add(qid)
                    except json.JSONDecodeError:
                        pass
            logger.info("Resume: %d question(s) already processed in %s", len(done), output_path)

        # Load pending prompts
        prompts_to_run: List[dict] = []
        total_in_file = 0
        with open(input_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total_in_file += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line in %s: %s", input_path, exc)
                    continue
                if rec.get("question_id") not in done:
                    prompts_to_run.append(rec)

        skipped = total_in_file - len(done) - len(prompts_to_run)
        logger.info(
            "run_from_jsonl: %d total | %d to run | %d already done | %d malformed",
            total_in_file, len(prompts_to_run), len(done), max(skipped, 0),
        )

        if not prompts_to_run:
            logger.info("Nothing to process — returning.")
            return 0

        wall_start = time.time()
        written = 0

        def _process_one(payload: dict):
            t0 = time.perf_counter()
            try:
                raw_output = self.run_single(payload)
                if isinstance(raw_output, str):
                    raw_output = {"text": raw_output}
                elif not isinstance(raw_output, dict):
                    raw_output = {"text": str(raw_output)}
                if "text" not in raw_output:
                    raw_output["text"] = ""
            except Exception as exc:
                tb = traceback.format_exc()
                qid = payload.get("question_id", "?")
                logger.error("run_single error for question_id=%s:\n%s", qid, tb)
                raw_output = {"text": f"ERROR: {exc}", "traceback": tb}
            elapsed = time.perf_counter() - t0
            with self._stats_lock:
                self._total_inference_time += elapsed
                self._total_inferences += 1
            return payload, raw_output, elapsed

        n_workers = self._concurrent_requests
        logger.info("Concurrent inference: %d worker(s) | %d prompts", n_workers, len(prompts_to_run))

        with open(output_path, "a") as out_fh:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_process_one, p) for p in prompts_to_run]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Inference",
                    unit="q",
                ):
                    payload, raw_output, elapsed = future.result()
                    result = {
                        "scene_id":         payload.get("scene_id", ""),
                        "sample_id":        payload.get("sample_id", ""),
                        "question_id":      payload.get("question_id", ""),
                        "prompt_id":        payload.get("prompt_id", ""),
                        "raw_output":       raw_output,
                        "inference_time_s": round(elapsed, 4),
                        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    out_fh.write(json.dumps(result) + "\n")
                    out_fh.flush()
                    written += 1

        wall_elapsed = time.time() - wall_start
        with self._stats_lock:
            avg = (
                self._total_inference_time / self._total_inferences
                if self._total_inferences else 0.0
            )
        logger.info(
            "run_from_jsonl done: %d written | wall=%.1fs | avg/q=%.3fs | workers=%d",
            written, wall_elapsed, avg, n_workers,
        )
        return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run {MODEL_ID} inference on the CausalDriveBench dataset"
    )
    parser.add_argument(
        "--bench-dir", default="",
        help=(
            "Path to a specific benchmark dataset directory "
            "(e.g. /workspace/data/causal_drive_bench/causal_nuscenes). "
            "Defaults to CDBConfig.get_bench_dir(--dataset) from eval_config.yaml."
        ),
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
        help="Scene name for mode=single (e.g. nuscenes-scene-0159)",
    )
    parser.add_argument(
        "--scenes",
        default=os.environ.get("SCENES") or None,
        help="Comma-separated list of scene IDs (overrides --mode/--scene).",
    )
    parser.add_argument(
        "--run-dir", default=None,
        help="Existing run directory to resume from.",
    )
    parser.add_argument(
        "--output-base", default=None,
        help="Override for cdb_outputs_dir from eval_config.yaml.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("GEMMA4_API_BASE_URL", CONFIG["default_api_base_url"]),
        help="Gemma 4 OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("GEMMA4_API_MODEL_NAME", CONFIG["default_model_name"]),
        help="Model name to pass in API requests.",
    )
    # --weights-path is kept for orchestrator / interface compatibility.
    # When provided it overrides --api-base-url.
    parser.add_argument(
        "--weights-path", default="",
        help="Alias for --api-base-url (legacy orchestrator compatibility).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=int(os.environ.get("IMAGE_SIZE", CONFIG["default_image_size"])),
        help="Resize images to this square pixel size before encoding.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("MAX_TOKENS", CONFIG["default_max_tokens"])),
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Unused for API-based inference (kept for interface compatibility).",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to eval_config.yaml (auto-detected if omitted).",
    )
    parser.add_argument(
        "--trajectory-eval",
        action="store_true",
        default=os.environ.get("TRAJECTORY_EVAL", "").lower() == "true",
        help="Also run trajectory evaluation alongside VQA.",
    )
    parser.add_argument(
        "--traj-only",
        action="store_true",
        default=False,
        help="Skip VQA inference; run trajectory evaluation only.",
    )
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    logger.info("Logging to %s", log_file)


def _has_pending_work(sample_ids: list, run_dir: Path, dataset_name: str) -> bool:
    for sid in sample_ids:
        outputs_file = (
            run_dir / dataset_name / sid["scene_id"] / sid["sample_id"] / "outputs.jsonl"
        )
        if not outputs_file.exists() or outputs_file.stat().st_size == 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg = CDBConfig(config_path=args.config)
    print(cfg.summary())

    bench_dir = Path(args.bench_dir) if args.bench_dir else cfg.get_bench_dir(args.dataset)
    causal_dataset_name = bench_dir.name

    # --weights-path overrides --api-base-url when provided (legacy compat)
    api_base_url = args.weights_path.strip() or args.api_base_url

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
    logger.info(
        "Mode: %s | model: %s | api: %s | image_size: %d | max_tokens: %d",
        args.mode, args.model_name, api_base_url, args.image_size, args.max_tokens,
    )

    dataset = BenchmarkDataset(bench_dir=bench_dir)
    model_cfg: dict = {"qa_types": cfg.qa_types, "batch_size": 1}
    loader = Gemma4DataLoader(dataset=dataset, model_cfg=model_cfg, cfg=cfg)

    if args.scenes:
        scenes_list = [s.strip() for s in args.scenes.split(",") if s.strip()]
        sample_ids = dataset.get_sample_ids(scenes_list=scenes_list)
    else:
        sample_ids = dataset.get_sample_ids(
            mode=args.mode,
            subset_size=args.subset_size,
            scene=args.scene,
        )
    logger.info("%d sample(s) selected (mode=%s)", len(sample_ids), args.mode)

    inference = Gemma4Inference(
        model_cfg=model_cfg,
        device=args.device,
        cfg=cfg,
        model_name=args.model_name,
        image_size=args.image_size,
        max_tokens=args.max_tokens,
    )

    if not _has_pending_work(sample_ids, run_dir, causal_dataset_name):
        logger.info("All samples already processed — nothing to do.")
        return

    try:
        inference.load_model(api_base_url)
    except Exception:
        logger.error("FATAL: could not connect to model server:\n%s", traceback.format_exc())
        raise

    total_prompts = 0
    total_outputs = 0

    do_vqa  = not args.traj_only
    do_traj = args.trajectory_eval or args.traj_only
    logger.info("VQA inference: %s | Trajectory inference: %s", do_vqa, do_traj)

    # ── Optimization 1: pre-generate all prompts.jsonl in parallel ────────
    # save_prompts/save_trajectory_prompts only write text (image paths, not
    # pixels) so they're fast and safe to parallelise.  We generate them all
    # up-front so the inference loop never blocks on I/O between samples.
    _prompt_gen_workers = int(os.environ.get("PROMPT_GEN_WORKERS", "64"))
    logger.info("Pre-generating prompts for %d samples (%d workers)…", len(sample_ids), _prompt_gen_workers)

    def _generate_prompts_for_sid(sid):
        s_dir = run_dir / causal_dataset_name / sid["scene_id"] / sid["sample_id"]
        s_dir.mkdir(parents=True, exist_ok=True)
        if do_vqa:
            pf = s_dir / "prompts.jsonl"
            if not pf.exists():
                try:
                    loader.save_prompts([sid], pf)
                except Exception:
                    logger.error("Prompt gen failed %s/%s:\n%s", sid["scene_id"], sid["sample_id"], traceback.format_exc())
        if do_traj:
            tf = s_dir / "trajectory_prompts.jsonl"
            if not tf.exists():
                try:
                    loader.save_trajectory_prompts([sid], tf)
                except NotImplementedError:
                    pass
                except Exception:
                    logger.error("Traj prompt gen failed %s/%s:\n%s", sid["scene_id"], sid["sample_id"], traceback.format_exc())

    from tqdm import tqdm as _tqdm
    with ThreadPoolExecutor(max_workers=_prompt_gen_workers) as _exe:
        list(_tqdm(
            _exe.map(_generate_prompts_for_sid, sample_ids),
            total=len(sample_ids),
            desc="Generating prompts",
            unit="sample",
        ))
    logger.info("Prompt pre-generation complete.")

    # ── Optimization 2: optionally pre-load all images into RAM ──────────
    # Set PRELOAD_IMAGES=true to encode all dataset images before inference.
    # Cost: ~2 min of wall time.  Benefit: eliminates all disk I/O and PIL
    # work from the inference hot path; every run_single() hits the dict cache.
    if os.environ.get("PRELOAD_IMAGES", "").lower() == "true":
        cache = preload_image_cache(
            sample_ids=sample_ids,
            dataset=dataset,
            cfg=cfg,
            image_size=args.image_size,
            jpeg_quality=CONFIG["jpeg_quality"],
            num_workers=int(os.environ.get("PRELOAD_WORKERS", "16")),
        )
        inference._image_cache = cache
        logger.info("Image preload done: %d images in RAM cache.", len(cache))

    # ── Optimization 3: optionally load persistent on-disk image cache ────
    # Set PERSISTENT_IMAGE_CACHE=<cache_dir> to load a pre-built cache from
    # disk (built once by build_image_cache.py).  Loads in seconds vs ~60 s
    # for PRELOAD_IMAGES=true.  Falls back to PRELOAD_IMAGES if not found.
    # Only used when PRELOAD_IMAGES is not already active.
    _persistent_cache_dir = os.environ.get("PERSISTENT_IMAGE_CACHE", "")
    if _persistent_cache_dir and not os.environ.get("PRELOAD_IMAGES", "").lower() == "true":
        from evaluation.common.image_cache import PersistentImageCache
        _pcache = PersistentImageCache()
        _pcd = Path(_persistent_cache_dir)
        _dataset_name = causal_dataset_name  # e.g. "causal_nuscenes"
        if _pcache.cache_exists(_pcd, _dataset_name, args.image_size):
            logger.info("Loading persistent image cache from %s …", _persistent_cache_dir)
            _t_cache = __import__("time").monotonic()
            _data_root = Path(args.bench_dir).parent
            inference._image_cache = _pcache.load(
                cache_dir=_pcd,
                dataset_name=_dataset_name,
                image_size=args.image_size,
                data_root=_data_root,
                num_workers=int(os.environ.get("PRELOAD_WORKERS", "8")),
            )
            logger.info(
                "Persistent cache loaded: %d entries in %.2f s.",
                len(inference._image_cache),
                __import__("time").monotonic() - _t_cache,
            )
        else:
            logger.warning(
                "Persistent cache not found at %s for dataset '%s' @ %dpx — "
                "falling back to on-demand image loading (set PRELOAD_IMAGES=true "
                "or run build_image_cache.py to build it).",
                _persistent_cache_dir, _dataset_name, args.image_size,
            )

    # ── Inference loop ────────────────────────────────────────────────────
    # SAMPLE_CONCURRENCY=K processes K samples in parallel, putting
    # K × (VQA + traj) questions into the server queue simultaneously.
    # This is the primary lever for filling --parallel N server slots:
    #   slots needed ≈ SAMPLE_CONCURRENCY × avg_questions_per_sample
    # Example: SAMPLE_CONCURRENCY=6, ~10 q/sample → ~60 q in flight → fills --parallel 64
    #          SAMPLE_CONCURRENCY=19, ~10 q/sample → ~190 q in flight → fills --parallel 192
    # Keep CONCURRENT_REQUESTS ≥ SAMPLE_CONCURRENCY × avg_q to avoid per-sample bottlenecks.
    _sample_concurrency = int(os.environ.get("SAMPLE_CONCURRENCY", "1"))
    logger.info(
        "Sample concurrency: %d (SAMPLE_CONCURRENCY) | API workers: %d (CONCURRENT_REQUESTS)",
        _sample_concurrency, inference._concurrent_requests,
    )

    import threading as _threading
    _total_lock = _threading.Lock()

    def _process_sample(sid):
        nonlocal total_prompts, total_outputs
        scene_id  = sid["scene_id"]
        sample_id = sid["sample_id"]
        sample_out_dir = run_dir / causal_dataset_name / scene_id / sample_id
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[main] ── %s/%s ──", scene_id, sample_id)

        prompts_file      = sample_out_dir / "prompts.jsonl"
        outputs_file      = sample_out_dir / "outputs.jsonl"
        traj_prompts_file = sample_out_dir / "trajectory_prompts.jsonl"
        traj_outputs_file = sample_out_dir / "trajectory_outputs.jsonl"

        # Re-generate prompts only if missing (pre-generation may have skipped
        # samples that already had the file from a prior run).
        if do_vqa and not prompts_file.exists():
            try:
                loader.save_prompts([sid], prompts_file)
            except Exception:
                logger.error("ERROR saving prompts for %s/%s:\n%s", scene_id, sample_id, traceback.format_exc())
                return
        if do_traj and not traj_prompts_file.exists():
            try:
                loader.save_trajectory_prompts([sid], traj_prompts_file)
            except NotImplementedError:
                pass
            except Exception:
                logger.error("ERROR saving traj prompts for %s/%s:\n%s", scene_id, sample_id, traceback.format_exc())

        # Count VQA prompts (skip sample if none)
        n_prompts = 0
        if do_vqa:
            try:
                n_prompts = sum(1 for ln in open(prompts_file) if ln.strip())
            except Exception:
                n_prompts = 0
            if n_prompts == 0:
                logger.warning("No prompts for %s/%s — skipping", scene_id, sample_id)
                return

        # ── Optimization 3: run VQA + trajectory concurrently per sample ─
        def _run_vqa():
            try:
                n = inference.run_from_jsonl(
                    input_path=prompts_file,
                    output_path=outputs_file,
                    batch_size=1,
                    resume=True,
                )
                logger.info("[main] VQA done %s/%s: %d written", scene_id, sample_id, n)
                return n
            except Exception:
                logger.error("ERROR VQA inference %s/%s:\n%s", scene_id, sample_id, traceback.format_exc())
                return 0

        def _run_traj():
            if not traj_prompts_file.exists():
                return 0
            try:
                n_traj_prompts = sum(1 for ln in open(traj_prompts_file) if ln.strip())
            except Exception:
                return 0
            if n_traj_prompts == 0:
                return 0
            try:
                inference.run_from_jsonl(
                    input_path=traj_prompts_file,
                    output_path=traj_outputs_file,
                    batch_size=1,
                    resume=True,
                )
                logger.info("[main] Traj done %s/%s", scene_id, sample_id)
                return n_traj_prompts
            except Exception:
                logger.error("ERROR traj inference %s/%s:\n%s", scene_id, sample_id, traceback.format_exc())
                return 0

        n_written = 0
        n_traj_written = 0
        if do_vqa and do_traj:
            with ThreadPoolExecutor(max_workers=2) as _sample_exe:
                _vqa_f  = _sample_exe.submit(_run_vqa)
                _traj_f = _sample_exe.submit(_run_traj)
                n_written = _vqa_f.result()
                n_traj_written = _traj_f.result()
        elif do_vqa:
            n_written = _run_vqa()
        elif do_traj:
            n_traj_written = _run_traj()

        with _total_lock:
            total_outputs += n_written
            total_prompts += n_prompts + n_traj_written

    if _sample_concurrency <= 1:
        for sid in sample_ids:
            _process_sample(sid)
    else:
        with ThreadPoolExecutor(max_workers=_sample_concurrency) as _outer_exe:
            _outer_futures = {_outer_exe.submit(_process_sample, sid): sid for sid in sample_ids}
            for _f in as_completed(_outer_futures):
                try:
                    _f.result()
                except Exception:
                    _sid = _outer_futures[_f]
                    logger.error("Sample worker failed %s/%s:\n%s",
                                 _sid["scene_id"], _sid["sample_id"], traceback.format_exc())

    logger.info("Inference complete.")
    logger.info("  Total prompts : %d", total_prompts)
    logger.info("  Total outputs : %d", total_outputs)
    logger.info("  Timing stats  : %s", inference.timing_stats)
    logger.info("  Outputs under : %s", run_dir)
    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    main()
