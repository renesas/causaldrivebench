"""
OmniDrive inference script.

Loads OmniDrive (EVA-ViT + LLaVA-Llama, mmdet3d framework) and runs inference
on the benchmark's causal reasoning questions.

OmniDrive uses a mmdet3d Petr3D detector with:
  - EVA-ViT backbone (640×640 images, ImageNet normalization)
  - PETRTemporalTransformer detection head
  - TinyLlama-based LLM head for VQA generation

ARCHITECTURE NOTES:
  The model requires a mmdet3d config file and a checkpoint file.
  Image preprocessing: resize to 640×640, normalize with ImageNet stats.
  The detection pipeline produces vision embeddings that are passed to
  the LLM head (lm_head.generate()) for text generation.

CALIBRATION:
  Production inference uses NuScenes lidar2img, intrinsics, extrinsics.
  For VQA evaluation we use identity-matrix dummy calibration which
  may reduce detection quality but allows LLM generation to proceed.

Usage (inside container):
    uv run python inference.py
    uv run python inference.py --mode subset --subset-size 50
    uv run python inference.py --bench-dir /workspace/data/causal_drive_bench/causal_nuscenes \\
                        --model-config /workspace/model/projects/configs/OmniDrive/eva_base_tinyllama.py \\
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

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ---------------------------------------------------------------------------
# Environment variable defaults
# Precedence: CLI flag > env var > hardcoded fallback
# ---------------------------------------------------------------------------
_MODEL_REPO = os.environ.get("MODEL_REPO", "/workspace/OmniDrive")
_MODEL_CONFIG = os.environ.get(
    "MODEL_CONFIG",
    f"{_MODEL_REPO}/projects/configs/OmniDrive/eva_base_tinyllama.py",
)
_MODEL_CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT",
    f"{_MODEL_REPO}/work_dirs/omnidrive.pth",
)
_DEVICE = os.environ.get("DEVICE", "cuda")

from evaluation.common.base_inference import ModelInference
from evaluation.common.dataset import BenchmarkDataset
from evaluation.common.utils.config_reader import CDBConfig
from evaluation.common.utils.exceptions import MissingFrameException, ModelNotReadyException
from dataloader import OmniDriveDataLoader, FRAME_ORDER, CAMERA_ORDER  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "omnidrive"

CONFIG = {
    "model_dtype": torch.float16,
    "max_new_tokens": 320,
    "temperature": 0.1,
    "top_p": 0.75,
    "do_sample": True,
    # EVA-ViT input resolution (from OmniDrive config)
    "image_size": 640,
    # ImageNet normalization (from OmniDrive img_norm_cfg)
    "img_mean": (123.675, 116.28, 103.53),   # BGR-like but applied to RGB
    "img_std": (58.395, 57.12, 57.375),
    "num_cameras": len(CAMERA_ORDER),
    "num_frames": len(FRAME_ORDER),
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert autonomous driving assistant. "
    "You are shown multi-view camera images from a self-driving vehicle and "
    "asked causal reasoning questions about the driving scene. "
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


def _preprocess_image(path_or_none, size: int = 640) -> np.ndarray:
    """
    Load and preprocess one image for OmniDrive's EVA-ViT backbone.

    Returns an (H, W, 3) float32 numpy array (normalized, not uint8).
    """
    from PIL import Image as PILImage
    import numpy as np

    if path_or_none and Path(path_or_none).exists():
        try:
            img = np.array(PILImage.open(path_or_none).convert("RGB"), dtype=np.float32)
        except Exception:
            img = np.full((900, 1600, 3), 128.0, dtype=np.float32)
    else:
        img = np.full((900, 1600, 3), 128.0, dtype=np.float32)

    # Resize to (size, size)
    import cv2
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    # ImageNet normalization: (pixel - mean) / std
    mean = np.array(CONFIG["img_mean"], dtype=np.float32)
    std = np.array(CONFIG["img_std"], dtype=np.float32)
    img = (img - mean) / std

    return img  # (H, W, 3) float32


def _load_calib(scene_id: str, sample_id: str, bench_dir: Path) -> dict:
    """
    Load calib.json for a sample. Returns dict keyed by camera name.
    Falls back to empty dict if file is missing.
    """
    calib_path = bench_dir / scene_id / sample_id / "calib.json"
    if calib_path.exists():
        import json
        return json.loads(calib_path.read_text())
    print(f"[calib] WARN: calib.json not found at {calib_path}, using identity")
    return {}


def _build_camera_matrices(calib: dict, camera_order: list, img_size: int) -> tuple:
    """
    Build intrinsic (3×3), extrinsic (4×4 ego→cam), and lidar2img (4×4)
    numpy arrays for each camera in camera_order.

    calib.json fields per camera:
      intrinsic  : 3×3 camera intrinsic matrix (original image coords)
      R_ego2cam  : 3×3 rotation from ego to camera frame
      t_ego2cam  : 3D translation from ego to camera frame
      image_size : [W, H] of original image

    lidar2img = K_4x4 @ [R | t; 0 0 0 1]
    where K is scaled to the resized image size (img_size × img_size).
    Assumes lidar ≈ ego (no separate lidar-to-ego transform in calib.json).

    Returns:
      intrinsics_list : list of (3, 3) np.float64
      extrinsics_list : list of (4, 4) np.float64
      lidar2img_list  : list of (4, 4) np.float64
    """
    intrinsics_list, extrinsics_list, lidar2img_list = [], [], []

    for cam in camera_order:
        cam_data = calib.get(cam, {})
        if cam_data:
            K = np.array(cam_data["intrinsic"], dtype=np.float64)   # (3, 3)
            R = np.array(cam_data["R_ego2cam"], dtype=np.float64)   # (3, 3)
            t = np.array(cam_data["t_ego2cam"], dtype=np.float64)   # (3,)
            orig_w, orig_h = cam_data.get("image_size", [1600, 900])

            # Scale intrinsics from original resolution to model input size
            scale_x = img_size / orig_w
            scale_y = img_size / orig_h
            K_scaled = K.copy()
            K_scaled[0] *= scale_x   # fx, cx
            K_scaled[1] *= scale_y   # fy, cy

            # Extrinsic: 4×4 ego-to-camera transform
            extrinsic = np.eye(4, dtype=np.float64)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t

            # lidar2img: K_4x4 @ extrinsic  (lidar ≈ ego frame)
            K4 = np.eye(4, dtype=np.float64)
            K4[:3, :3] = K_scaled
            lidar2img = K4 @ extrinsic
        else:
            K_scaled = np.eye(3, dtype=np.float64)
            extrinsic = np.eye(4, dtype=np.float64)
            lidar2img = np.eye(4, dtype=np.float64)

        intrinsics_list.append(K_scaled)
        extrinsics_list.append(extrinsic)
        lidar2img_list.append(lidar2img)

    return intrinsics_list, extrinsics_list, lidar2img_list


def _build_img_metas(
    sample_id: str,
    scene_id: str,
    n_cameras: int,
    img_shape: tuple,
    image_paths: list,
    camera_order: list,
    bench_dir: Path = None,
) -> list:
    """
    Build img_metas for OmniDrive's Petr3D forward pass using real calibration.
    """
    calib = _load_calib(scene_id, sample_id, bench_dir or Path())
    intrinsics_list, extrinsics_list, lidar2img_list = _build_camera_matrices(
        calib, camera_order, img_shape[0]
    )

    identity4 = np.eye(4, dtype=np.float64)
    filenames = [item["path"] for item in image_paths] if image_paths else [f"cam_{i}.jpg" for i in range(n_cameras)]

    meta = {
        "sample_idx": sample_id,
        "filename": filenames,
        "ori_shape": [img_shape] * n_cameras,
        "img_shape": [img_shape] * n_cameras,
        "pad_shape": [img_shape] * n_cameras,
        "scale_factor": 1.0,
        "flip": False,
        "lidar2img": lidar2img_list,
        "intrinsics": intrinsics_list,
        "extrinsics": extrinsics_list,
        "ego_pose": identity4,
        "ego_pose_inv": identity4,
        "timestamp": 0.0,
        "img_timestamp": [0.0] * n_cameras,
        "command": 0,
        "can_bus": np.zeros(18, dtype=np.float32),
        "vlm_labels": None,
        "img_norm_cfg": {"mean": CONFIG["img_mean"], "std": CONFIG["img_std"]},
    }
    return [meta]


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class OmniDriveInference(ModelInference):
    """
    Inference class for OmniDrive (EVA-ViT + TinyLlama, mmdet3d) on the
    causal benchmark.

    Loads the model via mmdet3d's build_model + load_checkpoint, then uses
    the model's LLM head (lm_head.generate()) for per-question generation.
    """

    def __init__(self, model_cfg: dict, device: str = "cuda") -> None:
        super().__init__(model_cfg, device)
        self.tokenizer = None
        self.system_prompt = _load_system_prompt()
        self._device_type = "cuda" if "cuda" in device else "cpu"
        self._model_config_path = model_cfg.get("model_config_path", "")
        print(
            f"[OmniDriveInference] Initialized | device={device} | dtype={CONFIG['model_dtype']}\n"
            f"  model_config={self._model_config_path}"
        )

    def load_model(self, weights_path: str) -> None:
        """
        Load OmniDrive via mmdet3d's build_model + load_checkpoint.

        Requires:
          - mmcv, mmdet, mmdet3d installed (see setup.sh)
          - OmniDrive plugin in projects/ on sys.path (MODEL_REPO)
          - A config file path set via model_cfg['model_config_path']

        Calls os.chdir(MODEL_REPO) so relative paths in the config
        (e.g. ``ckpts/tiny_llama/``, ``ckpts/eva02_petr_proj.pth``) resolve.
        """
        print(f"[load_model] Loading OmniDrive from {weights_path} …")

        omnidrive_root = Path(self.model_cfg.get("model_repo", _MODEL_REPO))
        if omnidrive_root.exists():
            if str(omnidrive_root) not in sys.path:
                sys.path.insert(0, str(omnidrive_root))
                print(f"[load_model] Added {omnidrive_root} to sys.path")
            os.chdir(omnidrive_root)
            print(f"[load_model] cwd → {omnidrive_root} (needed for relative config paths)")
        else:
            print(f"[load_model] WARN: MODEL_REPO={omnidrive_root} not found")

        try:
            import importlib
            import mmcv
            from mmcv import Config
            from mmcv.runner import load_checkpoint
            from mmdet3d.models import build_model

            if not self._model_config_path:
                raise RuntimeError(
                    "model_config_path is required for OmniDrive. "
                    "Set --model-config to the mmdet3d config file path."
                )

            cfg = Config.fromfile(self._model_config_path)
            cfg.model.pretrained = None
            cfg.model.train_cfg = None

            # Load OmniDrive plugin so Petr3D is registered before build_model
            if hasattr(cfg, "plugin") and cfg.plugin:
                plugin_dir = getattr(cfg, "plugin_dir", "projects/mmdet3d_plugin/")
                _module_path = ".".join(os.path.dirname(plugin_dir).split("/"))
                print(f"[load_model] Importing plugin: {_module_path}")
                importlib.import_module(_module_path)

            self.model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
            load_checkpoint(self.model, weights_path, map_location="cpu")
            self.model = self.model.to(self.device)
            self.model.eval()

            # Load tokenizer from the LLM head (TinyLlama)
            if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
                self.tokenizer = self.model.tokenizer
                print(f"[load_model] Tokenizer loaded: {type(self.tokenizer).__name__}")
            else:
                print("[load_model] WARN: model.tokenizer not available")

            print(f"[load_model] OmniDrive loaded on {self.device}")
        except Exception:
            print(f"[load_model] ERROR loading model:\n{traceback.format_exc()}")
            raise

    def run_single(self, prompt_payload: dict) -> dict:
        """
        Run inference for one QA prompt and return the raw response dict.

        Pipeline:
          1. Reload images from list → preprocess to (N_frames×N_cams, 3, 640, 640)
          2. Run extract_img_feat() → EVA-ViT features
          3. Run detection heads → vision embeddings (obj + map queries)
          4. Tokenize QA question → lm_head.generate()
          5. Decode generated token IDs → response text
        """
        if self.model is None:
            raise ModelNotReadyException("Call load_model() before run_single()")

        scene_id = prompt_payload.get("scene_id", "")
        sample_id = prompt_payload.get("sample_id", "?")
        question_id = prompt_payload.get("question_id", "?")
        print(f"\n[run_single] {scene_id}/{sample_id} | q={question_id}")

        bench_dir: Path = Path(prompt_payload.get("bench_dir", ""))
        image_paths: list = prompt_payload.get("image_paths", [])
        question_text: str = prompt_payload.get("question_text", "")
        prompt_context: str = prompt_payload.get("prompt_context", "")
        full_text = f"{prompt_context}\n\n{question_text}".strip() if prompt_context else question_text

        # Build lookup from list
        lookup = {(item["time_key"], item["camera_key"]): item["path"] for item in image_paths}

        # ------------------------------------------------------------------
        # Step 1: Preprocess images → (1, N, 3, H, W) tensor
        # ------------------------------------------------------------------
        imgs: List[np.ndarray] = []
        missing = 0

        for frame_key in FRAME_ORDER:
            for cam in CAMERA_ORDER:
                path = lookup.get((frame_key, cam))
                if path and not Path(path).exists():
                    try:
                        raise MissingFrameException(f"Image file not found: {path}")
                    except MissingFrameException as exc:
                        print(f"[run_single] WARN: {exc}")
                    missing += 1
                elif not path:
                    missing += 1
                img_np = _preprocess_image(path, size=CONFIG["image_size"])
                imgs.append(img_np)

        n_total = len(imgs)
        print(f"[run_single] {n_total - missing}/{n_total} images ({missing} placeholder(s))")

        # Stack: (N, H, W, 3) → (N, 3, H, W) → (1, N, 3, H, W)
        img_arr = np.stack(imgs, axis=0)  # (N, H, W, 3)
        img_tensor = torch.from_numpy(img_arr).permute(0, 3, 1, 2).unsqueeze(0)
        img_tensor = img_tensor.to(device=self.device, dtype=CONFIG["model_dtype"])

        img_shape = (CONFIG["image_size"], CONFIG["image_size"], 3)
        img_metas = _build_img_metas(
            sample_id=sample_id,
            scene_id=scene_id,
            n_cameras=n_total,
            img_shape=img_shape,
            image_paths=image_paths,
            camera_order=CAMERA_ORDER,
            bench_dir=bench_dir,
        )

        # ------------------------------------------------------------------
        # Step 2–3: Extract features and vision embeddings
        # ------------------------------------------------------------------
        with torch.no_grad():
            with torch.autocast(self._device_type, dtype=CONFIG["model_dtype"]):
                try:
                    dtype = CONFIG["model_dtype"]
                    data = {"img": img_tensor}
                    data["img_feats"] = self.model.extract_img_feat(data["img"])

                    # Scalar tensors required by StreamPETR temporal memory
                    data["timestamp"] = torch.zeros(1, dtype=torch.float32, device=self.device)
                    identity4 = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0)
                    data["ego_pose"]     = identity4
                    data["ego_pose_inv"] = identity4
                    data["command"] = torch.zeros(1, dtype=torch.float32, device=self.device)
                    # can_bus concatenated with command → 14-dim vector per sample
                    data["can_bus"] = torch.zeros(1, 13, dtype=torch.float32, device=self.device)

                    # Build calibration tensors from real calib.json data
                    # Shape: (1, N, 3, 3) for intrinsics, (1, N, 4, 4) for extrinsics/lidar2img
                    calib = _load_calib(scene_id, sample_id, bench_dir)
                    intrinsics_list, extrinsics_list, lidar2img_list = _build_camera_matrices(
                        calib, CAMERA_ORDER, CONFIG["image_size"]
                    )
                    # Replicate per-timestep: calib is shared across all 4 frames
                    n_frames = CONFIG["num_frames"]
                    n_cams = CONFIG["num_cameras"]
                    intr_np = np.stack(intrinsics_list * n_frames, axis=0)   # (N_total, 3, 3)
                    extr_np = np.stack(extrinsics_list * n_frames, axis=0)   # (N_total, 4, 4)
                    l2i_np  = np.stack(lidar2img_list  * n_frames, axis=0)   # (N_total, 4, 4)
                    data["intrinsics"] = torch.from_numpy(intr_np).to(dtype=dtype, device=self.device).unsqueeze(0)
                    data["extrinsics"] = torch.from_numpy(extr_np).to(dtype=dtype, device=self.device).unsqueeze(0)
                    # lidar2img must be float32: linalg.inv (used in position_embeding) does not support half
                    data["lidar2img"]  = torch.from_numpy(l2i_np).to(dtype=torch.float32, device=self.device).unsqueeze(0)

                    # Run detection heads to get query embeddings
                    location = self.model.prepare_location(img_metas, **data)
                    pos_embed = self.model.position_embeding(data, location, img_metas)

                    _, det_query = self.model.pts_bbox_head(img_metas, pos_embed, **data)
                    _, map_query = self.model.map_head(img_metas, pos_embed, **data)
                    vision_embedded = torch.cat([det_query, map_query], dim=1)
                except Exception:
                    print(
                        f"[run_single] WARN: detection pipeline failed, "
                        f"using zero vision embeddings:\n{traceback.format_exc()}"
                    )
                    # Fallback: zero vision embeddings (LLM still runs)
                    vision_embedded = torch.zeros(
                        1, 1, self.model.lm_head.config.hidden_size,
                        device=self.device, dtype=CONFIG["model_dtype"],
                    )

        # ------------------------------------------------------------------
        # Step 4: Tokenize QA and run LLM head
        # ------------------------------------------------------------------
        if self.tokenizer is None:
            return {"text": "[ERROR: tokenizer not loaded]"}

        full_prompt = f"{self.system_prompt}\n\n{full_text}"
        input_ids = self.tokenizer(
            full_prompt, return_tensors="pt", truncation=True, max_length=512
        ).input_ids.to(self.device)

        print(
            f"[run_single] Running lm_head.generate() | "
            f"max_new_tokens={CONFIG['max_new_tokens']}"
        )
        with torch.no_grad():
            with torch.autocast(self._device_type, dtype=CONFIG["model_dtype"]):
                output_ids = self.model.lm_head.generate(
                    inputs=input_ids,
                    images=vision_embedded,
                    do_sample=CONFIG["do_sample"],
                    temperature=CONFIG["temperature"],
                    top_p=CONFIG["top_p"],
                    num_beams=1,
                    max_new_tokens=CONFIG["max_new_tokens"],
                    use_cache=True,
                )

        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # Strip the prompt prefix from decoded output
        if full_prompt in response:
            response = response[len(full_prompt):].strip()
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
        "--model-repo",
        default=_MODEL_REPO,
        help="Path to the cloned OmniDrive repository (env: MODEL_REPO)",
    )
    parser.add_argument(
        "--model-config",
        default=_MODEL_CONFIG,
        help="Path to OmniDrive mmdet3d config (env: MODEL_CONFIG)",
    )
    parser.add_argument(
        "--weights-path",
        default=_MODEL_CHECKPOINT,
        help="Path to OmniDrive checkpoint .pth (env: MODEL_CHECKPOINT)",
    )
    parser.add_argument(
        "--device",
        default=_DEVICE,
        help="Torch device string (env: DEVICE)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
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
        "qa_types": cfg.qa_types,
        "batch_size": batch_size,
        "model_config_path": args.model_config,
        "model_repo": args.model_repo,
        "bench_dir": str(bench_dir),
        "raw_data_dir": str(cfg.dir_raw_data),
    }
    loader = OmniDriveDataLoader(dataset=dataset, model_cfg=model_cfg)

    sample_ids = dataset.get_sample_ids(
        mode=args.mode,
        subset_size=args.subset_size,
        scene=args.scene,
    )
    logger.info("%d sample(s) selected (mode=%s)", len(sample_ids), args.mode)

    inference = OmniDriveInference(model_cfg=model_cfg, device=args.device)

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
