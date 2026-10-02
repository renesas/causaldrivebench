"""
Orion inference script.

Loads Orion (EVA-ViT + QFormer + LLaMA-based, mmdet3d framework) and runs
inference on the benchmark's causal reasoning questions with three output modes:

  VQA:           Binary / MCQ answers via LM head generate().
  BTE Trajectory: Ego future trajectory prediction via VAE planning decoder.
  CF Trajectory:  Counterfactual trajectory via planning decoder + CF prompt.

VQA and trajectory share a single visual feature extraction pass (EVA-ViT +
detection + map heads). The planning trajectory uses the <waypoint_ego> special
token to route through the VAE decoder instead of text generation.

CHECKPOINTS:
  Orion.pth        — https://huggingface.co/poleyzdk/Orion
  pretrain_qformer — https://huggingface.co/exiawsh/pretrain_qformer
  (both to /workspace/ckpts; use --pretrain-qformer to override path)

TRAJECTORY OUTPUT:
  bte_waypoints   : [[x0,y0], …, [x5,y5]] cumulative ego positions at
                    t=0.5,1.0,1.5,2.0,2.5,3.0 s in ego T=0 frame.
  cf_waypoints    : same, but generated from a counterfactual prompt.
  If predicted length ≠ GT length the postprocessor linear-extrapolates.

Usage (inside conda env orion from the model directory):
    python inference.py
    python inference.py --mode subset --subset-size 50
    python inference.py --bench-dir /path/to/causal_nuscenes \\
                        --model-config Orion/adzoo/orion/configs/orion_stage3_fp16.py \\
                        --weights-path /workspace/ckpts/Orion.pth \\
                        --pretrain-qformer /workspace/ckpts/pretrain_qformer
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_inference import ModelInference
from evaluation.common.dataset import BenchmarkDataset
from evaluation.common.utils.config_reader import CDBConfig
from evaluation.common.utils.exceptions import MissingFrameException, ModelNotReadyException

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader import OrionDataLoader, FRAME_ORDER, CAMERA_ORDER  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_ID = "orion"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    "model_dtype": torch.float16,
    "max_new_tokens": 320,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": False,
    "image_size": 640,
    "img_mean": (123.675, 116.28, 103.53),
    "img_std": (58.395, 57.12, 57.375),
    "num_cameras": len(CAMERA_ORDER),
    "num_frames": len(FRAME_ORDER),
    # Timesteps produced by the VAE trajectory decoder (seconds from T=0)
    "traj_timesteps_s": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
}

_BTE_TIMESTEPS: List[float] = list(CONFIG["traj_timesteps_s"])

# Planning prompt templates (match Bench2Drive training format)
_PLANNING_USER_MSG = (
    "Based on the above information, please provide a safe, executable, and "
    "reasonable planning trajectory for the ego car."
)
_PLANNING_ASST_MSG = "Here is the planning trajectory <waypoint_ego>"

_CF_PLANNING_USER_TEMPLATE = (
    "Counterfactual scenario: {premise} "
    "Based on this hypothetical scenario, please provide a safe, executable, "
    "and reasonable planning trajectory for the ego car under these different "
    "conditions."
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert autonomous driving assistant. "
    "You are shown multi-view camera images from a self-driving vehicle across "
    "multiple timesteps and asked causal reasoning questions about the driving scene. "
    "For Binary (Yes/No) questions respond with 'Answer: Yes' or 'Answer: No' "
    "followed by a brief reasoning explanation. "
    "For Multiple Choice (A/B/C/D) questions respond with 'Answer: <letter>' "
    "followed by reasoning."
)


def _load_system_prompt() -> str:
    if _SYSTEM_PROMPT_PATH.exists():
        prompt = _SYSTEM_PROMPT_PATH.read_text().strip()
        logger.info("Loaded system prompt from %s", _SYSTEM_PROMPT_PATH)
        return prompt
    logger.info("system_prompt.txt not found, using default")
    return _DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _preprocess_image(path_or_none, size: int = 640) -> np.ndarray:
    """Load and preprocess one image for EVA-ViT backbone (H,W,3) float32."""
    from PIL import Image as PILImage
    import cv2

    if path_or_none and Path(path_or_none).exists():
        try:
            img = np.array(PILImage.open(path_or_none).convert("RGB"), dtype=np.float32)
        except Exception:
            img = np.full((900, 1600, 3), 128.0, dtype=np.float32)
    else:
        img = np.full((900, 1600, 3), 128.0, dtype=np.float32)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mean = np.array(CONFIG["img_mean"], dtype=np.float32)
    std = np.array(CONFIG["img_std"], dtype=np.float32)
    img = (img - mean) / std
    return img  # (H, W, 3) float32


# ---------------------------------------------------------------------------
# CAN bus
# ---------------------------------------------------------------------------

def _nuscenes_root(cfg: "CDBConfig | None") -> Path:
    """Return the NuScenes data root from CDBConfig, or a sensible default."""
    if cfg is not None:
        return cfg.dir_raw_data / "nuscenes"
    return Path("/workspace/raw_data/nuscenes")


def _resolve_image_path(rel_path: str, cfg: "CDBConfig | None") -> Optional[str]:
    """
    Resolve a relative image path from frames.json to an absolute path.

    Tries two formats:
      1. cfg.resolve_frame_path(rel_path)  — standard "raw_data/nuscenes/..." paths
      2. <nuscenes_root> / rel_path        — short "samples/CAM_FRONT/..." paths
    Returns None if neither resolves to an existing file.
    """
    if not rel_path:
        return None
    # Standard resolution via CDBConfig
    if cfg is not None:
        p = cfg.resolve_frame_path(rel_path)
        if p.exists():
            return str(p)
    # Fallback: frames.json stores short paths relative to the nuscenes root
    p2 = _nuscenes_root(cfg) / rel_path
    if p2.exists():
        return str(p2)
    # Last resort: try as-is (absolute or relative to CWD)
    p3 = Path(rel_path)
    if p3.exists():
        return str(p3)
    return None


def _can_bus_dir(cfg: "CDBConfig | None") -> Path:
    """Return the NuScenes can_bus directory."""
    return _nuscenes_root(cfg) / "can_bus"


def _load_can_bus(nuscenes_scene: str, timestamp: int, cfg: "CDBConfig | None" = None) -> np.ndarray:
    """Load 18-element can_bus vector from NuScenes pose file."""
    can_bus = np.zeros(18, dtype=np.float32)
    if not nuscenes_scene:
        return can_bus
    pose_path = _can_bus_dir(cfg) / f"{nuscenes_scene}_pose.json"
    if not pose_path.exists():
        return can_bus
    try:
        import json
        entries = json.loads(pose_path.read_text())
        if not entries:
            return can_bus
        closest = min(entries, key=lambda e: abs(e["utime"] - timestamp))
        pos = closest.get("pos", [0.0, 0.0, 0.0])
        orientation = closest.get("orientation", [1.0, 0.0, 0.0, 0.0])
        vel = closest.get("vel", [0.0, 0.0, 0.0])
        accel = closest.get("accel", [0.0, 0.0, 0.0])
        rotation_rate = closest.get("rotation_rate", [0.0, 0.0, 0.0])
        qw, qx, qy, qz = orientation
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
        half = yaw / 2.0
        rotation = [math.cos(half), 0.0, 0.0, math.sin(half)]
        if yaw < 0:
            yaw += 2.0 * math.pi
        can_bus[0:3] = pos
        can_bus[3:7] = rotation
        can_bus[7:10] = vel
        can_bus[10:13] = accel
        can_bus[13:16] = rotation_rate
        can_bus[16] = yaw
        can_bus[17] = yaw / math.pi * 180.0
    except Exception as exc:
        logger.warning("can_bus load failed for %s: %s", nuscenes_scene, exc)
    return can_bus


# ---------------------------------------------------------------------------
# img_metas builder
# ---------------------------------------------------------------------------

def _build_img_metas(
    sample_id: str,
    n_cameras: int,
    img_shape: tuple,
    calib: dict = None,
    camera_order: list = None,
    can_bus: np.ndarray = None,
    timestamp: float = 0.0,
) -> list:
    """Build img_metas with calibration and can_bus for Orion inference."""
    if calib is None:
        calib = {}
    if camera_order is None:
        camera_order = [f"cam_{i}" for i in range(n_cameras)]
    if can_bus is None:
        can_bus = np.zeros(18, dtype=np.float32)

    identity4 = np.eye(4, dtype=np.float64)
    identity3x3 = np.eye(3, dtype=np.float64)
    h_dst, w_dst = img_shape[0], img_shape[1]

    lidar2imgs, intrinsics, extrinsics = [], [], []
    for cam in camera_order:
        cam_calib = calib.get(cam)
        if cam_calib is None:
            lidar2imgs.append(identity4.copy())
            intrinsics.append(identity3x3.copy())
            extrinsics.append(identity4.copy())
            continue
        K_raw = np.array(cam_calib["intrinsic"], dtype=np.float64)
        orig_w, orig_h = cam_calib.get("image_size", [w_dst, h_dst])
        sx, sy = w_dst / orig_w, h_dst / orig_h
        K = K_raw.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        intrinsics.append(K)
        R = np.array(cam_calib["R_ego2cam"], dtype=np.float64)
        t = np.array(cam_calib["t_ego2cam"], dtype=np.float64)
        E = np.eye(4, dtype=np.float64)
        E[:3, :3] = R
        E[:3, 3] = t
        extrinsics.append(E)
        K_4x4 = np.eye(4, dtype=np.float64)
        K_4x4[:3, :3] = K
        lidar2imgs.append(K_4x4 @ E)

    meta = {
        "sample_idx":    sample_id,
        "scene_token":   sample_id,
        "frame_idx":     0,
        "filename":      [f"cam_{i}.jpg" for i in range(n_cameras)],
        "ori_shape":     [img_shape] * n_cameras,
        "img_shape":     [img_shape] * n_cameras,
        "pad_shape":     [img_shape] * n_cameras,
        "scale_factor":  1.0,
        "flip":          False,
        "lidar2img":     lidar2imgs,
        "intrinsics":    intrinsics,
        "extrinsics":    extrinsics,
        "ego_pose":      identity4,
        "ego_pose_inv":  identity4,
        "timestamp":     timestamp,
        "img_timestamp": [timestamp] * n_cameras,
        "command":       0,
        "can_bus":       can_bus,
        "vlm_labels":    None,
        "img_norm_cfg":  {"mean": CONFIG["img_mean"], "std": CONFIG["img_std"]},
    }
    return [meta]


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def _extrapolate_waypoints(
    waypoints: np.ndarray,
    src_times: List[float],
    dst_times: List[float],
) -> np.ndarray:
    """
    Linear interpolate / extrapolate (N, 2) waypoints from src_times to dst_times.

    Prepends origin (0, 0) at t=0 for extrapolation beyond the predicted range.
    """
    # Prepend ego origin so we can extrapolate backward/forward cleanly
    t_aug = [0.0] + list(src_times)
    xy_aug = np.vstack([np.zeros((1, 2)), waypoints])  # (N+1, 2)

    out = np.zeros((len(dst_times), 2), dtype=np.float64)
    for dim in range(2):
        out[:, dim] = np.interp(dst_times, t_aug, xy_aug[:, dim])
    return out


def _select_best_mode(ego_fut_preds: torch.Tensor) -> torch.Tensor:
    """
    Select the trajectory mode with the highest cumulative forward (x) distance.

    ego_fut_preds: (B, ego_fut_mode, fut_ts, 2) delta displacements
    Returns: (fut_ts, 2) for batch element 0, the best mode's delta sequence.
    """
    # Sum forward displacement (x) across timesteps per mode
    mode_forward = ego_fut_preds[0, :, :, 0].sum(dim=-1)  # (ego_fut_mode,)
    best_mode = mode_forward.argmax().item()
    return ego_fut_preds[0, best_mode]  # (fut_ts, 2) deltas


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class OrionInference(ModelInference):
    """
    Inference class for Orion on the causal driving benchmark.

    Produces:
      - VQA text answer via lm_head.generate()
      - BTE trajectory via VAE planning decoder (always)
      - CF trajectory via VAE planning decoder with counterfactual prompt
        (only when prompt_payload["trajectory_changes"] is True)

    All three share the same visual feature extraction (EVA-ViT + heads).
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
        self.system_prompt = _load_system_prompt()
        self._device_type = "cuda" if "cuda" in device else "cpu"
        self._model_config_path = model_cfg.get("model_config_path", "")
        self._pretrain_qformer_path = model_cfg.get("pretrain_qformer_path", "")
        # Cache vision_embedded per (scene_id, sample_id) — all questions for a
        # sample share identical images so we only run the backbone once per sample.
        self._vision_cache: dict = {}
        self._vision_cache_key: Optional[str] = None
        logger.info(
            "OrionInference initialized | device=%s | dtype=%s | config=%s",
            device, CONFIG["model_dtype"], self._model_config_path,
        )

    def load_model(self, weights_path: str) -> None:
        """
        Load Orion via mmdet3d build_model + load_checkpoint.

        Overrides llm_path in the config to point to pretrain_qformer.
        """
        logger.info("Loading Orion from %s …", weights_path)

        orion_root = Path(__file__).resolve().parent / "Orion"
        if orion_root.exists() and str(orion_root) not in sys.path:
            sys.path.insert(0, str(orion_root))
            logger.info("Added %s to sys.path", orion_root)

        try:
            from mmcv import Config
            from mmcv.utils import load_checkpoint
            from mmcv.models import build_model

            if not self._model_config_path:
                raise RuntimeError(
                    "model_config_path is required for Orion. "
                    "Pass --model-config pointing to orion_stage3_fp16.py."
                )

            cfg = Config.fromfile(self._model_config_path)
            cfg.model.pretrained = None
            cfg.model.train_cfg = None

            # Override QFormer / LLM path so we can specify it at runtime
            if self._pretrain_qformer_path:
                qp = str(Path(self._pretrain_qformer_path))
                cfg.model.tokenizer = qp
                cfg.model.lm_head = qp
                logger.info("Using pretrain_qformer: %s", qp)

            self.model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

            # Mirror Orion's original custom_wrap_fp16_model: enable FP16 on
            # modules that support it, but force map_head/pts_bbox_head to FP32
            # (these are numerically sensitive and produce NaN under fp16).
            if cfg.get("fp16", None) is not None:
                for m in self.model.modules():
                    if hasattr(m, "fp16_enabled"):
                        m.fp16_enabled = True
                for mod_name, enabled in {"map_head": False, "pts_bbox_head": False}.items():
                    sub = getattr(self.model, mod_name, None)
                    if sub is not None:
                        sub.fp16_enabled = enabled
                logger.info("Applied Orion custom_wrap_fp16_model (map_head/pts_bbox_head in fp32)")

            load_checkpoint(self.model, weights_path, map_location="cpu")
            self.model = self.model.to(self.device)
            self.model.eval()

            if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
                self.tokenizer = self.model.tokenizer
                logger.info("Tokenizer loaded: %s", type(self.tokenizer).__name__)
            else:
                logger.warning("model.tokenizer not available — VQA will be skipped")

            logger.info("Orion loaded on %s", self.device)
        except Exception:
            logger.error("ERROR loading model:\n%s", traceback.format_exc())
            raise

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def _build_vqa_input_ids(self, full_text: str) -> Optional[torch.Tensor]:
        """Tokenize a VQA question into LLaVA vicuna_v1 input_ids."""
        if self.tokenizer is None:
            return None
        try:
            from mmcv.datasets.data_utils import conversation as conversation_lib
            from mmcv.datasets.data_utils.data_utils import tokenizer_image_token
            from mmcv.datasets.data_utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

            _critical_q = (
                "Where are the critical objects in the scene and what impact "
                "do they have on the ego vehicle?"
            )
            user_content = f"{DEFAULT_IMAGE_TOKEN}\nYou are driving a car.{_critical_q} Additionally, {full_text}"
            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], user_content)
            conv.append_message(conv.roles[1], None)
            prompt_str = conv.get_prompt()
            return tokenizer_image_token(
                prompt_str, self.tokenizer,
                image_token_index=IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(self.device)
        except Exception:
            logger.warning("_build_vqa_input_ids fallback: %s", traceback.format_exc())
            ids = self.tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=512
            ).input_ids.to(self.device)
            return ids

    def _build_planning_input_ids(self, planning_user_msg: str) -> Optional[torch.Tensor]:
        """
        Tokenize a trajectory planning prompt.

        The assistant message ends with <waypoint_ego> so inference_ego()
        can extract the hidden state at that token position.
        """
        if self.tokenizer is None:
            return None
        try:
            from mmcv.datasets.data_utils import conversation as conversation_lib
            from mmcv.datasets.data_utils.data_utils import tokenizer_image_token
            from mmcv.datasets.data_utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

            user_content = f"{DEFAULT_IMAGE_TOKEN}\nYou are driving a car.{planning_user_msg}"
            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], user_content)
            conv.append_message(conv.roles[1], _PLANNING_ASST_MSG)
            prompt_str = conv.get_prompt()
            return tokenizer_image_token(
                prompt_str, self.tokenizer,
                image_token_index=IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(self.device)
        except Exception:
            logger.warning("_build_planning_input_ids failed: %s", traceback.format_exc())
            return None

    # ------------------------------------------------------------------
    # VQA generation
    # ------------------------------------------------------------------

    def _run_vqa(self, input_ids: torch.Tensor, vision_embedded: torch.Tensor) -> str:
        """Run LM head generate() and decode the response."""
        try:
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
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            logger.debug("VQA response (%d chars): %r…", len(response), response[:100])
            return response
        except Exception:
            logger.warning("_run_vqa failed: %s", traceback.format_exc())
            return ""

    # ------------------------------------------------------------------
    # Trajectory decoding
    # ------------------------------------------------------------------

    def _run_trajectory_decoder(
        self,
        planning_input_ids: torch.Tensor,
        vision_embedded: torch.Tensor,
    ) -> Optional[List[List[float]]]:
        """
        Run the Orion VAE trajectory decoder.

        Returns a list of [x, y] cumulative positions at
        t=0.5,1.0,...,3.0 s in the ego T=0 frame, or None on failure.
        """
        try:
            with torch.no_grad():
                with torch.autocast(self._device_type, dtype=CONFIG["model_dtype"]):
                    # 1. Get ego hidden feature at <waypoint_ego> token position
                    ego_feature = self.model.lm_head.inference_ego(
                        inputs=planning_input_ids,
                        images=vision_embedded,
                        return_ego_feature=True,
                    )
                ego_feature = ego_feature.to(torch.float32)  # (n_tokens, hidden)

                # inference_ego returns all <waypoint_ego> token positions;
                # take the last one if multiple (multi-turn edge case)
                if ego_feature.dim() == 2 and ego_feature.shape[0] > 1:
                    ego_feature = ego_feature[-1:, :]  # (1, hidden)
                elif ego_feature.dim() == 1:
                    ego_feature = ego_feature.unsqueeze(0)  # (1, hidden)

                # 2. Prepare inputs for distribution forward
                current_states = ego_feature.unsqueeze(1)  # (1, 1, hidden)

                # 3. Sample from prior distribution (no future GT at inference)
                sample, _ = self.model.distribution_forward(current_states, None, None)

                # 4. Predict future hidden states via GRU
                states_hs, _ = self.model.future_states_predict(
                    1, sample, current_states, current_states
                )

                # 5. Decode trajectory for each timestep
                fut_ts = getattr(self.model, "fut_ts", 6)
                ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                ego_fut_trajs_list = []
                for i in range(fut_ts):
                    delta = self.model.ego_fut_decoder(ego_query_hs[i]).reshape(
                        1, self.model.ego_fut_mode, 2
                    )
                    ego_fut_trajs_list.append(delta)

                # ego_fut_preds: (1, ego_fut_mode, fut_ts, 2) — delta displacements
                ego_fut_preds = torch.stack(ego_fut_trajs_list, dim=2)
                ego_fut_preds = ego_fut_preds.to(torch.float32)

                # 6. Select best mode (max cumulative forward distance)
                best_deltas = _select_best_mode(ego_fut_preds)  # (fut_ts, 2)
                cumulative = best_deltas.cumsum(dim=0)  # (fut_ts, 2)

            waypoints = cumulative.cpu().numpy().tolist()  # [[x,y], ...]
            logger.debug(
                "BTE trajectory: %d waypoints, final=(%.2f, %.2f)",
                len(waypoints), waypoints[-1][0], waypoints[-1][1],
            )
            return waypoints
        except Exception:
            logger.warning("_run_trajectory_decoder failed: %s", traceback.format_exc())
            return None

    # ------------------------------------------------------------------
    # Visual feature extraction
    # ------------------------------------------------------------------

    def _extract_visual_features(
        self,
        img_tensor: torch.Tensor,
        img_metas: list,
        n_total: int,
        n_cams: int,
        n_frames: int,
        can_bus: np.ndarray,
        utime: int,
        calib: dict,
        camera_order: list,
    ):
        """Run EVA-ViT + detection/map heads, return vision_embedded tensor."""
        img_shape = (CONFIG["image_size"], CONFIG["image_size"], 3)

        # Calibration tensors
        K_list = img_metas[0]["intrinsics"]
        cam_K_4x4 = np.zeros((n_cams, 4, 4), dtype=np.float64)
        for i, K in enumerate(K_list):
            cam_K_4x4[i, :3, :3] = K
            cam_K_4x4[i, 3, 3] = 1.0
        cam_intrinsic_np = np.tile(cam_K_4x4, (n_frames, 1, 1))[np.newaxis].astype(np.float32)
        cam_intrinsic_t = torch.from_numpy(cam_intrinsic_np).to(
            device=self.device, dtype=torch.float32
        )

        l2i_np = np.stack(img_metas[0]["lidar2img"], axis=0).astype(np.float32)
        lidar2img_t = torch.from_numpy(
            np.tile(l2i_np, (n_frames, 1, 1))[np.newaxis]
        ).to(device=self.device, dtype=torch.float32)

        ego_pose = torch.from_numpy(
            img_metas[0]["ego_pose"].astype(np.float32)
        ).unsqueeze(0).to(self.device)
        ego_pose_inv = torch.from_numpy(
            img_metas[0]["ego_pose_inv"].astype(np.float32)
        ).unsqueeze(0).to(self.device)
        can_bus_t = torch.from_numpy(can_bus).unsqueeze(0).to(self.device)
        command_t = torch.zeros(1, dtype=torch.long).to(self.device)
        timestamp_t = torch.tensor([utime / 1e6], dtype=torch.float32).to(self.device)

        data = {
            "img":           img_tensor,
            "cam_intrinsic": cam_intrinsic_t,
            "lidar2img":     lidar2img_t,
            "can_bus":       can_bus_t,
            "command":       command_t,
            "timestamp":     timestamp_t,
            "ego_pose":      ego_pose,
            "ego_pose_inv":  ego_pose_inv,
        }

        # img_backbone is half() (fp16_infer=True). Within the backbone,
        # FlashAttention's @auto_fp16(out_fp32=True) returns FP32, but the
        # following self.proj has FP16 weights — causing a dtype mismatch.
        # Wrap the backbone in autocast(fp16) so PyTorch auto-casts FP32
        # activations back to FP16 at op boundaries to match weights.
        # pts_bbox_head and map_head (FP32 modules) are run OUTSIDE autocast.
        with torch.no_grad():
            img_fp16 = data["img"].to(torch.float16)
            with torch.autocast(self._device_type, dtype=torch.float16):
                img_feats = self.model.extract_img_feat(img_fp16)
            data["img_feats"] = img_feats.to(torch.float32) if torch.is_tensor(img_feats) else img_feats
            location = self.model.prepare_location(img_metas, **data)
            pos_embed = self.model.position_embeding(data, location, img_metas)
            _, det_query = self.model.pts_bbox_head(img_metas, pos_embed, **data)
            _, map_query = self.model.map_head(img_metas, pos_embed, **data)
            vision_embedded = torch.cat([det_query, map_query], dim=1)

        return vision_embedded

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_single(self, prompt_payload: dict) -> dict:
        """
        Run VQA + BTE trajectory + optional CF trajectory for one question.

        Returns
        -------
        dict with keys:
          text          : VQA answer string
          bte_waypoints : [[x,y], …] cumulative ego trajectory (6 pts, T=0 frame)
          cf_waypoints  : [[x,y], …] counterfactual trajectory (or None)
        """
        if self.model is None:
            raise ModelNotReadyException("Call load_model() before run_single()")

        sample_id    = prompt_payload.get("sample_id", "?")
        question_id  = prompt_payload.get("question_id", "?")
        logger.info("[run_single] %s | q=%s", sample_id, question_id)

        image_paths: list    = prompt_payload.get("image_paths", [])
        question_text: str   = prompt_payload.get("question_text", "")
        prompt_context: str  = prompt_payload.get("prompt_context", "")
        trajectory_changes   = prompt_payload.get("trajectory_changes", False)
        cf_premise: str      = prompt_payload.get("counterfactual_premise", "") or ""
        nuscenes_scene: str  = prompt_payload.get("nuscenes_scene", "")
        utime: int           = int(prompt_payload.get("timestamp", 0))
        calib: dict          = prompt_payload.get("calib") or {}

        full_vqa_text = f"{prompt_context}\n\n{question_text}".strip() if prompt_context else question_text

        # Build image lookup
        lookup = {(item["time_key"], item["camera_key"]): item["path"] for item in image_paths}

        # ------------------------------------------------------------------
        # Step 1: Preprocess images → (1, N, 3, H, W)
        # ------------------------------------------------------------------
        imgs: List[np.ndarray] = []
        missing = 0
        for frame_key in FRAME_ORDER:
            for cam in CAMERA_ORDER:
                path = lookup.get((frame_key, cam))
                abs_path = _resolve_image_path(path, self.cfg) if path else None
                if abs_path is None:
                    missing += 1
                imgs.append(_preprocess_image(abs_path, size=CONFIG["image_size"]))

        n_total = len(imgs)
        n_cams  = len(CAMERA_ORDER)
        n_frames = len(FRAME_ORDER)
        logger.info("[run_single] Images: %d/%d (%d placeholder)", n_total - missing, n_total, missing)

        img_arr = np.stack(imgs, axis=0)  # (N, H, W, 3)
        img_tensor = torch.from_numpy(img_arr).permute(0, 3, 1, 2).unsqueeze(0)
        # Keep input in FP32; auto_fp16 decorators on fp16-enabled modules
        # (e.g. backbone) cast internally. map_head/pts_bbox_head stay FP32.
        img_tensor = img_tensor.to(device=self.device, dtype=torch.float32)

        img_shape = (CONFIG["image_size"], CONFIG["image_size"], 3)
        can_bus   = _load_can_bus(nuscenes_scene, utime, self.cfg)
        img_metas = _build_img_metas(
            sample_id=sample_id.replace("/", "_"),
            n_cameras=n_total,
            img_shape=img_shape,
            calib=calib,
            camera_order=CAMERA_ORDER,
            can_bus=can_bus,
            timestamp=utime / 1e6,
        )

        # ------------------------------------------------------------------
        # Step 2: Extract visual features (cached per sample — all questions
        # for the same scene/sample share identical images so backbone runs once)
        # ------------------------------------------------------------------
        cache_key = f"{prompt_payload.get('scene_id', '')}/{sample_id}"
        if cache_key == self._vision_cache_key and self._vision_cache.get(cache_key) is not None:
            vision_embedded = self._vision_cache[cache_key]
            logger.info("[run_single] Vision cache HIT for %s", cache_key)
        else:
            try:
                vision_embedded = self._extract_visual_features(
                    img_tensor, img_metas, n_total, n_cams, n_frames,
                    can_bus, utime, calib, CAMERA_ORDER,
                )
                # Store in cache; evict previous entry to free GPU memory
                self._vision_cache.clear()
                self._vision_cache[cache_key] = vision_embedded
                self._vision_cache_key = cache_key
                logger.info("[run_single] Vision cache MISS — computed and cached %s", cache_key)
            except Exception:
                logger.error("[run_single] Visual feature extraction failed:\n%s", traceback.format_exc())
                hidden_size = 2048
                try:
                    hidden_size = self.model.lm_head.config.hidden_size
                except Exception:
                    pass
                vision_embedded = torch.zeros(
                    1, 1, hidden_size, device=self.device, dtype=CONFIG["model_dtype"]
                )

        # ------------------------------------------------------------------
        # Step 3: VQA
        # ------------------------------------------------------------------
        vqa_text = ""
        if self.tokenizer is not None:
            vqa_ids = self._build_vqa_input_ids(full_vqa_text)
            if vqa_ids is not None:
                vqa_text = self._run_vqa(vqa_ids, vision_embedded)
        logger.info("[run_single] VQA (%d chars): %r…", len(vqa_text), vqa_text[:80])

        return {"text": vqa_text}

    # ------------------------------------------------------------------
    # Trajectory entry point (BTE / CTE)
    # ------------------------------------------------------------------

    def run_single_trajectory(self, prompt_payload: dict) -> dict:
        """
        Run Orion's planning decoder for one BTE or CTE trajectory prompt.

        prompt_payload is a record from trajectory_prompts.jsonl. Required
        fields (provided by OrionDataLoader.get_trajectory_prompt + base
        iter_trajectory_questions):
          eval_type            — "BTE" or "CTE_*"
          gt_timesteps         — timesteps that must appear in output
          image_paths          — list of {path, time_key, camera_key}
          nuscenes_scene       — for CAN-bus loading
          timestamp            — utime in µs
          calib                — per-cam calibration
          counterfactual_premise — text used in Orion's CF planning template

        Returns a dict whose "text" is a JSON string
            {"waypoints": [{"t": …, "x": …, "y": …}, …]}
        so the framework's parse_waypoints_from_text can consume it.
        """
        if self.model is None:
            raise ModelNotReadyException("Call load_model() before run_single_trajectory()")

        eval_type   = prompt_payload.get("eval_type", "BTE")
        gt_times    = prompt_payload.get("gt_timesteps") or [t for t in _BTE_TIMESTEPS]
        cf_premise  = prompt_payload.get("counterfactual_premise", "") or ""
        sample_id   = prompt_payload.get("sample_id", "?")
        question_id = prompt_payload.get("question_id", "?")

        logger.info(
            "[run_single_trajectory] %s | q=%s | eval_type=%s",
            sample_id, question_id, eval_type,
        )

        # Reuse the visual feature extraction path (cached by scene/sample)
        vision_embedded = self._prepare_vision_embedded(prompt_payload)

        # Build the planning prompt text
        if eval_type == "BTE" or not cf_premise:
            planning_msg = _PLANNING_USER_MSG
        else:
            planning_msg = _CF_PLANNING_USER_TEMPLATE.format(premise=cf_premise)

        waypoints = None
        planning_ids = self._build_planning_input_ids(planning_msg)
        if planning_ids is not None and hasattr(self.model, "distribution_forward"):
            waypoints = self._run_trajectory_decoder(planning_ids, vision_embedded)

        if not waypoints:
            logger.warning(
                "[run_single_trajectory] No waypoints produced for %s/%s",
                sample_id, question_id,
            )
            return {"text": json.dumps({"waypoints": []})}

        # Align prediction count to GT timesteps. Orion always returns 6 pts at
        # _BTE_TIMESTEPS; if GT asks for fewer, take the matching prefix.
        n = min(len(waypoints), len(gt_times))
        out_waypoints = [
            {"t": float(gt_times[i]), "x": float(waypoints[i][0]), "y": float(waypoints[i][1])}
            for i in range(n)
        ]
        return {"text": json.dumps({"waypoints": out_waypoints})}

    # ------------------------------------------------------------------
    # Internal: shared vision feature path used by both VQA and trajectory
    # ------------------------------------------------------------------

    def _prepare_vision_embedded(self, prompt_payload: dict) -> torch.Tensor:
        """
        Run the visual encoder for the given sample, with caching across all
        prompts (VQA + trajectory) belonging to the same scene/sample.
        """
        scene_id   = prompt_payload.get("scene_id", "")
        sample_id  = prompt_payload.get("sample_id", "?")
        cache_key  = f"{scene_id}/{sample_id}"
        if cache_key == self._vision_cache_key and self._vision_cache.get(cache_key) is not None:
            return self._vision_cache[cache_key]

        image_paths    = prompt_payload.get("image_paths", []) or []
        nuscenes_scene = prompt_payload.get("nuscenes_scene", "")
        utime          = int(prompt_payload.get("timestamp", 0))
        calib          = prompt_payload.get("calib") or {}

        lookup = {(item["time_key"], item["camera_key"]): item["path"] for item in image_paths}
        imgs: List[np.ndarray] = []
        missing = 0
        for frame_key in FRAME_ORDER:
            for cam in CAMERA_ORDER:
                path = lookup.get((frame_key, cam))
                abs_path = _resolve_image_path(path, self.cfg) if path else None
                if abs_path is None:
                    missing += 1
                imgs.append(_preprocess_image(abs_path, size=CONFIG["image_size"]))
        n_total = len(imgs)
        n_cams  = len(CAMERA_ORDER)
        n_frames = len(FRAME_ORDER)
        logger.info(
            "[trajectory] Images: %d/%d (%d placeholder)", n_total - missing, n_total, missing
        )

        img_arr = np.stack(imgs, axis=0)
        img_tensor = torch.from_numpy(img_arr).permute(0, 3, 1, 2).unsqueeze(0)
        img_tensor = img_tensor.to(device=self.device, dtype=torch.float32)

        img_shape = (CONFIG["image_size"], CONFIG["image_size"], 3)
        can_bus   = _load_can_bus(nuscenes_scene, utime, self.cfg)
        img_metas = _build_img_metas(
            sample_id=sample_id.replace("/", "_"),
            n_cameras=n_total,
            img_shape=img_shape,
            calib=calib,
            camera_order=CAMERA_ORDER,
            can_bus=can_bus,
            timestamp=utime / 1e6,
        )

        try:
            vision_embedded = self._extract_visual_features(
                img_tensor, img_metas, n_total, n_cams, n_frames,
                can_bus, utime, calib, CAMERA_ORDER,
            )
            self._vision_cache.clear()
            self._vision_cache[cache_key] = vision_embedded
            self._vision_cache_key = cache_key
            logger.info("[trajectory] Vision cache MISS — computed %s", cache_key)
        except Exception:
            logger.error(
                "[trajectory] Visual feature extraction failed:\n%s", traceback.format_exc()
            )
            hidden_size = 2048
            try:
                hidden_size = self.model.lm_head.config.hidden_size
            except Exception:
                pass
            vision_embedded = torch.zeros(
                1, 1, hidden_size, device=self.device, dtype=CONFIG["model_dtype"]
            )
        return vision_embedded

    # ------------------------------------------------------------------
    # Trajectory loop runner (mirrors run_from_jsonl)
    # ------------------------------------------------------------------

    def run_trajectory_from_jsonl(
        self,
        input_path: Path,
        output_path: Path,
        resume: bool = True,
    ) -> int:
        """
        Iterate trajectory_prompts.jsonl and append predictions to
        trajectory_outputs.jsonl. Same on-disk schema as run_from_jsonl.
        """
        import time as _time

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompts: List[dict] = []
        with open(input_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    prompts.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Malformed JSONL line in %s — skipped", input_path)

        done: set = set()
        if resume and output_path.exists():
            with open(output_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        done.add(rec.get("question_id", ""))
                    except json.JSONDecodeError:
                        pass

        to_run = [p for p in prompts if p.get("question_id", "") not in done]
        logger.info(
            "run_trajectory_from_jsonl: %d total | %d to run | %d already done",
            len(prompts), len(to_run), len(done),
        )

        written = 0
        with open(output_path, "a") as out_fh:
            for payload in to_run:
                t0 = _time.perf_counter()
                try:
                    raw_output = self.run_single_trajectory(payload)
                except Exception as exc:
                    tb = traceback.format_exc()
                    qid = payload.get("question_id", "?")
                    logger.error(
                        "run_single_trajectory error for question_id=%s:\n%s", qid, tb
                    )
                    raw_output = {"text": "", "error": str(exc)}
                elapsed = _time.perf_counter() - t0

                rec = {
                    "scene_id":         payload.get("scene_id", ""),
                    "sample_id":        payload.get("sample_id", ""),
                    "question_id":      payload.get("question_id", ""),
                    "prompt_id":        payload.get("prompt_id", ""),
                    "raw_output":       raw_output,
                    "inference_time_s": round(elapsed, 4),
                    "timestamp":        _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                }
                out_fh.write(json.dumps(rec) + "\n")
                out_fh.flush()
                written += 1

        logger.info("run_trajectory_from_jsonl done: %d written", written)
        return written


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


def _has_pending_work(
    sample_ids: list,
    run_dir: Path,
    causal_dataset_name: str,
    loader: "OrionDataLoader" = None,
) -> bool:
    """
    Return True if any sample has unfinished VQA *or* trajectory inference.

    Compares ``question_id``s expected by the loader (prompts.jsonl +
    trajectory_prompts.jsonl) against those already present in the matching
    *_outputs.jsonl. File-existence alone isn't enough — re-running with new
    CTE discovery would otherwise be a no-op when ``trajectory_outputs.jsonl``
    already exists with only BTE.
    """

    def _question_ids_in(path: Path) -> set:
        out: set = set()
        if not path.exists():
            return out
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = rec.get("question_id")
                if qid:
                    out.add(qid)
        return out

    for sid in sample_ids:
        sample_dir = run_dir / causal_dataset_name / sid["scene_id"] / sid["sample_id"]
        outputs_file       = sample_dir / "outputs.jsonl"
        traj_outputs_file  = sample_dir / "trajectory_outputs.jsonl"

        # Trivial first check — nothing on disk → pending.
        if not outputs_file.exists() or outputs_file.stat().st_size == 0:
            return True
        if not traj_outputs_file.exists() or traj_outputs_file.stat().st_size == 0:
            return True

        if loader is None:
            continue

        # Compare expected question_ids vs done question_ids.
        try:
            expected_vqa = {
                rec.get("question_id")
                for rec in loader.iter_questions([sid])
                if rec.get("question_id")
            }
            expected_traj = {
                rec.get("question_id")
                for rec in loader.iter_trajectory_questions([sid])
                if rec.get("question_id")
            }
        except Exception:
            # If discovery fails, conservatively assume work remains.
            return True

        done_vqa  = _question_ids_in(outputs_file)
        done_traj = _question_ids_in(traj_outputs_file)
        if (expected_vqa - done_vqa) or (expected_traj - done_traj):
            return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run {MODEL_ID} inference (VQA + trajectory) on CausalDriveBench"
    )
    parser.add_argument("--bench-dir", default="")
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
    parser.add_argument("--subset-size", type=int,
                        default=int(os.environ.get("SUBSET_SIZE", 0)) or None)
    parser.add_argument("--scene", default=os.environ.get("SCENE") or None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output-base", default=None)
    parser.add_argument(
        "--model-config",
        default=os.environ.get(
            "MODEL_CONFIG",
            str(Path(__file__).parent / "Orion/adzoo/orion/configs/orion_stage3_fp16.py"),
        ),
    )
    parser.add_argument(
        "--weights-path",
        default=os.environ.get("MODEL_CHECKPOINT", "/workspace/ckpts/Orion.pth"),
    )
    parser.add_argument(
        "--pretrain-qformer",
        default=os.environ.get("PRETRAIN_QFORMER_PATH", "/workspace/ckpts/pretrain_qformer"),
        help="Path to pretrain_qformer directory (tokenizer + LM head weights)",
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = CDBConfig(config_path=args.config)
    print(cfg.summary())

    bench_dir = Path(args.bench_dir) if args.bench_dir else cfg.get_bench_dir(args.dataset)
    causal_dataset_name = bench_dir.name

    weights_path = args.weights_path or str(cfg.dir_model_weights)
    batch_size   = args.batch_size if args.batch_size is not None else cfg.batch_size

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
        logger.info("Resuming existing run: %s", run_dir)
    else:
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path(args.output_base) if args.output_base else cfg.dir_cdb_outputs
        run_dir = output_base / f"{MODEL_ID}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir / "inference.log")
    logger.info("Run directory: %s", run_dir)
    logger.info("Bench directory: %s", bench_dir)
    logger.info("Mode: %s | batch_size: %d | device: %s", args.mode, batch_size, args.device)

    dataset = BenchmarkDataset(bench_dir=bench_dir)
    model_cfg: dict = {
        "qa_types":             cfg.qa_types,
        "batch_size":           batch_size,
        "model_config_path":    args.model_config,
        "pretrain_qformer_path": args.pretrain_qformer,
    }
    loader = OrionDataLoader(dataset=dataset, model_cfg=model_cfg, cfg=cfg)

    sample_ids = dataset.get_sample_ids(
        mode=args.mode,
        subset_size=args.subset_size,
        scene=args.scene,
    )
    logger.info("%d sample(s) selected (mode=%s)", len(sample_ids), args.mode)

    inference = OrionInference(model_cfg=model_cfg, device=args.device, cfg=cfg)

    if not _has_pending_work(sample_ids, run_dir, causal_dataset_name, loader=loader):
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
        scene_id  = sid["scene_id"]
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
            logger.warning("No prompts for %s/%s — skipping", scene_id, sample_id)
            continue

        total_prompts += n_prompts

        try:
            n_written = inference.run_from_jsonl(
                input_path=prompts_file,
                output_path=outputs_file,
                batch_size=batch_size,
                resume=True,
            )
            logger.info("[main] %s/%s: %d outputs", scene_id, sample_id, n_written)
            total_outputs += n_written
        except Exception:
            logger.error(
                "ERROR during inference for %s/%s:\n%s",
                scene_id, sample_id, traceback.format_exc(),
            )
            continue

        # ----------------------------------------------------------------
        # Trajectory inference (BTE + CTE) → trajectory_prompts/outputs.jsonl
        # ----------------------------------------------------------------
        traj_prompts_file = sample_out_dir / "trajectory_prompts.jsonl"
        traj_outputs_file = sample_out_dir / "trajectory_outputs.jsonl"
        try:
            n_traj_prompts = loader.save_trajectory_prompts(
                [sid], traj_prompts_file
            )
            logger.info(
                "[main] Traj prompts %s/%s: %d records",
                scene_id, sample_id, n_traj_prompts,
            )
        except NotImplementedError:
            n_traj_prompts = 0
        except Exception:
            logger.error(
                "ERROR saving trajectory prompts for %s/%s:\n%s",
                scene_id, sample_id, traceback.format_exc(),
            )
            n_traj_prompts = 0

        if n_traj_prompts > 0:
            total_prompts += n_traj_prompts
            try:
                n_traj_written = inference.run_trajectory_from_jsonl(
                    input_path=traj_prompts_file,
                    output_path=traj_outputs_file,
                    resume=True,
                )
                logger.info(
                    "[main] Traj outputs %s/%s: %d records",
                    scene_id, sample_id, n_traj_written,
                )
                total_outputs += n_traj_written
            except Exception:
                logger.error(
                    "ERROR during trajectory inference for %s/%s:\n%s",
                    scene_id, sample_id, traceback.format_exc(),
                )

    logger.info("Inference complete. prompts=%d outputs=%d", total_prompts, total_outputs)
    logger.info("Timing: %s", inference.timing_stats)
    logger.info("Run dir: %s", run_dir)
    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    main()
