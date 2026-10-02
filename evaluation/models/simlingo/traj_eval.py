"""
SimLingo trajectory evaluation using the native waypoint prediction head.

SimLingo (InternVL2-1B + LoRA + DrivingAdaptor) predicts 10 waypoints at 0.25s
intervals (t=0.25,0.50,...,2.50s). We compare against GT at t=0.5,1.0,1.5,2.0,2.5s
(indices [1,3,5,7,9] of the 10 predicted points).

Camera: front-camera only at T=0 (matches SimLingo's CARLA training distribution).

Usage:
    conda activate simlingo
    cd evaluation/models/simlingo
    python traj_eval.py --mode single --scene nuscenes-scene-0001
    python traj_eval.py --mode subset --subset-size 10
    python traj_eval.py --mode full
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
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

# ── add simlingo training package to path ─────────────────────────────────────
_SIMLINGO_REPO = Path(__file__).resolve().parent / "simlingo"
sys.path.insert(0, str(_SIMLINGO_REPO))

# redirect HF downloads to /workspace/ckpts
os.environ.setdefault("HF_HOME", "/workspace/ckpts/hf_home")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/ckpts/hf_home/hub")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CKPT_PATH = "/workspace/ckpts/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
BENCH_DIR  = "/workspace/raw_data/causal_nuscenes/nuscenes_full_ice"
NS_ROOT    = "/workspace/raw_data/nuscenes"
OUTPUT_DIR = "/workspace/traj_eval_outputs"

ENCODER_VARIANT = "OpenGVLab/InternVL2-1B"
INTERNVL2_CKPTS = "/workspace/ckpts"   # cache_root_dir for get_custom_chat_template

# Timesteps in our benchmark GT
GT_TIMES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # seconds

# SimLingo predicts 10 wps at 0.25 s each → t=0.25,0.5,...,2.5s
# Indices matching GT (0.5,1.0,1.5,2.0,2.5): [1,3,5,7,9]
SIMLINGO_DT     = 0.25
N_WPS           = 10                          # future_speed_waypoints
GT_MATCH_IDX    = [1, 3, 5, 7, 9]            # 0-based → t=0.5,1.0,1.5,2.0,2.5
GT_MATCH_TIMES  = [0.5, 1.0, 1.5, 2.0, 2.5]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 448
NUM_IMG_TOKENS = 256   # InternVL2-1B at 448×448 without dynamic tiling

NAV_CMD_MAP = {
    "GO STRAIGHT": "Go straight.",
    "TURN LEFT":   "Turn left.",
    "TURN RIGHT":  "Turn right.",
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _build_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class SimLingoTrajModel:
    """Wraps SimLingo's DrivingModel components for standalone trajectory inference."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.tokenizer = None
        self.vision_encoder = None
        self.lm = None
        self.wp_encoder = None
        self.driving_adaptor = None
        self.lang_adaptor = None
        self._transform = _build_transform()
        self._ready = False

    def load(self, ckpt_path: str) -> None:
        from transformers import AutoTokenizer

        from simlingo_training.models.encoder.internvl2_model import LingoInternVLModel
        from simlingo_training.models.language_model.llm import LLM
        from simlingo_training.models.adaptors.adaptors import (
            DrivingAdaptor, LanguageAdaptor, WaypointInputAdaptor,
        )

        logger.info("Loading tokenizer from %s", ENCODER_VARIANT)
        self.tokenizer = AutoTokenizer.from_pretrained(
            ENCODER_VARIANT, trust_remote_code=True, use_fast=False,
        )

        logger.info("Loading vision encoder …")
        self.vision_encoder = LingoInternVLModel(variant=ENCODER_VARIANT)
        self.vision_encoder.processor = self.tokenizer

        logger.info("Loading language model (with LoRA) …")
        self.lm = LLM(
            variant=ENCODER_VARIANT,
            lora=True, lora_r=32, lora_alpha=64, lora_dropout=0.1,
        )

        logger.info("Building adaptors …")
        hidden = self.lm.hidden_size
        self.wp_encoder      = WaypointInputAdaptor(token_size=hidden, hidden_size=256, hidden_size2=512)
        self.driving_adaptor = DrivingAdaptor(hidden, speed_wps_mode="2d", predict_route_as_wps=True)
        self.lang_adaptor    = LanguageAdaptor(self.lm)

        logger.info("Loading SimLingo weights from %s …", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Apply weights to each sub-module using matching key prefixes
        def _load_prefix(module, prefix):
            sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            missing, unexpected = module.load_state_dict(sub, strict=False)
            if missing:
                logger.debug("  [%s] missing keys: %s", prefix, missing[:5])
            return len(sub)

        n = _load_prefix(self.vision_encoder, "vision_model.")
        logger.info("  vision_encoder: %d keys loaded", n)
        n = _load_prefix(self.lm,           "language_model.")
        logger.info("  language_model: %d keys loaded", n)
        n = _load_prefix(self.driving_adaptor, "adaptors.driving.")
        logger.info("  driving_adaptor: %d keys loaded", n)
        n = _load_prefix(self.wp_encoder,    "wp_encoder.")
        logger.info("  wp_encoder: %d keys loaded", n)
        n = _load_prefix(self.lang_adaptor,  "adaptors.language.")
        logger.info("  lang_adaptor: %d keys loaded", n)

        # Move everything to device in bfloat16
        for mod in [self.vision_encoder, self.lm, self.wp_encoder,
                    self.driving_adaptor, self.lang_adaptor]:
            mod.to(self.device, dtype=torch.bfloat16).eval()

        self._ready = True
        logger.info("SimLingo loaded on %s", self.device)

    # ------------------------------------------------------------------

    def _build_lang_label(self, prompt_text: str):
        """Tokenise the inference prompt via InternVL2's chat template."""
        from simlingo_training.utils.internvl2_utils import get_custom_chat_template
        from simlingo_training.utils.custom_types import LanguageLabel

        conversation = [
            {"role": "user",      "content": [{"text": prompt_text}]},
            {"role": "assistant", "content": [{"text": "Waypoints:"}]},
        ]

        # cache_root_dir must be an absolute path (bypasses hydra.to_absolute_path)
        _, q_dict = get_custom_chat_template(
            [conversation], self.tokenizer, ENCODER_VARIANT,
            NUM_IMG_TOKENS, cache_root_dir=INTERNVL2_CKPTS,
        )

        label = LanguageLabel(
            phrase_ids     = q_dict["phrase_ids"].to(self.device),
            phrase_valid   = q_dict["phrase_valid"].to(self.device),
            phrase_mask    = q_dict["phrase_mask"].to(self.device),
            placeholder_values = [],     # no special placeholder tokens (empty → skips lookup)
            language_string    = q_dict["language_string"],
            loss_masking       = q_dict["loss_masking"].to(self.device),
        )
        return label

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, img_path: str, prompt_text: str) -> np.ndarray:
        """
        Run one trajectory prediction.

        Parameters
        ----------
        img_path    : absolute path to front-camera image at T=0
        prompt_text : full language prompt (from build_bte_prompt_text or
                      build_cte_prompt_text) — conditions the LM features that
                      the driving head operates on

        Returns
        -------
        np.ndarray, shape [10, 2], waypoints in ego-centric metres (x=fwd, y=lat).
        """
        assert self._ready, "call load() first"

        # ── 1. Preprocess image → [1,1,1,3,H,W] ─────────────────────────
        img = Image.open(img_path).convert("RGB")
        pv  = self._transform(img).to(device=self.device, dtype=torch.bfloat16)
        camera_images = pv.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,3,448,448]

        # ── 2. Tokenise prompt ────────────────────────────────────────────
        label = self._build_lang_label(prompt_text)

        # ── 3. Language embeddings + image injection ──────────────────────
        ids     = label.phrase_ids           # [1, seq]
        ids_val = label.phrase_valid         # [1, seq]

        # token embeddings via language adaptor embed_tokens
        embeds = self.lang_adaptor.embed_tokens(
            ids.clamp(0, self.lang_adaptor.embed_tokens.num_embeddings - 1)
        )  # [1, seq, hidden]

        # build minimal adaptor_dict expected by replace_placeholder_tokens
        adaptor_dict = {
            "language_inputs":      embeds,
            "language_inputs_mask": ids_val,
            "language__ids":        ids,
            "language__ids_mask":   ids_val,
            # placeholder for inputs / perm (used by replace_placeholder_tokens at end)
            "inputs":      embeds.clone(),
            "inputs_mask": ids_val.clone(),
            "perm":        torch.arange(embeds.size(1), device=self.device).unsqueeze(0),
        }

        adaptor_dict = self.vision_encoder.replace_placeholder_tokens(
            adaptor_dict     = adaptor_dict,
            pixel_values     = camera_images,
            placeholder_values = label.placeholder_values,
            wp_encoder       = self.wp_encoder,
        )

        input_embeds  = adaptor_dict["language_inputs"]   # [1, seq, hidden]
        attention_mask = ids_val                           # [1, seq]

        # ── 4. Greedy-sample language response (chain-of-thought) ─────────
        eos = self.tokenizer.added_tokens_encoder.get(
            "<|im_end|>",
            self.tokenizer.eos_token_id,
        )
        sampled_tokens, full_embeds = self.lm.greedy_sample(
            input_embeds,
            eos_token_id         = eos,
            max_new_tokens       = 60,
            input_embed_matrix   = self.lang_adaptor.embed_tokens.weight,
            logit_matrix         = self.lang_adaptor.lm_head.weight,
            attention_mask       = attention_mask,
        )

        # ── 5. Append driving query tokens → one more LM forward ─────────
        from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel as LL
        dummy_di = _DummyCameraInput(camera_images)

        driving_inputs = self.driving_adaptor(dummy_di)   # {"inputs": [1,N_drv,h], ...}
        drv_q  = driving_inputs["inputs"][0].unsqueeze(0) # [1, N_drv, h]

        concat = torch.cat([full_embeds, drv_q], dim=1)   # [1, seq+N_drv, h]
        features, logits = self.lm.forward(concat)

        n_drv = drv_q.size(1)
        drv_features = features[:, -n_drv:]
        drv_logits   = logits[:, -n_drv:]

        predictions = self.driving_adaptor.get_predictions(drv_features, drv_logits)
        speed_wps = predictions.get("speed_wps")   # [1, 10, 2]

        if speed_wps is None:
            raise RuntimeError("DrivingAdaptor returned no speed_wps")

        wps = speed_wps[0].cpu().float().numpy()    # [10, 2]
        return wps


class _DummyCameraInput:
    """Minimal object with .camera_images for DrivingAdaptor.forward()."""
    def __init__(self, camera_images):
        self.camera_images = camera_images


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _resolve_image(ns_root: str, rel_path: str) -> str:
    """Build absolute path; strips leading 'samples/' from frames.json paths."""
    if rel_path.startswith("samples/"):
        rel_path = rel_path[len("samples/"):]
    return str(Path(ns_root) / rel_path)


def _load_scene_samples(bench_dir: Path, scene_id: str, ns_root: str) -> List[dict]:
    """Return list of dicts with keys: sample_id, img_path, speed, nav_cmd, gt_future."""
    scene_path = bench_dir / scene_id
    samples = []

    for samp_dir in sorted(scene_path.iterdir()):
        if not samp_dir.is_dir():
            continue

        state_path  = samp_dir / "state.json"
        frames_path = samp_dir / "frames.json"
        if not state_path.exists() or not frames_path.exists():
            continue

        state  = json.loads(state_path.read_text())
        frames = json.loads(frames_path.read_text())

        ego       = state.get("ego", {})
        speed     = float(ego.get("speed", 0.0))
        nav_cmd   = state.get("nav_command", "GO STRAIGHT")
        gt_future = ego.get("future", [])
        if not gt_future:
            continue

        raw_frames = frames.get("frames", {})
        cam_path   = (raw_frames.get("Tp0p0") or {}).get("cam_front")
        if not cam_path:
            continue
        img_abs = _resolve_image(ns_root, cam_path)

        gt_wps = np.array([[p["x"], p["y"]] for p in gt_future], dtype=np.float32)

        samples.append({
            "sample_id": samp_dir.name,
            "img_path":  img_abs,
            "speed":     speed,
            "nav_cmd":   nav_cmd,
            "gt_wps":    gt_wps,   # [6, 2] at t=0.5,1,1.5,2,2.5,3s
            "state_json": state,
        })

    return samples


def discover_scenes(bench_dir: Path, mode: str, scene: Optional[str], subset_size: Optional[int]) -> List[str]:
    scenes = sorted(p.name for p in bench_dir.iterdir() if p.is_dir())
    if mode == "single":
        if scene not in scenes:
            raise ValueError(f"Scene {scene!r} not found in {bench_dir}")
        return [scene]
    if mode == "subset":
        return scenes[: subset_size or 5]
    return scenes   # full


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_wps: np.ndarray, gt_wps: np.ndarray) -> dict:
    """
    pred_wps : [10, 2] SimLingo predictions at 0.25 s intervals (t=0.25…2.5s)
    gt_wps   : [6, 2]  GT at t=0.5,1,1.5,2,2.5,3s

    Picks indices [1,3,5,7,9] → t=0.5,1,1.5,2,2.5s, then linearly extrapolates
    the last step (t=2.25→2.5s velocity vector) to produce a t=3.0s estimate,
    giving 6 comparable points for ADE/FDE.
    """
    pred_5 = pred_wps[GT_MATCH_IDX]               # [5, 2]  t=0.5,1,1.5,2,2.5s

    # linear extrapolation: same velocity as the last 0.25 s step, projected 0.5 s
    delta = pred_wps[9] - pred_wps[8]             # velocity over last 0.25 s step
    t3_pred = pred_wps[9] + 2.0 * delta           # +2 steps of 0.25 s = +0.5 s → t=3.0s
    pred_6 = np.vstack([pred_5, t3_pred])          # [6, 2]

    all_times = GT_MATCH_TIMES + [3.0]
    errors = np.linalg.norm(pred_6 - gt_wps[:6], axis=-1)   # [6]
    ade = float(errors.mean())
    fde = float(errors[-1])

    per_t = {str(t): float(e) for t, e in zip(all_times, errors)}

    return {"ade": ade, "fde": fde, "per_t": per_t, "n_pts": len(errors)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SimLingo trajectory evaluation (native driving head)")
    p.add_argument("--mode",         choices=["full", "subset", "single"], default="full")
    p.add_argument("--scene",        default=None)
    p.add_argument("--subset-size",  type=int, default=None)
    p.add_argument("--bench-dir",    default=BENCH_DIR)
    p.add_argument("--nuscenes-root",default=NS_ROOT)
    p.add_argument("--weights-path", default=CKPT_PATH)
    p.add_argument("--output-dir",   default=OUTPUT_DIR)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--resume",       action="store_true",
                   help="Skip scenes where output.json already exists")
    return p.parse_args()


def setup_logging(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, mode="a")],
    )


def main():
    args = parse_args()

    bench_dir = Path(args.bench_dir)
    ns_root   = args.nuscenes_root
    out_dir   = Path(args.output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"simlingo_traj_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir / "traj_eval.log")
    logger.info("Run dir: %s", run_dir)

    scenes = discover_scenes(bench_dir, args.mode, args.scene, args.subset_size)
    logger.info("%d scene(s) to process (mode=%s)", len(scenes), args.mode)

    # Load model once
    model = SimLingoTrajModel(device=args.device)
    model.load(args.weights_path)

    all_results = []

    for s_idx, scene_id in enumerate(scenes):
        logger.info("[%d/%d] %s", s_idx + 1, len(scenes), scene_id)

        samples = _load_scene_samples(bench_dir, scene_id, ns_root)
        if not samples:
            logger.warning("  No samples in %s — skipping", scene_id)
            continue

        for samp in samples:
            sample_id = samp["sample_id"]
            out_path  = run_dir / scene_id / sample_id / "output.json"

            if args.resume and out_path.exists():
                logger.info("  %s/%s already done — skipping", scene_id, sample_id)
                try:
                    r = json.loads(out_path.read_text())
                    all_results.append(r)
                except Exception:
                    pass
                continue

            try:
                from evaluation.common.trajectory_eval import BTE_TIMESTEPS, build_bte_prompt_text
                prompt_text = build_bte_prompt_text(samp["state_json"], BTE_TIMESTEPS)
                pred_wps = model.predict(samp["img_path"], prompt_text)
            except Exception:
                logger.error("  predict failed for %s/%s:\n%s",
                             scene_id, sample_id, traceback.format_exc())
                continue

            metrics = compute_metrics(pred_wps, samp["gt_wps"])

            # build 6-point pred list matching GT times (includes extrapolated t=3.0s)
            pred_5 = pred_wps[GT_MATCH_IDX]
            delta  = pred_wps[9] - pred_wps[8]
            t3_pred = pred_wps[9] + 2.0 * delta
            pred_6 = np.vstack([pred_5, t3_pred])
            all_times = GT_MATCH_TIMES + [3.0]

            result = {
                "scene_id":     scene_id,
                "sample_id":    sample_id,
                "nav_command":  samp["nav_cmd"],
                "pred_waypoints": [
                    {"t": t, "x": float(pred_6[i, 0]), "y": float(pred_6[i, 1])}
                    for i, t in enumerate(all_times)
                ],
                "gt_waypoints": [
                    {"t": GT_TIMES[i], "x": float(samp["gt_wps"][i, 0]), "y": float(samp["gt_wps"][i, 1])}
                    for i in range(len(samp["gt_wps"]))
                ],
                **metrics,
            }

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2))
            all_results.append(result)

            logger.info("  %s/%s | ADE=%.3f  FDE=%.3f",
                        scene_id, sample_id, metrics["ade"], metrics["fde"])

    # ── Aggregate ──────────────────────────────────────────────────────────
    if all_results:
        ades = [r["ade"] for r in all_results]
        fdes = [r["fde"] for r in all_results]
        per_t_keys = list(all_results[0]["per_t"].keys())
        per_t_means = {k: float(np.mean([r["per_t"][k] for r in all_results]))
                       for k in per_t_keys}

        report = {
            "aggregate": {
                "n_total":    len(all_results),
                "mean_ade":   round(float(np.mean(ades)), 4),
                "mean_fde":   round(float(np.mean(fdes)), 4),
                "per_t_mean_error": {k: round(v, 4) for k, v in per_t_means.items()},
            },
            "samples": all_results,
        }
        report_path = run_dir / "traj_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        logger.info("=" * 60)
        logger.info("RESULTS  n=%d", len(all_results))
        logger.info("  Mean ADE : %.4f m", report["aggregate"]["mean_ade"])
        logger.info("  Mean FDE : %.4f m", report["aggregate"]["mean_fde"])
        for t, e in per_t_means.items():
            logger.info("  t=%ss     : %.4f m", t, e)
        logger.info("Report: %s", report_path)
    else:
        logger.warning("No results collected.")

    logger.info("Run dir: %s", run_dir)


if __name__ == "__main__":
    main()
