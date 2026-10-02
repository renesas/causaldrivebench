"""
Alpamayo 1.5 inference script.

Loads the Alpamayo-1.5-10B model and runs inference on the benchmark's causal
reasoning questions using the model's native VQA interface.

DIFFERENCES FROM ALPAMAYO 1.0:

  1. MODEL CLASS
     Uses Alpamayo1_5 (from alpamayo1_5 package) instead of AlpamayoR1.
     Package lives at /workspace/model/src/alpamayo1_5/.

  2. VQA INTERFACE — KEY CHANGE
     Alpamayo 1.5 introduces dedicated VQA support via:
       helper.create_vqa_message(frames, question, camera_indices)
     This wraps the question in <|question_start|>/<|question_end|> special
     tokens and starts generation from <|answer_start|>.

     We call model.generate_text() instead of sample_trajectories_from_data_with_vlm_rollout().
     generate_text() returns extra["answer"] — a clean answer string without
     trajectory overhead. This is more appropriate for our binary/MCQ benchmark.

  3. CAMERA INDICES
     create_vqa_message() accepts camera_indices (shape: N_cameras) so the model
     sees proper camera display names ("Front camera", "Rear camera", etc.) in the
     prompt — matching the format used during training. We map NuScenes camera
     names to Alpamayo 1.5 camera indices using CAMERA_INDICES from dataloader.py.

  4. OUTPUT FIELD
     Extracts extra["answer"][0, 0] instead of extra["cot"][0].
     The answer text is cleaner (no CoC reasoning prefix) and directly parseable.

  5. OUTPUT DIRECTORY STRUCTURE (same as 1.0)
     <dir_cdb_outputs>/<MODEL_ID>_<timestamp>/
     ├── inference.log
     └── causal_nuscenes/
         └── <scene_id>/
             └── <sample_id>/
                 ├── prompts.jsonl
                 └── outputs.jsonl

Run modes:
    python inference.py --bench-dir /workspace/data/causal_drive_bench/causal_nuscenes
    python inference.py --bench-dir ... --mode subset --subset-size 20
    python inference.py --bench-dir ... --mode single --scene nuscenes-scene-0159
    python inference.py --run-dir /workspace/data/cdb_outputs/alpamayo_1_5_20260421_120000

Paths and batch size default to eval_config.yaml via CDBConfig.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.common.base_inference import ModelInference
from evaluation.common.dataset import BenchmarkDataset
from evaluation.common.utils.config_reader import CDBConfig
from evaluation.common.utils.errors import TensorShapeError
from evaluation.common.utils.exceptions import MissingFrameException, ModelNotReadyException

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader import (  # noqa: E402
    Alpamayo15DataLoader,
    FRAME_ORDER,
    CAMERA_ORDER,
    CAMERA_INDICES,
    _PLACEHOLDER_SIZE,
    interpolate_ego_history,
    CONFIG as DATALOADER_CONFIG,
)

logger = logging.getLogger(__name__)

MODEL_ID = "alpamayo_1_5"

CONFIG = {
    "model_dtype": torch.bfloat16,
    "top_p": 0.95,
    "temperature": 0.6,
    "num_samples": 1,
    "max_generation_length": 512,
    "num_history_steps": DATALOADER_CONFIG["num_history_steps"],
    "history_t_start_s": DATALOADER_CONFIG["history_t_start_s"],
    "history_t_end_s": DATALOADER_CONFIG["history_t_end_s"],
    "nuscenes_frame_times_s": DATALOADER_CONFIG["nuscenes_frame_times_s"],
    "placeholder_image_size_wh": _PLACEHOLDER_SIZE,
}

# camera_indices tensor for helper.create_vqa_message() — CAMERA-MAJOR order
# Shape: (N_cameras,) — one index per camera group, not per frame
_CAMERA_INDICES_TENSOR = torch.tensor(
    [CAMERA_INDICES[cam] for cam in CAMERA_ORDER], dtype=torch.long
)

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
    logger.info("system_prompt.txt not found, using default")
    return _DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class Alpamayo15Inference(ModelInference):
    """
    Inference class for the Alpamayo 1.5-10B model.

    Uses the model's native VQA interface (create_vqa_message + generate_text)
    which is designed for answering questions rather than trajectory prediction.
    This is more appropriate for the binary/MCQ benchmark than the CoC rollout
    used in Alpamayo 1.0.

    Key difference from 1.0:
      - create_vqa_message() wraps qa_text in <|question_start|>/<|question_end|>
      - generate_text() extracts extra["answer"] — the clean structured answer
      - camera_indices are passed to annotate images with display names
    """

    def __init__(
        self,
        model_cfg: dict,
        device: str = "cuda",
        cfg: CDBConfig = None,
    ) -> None:
        super().__init__(model_cfg, device)
        self.cfg = cfg
        self.processor = None
        self.helper = None
        self.system_prompt = _load_system_prompt()
        self._device_type = "cuda" if "cuda" in device else "cpu"
        logger.info(
            "Alpamayo15Inference initialized | device=%s | dtype=%s",
            device, CONFIG["model_dtype"],
        )

    def load_model(self, weights_path: str) -> None:
        """
        Load Alpamayo-1.5-10B from a local HuggingFace weights directory.

        Sets self.model, self.helper, self.processor.
        """
        logger.info("Loading Alpamayo1_5 from %s …", weights_path)

        _src = Path("/workspace/model/src")
        if _src.exists() and str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
            logger.info("Added %s to sys.path", _src)

        try:
            from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
            from alpamayo1_5 import helper as alpamayo_helper

            self.helper = alpamayo_helper
            logger.info("alpamayo1_5 imported successfully")

            self.model = Alpamayo1_5.from_pretrained(
                weights_path, dtype=CONFIG["model_dtype"]
            ).to(self.device)
            self.model.eval()
            logger.info("Alpamayo1_5 moved to %s, set to eval mode", self.device)

            self.processor = alpamayo_helper.get_processor(self.model.tokenizer)
            logger.info("Processor loaded: %s", type(self.processor).__name__)

        except Exception:
            logger.error("ERROR loading model:\n%s", traceback.format_exc())
            raise

    def run_single(self, prompt_payload: dict) -> dict:
        """
        Run VQA inference for one question and return the raw output dict.

        Pipeline:
          1. Load images from image_paths list → (16, 3, H, W) torch.Tensor
             in CAMERA-MAJOR order.
          2. Build VQA messages using helper.create_vqa_message() with
             camera_indices for proper camera labeling.
          3. Tokenise with processor.apply_chat_template().
          4. Call model.generate_text() — pure VLM generation, no trajectory.
          5. Extract extra["answer"][0, 0] and return {"text": answer_text}.

        Args:
            prompt_payload: prompt record from prompts.jsonl (framework + model fields)

        Returns:
            dict with "text" key containing the raw answer string.
        """
        if self.model is None:
            raise ModelNotReadyException("Call load_model() before run_single()")

        scene_id = prompt_payload.get("scene_id", "?")
        sample_id = prompt_payload.get("sample_id", "?")
        question_id = prompt_payload.get("question_id", "?")
        logger.info("[run_single] %s/%s | q=%s", scene_id, sample_id, question_id)

        image_paths: List[dict] = prompt_payload.get("image_paths", [])
        qa_text: str = prompt_payload.get("qa_text", "")

        # ------------------------------------------------------------------
        # Step 1: Load images → (16, 3, H, W) torch.Tensor, CAMERA-MAJOR
        # ------------------------------------------------------------------
        from PIL import Image as PILImage

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
                        logger.warning("[run_single] Image not found: %s", rel_path)
                    img_np = np.full(
                        (_PLACEHOLDER_SIZE[1], _PLACEHOLDER_SIZE[0], 3),
                        128, dtype=np.uint8,
                    )
                    missing_images += 1

                frame_tensors.append(torch.from_numpy(img_np).permute(2, 0, 1))

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
        # Step 2: Build VQA messages with camera labels
        # ------------------------------------------------------------------
        # create_vqa_message() wraps qa_text in <|question_start|>/<|question_end|>
        # and starts generation from <|answer_start|>. This matches training format.
        camera_indices = _CAMERA_INDICES_TENSOR.to(self.device)
        try:
            messages = self.helper.create_vqa_message(
                frames=image_frames,
                question=qa_text,
                camera_indices=camera_indices,
            )
            logger.info("[run_single] Messages built via create_vqa_message()")
        except Exception:
            logger.warning(
                "[run_single] create_vqa_message() failed, using minimal fallback:\n%s",
                traceback.format_exc(),
            )
            messages = _build_vqa_fallback_messages(image_frames, qa_text)

        # ------------------------------------------------------------------
        # Step 3: Tokenise
        # ------------------------------------------------------------------
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,   # continue from <|answer_start|>
            return_dict=True,
            return_tensors="pt",
        )

        # ------------------------------------------------------------------
        # Step 4: Generate answer via pure VLM (no trajectory sampling)
        # ------------------------------------------------------------------
        model_inputs = {"tokenized_data": inputs}
        model_inputs = self.helper.to_device(model_inputs, "cuda")

        logger.info(
            "[run_single] Running generate_text | top_p=%s temperature=%s max_gen=%s",
            CONFIG["top_p"], CONFIG["temperature"], CONFIG["max_generation_length"],
        )
        with torch.no_grad():
            with torch.autocast(self._device_type, dtype=CONFIG["model_dtype"]):
                extra = self.model.generate_text(
                    data=model_inputs,
                    top_p=CONFIG["top_p"],
                    temperature=CONFIG["temperature"],
                    num_samples=CONFIG["num_samples"],
                    max_generation_length=CONFIG["max_generation_length"],
                )

        # ------------------------------------------------------------------
        # Step 5: Extract answer text
        # ------------------------------------------------------------------
        # extra["answer"] has shape [B, num_samples]; B=1, num_samples=1
        answer = extra.get("answer", np.array([[""]]))
        try:
            result = str(answer[0, 0])
        except (IndexError, TypeError):
            result = str(answer)

        logger.info("[run_single] Answer (%d chars): %r…", len(result), result[:120])
        return {"text": result}

    def run_single_trajectory(self, prompt_payload: dict) -> dict:
        """
        Run trajectory inference using sample_trajectories_from_data_with_vlm_rollout.

        Uses helper.create_message() (not create_vqa_message) so the model
        receives <|traj_history_start|> tokens needed for trajectory generation.

        For CTE prompts (eval_type starts with "CTE"), also runs VQA via
        generate_text() to capture the counterfactual VQA answer.

        Returns {"text": json_string} where json_string is:
          BTE: '{"waypoints": [{"t": 0.5, "x": ..., "y": ...}, ...]}'
          CTE: '{"answer": "Yes", "waypoints": [{"t": 0.5, "x": ..., "y": ...}, ...]}'
        """
        if self.model is None:
            raise ModelNotReadyException("Call load_model() before run_single_trajectory()")

        scene_id = prompt_payload.get("scene_id", "?")
        sample_id = prompt_payload.get("sample_id", "?")
        question_id = prompt_payload.get("question_id", "?")
        eval_type = prompt_payload.get("eval_type", "BTE")
        is_cte = eval_type.startswith("CTE")

        logger.info(
            "[run_single_trajectory] %s/%s | q=%s | eval_type=%s",
            scene_id, sample_id, question_id, eval_type,
        )

        # ------------------------------------------------------------------
        # Step 1: Load images (same logic as run_single)
        # ------------------------------------------------------------------
        from PIL import Image as PILImage

        image_paths: List[dict] = prompt_payload.get("image_paths", [])
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
                            "[run_single_trajectory] Failed to load %s:\n%s",
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
                            "[run_single_trajectory] Image not found: %s", rel_path
                        )
                    img_np = np.full(
                        (_PLACEHOLDER_SIZE[1], _PLACEHOLDER_SIZE[0], 3),
                        128, dtype=np.uint8,
                    )
                    missing_images += 1

                frame_tensors.append(torch.from_numpy(img_np).permute(2, 0, 1))

        image_frames = torch.stack(frame_tensors, dim=0)  # (N_cams*N_frames, 3, H, W)
        n_total = len(CAMERA_ORDER) * len(FRAME_ORDER)
        logger.info(
            "[run_single_trajectory] Images: shape=%s (%d/%d real, %d placeholder)",
            tuple(image_frames.shape), n_total - missing_images, n_total, missing_images,
        )

        camera_indices = _CAMERA_INDICES_TENSOR.to(self.device)

        # ------------------------------------------------------------------
        # Step 2: Parse and interpolate ego history 4 → 16 steps
        # ------------------------------------------------------------------
        xyz_sparse = np.array(prompt_payload["ego_history_xyz"], dtype=np.float32)  # (4,3)
        rot_sparse = np.array(prompt_payload["ego_history_rot"], dtype=np.float32)  # (4,3,3)

        xyz_dense, rot_dense = interpolate_ego_history(xyz_sparse, rot_sparse)  # (16,3),(16,3,3)

        # Add batch dim (B=1) and group dim (G=1) → (1,1,16,3) / (1,1,16,3,3)
        xyz_t = (
            torch.from_numpy(xyz_dense)
            .unsqueeze(0).unsqueeze(0)
            .to(dtype=CONFIG["model_dtype"], device=self.device)
        )
        rot_t = (
            torch.from_numpy(rot_dense)
            .unsqueeze(0).unsqueeze(0)
            .to(dtype=CONFIG["model_dtype"], device=self.device)
        )
        logger.info(
            "[run_single_trajectory] ego_history_xyz=%s ego_history_rot=%s",
            tuple(xyz_t.shape), tuple(rot_t.shape),
        )

        # ------------------------------------------------------------------
        # Step 3 (CTE only): get VQA answer for the counterfactual question
        # ------------------------------------------------------------------
        vqa_answer: str | None = None
        if is_cte:
            qa_text: str = prompt_payload.get("qa_text", "")
            try:
                vqa_msgs = self.helper.create_vqa_message(
                    frames=image_frames,
                    question=qa_text,
                    camera_indices=camera_indices,
                )
            except Exception:
                logger.warning(
                    "[run_single_trajectory] create_vqa_message() failed; using fallback:\n%s",
                    traceback.format_exc(),
                )
                vqa_msgs = _build_vqa_fallback_messages(image_frames, qa_text)

            vqa_inputs = self.processor.apply_chat_template(
                vqa_msgs,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            vqa_model_in = self.helper.to_device({"tokenized_data": vqa_inputs}, "cuda")

            with torch.no_grad():
                with torch.autocast(self._device_type, dtype=CONFIG["model_dtype"]):
                    vqa_extra = self.model.generate_text(
                        data=vqa_model_in,
                        top_p=CONFIG["top_p"],
                        temperature=CONFIG["temperature"],
                        num_samples=CONFIG["num_samples"],
                        max_generation_length=CONFIG["max_generation_length"],
                    )

            raw_ans = vqa_extra.get("answer", np.array([[""]]))
            try:
                vqa_answer = str(raw_ans[0, 0])
            except (IndexError, TypeError):
                vqa_answer = str(raw_ans)
            vqa_answer = re.sub(r"<\|answer_end\|>.*", "", vqa_answer, flags=re.DOTALL).strip()
            logger.info("[run_single_trajectory] CTE VQA answer: %r", vqa_answer[:120])

        # ------------------------------------------------------------------
        # Step 4: Build trajectory messages via create_message (NOT create_vqa_message)
        # ------------------------------------------------------------------
        traj_msgs = self.helper.create_message(
            frames=image_frames,
            camera_indices=camera_indices,
        )
        logger.info("[run_single_trajectory] Trajectory messages built via create_message()")

        # ------------------------------------------------------------------
        # Step 5: Tokenize
        # ------------------------------------------------------------------
        traj_inputs = self.processor.apply_chat_template(
            traj_msgs,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        # ------------------------------------------------------------------
        # Step 6: Run trajectory sampling
        # ------------------------------------------------------------------
        model_inputs = {
            "tokenized_data": traj_inputs,
            "ego_history_xyz": xyz_t,
            "ego_history_rot": rot_t,
        }
        model_inputs = self.helper.to_device(model_inputs, "cuda")

        logger.info(
            "[run_single_trajectory] sample_trajectories_from_data_with_vlm_rollout | "
            "num_traj_samples=1 top_p=%s temperature=%s",
            CONFIG["top_p"], CONFIG["temperature"],
        )
        # cuSOLVER on this CUDA version does not support BFloat16 for Cholesky
        # operations (raises NotImplementedError in action_space/utils.py for
        # both torch.linalg.cholesky and torch.cholesky_solve).  The kinematic
        # functions have their own @torch.autocast("cuda", bfloat16) decorators
        # so we cannot avoid the cast by changing input dtypes.  Instead,
        # temporarily patch both ops to promote to float32 and cast back.
        _orig_cholesky       = torch.linalg.cholesky
        _orig_cholesky_solve = torch.cholesky_solve

        def _cholesky_f32(A, *args, **kwargs):
            return _orig_cholesky(A.float(), *args, **kwargs).to(A.dtype)

        def _cholesky_solve_f32(B, L, *args, **kwargs):
            return _orig_cholesky_solve(B.float(), L.float(), *args, **kwargs).to(B.dtype)

        torch.linalg.cholesky = _cholesky_f32
        torch.cholesky_solve  = _cholesky_solve_f32
        try:
            with torch.no_grad():
                with torch.autocast(self._device_type, dtype=CONFIG["model_dtype"]):
                    pred_xyz, _pred_rot, _extra = (
                        self.model.sample_trajectories_from_data_with_vlm_rollout(
                            data=model_inputs,
                            top_p=CONFIG["top_p"],
                            temperature=CONFIG["temperature"],
                            num_traj_samples=1,
                            return_extra=True,
                        )
                    )
        finally:
            torch.linalg.cholesky = _orig_cholesky
            torch.cholesky_solve  = _orig_cholesky_solve

        # pred_xyz: [B=1, num_traj_sets=1, num_traj_samples=1, 64, 3]
        waypoints_xyz = pred_xyz[0, 0, 0].cpu().float().numpy()  # (64, 3)
        logger.info(
            "[run_single_trajectory] pred_xyz shape=%s first_wp=%s",
            tuple(waypoints_xyz.shape), waypoints_xyz[0].tolist(),
        )

        # ------------------------------------------------------------------
        # Step 7: Sample at gt_timesteps
        # Model generates at 10 Hz: t[i] = (i + 1) * 0.1 s, i in [0, 63]
        # So for t=0.5s: idx = round(0.5/0.1) - 1 = 4
        # ------------------------------------------------------------------
        gt_timesteps: List[float] = prompt_payload.get("gt_timesteps") or [
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0
        ]
        waypoints = []
        for t_val in gt_timesteps:
            idx = int(round(t_val / 0.1)) - 1
            idx = max(0, min(idx, 63))
            wp = waypoints_xyz[idx]
            waypoints.append({"t": float(t_val), "x": float(wp[0]), "y": float(wp[1])})

        # ------------------------------------------------------------------
        # Step 8: Format output JSON
        # ------------------------------------------------------------------
        if is_cte and vqa_answer is not None:
            output_obj: dict = {"answer": vqa_answer, "waypoints": waypoints}
        else:
            output_obj = {"waypoints": waypoints}

        result_text = json.dumps(output_obj)
        logger.info("[run_single_trajectory] output: %s", result_text[:300])
        return {"text": result_text}

    def run_trajectory_from_jsonl(
        self,
        input_path: Path,
        output_path: Path,
        resume: bool = True,
    ) -> int:
        """
        Process all records in trajectory_prompts.jsonl using run_single_trajectory().

        Mirrors run_from_jsonl() output format so postprocess code works unchanged.
        Returns the number of records written.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompts: List[dict] = []
        with open(input_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        prompts.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Malformed JSONL line in %s — skipped", input_path)

        done: set = set()
        if resume and output_path.exists():
            with open(output_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            done.add(rec.get("question_id", ""))
                        except json.JSONDecodeError:
                            pass
            logger.info(
                "run_trajectory_from_jsonl: resume — %d already done", len(done)
            )

        to_run = [p for p in prompts if p.get("question_id", "") not in done]
        logger.info(
            "run_trajectory_from_jsonl: %d total | %d to run | %d skipped",
            len(prompts), len(to_run), len(done),
        )

        written = 0
        with open(output_path, "a") as out_fh:
            for payload in to_run:
                t0 = time.perf_counter()
                try:
                    raw_output = self.run_single_trajectory(payload)
                except Exception as exc:
                    tb = traceback.format_exc()
                    qid = payload.get("question_id", "?")
                    logger.error(
                        "run_single_trajectory error for question_id=%s:\n%s", qid, tb
                    )
                    raw_output = {"text": f"ERROR: {exc}", "traceback": tb}

                elapsed = time.perf_counter() - t0
                self._total_inference_time += elapsed
                self._total_inferences += 1

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
                logger.info(
                    "run_trajectory_from_jsonl: wrote %s (%.2fs)",
                    payload.get("question_id", "?"), elapsed,
                )

        logger.info("run_trajectory_from_jsonl done: %d written", written)
        return written


# ---------------------------------------------------------------------------
# Fallback message builder
# ---------------------------------------------------------------------------

def _build_vqa_fallback_messages(image_frames: torch.Tensor, qa_text: str) -> list:
    """
    Build a minimal VQA message list without helper.create_vqa_message().

    Does not include camera display names or special trajectory tokens.
    Used only when the alpamayo1_5 helper import fails.
    """
    image_content = [{"type": "image", "image": frame} for frame in image_frames]

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a driving assistant that generates safe and accurate actions."}],
        },
        {
            "role": "user",
            "content": image_content + [
                {
                    "type": "text",
                    "text": (
                        "These are camera views (front, front-left, front-right, back) "
                        "from a self-driving vehicle across 4 consecutive timesteps "
                        "(T-1.5s, T-1.0s, T-0.5s, T=0 current), shown in camera-major order."
                        f"\n<|question_start|>{qa_text}<|question_end|>"
                    ),
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|answer_start|>"}],
        },
    ]


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
        help="Path to Alpamayo 1.5 weights (local HuggingFace directory).",
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
        "--trajectory-eval",
        action="store_true",
        default=os.environ.get("TRAJECTORY_EVAL", "").lower() == "true",
        help=(
            "Also run trajectory evaluation: generates trajectory_prompts.jsonl "
            "and trajectory_outputs.jsonl alongside the standard VQA files. "
            "Adds BTE + CTE_R2/R3 questions for each sample."
        ),
    )
    parser.add_argument(
        "--traj-only",
        action="store_true",
        default=os.environ.get("TRAJ_ONLY", "").lower() == "true",
        help="Skip VQA inference; run trajectory evaluation only.",
    )
    return parser.parse_args()


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

def _has_pending_work(
    sample_ids: list,
    run_dir: Path,
    causal_dataset_name: str,
    check_trajectory: bool = False,
) -> bool:
    for sid in sample_ids:
        sample_dir = run_dir / causal_dataset_name / sid["scene_id"] / sid["sample_id"]
        if check_trajectory:
            traj_file = sample_dir / "trajectory_outputs.jsonl"
            if not traj_file.exists() or traj_file.stat().st_size == 0:
                return True
        else:
            outputs_file = sample_dir / "outputs.jsonl"
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
    loader = Alpamayo15DataLoader(dataset=dataset, model_cfg=model_cfg, cfg=cfg)

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

    inference = Alpamayo15Inference(model_cfg=model_cfg, device=args.device, cfg=cfg)

    do_vqa  = not args.traj_only
    do_traj = args.trajectory_eval or args.traj_only

    if not _has_pending_work(
        sample_ids, run_dir, causal_dataset_name,
        check_trajectory=args.traj_only,
    ):
        logger.info("All samples already processed — nothing to do.")
        return

    try:
        inference.load_model(weights_path)
    except Exception:
        logger.error("FATAL: model load failed:\n%s", traceback.format_exc())
        raise
    logger.info("VQA inference: %s | Trajectory inference: %s", do_vqa, do_traj)

    total_prompts = 0
    total_outputs = 0

    for sid in sample_ids:
        scene_id = sid["scene_id"]
        sample_id = sid["sample_id"]

        sample_out_dir = run_dir / causal_dataset_name / scene_id / sample_id
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[main] ── %s/%s ──", scene_id, sample_id)

        # ----------------------------------------------------------------
        # VQA inference
        # ----------------------------------------------------------------
        if do_vqa:
            prompts_file = sample_out_dir / "prompts.jsonl"
            outputs_file = sample_out_dir / "outputs.jsonl"

            try:
                n_prompts = loader.save_prompts([sid], prompts_file)
            except Exception:
                logger.error(
                    "ERROR saving VQA prompts for %s/%s:\n%s",
                    scene_id, sample_id, traceback.format_exc(),
                )
                n_prompts = 0

            if n_prompts > 0:
                total_prompts += n_prompts
                logger.info(
                    "[main] VQA: %s/%s %d prompts", scene_id, sample_id, n_prompts
                )
                try:
                    n_written = inference.run_from_jsonl(
                        input_path=prompts_file,
                        output_path=outputs_file,
                        batch_size=batch_size,
                        resume=True,
                    )
                    logger.info(
                        "[main] VQA done: %s/%s %d outputs", scene_id, sample_id, n_written
                    )
                    total_outputs += n_written
                except Exception:
                    logger.error(
                        "ERROR during VQA inference for %s/%s:\n%s",
                        scene_id, sample_id, traceback.format_exc(),
                    )
            else:
                logger.warning(
                    "No VQA prompts for %s/%s — skipping VQA", scene_id, sample_id
                )

        # ----------------------------------------------------------------
        # Trajectory inference
        # ----------------------------------------------------------------
        if do_traj:
            traj_prompts_file = sample_out_dir / "trajectory_prompts.jsonl"
            traj_outputs_file = sample_out_dir / "trajectory_outputs.jsonl"

            try:
                n_traj = loader.save_trajectory_prompts([sid], traj_prompts_file)
                logger.info(
                    "[main] Traj prompts: %s/%s %d records", scene_id, sample_id, n_traj
                )
            except NotImplementedError:
                logger.warning(
                    "get_trajectory_prompt not implemented — trajectory eval skipped"
                )
                n_traj = 0
            except Exception:
                logger.error(
                    "ERROR saving trajectory prompts for %s/%s:\n%s",
                    scene_id, sample_id, traceback.format_exc(),
                )
                n_traj = 0

            if n_traj > 0:
                total_prompts += n_traj
                try:
                    n_written = inference.run_trajectory_from_jsonl(
                        input_path=traj_prompts_file,
                        output_path=traj_outputs_file,
                        resume=True,
                    )
                    logger.info(
                        "[main] Traj done: %s/%s %d outputs", scene_id, sample_id, n_written
                    )
                    total_outputs += n_written
                except Exception:
                    logger.error(
                        "ERROR during trajectory inference for %s/%s:\n%s",
                        scene_id, sample_id, traceback.format_exc(),
                    )

    logger.info("Inference complete.")
    logger.info("  Total prompts : %d", total_prompts)
    logger.info("  Total outputs : %d", total_outputs)
    logger.info("  Timing stats  : %s", inference.timing_stats)
    logger.info("  Outputs under : %s", run_dir)
    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    main()
