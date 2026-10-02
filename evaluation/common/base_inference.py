"""
Abstract base class for model inference.

Each model implementation subclasses ModelInference and implements:
  - ``load_model(weights_path)``       — load weights into self.model
  - ``run_single(prompt_payload)``     — run one prompt, return raw output dict

Concrete helpers provided by this base class
--------------------------------------------
  - ``run_from_jsonl(input_path, output_path, batch_size, resume)``
        Reads prompts.jsonl, skips already-processed questions (resume),
        runs inference, writes outputs.jsonl.
  - ``timing_stats`` property
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelInference(ABC):
    """
    Wraps model loading and inference with timing, progress logging, and resume.

    Parameters
    ----------
    model_cfg : dict with model configuration (from eval_config.yaml or inline)
    device    : torch device string, e.g. ``"cuda"`` or ``"cpu"``
    """

    def __init__(self, model_cfg: dict, device: str = "cuda") -> None:
        self.model_cfg = model_cfg
        self.device = device
        self.model = None
        self._total_inference_time: float = 0.0
        self._total_inferences: int = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self, weights_path: str) -> None:
        """
        Load model weights and any associated processors/tokenisers.

        Sets ``self.model`` (and optionally ``self.tokenizer``, ``self.processor``).

        Parameters
        ----------
        weights_path : local path to model weights directory or checkpoint file
        """
        raise NotImplementedError

    @abstractmethod
    def run_single(self, prompt_payload: dict) -> Dict[str, Any]:
        """
        Run inference on one prompt payload and return a JSON-serialisable output dict.

        The returned dict **must** contain at least a ``"text"`` key with the
        model's raw string response.  Additional keys (e.g. ``"waypoints"``) are
        model-specific and are preserved as-is in outputs.jsonl.

        Parameters
        ----------
        prompt_payload : one prompt record dict from prompts.jsonl

        Returns
        -------
        dict, e.g. ``{"text": "Answer: A\\nReasoning: …", "waypoints": []}``
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete: JSONL inference loop
    # ------------------------------------------------------------------

    def run_from_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
        batch_size: int = 8,
        resume: bool = True,
    ) -> int:
        """
        Read prompts from a JSONL file, run inference, write results to JSONL.

        Output schema (one JSON object per line)::

            {
                "scene_id":         "nuscenes-scene-0159",
                "sample_id":        "SAMPLED_0",
                "question_id":      "Q1",
                "prompt_id":        "000001",
                "raw_output":       {"text": "Answer: A\\nReasoning: …"},
                "inference_time_s": 2.34,
                "timestamp":        "2026-04-10T15:00:00Z"
            }

        Parameters
        ----------
        input_path  : path to prompts.jsonl (written by DataLoader.save_prompts())
        output_path : path for outputs.jsonl (appended if *resume* and exists)
        batch_size  : number of prompts per progress-bar step (currently each
                      prompt still calls run_single(); override run_batch() for
                      true batching)
        resume      : if True and output_path exists, skip already-processed
                      question_ids (identified by question_id within the file)

        Returns
        -------
        Number of new outputs written in this call.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            raise FileNotFoundError(f"prompts.jsonl not found: {input_path}")

        # --- Resume: collect already-processed question_ids ---------------
        done: set[str] = set()
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

        # --- Load pending prompts -----------------------------------------
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

        # --- Inference loop -----------------------------------------------
        written = 0
        wall_start = time.time()

        with open(output_path, "a") as out_fh:
            for i in tqdm(
                range(0, len(prompts_to_run), batch_size),
                desc="Inference",
                unit="batch",
            ):
                batch = prompts_to_run[i : i + batch_size]
                for payload in batch:
                    t0 = time.perf_counter()
                    raw_output: Dict[str, Any]

                    try:
                        raw_output = self.run_single(payload)
                        # Ensure the return value is a dict with "text" key
                        if isinstance(raw_output, str):
                            raw_output = {"text": raw_output}
                        elif not isinstance(raw_output, dict):
                            raw_output = {"text": str(raw_output)}
                        if "text" not in raw_output:
                            raw_output["text"] = ""
                    except Exception as exc:
                        tb = traceback.format_exc()
                        qid = payload.get("question_id", "?")
                        logger.error(
                            "run_single error for question_id=%s:\n%s", qid, tb
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
                        "timestamp":        time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    }
                    out_fh.write(json.dumps(result) + "\n")
                    out_fh.flush()
                    written += 1

        wall_elapsed = time.time() - wall_start
        avg = (
            self._total_inference_time / self._total_inferences
            if self._total_inferences else 0.0
        )
        logger.info(
            "run_from_jsonl done: %d written | wall=%.1fs | avg/sample=%.3fs",
            written, wall_elapsed, avg,
        )
        return written

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def timing_stats(self) -> dict:
        """Return summary timing statistics."""
        avg = (
            self._total_inference_time / self._total_inferences
            if self._total_inferences else 0.0
        )
        return {
            "total_inferences": self._total_inferences,
            "total_time_s":     round(self._total_inference_time, 3),
            "avg_time_s":       round(avg, 4),
        }
