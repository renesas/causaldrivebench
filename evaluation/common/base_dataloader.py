"""
Abstract base class for model-specific data loaders.

Each model implementation subclasses ModelDataLoader and implements:

  - ``build_image_paths(record)``
        Return the ordered list of image path metadata dicts the model needs.
        Each dict: ``{"path": relative_str, "time_key": str, "camera_key": str}``.
        This is stored verbatim in prompts.jsonl and drives image loading
        during inference.

  - ``format_sample(record)``
        Convert a raw benchmark record into a model-specific input dict
        (e.g. load PIL images, tokenise text, interpolate ego history).

  - ``get_prompt(formatted_sample, question)``
        Build the model-specific prompt payload for one question.
        Return a JSON-serialisable dict with model-specific fields only
        (conversation turns, special tokens, etc.).  Framework fields
        (scene_id, sample_id, image_paths, …) are injected by the base class.

Concrete helpers provided by this base class
--------------------------------------------
  - ``iter_questions(sample_ids)``   → generator of full prompt record dicts
  - ``save_prompts(sample_ids, output_path)`` → write prompts.jsonl
  - ``format_question_text(question)``        → plain-text question string
  - ``format_qa_text(question)``              → question + GT answer/reasoning
  - ``interpolate_ego_history(state, n_steps)`` → smoothed ego history list
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generator, List, Optional

from evaluation.common.dataset import BenchmarkDataset

logger = logging.getLogger(__name__)

# Map internal qa_type → source filename (for prompts.jsonl metadata)
_QA_TYPE_TO_FILE: Dict[str, str] = {
    "ladder":     "active_qa.json",
    "dormant":    "dormant_qa.json",
    "distractor": "distractor_qa.json",
}


class ModelDataLoader(ABC):
    """
    Converts benchmark records and QA questions into model-ready prompt payloads.

    Parameters
    ----------
    dataset   : BenchmarkDataset instance
    model_cfg : dict with model configuration (from eval_config.yaml or inline)
                  Expected keys (all optional):
                    "qa_types" : list[str] — which QA types to include;
                                 default ["ladder", "dormant", "distractor"]
    """

    def __init__(self, dataset: BenchmarkDataset, model_cfg: dict) -> None:
        self.dataset = dataset
        self.model_cfg = model_cfg

    # ------------------------------------------------------------------
    # Abstract interface — implement in each model's dataloader.py
    # ------------------------------------------------------------------

    @abstractmethod
    def build_image_paths(self, record: dict) -> List[Dict[str, str]]:
        """
        Build the ordered list of image path metadata dicts for this model.

        This is the primary customisation point for controlling which cameras
        and time steps the model receives, and in what order.

        Parameters
        ----------
        record : dict as returned by BenchmarkDataset.load_record_by_name()

        Returns
        -------
        List of dicts, each with exactly these keys:

            path       : str — relative image path (relative to dir_data_root)
                               e.g. "raw_data/nuscenes/samples/CAM_FRONT/file.jpg"
            time_key   : str — one of ["Tm1p5", "Tm1p0", "Tm0p5", "Tp0p0"]
            camera_key : str — one of ["cam_front", "cam_front_left",
                               "cam_front_right", "cam_back",
                               "cam_back_left", "cam_back_right"]

        The list is stored verbatim in prompts.jsonl under ``image_paths``.
        At inference time, resolve each path with ``CDBConfig.resolve_frame_path()``.

        Example (front-camera only, all 4 timesteps):
        ::

            def build_image_paths(self, record):
                paths = []
                for time_key in ["Tm1p5", "Tm1p0", "Tm0p5", "Tp0p0"]:
                    p = record["frames"].get(time_key, {}).get("cam_front")
                    if p:
                        paths.append({
                            "path": p,
                            "time_key": time_key,
                            "camera_key": "cam_front",
                        })
                return paths
        """
        raise NotImplementedError

    @abstractmethod
    def format_sample(self, record: dict) -> dict:
        """
        Convert a raw benchmark record into a model-specific input dict.

        This is called once per sample and should perform all heavy data
        manipulation: loading PIL images, tokenising state text, interpolating
        ego history, etc.  The result is passed to ``get_prompt()`` for every
        question in the sample.

        Parameters
        ----------
        record : dict as returned by BenchmarkDataset.load_record_by_name()

        Returns
        -------
        Model-specific dict.  Must include at minimum:
            scene_id  : str
            sample_id : str
        Additional common keys (not enforced):
            images     : list of PIL.Image or tensors
            state_text : str
        """
        raise NotImplementedError

    @abstractmethod
    def get_prompt(self, formatted_sample: dict, question: dict) -> dict:
        """
        Build a model-specific prompt payload for one question.

        The returned dict is merged into the prompt record that goes into
        prompts.jsonl.  Framework fields (``scene_id``, ``sample_id``,
        ``question_id``, ``prompt_id``, ``is_evaluated``, ``image_paths``,
        ``question_text``, ``qa_text``, ``answer_format``) are always set by
        the base class and override any keys with the same name returned here.

        Parameters
        ----------
        formatted_sample : dict returned by ``format_sample()``
        question         : one QA dict from a qa file, e.g.::

                {
                    "id":             "Q1",
                    "question":       "Which element …?",
                    "answer_format":  "mcq",
                    "options":        ["A) …", "B) …", "C) …", "D) …"],
                    "correct_answer": "A",
                    "reasoning":      "…",
                    "_qa_type":       "ladder",
                }

        Returns
        -------
        JSON-serialisable dict with model-specific fields only, e.g.::

            {"conversation": [{"role": "system", "content": "…"}, …]}
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def iter_questions(
        self, sample_ids: List[Dict]
    ) -> Generator[Dict, None, None]:
        """
        Iterate over complete prompt record dicts for every question.

        Each yielded dict is a complete prompts.jsonl record, merging
        framework-level fields (scene_id, sample_id, image_paths, …)
        with model-specific fields returned by ``get_prompt()``.

        Failures at the record / format_sample / get_prompt level are caught
        and logged; affected samples or questions are skipped.

        Parameters
        ----------
        sample_ids : list of dicts from ``BenchmarkDataset.get_sample_ids()``
                     Each dict: ``{"scene_id": str, "sample_id": str, …}``

        Yields
        ------
        Complete prompt record dicts (JSON-serialisable).
        """
        qa_types: List[str] = self.model_cfg.get(
            "qa_types", ["ladder", "dormant", "distractor"]
        )
        prompt_counter = 0

        for sid in sample_ids:
            scene_id = sid["scene_id"]
            sample_id = sid["sample_id"]

            # --- Load record ----------------------------------------------
            try:
                record = self.dataset.load_record_by_name(scene_id, sample_id)
            except Exception as exc:
                logger.warning(
                    "Skipping %s/%s — could not load record: %s",
                    scene_id, sample_id, exc,
                )
                continue

            # --- Build image paths (model-specific) -----------------------
            try:
                image_paths = self.build_image_paths(record)
            except Exception as exc:
                logger.warning(
                    "build_image_paths failed for %s/%s: %s",
                    scene_id, sample_id, exc,
                )
                image_paths = []

            # --- Format sample (heavy data loading) ----------------------
            try:
                formatted = self.format_sample(record)
            except Exception as exc:
                logger.warning(
                    "format_sample failed for %s/%s: %s",
                    scene_id, sample_id, exc,
                )
                continue

            # --- Iterate questions ----------------------------------------
            for qa_type in qa_types:
                questions = record["qa"].get(qa_type, [])
                qa_file = _QA_TYPE_TO_FILE.get(qa_type, "")

                for q_idx, question in enumerate(questions):
                    question_id = question.get("id", f"{qa_type}_{q_idx:04d}")
                    prompt_counter += 1

                    try:
                        model_payload = self.get_prompt(formatted, question)
                    except Exception as exc:
                        logger.warning(
                            "get_prompt failed for %s/%s/%s: %s",
                            scene_id, sample_id, question_id, exc,
                        )
                        continue

                    # Build the complete prompt record.
                    # Framework fields always take precedence over model_payload keys.
                    prompt_record: Dict = {
                        **model_payload,
                        # --- Framework fields (always set, always correct) ---
                        "scene_id":          scene_id,
                        "sample_id":         sample_id,
                        "question_id":       question_id,
                        "prompt_id":         f"{prompt_counter:06d}",
                        "is_evaluated":      False,
                        "question_json_file": qa_file,
                        "answer_format":     question.get("answer_format", "binary"),
                        "question_text":     self.format_question_text(question),
                        "qa_text":           self._format_inference_text(question),
                        "image_paths":       image_paths,
                    }
                    yield prompt_record

    def save_prompts(
        self,
        sample_ids: List[Dict],
        output_path: str | Path,
    ) -> int:
        """
        Persist all prompt records to a JSONL file.

        One JSON object per line; each line is a complete prompts.jsonl record.

        Parameters
        ----------
        sample_ids  : list from ``BenchmarkDataset.get_sample_ids()``
        output_path : path to write JSONL (created; parent dirs made if needed)

        Returns
        -------
        Number of prompt records written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w") as fh:
            for record in self.iter_questions(sample_ids):
                fh.write(json.dumps(record) + "\n")
                count += 1

        logger.info("save_prompts: wrote %d records to %s", count, output_path)
        return count

    # ------------------------------------------------------------------
    # Static text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_question_text(question: dict) -> str:
        """
        Build a plain-text question string (no ground-truth answer).

        Handles both list-format options (``["A) …", "B) …"]``) and the
        legacy dict format (``{"A": "…", "B": "…"}``).

        Example output::

            Question: Which element is blocking you?
            A) The construction worker on the crosswalk
            B) The SUV behind you
            C) The traffic light
            D) The construction barrier
        """
        parts = [f"Question: {question.get('question', '')}"]
        options = question.get("options") or []

        if isinstance(options, list):
            parts.extend(str(o) for o in options)
        elif isinstance(options, dict):
            # Legacy dict format: {"A": "text", "B": "text", …}
            for letter in ["A", "B", "C", "D"]:
                if letter in options:
                    parts.append(f"{letter}) {options[letter]}")

        return "\n".join(parts)

    @staticmethod
    def _format_inference_text(question: dict) -> str:
        """
        Build the question text for model inference — no ground-truth answer.

        Includes the format hint so the model knows the expected output shape,
        but deliberately omits correct_answer and reasoning to prevent leakage.
        """
        question_text = ModelDataLoader.format_question_text(question)
        answer_format = question.get("answer_format", "binary")
        hint = "Answer: Yes or No" if answer_format == "binary" else "Answer: A, B, C, or D"
        return f"{question_text}\n\nFormat: {hint}"

    @staticmethod
    def format_qa_text(question: dict) -> str:
        """
        Build a text block including the question, answer format hint,
        and ground-truth answer and reasoning.

        Useful for few-shot prompts and ground-truth verification.

        Example output (binary)::

            Question: Would you still remain stopped?

            Format:
            Answer: Yes or No
            Answer: Yes
            Reasoning: You are stopped because …
        """
        question_text = ModelDataLoader.format_question_text(question)
        answer_format = question.get("answer_format", "binary")
        correct_answer = question.get("correct_answer", "")
        reasoning = question.get("reasoning", "")

        if answer_format == "binary":
            format_hint = "Answer: Yes or No"
        else:
            format_hint = "Answer: A, B, C, or D"

        return (
            f"{question_text}\n\n"
            f"Format:\n{format_hint}\n"
            f"Answer: {correct_answer}\n"
            f"Reasoning: {reasoning}"
        )

    # ------------------------------------------------------------------
    # Data interpolation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def interpolate_ego_history(
        state: Optional[dict],
        n_steps: int = 4,
    ) -> List[List[float]]:
        """
        Extract or interpolate ego vehicle XYZ positions from state.json.

        Returns a list of *n_steps* ``[x, y, z]`` coordinates in ego-centric
        coordinates, linearly interpolated from whatever ego history is
        available in *state*.

        Parameters
        ----------
        state   : state.json dict (may be None if missing)
        n_steps : number of history steps to return (default 4)

        Returns
        -------
        List of [x, y, z] floats, length == n_steps.
        Falls back to all-zeros if *state* is None or missing ego data.
        """
        if not state:
            return [[0.0, 0.0, 0.0]] * n_steps

        # Try "ego_history" list first, then "ego_states"
        history_raw: Optional[list] = (
            state.get("ego_history") or state.get("ego_states")
        )

        if not history_raw or not isinstance(history_raw, list):
            # No history: return all-zeros (T=0 origin)
            return [[0.0, 0.0, 0.0]] * n_steps

        # Extract [x, y, z] from each entry; support dict or list format
        def _to_xyz(entry) -> List[float]:
            if isinstance(entry, dict):
                return [
                    float(entry.get("x", 0.0)),
                    float(entry.get("y", 0.0)),
                    float(entry.get("z", 0.0)),
                ]
            if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                return [float(entry[0]), float(entry[1]), float(entry[2])]
            return [0.0, 0.0, 0.0]

        raw_xyz = [_to_xyz(e) for e in history_raw]

        if len(raw_xyz) == n_steps:
            return raw_xyz

        if len(raw_xyz) == 1:
            return [raw_xyz[0]] * n_steps

        # Linear interpolation to resize to n_steps
        import numpy as np  # lazy import; only needed here
        src = np.array(raw_xyz, dtype=float)  # (M, 3)
        t_src = np.linspace(0, 1, len(src))
        t_dst = np.linspace(0, 1, n_steps)
        interp = np.stack(
            [np.interp(t_dst, t_src, src[:, i]) for i in range(src.shape[1])],
            axis=1,
        )
        return interp.tolist()

    @staticmethod
    def build_chat_messages(
        system_prompt: str,
        user_content: str,
    ) -> List[Dict[str, str]]:
        """
        Build a minimal OpenAI-compatible chat message list.

        Parameters
        ----------
        system_prompt : system role content
        user_content  : user role content

        Returns
        -------
        ``[{"role": "system", "content": …}, {"role": "user", "content": …}]``
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Trajectory evaluation helpers (optional — override in model classes)
    # ------------------------------------------------------------------

    def get_trajectory_prompt(
        self,
        formatted_sample: dict,
        traj_type: str,
        context_question: Optional[dict] = None,
        gt_waypoints: Optional[list] = None,
    ) -> dict:
        """
        Build a model-specific trajectory prompt payload.

        Override this in the model's DataLoader to enable trajectory evaluation.
        The default implementation raises NotImplementedError so that
        iter_trajectory_questions() can skip models that don't implement it.

        Parameters
        ----------
        formatted_sample  : dict from format_sample()
        traj_type         : "BTE", "CTE_R2_CDE", "CTE_R3_CR", etc.
        context_question  : for CTE types, the source VQA question dict
        gt_waypoints      : [{t, x, y}, ...] ground-truth (for prompt timestep info)

        Returns
        -------
        JSON-serialisable dict with model-specific fields (qa_text, ego_history_*, …)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_trajectory_prompt(). "
            "Override this method to enable trajectory evaluation."
        )

    def iter_trajectory_questions(
        self,
        sample_ids: List[Dict],
        use_bte: bool = True,
        use_cte: bool = True,
    ) -> "Generator[Dict, None, None]":
        """
        Yield trajectory prompt records for BTE and CTE questions.

        One BTE record per sample; one CTE record per question with
        alternate_trajectory in active_qa.json.

        Skips samples/questions where get_trajectory_prompt() raises
        NotImplementedError (model doesn't support trajectory eval).
        All other errors are logged and skipped.

        Yields dicts suitable for trajectory_prompts.jsonl.
        """
        from evaluation.common.trajectory_eval import (
            BTE_TIMESTEPS,
            extract_bte_ground_truth,
            extract_cte_ground_truth,
            find_trajectory_questions,
            get_eval_type,
        )
        from evaluation.common.utils.exceptions import EvaluationException

        prompt_counter = 0

        for sid in sample_ids:
            scene_id = sid["scene_id"]
            sample_id = sid["sample_id"]

            try:
                record = self.dataset.load_record_by_name(scene_id, sample_id)
            except Exception as exc:
                logger.warning(
                    "iter_trajectory_questions: cannot load %s/%s: %s",
                    scene_id, sample_id, exc,
                )
                continue

            try:
                image_paths = self.build_image_paths(record)
            except Exception as exc:
                logger.warning(
                    "iter_trajectory_questions: build_image_paths failed %s/%s: %s",
                    scene_id, sample_id, exc,
                )
                image_paths = []

            try:
                formatted = self.format_sample(record)
            except Exception as exc:
                logger.warning(
                    "iter_trajectory_questions: format_sample failed %s/%s: %s",
                    scene_id, sample_id, exc,
                )
                continue

            state_json = record.get("state") or {}

            # ---- BTE --------------------------------------------------------
            if use_bte:
                try:
                    gt_wps = extract_bte_ground_truth(state_json)
                except EvaluationException as exc:
                    logger.warning(
                        "BTE GT unavailable for %s/%s: %s", scene_id, sample_id, exc
                    )
                    gt_wps = None

                if gt_wps is not None:
                    prompt_counter += 1
                    question_id = "traj_BTE"
                    try:
                        model_payload = self.get_trajectory_prompt(
                            formatted, "BTE",
                            context_question=None,
                            gt_waypoints=gt_wps,
                        )
                    except NotImplementedError:
                        logger.debug(
                            "get_trajectory_prompt not implemented; skipping trajectory eval"
                        )
                        return
                    except Exception as exc:
                        logger.warning(
                            "get_trajectory_prompt(BTE) failed %s/%s: %s",
                            scene_id, sample_id, exc,
                        )
                        model_payload = {}

                    gt_times = [w["t"] for w in gt_wps]
                    yield {
                        **model_payload,
                        "scene_id":          scene_id,
                        "sample_id":         sample_id,
                        "question_id":       question_id,
                        "prompt_id":         f"{prompt_counter:06d}",
                        "eval_type":         "BTE",
                        "trajectory_eval":   True,
                        "answer_format":     "trajectory",
                        "parent_question_id": None,
                        "gt_timesteps":      gt_times,
                        "gt_waypoints":      gt_wps,
                        "image_paths":       image_paths,
                    }

            # ---- CTE --------------------------------------------------------
            # CTE questions can appear in any of the three QA buckets
            # (active/ladder, dormant, distractor) — alternate_trajectory is
            # attached per-question, not per-bucket.
            if use_cte:
                qa_buckets = record.get("qa", {}) or {}
                bucket_aliases = {"ladder": "active"}  # active_qa.json on disk
                cte_iter = []
                for bucket_name, qs in qa_buckets.items():
                    parent_qa_type = bucket_aliases.get(bucket_name, bucket_name)
                    for q in find_trajectory_questions({"questions": qs or []}):
                        cte_iter.append((parent_qa_type, q))

                for parent_qa_type, question in cte_iter:
                    try:
                        gt_wps, rung, eval_type = extract_cte_ground_truth(question)
                    except EvaluationException as exc:
                        logger.warning(
                            "CTE GT unavailable for %s/%s q=%s: %s",
                            scene_id, sample_id, question.get("id"), exc,
                        )
                        continue

                    parent_qid = question.get("id", "")
                    question_id = f"traj_CTE_{parent_qid}"
                    prompt_counter += 1

                    try:
                        model_payload = self.get_trajectory_prompt(
                            formatted, eval_type,
                            context_question=question,
                            gt_waypoints=gt_wps,
                        )
                    except NotImplementedError:
                        return
                    except Exception as exc:
                        logger.warning(
                            "get_trajectory_prompt(%s) failed %s/%s q=%s: %s",
                            eval_type, scene_id, sample_id, parent_qid, exc,
                        )
                        model_payload = {}

                    gt_times = [w["t"] for w in gt_wps]
                    q_type = question.get("category", "")
                    yield {
                        **model_payload,
                        "scene_id":           scene_id,
                        "sample_id":          sample_id,
                        "question_id":        question_id,
                        "prompt_id":          f"{prompt_counter:06d}",
                        "eval_type":          eval_type,
                        "trajectory_eval":    True,
                        "answer_format":      "vqa+trajectory",
                        "parent_question_id": parent_qid,
                        "parent_qa_type":     parent_qa_type,
                        "rung":               rung,
                        "question_type":      q_type,
                        "gt_answer":          question.get("correct_answer", ""),
                        "gt_timesteps":       gt_times,
                        "gt_waypoints":       gt_wps,
                        "vqa_question_text":  question.get("question", ""),
                        "image_paths":        image_paths,
                    }

    def save_trajectory_prompts(
        self,
        sample_ids: List[Dict],
        output_path: "str | Path",
        use_bte: bool = True,
        use_cte: bool = True,
    ) -> int:
        """
        Persist trajectory prompt records to trajectory_prompts.jsonl.

        Returns number of records written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w") as fh:
            for record in self.iter_trajectory_questions(
                sample_ids, use_bte=use_bte, use_cte=use_cte
            ):
                fh.write(json.dumps(record) + "\n")
                count += 1

        logger.info(
            "save_trajectory_prompts: wrote %d records to %s", count, output_path
        )
        return count

    @staticmethod
    def handle_missing_frame(
        path: Optional[str],
        label: str = "frame",
    ) -> Optional[str]:
        """
        Validate that an image path is non-empty.

        Parameters
        ----------
        path  : relative image path from frames.json (may be None / empty)
        label : human-readable label for logging

        Returns
        -------
        The path string if valid, else None (with a debug log).
        """
        if not path:
            logger.debug("Missing %s path", label)
            return None
        return path
