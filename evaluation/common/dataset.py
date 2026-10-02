"""
BenchmarkDataset: structured access to CausalDriveBench records and QA.

Directory layout (see evaluation/docs/data_formats.md):

    <bench_dir>/
    ├── manifest.json
    ├── <scene_id>/                        # e.g. nuscenes-scene-0159/
    │   ├── SAMPLED_0/
    │   │   ├── frames.json                — per-frame image paths (relative)
    │   │   ├── state.json                 — ego + agent state at T=0
    │   │   ├── calib.json                 — camera calibration matrices
    │   │   ├── meta.json                  — scene metadata
    │   │   ├── graph.json                 — pruned causal graph
    │   │   └── qa/
    │   │       ├── active_qa.json         — ladder / active questions
    │   │       ├── dormant_qa.json        — dormant link questions
    │   │       └── distractor_qa.json     — distractor node questions
    │   └── SAMPLED_N/
    └── <scene_id>/

Important
---------
- SAMPLED indices are NOT guaranteed to be sequential (gaps are expected).
- QA files may be missing; individual questions may be invalid — both are
  handled gracefully with warnings.
- ``data_root`` in frames.json is intentionally ignored.  Callers must use
  ``CDBConfig.resolve_frame_path()`` to obtain absolute image paths.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, Generator, List, Optional

from evaluation.common.utils.exceptions import DataException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All temporal frame keys; Tp0p0 is T=0 (the "current" instant)
FRAMES: list[str] = ["Tm1p5", "Tm1p0", "Tm0p5", "Tp0p0"]

# Union of all camera keys across nuscenes / argoverse2 / openscene.
# Actual keys per sample depend on the source dataset.
CAMERAS: list[str] = [
    "cam_front",
    "cam_front_left",
    "cam_front_right",
    "cam_back",
    "cam_back_left",
    "cam_back_right",
]

SAMPLED_PREFIX = "SAMPLED_"

# Filename → internal QA type key mapping
_QA_FILE_TO_TYPE: Dict[str, str] = {
    "active_qa.json":     "ladder",
    "dormant_qa.json":    "dormant",
    "distractor_qa.json": "distractor",
}

# Required fields for a question to be considered valid
_QUESTION_REQUIRED_FIELDS: tuple[str, ...] = (
    "id", "question", "answer_format", "correct_answer", "reasoning"
)


# ---------------------------------------------------------------------------
# BenchmarkDataset
# ---------------------------------------------------------------------------

class BenchmarkDataset:
    """
    Provides structured access to benchmark records and QA files.

    Parameters
    ----------
    bench_dir : str | Path
        Dataset-specific subdirectory of causal_drive_bench, e.g.
        ``/causal_drive_bench/causal_nuscenes``.
        Direct children are scene directories (e.g. ``nuscenes-scene-0159/``),
        each containing ``SAMPLED_N/`` subdirectories.
    """

    def __init__(self, bench_dir: str | Path) -> None:
        self.bench_dir = Path(bench_dir)
        if not self.bench_dir.exists():
            raise FileNotFoundError(f"bench_dir does not exist: {self.bench_dir}")
        if not self.bench_dir.is_dir():
            raise NotADirectoryError(f"bench_dir is not a directory: {self.bench_dir}")
        logger.info("BenchmarkDataset: bench_dir=%s", self.bench_dir)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_scenes(self) -> List[str]:
        """
        Return sorted list of scene IDs found directly under bench_dir.

        Only directories that contain at least one SAMPLED_N/ subdirectory
        are returned; files (e.g. manifest.json) are ignored.
        """
        scenes: List[str] = []
        for entry in sorted(self.bench_dir.iterdir()):
            if not entry.is_dir():
                continue
            if any(
                s.is_dir() and s.name.startswith(SAMPLED_PREFIX)
                for s in entry.iterdir()
            ):
                scenes.append(entry.name)
        logger.debug("list_scenes: %d scenes in %s", len(scenes), self.bench_dir)
        return scenes

    def list_samples(self, scene: str) -> List[str]:
        """
        Return sorted list of sample directory names for a scene.

        Names follow the ``SAMPLED_N`` pattern; indices need not be sequential.

        Raises
        ------
        FileNotFoundError  if the scene directory does not exist
        DataException      if no SAMPLED_N subdirectories are found
        """
        scene_dir = self._scene_dir(scene)
        if not scene_dir.exists():
            raise FileNotFoundError(
                f"Scene directory not found: {scene_dir}"
            )
        samples = sorted(
            d.name
            for d in scene_dir.iterdir()
            if d.is_dir() and d.name.startswith(SAMPLED_PREFIX)
        )
        if not samples:
            raise DataException(
                f"No {SAMPLED_PREFIX}N directories found for scene '{scene}' in {scene_dir}"
            )
        logger.debug("list_samples(%s): %s", scene, samples)
        return samples

    def _scene_dir(self, scene: str) -> Path:
        return self.bench_dir / scene

    def _sample_dir(self, scene: str, sample: str) -> Path:
        return self.bench_dir / scene / sample

    # ------------------------------------------------------------------
    # Record loading
    # ------------------------------------------------------------------

    def load_record(self, scene: str, sample_idx: int = 0) -> Dict:
        """
        Load a full benchmark record by scene name and sample index.

        Parameters
        ----------
        scene      : scene directory name (e.g. ``"nuscenes-scene-0159"``)
        sample_idx : index into ``list_samples(scene)``; default 0

        Returns
        -------
        Record dict — see ``load_record_by_name`` for the full schema.
        """
        samples = self.list_samples(scene)
        if sample_idx >= len(samples):
            raise IndexError(
                f"sample_idx {sample_idx} out of range for scene '{scene}' "
                f"({len(samples)} samples: {samples})"
            )
        return self.load_record_by_name(scene, samples[sample_idx])

    def load_record_by_name(self, scene: str, sample: str) -> Dict:
        """
        Load a full benchmark record by explicit scene and sample names.

        Parameters
        ----------
        scene  : scene directory name (e.g. ``"nuscenes-scene-0159"``)
        sample : sample directory name (e.g. ``"SAMPLED_0"``)

        Returns
        -------
        dict with keys:

            scene_id  : str — scene directory name
            sample_id : str — sample directory name (e.g. "SAMPLED_0")
            meta      : dict — meta.json contents ({} if absent)
            frames    : dict[time_key → dict[cam_key → relative_path]]
                        Relative paths are joined with CDBConfig.dir_data_root
                        to produce absolute paths.
            state     : dict | None — state.json contents
            calib     : dict | None — calib.json contents
            qa        : dict[qa_type → List[dict]]
                        qa_type ∈ {"ladder", "dormant", "distractor"}
        """
        sample_dir = self._sample_dir(scene, sample)
        if not sample_dir.exists():
            raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

        logger.debug("load_record_by_name: %s / %s", scene, sample)

        return {
            "scene_id":  scene,
            "sample_id": sample,
            "meta":      self._load_json_optional(sample_dir / "meta.json", default={}),
            "state":     self._load_json_optional(sample_dir / "state.json"),
            "calib":     self._load_json_optional(sample_dir / "calib.json"),
            "frames":    self._load_frames(sample_dir),
            "qa":        self._load_qa(sample_dir),
        }

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_json_optional(self, path: Path, default=None):
        """
        Load a JSON file and return its contents.

        Returns *default* if the file does not exist.
        Raises DataException if the file exists but is malformed.
        """
        if not path.exists():
            logger.debug("Optional file absent: %s", path)
            return default
        try:
            with open(path) as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            raise DataException(f"Malformed JSON in {path}: {exc}") from exc
        except OSError as exc:
            raise DataException(f"Cannot read {path}: {exc}") from exc

    def _load_frames(self, sample_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Load frames.json and return relative image paths keyed by time → camera.

        The ``data_root`` field in frames.json is intentionally ignored.
        Paths are resolved externally via ``CDBConfig.resolve_frame_path()``.

        Returns an empty camera dict for any missing time key.
        """
        frames_path = sample_dir / "frames.json"
        if not frames_path.exists():
            logger.warning("frames.json absent: %s", frames_path)
            return {k: {} for k in FRAMES}

        try:
            with open(frames_path) as fh:
                raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise DataException(
                f"Malformed frames.json in {sample_dir}: {exc}"
            ) from exc

        # frames.json may wrap camera paths under a "frames" key
        frame_data: dict = raw.get("frames", raw)

        frames: Dict[str, Dict[str, str]] = {}
        for time_key in FRAMES:
            cam_paths = frame_data.get(time_key, {})
            if not isinstance(cam_paths, dict):
                logger.warning(
                    "Unexpected type for time_key '%s' in %s; skipping",
                    time_key, frames_path,
                )
                cam_paths = {}
            # Keep only non-empty string values; ignore None / non-string entries
            frames[time_key] = {
                k: v for k, v in cam_paths.items()
                if isinstance(v, str) and v
            }
        return frames

    def _load_qa(self, sample_dir: Path) -> Dict[str, List[Dict]]:
        """
        Load QA from the qa/ subfolder.

        Reads active_qa.json → "ladder", dormant_qa.json → "dormant",
        distractor_qa.json → "distractor".

        Missing files are silently skipped.
        Malformed files raise DataException.
        Invalid individual questions are skipped with a warning.
        """
        qa: Dict[str, List[Dict]] = {"ladder": [], "dormant": [], "distractor": []}
        qa_dir = sample_dir / "qa"

        if not qa_dir.exists():
            logger.warning("qa/ folder absent: %s", qa_dir)
            return qa

        for filename, qa_type in _QA_FILE_TO_TYPE.items():
            qa_file = qa_dir / filename
            if not qa_file.exists():
                logger.debug("QA file absent: %s", qa_file)
                continue

            questions = self._parse_qa_file(qa_file, qa_type)
            qa[qa_type].extend(questions)
            logger.debug(
                "Loaded %d %s questions from %s", len(questions), qa_type, qa_file.name
            )

        return qa

    def _parse_qa_file(self, path: Path, qa_type: str) -> List[Dict]:
        """
        Parse a single QA JSON file and return validated question dicts.

        Raises DataException for IO errors or structural issues.
        Skips (with warning) individual questions that fail validation.
        """
        try:
            with open(path) as fh:
                loaded = json.load(fh)
        except json.JSONDecodeError as exc:
            raise DataException(f"Malformed QA file {path}: {exc}") from exc
        except OSError as exc:
            raise DataException(f"Cannot read QA file {path}: {exc}") from exc

        # Support {"questions": [...]} wrapper or bare list
        if isinstance(loaded, dict) and "questions" in loaded:
            raw_questions = loaded["questions"]
        elif isinstance(loaded, list):
            raw_questions = loaded
        else:
            raise DataException(
                f"Unexpected QA format in {path}: expected dict with 'questions' "
                f"key or a bare list; got {type(loaded).__name__}"
            )

        if not isinstance(raw_questions, list):
            raise DataException(
                f"'questions' value is not a list in {path}: got {type(raw_questions).__name__}"
            )

        validated: List[Dict] = []
        for i, q in enumerate(raw_questions):
            if not isinstance(q, dict):
                logger.warning(
                    "Skipping non-dict question at index %d in %s", i, path.name
                )
                continue
            error = _validate_question(q)
            if error:
                logger.warning(
                    "Skipping invalid question at index %d in %s: %s",
                    i, path.name, error,
                )
                continue
            # Annotate with internal metadata
            q.setdefault("_qa_type", qa_type)
            q.setdefault("_source_file", path.name)
            validated.append(q)

        return validated

    # ------------------------------------------------------------------
    # Sample ID helpers
    # ------------------------------------------------------------------

    def get_sample_ids(
        self,
        mode: str = "full",
        subset_size: Optional[int] = None,
        scene: Optional[str] = None,
        scenes_list: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Return a list of sample identifier dicts.

        Each dict has the form::

            {
                "scene_id":   str,   # e.g. "nuscenes-scene-0159"
                "sample_id":  str,   # e.g. "SAMPLED_0"
                "sample_idx": int,   # position in list_samples()
            }

        Parameters
        ----------
        mode        : ``"full"``   — all scenes and samples
                      ``"subset"`` — random subset of *subset_size* scenes
                      ``"single"`` — single scene (provide *scene* parameter)
        subset_size : required when mode="subset"
        scene       : required when mode="single"
        scenes_list : explicit list of scene IDs — overrides mode/scene/subset_size
        """
        # If explicit list of scenes is given, override mode completely
        if scenes_list:
            scenes = list(scenes_list)
        elif mode == "single":
            if not scene:
                raise ValueError("mode='single' requires the 'scene' parameter")
            scenes = [scene]
        else:
            scenes = self.list_scenes()

        if mode == "subset":
            if subset_size is None:
                raise ValueError("mode='subset' requires 'subset_size'")
            if subset_size < len(scenes):
                scenes = random.sample(scenes, subset_size)

        sample_ids: List[Dict] = []
        for sc in scenes:
            try:
                samples = self.list_samples(sc)
            except (FileNotFoundError, DataException) as exc:
                logger.warning("Skipping scene '%s': %s", sc, exc)
                continue
            for idx, samp in enumerate(samples):
                sample_ids.append(
                    {"scene_id": sc, "sample_id": samp, "sample_idx": idx}
                )

        logger.info(
            "get_sample_ids(mode=%s): %d samples across %d scene(s)",
            mode, len(sample_ids), len(scenes),
        )
        return sample_ids

    # ------------------------------------------------------------------
    # Convenience iterator
    # ------------------------------------------------------------------

    def iter_records(
        self,
        mode: str = "full",
        subset_size: Optional[int] = None,
        scene: Optional[str] = None,
    ) -> Generator[Dict, None, None]:
        """
        Iterate over loaded records for the specified sample set.

        Yields dicts from ``load_record_by_name()``.
        Samples that fail to load are skipped with a warning.
        """
        for sid in self.get_sample_ids(mode=mode, subset_size=subset_size, scene=scene):
            try:
                yield self.load_record_by_name(sid["scene_id"], sid["sample_id"])
            except (DataException, FileNotFoundError, IndexError) as exc:
                logger.warning(
                    "Skipping %s/%s: %s", sid["scene_id"], sid["sample_id"], exc
                )


# ---------------------------------------------------------------------------
# Question validation
# ---------------------------------------------------------------------------

def _validate_question(q: dict) -> Optional[str]:
    """
    Return an error string if the question dict is invalid, else None.

    Required fields: id, question, answer_format, correct_answer, reasoning.
    MCQ questions must also have a non-empty options list.
    """
    for field in _QUESTION_REQUIRED_FIELDS:
        if field not in q:
            return f"missing required field '{field}'"
        val = q[field]
        if not isinstance(val, (str, int, float, bool)):
            return f"field '{field}' has unexpected type {type(val).__name__}"
        if isinstance(val, str) and not val.strip():
            return f"field '{field}' is an empty string"

    if q.get("answer_format") == "mcq":
        opts = q.get("options")
        if not opts:
            return "answer_format=mcq but 'options' is missing or empty"

    return None
