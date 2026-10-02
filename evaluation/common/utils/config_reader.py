"""
CausalDriveBench configuration reader.

Reads eval_config.yaml and resolves all directory paths for both host and
Docker runtime environments.  The EVAL_RUNTIME environment variable selects
the runtime mode:

  - ``"host"``   (default): host paths from eval_config.yaml
  - ``"docker"``: fixed Docker volume-mount paths (constants)

Image paths stored in frames.json are *relative* to ``dir_data_root`` (the
parent directory that contains raw_data/).  Use ``resolve_frame_path()`` to
convert them to absolute paths; never read ``data_root`` from frames.json.

Usage
-----
    # Auto-detect config and runtime
    cfg = CDBConfig()

    # Explicit config path
    cfg = CDBConfig("/path/to/eval_config.yaml")

    # Force docker paths regardless of EVAL_RUNTIME env var
    cfg = CDBConfig(runtime="docker")

    # Resolve an image path from frames.json
    abs_path = cfg.resolve_frame_path("raw_data/nuscenes/samples/CAM_FRONT/file.jpg")

    # Get bench directory for a dataset
    bench_dir = cfg.get_bench_dir("nuscenes")  # → /data/causal_drive_bench/causal_nuscenes
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from evaluation.common.utils.errors import ConfigurationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_RUNTIME_ENV = "EVAL_RUNTIME"

# Fixed Docker volume-mount paths — these are constants inside the container.
_DOCKER_WORKSPACE = "/workspace"
_DOCKER_DATA_ROOT = "/workspace/data"
_DOCKER_PATHS: dict[str, str] = {
    "raw_data":            f"{_DOCKER_DATA_ROOT}/raw_data",
    "causal_drive_bench":  f"{_DOCKER_DATA_ROOT}/causal_drive_bench",
    "cdb_outputs":         f"{_DOCKER_DATA_ROOT}/cdb_outputs",
    "cdb_reports":         f"{_DOCKER_DATA_ROOT}/cdb_reports",
    "model_code":          f"{_DOCKER_WORKSPACE}/model",
    "model_weights":       f"{_DOCKER_WORKSPACE}/weights",
}


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------

def _find_eval_config() -> Path:
    """
    Locate eval_config.yaml by searching standard locations.

    Search order:
      1. Current working directory
      2. evaluation/ subdirectory of cwd
      3. Two levels up from this file (evaluation/common/utils/ → evaluation/)
    """
    candidates = [
        Path.cwd() / "eval_config.yaml",
        Path.cwd() / "evaluation" / "eval_config.yaml",
        Path(__file__).parents[2] / "eval_config.yaml",  # evaluation/
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "eval_config.yaml not found. Searched:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


# ---------------------------------------------------------------------------
# CDBConfig
# ---------------------------------------------------------------------------

class CDBConfig:
    """
    Resolves CausalDriveBench directory paths for the current runtime.

    Parameters
    ----------
    config_path : optional path to eval_config.yaml.  Auto-detected if omitted.
    runtime     : ``"host"`` or ``"docker"``.  Reads EVAL_RUNTIME env var if omitted.
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        runtime: Optional[str] = None,
    ) -> None:
        self.runtime: str = (runtime or os.environ.get(EVAL_RUNTIME_ENV, "host")).lower()
        if self.runtime not in ("host", "docker"):
            raise ConfigurationError(
                f"EVAL_RUNTIME must be 'host' or 'docker', got '{self.runtime}'"
            )

        config_path = Path(config_path) if config_path else _find_eval_config()
        if not config_path.exists():
            raise FileNotFoundError(f"eval_config.yaml not found at {config_path}")

        self.config_path = config_path
        with open(config_path) as fh:
            self._cfg: dict = yaml.safe_load(fh) or {}

        logger.info(
            "CDBConfig: runtime=%s  config=%s", self.runtime, self.config_path
        )

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    @property
    def is_docker(self) -> bool:
        return self.runtime == "docker"

    def _host_path(self, key: str, fallback: str) -> Path:
        """Return Path from config key, or *fallback* if absent."""
        return Path(self._cfg.get(key) or fallback)

    # ------------------------------------------------------------------
    # Directory properties
    # ------------------------------------------------------------------

    @property
    def dir_raw_data(self) -> Path:
        """
        Root directory for raw dataset files (nuscenes/, argoverse2/, openscene/).

        Docker: ``/workspace/data/raw_data``
        Host:   ``raw_data_dir`` from eval_config.yaml
        """
        if self.is_docker:
            return Path(_DOCKER_PATHS["raw_data"])
        return self._host_path("raw_data_dir", "/raw_data")

    @property
    def dir_data_root(self) -> Path:
        """
        Parent of raw_data/.

        Used to resolve relative image paths from frames.json.
        E.g. ``raw_data/nuscenes/.../file.jpg`` → ``dir_data_root / relative_path``.
        """
        return self.dir_raw_data.parent

    @property
    def dir_causal_drive_bench(self) -> Path:
        """
        Root of CausalDriveBench annotations (causal_nuscenes/, causal_argoverse2/, …).

        Docker: ``/workspace/data/causal_drive_bench``
        Host:   ``causal_drive_bench_dir`` from eval_config.yaml,
                 or ``dir_data_root/causal_drive_bench`` if absent
        """
        if self.is_docker:
            return Path(_DOCKER_PATHS["causal_drive_bench"])
        return self._host_path(
            "causal_drive_bench_dir",
            str(self.dir_data_root / "causal_drive_bench"),
        )

    @property
    def dir_cdb_outputs(self) -> Path:
        """
        Root of per-run inference output directories.

        Docker: ``/workspace/data/cdb_outputs``
        Host:   ``cdb_outputs_dir`` from eval_config.yaml,
                 or ``dir_data_root/cdb_outputs`` if absent
        """
        if self.is_docker:
            return Path(_DOCKER_PATHS["cdb_outputs"])
        return self._host_path(
            "cdb_outputs_dir",
            str(self.dir_data_root / "cdb_outputs"),
        )

    @property
    def dir_cdb_reports(self) -> Path:
        """
        Root of evaluation report directories.

        Docker: ``/workspace/data/cdb_reports``
        Host:   ``cdb_reports_dir`` from eval_config.yaml,
                 or ``dir_data_root/cdb_reports`` if absent
        """
        if self.is_docker:
            return Path(_DOCKER_PATHS["cdb_reports"])
        return self._host_path(
            "cdb_reports_dir",
            str(self.dir_data_root / "cdb_reports"),
        )

    @property
    def dir_model_weights(self) -> Path:
        """
        Default model weights directory.

        Docker: ``/workspace/weights``
        Host:   ``model_weights_dir`` from eval_config.yaml, or ``/weights``
        """
        if self.is_docker:
            return Path(_DOCKER_PATHS["model_weights"])
        return self._host_path("model_weights_dir", "/weights")

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def get_bench_dir(self, dataset_name: str) -> Path:
        """
        Return the benchmark directory for a named dataset.

        Checks ``datasets.<name>.bench_dir`` in eval_config.yaml first;
        falls back to ``dir_causal_drive_bench/causal_<dataset_name>``.

        Parameters
        ----------
        dataset_name : ``"nuscenes"``, ``"argoverse2"``, or ``"openscene"``
        """
        if self.is_docker:
            return self.dir_causal_drive_bench / f"causal_{dataset_name}"

        datasets: dict = self._cfg.get("datasets", {})
        ds = datasets.get(dataset_name, {})
        bench_dir = ds.get("bench_dir")
        if bench_dir:
            return Path(bench_dir)
        
        return self.dir_causal_drive_bench / f"causal_{dataset_name}"

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def resolve_frame_path(self, relative_path: str) -> Path:
        """
        Resolve a relative image path from frames.json to an absolute path.

        The ``data_root`` field inside frames.json is intentionally ignored;
        ``dir_data_root`` from this config is always the authoritative root.

        Parameters
        ----------
        relative_path : as stored in frames.json, e.g.
            ``"raw_data/nuscenes/samples/CAM_FRONT/file.jpg"``

        Returns
        -------
        Absolute ``Path`` object.
        """
        return self.dir_data_root / relative_path

    # ------------------------------------------------------------------
    # Evaluation config shortcuts
    # ------------------------------------------------------------------

    @property
    def evaluation(self) -> dict:
        return self._cfg.get("evaluation", {})

    @property
    def batch_size(self) -> int:
        return int(self.evaluation.get("batch_size", 4))

    @property
    def qa_types(self) -> list[str]:
        return list(self.evaluation.get("qa_types", ["ladder", "dormant", "distractor"]))

    @property
    def use_llm_judge(self) -> bool:
        return bool(self.evaluation.get("use_llm_judge", False))

    @property
    def judge_config(self) -> dict:
        return self.evaluation.get("judge", {})

    def get_model_config(self, model_name: str) -> dict:
        """Return the model-specific config block from eval_config.yaml."""
        return self._cfg.get("models", {}).get(model_name, {})

    # ------------------------------------------------------------------
    # Reporting config shortcuts
    # ------------------------------------------------------------------

    @property
    def reporting(self) -> dict:
        return self._cfg.get("reporting", {})

    @property
    def schema_version(self) -> str:
        return str(self.reporting.get("schema_version", "1.0"))

    @property
    def report_metrics(self) -> list[str]:
        return list(self.reporting.get("metrics", [
            "overall", "per_qa_type", "per_question_type", "per_graph_structure",
            "per_rung", "answer_stats", "binary_metrics", "reasoning_score",
        ]))

    @property
    def html_enabled(self) -> bool:
        return bool(self.reporting.get("html", {}).get("enabled", True))

    # ------------------------------------------------------------------
    # Debug / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of all resolved paths."""
        lines = [
            f"CDBConfig  runtime={self.runtime}  config={self.config_path}",
            f"  dir_data_root         : {self.dir_data_root}",
            f"  dir_raw_data          : {self.dir_raw_data}",
            f"  dir_causal_drive_bench: {self.dir_causal_drive_bench}",
            f"  dir_cdb_outputs       : {self.dir_cdb_outputs}",
            f"  dir_cdb_reports       : {self.dir_cdb_reports}",
            f"  dir_model_weights     : {self.dir_model_weights}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CDBConfig(runtime={self.runtime!r}, config={self.config_path})"
