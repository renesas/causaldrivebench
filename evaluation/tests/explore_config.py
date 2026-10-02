"""
Exploratory script for CDBConfig.

Run from the repo root or from evaluation/:

    python evaluation/tests/explore_config.py
    python evaluation/tests/explore_config.py --config /path/to/eval_config.yaml
    python evaluation/tests/explore_config.py --runtime docker
    python evaluation/tests/explore_config.py --dataset argoverse2
    python evaluation/tests/explore_config.py --frame raw_data/nuscenes/samples/CAM_FRONT/n008.jpg

Each section is printed with a clear header so you can read the output top-to-bottom
and get a complete picture of what CDBConfig resolves in your current environment.
"""

import argparse
import sys
from pathlib import Path

# Allow running from any working directory without installing the package.
_EVAL_ROOT = Path(__file__).parents[1]
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT.parent))

from evaluation.common.utils.config_reader import CDBConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def _row(label: str, value) -> None:
    print(f"  {label:<30} {value}")


def _check_exists(path: Path) -> str:
    return "EXISTS" if path.exists() else "MISSING"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_identity(cfg: CDBConfig) -> None:
    _header("Identity")
    _row("repr", repr(cfg))
    _row("runtime", cfg.runtime)
    _row("is_docker", cfg.is_docker)
    _row("config_path", cfg.config_path)
    _row("config_path exists", _check_exists(cfg.config_path))


def section_directories(cfg: CDBConfig) -> None:
    _header("Resolved Directory Paths")
    dirs = {
        "dir_data_root":          cfg.dir_data_root,
        "dir_raw_data":           cfg.dir_raw_data,
        "dir_causal_drive_bench": cfg.dir_causal_drive_bench,
        "dir_cdb_outputs":        cfg.dir_cdb_outputs,
        "dir_cdb_reports":        cfg.dir_cdb_reports,
        "dir_model_weights":      cfg.dir_model_weights,
    }
    for name, path in dirs.items():
        _row(name, f"{path}  [{_check_exists(path)}]")


def section_dataset_bench_dirs(cfg: CDBConfig, datasets: list[str]) -> None:
    _header("Dataset Bench Directories")
    for ds in datasets:
        path = cfg.get_bench_dir(ds)
        _row(f"get_bench_dir({ds!r})", f"{path}  [{_check_exists(path)}]")


def section_evaluation_params(cfg: CDBConfig) -> None:
    _header("Evaluation Parameters")
    _row("batch_size", cfg.batch_size)
    _row("qa_types", cfg.qa_types)
    _row("use_llm_judge", cfg.use_llm_judge)
    _row("judge_config", cfg.judge_config)


def section_model_configs(cfg: CDBConfig) -> None:
    _header("Model Configs (from eval_config.yaml)")
    models_raw: dict = cfg._cfg.get("models", {})
    if not models_raw:
        print("  (no models defined)")
        return
    for model_name in models_raw:
        mc = cfg.get_model_config(model_name)
        _row(model_name, mc)


def section_frame_resolution(cfg: CDBConfig, frame_path: str) -> None:
    _header("Frame Path Resolution")
    _row("relative_path", frame_path)
    abs_path = cfg.resolve_frame_path(frame_path)
    _row("resolved", abs_path)
    _row("exists", _check_exists(abs_path))


def section_summary(cfg: CDBConfig) -> None:
    _header("Built-in Summary")
    print(cfg.summary())


def section_raw_config(cfg: CDBConfig) -> None:
    _header("Raw Config Keys")
    for key, value in cfg._cfg.items():
        _row(key, value)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore CDBConfig properties interactively.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="Explicit path to eval_config.yaml (auto-detected if omitted).",
    )
    parser.add_argument(
        "--runtime", choices=["host", "docker"],
        help="Override runtime mode (default: reads EVAL_RUNTIME env var, falls back to 'host').",
    )
    parser.add_argument(
        "--dataset", metavar="NAME", action="append", dest="datasets",
        default=["nuscenes", "argoverse2", "openscene"],
        help="Dataset name(s) to test with get_bench_dir(). May be repeated. "
             "Defaults to nuscenes, argoverse2, openscene.",
    )
    parser.add_argument(
        "--frame", metavar="REL_PATH",
        default="raw_data/nuscenes/samples/CAM_FRONT/example.jpg",
        help="Relative frame path to resolve (as stored in frames.json).",
    )
    args = parser.parse_args()

    try:
        cfg = CDBConfig(
            config_path=args.config,
            runtime=args.runtime,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    section_identity(cfg)
    section_directories(cfg)
    section_dataset_bench_dirs(cfg, args.datasets)
    section_evaluation_params(cfg)
    section_model_configs(cfg)
    section_frame_resolution(cfg, args.frame)
    section_summary(cfg)
    section_raw_config(cfg)

    print()


if __name__ == "__main__":
    main()
