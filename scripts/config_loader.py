"""
Shared config loader for the pipeline scripts.

Loads config.yaml from the repo root and resolves ${ENV_VAR} references.
CLI arguments always override config values.
"""
import os
import re
import yaml
from pathlib import Path


def _resolve_env_vars(value):
    """Replace ${VAR} with environment variable values."""
    if not isinstance(value, str):
        return value
    def replacer(match):
        var = match.group(1)
        return os.environ.get(var, match.group(0))
    return re.sub(r'\$\{(\w+)\}', replacer, value)


def _resolve_dict(d):
    """Recursively resolve env vars in a dict."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _resolve_dict(v)
        else:
            out[k] = _resolve_env_vars(v)
    return out


def find_config():
    """Find config.yaml by walking up from this script's location."""
    d = Path(__file__).resolve().parent
    for _ in range(5):
        cfg = d / "config.yaml"
        if cfg.exists():
            return cfg
        d = d.parent
    return None


def load_config(dataset=None):
    """
    Load config.yaml and return the resolved config dict.

    Args:
        dataset: If provided, returns only that dataset's section
                 merged with top-level keys (api_key, model, reasoning).

    Returns:
        dict with config values, or None if config.yaml not found.
    """
    cfg_path = find_config()
    if cfg_path is None:
        return None

    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return None

    cfg = _resolve_dict(raw)

    if dataset:
        ds = cfg.get(dataset, {})
        # Merge top-level keys into dataset config
        ds["api_key"] = cfg.get("api_key")
        ds["model"] = cfg.get("model", "gpt-5.2")
        ds["reasoning"] = cfg.get("reasoning", "none")
        ds["sampling_step"] = cfg.get("sampling_step", 6)
        ds["history_frames"] = cfg.get("history_frames", 3)
        ds["future_frames"] = cfg.get("future_frames", 6)
        return ds

    return cfg


def apply_config(args, dataset):
    """
    Fill in missing argparse values from config.yaml.

    CLI args always win. Only fills in args that are None or at their default.

    Args:
        args: argparse.Namespace
        dataset: dataset key in config.yaml (nuscenes, openscene, argoverse2)

    Returns:
        Modified args namespace.
    """
    cfg = load_config(dataset)
    if cfg is None:
        return args

    # Map config keys → argparse attribute names per dataset
    key_map = {
        "nuscenes": {
            "data": "data",
            "version": "version",
            "maps": "maps",
            "records": "records",
            "num_scenes": "num_scenes",
        },
        "openscene": {
            "logs": "logs",
            "blobs": "blobs",
            "maps": "maps",
            "records": "records",
            "num_scenes": "num_scenes",
        },
        "argoverse2": {
            "data": "data",
            "records": "records",
            "num_scenes": "num_scenes",
        },
        "graph": {
            "model": "model",
            "reasoning": "reasoning",
        },
        "qa": {
            "model": "model",
            "reasoning": "reasoning",
        },
    }

    mapping = key_map.get(dataset, {})

    for cfg_key, arg_key in mapping.items():
        if cfg_key in cfg and hasattr(args, arg_key):
            current = getattr(args, arg_key)
            # Only override if arg is None (not provided via CLI)
            if current is None:
                setattr(args, arg_key, cfg[cfg_key])

    # Set API key from config if not in env
    if cfg.get("api_key") and not os.environ.get("OPENAI_API_KEY"):
        api_key = cfg["api_key"]
        if not api_key.startswith("${"):  # Skip unresolved env refs
            os.environ["OPENAI_API_KEY"] = api_key

    return args
