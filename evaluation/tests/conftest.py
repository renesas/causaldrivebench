import os
import pytest
from pathlib import Path


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring live API access (skipped when env vars are absent)",
    )
    _load_dev_env_if_present()


def _load_dev_env_if_present() -> None:
    """
    Load API keys from a dev.env file if present, without overwriting existing env vars.

    Checks the following locations in order:
      1. $DEV_ENV_PATH (explicit override)
      2. evaluation/dev.env (repo-local, gitignored)

    The dev.env file uses KEY=VALUE or KEY="VALUE" syntax (bash-style).
    Only sets variables that are not already present in the environment.
    Never commits secrets — dev.env is listed in .gitignore.

    To run integration tests with API keys:
        DEV_ENV_PATH=/path/to/your/dev.env pytest evaluation/tests/ -m integration
    """
    candidates = [
        os.environ.get("DEV_ENV_PATH"),
        str(Path(__file__).parents[1] / "dev.env"),
    ]

    for path in candidates:
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        break
