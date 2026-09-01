from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """Load configuration and apply the intentionally small environment surface."""
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = deepcopy(config)
    config["runtime"]["mode"] = os.getenv("A2R_RUNTIME_MODE", config["runtime"]["mode"])
    config["llm"]["ollama_base_url"] = os.getenv(
        "A2R_OLLAMA_BASE_URL", config["llm"]["ollama_base_url"]
    )
    return config


def project_path(value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else DEFAULT_CONFIG_PATH.parent / value
