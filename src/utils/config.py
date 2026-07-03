"""Configuration management using YAML files.

Provides simple load/save helpers and a merge utility so CLI arguments can
override YAML defaults without boilerplate.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dict of configuration values.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save a configuration dict to a YAML file.

    Args:
        config: Configuration dictionary.
        path: Destination path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dicts, later dicts taking priority.

    Args:
        *configs: Configuration dicts in priority order (lowest to highest).

    Returns:
        Merged configuration dict.
    """
    result: Dict[str, Any] = {}
    for cfg in configs:
        result.update(cfg)
    return result
