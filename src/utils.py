"""Shared helpers: config loading, seeding, logging."""
import random
import logging
from pathlib import Path

import numpy as np
import yaml


def load_config(config_path: str, experiment: str | None = None) -> dict:
    """Load the YAML config, optionally overlaying a named experiment's overrides.

    Experiment overrides are shallow-merged into the base config, so an
    experiment only needs to specify the keys it changes.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if experiment:
        experiments = config.get("experiments", {})
        if experiment not in experiments:
            raise ValueError(
                f"Unknown experiment '{experiment}'. Available: {list(experiments.keys())}"
            )
        overrides = experiments[experiment]
        config = _deep_merge(config, overrides)
        config["run_name"] = experiment
    else:
        config["run_name"] = "baseline"

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
