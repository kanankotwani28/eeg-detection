"""YAML config loading with simple deep-merge and `extends:` support."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import CONFIGS


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml."""
    return load_yaml(CONFIGS / f"{name}.yaml")


def load_experiment(exp_id: str, overrides: dict | None = None) -> dict:
    """Experiment config = base configs merged with configs/experiments/<id>.yaml (+ extends chain)."""
    path = CONFIGS / "experiments" / f"{exp_id}.yaml"
    exp = load_yaml(path)
    chain = [exp]
    while "extends" in chain[-1]:
        chain.append(load_yaml(CONFIGS / "experiments" / f"{chain[-1]['extends']}.yaml"))
    merged: dict[str, Any] = {}
    for c in reversed(chain):
        merged = deep_merge(merged, {k: v for k, v in c.items() if k != "extends"})
    base = {
        "data": load_config("data"),
        "preprocess": load_config("preprocess"),
        "models": load_config("models"),
        "fl": load_config("fl"),
        "dp": load_config("dp"),
        "audit": load_config("audit"),
    }
    cfg = deep_merge(base, merged)
    cfg = deep_merge(cfg, overrides or {})
    cfg["id"] = exp_id
    return cfg
