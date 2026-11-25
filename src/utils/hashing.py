"""Stable hashes for configs, split files and input tables."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode())


def git_commit() -> str:
    import subprocess

    from src.utils.paths import ROOT

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return out.stdout.strip() + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"
