"""Central path definitions. Everything derived from the repository root."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
ARTIFACTS = Path(os.environ.get("EEGFL_ARTIFACTS", ROOT / "artifacts"))
CACHE = ARTIFACTS / "cache"
RESULTS = ARTIFACTS / "results"
DATA = ROOT / "data"
SPLITS_FILE = ARTIFACTS / "splits.json"
PARTICIPANTS_FILE = ARTIFACTS / "participants.parquet"
MLRUNS = Path(os.environ.get("EEGFL_MLRUNS", ROOT / "mlruns"))


def ensure_dirs() -> None:
    for p in (ARTIFACTS, CACHE, RESULTS):
        p.mkdir(parents=True, exist_ok=True)


def standard_montage():
    """10-20 montage; MNE >= 1.13 renamed 'standard_1020' to 'colin27_1020'."""
    import mne

    names = mne.channels.get_builtin_montages()
    return mne.channels.make_standard_montage("colin27_1020" if "colin27_1020" in names else "standard_1020")
