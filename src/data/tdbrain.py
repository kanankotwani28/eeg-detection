"""TDBRAIN access: resolve the data root (local / kaggle / synthetic), find recordings, load raw EEG.

TDBRAIN is distributed under a data-use agreement that forbids redistribution. Nothing here copies
data anywhere except the local cache in artifacts/cache (gitignored).
"""
from __future__ import annotations

import os
from pathlib import Path

from src.utils.logging import get_logger
from src.utils.paths import ROOT

log = get_logger(__name__)

EEG_EXTS = (".vhdr", ".edf", ".bdf", ".set", ".fif")
NON_EEG_HINTS = ("VPVA", "VNVB", "HPHL", "HNHR", "Erbs", "OrbOcc", "Mass", "EOG", "ECG", "EMG", "Status")


def resolve_root(data_cfg: dict) -> Path:
    t = data_cfg["tdbrain"]
    env = os.environ.get("TDBRAIN_PATH")
    p = Path(env) if env else Path(t["local_path"])
    if not p.is_absolute():
        p = ROOT / p
    return p


def find_participants_file(root: Path) -> Path:
    """participants.tsv (BIDS) or the V3 release's TDBRAIN_participants_V3.xlsx, in the root or its parent.
    macOS resource-fork files ('._*') and Excel lock files ('~$*') are ignored."""
    cands = [root / "participants.tsv"]
    for d in (root, root.parent):
        for pat in ("*participants*.tsv", "*participants*.xlsx", "*participants*.csv"):
            cands += sorted(d.glob(pat))
    for c in cands:
        if c.exists() and not c.name.startswith(("._", "~$")):
            return c
    raise FileNotFoundError(
        f"No participants file (participants.tsv or TDBRAIN_participants_*.xlsx) in {root} or its parent. Set "
        "tdbrain.local_path in configs/data.yaml or TDBRAIN_PATH to the folder that holds sub-*/.")


def read_participants(path: Path):
    import pandas as pd

    if path.suffix == ".xlsx":
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", dtype=str)


def verify_local(root: Path, condition: str = "restEC") -> None:
    """Check a participants file exists and at least one restEC recording is readable."""
    find_participants_file(root)
    for f in sorted(root.glob(f"sub-*/ses-*/eeg/*{condition}*")):
        if f.suffix in EEG_EXTS and not f.name.startswith("._"):
            raw = load_raw(f, preload=False)
            log.info("Readable %s file: %s (%d channels, %.0f Hz)", condition, f.name, len(raw.ch_names),
                     raw.info["sfreq"])
            return
    raise FileNotFoundError(f"No readable sub-*/ses-*/eeg/*{condition}* recording under {root}")


def find_recording(root: Path, sub: str, ses: str, condition: str) -> Path | None:
    eeg_dir = root / sub / ses / "eeg"
    if not eeg_dir.exists():
        return None
    for ext in EEG_EXTS:
        hits = sorted(eeg_dir.glob(f"*task-{condition}*{ext}")) or sorted(eeg_dir.glob(f"*{condition}*{ext}"))
        hits = [h for h in hits if not h.name.startswith("._")]
        if hits:
            return hits[0]
    return None


def list_sessions(root: Path, sub: str) -> list[str]:
    return sorted(p.name for p in (root / sub).glob("ses-*") if p.is_dir())


def load_raw(path: str | Path, preload: bool = True):
    import mne

    path = Path(path)
    kw = dict(preload=preload, verbose="ERROR")
    if path.suffix == ".vhdr":
        return mne.io.read_raw_brainvision(path, **kw)
    if path.suffix in (".edf", ".bdf"):
        return mne.io.read_raw_edf(path, **kw) if path.suffix == ".edf" else mne.io.read_raw_bdf(path, **kw)
    if path.suffix == ".set":
        return mne.io.read_raw_eeglab(path, **kw)
    return mne.io.read_raw_fif(path, **kw)
