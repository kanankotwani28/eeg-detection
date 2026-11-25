"""Phase 2: per-recording preprocessing. No statistic is computed across participants.

Order: load -> drop non-EEG -> rename -> pick 19 channels (fixed order) -> V->uV -> band-pass FIR
zero-phase -> notch -> (within-recording bad-channel interpolation, counted) -> average reference ->
resample -> fixed-length epochs -> amplitude/flat rejection -> keep first N clean epochs.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.hashing import sha256_obj
from src.utils.logging import get_logger
from src.utils.paths import CACHE, PARTICIPANTS_FILE, ensure_dirs, standard_montage

log = get_logger(__name__)
CANON = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8",
         "O1", "O2", "FC3", "FCz", "FC4", "CP3", "CPz", "CP4", "Oz", "T3", "T4", "T5", "T6", "A1", "A2"]
_CANON_UPPER = {c.upper(): c for c in CANON}


def clean_channel_name(name: str) -> str:
    """'EEG Fp1-LE' -> 'Fp1'; 'FP1' -> 'Fp1'."""
    s = re.sub(r"^(EEG|EOG|ECG)\s+", "", name.strip(), flags=re.I)
    s = re.sub(r"-(LE|REF|RE|AV|A1|A2|M1|M2|AVG)$", "", s, flags=re.I)
    return _CANON_UPPER.get(s.upper(), s)


@dataclass
class RecordingResult:
    epochs: np.ndarray                     # (n_kept, n_ch, n_samples) float32, microvolts
    rejected: np.ndarray                   # (n_total,) bool
    reasons: list[str]
    kept_mask: np.ndarray                  # (n_total,) bool: stored in `epochs` (clean and within cap)
    n_interpolated: int = 0
    bad_channels: list[str] = field(default_factory=list)
    sfreq: float = 256.0
    ch_names: list[str] = field(default_factory=list)


def pick_and_rename(raw, cfg: dict):
    mapping = {}
    for ch in raw.ch_names:
        c = clean_channel_name(ch)
        mapping[ch] = cfg["rename"].get(c, c)
    # avoid duplicate targets: keep the first occurrence
    seen, drop = set(), []
    for ch, tgt in mapping.items():
        if tgt in seen:
            drop.append(ch)
        seen.add(tgt)
    if drop:
        raw.drop_channels(drop)
        mapping = {k: v for k, v in mapping.items() if k not in drop}
    raw.rename_channels(mapping)
    missing = [c for c in cfg["channels"] if c not in raw.ch_names]
    if missing:
        raise ValueError(f"Recording lacks required channels {missing}; interpolation of missing channels is not allowed")
    raw.set_channel_types({c: "eeg" for c in cfg["channels"]}, verbose="ERROR")
    raw.pick(cfg["channels"])  # drops every non-EEG / extra channel
    raw.reorder_channels(cfg["channels"])
    return raw


def detect_bad_channels(x_uv: np.ndarray, cfg: dict) -> list[int]:
    sd = x_uv.std(axis=1)
    bad = set(np.where(sd < cfg.get("bad_channel_flat_uv", 0.5))[0].tolist())
    lv = np.log(np.maximum(sd, 1e-6) ** 2)
    med = np.median(lv)
    mad = np.median(np.abs(lv - med)) * 1.4826 + 1e-9
    bad |= set(np.where((lv - med) / mad > cfg.get("bad_channel_z", 5.0))[0].tolist())
    return sorted(bad)


def epoch_and_reject(x: np.ndarray, sfreq: float, cfg: dict):
    n_len = int(round(cfg["epoch_len_s"] * sfreq))
    step = int(round(n_len * (1 - cfg.get("epoch_overlap", 0.0))))
    starts = np.arange(0, x.shape[1] - n_len + 1, step)
    ep = np.stack([x[:, s:s + n_len] for s in starts]) if len(starts) else np.zeros((0, x.shape[0], n_len))
    ptp = ep.max(axis=2) - ep.min(axis=2) if len(ep) else np.zeros((0, x.shape[0]))
    too_big = (ptp > cfg["reject_ptp_uv"]).any(axis=1)
    flat = (ptp < cfg["flat_uv"]).any(axis=1)
    rejected = too_big | flat
    reasons = ["ptp" if b else ("flat" if f else "") for b, f in zip(too_big, flat, strict=True)]
    clean_idx = np.where(~rejected)[0][: cfg["max_epochs_per_person"]]
    kept = np.zeros(len(ep), bool)
    kept[clean_idx] = True
    return ep, rejected, reasons, kept


def preprocess_raw(raw, cfg: dict) -> RecordingResult:

    raw = raw.copy().load_data(verbose="ERROR")
    raw = pick_and_rename(raw, cfg)
    sf = raw.info["sfreq"]
    lo, hi = cfg["bandpass"]
    raw.filter(lo, hi, method="fir", phase="zero", fir_design="firwin", verbose="ERROR")
    if cfg.get("notch") and cfg["notch"] < sf / 2:
        raw.notch_filter(cfg["notch"], method="fir", phase="zero", verbose="ERROR")
    bads: list[str] = []
    if cfg.get("interpolate_bads", True):
        idx = detect_bad_channels(raw.get_data() * 1e6, cfg)
        bads = [raw.ch_names[i] for i in idx]
        if bads and len(bads) <= 3:
            raw.set_montage(standard_montage(), on_missing="ignore", verbose="ERROR")
            raw.info["bads"] = bads
            raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    if cfg.get("ica"):
        raw = _ica_clean(raw)
    if cfg.get("reference", "average") == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.resample(cfg["resample"], verbose="ERROR")
    x = raw.get_data() * 1e6  # V -> microvolts
    ep, rej, reasons, kept = epoch_and_reject(x, raw.info["sfreq"], cfg)
    return RecordingResult(ep[kept].astype(np.float32), rej, reasons, kept, len(bads) if len(bads) <= 3 else 0,
                           bads, raw.info["sfreq"], list(raw.ch_names))


def _ica_clean(raw):
    """E15 sensitivity only: ICA with frontal-proxy EOG detection (Fp1/Fp2)."""
    import mne

    ica = mne.preprocessing.ICA(n_components=15, random_state=0, max_iter="auto", verbose="ERROR")
    ica.fit(raw, verbose="ERROR")
    idx, _ = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2"], verbose="ERROR")
    ica.exclude = idx[:2]
    return ica.apply(raw, verbose="ERROR")


def preprocess_file(path: str | Path, cfg: dict) -> RecordingResult:
    from src.data.tdbrain import load_raw

    return preprocess_raw(load_raw(path, preload=True), cfg)


def cache_paths(dataset: str, cond: str) -> dict:
    return {"epochs": CACHE / f"epochs_{dataset}_{cond}.npy",
            "index": CACHE / f"epoch_index_{dataset}_{cond}.parquet",
            "meta": CACHE / f"meta_{dataset}_{cond}.json"}


def run_dataset(items: list[tuple[str, str]], cfg: dict, dataset: str, cond: str, n_jobs: int = 1,
                force: bool = False) -> dict:
    """items: [(participant_id, path)]. Writes the epoch cache and returns the meta dict."""
    from joblib import Parallel, delayed

    ensure_dirs()
    cp = cache_paths(dataset, cond)
    cfg_hash = sha256_obj(cfg)
    if cp["meta"].exists() and not force:
        meta = json.loads(cp["meta"].read_text())
        if meta.get("config_hash") == cfg_hash and meta.get("participants") == sorted(p for p, _ in items):
            log.info("Cache up to date for %s/%s", dataset, cond)
            return meta

    def one(pid, path):
        try:
            return pid, preprocess_file(path, cfg), None
        except Exception as e:  # recorded as exclusion, never silently dropped
            return pid, None, f"{type(e).__name__}: {e}"

    results = Parallel(n_jobs=n_jobs)(delayed(one)(p, f) for p, f in items)
    arrays, rows, per_person = [], [], {}
    row = 0
    for pid, res, err in sorted(results, key=lambda r: r[0]):
        if res is None:
            per_person[pid] = {"n_total": 0, "n_clean": 0, "n_kept": 0, "excluded": True, "error": err,
                               "n_interpolated": 0, "bad_channels": []}
            continue
        n_clean = int((~res.rejected).sum())
        excluded = n_clean < cfg["min_epochs_per_person"]
        per_person[pid] = {"n_total": int(len(res.rejected)), "n_clean": n_clean, "n_kept": 0 if excluded else
                           int(res.kept_mask.sum()), "excluded": bool(excluded), "error": None,
                           "n_interpolated": res.n_interpolated, "bad_channels": res.bad_channels}
        k_iter = iter(range(row, row + int(res.kept_mask.sum())))
        for e in range(len(res.rejected)):
            stored = bool(res.kept_mask[e]) and not excluded
            rows.append({"participant_id": pid, "epoch": e, "rejected": bool(res.rejected[e]),
                         "reason": res.reasons[e], "kept": stored, "array_row": next(k_iter) if stored else -1})
        if not excluded:
            arrays.append(res.epochs)
            row += len(res.epochs)
    n_ch, n_s = len(cfg["channels"]), int(round(cfg["epoch_len_s"] * cfg["resample"]))
    X = np.concatenate(arrays).astype(np.float32) if arrays else np.zeros((0, n_ch, n_s), np.float32)
    np.save(cp["epochs"], X)
    pd.DataFrame(rows).to_parquet(cp["index"], index=False)
    meta = {"dataset": dataset, "cond": cond, "config_hash": cfg_hash, "shape": list(X.shape),
            "participants": sorted(p for p, _ in items), "per_person": per_person, "channels": cfg["channels"],
            "sfreq": cfg["resample"]}
    cp["meta"].write_text(json.dumps(meta, indent=1))
    log.info("%s/%s: %s epochs from %d people (%d excluded)", dataset, cond, X.shape,
             len(items), sum(v["excluded"] for v in per_person.values()))
    return meta


def load_cache(dataset: str, cond: str, mmap: bool = True):
    cp = cache_paths(dataset, cond)
    X = np.load(cp["epochs"], mmap_mode="r" if mmap else None)
    idx = pd.read_parquet(cp["index"])
    meta = json.loads(cp["meta"].read_text())
    return X, idx, meta


def participant_epochs(X, idx: pd.DataFrame) -> dict[str, np.ndarray]:
    """participant_id -> array rows of kept epochs."""
    k = idx[idx["kept"]]
    return {p: g["array_row"].to_numpy() for p, g in k.groupby("participant_id")}


def tdbrain_items(cond: str = "restEC") -> list[tuple[str, str]]:
    df = pd.read_parquet(PARTICIPANTS_FILE)
    sel = df[((df["T1_label"] >= 0) | (df["T2_label"] >= 0)) & df[f"has_{cond}"]]
    return list(zip(sel["participant_id"], sel[f"path_{cond}"], strict=True))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="restEC")
    ap.add_argument("--dataset", default="tdbrain")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default="preprocess")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    run_dataset(tdbrain_items(a.cond), cfg, a.dataset, a.cond, cfg.get("n_jobs", 1), a.force)
    from src.preprocess.qc import write_qc

    write_qc(a.dataset, a.cond)


if __name__ == "__main__":
    main()
