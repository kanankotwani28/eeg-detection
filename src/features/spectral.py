"""Track A features: Welch PSD per epoch -> log absolute band power (19 ch x 4 bands = 76) + FAA,
averaged per participant. Gamma is excluded (muscle contamination near the 40 Hz filter edge).

FAA = ln(alpha F4) - ln(alpha F3). It is included as a model input, not as a biomarker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

DEFAULT_BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def epoch_band_power(X: np.ndarray, sfreq: float, bands: dict = DEFAULT_BANDS, win_s: float = 2.0,
                     overlap: float = 0.5) -> np.ndarray:
    """X (n_ep, n_ch, n_t) in uV -> absolute band power (n_ep, n_ch, n_bands) in uV^2."""
    nper = int(win_s * sfreq)
    f, p = welch(np.asarray(X, dtype=np.float64), fs=sfreq, nperseg=nper, noverlap=int(nper * overlap), axis=-1)
    df = f[1] - f[0]
    out = []
    for lo, hi in bands.values():
        m = (f >= lo) & (f < hi)
        out.append(p[..., m].sum(axis=-1) * df)
    return np.stack(out, axis=-1)


def feature_names(channels: list[str], bands: dict = DEFAULT_BANDS, include_faa: bool = True) -> list[str]:
    names = [f"{b}_{c}" for b in bands for c in channels]
    return names + (["FAA"] if include_faa else [])


def epoch_features(X, sfreq, channels, bands=DEFAULT_BANDS, include_faa=True, **kw) -> np.ndarray:
    bp = epoch_band_power(X, sfreq, bands, **kw)  # (n, ch, b)
    logbp = np.log(np.maximum(bp, 1e-12))
    feats = logbp.transpose(0, 2, 1).reshape(len(X), -1)  # band-major ordering, matches feature_names
    if include_faa:
        ai = list(bands).index("alpha")
        faa = logbp[:, channels.index("F4"), ai] - logbp[:, channels.index("F3"), ai]
        feats = np.concatenate([feats, faa[:, None]], axis=1)
    return feats


def participant_features(X, rows_by_pid: dict, sfreq: float, channels: list[str], bands=DEFAULT_BANDS,
                         include_faa: bool = True, max_epochs: int | None = None, win_s=2.0, overlap=0.5) -> pd.DataFrame:
    """One row per participant: mean over that participant's epochs (no cross-subject statistics)."""
    names = feature_names(channels, bands, include_faa)
    recs = []
    for pid, rows in sorted(rows_by_pid.items()):
        r = rows[:max_epochs] if max_epochs else rows
        f = epoch_features(np.asarray(X[r]), sfreq, channels, bands, include_faa, win_s=win_s, overlap=overlap)
        recs.append(pd.Series(f.mean(axis=0), index=names, name=pid))
    df = pd.DataFrame(recs)
    df.index.name = "participant_id"
    return df


def feature_table(dataset: str, cond: str, model_cfg: dict, max_epochs: int | None = None) -> pd.DataFrame:
    """Load (or compute and cache) the participant-level feature table for a cached epoch set."""
    from src.preprocess.pipeline import load_cache, participant_epochs
    from src.utils.hashing import sha256_obj
    from src.utils.paths import CACHE

    X, idx, meta = load_cache(dataset, cond)
    key = sha256_obj({"meta": meta["config_hash"], "bands": model_cfg["bands"], "faa": model_cfg["include_faa"],
                      "max": max_epochs, "w": model_cfg["welch_window_s"], "o": model_cfg["welch_overlap"]})[:12]
    path = CACHE / f"features_{dataset}_{cond}_{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = participant_features(X, participant_epochs(X, idx), meta["sfreq"], meta["channels"],
                              {k: tuple(v) for k, v in model_cfg["bands"].items()}, model_cfg["include_faa"],
                              max_epochs, model_cfg["welch_window_s"], model_cfg["welch_overlap"])
    df.to_parquet(path)
    return df


def participant_relative(feat: pd.DataFrame, bands=DEFAULT_BANDS) -> pd.DataFrame:
    """Data-independent, per-participant re-expression of log band power (no cross-subject statistics, so it
    costs no privacy budget): for each band, log power relative to the participant's own mean over channels,
    plus each band's level relative to the participant's mean over bands. FAA is kept as is. Drops only the
    participant's overall log-power offset (largely amplifier/reference dependent)."""
    out = {}
    levels = {}
    for b in bands:
        cols = [c for c in feat.columns if c.startswith(f"{b}_")]
        if not cols:
            continue
        m = feat[cols].mean(axis=1)
        levels[b] = m
        for c in cols:
            out[c] = feat[c] - m
    if levels:
        g = pd.DataFrame(levels)
        for b in g.columns:
            out[f"bandlevel_{b}"] = g[b] - g.mean(axis=1)
    if "FAA" in feat.columns:
        out["FAA"] = feat["FAA"]
    res = pd.DataFrame(out, index=feat.index)
    res.index.name = feat.index.name
    return res
