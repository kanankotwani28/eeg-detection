"""Band occlusion for EEGNet: remove one frequency band from the test input (zero-phase band-stop)
and measure the drop in participant-level balanced accuracy. Explains the MODEL, not the disorder."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.eval.metrics import balanced_accuracy
from src.train.torch_core import PData, predict_participants

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def bandstop(x: np.ndarray, lo: float, hi: float, sfreq: float) -> np.ndarray:
    lo = max(lo, 0.5)
    sos = butter(4, [lo, hi], btype="bandstop", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def band_occlusion(model, data: PData, threshold: float, sfreq: float, device="cpu") -> dict:
    base = balanced_accuracy(data.y, (predict_participants(model, data, device) >= threshold).astype(int))
    out = {"baseline_ba": base}
    for band, (lo, hi) in BANDS.items():
        occl = PData(data.ids, data.y, [bandstop(s, lo, hi, sfreq) for s in data.samples])
        ba = balanced_accuracy(data.y, (predict_participants(model, occl, device) >= threshold).astype(int))
        out[band] = base - ba
    return out
