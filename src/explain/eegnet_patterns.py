"""EEGNet spatial patterns of the depthwise (spatial) filters, via the Haufe et al. (2014) transform
a = Cov(x) w / var(w^T x), estimated on training-set epochs. Explains the MODEL, not the disorder."""
from __future__ import annotations

import numpy as np
import pandas as pd


def spatial_patterns(model, X_epochs: np.ndarray) -> np.ndarray:
    """model: EEGNet; X_epochs (n, C, T) z-scored. Returns (n_filters, C) patterns (unit-max abs)."""
    W = model.conv_spatial.weight.detach().cpu().numpy()[:, 0, :, 0]  # (F1*D, C)
    Xc = X_epochs.transpose(1, 0, 2).reshape(X_epochs.shape[1], -1)
    Xc = Xc - Xc.mean(axis=1, keepdims=True)
    cov = (Xc @ Xc.T) / Xc.shape[1]
    A = []
    for w in W:
        a = cov @ w / max(float(w @ cov @ w), 1e-12)
        A.append(a / (np.abs(a).max() + 1e-12))
    return np.array(A)


def patterns_table(A: np.ndarray, channels: list[str], fold, seed) -> pd.DataFrame:
    return pd.DataFrame([{"filter": i, "channel": c, "value": float(A[i, j]), "fold": fold, "seed": seed}
                         for i in range(A.shape[0]) for j, c in enumerate(channels)])


def save_topomaps(A: np.ndarray, channels: list[str], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    from src.utils.paths import standard_montage

    info = mne.create_info(channels, 256.0, "eeg")
    info.set_montage(standard_montage())
    n = A.shape[0]
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(2 * np.ceil(n / 2), 4.4))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i < n:
            mne.viz.plot_topomap(A[i], info, axes=ax, show=False, contours=0)
            ax.set_title(f"filter {i}", fontsize=8)
        else:
            ax.axis("off")
    fig.suptitle("EEGNet spatial patterns (explains the model, not the disorder)", fontsize=9)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
