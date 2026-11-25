"""Participant-level metrics. Epoch probabilities are averaged per participant before scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score, roc_curve


def aggregate_participants(pids: np.ndarray, probs: np.ndarray) -> pd.Series:
    return pd.Series(probs).groupby(np.asarray(pids)).mean()


def balanced_accuracy(y, yhat) -> float:
    y, yhat = np.asarray(y), np.asarray(yhat)
    rec = [np.mean(yhat[y == c] == c) for c in (0, 1) if np.any(y == c)]
    return float(np.mean(rec)) if rec else float("nan")


def best_threshold(y, p) -> float:
    """Threshold maximising balanced accuracy (fitted on validation participants only)."""
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return 0.5
    cand = np.unique(p)
    mids = np.concatenate([[0.0], (cand[:-1] + cand[1:]) / 2, [1.0]])
    bas = [balanced_accuracy(y, (p >= t).astype(int)) for t in mids]
    best = np.max(bas)
    good = mids[np.isclose(bas, best)]
    return float(good[np.argmin(np.abs(good - 0.5))])  # among ties prefer closest to 0.5


def ece(y, p, n_bins=10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def all_metrics(y, p, thr: float) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    yhat = (p >= thr).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "balanced_accuracy": balanced_accuracy(y, yhat),
        "auc": auc(y, p),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, yhat)) if len(np.unique(yhat)) > 1 else 0.0,
        "ece": ece(y, p),
        "threshold": float(thr),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def roc_points(y, p) -> pd.DataFrame:
    fpr, tpr, thr = roc_curve(y, p)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})
