"""Participant-level bootstrap CIs (2,000 resamples, stratified by class) and paired bootstrap of
differences. Vectorised over resamples."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _strat_indices(y: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for c in (0, 1):
        m = np.where(y == c)[0]
        if len(m):
            parts.append(m[rng.integers(0, len(m), size=(n_boot, len(m)))])
    return np.concatenate(parts, axis=1)


def _ba(y, yhat):
    """y, yhat: (B, n) -> (B,)"""
    pos, neg = y == 1, y == 0
    sens = (yhat & pos).sum(1) / np.maximum(pos.sum(1), 1)
    spec = (~yhat & neg).sum(1) / np.maximum(neg.sum(1), 1)
    has = (pos.sum(1) > 0) & (neg.sum(1) > 0)
    return np.where(has, (sens + spec) / 2, np.where(pos.sum(1) > 0, sens, spec))


def _auc(y, p):
    r = rankdata(p, axis=1)
    n1 = (y == 1).sum(1)
    n0 = (y == 0).sum(1)
    u = (r * (y == 1)).sum(1) - n1 * (n1 + 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((n1 > 0) & (n0 > 0), u / (n1 * n0), np.nan)


def _metric(name, y, p, thr):
    if name == "balanced_accuracy":
        return _ba(y, p >= thr)
    if name == "auc":
        return _auc(y, p)
    if name == "sensitivity":
        pos = y == 1
        return np.where(pos.sum(1) > 0, ((p >= thr) & pos).sum(1) / np.maximum(pos.sum(1), 1), np.nan)
    if name == "specificity":
        neg = y == 0
        return np.where(neg.sum(1) > 0, ((p < thr) & neg).sum(1) / np.maximum(neg.sum(1), 1), np.nan)
    raise ValueError(name)


def point(y, p, thr, metric):
    return float(_metric(metric, np.asarray(y)[None].astype(int), np.asarray(p, float)[None], thr)[0])


def bootstrap_ci(y, p, thr, metric="balanced_accuracy", n_boot=2000, alpha=0.05, seed=0):
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    if len(y) == 0:
        return np.nan, np.nan, np.nan
    idx = _strat_indices(y, n_boot, np.random.default_rng(seed))
    vals = _metric(metric, y[idx], p[idx], thr)
    vals = vals[~np.isnan(vals)]
    pt = point(y, p, thr, metric)
    if not len(vals):
        return pt, np.nan, np.nan
    return pt, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(y, p_a, p_b, thr_a, thr_b, metric="balanced_accuracy", n_boot=2000, seed=0):
    """Difference A - B on the same participants. Returns (diff, lo, hi, fraction of resamples with diff <= 0)."""
    y = np.asarray(y).astype(int)
    p_a, p_b = np.asarray(p_a, float), np.asarray(p_b, float)
    idx = _strat_indices(y, n_boot, np.random.default_rng(seed))
    d = _metric(metric, y[idx], p_a[idx], thr_a) - _metric(metric, y[idx], p_b[idx], thr_b)
    d = d[~np.isnan(d)]
    pt = point(y, p_a, thr_a, metric) - point(y, p_b, thr_b, metric)
    return float(pt), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float(np.mean(d <= 0))
