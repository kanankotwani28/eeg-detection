"""Privacy accounting with the Opacus RDP accountant.

The privacy unit is the PARTICIPANT. Each client's mechanism is a composition of
(noise_multiplier, sample_rate=q, steps) Gaussian-subsampled steps, plus optional one-off Gaussian
releases (e.g. DP feature-scaler statistics, sample_rate=1). Each participant belongs to exactly one
client, so the max over clients is the reported participant-level epsilon.

`epoch_level_epsilon` reproduces what standard Opacus would report if each EEG epoch were treated as
an independent record. It is NOT a participant guarantee and is reported only as a labelled comparison.
"""
from __future__ import annotations

import math

from opacus.accountants import RDPAccountant


def epsilon(history: list[tuple[float, float, int]], delta: float) -> float:
    hist = [(float(s), float(q), int(t)) for s, q, t in history if t > 0]
    if not hist:
        return 0.0
    if any(s <= 0 for s, _, _ in hist):
        return math.inf
    acc = RDPAccountant()
    acc.history = hist
    return float(acc.get_epsilon(delta=delta))


def calibrate_sigma(target_eps: float, delta: float, q: float, steps: int,
                    extra: list[tuple[float, float, int]] | None = None, tol: float = 1e-3) -> float:
    """Smallest noise multiplier sigma such that epsilon([(sigma,q,steps)] + extra) <= target_eps."""
    if math.isinf(target_eps):
        return 0.0
    extra = extra or []
    if epsilon(extra, delta) >= target_eps:
        raise ValueError("The fixed extra releases alone exceed the target epsilon")
    lo, hi = 1e-2, 1.0
    while epsilon([(hi, q, steps)] + extra, delta) > target_eps:
        hi *= 2
        if hi > 1e4:
            raise ValueError("Cannot reach target epsilon")
    while hi - lo > tol * hi:
        mid = (lo + hi) / 2
        if epsilon([(mid, q, steps)] + extra, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


def epoch_level_epsilon(sigma: float, expected_batch_epochs: float, n_epochs_total: int, steps: int,
                        delta: float) -> float:
    """What a per-sample (per-epoch) accountant would report. NOT a participant-level guarantee."""
    q_e = min(1.0, expected_batch_epochs / max(n_epochs_total, 1))
    return epsilon([(sigma, q_e, steps)], delta)
