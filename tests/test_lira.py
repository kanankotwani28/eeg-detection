import numpy as np

from src.privacy.lira import IGNORE, IN, OUT, lira_likelihood_ratios, summarise


def _status(S, N, seed):
    r = np.random.default_rng(seed)
    st = np.where(r.random((S, N)) < 0.5, IN, OUT)
    st[r.random((S, N)) < 0.05] = IGNORE
    return st


def test_trivially_separable_members_auc_high():
    S, N = 16, 200
    st = _status(S, N, 0)
    r = np.random.default_rng(1)
    base = r.normal(0, 1, N)  # per-participant difficulty
    scores = base + np.where(st == IN, 3.0, 0.0) + r.normal(0, 0.3, (S, N))
    s = summarise(lira_likelihood_ratios(scores, st))
    assert s["auc"] > 0.9
    assert s["tpr_at_fpr_0.01"] > 0.3


def test_random_labels_auc_half():
    S, N = 16, 300
    st = _status(S, N, 2)
    scores = np.random.default_rng(3).normal(0, 1, (S, N))  # membership has no effect
    s = summarise(lira_likelihood_ratios(scores, st))
    assert abs(s["auc"] - 0.5) < 0.05
    assert abs(s["sanity_auc_nonmembers_only"] - 0.5) < 0.1


def test_tpr_at_0p1_requires_enough_nonmembers():
    S, N = 4, 40
    st = _status(S, N, 4)
    scores = np.random.default_rng(5).normal(0, 1, (S, N))
    s = summarise(lira_likelihood_ratios(scores, st), min_nonmembers_for_0p1=1000)
    assert s["tpr_at_fpr_0.001"] is None and "cannot be estimated" in s["tpr_at_fpr_0.001_note"]
