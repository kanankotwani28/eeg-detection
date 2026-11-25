import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.eval.bootstrap import bootstrap_ci, paired_bootstrap_diff
from src.eval.metrics import all_metrics, best_threshold


def test_metrics_match_sklearn():
    r = np.random.default_rng(0)
    y = r.integers(0, 2, 300)
    p = np.clip(0.3 * y + 0.7 * r.random(300), 0, 1)
    m = all_metrics(y, p, 0.5)
    assert np.isclose(m["balanced_accuracy"], balanced_accuracy_score(y, p >= 0.5))
    assert np.isclose(m["auc"], roc_auc_score(y, p))
    pt, lo, hi = bootstrap_ci(y, p, 0.5, "auc", 500)
    assert lo < pt < hi and np.isclose(pt, roc_auc_score(y, p))
    d, lo, hi, _ = paired_bootstrap_diff(y, p, p, 0.5, 0.5, "balanced_accuracy", 200)
    assert d == lo == hi == 0


def test_threshold_maximises_ba():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    t = best_threshold(y, p)
    assert 0.3 < t < 0.6
