"""Track A model. Centralised: StandardScaler (train only) -> L2 LogisticRegression(class_weight=balanced),
C tuned on the inner validation participants. Federated/DP: the same linear model in torch."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn


class LinearModel(nn.Module):
    """Logistic regression as 2 logits ([0, w.x+b]) so it shares the EEGNet training/eval code."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        z = self.linear(x.reshape(x.shape[0], -1))
        return torch.cat([torch.zeros_like(z), z], dim=1)

    def apply_max_norm(self):
        pass


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    return StandardScaler().fit(X_train)


def fit_logreg_tuned(X_tr, y_tr, X_va, y_va, C_grid, max_iter=5000, seed=0):
    """Returns (scaler, model, chosen C, val AUC per C). Scaler fitted on X_tr only."""
    scaler = fit_scaler(X_tr)
    Xs_tr, Xs_va = scaler.transform(X_tr), scaler.transform(X_va)
    scores = {}
    for C in C_grid:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, random_state=seed)
        m.fit(Xs_tr, y_tr)
        p = m.predict_proba(Xs_va)[:, 1]
        scores[C] = roc_auc_score(y_va, p) if len(np.unique(y_va)) > 1 else 0.5
    best = max(C_grid, key=lambda c: (round(scores[c], 6), -abs(np.log10(c))))  # tie -> closest to C=1
    model = LogisticRegression(C=best, class_weight="balanced", max_iter=max_iter, random_state=seed).fit(Xs_tr, y_tr)
    return scaler, model, best, scores
