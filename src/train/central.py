"""Centralised training for both tracks (also used per-client by local-only training)."""
from __future__ import annotations

import numpy as np
import torch

from src.eval.metrics import best_threshold
from src.models.eegnet import build_eegnet
from src.models.logreg import fit_logreg_tuned
from src.train.torch_core import eeg_pdata, predict_participants, train_early_stopping
from src.utils.seed import set_seed


def train_logreg(feat, train_ids, val_ids, labels, model_cfg, seed, test_ids=()):
    """Scaler + C + threshold fitted on train / inner-validation participants only."""
    Xtr = feat.loc[train_ids].to_numpy()
    Xva = feat.loc[val_ids].to_numpy()
    ytr = np.array([labels[p] for p in train_ids])
    yva = np.array([labels[p] for p in val_ids])
    scaler, model, C, scores = fit_logreg_tuned(Xtr, ytr, Xva, yva, model_cfg["C_grid"], model_cfg["max_iter"], seed)
    p_va = model.predict_proba(scaler.transform(Xva))[:, 1]
    thr = best_threshold(yva, p_va)
    p_te = model.predict_proba(scaler.transform(feat.loc[list(test_ids)].to_numpy()))[:, 1] if len(test_ids) else np.zeros(0)
    return {"scaler": scaler, "model": model, "C": C, "val_auc_by_C": scores, "threshold": thr,
            "p_val": p_va, "p_test": p_te}


def train_eegnet(td, train_ids, val_ids, labels, model_cfg, seed, device, test_ids=(), max_epochs=None):
    set_seed(seed)
    tr = eeg_pdata(td.X, td.rows, train_ids, labels, td.eeg_max_epochs)
    va = eeg_pdata(td.X, td.rows, val_ids, labels, td.eeg_max_epochs)
    model = build_eegnet(model_cfg, n_channels=tr.samples[0].shape[1], n_times=tr.samples[0].shape[2])
    model, info = train_early_stopping(model, tr, va, "eegnet", model_cfg, seed, device,
                                       max_epochs or model_cfg["max_epochs"], model_cfg["patience"])
    p_va = predict_participants(model, va, device)
    thr = best_threshold(va.y, p_va)
    p_te = np.zeros(0)
    if len(test_ids):
        te = eeg_pdata(td.X, td.rows, list(test_ids), td.labels, td.eeg_max_epochs)  # true labels
        p_te = predict_participants(model, te, device)
    return {"model": model, "threshold": thr, "p_val": p_va, "p_test": p_te, **info}


def state_dict_cpu(model) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()} if isinstance(model, torch.nn.Module) else {}
