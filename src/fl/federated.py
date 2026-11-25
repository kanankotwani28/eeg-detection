"""Federated (+ participant-level DP) regime for both tracks, on one outer fold."""
from __future__ import annotations

import math

import numpy as np

from src.eval.metrics import best_threshold
from src.models.eegnet import build_eegnet
from src.models.logreg import LinearModel
from src.privacy.accountant import calibrate_sigma, epoch_level_epsilon
from src.train.torch_core import PData, eeg_pdata, feat_pdata, predict_participants
from src.utils.seed import set_seed

from .server import run_federated


def federated_scaler(client_feats: list[np.ndarray]):
    """Pooled training mean/SD from client sufficient statistics (sum, sum of squares, n) - identical to a
    centralised StandardScaler on the union of the clients' training participants. Not private: refused under DP."""
    n = sum(len(z) for z in client_feats)
    s1 = sum(z.sum(0) for z in client_feats)
    s2 = sum((z**2).sum(0) for z in client_feats)
    mean = s1 / n
    return mean, np.sqrt(np.maximum(s2 / n - mean**2, 1e-12))


def dp_settings(cfg: dict, model_name: str, eps_target: float, n_rounds: int, local_epochs: int,
                n_participants_max: int, n_epochs_total: int) -> dict | None:
    if eps_target is None or math.isinf(eps_target):
        return None
    d = cfg["dp"]
    q = float(d["sample_rate"])
    spr = int(d.get("steps_per_round") or max(1, round(local_epochs / q)))
    steps = n_rounds * spr
    sigma = calibrate_sigma(float(eps_target), float(d["delta"]), q, steps)
    lr = d["dp_lr"][model_name] if isinstance(d["dp_lr"], dict) else d["dp_lr"]
    opt = d.get("dp_optimizer", {}).get(model_name, "sgd") if isinstance(d.get("dp_optimizer"), dict) else d.get("dp_optimizer", "sgd")
    return {"sigma": sigma, "sample_rate": q, "clip_norm": float(d["clip_norm"]), "dp_lr": float(lr),
            "steps_per_round": spr, "delta": float(d["delta"]), "grad_mode": d.get("grad_mode", "one_epoch"),
            "target_epsilon": float(eps_target), "optimizer": opt,
            # LR sees one averaged feature vector per participant: there is no epoch-level unit to compare
            "epoch_level_epsilon_reference": None if model_name == "logreg" else
            epoch_level_epsilon(sigma, q * n_participants_max, n_epochs_total, steps, float(d["delta"]))}


def train_federated(model_name, td, fold_ids, labels, cfg, seed, device, eps_target=None, engine=None):
    set_seed(seed)
    fl = dict(cfg["fl"])
    K = int(fl["n_clients"])
    client_ids = [fold_ids["train"]] if K == 1 else fold_ids["clients"][fl["partition"]][str(K)]
    mc = cfg["models"][model_name]
    local_epochs = int(mc.get("fl_local_epochs", fl["local_epochs"]))
    fl["local_epochs"] = local_epochs
    true = td.labels
    if model_name == "logreg":
        feat = td.features()
        cfeat = [feat.loc[ids].to_numpy(np.float64) for ids in client_ids]
        n_p = max(len(c) for c in client_ids)
        dp = dp_settings(cfg, model_name, eps_target, fl["rounds"], local_epochs, n_p, n_p)
        if cfg.get("feature_transform") == "participant_relative":
            mean, scale = np.zeros(feat.shape[1]), np.ones(feat.shape[1])  # no cross-subject statistics
        elif dp:
            raise ValueError("DP logistic regression needs feature_transform: participant_relative; a pooled "
                             "feature scaler would leak participant information outside the DP accounting")
        else:
            mean, scale = federated_scaler(cfeat)
        clients = [feat_pdata(feat, ids, labels, mean, scale) for ids in client_ids]
        val = feat_pdata(feat, fold_ids["val"], labels, mean, scale)
        test = feat_pdata(feat, fold_ids["test"], true, mean, scale)
        n_f = feat.shape[1]

        def builder():
            return LinearModel(n_f)
        kind = "linear"
        extra_out = {"scaler_mean": mean, "scaler_scale": scale}
    else:
        clients = [eeg_pdata(td.X, td.rows, ids, labels, td.eeg_max_epochs) for ids in client_ids]
        val = eeg_pdata(td.X, td.rows, fold_ids["val"], labels, td.eeg_max_epochs)
        test = eeg_pdata(td.X, td.rows, fold_ids["test"], true, td.eeg_max_epochs)
        n_p = max(len(c) for c in clients)
        n_e = max(c.n_samples for c in clients)
        dp = dp_settings(cfg, model_name, eps_target, fl["rounds"], local_epochs, n_p, n_e)
        C, T = clients[0].samples[0].shape[1:]

        def builder():
            return build_eegnet(mc, n_channels=C, n_times=T)
        kind = "eegnet"
        extra_out = {}
    res = run_federated(builder, kind, clients, val, mc, fl, dp, seed, device, engine or fl.get("engine"))
    model = res["model"]
    p_va = predict_participants(model, val, device)
    thr = best_threshold(val.y, p_va)
    p_te = predict_participants(model, test, device) if len(test) else np.zeros(0)
    out = {**res, **extra_out, "threshold": thr, "p_val": p_va, "p_test": p_te, "test_ids": test.ids,
           "dp": dict(dp or {}), "n_clients": K,
           "client_sizes": [len(c) for c in clients]}
    return out


__all__ = ["train_federated", "federated_scaler", "dp_settings", "PData"]
