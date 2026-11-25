"""Shared torch machinery for EEGNet and the torch linear model: participant data containers,
non-private local training (optionally FedProx), early stopping and participant-level prediction."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.eval.metrics import auc, balanced_accuracy


@dataclass
class PData:
    """Participant-structured data. samples[i] has shape (n_i, ...) for participant ids[i]."""

    ids: list[str]
    y: np.ndarray
    samples: list[np.ndarray]

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.ids)

    def flat(self):
        X = np.concatenate(self.samples).astype(np.float32)
        owner = np.concatenate([np.full(len(s), i) for i, s in enumerate(self.samples)])
        return X, self.y[owner], owner

    def subset(self, ids):
        pos = {p: i for i, p in enumerate(self.ids)}
        sel = [pos[p] for p in ids]
        return PData([self.ids[i] for i in sel], self.y[sel], [self.samples[i] for i in sel])

    @property
    def n_samples(self):
        return int(sum(len(s) for s in self.samples))


def zscore_recording(ep: np.ndarray) -> np.ndarray:
    """Per-recording, per-channel z-score: statistics from this participant's own epochs only."""
    ep = np.asarray(ep, dtype=np.float32)
    mu = ep.mean(axis=(0, 2), keepdims=True)
    sd = ep.std(axis=(0, 2), keepdims=True) + 1e-6
    return (ep - mu) / sd


def eeg_pdata(X, rows_by_pid: dict, ids: list[str], labels: dict, max_epochs: int | None = None) -> PData:
    ids = [p for p in ids if p in rows_by_pid]
    samples = [zscore_recording(np.asarray(X[rows_by_pid[p][:max_epochs] if max_epochs else rows_by_pid[p]]))
               for p in ids]
    return PData(ids, np.array([labels[p] for p in ids]), samples)


def feat_pdata(feat_df, ids: list[str], labels: dict, mean: np.ndarray, scale: np.ndarray) -> PData:
    ids = [p for p in ids if p in feat_df.index]
    Z = ((feat_df.loc[ids].to_numpy(np.float64) - mean) / scale).astype(np.float32)
    return PData(ids, np.array([labels[p] for p in ids]), [z[None, :] for z in Z])


def class_weights(y: np.ndarray) -> torch.Tensor:
    n = np.bincount(np.asarray(y), minlength=2).astype(float)
    w = n.sum() / (2 * np.maximum(n, 1))
    return torch.tensor(w, dtype=torch.float32)


def make_optimizer(model, kind: str, cfg: dict):
    if kind == "eegnet":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    if cfg.get("optimizer", "adam") == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.get("adam_lr", 0.01), weight_decay=cfg.get("l2", 0.0))
    return torch.optim.SGD(model.parameters(), lr=cfg["sgd_lr"], weight_decay=cfg.get("l2", 0.0))


def train_epochs(model: nn.Module, data: PData, kind: str, cfg: dict, n_epochs: int, seed: int, device,
                 weights: torch.Tensor | None = None, prox_mu: float = 0.0, global_params=None) -> int:
    """Standard (non-private) minibatch training over samples. Returns number of optimiser steps."""
    model.to(device).train()
    X, y, _ = data.flat()
    w = (weights if weights is not None else class_weights(y)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = make_optimizer(model, kind, cfg)
    g = torch.Generator().manual_seed(seed)
    bs = cfg["batch_size"]
    gp = [p.detach().clone().to(device) for p in global_params] if (prox_mu > 0 and global_params is not None) else None
    steps = 0
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(n_epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb, yb = Xt[b].to(device), yt[b].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if gp is not None:
                loss = loss + 0.5 * prox_mu * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), gp, strict=True))
            loss.backward()
            opt.step()
            model.apply_max_norm()
            steps += 1
    return steps


@torch.no_grad()
def predict_samples(model: nn.Module, data: PData, device, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Returns per-sample softmax P(class 1) and per-sample owner index."""
    model.to(device).eval()
    X, _, owner = data.flat()
    out = []
    for i in range(0, len(X), batch):
        out.append(torch.softmax(model(torch.from_numpy(X[i:i + batch]).to(device)), dim=1)[:, 1].cpu().numpy())
    return (np.concatenate(out) if out else np.zeros(0)), owner


def predict_participants(model, data: PData, device) -> np.ndarray:
    """Per-person probability = mean of epoch softmax probabilities (aligned with data.ids)."""
    p, owner = predict_samples(model, data, device)
    return np.array([p[owner == i].mean() for i in range(len(data))])


def lira_scores(model, data: PData, device) -> np.ndarray:
    """Per-participant mean over epochs of the logit-scaled confidence of the correct class."""
    p, owner = predict_samples(model, data, device)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    yy = data.y[owner]
    pc = np.where(yy == 1, p, 1 - p)
    s = np.log(pc) - np.log(1 - pc)
    return np.array([s[owner == i].mean() for i in range(len(data))])


def val_score(model, data: PData, device) -> tuple[float, float]:
    p = predict_participants(model, data, device)
    return balanced_accuracy(data.y, (p >= 0.5).astype(int)), auc(data.y, p)


def train_early_stopping(model, train: PData, val: PData, kind: str, cfg: dict, seed: int, device,
                         max_epochs: int, patience: int):
    """Early stopping on inner-validation participant-level balanced accuracy (ties -> AUC)."""
    best, best_state, best_ep, wait, hist = (-1.0, -1.0), copy.deepcopy(model.state_dict()), 0, 0, []
    weights = class_weights(train.flat()[1])
    for ep in range(max_epochs):
        train_epochs(model, train, kind, cfg, 1, seed * 1000 + ep, device, weights)
        ba, au = val_score(model, val, device)
        hist.append({"epoch": ep, "val_ba": ba, "val_auc": au})
        key = (round(ba, 6), 0.0 if np.isnan(au) else round(au, 6))
        if key > best:
            best, best_state, best_ep, wait = key, copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep, "best_val_ba": best[0], "best_val_auc": best[1], "history": hist}
