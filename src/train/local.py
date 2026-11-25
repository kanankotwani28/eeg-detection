"""Local-only regime: each client trains alone on its own participants and is evaluated on the same
outer test fold. Validation (C, threshold, early stopping) uses the fold's inner-validation participants."""
from __future__ import annotations

from src.train.central import train_eegnet, train_logreg


def train_local_clients(model_name, td, fold_ids, clients, labels, model_cfg, seed, device):
    out = []
    for k, ids in enumerate(clients):
        if len({labels[p] for p in ids}) < 2:
            out.append({"client": k, "skipped": "single class"})
            continue
        if model_name == "logreg":
            r = train_logreg(td.features(), ids, fold_ids["val"], labels, model_cfg, seed, fold_ids["test"])
        else:
            r = train_eegnet(td, ids, fold_ids["val"], labels, model_cfg, seed, device, fold_ids["test"])
        r["client"] = k
        r["n_train"] = len(ids)
        out.append(r)
    return out
