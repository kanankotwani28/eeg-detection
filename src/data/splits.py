"""Phase 1: participant-level splits, made once and saved to artifacts/splits.json.

Outer: StratifiedGroupKFold(5, shuffle, 42) grouped by participant, stratified by label (+ age band
when every stratum is large enough). Inner: 80/20 group split of each outer-train fold for validation.
Clients: the remaining training participants are assigned to K clients, IID or label-skew (Dirichlet).
A participant appears in exactly one of train/val/test per fold and in at most one client.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.config import load_config
from src.utils.hashing import sha256_obj
from src.utils.logging import get_logger
from src.utils.paths import PARTICIPANTS_FILE, SPLITS_FILE, ensure_dirs

log = get_logger(__name__)
K_VARIANTS = (1, 2, 4, 8)


def strat_key(y: np.ndarray, age: np.ndarray, n_splits: int) -> tuple[np.ndarray, bool]:
    """Label x age-tertile key if every stratum has >= n_splits members, else label only."""
    if np.isnan(age).any():
        return y.astype(str), False
    edges = np.quantile(age, [1 / 3, 2 / 3])
    band = np.digitize(age, edges)
    key = np.array([f"{a}_{b}" for a, b in zip(y, band, strict=True)])
    _, counts = np.unique(key, return_counts=True)
    if counts.min() >= n_splits:
        return key, True
    return y.astype(str), False


def group_split(ids: np.ndarray, key: np.ndarray, n_splits: int, seed: int):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(ids)), key, groups=ids))


def assign_iid(ids: list[str], k: int, rng: np.random.Generator) -> list[list[str]]:
    perm = list(rng.permutation(ids))
    return [sorted(perm[i::k]) for i in range(k)]


def assign_label_skew(ids: list[str], y: list[int], k: int, alpha: float, min_per_class: int,
                      rng: np.random.Generator) -> list[list[str]]:
    ids, y = np.asarray(ids), np.asarray(y)
    clients: list[list[str]] = [[] for _ in range(k)]
    lab_of = dict(zip(ids, y, strict=True))
    for c in np.unique(y):
        members = rng.permutation(ids[y == c])
        p = rng.dirichlet(alpha * np.ones(k))
        cuts = (np.cumsum(p) * len(members)).astype(int)[:-1]
        for i, part in enumerate(np.split(members, cuts)):
            clients[i].extend(part.tolist())
    # repair: >= min_per_class per class per client where possible
    for c in np.unique(y):
        n_c = int((y == c).sum())
        target = min(min_per_class, n_c // k)
        for _ in range(10 * len(ids)):
            counts = [sum(lab_of[i] == c for i in cl) for cl in clients]
            lo, hi = int(np.argmin(counts)), int(np.argmax(counts))
            if counts[lo] >= target or counts[hi] <= target:
                break
            donor = [i for i in clients[hi] if lab_of[i] == c]
            mv = donor[int(rng.integers(len(donor)))]
            clients[hi].remove(mv)
            clients[lo].append(mv)
    return [sorted(cl) for cl in clients]


def make_clients(ids, y, seed, alpha, min_per_class) -> dict:
    out = {"iid": {}, "label_skew": {}}
    for k in K_VARIANTS:
        if k > len(ids):
            continue
        out["iid"][str(k)] = assign_iid(list(ids), k, np.random.default_rng(seed + 1000 * k))
        out["label_skew"][str(k)] = assign_label_skew(list(ids), list(y), k, alpha, min_per_class,
                                                      np.random.default_rng(seed + 2000 * k))
    return out


def inner_split(ids: np.ndarray, y: np.ndarray, age: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    key, _ = strat_key(y, age, 5)
    tr, va = group_split(ids, key, 5, seed)[0]
    return ids[tr], ids[va]


def make_task_splits(df: pd.DataFrame, task: str, n_splits: int, seed: int, alpha: float, min_pc: int) -> dict:
    d = df[df[f"{task}_label"] >= 0].sort_values("participant_id").reset_index(drop=True)
    ids = d["participant_id"].to_numpy()
    y = d[f"{task}_label"].to_numpy().astype(int)
    age = d["age"].to_numpy(dtype=float)
    labels = dict(zip(ids, y.tolist(), strict=True))
    if len(ids) < n_splits or min(np.bincount(y, minlength=2)) < n_splits:
        log.warning("%s: too few participants for %d folds (n=%d)", task, n_splits, len(ids))
        return {"n": int(len(ids)), "folds": [], "labels": labels, "age_stratified": False}
    key, age_strat = strat_key(y, age, n_splits)
    folds = []
    for f, (tr_idx, te_idx) in enumerate(group_split(ids, key, n_splits, seed)):
        tr_ids, va_ids = inner_split(ids[tr_idx], y[tr_idx], age[tr_idx], seed + f)
        ytr = [labels[i] for i in tr_ids]
        folds.append({
            "fold": f,
            "test": sorted(ids[te_idx].tolist()),
            "train": sorted(tr_ids.tolist()),
            "val": sorted(va_ids.tolist()),
            "clients": make_clients(tr_ids, ytr, seed + 10 * f, alpha, min_pc),
        })
    # final split (all participants) for the frozen external-test models
    tr_ids, va_ids = inner_split(ids, y, age, seed + 99)
    final = {"train": sorted(tr_ids.tolist()), "val": sorted(va_ids.tolist()),
             "clients": make_clients(tr_ids, [labels[i] for i in tr_ids], seed + 999, alpha, min_pc)}
    return {"n": int(len(ids)), "labels": labels, "age_stratified": age_strat, "folds": folds, "final": final}


def make_splits(df: pd.DataFrame, n_splits: int = 5, seed: int = 42, alpha: float = 0.5, min_pc: int = 5) -> dict:
    body = {
        "seed": seed, "n_splits": n_splits, "dirichlet_alpha": alpha, "min_per_class_per_client": min_pc,
        "tasks": {t: make_task_splits(df, t, n_splits, seed, alpha, min_pc) for t in ("T1", "T2")},
    }
    body["sha256"] = sha256_obj(body)
    return body


def load_splits(path=SPLITS_FILE) -> dict:
    with open(path) as f:
        s = json.load(f)
    body = {k: v for k, v in s.items() if k != "sha256"}
    if sha256_obj(body) != s["sha256"]:
        raise ValueError("splits.json hash mismatch: the split file was modified after creation")
    return s


def get_fold(splits: dict, task: str, fold: int | str) -> dict:
    t = splits["tasks"][task]
    return t["final"] if fold == "final" else t["folds"][int(fold)]


def get_clients(fold_split: dict, partition: str, k: int) -> list[list[str]]:
    return fold_split["clients"][partition][str(k)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits.json")
    a = ap.parse_args(argv)
    ensure_dirs()
    if SPLITS_FILE.exists() and not a.force:
        s = load_splits()
        log.info("splits.json exists (sha256 %s); not re-splitting. Use --force to rebuild.", s["sha256"][:12])
        return
    fl = load_config("fl")
    df = pd.read_parquet(PARTICIPANTS_FILE)
    s = make_splits(df, 5, 42, fl["dirichlet_alpha"], fl["min_per_class_per_client"])
    SPLITS_FILE.write_text(json.dumps(s, indent=1))
    for t, v in s["tasks"].items():
        sizes = [(len(f["train"]), len(f["val"]), len(f["test"])) for f in v["folds"]]
        log.info("%s n=%d age-stratified=%s (train,val,test) per fold: %s", t, v["n"], v["age_stratified"], sizes)
    log.info("Saved %s sha256=%s", SPLITS_FILE, s["sha256"])


if __name__ == "__main__":
    main()
