"""LR explainability: standardised coefficients and permutation importance per fold, grouped by
band x region; only features whose coefficient sign is stable in >= 4/5 folds are kept.
This explains the MODEL, not the disorder."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.train.central import train_logreg

REGIONS = {"frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4"],
           "central": ["C3", "Cz", "C4", "CP3", "CPz", "CP4"], "temporal": ["T7", "T8"],
           "parietal": ["P7", "P3", "Pz", "P4", "P8"], "occipital": ["O1", "O2", "Oz"]}


def region_of(ch: str) -> str:
    return next((r for r, chs in REGIONS.items() if ch in chs), "other")


def lr_explain(td, cfg, folds=(0, 1, 2, 3, 4), seed=0, min_stable=4) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat = td.features()
    coefs, perms = [], []
    for f in folds:
        if f >= len(td.splits["tasks"][td.task]["folds"]):
            continue
        ids = td.fold(f)
        r = train_logreg(feat, ids["train"], ids["val"], td.labels, cfg["models"]["logreg"], seed, ids["test"])
        coefs.append(pd.Series(r["model"].coef_[0], index=feat.columns, name=f))
        Xte = r["scaler"].transform(feat.loc[ids["test"]].to_numpy())
        yte = np.array([td.labels[p] for p in ids["test"]])
        pi = permutation_importance(r["model"], Xte, yte, scoring="balanced_accuracy", n_repeats=20, random_state=seed)
        perms.append(pd.Series(pi.importances_mean, index=feat.columns, name=f))
    C, P = pd.DataFrame(coefs), pd.DataFrame(perms)
    n = len(C)
    need = min(min_stable, n) if n < 5 else min_stable
    sign_pos = (C > 0).sum()
    stable = (sign_pos >= need) | ((n - sign_pos) >= need)
    rows = []
    for name in C.columns:
        band, ch = (name.split("_", 1) + [""])[:2] if name != "FAA" else ("alpha", "F4-F3")
        rows.append({"feature": name, "band": band if name != "FAA" else "FAA", "channel": ch,
                     "region": "frontal" if name == "FAA" else region_of(ch), "coef_mean": C[name].mean(),
                     "coef_sd": C[name].std(ddof=0), "perm_importance_mean": P[name].mean(),
                     "sign_stable_folds": int(max(sign_pos[name], n - sign_pos[name])), "n_folds": n,
                     "stable": bool(stable[name])})
    per_feat = pd.DataFrame(rows)
    grouped = (per_feat[per_feat["stable"]].groupby(["band", "region"], as_index=False)
               .agg(coef_mean=("coef_mean", "mean"), perm_importance=("perm_importance_mean", "sum"),
                    n_features=("feature", "size")))
    return per_feat, grouped
