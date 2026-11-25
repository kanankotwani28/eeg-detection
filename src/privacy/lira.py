"""Phase 7: participant-level LiRA membership-inference audit (Carlini et al., 2022).

Threat model: an honest-but-curious server, or anyone with query access to the released model,
wants to infer whether a given participant's EEG was used in training.

Procedure: S models are trained with the TARGET recipe, each on a random 50% of the pool (its
in-set is further split 80/20 into train/val; val participants are excluded from that model's
evaluation). Score per participant = mean over their epochs of the logit-scaled confidence of the
correct class. Each model in turn plays the target; the others are shadows: per participant, fit
Gaussians to in- and out-shadow scores and compute the likelihood ratio of the target's score.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

from src.utils.logging import get_logger

log = get_logger(__name__)
IN, OUT, IGNORE = 1, 0, -1


def lira_likelihood_ratios(scores: np.ndarray, status: np.ndarray, fixed_variance: bool = True,
                           min_shadow: int = 1) -> pd.DataFrame:
    """scores, status: (n_models, n_participants). Returns one row per (target model, participant)
    with the LiRA statistic and true membership."""
    S, N = scores.shape
    rows = []
    for t in range(S):
        others = np.arange(S) != t
        sc, st = scores[others], status[others]
        global_sd_in = np.nanstd(np.where(st == IN, sc, np.nan))
        global_sd_out = np.nanstd(np.where(st == OUT, sc, np.nan))
        for i in range(N):
            if status[t, i] == IGNORE:
                continue
            s_in, s_out = sc[st[:, i] == IN, i], sc[st[:, i] == OUT, i]
            if len(s_in) < min_shadow or len(s_out) < min_shadow:
                continue
            sd_in = global_sd_in if fixed_variance else max(s_in.std(), 1e-3)
            sd_out = global_sd_out if fixed_variance else max(s_out.std(), 1e-3)
            x = scores[t, i]
            lr = norm.logpdf(x, s_in.mean(), max(sd_in, 1e-3)) - norm.logpdf(x, s_out.mean(), max(sd_out, 1e-3))
            rows.append({"target": t, "participant": i, "member": int(status[t, i] == IN), "lr": float(lr)})
    return pd.DataFrame(rows, columns=["target", "participant", "member", "lr"])


def tpr_at_fpr(y, s, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def summarise(df: pd.DataFrame, fprs=(0.001, 0.01, 0.05), min_nonmembers_for_0p1: int = 1000, seed: int = 0) -> dict:
    if df.empty or df["member"].nunique() < 2:
        return {"auc": float("nan"), "n_member_evals": int(df["member"].sum()) if len(df) else 0,
                "n_nonmember_evals": int((df["member"] == 0).sum()) if len(df) else 0,
                **{f"tpr_at_fpr_{f:g}": None for f in fprs}, "note": "too few shadow models for LiRA"}
    y, s = df["member"].to_numpy(), df["lr"].to_numpy()
    n_non = int((y == 0).sum())
    out = {"auc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan"),
           "n_member_evals": int(y.sum()), "n_nonmember_evals": n_non}
    for f in fprs:
        key = f"tpr_at_fpr_{f:g}"
        if f < 0.01 and n_non < min_nonmembers_for_0p1:
            out[key] = None
            out[key + "_note"] = f"cannot be estimated: only {n_non} non-member evaluations (< {min_nonmembers_for_0p1})"
        else:
            out[key] = tpr_at_fpr(y, s, f)
    # sanity: among NON-members only, a random half relabelled "member" -> AUC ~ 0.5
    non = df[df["member"] == 0]
    if len(non) > 10:
        fake = np.random.default_rng(seed).random(len(non)) < 0.5
        out["sanity_auc_nonmembers_only"] = float(roc_auc_score(fake, non["lr"])) if 0 < fake.sum() < len(fake) else float("nan")
    return out


def roc_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df["member"].nunique() < 2:
        return pd.DataFrame({"fpr": [], "tpr": []})
    fpr, tpr, _ = roc_curve(df["member"], df["lr"])
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def make_shadow_folds(pool: list[str], labels: dict, n_models: int, k_clients: int, seed: int):
    """Each model: random 50% in-set (stratified by label), split 80/20 into train/val; IID clients from train."""
    from src.data.splits import assign_iid

    rng = np.random.default_rng(seed)
    pool = np.array(sorted(pool))
    y = np.array([labels[p] for p in pool])
    status = np.full((n_models, len(pool)), OUT)
    folds = []
    for m in range(n_models):
        inset = np.zeros(len(pool), bool)
        for c in (0, 1):
            idx = np.where(y == c)[0]
            inset[rng.choice(idx, len(idx) // 2, replace=False)] = True
        in_idx = rng.permutation(np.where(inset)[0])
        n_val = max(2, int(round(0.2 * len(in_idx))))
        val_idx, tr_idx = in_idx[:n_val], in_idx[n_val:]
        status[m, tr_idx] = IN
        status[m, val_idx] = IGNORE
        tr = sorted(pool[tr_idx].tolist())
        clients = {"iid": {str(k_clients): assign_iid(tr, k_clients, np.random.default_rng(seed + m)), "1": [tr]}}
        folds.append({"train": tr, "val": sorted(pool[val_idx].tolist()), "test": [], "clients": clients})
    return pool.tolist(), folds, status


def score_pool(model_name: str, regime: str, r: dict, td, pool: list[str], device) -> np.ndarray:
    from src.train.torch_core import eeg_pdata, feat_pdata, lira_scores

    if model_name == "logreg" and regime == "central":
        ex = r["extra"]
        p = ex["model"].predict_proba(ex["scaler"].transform(td.features().loc[pool].to_numpy()))[:, 1]
        p = np.clip(p, 1e-7, 1 - 1e-7)
        y = np.array([td.labels[i] for i in pool])
        pc = np.where(y == 1, p, 1 - p)
        return np.log(pc) - np.log(1 - pc)
    if model_name == "logreg":
        ex = r["extra"]
        data = feat_pdata(td.features(), pool, td.labels, ex["scaler_mean"], ex["scaler_scale"])
    else:
        data = eeg_pdata(td.X, td.rows, pool, td.labels, td.eeg_max_epochs)
    return lira_scores(r["model"], data, device)


def run_audit(model_name: str, regime: str, eps, td, cfg: dict, device, n_models: int, seed: int = 0):
    from src.experiments import run_one

    pool = sorted(p for p in td.labels if p in td.rows and p in td.features().index)
    k = cfg["fl"]["n_clients"]
    pool, folds, status = make_shadow_folds(pool, td.labels, n_models, k, seed)
    cfg = dict(cfg)
    cfg["fl"] = {**cfg["fl"], "partition": "iid"}
    scores = np.zeros((n_models, len(pool)))
    eps_seen = []
    for m, f in enumerate(folds):
        r = run_one(model_name, regime, td, f, seed * 1000 + m, cfg, device, eps)
        scores[m] = score_pool(model_name, regime, r, td, pool, device)
        if r["info"].get("epsilon") is not None:
            eps_seen.append(r["info"]["epsilon"])
    df = lira_likelihood_ratios(scores, status)
    summ = summarise(df, tuple(cfg["audit"]["fprs"]), cfg["audit"]["min_nonmembers_for_0p1"], seed)
    summ.update({"model": model_name, "regime": regime, "eps": "inf" if eps is None or math.isinf(eps) else f"{eps:g}",
                 "n_models": n_models, "pool_size": len(pool),
                 "achieved_epsilon": max(eps_seen) if eps_seen else None})
    return summ, roc_table(df), df


def run_audit_experiment(exp_id: str, cfg: dict, smoke: bool = False) -> list[dict]:
    from src.experiments import _eps_list, _is_synthetic
    from src.train.data import TaskData
    from src.utils import tracking
    from src.utils.hashing import git_commit, sha256_obj
    from src.utils.seed import get_device

    tracking.setup()
    td = TaskData({**cfg, "task": cfg["audit"].get("pool_task", cfg.get("task", "T1"))})
    device = get_device(cfg["models"]["train"].get("device", "auto"))
    out = []
    for model_name in cfg["run_models"]:
        n_models = cfg["audit"]["n_shadows_logreg" if model_name == "logreg" else "n_shadows_eegnet"]
        for regime in cfg.get("audit_regimes", ["central", "federated"]):
            eps_cfg = cfg["audit"].get(f"{model_name}_target_epsilons") or cfg["dp"]["target_epsilons"]
            eps_list = _eps_list(eps_cfg) if regime == "federated" else [None]
            for eps in eps_list:
                summ, roc, raw = run_audit(model_name, regime, eps, td, cfg, device, n_models)
                name = f"{exp_id}/{model_name}/{regime}/eps{summ['eps']}"
                log.info("%s LiRA AUC=%s TPR@1%%=%s TPR@5%%=%s", name, summ["auc"], summ.get("tpr_at_fpr_0.01"),
                         summ.get("tpr_at_fpr_0.05"))
                tags = {"experiment_id": exp_id, "model": model_name, "regime": regime, "eps": summ["eps"],
                        "config_hash": sha256_obj(cfg), "split_hash": td.splits["sha256"], "git_commit": git_commit(),
                        "synthetic": _is_synthetic(), "kind": "lira"}
                metrics = {k: v for k, v in summ.items() if isinstance(v, int | float) and v is not None}
                tracking.log_run(name, {"n_models": n_models, "regime": regime, "model": model_name}, tags, metrics,
                                 roc.assign(model=model_name, regime=regime, eps=summ["eps"]),
                                 {"lira_summary.json": summ}, None, cfg)
                out.append({"summary": summ, "roc": roc})
    return out
