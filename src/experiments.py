"""Experiment runner: `python -m src.experiments E04` (or `make exp E=E04`).

One MLflow run per (experiment, variant, model, fold, seed), logging config + hashes + git commit,
seeds, participant-level metrics, confusion matrix, epsilon, checkpoint and per-person predictions.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time

import numpy as np
import pandas as pd

from src.eval.metrics import all_metrics, auc, balanced_accuracy, best_threshold
from src.fl.federated import train_federated
from src.train.central import state_dict_cpu, train_eegnet, train_logreg
from src.train.data import TaskData
from src.train.local import train_local_clients
from src.utils import tracking
from src.utils.config import deep_merge, load_experiment, load_yaml
from src.utils.hashing import git_commit, sha256_obj
from src.utils.logging import get_logger
from src.utils.paths import CONFIGS
from src.utils.seed import get_device, set_seed

log = get_logger(__name__)
ESSENTIAL_ORDER = ["E01", "E02", "E16", "E03", "E04", "E05", "E06", "E09", "E10", "E12", "E11", "E13"]
OPTIONAL = ["E07", "E08", "E14", "E15", "E17", "E18", "E19", "E20"]


def _eps_list(v):
    return [math.inf if (isinstance(e, str) and e.lower() in ("inf", "infinity")) else float(e) for e in v]


def run_one(model_name: str, regime: str, td: TaskData, fold, seed: int, cfg: dict, device, eps=None,
            engine=None) -> dict:
    """Train one configuration on one fold/seed. Returns test predictions, metrics and diagnostics."""
    set_seed(seed)
    f = td.fold(fold)
    labels = td.labels_for(f, seed)
    mc = cfg["models"][model_name]
    if regime == "central":
        if model_name == "logreg":
            r = train_logreg(td.features(), f["train"], f["val"], labels, mc, seed, f["test"])
            info = {"C": r["C"], "val_auc_by_C": r["val_auc_by_C"]}
            state = {"coef": r["model"].coef_.tolist(), "intercept": r["model"].intercept_.tolist(),
                     "scaler_mean": r["scaler"].mean_.tolist(), "scaler_scale": r["scaler"].scale_.tolist(),
                     "features": list(td.features().columns)}
        else:
            r = train_eegnet(td, f["train"], f["val"], labels, mc, seed, device, f["test"])
            info = {"best_epoch": r["best_epoch"], "history": r["history"]}
            state = state_dict_cpu(r["model"])
        preds = [(f["test"], r["p_test"], r["threshold"], None)]
        return {"preds": preds, "info": info, "state": state, "p_val": r["p_val"], "val_ids": f["val"],
                "model": r["model"], "fold_ids": f, "extra": r}
    if regime == "local":
        k = cfg["fl"]["n_clients"]
        clients = f["clients"][cfg["fl"]["partition"]][str(k)]
        rs = train_local_clients(model_name, td, f, clients, labels, mc, seed, device)
        preds = [(f["test"], r["p_test"], r["threshold"], r["client"]) for r in rs if "skipped" not in r]
        info = {"clients": [{"client": r["client"], "n_train": r.get("n_train"), "skipped": r.get("skipped")}
                            for r in rs]}
        return {"preds": preds, "info": info, "state": None, "fold_ids": f}
    if regime == "federated":
        r = train_federated(model_name, td, f, labels, cfg, seed, device, eps, engine)
        info = {k: r[k] for k in ("best_round", "best_val_ba", "best_val_auc", "history", "participation",
                                  "client_sizes", "n_clients")}
        info["dp"] = r["dp"]
        if "epsilon" in r:
            info["epsilon"], info["epsilon_per_client"] = r["epsilon"], r["epsilon_per_client"]
        state = state_dict_cpu(r["model"])
        if "scaler_mean" in r:
            state = {**state, "scaler_mean": r["scaler_mean"].tolist(), "scaler_scale": r["scaler_scale"].tolist()}
        return {"preds": [(r["test_ids"], r["p_test"], r["threshold"], None)], "info": info, "state": state,
                "p_val": r["p_val"], "model": r["model"], "fold_ids": f, "extra": r}
    raise ValueError(regime)


def run_segment_split(td: TaskData, cfg: dict, fold: int, seed: int, device) -> dict:
    """E12 - DELIBERATELY WRONG: split EPOCHS (not participants) into folds. Leakage demonstration only."""
    from src.models.eegnet import build_eegnet
    from src.train.torch_core import PData, predict_participants, train_early_stopping, zscore_recording

    set_seed(seed)
    ids = sorted(p for p in td.labels if p in td.rows)
    items = []
    for p in ids:
        ep = zscore_recording(np.asarray(td.X[td.rows[p][: td.eeg_max_epochs] if td.eeg_max_epochs else td.rows[p]]))
        items += [(p, e) for e in ep]
    rng = np.random.default_rng(42)
    fold_of = rng.integers(0, 5, len(items))
    inner = np.random.default_rng(seed).random(len(items)) < 0.2

    def mk(mask):
        sel = np.where(mask)[0]
        return PData([f"{items[i][0]}#{i}" for i in sel], [td.labels[items[i][0]] for i in sel],
                     [items[i][1][None] for i in sel])
    te = mk(fold_of == fold)
    tr = mk((fold_of != fold) & ~inner)
    va = mk((fold_of != fold) & inner)
    mc = cfg["models"]["eegnet"]
    model = build_eegnet(mc, n_channels=tr.samples[0].shape[1], n_times=tr.samples[0].shape[2])
    model, info = train_early_stopping(model, tr, va, "eegnet", mc, seed, device, mc["max_epochs"], mc["patience"])
    thr = best_threshold(va.y, predict_participants(model, va, device))
    p = predict_participants(model, te, device)
    owners = [s.split("#")[0] for s in te.ids]
    pp = pd.DataFrame({"participant_id": owners, "p": p}).groupby("participant_id")["p"].mean()
    return {"preds": [(list(pp.index), pp.to_numpy(), thr, None)],
            "info": {"segment_level_ba": balanced_accuracy(te.y, (p >= thr).astype(int)), "segment_level_auc": auc(te.y, p),
                     "n_test_segments": len(te), "best_epoch": info["best_epoch"],
                     "note": "DELIBERATELY WRONG segment-level split (E12 leakage demonstration)"},
            "state": None}


def _metrics_from_preds(preds, labels) -> tuple[dict, pd.DataFrame]:
    rows, per = [], []
    for ids, p, thr, client in preds:
        y = np.array([labels[i] for i in ids])
        m = all_metrics(y, p, thr)
        m["client"] = client
        per.append(m)
        rows.append(pd.DataFrame({"participant_id": ids, "y": y, "prob": p, "threshold": thr, "client": client}))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(per) == 1:
        met = {k: v for k, v in per[0].items() if isinstance(v, int | float) and k != "client"}
        met_cm = per[0]["cm"]
    else:  # local-only: mean and worst across clients
        met = {}
        for k in ("balanced_accuracy", "auc", "sensitivity", "specificity", "f1", "mcc", "ece"):
            vals = [m[k] for m in per]
            met[k] = float(np.nanmean(vals))
            met[f"worst_{k}"] = float(np.nanmin(vals))
        met["n"] = per[0]["n"] if per else 0
        met_cm = None
    return met | ({"cm": met_cm} if met_cm else {}), df


def expand_variants(cfg: dict) -> list[tuple[str, dict]]:
    grid = cfg.get("grid") or [{"variant": "default"}]
    out = []
    for g in grid:
        g = copy.deepcopy(g)
        name = g.pop("variant")
        out.append((name, deep_merge(cfg, g)))
    return out


def load_overlay(name: str | None) -> dict:
    """Named override sets in configs/experiments/<name>.yaml (key `overrides`), e.g. smoke, budget."""
    return load_yaml(CONFIGS / "experiments" / f"{name}.yaml").get("overrides", {}) if name else {}


def run_experiment(exp_id: str, smoke: bool = False, overrides: dict | None = None, folds=None, seeds=None,
                   log_mlflow: bool = True, overlay: str | None = None) -> list[dict]:
    smoke_over = load_overlay("smoke") if smoke else {}
    cfg = load_experiment(exp_id, deep_merge(deep_merge(smoke_over, load_overlay(overlay)), overrides or {}))
    cfg["overlay"] = overlay or ("smoke" if smoke else None)
    if cfg.get("regime") in ("external", "probe", "mumtaz_cv"):
        from src.eval.external import run_external_experiment

        return run_external_experiment(exp_id, cfg, smoke)
    if cfg.get("regime") == "audit":
        from src.privacy.lira import run_audit_experiment

        return run_audit_experiment(exp_id, cfg, smoke)
    if cfg.get("regime") == "blinded_export":
        from src.eval.blinded import run_blinded_export

        return run_blinded_export(exp_id, cfg)
    device = get_device(cfg["models"]["train"].get("device", "auto"))
    if log_mlflow:
        tracking.setup()
    from src.data.splits import load_splits

    splits = load_splits()
    ptsv_hash = _ptsv_hash()
    commit = git_commit()
    out = []
    for variant, vcfg in expand_variants(cfg):
        td = TaskData(vcfg, splits)
        folds_ = folds or vcfg["models"]["train"]["folds"]
        seeds_ = seeds or vcfg["models"]["train"]["seeds"]
        eps_list = _eps_list(vcfg["dp"]["target_epsilons"]) if vcfg.get("dp", {}).get("enabled") else [None]
        for model_name in vcfg["run_models"]:
            if model_name not in vcfg["models"]:
                continue
            model_seeds = seeds or vcfg["models"][model_name].get("seeds") or seeds_  # per-model seed budget
            for eps in eps_list:
                for fold in folds_:
                    for seed in model_seeds:
                        t0 = time.time()
                        if vcfg.get("split_mode") == "segment":
                            r = run_segment_split(td, vcfg, fold, seed, device)
                        else:
                            r = run_one(model_name, vcfg["regime"], td, fold, seed, vcfg, device, eps)
                        met, pred = _metrics_from_preds(r["preds"], td.labels)
                        cm = met.pop("cm", None)
                        eps_tag = "inf" if eps is None or math.isinf(eps) else f"{eps:g}"
                        if r["info"].get("epsilon") is not None:
                            met["epsilon"] = r["info"]["epsilon"]
                            met["epsilon_epoch_level_reference"] = r["info"]["dp"].get("epoch_level_epsilon_reference")
                            met["sigma"] = r["info"]["dp"].get("sigma")
                        if "best_round" in r["info"]:
                            met["best_round"] = r["info"]["best_round"]
                        for k in ("segment_level_ba", "segment_level_auc"):
                            if k in r["info"]:
                                met[k] = r["info"][k]
                        met["seconds"] = time.time() - t0
                        pred = pred.assign(experiment=exp_id, variant=variant, model=model_name, fold=str(fold),
                                           seed=seed, eps=eps_tag, regime=vcfg["regime"], task=td.task)
                        tags = {"experiment_id": exp_id, "variant": variant, "model": model_name, "fold": fold,
                                "seed": seed, "regime": vcfg["regime"], "task": td.task, "eps": eps_tag,
                                "config_hash": sha256_obj(vcfg), "split_hash": splits["sha256"],
                                "participants_tsv_hash": ptsv_hash, "git_commit": commit,
                                "dataset": td.dataset, "cond": td.cond, "synthetic": _is_synthetic(),
                                "split_mode": vcfg.get("split_mode", "subject"),
                                "partition": vcfg["fl"]["partition"], "n_clients": vcfg["fl"]["n_clients"],
                                "strategy": vcfg["fl"]["strategy"], "description": vcfg.get("description", ""),
                                "overlay": vcfg.get("overlay") or "full"}
                        name = f"{exp_id}/{variant}/{model_name}/eps{eps_tag}/f{fold}/s{seed}"
                        log.info("%s  BA=%.3f AUC=%.3f (%.0fs)", name, met.get("balanced_accuracy", np.nan),
                                 met.get("auc", np.nan), met["seconds"])
                        if log_mlflow:
                            tracking.log_run(name, {"seed": seed, "fold": fold, **_param_view(vcfg, model_name)}, tags,
                                             met, pred, {"info.json": r["info"], "confusion_matrix.json": cm or {}},
                                             r.get("state") if isinstance(r.get("state"), dict) else None, vcfg)
                        out.append({"name": name, "metrics": met, "tags": tags, "pred": pred, "info": r["info"]})
    return out


def _param_view(cfg, model_name):
    return {"task": cfg.get("task"), "regime": cfg.get("regime"), "model": model_name,
            "model_cfg": cfg["models"][model_name], "fl": cfg["fl"],
            "dp": cfg["dp"] if cfg.get("dp", {}).get("enabled") else {"enabled": False},
            "shuffle_labels": cfg.get("shuffle_labels", False), "features": cfg.get("features", "all"),
            "dataset": cfg.get("dataset", "tdbrain"), "cond": cfg.get("cond", "restEC"),
            "max_epochs_per_person": cfg.get("max_epochs_per_person")}


def _ptsv_hash():
    from src.data.tdbrain import find_participants_file, resolve_root
    from src.utils.config import load_config
    from src.utils.hashing import sha256_file

    try:
        return sha256_file(find_participants_file(resolve_root(load_config("data"))))
    except FileNotFoundError:
        return "unknown"


def _is_synthetic() -> bool:
    from src.data.tdbrain import resolve_root
    from src.utils.config import load_config

    return (resolve_root(load_config("data")) / "SYNTHETIC").exists()


def check_e01(results: list[dict], floor: float = 0.08) -> dict:
    """E01 must be ~50%. Each (fold, seed) run is one independent null model: with shuffled labels a model is a
    random direction in feature space, whose test BA/AUC can sit far from 0.5 on any single run (symmetrically).
    So the check is on the MEAN over runs: |mean - 0.5| <= max(floor, t_{0.975, n-1} * SD / sqrt(n)) for both BA and
    AUC. A leak shows up as a mean well above 0.5 (e.g. a segment split gives > 0.8)."""
    from scipy.stats import t as t_dist

    out = {}
    df = pd.concat([r["pred"] for r in results])
    for m, g in df.groupby("model"):
        per = [(balanced_accuracy(gg["y"], (gg["prob"] >= gg["threshold"]).astype(int)), auc(gg["y"], gg["prob"]))
               for _, gg in g.groupby(["fold", "seed"])]
        res = {"n_runs": len(per)}
        ok = True
        for i, name in enumerate(("ba", "auc")):
            v = np.array([p[i] for p in per], float)
            v = v[~np.isnan(v)]
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.inf
            tol = max(floor, float(t_dist.ppf(0.975, len(v) - 1)) * se if len(v) > 1 else 1.0)
            res[f"{name}_mean"], res[f"{name}_tol"] = float(v.mean()), tol
            ok &= abs(v.mean() - 0.5) <= tol
        res["passes"] = bool(ok)
        out[m] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", nargs="+", help="experiment IDs, or 'all' / 'essential'")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--overlay", help="named override set in configs/experiments/<name>.yaml, e.g. budget")
    ap.add_argument("--folds", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--models", type=lambda s: s.split(","), help="restrict run_models, e.g. logreg or eegnet")
    a = ap.parse_args(argv)
    ids = ESSENTIAL_ORDER if a.exp in (["all"], ["essential"]) else a.exp
    if a.exp == ["all"]:
        ids = ESSENTIAL_ORDER + OPTIONAL
    for e in ids:
        over = {"run_models": a.models} if a.models else None
        res = run_experiment(e, a.smoke, over, folds=a.folds, seeds=a.seeds, log_mlflow=not a.no_mlflow, overlay=a.overlay)
        if e == "E01":
            chk = check_e01(res)
            print(json.dumps({"E01_check": chk}, indent=1))
            if not all(v["passes"] for v in chk.values()):
                raise SystemExit("STOP: E01 shuffled-label balanced accuracy is not ~50%. Find the leak before "
                                 "running anything else.")


if __name__ == "__main__":
    main()
