"""Phase 8: the ONLY module that reads Mumtaz. Every access is logged to
artifacts/results/mumtaz_access.log.

E11: models retrained on all TDBRAIN T1 participants (CV-chosen hyperparameters; threshold fixed from
     TDBRAIN validation) are evaluated ONCE on Mumtaz eyes-closed, with 2,000-sample bootstrap CIs.
E13: dataset-identity probe (TDBRAIN-HC vs Mumtaz-HC from band power, group CV). High AUC = the
     datasets are trivially separable, so pooled training would risk shortcut learning.
E20: within-Mumtaz subject-level vs segment-level CV.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import mumtaz as mz
from src.eval.bootstrap import bootstrap_ci
from src.eval.metrics import all_metrics, auc, balanced_accuracy, best_threshold
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.paths import RESULTS, ROOT, ensure_dirs

log = get_logger(__name__)
DATASET, COND = "mumtaz", "EC"


def access_log(msg: str) -> None:
    ensure_dirs()
    with open(RESULTS / "mumtaz_access.log", "a") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")


def mumtaz_root() -> Path:
    c = load_config("data")["mumtaz"]
    p = Path(os.environ.get("MUMTAZ_PATH", c["local_path"]))
    return p if p.is_absolute() else ROOT / p


def fetch_mumtaz() -> Path:
    from src.data.download import figshare_download

    root = mumtaz_root()
    marker = root / ".download_complete"
    if not marker.exists():
        access_log(f"download figshare -> {root}")
        figshare_download(load_config("data")["mumtaz"]["figshare_article_id"], root)
        marker.write_text("ok\n")
    return root


def prepare_mumtaz(cond: str = COND) -> pd.DataFrame:
    """Index, verify counts, and preprocess Mumtaz with the identical preprocessing config."""
    from src.preprocess.pipeline import cache_paths, run_dataset

    root = fetch_mumtaz()
    access_log(f"index {root}")
    idx = mz.index_folder(root)
    print("Mumtaz files per group x condition:\n", mz.count_table(idx).to_string())
    sel = idx[idx["cond"] == cond]
    pre = load_config("preprocess")
    if not cache_paths(DATASET, cond)["meta"].exists():
        access_log(f"preprocess {len(sel)} {cond} recordings")
        meta = run_dataset(list(zip(sel["participant_id"], sel["path"], strict=True)), pre, DATASET, cond,
                           pre.get("n_jobs", 1))
        if meta["channels"] != pre["channels"]:
            raise AssertionError("Mumtaz channel order differs from the 19-channel config")
        from src.preprocess.qc import write_qc

        write_qc(DATASET, cond, dict(zip(sel["participant_id"], sel["group"], strict=True)))
    return sel


def mumtaz_labels(sel: pd.DataFrame) -> dict:
    return dict(zip(sel["participant_id"], sel["label"].astype(int), strict=True))


def _predict_mumtaz(src: dict, r: dict, labels: dict, model_cfg: dict, device):
    from src.features.spectral import feature_table, participant_relative
    from src.preprocess.pipeline import load_cache, participant_epochs
    from src.train.torch_core import eeg_pdata, feat_pdata, predict_participants

    access_log(f"predict with frozen model {src['name']}")
    if src["model"] == "logreg":
        feat = feature_table(DATASET, COND, model_cfg["logreg"])
        if r.get("feature_transform") == "participant_relative":
            feat = participant_relative(feat)
        feat = feat[r["feature_columns"]]
        ids = [p for p in feat.index if p in labels]
        if src["regime"] == "central":
            ex = r["extra"]
            p = ex["model"].predict_proba(ex["scaler"].transform(feat.loc[ids].to_numpy()))[:, 1]
            return ids, p
        ex = r["extra"]
        data = feat_pdata(feat, ids, labels, ex["scaler_mean"], ex["scaler_scale"])
    else:
        X, idx, _ = load_cache(DATASET, COND)
        rows = participant_epochs(X, idx)
        data = eeg_pdata(X, rows, [p for p in sorted(rows) if p in labels], labels)
        ids = data.ids
    return data.ids, predict_participants(r["model"], data, device)


def run_e11(exp_id: str, cfg: dict, smoke: bool, force: bool = False) -> list[dict]:
    from src.experiments import _is_synthetic, run_one
    from src.train.data import TaskData
    from src.utils import tracking
    from src.utils.hashing import git_commit, sha256_obj
    from src.utils.seed import get_device

    marker = RESULTS / "E11_evaluated_once.json"
    if marker.exists() and not force and not smoke:
        raise SystemExit(f"E11 already evaluated once ({marker}). The external test is run ONCE; pass --force "
                         "only if you are certain nothing was tuned on Mumtaz.")
    tracking.setup()
    device = get_device("auto")
    from src.utils.config import deep_merge

    trained = []
    for src in cfg["sources"]:  # train BEFORE Mumtaz is loaded: no Mumtaz information can reach training
        scfg = deep_merge(cfg, src.get("overrides", {}))
        td = TaskData(scfg)
        seed = cfg["models"]["train"]["seeds"][0]
        r = run_one(src["model"], src["regime"], td, "final", seed, scfg, device, src.get("eps"))
        r["feature_columns"] = list(td.features().columns)
        r["feature_transform"] = scfg.get("feature_transform")
        trained.append((src, r))
    sel = prepare_mumtaz()
    labels = mumtaz_labels(sel)
    out = []
    for src, r in trained:
        ids, p = _predict_mumtaz(src, r, labels, cfg["models"], device)
        y = np.array([labels[i] for i in ids])
        thr = r["extra"]["threshold"]
        met = all_metrics(y, p, thr)
        for m in ("balanced_accuracy", "auc", "sensitivity", "specificity"):
            _, lo, hi = bootstrap_ci(y, p, thr, m, 2000)
            met[f"{m}_lo"], met[f"{m}_hi"] = lo, hi
        cm = met.pop("cm")
        eps = r["info"].get("epsilon")
        if eps is not None:
            met["epsilon"] = eps
        name = f"{exp_id}/{src['name']}"
        log.info("%s Mumtaz BA=%.3f [%.3f, %.3f] AUC=%.3f", name, met["balanced_accuracy"], met["balanced_accuracy_lo"],
                 met["balanced_accuracy_hi"], met["auc"])
        tags = {"experiment_id": exp_id, "kind": "external", "source": src["name"], "model": src["model"],
                "regime": src["regime"], "eps": str(src.get("eps", "inf")), "config_hash": sha256_obj(cfg),
                "split_hash": td.splits["sha256"], "git_commit": git_commit(), "synthetic": _is_synthetic()}
        pred = pd.DataFrame({"participant_id": ids, "y": y, "prob": p, "threshold": thr, "source": src["name"]})
        tracking.log_run(name, {"source": src, "threshold": thr}, tags, met, pred, {"confusion_matrix.json": cm}, None, cfg)
        out.append({"name": name, "metrics": met})
    marker.write_text(json.dumps({"when": dt.datetime.now().isoformat(), "sources": [s["name"] for s in cfg["sources"]],
                                  "smoke": smoke}, indent=1))
    return out


def run_e13(exp_id: str, cfg: dict) -> list[dict]:
    """Dataset-identity probe: TDBRAIN-HC vs Mumtaz-HC from band-power features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    from src.experiments import _is_synthetic
    from src.features.spectral import feature_table
    from src.train.data import TaskData
    from src.utils import tracking
    from src.utils.hashing import git_commit

    tracking.setup()
    td = TaskData({**cfg, "task": "T1"})
    tf = td.features()
    hc_t = [p for p, y in td.labels.items() if y == 0 and p in tf.index]
    sel = prepare_mumtaz()
    access_log("E13 dataset-identity probe")
    mf = feature_table(DATASET, COND, cfg["models"]["logreg"])
    hc_m = [p for p in mf.index if p in set(sel.loc[sel["label"] == 0, "participant_id"])]
    X = np.vstack([tf.loc[hc_t, mf.columns].to_numpy(), mf.loc[hc_m].to_numpy()])
    y = np.array([0] * len(hc_t) + [1] * len(hc_m))
    groups = np.array(hc_t + hc_m)
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    a, lo, hi = bootstrap_ci(y, oof, 0.5, "auc", 2000)
    met = {"auc": a, "auc_lo": lo, "auc_hi": hi, "n_tdbrain_hc": len(hc_t), "n_mumtaz_hc": len(hc_m),
           "balanced_accuracy": balanced_accuracy(y, (oof >= 0.5).astype(int))}
    log.info("E13 dataset-identity AUC=%.3f [%.3f, %.3f]", a, lo, hi)
    tracking.log_run(f"{exp_id}/probe", {"C": 1.0}, {"experiment_id": exp_id, "kind": "probe", "git_commit": git_commit(),
                                                    "synthetic": _is_synthetic()}, met,
                     pd.DataFrame({"participant_id": groups, "y": y, "prob": oof}), None, None, cfg)
    return [{"metrics": met}]


def run_e20(exp_id: str, cfg: dict) -> list[dict]:
    """Within-Mumtaz: subject-level (group) CV vs segment-level CV, LR and EEGNet."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from src.experiments import _is_synthetic
    from src.features.spectral import epoch_features
    from src.preprocess.pipeline import load_cache, participant_epochs
    from src.utils import tracking
    from src.utils.hashing import git_commit

    tracking.setup()
    sel = prepare_mumtaz()
    labels = mumtaz_labels(sel)
    access_log("E20 within-Mumtaz CV")
    X, idx, meta = load_cache(DATASET, COND)
    rows = participant_epochs(X, idx)
    pids = [p for p in sorted(rows) if p in labels]
    out = []
    for model_name in cfg["run_models"]:
        for mode in ("subject", "segment"):
            if model_name == "logreg":
                feats, owner = [], []
                for p in pids:
                    f = epoch_features(np.asarray(X[rows[p]]), meta["sfreq"], meta["channels"])
                    feats.append(f)
                    owner += [p] * len(f)
                F, owner = np.vstack(feats), np.array(owner)
            else:
                F = None
                owner = np.concatenate([[p] * len(rows[p]) for p in pids])
            y = np.array([labels[o] for o in owner])
            splitter = (StratifiedGroupKFold(5, shuffle=True, random_state=42).split(owner, y, owner) if mode == "subject"
                        else StratifiedKFold(5, shuffle=True, random_state=42).split(owner, y))
            oof = np.zeros(len(y))
            for tr, te in splitter:
                if model_name == "logreg":
                    sc = StandardScaler().fit(F[tr])
                    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(F[tr]), y[tr])
                    oof[te] = m.predict_proba(sc.transform(F[te]))[:, 1]
                else:
                    oof[te] = _eegnet_segment_cv(X, rows, pids, owner, labels, tr, te, cfg)
            pp = pd.DataFrame({"p": oof, "o": owner}).groupby("o")["p"].mean()
            yp = np.array([labels[o] for o in pp.index])
            met = {"participant_ba": balanced_accuracy(yp, (pp.to_numpy() >= 0.5).astype(int)), "participant_auc": auc(yp, pp),
                   "segment_ba": balanced_accuracy(y, (oof >= 0.5).astype(int)), "segment_auc": auc(y, oof)}
            log.info("E20 %s %s-level CV: segment BA=%.3f participant BA=%.3f", model_name, mode, met["segment_ba"],
                     met["participant_ba"])
            tracking.log_run(f"{exp_id}/{model_name}/{mode}", {"mode": mode},
                             {"experiment_id": exp_id, "kind": "mumtaz_cv", "model": model_name, "mode": mode,
                              "git_commit": git_commit(), "synthetic": _is_synthetic()}, met, None, None, None, cfg)
            out.append({"model": model_name, "mode": mode, "metrics": met})
    return out


def _eegnet_segment_cv(X, rows, pids, owner, labels, tr, te, cfg):
    import torch

    from src.models.eegnet import build_eegnet
    from src.train.torch_core import PData, predict_samples, train_epochs, zscore_recording

    Z = {p: zscore_recording(np.asarray(X[rows[p]])) for p in pids}
    flat = np.concatenate([Z[p] for p in pids])
    mc = dict(cfg["models"]["eegnet"])
    trd = PData([f"s{i}" for i in tr], [labels[owner[i]] for i in tr], [flat[i][None] for i in tr])
    ted = PData([f"s{i}" for i in te], [labels[owner[i]] for i in te], [flat[i][None] for i in te])
    model = build_eegnet(mc, flat.shape[1], flat.shape[2])
    train_epochs(model, trd, "eegnet", mc, min(mc["max_epochs"], 20), 0, torch.device("cpu"))
    p, _ = predict_samples(model, ted, torch.device("cpu"))
    return p


def run_external_experiment(exp_id: str, cfg: dict, smoke: bool = False) -> list[dict]:
    reg = cfg["regime"]
    if reg == "external":
        return run_e11(exp_id, cfg, smoke, force=bool(os.environ.get("EEGFL_FORCE_EXTERNAL")))
    if reg == "probe":
        return run_e13(exp_id, cfg)
    if reg == "mumtaz_cv":
        return run_e20(exp_id, cfg)
    raise ValueError(reg)


__all__ = ["run_external_experiment", "prepare_mumtaz", "access_log", "best_threshold"]
