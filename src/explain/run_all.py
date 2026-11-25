"""Compute and save all explainability outputs for the dashboard (artifacts/results/explain_*)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.report import mlflow_io
from src.utils.config import load_experiment
from src.utils.logging import get_logger
from src.utils.paths import RESULTS

log = get_logger(__name__)


def run_all(max_eegnet_runs: int = 15) -> None:
    from src.explain.band_occlusion import band_occlusion
    from src.explain.coefficients import lr_explain
    from src.explain.eegnet_patterns import patterns_table, save_topomaps, spatial_patterns
    from src.models.eegnet import build_eegnet
    from src.train.data import TaskData
    from src.train.torch_core import eeg_pdata

    e02 = mlflow_io.runs("tags.experiment_id = 'E02'")
    if e02.empty:
        log.info("No E02 runs: skipping explainability")
        return
    ds = str(e02["tags.dataset"].iloc[0])
    over = {"subset_n": int(ds.split("@smoke")[1])} if "@smoke" in ds else {}
    cfg = load_experiment("E02", over)
    try:
        td = TaskData(cfg)
        per, grouped = lr_explain(td, cfg, folds=tuple(cfg["models"]["train"]["folds"]))
        per.to_parquet(RESULTS / "explain_lr_features.parquet", index=False)
        grouped.to_parquet(RESULTS / "explain_lr_grouped.parquet", index=False)
    except Exception as e:  # explainability must never block the export
        log.warning("LR explainability skipped: %s", e)
        return
    runs = mlflow_io.runs("tags.experiment_id = 'E04' and tags.variant = 'T1'")
    pats, occ = [], []
    first = True
    for _, r in runs.head(max_eegnet_runs).iterrows():
        state = mlflow_io.read_state(r)
        if not state:
            continue
        fold, seed = int(r["tags.fold"]), int(r["tags.seed"])
        ids = td.fold(fold)
        _ = td.X  # loads the cache and td.meta
        m = build_eegnet(cfg["models"]["eegnet"], n_channels=len(td.meta["channels"]), n_times=td.X.shape[2])
        m.load_state_dict(state)
        tr = eeg_pdata(td.X, td.rows, ids["train"][:100], td.labels)
        A = spatial_patterns(m, np.concatenate(tr.samples))
        pats.append(patterns_table(A, td.meta["channels"], fold, seed))
        if first:
            save_topomaps(A, td.meta["channels"], RESULTS / "explain_eegnet_topomaps.png")
            first = False
        te = eeg_pdata(td.X, td.rows, ids["test"], td.labels)
        thr = float(mlflow_io.read_predictions(r)["threshold"].iloc[0])
        occ.append({**band_occlusion(m, te, thr, td.meta["sfreq"]), "fold": fold, "seed": seed})
    if pats:
        pd.concat(pats).to_parquet(RESULTS / "explain_eegnet_patterns.parquet", index=False)
        pd.DataFrame(occ).to_parquet(RESULTS / "explain_band_occlusion.parquet", index=False)
    log.info("Explainability outputs written (%d EEGNet models)", len(pats))
