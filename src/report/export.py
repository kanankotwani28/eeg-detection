"""Aggregate MLflow runs into artifacts/results/*.parquet + summary.json (the dashboard reads only these).

Statistics: mean +- SD across folds x seeds of the per-run metrics; participant-level bootstrap 95% CIs
(2,000 resamples) on out-of-fold predictions averaged over seeds; paired bootstrap of differences on
the participants common to both configurations. No p-values from 5-fold tests are reported.
"""
from __future__ import annotations

import json
import platform
from importlib import metadata

import numpy as np
import pandas as pd

from src.eval.bootstrap import bootstrap_ci, paired_bootstrap_diff
from src.eval.metrics import all_metrics
from src.report import mlflow_io
from src.utils.hashing import git_commit, sha256_file
from src.utils.logging import get_logger
from src.utils.paths import RESULTS, SPLITS_FILE, ensure_dirs

log = get_logger(__name__)
KEY = ["experiment", "variant", "model", "eps", "regime", "task"]
COMPARISONS = [
    # (label, A, B) with A/B = dict of KEY values; A - B
    ("EEGNet - LR (central, T1)", {"experiment": "E04", "variant": "T1", "model": "eegnet"},
     {"experiment": "E02", "model": "logreg"}),
    ("LR FedAvg IID - LR central", {"experiment": "E06", "variant": "iid", "model": "logreg"},
     {"experiment": "E02", "model": "logreg"}),
    ("EEGNet FedAvg IID - EEGNet central", {"experiment": "E06", "variant": "iid", "model": "eegnet"},
     {"experiment": "E04", "variant": "T1", "model": "eegnet"}),
    ("LR FedAvg label-skew - IID", {"experiment": "E06", "variant": "label_skew", "model": "logreg"},
     {"experiment": "E06", "variant": "iid", "model": "logreg"}),
    ("EEGNet FedAvg label-skew - IID", {"experiment": "E06", "variant": "label_skew", "model": "eegnet"},
     {"experiment": "E06", "variant": "iid", "model": "eegnet"}),
    ("FAA only - full LR", {"experiment": "E16", "model": "logreg"}, {"experiment": "E02", "model": "logreg"}),
    ("E12 segment split (WRONG) - E04 subject split", {"experiment": "E12", "model": "eegnet"},
     {"experiment": "E04", "variant": "T1", "model": "eegnet"}),
] + [(f"{m} DP eps={e} - non-DP", {"experiment": "E09", "model": m, "eps": e}, {"experiment": "E09", "model": m, "eps": "inf"})
     for m in ("logreg", "eegnet") for e in ("1", "3", "8")]


def _sel(pred, crit):
    m = np.ones(len(pred), bool)
    for k, v in crit.items():
        m &= pred[k].astype(str).to_numpy() == str(v)
    return pred[m]


def seed_averaged(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per participant (per client for local-only): prob and margin averaged across seeds."""
    p = pred.assign(margin=pred["prob"] - pred["threshold"], client=pred["client"].fillna(-1))
    return p.groupby(["participant_id", "client"], as_index=False).agg(y=("y", "first"), prob=("prob", "mean"),
                                                                     margin=("margin", "mean"), n=("prob", "size"))


def pooled_stats(pred: pd.DataFrame, n_boot=2000) -> dict:
    out = {}
    per_client = []
    for c, g in seed_averaged(pred).groupby("client"):
        y, p, mg = g["y"].to_numpy(), g["prob"].to_numpy(), g["margin"].to_numpy()
        row = {"client": str(int(c)) if c >= 0 else "all", "n_participants": len(g)}
        for metric, (s, thr) in {"balanced_accuracy": (mg, 0.0), "auc": (p, 0.5), "sensitivity": (mg, 0.0),
                                 "specificity": (mg, 0.0)}.items():
            v, lo, hi = bootstrap_ci(y, s, thr, metric, n_boot)
            row[metric], row[f"{metric}_lo"], row[f"{metric}_hi"] = v, lo, hi
        per_client.append(row)
    pc = pd.DataFrame(per_client)
    if len(pc) == 1:
        out = pc.iloc[0].to_dict()
    else:  # local-only: mean client, and worst client by balanced accuracy
        worst = pc.sort_values("balanced_accuracy").iloc[0]
        out = {k: float(pc[k].mean()) for k in pc.columns if k not in ("client", "n_participants")}
        out.update({f"worst_{k}": worst[k] for k in ("balanced_accuracy", "balanced_accuracy_lo", "balanced_accuracy_hi",
                                                      "auc", "auc_lo", "auc_hi")})
        out["n_participants"] = int(pc["n_participants"].max())
        out["client"] = "mean_over_clients"
    return out


def export() -> dict:
    ensure_dirs()
    df = mlflow_io.runs()
    if df.empty:
        raise SystemExit("No MLflow runs found. Run experiments first (make exp-all).")
    kind = df.get("tags.kind", pd.Series([None] * len(df))).fillna("train")
    main, lira, ext = df[kind == "train"], df[kind == "lira"], df[kind.isin(["external", "probe", "mumtaz_cv"])]
    # ---- per-run metrics
    met_cols = [c for c in df.columns if c.startswith("metrics.")]
    tag_cols = [c for c in df.columns if c.startswith("tags.")]
    runs = df[["run_id", "start_time", "artifact_uri"] + tag_cols + met_cols].copy()
    runs.columns = [c.replace("metrics.", "").replace("tags.", "tag_") for c in runs.columns]
    runs.drop(columns=["artifact_uri"]).to_parquet(RESULTS / "runs.parquet", index=False)
    # ---- predictions + FL histories
    preds, hist = [], []
    for _, r in main.iterrows():
        p = mlflow_io.read_predictions(r)
        if p is not None:
            preds.append(p.assign(run_id=r["run_id"]))
        info = mlflow_io.read_json(r, "info.json") or {}
        for h in info.get("history", []) if r.get("tags.regime") == "federated" else []:
            hist.append({**h, "experiment": r["tags.experiment_id"], "variant": r["tags.variant"], "model": r["tags.model"],
                         "eps": r["tags.eps"], "fold": r["tags.fold"], "seed": r["tags.seed"]})
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    pred["client"] = pred["client"].astype("float") if "client" in pred else np.nan
    pred.to_parquet(RESULTS / "predictions.parquet", index=False)
    pd.DataFrame(hist).to_parquet(RESULTS / "fl_history.parquet", index=False)
    # ---- summary per configuration
    rows = []
    mm = runs[runs["tag_kind"].fillna("train") == "train"] if "tag_kind" in runs else runs
    for key, g in pred.groupby(KEY):
        rr = mm[(mm["tag_experiment_id"] == key[0]) & (mm["tag_variant"] == key[1]) & (mm["tag_model"] == key[2])
                & (mm["tag_eps"] == key[3])]
        row = dict(zip(KEY, key, strict=True))
        row.update(pooled_stats(g))
        for m in ("balanced_accuracy", "auc", "epsilon", "epsilon_epoch_level_reference", "sigma", "worst_balanced_accuracy"):
            if m in rr and rr[m].notna().any():
                row[f"{m}_mean"], row[f"{m}_sd"] = float(rr[m].mean()), float(rr[m].std(ddof=0))
        row["n_runs"] = int(len(rr))
        row["description"] = rr["tag_description"].iloc[0] if len(rr) and "tag_description" in rr else ""
        row["partition"] = rr["tag_partition"].iloc[0] if len(rr) else ""
        row["strategy"] = rr["tag_strategy"].iloc[0] if len(rr) else ""
        row["n_clients"] = rr["tag_n_clients"].iloc[0] if len(rr) else ""
        row["split_mode"] = rr["tag_split_mode"].iloc[0] if len(rr) and "tag_split_mode" in rr else "subject"
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_parquet(RESULTS / "summary_table.parquet", index=False)
    # ---- ROC / calibration / confusion for each configuration (seed-averaged)
    roc, cal, cms = [], [], []
    from sklearn.metrics import roc_curve

    for key, g in pred.groupby(KEY):
        if g["client"].notna().any():  # local-only: shown per client on the FL page instead
            continue
        sa = seed_averaged(g)
        if sa["y"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(sa["y"], sa["prob"])
        roc.append(pd.DataFrame({"fpr": fpr, "tpr": tpr}).assign(**dict(zip(KEY, key, strict=True))))
        bins = pd.cut(sa["prob"], np.linspace(0, 1, 11), include_lowest=True)
        c = sa.groupby(bins, observed=True).agg(mean_pred=("prob", "mean"), frac_pos=("y", "mean"), n=("y", "size")).reset_index(drop=True)
        cal.append(c.assign(**dict(zip(KEY, key, strict=True))))
        m = all_metrics(sa["y"], sa["margin"] + 0.5, 0.5)
        cms.append({**dict(zip(KEY, key, strict=True)), "tn": m["cm"][0][0], "fp": m["cm"][0][1], "fn": m["cm"][1][0],
                    "tp": m["cm"][1][1]})
    pd.concat(roc).to_parquet(RESULTS / "roc.parquet", index=False) if roc else None
    pd.concat(cal).to_parquet(RESULTS / "calibration.parquet", index=False) if cal else None
    pd.DataFrame(cms).to_parquet(RESULTS / "confusion.parquet", index=False)
    # ---- paired comparisons
    comps = []
    for label, A, B in COMPARISONS:
        a, b = _sel(pred, A), _sel(pred, B)
        if a.empty or b.empty or a["client"].notna().any() or b["client"].notna().any():
            continue
        sa, sb = seed_averaged(a), seed_averaged(b)
        j = sa.merge(sb, on="participant_id", suffixes=("_a", "_b"))
        if len(j) < 4 or j["y_a"].nunique() < 2:
            continue
        for metric, ca, cb, t in (("balanced_accuracy", "margin_a", "margin_b", 0.0), ("auc", "prob_a", "prob_b", 0.5)):
            d, lo, hi, pneg = paired_bootstrap_diff(j["y_a"], j[ca], j[cb], t, t, metric, 2000)
            comps.append({"comparison": label, "metric": metric, "diff": d, "lo": lo, "hi": hi,
                          "boot_frac_le_0": pneg, "n_participants": len(j)})
    pd.DataFrame(comps).to_parquet(RESULTS / "comparisons.parquet", index=False)
    # ---- LiRA
    lr_rows, lr_roc = [], []
    for _, r in lira.iterrows():
        s = mlflow_io.read_json(r, "lira_summary.json")
        if s:
            lr_rows.append(s)
        p = mlflow_io.read_predictions(r)
        if p is not None:
            lr_roc.append(p)
    pd.DataFrame(lr_rows).to_parquet(RESULTS / "lira.parquet", index=False)
    (pd.concat(lr_roc) if lr_roc else pd.DataFrame()).to_parquet(RESULTS / "lira_roc.parquet", index=False)
    # ---- external / probe / Mumtaz CV
    e_rows = []
    for _, r in ext.iterrows():
        row = {"kind": r["tags.kind"], "name": r["tags.mlflow.runName"], "source": r.get("tags.source"),
               "model": r.get("tags.model"), "regime": r.get("tags.regime"), "mode": r.get("tags.mode")}
        row.update({c.replace("metrics.", ""): r[c] for c in met_cols if pd.notna(r[c])})
        e_rows.append(row)
    pd.DataFrame(e_rows).to_parquet(RESULTS / "external.parquet", index=False)
    # ---- reproducibility
    pkgs = {}
    for p in ("numpy", "scipy", "pandas", "scikit-learn", "torch", "flwr", "opacus", "mne", "mlflow", "streamlit", "plotly"):
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = None
    meta = {
        "git_commit": git_commit(), "split_sha256": json.loads(SPLITS_FILE.read_text())["sha256"] if SPLITS_FILE.exists() else None,
        "splits_file_sha256": sha256_file(SPLITS_FILE) if SPLITS_FILE.exists() else None,
        "python": platform.python_version(), "platform": platform.platform(), "packages": pkgs,
        "n_runs": int(len(df)), "synthetic": bool((df.get("tags.synthetic", pd.Series(["False"])) == "True").any()),
        "config_hashes": sorted(set(df.get("tags.config_hash", pd.Series(dtype=str)).dropna())),
        "participants_tsv_hashes": sorted(set(df.get("tags.participants_tsv_hash", pd.Series(dtype=str)).dropna())),
        "seeds": sorted(set(df.get("tags.seed", pd.Series(dtype=str)).dropna())),
        "experiments": sorted(set(df["tags.experiment_id"].dropna())),
        "mlflow_runs": runs[["run_id", "tag_mlflow.runName"]].rename(columns={"tag_mlflow.runName": "name"}).to_dict("records"),
    }
    (RESULTS / "summary.json").write_text(json.dumps(meta, indent=1, default=str))
    log.info("Exported %d runs, %d configurations, %d comparisons to %s", len(df), len(summary), len(comps), RESULTS)
    return meta


def main():
    export()
    from src.explain.run_all import run_all

    run_all()


if __name__ == "__main__":
    main()
