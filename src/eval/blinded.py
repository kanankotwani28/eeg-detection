"""E21 (optional, needs the user): predictions for TDBRAIN's blinded REPLICATION sample, to be submitted
by the user to the TDBRAIN curators for their diagnostic validation. Labels are never seen here."""
from __future__ import annotations

import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.paths import PARTICIPANTS_FILE, RESULTS

log = get_logger(__name__)


def run_blinded_export(exp_id: str, cfg: dict) -> list[dict]:
    from src.experiments import run_one
    from src.features.spectral import feature_table
    from src.preprocess.pipeline import cache_paths, run_dataset
    from src.train.data import TaskData
    from src.utils.seed import get_device

    src = cfg["source"]
    if src["model"] != "logreg" or src["regime"] != "central":
        raise NotImplementedError("E21 currently exports the centralised logistic-regression model only")
    td = TaskData(cfg)
    r = run_one("logreg", "central", td, "final", 0, cfg, get_device("cpu"))
    df = pd.read_parquet(PARTICIPANTS_FILE)
    bl = df[df["blinded"] & df["has_restEC"]]
    if not cache_paths("tdbrain-blinded", "restEC")["meta"].exists():
        pre = load_config("preprocess")
        run_dataset(list(zip(bl["participant_id"], bl["path_restEC"], strict=True)), pre, "tdbrain-blinded", "restEC")
    feat = feature_table("tdbrain-blinded", "restEC", cfg["models"]["logreg"])
    ex = r["extra"]
    p = ex["model"].predict_proba(ex["scaler"].transform(feat[td.features().columns].to_numpy()))[:, 1]
    out = pd.DataFrame({"participant_id": feat.index, "prob_mdd": p, "pred_mdd": (p >= ex["threshold"]).astype(int)})
    path = RESULTS / "E21_blinded_predictions.csv"
    out.to_csv(path, index=False)
    log.info("Wrote %d blinded predictions to %s (submit them to the TDBRAIN curators yourself)", len(out), path)
    return [{"n": len(out), "path": str(path)}]
