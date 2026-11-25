"""Read logged runs and their artifacts back from the local MLflow store."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import mlflow
import pandas as pd

from src.utils import tracking


def runs(filter_string: str = "") -> pd.DataFrame:
    tracking.setup()
    df = mlflow.search_runs(experiment_names=["eeg-fl-dp"], filter_string=filter_string, max_results=100000)
    if df.empty:
        return df
    # keep only the latest run per name (re-runs supersede earlier ones)
    df = df.sort_values("start_time").drop_duplicates("tags.mlflow.runName", keep="last")
    return df.reset_index(drop=True)


def artifact_dir(row) -> Path:
    return Path(unquote(urlparse(row["artifact_uri"]).path))


def read_json(row, name: str):
    p = artifact_dir(row) / name
    return json.loads(p.read_text()) if p.exists() else None


def read_predictions(row) -> pd.DataFrame | None:
    p = artifact_dir(row) / "predictions.parquet"
    return pd.read_parquet(p) if p.exists() else None


def read_state(row):
    import torch

    p = artifact_dir(row) / "model.pt"
    return torch.load(p, weights_only=False) if p.exists() else None
