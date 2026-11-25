"""MLflow tracking (local SQLite store + artifacts under ./mlruns)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from src.utils.paths import MLRUNS


def setup(experiment: str = "eeg-fl-dp") -> str:
    MLRUNS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS / 'mlflow.db'}")
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        mlflow.create_experiment(experiment, artifact_location=(MLRUNS / "artifacts").as_uri())
    mlflow.set_experiment(experiment)
    return experiment


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def log_run(name: str, params: dict, tags: dict, metrics: dict, predictions: pd.DataFrame | None = None,
            json_artifacts: dict | None = None, model_state: dict | None = None, config: dict | None = None) -> str:
    import torch

    with mlflow.start_run(run_name=name) as run:
        flat = {k: str(v)[:500] for k, v in _flatten(params).items()}
        for i in range(0, len(flat), 90):
            mlflow.log_params(dict(list(flat.items())[i:i + 90]))
        mlflow.set_tags({k: str(v) for k, v in tags.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and pd.notna(v)})
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            if predictions is not None:
                predictions.to_parquet(d / "predictions.parquet", index=False)
                mlflow.log_artifact(str(d / "predictions.parquet"))
            for fname, obj in (json_artifacts or {}).items():
                (d / fname).write_text(json.dumps(obj, indent=1, default=float))
                mlflow.log_artifact(str(d / fname))
            if config is not None:
                (d / "config.json").write_text(json.dumps(config, indent=1, default=str))
                mlflow.log_artifact(str(d / "config.json"))
            if model_state:
                torch.save(model_state, d / "model.pt")
                mlflow.log_artifact(str(d / "model.pt"))
        return run.info.run_id
