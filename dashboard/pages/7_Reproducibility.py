import pandas as pd
import streamlit as st

from common import header, js, table  # noqa: I001
from src.utils.paths import MLRUNS  # noqa: E402

header("Reproducibility")
m = js("summary.json")
if m:
    st.write({"git commit": m.get("git_commit"), "split hash": m.get("split_sha256"),
              "participants.tsv hash": m.get("participants_tsv_hashes"), "seeds": m.get("seeds"),
              "python": m.get("python"), "platform": m.get("platform"), "MLflow runs": m.get("n_runs")})
    st.subheader("Package versions")
    st.table(pd.Series(m.get("packages", {}), name="version"))
    st.subheader("Config hashes")
    st.write(m.get("config_hashes"))
runs = table("runs.parquet")
if len(runs):
    st.subheader("MLflow runs")
    st.caption(f"Local tracking store: {MLRUNS}. Browse with `mlflow ui --backend-store-uri sqlite:///{MLRUNS}/mlflow.db`.")
    cols = [c for c in runs.columns if c in ("run_id", "tag_mlflow.runName", "tag_config_hash", "tag_split_hash", "tag_git_commit",
                                             "tag_seed", "balanced_accuracy", "auc", "epsilon")]
    st.dataframe(runs[cols], width="stretch")
