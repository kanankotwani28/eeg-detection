"""Test fixtures. All tests run on SYNTHETIC data in a temporary folder; no real EEG is needed or touched.
Environment variables are set before any `src` import so that src.utils.paths picks them up."""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="eegfl_tests_"))
os.environ["EEGFL_ARTIFACTS"] = str(_TMP / "artifacts")
os.environ["EEGFL_MLRUNS"] = str(_TMP / "mlruns")
os.environ["TDBRAIN_PATH"] = str(_TMP / "tdbrain")
os.environ["MUMTAZ_PATH"] = str(_TMP / "mumtaz")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_env():
    """Synthetic TDBRAIN (BIDS/BrainVision) -> participant index -> splits -> preprocessed epoch cache."""
    import json

    import pandas as pd

    from src.data.index import assign_tasks, build_index, cohort_report
    from src.data.splits import make_splits
    from src.data.synthetic import make_tdbrain
    from src.preprocess.pipeline import run_dataset, tdbrain_items
    from src.utils.config import load_config
    from src.utils.paths import PARTICIPANTS_FILE, RESULTS, SPLITS_FILE, ensure_dirs

    ensure_dirs()
    root = Path(os.environ["TDBRAIN_PATH"])
    if not (root / "participants.tsv").exists():
        make_tdbrain(root, {"MDD": 34, "HEALTHY": 30, "ADHD": 8, "OCD": 4, "SMC": 4, "UNKNOWN": 3, "REPLICATION": 3},
                     dur_s=48.0, seed=0, effect=0.6)
    data_cfg = load_config("data")
    df = assign_tasks(build_index(data_cfg), data_cfg["cohort"])
    rep = cohort_report(df, root, data_cfg["cohort"])
    df.attrs = {}
    df.to_parquet(PARTICIPANTS_FILE, index=False)
    (RESULTS / "cohort.json").write_text(json.dumps(rep, default=str))
    s = make_splits(df)
    SPLITS_FILE.write_text(json.dumps(s))
    pre = load_config("preprocess")
    pre["min_epochs_per_person"] = 5
    run_dataset(tdbrain_items("restEC"), pre, "tdbrain", "restEC")
    return {"df": pd.read_parquet(PARTICIPANTS_FILE), "splits": s, "report": rep}
