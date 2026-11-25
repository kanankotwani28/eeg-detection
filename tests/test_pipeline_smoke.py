"""Integration on synthetic data: index -> splits -> preprocess -> E01 (must be ~50%) and E02."""
import numpy as np


def test_index_rules(synthetic_env):
    df, rep = synthetic_env["df"], synthetic_env["report"]
    assert df["participant_id"].is_unique
    assert (df.loc[df["blinded"], "T1_label"] == -1).all()  # REPLICATION never labelled
    assert (df.loc[df["primary_indication"] == "UNKNOWN", "T1_label"] == -1).all()  # UNKNOWN is not healthy
    assert not rep["mumtaz_19_missing_in_tdbrain"]


def test_e01_shuffled_labels_is_chance(synthetic_env):
    from src.eval.metrics import balanced_accuracy
    from src.experiments import run_experiment

    res = run_experiment("E01", overrides={"run_models": ["logreg"], "models": {"train": {"seeds": [0, 1, 2, 3]}}},
                         log_mlflow=False)
    bas = [balanced_accuracy(r["pred"]["y"], (r["pred"]["prob"] >= r["pred"]["threshold"]).astype(int)) for r in res]
    assert abs(np.mean(bas) - 0.5) < 0.12, bas
    from src.experiments import check_e01

    assert check_e01(res)["logreg"]["passes"]


def test_e02_runs_and_logs(synthetic_env):
    from src.experiments import run_experiment

    res = run_experiment("E02", folds=[0], seeds=[0])
    assert len(res) == 1 and 0 <= res[0]["metrics"]["balanced_accuracy"] <= 1
    assert res[0]["tags"]["split_hash"] == synthetic_env["splits"]["sha256"]


def test_e01_check_flags_a_leak():
    """Positive control: predictions that track the true labels must FAIL the E01 check."""
    import pandas as pd

    from src.experiments import check_e01

    r = np.random.default_rng(0)
    res = []
    for fold in range(5):
        y = r.integers(0, 2, 60)
        leaky = pd.DataFrame({"y": y, "prob": np.clip(0.35 * y + 0.65 * r.random(60), 0, 1), "threshold": 0.5,
                              "model": "logreg", "fold": str(fold), "seed": 0})
        res.append({"pred": leaky})
    assert not check_e01(res)["logreg"]["passes"]
    null = [{"pred": p["pred"].assign(prob=r.random(60))} for p in res]
    assert check_e01(null)["logreg"]["passes"]


def test_v3_xlsx_participants_file_in_parent(tmp_path):
    """TDBRAIN V3 ships TDBRAIN_participants_V3.xlsx next to the dataset folder; '._' files are ignored."""
    import pandas as pd

    from src.data.tdbrain import find_participants_file, read_participants

    ds = tmp_path / "TDBRAIN_Dataset_V3_1"
    ds.mkdir()
    (tmp_path / "._TDBRAIN_participants_V3.xlsx").write_bytes(b"junk")
    pd.DataFrame({"participants_ID": ["sub-1"], "indication": ["MDD"]}).to_excel(
        tmp_path / "TDBRAIN_participants_V3.xlsx", index=False)
    f = find_participants_file(ds)
    assert f.name == "TDBRAIN_participants_V3.xlsx"
    assert read_participants(f)["indication"].iloc[0] == "MDD"


def test_v3_column_names_resolve():
    import pandas as pd

    from src.data.index import ALIASES, _col

    cols = ["TDBRAIN_ID", "DISC/REP", "indication", "formal_status", "age", "gender", "sessID", "BDI_pre"]
    got = {k: _col(pd.DataFrame(columns=cols), k, False) for k in ALIASES}
    assert got == {"id": "TDBRAIN_ID", "session": "sessID", "indication": "indication", "diagnosis": "formal_status",
                   "age": "age", "sex": "gender", "bdi": "BDI_pre"}
