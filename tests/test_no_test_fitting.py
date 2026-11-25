"""Scalers, thresholds, hyperparameters and early stopping see only training / inner-validation participants."""
from unittest import mock

import numpy as np
import pytest


@pytest.fixture(scope="module")
def td(synthetic_env):
    from src.train.data import TaskData
    from src.utils.config import load_experiment

    cfg = load_experiment("E02", {"models": {"eegnet": {"max_epochs": 2, "patience": 1}}, "fl": {"rounds": 2, "engine": "inprocess"}})
    return TaskData(cfg), cfg


def test_logreg_scaler_and_threshold_use_train_and_val_only(td):
    import src.models.logreg as lr_mod
    import src.train.central as central
    from src.experiments import run_one

    data, cfg = td
    f = data.fold(0)
    feat = data.features()
    seen = {}
    real_fit_scaler, real_thr = lr_mod.fit_scaler, central.best_threshold

    def spy_scaler(X):
        seen["scaler_X"] = np.asarray(X)
        return real_fit_scaler(X)

    def spy_thr(y, p):
        seen["thr_y"] = np.asarray(y)
        seen["thr_n"] = len(p)
        return real_thr(y, p)

    with mock.patch.object(lr_mod, "fit_scaler", spy_scaler), mock.patch.object(central, "best_threshold", spy_thr):
        run_one("logreg", "central", data, 0, 0, cfg, "cpu")
    assert np.allclose(seen["scaler_X"], feat.loc[f["train"]].to_numpy())
    assert seen["thr_n"] == len(f["val"])
    assert np.array_equal(seen["thr_y"], np.array([data.labels[p] for p in f["val"]]))
    assert not set(f["test"]) & (set(f["train"]) | set(f["val"]))


def test_eegnet_early_stopping_and_threshold_use_val_only(td):
    import src.train.central as central
    from src.experiments import run_one

    data, cfg = td
    f = data.fold(1)
    seen = {}
    real = central.train_early_stopping

    def spy(model, train, val, *a, **k):
        seen["train"], seen["val"] = set(train.ids), set(val.ids)
        return real(model, train, val, *a, **k)

    with mock.patch.object(central, "train_early_stopping", spy):
        run_one("eegnet", "central", data, 1, 0, cfg, "cpu")
    assert seen["train"] == set(f["train"]) and seen["val"] == set(f["val"])
    assert not seen["val"] & set(f["test"])


def test_federated_scaler_uses_client_training_data_only(td):
    import src.fl.federated as fed
    from src.experiments import run_one

    data, cfg = td
    f = data.fold(2)
    seen = {}
    real = fed.federated_scaler

    def spy(client_feats):
        seen["rows"] = np.vstack(client_feats)
        return real(client_feats)

    with mock.patch.object(fed, "federated_scaler", spy):
        run_one("logreg", "federated", data, 2, 0, cfg, "cpu")
    train_rows = data.features().loc[f["train"]].to_numpy()
    assert len(seen["rows"]) == len(train_rows)
    assert {tuple(r) for r in np.round(seen["rows"], 8)} == {tuple(r) for r in np.round(train_rows, 8)}


def test_shuffled_labels_never_touch_test(td):
    data, cfg = td
    d2 = type(data)({**cfg, "shuffle_labels": True}, data.splits)
    f = d2.fold(0)
    lab = d2.labels_for(f, seed=3)
    assert all(lab[p] == d2.labels[p] for p in f["test"])
    assert any(lab[p] != d2.labels[p] for p in f["train"] + f["val"])


def test_dp_logreg_refuses_pooled_scaler(td):
    from src.experiments import run_one

    data, cfg = td
    with pytest.raises(ValueError, match="participant_relative"):
        run_one("logreg", "federated", data, 0, 0, cfg, "cpu", eps=3.0)
