import numpy as np

from src.eval.metrics import auc
from src.fl.server import run_federated
from src.fl.strategy import select_clients, weighted_average
from src.models.logreg import LinearModel
from src.train.torch_core import PData, predict_participants, train_epochs
from src.utils.config import load_config

MC = load_config("models")["logreg"]


def lin(n, seed):
    r = np.random.default_rng(seed)
    y = r.integers(0, 2, n)
    X = r.standard_normal((n, 10)).astype(np.float32)
    X[:, 0] += 1.5 * y
    return PData([f"{seed}-{i}" for i in range(n)], y, [x[None] for x in X])


def test_weighted_average():
    a = [np.ones(3)], 1
    b = [np.zeros(3)], 3
    assert np.allclose(weighted_average([a, b])[0], 0.25)


def test_dropout_is_seeded():
    assert select_clients(4, 7, 0.2, 0) == select_clients(4, 7, 0.2, 0)
    rounds = [len(select_clients(4, r, 0.2, 0)) for r in range(1, 500)]
    assert 0.75 < np.mean(rounds) / 4 < 0.85


def test_k1_federated_matches_centralised():
    train, val, test = lin(200, 0), lin(80, 1), lin(200, 2)
    fl = {"rounds": 20, "local_epochs": 1, "strategy": "fedavg", "engine": "inprocess"}
    r = run_federated(lambda: LinearModel(10), "linear", [train], val, MC, fl, None, 0, "cpu")
    central = LinearModel(10)
    train_epochs(central, train, "linear", MC, 20, 0, "cpu")
    a_fl = auc(test.y, predict_participants(r["model"], test, "cpu"))
    a_c = auc(test.y, predict_participants(central, test, "cpu"))
    assert abs(a_fl - a_c) < 0.03, (a_fl, a_c)
    w_fl = r["model"].linear.weight.detach().numpy()
    w_c = central.linear.weight.detach().numpy()
    assert np.corrcoef(w_fl.ravel(), w_c.ravel())[0, 1] > 0.95


def test_flower_engine_matches_inprocess():
    clients, val = [lin(30, 10), lin(30, 11), lin(30, 12)], lin(30, 13)
    fl = {"rounds": 2, "local_epochs": 1, "strategy": "fedavg", "dropout_p": 0.2}
    a = run_federated(lambda: LinearModel(10), "linear", clients, val, MC, fl, None, 0, "cpu", "inprocess")
    b = run_federated(lambda: LinearModel(10), "linear", clients, val, MC, fl, None, 0, "cpu", "flower")
    assert a["participation"] == list(b["participation"])
    assert np.allclose(a["model"].linear.weight.detach().numpy(), b["model"].linear.weight.detach().numpy(), atol=1e-6)
