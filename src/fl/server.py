"""Federated training driver (phase 5/6).

engine="flower": Flower simulation engine (ServerApp + ClientApp, Ray backend).
engine="inprocess": identical client logic, selection and aggregation, executed in one process
(used for tests and the smoke pipeline; results match the Flower engine for the same seeds).
Server-side evaluation on the inner-validation participants every round; the best round is kept.
"""
from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.fl.client import ClientLogic, get_params, make_flower_client_app, set_params
from src.fl.strategy import norm_mask, select_clients, weighted_average
from src.privacy.accountant import epsilon
from src.train.torch_core import PData, class_weights, val_score
from src.utils.logging import get_logger
from src.utils.paths import CACHE

log = get_logger(__name__)


def _round_eval(model_builder, params, val: PData, device):
    m = model_builder()
    set_params(m, params)
    ba, au = val_score(m, val, device)
    return ba, (0.0 if np.isnan(au) else au)


def _save_clients(clients: list[PData], root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(clients):
        X, _, owner = c.flat() if len(c) else (np.zeros((0, 1), np.float32), None, np.zeros(0, int))
        np.save(root / f"c{i}_X.npy", X)
        np.save(root / f"c{i}_owner.npy", owner)
        np.save(root / f"c{i}_y.npy", c.y)
        (root / f"c{i}_ids.txt").write_text("\n".join(c.ids))


def _load_client(root: Path, i: int) -> PData:
    X = np.load(root / f"c{i}_X.npy", mmap_mode="r")
    owner = np.load(root / f"c{i}_owner.npy")
    y = np.load(root / f"c{i}_y.npy")
    ids = [s for s in (root / f"c{i}_ids.txt").read_text().split("\n") if s]
    return PData(ids, y, [np.asarray(X[owner == k]) for k in range(len(ids))])


def run_federated(model_builder, kind: str, clients: list[PData], val: PData, model_cfg: dict, fl_cfg: dict,
                  dp_cfg: dict | None, seed: int, device="cpu", engine: str | None = None) -> dict:
    engine = engine or fl_cfg.get("engine", "inprocess")
    n_clients = len(clients)
    rounds = int(fl_cfg["rounds"])
    torch.manual_seed(seed)
    init_model = model_builder()
    params0 = get_params(init_model)
    names = [n for n, _ in init_model.named_parameters()]
    fedbn = fl_cfg.get("strategy") == "fedbn"
    mask = norm_mask(names)
    # client class weights from the client's own labels
    cw = [class_weights(c.flat()[1]) if len(c) else torch.ones(2) for c in clients]
    state = {"best": (-1.0, -1.0), "best_params": params0, "best_round": 0, "history": []}

    def on_round(rnd, params):
        ba, au = _round_eval(model_builder, params, val, device)
        state["history"].append({"round": rnd, "val_ba": ba, "val_auc": au})
        if rnd > 0 and (round(ba, 6), round(au, 6)) > state["best"]:
            state["best"], state["best_params"], state["best_round"] = (round(ba, 6), round(au, 6)), copy.deepcopy(params), rnd

    participation = np.zeros(n_clients, int)
    if engine == "inprocess":
        logics = [ClientLogic(i, c, model_builder, kind, model_cfg, fl_cfg, dp_cfg, seed, device, cw[i])
                  for i, c in enumerate(clients)]
        params = params0
        local_norm: dict[int, list[np.ndarray]] = {}
        on_round(0, params)
        for rnd in range(1, rounds + 1):
            chosen = select_clients(n_clients, rnd, fl_cfg.get("dropout_p", 0.0), seed)
            res = []
            for c in chosen:
                p_in = params
                if fedbn and c in local_norm:
                    it = iter(local_norm[c])
                    p_in = [next(it) if m else g for g, m in zip(params, mask, strict=True)]
                out, n, _ = logics[c].fit(p_in, rnd)
                participation[c] += 1
                if fedbn:
                    local_norm[c] = [p for p, m in zip(out, mask, strict=True) if m]
                res.append((out, n))
            if res:
                params = weighted_average(res)
            on_round(rnd, params)
    elif engine == "flower":
        from flwr.server import ServerApp, ServerAppComponents, ServerConfig
        from flwr.server.strategy import FedAvg, FedProx
        from flwr.simulation import run_simulation

        from src.fl.strategy import make_flower_strategy

        tmp = Path(tempfile.mkdtemp(prefix="fl_", dir=CACHE if CACHE.exists() else None))
        _save_clients(clients, tmp)
        cw_list = [w.numpy().tolist() for w in cw]
        cache: dict[int, ClientLogic] = {}

        def loader(i):
            if i not in cache:
                cache[i] = ClientLogic(i, _load_client(tmp, i), model_builder, kind, model_cfg, fl_cfg, dp_cfg, seed,
                                       "cpu", torch.tensor(cw_list[i]))
            return cache[i]

        holder = {}

        def evaluate_fn(rnd, nd, cfg):
            on_round(rnd, [np.asarray(a) for a in nd])
            return 0.0, {}

        def server_fn(context):
            base = FedProx if fl_cfg.get("strategy") == "fedprox" else FedAvg
            st = make_flower_strategy(base, n_clients=n_clients, dropout_p=fl_cfg.get("dropout_p", 0.0), seed=seed,
                                      fedbn=fedbn, param_names=names, on_round=evaluate_fn, initial=params0,
                                      fit_config=lambda r: {"round": r})
            if base is FedProx:
                st.proximal_mu = fl_cfg.get("proximal_mu", 0.0)
            holder["st"] = st
            return ServerAppComponents(strategy=st, config=ServerConfig(num_rounds=rounds))

        cpus = max(1, (torch.get_num_threads() or 1) // max(n_clients, 1))
        try:
            run_simulation(server_app=ServerApp(server_fn=server_fn), client_app=make_flower_client_app(loader),
                           num_supernodes=n_clients,
                           backend_config={"client_resources": {"num_cpus": cpus, "num_gpus": 0.0},
                                           "init_args": {"logging_level": 30, "log_to_driver": False}})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        participation = holder["st"].participation
    else:
        raise ValueError(engine)

    final_model = model_builder()
    set_params(final_model, state["best_params"])
    out = {"model": final_model, "best_round": state["best_round"], "best_val_ba": state["best"][0],
           "best_val_auc": state["best"][1], "history": state["history"], "participation": participation.tolist()}
    if dp_cfg:
        eps = []
        for c in range(n_clients):
            hist = [(dp_cfg["sigma"], dp_cfg["sample_rate"], int(participation[c]) * dp_cfg["steps_per_round"])]
            hist += dp_cfg.get("extra_history", [])
            eps.append(epsilon(hist, dp_cfg["delta"]))
        out["epsilon_per_client"] = eps
        out["epsilon"] = max(eps)
    return out
