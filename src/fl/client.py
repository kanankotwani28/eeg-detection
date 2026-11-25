"""Federated client: local training on one client's participants (optionally FedProx or participant-level DP)."""
from __future__ import annotations

import numpy as np
import torch

from src.privacy.participant_dp import dp_local_train
from src.train.torch_core import PData, class_weights, train_epochs


def get_params(model) -> list[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def set_params(model, params: list[np.ndarray]) -> None:
    with torch.no_grad():
        for p, v in zip(model.parameters(), params, strict=True):
            p.copy_(torch.from_numpy(np.asarray(v, dtype=np.float32)))


class ClientLogic:
    """Pure-python client used by both engines."""

    def __init__(self, partition: int, data: PData, model_builder, kind: str, model_cfg: dict, fl_cfg: dict,
                 dp_cfg: dict | None, seed: int, device="cpu", class_w: torch.Tensor | None = None):
        self.partition, self.data, self.kind = partition, data, kind
        self.model_builder, self.model_cfg, self.fl_cfg, self.dp = model_builder, model_cfg, fl_cfg, dp_cfg
        self.seed, self.device = seed, torch.device(device)
        self.class_w = class_w if class_w is not None else class_weights(data.flat()[1] if len(data) else np.array([0, 1]))

    @property
    def n_examples(self) -> int:
        return len(self.data) if self.dp else self.data.n_samples

    def fit(self, params: list[np.ndarray], rnd: int) -> tuple[list[np.ndarray], int, dict]:
        torch.set_num_threads(max(1, torch.get_num_threads()))
        model = self.model_builder()
        set_params(model, params)
        s = int(np.random.default_rng([self.seed, rnd, self.partition]).integers(2**31))
        if len(self.data) == 0:
            return get_params(model), 0, {"partition": self.partition, "steps": 0}
        if self.dp:
            info = dp_local_train(model, self.data, self.dp["steps_per_round"], self.dp["sample_rate"],
                                  self.dp["sigma"], self.dp["clip_norm"], self.dp["dp_lr"], s, self.device,
                                  self.dp.get("grad_mode", "one_epoch"), self.class_w,
                                  optimizer=self.dp.get("optimizer", "sgd"))
            steps = info["steps"]
        else:
            mu = self.fl_cfg.get("proximal_mu", 0.0) if self.fl_cfg.get("strategy") == "fedprox" else 0.0
            steps = train_epochs(model, self.data, self.kind, self.model_cfg, self.fl_cfg.get("local_epochs", 1), s,
                                 self.device, self.class_w, prox_mu=mu, global_params=[torch.from_numpy(p) for p in params])
        return get_params(model), self.n_examples, {"partition": self.partition, "steps": int(steps)}


def make_flower_client_app(logics_loader):
    """logics_loader(partition_id) -> ClientLogic (called inside the simulation worker)."""
    from flwr.client import NumPyClient
    from flwr.clientapp import ClientApp

    class FlowerClient(NumPyClient):
        def __init__(self, logic: ClientLogic):
            self.logic = logic

        def get_properties(self, config):
            return {"partition": self.logic.partition}

        def fit(self, parameters, config):
            return self.logic.fit(parameters, int(config.get("round", 0)))

    def client_fn(context):
        torch.set_num_threads(1)
        return FlowerClient(logics_loader(int(context.node_config["partition-id"]))).to_client()

    return ClientApp(client_fn=client_fn)
