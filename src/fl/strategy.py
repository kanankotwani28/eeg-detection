"""Aggregation and client selection shared by the in-process engine and the Flower strategy.

- FedAvg: weighted by client sample size (number of participants for DP clients, epochs otherwise).
- FedProx: same aggregation; clients add (mu/2)||w - w_global||^2 locally.
- FedBN: normalisation parameters stay local to each client; the server averages the rest.
  For server-side/test evaluation the global model carries the average of the clients' norm params.
- Dropout: each client is unavailable in a round with probability p (seeded, identical in both engines).
"""
from __future__ import annotations

import numpy as np


def weighted_average(results: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    total = sum(n for _, n in results)
    return [sum(p[i] * (n / total) for p, n in results) for i in range(len(results[0][0]))]


def select_clients(n_clients: int, rnd: int, dropout_p: float, seed: int) -> list[int]:
    if dropout_p <= 0:
        return list(range(n_clients))
    r = np.random.default_rng([seed, rnd])
    return [c for c in range(n_clients) if r.random() >= dropout_p]


def norm_mask(param_names: list[str]) -> list[bool]:
    return ["norm" in n for n in param_names]


def make_flower_strategy(base_cls, *, n_clients, dropout_p, seed, fedbn, param_names, on_round,
                         initial, fit_config):
    """Build a Flower strategy subclass (legacy Strategy API) with seeded dropout, per-partition
    configs/parameters (FedBN) and participation counting for DP accounting."""
    from flwr.common import FitIns, GetPropertiesIns, ndarrays_to_parameters, parameters_to_ndarrays

    mask = norm_mask(param_names)

    class Tracked(base_cls):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.cid_to_part: dict[str, int] = {}
            self.local_norm: dict[int, list[np.ndarray]] = {}
            self.participation = np.zeros(n_clients, int)

        def _map(self, client_manager):
            client_manager.wait_for(n_clients)
            allc = client_manager.all()
            if len(self.cid_to_part) < len(allc):
                for cid, prox in allc.items():
                    if cid not in self.cid_to_part:
                        r = prox.get_properties(GetPropertiesIns({}), timeout=None, group_id=0)
                        self.cid_to_part[cid] = int(r.properties["partition"])
            return allc

        def configure_fit(self, server_round, parameters, client_manager):
            allc = self._map(client_manager)
            chosen = set(select_clients(n_clients, server_round, dropout_p, seed))
            glob = parameters_to_ndarrays(parameters)
            out = []
            for cid, prox in sorted(allc.items(), key=lambda kv: self.cid_to_part[kv[0]]):
                part = self.cid_to_part[cid]
                if part not in chosen:
                    continue
                params = glob
                if fedbn and part in self.local_norm:
                    loc = iter(self.local_norm[part])
                    params = [next(loc) if m else g for g, m in zip(glob, mask, strict=True)]
                cfg = dict(fit_config(server_round))
                out.append((prox, FitIns(ndarrays_to_parameters(params), cfg)))
                self.participation[part] += 1
            return out

        def aggregate_fit(self, server_round, results, failures):
            if failures:
                raise RuntimeError(f"Client failures in round {server_round}: {failures[:1]}")
            if not results:
                return None, {}
            res = []
            for prox, fr in results:
                nd = parameters_to_ndarrays(fr.parameters)
                part = self.cid_to_part[prox.cid]
                if fedbn:
                    self.local_norm[part] = [p for p, m in zip(nd, mask, strict=True) if m]
                res.append((nd, fr.num_examples))
            agg = weighted_average(res)
            return ndarrays_to_parameters(agg), {}

        def configure_evaluate(self, server_round, parameters, client_manager):
            return []

    return Tracked(fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=1, min_available_clients=n_clients,
                   initial_parameters=ndarrays_to_parameters(initial), evaluate_fn=on_round)
