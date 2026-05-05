"""
fl/baselines/foolsgold_strategy.py
FoolsGold: Defending Against Sybil Attacks in Federated Learning.
Reference: Fung et al., 2018. arXiv:1808.04866

Key mechanism:
  Maintains a cumulative gradient history per client.
  Clients with highly similar histories (colluding sybils) are penalized
  by reducing their effective learning rate contribution.
  Does NOT require knowledge of f (number of adversaries).

Weakness: Underperforms against independent non-colluding attackers.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FoolsGoldStrategy(FedAvg):
    """
    FoolsGold aggregation strategy.

    Args:
        num_clients: Total number of clients in federation.
    """

    def __init__(self, num_clients: int, **kwargs):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.histories: Dict[int, np.ndarray] = {}   # client_id → cumulative gradient
        self.round_logs: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # Extract params and flatten each update
        client_ids = [int(proxy.cid) for proxy, _ in results]
        params_list = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        flat_updates = [
            np.concatenate([p.flatten() for p in params]).astype(np.float64)
            for params in params_list
        ]

        # ── Update cumulative gradient histories ────────────────────────
        for cid, flat in zip(client_ids, flat_updates):
            if cid not in self.histories:
                self.histories[cid] = np.zeros_like(flat)
            self.histories[cid] = self.histories[cid] + flat

        # ── Compute pairwise cosine similarity of histories ─────────────
        history_vecs = np.array([self.histories[cid] for cid in client_ids])
        norms = np.linalg.norm(history_vecs, axis=1, keepdims=True) + 1e-8
        normalized = history_vecs / norms
        cosine_sim = normalized @ normalized.T   # (n, n) similarity matrix

        n = len(client_ids)

        # ── FoolsGold contribution weights ─────────────────────────────
        lr_frac = np.ones(n, dtype=np.float64)
        for i in range(n):
            sims = [cosine_sim[i, j] for j in range(n) if j != i]
            max_sim = max(sims) if sims else 0.0
            lr_frac[i] = 1.0 - max_sim   # High similarity → low contribution

        lr_frac = np.clip(lr_frac, 0.0, 1.0)

        total = lr_frac.sum()
        if total < 1e-8:
            weights = np.ones(n) / n
        else:
            weights = lr_frac / total

        # ── Weighted average ───────────────────────────────────────────
        n_layers = len(params_list[0])
        aggregated = [
            np.sum(
                [weights[i] * params_list[i][layer] for i in range(n)],
                axis=0,
            )
            for layer in range(n_layers)
        ]

        log = {
            "round":                    server_round,
            "foolsgold_mean_lr":        float(np.mean(lr_frac)),
            "foolsgold_penalized":      int(np.sum(lr_frac < 0.5)),
        }
        self.round_logs.append(log)

        return ndarrays_to_parameters(aggregated), log
