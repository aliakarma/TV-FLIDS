"""
fl/baselines/krum_strategy.py
Multi-Krum Byzantine-resilient aggregation.
Reference: Blanchard et al., NeurIPS 2017.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from attacks.adversarial import apply_min_max_attack_to_params


class KrumStrategy(FedAvg):
    """
    Multi-Krum aggregation.

    Selects m clients whose updates have the smallest sum of squared
    distances to their n-f-2 nearest neighbors. Averages selected clients.

    Weakness: O(n²·d) computation; requires approximate knowledge of f.

    Args:
        num_clients:    Total federation size.
        num_byzantine:  Estimated number of adversarial clients.
        m:              Number of clients to select (default: n-f-2).
    """

    def __init__(
        self,
        num_clients: int,
        num_byzantine: int,
        m: Optional[int] = None,
        global_model: Optional[object] = None,
        attack_type: Optional[str] = None,
        attack_kwargs: Optional[dict] = None,
        malicious_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = num_clients
        self.f = num_byzantine
        self.m = m if m is not None else max(1, num_clients - num_byzantine - 2)
        self.global_model = global_model
        self.attack_type = attack_type
        self.attack_kwargs = attack_kwargs or {}
        self.malicious_ids = malicious_ids or []
        self.round_logs: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        client_ids = [int(proxy.cid) for proxy, _ in results]
        params_list = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        if self.attack_type == "min_max" and self.global_model is not None:
            global_params = self.global_model.get_parameters()
            params_list = apply_min_max_attack_to_params(
                params_list,
                global_params,
                client_ids,
                self.malicious_ids,
                gamma=self.attack_kwargs.get("gamma", 2.0),
            )

        # Flatten all parameters for distance computation
        flat = np.array([
            np.concatenate([p.flatten() for p in params])
            for params in params_list
        ], dtype=np.float64)  # (n, d)

        n = len(flat)
        if n <= 2:
            # Fallback to mean if too few clients
            agg = [np.mean([p[i] for p in params_list], axis=0)
                   for i in range(len(params_list[0]))]
            if self.global_model is not None:
                self.global_model.set_parameters(agg)
            return ndarrays_to_parameters(agg), {"round": server_round}

        # Pairwise squared Euclidean distances
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.sum((flat[i] - flat[j]) ** 2))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Krum score: sum of distances to k nearest neighbors
        k = max(1, n - self.f - 2)
        scores = []
        for i in range(n):
            sorted_dists = np.sort(dist_matrix[i])
            scores.append(float(np.sum(sorted_dists[1:k + 1])))  # exclude self

        # Select m clients with lowest Krum scores
        m_select = min(self.m, n)
        selected = np.argsort(scores)[:m_select]

        # Average selected clients' parameters
        selected_params = [params_list[i] for i in selected]
        n_layers = len(selected_params[0])
        aggregated = [
            np.mean([sp[layer] for sp in selected_params], axis=0)
            for layer in range(n_layers)
        ]

        log = {
            "round":          server_round,
            "krum_selected":  selected.tolist(),
            "krum_m":         int(m_select),
        }
        self.round_logs.append(log)

        if self.global_model is not None:
            self.global_model.set_parameters(aggregated)

        return ndarrays_to_parameters(aggregated), log
