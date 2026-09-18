"""
fl/baselines/multikrum_strategy.py
Multi-Krum Byzantine-resilient aggregation.
Reference: Blanchard et al., NeurIPS 2017.
Supplementary Material Section S2 & Table S2.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from attacks.adversarial import apply_round_attacks, apply_min_max_attack_to_params
from fl.ordering import order_results


class MultiKrumStrategy(FedAvg):
    """
    Multi-Krum aggregation (Blanchard et al., NeurIPS 2017).

    Selects m clients whose updates have the smallest sum of squared
    Euclidean distances to their n - f' - 2 nearest neighbors.
    Averages the parameters of the selected m clients.

    Args:
        num_clients:    Total federation size (N).
        num_byzantine:  Estimated number of adversarial clients (f').
        m:              Number of clients to select and average (default: max(1, D - f')).
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
        n_active = max(2, int(num_clients * kwargs.get("fraction_fit", 0.5)))
        # Default m: D - f' (or D - 2f', bounded below by 1)
        self.m = m if m is not None else max(1, n_active - num_byzantine)
        self.m = max(1, self.m)
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

        # Deterministic aggregation order
        results = order_results(results)

        client_ids = [int(proxy.cid) for proxy, _ in results]
        params_list = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        if self.attack_type and self.global_model is not None:
            global_params = self.global_model.get_parameters()
            params_list = apply_round_attacks(
                client_params=params_list,
                global_params=global_params,
                client_ids=client_ids,
                malicious_ids=self.malicious_ids,
                attack_type=self.attack_type,
                attack_kwargs=self.attack_kwargs,
                global_model=self.global_model,
                server_round=server_round,
            )

        # Flatten all parameters for distance computation
        flat = np.array([
            np.concatenate([p.flatten() for p in params])
            for params in params_list
        ], dtype=np.float64)  # (n, d)

        n = len(flat)
        if n <= 2:
            # Fallback to mean if too few clients
            agg = [
                np.mean([p[i] for p in params_list], axis=0)
                for i in range(len(params_list[0]))
            ]
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

        # Krum score: sum of distances to k nearest neighbors (excluding self)
        k = max(1, n - self.f - 2)
        scores = []
        for i in range(n):
            sorted_dists = np.sort(dist_matrix[i])
            scores.append(float(np.sum(sorted_dists[1:k + 1])))

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
            "round":               server_round,
            "multikrum_selected":  selected.tolist(),
            "multikrum_m":         int(m_select),
            "multikrum_k":         int(k),
        }
        self.round_logs.append(log)

        if self.global_model is not None:
            self.global_model.set_parameters(aggregated)

        return ndarrays_to_parameters(aggregated), log
