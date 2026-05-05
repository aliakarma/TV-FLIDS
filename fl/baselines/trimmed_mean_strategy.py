"""
fl/baselines/trimmed_mean_strategy.py
Coordinate-wise Trimmed Mean aggregation.
Reference: Yin et al., ICML 2018.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class TrimmedMeanStrategy(FedAvg):
    """
    Coordinate-wise Trimmed Mean.

    For each model parameter dimension, trims the top and bottom
    beta fraction of client values, then averages the remainder.

    At 30% adversarial ratio, use beta >= 0.30 (trim more than attack fraction).

    Args:
        beta: Fraction to trim from each tail (0.1 = trim 10% each side).
    """

    def __init__(self, beta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.round_logs: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        params_list = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        n = len(params_list)
        k = int(np.floor(self.beta * n))   # Values to trim from each tail

        # Safety: cannot trim more than half
        if 2 * k >= n:
            k = 0

        n_layers = len(params_list[0])
        aggregated = []

        for layer_idx in range(n_layers):
            # Stack: shape (n, *layer_shape)
            stacked = np.stack(
                [params_list[i][layer_idx] for i in range(n)], axis=0
            )
            # Sort along client axis (axis=0) and trim both tails
            sorted_stacked = np.sort(stacked, axis=0)
            if k > 0:
                trimmed = sorted_stacked[k:-k]
            else:
                trimmed = sorted_stacked

            aggregated.append(np.mean(trimmed, axis=0))

        log = {
            "round":               server_round,
            "trimmed_mean_k":      int(k),
            "trimmed_mean_beta":   self.beta,
            "num_clients":         n,
        }
        self.round_logs.append(log)

        return ndarrays_to_parameters(aggregated), log
