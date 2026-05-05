"""
fl/baselines/fedavg_strategy.py
Vanilla FedAvg strategy wrapper.
Reference: McMahan et al., AISTATS 2017.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedAvgStrategy(FedAvg):
    """
    Standard FedAvg aggregation.
    Uses data-size weighted averaging of client parameters.
    No defense mechanism — serves as the attack vulnerability baseline.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_logs: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # Weighted average by number of local training samples
        total_samples = sum(fit_res.num_examples for _, fit_res in results)
        params_list = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        n_layers = len(params_list[0][0])
        aggregated = [
            np.sum(
                [params[layer] * (n / total_samples) for params, n in params_list],
                axis=0,
            )
            for layer in range(n_layers)
        ]

        log = {"round": server_round, "num_clients": len(results)}
        self.round_logs.append(log)

        return ndarrays_to_parameters(aggregated), log
