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
from attacks.adversarial import apply_min_max_attack_to_params


class FedAvgStrategy(FedAvg):
    """
    Standard FedAvg aggregation.
    Uses data-size weighted averaging of client parameters.
    No defense mechanism — serves as the attack vulnerability baseline.
    """

    def __init__(
        self,
        global_model: Optional[object] = None,
        attack_type: Optional[str] = None,
        attack_kwargs: Optional[dict] = None,
        malicious_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
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

        # Weighted average by number of local training samples
        total_samples = sum(fit_res.num_examples for _, fit_res in results)
        params_list = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        if self.attack_type == "min_max" and self.global_model is not None:
            global_params = self.global_model.get_parameters()
            params_only = [p for p, _ in params_list]
            params_only = apply_min_max_attack_to_params(
                params_only,
                global_params,
                client_ids,
                self.malicious_ids,
                gamma=self.attack_kwargs.get("gamma", 2.0),
            )
            params_list = list(zip(params_only, [n for _, n in params_list]))

        n_layers = len(params_list[0][0])
        aggregated = [
            np.sum(
                [params[layer] * (n / total_samples) for params, n in params_list],
                axis=0,
            )
            for layer in range(n_layers)
        ]

        if self.global_model is not None:
            self.global_model.set_parameters(aggregated)

        log = {"round": server_round, "num_clients": len(results)}
        self.round_logs.append(log)

        return ndarrays_to_parameters(aggregated), log
