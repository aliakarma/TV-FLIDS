"""
fl/baselines/rfa_strategy.py
RFA: Robust Aggregation for Federated Learning (Geometric Median).
Reference: Pillutla et al., IEEE TSP 2022.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from attacks.adversarial import apply_min_max_attack_to_params


class RFAStrategy(FedAvg):
    """
    RFA aggregation strategy using the smoothed Weiszfeld algorithm.

    Args:
        max_iter:  Maximum number of Weiszfeld iterations.
        tol:       Convergence tolerance.
        eps:       Smoothing term to avoid division by zero.
    """

    def __init__(
        self,
        max_iter: int = 10,
        tol: float = 1e-5,
        eps: float = 1e-6,
        global_model: Optional[object] = None,
        attack_type: Optional[str] = None,
        attack_kwargs: Optional[dict] = None,
        malicious_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
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
        num_examples = np.array([fit_res.num_examples for _, fit_res in results], dtype=np.float64)

        if self.global_model is None:
            global_params = params_list[0]
        else:
            global_params = self.global_model.get_parameters()

        if self.attack_type == "min_max" and self.global_model is not None:
            params_list = apply_min_max_attack_to_params(
                params_list,
                global_params,
                client_ids,
                self.malicious_ids,
                gamma=self.attack_kwargs.get("gamma", 2.0),
            )

        shapes = [p.shape for p in params_list[0]]

        def flatten(params: List[np.ndarray]) -> np.ndarray:
            return np.concatenate([p.flatten() for p in params]).astype(np.float64)

        def unflatten(flat: np.ndarray) -> List[np.ndarray]:
            out = []
            idx = 0
            for shape in shapes:
                size = int(np.prod(shape))
                out.append(flat[idx:idx + size].reshape(shape).astype(np.float32))
                idx += size
            return out

        flat_params = np.stack([flatten(p) for p in params_list], axis=0)

        if num_examples.sum() <= 0:
            weights = np.ones(len(params_list), dtype=np.float64) / len(params_list)
        else:
            weights = num_examples / num_examples.sum()

        x = np.sum(flat_params * weights[:, None], axis=0)
        iters = 0
        for iters in range(1, self.max_iter + 1):
            diffs = flat_params - x
            distances = np.linalg.norm(diffs, axis=1)
            distances = np.maximum(distances, self.eps)
            w = weights / distances
            x_new = np.sum(flat_params * w[:, None], axis=0) / np.sum(w)
            if np.linalg.norm(x_new - x) < self.tol:
                x = x_new
                break
            x = x_new

        aggregated = unflatten(x)

        if self.global_model is not None:
            self.global_model.set_parameters(aggregated)

        log = {
            "round": server_round,
            "rfa_iters": int(iters),
            "rfa_tol": float(self.tol),
        }
        self.round_logs.append(log)
        return ndarrays_to_parameters(aggregated), log
