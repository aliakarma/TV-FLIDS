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
from attacks.adversarial import apply_round_attacks, apply_min_max_attack_to_params
from evaluation.overhead import OverheadTracker
from fl.ordering import order_results


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
        config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_model = global_model
        self.attack_type = attack_type
        self.attack_kwargs = attack_kwargs or {}
        self.malicious_ids = malicious_ids or []
        self.round_logs: List[Dict] = []

        # Paper Table XII reports TV-FLIDS's per-round server cost *relative to
        # FedAvg*. That ratio needs FedAvg's own server-side aggregate_fit wall
        # time measured under the identical harness; without it the relative
        # figure has no measured denominator. The timing is symmetric with
        # TVFLIDSStrategy's own tracker (same OverheadTracker, same phase name,
        # same enable flag) so the two numbers are comparable by construction.
        self.track_overhead = bool((config or {}).get(
            'enable_overhead_tracking', True))
        self.overhead_tracker = OverheadTracker()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # Deterministic aggregation order (see fl/ordering.py):
        # Ray yields results in completion order, and float32
        # summation is not associative, so an unsorted round made
        # the same seed drift run to run.
        results = order_results(results)

        # Record the round's mean per-client local-training time (paper
        # Table XII, first row). Clients report it in their FitRes metrics.
        if self.track_overhead:
            _ct = [float(fr.metrics.get("train_time_ms", 0.0))
                   for _, fr in results
                   if getattr(fr, "metrics", None)
                   and "train_time_ms" in fr.metrics]
            if _ct:
                self.overhead_tracker.timings["client_training"].append(
                    float(np.mean(_ct)) / 1000.0)

        _timer = None
        if self.track_overhead:
            _timer = self.overhead_tracker.time_phase("fedavg_total")
            _timer.__enter__()

        client_ids = [int(proxy.cid) for proxy, _ in results]

        # Weighted average by number of local training samples
        total_samples = sum(fit_res.num_examples for _, fit_res in results)
        params_list = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        if self.attack_type and self.global_model is not None:
            global_params = self.global_model.get_parameters()
            params_only = [p for p, _ in params_list]
            params_only = apply_round_attacks(
                client_params=params_only,
                global_params=global_params,
                client_ids=client_ids,
                malicious_ids=self.malicious_ids,
                attack_type=self.attack_type,
                attack_kwargs=self.attack_kwargs,
                global_model=self.global_model,
                server_round=server_round,
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
        if _timer is not None:
            _timer.__exit__(None, None, None)
            _times = self.overhead_tracker.timings.get("fedavg_total")
            if _times:
                log["time_fedavg_total_ms"] = float(_times[-1] * 1000.0)
        self.round_logs.append(log)

        return ndarrays_to_parameters(aggregated), log

    def get_overhead_summary(self) -> Dict[str, float]:
        """Mean per-round server aggregation time, in the same schema
        TVFLIDSStrategy uses, so the two are directly comparable."""
        return self.overhead_tracker.get_summary() if self.track_overhead else {}
