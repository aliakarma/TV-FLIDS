"""
fl/baselines/bucketing_strategy.py
Bucketing: Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing.
Reference: Karimireddy et al., ICLR 2022.

Key mechanism ("bucket-then-trim"):
  - Randomly shuffle and partition the n received client updates into
    buckets of size s (default s=2).
  - Average the updates within each bucket to obtain one "bucket update"
    per bucket (this pre-averaging reduces the effective heterogeneity/
    variance seen by the base aggregator, which is the point of bucketing).
  - Apply a base robust aggregator -- here, coordinate-wise trimmed mean,
    matching fl/baselines/trimmed_mean_strategy.py's trim logic and beta
    parameter -- over the bucket-level averages instead of the raw
    per-client updates.
  - The bucket assignment is reshuffled every round, seeded on
    `seed + server_round`, for reproducibility.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from attacks.adversarial import apply_min_max_attack_to_params
from fl.ordering import order_results


class BucketingStrategy(FedAvg):
    """
    Bucket-then-trim aggregation.

    Args:
        bucket_size: Number of client updates averaged into a single
            "bucket" before the base aggregator runs (paper default s=2).
        beta:        Trimmed-mean fraction removed from each tail of the
            bucket-level averages (see TrimmedMeanStrategy).
        seed:        Base seed for the per-round bucket shuffle
            (actual round seed is `seed + server_round`).
    """

    def __init__(
        self,
        bucket_size: int = 2,
        beta: float = 0.1,
        global_model: Optional[object] = None,
        attack_type: Optional[str] = None,
        attack_kwargs: Optional[dict] = None,
        malicious_ids: Optional[List[int]] = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.s = max(1, int(bucket_size))
        self.beta = beta
        self.global_model = global_model
        self.attack_type = attack_type
        self.attack_kwargs = attack_kwargs or {}
        self.malicious_ids = malicious_ids or []
        self.seed = seed
        self.round_logs: List[Dict] = []

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

        n = len(params_list)
        n_layers = len(params_list[0])

        # Reshuffle bucket assignment each round (seeded for reproducibility).
        rng = np.random.default_rng(self.seed + server_round)
        order = rng.permutation(n)
        buckets = [order[i:i + self.s] for i in range(0, n, self.s)]

        # Average within each bucket -> one "bucket update" per bucket.
        bucket_params = []
        for bucket in buckets:
            bucket_avg = [
                np.mean([params_list[idx][layer] for idx in bucket], axis=0)
                for layer in range(n_layers)
            ]
            bucket_params.append(bucket_avg)

        n_buckets = len(bucket_params)
        k = int(np.floor(self.beta * n_buckets))  # Trim per tail, bucket-level
        if 2 * k >= n_buckets:
            k = 0

        aggregated = []
        for layer_idx in range(n_layers):
            stacked = np.stack(
                [bucket_params[i][layer_idx] for i in range(n_buckets)], axis=0
            )
            sorted_stacked = np.sort(stacked, axis=0)
            trimmed = sorted_stacked[k:-k] if k > 0 else sorted_stacked
            aggregated.append(np.mean(trimmed, axis=0))

        log = {
            "round":                  server_round,
            "bucketing_bucket_size":  int(self.s),
            "bucketing_num_buckets":  int(n_buckets),
            "bucketing_trim_k":       int(k),
            "bucketing_beta":         self.beta,
            "num_clients":            n,
        }
        self.round_logs.append(log)

        if self.global_model is not None:
            self.global_model.set_parameters(aggregated)

        return ndarrays_to_parameters(aggregated), log
