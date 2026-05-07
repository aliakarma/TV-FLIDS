"""
fl/baselines/flame_strategy.py
FLAME: Taming Backdoors in Federated Learning.
Reference: Nguyen et al., USENIX Security 2022.

Key mechanism:
  - HDBSCAN clustering over client updates to identify the largest benign cluster
  - Adaptive norm clipping of benign updates
  - Gaussian noise injection proportional to the clipping norm
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

try:
    import hdbscan
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False

from attacks.adversarial import apply_min_max_attack_to_params


class FLAMEStrategy(FedAvg):
    """
    FLAME aggregation strategy.

    Args:
        num_clients:        Total federation size (used for defaults).
        min_cluster_size:   HDBSCAN minimum cluster size (defaults to 50% of active clients).
        min_samples:        HDBSCAN min_samples.
        noise_multiplier:   Noise scale for adaptive Gaussian injection.
        global_model:       Global model for computing updates.
    """

    def __init__(
        self,
        num_clients: int,
        min_cluster_size: Optional[int] = None,
        min_samples: int = 1,
        noise_multiplier: float = 0.01,
        global_model: Optional[object] = None,
        attack_type: Optional[str] = None,
        attack_kwargs: Optional[dict] = None,
        malicious_ids: Optional[List[int]] = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.noise_multiplier = noise_multiplier
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

        if not _HDBSCAN_AVAILABLE:
            raise ImportError(
                "hdbscan is required for FLAME. Install with: pip install hdbscan"
            )

        client_ids = [int(proxy.cid) for proxy, _ in results]
        params_list = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        num_examples = [fit_res.num_examples for _, fit_res in results]

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

        # Compute updates for clustering
        updates = [[c - g for c, g in zip(cp, global_params)] for cp in params_list]
        flat_updates = np.array(
            [np.concatenate([u.flatten() for u in upd]) for upd in updates],
            dtype=np.float64,
        )

        n_active = len(flat_updates)
        min_cluster_size = self.min_cluster_size
        if min_cluster_size is None:
            min_cluster_size = max(2, int(n_active * 0.5))

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(flat_updates)

        # Largest non-noise cluster is treated as benign
        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            benign_idx = list(range(n_active))
        else:
            unique, counts = np.unique(valid_labels, return_counts=True)
            benign_label = unique[int(np.argmax(counts))]
            benign_idx = np.where(labels == benign_label)[0].tolist()

        if not benign_idx:
            benign_idx = list(range(n_active))

        benign_updates = [updates[i] for i in benign_idx]
        benign_sizes = np.array([num_examples[i] for i in benign_idx], dtype=np.float64)

        flat_benign = flat_updates[benign_idx]
        norms = np.linalg.norm(flat_benign, axis=1)
        clip_norm = float(np.median(norms)) if len(norms) > 0 else 1.0

        clipped_updates = []
        for upd, norm in zip(benign_updates, norms):
            scale = 1.0 if norm < 1e-12 else min(1.0, clip_norm / norm)
            clipped_updates.append([u * scale for u in upd])

        if benign_sizes.sum() <= 0:
            weights = np.ones(len(clipped_updates), dtype=np.float64) / len(clipped_updates)
        else:
            weights = benign_sizes / benign_sizes.sum()

        n_layers = len(global_params)
        aggregated_delta = [
            np.sum(
                [weights[i] * clipped_updates[i][layer] for i in range(len(clipped_updates))],
                axis=0,
            )
            for layer in range(n_layers)
        ]

        noise_std = float(self.noise_multiplier * clip_norm)
        if noise_std > 0:
            rng = np.random.default_rng(self.seed + server_round)
            for layer in range(n_layers):
                aggregated_delta[layer] = aggregated_delta[layer] + rng.normal(
                    0.0, noise_std, aggregated_delta[layer].shape
                ).astype(np.float32)

        aggregated = [g + d for g, d in zip(global_params, aggregated_delta)]

        if self.global_model is not None:
            self.global_model.set_parameters(aggregated)

        log = {
            "round": server_round,
            "flame_benign": int(len(benign_idx)),
            "flame_outliers": int(n_active - len(benign_idx)),
            "flame_clip_norm": float(clip_norm),
            "flame_noise_std": noise_std,
        }
        self.round_logs.append(log)
        return ndarrays_to_parameters(aggregated), log
