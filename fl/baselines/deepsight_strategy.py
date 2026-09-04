"""
fl/baselines/deepsight_strategy.py
DeepSight: Mitigating Backdoor Attacks in Federated Learning through
Deep Model Inspection.
Reference: Rieger et al., NDSS 2022.

Key mechanism (simplified-fidelity implementation, matching the fidelity
level already used for FLAME/RFA in this repo -- see flame_strategy.py
and rfa_strategy.py -- rather than a paper-exact re-derivation of
DeepSight's full NEUPs/DDifs/cosine ensemble):
  - Clustering-by-bias-gradient: cluster client updates by the cosine
    similarity of their output-layer bias update (for IDSMLP, the final
    nn.Linear's bias -- the last array returned by get_parameters(), a
    cheap proxy for the label distribution an update is pushing towards,
    which is exactly the signal DeepSight's original NEUP/bias-variance
    features are approximating).
  - Majority/minority split: the largest cluster is treated as benign;
    every other (minority) cluster is treated as suspicious and dropped.
  - Weight-clip: each accepted (benign) update's norm is clipped to the
    median norm of the accepted set, then the clipped updates are
    averaged (sample-size weighted). Clipping is what defends against
    magnitude/scaling attacks that preserve update *direction* (and thus
    would not be separated out by the bias-cosine clustering step alone).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from sklearn.cluster import AgglomerativeClustering

from attacks.adversarial import apply_min_max_attack_to_params
from fl.ordering import order_results


class DeepSightStrategy(FedAvg):
    """
    DeepSight aggregation strategy.

    Args:
        distance_threshold: Cosine-distance threshold used by
            AgglomerativeClustering to decide where clusters split over
            the output-layer bias updates.
        global_model:       Global model, used to compute per-client
            updates (delta = client_params - global_params).
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        global_model: Optional[object] = None,
        attack_type: Optional[str] = None,
        attack_kwargs: Optional[dict] = None,
        malicious_ids: Optional[List[int]] = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.distance_threshold = distance_threshold
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

        updates = [[c - g for c, g in zip(cp, global_params)] for cp in params_list]
        n_active = len(updates)
        n_layers = len(global_params)

        if n_active < 2:
            aggregated = [
                np.mean([p[i] for p in params_list], axis=0)
                for i in range(n_layers)
            ]
            if self.global_model is not None:
                self.global_model.set_parameters(aggregated)
            log = {
                "round": server_round,
                "deepsight_benign": int(n_active),
                "deepsight_suspicious": 0,
                "deepsight_clip_norm": 0.0,
            }
            self.round_logs.append(log)
            return ndarrays_to_parameters(aggregated), log

        # Output-layer bias update: last array in the parameter list
        # (for IDSMLP.get_parameters(), the final nn.Linear's bias).
        bias_updates = np.stack(
            [upd[-1].flatten() for upd in updates], axis=0
        ).astype(np.float64)

        if n_active < 3:
            # Too few points for a meaningful cluster split.
            benign_idx = list(range(n_active))
        else:
            # Guard against degenerate (all-zero) bias rows, which are
            # ill-defined under cosine distance.
            norms = np.linalg.norm(bias_updates, axis=1, keepdims=True)
            safe_bias = bias_updates / np.where(norms < 1e-12, 1.0, norms)
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="cosine",
                linkage="average",
            )
            labels = clusterer.fit_predict(safe_bias)
            unique, counts = np.unique(labels, return_counts=True)
            benign_label = unique[int(np.argmax(counts))]
            benign_idx = np.where(labels == benign_label)[0].tolist()

        if not benign_idx:
            benign_idx = list(range(n_active))

        benign_updates = [updates[i] for i in benign_idx]
        flat_benign = np.array(
            [np.concatenate([u.flatten() for u in upd]) for upd in benign_updates],
            dtype=np.float64,
        )
        norms = np.linalg.norm(flat_benign, axis=1)
        clip_norm = float(np.median(norms)) if len(norms) > 0 else 1.0

        clipped_updates = []
        for upd, norm in zip(benign_updates, norms):
            scale = 1.0 if norm < 1e-12 else min(1.0, clip_norm / norm)
            clipped_updates.append([u * scale for u in upd])

        benign_sizes = np.array([num_examples[i] for i in benign_idx], dtype=np.float64)
        if benign_sizes.sum() <= 0:
            weights = np.ones(len(clipped_updates), dtype=np.float64) / len(clipped_updates)
        else:
            weights = benign_sizes / benign_sizes.sum()

        aggregated_delta = [
            np.sum(
                [weights[i] * clipped_updates[i][layer] for i in range(len(clipped_updates))],
                axis=0,
            )
            for layer in range(n_layers)
        ]
        aggregated = [g + d for g, d in zip(global_params, aggregated_delta)]

        log = {
            "round":                  server_round,
            "deepsight_benign":       int(len(benign_idx)),
            "deepsight_suspicious":   int(n_active - len(benign_idx)),
            "deepsight_clip_norm":    float(clip_norm),
        }
        self.round_logs.append(log)

        if self.global_model is not None:
            self.global_model.set_parameters(aggregated)

        return ndarrays_to_parameters(aggregated), log
