"""
fl/baselines/norm_clipping_strategy.py
Norm Clipping Byzantine-resilient aggregation.
Reference: Sun et al., 2019 ("Can You Really Backdoor Federated Learning?").
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


class NormClippingStrategy(FedAvg):
    """
    Norm Clipping aggregation (Sun et al., 2019).

    Computes updates relative to the previous global model, determines a clipping
    threshold C = clip_factor * median(||Delta_i||), scales any update exceeding C
    down to the spherical boundary C, and averages the clipped updates.

    Does NOT use TV-FLIDS validation gating, trust memory, or adaptive weights.

    Args:
        clip_factor:   Multiplier applied to the median update norm (default: 1.0).
        clip_radius:   Optional fixed clipping threshold override.
        weighted:      Whether to weight clipped updates by sample size (default: False, simple average).
    """

    def __init__(
        self,
        clip_factor: float = 1.0,
        clip_radius: Optional[float] = None,
        weighted: bool = False,
        global_model: Optional[object] = None,
        attack_type: Optional[str] = None,
        attack_kwargs: Optional[dict] = None,
        malicious_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clip_factor = float(clip_factor)
        self.clip_radius = clip_radius
        self.weighted = weighted
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
        sample_counts = [fit_res.num_examples for _, fit_res in results]
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

        n = len(params_list)
        n_layers = len(params_list[0])
        if self.global_model is not None:
            global_params = self.global_model.get_parameters()
            if len(global_params) != n_layers:
                # If test supplied a subset or dummy shape
                global_params = global_params[:n_layers]
        else:
            # Reconstruct baseline from arithmetic mean if no global model provided
            global_params = [
                np.mean([p[l] for p in params_list], axis=0)
                for l in range(n_layers)
            ]

        # ── Step 1: Compute client updates Delta_i = w_i - w^(t) ─────────────
        updates: List[List[np.ndarray]] = []
        raw_norms: List[float] = []

        for p_client in params_list:
            upd = [c - g for c, g in zip(p_client, global_params)]
            flat_upd = np.concatenate([u.flatten() for u in upd])
            norm = float(np.linalg.norm(flat_upd))
            updates.append(upd)
            raw_norms.append(norm)

        # ── Step 2: Determine clipping radius C ──────────────────────────────
        if self.clip_radius is not None:
            c_thresh = float(self.clip_radius)
        else:
            median_norm = float(np.median(raw_norms)) if raw_norms else 0.0
            c_thresh = float(self.clip_factor * median_norm)

        # ── Step 3: Clip updates to radius C ─────────────────────────────────
        clipped_updates: List[List[np.ndarray]] = []
        clipped_count = 0
        eps = 1e-12

        for upd, norm in zip(updates, raw_norms):
            if norm > c_thresh and c_thresh > 0:
                scale = c_thresh / (norm + eps)
                clipped_upd = [u * scale for u in upd]
                clipped_count += 1
            else:
                clipped_upd = upd
            clipped_updates.append(clipped_upd)

        # ── Step 4: Aggregate clipped updates ────────────────────────────────
        n_layers = len(global_params)
        if self.weighted:
            total_samples = sum(sample_counts)
            weights = [s / total_samples for s in sample_counts]
            avg_update = [
                np.sum([w * cupd[l] for w, cupd in zip(weights, clipped_updates)], axis=0)
                for l in range(n_layers)
            ]
        else:
            avg_update = [
                np.mean([cupd[l] for cupd in clipped_updates], axis=0)
                for l in range(n_layers)
            ]

        aggregated = [g + u for g, u in zip(global_params, avg_update)]

        log = {
            "round":          server_round,
            "clip_radius":    c_thresh,
            "clipped_count":  int(clipped_count),
            "num_clients":    int(n),
        }
        self.round_logs.append(log)

        if self.global_model is not None:
            self.global_model.set_parameters(aggregated)

        return ndarrays_to_parameters(aggregated), log
