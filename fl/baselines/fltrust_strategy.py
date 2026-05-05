"""
fl/baselines/fltrust_strategy.py
FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping.
Reference: Cao et al., NDSS 2021. https://arxiv.org/abs/2012.13995

Key mechanism:
  Server trains on a small root dataset to produce a reference update.
  Each client's update is scored by cosine similarity with this reference.
  Updates pointing away (cosine < 0) receive zero weight.
  All selected updates are magnitude-normalized before averaging.

Required: server_root_loader — clean, class-stratified, 1-5% of total data.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FLTrustStrategy(FedAvg):
    """
    FLTrust aggregation strategy.

    Args:
        server_model:      Global model instance.
        server_root_loader: DataLoader for the server's clean root dataset.
        device:            Torch device.
        local_epochs:      Epochs for server root model training.
        lr:                Learning rate for root model training.
    """

    def __init__(
        self,
        server_model: nn.Module,
        server_root_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 1,
        lr: float = 0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_model = server_model
        self.root_loader = server_root_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.round_logs: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        global_params = self.server_model.get_parameters()

        # ── Step 1: Compute server root update ────────────────────────
        root_update = self._compute_root_update(global_params)
        flat_root = np.concatenate([r.flatten() for r in root_update])
        root_norm = float(np.linalg.norm(flat_root))

        # ── Step 2: Score each client update ─────────────────────────
        trust_scores: List[float] = []
        normalized_updates: List[List[np.ndarray]] = []

        for proxy, fit_res in results:
            client_params = parameters_to_ndarrays(fit_res.parameters)
            delta = [c - g for c, g in zip(client_params, global_params)]
            flat_delta = np.concatenate([d.flatten() for d in delta])
            client_norm = float(np.linalg.norm(flat_delta))

            # Cosine similarity with server root update
            if root_norm < 1e-8 or client_norm < 1e-8:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(flat_delta, flat_root) /
                                (client_norm * root_norm))

            # Trust weight: ReLU(cosine) — negative → zero weight
            trust = max(0.0, cos_sim)
            trust_scores.append(trust)

            # Magnitude normalization: scale client update to server norm
            if client_norm > 1e-8:
                scale = root_norm / client_norm
                delta_norm = [d * scale for d in delta]
            else:
                delta_norm = delta

            normalized_updates.append(delta_norm)

        # ── Step 3: Trust-weighted aggregation ────────────────────────
        total_trust = sum(trust_scores)

        if total_trust < 1e-8:
            # All rejected — keep global model
            log = {
                "round": server_round,
                "fltrust_mean_trust": 0.0,
                "fltrust_zero_weight": len(trust_scores),
            }
            self.round_logs.append(log)
            return ndarrays_to_parameters(global_params), log

        n_layers = len(global_params)
        aggregated_delta = [
            np.sum(
                [trust_scores[i] * normalized_updates[i][layer] / total_trust
                 for i in range(len(results))],
                axis=0,
            )
            for layer in range(n_layers)
        ]
        aggregated = [g + d for g, d in zip(global_params, aggregated_delta)]
        self.server_model.set_parameters(aggregated)

        n_zero = sum(1 for t in trust_scores if t < 1e-6)
        log = {
            "round":                server_round,
            "fltrust_mean_trust":   float(np.mean(trust_scores)),
            "fltrust_zero_weight":  n_zero,
        }
        self.round_logs.append(log)

        return ndarrays_to_parameters(aggregated), log

    # ── Root Update Computation ───────────────────────────────────────────

    def _compute_root_update(
        self, original_params: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Train server model on root dataset for local_epochs, return Δw.
        Restores model to original params after computing the update.
        """
        # Load original params
        self.server_model.set_parameters(original_params)
        self.server_model.train()

        optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.lr
        )
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.local_epochs):
            for X_batch, y_batch in self.root_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.server_model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        new_params = self.server_model.get_parameters()
        root_update = [n - o for n, o in zip(new_params, original_params)]

        # Restore to original — root training is for scoring only
        self.server_model.set_parameters(original_params)
        return root_update
