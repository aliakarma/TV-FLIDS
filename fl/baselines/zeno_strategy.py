"""
fl/baselines/zeno_strategy.py
Zeno: Distributed Machine Learning with Byzantine Heuristics.
Reference: Xie et al., ICML 2019. https://arxiv.org/abs/1905.10053
Supplementary Material Section S2 & Table S2.
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
from attacks.adversarial import apply_round_attacks, apply_min_max_attack_to_params
from fl.ordering import order_results


def _eval_loss_on_loader(
    model: nn.Module,
    parameters: List[np.ndarray],
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate mean cross-entropy loss of given parameters on a dataloader."""
    orig_params = model.get_parameters()
    model.set_parameters(parameters)
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(device), batch[1].to(device)
            else:
                x = batch["features"].to(device)
                y = batch["label"].to(device)

            out = model(x)
            loss = criterion(out, y)
            total_loss += float(loss.item())
            total_samples += len(y)

    model.set_parameters(orig_params)
    return total_loss / max(1, total_samples)


class ZenoStrategy(FedAvg):
    """
    Zeno aggregation strategy (Xie et al., ICML 2019).

    Scores each client's candidate update on a validation dataset as:
        Score_i = (l(w^(t)) - l(w^(t) + eta * Delta_i)) - rho_Z * ||Delta_i||^2
    Selects the top m = D - b_Z highest scoring updates and averages them.

    Args:
        server_model:   Global model instance (nn.Module).
        val_loader:     Server validation dataset loader (D_val).
        device:         Torch device.
        rho:            Update-norm penalty coefficient rho_Z (default: 0.001).
        b:              Number of trimmed clients b_Z (default: 3).
        eta:            Step size multiplier (default: 1.0).
    """

    def __init__(
        self,
        server_model: Optional[nn.Module] = None,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        rho: float = 0.001,
        b: int = 3,
        eta: float = 1.0,
        global_model: Optional[object] = None,
        attack_type: Optional[str] = None,
        attack_kwargs: Optional[dict] = None,
        malicious_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_model = server_model or global_model
        self.val_loader = val_loader
        self.device = device or torch.device("cpu")
        self.rho = float(rho)
        self.b = int(b)
        self.eta = float(eta)
        self.global_model = self.server_model
        self.attack_type = attack_type
        self.attack_kwargs = attack_kwargs or {}
        self.malicious_ids = malicious_ids or []
        self.round_logs: List[Dict] = []
        self.val_data = None
        if val_loader is not None:
            val_X_list, val_y_list = [], []
            for bx, by in val_loader:
                val_X_list.append(bx.numpy() if hasattr(bx, "numpy") else np.array(bx))
                val_y_list.append(by.numpy() if hasattr(by, "numpy") else np.array(by))
            if val_X_list:
                self.val_data = (np.concatenate(val_X_list, axis=0), np.concatenate(val_y_list, axis=0))

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
                val_data=self.val_data,
                server_round=server_round,
            )

        n = len(params_list)
        n_layers = len(params_list[0])

        if self.global_model is not None:
            global_params = self.global_model.get_parameters()
            if len(global_params) != n_layers:
                global_params = global_params[:n_layers]
        else:
            global_params = [
                np.mean([p[l] for p in params_list], axis=0)
                for l in range(n_layers)
            ]

        # ── Step 1: Compute validation loss of reference global model ────────
        if self.val_loader is not None and self.server_model is not None and isinstance(self.server_model, nn.Module):
            l0 = _eval_loss_on_loader(self.server_model, global_params, self.val_loader, self.device)
        else:
            l0 = 0.0

        # ── Step 2: Compute Zeno score for each client ───────────────────────
        zeno_scores: List[float] = []
        for p in params_list:
            upd = [c - g for c, g in zip(p, global_params)]
            flat_upd = np.concatenate([u.flatten() for u in upd])
            sq_norm = float(np.sum(flat_upd ** 2))

            if self.eta != 1.0:
                p_cand = [g + self.eta * u for g, u in zip(global_params, upd)]
            else:
                p_cand = p

            if self.val_loader is not None and self.server_model is not None and isinstance(self.server_model, nn.Module):
                l_cand = _eval_loss_on_loader(self.server_model, p_cand, self.val_loader, self.device)
            else:
                # If no validation set provided, descent is 0
                l_cand = 0.0

            # Score = (l0 - l_cand) - rho_Z * ||Delta_i||^2
            score = (l0 - l_cand) - self.rho * sq_norm
            zeno_scores.append(float(score))

        # ── Step 3: Select top m = max(1, n - b_Z) clients ───────────────────
        m_select = max(1, n - self.b)
        m_select = min(m_select, n)

        # Higher score is better -> sort descending
        ranked_indices = np.argsort(zeno_scores)[::-1]
        selected_indices = ranked_indices[:m_select]

        selected_params = [params_list[i] for i in selected_indices]
        aggregated = [
            np.mean([sp[l] for sp in selected_params], axis=0)
            for l in range(n_layers)
        ]

        log = {
            "round":          server_round,
            "zeno_selected":  [int(client_ids[i]) for i in selected_indices],
            "zeno_m":         int(m_select),
            "zeno_scores":    [float(zeno_scores[i]) for i in selected_indices],
        }
        self.round_logs.append(log)

        if self.global_model is not None:
            self.global_model.set_parameters(aggregated)

        return ndarrays_to_parameters(aggregated), log
