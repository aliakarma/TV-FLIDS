"""
fl/baselines/baffle_strategy.py
BaFFLe: Backdoor Federated Learning Filter.
Reference: Andreina et al., 2021. https://arxiv.org/abs/2011.02167
Supplementary Material Section S2 & Table S2.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Set, Tuple, Union

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from attacks.adversarial import apply_round_attacks, apply_min_max_attack_to_params
from fl.ordering import order_results


def _compute_per_class_error_rates(
    model: nn.Module,
    parameters: List[np.ndarray],
    loader: DataLoader,
    device: torch.device,
) -> Dict[int, float]:
    """Compute error rate (1 - recall) per class on validation data."""
    orig_params = model.get_parameters()
    model.set_parameters(parameters)
    model.eval()
    model.to(device)

    class_correct: Dict[int, int] = {}
    class_total: Dict[int, int] = {}

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(device), batch[1].to(device)
            else:
                x = batch["features"].to(device)
                y = batch["label"].to(device)

            out = model(x)
            preds = torch.argmax(out, dim=1)

            for p, target in zip(preds.view(-1), y.view(-1)):
                c = int(target.item())
                class_total[c] = class_total.get(c, 0) + 1
                if int(p.item()) == c:
                    class_correct[c] = class_correct.get(c, 0) + 1

    model.set_parameters(orig_params)
    error_rates: Dict[int, float] = {}
    for c, tot in class_total.items():
        corr = class_correct.get(c, 0)
        error_rates[c] = 1.0 - (corr / max(1, tot))

    return error_rates


class BaFFLeStrategy(FedAvg):
    """
    BaFFLe aggregation strategy (Andreina et al., 2021).

    Sends candidate global models to a set of validating participants who compare
    per-class error rates with those of recent accepted models. The candidate is
    discarded (keeping the previous model) when rejecting votes reach the quorum.

    Args:
        num_clients:     Total number of clients in federation (default: 20).
        num_validators:  Number of validating clients sampled per round K_val (default: 5).
        quorum:          Rejection threshold fraction (default: 0.3).
        lookback:        Number of past accepted models to compare against L (default: 10).
        error_threshold: Tolerable per-class error rate degradation tau_baffle (default: 0.05).
        val_loader:      Server/client validation dataset loader for error rate evaluation.
        device:          Torch device.
    """

    def __init__(
        self,
        num_clients: int = 20,
        num_validators: int = 5,
        quorum: float = 0.3,
        lookback: int = 10,
        error_threshold: float = 0.05,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        global_model: Optional[object] = None,
        attack_type: Optional[str] = None,
        attack_kwargs: Optional[dict] = None,
        malicious_ids: Optional[List[int]] = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.num_validators = min(num_validators, num_clients)
        self.quorum = float(quorum)
        self.lookback = int(lookback)
        self.error_threshold = float(error_threshold)
        self.val_loader = val_loader
        self.device = device or torch.device("cpu")
        self.global_model = global_model
        self.attack_type = attack_type
        self.attack_kwargs = attack_kwargs or {}
        self.malicious_ids = malicious_ids or []
        self.seed = seed
        self.round_logs: List[Dict] = []
        self.val_data = None
        if val_loader is not None:
            val_X_list, val_y_list = [], []
            for bx, by in val_loader:
                val_X_list.append(bx.numpy() if hasattr(bx, "numpy") else np.array(bx))
                val_y_list.append(by.numpy() if hasattr(by, "numpy") else np.array(by))
            if val_X_list:
                self.val_data = (np.concatenate(val_X_list, axis=0), np.concatenate(val_y_list, axis=0))

        # BaFFLe Lookback History
        self.accepted_history: List[List[np.ndarray]] = []
        self.last_accepted_params: Optional[List[np.ndarray]] = None

    def reset(self):
        """Reset historical lookback state across simulation runs."""
        self.accepted_history.clear()
        self.last_accepted_params = None
        self.round_logs.clear()

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

        # ── Step 1: Form candidate global model via FedAvg ───────────────────
        total_samples = sum(sample_counts)
        weights = [s / total_samples for s in sample_counts]
        w_cand = [
            np.sum([w * p[l] for w, p in zip(weights, params_list)], axis=0)
            for l in range(n_layers)
        ]

        # ── Step 2: Select validator clients ─────────────────────────────────
        rng = np.random.RandomState(self.seed + server_round * 1000)
        validator_ids = rng.choice(self.num_clients, size=self.num_validators, replace=False).tolist()

        # ── Step 3: Compute candidate vs baseline error rates ────────────────
        votes: Dict[int, int] = {}  # 1 = Reject, 0 = Accept

        has_eval_infrastructure = (
            self.val_loader is not None
            and self.global_model is not None
            and isinstance(self.global_model, nn.Module)
        )

        cand_errs: Dict[int, float] = {}
        base_errs: Dict[int, float] = {}
        if has_eval_infrastructure:
            cand_errs = _compute_per_class_error_rates(self.global_model, w_cand, self.val_loader, self.device)
            if self.last_accepted_params is not None:
                base_errs = _compute_per_class_error_rates(self.global_model, self.last_accepted_params, self.val_loader, self.device)
            else:
                base_errs = _compute_per_class_error_rates(self.global_model, global_params, self.val_loader, self.device)

        for vid in validator_ids:
            if vid in self.malicious_ids:
                # Byzantine validator: votes to accept poisoned candidates (0) and reject clean ones (1)
                # Check if malicious clients participated in candidate aggregation
                mal_in_round = any(cid in self.malicious_ids for cid in client_ids)
                vote = 0 if mal_in_round else 1
            else:
                # Honest validator: votes to reject if error rate on any class increased beyond threshold
                if has_eval_infrastructure:
                    max_degradation = 0.0
                    for c, cand_e in cand_errs.items():
                        base_e = base_errs.get(c, cand_e)
                        deg = cand_e - base_e
                        if deg > max_degradation:
                            max_degradation = deg
                    vote = 1 if max_degradation > self.error_threshold else 0
                else:
                    vote = 0
            votes[vid] = vote

        # ── Step 4: Quorum Decision ──────────────────────────────────────────
        reject_count = sum(votes.values())
        reject_ratio = float(reject_count) / max(1, len(votes))
        is_rejected = bool(reject_ratio >= self.quorum)

        if is_rejected:
            # Rejection quorum reached: discard candidate, retain previous global model
            aggregated = global_params
        else:
            # Accepted: update global model and append to lookback history
            aggregated = w_cand
            self.last_accepted_params = [p.copy() for p in aggregated]
            self.accepted_history.append([p.copy() for p in aggregated])
            if len(self.accepted_history) > self.lookback:
                self.accepted_history.pop(0)

        log = {
            "round":          server_round,
            "baffle_rejected": is_rejected,
            "reject_ratio":   reject_ratio,
            "quorum":         self.quorum,
            "num_validators": int(len(validator_ids)),
        }
        self.round_logs.append(log)

        if self.global_model is not None:
            self.global_model.set_parameters(aggregated)

        return ndarrays_to_parameters(aggregated), log
