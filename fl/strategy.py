"""
fl/strategy.py — TVFLIDSStrategy: Verification → Trust → Weighted Aggregation.
Reference: Guide §14.1
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from trust.trust_scorer import TrustScorer
from trust.adaptive_trust_scorer import AdaptiveTrustScorer
from trust.verification import VerificationModule


class TVFLIDSStrategy(FedAvg):
    """
    Custom Flower strategy: three-criteria verification → dynamic trust scoring
    → trust-weighted aggregation.
    """
    def __init__(self, num_clients: int, config: dict, val_loader: DataLoader,
                 model: nn.Module, device: torch.device, adaptive: bool = True,
                 use_adaptive_thresholds: bool = False, evaluate_fn=None,
                 **kwargs):
        super().__init__(evaluate_fn=evaluate_fn, **kwargs)
        self.num_clients = num_clients
        self.config = config
        self.val_loader = val_loader
        self.model = model
        self.device = device
        self.adaptive = adaptive
        self.use_adaptive_thresholds = use_adaptive_thresholds

        t = config.get('trust', {})
        v = config.get('verification', {})

        self.trust_scorer = (
            AdaptiveTrustScorer(num_clients=num_clients,
                                memory_decay=t.get('memory_decay', 0.9),
                                min_trust=t.get('min_trust', 0.01),
                                meta_lr=t.get('meta_lr', 0.01))
            if adaptive else
            TrustScorer(num_clients=num_clients,
                        alpha=t.get('alpha', 0.4), beta=t.get('beta', 0.4),
                        gamma=t.get('gamma', 0.2),
                        memory_decay=t.get('memory_decay', 0.9),
                        min_trust=t.get('min_trust', 0.01))
        )

        self.verifier = VerificationModule(
            loss_threshold=v.get('loss_threshold', 0.0),
            cosine_threshold=v.get('cosine_threshold', 0.0),
            zscore_threshold=v.get('zscore_threshold', 2.5),
        )
        self.warmup_rounds = v.get('warmup_rounds', 20)
        self.round_logs: List[Dict] = []

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        client_params = [parameters_to_ndarrays(r.parameters) for _, r in results]
        client_ids    = [int(p.cid) for p, _ in results]
        global_params = self.model.get_parameters()

        # Δw_i = w_i^trained − w_global
        updates = [[c - g for c, g in zip(cp, global_params)] for cp in client_params]

        # Adaptive threshold update
        if self.use_adaptive_thresholds:
            self.verifier.zscore_threshold = VerificationModule.adaptive_zscore_threshold(
                self.config['verification'].get('zscore_threshold', 2.5), server_round, self.warmup_rounds)
            self.verifier.loss_threshold = VerificationModule.adaptive_loss_threshold(
                server_round, initial=-0.1, final=0.0, transition=30)

        # ── STEP 1: Verify ────────────────────────────────────────────
        global_loss = self._eval_model(global_params)
        vr = self.verifier.verify_all(
            updates, client_ids, global_loss, global_params,
            self.model, self.device, self.val_loader)

        active = vr['verified'] + vr['flagged']
        if not active:
            log = {'round': server_round, 'all_rejected': 1,
                   'num_verified': 0, 'num_flagged': 0, 'num_rejected': len(vr['rejected'])}
            self.round_logs.append(log)
            return ndarrays_to_parameters(global_params), log

        a_ids  = [cid for cid, _ in active]
        a_upds = [upd for _, upd in active]
        a_pars = [[g + u for g, u in zip(global_params, upd)] for upd in a_upds]

        # ── STEP 2: Trust signals ─────────────────────────────────────
        mean_upd = [np.mean([u[i] for u in a_upds], axis=0) for i in range(len(global_params))]
        sim  = self.trust_scorer.compute_similarity_scores(a_upds, mean_upd)
        va   = [self._eval_model(p) for p in a_pars]
        acc  = self.trust_scorer.compute_accuracy_scores(global_loss, va)
        anom = self.trust_scorer.compute_anomaly_scores(a_upds)

        # ── STEP 3: Update trust + optional meta-gradient ─────────────
        self.trust_scorer.update_trust(a_ids, sim, acc, anom)

        if self.adaptive and isinstance(self.trust_scorer, AdaptiveTrustScorer):
            _sim, _acc, _anom = sim.copy(), acc.copy(), anom.copy()
            _apars = a_pars[:]
            _gpar  = global_params

            def _val_fn(alpha, beta, gamma):
                """
                Differentiable trust-weighted aggregation loss.
                Connects alpha/beta/gamma to validation loss via per-client val losses.
                """
                per_client_losses = torch.tensor(
                    [self._eval_model(_apars[i]) for i in range(len(a_ids))],
                    dtype=torch.float32,
                )
                sim_t = torch.tensor(_sim, dtype=torch.float32)
                acc_t = torch.tensor(_acc, dtype=torch.float32)
                anom_t = torch.tensor(_anom, dtype=torch.float32)

                raw_scores = torch.clamp(
                    alpha * sim_t + beta * acc_t - gamma * anom_t, 0.0, 1.0
                )
                total = raw_scores.sum()
                weights = raw_scores / (total + 1e-8)

                weighted_val_loss = (weights * per_client_losses).sum()
                return weighted_val_loss

            try:
                self.trust_scorer.meta_update(_val_fn)
            except Exception:
                pass

        # ── STEP 4: Weighted aggregation ──────────────────────────────
        weights = self.trust_scorer.get_aggregation_weights(a_ids)
        aggregated = [
            np.sum([weights[i] * a_pars[i][l] for i in range(len(a_ids))], axis=0)
            for l in range(len(global_params))
        ]
        self.model.set_parameters(aggregated)

        ts = self.trust_scorer.get_summary()
        log = {
            'round': server_round, 'global_loss': float(global_loss),
            'num_verified': len(vr['verified']), 'num_flagged': len(vr['flagged']),
            'num_rejected': len(vr['rejected']), 'all_rejected': 0,
            **{f'trust_{k}': float(v) for k, v in ts.items()},
        }
        self.round_logs.append(log)
        return ndarrays_to_parameters(aggregated), log

    def _eval_model(self, params: List[np.ndarray]) -> float:
        orig = self.model.get_parameters()
        self.model.set_parameters(params)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total, n = 0.0, 0
        with torch.no_grad():
            for X, y in self.val_loader:
                total += criterion(self.model(X.to(self.device)), y.to(self.device)).item()
                n += 1
        self.model.set_parameters(orig)
        self.model.train()
        return total / max(n, 1)

    def get_trust_history(self) -> Dict[int, List[float]]:
        return self.trust_scorer.trust_history

    def reset_trust(self) -> None:
        self.trust_scorer.reset()
        self.round_logs.clear()
