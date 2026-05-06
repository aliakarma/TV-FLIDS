"""
trust/adaptive_trust_scorer.py — Meta-gradient adaptive trust scorer.
Learns α, β, γ via softmax-projected gradient descent on server validation loss.
Reference: Guide §7.3
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional

from trust.trust_scorer import TrustScorer


class AdaptiveTrustScorer(TrustScorer):
    """
    Extends TrustScorer with meta-gradient weight adaptation.
    α, β, γ are learned parameters; softmax enforces positivity + sum-to-1.
    """

    def __init__(self, num_clients: int, memory_decay: float = 0.9,
                 min_trust: float = 0.01, meta_lr: float = 0.01):
        super().__init__(num_clients=num_clients, alpha=1/3, beta=1/3, gamma=1/3,
                         memory_decay=memory_decay, min_trust=min_trust)
        self.meta_lr = meta_lr
        # Unconstrained: softmax([0,0,0]) = [1/3, 1/3, 1/3]
        self.log_weights = nn.Parameter(torch.zeros(3, dtype=torch.float32), requires_grad=True)
        self.meta_optimizer = torch.optim.Adam([self.log_weights], lr=meta_lr)
        self.weight_history: List[Dict[str, float]] = []

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.log_weights, dim=0)

    def get_current_weights(self) -> Dict[str, float]:
        w = self.weights.detach().cpu().numpy()
        return {"alpha": float(w[0]), "beta": float(w[1]), "gamma": float(w[2])}

    def _sync_weights(self) -> None:
        snap = self.get_current_weights()
        self.alpha = snap["alpha"]
        self.beta  = snap["beta"]
        self.gamma = snap["gamma"]

    def meta_update(self, compute_val_loss_fn: Callable) -> Dict[str, float]:
        """One meta-gradient step on [α,β,γ] to minimize server validation loss."""
        self.meta_optimizer.zero_grad()
        w = self.weights
        try:
            loss = compute_val_loss_fn(w[0], w[1], w[2])
            assert isinstance(loss, torch.Tensor), "val_fn must return a Tensor"
            if not loss.requires_grad:
                raise ValueError(
                    "val_fn returned requires_grad=False. "
                    "Ensure computation graph connects to alpha/beta/gamma."
                )
            loss.backward()
            self.meta_optimizer.step()
        except AssertionError:
            raise
        except Exception as e:
            print(f"[AdaptiveTrust] meta_update skipped: {e}")
        self._sync_weights()
        snap = self.get_current_weights()
        self.weight_history.append(snap)
        return snap

    def update_trust(self, client_ids: List[int], similarity_scores: np.ndarray,
                     accuracy_scores: np.ndarray, anomaly_scores: np.ndarray) -> np.ndarray:
        self._sync_weights()
        return super().update_trust(client_ids, similarity_scores, accuracy_scores, anomaly_scores)

    def get_summary(self) -> Dict[str, float]:
        summary = super().get_summary()
        snap = self.get_current_weights()
        summary.update({"adaptive_alpha": snap["alpha"], "adaptive_beta": snap["beta"],
                        "adaptive_gamma": snap["gamma"]})
        return summary

    def reset(self) -> None:
        super().reset()
        with torch.no_grad():
            self.log_weights.fill_(0.0)
        self.weight_history = []
        self.alpha = self.beta = self.gamma = 1/3
