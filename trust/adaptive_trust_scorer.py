"""
trust/adaptive_trust_scorer.py — Meta-gradient adaptive trust scorer.
Learns α, β, γ via softmax-projected Adam optimization on server validation loss.
Reference: Paper §IV (Stage 4), Eq. (8), Algorithm 1, Lemma 4, Supplementary §S2.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from trust.trust_scorer import TrustScorer
from utils.ste import clip_ste


class AdaptiveTrustScorer(TrustScorer):
    """
    Extends TrustScorer with online meta-gradient weight adaptation (Paper §IV, Stage 4).

    Log-weights: v = (v_alpha, v_beta, v_gamma) in R^3, initialized to (0, 0, 0).
    Softmax projection: (alpha, beta, gamma) = softmax(v) = (1/3, 1/3, 1/3).

    Meta-loss objective:
        L_meta = sum_{i in A} hat{w}_i * l_val(tilde{w}_i)
        hat{w}_i = clip_[0, 1](u_i) / (sum_{j in A} clip_[0, 1](u_j) + eps)
        u_i = alpha * S_i + beta * A_i - gamma * O_i

    Straight-Through Estimator (STE) for clip_[0, 1]:
        Forward: hard clamp to [0, 1]
        Backward: identity gradient (d clip_ste(u) / du = 1)

    Optimization:
        One Adam step on v per round with learning rate eta_meta = 0.01.
        Adam state persists within each simulation run and is reset cleanly between runs.
    """

    def __init__(
        self,
        num_clients: int,
        lambda_up: float = 0.9,
        lambda_down: float = 0.7,
        min_trust: float = 0.01,
        initial_trust: float = 0.5,
        meta_lr: float = 0.01,
        memory_decay: Optional[float] = None,
    ):
        super().__init__(
            num_clients=num_clients,
            alpha=1/3,
            beta=1/3,
            gamma=1/3,
            lambda_up=lambda_up,
            lambda_down=lambda_down,
            min_trust=min_trust,
            initial_trust=initial_trust,
            memory_decay=memory_decay,
        )
        self.meta_lr = float(meta_lr)
        # Log-weights v in R^3, initialized at (0, 0, 0) => softmax(v) = (1/3, 1/3, 1/3)
        self.log_weights = nn.Parameter(torch.zeros(3, dtype=torch.float32), requires_grad=True)
        self.meta_optimizer = torch.optim.Adam([self.log_weights], lr=self.meta_lr)
        self.weight_history: List[Dict[str, float]] = []
        self.saturation_history: List[bool] = []

    @property
    def weights(self) -> torch.Tensor:
        """Normalized positive weights (alpha, beta, gamma) = softmax(v)."""
        return torch.softmax(self.log_weights, dim=0)

    def get_current_weights(self) -> Dict[str, float]:
        """Return current alpha, beta, gamma as a float dictionary."""
        w = self.weights.detach().cpu().numpy()
        return {"alpha": float(w[0]), "beta": float(w[1]), "gamma": float(w[2])}

    def _sync_weights(self) -> None:
        """Synchronize parent TrustScorer scalar weights with current softmax projection."""
        snap = self.get_current_weights()
        self.alpha = snap["alpha"]
        self.beta  = snap["beta"]
        self.gamma = snap["gamma"]

    def compute_meta_loss(
        self,
        similarity_scores: Union[np.ndarray, List[float], torch.Tensor],
        accuracy_scores: Union[np.ndarray, List[float], torch.Tensor],
        anomaly_scores: Union[np.ndarray, List[float], torch.Tensor],
        val_losses: Union[np.ndarray, List[float], torch.Tensor],
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, bool, torch.Tensor, torch.Tensor]:
        """Compute meta-loss L_meta = sum_{i in A} hat{w}_i * l_val(tilde{w}_i) (Paper Eq. (8)).

        All input signals and candidate validation losses are treated as detached constants.
        Gradients flow strictly to log_weights through the STE path:
            v -> softmax(v) = (alpha, beta, gamma) -> u_i -> clip_ste(u_i) -> hat{w}_i -> L_meta

        Returns:
            (meta_loss, is_saturated, raw_u, meta_weights)
        """
        # Detached constant tensors on CPU
        sim_t = torch.as_tensor(similarity_scores, dtype=torch.float32, device=self.log_weights.device).detach()
        acc_t = torch.as_tensor(accuracy_scores, dtype=torch.float32, device=self.log_weights.device).detach()
        anom_t = torch.as_tensor(anomaly_scores, dtype=torch.float32, device=self.log_weights.device).detach()
        loss_t = torch.as_tensor(val_losses, dtype=torch.float32, device=self.log_weights.device).detach()

        w = self.weights
        alpha, beta, gamma = w[0], w[1], w[2]

        # Mixed instantaneous signal u_i = alpha * S_i + beta * A_i - gamma * O_i
        raw_u = alpha * sim_t + beta * acc_t - gamma * anom_t

        # Diagnostic: check if clip binds on any accepted client
        is_saturated = bool(((raw_u < 0.0) | (raw_u > 1.0)).any().item()) if len(raw_u) > 0 else False

        # Straight-through estimator for [0, 1] clip
        clipped_u = clip_ste(raw_u, 0.0, 1.0)
        total = clipped_u.sum()

        if total.item() < eps:
            # Saturated / zero-weight boundary: assign equal weights to avoid division by zero
            meta_weights = torch.ones_like(clipped_u) / max(len(clipped_u), 1)
        else:
            meta_weights = clipped_u / (total + eps)

        meta_loss = (meta_weights * loss_t).sum()
        return meta_loss, is_saturated, raw_u, meta_weights

    def adapt_weights(
        self,
        similarity_scores: Union[np.ndarray, List[float], torch.Tensor],
        accuracy_scores: Union[np.ndarray, List[float], torch.Tensor],
        anomaly_scores: Union[np.ndarray, List[float], torch.Tensor],
        val_losses: Union[np.ndarray, List[float], torch.Tensor],
        eps: float = 1e-8,
    ) -> Dict[str, Union[float, bool]]:
        """Perform one Adam meta-gradient step on log-weights v (Paper §IV, Stage 4, Alg. 1 line 9).

        Handles empty accepted cohorts gracefully without error or state update.
        """
        n_accepted = len(val_losses)
        if n_accepted == 0:
            # Empty accepted cohort: no meta-update performed (Algorithm 1 line 8)
            snap = self.get_current_weights()
            return {
                "alpha": snap["alpha"],
                "beta": snap["beta"],
                "gamma": snap["gamma"],
                "loss": 0.0,
                "saturated": False,
                "step_taken": False,
            }

        self.meta_optimizer.zero_grad()
        meta_loss, is_saturated, _, _ = self.compute_meta_loss(
            similarity_scores, accuracy_scores, anomaly_scores, val_losses, eps=eps
        )

        if not torch.isfinite(meta_loss):
            raise ValueError(f"Non-finite meta-loss encountered: {meta_loss.item()}")

        meta_loss.backward()

        if not torch.isfinite(self.log_weights.grad).all():
            raise ValueError(f"Non-finite meta-gradient encountered on log_weights: {self.log_weights.grad}")

        self.meta_optimizer.step()
        self._sync_weights()

        snap = self.get_current_weights()
        self.weight_history.append(snap)
        self.saturation_history.append(is_saturated)

        return {
            "alpha": snap["alpha"],
            "beta": snap["beta"],
            "gamma": snap["gamma"],
            "loss": float(meta_loss.item()),
            "saturated": is_saturated,
            "step_taken": True,
        }

    def meta_update(self, compute_val_loss_fn: Callable) -> Dict[str, float]:
        """One meta-gradient step on [alpha, beta, gamma] given a differentiable loss function.

        Maintained for backward-compatible custom loss functions and test closures.
        """
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
            if not torch.isfinite(loss):
                raise ValueError(f"val_fn returned non-finite loss: {loss.item()}")
            loss.backward()
            if not torch.isfinite(self.log_weights.grad).all():
                raise ValueError(f"Non-finite gradient on log_weights: {self.log_weights.grad}")
            self.meta_optimizer.step()
        except AssertionError:
            raise
        except Exception as e:
            print(f"[AdaptiveTrust] meta_update skipped: {e}")
            return self.get_current_weights()

        self._sync_weights()
        snap = self.get_current_weights()
        self.weight_history.append(snap)
        return snap

    def update_trust(
        self,
        client_ids: List[int],
        similarity_scores: np.ndarray,
        accuracy_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        participant_ids: Optional[List[int]] = None,
        rejected_ids: Optional[List[int]] = None,
    ) -> np.ndarray:
        self._sync_weights()
        return super().update_trust(
            client_ids,
            similarity_scores,
            accuracy_scores,
            anomaly_scores,
            participant_ids=participant_ids,
            rejected_ids=rejected_ids,
        )

    def get_summary(self) -> Dict[str, float]:
        summary = super().get_summary()
        snap = self.get_current_weights()
        summary.update({
            "adaptive_alpha": snap["alpha"],
            "adaptive_beta": snap["beta"],
            "adaptive_gamma": snap["gamma"],
        })
        if self.saturation_history:
            summary["clip_saturation_rate"] = float(np.mean(self.saturation_history))
        return summary

    def reset(self) -> None:
        """Reset trust scores, log-weights v to (0,0,0), and clean Adam optimizer state."""
        super().reset()
        with torch.no_grad():
            self.log_weights.fill_(0.0)
        # Re-instantiate Adam optimizer so no momentum/variance leaks into next run
        self.meta_optimizer = torch.optim.Adam([self.log_weights], lr=self.meta_lr)
        self.weight_history = []
        self.saturation_history = []
        self.alpha = self.beta = self.gamma = 1/3
