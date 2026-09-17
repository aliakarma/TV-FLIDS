"""
trust/verification.py — Stage 1 (Median Norm Clipping) & Stage 2 (Single Validation Gate).
Reference: Paper §IV, Eq. (3)–(4), Algorithm 1.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import hashlib
import pickle


def clip_updates(updates: List[List[np.ndarray]],
                 cohort_norms: Optional[List[float]] = None
                 ) -> Tuple[List[List[np.ndarray]], float, List[float]]:
    """Stage 1: Median-radius norm clipping (Paper §IV, Eq. (3)).

    C^{(t)} = median_{j in P} ||Delta_j^{(t)}||
    tilde{Delta}_i^{(t)} = Delta_i^{(t)} * min(1, C^{(t)} / ||Delta_i^{(t)}||)

    The median is computed over the FULL sampled participant cohort P.
    Handles zero-norm updates safely without division-by-zero.
    Updates below or on the median radius remain unchanged.
    Updates above the median radius are scaled onto the median-radius boundary.

    Returns:
        (clipped_updates, clipping_radius C, raw_norms)
    """
    n = len(updates)
    if n == 0:
        return [], 0.0, []

    if cohort_norms is None:
        raw_norms = [
            float(np.linalg.norm(np.concatenate([x.flatten() for x in u])))
            for u in updates
        ]
    else:
        raw_norms = [float(val) for val in cohort_norms]

    # Median over the full sampled participant cohort P
    # For D values, mean of order statistics of ranks ceil(D/2) and floor(D/2)+1
    c_radius = float(np.median(raw_norms))

    clipped_updates = []
    for idx, upd in enumerate(updates):
        norm_i = raw_norms[idx]
        if norm_i == 0.0 or norm_i <= c_radius:
            # Below or equal to median (or zero): unchanged
            clipped_updates.append([p.copy() for p in upd])
        else:
            # Scaled exactly onto the median-radius boundary
            scale = c_radius / norm_i
            clipped_updates.append([p * scale for p in upd])

    return clipped_updates, c_radius, raw_norms


class VerificationModule:
    """
    TV-FLIDS Pre-aggregation module:
    Stage 1: Median-radius norm clipping
    Stage 2: Single validation-loss improvement gate (Paper §IV, Eq. (4))
    """
    clip_updates = staticmethod(clip_updates)

    def __init__(self, loss_threshold: float = 0.0, cosine_threshold: float = 0.0,
                 zscore_threshold: float = 2.5):
        self.loss_threshold = loss_threshold
        self.cosine_threshold = cosine_threshold
        self.zscore_threshold = zscore_threshold
        self.verification_log: List[Dict] = []

    def evaluate_validation_gate(
        self,
        clipped_updates: List[List[np.ndarray]],
        client_ids: List[int],
        global_loss: float,
        global_params: List[np.ndarray],
        model: nn.Module,
        device: torch.device,
        val_loader: torch.utils.data.DataLoader,
        eval_cache: Optional[Dict[str, float]] = None,
        loss_threshold: Optional[float] = None,
    ) -> Dict:
        """Stage 2: Single validation-loss gate (Paper §IV, Eq. (4)).

        delta_i = l_val(w^{(t)}) - l_val(tilde{w}_i^{(t)}) >= tau_L(t)

        Evaluates candidate models constructed from CLIPPED updates.
        Rejection is based solely on delta_i < tau_L(t).
        Direction (cosine) and norm (z-score) are not hard filters.
        """
        n = len(clipped_updates)
        if n == 0:
            return {
                'accepted': [], 'rejected': [],
                'verified': [], 'flagged': [],
                'deltas': {}, 'val_losses': {},
                'threshold': float(loss_threshold if loss_threshold is not None else self.loss_threshold),
                'reasons': {},
            }

        tau_l = self.loss_threshold if loss_threshold is None else float(loss_threshold)

        accepted: List[Tuple[int, List[np.ndarray]]] = []
        rejected: List[Tuple[int, List[np.ndarray]]] = []
        deltas: Dict[int, float] = {}
        val_losses: Dict[int, float] = {}
        reasons: Dict[int, str] = {}

        for idx, (cid, upd) in enumerate(zip(client_ids, clipped_updates)):
            # Candidate model from clipped update: tilde{w}_i = w^{(t)} + tilde{Delta}_i
            tentative = [g + u for g, u in zip(global_params, upd)]
            loss_after = self._eval_params(model, tentative, val_loader, device, eval_cache=eval_cache)
            delta = float(global_loss - loss_after)
            deltas[cid] = delta
            val_losses[cid] = float(loss_after)

            if delta < tau_l:
                rejected.append((cid, upd))
                reasons[cid] = f'REJECTED: loss_degradation(ΔL={delta:.4f} < tau_L={tau_l:.4f})'
            else:
                accepted.append((cid, upd))
                reasons[cid] = f'ACCEPTED: loss_improvement(ΔL={delta:.4f} >= tau_L={tau_l:.4f})'

        result = {
            'accepted': accepted,
            'rejected': rejected,
            'verified': accepted,   # backward compatibility alias
            'flagged': [],          # no independent hard flags in single-gate
            'deltas': deltas,
            'val_losses': val_losses,
            'threshold': float(tau_l),
            'reasons': reasons,
        }

        self.verification_log.append({
            'num_accepted': len(accepted),
            'num_verified': len(accepted),
            'num_flagged': 0,
            'num_rejected': len(rejected),
            'loss_threshold': float(tau_l),
            'reasons': reasons,
        })
        return result

    def verify_all(
        self,
        client_updates: List[List[np.ndarray]],
        client_ids: List[int],
        global_loss: float,
        global_params: List[np.ndarray],
        model: nn.Module,
        device: torch.device,
        val_loader: torch.utils.data.DataLoader,
        eval_cache: Optional[Dict[str, float]] = None,
        clip_first: bool = True,
        loss_threshold: Optional[float] = None,
    ) -> Dict:
        """Backward-compatible verification entry point.

        If clip_first is True (default), performs Stage 1 median norm clipping
        before Stage 2 validation gate evaluation.
        """
        if clip_first:
            clipped, _, _ = self.clip_updates(client_updates)
        else:
            clipped = client_updates

        return self.evaluate_validation_gate(
            clipped, client_ids, global_loss, global_params,
            model, device, val_loader, eval_cache=eval_cache,
            loss_threshold=loss_threshold,
        )

    # ── Adaptive threshold helpers ────────────────────────────────────────
    # Both schedules implement the warmup annealing of paper Section IV-A
    # and are driven by the authoritative configuration knob,
    # ``verification.warmup_rounds`` (T_warm, default 20 per paper Table IV).

    @staticmethod
    def adaptive_zscore_threshold(base: float, round_num: int,
                                   warmup_rounds: int = 20,
                                   warmup_offset: float = 0.5) -> float:
        """Check-3 threshold tau_z(t), paper Section IV-A (label eq:tau_anneal).

            tau_z(t) = base + warmup_offset * max(0, 1 - t / T_warm)
        """
        if warmup_rounds <= 0:
            return float(base)
        return float(base + warmup_offset * max(0.0, 1.0 - round_num / warmup_rounds))

    @staticmethod
    def adaptive_loss_threshold(round_num: int, initial: float = -0.1,
                                 final: float = 0.0,
                                 warmup_rounds: int = 20) -> float:
        """Check-1 threshold tau_L(t), paper Section IV-A, Eq. (4).

            tau_L(t) = initial + (final - initial) * min(1, t / T_warm)
            = -0.1 + 0.1 * min(1, t / 20)
        """
        if warmup_rounds <= 0:
            return float(final)
        a = min(round_num / warmup_rounds, 1.0)
        return float(initial * (1 - a) + final * a)

    # ── Private helpers ────────────────────────────────────────────────────
    def _eval_params(self, model: nn.Module, params: List[np.ndarray],
                     val_loader, device: torch.device,
                     eval_cache: Optional[Dict[str, float]] = None) -> float:
        try:
            key = hashlib.sha256(pickle.dumps(params)).hexdigest()
        except Exception:
            key = None

        if key is not None and eval_cache is not None and key in eval_cache:
            return eval_cache[key]

        orig = model.get_parameters()
        model.set_parameters(params)
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total, n = 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                total += criterion(model(X.to(device)), y.to(device)).item()
                n += 1
        model.set_parameters(orig)
        model.train()

        loss = total / max(n, 1)
        if key is not None and eval_cache is not None:
            eval_cache[key] = loss
        return loss

    def _mean_update(self, updates: List[List[np.ndarray]]) -> List[np.ndarray]:
        if not updates:
            return []
        return [np.mean([u[i] for u in updates], axis=0) for i in range(len(updates[0]))]

    def _flatten(self, p: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([x.flatten() for x in p])

    def _norm(self, p: List[np.ndarray]) -> float:
        return float(np.linalg.norm(self._flatten(p)))

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0

    def get_round_summary(self) -> Optional[Dict]:
        return self.verification_log[-1] if self.verification_log else None

