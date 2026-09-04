"""
trust/verification.py — Three-criteria pre-aggregation verification gate.
Checks: (1) loss consistency, (2) cosine similarity, (3) z-score norm outlier.
Reference: Guide §8
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import hashlib
import pickle


class VerificationModule:
    """
    Pre-aggregation gate: Rejected → excluded; Flagged → reduced trust; Verified → normal.
    """
    def __init__(self, loss_threshold: float = 0.0, cosine_threshold: float = 0.0,
                 zscore_threshold: float = 2.5):
        self.loss_threshold = loss_threshold
        self.cosine_threshold = cosine_threshold
        self.zscore_threshold = zscore_threshold
        self.verification_log: List[Dict] = []

    def verify_all(self, client_updates: List[List[np.ndarray]], client_ids: List[int],
                   global_loss: float, global_params: List[np.ndarray],
                   model: nn.Module, device: torch.device,
                   val_loader: torch.utils.data.DataLoader,
                   eval_cache: Optional[Dict[str, float]] = None) -> Dict:
        """Run all 3 checks on each client update. Returns verified/flagged/rejected dicts."""
        n = len(client_updates)
        if n == 0:
            return {'verified': [], 'flagged': [], 'rejected': []}

        pseudo = self._mean_update(client_updates)
        norms = np.array([self._norm(u) for u in client_updates], dtype=np.float64)
        mu, sigma = np.mean(norms), np.std(norms) + 1e-8

        results = {'verified': [], 'flagged': [], 'rejected': []}
        reasons: Dict[int, str] = {}

        for idx, (cid, upd) in enumerate(zip(client_ids, client_updates)):
            flags = []

            # CHECK 1: Loss consistency — does update improve server validation loss?
            tentative = [g + u for g, u in zip(global_params, upd)]
            loss_after = self._eval_params(model, tentative, val_loader, device, eval_cache=eval_cache)
            delta = global_loss - loss_after   # positive = improvement

            if delta < self.loss_threshold:
                results['rejected'].append((cid, upd))
                reasons[cid] = f'REJECTED: loss_degradation(ΔL={delta:.4f})'
                continue

            # CHECK 2: Cosine similarity with pseudo-gradient
            cos = self._cosine(self._flatten(upd), self._flatten(pseudo))
            if cos < self.cosine_threshold:
                flags.append(f'direction_anomaly(cos={cos:.3f})')

            # CHECK 3: Z-score norm outlier
            z = abs((norms[idx] - mu) / sigma)
            if z > self.zscore_threshold:
                flags.append(f'norm_outlier(z={z:.3f})')

            if flags:
                results['flagged'].append((cid, upd))
                reasons[cid] = 'FLAGGED: ' + ', '.join(flags)
            else:
                results['verified'].append((cid, upd))
                reasons[cid] = 'VERIFIED'

        self.verification_log.append({
            'num_verified': len(results['verified']),
            'num_flagged':  len(results['flagged']),
            'num_rejected': len(results['rejected']),
            'reasons': reasons,
        })
        return results

    # ── Adaptive threshold helpers ────────────────────────────────────────
    # Both schedules implement the warmup annealing of paper Section IV-A
    # ("Stage 1: Three-Criteria Verification Gate") and are driven by the
    # SAME authoritative configuration knob, ``verification.warmup_rounds``
    # (T_warm, default 20 per paper Table IV). Neither carries its own
    # independent transition length.

    @staticmethod
    def adaptive_zscore_threshold(base: float, round_num: int,
                                   warmup_rounds: int = 20,
                                   warmup_offset: float = 0.5) -> float:
        """Check-3 threshold tau_z(t), paper Section IV-A (label eq:tau_anneal).

            tau_z(t) = base + warmup_offset * max(0, 1 - t / T_warm)

        With the paper's defaults (base = tau_z = 2.5, warmup_offset = 0.5,
        T_warm = 20) this evaluates to 3.0 at t = 0, decays *linearly* to the
        nominal 2.5 at t = T_warm, and stays at 2.5 for every t > T_warm.

        The annealing is ADDITIVE (an offset above the nominal threshold),
        not multiplicative. A previous implementation multiplied ``base`` by
        a scale factor that started at 3.0 (giving tau_z(0) = 7.5, three times
        the intended leniency) and, worse, switched at t = T_warm to a second,
        exponential branch that jumped the threshold back up to ~4.88 at
        t = T_warm + 1 before decaying — a discontinuity the paper's schedule
        does not have and which made the gate *more* permissive after warmup
        than during it. See tests/test_warmup_schedule.py.
        """
        if warmup_rounds <= 0:
            return float(base)
        return float(base + warmup_offset * max(0.0, 1.0 - round_num / warmup_rounds))

    @staticmethod
    def adaptive_loss_threshold(round_num: int, initial: float = -0.1,
                                 final: float = 0.0,
                                 warmup_rounds: int = 20) -> float:
        """Check-1 threshold tau_L(t), paper Section IV-A.

            tau_L(t) = initial + (final - initial) * min(1, t / T_warm)

        Linearly annealed from ``initial`` (-0.1) to ``final`` (0.0) over the
        first T_warm rounds, then held at ``final``. T_warm is the SAME
        ``warmup_rounds`` that drives Eq. (5); the previous signature took an
        independent ``transition`` argument that call sites hardcoded to 30,
        contradicting both Table IV (T_warm = 20) and the tau_z schedule.
        """
        if warmup_rounds <= 0:
            return float(final)
        a = min(round_num / warmup_rounds, 1.0)
        return float(initial * (1 - a) + final * a)

    # ── Private helpers ────────────────────────────────────────────────────
    def _eval_params(self, model: nn.Module, params: List[np.ndarray],
                     val_loader, device: torch.device,
                     eval_cache: Optional[Dict[str, float]] = None) -> float:
        # Try to compute a deterministic key for these params and consult cache
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
