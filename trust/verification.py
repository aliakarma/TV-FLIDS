"""
trust/verification.py — Three-criteria pre-aggregation verification gate.
Checks: (1) loss consistency, (2) cosine similarity, (3) z-score norm outlier.
Reference: Guide §8
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


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
                   val_loader: torch.utils.data.DataLoader) -> Dict:
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
            loss_after = self._eval_params(model, tentative, val_loader, device)
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
    @staticmethod
    def adaptive_zscore_threshold(base: float, round_num: int,
                                   warmup_rounds: int = 20, max_scale: float = 2.0) -> float:
        if round_num <= warmup_rounds:
            scale = max_scale * (1 - round_num / warmup_rounds) + 1.0
        else:
            scale = 1.0 + (max_scale - 1.0) * np.exp(-(round_num - warmup_rounds) / 20.0)
        return base * scale

    @staticmethod
    def adaptive_loss_threshold(round_num: int, initial: float = -0.1,
                                 final: float = 0.0, transition: int = 30) -> float:
        a = min(round_num / transition, 1.0)
        return initial * (1 - a) + final * a

    # ── Private helpers ────────────────────────────────────────────────────
    def _eval_params(self, model: nn.Module, params: List[np.ndarray],
                     val_loader, device: torch.device) -> float:
        tmp = copy.deepcopy(model)
        tmp.set_parameters(params)
        tmp.eval()
        criterion = nn.CrossEntropyLoss()
        total, n = 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                total += criterion(tmp(X.to(device)), y.to(device)).item()
                n += 1
        del tmp
        return total / max(n, 1)

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
