"""
trust/trust_scorer.py — Dynamic trust scoring for FL clients.
Formula: T_i(t) = decay * T_i(t-1) + (1-decay) * [α·S_i + β·A_i − γ·O_i]
Reference: Guide §7.1
"""

import numpy as np
from typing import Dict, List, Optional


class TrustScorer:
    def __init__(self, num_clients: int, alpha: float = 0.4, beta: float = 0.4,
                 gamma: float = 0.2, memory_decay: float = 0.9, min_trust: float = 0.01):
        assert abs(alpha + beta + gamma - 1.0) < 1e-5, \
            f"α+β+γ must equal 1.0, got {alpha+beta+gamma:.4f}"
        self.num_clients = num_clients
        self.alpha = alpha; self.beta = beta; self.gamma = gamma
        self.decay = memory_decay; self.min_trust = min_trust
        self.trust_scores = np.ones(num_clients, dtype=np.float64)
        self.trust_history: Dict[int, List[float]] = {i: [1.0] for i in range(num_clients)}

    def compute_similarity_scores(self, client_updates: List, reference_update: List) -> np.ndarray:
        flat_ref = self._flatten(reference_update)
        scores = []
        for u in client_updates:
            flat_u = self._flatten(u)
            cos = self._cosine_sim(flat_u, flat_ref)
            scores.append((cos + 1.0) / 2.0)
        return np.array(scores, dtype=np.float64)

    def compute_accuracy_scores(self, global_loss: float, val_losses_after: List[float]) -> np.ndarray:
        scores = []
        for la in val_losses_after:
            imp = (global_loss - la) / (global_loss + 1e-8)
            scores.append(float(np.clip(imp, 0.0, 1.0)))
        return np.array(scores, dtype=np.float64)

    def compute_anomaly_scores(self, client_updates: List) -> np.ndarray:
        norms = np.array([np.linalg.norm(self._flatten(u)) for u in client_updates], dtype=np.float64)
        mu, sigma = np.mean(norms), np.std(norms) + 1e-8
        z = np.abs((norms - mu) / sigma)
        return 1.0 - np.exp(-z / 2.5)

    def update_trust(self, client_ids: List[int], similarity_scores: np.ndarray,
                     accuracy_scores: np.ndarray, anomaly_scores: np.ndarray) -> np.ndarray:
        for idx, cid in enumerate(client_ids):
            current = float(np.clip(
                self.alpha * similarity_scores[idx] +
                self.beta  * accuracy_scores[idx] -
                self.gamma * anomaly_scores[idx], 0.0, 1.0))
            new_trust = self.decay * self.trust_scores[cid] + (1.0 - self.decay) * current
            self.trust_scores[cid] = max(float(new_trust), self.min_trust)
            self.trust_history[cid].append(self.trust_scores[cid])
        return self.trust_scores.copy()

    def get_aggregation_weights(self, client_ids: List[int]) -> np.ndarray:
        selected = self.trust_scores[np.array(client_ids)]
        total = selected.sum()
        return selected / total if total >= 1e-8 else np.ones(len(client_ids)) / len(client_ids)

    def get_summary(self) -> Dict[str, float]:
        return {"mean": float(np.mean(self.trust_scores)), "min": float(np.min(self.trust_scores)),
                "max": float(np.max(self.trust_scores)), "std": float(np.std(self.trust_scores))}

    def reset(self) -> None:
        self.trust_scores = np.ones(self.num_clients, dtype=np.float64)
        self.trust_history = {i: [1.0] for i in range(self.num_clients)}

    def _flatten(self, params) -> np.ndarray:
        return np.concatenate([p.flatten() for p in params])

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0
