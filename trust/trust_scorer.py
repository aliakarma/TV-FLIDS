r"""
trust/trust_scorer.py — Dynamic trust scoring for FL clients.
Formula:
    u_i = alpha * S_i + beta * A_i - gamma * O_i
    s_i = clip_{[0, 1]}(u_i) for accepted clients i in A; s_i = 0 for rejected clients i in P \ A
    T_i^{(t)} = max(tau_min, lambda_i * T_i^{(t-1)} + (1 - lambda_i) * s_i)
    lambda_i = lambda_up (0.9) if s_i >= T_i^{(t-1)} else lambda_down (0.7)
    Initial trust T_i^{(0)} = 0.5, trust floor tau_min = 0.01.
Reference: Paper §IV, Eq. (5)–(7), Algorithm 1.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class TrustScorer:
    def __init__(
        self,
        num_clients: int,
        alpha: float = 1/3,
        beta: float = 1/3,
        gamma: float = 1/3,
        lambda_up: float = 0.9,
        lambda_down: float = 0.7,
        min_trust: float = 0.01,
        initial_trust: float = 0.5,
        memory_decay: Optional[float] = None,
    ):
        assert abs(alpha + beta + gamma - 1.0) < 1e-5, \
            f"α+β+γ must equal 1.0, got {alpha+beta+gamma:.4f}"
        self.num_clients = num_clients
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        # Asymmetric trust decay (Paper §IV Eq. (7))
        if memory_decay is not None and lambda_down == 0.7 and lambda_up == 0.9:
            self.lambda_up = float(memory_decay)
            self.lambda_down = float(memory_decay)
            self.decay = float(memory_decay)
        else:
            self.lambda_up = float(lambda_up)
            self.lambda_down = float(lambda_down)
            self.decay = float(lambda_up)

        self.min_trust = float(min_trust)
        self.initial_trust = float(initial_trust)
        self.trust_scores = np.full(num_clients, self.initial_trust, dtype=np.float64)
        self.trust_history: Dict[int, List[float]] = {
            i: [self.initial_trust] for i in range(num_clients)
        }

    def compute_similarity_scores(self, client_updates: List, reference_update: List) -> np.ndarray:
        """Paper §IV Eq. (5): S_i = 1/2 * (cos(tilde{Delta}_i, bar{Delta}_A) + 1).

        Computed on clipped updates relative to mean clipped update of accepted cohort.
        Safe for zero-norm updates (returns 0.5).
        """
        flat_ref = self._flatten(reference_update)
        scores = []
        for u in client_updates:
            flat_u = self._flatten(u)
            cos = self._cosine_sim(flat_u, flat_ref)
            scores.append((cos + 1.0) / 2.0)
        return np.array(scores, dtype=np.float64)

    def compute_accuracy_scores(self, global_loss: float, val_losses_after: List[float],
                                eps: float = 1e-8) -> np.ndarray:
        """Paper §IV Eq. (5):
        A_i = clip_{[-1, 1]}( (l_bal(w^{(t)}) - l_bal(tilde{w}_i)) / (l_bal(w^{(t)}) + eps) )

        Note: A_i can be negative (when candidate worsens class-balanced validation loss).
        It is clipped to [-1, 1], NEVER to [0, 1].
        """
        scores = []
        for la in val_losses_after:
            imp = (global_loss - la) / (global_loss + eps)
            scores.append(float(np.clip(imp, -1.0, 1.0)))
        return np.array(scores, dtype=np.float64)

    def compute_anomaly_scores(
        self,
        client_updates: List,
        tau_z: float = 2.5,
        target_updates: Optional[List] = None,
        cohort_norms: Optional[List[float]] = None,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Paper §IV Eq. (5):
        O_i = 1 - exp(-z_i / tau_z)
        z_i = | ||Delta_i|| - mu_P | / (sigma_P + eps)

        Computed on UNCLIPPED updates over the full participant cohort P.
        If target_updates is provided, returns O_i for target_updates using
        mu_P and sigma_P computed over client_updates (or cohort_norms).
        If target_updates is None, returns O_i for all client_updates.
        """
        if cohort_norms is not None:
            norms = np.array(cohort_norms, dtype=np.float64)
        else:
            norms = np.array([np.linalg.norm(self._flatten(u)) for u in client_updates], dtype=np.float64)

        mu = float(np.mean(norms))
        sigma = float(np.std(norms)) + eps

        if target_updates is not None:
            eval_norms = np.array([np.linalg.norm(self._flatten(u)) for u in target_updates], dtype=np.float64)
        else:
            eval_norms = norms

        z = np.abs((eval_norms - mu) / sigma)
        return 1.0 - np.exp(-z / max(float(tau_z), eps))

    def compute_instantaneous_signal(self, similarity: float, accuracy: float, anomaly: float) -> Tuple[float, float]:
        """Paper §IV Eq. (5):
        u_i = alpha * S_i + beta * A_i - gamma * O_i
        s_i = clip_{[0, 1]}(u_i)
        """
        u = float(self.alpha * similarity + self.beta * accuracy - self.gamma * anomaly)
        s = float(np.clip(u, 0.0, 1.0))
        return u, s

    def update_trust(
        self,
        client_ids: List[int],
        similarity_scores: np.ndarray,
        accuracy_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        participant_ids: Optional[List[int]] = None,
        rejected_ids: Optional[List[int]] = None,
    ) -> np.ndarray:
        r"""Update trust scores according to Paper §IV Eq. (7) and Algorithm 1.

        - For accepted clients i in A:
            u_i = alpha * S_i + beta * A_i - gamma * O_i
            s_i = clip_{[0, 1]}(u_i)
        - For rejected participants j in P \ A:
            s_j = 0.0
        - Asymmetric trust memory update:
            lambda_i = lambda_up (0.9) if s_i >= T_i^{(t-1)} else lambda_down (0.7)
            T_i^{(t)} = max(tau_min, lambda_i * T_i^{(t-1)} + (1 - lambda_i) * s_i)
        - Non-participating clients k not in P:
            T_k^{(t)} = T_k^{(t-1)} (untouched)
        """
        accepted_set = set(client_ids)

        if participant_ids is not None:
            all_participants = list(participant_ids)
            if rejected_ids is not None:
                rej_set = set(rejected_ids)
            else:
                rej_set = set(all_participants) - accepted_set
        elif rejected_ids is not None:
            rej_set = set(rejected_ids)
            all_participants = list(accepted_set | rej_set)
        else:
            # Legacy caller: all client_ids treated as participants
            all_participants = list(client_ids)
            rej_set = set()

        # Update accepted participants
        for idx, cid in enumerate(client_ids):
            sim = float(similarity_scores[idx])
            acc = float(accuracy_scores[idx])
            anom = float(anomaly_scores[idx])
            _, s = self.compute_instantaneous_signal(sim, acc, anom)

            old_t = self.trust_scores[cid]
            lam = self.lambda_up if s >= old_t else self.lambda_down
            new_t = max(self.min_trust, lam * old_t + (1.0 - lam) * s)
            self.trust_scores[cid] = float(new_t)
            self.trust_history[cid].append(float(new_t))

        # Update rejected participants (s_i = 0)
        for cid in rej_set:
            old_t = self.trust_scores[cid]
            s = 0.0
            # s = 0.0 is strictly less than old_t (since old_t >= min_trust > 0), so lam = lambda_down
            lam = self.lambda_up if s >= old_t else self.lambda_down
            new_t = max(self.min_trust, lam * old_t + (1.0 - lam) * s)
            self.trust_scores[cid] = float(new_t)
            self.trust_history[cid].append(float(new_t))

        # Non-participating clients retain their previous trust scores without update.
        return self.trust_scores.copy()

    def get_aggregation_weights(self, client_ids: List[int]) -> np.ndarray:
        selected = self.trust_scores[np.array(client_ids)]
        total = selected.sum()
        return selected / total if total >= 1e-8 else np.ones(len(client_ids)) / len(client_ids)

    def get_summary(self) -> Dict[str, float]:
        return {"mean": float(np.mean(self.trust_scores)), "min": float(np.min(self.trust_scores)),
                "max": float(np.max(self.trust_scores)), "std": float(np.std(self.trust_scores))}

    def reset(self) -> None:
        self.trust_scores = np.full(self.num_clients, self.initial_trust, dtype=np.float64)
        self.trust_history = {i: [self.initial_trust] for i in range(self.num_clients)}

    def _flatten(self, params) -> np.ndarray:
        return np.concatenate([p.flatten() for p in params])

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            cos = float(np.dot(a, b) / (na * nb))
            return float(np.clip(cos, -1.0, 1.0))
        return 0.0

