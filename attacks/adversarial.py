"""
attacks/adversarial.py
Four adversarial attack implementations for FL poisoning experiments.
Reference: Guide §9
"""

import numpy as np
from typing import List, Optional, Tuple


class AdversarialAttackFactory:
    """
    Factory class for creating adversarial perturbations.

    All attacks are implemented as static methods.
    Injection points:
      - Data-level  (before local training): label_flip, backdoor
      - Gradient-level (after local training): gradient_scale, noise_injection
    """

    @staticmethod
    def label_flip(y: np.ndarray, source_class: Optional[int] = None,
                   target_class: int = 0, flip_ratio: float = 1.0,
                   seed: int = 42) -> np.ndarray:
        """
        Label Flipping Attack (Data Poisoning).
        Flips attack labels -> 'normal' (0) to evade IDS detection.

        Args:
            y:            Label array.
            source_class: Which class to flip (None = all non-target classes).
            target_class: Class to flip to (0 = normal).
            flip_ratio:   Fraction of eligible labels to flip.
            seed:         Random seed for reproducibility.
        """
        rng = np.random.RandomState(seed)
        y_corrupt = y.copy()

        if source_class is None:
            eligible = np.where(y_corrupt != target_class)[0]
        else:
            eligible = np.where(y_corrupt == source_class)[0]

        if len(eligible) == 0:
            return y_corrupt

        n_flip = max(1, int(len(eligible) * flip_ratio))
        flip_idx = rng.choice(eligible, size=n_flip, replace=False)
        y_corrupt[flip_idx] = target_class
        return y_corrupt

    @staticmethod
    def gradient_scale(client_params: List[np.ndarray],
                        global_params: List[np.ndarray],
                        scale_factor: float = 10.0) -> List[np.ndarray]:
        """
        Gradient Scaling Attack (Model Poisoning).
        Amplifies gradient update by scale_factor to dominate aggregation.

        Injection point: after local training (gradient-level).
        """
        delta = [c - g for c, g in zip(client_params, global_params)]
        scaled_delta = [scale_factor * d for d in delta]
        return [g + sd for g, sd in zip(global_params, scaled_delta)]

    @staticmethod
    def noise_injection(client_params: List[np.ndarray],
                        noise_std: float = 0.5,
                        seed: int = 42) -> List[np.ndarray]:
        """
        Noise Injection Attack.
        Adds large Gaussian noise to all parameters.

        Injection point: after local training (gradient-level).
        """
        rng = np.random.RandomState(seed)
        return [
            p + rng.normal(0, noise_std, p.shape).astype(np.float32)
            for p in client_params
        ]

    @staticmethod
    def min_max_attack(client_params: List[np.ndarray],
                       global_params: List[np.ndarray],
                       all_updates: List[List[np.ndarray]],
                       gamma: float = 2.0) -> List[np.ndarray]:
        """
        Min-Max Attack (Shejwalkar & Houmansadr, NDSS 2021).
        Maximizes deviation from honest aggregate while staying within the
        norm ball of honest updates, minimizing detectability.

        Reference: https://arxiv.org/abs/2103.06820
        """
        honest_norms = [
            np.linalg.norm(np.concatenate([p.flatten() for p in u]))
            for u in all_updates
        ]
        bound = np.mean(honest_norms) + gamma * np.std(honest_norms)

        delta = [c - g for c, g in zip(client_params, global_params)]
        flat = np.concatenate([d.flatten() for d in delta])
        scale = min(bound / (np.linalg.norm(flat) + 1e-8), gamma)
        return [g + scale * d for g, d in zip(global_params, delta)]

    @staticmethod
    def backdoor_attack(X: np.ndarray, y: np.ndarray,
                        trigger_feature_idx: int = 0,
                        trigger_value: float = 1.0,
                        target_class: int = 0,
                        poison_ratio: float = 0.1,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backdoor Attack (Advanced Data Poisoning).
        Inserts a trigger pattern into training samples, labels them target_class.

        In IoT IDS context: specific feature value acts as trigger causing the
        model to predict 'normal' regardless of actual traffic.

        Injection point: before local training (data-level).
        """
        rng = np.random.RandomState(seed)
        X_poison = X.copy()
        y_poison = y.copy()

        n_poison = max(1, int(len(X) * poison_ratio))
        poison_idx = rng.choice(len(X), size=n_poison, replace=False)

        X_poison[poison_idx, trigger_feature_idx] = trigger_value
        y_poison[poison_idx] = target_class

        return X_poison, y_poison


# Pre-defined attack configurations matching Guide §9.2
ATTACK_CONFIGS = {
    "no_attack":          {"ratio": 0.0,  "type": None},
    "label_flip_10":      {"ratio": 0.10, "type": "label_flip"},
    "label_flip_20":      {"ratio": 0.20, "type": "label_flip"},
    "label_flip_30":      {"ratio": 0.30, "type": "label_flip"},
    "gradient_scale_10":  {"ratio": 0.10, "type": "gradient_scale", "factor": 10.0},
    "gradient_scale_30":  {"ratio": 0.30, "type": "gradient_scale", "factor": 10.0},
    "noise_30":           {"ratio": 0.30, "type": "noise", "std": 0.5},
    "backdoor_20":        {"ratio": 0.20, "type": "backdoor", "poison_ratio": 0.1},
    "min_max_30":         {"ratio": 0.30, "type": "min_max", "gamma": 2.0},
}


def apply_min_max_attack_to_params(
    client_params: List[List[np.ndarray]],
    global_params: List[np.ndarray],
    client_ids: List[int],
    malicious_ids: List[int],
    gamma: float = 2.0,
) -> List[List[np.ndarray]]:
    """Apply Min-Max attack to malicious client parameters in-place."""
    if not malicious_ids:
        return client_params
    malicious_set = set(malicious_ids)
    if not malicious_set:
        return client_params

    all_updates = [
        [c - g for c, g in zip(params, global_params)]
        for params in client_params
    ]

    for idx, cid in enumerate(client_ids):
        if cid in malicious_set:
            client_params[idx] = AdversarialAttackFactory.min_max_attack(
                client_params[idx], global_params, all_updates, gamma=gamma
            )
    return client_params


def get_malicious_client_ids(
    num_clients: int,
    attack_ratio: float,
    seed: int = 42,
) -> List[int]:
    """
    Deterministically select which clients are adversarial.

    Args:
        num_clients:  Total number of FL clients.
        attack_ratio: Fraction to designate as malicious (e.g. 0.30).
        seed:         Random seed for reproducibility.

    Returns:
        Sorted list of adversarial client IDs.
    """
    rng = np.random.default_rng(seed)
    n_malicious = max(0, int(num_clients * attack_ratio))
    if n_malicious == 0:
        return []
    all_ids = np.arange(num_clients)
    malicious = rng.choice(all_ids, n_malicious, replace=False).tolist()
    return sorted(int(x) for x in malicious)
