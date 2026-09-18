"""
attacks/knowledge.py
Knowledge tier definitions and validation estimate providers for adversarial attacks.
Reference: IEEE TIFS Manuscript §III-B, Supplementary §S4.

Threat model knowledge tiers:
  - K0: Zero server validation knowledge. The coalition only pools its own local
        training data and resamples to 2,000 records using public class quotas.
        Zero access to server validation data D_val, tuning data D_tune, or peer data.
  - K1: Validation-distribution knowledge. The coalition receives 2,000 records
        with the public class quotas drawn from the dataset outside D_val and D_tune.
        Zero access to D_val or D_tune.
  - K2: Omniscient server validation knowledge. The coalition receives D_val itself
        (an upper bound on adaptive power).
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class KnowledgeTier(str, Enum):
    K0 = "K0"
    K1 = "K1"
    K2 = "K2"


# Table I NSL-KDD Class Quotas for 2,000-sample validation set
NSLKDD_VAL_QUOTAS = {
    0: 1016,  # Normal
    1: 693,   # DoS
    2: 176,   # Probe
    3: 100,   # R2L
    4: 15,    # U2R
}


class ValidationEstimateProvider:
    """
    Provides validation estimate datasets to adversarial attacks strictly respecting
    the boundaries of Knowledge Tiers K0, K1, and K2.
    """

    @staticmethod
    def sample_with_quotas(
        X: np.ndarray,
        y: np.ndarray,
        quotas: Dict[int, int],
        total_samples: int = 2000,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stratified sampling/resampling from (X, y) to match target class quotas.
        If a class is underrepresented, samples with replacement.
        If a class is completely absent, allocates remaining quota to available classes.
        """
        rng = np.random.default_rng(seed)
        selected_indices = []

        all_classes = sorted(quotas.keys())
        remaining_quota = 0

        for c, target_count in quotas.items():
            class_indices = np.where(y == c)[0]
            if len(class_indices) == 0:
                # Class absent in source pool: defer count to fallback
                remaining_quota += target_count
                continue

            if len(class_indices) >= target_count:
                chosen = rng.choice(class_indices, size=target_count, replace=False)
            else:
                chosen = rng.choice(class_indices, size=target_count, replace=True)
            selected_indices.extend(chosen)

        # Handle any unmet quota due to absent classes
        if remaining_quota > 0 and len(selected_indices) > 0:
            available_indices = np.arange(len(y))
            if len(available_indices) > 0:
                fill = rng.choice(available_indices, size=remaining_quota, replace=True)
                selected_indices.extend(fill)

        if len(selected_indices) == 0:
            # Fallback if source data is completely empty
            if len(X) == 0:
                raise ValueError("Cannot sample validation estimate from empty dataset.")
            selected_indices = rng.choice(len(X), size=min(total_samples, len(X)), replace=True)

        selected_indices = np.array(selected_indices)
        # Shuffle final selection
        rng.shuffle(selected_indices)
        return X[selected_indices].copy(), y[selected_indices].copy()

    @classmethod
    def get_validation_estimate(
        cls,
        tier: Union[KnowledgeTier, str],
        coalition_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        background_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        server_val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        val_quotas: Optional[Dict[int, int]] = None,
        total_samples: int = 2000,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve validation estimate according to the specified Knowledge Tier.

        Args:
            tier: KnowledgeTier enum or string ('K0', 'K1', 'K2')
            coalition_data: List of (X, y) client shards held by coalition members (Required for K0)
            background_data: (X_bg, y_bg) from training pool outside D_val and D_tune (Required for K1)
            server_val_data: (X_val, y_val) actual server validation set D_val (Required for K2)
            val_quotas: Per-class target quotas (defaults to NSLKDD_VAL_QUOTAS)
            total_samples: Total validation estimate size (default 2,000)
            seed: Random seed for deterministic resampling

        Returns:
            (X_hat_val, y_hat_val): NumPy arrays of validation estimate
        """
        tier = KnowledgeTier(tier)
        quotas = val_quotas or NSLKDD_VAL_QUOTAS

        if tier == KnowledgeTier.K0:
            if not coalition_data:
                raise ValueError("Knowledge tier K0 requires coalition local data shards.")
            # Pool only coalition members' local data
            X_pool = np.concatenate([X for X, y in coalition_data], axis=0)
            y_pool = np.concatenate([y for X, y in coalition_data], axis=0)
            return cls.sample_with_quotas(X_pool, y_pool, quotas, total_samples=total_samples, seed=seed)

        elif tier == KnowledgeTier.K1:
            if background_data is None:
                raise ValueError("Knowledge tier K1 requires background training pool data.")
            X_bg, y_bg = background_data
            return cls.sample_with_quotas(X_bg, y_bg, quotas, total_samples=total_samples, seed=seed)

        elif tier == KnowledgeTier.K2:
            if server_val_data is None:
                raise ValueError("Knowledge tier K2 requires server validation data D_val.")
            X_val, y_val = server_val_data
            return X_val.copy(), y_val.copy()

        else:
            raise ValueError(f"Unknown knowledge tier: {tier}")
