"""
data/partitioning.py
Client data partitioning for FL simulation.

Implements IID (uniform) and Non-IID (Dirichlet) partitioning strategies.
Reference: Guide §3.4
"""

import numpy as np
from typing import List, Tuple


class IIDPartitioner:
    """
    Uniform IID partitioning across clients.
    Each client receives an equal random subset of the global dataset.
    """

    def partition(self, X: np.ndarray, y: np.ndarray,
                  num_clients: int, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        np.random.seed(seed)
        n = len(X)
        indices = np.random.permutation(n)
        splits = np.array_split(indices, num_clients)
        return [(X[s].copy(), y[s].copy()) for s in splits]


class NonIIDPartitioner:
    """
    Dirichlet non-IID partitioning.

    Each client's class distribution is sampled from Dir(alpha).
    - alpha=0.5: Realistic IoT heterogeneity.
    - alpha=0.1: Extreme non-IID (stress test).
    - alpha=100: Approaches IID.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def partition(self, X: np.ndarray, y: np.ndarray,
                  num_clients: int, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        np.random.seed(seed)
        num_classes = len(np.unique(y))
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]

        for cls in range(num_classes):
            cls_idx = np.where(y == cls)[0].copy()
            np.random.shuffle(cls_idx)

            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet(
                alpha=np.repeat(self.alpha, num_clients)
            )
            proportions = (proportions * len(cls_idx)).astype(int)
            # Fix rounding so we use all samples
            proportions[-1] = len(cls_idx) - proportions[:-1].sum()

            start = 0
            for client_id, count in enumerate(proportions):
                count = max(count, 0)
                client_indices[client_id].extend(cls_idx[start:start + count].tolist())
                start += count

        result = []
        for idx_list in client_indices:
            if len(idx_list) == 0:
                # Safety: give client a tiny random sample if empty
                idx_list = np.random.choice(len(X), 10, replace=False).tolist()
            arr = np.array(idx_list)
            result.append((X[arr].copy(), y[arr].copy()))

        return result


def get_partitioner(partition_type: str, alpha: float = 0.5):
    """Factory function for partitioners."""
    if partition_type == "iid":
        return IIDPartitioner()
    elif partition_type == "noniid":
        return NonIIDPartitioner(alpha=alpha)
    else:
        raise ValueError(f"Unknown partition type: {partition_type}. "
                         f"Choose 'iid' or 'noniid'.")


def create_server_validation_set(X: np.ndarray, y: np.ndarray,
                                  val_size: int = 2000,
                                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a stratified server-held validation set.
    This set is NEVER shared with clients.

    Args:
        X: Full training features.
        y: Full training labels.
        val_size: Number of validation samples.
        seed: Random seed.

    Returns:
        (X_val, y_val) tuple.
    """
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(
        X, y,
        test_size=val_size / len(X),
        stratify=y,
        random_state=seed,
    )
    return X_val.astype(np.float32), y_val.astype(np.int64)
