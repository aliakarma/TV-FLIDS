import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import torch

from attacks.adversarial import apply_round_attacks, min_sum_attack, min_max_attack
from models.mlp import IDSMLP


def make_synthetic_val_data(num_samples: int = 40, num_features: int = 122, num_classes: int = 5):
    torch.manual_seed(42)
    X = torch.randn(num_samples, num_features).numpy()
    y = np.array([i % num_classes for i in range(num_samples)], dtype=np.int64)
    return X, y


def test_clean_client_updates_unmutated():
    """Verify that apply_round_attacks leaves clean client parameters bitwise untouched."""
    num_clients = 10
    malicious_ids = [7, 8, 9]
    clean_ids = [i for i in range(num_clients) if i not in malicious_ids]

    model = IDSMLP(input_dim=122, num_classes=5)
    global_params = model.get_parameters()

    # Generate synthetic client params
    np.random.seed(42)
    client_params = []
    original_client_params = {}

    for i in range(num_clients):
        w = [p + np.random.randn(*p.shape) * 0.01 for p in global_params]
        client_params.append(w)
        original_client_params[i] = [np.copy(p) for p in w]

    client_ids = list(range(num_clients))
    val_data = make_synthetic_val_data()

    # Apply Min-Sum attack via interceptor
    attacked_params = apply_round_attacks(
        client_params=client_params,
        global_params=global_params,
        client_ids=client_ids,
        malicious_ids=malicious_ids,
        attack_type="min_sum",
        global_model=model,
        val_data=val_data,
        server_round=1,
        seed=42,
    )

    # Verify clean clients are unmodified
    for cid in clean_ids:
        for p_orig, p_curr in zip(original_client_params[cid], attacked_params[cid]):
            np.testing.assert_array_equal(p_orig, p_curr)

    # Verify malicious clients were modified
    for cid in malicious_ids:
        diff = any(not np.array_equal(p_orig, p_curr) for p_orig, p_curr in zip(original_client_params[cid], attacked_params[cid]))
        assert diff, f"Malicious client {cid} parameters were not modified"


def test_cross_attack_independence():
    """Verify that running Attack A then Attack B does not affect the output of Attack B run in isolation."""
    model = IDSMLP(input_dim=122, num_classes=5)
    global_params = model.get_parameters()
    val_data = make_synthetic_val_data()

    def create_fresh_client_params():
        np.random.seed(123)
        return [
            [p + np.random.randn(*p.shape) * 0.01 for p in global_params]
            for _ in range(5)
        ]

    # Isolated Run of Min-Sum
    params_a = create_fresh_client_params()
    out_ms_isolated = apply_round_attacks(
        client_params=params_a,
        global_params=global_params,
        client_ids=list(range(5)),
        malicious_ids=[3, 4],
        attack_type="min_sum",
        global_model=model,
        val_data=val_data,
        server_round=1,
        seed=42,
    )

    # Sequential Run: Min-Max then Min-Sum on fresh input
    params_mm = create_fresh_client_params()
    _ = apply_round_attacks(
        client_params=params_mm,
        global_params=global_params,
        client_ids=list(range(5)),
        malicious_ids=[3, 4],
        attack_type="min_max",
        global_model=model,
        val_data=val_data,
        server_round=1,
        seed=42,
    )

    params_ms_seq = create_fresh_client_params()
    out_ms_seq = apply_round_attacks(
        client_params=params_ms_seq,
        global_params=global_params,
        client_ids=list(range(5)),
        malicious_ids=[3, 4],
        attack_type="min_sum",
        global_model=model,
        val_data=val_data,
        server_round=1,
        seed=42,
    )

    # Isolated and sequential Min-Sum outputs must match bitwise
    for cid in range(5):
        for p_iso, p_seq in zip(out_ms_isolated[cid], out_ms_seq[cid]):
            np.testing.assert_array_equal(p_iso, p_seq)
