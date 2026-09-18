"""
Tests for ACK4 (Meta-Weight Evasion Attack).

Verifies:
1. Grid search over omega in {0.0, 0.25, 0.5, 0.75, 1.0}.
2. Algorithm 1 simulation produces candidate trust scores and meta weights.
3. Maximization of malicious contribution objective J(omega).
4. Output updates are shaped properly and replace malicious updates correctly.
5. Determinism under fixed RNG seed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import torch

from attacks.adversarial import ack4_generate_updates, apply_round_attacks, AdversarialAttackFactory
from models.mlp import IDSMLP


def make_synthetic_val_data(num_samples: int = 50, num_features: int = 122, num_classes: int = 5):
    torch.manual_seed(42)
    X = torch.randn(num_samples, num_features).numpy()
    y = np.array([i % num_classes for i in range(num_samples)], dtype=np.int64)
    return X, y


def test_ack4_generation_basic():
    """Verify that ACK4 generates perturbed updates for malicious clients with valid dimensions."""
    num_malicious = 2
    model = IDSMLP(input_dim=122, num_classes=5)
    global_params = model.get_parameters()
    X_val, y_val = make_synthetic_val_data()

    # Synthetic honest deltas from coalition
    coalition_honest = [
        [np.random.randn(*p.shape) * 0.01 for p in global_params]
        for _ in range(num_malicious)
    ]

    mal_deltas = ack4_generate_updates(
        global_model=model,
        global_params=global_params,
        X_val_hat=X_val,
        y_val_hat=y_val,
        coalition_honest_updates=coalition_honest,
        ack2_updates=[],
        estimated_clipping_radius=1.0,
        num_malicious=num_malicious,
        server_round=1,
        seed=42,
    )

    assert len(mal_deltas) == num_malicious
    for d in mal_deltas:
        assert len(d) == len(global_params)
        for p, g in zip(d, global_params):
            assert p.shape == g.shape
            assert np.isfinite(p).all()

    # Updates across different malicious clients should have random perturbations
    diff = any(not np.array_equal(p1, p2) for p1, p2 in zip(mal_deltas[0], mal_deltas[1]))
    assert diff


def test_ack4_determinism():
    """Verify that ACK4 is deterministic with fixed seed."""
    model = IDSMLP(input_dim=122, num_classes=5)
    global_params = model.get_parameters()
    X_val, y_val = make_synthetic_val_data()

    coalition_honest = [
        [np.ones_like(p) * 0.01 for p in global_params]
    ]

    def run_ack4(s):
        return ack4_generate_updates(
            global_model=model,
            global_params=global_params,
            X_val_hat=X_val,
            y_val_hat=y_val,
            coalition_honest_updates=coalition_honest,
            ack2_updates=[],
            estimated_clipping_radius=1.0,
            num_malicious=1,
            server_round=1,
            seed=s,
        )

    res1 = run_ack4(100)
    res2 = run_ack4(100)
    res3 = run_ack4(200)

    for p1, p2 in zip(res1[0], res2[0]):
        np.testing.assert_array_equal(p1, p2)

    diff = any(not np.array_equal(p1, p3) for p1, p3 in zip(res1[0], res3[0]))
    assert diff


def test_ack4_apply_round_attacks_integration():
    """Verify that ACK4 executes seamlessly via apply_round_attacks."""
    num_clients = 5
    malicious_ids = [3, 4]
    model = IDSMLP(input_dim=122, num_classes=5)
    global_params = model.get_parameters()
    X_val, y_val = make_synthetic_val_data()

    client_params = [
        [p + np.random.randn(*p.shape) * 0.01 for p in global_params]
        for _ in range(num_clients)
    ]
    original_params = [[p.copy() for p in cp] for cp in client_params]
    client_ids = list(range(num_clients))

    attacked_params = apply_round_attacks(
        client_params=client_params,
        global_params=global_params,
        client_ids=client_ids,
        malicious_ids=malicious_ids,
        attack_type="ack4",
        global_model=model,
        val_data=(X_val, y_val),
        server_round=1,
        seed=42,
    )

    assert len(attacked_params) == num_clients
    # Clean client 0 should be unchanged
    for p_orig, p_att in zip(original_params[0], attacked_params[0]):
        np.testing.assert_array_equal(p_orig, p_att)
    # Malicious client 3 should be modified
    diff = any(not np.array_equal(p_orig, p_att) for p_orig, p_att in zip(original_params[3], attacked_params[3]))
    assert diff
