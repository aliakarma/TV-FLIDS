import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import torch

from attacks.knowledge import KnowledgeTier, ValidationEstimateProvider
from fl.client import TVFLIDSClient
from models.mlp import IDSMLP


def make_synthetic_data(num_samples: int = 100, num_features: int = 122, num_classes: int = 5):
    torch.manual_seed(42)
    X = torch.randn(num_samples, num_features).numpy()
    y = np.array([i % num_classes for i in range(num_samples)], dtype=np.int64)
    return X, y


def test_ack3_relabeling_dos_probe_only():
    """Verify that ACK3 relabels only DoS (1) and Probe (2) to Normal (0), leaving 3 and 4 unchanged."""
    y = np.array([0, 1, 2, 3, 4] * 10, dtype=np.int64)
    y_ack3 = y.copy()
    dos_probe_idx = np.where((y_ack3 == 1) | (y_ack3 == 2))[0]
    y_ack3[dos_probe_idx] = 0

    for orig, adv in zip(y, y_ack3):
        if orig in (1, 2):
            assert adv == 0, f"Class {orig} was not relabeled to 0 (got {adv})"
        else:
            assert adv == orig, f"Class {orig} should have remained {orig} (got {adv})"


def test_ack3_client_fit_execution():
    """Verify that ACK3 runs end-to-end in client.fit and returns valid weights and metrics."""
    X_train, y_train = make_synthetic_data(num_samples=100)
    X_val, y_val = make_synthetic_data(num_samples=40)
    device = torch.device("cpu")
    config = {"local_epochs": 2, "batch_size": 16, "local_lr": 0.001}

    proxy_val = ValidationEstimateProvider.get_validation_estimate(
        tier=KnowledgeTier.K1,
        background_data=(X_train, y_train),
        val_quotas={0: 10, 1: 10, 2: 10, 3: 5, 4: 5},
        seed=42,
    )

    client = TVFLIDSClient(
        client_id=1,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        device=device,
        config=config,
        is_malicious=True,
        attack_type="ack3",
        attack_kwargs={"lambda_1": 0.5, "lambda_2": 0.5, "tau_1": 0.1, "tau_2": 0.1},
        model_kwargs={"input_dim": 122, "num_classes": 5},
        proxy_val_data=proxy_val,
    )

    model = IDSMLP(input_dim=122, num_classes=5)
    initial_weights = model.get_parameters()

    weights, num_samples, metrics = client.fit(initial_weights, {"current_round": 1})

    assert num_samples == len(X_train)
    assert "val_loss" in metrics
    assert "train_time_ms" in metrics
    assert len(weights) == len(initial_weights)

    # Model weights should have changed
    for w_init, w_new in zip(initial_weights, weights):
        assert not np.allclose(w_init, w_new)


def test_ack3_determinism():
    """Verify that ACK3 is deterministic given fixed seed."""
    X_train, y_train = make_synthetic_data(num_samples=60)
    X_val, y_val = make_synthetic_data(num_samples=30)
    device = torch.device("cpu")
    config = {"local_epochs": 1, "batch_size": 16, "local_lr": 0.001}

    def run_ack3(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        client = TVFLIDSClient(
            client_id=2,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            device=device,
            config=config,
            is_malicious=True,
            attack_type="ack3",
            attack_kwargs={"seed": seed},
            model_kwargs={"input_dim": 122, "num_classes": 5},
            proxy_val_data=(X_val, y_val),
        )
        model = IDSMLP(input_dim=122, num_classes=5)
        init_w = model.get_parameters()
        weights, _, metrics = client.fit(init_w, {"current_round": 1})
        return weights, metrics

    w1, m1 = run_ack3(123)
    w2, m2 = run_ack3(123)
    w3, m3 = run_ack3(999)

    for p1, p2 in zip(w1, w2):
        np.testing.assert_array_equal(p1, p2)
    assert m1["val_loss"] == m2["val_loss"]

    # Different seeds should produce different updates
    different = any(not np.allclose(p1, p3) for p1, p3 in zip(w1, w3))
    assert different
