import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from attacks.adversarial import label_flip_random
from fl.client import TVFLIDSClient
from models.mlp import IDSMLP


def make_synthetic_data(num_samples: int = 100, num_features: int = 122, num_classes: int = 5):
    torch.manual_seed(42)
    X = torch.randn(num_samples, num_features).numpy()
    y = np.array([i % num_classes for i in range(num_samples)], dtype=np.int64)
    return X, y


def test_lfr_class_remapping_rules():
    """Verify that LF-R leaves normal traffic intact and remaps attack classes to distinct classes."""
    y = np.array([0, 1, 2, 3, 4] * 20)
    adv_y = label_flip_random(y, num_classes=5, seed=42)

    for orig, adv in zip(y, adv_y):
        if orig == 0:
            assert adv == 0, f"Normal sample (class 0) was flipped to {adv}"
        else:
            assert adv != orig, f"Attack sample (class {orig}) was not flipped (remained {adv})"
            assert 1 <= adv < 5, f"Flipped label {adv} out of bounds or mapped to normal"


def test_lfr_determinism():
    """Verify that LF-R produces identical mappings with identical seeds and different mappings with different seeds."""
    y = np.array([0, 1, 2, 3, 4] * 10)

    adv1 = label_flip_random(y, num_classes=5, seed=123)
    adv2 = label_flip_random(y, num_classes=5, seed=123)
    adv3 = label_flip_random(y, num_classes=5, seed=999)

    np.testing.assert_array_equal(adv1, adv2)
    assert not np.array_equal(adv1, adv3)


def test_lfr_client_training():
    """Verify that TVFLIDSClient with attack='lf_r' trains and produces valid weights."""
    X_train, y_train = make_synthetic_data(num_samples=60, num_classes=5)
    X_val, y_val = make_synthetic_data(num_samples=20, num_classes=5)
    device = torch.device("cpu")
    config = {"local_epochs": 1, "batch_size": 16, "local_lr": 0.001}

    client = TVFLIDSClient(
        client_id=5,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        device=device,
        config=config,
        is_malicious=True,
        attack_type="lf_r",
        attack_kwargs={"seed": 42},
        model_kwargs={"input_dim": 122, "num_classes": 5},
    )

    model = IDSMLP(input_dim=122, num_classes=5)
    init_weights = model.get_parameters()

    weights, num_samples, metrics = client.fit(init_weights, {"current_round": 1})
    assert num_samples == len(X_train)
    assert len(weights) == len(init_weights)
    assert "val_loss" in metrics
    assert "train_time_ms" in metrics
