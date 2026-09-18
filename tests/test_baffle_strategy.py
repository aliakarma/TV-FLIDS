"""
tests/test_baffle_strategy.py
Tests for BaFFLe baseline strategy.
Reference: Andreina et al., 2021; Supplementary Table S2.
"""

import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from models.mlp import IDSMLP
from fl.baselines.baffle_strategy import BaFFLeStrategy, _compute_per_class_error_rates


class _FakeProxy:
    def __init__(self, cid: int):
        self.cid = str(cid)


class _FakeFitRes:
    def __init__(self, params: list, num_examples: int = 10):
        self.parameters = ndarrays_to_parameters(params)
        self.num_examples = num_examples
        self.metrics = {}


def make_test_fixture():
    torch.manual_seed(42)
    model = IDSMLP(input_dim=4, num_classes=2)
    X = torch.randn(40, 4)
    y = torch.tensor([0] * 20 + [1] * 20, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=20)
    device = torch.device("cpu")
    return model, loader, device


def test_baffle_error_rate_calculation():
    """Verify per-class error rate computation."""
    model, loader, device = make_test_fixture()
    errs = _compute_per_class_error_rates(model, model.get_parameters(), loader, device)
    assert 0 in errs
    assert 1 in errs
    assert 0.0 <= errs[0] <= 1.0
    assert 0.0 <= errs[1] <= 1.0


def test_baffle_quorum_acceptance_and_rejection():
    """
    Test BaFFLe acceptance when candidate is clean vs rejection when candidate causes class degradation.
    """
    model, loader, device = make_test_fixture()
    w0 = model.get_parameters()

    strategy = BaFFLeStrategy(
        num_clients=10,
        num_validators=5,
        quorum=0.4,
        lookback=5,
        error_threshold=0.05,
        val_loader=loader,
        device=device,
        global_model=model,
    )

    # Round 1: Honest small updates -> accepted
    clean_p = [
        [p + np.random.normal(0, 0.0001, p.shape).astype(np.float32) for p in w0]
        for _ in range(4)
    ]
    results_clean = [(_FakeProxy(i), _FakeFitRes(p)) for i, p in enumerate(clean_p)]
    agg_clean, log1 = strategy.aggregate_fit(server_round=1, results=results_clean, failures=[])

    assert log1["baffle_rejected"] is False
    assert len(strategy.accepted_history) == 1

    # Round 2: Massive corrupt updates that ruin classification -> rejected
    corrupt_p = [
        [p + np.random.normal(5.0, 1.0, p.shape).astype(np.float32) for p in w0]
        for _ in range(4)
    ]
    results_corrupt = [(_FakeProxy(i), _FakeFitRes(p)) for i, p in enumerate(corrupt_p)]
    agg_corrupt, log2 = strategy.aggregate_fit(server_round=2, results=results_corrupt, failures=[])

    # Should be rejected because error rate degraded dramatically
    assert log2["baffle_rejected"] is True
    # Aggregated model should match previous model (rollback)
    agg_arr = parameters_to_ndarrays(agg_corrupt)
    prev_arr = parameters_to_ndarrays(agg_clean)
    for a, p in zip(agg_arr, prev_arr):
        np.testing.assert_allclose(a, p, rtol=1e-5)


def test_baffle_reset_isolation():
    """Verify reset() clears lookback history."""
    strategy = BaFFLeStrategy()
    strategy.accepted_history.append([np.array([1.0])])
    strategy.last_accepted_params = [np.array([1.0])]

    strategy.reset()
    assert len(strategy.accepted_history) == 0
    assert strategy.last_accepted_params is None


def test_baffle_isolation_no_tvflids_state():
    """Verify BaFFLeStrategy has no TV-FLIDS trust memory, gate, or meta-weights."""
    strategy = BaFFLeStrategy()

    assert not hasattr(strategy, "trust_scorer")
    assert not hasattr(strategy, "verifier")
    assert not hasattr(strategy, "adaptive")
    assert not hasattr(strategy, "_eval_cache")
    assert not hasattr(strategy, "log_weights")
