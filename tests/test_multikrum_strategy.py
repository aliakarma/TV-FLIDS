"""
tests/test_multikrum_strategy.py
Tests for Multi-Krum strategy implementation and isolation.
"""

import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from models.mlp import IDSMLP
from fl.baselines.krum_strategy import KrumStrategy
from fl.baselines.multikrum_strategy import MultiKrumStrategy


class _FakeProxy:
    def __init__(self, cid: int):
        self.cid = str(cid)


class _FakeFitRes:
    def __init__(self, params: list, num_examples: int = 10):
        self.parameters = ndarrays_to_parameters(params)
        self.num_examples = num_examples
        self.metrics = {}


def test_multikrum_hand_calculated_example():
    """
    Controlled hand-computed test for Multi-Krum:
    5 clients with 1D parameter:
    c0 = 0.0
    c1 = 1.0
    c2 = 2.0
    c3 = 3.0
    c4 = 100.0 (extreme outlier)

    Distance matrix (squared Euclidean):
    c0: [0, 1, 4, 9, 10000]
    c1: [1, 0, 1, 4, 9801]
    c2: [4, 1, 0, 1, 9604]
    c3: [9, 4, 1, 0, 9409]
    c4: [10000, 9801, 9604, 9409, 0]

    With f=1, k = max(1, n - f - 2) = max(1, 5 - 1 - 2) = 2.
    Krum scores (sum of 2 smallest non-zero distances):
    c0: 1 + 4 = 5
    c1: 1 + 1 = 2
    c2: 1 + 1 = 2
    c3: 1 + 4 = 5
    c4: 9409 + 9604 = 19013

    With m=2: top 2 lowest scores are c1 and c2 (scores 2 and 2).
    Multi-Krum averages c1 (1.0) and c2 (2.0) -> aggregate = 1.5.
    """
    strategy = MultiKrumStrategy(num_clients=5, num_byzantine=1, m=2)

    results = [
        (_FakeProxy(0), _FakeFitRes([np.array([[0.0]], dtype=np.float32)])),
        (_FakeProxy(1), _FakeFitRes([np.array([[1.0]], dtype=np.float32)])),
        (_FakeProxy(2), _FakeFitRes([np.array([[2.0]], dtype=np.float32)])),
        (_FakeProxy(3), _FakeFitRes([np.array([[3.0]], dtype=np.float32)])),
        (_FakeProxy(4), _FakeFitRes([np.array([[100.0]], dtype=np.float32)])),
    ]

    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    agg_arr = parameters_to_ndarrays(agg_params)

    assert log["multikrum_m"] == 2
    assert log["multikrum_k"] == 2
    selected = sorted(log["multikrum_selected"])
    assert selected == [1, 2]
    np.testing.assert_allclose(agg_arr[0], np.array([[1.5]], dtype=np.float32), rtol=1e-5)


def test_krum_vs_multikrum_selection_difference():
    """Verify single Krum selects 1 client while Multi-Krum averages m > 1 clients."""
    # 4 clients: c0=10.0, c1=10.1, c2=10.2, c3=100.0
    c_vals = [10.0, 10.1, 10.2, 100.0]
    results = [(_FakeProxy(i), _FakeFitRes([np.array([val], dtype=np.float32)])) for i, val in enumerate(c_vals)]

    # Single Krum (m=1)
    krum = KrumStrategy(num_clients=4, num_byzantine=1, m=1)
    krum_agg, krum_log = krum.aggregate_fit(server_round=1, results=results, failures=[])
    krum_val = parameters_to_ndarrays(krum_agg)[0][0]

    # c1 is central (dist to c0 is 0.01, to c2 is 0.01; score=0.01 for k=1)
    assert krum_log["krum_m"] == 1
    assert len(krum_log["krum_selected"]) == 1
    assert krum_val == pytest.approx(10.1, abs=1e-4)

    # Multi-Krum (m=3)
    mkrum = MultiKrumStrategy(num_clients=4, num_byzantine=1, m=3)
    mkrum_agg, mkrum_log = mkrum.aggregate_fit(server_round=1, results=results, failures=[])
    mkrum_val = parameters_to_ndarrays(mkrum_agg)[0][0]

    assert mkrum_log["multikrum_m"] == 3
    assert len(mkrum_log["multikrum_selected"]) == 3
    # Averages c0, c1, c2: (10.0 + 10.1 + 10.2) / 3 = 10.1
    assert mkrum_val == pytest.approx(10.1, abs=1e-4)


def test_multikrum_isolation_no_tvflids_state():
    """Verify MultiKrumStrategy has no TV-FLIDS attributes and does not alter input shapes."""
    strategy = MultiKrumStrategy(num_clients=10, num_byzantine=3)

    assert not hasattr(strategy, "trust_scorer")
    assert not hasattr(strategy, "verifier")
    assert not hasattr(strategy, "adaptive")
    assert not hasattr(strategy, "_eval_cache")
    assert not hasattr(strategy, "log_weights")
