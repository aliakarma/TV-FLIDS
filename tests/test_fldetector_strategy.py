"""
tests/test_fldetector_strategy.py
Tests for FLDetector baseline strategy.
Reference: Zhang et al., KDD 2022; Supplementary Table S2.
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
from fl.baselines.fldetector_strategy import FLDetectorStrategy, _lbfgs_hessian_vector_product


class _FakeProxy:
    def __init__(self, cid: int):
        self.cid = str(cid)


class _FakeFitRes:
    def __init__(self, params: list, num_examples: int = 10):
        self.parameters = ndarrays_to_parameters(params)
        self.num_examples = num_examples
        self.metrics = {}


class _DummyModel:
    def __init__(self, params):
        self.params = params

    def get_parameters(self):
        return self.params

    def set_parameters(self, p):
        self.params = p


def test_lbfgs_hessian_vector_product_identity():
    """Verify L-BFGS with empty history returns identity copy."""
    v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    r = _lbfgs_hessian_vector_product([], [], v)
    np.testing.assert_allclose(r, v)


def test_fldetector_warmup_and_detection():
    """
    Test FLDetector warm-up rounds (t < start_round) vs active detection rounds (t >= start_round).
    """
    model = _DummyModel([np.array([0.0, 0.0], dtype=np.float32)])
    strategy = FLDetectorStrategy(window=5, start_round=3, beta=0.0, num_clients=4, global_model=model)

    # Round 1 (warm-up): 4 clients submit consistent gradients
    r1 = [
        (_FakeProxy(0), _FakeFitRes([np.array([1.0, 1.0], dtype=np.float32)])),
        (_FakeProxy(1), _FakeFitRes([np.array([1.1, 1.0], dtype=np.float32)])),
        (_FakeProxy(2), _FakeFitRes([np.array([0.9, 1.0], dtype=np.float32)])),
        (_FakeProxy(3), _FakeFitRes([np.array([1.0, 0.9], dtype=np.float32)])),
    ]
    _, log1 = strategy.aggregate_fit(server_round=1, results=r1, failures=[])
    assert log1["num_accepted"] == 4
    assert len(log1["fldetector_flagged"]) == 0

    # Round 2 (warm-up)
    r2 = [
        (_FakeProxy(0), _FakeFitRes([np.array([2.0, 2.0], dtype=np.float32)])),
        (_FakeProxy(1), _FakeFitRes([np.array([2.1, 2.0], dtype=np.float32)])),
        (_FakeProxy(2), _FakeFitRes([np.array([1.9, 2.0], dtype=np.float32)])),
        (_FakeProxy(3), _FakeFitRes([np.array([2.0, 1.9], dtype=np.float32)])),
    ]
    _, log2 = strategy.aggregate_fit(server_round=2, results=r2, failures=[])
    assert log2["num_accepted"] == 4

    # Round 3 (detection active): client 3 suddenly flips gradient sign (malicious inversion)
    r3 = [
        (_FakeProxy(0), _FakeFitRes([np.array([3.0, 3.0], dtype=np.float32)])),
        (_FakeProxy(1), _FakeFitRes([np.array([3.1, 3.0], dtype=np.float32)])),
        (_FakeProxy(2), _FakeFitRes([np.array([2.9, 3.0], dtype=np.float32)])),
        (_FakeProxy(3), _FakeFitRes([np.array([-50.0, -50.0], dtype=np.float32)])),  # Adversary
    ]
    _, log3 = strategy.aggregate_fit(server_round=3, results=r3, failures=[])
    # Client 3 should be flagged
    assert 3 in log3["fldetector_flagged"]
    assert log3["num_accepted"] == 3


def test_fldetector_reset_isolation():
    """Verify reset() clears all history and detected client state."""
    strategy = FLDetectorStrategy(window=5, start_round=2)
    strategy.s_history.append(np.array([1.0, 2.0]))
    strategy.y_history.append(np.array([0.1, 0.2]))
    strategy.client_history[0] = np.array([1.0])
    strategy.detected_clients.add(0)

    strategy.reset()

    assert len(strategy.s_history) == 0
    assert len(strategy.y_history) == 0
    assert len(strategy.client_history) == 0
    assert len(strategy.detected_clients) == 0


def test_fldetector_isolation_no_tvflids_state():
    """Verify FLDetectorStrategy has no TV-FLIDS trust, gating, or meta-weight attributes."""
    strategy = FLDetectorStrategy()

    assert not hasattr(strategy, "trust_scorer")
    assert not hasattr(strategy, "verifier")
    assert not hasattr(strategy, "adaptive")
    assert not hasattr(strategy, "_eval_cache")
    assert not hasattr(strategy, "log_weights")
