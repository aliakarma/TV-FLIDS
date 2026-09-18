"""
tests/test_norm_clipping_strategy.py
Tests for Norm Clipping baseline strategy.
Reference: Sun et al., 2019; Supplementary Table S2.
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
from fl.baselines.norm_clipping_strategy import NormClippingStrategy


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


def test_norm_clipping_exact_math():
    """
    Test exact Norm Clipping mathematics on a controlled 1D update setting.
    Global model: w = [0.0]
    4 clients submit updates:
      c0: +1.0 (norm = 1.0)
      c1: +2.0 (norm = 2.0)
      c2: +3.0 (norm = 3.0)
      c3: +10.0 (norm = 10.0)

    Raw norms: [1.0, 2.0, 3.0, 10.0]
    Median norm M = median([1.0, 2.0, 3.0, 10.0]) = (2.0 + 3.0)/2 = 2.5.
    With clip_factor = 1.0, C = 2.5.

    Clipped updates:
      c0: 1.0 <= 2.5 -> 1.0
      c1: 2.0 <= 2.5 -> 2.0
      c2: 3.0 > 2.5  -> 2.5 * (3.0 / 3.0) = 2.5
      c3: 10.0 > 2.5 -> 2.5 * (10.0 / 10.0) = 2.5

    Average clipped update: (1.0 + 2.0 + 2.5 + 2.5) / 4 = 8.0 / 4 = 2.0.
    New global model: 0.0 + 2.0 = 2.0.
    """
    model = _DummyModel([np.array([0.0], dtype=np.float32)])

    strategy = NormClippingStrategy(clip_factor=1.0, global_model=model)

    results = [
        (_FakeProxy(0), _FakeFitRes([np.array([1.0], dtype=np.float32)])),
        (_FakeProxy(1), _FakeFitRes([np.array([2.0], dtype=np.float32)])),
        (_FakeProxy(2), _FakeFitRes([np.array([3.0], dtype=np.float32)])),
        (_FakeProxy(3), _FakeFitRes([np.array([10.0], dtype=np.float32)])),
    ]

    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    agg_arr = parameters_to_ndarrays(agg_params)

    assert log["clip_radius"] == pytest.approx(2.5, abs=1e-5)
    assert log["clipped_count"] == 2  # c2 and c3 clipped
    np.testing.assert_allclose(agg_arr[0], np.array([2.0], dtype=np.float32), rtol=1e-5)


def test_norm_clipping_multiplier_grid():
    """Verify clip_factor multiplier scaling (radius in {0.5, 1, 1.5, 2} x median norm)."""
    results = [
        (_FakeProxy(0), _FakeFitRes([np.array([2.0], dtype=np.float32)])),
        (_FakeProxy(1), _FakeFitRes([np.array([4.0], dtype=np.float32)])),
    ]
    # Median norm is 3.0 when global model is 0.0

    # clip_factor = 0.5 -> C = 1.5. Both clipped to 1.5. Average = 1.5.
    model_05 = _DummyModel([np.array([0.0], dtype=np.float32)])
    s05 = NormClippingStrategy(clip_factor=0.5, global_model=model_05)
    agg, log = s05.aggregate_fit(server_round=1, results=results, failures=[])
    assert log["clip_radius"] == pytest.approx(1.5, abs=1e-5)
    assert parameters_to_ndarrays(agg)[0][0] == pytest.approx(1.5, abs=1e-5)

    # clip_factor = 2.0 -> C = 6.0. Neither clipped. Average = 3.0.
    model_20 = _DummyModel([np.array([0.0], dtype=np.float32)])
    s20 = NormClippingStrategy(clip_factor=2.0, global_model=model_20)
    agg, log = s20.aggregate_fit(server_round=1, results=results, failures=[])
    assert log["clip_radius"] == pytest.approx(6.0, abs=1e-5)
    assert parameters_to_ndarrays(agg)[0][0] == pytest.approx(3.0, abs=1e-5)


def test_norm_clipping_isolation_no_tvflids_state():
    """Verify NormClippingStrategy has no TV-FLIDS trust or gating state."""
    strategy = NormClippingStrategy(clip_factor=1.0)

    assert not hasattr(strategy, "trust_scorer")
    assert not hasattr(strategy, "verifier")
    assert not hasattr(strategy, "adaptive")
    assert not hasattr(strategy, "_eval_cache")
    assert not hasattr(strategy, "log_weights")
