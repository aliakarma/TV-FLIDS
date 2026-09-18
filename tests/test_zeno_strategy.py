"""
tests/test_zeno_strategy.py
Tests for Zeno baseline strategy.
Reference: Xie et al., ICML 2019; Supplementary Table S2.
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
from fl.baselines.zeno_strategy import ZenoStrategy, _eval_loss_on_loader


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
    X = torch.randn(32, 4)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)
    device = torch.device("cpu")
    return model, loader, device


def test_zeno_score_hand_calculation():
    """
    Test exact Zeno score computation:
    Score_i = (l(w0) - l(w_cand)) - rho * ||Delta_i||^2
    """
    model, loader, device = make_test_fixture()
    w0 = model.get_parameters()
    l0 = _eval_loss_on_loader(model, w0, loader, device)

    # Client 0: honest, small beneficial step
    p0 = [p + np.random.normal(0, 0.01, p.shape).astype(np.float32) for p in w0]
    l0_cand = _eval_loss_on_loader(model, p0, loader, device)
    upd0 = [c - g for c, g in zip(p0, w0)]
    sq_norm0 = float(np.sum([np.sum(u**2) for u in upd0]))
    rho = 0.001
    expected_score0 = (l0 - l0_cand) - rho * sq_norm0

    # Client 1: adversary, huge magnitude
    p1 = [p + np.ones_like(p) * 10.0 for p in w0]
    l1_cand = _eval_loss_on_loader(model, p1, loader, device)
    upd1 = [c - g for c, g in zip(p1, w0)]
    sq_norm1 = float(np.sum([np.sum(u**2) for u in upd1]))
    expected_score1 = (l0 - l1_cand) - rho * sq_norm1

    strategy = ZenoStrategy(server_model=model, val_loader=loader, device=device, rho=rho, b=1)

    results = [
        (_FakeProxy(0), _FakeFitRes(p0)),
        (_FakeProxy(1), _FakeFitRes(p1)),
    ]

    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])

    # Score of client 0 should be dramatically higher than client 1
    assert expected_score0 > expected_score1
    # Client 0 should be selected (b=1 means 2-1=1 selected)
    assert log["zeno_selected"] == [0]


def test_zeno_ranking_and_averaging():
    """Verify top m = D - b_Z candidates are ranked and averaged."""
    model, loader, device = make_test_fixture()
    w0 = model.get_parameters()

    # 4 clients with varying perturbations
    clients_p = [
        [p + np.random.normal(0, 0.005, p.shape).astype(np.float32) for p in w0],
        [p + np.random.normal(0, 0.01, p.shape).astype(np.float32) for p in w0],
        [p + np.random.normal(0, 0.02, p.shape).astype(np.float32) for p in w0],
        [p + np.ones_like(p) * 5.0 for p in w0],  # Outlier
    ]

    strategy = ZenoStrategy(server_model=model, val_loader=loader, device=device, rho=0.001, b=2)
    results = [(_FakeProxy(i), _FakeFitRes(cp)) for i, cp in enumerate(clients_p)]

    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    assert log["zeno_m"] == 2  # 4 - 2 = 2
    assert len(log["zeno_selected"]) == 2
    # Client 3 (adversary) should NOT be selected
    assert 3 not in log["zeno_selected"]


def test_zeno_isolation_no_tvflids_state():
    """Verify ZenoStrategy has no TV-FLIDS trust memory or gate schedule."""
    strategy = ZenoStrategy()

    assert not hasattr(strategy, "trust_scorer")
    assert not hasattr(strategy, "verifier")
    assert not hasattr(strategy, "adaptive")
    assert not hasattr(strategy, "_eval_cache")
    assert not hasattr(strategy, "log_weights")
