r"""
tests/test_aggregation_alignment.py
Rigorously test Stage 5 (TV-FLIDS Final Model Aggregation), strictly aligned
with IEEE TIFS manuscript Section IV (Eq. 11, Alg. 1) and Part I specifications.

Canonical Equation (Paper §IV, Eq. 11):
    w^{(t+1)} = \sum_{i \in \mathcal{A}} \frac{T_i^{(t)}}{\sum_{j \in \mathcal{A}} T_j^{(t)}} \tilde{w}_i
where \tilde{w}_i = w^{(t)} + \tilde{\Delta}_i is the Stage-1 clipped candidate model.
"""

import os
import sys
import copy
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from trust.trust_scorer import TrustScorer
from trust.adaptive_trust_scorer import AdaptiveTrustScorer
from trust.verification import VerificationModule, clip_updates
from fl.strategy import TVFLIDSStrategy
from models.mlp import IDSMLP


# ── Synthetic Test Fixtures & Helpers ──────────────────────────────────────────

def make_test_model(input_dim: int = 10, num_classes: int = 2) -> nn.Module:
    torch.manual_seed(42)
    return IDSMLP(input_dim=input_dim, num_classes=num_classes)


def make_test_dataloader(input_dim: int = 10, num_classes: int = 2, n_samples: int = 32) -> DataLoader:
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)


class _FakeProxy:
    def __init__(self, cid: int):
        self.cid = str(cid)


class _FakeFitRes:
    def __init__(self, params: list):
        self.parameters = ndarrays_to_parameters(params)
        self.num_examples = 10
        self.metrics = {}


def make_strategy_instance(model: nn.Module, val_loader: DataLoader, num_clients: int = 10,
                           adaptive: bool = True, initial_trust: float = 0.5) -> TVFLIDSStrategy:
    device = torch.device("cpu")
    config = {
        "trust": {
            "lambda_up": 0.9,
            "lambda_down": 0.7,
            "min_trust": 0.01,
            "initial_trust": initial_trust,
            "meta_lr": 0.01,
        },
        "verification": {
            "loss_threshold": 0.0,
            "warmup_rounds": 20,
            "adaptive_thresholds": False,
        },
        "enable_overhead_tracking": True,
    }
    return TVFLIDSStrategy(
        num_clients=num_clients,
        config=config,
        val_loader=val_loader,
        model=model,
        device=device,
        adaptive=adaptive,
        use_adaptive_thresholds=False,
        seed=42,
    )


# ── Test Suite: 12 Rigorous Stage 5 Aggregation Tests ─────────────────────────

def test_1_trust_weights_sum_to_one():
    """Test 1 — Trust weights sum to one (to numerical precision < 1e-12)."""
    scorer = TrustScorer(num_clients=10, initial_trust=0.5)
    rng = np.random.default_rng(42)
    scorer.trust_scores = rng.uniform(0.01, 1.0, size=10)

    for subset_size in [1, 2, 3, 5, 8, 10]:
        cohort = list(range(subset_size))
        weights = scorer.get_aggregation_weights(cohort)
        assert len(weights) == subset_size
        assert pytest.approx(float(np.sum(weights)), abs=1e-12) == 1.0
        assert np.all(weights > 0.0)


def test_2_accepted_clients_only():
    """Test 2 — Accepted clients only.
    Construct accepted/rejected clients with deliberately different models.
    Verify rejected models have zero contribution.
    """
    model = make_test_model()
    val_loader = make_test_dataloader()
    strategy = make_strategy_instance(model, val_loader, num_clients=4)

    global_params = model.get_parameters()

    # Client 0: good update (improves loss, will be accepted)
    # Construct update along negative gradient
    X_val, y_val = next(iter(val_loader))
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.zero_grad()
    loss = criterion(model(X_val), y_val)
    loss.backward()
    grad_update = [-0.1 * p.grad.detach().numpy() for p in model.parameters()]
    model.eval()

    c0_params = [g + u for g, u in zip(global_params, grad_update)]

    # Client 1: huge poison update filled with 999.0 (degrades loss, will be rejected)
    c1_params = [g + np.full_like(g, 999.0) for g in global_params]

    results = [
        (_FakeProxy(0), _FakeFitRes(c0_params)),
        (_FakeProxy(1), _FakeFitRes(c1_params)),
    ]

    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    assert agg_params is not None
    assert log["num_accepted"] == 1
    assert log["num_rejected"] == 1

    agg_ndarrays = parameters_to_ndarrays(agg_params)
    # Client 0's candidate model was the ONLY accepted model, so aggregated model
    # should be client 0's clipped candidate model exactly (and contain NO 999.0)
    for p_agg, p_c0 in zip(agg_ndarrays, c0_params):
        # Difference between agg and c0 should be zero or bounded by clipping if norm > median
        # In this 2-client setup, median norm of [norm(u0), norm(u1)] = (norm(u0) + norm(u1))/2
        # Since norm(u0) < C, u0 is unclipped.
        np.testing.assert_allclose(p_agg, p_c0, atol=1e-5)
        # Ensure no 999.0 leaked
        assert not np.any(np.isclose(p_agg, 999.0, atol=10.0))


def test_3_clipped_candidate_models_are_aggregated():
    """Test 3 — Clipped candidate models are aggregated.
    Construct raw updates where one client exceeds median radius C.
    Verify aggregation uses clipped candidate \tilde{w}_i, NOT raw client model w_i.
    """
    model = make_test_model()
    val_loader = make_test_dataloader()
    strategy = make_strategy_instance(model, val_loader, num_clients=3)
    # Set gate threshold very permissive so all clients pass the gate
    strategy.verifier.loss_threshold = -1e9

    global_params = model.get_parameters()

    # Client 0: update norm 1.0
    u0 = [np.ones_like(p) * 0.01 for p in global_params]
    flat0 = np.concatenate([u.flatten() for u in u0])
    norm0 = float(np.linalg.norm(flat0))

    # Client 1: update norm 1.0 (same as 0)
    u1 = [u.copy() for u in u0]

    # Client 2: massive update norm 10.0
    u2 = [u * 10.0 for u in u0]
    norm2 = float(np.linalg.norm(np.concatenate([u.flatten() for u in u2])))

    # Median of [norm0, norm0, norm2] is norm0 (since 2 of 3 are norm0)
    # So C = norm0, and Client 2's update will be clipped to norm0 (scaling factor = norm0 / norm2 = 0.1)
    # Clipped update for Client 2: u2_clipped = u2 * 0.1 = u0.

    c0_params = [g + u for g, u in zip(global_params, u0)]
    c1_params = [g + u for g, u in zip(global_params, u1)]
    c2_params = [g + u for g, u in zip(global_params, u2)]

    results = [
        (_FakeProxy(0), _FakeFitRes(c0_params)),
        (_FakeProxy(1), _FakeFitRes(c1_params)),
        (_FakeProxy(2), _FakeFitRes(c2_params)),
    ]

    # Equal initial trust for all 3 clients
    strategy.trust_scorer.trust_scores = np.array([0.5, 0.5, 0.5, 0.5])

    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    agg_ndarrays = parameters_to_ndarrays(agg_params)

    # All 3 clipped updates are identical to u0!
    # Therefore, the aggregated model offset should be exactly u0, NOT (u0 + u1 + u2)/3 = (12/3)*u0 = 4*u0
    for l, (p_agg, g, u_exp) in enumerate(zip(agg_ndarrays, global_params, u0)):
        expected = g + u_exp
        np.testing.assert_allclose(p_agg, expected, rtol=1e-4, atol=1e-5)


def test_4_single_accepted_client():
    """Test 4 — Single accepted client.
    Verify the next global model equals that client's clipped candidate exactly.
    """
    model = make_test_model()
    val_loader = make_test_dataloader()
    strategy = make_strategy_instance(model, val_loader, num_clients=2)
    strategy.verifier.loss_threshold = -1e9

    global_params = model.get_parameters()
    u0 = [np.random.default_rng(42).normal(scale=0.01, size=p.shape).astype(np.float32) for p in global_params]
    c0_params = [g + u for g, u in zip(global_params, u0)]

    results = [(_FakeProxy(0), _FakeFitRes(c0_params))]
    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    agg_ndarrays = parameters_to_ndarrays(agg_params)

    for p_agg, p_exp in zip(agg_ndarrays, c0_params):
        np.testing.assert_allclose(p_agg, p_exp, atol=1e-6)


def test_5_equal_trust_arithmetic_average():
    """Test 5 — Equal trust: two accepted clients with equal trust produce arithmetic average."""
    scorer = TrustScorer(num_clients=4, initial_trust=0.5)
    weights = scorer.get_aggregation_weights([0, 1])
    np.testing.assert_allclose(weights, np.array([0.5, 0.5]))


def test_6_unequal_trust_mathematical_formula():
    """Test 6 — Unequal trust: weighted averaging matches exact formula T_i / sum(T_j)."""
    scorer = TrustScorer(num_clients=4, initial_trust=0.5)
    scorer.trust_scores[0] = 0.8
    scorer.trust_scores[1] = 0.2
    scorer.trust_scores[2] = 0.5

    # Test pair [0, 1]
    w01 = scorer.get_aggregation_weights([0, 1])
    np.testing.assert_allclose(w01, np.array([0.8, 0.2]))

    # Test trio [0, 1, 2] -> sum = 1.5 -> weights = [0.8/1.5, 0.2/1.5, 0.5/1.5]
    w012 = scorer.get_aggregation_weights([0, 1, 2])
    expected = np.array([0.8 / 1.5, 0.2 / 1.5, 0.5 / 1.5])
    np.testing.assert_allclose(w012, expected)


def test_7_all_trust_at_floor():
    """Test 7 — All trust at floor: normalization remains stable and equal."""
    scorer = TrustScorer(num_clients=5, min_trust=0.01)
    scorer.trust_scores[:] = 0.01

    weights = scorer.get_aggregation_weights([0, 1, 2, 3, 4])
    assert pytest.approx(float(np.sum(weights)), abs=1e-12) == 1.0
    np.testing.assert_allclose(weights, np.full(5, 0.2))


def test_8_empty_accepted_cohort():
    """Test 8 — Empty accepted cohort: global model remains completely unchanged and trust is penalized."""
    model = make_test_model()
    val_loader = make_test_dataloader()
    strategy = make_strategy_instance(model, val_loader, num_clients=4)
    # Set impossible loss threshold so all clients are rejected
    strategy.verifier.loss_threshold = 1e9

    global_params_before = [p.copy() for p in model.get_parameters()]
    initial_trust_before = strategy.trust_scorer.trust_scores.copy()

    # Participants submit updates
    results = [
        (_FakeProxy(0), _FakeFitRes([g + 0.1 for g in global_params_before])),
        (_FakeProxy(1), _FakeFitRes([g + 0.2 for g in global_params_before])),
    ]

    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    assert log["all_rejected"] == 1
    assert log["num_accepted"] == 0
    assert log["num_rejected"] == 2

    # Global model returned is unchanged
    agg_ndarrays = parameters_to_ndarrays(agg_params)
    for p_agg, p_before in zip(agg_ndarrays, global_params_before):
        np.testing.assert_allclose(p_agg, p_before, atol=1e-7)

    # Participants 0 and 1 had trust penalized (lambda_down = 0.7)
    # T_new = max(0.01, 0.7 * 0.5 + 0.3 * 0.0) = 0.35
    assert pytest.approx(strategy.trust_scorer.trust_scores[0], abs=1e-5) == 0.35
    assert pytest.approx(strategy.trust_scorer.trust_scores[1], abs=1e-5) == 0.35
    # Non-participants 2 and 3 retain initial trust 0.5
    assert pytest.approx(strategy.trust_scorer.trust_scores[2], abs=1e-5) == 0.5
    assert pytest.approx(strategy.trust_scorer.trust_scores[3], abs=1e-5) == 0.5


def test_9_rejected_clients_do_not_affect_aggregation():
    """Test 9 — Rejected clients do not affect aggregation: changing rejected model parameters
    dramatically (10^6x) leaves the aggregated global model 100% unchanged.
    """
    model = make_test_model()
    val_loader = make_test_dataloader()
    strategy = make_strategy_instance(model, val_loader, num_clients=4)
    strategy.verifier.loss_threshold = 0.0

    global_params = model.get_parameters()

    # Clean update that improves loss
    X_val, y_val = next(iter(val_loader))
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.zero_grad()
    loss = criterion(model(X_val), y_val)
    loss.backward()
    clean_upd = [-0.05 * p.grad.detach().numpy() for p in model.parameters()]
    model.eval()
    good_params = [g + u for g, u in zip(global_params, clean_upd)]

    # Rejected update 1 (bad loss)
    bad_params_1 = [g + np.ones_like(g) * 5.0 for g in global_params]

    # Run 1: with bad_params_1
    strategy_1 = copy.deepcopy(strategy)
    res_1 = [
        (_FakeProxy(0), _FakeFitRes(good_params)),
        (_FakeProxy(1), _FakeFitRes(bad_params_1)),
    ]
    agg_1, log_1 = strategy_1.aggregate_fit(server_round=1, results=res_1, failures=[])

    # Run 2: with bad_params_2 (10^6x larger bad params)
    strategy_2 = copy.deepcopy(strategy)
    bad_params_2 = [g + np.ones_like(g) * 1e6 for g in global_params]
    res_2 = [
        (_FakeProxy(0), _FakeFitRes(good_params)),
        (_FakeProxy(1), _FakeFitRes(bad_params_2)),
    ]
    agg_2, log_2 = strategy_2.aggregate_fit(server_round=1, results=res_2, failures=[])

    assert log_1["num_accepted"] == 1
    assert log_2["num_accepted"] == 1

    agg1_arr = parameters_to_ndarrays(agg_1)
    agg2_arr = parameters_to_ndarrays(agg_2)

    for p1, p2 in zip(agg1_arr, agg2_arr):
        np.testing.assert_allclose(p1, p2, atol=1e-7)


def test_10_non_participant_state_isolation():
    """Test 10 — Non-participant state: non-participants have zero contribution and trust is untouched."""
    model = make_test_model()
    val_loader = make_test_dataloader()
    strategy = make_strategy_instance(model, val_loader, num_clients=6)
    strategy.verifier.loss_threshold = -1e9

    global_params = model.get_parameters()
    # Participants are 0, 1, 2
    results = [
        (_FakeProxy(0), _FakeFitRes([g + 0.01 for g in global_params])),
        (_FakeProxy(1), _FakeFitRes([g + 0.02 for g in global_params])),
        (_FakeProxy(2), _FakeFitRes([g + 0.03 for g in global_params])),
    ]

    strategy.aggregate_fit(server_round=1, results=results, failures=[])

    # Non-participants 3, 4, 5 retain initial trust 0.5
    for cid in [3, 4, 5]:
        assert strategy.trust_scorer.trust_scores[cid] == 0.5
        assert strategy.trust_scorer.trust_history[cid] == [0.5]


def test_11_repeated_aggregation_stability():
    """Test 11 — Repeated aggregation stability over 20 rounds."""
    model = make_test_model()
    val_loader = make_test_dataloader()
    strategy = make_strategy_instance(model, val_loader, num_clients=4)
    strategy.verifier.loss_threshold = -1e9

    for r in range(1, 21):
        global_params = model.get_parameters()
        results = [
            (_FakeProxy(0), _FakeFitRes([g + 0.001 * np.sin(r) for g in global_params])),
            (_FakeProxy(1), _FakeFitRes([g + 0.001 * np.cos(r) for g in global_params])),
        ]
        agg_params, log = strategy.aggregate_fit(server_round=r, results=results, failures=[])
        assert agg_params is not None
        agg_arr = parameters_to_ndarrays(agg_params)
        for p in agg_arr:
            assert np.all(np.isfinite(p))

        # Check trust scores remain in [tau_min, 1.0]
        assert np.all(strategy.trust_scorer.trust_scores >= 0.01)
        assert np.all(strategy.trust_scorer.trust_scores <= 1.0)


def test_12_near_zero_trust_normalization_safety():
    """Test 12 — Near-zero trust normalization safety: fallback to equal weights if total trust < 1e-8."""
    scorer = TrustScorer(num_clients=3, initial_trust=0.0)
    scorer.trust_scores[:] = 0.0  # Force zero trust

    weights = scorer.get_aggregation_weights([0, 1, 2])
    assert pytest.approx(float(np.sum(weights)), abs=1e-12) == 1.0
    np.testing.assert_allclose(weights, np.full(3, 1.0 / 3.0))
