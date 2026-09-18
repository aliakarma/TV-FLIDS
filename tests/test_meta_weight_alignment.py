"""
tests/test_meta_weight_alignment.py — Comprehensive Stage 4 Alignment Tests.

Verifies:
1. Log-weight representation v in R^3 and initialization v^(0) = (0, 0, 0).
2. Softmax projection (alpha, beta, gamma) on probability simplex.
3. Positivity and numerical stability of weights.
4. Instantaneous signal u_i = alpha*S_i + beta*A_i - gamma*O_i.
5. Straight-Through Estimator (STE) forward and backward behavior.
6. STE meta-weights hat{w}_i and normalization.
7. Exact meta-loss L_meta = sum_{i in A} hat{w}_i * l_val(tilde{w}_i).
8. Autograd gradient vs numerical finite differences (unsaturated & STE proxy).
9. Lemma 4 closed-form gradient formula verification.
10. Adam optimizer step, learning rate eta_meta = 0.01, and moment persistence.
11. Run isolation and reset behavior.
12. Empty accepted cohort fallback safety.
13. Full FL round integration in TVFLIDSStrategy.
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

from trust.adaptive_trust_scorer import AdaptiveTrustScorer
from trust.trust_scorer import TrustScorer
from utils.ste import clip_ste
from fl.strategy import TVFLIDSStrategy
from models.mlp import IDSMLP
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


class _FakeProxy:
    def __init__(self, cid: int):
        self.cid = str(cid)


class _FakeFitRes:
    def __init__(self, params: list):
        self.parameters = ndarrays_to_parameters(params)
        self.num_examples = 10
        self.metrics = {"train_time_ms": 5.0}


# ── Test 1: Initialization ────────────────────────────────────────────────────

def test_1_initialization():
    """Verify v = (0,0,0) and (alpha, beta, gamma) = (1/3, 1/3, 1/3)."""
    ats = AdaptiveTrustScorer(num_clients=10, meta_lr=0.01)
    # Check log_weights tensor
    assert torch.equal(ats.log_weights.data, torch.zeros(3, dtype=torch.float32))
    assert ats.log_weights.requires_grad is True

    # Check softmax weights
    w = ats.get_current_weights()
    assert w["alpha"] == pytest.approx(1/3, abs=1e-6)
    assert w["beta"] == pytest.approx(1/3, abs=1e-6)
    assert w["gamma"] == pytest.approx(1/3, abs=1e-6)

    # Check parent class attributes
    assert ats.alpha == pytest.approx(1/3, abs=1e-6)
    assert ats.beta == pytest.approx(1/3, abs=1e-6)
    assert ats.gamma == pytest.approx(1/3, abs=1e-6)


# ── Test 2: Softmax Simplex ───────────────────────────────────────────────────

def test_2_softmax_simplex():
    """Verify alpha + beta + gamma = 1.0 to numerical precision."""
    ats = AdaptiveTrustScorer(num_clients=5)
    # Test at initialization
    w = ats.get_current_weights()
    assert (w["alpha"] + w["beta"] + w["gamma"]) == pytest.approx(1.0, abs=1e-6)

    # Test under arbitrary log_weights
    test_vectors = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([-5.0, 0.0, 5.0]),
        torch.tensor([10.0, -10.0, 0.0]),
        torch.tensor([-100.0, -100.0, -100.0]),
    ]
    for v in test_vectors:
        with torch.no_grad():
            ats.log_weights.copy_(v)
        ats._sync_weights()
        curr = ats.get_current_weights()
        assert (curr["alpha"] + curr["beta"] + curr["gamma"]) == pytest.approx(1.0, abs=1e-6)


# ── Test 3: Positive Weights ──────────────────────────────────────────────────

def test_3_positive_weights():
    """Verify all weights remain strictly positive (> 0) under non-underflowing parameter regimes."""
    ats = AdaptiveTrustScorer(num_clients=5)
    test_vectors = [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([5.0, -5.0, 0.0]),
        torch.tensor([-10.0, -10.0, -10.0]),
        torch.tensor([2.0, 5.0, -3.0]),
        torch.tensor([0.5, -0.8, 1.2]),
    ]
    for v in test_vectors:
        with torch.no_grad():
            ats.log_weights.copy_(v)
        ats._sync_weights()
        curr = ats.get_current_weights()
        assert curr["alpha"] > 0.0
        assert curr["beta"] > 0.0
        assert curr["gamma"] > 0.0
        assert (curr["alpha"] + curr["beta"] + curr["gamma"]) == pytest.approx(1.0, abs=1e-6)


# ── Test 4: Exact u_i Formula ─────────────────────────────────────────────────

def test_4_exact_ui_formula():
    """Verify u_i = alpha * S_i + beta * A_i - gamma * O_i (with negative gamma sign)."""
    ats = AdaptiveTrustScorer(num_clients=3)
    sim = np.array([0.9, 0.5, 0.2], dtype=np.float32)
    acc = np.array([0.8, -0.2, 0.1], dtype=np.float32)
    anom = np.array([0.1, 0.7, 0.9], dtype=np.float32)
    val_losses = np.array([0.5, 1.0, 1.5], dtype=np.float32)

    with torch.no_grad():
        ats.log_weights.copy_(torch.tensor([0.5, 1.0, -0.5]))
    ats._sync_weights()
    w = ats.get_current_weights()
    a, b, g = w["alpha"], w["beta"], w["gamma"]

    # Compute expected u_i analytically
    expected_u = a * sim + b * acc - g * anom

    _, _, raw_u, _ = ats.compute_meta_loss(sim, acc, anom, val_losses)
    np.testing.assert_allclose(raw_u.detach().numpy(), expected_u, rtol=1e-5, atol=1e-5)


# ── Test 5: STE Forward Values ────────────────────────────────────────────────

def test_5_ste_forward_values():
    """Verify forward value of clip_ste(u) is bit-for-bit identical to clamp(u, 0, 1)."""
    u_vals = torch.tensor([-2.5, -0.5, 0.0, 0.35, 0.99, 1.0, 1.01, 5.0])
    clipped_ste = clip_ste(u_vals, 0.0, 1.0)
    clamped = torch.clamp(u_vals, 0.0, 1.0)
    assert torch.equal(clipped_ste, clamped)


# ── Test 6: STE Gradient ──────────────────────────────────────────────────────

def test_6_ste_gradient():
    """Verify backward pass of clip_ste yields d/du = 1 everywhere."""
    u_vals = torch.tensor([-2.0, -0.5, 0.0, 0.4, 1.0, 1.5, 3.0], requires_grad=True)
    out = clip_ste(u_vals, 0.0, 1.0)
    out.sum().backward()

    # Gradient must be 1.0 for every element, including saturated region
    assert torch.equal(u_vals.grad, torch.ones_like(u_vals))

    # Contrast with torch.clamp which zeroes gradient outside [0, 1]
    u_clamp = torch.tensor([-2.0, -0.5, 0.0, 0.4, 1.0, 1.5, 3.0], requires_grad=True)
    torch.clamp(u_clamp, 0.0, 1.0).sum().backward()
    assert u_clamp.grad[0].item() == 0.0
    assert u_clamp.grad[1].item() == 0.0
    assert u_clamp.grad[5].item() == 0.0
    assert u_clamp.grad[6].item() == 0.0


# ── Test 7: Meta-Weight Normalization ─────────────────────────────────────────

def test_7_meta_weight_normalization():
    """Verify sum_{i in A} hat{w}_i = 1.0 for valid accepted cohorts."""
    ats = AdaptiveTrustScorer(num_clients=4)
    sim = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    acc = np.array([0.5, 0.4, 0.3, 0.2], dtype=np.float32)
    anom = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    val_losses = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32)

    _, _, _, meta_w = ats.compute_meta_loss(sim, acc, anom, val_losses)
    assert meta_w.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert (meta_w >= 0.0).all()


# ── Test 8: Meta-Loss ─────────────────────────────────────────────────────────

def test_8_exact_meta_loss():
    """Verify L_meta = sum_{i in A} hat{w}_i * l_val(tilde{w}_i) matches hand calculation."""
    ats = AdaptiveTrustScorer(num_clients=2)
    # Set weights to uniform (1/3, 1/3, 1/3)
    sim = np.array([0.9, 0.3], dtype=np.float32)
    acc = np.array([0.6, 0.0], dtype=np.float32)
    anom = np.array([0.0, 0.6], dtype=np.float32)
    val_losses = np.array([0.4, 1.6], dtype=np.float32)

    # u_0 = (0.9 + 0.6 - 0.0) / 3 = 1.5 / 3 = 0.5 -> clip = 0.5
    # u_1 = (0.3 + 0.0 - 0.6) / 3 = -0.3 / 3 = -0.1 -> clip = 0.0
    # total = 0.5 + 0.0 = 0.5 (+ 1e-8)
    # hat{w}_0 = 0.5 / 0.5 = 1.0
    # hat{w}_1 = 0.0 / 0.5 = 0.0
    # L_meta = 1.0 * 0.4 + 0.0 * 1.6 = 0.4

    loss, is_sat, _, meta_w = ats.compute_meta_loss(sim, acc, anom, val_losses)
    assert is_sat is True  # client 1 was saturated
    assert meta_w[0].item() == pytest.approx(1.0, abs=1e-5)
    assert meta_w[1].item() == pytest.approx(0.0, abs=1e-5)
    assert loss.item() == pytest.approx(0.4, abs=1e-5)


# ── Test 9: Gradient Direction ────────────────────────────────────────────────

def test_9_gradient_direction():
    """Verify gradient direction: increasing alpha reduces loss when high similarity client has lower loss."""
    ats = AdaptiveTrustScorer(num_clients=2, meta_lr=0.01)
    # Client 0: high similarity, low loss (0.1)
    # Client 1: low similarity, high loss (2.0)
    sim = np.array([0.9, 0.1], dtype=np.float32)
    acc = np.array([0.5, 0.5], dtype=np.float32)
    anom = np.array([0.2, 0.2], dtype=np.float32)
    val_losses = np.array([0.1, 2.0], dtype=np.float32)

    ats.meta_optimizer.zero_grad()
    loss, _, _, _ = ats.compute_meta_loss(sim, acc, anom, val_losses)
    loss.backward()

    # d L_meta / d v_alpha should be negative (increasing v_alpha lowers loss)
    grad_v = ats.log_weights.grad
    assert grad_v[0].item() < 0.0  # dL/d v_alpha < 0


# ── Test 10: Finite-Difference Gradient Check (Unsaturated & STE Proxy) ────────

def test_10_finite_difference_gradient_check_unsaturated():
    """Compare autograd gradient against central finite differences in the unsaturated regime."""
    ats = AdaptiveTrustScorer(num_clients=4)

    # Set arbitrary log-weights
    with torch.no_grad():
        ats.log_weights.copy_(torch.tensor([0.2, -0.5, 0.3], dtype=torch.float32))

    # Construct strictly unsaturated cohort where u_i in (0, 1) for all i
    sim = np.array([0.85, 0.70, 0.60, 0.50], dtype=np.float64)
    acc = np.array([0.80, 0.65, 0.50, 0.40], dtype=np.float64)
    anom = np.array([0.10, 0.15, 0.20, 0.25], dtype=np.float64)
    val_losses = np.array([0.25, 0.45, 0.85, 1.50], dtype=np.float64)

    # Check unsaturated
    w = torch.softmax(ats.log_weights, dim=0).detach().numpy()
    u = w[0] * sim + w[1] * acc - w[2] * anom
    assert (u > 0.0).all() and (u < 1.0).all(), "Must be strictly unsaturated"

    # Compute autograd gradient
    ats.meta_optimizer.zero_grad()
    loss, is_sat, _, _ = ats.compute_meta_loss(sim, acc, anom, val_losses)
    assert is_sat is False
    loss.backward()
    autograd_grad = ats.log_weights.grad.clone().detach().numpy().astype(np.float64)

    # Compute central finite differences with optimal float32 step size eps = 1e-3
    eps = 1e-3
    fd_grad = np.zeros(3, dtype=np.float64)
    for k in range(3):
        v_plus = ats.log_weights.clone().detach()
        v_minus = ats.log_weights.clone().detach()
        v_plus[k] += eps
        v_minus[k] -= eps

        with torch.no_grad():
            ats.log_weights.copy_(v_plus)
            loss_plus, _, _, _ = ats.compute_meta_loss(sim, acc, anom, val_losses)
            ats.log_weights.copy_(v_minus)
            loss_minus, _, _, _ = ats.compute_meta_loss(sim, acc, anom, val_losses)

        fd_grad[k] = (loss_plus.item() - loss_minus.item()) / (2 * eps)

    # Reset log-weights
    with torch.no_grad():
        ats.log_weights.copy_(torch.tensor([0.2, -0.5, 0.3], dtype=torch.float32))

    # Check agreement between autograd and finite differences
    np.testing.assert_allclose(autograd_grad, fd_grad, rtol=1e-2, atol=1e-3)


def test_10b_lemma4_closed_form_gradient():
    """Verify autograd matches Lemma 4 closed-form formula in the unsaturated regime."""
    # Setup unsaturated cohort
    sim = np.array([0.9, 0.7, 0.5, 0.3], dtype=np.float64)
    acc = np.array([0.8, 0.6, 0.4, 0.2], dtype=np.float64)
    anom = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    losses = np.array([0.3, 0.5, 0.7, 1.2], dtype=np.float64)

    ats = AdaptiveTrustScorer(num_clients=4)
    with torch.no_grad():
        ats.log_weights.copy_(torch.tensor([0.1, 0.2, -0.3], dtype=torch.float32))

    w_tensor = ats.weights
    w = w_tensor.detach().numpy().astype(np.float64)
    alpha, beta, gamma = w[0], w[1], w[2]

    # Instantaneous signals
    u = alpha * sim + beta * acc - gamma * anom
    assert (u > 0).all() and (u < 1).all(), "Must be strictly unsaturated"

    u_bar = np.mean(u)
    # Signals dictionary
    X_dict = {0: sim, 1: acc, 2: -anom}

    # Lemma 4: d L_meta / d theta = (u_bar * Cov(X, l) - X_bar * Cov(u, l)) / (u_bar^2)
    # where Cov is population covariance 1/N sum (X - X_bar)(l - l_bar)
    l_bar = np.mean(losses)
    cov_u_l = np.mean((u - u_bar) * (losses - l_bar))

    g_theta = np.zeros(3, dtype=np.float64)
    for k in range(3):
        X_k = X_dict[k]
        X_bar_k = np.mean(X_k)
        cov_X_l = np.mean((X_k - X_bar_k) * (losses - l_bar))
        g_theta[k] = (u_bar * cov_X_l - X_bar_k * cov_u_l) / (u_bar ** 2)

    # Transform to log-weights Jacobian (Corollary 1(iii)):
    # d L_meta / d v_k = theta_k * (g_k - sum_m theta_m * g_m)
    dot_w_g = np.dot(w, g_theta)
    grad_v_lemma4 = w * (g_theta - dot_w_g)

    # Compute autograd
    ats.meta_optimizer.zero_grad()
    loss, is_sat, _, _ = ats.compute_meta_loss(sim, acc, anom, losses)
    loss.backward()
    autograd_grad = ats.log_weights.grad.detach().numpy().astype(np.float64)

    np.testing.assert_allclose(autograd_grad, grad_v_lemma4, rtol=1e-4, atol=1e-5)


# ── Test 11: One Adam Step ────────────────────────────────────────────────────

def test_11_one_adam_step():
    """Verify that calling adapt_weights actually changes log_weights and softmax weights."""
    ats = AdaptiveTrustScorer(num_clients=3, meta_lr=0.01)
    sim = np.array([0.9, 0.5, 0.1], dtype=np.float32)
    acc = np.array([0.8, 0.4, -0.1], dtype=np.float32)
    anom = np.array([0.1, 0.3, 0.8], dtype=np.float32)
    val_losses = np.array([0.2, 0.6, 1.8], dtype=np.float32)

    v_before = ats.log_weights.clone().detach().numpy()
    w_before = ats.get_current_weights().copy()

    res = ats.adapt_weights(sim, acc, anom, val_losses)
    assert res["step_taken"] is True

    v_after = ats.log_weights.detach().numpy()
    w_after = ats.get_current_weights()

    # Log weights must have changed
    assert not np.allclose(v_before, v_after)
    # Softmax weights must have changed
    assert (w_before["alpha"] != w_after["alpha"]) or (w_before["beta"] != w_after["beta"])


# ── Test 12: Weight Response ──────────────────────────────────────────────────

def test_12_weight_response():
    """Verify that after an update, alpha rises when similarity favors low loss."""
    ats = AdaptiveTrustScorer(num_clients=2, meta_lr=0.05)
    sim = np.array([0.95, 0.05], dtype=np.float32)
    acc = np.array([0.50, 0.50], dtype=np.float32)
    anom = np.array([0.20, 0.20], dtype=np.float32)
    val_losses = np.array([0.1, 2.5], dtype=np.float32)

    w_initial = ats.get_current_weights()
    ats.adapt_weights(sim, acc, anom, val_losses)
    w_after = ats.get_current_weights()

    assert w_after["alpha"] > w_initial["alpha"]


# ── Test 13: Optimizer State Persistence ──────────────────────────────────────

def test_13_optimizer_state_persistence():
    """Verify Adam optimizer state persists across steps in the same run."""
    ats = AdaptiveTrustScorer(num_clients=3, meta_lr=0.01)
    sim = np.array([0.9, 0.5, 0.1], dtype=np.float32)
    acc = np.array([0.8, 0.4, 0.0], dtype=np.float32)
    anom = np.array([0.1, 0.3, 0.7], dtype=np.float32)
    val_losses = np.array([0.2, 0.5, 1.2], dtype=np.float32)

    # Before step: no state for parameter
    param = ats.log_weights
    assert param not in ats.meta_optimizer.state or len(ats.meta_optimizer.state[param]) == 0

    # Step 1
    ats.adapt_weights(sim, acc, anom, val_losses)
    state = ats.meta_optimizer.state[param]
    assert "step" in state
    step1 = state["step"]
    assert (step1.item() if isinstance(step1, torch.Tensor) else step1) == 1
    exp_avg_1 = state["exp_avg"].clone()

    # Step 2
    ats.adapt_weights(sim, acc, anom, val_losses)
    state = ats.meta_optimizer.state[param]
    step2 = state["step"]
    assert (step2.item() if isinstance(step2, torch.Tensor) else step2) == 2
    # Momentum buffer evolved
    assert not torch.equal(state["exp_avg"], exp_avg_1)


# ── Test 14: Run Isolation ────────────────────────────────────────────────────

def test_14_run_isolation():
    """Verify that reset() cleanly wipes optimizer state, step counter, and weights."""
    ats1 = AdaptiveTrustScorer(num_clients=3, meta_lr=0.01)
    sim = np.array([0.9, 0.5, 0.1], dtype=np.float32)
    acc = np.array([0.8, 0.4, 0.0], dtype=np.float32)
    anom = np.array([0.1, 0.3, 0.7], dtype=np.float32)
    val_losses = np.array([0.2, 0.5, 1.2], dtype=np.float32)

    # Perform 5 updates on ats1
    for _ in range(5):
        ats1.adapt_weights(sim, acc, anom, val_losses)

    assert len(ats1.weight_history) == 5
    param1 = ats1.log_weights
    step_val = ats1.meta_optimizer.state[param1]["step"]
    assert (step_val.item() if isinstance(step_val, torch.Tensor) else step_val) == 5

    # Reset ats1
    ats1.reset()

    # Check reset properties
    assert len(ats1.weight_history) == 0
    assert ats1.alpha == pytest.approx(1/3, abs=1e-6)
    assert ats1.beta == pytest.approx(1/3, abs=1e-6)
    assert ats1.gamma == pytest.approx(1/3, abs=1e-6)
    assert torch.equal(ats1.log_weights.data, torch.zeros(3, dtype=torch.float32))
    assert ats1.log_weights not in ats1.meta_optimizer.state or len(ats1.meta_optimizer.state[ats1.log_weights]) == 0

    # Test two independent instances
    ats_a = AdaptiveTrustScorer(num_clients=3, meta_lr=0.01)
    ats_b = AdaptiveTrustScorer(num_clients=3, meta_lr=0.01)

    ats_a.adapt_weights(sim, acc, anom, val_losses)
    assert len(ats_a.weight_history) == 1
    assert len(ats_b.weight_history) == 0
    assert ats_b.log_weights not in ats_b.meta_optimizer.state or len(ats_b.meta_optimizer.state[ats_b.log_weights]) == 0


# ── Test 15: Empty Cohort Handling ────────────────────────────────────────────

def test_15_empty_cohort_handling():
    """Verify empty accepted cohort returns cleanly without division by zero or state change."""
    ats = AdaptiveTrustScorer(num_clients=3, meta_lr=0.01)
    v_before = ats.log_weights.clone().detach()

    res = ats.adapt_weights(
        similarity_scores=np.array([], dtype=np.float32),
        accuracy_scores=np.array([], dtype=np.float32),
        anomaly_scores=np.array([], dtype=np.float32),
        val_losses=np.array([], dtype=np.float32),
    )

    assert res["step_taken"] is False
    assert res["loss"] == 0.0
    assert torch.equal(ats.log_weights.data, v_before)
    assert len(ats.weight_history) == 0


# ── Test 16: Full FL Round Integration ────────────────────────────────────────

def test_16_full_round_integration():
    """Verify TVFLIDSStrategy executes Stage 4 meta-gradient step and logs diagnostics."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = IDSMLP(input_dim=41, num_classes=5)
    dummy_x = torch.randn(20, 41)
    dummy_y = torch.tensor([0, 1, 2, 3, 4] * 4, dtype=torch.long)
    val_loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=20)

    config = {
        'trust': {'meta_lr': 0.01, 'initial_trust': 0.5, 'min_trust': 0.01},
        'verification': {'loss_threshold': -0.5, 'warmup_rounds': 0, 'adaptive_thresholds': False},
        'enable_overhead_tracking': True,
    }

    strategy = TVFLIDSStrategy(
        num_clients=4,
        config=config,
        val_loader=val_loader,
        model=model,
        device=torch.device('cpu'),
        adaptive=True,
    )

    # 4 participating clients
    global_params = model.get_parameters()
    results = []
    for cid in range(4):
        # Slightly perturbed models
        client_p = [p + np.random.randn(*p.shape).astype(np.float32) * 0.01 for p in global_params]
        results.append((_FakeProxy(cid), _FakeFitRes(client_p)))

    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])

    assert agg_params is not None
    assert log["num_accepted"] > 0
    assert "adaptive_alpha" in log
    assert "adaptive_beta" in log
    assert "adaptive_gamma" in log
    assert "meta_loss" in log
    assert log["adaptive_alpha"] + log["adaptive_beta"] + log["adaptive_gamma"] == pytest.approx(1.0, abs=1e-5)
    assert len(strategy.trust_scorer.weight_history) == 1


# ── Test 17: Extreme Log-Weights Stability ────────────────────────────────────

def test_17_extreme_log_weights_stability():
    """Verify numerical stability of softmax projection under extreme values."""
    ats = AdaptiveTrustScorer(num_clients=3)
    extreme_vectors = [
        torch.tensor([500.0, -500.0, 0.0]),
        torch.tensor([-1000.0, -1000.0, -1000.0]),
        torch.tensor([1000.0, 1000.0, 1000.0]),
    ]
    for v in extreme_vectors:
        with torch.no_grad():
            ats.log_weights.copy_(v)
        ats._sync_weights()
        w = ats.get_current_weights()
        assert not np.isnan(w["alpha"]) and not np.isinf(w["alpha"])
        assert not np.isnan(w["beta"]) and not np.isinf(w["beta"])
        assert not np.isnan(w["gamma"]) and not np.isinf(w["gamma"])
        assert (w["alpha"] + w["beta"] + w["gamma"]) == pytest.approx(1.0, abs=1e-5)


# ── Test 18: All Saturated Cohort ─────────────────────────────────────────────

def test_18_all_saturated_cohort():
    """Verify behavior when all accepted clients have negative raw signals u_i < 0."""
    ats = AdaptiveTrustScorer(num_clients=3)
    # Huge anomaly and negative accuracy to force raw u < 0
    sim = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    acc = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    anom = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    val_losses = np.array([1.0, 1.5, 2.0], dtype=np.float32)

    loss, is_sat, raw_u, meta_w = ats.compute_meta_loss(sim, acc, anom, val_losses)
    assert is_sat is True
    assert (raw_u < 0).all()
    # Loss should remain finite
    assert torch.isfinite(loss)
    # Meta-weights fallback to uniform
    assert meta_w.sum().item() == pytest.approx(1.0, abs=1e-5)


# ── Test 19: Gradient Isolation (No Accidental Leaks) ──────────────────────────

def test_19_gradient_isolation():
    """Verify gradients do NOT flow into input signals or validation losses."""
    ats = AdaptiveTrustScorer(num_clients=3)
    sim = torch.tensor([0.9, 0.5, 0.1], requires_grad=True)
    acc = torch.tensor([0.8, 0.4, -0.1], requires_grad=True)
    anom = torch.tensor([0.1, 0.3, 0.8], requires_grad=True)
    val_losses = torch.tensor([0.2, 0.6, 1.8], requires_grad=True)

    ats.adapt_weights(sim, acc, anom, val_losses)

    # Input tensors must NOT have gradients accumulated
    assert sim.grad is None
    assert acc.grad is None
    assert anom.grad is None
    assert val_losses.grad is None
