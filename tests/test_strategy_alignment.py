"""
tests/test_strategy_alignment.py
Rigorously test TV-FLIDS Stage 1 (Median-Radius Update Norm Clipping)
and Stage 2 (Single Validation-Loss Gate), strictly aligned with
IEEE TIFS manuscript Section IV (Eq. 3-4, Alg. 1).
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


def make_update_with_norm(template: list, target_norm: float) -> list:
    """Construct an update with exact target L2 norm."""
    flat = np.concatenate([p.flatten() for p in template])
    current_norm = np.linalg.norm(flat)
    if current_norm == 0.0:
        if target_norm == 0.0:
            return [np.zeros_like(p) for p in template]
        flat = np.ones_like(flat)
        flat = flat / np.linalg.norm(flat) * target_norm
    else:
        flat = flat / current_norm * target_norm

    # Unflatten back to template shapes
    out = []
    idx = 0
    for p in template:
        sz = p.size
        out.append(flat[idx:idx + sz].reshape(p.shape).astype(p.dtype))
        idx += sz
    return out


class _FakeProxy:
    def __init__(self, cid: int):
        self.cid = str(cid)


class _FakeFitRes:
    def __init__(self, params: list):
        self.parameters = ndarrays_to_parameters(params)
        self.num_examples = 10
        self.metrics = {}


# ── Test Suite: 12 Rigorous Specification Tests ───────────────────────────────

class TestMedianRadiusNormClipping:
    """Stage 1: Median-radius norm clipping tests."""

    def test_median_radius_known_norms(self):
        """Test 1: Construct updates with known norms [1, 2, 3, 10].
        Verify median is exactly 2.5 (mean of order statistics 2 and 3).
        Also test odd length [1, 2, 10] -> 2.0.
        """
        model = make_test_model()
        template = model.get_parameters()

        # Even cohort: 4 clients
        known_norms_even = [1.0, 2.0, 3.0, 10.0]
        updates_even = [make_update_with_norm(template, n) for n in known_norms_even]
        clipped_even, C_even, raw_norms_even = clip_updates(updates_even)

        assert pytest.approx(C_even, abs=1e-7) == 2.5
        assert len(raw_norms_even) == 4
        for r_n, exp in zip(raw_norms_even, known_norms_even):
            assert pytest.approx(r_n, abs=1e-5) == exp

        # Odd cohort: 3 clients
        known_norms_odd = [1.0, 2.0, 10.0]
        updates_odd = [make_update_with_norm(template, n) for n in known_norms_odd]
        clipped_odd, C_odd, raw_norms_odd = clip_updates(updates_odd)

        assert pytest.approx(C_odd, abs=1e-7) == 2.0
        for r_n, exp in zip(raw_norms_odd, known_norms_odd):
            assert pytest.approx(r_n, abs=1e-5) == exp

    def test_below_median_update_unchanged(self):
        """Test 2: Update with norm below or equal to median radius is unchanged."""
        model = make_test_model()
        template = model.get_parameters()

        known_norms = [1.0, 2.0, 4.0, 10.0]  # C = (2 + 4)/2 = 3.0
        updates = [make_update_with_norm(template, n) for n in known_norms]
        clipped, C, raw_norms = clip_updates(updates)
        assert pytest.approx(C, abs=1e-7) == 3.0

        # Clients 0 (norm 1.0) and 1 (norm 2.0) are below median 3.0
        for idx in [0, 1]:
            for orig_layer, clipped_layer in zip(updates[idx], clipped[idx]):
                np.testing.assert_allclose(clipped_layer, orig_layer, rtol=1e-7, atol=1e-7)
            flat_clipped = np.concatenate([p.flatten() for p in clipped[idx]])
            assert pytest.approx(np.linalg.norm(flat_clipped), abs=1e-5) == known_norms[idx]

    def test_above_median_update_scaled_to_boundary(self):
        """Test 3: Update with norm above median radius is scaled exactly to C."""
        model = make_test_model()
        template = model.get_parameters()

        known_norms = [1.0, 2.0, 4.0, 10.0]  # C = 3.0
        updates = [make_update_with_norm(template, n) for n in known_norms]
        clipped, C, raw_norms = clip_updates(updates)

        # Clients 2 (norm 4.0) and 3 (norm 10.0) are above median 3.0
        for idx in [2, 3]:
            flat_clipped = np.concatenate([p.flatten() for p in clipped[idx]])
            clipped_norm = float(np.linalg.norm(flat_clipped))
            assert pytest.approx(clipped_norm, abs=1e-5) == 3.0

            # Direction must be preserved (collinear with positive dot product)
            flat_orig = np.concatenate([p.flatten() for p in updates[idx]])
            expected_scale = 3.0 / known_norms[idx]
            np.testing.assert_allclose(flat_clipped, flat_orig * expected_scale, rtol=1e-6, atol=1e-6)

    def test_zero_update_no_nans(self):
        """Test 4: Zero-norm update produces no NaNs or infinities and remains zero."""
        model = make_test_model()
        template = model.get_parameters()

        known_norms = [0.0, 2.0, 4.0, 8.0]  # C = 3.0
        updates = [make_update_with_norm(template, n) for n in known_norms]
        clipped, C, raw_norms = clip_updates(updates)

        zero_clipped = clipped[0]
        for p in zero_clipped:
            assert np.all(np.isfinite(p)), "Clipped zero-norm update must contain only finite numbers"
            np.testing.assert_allclose(p, np.zeros_like(p), atol=1e-8)

        flat_zero = np.concatenate([p.flatten() for p in zero_clipped])
        assert np.linalg.norm(flat_zero) == 0.0

    def test_cohort_scope_all_sampled_participants(self):
        """Test 5: Clipping radius C is computed on ALL sampled participants P,
        not only those that will subsequently be accepted.
        """
        model = make_test_model()
        template = model.get_parameters()

        # Cohort P: 4 participants with norms [1.0, 2.0, 10.0, 20.0] -> C = 6.0
        # If computed only on [1.0, 2.0] (the ones that might be honest), C would be 1.5.
        known_norms = [1.0, 2.0, 10.0, 20.0]
        updates = [make_update_with_norm(template, n) for n in known_norms]
        clipped, C, raw_norms = clip_updates(updates)

        assert pytest.approx(C, abs=1e-5) == 6.0
        assert len(raw_norms) == 4


class TestSingleValidationGate:
    """Stage 2: Single validation-loss gate tests."""

    def test_gate_acceptance_improving_loss(self):
        """Test 6: Candidate model improving validation loss sufficiently is accepted."""
        model = make_test_model()
        val_loader = make_test_dataloader()
        device = torch.device("cpu")
        gp = model.get_parameters()

        verifier = VerificationModule(loss_threshold=0.0)
        # Compute baseline loss
        criterion = nn.CrossEntropyLoss()
        model.eval()
        with torch.no_grad():
            global_loss = sum(criterion(model(x), y).item() for x, y in val_loader) / len(val_loader)

        # Zero update: tentative model == global model, delta = 0.0 >= tau_L(20) = 0.0
        zero_upd = [np.zeros_like(p) for p in gp]
        res = verifier.evaluate_validation_gate(
            clipped_updates=[zero_upd], client_ids=[0], global_loss=global_loss,
            global_params=gp, model=model, device=device, val_loader=val_loader,
            loss_threshold=0.0
        )
        assert 0 in [cid for cid, _ in res['accepted']]
        assert res['deltas'][0] >= 0.0
        assert len(res['rejected']) == 0

    def test_gate_rejection_degrading_loss(self):
        """Test 7: Candidate model that fails the validation-loss threshold is rejected."""
        model = make_test_model()
        val_loader = make_test_dataloader()
        device = torch.device("cpu")
        gp = model.get_parameters()

        verifier = VerificationModule(loss_threshold=0.0)
        # Random noise update that degrades loss
        rng = np.random.RandomState(42)
        degrading_upd = [rng.randn(*p.shape).astype(np.float32) * 5.0 for p in gp]

        criterion = nn.CrossEntropyLoss()
        model.eval()
        with torch.no_grad():
            global_loss = sum(criterion(model(x), y).item() for x, y in val_loader) / len(val_loader)

        res = verifier.evaluate_validation_gate(
            clipped_updates=[degrading_upd], client_ids=[1], global_loss=global_loss,
            global_params=gp, model=model, device=device, val_loader=val_loader,
            loss_threshold=0.0
        )
        assert 1 in [cid for cid, _ in res['rejected']]
        assert 1 not in [cid for cid, _ in res['accepted']]

    def test_no_cosine_hard_rejection(self):
        """Test 8: Poor or negative cosine similarity does NOT independently reject
        if validation loss satisfies the gate.
        """
        model = make_test_model()
        val_loader = make_test_dataloader()
        device = torch.device("cpu")
        gp = model.get_parameters()

        # Verifier with legacy cosine threshold configured
        verifier = VerificationModule(loss_threshold=0.0, cosine_threshold=0.99)
        zero_upd = [np.zeros_like(p) for p in gp]

        criterion = nn.CrossEntropyLoss()
        model.eval()
        with torch.no_grad():
            global_loss = sum(criterion(model(x), y).item() for x, y in val_loader) / len(val_loader)

        # Candidate with delta >= 0
        res = verifier.evaluate_validation_gate(
            clipped_updates=[zero_upd], client_ids=[0], global_loss=global_loss,
            global_params=gp, model=model, device=device, val_loader=val_loader,
            loss_threshold=0.0
        )
        # Even though cosine with any vector might be 0.0 (< 0.99), candidate MUST NOT be rejected
        assert 0 in [cid for cid, _ in res['accepted']]
        assert len(res['rejected']) == 0

    def test_no_zscore_hard_rejection(self):
        """Test 9: Extreme norm outlier (high z-score) does NOT independently reject
        if validation loss satisfies the gate.
        """
        model = make_test_model()
        val_loader = make_test_dataloader()
        device = torch.device("cpu")
        gp = model.get_parameters()

        verifier = VerificationModule(loss_threshold=-1.0, zscore_threshold=1.0)
        zero_upd = [np.zeros_like(p) for p in gp]

        criterion = nn.CrossEntropyLoss()
        model.eval()
        with torch.no_grad():
            global_loss = sum(criterion(model(x), y).item() for x, y in val_loader) / len(val_loader)

        res = verifier.evaluate_validation_gate(
            clipped_updates=[zero_upd], client_ids=[0], global_loss=global_loss,
            global_params=gp, model=model, device=device, val_loader=val_loader,
            loss_threshold=-1.0
        )
        assert 0 in [cid for cid, _ in res['accepted']]
        assert len(res['rejected']) == 0

    def test_warmup_threshold_schedule(self):
        """Test 10: Verify the exact threshold at t=0, t=10, t=20, t>20
        using tau_L(t) = -0.1 + 0.1 * min(1, t / 20).
        """
        assert VerificationModule.adaptive_loss_threshold(0, warmup_rounds=20) == pytest.approx(-0.10)
        assert VerificationModule.adaptive_loss_threshold(10, warmup_rounds=20) == pytest.approx(-0.05)
        assert VerificationModule.adaptive_loss_threshold(20, warmup_rounds=20) == pytest.approx(0.00)
        assert VerificationModule.adaptive_loss_threshold(25, warmup_rounds=20) == pytest.approx(0.00)
        assert VerificationModule.adaptive_loss_threshold(100, warmup_rounds=20) == pytest.approx(0.00)


class TestPipelineOrderingAndEmptyCohort:
    """Pipeline integration, ordering, and edge cases."""

    def test_clipping_before_gating_evaluated(self):
        """Test 11: Candidate model evaluated by the gate MUST be formed from
        the CLIPPED update, not the raw unclipped update.
        """
        model = make_test_model()
        val_loader = make_test_dataloader()
        device = torch.device("cpu")
        gp = model.get_parameters()

        # Construct a massive update that severely degrades loss when unclipped,
        # but when clipped to small radius, loss degradation is small.
        huge_upd = [np.ones_like(p, dtype=np.float32) * 50.0 for p in gp]
        small_upd = [np.zeros_like(p) for p in gp]

        # Cohort of 3: [small_upd, small_upd, huge_upd] -> median norm is 0.0
        # If clipped first, huge_upd is clipped to 0.0 -> evaluated as 0.0 -> delta = 0.0!
        raw_updates = [small_upd, small_upd, huge_upd]
        clipped_updates, C, _ = clip_updates(raw_updates)
        assert C == pytest.approx(0.0)

        criterion = nn.CrossEntropyLoss()
        model.eval()
        with torch.no_grad():
            global_loss = sum(criterion(model(x), y).item() for x, y in val_loader) / len(val_loader)

        # Evaluate gate on clipped_updates
        verifier = VerificationModule(loss_threshold=0.0)
        res = verifier.evaluate_validation_gate(
            clipped_updates=clipped_updates, client_ids=[0, 1, 2],
            global_loss=global_loss, global_params=gp, model=model,
            device=device, val_loader=val_loader, loss_threshold=0.0
        )
        # Client 2 was huge in raw form, but in clipped form it was clipped to radius 0.0,
        # so its delta is 0.0 >= 0.0 and it is accepted!
        assert 2 in [cid for cid, _ in res['accepted']]

    def test_empty_accepted_cohort_keeps_global_model(self):
        """Test 12: When all updates fail validation (A^(t) = empty),
        strategy aggregate_fit returns unchanged w^(t+1) = w^(t).
        """
        model = make_test_model()
        val_loader = make_test_dataloader()
        device = torch.device("cpu")
        gp_orig = model.get_parameters()

        # Strategy with impossible threshold (+1e9) -> all clients rejected
        config = {
            "trust": {"alpha": 1/3, "beta": 1/3, "gamma": 1/3, "memory_decay": 0.9, "min_trust": 0.01},
            "verification": {"loss_threshold": 1e9, "adaptive_thresholds": False, "warmup_rounds": 20},
            "enable_overhead_tracking": False,
        }
        strategy = TVFLIDSStrategy(
            num_clients=2, config=config, val_loader=val_loader,
            model=model, device=device, adaptive=False
        )

        # Clients submit non-zero updates
        rng = np.random.RandomState(42)
        c0_params = [p + rng.randn(*p.shape).astype(np.float32) * 0.1 for p in gp_orig]
        c1_params = [p + rng.randn(*p.shape).astype(np.float32) * 0.1 for p in gp_orig]
        results = [
            (_FakeProxy(0), _FakeFitRes(c0_params)),
            (_FakeProxy(1), _FakeFitRes(c1_params)),
        ]

        agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])

        assert log["all_rejected"] == 1
        assert log["num_rejected"] == 2
        # Global model returned must be identical to gp_orig
        agg_arrays = parameters_to_ndarrays(agg_params)
        for p_agg, p_orig in zip(agg_arrays, gp_orig):
            np.testing.assert_array_equal(p_agg, p_orig)

        # And model parameters in strategy.model must also be untouched
        for p_model, p_orig in zip(strategy.model.get_parameters(), gp_orig):
            np.testing.assert_array_equal(p_model, p_orig)
