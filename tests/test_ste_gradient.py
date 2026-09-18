"""
tests/test_ste_gradient.py
Gradient-level tests for the straight-through estimator used in the
aggregation-weight equation eq:hatw.

Paper claim (Section IV-C, after the aggregation-weight equation):
    "Wherever the clip is active ... we employ the straight-through estimator:
     the forward pass uses the clipped value, while the backward pass treats
     the clip as the identity and passes the incoming gradient through
     unchanged [bengio2013ste]"

These tests check the gradient MATHEMATICALLY against closed-form derivatives,
not merely that `.backward()` runs. Pure torch on CPU; no Flower, no Ray.
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.ste import clip_ste  # noqa: E402


# ── The estimator itself ─────────────────────────────────────────────────────

class TestClipSTE:

    def test_forward_is_exactly_clamp(self):
        x = torch.tensor([-3.0, -0.5, 0.0, 0.25, 1.0, 1.5, 9.0])
        assert torch.equal(clip_ste(x, 0.0, 1.0), torch.clamp(x, 0.0, 1.0))

    def test_backward_is_identity_everywhere(self):
        """d clip_ste(x) / dx == 1 for every x, saturated region included."""
        x = torch.tensor([-3.0, -0.5, 0.0, 0.25, 1.0, 1.5, 9.0],
                         requires_grad=True)
        clip_ste(x).sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))

    def test_differs_from_clamp_exactly_in_the_saturated_region(self):
        """This is the substantive difference the paper's STE buys.

        Note torch.clamp already passes gradient AT the boundary (x == 0 or
        x == 1); what it kills is the SATURATED region (x < 0 or x > 1). So a
        hard clamp does not implement the cited estimator.
        """
        vals = [-2.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        x1 = torch.tensor(vals, requires_grad=True)
        x2 = torch.tensor(vals, requires_grad=True)
        clip_ste(x1).sum().backward()
        torch.clamp(x2, 0.0, 1.0).sum().backward()

        saturated = torch.tensor([v < 0.0 or v > 1.0 for v in vals])
        assert torch.equal(x2.grad[saturated], torch.zeros(int(saturated.sum())))
        assert torch.equal(x1.grad[saturated], torch.ones(int(saturated.sum())))
        # Identical where the clip does not bind.
        assert torch.equal(x1.grad[~saturated], x2.grad[~saturated])

    def test_chain_rule_scales_upstream_gradient(self):
        """STE is the identity, not a constant-1 override: d(c*clip(x))/dx = c."""
        x = torch.tensor([-5.0, 0.5, 7.0], requires_grad=True)
        (clip_ste(x) * torch.tensor([2.0, 3.0, 4.0])).sum().backward()
        assert torch.allclose(x.grad, torch.tensor([2.0, 3.0, 4.0]))

    def test_gradient_flows_through_an_upstream_parameter(self):
        """y = clip_ste(w * x) with w*x saturated => dy/dw == x, not 0."""
        w = torch.tensor(3.0, requires_grad=True)
        x = torch.tensor(5.0)                 # w*x = 15, far above the clip
        clip_ste(w * x).backward()
        assert w.grad.item() == pytest.approx(5.0)

        w2 = torch.tensor(3.0, requires_grad=True)
        torch.clamp(w2 * x, 0.0, 1.0).backward()
        assert w2.grad.item() == pytest.approx(0.0)   # hard clip: no signal

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError):
            clip_ste(torch.tensor([0.5]), lo=1.0, hi=0.0)

    def test_custom_bounds(self):
        x = torch.tensor([-1.0, 0.5, 3.0], requires_grad=True)
        y = clip_ste(x, -0.5, 2.0)
        assert torch.allclose(y, torch.tensor([-0.5, 0.5, 2.0]))
        y.sum().backward()
        assert torch.equal(x.grad, torch.ones(3))


# ── eq:hatw in situ: the meta-gradient proxy loss ────────────────────────────

def _meta_loss(alpha, beta, gamma, sim, acc, anom, losses, clip):
    """Reproduces fl/strategy.py::aggregate_fit::_val_fn with a swappable clip."""
    raw = clip(alpha * sim + beta * acc - gamma * anom)
    w = raw / (raw.sum() + 1e-8)
    return (w * losses).sum()


class TestMetaGradientPath:

    @staticmethod
    def _signals():
        # Client 2's raw signal is deliberately negative (gamma*O dominates),
        # i.e. the lower clip binds for it — exactly the "binds transiently in
        # early rounds" regime the paper describes.
        sim = torch.tensor([0.90, 0.80, 0.05])
        acc = torch.tensor([0.70, 0.60, 0.00])
        anom = torch.tensor([0.10, 0.20, 0.99])
        losses = torch.tensor([0.30, 0.35, 2.50])
        return sim, acc, anom, losses

    def _grad(self, clip):
        sim, acc, anom, losses = self._signals()
        v = torch.zeros(3, requires_grad=True)       # softmax(0,0,0) = 1/3 each
        w = torch.softmax(v, dim=0)
        _meta_loss(w[0], w[1], w[2], sim, acc, anom, losses, clip).backward()
        return v.grad.clone()

    def test_saturated_client_binds_in_this_fixture(self):
        """Guard: the fixture must actually exercise the saturated branch."""
        sim, acc, anom, _ = self._signals()
        raw = (sim + acc - anom) / 3.0
        assert (raw < 0).any(), "fixture no longer exercises the clip"

    def test_forward_loss_identical_under_both_clips(self):
        """The STE must not change the model's forward behaviour at all."""
        sim, acc, anom, losses = self._signals()
        w = torch.softmax(torch.zeros(3), dim=0)
        a = _meta_loss(w[0], w[1], w[2], sim, acc, anom, losses, clip_ste)
        b = _meta_loss(w[0], w[1], w[2], sim, acc, anom, losses,
                       lambda t: torch.clamp(t, 0.0, 1.0))
        assert torch.equal(a, b)

    def test_ste_and_clamp_give_different_meta_gradients(self):
        g_ste = self._grad(clip_ste)
        g_clamp = self._grad(lambda t: torch.clamp(t, 0.0, 1.0))
        assert not torch.allclose(g_ste, g_clamp), \
            "STE must change dL_meta/dv when the clip binds"

    def test_ste_gradient_matches_analytic_unclipped_derivative(self):
        """With STE the clip is the identity in backward, so dL/dv must equal
        the gradient of the SAME expression computed with no clip at all."""
        sim, acc, anom, losses = self._signals()

        v1 = torch.zeros(3, requires_grad=True)
        w1 = torch.softmax(v1, dim=0)
        _meta_loss(w1[0], w1[1], w1[2], sim, acc, anom, losses, clip_ste).backward()

        v2 = torch.zeros(3, requires_grad=True)
        w2 = torch.softmax(v2, dim=0)
        raw = w2[0] * sim + w2[1] * acc - w2[2] * anom   # identity, no clip
        # forward must be re-clipped to keep the same value; detach the
        # correction so only the identity path carries gradient
        raw = raw + (torch.clamp(raw, 0.0, 1.0) - raw).detach()
        wts = raw / (raw.sum() + 1e-8)
        (wts * losses).sum().backward()

        assert torch.allclose(v1.grad, v2.grad, atol=1e-6)

    def test_softmax_weights_stay_on_the_simplex(self):
        """alpha+beta+gamma == 1 must survive an STE-driven Adam step."""
        sim, acc, anom, losses = self._signals()
        v = torch.zeros(3, requires_grad=True)
        opt = torch.optim.Adam([v], lr=0.01)
        for _ in range(10):
            opt.zero_grad()
            w = torch.softmax(v, dim=0)
            _meta_loss(w[0], w[1], w[2], sim, acc, anom, losses, clip_ste).backward()
            opt.step()
            w = torch.softmax(v, dim=0)
            assert w.sum().item() == pytest.approx(1.0, abs=1e-6)
            assert (w > 0).all()

    def test_upper_clip_never_binds_under_paper_ranges(self):
        """Section IV-B range argument: with a+b+g=1, S,A in [0,1], O in [0,1),
        the raw signal lies in (-gamma, 1-gamma], so the UPPER bound at 1 is
        unreachable and only the lower boundary is ever active."""
        g = torch.rand(500) * 0.98 + 0.01          # gamma in (0.01, 0.99)
        rest = 1.0 - g
        a = rest * torch.rand(500)
        b = rest - a
        S, A, O = torch.rand(500), torch.rand(500), torch.rand(500)
        raw = a * S + b * A - g * O
        assert (raw <= 1.0 - g + 1e-6).all()
        assert (raw > -g - 1e-6).all()
        assert (raw < 1.0).all()


class TestStrategyUsesSTE:

    def test_strategy_imports_the_estimator(self):
        """Regression guard: the meta-gradient path must not silently revert
        to torch.clamp, which would contradict Section IV-C."""
        import inspect
        from trust.adaptive_trust_scorer import AdaptiveTrustScorer

        src = inspect.getsource(AdaptiveTrustScorer.compute_meta_loss)
        assert "clip_ste(" in src
        assert "torch.clamp(" not in src

