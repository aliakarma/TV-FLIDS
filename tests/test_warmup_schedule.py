"""
tests/test_warmup_schedule.py
Deterministic tests for the verification-gate warmup annealing schedules
(paper Section IV-A: tau_z via eq:tau_anneal, tau_L via the prose schedule).

These are pure-arithmetic tests: no Flower, no Ray, no dataset, no randomness.
They pin the exact threshold values the paper specifies at every phase of the
schedule, so a regression in either formula fails loudly.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trust.verification import VerificationModule  # noqa: E402


TAU_Z_NOMINAL = 2.5      # paper Table IV
TAU_Z_WARM_OFFSET = 0.5  # so tau_z(0) = 3.0
T_WARM = 20              # paper Table IV


def tau_z(t, warmup=T_WARM):
    return VerificationModule.adaptive_zscore_threshold(
        TAU_Z_NOMINAL, t, warmup_rounds=warmup, warmup_offset=TAU_Z_WARM_OFFSET)


def tau_L(t, warmup=T_WARM):
    return VerificationModule.adaptive_loss_threshold(
        t, initial=-0.1, final=0.0, warmup_rounds=warmup)


# ── eq:tau_anneal: tau_z(t) = 2.5 + 0.5 * max(0, 1 - t/T_warm) ───────────────

class TestZScoreSchedule:

    def test_endpoints_match_paper(self):
        """3.0 at t=0, exactly the nominal 2.5 at t=T_warm."""
        assert tau_z(0) == pytest.approx(3.0)
        assert tau_z(T_WARM) == pytest.approx(TAU_Z_NOMINAL)

    @pytest.mark.parametrize("t,expected", [
        (0,  3.000), (1,  2.975), (5,  2.875), (10, 2.750),
        (15, 2.625), (19, 2.525), (20, 2.500),
    ])
    def test_exact_warmup_values(self, t, expected):
        """Linear decay: every warmup round has a pinned closed-form value."""
        assert tau_z(t) == pytest.approx(expected, abs=1e-9)

    def test_linear_during_warmup(self):
        """Equal steps => equal decrements (the schedule is affine in t)."""
        steps = [tau_z(t) - tau_z(t + 1) for t in range(T_WARM)]
        assert all(s == pytest.approx(steps[0], abs=1e-12) for s in steps)
        assert steps[0] == pytest.approx(TAU_Z_WARM_OFFSET / T_WARM)

    @pytest.mark.parametrize("t", [21, 25, 30, 50, 100, 1000])
    def test_constant_after_warmup(self, t):
        """"remaining fixed thereafter" — no post-warmup rebound."""
        assert tau_z(t) == pytest.approx(TAU_Z_NOMINAL, abs=1e-12)

    def test_no_discontinuity_at_t_warm(self):
        """Regression: the old implementation jumped 2.5 -> ~4.88 at T_warm+1.

        That made the gate strictly MORE permissive just after warmup than at
        the end of warmup, inverting the schedule's intent.
        """
        assert tau_z(T_WARM + 1) == pytest.approx(tau_z(T_WARM), abs=1e-12)

    def test_monotone_non_increasing(self):
        vals = [tau_z(t) for t in range(0, 60)]
        assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))

    def test_never_below_nominal(self):
        """Annealing is an offset ABOVE nominal; it must never undershoot."""
        assert all(tau_z(t) >= TAU_Z_NOMINAL - 1e-12 for t in range(0, 200))

    def test_additive_not_multiplicative(self):
        """tau_z(0) is nominal + 0.5, not nominal * 3 (the old bug gave 7.5)."""
        assert tau_z(0) == pytest.approx(TAU_Z_NOMINAL + TAU_Z_WARM_OFFSET)
        assert tau_z(0) != pytest.approx(TAU_Z_NOMINAL * 3.0)

    def test_warmup_rounds_is_authoritative(self):
        """T_warm is the only knob controlling the schedule length."""
        for warm in (5, 10, 20, 30):
            assert tau_z(0, warm) == pytest.approx(3.0)
            assert tau_z(warm, warm) == pytest.approx(TAU_Z_NOMINAL)
            assert tau_z(warm + 1, warm) == pytest.approx(TAU_Z_NOMINAL)
            # halfway through warmup is halfway through the offset
            assert tau_z(warm / 2, warm) == pytest.approx(
                TAU_Z_NOMINAL + TAU_Z_WARM_OFFSET / 2)

    def test_disabled_warmup_returns_base(self):
        assert tau_z(0, warmup=0) == pytest.approx(TAU_Z_NOMINAL)
        assert tau_z(7, warmup=0) == pytest.approx(TAU_Z_NOMINAL)


# ── tau_L(t): -0.1 -> 0.0 linearly over the first T_warm rounds ───────────────

class TestLossThresholdSchedule:

    def test_endpoints_match_paper(self):
        assert tau_L(0) == pytest.approx(-0.1)
        assert tau_L(T_WARM) == pytest.approx(0.0)

    @pytest.mark.parametrize("t,expected", [
        (0, -0.100), (5, -0.075), (10, -0.050), (15, -0.025), (20, 0.0),
    ])
    def test_exact_warmup_values(self, t, expected):
        assert tau_L(t) == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("t", [21, 30, 31, 100])
    def test_constant_after_warmup(self, t):
        assert tau_L(t) == pytest.approx(0.0, abs=1e-12)

    def test_reaches_zero_at_t_warm_not_at_30(self):
        """Regression: the call site hardcoded transition=30, contradicting
        Table IV's T_warm=20. At t=20 the threshold must already be 0.0."""
        assert tau_L(20) == pytest.approx(0.0, abs=1e-12)
        # under the old transition=30 this was -0.1 * (1 - 20/30) = -0.0333
        assert tau_L(20) != pytest.approx(-1.0 / 30.0, abs=1e-4)

    def test_monotone_non_decreasing(self):
        vals = [tau_L(t) for t in range(0, 60)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))

    def test_warmup_rounds_is_authoritative(self):
        for warm in (5, 10, 20, 30):
            assert tau_L(0, warm) == pytest.approx(-0.1)
            assert tau_L(warm, warm) == pytest.approx(0.0, abs=1e-12)
            assert tau_L(warm / 2, warm) == pytest.approx(-0.05, abs=1e-12)

    def test_disabled_warmup_returns_final(self):
        assert tau_L(0, warmup=0) == pytest.approx(0.0)


# ── Both schedules share one T_warm and are wired into the strategy ───────────

class TestScheduleWiring:

    @staticmethod
    def _config(warmup=20, enabled=True):
        return {
            "trust": {"memory_decay": 0.9, "min_trust": 0.01, "meta_lr": 0.01},
            "verification": {
                "loss_threshold": 0.0, "cosine_threshold": 0.0,
                "zscore_threshold": 2.5, "warmup_rounds": warmup,
                "adaptive_thresholds": enabled,
                "warmup_loss_threshold": -0.1, "warmup_zscore_offset": 0.5,
            },
        }

    def _strategy(self, **kw):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from fl.strategy import TVFLIDSStrategy
        from models.mlp import IDSMLP

        model = IDSMLP(input_dim=8, num_classes=3)
        loader = DataLoader(
            TensorDataset(torch.zeros(4, 8), torch.zeros(4, dtype=torch.long)),
            batch_size=4)
        return TVFLIDSStrategy(
            num_clients=3, config=self._config(**kw), val_loader=loader,
            model=model, device=torch.device("cpu"), adaptive=True)

    def test_enabled_by_default_from_config(self):
        """The paper describes annealing as part of the method; the previous
        production call site hardcoded it off."""
        assert self._strategy().use_adaptive_thresholds is True

    def test_config_can_disable(self):
        assert self._strategy(enabled=False).use_adaptive_thresholds is False

    def test_explicit_argument_overrides_config(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from fl.strategy import TVFLIDSStrategy
        from models.mlp import IDSMLP
        model = IDSMLP(input_dim=8, num_classes=3)
        loader = DataLoader(
            TensorDataset(torch.zeros(4, 8), torch.zeros(4, dtype=torch.long)),
            batch_size=4)
        s = TVFLIDSStrategy(
            num_clients=3, config=self._config(enabled=True), val_loader=loader,
            model=model, device=torch.device("cpu"),
            use_adaptive_thresholds=False)
        assert s.use_adaptive_thresholds is False

    def test_strategy_reads_warmup_rounds_from_config(self):
        assert self._strategy(warmup=7).warmup_rounds == 7

    def test_nominal_thresholds_preserved_separately(self):
        """The live verifier thresholds get overwritten each round, so the
        nominal values must be stored independently or the schedule would
        compound on itself round over round."""
        s = self._strategy()
        assert s._nominal_zscore_threshold == pytest.approx(2.5)
        assert s._nominal_loss_threshold == pytest.approx(0.0)

    def test_repeated_annealing_does_not_compound(self):
        """Calling the schedule for rounds 1..30 in sequence must give the
        same values as calling it for each round independently."""
        s = self._strategy()
        seen = []
        for r in range(1, 31):
            s.verifier.zscore_threshold = \
                VerificationModule.adaptive_zscore_threshold(
                    s._nominal_zscore_threshold, r,
                    warmup_rounds=s.warmup_rounds,
                    warmup_offset=s._warmup_zscore_offset)
            seen.append(s.verifier.zscore_threshold)
        assert seen == [pytest.approx(tau_z(r)) for r in range(1, 31)]
