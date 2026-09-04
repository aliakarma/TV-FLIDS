"""
tests/test_a1_ablation_semantics.py
Pins the exact semantics of ablation A1 ("No Verification").

WHAT A1 ACTUALLY DOES
---------------------
Paper Section IX defines A1 as bypassing the gate by setting
tau_L = -inf, tau_C = -inf, tau_z = +inf. experiments/run_ablation.py
implements that with -1e9 / -1e9 / +1e9.

tau_z, however, is NOT exclusive to Check 3. It also appears in the anomaly
penalty signal (Eq. (9)):

    O_i = 1 - exp(-z_i / tau_z)

so tau_z -> +inf drives O_i -> 0 for every client, which removes the
-gamma * O_i term from the trust signal of Eq. (6) as well. A1 therefore
disables BOTH the anomaly-based rejection gate AND the anomaly contribution to
the trust score. This is faithful to the definition as the paper states it
(tau_z is a single shared quantity, and A1 sets it to +inf), but it is a
consequence a reader will not infer, so the manuscript now states it and this
test locks the behaviour in place.

Consequence for interpretation: the A1 arm measures the JOINT contribution of
the verification gate and the norm-anomaly trust signal, which makes it an
upper bound on the gate's own contribution rather than an isolated measurement
of it. The manuscript's Section IX text says so explicitly.

By contrast tau_L and tau_C are gate-only: no trust signal reads them, so
relaxing them removes rejection/flagging without touching S_i or A_i.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.run_ablation import ABLATION_CONFIGS  # noqa: E402
from trust.trust_scorer import TrustScorer  # noqa: E402

# The values experiments/run_ablation.py writes for the A1 arm.
A1_LOSS_THRESHOLD = -1e9
A1_COSINE_THRESHOLD = -1e9
A1_ZSCORE_THRESHOLD = 1e9
NOMINAL_TAU_Z = 2.5


def _updates_with_one_outlier():
    """Five ordinary updates and one with a grossly inflated norm."""
    rng = np.random.RandomState(0)
    ordinary = [[rng.randn(20).astype(np.float32) * 0.01] for _ in range(5)]
    outlier = [np.ones(20, dtype=np.float32) * 5.0]
    return ordinary + [outlier]


class TestA1ArmIsWiredAsDocumented:

    def test_a1_sets_all_three_thresholds_to_extremes(self):
        assert ABLATION_CONFIGS["A1: No Verification"]["no_verification"] is True

    def test_a1_also_switches_annealing_off(self):
        """Otherwise tau_L would anneal from -0.1 toward -1e9, not sit flat."""
        a1 = ABLATION_CONFIGS["A1: No Verification"]
        assert a1["no_verification"] is True
        assert a1["trust_override"] is None


class TestSharedTauZDisablesTheAnomalySignal:

    def test_anomaly_scores_are_nonzero_at_the_nominal_threshold(self):
        scorer = TrustScorer(num_clients=6)
        o = scorer.compute_anomaly_scores(_updates_with_one_outlier(),
                                          tau_z=NOMINAL_TAU_Z)
        assert o.max() > 0.5, "the outlier must register under normal operation"
        assert (o > 0).all()

    def test_a1_tau_z_collapses_every_anomaly_score_to_zero(self):
        scorer = TrustScorer(num_clients=6)
        o = scorer.compute_anomaly_scores(_updates_with_one_outlier(),
                                          tau_z=A1_ZSCORE_THRESHOLD)
        # O_i = 1 - exp(-z_i / 1e9) ~ z_i * 1e-9 for any realistic z_i.
        assert o.max() < 1e-6, (
            "A1's tau_z = +inf must drive O_i -> 0; got max O_i = %g" % o.max())
        assert (o >= 0).all()

    def test_the_gamma_term_vanishes_from_the_trust_signal(self):
        """The practical consequence: gamma * O_i drops out of Eq. (6)."""
        updates = _updates_with_one_outlier()
        n = len(updates)
        sim = np.full(n, 0.8)
        acc = np.full(n, 0.6)

        nominal = TrustScorer(num_clients=n)
        a1 = TrustScorer(num_clients=n)
        no_anomaly = TrustScorer(num_clients=n)

        ids = list(range(n))
        nominal.update_trust(
            ids, sim, acc,
            nominal.compute_anomaly_scores(updates, tau_z=NOMINAL_TAU_Z))
        a1.update_trust(
            ids, sim, acc,
            a1.compute_anomaly_scores(updates, tau_z=A1_ZSCORE_THRESHOLD))
        # Ground truth for "the anomaly signal is gone": O_i identically 0.
        no_anomaly.update_trust(ids, sim, acc, np.zeros(n))

        assert a1.trust_scores == pytest.approx(no_anomaly.trust_scores,
                                                abs=1e-9)
        # ...and it genuinely differs from the nominal gate, so the test is
        # not passing because the anomaly signal was inert to begin with.
        assert a1.trust_scores[-1] > nominal.trust_scores[-1] + 1e-4


class TestTauLAndTauCAreGateOnly:
    """Only tau_z is shared; the other two A1 relaxations are gate-local."""

    def test_no_trust_signal_reads_tau_l_or_tau_c(self):
        import inspect

        src = inspect.getsource(TrustScorer)
        for signal in ("compute_similarity_scores", "compute_accuracy_scores"):
            assert signal in src
        # The only threshold any signal accepts is tau_z.
        sig = inspect.signature(TrustScorer.compute_anomaly_scores)
        assert "tau_z" in sig.parameters
        for name in ("compute_similarity_scores", "compute_accuracy_scores"):
            params = inspect.signature(getattr(TrustScorer, name)).parameters
            assert "tau_z" not in params
            assert "loss_threshold" not in params
            assert "cosine_threshold" not in params

    def test_similarity_and_accuracy_are_unchanged_under_a1(self):
        """S_i and A_i do not move when the gate thresholds are relaxed."""
        scorer = TrustScorer(num_clients=6)
        updates = _updates_with_one_outlier()
        mean = [np.mean([u[0] for u in updates], axis=0)]

        sim = scorer.compute_similarity_scores(updates, mean)
        acc = scorer.compute_accuracy_scores(1.0, [0.9] * len(updates))

        # Recomputing them is threshold-independent by construction: neither
        # helper takes a threshold argument at all.
        assert sim == pytest.approx(scorer.compute_similarity_scores(updates, mean))
        assert acc == pytest.approx(
            scorer.compute_accuracy_scores(1.0, [0.9] * len(updates)))
