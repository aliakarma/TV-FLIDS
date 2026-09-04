"""
tests/test_proposition1_domain.py
Domain and boundary-case tests for Proposition 1 (Bounded Byzantine Influence),
paper Section V, as implemented by theory/proposition1_verification.py.

The accepted-set formulation of Section V-A scopes H_A and B_A to the Stage-1
OUTPUT rather than to the client population, which makes two boundary cases
reachable that the population-level formulation did not:

  1. H_A = emptyset. OUTSIDE the proposition's domain. w* (Eq. (16)) and
     tau_bar_{H,A} are undefined, and the bound's denominator
     N_{H,A} * tau_bar_{H,A} is 0. The proposition therefore carries the
     explicit hypothesis N_{H,A} >= 1, and the verification code must report
     this case as out-of-domain rather than as a pass or a failure.

  2. B_A = emptyset. INSIDE the domain, under the convention stated in the
     proposition: the right-hand side of Eq. (18) is 0, w^(t+1) = w*, and the
     bound holds with equality.

Nothing here runs a federated experiment; all cases are constructed directly.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from theory.proposition1_verification import verify_proposition1  # noqa: E402

TAU_MIN = 0.01
DIM = 8


def _params(values):
    """A single-tensor 'model' so the flattening helpers have something real."""
    return [np.asarray(values, dtype=np.float32)]


def _case(n_honest, n_byzantine, seed=0):
    """Build one accepted set: honest near the origin, Byzantine far from it."""
    rng = np.random.RandomState(seed)
    n_total = n_honest + n_byzantine
    honest_ids = list(range(n_honest))
    byzantine_ids = list(range(n_honest, n_total))

    trust = np.ones(n_total, dtype=np.float64)
    if n_honest:
        trust[np.array(honest_ids)] = rng.uniform(0.6, 1.0, n_honest)
    if n_byzantine:
        trust[np.array(byzantine_ids)] = TAU_MIN

    global_params = _params(np.zeros(DIM))
    honest_params = [_params(rng.randn(DIM) * 0.05) for _ in range(n_honest)]
    byz_params = [_params(rng.randn(DIM) * 2.0) for _ in range(n_byzantine)]

    return dict(trust_scores=trust, honest_ids=honest_ids,
                byzantine_ids=byzantine_ids, global_params=global_params,
                honest_params=honest_params, byzantine_params=byz_params,
                tau_min=TAU_MIN)


class TestValidCase:
    """N_{H,A} >= 1 and B_A non-empty: the proposition's normal regime."""

    @pytest.mark.parametrize("n_honest,n_byzantine",
                             [(1, 1), (14, 6), (19, 1), (10, 9)])
    def test_bound_holds_and_is_well_posed(self, n_honest, n_byzantine):
        r = verify_proposition1(**_case(n_honest, n_byzantine))

        assert r.get("outside_domain") is False
        assert r["N_H"] == n_honest and r["N_H"] >= 1
        assert r["f"] == n_byzantine
        # Denominator N_{H,A} * tau_bar_{H,A} is strictly positive, which is
        # what N_{H,A} >= 1 plus the Eq. (6) floor clip guarantees.
        assert r["tau_bar_H"] > 0
        assert r["N_H"] * r["tau_bar_H"] > 0
        assert r["theoretical_bound"] > 0
        assert r["bound_holds"] is True
        assert r["observed_deviation"] <= r["theoretical_bound"] + 1e-8

    def test_single_honest_client_is_in_domain(self):
        """N_{H,A} = 1 is the tightest admissible case, not an excluded one."""
        r = verify_proposition1(**_case(1, 3))
        assert r.get("outside_domain") is False
        assert r["bound_holds"] is True


class TestEmptyHonestSetIsOutOfDomain:
    """H_A = emptyset must be rejected, not silently scored."""

    @pytest.mark.parametrize("n_byzantine", [1, 6])
    def test_reported_as_outside_domain(self, n_byzantine):
        r = verify_proposition1(**_case(0, n_byzantine))

        assert r.get("outside_domain") is True
        assert r["domain_condition"] == "N_{H,A} >= 1"
        assert r["N_H"] == 0
        # No verdict is offered, because the proposition makes no claim here.
        assert "bound_holds" not in r
        assert "theoretical_bound" not in r
        assert "observed_deviation" not in r

    def test_not_reported_as_a_passing_verification(self):
        """Guard against the failure mode of counting this case as a pass."""
        r = verify_proposition1(**_case(0, 4))
        assert r.get("bound_holds", False) is False


class TestEmptyByzantineSetUsesEqualityConvention:
    """B_A = emptyset: RHS is 0, w^(t+1) = w*, bound holds with equality."""

    @pytest.mark.parametrize("n_honest", [1, 5, 14])
    def test_equality_convention(self, n_honest):
        r = verify_proposition1(**_case(n_honest, 0))

        assert r.get("outside_domain") is False
        assert r["byzantine_set_empty"] is True
        assert r["f"] == 0
        assert r["max_byz_deviation"] == 0.0
        assert r["theoretical_bound"] == 0.0
        # w_TV is the honest aggregate itself, so deviation is 0 up to the
        # floating-point error of summing the same weighted mean twice.
        assert r["observed_deviation"] == pytest.approx(0.0, abs=1e-5)
        assert r["bound_holds"] is True

    def test_both_sides_vanish_together(self):
        r = verify_proposition1(**_case(6, 0))
        assert r["observed_deviation"] == pytest.approx(r["theoretical_bound"],
                                                        abs=1e-5)
