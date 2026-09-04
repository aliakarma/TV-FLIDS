"""
tests/test_forward_pass_count.py
Regression test for the validation forward-pass count claimed in paper
Sections XII-A and XII-C.

THE CLAIM UNDER TEST
--------------------
Per server round, TVFLIDSStrategy.aggregate_fit performs exactly

    forward_passes = |P| + 1

distinct sweeps of D_val, where P is the round's submitted cohort:

  * 1 shared baseline pass for l_val(w^(t));
  * 1 Check-1 pass l_val(w^(t) + Delta_i) for EVERY i in P -- Check 1 must
    evaluate a client BEFORE it can know whether to accept it, so rejected
    clients consume their pass too (trust/verification.py::verify_all
    evaluates, then continues on rejection);
  * 0 additional passes in Stage 2: the accuracy signal A_i (Eq. (8)) reuses
    the two quantities Check 1 already cached, and the parameter-hash cache in
    fl/strategy.py::_eval_model returns them.

WHY |A| + 1 IS A LOWER BOUND, NOT AN UPPER BOUND
------------------------------------------------
|A| <= |P| always, with strict inequality whenever Check 1 rejects anything.
So |A| + 1 <= |P| + 1: it under-counts the work actually done. An earlier
revision of Section XII-C stated |A| + 1 as an exact count and then as an
upper bound; both readings are wrong, and test_accepted_count_is_a_lower_bound
pins the direction.

Runs the strategy in-process (no Flower simulation engine, no Ray).
"""

import os
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flwr.common import ndarrays_to_parameters  # noqa: E402

from fl.strategy import TVFLIDSStrategy  # noqa: E402
from models.mlp import IDSMLP  # noqa: E402

INPUT_DIM = 12
NUM_CLASSES = 3
COHORT_SIZE = 6          # |P|


class _CountingLoader:
    """DataLoader proxy that counts complete sweeps of the validation set.

    Both evaluation sites -- fl/strategy.py::_eval_model and
    trust/verification.py::_eval_params -- consume the loader with
    ``for X, y in val_loader``, so one __iter__ == one forward pass over D_val.
    """

    def __init__(self, loader):
        self._loader = loader
        self.sweeps = 0

    def __iter__(self):
        self.sweeps += 1
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)


class _FakeProxy:
    def __init__(self, cid):
        self.cid = str(cid)


class _FakeFitRes:
    def __init__(self, params):
        self.parameters = ndarrays_to_parameters(params)
        self.num_examples = 10
        self.metrics = {}


def _config(loss_threshold):
    return {
        "trust": {"alpha": 1 / 3, "beta": 1 / 3, "gamma": 1 / 3,
                  "memory_decay": 0.9, "min_trust": 0.01, "meta_lr": 0.01},
        "verification": {"loss_threshold": loss_threshold,
                         "cosine_threshold": -1.0,
                         "zscore_threshold": 10.0,
                         "warmup_rounds": 20,
                         # Off, so loss_threshold is used verbatim from round 1
                         # instead of being annealed away from it.
                         "adaptive_thresholds": False},
        "enable_overhead_tracking": False,
    }


def _make_strategy(loss_threshold=-1e9):
    torch.manual_seed(0)
    model = IDSMLP(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    g = torch.Generator().manual_seed(0)
    X = torch.rand(32, INPUT_DIM, generator=g)
    y = torch.randint(0, NUM_CLASSES, (32,), generator=g)
    loader = _CountingLoader(DataLoader(TensorDataset(X, y), batch_size=16))
    strategy = TVFLIDSStrategy(
        num_clients=COHORT_SIZE, config=_config(loss_threshold),
        val_loader=loader, model=model, device=torch.device("cpu"),
        adaptive=True, seed=42)
    return strategy, loader


def _cohort(strategy, scales):
    """One submitted update per entry of ``scales`` (so |P| == len(scales))."""
    rng = np.random.RandomState(0)
    base = strategy.model.get_parameters()
    out = []
    for cid, scale in enumerate(scales):
        params = [p + rng.randn(*p.shape).astype(np.float32) * scale
                  for p in base]
        out.append((_FakeProxy(cid), _FakeFitRes(params)))
    return out


class TestExactCount:

    def test_all_accepted_costs_cohort_plus_one(self):
        strategy, loader = _make_strategy(loss_threshold=-1e9)
        results = _cohort(strategy, [0.01] * COHORT_SIZE)
        log = strategy.aggregate_fit(1, results, [])[1]

        assert log["num_rejected"] == 0, "precondition: nothing rejected here"
        assert loader.sweeps == COHORT_SIZE + 1, (
            "expected |P| + 1 = %d forward passes, got %d"
            % (COHORT_SIZE + 1, loader.sweeps))

    def test_rejected_clients_still_consume_their_pass(self):
        # tau_L = -0.5: a scale-10 perturbation degrades val loss far past
        # that, a scale-0.001 perturbation cannot move cross-entropy by 0.5.
        strategy, loader = _make_strategy(loss_threshold=-0.5)
        scales = [0.001, 0.001, 0.001, 10.0, 10.0, 10.0]
        log = strategy.aggregate_fit(1, _cohort(strategy, scales), [])[1]

        n_accepted = log["num_verified"] + log["num_flagged"]
        # Self-validating: the split must actually have happened, otherwise
        # this test would pass vacuously against the |A| + 1 formula too.
        assert log["num_rejected"] > 0
        assert n_accepted > 0
        assert n_accepted + log["num_rejected"] == COHORT_SIZE

        assert loader.sweeps == COHORT_SIZE + 1, (
            "Check 1 evaluates every submitted client before acceptance is "
            "known, so the count is |P| + 1 = %d, not |A| + 1 = %d; got %d"
            % (COHORT_SIZE + 1, n_accepted + 1, loader.sweeps))

    def test_accepted_count_is_a_lower_bound(self):
        strategy, loader = _make_strategy(loss_threshold=-0.5)
        scales = [0.001, 0.001, 0.001, 10.0, 10.0, 10.0]
        log = strategy.aggregate_fit(1, _cohort(strategy, scales), [])[1]

        n_accepted = log["num_verified"] + log["num_flagged"]
        assert n_accepted + 1 < loader.sweeps          # strict: |A| < |P|
        assert loader.sweeps == COHORT_SIZE + 1

    def test_stage2_adds_no_pass_for_accepted_clients(self):
        """A_i (Eq. (8)) must be free: it reuses Check 1's cached losses."""
        strategy, loader = _make_strategy(loss_threshold=-1e9)
        results = _cohort(strategy, [0.01] * COHORT_SIZE)
        strategy.aggregate_fit(1, results, [])

        # The per-round cache holds one entry per distinct evaluated vector:
        # the global model plus each of the |P| submitted models.
        assert len(strategy._eval_cache) == COHORT_SIZE + 1
        # If Stage 2 had recomputed l_val(w_i) independently, the sweep count
        # would be the naive 2|P| + 1 rather than |P| + 1.
        assert loader.sweeps == COHORT_SIZE + 1
        assert loader.sweeps < 2 * COHORT_SIZE + 1

    def test_count_holds_across_rounds(self):
        strategy, loader = _make_strategy(loss_threshold=-1e9)
        for r in range(1, 4):
            strategy.aggregate_fit(r, _cohort(strategy, [0.01] * COHORT_SIZE), [])
        assert loader.sweeps == 3 * (COHORT_SIZE + 1)


class TestCacheOnlyReduces:

    def test_bit_identical_submissions_collapse_to_one_pass(self):
        """A parameter-hash collision can only LOWER the count, never raise it.

        Section XII-C argues this collision has negligible probability for
        distinct clients trained on disjoint shards; the point here is only
        that its effect is monotone downward, which is what makes |P| + 1 the
        exact count in the collision-free case and an upper bound in general.
        """
        strategy, loader = _make_strategy(loss_threshold=-1e9)
        rng = np.random.RandomState(0)
        base = strategy.model.get_parameters()
        shared = [p + rng.randn(*p.shape).astype(np.float32) * 0.01
                  for p in base]
        results = [(_FakeProxy(cid), _FakeFitRes(shared))
                   for cid in range(COHORT_SIZE)]

        strategy.aggregate_fit(1, results, [])
        # 1 baseline + 1 shared client vector.
        assert loader.sweeps == 2
        assert loader.sweeps < COHORT_SIZE + 1


class TestDerivation:
    """Pure arithmetic form of the Section XII-C claim, no model needed."""

    @pytest.mark.parametrize("n_cohort,n_rejected",
                             [(10, 0), (10, 3), (10, 9), (1, 0)])
    def test_formula(self, n_cohort, n_rejected):
        n_accepted = n_cohort - n_rejected
        exact = n_cohort + 1
        naive = 2 * n_cohort + 1

        assert exact <= naive
        assert n_accepted + 1 <= exact          # |A| + 1 is a LOWER bound
        if n_rejected > 0:
            assert n_accepted + 1 < exact
