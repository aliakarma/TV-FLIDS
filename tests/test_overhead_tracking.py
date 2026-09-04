"""
tests/test_overhead_tracking.py
Verifies that the per-stage timing instrumentation backing paper Table XII is
actually entered during TVFLIDSStrategy.aggregate_fit.

This test asserts INSTRUMENTATION PRESENCE ONLY. It deliberately makes no claim
about the magnitudes in Table XII: those are hardware-specific measurements that
require a genuine full-scale run on the paper's stated environment. What is
checked here is that every phase the table reports has a live timer on the real
code path, that the timings are recorded under the expected keys, and that
aggregation still produces a correct result with timing enabled.

Runs the strategy in-process (no Flower simulation engine, no Ray), so it is
unaffected by the known Windows Ray/PyTorch DLL problem.
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

from evaluation.overhead import OverheadTracker  # noqa: E402
from fl.strategy import TVFLIDSStrategy  # noqa: E402
from models.mlp import IDSMLP  # noqa: E402


# Phases that must be separately measured. These mirror the rows of Table XII:
# the table breaks TV-FLIDS's additions into verification gate, trust scoring
# and meta-gradient update, on top of the aggregation and per-round total.
REQUIRED_PHASES = [
    "client_processing",
    "verification",
    "trust_scoring",
    "meta_gradient",
    "aggregation",
    "total",
]

INPUT_DIM = 12
NUM_CLASSES = 3
NUM_CLIENTS = 4


def _config(track=True):
    return {
        "trust": {"alpha": 0.4, "beta": 0.4, "gamma": 0.2,
                  "memory_decay": 0.9, "min_trust": 0.01, "meta_lr": 0.01},
        "verification": {"loss_threshold": -10.0, "cosine_threshold": -1.0,
                         "zscore_threshold": 10.0, "warmup_rounds": 20,
                         "adaptive_thresholds": True},
        "enable_overhead_tracking": track,
    }


class _FakeProxy:
    def __init__(self, cid):
        self.cid = str(cid)


class _FakeFitRes:
    def __init__(self, params):
        self.parameters = ndarrays_to_parameters(params)
        self.num_examples = 10
        self.metrics = {}


def _make_strategy(track=True, adaptive=True):
    torch.manual_seed(0)
    model = IDSMLP(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    g = torch.Generator().manual_seed(0)
    X = torch.rand(32, INPUT_DIM, generator=g)
    y = torch.randint(0, NUM_CLASSES, (32,), generator=g)
    loader = DataLoader(TensorDataset(X, y), batch_size=16)
    return TVFLIDSStrategy(
        num_clients=NUM_CLIENTS, config=_config(track), val_loader=loader,
        model=model, device=torch.device("cpu"), adaptive=adaptive, seed=42)


def _fake_results(strategy, scale=0.01):
    rng = np.random.RandomState(0)
    base = strategy.model.get_parameters()
    out = []
    for cid in range(NUM_CLIENTS):
        params = [p + rng.randn(*p.shape).astype(np.float32) * scale for p in base]
        out.append((_FakeProxy(cid), _FakeFitRes(params)))
    return out


@pytest.fixture
def strategy():
    return _make_strategy()


class TestPhasesAreEntered:

    def test_every_required_phase_records_a_timing(self, strategy):
        strategy.aggregate_fit(1, _fake_results(strategy), [])
        for phase in REQUIRED_PHASES:
            times = strategy.overhead_tracker.timings.get(phase)
            assert times, f"phase '{phase}' was never entered in aggregate_fit"
            assert len(times) == 1
            assert times[0] >= 0.0

    def test_timings_accumulate_one_entry_per_round(self, strategy):
        for r in range(1, 4):
            strategy.aggregate_fit(r, _fake_results(strategy), [])
        for phase in REQUIRED_PHASES:
            assert len(strategy.overhead_tracker.timings[phase]) == 3

    def test_total_covers_the_individual_stages(self, strategy):
        strategy.aggregate_fit(1, _fake_results(strategy), [])
        t = strategy.overhead_tracker.timings
        stages = sum(t[p][0] for p in
                     ["client_processing", "verification", "trust_scoring",
                      "meta_gradient", "aggregation"])
        # The total spans the whole round, so it cannot be smaller than the
        # sum of the disjoint sub-stages it contains (modulo timer resolution).
        assert t["total"][0] >= stages - 1e-6

    def test_meta_gradient_absent_when_adaptation_is_off(self):
        s = _make_strategy(adaptive=False)
        s.aggregate_fit(1, _fake_results(s), [])
        assert not s.overhead_tracker.timings.get("meta_gradient")
        # ...but the always-on stages are still measured.
        for phase in ["verification", "trust_scoring", "aggregation", "total"]:
            assert s.overhead_tracker.timings[phase]


class TestSummaryKeys:

    def test_expected_summary_keys_present(self, strategy):
        strategy.aggregate_fit(1, _fake_results(strategy), [])
        summary = strategy.get_overhead_summary()
        for phase in REQUIRED_PHASES:
            assert f"{phase}_mean_ms" in summary
            assert f"{phase}_std_ms" in summary
            assert summary[f"{phase}_mean_ms"] >= 0.0

    def test_round_log_carries_per_stage_timings(self, strategy):
        _, log = strategy.aggregate_fit(1, _fake_results(strategy), [])
        for phase in REQUIRED_PHASES:
            assert f"time_{phase}_ms" in log

    def test_round_log_carries_live_gate_thresholds(self, strategy):
        """Round 1 is inside warmup, so the annealed values must be visible.

        This fixture uses deliberately permissive nominal thresholds
        (tau_z = 10.0, tau_L = -10.0) so that nothing is rejected; the
        schedules still anneal from tau_z + 0.5 and from -0.1 toward those
        nominal values over T_warm = 20 rounds.
        """
        _, log = strategy.aggregate_fit(1, _fake_results(strategy), [])
        a = 1 / 20
        assert log["tau_z"] == pytest.approx(10.0 + 0.5 * (1 - a))
        assert log["tau_L"] == pytest.approx(-0.1 * (1 - a) + (-10.0) * a)

    def test_summary_empty_when_tracking_disabled(self):
        s = _make_strategy(track=False)
        s.aggregate_fit(1, _fake_results(s), [])
        assert s.get_overhead_summary() == {}
        assert not any(s.overhead_tracker.timings.values())


class TestAggregationUnaffected:

    def test_aggregation_result_identical_with_and_without_timing(self):
        on, off = _make_strategy(track=True), _make_strategy(track=False)
        p_on, _ = on.aggregate_fit(1, _fake_results(on), [])
        p_off, _ = off.aggregate_fit(1, _fake_results(off), [])
        from flwr.common import parameters_to_ndarrays
        for a, b in zip(parameters_to_ndarrays(p_on),
                        parameters_to_ndarrays(p_off)):
            assert np.allclose(a, b, atol=1e-6)

    def test_aggregate_fit_still_returns_params_and_log(self, strategy):
        params, log = strategy.aggregate_fit(1, _fake_results(strategy), [])
        assert params is not None
        assert log["round"] == 1
        assert log["num_verified"] + log["num_flagged"] + log["num_rejected"] \
            == NUM_CLIENTS

    def test_all_rejected_path_closes_the_round_timer(self):
        """The early-return branch must not leave the total timer open."""
        s = _make_strategy()
        s.verifier.loss_threshold = 1e9      # reject everything
        s.use_adaptive_thresholds = False
        s.aggregate_fit(1, _fake_results(s), [])
        assert len(s.overhead_tracker.timings["total"]) == 1

    def test_reset_clears_timings(self, strategy):
        strategy.aggregate_fit(1, _fake_results(strategy), [])
        strategy.reset_trust()
        assert not any(strategy.overhead_tracker.timings.values())


class TestTrackerUnit:

    def test_tracker_declares_all_table_xii_phases(self):
        t = OverheadTracker()
        for phase in REQUIRED_PHASES + ["fedavg_total"]:
            assert phase in t.timings

    def test_overhead_pct_computed_against_fedavg(self):
        t = OverheadTracker()
        t.timings["total"] = [0.115]
        t.timings["fedavg_total"] = [0.100]
        assert t.get_summary()["overhead_pct"] == pytest.approx(15.0, abs=1e-6)
