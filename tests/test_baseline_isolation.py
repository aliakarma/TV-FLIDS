"""
tests/test_baseline_isolation.py
Rigorously test Baseline Strategy Isolation and Cross-Method State Leakage.

Specification:
1. Strict isolation of all 14 baseline strategies from TV-FLIDS-specific mechanisms
   (no TV-FLIDS median norm clipping, no validation-loss gating, no multi-signal trust memory,
   no asymmetric EMA, no online meta-weight adaptation).
2. Independent method-specific server logic.
3. Clean dispatch boundary in strategy factory.
4. Zero cross-method state leakage between runs.
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
from models.mlp import IDSMLP
from fl.strategy import TVFLIDSStrategy
from trust.adaptive_trust_scorer import AdaptiveTrustScorer
from fl.baselines.fedavg_strategy import FedAvgStrategy
from fl.baselines.krum_strategy import KrumStrategy
from fl.baselines.multikrum_strategy import MultiKrumStrategy
from fl.baselines.trimmed_mean_strategy import TrimmedMeanStrategy
from fl.baselines.norm_clipping_strategy import NormClippingStrategy
from fl.baselines.rfa_strategy import RFAStrategy
from fl.baselines.bucketing_strategy import BucketingStrategy
from fl.baselines.foolsgold_strategy import FoolsGoldStrategy
from fl.baselines.flame_strategy import FLAMEStrategy
from fl.baselines.deepsight_strategy import DeepSightStrategy
from fl.baselines.fldetector_strategy import FLDetectorStrategy
from fl.baselines.zeno_strategy import ZenoStrategy
from fl.baselines.fltrust_strategy import FLTrustStrategy
from fl.baselines.baffle_strategy import BaFFLeStrategy
from experiments.run_experiment import make_strategy


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
    def __init__(self, params: list, num_examples: int = 10):
        self.parameters = ndarrays_to_parameters(params)
        self.num_examples = num_examples
        self.metrics = {}


# ── Part 1: Baseline Method-Specific Isolation Tests ─────────────────────────

def test_fedavg_isolation_no_tvflids_mechanisms():
    """Verify FedAvg does not inherit TV-FLIDS clipping, gate, trust memory, or meta-weights."""
    model = make_test_model()
    strategy = FedAvgStrategy(global_model=model)

    # 1. Verify no TV-FLIDS attributes exist
    assert not hasattr(strategy, "trust_scorer")
    assert not hasattr(strategy, "verifier")
    assert not hasattr(strategy, "adaptive")
    assert not hasattr(strategy, "_eval_cache")

    # 2. Verify no clipping: client with 1000x norm update is NOT clipped
    global_params = model.get_parameters()
    u0 = [np.ones_like(p) * 1.0 for p in global_params]
    u1 = [np.ones_like(p) * 1000.0 for p in global_params]

    results = [
        (_FakeProxy(0), _FakeFitRes([g + u for g, u in zip(global_params, u0)], num_examples=10)),
        (_FakeProxy(1), _FakeFitRes([g + u for g, u in zip(global_params, u1)], num_examples=10)),
    ]

    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    agg_arr = parameters_to_ndarrays(agg_params)

    # FedAvg weighted mean: (1.0 + 1000.0) / 2 = 500.5
    for p_agg, g in zip(agg_arr, global_params):
        expected = g + 500.5
        np.testing.assert_allclose(p_agg, expected, rtol=1e-5)


def test_krum_isolation():
    """Verify Krum executes its own distance scoring and does not inherit TV-FLIDS logic."""
    model = make_test_model()
    strategy = KrumStrategy(num_clients=4, num_byzantine=1, m=1, global_model=model)

    assert not hasattr(strategy, "trust_scorer")
    assert not hasattr(strategy, "verifier")

    global_params = model.get_parameters()
    # 3 honest clients close to each other, 1 outlier far away
    u0 = [np.zeros_like(p) for p in global_params]
    u1 = [np.ones_like(p) * 0.01 for p in global_params]
    u2 = [np.ones_like(p) * -0.01 for p in global_params]
    u3 = [np.ones_like(p) * 100.0 for p in global_params]  # Outlier

    results = [
        (_FakeProxy(0), _FakeFitRes([g + u for g, u in zip(global_params, u0)])),
        (_FakeProxy(1), _FakeFitRes([g + u for g, u in zip(global_params, u1)])),
        (_FakeProxy(2), _FakeFitRes([g + u for g, u in zip(global_params, u2)])),
        (_FakeProxy(3), _FakeFitRes([g + u for g, u in zip(global_params, u3)])),
    ]

    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    assert "krum_selected" in log
    # Client 0 is closest to clients 1 and 2, so client 0 should be selected
    assert 0 in log["krum_selected"]
    assert 3 not in log["krum_selected"]


def test_trimmed_mean_isolation():
    """Verify Trimmed Mean executes coordinate-wise trimming independently."""
    model = make_test_model()
    strategy = TrimmedMeanStrategy(beta=0.25, global_model=model)

    assert not hasattr(strategy, "trust_scorer")
    assert not hasattr(strategy, "verifier")

    global_params = model.get_parameters()
    # 4 clients: [-10.0, 1.0, 2.0, 100.0] -> beta=0.25 trims 1 from each side -> remainder is [1.0, 2.0] -> mean = 1.5
    results = [
        (_FakeProxy(0), _FakeFitRes([g - 10.0 for g in global_params])),
        (_FakeProxy(1), _FakeFitRes([g + 1.0 for g in global_params])),
        (_FakeProxy(2), _FakeFitRes([g + 2.0 for g in global_params])),
        (_FakeProxy(3), _FakeFitRes([g + 100.0 for g in global_params])),
    ]

    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    agg_arr = parameters_to_ndarrays(agg_params)
    for p_agg, g in zip(agg_arr, global_params):
        np.testing.assert_allclose(p_agg, g + 1.5, rtol=1e-5)


def test_rfa_isolation():
    """Verify RFA executes smoothed Weiszfeld geometric median aggregation."""
    model = make_test_model()
    strategy = RFAStrategy(max_iter=10, global_model=model)

    assert not hasattr(strategy, "trust_scorer")
    assert not hasattr(strategy, "verifier")

    global_params = model.get_parameters()
    results = [
        (_FakeProxy(0), _FakeFitRes([g + 1.0 for g in global_params])),
        (_FakeProxy(1), _FakeFitRes([g + 1.0 for g in global_params])),
        (_FakeProxy(2), _FakeFitRes([g + 100.0 for g in global_params])),
    ]

    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    assert "rfa_iters" in log
    agg_arr = parameters_to_ndarrays(agg_params)
    # Geometric median of [1, 1, 100] is very close to 1.0
    for p_agg, g in zip(agg_arr, global_params):
        np.testing.assert_allclose(p_agg, g + 1.0, atol=1.0)


def test_fltrust_isolation():
    """Verify FLTrust uses server-root cosine weighting and does not use TV-FLIDS trust memory."""
    model = make_test_model()
    val_loader = make_test_dataloader()
    device = torch.device("cpu")
    strategy = FLTrustStrategy(
        server_model=model,
        server_root_loader=val_loader,
        device=device,
        local_epochs=1,
        lr=0.001,
    )

    # Verify absence of TV-FLIDS trust scorer and verifier
    assert not hasattr(strategy, "trust_scorer")
    assert not hasattr(strategy, "verifier")

    global_params = model.get_parameters()
    # Compute root update direction
    root_upd = strategy._compute_root_update(global_params)
    flat_root = np.concatenate([r.flatten() for r in root_upd])
    norm_root = np.linalg.norm(flat_root)

    # Client 0: update perfectly aligned with root update
    c0_params = [g + r for g, r in zip(global_params, root_upd)]
    # Client 1: update pointing directly opposite to root update (cosine < 0)
    c1_params = [g - r for g, r in zip(global_params, root_upd)]

    results = [
        (_FakeProxy(0), _FakeFitRes(c0_params)),
        (_FakeProxy(1), _FakeFitRes(c1_params)),
    ]

    agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    assert log["fltrust_zero_weight"] == 1  # Client 1 received zero weight

    # Aggregated update should be exactly along the root update direction
    agg_arr = parameters_to_ndarrays(agg_params)
    for p_agg, p_c0 in zip(agg_arr, c0_params):
        np.testing.assert_allclose(p_agg, p_c0, atol=1e-5)


def test_clustering_and_sybil_baselines_isolation():
    """Verify FoolsGold, Bucketing, DeepSight, and FLAME strategies maintain independent isolation."""
    model = make_test_model()
    global_params = model.get_parameters()

    # FoolsGold
    fg = FoolsGoldStrategy(num_clients=4, global_model=model)
    assert not hasattr(fg, "trust_scorer")
    assert not hasattr(fg, "verifier")

    # Bucketing
    bk = BucketingStrategy(bucket_size=2, beta=0.1, global_model=model)
    assert not hasattr(bk, "trust_scorer")
    assert not hasattr(bk, "verifier")

    # DeepSight
    ds = DeepSightStrategy(global_model=model)
    assert not hasattr(ds, "trust_scorer")
    assert not hasattr(ds, "verifier")

    # FLAME
    flame = FLAMEStrategy(num_clients=4, global_model=model)
    assert not hasattr(flame, "trust_scorer")
    assert not hasattr(flame, "verifier")


# ── Part 2: Cross-Method State Leakage and Run Isolation Tests ───────────────

def test_cross_method_state_leakage_tvflids_then_fedavg():
    """Verify running TV-FLIDS leaves NO residual state that alters a subsequent FedAvg run."""
    model = make_test_model()
    val_loader = make_test_dataloader()
    global_params = model.get_parameters()

    # Phase 1: Run TV-FLIDS for 5 rounds
    config = {
        "trust": {"lambda_up": 0.9, "lambda_down": 0.7, "min_trust": 0.01, "initial_trust": 0.5, "meta_lr": 0.01},
        "verification": {"loss_threshold": 0.0, "warmup_rounds": 20, "adaptive_thresholds": False},
    }
    tvflids = TVFLIDSStrategy(
        num_clients=4, config=config, val_loader=val_loader,
        model=model, device=torch.device("cpu"), adaptive=True, seed=42
    )

    for r in range(1, 6):
        res = [
            (_FakeProxy(0), _FakeFitRes([g + 0.01 * r for g in global_params])),
            (_FakeProxy(1), _FakeFitRes([g - 0.01 * r for g in global_params])),
        ]
        tvflids.aggregate_fit(server_round=r, results=res, failures=[])

    # TV-FLIDS has adapted meta-weights and updated trust scores
    assert np.any(tvflids.trust_scorer.trust_scores != 0.5)

    # Phase 2: Create a fresh FedAvg instance
    fedavg = FedAvgStrategy(global_model=model)

    # FedAvg must have no TV-FLIDS state
    assert not hasattr(fedavg, "trust_scorer")
    assert not hasattr(fedavg, "verifier")

    # Run FedAvg
    res_fedavg = [
        (_FakeProxy(0), _FakeFitRes([g + 1.0 for g in global_params], num_examples=20)),
        (_FakeProxy(1), _FakeFitRes([g + 3.0 for g in global_params], num_examples=10)),
    ]
    agg_fedavg, log_fedavg = fedavg.aggregate_fit(server_round=1, results=res_fedavg, failures=[])
    agg_fedavg_arr = parameters_to_ndarrays(agg_fedavg)

    # Expected: weighted mean (20 * 1.0 + 10 * 3.0) / 30 = 50 / 30 = 1.666667
    for p_agg, g in zip(agg_fedavg_arr, global_params):
        expected = g + (50.0 / 30.0)
        np.testing.assert_allclose(p_agg, expected, rtol=1e-5)


def test_cross_method_state_leakage_tvflids_then_multiple_baselines():
    """Verify TV-FLIDS execution does not leak into Krum, Trimmed Mean, RFA, or FLTrust."""
    model = make_test_model()
    val_loader = make_test_dataloader()
    global_params = model.get_parameters()

    # Run TV-FLIDS
    config = {
        "trust": {"lambda_up": 0.9, "lambda_down": 0.7, "min_trust": 0.01, "initial_trust": 0.5, "meta_lr": 0.01},
        "verification": {"loss_threshold": -1e9, "warmup_rounds": 20, "adaptive_thresholds": False},
    }
    tvflids = TVFLIDSStrategy(
        num_clients=4, config=config, val_loader=val_loader,
        model=model, device=torch.device("cpu"), adaptive=True, seed=42
    )
    res = [(_FakeProxy(0), _FakeFitRes([g + 0.01 for g in global_params]))]
    tvflids.aggregate_fit(server_round=1, results=res, failures=[])

    # Instantiate fresh baselines
    krum = KrumStrategy(num_clients=4, num_byzantine=1, global_model=model)
    tm = TrimmedMeanStrategy(beta=0.1, global_model=model)
    rfa = RFAStrategy(global_model=model)
    fltrust = FLTrustStrategy(server_model=model, server_root_loader=val_loader, device=torch.device("cpu"))

    for baseline in [krum, tm, rfa, fltrust]:
        assert not hasattr(baseline, "trust_scorer")
        assert not hasattr(baseline, "verifier")
        assert not hasattr(baseline, "use_adaptive_thresholds")


def test_fresh_tvflids_instance_isolation_from_previous_runs():
    """Verify new TVFLIDSStrategy instance is completely fresh and unaffected by prior strategy instances."""
    model = make_test_model()
    val_loader = make_test_dataloader()
    config = {
        "trust": {"lambda_up": 0.9, "lambda_down": 0.7, "min_trust": 0.01, "initial_trust": 0.5, "meta_lr": 0.01},
        "verification": {"loss_threshold": 0.0, "warmup_rounds": 20, "adaptive_thresholds": False},
    }

    # Run instance 1 for 10 rounds
    strat1 = TVFLIDSStrategy(num_clients=4, config=config, val_loader=val_loader, model=model, device=torch.device("cpu"), seed=42)
    for r in range(1, 11):
        g = model.get_parameters()
        strat1.aggregate_fit(server_round=r, results=[(_FakeProxy(0), _FakeFitRes([p + 0.01 for p in g]))], failures=[])

    assert len(strat1.round_logs) == 10

    # Instantiate instance 2
    strat2 = TVFLIDSStrategy(num_clients=4, config=config, val_loader=val_loader, model=model, device=torch.device("cpu"), seed=42)

    # Verify instance 2 is cleanly initialized
    assert len(strat2.round_logs) == 0
    np.testing.assert_allclose(strat2.trust_scorer.trust_scores, np.full(4, 0.5))
    if isinstance(strat2.trust_scorer, AdaptiveTrustScorer):
        np.testing.assert_allclose(strat2.trust_scorer.log_weights.detach().numpy(), np.zeros(3))
        snap = strat2.trust_scorer.get_current_weights()
        assert pytest.approx(snap["alpha"], abs=1e-6) == 1/3
        assert pytest.approx(snap["beta"], abs=1e-6) == 1/3
        assert pytest.approx(snap["gamma"], abs=1e-6) == 1/3
    assert len(strat2._eval_cache) == 0
    assert len(strat2._eval_bal_cache) == 0


# ── Part 3: Strategy Factory Dispatch Verification ────────────────────────────

def test_strategy_factory_dispatch_routes_correctly():
    """Verify make_strategy dispatch logic instantiates the exact requested strategy class."""
    model = make_test_model()
    val_loader = make_test_dataloader()
    root_loader = make_test_dataloader()
    device = torch.device("cpu")
    fl_cfg = {"fraction_fit": 0.5, "fraction_evaluate": 0.3, "local_lr": 0.001}
    config = {
        "trust": {"lambda_up": 0.9, "lambda_down": 0.7, "min_trust": 0.01, "initial_trust": 0.5, "meta_lr": 0.01},
        "verification": {"loss_threshold": 0.0, "warmup_rounds": 20, "adaptive_thresholds": False},
        "adversarial": {"attack_ratio": 0.3},
    }

    # FedAvg
    s_fedavg = make_strategy("fedavg", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_fedavg, FedAvgStrategy)

    # Krum
    s_krum = make_strategy("krum", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_krum, KrumStrategy)

    # Multi-Krum
    s_mkrum = make_strategy("multikrum", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_mkrum, MultiKrumStrategy)

    s_multi_krum = make_strategy("multi_krum", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_multi_krum, MultiKrumStrategy)

    # Trimmed Mean
    s_tm = make_strategy("trimmed_mean", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_tm, TrimmedMeanStrategy)

    # Norm Clipping
    s_nc = make_strategy("norm_clipping", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_nc, NormClippingStrategy)

    # FLTrust
    s_fltrust = make_strategy("fltrust", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_fltrust, FLTrustStrategy)

    # FoolsGold
    s_fg = make_strategy("foolsgold", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_fg, FoolsGoldStrategy)

    # FLAME
    s_flame = make_strategy("flame", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_flame, FLAMEStrategy)

    # RFA
    s_rfa = make_strategy("rfa", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_rfa, RFAStrategy)

    # Bucketing
    s_bk = make_strategy("bucketing", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_bk, BucketingStrategy)

    # DeepSight
    s_ds = make_strategy("deepsight", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_ds, DeepSightStrategy)

    # FLDetector
    s_fldet = make_strategy("fldetector", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_fldet, FLDetectorStrategy)

    # Zeno
    s_zeno = make_strategy("zeno", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_zeno, ZenoStrategy)

    # BaFFLe
    s_baffle = make_strategy("baffle", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_baffle, BaFFLeStrategy)

    # TV-FLIDS Adaptive
    s_tvflids = make_strategy("tvflids", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_tvflids, TVFLIDSStrategy)
    assert s_tvflids.adaptive is True

    # TV-FLIDS Fixed
    s_tvflids_fixed = make_strategy("tvflids_fixed", config, model, val_loader, root_loader, device, 10, fl_cfg)
    assert isinstance(s_tvflids_fixed, TVFLIDSStrategy)
    assert s_tvflids_fixed.adaptive is False

    # Invalid strategy name raises ValueError
    with pytest.raises(ValueError, match="Unknown strategy: invalid_name"):
        make_strategy("invalid_name", config, model, val_loader, root_loader, device, 10, fl_cfg)


def test_all_14_baselines_isolation_attributes():
    """Verify none of the 14 baseline strategies has TV-FLIDS trust, gating, or meta-weights."""
    model = make_test_model()
    val_loader = make_test_dataloader()
    root_loader = make_test_dataloader()
    device = torch.device("cpu")
    fl_cfg = {"fraction_fit": 0.5, "fraction_evaluate": 0.3, "local_lr": 0.001}
    config = {
        "trust": {"lambda_up": 0.9, "lambda_down": 0.7, "min_trust": 0.01, "initial_trust": 0.5, "meta_lr": 0.01},
        "verification": {"loss_threshold": 0.0, "warmup_rounds": 20, "adaptive_thresholds": False},
        "adversarial": {"attack_ratio": 0.3},
    }

    all_baseline_names = [
        "fedavg", "krum", "multikrum", "trimmed_mean", "norm_clipping",
        "rfa", "bucketing", "foolsgold", "flame", "deepsight",
        "fldetector", "zeno", "fltrust", "baffle"
    ]

    for name in all_baseline_names:
        strat = make_strategy(name, config, model, val_loader, root_loader, device, 10, fl_cfg)
        assert not hasattr(strat, "trust_scorer"), f"{name} should not have trust_scorer"
        assert not hasattr(strat, "verifier"), f"{name} should not have verifier"
        assert not hasattr(strat, "_eval_cache"), f"{name} should not have _eval_cache"
        assert not hasattr(strat, "log_weights"), f"{name} should not have log_weights"

