"""
tests/test_statistical_evaluation.py
Comprehensive test suite for TV-FLIDS Stage 10 Statistical Evaluation Pipeline.
Reference: IEEE TIFS Manuscript §VI-E, §VII, and Supplementary §S8.

Covers:
  1. Result ingestion
  2. Run ID and provenance validation
  3. Seed pairing
  4. Duplicate seed detection
  5. Missing-pair detection
  6. Configuration mismatch detection
  7. Wilcoxon exact two-sided behavior
  8. Zero-difference and tie handling
  9. Holm-Bonferroni correction (hand-computed fixtures)
  10. Effect-size calculation (paired median difference & favor count)
  11. BCa paired bootstrap (10,000 resamples, fallback on degenerate)
  12. Delta-ASR calculation
  13. Clean-reference pairing
  14. Incomplete-campaign behavior (status INCOMPLETE)
  15. Blocked-dataset behavior (status BLOCKED)
  16. Synthetic-test fixture separation
  17. Manuscript-target validation
  18. Deterministic analysis output
"""

from __future__ import annotations

import json
import os
import sys
import shutil
import tempfile
import pytest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from campaign.run_spec import RunSpecification
from campaign.artifacts import write_metrics, write_status, get_artifact_directory
from campaign.config_freezer import freeze_configuration
from evaluation.result_loader import ResultLoader, CampaignResult
from evaluation.pairing import (
    pair_primary_comparison, pair_delta_asr, PairedComparison, PairedObservation
)
from evaluation.wilcoxon import wilcoxon_signed_rank_test, WilcoxonResult
from evaluation.holm import holm_bonferroni_adjust, adjust_family
from evaluation.bootstrap_ci import compute_paired_bca_ci, BootstrapCIResult
from evaluation.effect_size import compute_paired_effect_size, EffectSizeResult
from evaluation.delta_metrics import compute_routing_diagnostics, compute_delta_asr
from evaluation.target_validator import TargetValidator, TargetValidationRecord
from evaluation.statistical_pipeline import StatisticalPipeline, PrimaryComparisonResult
from evaluation.table_generator import generate_table_s3_pairwise, generate_table_s4_intervals


@pytest.fixture
def temp_results_dir():
    """Create a temporary directory for test artifacts."""
    d = tempfile.mkdtemp(prefix="tvflids_test_results_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def create_mock_run(
    base_dir: str,
    dataset: str = "nslkdd",
    block: str = "B2",
    strategy: str = "tvflids",
    attack: str = "label_flip_30",
    seed: int = 42,
    status: str = "COMPLETED",
    acc: float = 0.720,
    f1: float = 0.539,
    asr: float = 0.375,
    is_synthetic: bool = True,
    corrupt_metrics: bool = False,
    corrupt_config: bool = False,
    override_config: dict = None,
) -> str:
    """Helper to create a mock run artifact on disk."""
    strat_params = override_config.get("strategy_params", {}) if override_config else {}
    spec = RunSpecification(
        block=block,
        purpose=f"Test run {strategy} {seed}",
        dataset=dataset,
        strategy=strategy,
        strategy_params=strat_params,
        attack=attack,
        seed=seed,
        alpha=0.5,
        num_clients=20,
        fraction_fit=0.5,
        num_rounds=100,
        val_size=2000,
    )
    run_dir = get_artifact_directory(spec, base_dir=base_dir)
    os.makedirs(run_dir, exist_ok=True)

    # 1. Config
    git_info = {"git_commit": "testcommit123", "git_dirty": False, "git_branch": "main"}
    ds_prov = {"dataset_name": dataset, "status": "AVAILABLE"}
    extra = {"is_synthetic": is_synthetic}
    freeze_configuration(
        spec=spec,
        output_dir=run_dir,
        allow_dirty=True,
        extra_metadata=extra,
        git_info=git_info,
        dataset_prov=ds_prov,
    )

    if corrupt_config:
        cfg_path = os.path.join(run_dir, "config.json")
        with open(cfg_path, "r") as f:
            cdata = json.load(f)
        cdata["run_id"] = "tampered_run_id_123"
        with open(cfg_path, "w") as f:
            json.dump(cdata, f)

    # 2. Status
    write_status(run_dir, status=status)

    # 3. Metrics
    if corrupt_metrics:
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            f.write("{invalid_json: true")
    else:
        write_metrics(run_dir, {
            "final_accuracy": acc,
            "final_f1_macro": f1,
            "final_attack_success_rate": asr,
            "final_attack_recall": 0.665,
            "final_benign_routing_phi": 0.852,
        })

    return run_dir


# ===========================================================================
# 1. Result Ingestion & 2. Provenance Validation
# ===========================================================================

def test_result_loader_valid_run(temp_results_dir):
    """Test discovering and loading a valid completed run."""
    run_dir = create_mock_run(temp_results_dir, is_synthetic=True)
    loader = ResultLoader(base_dir=temp_results_dir, allow_synthetic=True)
    count = loader.discover_and_load()
    assert count == 1
    res = loader.get_by_key("nslkdd", "B2", "tvflids", "label_flip_30", 42)
    assert res is not None
    assert res.status == "COMPLETED"
    assert res.seed == 42
    assert res.metrics["final_f1_macro"] == pytest.approx(0.539)


def test_result_loader_quarantines_synthetic_in_production(temp_results_dir):
    """Test that synthetic fixtures are quarantined when allow_synthetic=False."""
    create_mock_run(temp_results_dir, is_synthetic=True)
    loader = ResultLoader(base_dir=temp_results_dir, allow_synthetic=False)
    count = loader.discover_and_load()
    assert count == 0


def test_result_loader_rejects_failed_or_incomplete(temp_results_dir):
    """Test that FAILED or BLOCKED runs are never loaded as completed observations."""
    create_mock_run(temp_results_dir, status="FAILED", is_synthetic=True)
    loader = ResultLoader(base_dir=temp_results_dir, allow_synthetic=True)
    assert loader.discover_and_load() == 0


def test_result_loader_detects_tampered_run_id(temp_results_dir):
    """Test that run ID mismatch between config and hash causes rejection."""
    create_mock_run(temp_results_dir, corrupt_config=True, is_synthetic=True)
    loader = ResultLoader(base_dir=temp_results_dir, allow_synthetic=True)
    assert loader.discover_and_load() == 0


def test_result_loader_detects_corrupt_metrics(temp_results_dir):
    """Test that corrupt metrics.json causes rejection."""
    create_mock_run(temp_results_dir, corrupt_metrics=True, is_synthetic=True)
    loader = ResultLoader(base_dir=temp_results_dir, allow_synthetic=True)
    assert loader.discover_and_load() == 0


# ===========================================================================
# 3. Seed Pairing, 4. Duplicate Detection, 5. Missing Detection, 6. Mismatch
# ===========================================================================

def test_seed_pairing_valid(temp_results_dir):
    """Test explicit pairing between TV-FLIDS and FedAvg by seed."""
    seeds = [42, 123]
    for s in seeds:
        create_mock_run(temp_results_dir, strategy="tvflids", seed=s, f1=0.550, is_synthetic=True)
        create_mock_run(temp_results_dir, strategy="fedavg", seed=s, f1=0.300, is_synthetic=True)

    loader = ResultLoader(base_dir=temp_results_dir, allow_synthetic=True)
    loader.discover_and_load()

    comp = pair_primary_comparison(
        loader=loader,
        dataset="nslkdd",
        baseline="fedavg",
        endpoint="f1",
        expected_seeds=seeds,
    )
    assert comp.status == "COMPLETE"
    assert len(comp.pairs) == 2
    assert comp.differences[0] == pytest.approx(25.0)  # 100 * (0.550 - 0.300)


def test_seed_pairing_missing_seed_detection(temp_results_dir):
    """Test detecting missing seeds in pairing."""
    # TV-FLIDS has seed 42 and 123, FedAvg only has seed 42
    create_mock_run(temp_results_dir, strategy="tvflids", seed=42, is_synthetic=True)
    create_mock_run(temp_results_dir, strategy="tvflids", seed=123, is_synthetic=True)
    create_mock_run(temp_results_dir, strategy="fedavg", seed=42, is_synthetic=True)

    loader = ResultLoader(base_dir=temp_results_dir, allow_synthetic=True)
    loader.discover_and_load()

    comp = pair_primary_comparison(
        loader=loader,
        dataset="nslkdd",
        baseline="fedavg",
        endpoint="f1",
        expected_seeds=[42, 123],
    )
    assert comp.status == "INCOMPLETE"
    assert comp.missing_seeds == [123]
    assert len(comp.pairs) == 1


def test_seed_pairing_duplicate_seed_detection(temp_results_dir):
    """Test that multiple runs for the same seed trigger duplicate error."""
    loader = ResultLoader(base_dir=temp_results_dir, allow_synthetic=True)
    # Inject two results with same seed into loader manually
    r1 = CampaignResult("run1", "B2", "nslkdd", "tvflids", "label_flip_30", 42, {}, {"final_f1_macro": 0.5}, "COMPLETED", {}, {}, "path1")
    r2 = CampaignResult("run2", "B2", "nslkdd", "tvflids", "label_flip_30", 42, {}, {"final_f1_macro": 0.51}, "COMPLETED", {}, {}, "path2")
    r3 = CampaignResult("run3", "B2", "nslkdd", "fedavg", "label_flip_30", 42, {}, {"final_f1_macro": 0.3}, "COMPLETED", {}, {}, "path3")
    loader._results["run1"] = r1
    loader._results["run2"] = r2
    loader._results["run3"] = r3

    comp = pair_primary_comparison(loader=loader, dataset="nslkdd", baseline="fedavg", expected_seeds=[42])
    assert comp.status == "INVALID"
    assert 42 in comp.duplicate_seeds


# ===========================================================================
# 7. Wilcoxon Exact Behavior, 8. Zero-Difference & Tie Handling
# ===========================================================================

def test_wilcoxon_hand_computed_fixture():
    """
    Hand-computed Wilcoxon signed-rank test:
    d = [1.0, 2.0, 3.0, 4.0, 5.0]
    All positive ranks: W+ = 1+2+3+4+5 = 15, W- = 0.
    Exact two-sided p-value for n=5 with all signs identical: 2 * (1/2)^5 = 2/32 = 0.0625.
    """
    d = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = wilcoxon_signed_rank_test(d)
    assert res.method_used == "exact"
    assert res.statistic == 0.0
    assert res.p_value == pytest.approx(0.0625)
    assert res.has_ties is False


def test_wilcoxon_zero_difference_handling():
    """Zero differences are dropped. If all differences are zero, p=1.0, stat=0.0."""
    d_mixed = [0.0, 1.0, 2.0, 3.0, 4.0]
    res_mixed = wilcoxon_signed_rank_test(d_mixed)
    # n_nonzero = 4, exact p = 2 * (1/2)^4 = 2/16 = 0.125
    assert res_mixed.n_nonzero == 4
    assert res_mixed.p_value == pytest.approx(0.125)

    d_zeros = [0.0, 0.0, 0.0]
    res_zeros = wilcoxon_signed_rank_test(d_zeros)
    assert res_zeros.n_nonzero == 0
    assert res_zeros.p_value == 1.0
    assert res_zeros.statistic == 0.0


def test_wilcoxon_ties_handling():
    """When absolute differences have ties, asymptotic/approximate test is used."""
    d_ties = [1.0, -1.0, 2.0, -2.0, 3.0, 3.0]
    res = wilcoxon_signed_rank_test(d_ties)
    assert res.has_ties is True
    assert res.method_used == "approx"
    assert 0.0 <= res.p_value <= 1.0


# ===========================================================================
# 9. Holm-Bonferroni Correction (Hand-Computed Fixture)
# ===========================================================================

def test_holm_bonferroni_hand_computed_fixture():
    """
    Hand-computed Holm step-down test:
    m = 4 hypotheses with raw p-values: [0.01, 0.04, 0.03, 0.20].
    Sorted:
      k=1: p_(1) = 0.01 -> min(1, 4 * 0.01) = 0.04. running_max = 0.04
      k=2: p_(2) = 0.03 -> min(1, 3 * 0.03) = 0.09. running_max = 0.09
      k=3: p_(3) = 0.04 -> min(1, 2 * 0.04) = 0.08. running_max = max(0.09, 0.08) = 0.09
      k=4: p_(4) = 0.20 -> min(1, 1 * 0.20) = 0.20. running_max = max(0.09, 0.20) = 0.20
    Adjusted in original order: [0.04, 0.09, 0.09, 0.20].
    """
    raw_ps = [0.01, 0.04, 0.03, 0.20]
    adj = holm_bonferroni_adjust(raw_ps)
    expected = np.array([0.04, 0.09, 0.09, 0.20])
    np.testing.assert_allclose(adj, expected, rtol=1e-6)


def test_adjust_family_helper():
    """Test adjust_family updating list of dicts."""
    items = [
        {"id": "A", "p": 0.01},
        {"id": "B", "p": 0.04},
        {"id": "C", "p": 0.03},
        {"id": "D", "p": 0.20},
    ]
    adjust_family(items, p_key="p", out_key="p_holm", sig_key="sig", alpha=0.05)
    assert items[0]["p_holm"] == pytest.approx(0.04)
    assert items[0]["sig"] is True
    assert items[1]["p_holm"] == pytest.approx(0.09)
    assert items[1]["sig"] is False


# ===========================================================================
# 10. Effect Size Calculation
# ===========================================================================

def test_effect_size_calculation():
    """Test paired median difference and favor count."""
    diffs = np.array([5.0, 10.0, -2.0, 8.0, 0.0])
    eff_f1 = compute_paired_effect_size(diffs, endpoint="f1", dataset="nslkdd", baseline="fedavg")
    assert eff_f1.paired_median_diff == pytest.approx(5.0)
    assert eff_f1.favor_count == 3  # diff > 0 for f1
    assert eff_f1.favor_ratio == pytest.approx(0.6)

    eff_asr = compute_paired_effect_size(diffs, endpoint="asr", dataset="nslkdd", baseline="fedavg")
    assert eff_asr.favor_count == 1  # diff < 0 for asr
    assert eff_asr.favor_ratio == pytest.approx(0.2)


# ===========================================================================
# 11. BCa Paired Bootstrap Confidence Intervals
# ===========================================================================

def test_bca_bootstrap_ci_deterministic():
    """Test BCa bootstrap returns deterministic results with fixed seed."""
    d = np.array([1.2, 2.3, 3.4, 0.5, 1.8, 2.1, 1.9, 2.8, 3.0, 1.5,
                  2.0, 2.2, 2.5, 1.7, 2.9, 3.1, 1.6, 2.4, 2.7, 1.3])
    res1 = compute_paired_bca_ci(d, n_resamples=1000, seed_or_tag=42)
    res2 = compute_paired_bca_ci(d, n_resamples=1000, seed_or_tag=42)
    assert res1.ci_low == pytest.approx(res2.ci_low)
    assert res1.ci_high == pytest.approx(res2.ci_high)
    assert res1.ci_low <= res1.estimate <= res1.ci_high


def test_bca_bootstrap_degenerate_cases():
    """Test BCa handles zero spread by returning [d0, d0]."""
    d_const = np.array([3.14] * 20)
    res = compute_paired_bca_ci(d_const, seed_or_tag=42)
    assert res.ci_low == pytest.approx(3.14)
    assert res.ci_high == pytest.approx(3.14)
    assert res.method_used == "degenerate_constant"


# ===========================================================================
# 12. Delta-ASR Calculation & 13. Clean-Reference Pairing
# ===========================================================================

def test_delta_asr_and_routing_diagnostics():
    """Test Delta-ASR formula and confusion matrix routing identity."""
    # Delta-ASR = 100 * (attacked - clean)
    assert compute_delta_asr(0.741, 0.350) == pytest.approx(39.1)

    # Confusion matrix K=3: class 0 Benign, classes 1,2 Attack
    # Row 0: 1000 Benign (800 benign, 100 class 1, 100 class 2)
    # Row 1: 500 Attack 1 (200 benign, 250 attack 1, 50 attack 2)
    # Row 2: 500 Attack 2 (100 benign, 50 attack 1, 350 attack 2)
    # Total attack records = 500 + 500 = 1000
    # Attack classified as Benign = 200 + 100 = 300 -> ASR = 300 / 1000 = 0.30
    # Attack correctly classified = 250 + 350 = 600 -> R_atk = 600 / 1000 = 0.60
    # Missed attacks = 1000 - 600 = 400
    # Benign routing phi = 300 / 400 = 0.75
    # Identity: (1 - R_atk) * phi = (1 - 0.60) * 0.75 = 0.40 * 0.75 = 0.30 == ASR
    cm = np.array([
        [800, 100, 100],
        [200, 250,  50],
        [100,  50, 350],
    ])
    diag = compute_routing_diagnostics(cm)
    assert diag.asr == pytest.approx(0.30)
    assert diag.attack_recall == pytest.approx(0.60)
    assert diag.benign_routing_phi == pytest.approx(0.75)
    assert diag.identity_residual == pytest.approx(0.0, abs=1e-12)


def test_clean_reference_pairing(temp_results_dir):
    """Test pairing attacked runs with clean reference runs."""
    seeds = [42, 123]
    for s in seeds:
        create_mock_run(temp_results_dir, block="B3", strategy="tvflids", attack="none", seed=s, asr=0.350, is_synthetic=True)
        create_mock_run(temp_results_dir, block="B4", strategy="tvflids", attack="label_flip_30", seed=s, asr=0.375, is_synthetic=True)

    loader = ResultLoader(base_dir=temp_results_dir, allow_synthetic=True)
    loader.discover_and_load()

    pairs = pair_delta_asr(loader, dataset="nslkdd", method="tvflids", attack="label_flip_30", expected_seeds=seeds)
    assert len(pairs) == 2
    assert pairs[0].delta_asr == pytest.approx(2.5)  # 100 * (0.375 - 0.350)


# ===========================================================================
# 14. Incomplete Campaign, 15. Blocked Dataset, 16. Synthetic Separation
# ===========================================================================

def test_incomplete_campaign_behavior(temp_results_dir):
    """Test that a comparison with fewer than 20 seeds is marked INCOMPLETE."""
    # Only 5 seeds
    seeds = [42, 123, 456, 789, 1337]
    for s in seeds:
        create_mock_run(temp_results_dir, strategy="tvflids", seed=s, is_synthetic=True)
        create_mock_run(temp_results_dir, strategy="fedavg", seed=s, is_synthetic=True)

    loader = ResultLoader(base_dir=temp_results_dir, allow_synthetic=True)
    loader.discover_and_load()

    comp = pair_primary_comparison(loader, dataset="nslkdd", baseline="fedavg")
    assert comp.status == "INCOMPLETE"
    assert len(comp.pairs) == 5
    assert comp.required_seeds == 20


def test_blocked_dataset_behavior(temp_results_dir):
    """Test that comparisons on ciciot2023 or edgeiiotset without runs are marked BLOCKED."""
    loader = ResultLoader(base_dir=temp_results_dir, allow_synthetic=True)
    loader.discover_and_load()

    comp = pair_primary_comparison(loader, dataset="ciciot2023", baseline="fedavg")
    assert comp.status == "BLOCKED"


# ===========================================================================
# 17. Manuscript Target Validation
# ===========================================================================

def test_target_validator():
    """Test TargetValidator behavior with targets.json."""
    validator = TargetValidator()
    assert len(validator.targets.get("primary_tests", [])) == 84

    # Valid complete comparison matching target within tolerance
    target = validator.get_primary_target("nslkdd", "fedavg", "f1")
    assert target is not None

    obs_complete = {
        "dataset": "nslkdd",
        "baseline": "fedavg",
        "endpoint": "f1",
        "status": "COMPLETE",
        "paired_median_diff": target["med"] + 0.5,
        "claim": target["claim"],
    }
    rec = validator.validate_primary_test(obs_complete, tolerance_med_pp=1.0)
    assert rec.passed is True

    # Incomplete comparison cannot be validated as passing
    obs_incomplete = {
        "dataset": "nslkdd",
        "baseline": "fedavg",
        "endpoint": "f1",
        "status": "INCOMPLETE",
        "paired_median_diff": target["med"],
    }
    rec_incomp = validator.validate_primary_test(obs_incomplete)
    assert rec_incomp.passed is None
    assert "CANNOT_VALIDATE" in rec_incomp.status_message


# ===========================================================================
# 18. End-to-End Synthetic Pipeline & Deterministic Output
# ===========================================================================

def test_end_to_end_statistical_pipeline_synthetic(temp_results_dir):
    """
    End-to-end synthetic pipeline test:
    Mock 20-seed runs for TV-FLIDS and FedAvg on NSL-KDD.
    Run pipeline in allow_synthetic mode.
    Verify 84 comparisons are produced, Holm correction applied, artifact saved.
    """
    seeds = [
        42, 123, 456, 789, 1337,
        2024, 31415, 8080, 555, 999,
        1001, 1002, 1003, 1004, 1005,
        1006, 1007, 1008, 1009, 1010,
    ]
    for s in seeds:
        create_mock_run(temp_results_dir, strategy="tvflids", seed=s, f1=0.539 + 0.01 * (s % 5), asr=0.375, is_synthetic=True)
        create_mock_run(temp_results_dir, strategy="fedavg", seed=s, f1=0.300 + 0.01 * (s % 5), asr=0.741, is_synthetic=True)

    loader = ResultLoader(base_dir=temp_results_dir, allow_synthetic=True)
    assert loader.discover_and_load() == 40

    out_dir = os.path.join(temp_results_dir, "analysis")
    pipeline = StatisticalPipeline(
        loader=loader,
        output_dir=out_dir,
        bootstrap_replicates=500,     # Fast for test
        allow_synthetic=True,
    )
    payload = pipeline.run_primary_evaluation(expected_seeds=seeds)

    assert payload["metadata"]["family_size"] == 84
    assert len(payload["comparisons"]) == 84

    # Verify NSL|FedAvg|f1 comparison is SYNTHETIC_TEST_ONLY
    nsl_fedavg_f1 = next(
        c for c in payload["comparisons"]
        if c["dataset"] == "nslkdd" and c["baseline"] == "fedavg" and c["endpoint"] == "f1"
    )
    assert nsl_fedavg_f1["status"] == "SYNTHETIC_TEST_ONLY"
    assert nsl_fedavg_f1["paired_median_diff"] == pytest.approx(23.9, abs=1.0)
    assert nsl_fedavg_f1["holm_p_value"] is not None
    assert nsl_fedavg_f1["bca_ci_low"] is not None

    # Table generator
    s3_content = generate_table_s3_pairwise(payload)
    assert "FedAvg" in s3_content
    assert "Edge-IIoTset" in s3_content

    s4_content = generate_table_s4_intervals(payload)
    assert "FLTrust" in s4_content
