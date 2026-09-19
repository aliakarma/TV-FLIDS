"""
tests/test_campaign_orchestration.py
Comprehensive test suite for the TV-FLIDS experimental campaign orchestration
and reproducibility infrastructure (Stage 9).

Covers:
  1. Canonical run count derivation & invariant verification (6,111 total, 693 in B1).
  2. Unique run identity & determinism (same config -> same ID, any change -> different ID).
  3. Parameter ordering invariance & cross-process stability.
  4. Complete block coverage (B1 through B11) and filtering.
  5. Configuration freezing & model architecture resolution.
  6. Checkpointing, resume, and configuration mismatch detection.
  7. Failure handling & explicit status recording (PENDING, RUNNING, COMPLETED, FAILED, BLOCKED, SKIPPED).
  8. Dataset blocker handling for absent raw files (CIC-IoT-2023, Edge-IIoTset).
  9. Seed isolation and run-order independence.
  10. Dirty tree protection & allow_dirty override.
  11. Artifact hierarchy and canonical result schema validation.
"""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from campaign.run_spec import RunSpecification, compute_run_id
from campaign.enumerator import (
    CampaignEnumerator,
    CANONICAL_20_SEEDS,
    CANONICAL_10_SEEDS,
    CANONICAL_3_SEEDS,
    ALL_15_METHODS,
    FIVE_EVAL_METHODS,
    ALL_3_DATASETS,
)
from campaign.config_freezer import freeze_configuration, resolve_model_configuration
from campaign.artifacts import (
    get_artifact_directory,
    write_status,
    read_status,
    write_metrics,
    read_metrics,
    validate_result_schema,
    VALID_STATUSES,
)
from campaign.runner import CampaignRunner, check_dataset_availability


# ── 1. Canonical Run Count & Invariant Tests ──────────────────────────────────

class TestCampaignRunCount:
    """Verify campaign size derived manually (Method 1) matches programmatic enumeration (Method 2)."""

    def test_block_1_tuning_configuration_count(self):
        """
        Verify Block 1 tuning configurations:
        FedAvg (0) + Krum (3) + Multi-Krum (6) + Trimmed Mean (4) + Norm Clipping (4) +
        RFA (4) + Bucketing (6) + FoolsGold (4) + FLAME (9) + DeepSight (3) +
        FLDetector (4) + Zeno (6) + FLTrust (4) + BaFFLe (8) + TV-FLIDS (12) = 77 configs.
        Runs = 77 configs x 3 seeds x 3 datasets = 693 runs.
        """
        b1_runs = CampaignEnumerator.enumerate_block_1()
        assert len(b1_runs) == 693, f"Expected 693 B1 runs, got {len(b1_runs)}"

        # Check unique configurations
        unique_configs = set()
        for r in b1_runs:
            cfg_key = (r.strategy, json.dumps(r.strategy_params, sort_keys=True))
            unique_configs.add(cfg_key)
        assert len(unique_configs) == 77, f"Expected 77 unique B1 configs, got {len(unique_configs)}"

    def test_block_by_block_run_counts(self):
        """Verify each block matches the paper and supplementary specification."""
        assert len(CampaignEnumerator.enumerate_block_1()) == 693   # 77 configs x 3 x 3
        assert len(CampaignEnumerator.enumerate_block_2()) == 900   # 15 x 3 x 20
        assert len(CampaignEnumerator.enumerate_block_3()) == 900   # 15 x 3 x 20
        assert len(CampaignEnumerator.enumerate_block_4()) == 1050  # 15 x 7 x 10
        assert len(CampaignEnumerator.enumerate_block_5()) == 300   # 5 x 6 x 10
        assert len(CampaignEnumerator.enumerate_block_6()) == 100   # (4 + 6) x 10
        assert len(CampaignEnumerator.enumerate_block_7()) == 480   # 12 x 2 x 20
        assert len(CampaignEnumerator.enumerate_block_8()) == 1010  # (5 x 17 + 2 x 8) x 10
        assert len(CampaignEnumerator.enumerate_block_9()) == 200   # 5 x 40
        assert len(CampaignEnumerator.enumerate_block_10()) == 28   # 2 x 10 + 8
        assert len(CampaignEnumerator.enumerate_block_11()) == 450  # 15 x 3 x 10

    def test_total_canonical_campaign_count(self):
        """
        Invariant: Total canonical campaign count is exactly 6,111 runs
        (693 tuning + 5,418 evaluation), bounded by <= 7,038 theoretical paper upper bound.
        """
        all_runs = CampaignEnumerator.enumerate_all()
        assert len(all_runs) == 6111, f"Expected 6,111 runs, got {len(all_runs)}"
        assert len(all_runs) <= 7038

    def test_no_duplicate_run_specifications(self):
        """Verify all 6,111 runs have unique run IDs."""
        all_runs = CampaignEnumerator.enumerate_all()
        seen = set()
        duplicates = []
        for r in all_runs:
            if r.run_id in seen:
                duplicates.append(r.run_id)
            seen.add(r.run_id)
        assert len(duplicates) == 0, f"Found duplicate run IDs: {duplicates}"


# ── 2. Run Identity & Determinism Tests ───────────────────────────────────────

class TestRunIdentity:

    def test_same_configuration_same_id(self):
        spec1 = RunSpecification(block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids", seed=42)
        spec2 = RunSpecification(block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids", seed=42)
        assert spec1.run_id == spec2.run_id

    @pytest.mark.parametrize("param_override", [
        {"block": "B3"},
        {"dataset": "ciciot2023"},
        {"strategy": "fedavg"},
        {"seed": 43},
        {"num_rounds": 50},
        {"alpha": 0.1},
        {"val_size": 500},
        {"num_clients": 50},
        {"fraction_fit": 0.3},
        {"protocol": "leakage_free"},
        {"strategy_params": {"lambda_up": 0.8}},
        {"attack": "noise_30"},
        {"attack_params": {"variant": "omniscient"}},
        {"knowledge_tier": "K2"},
        {"on_off_k": 20},
        {"local_epochs": 10},
        {"local_lr": 0.01},
        {"dp_noise_multiplier": 1.0},
        {"is_profiling": True},
    ])
    def test_any_scientific_parameter_change_changes_id(self, param_override):
        base_kwargs = dict(
            block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids",
            seed=42, num_rounds=100, alpha=0.5, val_size=2000, num_clients=20,
            fraction_fit=0.5, protocol="main", strategy_params={"lambda_up": 0.9},
            attack="label_flip_30", attack_params={}, knowledge_tier="K1",
            local_epochs=5, local_lr=0.001
        )
        spec_base = RunSpecification(**base_kwargs)
        modified_kwargs = copy.deepcopy(base_kwargs)
        modified_kwargs.update(param_override)
        spec_mod = RunSpecification(**modified_kwargs)

        assert spec_base.run_id != spec_mod.run_id, f"Run ID did not change with override: {param_override}"

    def test_parameter_ordering_invariance(self):
        """Dictionary key order in strategy_params or attack_params must not change the ID."""
        spec1 = RunSpecification(
            block="B1", purpose="Order test", dataset="nslkdd", strategy="tvflids",
            strategy_params={"b": 2, "a": 1, "nested": {"z": 9, "y": 8}},
            attack_params={"k": 30, "psi": 0.85},
            seed=42,
        )
        spec2 = RunSpecification(
            block="B1", purpose="Order test", dataset="nslkdd", strategy="tvflids",
            strategy_params={"nested": {"y": 8, "z": 9}, "a": 1, "b": 2},
            attack_params={"psi": 0.85, "k": 30},
            seed=42,
        )
        assert spec1.run_id == spec2.run_id

    def test_cross_process_stability(self):
        """Run ID must be identical across separate Python process invocations."""
        code = (
            "from campaign.run_spec import RunSpecification\n"
            "spec = RunSpecification(block='B2', purpose='P', dataset='nslkdd', strategy='tvflids', seed=42)\n"
            "print(spec.run_id)\n"
        )
        cmd = [sys.executable, "-c", code]
        res1 = subprocess.check_output(cmd, text=True).strip()
        res2 = subprocess.check_output(cmd, text=True).strip()
        spec_local = RunSpecification(block="B2", purpose="P", dataset="nslkdd", strategy="tvflids", seed=42)
        assert res1 == res2 == spec_local.run_id


# ── 3. Enumerator & Filtering Tests ──────────────────────────────────────────

class TestEnumeratorFiltering:

    def test_filtering_by_block(self):
        all_runs = CampaignEnumerator.enumerate_all()
        b2_runs = CampaignEnumerator.filter_runs(all_runs, block="B2")
        assert len(b2_runs) == 900
        assert all(r.block == "B2" for r in b2_runs)

    def test_filtering_by_dataset(self):
        all_runs = CampaignEnumerator.enumerate_all()
        ciciot_runs = CampaignEnumerator.filter_runs(all_runs, dataset="ciciot2023")
        # B1 (231) + B2 (300) + B3 (300) = 831 runs
        assert len(ciciot_runs) == 831
        assert all(r.dataset == "ciciot2023" for r in ciciot_runs)

    def test_filtering_by_strategy(self):
        all_runs = CampaignEnumerator.enumerate_all()
        fedavg_runs = CampaignEnumerator.filter_runs(all_runs, strategy="fedavg")
        # FedAvg: B1 (0) + B2 (60) + B3 (60) + B4 (70) + B11 (30) = 220 runs
        assert len(fedavg_runs) == 220

    def test_filtering_by_seed(self):
        all_runs = CampaignEnumerator.enumerate_all()
        seed42_runs = CampaignEnumerator.filter_runs(all_runs, seed=42)
        assert len(seed42_runs) > 0
        assert all(r.seed == 42 for r in seed42_runs)


# ── 4. Configuration Freezing Tests ──────────────────────────────────────────

class TestConfigurationFreezing:

    def test_model_configuration_formula_match(self):
        m_nsl = resolve_model_configuration("nslkdd")
        assert m_nsl["parameter_count"] == 53125
        m_cic = resolve_model_configuration("ciciot2023")
        assert m_cic["parameter_count"] == 54600
        m_edge = resolve_model_configuration("edgeiiotset")
        assert m_edge["parameter_count"] == 58310

    def test_freeze_configuration_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = RunSpecification(block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids", seed=42)
            frozen = freeze_configuration(spec, tmpdir, allow_dirty=True)
            cfg_file = os.path.join(tmpdir, "config.json")
            assert os.path.exists(cfg_file)
            with open(cfg_file, "r") as f:
                loaded = json.load(f)
            assert loaded["run_id"] == spec.run_id
            assert loaded["scientific_configuration"]["dataset"] == "nslkdd"
            assert "git_provenance" in loaded
            assert "environment" in loaded

    def test_dirty_tree_protection(self):
        """When working tree is dirty and allow_dirty=False, freeze_configuration must raise RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = RunSpecification(block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids", seed=42)
            # If git working tree is dirty:
            from utils.provenance import git_state
            is_dirty = git_state().get("git_dirty", False)
            if is_dirty:
                with pytest.raises(RuntimeError, match="Refusing to execute experiment"):
                    freeze_configuration(spec, tmpdir, allow_dirty=False)


# ── 5. Checkpointing, Resume & Mismatch Tests ────────────────────────────────

class TestCheckpointAndResume:

    def test_completed_run_resumes_without_rerun(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CampaignRunner(base_results_dir=tmpdir, allow_dirty=True)
            spec = RunSpecification(block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids", seed=42)
            out_dir = get_artifact_directory(spec, base_dir=tmpdir)

            # Manually simulate a completed run
            os.makedirs(out_dir, exist_ok=True)
            freeze_configuration(spec, out_dir, allow_dirty=True)
            write_status(out_dir, status="COMPLETED")
            write_metrics(out_dir, {
                "final_accuracy": 0.88,
                "final_f1_macro": 0.85,
                "final_attack_success_rate": 0.05,
            })

            # Execute run: must detect COMPLETED and return cached
            res = runner.execute_run(spec, dry_run=False, verbose=False)
            assert res["status"] == "COMPLETED"
            assert res.get("resumed") is True
            assert res["metrics"]["final_accuracy"] == 0.88

    def test_configuration_mismatch_during_resume_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CampaignRunner(base_results_dir=tmpdir, allow_dirty=True)
            spec = RunSpecification(block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids", seed=42)
            out_dir = get_artifact_directory(spec, base_dir=tmpdir)
            os.makedirs(out_dir, exist_ok=True)

            # Write a config.json with a DIFFERENT run_id
            mismatched_config = {
                "run_id": "run_DIFFERENT_HASH",
                "scientific_configuration": {"dataset": "different"},
            }
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(mismatched_config, f)
            write_status(out_dir, status="COMPLETED")
            write_metrics(out_dir, {"final_accuracy": 0.9, "final_f1_macro": 0.9, "final_attack_success_rate": 0.1})

            with pytest.raises(RuntimeError, match="Configuration mismatch during resume"):
                runner.execute_run(spec, dry_run=False, verbose=False)

    def test_interrupted_run_detected_and_restarted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CampaignRunner(base_results_dir=tmpdir, allow_dirty=True)
            spec = RunSpecification(block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids", seed=42)
            out_dir = get_artifact_directory(spec, base_dir=tmpdir)

            # Simulate interrupted run (status RUNNING)
            os.makedirs(out_dir, exist_ok=True)
            write_status(out_dir, status="RUNNING")

            # Execute with dry_run: should restart cleanly and update to SKIPPED
            res = runner.execute_run(spec, dry_run=True, verbose=False)
            assert res["status"] == "SKIPPED"
            assert read_status(out_dir)["status"] == "SKIPPED"


# ── 6. Dataset Blocker Handling Tests ─────────────────────────────────────────

class TestDatasetBlockerHandling:

    def test_ciciot2023_execution_is_blocked(self):
        """CIC-IoT-2023 runs must be marked BLOCKED when raw files are absent."""
        available, reason = check_dataset_availability("ciciot2023")
        assert available is False
        assert "BLOCKED" in reason

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CampaignRunner(base_results_dir=tmpdir, allow_dirty=True)
            spec = RunSpecification(block="B2", purpose="Test", dataset="ciciot2023", strategy="tvflids", seed=42)
            res = runner.execute_run(spec, dry_run=False, verbose=False)
            assert res["status"] == "BLOCKED"
            out_dir = get_artifact_directory(spec, base_dir=tmpdir)
            st = read_status(out_dir)
            assert st["status"] == "BLOCKED"
            assert "CIC-IoT-2023" in st["error_message"]

    def test_edgeiiotset_execution_is_blocked(self):
        """Edge-IIoTset runs must be marked BLOCKED when raw files are absent."""
        available, reason = check_dataset_availability("edgeiiotset")
        assert available is False
        assert "BLOCKED" in reason

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CampaignRunner(base_results_dir=tmpdir, allow_dirty=True)
            spec = RunSpecification(block="B2", purpose="Test", dataset="edgeiiotset", strategy="tvflids", seed=42)
            res = runner.execute_run(spec, dry_run=False, verbose=False)
            assert res["status"] == "BLOCKED"
            out_dir = get_artifact_directory(spec, base_dir=tmpdir)
            st = read_status(out_dir)
            assert st["status"] == "BLOCKED"
            assert "Edge-IIoTset" in st["error_message"]


# ── 7. Seed Isolation & Reproducibility Tests ─────────────────────────────────

class TestSeedIsolation:

    def test_seed_isolation_run_order_independence(self):
        """
        Verify that running seed 42 after seed 1042 produces identical RNG
        sequences as running seed 42 directly.
        """
        import numpy as np
        import torch
        from utils.seed import set_all_seeds

        # Sequence 1: Seed 42 alone
        set_all_seeds(42)
        np_seq_alone = np.random.rand(10)
        torch_seq_alone = torch.rand(10)

        # Sequence 2: Seed 1042, then Seed 42
        set_all_seeds(1042)
        _ = np.random.rand(50)
        _ = torch.rand(50)

        set_all_seeds(42)
        np_seq_after = np.random.rand(10)
        torch_seq_after = torch.rand(10)

        assert np.allclose(np_seq_alone, np_seq_after)
        assert torch.allclose(torch_seq_alone, torch_seq_after)


# ── 8. Artifact Hierarchy & Schema Tests ──────────────────────────────────────

class TestArtifactSchema:

    def test_artifact_hierarchy_path(self):
        spec = RunSpecification(block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids", attack="label_flip_30", seed=42)
        path = get_artifact_directory(spec, base_dir="results")
        expected = os.path.join("results", "nslkdd", "B2", "tvflids", "label_flip_30", spec.run_id)
        assert path == expected

    def test_result_schema_validation(self):
        valid = {
            "final_accuracy": 0.85,
            "final_f1_macro": 0.82,
            "final_attack_success_rate": 0.03,
            "final_attack_recall": 0.95,
        }
        assert validate_result_schema(valid) is True

        invalid_missing = {
            "final_accuracy": 0.85,
        }
        assert validate_result_schema(invalid_missing) is False

        invalid_type = {
            "final_accuracy": "0.85",
            "final_f1_macro": 0.82,
            "final_attack_success_rate": 0.03,
        }
        assert validate_result_schema(invalid_type) is False


# ── 9. Provenance-Safe Resume & Cache Reuse Tests ─────────────────────────────

class TestProvenanceSafeResume:

    def test_completed_run_same_commit_allowed(self):
        """1. completed run + same commit -> cached reuse allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = RunSpecification(block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids", seed=42)
            out_dir = get_artifact_directory(spec, base_dir=tmpdir)

            # Freeze config and record completed status & metrics
            freeze_configuration(spec, out_dir, allow_dirty=True)
            write_status(out_dir, status="COMPLETED")
            write_metrics(out_dir, {
                "final_accuracy": 0.91,
                "final_f1_macro": 0.89,
                "final_attack_success_rate": 0.02,
            })

            # Read frozen commit
            with open(os.path.join(out_dir, "config.json"), "r") as f:
                stored_cfg = json.load(f)
            stored_commit = stored_cfg["git_provenance"]["git_commit"]
            stored_dataset_prov = stored_cfg["dataset_provenance"]

            # Run with runner configured with matching commit and dataset provenance
            runner = CampaignRunner(
                base_results_dir=tmpdir,
                allow_dirty=True,
                current_git={"git_commit": stored_commit, "git_dirty": False},
                current_dataset_prov=stored_dataset_prov,
            )

            res = runner.execute_run(spec, dry_run=False, verbose=False)
            assert res["status"] == "COMPLETED"
            assert res.get("resumed") is True
            assert res["metrics"]["final_accuracy"] == 0.91

    def test_completed_run_different_commit_rejected(self):
        """2. completed run + different commit -> cached reuse rejected with explicit error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = RunSpecification(block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids", seed=42)
            out_dir = get_artifact_directory(spec, base_dir=tmpdir)

            freeze_configuration(spec, out_dir, allow_dirty=True)
            write_status(out_dir, status="COMPLETED")
            write_metrics(out_dir, {
                "final_accuracy": 0.91,
                "final_f1_macro": 0.89,
                "final_attack_success_rate": 0.02,
            })

            # Manually set stored commit to C1
            cfg_path = os.path.join(out_dir, "config.json")
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            cfg["git_provenance"]["git_commit"] = "commit_C1_hash"
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)

            # Runner is at commit C2
            runner = CampaignRunner(
                base_results_dir=tmpdir,
                allow_dirty=True,
                current_git={"git_commit": "commit_C2_hash", "git_dirty": False},
                current_dataset_prov=cfg["dataset_provenance"],
            )

            with pytest.raises(RuntimeError) as exc_info:
                runner.execute_run(spec, dry_run=False, verbose=False)

            err_msg = str(exc_info.value)
            assert f"Completed artifact exists for run_id={spec.run_id}, but provenance differs:" in err_msg
            assert "stored_commit=commit_C1_hash" in err_msg
            assert "current_commit=commit_C2_hash" in err_msg
            assert "Refusing cached-result reuse." in err_msg

    def test_completed_run_different_dataset_provenance_rejected(self):
        """3. completed run + same commit + different dataset provenance -> cached reuse rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = RunSpecification(block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids", seed=42)
            out_dir = get_artifact_directory(spec, base_dir=tmpdir)

            freeze_configuration(spec, out_dir, allow_dirty=True)
            write_status(out_dir, status="COMPLETED")
            write_metrics(out_dir, {
                "final_accuracy": 0.91,
                "final_f1_macro": 0.89,
                "final_attack_success_rate": 0.02,
            })

            # Config has dataset_provenance version v1
            cfg_path = os.path.join(out_dir, "config.json")
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            commit = cfg["git_provenance"]["git_commit"]
            cfg["dataset_provenance"]["raw_files"]["train_hash"] = "hash_v1"
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)

            # Current environment has dataset_provenance version v2
            different_dataset_prov = dict(cfg["dataset_provenance"])
            different_dataset_prov["raw_files"] = dict(cfg["dataset_provenance"]["raw_files"])
            different_dataset_prov["raw_files"]["train_hash"] = "hash_v2_altered"

            runner = CampaignRunner(
                base_results_dir=tmpdir,
                allow_dirty=True,
                current_git={"git_commit": commit, "git_dirty": False},
                current_dataset_prov=different_dataset_prov,
            )

            with pytest.raises(RuntimeError) as exc_info:
                runner.execute_run(spec, dry_run=False, verbose=False)

            err_msg = str(exc_info.value)
            assert "dataset provenance differs" in err_msg
            assert "Refusing cached-result reuse." in err_msg

    def test_completed_run_mismatched_scientific_configuration_rejected(self):
        """4. completed run + mismatched scientific configuration -> cached reuse rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = RunSpecification(block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids", seed=42)
            out_dir = get_artifact_directory(spec, base_dir=tmpdir)

            freeze_configuration(spec, out_dir, allow_dirty=True)
            write_status(out_dir, status="COMPLETED")
            write_metrics(out_dir, {
                "final_accuracy": 0.91,
                "final_f1_macro": 0.89,
                "final_attack_success_rate": 0.02,
            })

            # Alter scientific configuration in config.json while keeping run_id
            cfg_path = os.path.join(out_dir, "config.json")
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            cfg["scientific_configuration"]["num_rounds"] = 999
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)

            runner = CampaignRunner(base_results_dir=tmpdir, allow_dirty=True)

            with pytest.raises(RuntimeError) as exc_info:
                runner.execute_run(spec, dry_run=False, verbose=False)

            err_msg = str(exc_info.value)
            assert "scientific configuration differs" in err_msg
            assert "Refusing cached-result reuse." in err_msg

    def test_blocked_dataset_behavior_remains_unchanged(self):
        """5. blocked dataset behavior remains unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CampaignRunner(base_results_dir=tmpdir, allow_dirty=True)
            spec_cic = RunSpecification(block="B2", purpose="Test", dataset="ciciot2023", strategy="tvflids", seed=42)
            out_dir_cic = get_artifact_directory(spec_cic, base_dir=tmpdir)

            res_cic = runner.execute_run(spec_cic, dry_run=False, verbose=False)
            assert res_cic["status"] == "BLOCKED"
            assert read_status(out_dir_cic)["status"] == "BLOCKED"
            assert read_metrics(out_dir_cic) is None

            spec_edge = RunSpecification(block="B2", purpose="Test", dataset="edgeiiotset", strategy="tvflids", seed=42)
            out_dir_edge = get_artifact_directory(spec_edge, base_dir=tmpdir)

            res_edge = runner.execute_run(spec_edge, dry_run=False, verbose=False)
            assert res_edge["status"] == "BLOCKED"
            assert read_status(out_dir_edge)["status"] == "BLOCKED"
            assert read_metrics(out_dir_edge) is None

    def test_fresh_run_after_provenance_mismatch_produces_new_valid_artifact(self):
        """6. fresh run after provenance mismatch produces a new valid artifact rather than silently reusing old one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = RunSpecification(block="B2", purpose="Test", dataset="nslkdd", strategy="tvflids", seed=42)
            out_dir = get_artifact_directory(spec, base_dir=tmpdir)

            # Old artifact from commit C1
            freeze_configuration(spec, out_dir, allow_dirty=True)
            cfg_path = os.path.join(out_dir, "config.json")
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            cfg["git_provenance"]["git_commit"] = "commit_C1_hash"
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            write_status(out_dir, status="COMPLETED")
            write_metrics(out_dir, {
                "final_accuracy": 0.50,
                "final_f1_macro": 0.40,
                "final_attack_success_rate": 0.60,
            })

            # Runner at commit C2
            runner = CampaignRunner(
                base_results_dir=tmpdir,
                allow_dirty=True,
                current_git={"git_commit": "commit_C2_hash", "git_dirty": False},
                current_dataset_prov=cfg["dataset_provenance"],
            )

            # Normal execute_run refuses reuse
            with pytest.raises(RuntimeError, match="provenance differs"):
                runner.execute_run(spec, dry_run=False, verbose=False)

            # Executing with force=True and dry_run=True overrides stale cache and writes fresh artifact
            res = runner.execute_run(spec, dry_run=True, force=True, verbose=False)
            assert res["status"] == "SKIPPED"
            assert res["dry_run"] is True

            # Verify that the artifact was updated to commit C2
            with open(cfg_path, "r") as f:
                updated_cfg = json.load(f)
            assert updated_cfg["git_provenance"]["git_commit"] == "commit_C2_hash"
            assert read_status(out_dir)["status"] == "SKIPPED"
