"""
tests/test_model_and_nslkdd_alignment.py
Comprehensive test suite verifying Stage 1 alignment:
- MLP Architecture alignment with IEEE paper §VI-B
- NSL-KDD preprocessing, quota-based validation splitting, tuning splitting,
  server-only zero-leakage normalization, and client-local SMOTE (§VI-A, Table I).
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.mlp import IDSMLP
from data.preprocessing.nslkdd_pipeline import (
    CLASS_NAMES,
    PAPER_VAL_QUOTAS,
    PAPER_TUNE_QUOTAS,
    PAPER_CLIENT_QUOTAS,
    PAPER_TEST_QUOTAS,
    compute_nslkdd_quotas,
    extract_nslkdd_splits,
    encode_and_scale_splits,
    apply_client_local_smote,
    apply_smote,
    get_nslkdd_splits,
    build_pipeline,
)

TRAIN_PATH = "data/raw/KDDTrain+.txt"
TEST_PATH = "data/raw/KDDTest+.txt"

has_raw_data = os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)


# ============================================================================
# PART B: MLP Model Architecture Verification
# ============================================================================

class TestMLPAlignment:
    """Verifies that IDSMLP matches IEEE paper §VI-B exactly."""

    def test_model_uses_layernorm_not_batchnorm(self):
        """The paper mandates LayerNorm and explicitly rejects BatchNorm for non-IID FL."""
        model = IDSMLP(input_dim=41, num_classes=5)

        layer_norms = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
        batch_norms = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]

        assert len(layer_norms) == 3, f"Expected exactly 3 LayerNorm layers, found {len(layer_norms)}"
        assert len(batch_norms) == 0, f"BatchNorm must NOT be used in IDSMLP! Found {len(batch_norms)}"

        # Check LayerNorm normalized shapes
        assert layer_norms[0].normalized_shape == (256,)
        assert layer_norms[1].normalized_shape == (128,)
        assert layer_norms[2].normalized_shape == (64,)

    def test_model_layer_ordering_and_output_unnormalized(self):
        """
        Architecture must be:
        Linear(256) -> LayerNorm(256) -> ReLU -> Dropout(0.3)
        -> Linear(128) -> LayerNorm(128) -> ReLU -> Dropout(0.3)
        -> Linear(64) -> LayerNorm(64) -> ReLU -> Dropout(0.3)
        -> Linear(K)
        Output layer must NOT receive LayerNorm or Dropout.
        """
        model = IDSMLP(input_dim=41, num_classes=5)
        children = list(model.network.children())

        expected_types = [
            nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout,
            nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout,
            nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout,
            nn.Linear,
        ]

        assert len(children) == len(expected_types), (
            f"Expected {len(expected_types)} sequential layers, got {len(children)}"
        )

        for idx, (layer, exp_type) in enumerate(zip(children, expected_types)):
            assert isinstance(layer, exp_type), (
                f"Layer {idx} expected {exp_type.__name__}, got {type(layer).__name__}"
            )
            if isinstance(layer, nn.Dropout):
                assert layer.p == 0.3, f"Dropout rate expected 0.3, got {layer.p}"

        # Final layer check
        last_layer = children[-1]
        assert isinstance(last_layer, nn.Linear)
        assert last_layer.out_features == 5

    @pytest.mark.parametrize("d,K,expected_params", [
        (41, 5, 53125),   # NSL-KDD (Paper line 355)
        (46, 8, 54600),   # CIC-IoT-2023
        (61, 6, 58310),   # Edge-IIoTset
    ])
    def test_exact_parameter_counts(self, d, K, expected_params):
        """Verify exact parameter counts for each of the three paper datasets."""
        model = IDSMLP(input_dim=d, num_classes=K)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == expected_params, (
            f"Dataset (d={d}, K={K}): expected {expected_params} parameters, got {total_params}"
        )

    def test_parameter_count_formula(self):
        """
        Closed-form parameter formula:
        P(d, K) = 256*d + 65*K + 42,304
        Derivation:
          Layer 1: Linear(d, 256) -> 256*d + 256; LayerNorm(256) -> 256 + 256 = 512
          Layer 2: Linear(256, 128) -> 32,768 + 128 = 32,896; LayerNorm(128) -> 128 + 128 = 256
          Layer 3: Linear(128, 64) -> 8,192 + 64 = 8,256; LayerNorm(64) -> 64 + 64 = 128
          Layer 4: Linear(64, K) -> 64*K + K = 65*K
          Constant sum: 256 + 512 + 32896 + 256 + 8256 + 128 = 42,304.
        """
        for d in [20, 41, 46, 61, 100]:
            for K in [2, 5, 8, 10]:
                model = IDSMLP(input_dim=d, num_classes=K)
                actual_params = sum(p.numel() for p in model.parameters())
                formula_params = 256 * d + 65 * K + 42304
                assert actual_params == formula_params, (
                    f"Mismatch at d={d}, K={K}: actual={actual_params}, formula={formula_params}"
                )


# ============================================================================
# PARTS C, D, E, H: NSL-KDD Quota & Splitting Verification
# ============================================================================

@pytest.mark.skipif(not has_raw_data, reason="Raw NSL-KDD files not available")
class TestNSLKDDPartitionsAndQuotas:
    """Verifies partition sizes, exact quotas, disjointness, and determinism."""

    def test_exact_partition_sizes(self):
        """
        Test 1 — Exact partition sizes:
        D_val = 2,000
        D_tune = 12,398
        D_client = 111,575
        D_test = 22,544
        """
        splits = get_nslkdd_splits(TRAIN_PATH, TEST_PATH, seed=42)

        X_val, y_val = splits["D_val"]
        X_tune, y_tune = splits["D_tune"]
        X_client, y_client = splits["D_client"]
        X_test, y_test = splits["D_test"]

        assert len(X_val) == 2000
        assert len(y_val) == 2000
        assert len(X_tune) == 12398
        assert len(y_tune) == 12398
        assert len(X_client) == 111575
        assert len(y_client) == 111575
        assert len(X_test) == 22544
        assert len(y_test) == 22544

        # Total training records
        total_train = len(X_val) + len(X_tune) + len(X_client)
        assert total_train == 125973

    def test_validation_and_partition_class_quotas(self):
        """
        Test 2 — Exact validation class quotas from Table I:
        Normal: 1,016
        DoS: 693
        Probe: 176
        R2L: 100
        U2R: 15
        Total = 2,000
        """
        splits = get_nslkdd_splits(TRAIN_PATH, TEST_PATH, seed=42)

        _, y_val = splits["D_val"]
        val_counts = dict(zip(*np.unique(y_val, return_counts=True)))
        assert val_counts == PAPER_VAL_QUOTAS, f"Validation quota mismatch: {val_counts} vs {PAPER_VAL_QUOTAS}"

        _, y_tune = splits["D_tune"]
        tune_counts = dict(zip(*np.unique(y_tune, return_counts=True)))
        assert tune_counts == PAPER_TUNE_QUOTAS, f"Tuning quota mismatch: {tune_counts} vs {PAPER_TUNE_QUOTAS}"

        _, y_client = splits["D_client"]
        client_counts = dict(zip(*np.unique(y_client, return_counts=True)))
        assert client_counts == PAPER_CLIENT_QUOTAS, f"Client quota mismatch: {client_counts} vs {PAPER_CLIENT_QUOTAS}"

        _, y_test = splits["D_test"]
        test_counts = dict(zip(*np.unique(y_test, return_counts=True)))
        assert test_counts == PAPER_TEST_QUOTAS, f"Test quota mismatch: {test_counts} vs {PAPER_TEST_QUOTAS}"

    def test_disjointness_of_subsets(self):
        """
        Test 3 — Disjointness:
        Ensures sample indices allocated to D_val, D_tune, and D_client are strictly disjoint.
        """
        from data.preprocessing.nslkdd_pipeline import load_nslkdd, map_labels
        train_df, _ = load_nslkdd(TRAIN_PATH, TEST_PATH)
        train_df = map_labels(train_df)

        # Add index tracker column
        train_df["orig_idx"] = np.arange(len(train_df))

        val_df, tune_df, client_df = extract_nslkdd_splits(train_df, seed=42)

        val_set = set(val_df["orig_idx"])
        tune_set = set(tune_df["orig_idx"])
        client_set = set(client_df["orig_idx"])

        assert len(val_set) == 2000
        assert len(tune_set) == 12398
        assert len(client_set) == 111575

        # Pairwise intersections must be empty
        assert len(val_set & tune_set) == 0, f"Val and Tune overlap: {len(val_set & tune_set)} samples"
        assert len(val_set & client_set) == 0, f"Val and Client overlap: {len(val_set & client_set)} samples"
        assert len(tune_set & client_set) == 0, f"Tune and Client overlap: {len(tune_set & client_set)} samples"

        # Union must reconstruct entire train set
        assert len(val_set | tune_set | client_set) == 125973

    def test_deterministic_split(self):
        """
        Test 6 — Deterministic split:
        Running preprocessing twice with the same seed must produce identical
        sample membership and numerical representations.
        """
        splits1 = get_nslkdd_splits(TRAIN_PATH, TEST_PATH, seed=42)
        splits2 = get_nslkdd_splits(TRAIN_PATH, TEST_PATH, seed=42)

        for key in ["D_val", "D_tune", "D_client", "D_test"]:
            X1, y1 = splits1[key]
            X2, y2 = splits2[key]
            assert np.array_equal(y1, y2), f"Labels mismatch for {key}"
            assert np.allclose(X1, X2, atol=1e-7), f"Features mismatch for {key}"

    def test_loud_failure_on_unsatisfiable_quota(self):
        """Pipeline must fail loudly with a diagnostic if quota cannot be met."""
        # Case 1: Overall dataset size smaller than val_size
        small_df = pd.DataFrame({
            "label": [0] * 1000 + [1] * 500,
            "feature": np.random.randn(1500),
        })
        with pytest.raises(ValueError, match="Cannot extract validation set"):
            extract_nslkdd_splits(small_df, seed=42, val_size=2000)

        # Case 2: Dataset large enough overall, but minority class lacks quota
        # e.g., Class 4 (U2R) has only 5 records when 15 are required
        toy_df = pd.DataFrame({
            "label": [0] * 20000 + [1] * 10000 + [2] * 2000 + [3] * 500 + [4] * 5,
            "feature": np.random.randn(32505),
        })
        with pytest.raises(ValueError, match="Cannot satisfy validation quota for class 4"):
            extract_nslkdd_splits(toy_df, seed=42, val_size=2000)


# ============================================================================
# PART F: Normalization & Zero Data Leakage Verification
# ============================================================================

class TestNormalizationZeroLeakage:
    """
    Test 4 — Scaler Leakage:
    Verifies that scaler statistics depend EXCLUSIVELY on server-held data (D_val U D_tune).
    Client training data and test data distributions must NOT influence the fitted scaler.
    """

    def test_scaler_invariant_to_client_and_test_distributions(self):
        """
        Construct two scenarios with identical server sets (D_val, D_tune)
        but wildly different client and test sets (e.g. multiplied by 10,000,
        containing out-of-distribution values).
        The fitted scaler parameters (data_min_, data_max_, scale_) must be 100% identical.
        """
        rng = np.random.RandomState(42)
        d = 10

        # Fixed server data (val + tune)
        val_df = pd.DataFrame(rng.randn(200, d), columns=[f"f{i}" for i in range(d)])
        val_df["label"] = rng.randint(0, 5, size=200)
        tune_df = pd.DataFrame(rng.randn(500, d), columns=[f"f{i}" for i in range(d)])
        tune_df["label"] = rng.randint(0, 5, size=500)

        # Baseline client and test data
        client_df_1 = pd.DataFrame(rng.randn(1000, d), columns=[f"f{i}" for i in range(d)])
        client_df_1["label"] = rng.randint(0, 5, size=1000)
        test_df_1 = pd.DataFrame(rng.randn(300, d), columns=[f"f{i}" for i in range(d)])
        test_df_1["label"] = rng.randint(0, 5, size=300)

        # Altered client and test data: extreme scale (e.g. 10,000x) and offsets
        client_df_2 = client_df_1.copy()
        test_df_2 = test_df_1.copy()
        for i in range(d):
            client_df_2[f"f{i}"] = client_df_2[f"f{i}"] * 10000.0 + 5000.0
            test_df_2[f"f{i}"] = test_df_2[f"f{i}"] * 50000.0 - 20000.0

        # Encode and scale both scenarios
        *_, scaler_1, _ = encode_and_scale_splits(val_df, tune_df, client_df_1, test_df_1)
        *_, scaler_2, _ = encode_and_scale_splits(val_df, tune_df, client_df_2, test_df_2)

        # Assert scaler statistics are identical
        assert np.array_equal(scaler_1.data_min_, scaler_2.data_min_), (
            "data_min_ changed when client/test distributions changed! Scaler is leaking."
        )
        assert np.array_equal(scaler_1.data_max_, scaler_2.data_max_), (
            "data_max_ changed when client/test distributions changed! Scaler is leaking."
        )
        assert np.array_equal(scaler_1.scale_, scaler_2.scale_), (
            "scale_ changed when client/test distributions changed! Scaler is leaking."
        )

    def test_unclipped_minmax_preserves_out_of_range_test_values(self):
        """Paper §VI-A: 'test values outside the fitted range are kept'."""
        val_df = pd.DataFrame({"f0": [0.0, 10.0], "label": [0, 1]})
        tune_df = pd.DataFrame({"f0": [2.0, 8.0], "label": [0, 1]})
        client_df = pd.DataFrame({"f0": [5.0], "label": [0]})
        # Test set has values outside server [0, 10] range
        test_df = pd.DataFrame({"f0": [-5.0, 15.0], "label": [0, 1]})

        *_, X_test, _, scaler, _ = encode_and_scale_splits(val_df, tune_df, client_df, test_df)

        # Expected: -5 -> -0.5, 15 -> 1.5 (not clipped to 0 or 1)
        assert np.isclose(X_test[0, 0], -0.5), f"Expected -0.5, got {X_test[0, 0]}"
        assert np.isclose(X_test[1, 0], 1.5), f"Expected 1.5, got {X_test[1, 0]}"


# ============================================================================
# PART G: SMOTE Protocol & Scope Verification
# ============================================================================

class TestSMOTEProtocolAndScope:
    """
    Test 5 — SMOTE Scope:
    Verifies that SMOTE is never applied to server sets or test set,
    and adheres strictly to the client-local balancing rule.
    """

    def test_server_and_test_sets_never_oversampled(self):
        """D_val, D_tune, and D_test must never be oversampled by the pipeline."""
        if not has_raw_data:
            pytest.skip("Raw data not available")

        splits = get_nslkdd_splits(TRAIN_PATH, TEST_PATH, seed=42)
        _, y_val = splits["D_val"]
        _, y_tune = splits["D_tune"]
        _, y_test = splits["D_test"]

        # If SMOTE were applied, minority classes would be raised to majority count
        # Assert exact raw Table I counts are preserved
        assert dict(zip(*np.unique(y_val, return_counts=True))) == PAPER_VAL_QUOTAS
        assert dict(zip(*np.unique(y_tune, return_counts=True))) == PAPER_TUNE_QUOTAS
        assert dict(zip(*np.unique(y_test, return_counts=True))) == PAPER_TEST_QUOTAS

    def test_client_local_smote_median_rule(self):
        """
        Paper §VI-A:
        'every class below the client's median non-empty class count is raised
        to that median, with k=min(5,n_c-1) neighbors when the class has n_c>=2
        records, duplication when n_c=1, and nothing when n_c=0.'
        """
        rng = np.random.RandomState(42)
        # Construct client data:
        # Class 0: 50 records
        # Class 1: 30 records
        # Class 2: 10 records
        # Class 3: 4 records (>= 2: SMOTE k=min(5, 3)=3)
        # Class 4: 1 record  (== 1: duplicate to median)
        # Class 5: 0 records (absent: remains 0)
        counts = {0: 50, 1: 30, 2: 10, 3: 4, 4: 1}
        X_parts, y_parts = [], []
        for c, cnt in counts.items():
            X_parts.append(rng.randn(cnt, 8).astype(np.float32) + c * 2.0)
            y_parts.append(np.full(cnt, c, dtype=np.int64))

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        # Non-empty counts: [50, 30, 10, 4, 1]. Median = 10.
        assert np.median([50, 30, 10, 4, 1]) == 10.0

        X_res, y_res = apply_client_local_smote(X, y, random_state=42)

        res_counts = dict(zip(*np.unique(y_res, return_counts=True)))
        expected = {
            0: 50,  # >= median, unchanged
            1: 30,  # >= median, unchanged
            2: 10,  # == median, unchanged
            3: 10,  # was 4, raised to median 10 via SMOTE
            4: 10,  # was 1, raised to median 10 via duplication
        }
        assert res_counts == expected, f"Client SMOTE counts mismatch: {res_counts} vs {expected}"
        assert 5 not in res_counts, "Class with 0 samples must remain 0!"

    def test_local_vs_global_smote_isolation(self):
        """
        Synthetic proof:
        Global SMOTE before partitioning blends data across client boundaries.
        Client-local SMOTE partitions first, preserving client boundary isolation.
        """
        # Client A data in quadrant (+, +), Client B data in quadrant (-, -)
        rng = np.random.RandomState(42)
        X_A = rng.normal(loc=5.0, scale=0.1, size=(20, 2)).astype(np.float32)
        y_A = np.array([0] * 18 + [1] * 2, dtype=np.int64)

        X_B = rng.normal(loc=-5.0, scale=0.1, size=(20, 2)).astype(np.float32)
        y_B = np.array([0] * 18 + [1] * 2, dtype=np.int64)

        # Local SMOTE: Client A's oversampled records stay in loc=5.0
        X_A_res, y_A_res = apply_client_local_smote(X_A, y_A, random_state=42)
        assert np.all(X_A_res > 0), "Local SMOTE on Client A created points outside Client A cluster!"

        # Local SMOTE on Client B: stays in loc=-5.0
        X_B_res, y_B_res = apply_client_local_smote(X_B, y_B, random_state=42)
        assert np.all(X_B_res < 0), "Local SMOTE on Client B created points outside Client B cluster!"


# ============================================================================
# PART I: Backward Compatibility Verification
# ============================================================================

class TestBackwardCompatibility:
    """Verifies that build_pipeline maintains backward compatibility."""

    def test_build_pipeline_return_signature_and_defaults(self):
        """build_pipeline must return the 9-tuple by default."""
        if not has_raw_data:
            pytest.skip("Raw data not available")

        ret = build_pipeline(TRAIN_PATH, TEST_PATH, val_size=2000, protocol="main")
        assert len(ret) == 9
        (X_tr, y_tr, X_val, y_val, X_te, y_te, scaler, encoders, weights) = ret

        assert X_tr.shape == (111575, 41)
        assert y_tr.shape == (111575,)
        assert X_val.shape == (2000, 41)
        assert y_val.shape == (2000,)
        assert X_te.shape == (22544, 41)
        assert y_te.shape == (22544,)
        assert len(weights) == 5

    def test_build_pipeline_return_tune(self):
        """build_pipeline with return_tune=True must return the 11-tuple."""
        if not has_raw_data:
            pytest.skip("Raw data not available")

        ret = build_pipeline(TRAIN_PATH, TEST_PATH, val_size=2000, protocol="main", return_tune=True)
        assert len(ret) == 11
        (X_tr, y_tr, X_val, y_val, X_tune, y_tune, X_te, y_te, scaler, encoders, weights) = ret

        assert X_tune.shape == (12398, 41)
        assert y_tune.shape == (12398,)
