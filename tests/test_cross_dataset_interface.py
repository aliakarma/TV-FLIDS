"""
tests/test_cross_dataset_interface.py

Cross-dataset interface tests verifying:
1. Consistent DatasetBundle interface across NSL-KDD, CIC-IoT-2023, Edge-IIoTset.
2. Canonical model parameter count formula P(d, K) = 256*d + 65*K + 42,304.
   - NSL-KDD (41, 5): 53,125
   - CIC-IoT-2023 (46, 8): 54,600
   - Edge-IIoTset (61, 6): 58,310
3. Real PyTorch parameter count matching formula exactly.
4. Dataset manifest generation and serialization.
"""

import os
import sys
import tempfile
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset_bundle import (
    DatasetBundle,
    canonical_param_count,
    verify_parameter_count,
    generate_dataset_manifest,
    DATASET_SPECS,
)
from models.mlp import IDSMLP


class TestCrossDatasetInterface:
    @pytest.mark.parametrize("dataset_name,d,K,expected_params", [
        ("nslkdd", 41, 5, 53125),
        ("ciciot2023", 46, 8, 54600),
        ("edgeiiotset", 61, 6, 58310),
    ])
    def test_canonical_param_counts(self, dataset_name, d, K, expected_params):
        """Verify formula P(d, K) = 256*d + 65*K + 42,304 against paper specification."""
        calc = canonical_param_count(d, K)
        assert calc == expected_params
        assert DATASET_SPECS[dataset_name]["param_count"] == expected_params

        # Verify against actual PyTorch IDSMLP
        model = IDSMLP(input_dim=d, num_classes=K)
        actual_params = sum(p.numel() for p in model.parameters())
        assert actual_params == expected_params

        # Test verification helper
        report = verify_parameter_count(dataset_name, model)
        assert report["matches"] is True
        assert report["expected_params"] == expected_params
        assert report["actual_params"] == expected_params

    def test_dataset_bundle_validation(self):
        """Verify DatasetBundle enforces internal consistency."""
        X_train = np.ones((100, 41), dtype=np.float32)
        y_train = np.zeros(100, dtype=np.int64)
        X_val = np.ones((20, 41), dtype=np.float32)
        y_val = np.zeros(20, dtype=np.int64)
        X_test = np.ones((30, 41), dtype=np.float32)
        y_test = np.zeros(30, dtype=np.int64)
        client_data = [(X_train[:50], y_train[:50]), (X_train[50:], y_train[50:])]
        class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]

        bundle = DatasetBundle(
            name="nslkdd",
            input_dim=41,
            num_classes=5,
            class_names=class_names,
            client_data=client_data,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

        assert bundle.input_dim == 41
        assert bundle.num_classes == 5
        assert bundle.num_clients == 2
        assert bundle.total_train_samples == 100

        # Dimension mismatch should raise ValueError
        with pytest.raises(ValueError, match="Input dimension mismatch"):
            DatasetBundle(
                name="nslkdd",
                input_dim=46,  # wrong
                num_classes=5,
                class_names=class_names,
                client_data=client_data,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
            )

    def test_manifest_generation(self):
        """Verify machine-readable manifest generation."""
        manifest = generate_dataset_manifest(
            dataset_name="edgeiiotset",
            raw_files=["edgeiiotset_train.csv", "edgeiiotset_test.csv"],
            preprocessing_config={"scaler": "MinMaxScaler(clip=False)", "caps": [150000, 30000]},
            num_clients=10,
            partition_type="noniid",
            alpha=0.5,
            seed=42,
        )

        assert manifest["dataset_name"] == "edgeiiotset"
        assert manifest["input_dim"] == 61
        assert manifest["num_classes"] == 6
        assert manifest["param_count"] == 58310
        assert manifest["partition"]["alpha"] == 0.5
        assert manifest["partition"]["num_clients"] == 10
        assert "timestamp" in manifest
        assert "is_synthetic_fixture" in manifest
        assert "execution_type" in manifest
        assert "raw_data_status" in manifest

    def test_bundle_manifest_synthetic_distinction(self):
        """Verify DatasetBundle to_manifest_dict distinguishes synthetic fixture from real data."""
        X_dummy = np.ones((10, 46), dtype=np.float32)
        y_dummy = np.zeros(10, dtype=np.int64)
        bundle = DatasetBundle(
            name="ciciot2023",
            input_dim=46,
            num_classes=8,
            class_names=["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "WebBased", "BruteForce"],
            client_data=[(X_dummy, y_dummy)],
            X_val=X_dummy,
            y_val=y_dummy,
            X_test=X_dummy,
            y_test=y_dummy,
            manifest={"is_synthetic_fixture": True, "execution_type": "synthetic_fixture"},
        )
        manifest_dict = bundle.to_manifest_dict()
        assert manifest_dict["is_synthetic_fixture"] is True
        assert manifest_dict["execution_type"] == "synthetic_fixture"
        assert manifest_dict["raw_data_status"] == "blocked_missing_raw_files"

    def test_cross_dataset_paper_val_quotas(self):
        """Verify all three dataset pipelines export PAPER_VAL_QUOTAS summing to exactly 2,000."""
        from data.preprocessing.nslkdd_pipeline import PAPER_VAL_QUOTAS as KDD_QUOTAS
        from data.preprocessing.ciciot2023_pipeline import PAPER_VAL_QUOTAS as CIC_QUOTAS
        from data.preprocessing.edgeiiotset_pipeline import PAPER_VAL_QUOTAS as EDGE_QUOTAS

        assert sum(KDD_QUOTAS.values()) == 2000
        assert sum(CIC_QUOTAS.values()) == 2000
        assert sum(EDGE_QUOTAS.values()) == 2000

        assert len(KDD_QUOTAS) == 5
        assert len(CIC_QUOTAS) == 8
        assert len(EDGE_QUOTAS) == 6

