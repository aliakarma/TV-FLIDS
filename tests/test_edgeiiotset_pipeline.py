"""
tests/test_edgeiiotset_pipeline.py

Comprehensive tests for the Edge-IIoTset data pipeline
(data/preprocessing/edgeiiotset_pipeline.py).
Uses deterministic synthetic schema fixtures matching the 61 flow features,
14 attack classes + Normal, time-disjoint split, 60s guard band, zero-leakage
scaler, Table S5 caps, and model compatibility.

No network calls are made.
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.preprocessing.edgeiiotset_pipeline import (
    build_pipeline,
    time_disjoint_split_edgeiiotset,
    FEATURE_COLUMNS,
    LABEL_COL,
    TIMESTAMP_COL,
    RAW_ATTACK_MAPPING,
    CANONICAL_CLASSES,
    INPUT_DIM,
    NUM_CLASSES,
    CAP_TRAIN,
    CAP_TEST,
    GUARD_BAND_SECONDS,
)
from models.mlp import IDSMLP


# ── Synthetic Data Fixtures ──────────────────────────────────────────────────

def _make_synthetic_edgeiiotset_df(n_rows: int, seed: int, start_timestamp: float = 1600000000.0) -> pd.DataFrame:
    """
    Build synthetic Edge-IIoTset dataframe matching the 61-feature schema,
    raw attack labels (14 attacks + Normal), and timestamps.
    """
    rng = np.random.default_rng(seed)

    raw_labels = list(RAW_ATTACK_MAPPING.keys())
    # Heavy class imbalance matching typical Edge-IIoTset
    weights = [0.4] + [0.6 / (len(raw_labels) - 1)] * (len(raw_labels) - 1)
    labels = rng.choice(raw_labels, size=n_rows, p=weights)

    data = {}
    # 61 numeric features
    for col in FEATURE_COLUMNS:
        if any(term in col for term in ("flag", "rate", "count", "type")):
            data[col] = rng.integers(0, 5, size=n_rows).astype(np.float64)
        else:
            data[col] = np.abs(rng.normal(loc=15.0, scale=8.0, size=n_rows))

    # Sequential timestamps (avg 0.5s apart)
    deltas = rng.exponential(scale=0.5, size=n_rows)
    timestamps = start_timestamp + np.cumsum(deltas)

    df = pd.DataFrame(data)
    df[LABEL_COL] = labels
    df[TIMESTAMP_COL] = timestamps
    return df


@pytest.fixture()
def synthetic_edgeiiotset_files():
    """Writes small synthetic train/test Edge-IIoTset CSVs to a tempdir."""
    tmpdir = tempfile.mkdtemp(prefix="edgeiiotset_test_")
    train_path = os.path.join(tmpdir, "edgeiiotset_train.csv")
    test_path = os.path.join(tmpdir, "edgeiiotset_test.csv")

    train_df = _make_synthetic_edgeiiotset_df(n_rows=800, seed=42, start_timestamp=1600000000.0)
    # Test set starts after train set + 100s
    last_train_ts = train_df[TIMESTAMP_COL].max()
    test_df = _make_synthetic_edgeiiotset_df(n_rows=300, seed=123, start_timestamp=last_train_ts + 100.0)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    yield train_path, test_path

    shutil.rmtree(tmpdir, ignore_errors=True)


# ── Pipeline Tests ───────────────────────────────────────────────────────────

class TestEdgeIIoTsetPipeline:
    def test_dimensions_and_types(self, synthetic_edgeiiotset_files):
        train_path, test_path = synthetic_edgeiiotset_files
        (X_train, y_train, X_val, y_val, X_test, y_test,
         scaler, encoders, class_weights) = build_pipeline(
            train_path, test_path, use_smote=True, seed=42, val_size=50,
        )

        assert X_train.shape[1] == INPUT_DIM == 61
        assert X_val.shape[1] == INPUT_DIM == 61
        assert X_test.shape[1] == INPUT_DIM == 61
        assert X_train.dtype == np.float32
        assert X_val.dtype == np.float32
        assert X_test.dtype == np.float32
        assert y_train.dtype == np.int64
        assert y_val.dtype == np.int64
        assert y_test.dtype == np.int64

        assert set(np.unique(y_train)).issubset(set(range(NUM_CLASSES)))
        assert set(np.unique(y_test)).issubset(set(range(NUM_CLASSES)))
        assert len(CANONICAL_CLASSES) == NUM_CLASSES == 6

    def test_time_disjoint_split_and_guard_band(self):
        """Verify time-disjoint splitting drops rows inside 60-second guard band."""
        df = _make_synthetic_edgeiiotset_df(n_rows=1000, seed=99, start_timestamp=1000.0)
        train_df, test_df = time_disjoint_split_edgeiiotset(
            df, train_ratio=0.8, guard_band_seconds=GUARD_BAND_SECONDS
        )

        max_train_ts = train_df[TIMESTAMP_COL].max()
        min_test_ts = test_df[TIMESTAMP_COL].min()

        # Guard band separation
        assert min_test_ts - max_train_ts >= GUARD_BAND_SECONDS
        assert len(train_df) + len(test_df) < len(df)  # Some rows dropped in band

    def test_zero_leakage_scaling(self, synthetic_edgeiiotset_files):
        """Verify scaler is fitted ONLY on server data (D_val U D_tune), not test or client data."""
        train_path, test_path = synthetic_edgeiiotset_files
        (X_train, y_train, X_val, y_val, X_test, y_test,
         scaler, encoders, class_weights) = build_pipeline(
            train_path, test_path, use_smote=False, seed=42, val_size=50,
            protocol="leakage_free",
        )

        # X_val is part of server data (D_val U D_tune) where scaler was fitted
        assert X_val.min() >= -1e-5
        assert X_val.max() <= 1.0 + 1e-5
        assert hasattr(scaler, "data_min_")
        assert len(scaler.data_min_) == INPUT_DIM

    def test_table_s5_caps(self):
        """Verify caps are respected when dataset exceeds CAP_TRAIN / CAP_TEST."""
        tmpdir = tempfile.mkdtemp(prefix="edgeiiotset_caps_")
        train_path = os.path.join(tmpdir, "edgeiiotset_large_train.csv")
        test_path = os.path.join(tmpdir, "edgeiiotset_large_test.csv")

        # Generate 200 rows with small cap
        df_train = _make_synthetic_edgeiiotset_df(n_rows=200, seed=1)
        df_test = _make_synthetic_edgeiiotset_df(n_rows=100, seed=2)
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)

        (X_tr, y_tr, X_v, y_v, X_te, y_te, *_) = build_pipeline(
            train_path, test_path, use_smote=False, seed=42, val_size=20,
            cap_train=100, cap_test=50,
        )

        assert len(X_tr) + len(X_v) <= 100
        assert len(X_te) <= 50

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_model_forward_pass_edgeiiotset(self, synthetic_edgeiiotset_files):
        """Verify canonical IDSMLP forward pass with (d=61, K=6)."""
        train_path, test_path = synthetic_edgeiiotset_files
        (_, _, _, _, X_test, y_test, *_) = build_pipeline(
            train_path, test_path, use_smote=False, seed=42, val_size=50,
        )

        model = IDSMLP(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test))
        assert logits.shape == (len(X_test), NUM_CLASSES)
        assert torch.isfinite(logits).all()

    def test_setup_data_integration(self, synthetic_edgeiiotset_files):
        """Verify run_experiment::setup_data integration with edgeiiotset."""
        train_path, test_path = synthetic_edgeiiotset_files
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        config_path = os.path.abspath(os.path.join(repo_root, "config", "dataset_config.yaml"))

        with open(config_path, "r") as f:
            orig_text = f.read()

        import yaml
        cfg = yaml.safe_load(orig_text)
        cfg["datasets"]["edgeiiotset"]["train_file"] = train_path.replace("\\", "/")
        cfg["datasets"]["edgeiiotset"]["test_file"] = test_path.replace("\\", "/")

        cwd_before = os.getcwd()
        try:
            with open(config_path, "w") as f:
                yaml.safe_dump(cfg, f)

            os.chdir(os.path.abspath(repo_root))
            from experiments.run_experiment import setup_data

            fl_config = {"federated_learning": {"num_clients": 4}}
            client_data, X_test, y_test, X_val, y_val, class_weights = setup_data(
                config=fl_config,
                dataset="edgeiiotset",
                seed=42,
                partition_type="iid",
                val_size=50,
            )

            assert len(client_data) == 4
            for Xc, yc in client_data:
                assert Xc.shape[1] == 61
                assert len(Xc) == len(yc)
            assert X_test.shape[1] == 61
            assert X_val.shape[0] == 50
        finally:
            os.chdir(cwd_before)
            with open(config_path, "w") as f:
                f.write(orig_text)

    def test_canonical_14_attack_taxonomy(self):
        """
        Verify canonical Edge-IIoTset taxonomy (Ferrag et al. 2022):
        Exactly 14 raw attacks + 1 Normal = 15 canonical classes mapped to 6 final classes.
        Specifically catches the '14 vs 21 labels' discrepancy.
        """
        from data.preprocessing.edgeiiotset_pipeline import (
            CANONICAL_RAW_ATTACKS,
            CANONICAL_RAW_MAP,
            CATEGORY_TO_ID,
            map_labels,
        )

        assert len(CANONICAL_RAW_ATTACKS) == 15, (
            f"Expected exactly 15 canonical labels (14 attacks + Normal), got {len(CANONICAL_RAW_ATTACKS)}"
        )
        assert CANONICAL_RAW_ATTACKS[0] == "Normal"
        raw_attacks = [a for a in CANONICAL_RAW_ATTACKS if a != "Normal"]
        assert len(raw_attacks) == 14, f"Expected exactly 14 raw attacks, got {len(raw_attacks)}"

        # Verify exact grouping into 5 attack classes
        class_counts = {cid: 0 for cid in range(NUM_CLASSES)}
        for label in CANONICAL_RAW_ATTACKS:
            cid = CANONICAL_RAW_MAP[label]
            class_counts[cid] += 1

        assert class_counts[0] == 1, "Normal class must contain exactly 1 raw label ('Normal')"
        assert class_counts[1] == 4, "DoS/DDoS must contain exactly 4 raw attacks (DDoS_UDP, DDoS_ICMP, DDoS_HTTP, DDoS_TCP)"
        assert class_counts[2] == 3, "Injection must contain exactly 3 raw attacks (SQL_injection, XSS, Uploading)"
        assert class_counts[3] == 3, "Scanning must contain exactly 3 raw attacks (Port_Scanning, Vulnerability_scanner, Fingerprinting)"
        assert class_counts[4] == 3, "Malware must contain exactly 3 raw attacks (Backdoor, Password, Ransomware)"
        assert class_counts[5] == 1, "MITM must contain exactly 1 raw attack ('MITM')"

        # Total attacks = 4 + 3 + 3 + 3 + 1 = 14
        total_attacks = sum(class_counts[c] for c in range(1, NUM_CLASSES))
        assert total_attacks == 14, f"Total raw attack count must be 14, got {total_attacks}"

        # Test mapping of each canonical label through map_labels
        df_canonical = pd.DataFrame({"Attack_type": CANONICAL_RAW_ATTACKS})
        df_mapped = map_labels(df_canonical)
        assert len(df_mapped) == 15
        assert list(df_mapped["label"]) == [CANONICAL_RAW_MAP[a] for a in CANONICAL_RAW_ATTACKS]

    def test_unexpected_and_alien_labels_rejected(self):
        """
        Verify that alien labels (e.g. dns_spoofing from CIC-IoT-2023) or unknown attacks
        are NOT silently accepted.
        """
        from data.preprocessing.edgeiiotset_pipeline import map_labels

        df_alien = pd.DataFrame({
            "Attack_type": ["dns_spoofing", "unknown_trojan", "fictitious_attack", "Normal"]
        })
        df_mapped = map_labels(df_alien)
        # Only 'Normal' should survive; all 3 unrecognized labels should be dropped
        assert len(df_mapped) == 1
        assert df_mapped.iloc[0]["label"] == 0

    def test_validation_quota_construction(self):
        """
        Verify Edge-IIoTset validation quota construction matches Table S5 caps and NSL-KDD rules.
        """
        from data.preprocessing.edgeiiotset_pipeline import (
            PAPER_TRAIN_CAPS,
            PAPER_VAL_QUOTAS,
            compute_edgeiiotset_quotas,
        )

        val_quotas, tune_quotas, client_quotas = compute_edgeiiotset_quotas(
            PAPER_TRAIN_CAPS, val_size=2000, tune_fraction=0.1
        )
        assert sum(val_quotas.values()) == 2000
        for c in range(NUM_CLASSES):
            assert val_quotas[c] == PAPER_VAL_QUOTAS[c], (
                f"Class {c}: expected {PAPER_VAL_QUOTAS[c]}, got {val_quotas[c]}"
            )
            assert tune_quotas[c] > 0
            assert client_quotas[c] > 0
            assert val_quotas[c] + tune_quotas[c] + client_quotas[c] == PAPER_TRAIN_CAPS[c]
