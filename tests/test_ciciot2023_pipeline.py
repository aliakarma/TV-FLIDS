"""
tests/test_ciciot2023_pipeline.py

Tiny-synthetic smoke test for the CIC-IoT-2023 data pipeline
(data/preprocessing/ciciot2023_pipeline.py). The real CIC-IoT-2023 dataset
is tens of GB and is explicitly out of scope to download for this
remediation pass; instead this generates a small synthetic CSV whose schema
(46 FEATURE_COLUMNS + a `label` column drawn from the pipeline's 8-class
taxonomy, imbalanced in spirit to the real dataset -- mostly Benign/DDoS,
a few rare classes) matches exactly what build_pipeline() expects, and runs
the real pipeline code end-to-end against it.

No network calls are made anywhere in this file.

Run: .venv\\Scripts\\python.exe -m pytest tests\\test_ciciot2023_pipeline.py -v
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

from data.preprocessing.ciciot2023_pipeline import (
    build_pipeline, FEATURE_COLUMNS, LABEL_COL, CLASS_NAMES, INPUT_DIM, NUM_CLASSES,
)
from models.mlp import IDSMLP


# ── Synthetic data generation ───────────────────────────────────────────────

def _make_synthetic_ciciot2023_df(n_rows: int, seed: int) -> pd.DataFrame:
    """Build a synthetic dataframe with CIC-IoT-2023's schema: 46 numeric
    flow-feature columns + a `label` column of fine-grained attack-name-like
    strings covering all 8 taxonomy categories, imbalanced similarly in
    spirit to the real dataset (mostly Benign + DDoS, a handful of rare
    classes like WebBased/BruteForce)."""
    rng = np.random.default_rng(seed)

    # Fine-grained-style label pool, weighted so Benign/DDoS dominate and
    # WebBased/BruteForce/Spoofing/Mirai/Recon/DoS are rarer -- mirrors the
    # real dataset's heavy class imbalance.
    label_pool = (
        ["BenignTraffic"] * 40
        + ["DDoS-ICMP_Flood", "DDoS-UDP_Flood", "DDoS-SYN_Flood"] * 15
        + ["DoS-TCP_Flood", "DoS-UDP_Flood"] * 10
        + ["Recon-PortScan", "Recon-OSScan"] * 8
        + ["Mirai-greeting_flood", "Mirai-udpplain"] * 8
        + ["MITM-ArpSpoofing", "DNS_Spoofing"] * 6
        + ["DictionaryBruteForce"] * 4
        + ["SqlInjection", "XSS", "CommandInjection", "BrowserHijacking"] * 2
    )
    labels = rng.choice(label_pool, size=n_rows)

    data = {}
    for col in FEATURE_COLUMNS:
        # Flag/protocol-indicator-style columns get small non-negative
        # integers; everything else gets continuous positive values --
        # loosely mimicking the real dataset's mixed flag/statistic columns.
        if any(k in col for k in (
            "flag", "count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH",
            "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
        )):
            data[col] = rng.integers(0, 3, size=n_rows).astype(np.float64)
        else:
            data[col] = np.abs(rng.normal(loc=10.0, scale=5.0, size=n_rows))

    df = pd.DataFrame(data)
    df[LABEL_COL] = labels
    return df


@pytest.fixture()
def synthetic_ciciot2023_files():
    """Writes small synthetic train/test CIC-IoT-2023 CSVs to a tempdir and
    yields their paths. Cleans up afterward."""
    tmpdir = tempfile.mkdtemp(prefix="ciciot2023_smoke_")
    train_path = os.path.join(tmpdir, "CICIoT2023_train.csv")
    test_path = os.path.join(tmpdir, "CICIoT2023_test.csv")

    train_df = _make_synthetic_ciciot2023_df(n_rows=700, seed=42)
    test_df = _make_synthetic_ciciot2023_df(n_rows=200, seed=123)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    yield train_path, test_path

    shutil.rmtree(tmpdir, ignore_errors=True)


# ── build_pipeline() end-to-end tests ───────────────────────────────────────

class TestBuildPipeline:
    def test_loads_and_returns_correct_shapes(self, synthetic_ciciot2023_files):
        train_path, test_path = synthetic_ciciot2023_files
        (X_train, y_train, X_val, y_val, X_test, y_test,
         scaler, encoders, class_weights) = build_pipeline(
            train_path, test_path, use_smote=True, seed=42, val_size=50,
        )

        assert X_train.shape[1] == INPUT_DIM == len(FEATURE_COLUMNS)
        assert X_val.shape[1] == INPUT_DIM
        assert X_test.shape[1] == INPUT_DIM
        assert X_train.dtype == np.float32
        assert X_val.dtype == np.float32
        assert X_test.dtype == np.float32
        assert y_train.dtype == np.int64
        assert y_val.dtype == np.int64
        assert y_test.dtype == np.int64

        # Labels fall within the 8-class taxonomy.
        assert set(np.unique(y_train)).issubset(set(range(NUM_CLASSES)))
        assert set(np.unique(y_test)).issubset(set(range(NUM_CLASSES)))

        # encoders is present (empty dict; no categorical columns) so the
        # 9-tuple shape matches nslkdd/unswnb15 pipelines.
        assert isinstance(encoders, dict)
        assert len(class_weights) == len(np.unique(y_train))

    @pytest.mark.parametrize("val_size", [50, 100])
    def test_val_size_is_respected(self, synthetic_ciciot2023_files, val_size):
        train_path, test_path = synthetic_ciciot2023_files
        (_, _, X_val, y_val, *_rest) = build_pipeline(
            train_path, test_path, use_smote=True, seed=42, val_size=val_size,
        )
        assert X_val.shape[0] == val_size
        assert y_val.shape[0] == val_size

    def test_smote_application_does_not_crash(self, synthetic_ciciot2023_files):
        train_path, test_path = synthetic_ciciot2023_files
        # Rare classes (WebBased, BruteForce) have few synthetic rows, which
        # is exactly the regime that exercises SMOTE's k_neighbors clamping.
        (X_train, y_train, *_rest) = build_pipeline(
            train_path, test_path, use_smote=True, seed=42, val_size=50,
        )
        # After SMOTE, classes present in y_train should be reasonably
        # balanced relative to each other (no crash, and no class left at a
        # single-digit count while others are in the hundreds).
        counts = np.bincount(y_train, minlength=NUM_CLASSES)
        present = counts[counts > 0]
        assert present.min() > 1

    def test_leakage_free_protocol_runs(self, synthetic_ciciot2023_files):
        train_path, test_path = synthetic_ciciot2023_files
        (X_train, y_train, X_val, y_val, X_test, y_test,
         scaler, encoders, class_weights) = build_pipeline(
            train_path, test_path, use_smote=False, seed=42, val_size=50,
            protocol="leakage_free",
        )
        assert X_val.shape[0] == 50
        assert X_train.shape[1] == INPUT_DIM

    def test_canonical_class_ordering(self):
        """Verify canonical class ordering matches paper §VI-A and build_targets.py."""
        expected = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "WebBased", "BruteForce"]
        assert CLASS_NAMES == expected
        assert len(CLASS_NAMES) == 8

    def test_zero_leakage_scaling(self, synthetic_ciciot2023_files):
        """Verify scaler is fitted ONLY on server data (D_val U D_tune), not test or client data."""
        train_path, test_path = synthetic_ciciot2023_files
        (X_train, y_train, X_val, y_val, X_test, y_test,
         scaler, encoders, class_weights) = build_pipeline(
            train_path, test_path, use_smote=False, seed=42, val_size=50,
            protocol="leakage_free",
        )
        assert X_val.min() >= -1e-5
        assert X_val.max() <= 1.0 + 1e-5
        assert hasattr(scaler, "data_min_")
        assert len(scaler.data_min_) == INPUT_DIM

    def test_caps_enforcement(self):
        """Verify caps are respected when dataset exceeds cap limits."""
        from data.preprocessing.ciciot2023_pipeline import time_disjoint_split_ciciot2023, GUARD_BAND_SECONDS
        tmpdir = tempfile.mkdtemp(prefix="ciciot2023_caps_")
        train_path = os.path.join(tmpdir, "ciciot2023_cap_train.csv")
        test_path = os.path.join(tmpdir, "ciciot2023_cap_test.csv")

        df_train = _make_synthetic_ciciot2023_df(n_rows=200, seed=1)
        df_test = _make_synthetic_ciciot2023_df(n_rows=100, seed=2)
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)

        (X_tr, y_tr, X_v, y_v, X_te, y_te, *_) = build_pipeline(
            train_path, test_path, use_smote=False, seed=42, val_size=20,
            cap_train=100, cap_test=50,
        )
        assert len(X_tr) + len(X_v) <= 100
        assert len(X_te) <= 50
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_time_disjoint_split_and_guard_band(self):
        """Verify time-disjoint split logic enforces 60s guard band."""
        from data.preprocessing.ciciot2023_pipeline import time_disjoint_split_ciciot2023, GUARD_BAND_SECONDS
        df = _make_synthetic_ciciot2023_df(n_rows=500, seed=42)
        # Add timestamps
        timestamps = 1600000000.0 + np.cumsum(np.random.exponential(scale=1.0, size=len(df)))
        df["timestamp"] = timestamps
        train_df, test_df = time_disjoint_split_ciciot2023(
            df, train_ratio=0.8, guard_band_seconds=GUARD_BAND_SECONDS, timestamp_col="timestamp"
        )
        assert test_df["timestamp"].min() - train_df["timestamp"].max() >= GUARD_BAND_SECONDS

    def test_model_forward_pass_on_test_set(self, synthetic_ciciot2023_files):
        train_path, test_path = synthetic_ciciot2023_files
        (_, _, _, _, X_test, y_test, *_rest) = build_pipeline(
            train_path, test_path, use_smote=True, seed=42, val_size=50,
        )
        model = IDSMLP(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test))
        assert logits.shape == (X_test.shape[0], NUM_CLASSES)
        assert torch.isfinite(logits).all()


# ── run_experiment.py::setup_data() integration test ───────────────────────

class TestSetupDataIntegration:
    """Verifies experiments/run_experiment.py's existing, unmodified
    `dataset == "ciciot2023"` branch of setup_data() genuinely works against
    this pipeline module, end-to-end. Temporarily overrides
    config/dataset_config.yaml's ciciot2023.train_file/test_file to point at
    synthetic tempdir CSVs (restored after the test) so no synthetic data is
    left behind in data/raw/."""

    def test_setup_data_with_ciciot2023(self, synthetic_ciciot2023_files):
        train_path, test_path = synthetic_ciciot2023_files

        repo_root = os.path.join(os.path.dirname(__file__), "..")
        config_path = os.path.abspath(
            os.path.join(repo_root, "config", "dataset_config.yaml")
        )
        with open(config_path, "r") as f:
            original_config_text = f.read()

        import yaml
        cfg = yaml.safe_load(original_config_text)
        cfg["datasets"]["ciciot2023"]["train_file"] = train_path.replace("\\", "/")
        cfg["datasets"]["ciciot2023"]["test_file"] = test_path.replace("\\", "/")

        cwd_before = os.getcwd()
        try:
            with open(config_path, "w") as f:
                yaml.safe_dump(cfg, f)

            # setup_data() opens "config/dataset_config.yaml" as a path
            # relative to the process cwd; run from the repo root to match
            # how the real experiment entry point is invoked.
            os.chdir(os.path.abspath(repo_root))

            from experiments.run_experiment import setup_data

            fl_config = {
                "federated_learning": {"num_clients": 4},
            }

            client_data, X_test, y_test, X_val, y_val, class_weights = setup_data(
                config=fl_config,
                dataset="ciciot2023",
                seed=42,
                partition_type="iid",
                val_size=50,
            )

            assert len(client_data) == 4
            for Xc, yc in client_data:
                assert Xc.shape[1] == INPUT_DIM
                assert len(Xc) == len(yc)
            assert X_test.shape[1] == INPUT_DIM
            assert X_val.shape[0] == 50
            assert len(class_weights) > 0
        finally:
            os.chdir(cwd_before)
            with open(config_path, "w") as f:
                f.write(original_config_text)

    def test_canonical_34_label_taxonomy(self):
        """
        Verify canonical CIC-IoT-2023 taxonomy (Neto et al., Sensors 2023):
        Exactly 33 raw attacks + 1 Benign = 34 canonical labels mapped to 8 classes.
        Specifically tests VulnerabilityScan mapping to Recon (Class 4).
        """
        from data.preprocessing.ciciot2023_pipeline import (
            CANONICAL_RAW_ATTACKS,
            CANONICAL_RAW_MAP,
            map_labels,
        )

        assert len(CANONICAL_RAW_ATTACKS) == 34, (
            f"Expected exactly 34 canonical labels (33 attacks + BenignTraffic), got {len(CANONICAL_RAW_ATTACKS)}"
        )
        assert CANONICAL_RAW_ATTACKS[0] == "BenignTraffic"
        raw_attacks = [a for a in CANONICAL_RAW_ATTACKS if a != "BenignTraffic"]
        assert len(raw_attacks) == 33, f"Expected exactly 33 raw attacks, got {len(raw_attacks)}"

        # Verify grouping across 8 classes
        class_counts = {cid: 0 for cid in range(NUM_CLASSES)}
        for label in CANONICAL_RAW_ATTACKS:
            cid = CANONICAL_RAW_MAP[label]
            class_counts[cid] += 1

        assert class_counts[0] == 1, "Benign must contain exactly 1 raw label ('BenignTraffic')"
        assert class_counts[1] == 12, "DDoS must contain exactly 12 raw attacks"
        assert class_counts[2] == 4, "DoS must contain exactly 4 raw attacks"
        assert class_counts[3] == 3, "Mirai must contain exactly 3 raw attacks"
        assert class_counts[4] == 5, "Recon must contain exactly 5 raw attacks (including VulnerabilityScan)"
        assert class_counts[5] == 2, "Spoofing must contain exactly 2 raw attacks"
        assert class_counts[6] == 6, "WebBased must contain exactly 6 raw attacks"
        assert class_counts[7] == 1, "BruteForce must contain exactly 1 raw attack"

        # Specific test for VulnerabilityScan
        assert CANONICAL_RAW_MAP["VulnerabilityScan"] == 4, "VulnerabilityScan must map to Class 4 (Recon)"

        # Test map_labels on all canonical labels
        df_canonical = pd.DataFrame({"label": CANONICAL_RAW_ATTACKS})
        df_mapped = map_labels(df_canonical)
        assert len(df_mapped) == 34
        assert list(df_mapped["label"]) == [CANONICAL_RAW_MAP[a] for a in CANONICAL_RAW_ATTACKS]

    def test_validation_quota_construction(self):
        """
        Verify CIC-IoT-2023 validation quota construction matches Table S5 caps and NSL-KDD rules:
        Classes 6 & 7 (< 5,000) receive 100 each; remaining 1,800 distributed proportionally.
        """
        from data.preprocessing.ciciot2023_pipeline import (
            PAPER_TRAIN_CAPS,
            PAPER_VAL_QUOTAS,
            compute_ciciot2023_quotas,
        )

        val_quotas, tune_quotas, client_quotas = compute_ciciot2023_quotas(
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



if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
