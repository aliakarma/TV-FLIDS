"""
data/dataset_bundle.py
Unified cross-dataset interface and provenance manifest generator.

Exposes a consistent abstraction across all three TV-FLIDS datasets:
- NSL-KDD (d=41, K=5)
- CIC-IoT-2023 (d=46, K=8)
- Edge-IIoTset (d=61, K=6)

Guarantees:
1. Uniform dataset bundle interface with train/client pool, validation, tuning, test,
   feature dimension, class mapping, and metadata manifest.
2. Exact parameter closed-form verification: P(d, K) = 256d + 65K + 42,304.
3. Machine-readable provenance and partition audit manifests.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from data.partitioning import get_partitioner


DATASET_SPECS = {
    "nslkdd": {"d": 41, "K": 5, "param_count": 53125},
    "ciciot2023": {"d": 46, "K": 8, "param_count": 54600},
    "edgeiiotset": {"d": 61, "K": 6, "param_count": 58310},
}


@dataclass
class DatasetBundle:
    """
    Standardized container for federated IDS datasets.
    """
    name: str
    input_dim: int
    num_classes: int
    class_names: List[str]
    client_data: List[Tuple[np.ndarray, np.ndarray]]
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_weights: Optional[np.ndarray] = None
    val_quotas: Optional[Dict[int, int]] = None
    X_tune: Optional[np.ndarray] = None
    y_tune: Optional[np.ndarray] = None
    manifest: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.class_weights is None:
            self.class_weights = np.array([])
        if self.val_quotas is None:
            self.val_quotas = {}
        # Dimension validation
        if self.X_val is not None and len(self.X_val) > 0 and self.X_val.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch in X_val: expected {self.input_dim}, got {self.X_val.shape[1]}")
        if self.X_test is not None and len(self.X_test) > 0 and self.X_test.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch in X_test: expected {self.input_dim}, got {self.X_test.shape[1]}")
        for cid, (Xc, yc) in enumerate(self.client_data):
            if len(Xc) > 0 and Xc.shape[1] != self.input_dim:
                raise ValueError(f"Input dimension mismatch in client {cid}: expected {self.input_dim}, got {Xc.shape[1]}")

    @property
    def expected_parameter_count(self) -> int:
        """Derive expected parameter count using canonical formula P(d, K) = 256d + 65K + 42,304."""
        return 256 * self.input_dim + 65 * self.num_classes + 42304

    @property
    def num_clients(self) -> int:
        return len(self.client_data)

    @property
    def total_train_samples(self) -> int:
        return sum(len(Xc) for Xc, _ in self.client_data)

    def to_manifest_dict(self) -> Dict[str, Any]:
        """Produce machine-readable provenance dictionary for scientific reproducibility."""
        client_summary = []
        for cid, (Xc, yc) in enumerate(self.client_data):
            unique, counts = np.unique(yc, return_counts=True)
            dist = {int(k): int(v) for k, v in zip(unique, counts)}
            client_summary.append({
                "client_id": cid,
                "num_samples": len(Xc),
                "class_distribution": dist,
            })

        val_unique, val_counts = np.unique(self.y_val, return_counts=True)
        test_unique, test_counts = np.unique(self.y_test, return_counts=True)

        val_hash = hashlib.sha256(self.X_val.tobytes() + self.y_val.tobytes()).hexdigest()[:16]
        test_hash = hashlib.sha256(self.X_test.tobytes() + self.y_test.tobytes()).hexdigest()[:16]

        # Distinguish synthetic fixture validation from real-data execution per Part F/I
        is_synthetic = self.manifest.get("is_synthetic_fixture", True if self.name in ("ciciot2023", "edgeiiotset") else False)
        execution_type = self.manifest.get("execution_type", "synthetic_fixture" if is_synthetic else "real_data")
        raw_data_status = self.manifest.get("raw_data_status", "blocked_missing_raw_files" if is_synthetic else "verified_real_data")

        return {
            "dataset_name": self.name,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "expected_parameter_count": self.expected_parameter_count,
            "is_synthetic_fixture": is_synthetic,
            "execution_type": execution_type,
            "raw_data_status": raw_data_status,
            "val_samples": len(self.X_val),
            "val_distribution": {int(k): int(v) for k, v in zip(val_unique, val_counts)},
            "val_hash": val_hash,
            "tune_samples": len(self.X_tune) if self.X_tune is not None else 0,
            "test_samples": len(self.X_test),
            "test_distribution": {int(k): int(v) for k, v in zip(test_unique, test_counts)},
            "test_hash": test_hash,
            "num_clients": len(self.client_data),
            "clients": client_summary,
            "timestamp": datetime.utcnow().isoformat(),
            **self.manifest,
        }

    def save_manifest(self, filepath: str) -> None:
        """Write manifest JSON to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_manifest_dict(), f, indent=2)


def canonical_parameter_count(d: int, K: int) -> int:
    """Canonical model parameter count formula: P(d, K) = 256d + 65K + 42,304."""
    return 256 * d + 65 * K + 42304


canonical_param_count = canonical_parameter_count


def verify_parameter_count(dataset_name: str, model: Any) -> Dict[str, Any]:
    """
    Verify model parameter count against canonical closed-form formula P(d, K) = 256d + 65K + 42,304.
    """
    dataset_key = dataset_name.lower().replace("-", "").replace("_", "")
    if dataset_key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    spec = DATASET_SPECS[dataset_key]
    expected = spec["param_count"]
    actual = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else None
    return {
        "dataset": dataset_name,
        "d": spec["d"],
        "K": spec["K"],
        "expected_params": expected,
        "actual_params": actual,
        "matches": (actual == expected),
    }


def generate_dataset_manifest(
    dataset_name: str,
    raw_files: List[str],
    preprocessing_config: Dict[str, Any],
    num_clients: int,
    partition_type: str,
    alpha: Optional[float],
    seed: int,
    is_synthetic_fixture: bool = False,
    execution_type: str = "real_data",
    raw_data_status: str = "verified_real_data",
) -> Dict[str, Any]:
    """
    Generate machine-readable provenance manifest dictionary.
    Distinguishes target configuration, synthetic-fixture validation, and real-data execution.
    """
    dataset_key = dataset_name.lower().replace("-", "").replace("_", "")
    spec = DATASET_SPECS.get(dataset_key, {})
    return {
        "dataset_name": dataset_name,
        "input_dim": spec.get("d"),
        "num_classes": spec.get("K"),
        "param_count": spec.get("param_count"),
        "is_synthetic_fixture": is_synthetic_fixture,
        "execution_type": execution_type,
        "raw_data_status": raw_data_status,
        "raw_files": raw_files,
        "preprocessing_config": preprocessing_config,
        "partition": {
            "num_clients": num_clients,
            "partition_type": partition_type,
            "alpha": alpha,
            "seed": seed,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


def load_dataset(
    dataset_name: str,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    seed: int = 42,
    partition_type: str = "noniid",
    alpha: float = 0.5,
    num_clients: int = 20,
    val_size: int = 2000,
    protocol: str = "main",
    return_tune: bool = True,
) -> DatasetBundle:
    """
    Factory function loading any of the three benchmark datasets and producing a DatasetBundle.
    """
    dataset_key = dataset_name.lower().replace("-", "").replace("_", "")

    if dataset_key in ("nslkdd", "kdd"):
        from data.preprocessing.nslkdd_pipeline import (
            build_pipeline, CLASS_NAMES, FEATURE_COLUMNS, PAPER_VAL_QUOTAS, apply_smote
        )
        tr_path = train_path or "data/raw/KDDTrain+.txt"
        te_path = test_path or "data/raw/KDDTest+.txt"
        d = 41
        K = 5
        class_names = CLASS_NAMES
        val_quotas = PAPER_VAL_QUOTAS.copy()

        (
            X_client, y_client,
            X_val, y_val,
            X_tune, y_tune,
            X_test, y_test,
            scaler, encoders, weights,
        ) = build_pipeline(
            tr_path, te_path, seed=seed, val_size=val_size,
            protocol=protocol, return_tune=True
        )

    elif dataset_key in ("ciciot2023", "ciciot"):
        from data.preprocessing.ciciot2023_pipeline import (
            build_pipeline, CLASS_NAMES, FEATURE_COLUMNS, PAPER_VAL_QUOTAS, apply_smote
        )
        tr_path = train_path or "data/raw/CICIoT2023_train.csv"
        te_path = test_path or "data/raw/CICIoT2023_test.csv"
        d = 46
        K = 8
        class_names = CLASS_NAMES
        val_quotas = PAPER_VAL_QUOTAS.copy()

        (
            X_client, y_client,
            X_val, y_val,
            X_tune, y_tune,
            X_test, y_test,
            scaler, encoders, weights,
        ) = build_pipeline(
            tr_path, te_path, seed=seed, val_size=val_size,
            protocol=protocol, return_tune=True
        )

    elif dataset_key in ("edgeiiotset", "edgeiiot", "edge"):
        from data.preprocessing.edgeiiotset_pipeline import (
            build_pipeline, CLASS_NAMES, FEATURE_COLUMNS, PAPER_VAL_QUOTAS,
            compute_edgeiiotset_quotas, apply_smote
        )
        tr_path = train_path or "data/raw/EdgeIIoTset_train.csv"
        te_path = test_path or "data/raw/EdgeIIoTset_test.csv"
        d = 61
        K = 6
        class_names = CLASS_NAMES

        (
            X_client, y_client,
            X_val, y_val,
            X_tune, y_tune,
            X_test, y_test,
            scaler, encoders, weights,
        ) = build_pipeline(
            tr_path, te_path, seed=seed, val_size=val_size,
            protocol=protocol, return_tune=True
        )
        if val_size == 2000:
            val_quotas = PAPER_VAL_QUOTAS.copy()
        else:
            val_unique, val_counts = np.unique(y_val, return_counts=True)
            val_quotas = {int(k): int(v) for k, v in zip(val_unique, val_counts)}

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from 'nslkdd', 'ciciot2023', 'edgeiiotset'."
        )

    # Client Partitioning
    partitioner = get_partitioner(partition_type, alpha=alpha)
    client_data = partitioner.partition(X_client, y_client, num_clients, seed=seed)

    # Client-local SMOTE applied after partitioning
    client_data = [
        apply_smote(Xc, yc, random_state=seed + cid)
        for cid, (Xc, yc) in enumerate(client_data)
    ]

    manifest = {
        "partition_type": partition_type,
        "alpha": alpha if partition_type == "noniid" else None,
        "protocol": protocol,
        "seed": seed,
    }

    bundle = DatasetBundle(
        name=dataset_name,
        input_dim=d,
        num_classes=K,
        class_names=class_names,
        client_data=client_data,
        X_val=X_val,
        y_val=y_val,
        X_tune=X_tune if return_tune else None,
        y_tune=y_tune if return_tune else None,
        X_test=X_test,
        y_test=y_test,
        class_weights=weights,
        val_quotas=val_quotas,
        manifest=manifest,
    )

    return bundle
