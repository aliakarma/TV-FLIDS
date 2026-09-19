"""
campaign/config_freezer.py
Pre-execution configuration resolution and freezing.
Reference: IEEE TIFS Manuscript §VI, §VII, and Supplementary §S3–S8.

Guarantees:
  - Generates an immutable, fully resolved configuration artifact before execution starts.
  - Records exact Git commit provenance and detects uncommitted working tree modifications.
  - Refuses to execute on a dirty Git working tree unless an explicit override is supplied.
  - Records software runtime environment versions and hardware characteristics.
  - Experiments operate exclusively from the resolved configuration, preventing reliance on mutable defaults.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import hashlib
from campaign.run_spec import RunSpecification
from utils.provenance import git_state, package_versions, hardware
from models.mlp import IDSMLP


def resolve_dataset_provenance(dataset: str) -> Dict[str, Any]:
    """
    Resolve dataset provenance including version, raw file hashes, and protocol.
    Designed for extensibility with dataset version, raw-data hash, preprocessing
    manifest hash, and dataset configuration hash.
    """
    if dataset == "nslkdd":
        train_path = os.path.join(ROOT, "data", "raw", "KDDTrain+.txt")
        test_path = os.path.join(ROOT, "data", "raw", "KDDTest+.txt")

        raw_files: Dict[str, Any] = {}
        if os.path.exists(train_path):
            with open(train_path, "rb") as f:
                raw_files["train_hash"] = hashlib.sha256(f.read()).hexdigest()
            raw_files["train_size"] = os.path.getsize(train_path)
            raw_files["train_path"] = "data/raw/KDDTrain+.txt"

        if os.path.exists(test_path):
            with open(test_path, "rb") as f:
                raw_files["test_hash"] = hashlib.sha256(f.read()).hexdigest()
            raw_files["test_size"] = os.path.getsize(test_path)
            raw_files["test_path"] = "data/raw/KDDTest+.txt"

        return {
            "dataset_name": "nslkdd",
            "dataset_version": "standard_nslkdd_125973_22544",
            "protocol": "ieee_tifs_table1",
            "raw_files": raw_files,
            "status": "AVAILABLE" if ("train_hash" in raw_files and "test_hash" in raw_files) else "MISSING_RAW_FILES",
        }

    elif dataset == "ciciot2023":
        return {
            "dataset_name": "ciciot2023",
            "dataset_version": "ciciot2023_8class_table_s5",
            "protocol": "time_disjoint_80_20_guard_band",
            "status": "BLOCKED",
            "reason": "Raw multi-GB shards for CIC-IoT-2023 are not present on disk (Stage 8 BLOCKED state).",
        }

    elif dataset == "edgeiiotset":
        return {
            "dataset_name": "edgeiiotset",
            "dataset_version": "edgeiiotset_6class_table_s5",
            "protocol": "time_disjoint_80_20_guard_band",
            "status": "BLOCKED",
            "reason": "Raw multi-GB files for Edge-IIoTset are not present on disk (Stage 8 BLOCKED state).",
        }

    return {
        "dataset_name": dataset,
        "status": "UNKNOWN",
    }


def resolve_model_configuration(dataset: str) -> Dict[str, Any]:
    """Resolve exact model parameters and theoretical closed-form parameter count."""
    if dataset == "nslkdd":
        d, k = 41, 5
    elif dataset == "ciciot2023":
        d, k = 46, 8
    elif dataset == "edgeiiotset":
        d, k = 61, 6
    else:
        d, k = 41, 5

    expected_params = IDSMLP.parameter_count_formula(d, k)
    return {
        "architecture": "4-layer MLP: d -> 256 -> 128 -> 64 -> K",
        "input_dim": d,
        "num_classes": k,
        "hidden_layers": [256, 128, 64],
        "normalization": "nn.LayerNorm",
        "activation": "nn.ReLU",
        "dropout": 0.3,
        "parameter_count": expected_params,
    }


def freeze_configuration(
    spec: RunSpecification,
    output_dir: str,
    allow_dirty: bool = False,
    extra_metadata: Optional[Dict[str, Any]] = None,
    git_info: Optional[Dict[str, Any]] = None,
    dataset_prov: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve and freeze all scientific and provenance parameters for a run.
    Writes `config.json` inside output_dir.

    Raises:
        RuntimeError: If the Git working tree is dirty and allow_dirty is False.
    """
    os.makedirs(output_dir, exist_ok=True)
    if git_info is None:
        git_info = git_state()

    # Dirty tree protection
    if git_info.get("git_dirty", False) and not allow_dirty:
        dirty_files = git_info.get("git_dirty_paths", [])
        raise RuntimeError(
            f"Refusing to execute experiment {spec.run_id}: working tree has "
            f"{len(dirty_files)} uncommitted changes. Commit all changes or supply "
            f"allow_dirty=True override.\nDirty paths: {dirty_files[:10]}"
        )

    scientific_config = spec.to_scientific_dict()
    model_config = resolve_model_configuration(spec.dataset)
    if dataset_prov is None:
        dataset_prov = resolve_dataset_provenance(spec.dataset)

    frozen_config = {
        "run_id": spec.run_id,
        "block": spec.block,
        "purpose": spec.purpose,
        "scientific_configuration": scientific_config,
        "model_configuration": model_config,
        "dataset_provenance": dataset_prov,
        "git_provenance": git_info,
        "environment": {
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": package_versions(),
            "hardware": hardware(),
        },
        "extra_metadata": extra_metadata or {},
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(frozen_config, f, indent=2, sort_keys=True)

    return frozen_config
