"""
campaign/run_spec.py
Deterministic run specification and scientific identity generation.
Reference: IEEE TIFS Manuscript §VI, §VII, and Supplementary §S3–S8.

Guarantees:
  - Every experiment is defined by an immutable, canonical scientific configuration.
  - Run ID is a deterministic function of the complete scientific specification.
  - Scientific identity is invariant to parameter serialization ordering and cross-process invocation.
  - Scientific identity excludes wall-clock time, hardware characteristics, and execution order.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RunSpecification:
    """
    Complete, immutable specification of a single experimental run.
    Contains every parameter capable of altering the scientific outcome.
    """
    # Campaign Block
    block: str                                     # "B1" through "B11"
    purpose: str                                   # Description of purpose

    # Dataset & Partitioning
    dataset: str                                   # "nslkdd", "ciciot2023", "edgeiiotset"
    protocol: str = "main"                         # "main", "leakage_free", "smote_per_client_after", etc.
    partition_type: str = "noniid"                 # "noniid", "iid"
    alpha: float = 0.5                             # Dirichlet concentration parameter
    val_size: int = 2000                           # Validation set size |D_val|
    num_clients: int = 20                          # Total client count N
    fraction_fit: float = 0.5                      # Client participation fraction rho = D/N
    fraction_evaluate: float = 0.3                 # Client evaluation fraction

    # Strategy & Defense Hyperparameters
    strategy: str = "tvflids"                      # Strategy name (15 methods)
    strategy_params: Dict[str, Any] = field(default_factory=dict)  # Tuning/defense overrides

    # Adversarial Threat Model
    attack: str = "label_flip_30"                  # Attack name or configuration key
    attack_params: Dict[str, Any] = field(default_factory=dict)    # Attack parameters
    attack_variant: Optional[str] = None           # "partial", "omniscient"
    knowledge_tier: Optional[str] = None           # "K0", "K1", "K2"
    attack_strength: Optional[float] = None        # Scale factor, noise std, etc.
    on_off_k: Optional[int] = None                 # Honest phase length k in {10, 20, 30, 50}

    # Model & Local Training
    model_type: str = "mlp"                        # "mlp"
    num_rounds: int = 100                          # Federated rounds
    local_epochs: int = 5                          # Local training epochs E
    local_lr: float = 0.001                        # Local learning rate
    local_batch_size: int = 256                    # Local batch size

    # Random Seed
    seed: int = 42                                 # Global simulation seed

    # Special Block Flags
    dp_noise_multiplier: Optional[float] = None    # Differential privacy noise (B10)
    is_profiling: bool = False                     # Profiling run flag (B10)
    profiling_d: Optional[int] = None              # Participants D for profiling
    profiling_n: Optional[int] = None              # Clients N for profiling

    def to_scientific_dict(self) -> Dict[str, Any]:
        """
        Convert to a normalized dictionary containing all scientific parameters.
        Keys and nested dictionaries are strictly sorted for deterministic serialization.
        """
        data = {
            "block": self.block,
            "dataset": self.dataset,
            "protocol": self.protocol,
            "partition_type": self.partition_type,
            "alpha": float(self.alpha),
            "val_size": int(self.val_size),
            "num_clients": int(self.num_clients),
            "fraction_fit": float(self.fraction_fit),
            "fraction_evaluate": float(self.fraction_evaluate),
            "strategy": self.strategy,
            "strategy_params": _normalize_dict(self.strategy_params),
            "attack": self.attack,
            "attack_params": _normalize_dict(self.attack_params),
            "attack_variant": self.attack_variant,
            "knowledge_tier": self.knowledge_tier,
            "attack_strength": float(self.attack_strength) if self.attack_strength is not None else None,
            "on_off_k": int(self.on_off_k) if self.on_off_k is not None else None,
            "model_type": self.model_type,
            "num_rounds": int(self.num_rounds),
            "local_epochs": int(self.local_epochs),
            "local_lr": float(self.local_lr),
            "local_batch_size": int(self.local_batch_size),
            "seed": int(self.seed),
            "dp_noise_multiplier": float(self.dp_noise_multiplier) if self.dp_noise_multiplier is not None else None,
            "is_profiling": bool(self.is_profiling),
            "profiling_d": int(self.profiling_d) if self.profiling_d is not None else None,
            "profiling_n": int(self.profiling_n) if self.profiling_n is not None else None,
        }
        return data

    def to_dict(self) -> Dict[str, Any]:
        """Convert RunSpecification to a dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunSpecification:
        """Construct RunSpecification from dictionary, filtering unknown keys."""
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @property
    def run_id(self) -> str:
        """Deterministic run ID computed from the scientific configuration."""
        return compute_run_id(self)


def _normalize_dict(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively sort dictionary keys and normalize numeric types."""
    if not d:
        return {}
    normalized = {}
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, dict):
            normalized[k] = _normalize_dict(v)
        elif isinstance(v, list):
            normalized[k] = [_normalize_dict(item) if isinstance(item, dict) else item for item in v]
        elif isinstance(v, (int, float, str, bool)) or v is None:
            normalized[k] = v
        else:
            normalized[k] = str(v)
    return normalized


def compute_run_id(spec: RunSpecification) -> str:
    """
    Compute a deterministic 16-character hexadecimal run identifier from a RunSpecification.
    Two runs with identical scientific configurations produce the same ID.
    Any change in scientific parameters produces a distinct ID.
    """
    scientific_dict = spec.to_scientific_dict()
    canonical_json = json.dumps(scientific_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    hash_digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return f"run_{hash_digest[:16]}"
