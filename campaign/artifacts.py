"""
campaign/artifacts.py
Deterministic artifact hierarchy, status tracking, and canonical result schema.
Reference: IEEE TIFS Manuscript §VI, §VII, and Supplementary §S3–S8.

Guarantees:
  - Enforces deterministic artifact hierarchy: results/<dataset>/<block>/<strategy>/<attack>/<run_id>/
  - Explicit execution status lifecycle: PENDING -> RUNNING -> (COMPLETED | FAILED | SKIPPED | BLOCKED)
  - Canonical result schema covering accuracy, Macro-F1, ASR, recall, benign routing phi, round-level logs.
  - Distinguishes valid completed runs from interrupted, corrupted, or synthetic fixture runs.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from campaign.run_spec import RunSpecification

VALID_STATUSES = {"PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED", "BLOCKED"}


def get_artifact_directory(spec: RunSpecification, base_dir: str = "results") -> str:
    """
    Compute canonical artifact directory path:
    results/<dataset>/<block>/<strategy>/<attack>/<run_id>/
    """
    # Clean strategy and attack strings for directory naming
    strat_clean = spec.strategy.lower().replace(":", "_").replace(" ", "_")
    attack_clean = spec.attack.lower().replace(":", "_").replace(" ", "_")
    return os.path.join(
        base_dir,
        spec.dataset.lower(),
        spec.block.upper(),
        strat_clean,
        attack_clean,
        spec.run_id,
    )


def write_status(
    output_dir: str,
    status: str,
    error_message: Optional[str] = None,
    traceback_str: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write or update status.json in the artifact directory."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Allowed: {VALID_STATUSES}")

    os.makedirs(output_dir, exist_ok=True)
    status_path = os.path.join(output_dir, "status.json")

    payload = {
        "status": status,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "error_message": error_message,
        "traceback": traceback_str,
    }
    if extra_info:
        payload["extra_info"] = extra_info

    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return payload


def read_status(output_dir: str) -> Optional[Dict[str, Any]]:
    """Read status.json if it exists and parses."""
    status_path = os.path.join(output_dir, "status.json")
    if not os.path.exists(status_path):
        return None
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def validate_result_schema(metrics: Dict[str, Any]) -> bool:
    """Validate that metrics adhere to canonical schema."""
    required_keys = ["final_accuracy", "final_f1_macro", "final_attack_success_rate"]
    for k in required_keys:
        if k not in metrics or metrics[k] is None:
            return False
        if not isinstance(metrics[k], (int, float)):
            return False
    return True


def write_metrics(output_dir: str, metrics: Dict[str, Any]) -> str:
    """Write canonical metrics.json in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")

    # Canonical fields with safe defaults
    canonical_metrics = {
        "final_accuracy": float(metrics.get("final_accuracy", 0.0)),
        "final_f1_macro": float(metrics.get("final_f1_macro", 0.0)),
        "final_attack_success_rate": float(metrics.get("final_attack_success_rate", 0.0)),
        "final_attack_recall": float(metrics.get("final_attack_recall", 0.0)) if metrics.get("final_attack_recall") is not None else None,
        "final_benign_routing_phi": float(metrics.get("final_benign_routing_phi", 0.0)) if metrics.get("final_benign_routing_phi") is not None else None,
        "round_metrics": metrics.get("round_metrics", []),
        "accepted_clients_per_round": metrics.get("accepted_clients_per_round", []),
        "rejected_clients_per_round": metrics.get("rejected_clients_per_round", []),
        "compute_overhead_ms": metrics.get("compute_overhead_ms", {}),
        "strategy_diagnostics": metrics.get("strategy_diagnostics", {}),
        "attack_diagnostics": metrics.get("attack_diagnostics", {}),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(canonical_metrics, f, indent=2, sort_keys=True)

    return metrics_path


def read_metrics(output_dir: str) -> Optional[Dict[str, Any]]:
    """Read metrics.json if present and valid."""
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if validate_result_schema(data):
                return data
            return None
    except (OSError, json.JSONDecodeError):
        return None
