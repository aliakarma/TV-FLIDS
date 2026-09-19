"""
evaluation/target_validator.py
Validation of empirical results against manuscript targets (targets.json).
Reference: IEEE TIFS Manuscript §VI, §VII, and targets.json.

Guarantees:
  - Compares actual campaign-derived statistics against manuscript declared targets ONLY when real completed data exist.
  - Distinguishes:
      * target definition exists
      * source result exists
      * comparison is complete
      * observed value
      * tolerance / decision rule
      * pass / fail status
  - Refuses to validate incomplete, blocked, or synthetic runs as real target matches.
  - Never mutates experimental results to satisfy targets.
  - Never optimizes code toward manuscript target numbers.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@dataclass(frozen=True)
class TargetValidationRecord:
    """Record of a comparison between an observed metric and a manuscript target."""
    target_key: str
    target_type: str                  # "primary_test", "table_main", "dasr", etc.
    target_exists: bool
    source_result_exists: bool
    is_complete: bool
    target_value: Any
    observed_value: Any
    tolerance: Optional[float]
    passed: Optional[bool]            # True/False if evaluated, None if data absent/incomplete
    status_message: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class TargetValidator:
    """
    Validates experimental observations against canonical manuscript targets.
    """

    DEFAULT_TARGETS_PATH = os.path.join(
        ROOT, "Paper", "IEEE", "analysis", "targets", "targets.json"
    )

    def __init__(self, targets_path: Optional[str] = None):
        self.targets_path = targets_path or self.DEFAULT_TARGETS_PATH
        self.targets: Dict[str, Any] = {}
        self._load_targets()

    def _load_targets(self) -> None:
        if os.path.exists(self.targets_path):
            try:
                with open(self.targets_path, "r", encoding="utf-8") as f:
                    self.targets = json.load(f)
            except (OSError, json.JSONDecodeError):
                self.targets = {}

    def get_primary_target(self, dataset: str, baseline: str, endpoint: str) -> Optional[Dict[str, Any]]:
        """Look up canonical primary test target in targets.json."""
        ds_map = {"nslkdd": "NSL", "ciciot2023": "CIC", "edgeiiotset": "EDGE", "nsl": "NSL", "cic": "CIC", "edge": "EDGE"}
        ds_code = ds_map.get(dataset.lower(), dataset.upper())

        # Normalize baseline name
        base_norm = baseline.replace("_", " ").title()
        if base_norm.lower() == "tvflids":
            base_norm = "TV-FLIDS"
        elif base_norm.lower() == "fedavg":
            base_norm = "FedAvg"
        elif base_norm.lower() == "multikrum":
            base_norm = "Multi-Krum"
        elif base_norm.lower() == "trimmedmean":
            base_norm = "Trimmed Mean"
        elif base_norm.lower() == "normclipping":
            base_norm = "Norm Clipping"
        elif base_norm.lower() == "foolsgold":
            base_norm = "FoolsGold"
        elif base_norm.lower() == "deepsight":
            base_norm = "DeepSight"
        elif base_norm.lower() == "fldetector":
            base_norm = "FLDetector"
        elif base_norm.lower() == "fltrust":
            base_norm = "FLTrust"
        elif base_norm.lower() == "baffle":
            base_norm = "BaFFLe"

        prim_list = self.targets.get("primary_tests", [])
        for t in prim_list:
            if t.get("ds") == ds_code and t.get("base", "").lower() == base_norm.lower() and t.get("ep") == endpoint:
                return t
        return None

    def validate_primary_test(
        self,
        observed: Dict[str, Any],
        tolerance_med_pp: float = 2.0,
    ) -> TargetValidationRecord:
        """
        Validate an observed primary comparison against the corresponding target in targets.json.

        Args:
            observed: Dict containing dataset, baseline, endpoint, status, paired_median_diff, etc.
            tolerance_med_pp: Allowed difference in percentage points between observed and target median.

        Returns:
            TargetValidationRecord.
        """
        ds = observed.get("dataset", "")
        base = observed.get("baseline", "")
        ep = observed.get("endpoint", "")
        status = observed.get("status", "INCOMPLETE")
        is_complete = (status == "COMPLETE")
        source_exists = "paired_median_diff" in observed and observed["paired_median_diff"] is not None

        target = self.get_primary_target(ds, base, ep)
        target_key = f"{ds}|{base}|{ep}"

        if not target:
            return TargetValidationRecord(
                target_key=target_key,
                target_type="primary_test",
                target_exists=False,
                source_result_exists=source_exists,
                is_complete=is_complete,
                target_value=None,
                observed_value=observed.get("paired_median_diff"),
                tolerance=tolerance_med_pp,
                passed=None,
                status_message="TARGET_NOT_FOUND: No matching target defined in targets.json",
            )

        if not is_complete or not source_exists:
            return TargetValidationRecord(
                target_key=target_key,
                target_type="primary_test",
                target_exists=True,
                source_result_exists=source_exists,
                is_complete=is_complete,
                target_value=target.get("med"),
                observed_value=observed.get("paired_median_diff"),
                tolerance=tolerance_med_pp,
                passed=None,
                status_message=f"CANNOT_VALIDATE: Data status is '{status}'. Target requires 20 completed real seeds.",
            )

        obs_med = float(observed["paired_median_diff"])
        tgt_med = float(target["med"])
        med_diff = abs(obs_med - tgt_med)

        # Decision rule agreement
        obs_claim = bool(observed.get("claim", False))
        tgt_claim = bool(target.get("claim", False))

        passed = bool(med_diff <= tolerance_med_pp and obs_claim == tgt_claim)
        msg = (
            f"PASSED (med diff: {med_diff:.2f} pp <= {tolerance_med_pp} pp; claim matches: {obs_claim})"
            if passed
            else f"FAILED (med diff: {med_diff:.2f} pp, obs_claim={obs_claim} vs tgt_claim={tgt_claim})"
        )

        return TargetValidationRecord(
            target_key=target_key,
            target_type="primary_test",
            target_exists=True,
            source_result_exists=True,
            is_complete=True,
            target_value=tgt_med,
            observed_value=obs_med,
            tolerance=tolerance_med_pp,
            passed=passed,
            status_message=msg,
        )
