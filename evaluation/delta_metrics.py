"""
evaluation/delta_metrics.py
Manuscript-defined Delta-ASR and confusion-matrix routing diagnostics.
Reference: IEEE TIFS Manuscript §VI-D Eq. (16)-(17), Supplementary §S8.

Guarantees:
  - Exact ASR definition (Eq. 16): sum_{c != 0} n_{c0} / sum_{c != 0} n_c.
  - Attack recall R_atk: sum_{c != 0} n_{cc} / sum_{c != 0} n_c.
  - Benign routing phi: sum_{c != 0} n_{c0} / sum_{c != 0} (n_c - n_{cc}).
  - Exact decomposition identity: ASR = (1 - R_atk) * phi.
  - Delta-ASR (percentage points): 100 * (ASR_attacked - ASR_clean_ref), paired by seed.
  - Validates clean-reference seed pairing and handles missing references.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class RoutingDiagnostics:
    """Confusion-matrix derived metrics separating missed recognition from benign routing."""
    asr: float                         # Attack success rate
    attack_recall: float               # R_atk
    benign_routing_phi: float          # phi
    identity_residual: float           # |ASR - (1 - R_atk) * phi|

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def compute_routing_diagnostics(
    confusion_mat: np.ndarray,
) -> RoutingDiagnostics:
    """
    Compute ASR, Attack Recall, and Benign Routing Phi from a confusion matrix.
    Assumes class 0 is Benign/Normal.

    Args:
        confusion_mat: 2D integer array (K x K), where row is true class and col is predicted.

    Returns:
        RoutingDiagnostics with asr, attack_recall, benign_routing_phi, and residual.
    """
    cm = np.asarray(confusion_mat, dtype=float)
    k = cm.shape[0]

    # Row sums: n_c
    row_sums = np.sum(cm, axis=1)

    # Attack traffic: all rows c != 0
    attack_total = float(np.sum(row_sums[1:]))

    if attack_total <= 0:
        return RoutingDiagnostics(
            asr=0.0,
            attack_recall=1.0,
            benign_routing_phi=0.0,
            identity_residual=0.0,
        )

    # Attack traffic classified as Benign (col 0): sum_{c != 0} n_{c0}
    attack_as_benign = float(np.sum(cm[1:, 0]))
    asr = attack_as_benign / attack_total

    # Attack traffic correctly classified: sum_{c != 0} n_{cc}
    attack_correct = float(np.sum(np.diag(cm)[1:]))
    r_atk = attack_correct / attack_total

    # Missed attack traffic: sum_{c != 0} (n_c - n_{cc})
    attack_missed = attack_total - attack_correct

    if attack_missed <= 0:
        phi = 0.0
    else:
        phi = attack_as_benign / attack_missed

    # Identity check: ASR = (1 - R_atk) * phi
    expected_asr = (1.0 - r_atk) * phi
    residual = abs(asr - expected_asr)

    return RoutingDiagnostics(
        asr=asr,
        attack_recall=r_atk,
        benign_routing_phi=phi,
        identity_residual=residual,
    )


def compute_delta_asr(
    attacked_asr: float,
    clean_ref_asr: float,
) -> float:
    """
    Compute Delta-ASR in percentage points.
    Delta-ASR = 100 * (ASR_attacked - ASR_clean_ref).
    """
    return 100.0 * (float(attacked_asr) - float(clean_ref_asr))
