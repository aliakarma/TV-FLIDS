"""
evaluation/holm.py
Holm-Bonferroni step-down family-wise error rate control.
Reference: IEEE TIFS Manuscript §VI-E line 381, Supplementary §S8.

Guarantees:
  - Adjusts across the complete family of primary comparisons (m = 84 = 14 baselines x 3 datasets x 2 endpoints).
  - Also supports secondary families: ablations (m = 48), RQ4 worst case (m = 4), RQ6 (m = 4).
  - Monotone step-down adjustment:
      p'_(1) = min(1, m * p_(1))
      p'_(k) = max(p'_(k-1), min(1, (m - k + 1) * p_(k)))
  - Deterministic sorting and rank preservation.
  - Preserves comparison identifiers and maps adjusted p-values back to original inputs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class HolmAdjustmentItem:
    """Individual item in a Holm-adjusted family."""
    identifier: Any
    raw_p: float
    holm_p: float
    rank: int                          # 1-indexed rank in sorted order
    multiplier: int                    # (m - rank + 1)
    significant: bool                  # holm_p < alpha


def holm_bonferroni_adjust(
    p_values: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """
    Compute Holm-Bonferroni adjusted p-values for an array of p-values.

    Formula:
        p'_(k) = max_{j <= k} min(1.0, (m - j + 1) * p_(j))
    where p_(1) <= p_(2) <= ... <= p_(m).

    Returns:
        np.ndarray of adjusted p-values in the original order.
    """
    ps = np.asarray(p_values, dtype=float)
    m = len(ps)
    if m == 0:
        return np.array([], dtype=float)
    if m == 1:
        return np.clip(ps, 0.0, 1.0)

    # Sort indices ascending
    sort_idx = np.argsort(ps)
    adj = np.empty(m, dtype=float)

    running_max = 0.0
    for i, idx in enumerate(sort_idx):
        k = i + 1                     # 1-based rank
        multiplier = m - k + 1
        candidate = min(1.0, multiplier * ps[idx])
        running_max = max(running_max, candidate)
        adj[idx] = running_max

    return adj


def adjust_family(
    items: List[Dict[str, Any]],
    p_key: str = "p_value",
    out_key: str = "p_holm",
    sig_key: str = "significant_holm",
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Apply Holm-Bonferroni adjustment to a list of dicts representing comparisons.

    Args:
        items: List of comparison dicts, each containing p_key.
        p_key: Key in each dict containing raw p-value.
        out_key: Key to write adjusted p-value into.
        sig_key: Key to write boolean significance decision into.
        alpha: Significance threshold (default 0.05).

    Returns:
        The updated list of dicts (modified in place and returned).
    """
    if not items:
        return items

    raw_ps = [float(item[p_key]) for item in items]
    adj_ps = holm_bonferroni_adjust(raw_ps)

    for item, adj in zip(items, adj_ps):
        item[out_key] = float(adj)
        item[sig_key] = bool(adj < alpha)

    return items
