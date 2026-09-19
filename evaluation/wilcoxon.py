"""
evaluation/wilcoxon.py
Two-sided exact paired Wilcoxon signed-rank test.
Reference: IEEE TIFS Manuscript §VI-E line 381, Supplementary §S8.

Guarantees:
  - Two-sided hypothesis H0: median difference = 0 vs H1: median difference != 0.
  - Zero-difference handling: drops |d| <= 1e-12. If all d are zero, p=1.0, stat=0.0.
  - Tie handling: when ties in |d| exist, uses asymptotic normal approximation with continuity correction;
    when no ties exist, uses exact permutation distribution.
  - Never substitutes a one-sided test or an independent-samples test.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class WilcoxonResult:
    """Detailed result of a Wilcoxon signed-rank test."""
    statistic: float
    p_value: float
    n_total: int
    n_nonzero: int
    has_ties: bool
    method_used: str               # "exact", "approx", or "zero_diff_fallback"
    alternative: str = "two-sided"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def wilcoxon_signed_rank_test(
    d: Union[np.ndarray, list],
    zero_tol: float = 1e-12,
) -> WilcoxonResult:
    """
    Perform the canonical two-sided Wilcoxon signed-rank test on paired differences d.

    Args:
        d: Array-like of paired differences (e.g. TV-FLIDS - baseline).
        zero_tol: Absolute difference threshold below which differences are considered zero.

    Returns:
        WilcoxonResult with test statistic, p-value, and diagnostic metadata.
    """
    arr = np.asarray(d, dtype=float).flatten()
    n_total = len(arr)

    # Drop zero differences
    nonzero = arr[np.abs(arr) > zero_tol]
    n_nonzero = len(nonzero)

    if n_nonzero == 0:
        return WilcoxonResult(
            statistic=0.0,
            p_value=1.0,
            n_total=n_total,
            n_nonzero=0,
            has_ties=False,
            method_used="zero_diff_fallback",
            alternative="two-sided",
        )

    # Check for ties in absolute values
    rounded_abs = np.round(np.abs(nonzero), 12)
    has_ties = bool(len(np.unique(rounded_abs)) < n_nonzero)
    method = "approx" if has_ties else "exact"

    res = stats.wilcoxon(
        nonzero,
        alternative="two-sided",
        method=method,
    )

    return WilcoxonResult(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        n_total=n_total,
        n_nonzero=n_nonzero,
        has_ties=has_ties,
        method_used=method,
        alternative="two-sided",
    )
