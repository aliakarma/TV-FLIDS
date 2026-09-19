"""
evaluation/wilcoxon.py
Two-sided exact paired Wilcoxon signed-rank test supporting ties via exact null distribution.
Reference: IEEE TIFS Manuscript §VI-E line 381, Supplementary §S8.

Guarantees:
  - Two-sided hypothesis H0: median difference = 0 vs H1: median difference != 0.
  - Zero-difference handling: drops |d| <= 1e-12. If all d are zero, p=1.0, stat=0.0.
  - Exact tie handling: when ties in |d| exist, ranks are assigned via the standard average-rank
    rule. The exact null distribution under the observed tied ranks is computed via dynamic
    programming over all 2^N equally likely sign assignments.
  - Never falls back to an asymptotic or normal approximation.
  - Numerical precision policy: |d| <= 1e-12 is treated as zero to prevent floating-point
    cancellation residuals from introducing spurious ranks.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class WilcoxonResult:
    """Detailed result of a Wilcoxon signed-rank test."""
    statistic: float
    p_value: float
    n_total: int
    n_nonzero: int
    has_ties: bool
    method_used: str               # Always "exact" (or "exact_zero_fallback")
    alternative: str = "two-sided"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def wilcoxon_signed_rank_test(
    d: Union[np.ndarray, list],
    zero_tol: float = 1e-12,
) -> WilcoxonResult:
    """
    Perform the canonical two-sided exact Wilcoxon signed-rank test on paired differences d.

    Algorithm:
      1. Filter out differences with |d_i| <= zero_tol.
      2. If N == 0: return stat=0.0, p=1.0.
      3. Compute tied ranks R_i on |d_i| using the average-rank rule.
         By mathematical theorem, for any tie group of size k starting at rank a,
         the average rank is a + (k - 1) / 2, so 2 * R_i is ALWAYS an exact integer.
      4. Compute W+ = sum_{d_i > 0} R_i and W- = sum_{d_i < 0} R_i.
         Test statistic is W_obs = min(W+, W-).
      5. Compute the exact distribution of w+ = sum_{i=1}^N B_i R_i under H0
         (where B_i in {0, 1} i.i.d. Bernoulli(1/2)) via dynamic programming over 2 * R_i.
      6. Two-sided exact p-value = min(1.0, 2 * Pr(w+ <= W_obs)). If W_obs >= Total/2, p = 1.0.

    Args:
        d: Array-like of paired differences (e.g. TV-FLIDS - baseline).
        zero_tol: Absolute difference threshold below which differences are considered zero.

    Returns:
        WilcoxonResult with test statistic, exact p-value, and diagnostic metadata.
    """
    arr = np.asarray(d, dtype=float).flatten()
    n_total = len(arr)

    # 1. Drop zero differences below precision threshold
    nonzero = arr[np.abs(arr) > zero_tol]
    n_nonzero = len(nonzero)

    if n_nonzero == 0:
        return WilcoxonResult(
            statistic=0.0,
            p_value=1.0,
            n_total=n_total,
            n_nonzero=0,
            has_ties=False,
            method_used="exact_zero_fallback",
            alternative="two-sided",
        )

    # 2. Assign average ranks for ties on absolute differences
    abs_nonzero = np.abs(nonzero)
    ranks = rankdata(abs_nonzero, method="average")

    # Check if ties occurred in absolute values
    rounded_abs = np.round(abs_nonzero, 12)
    has_ties = bool(len(np.unique(rounded_abs)) < n_nonzero)

    # 3. Compute signed-rank sums
    w_plus = float(np.sum(ranks[nonzero > 0]))
    w_minus = float(np.sum(ranks[nonzero < 0]))
    w_obs = min(w_plus, w_minus)
    total_ranks = float(np.sum(ranks))

    # 4. Exact distribution computation via Dynamic Programming
    # Since 2 * R_i is always an exact integer, we scale by 2 to work entirely with integers.
    if w_obs >= total_ranks / 2.0:
        p_val = 1.0
    else:
        # Scale ranks to exact integers
        scaled_ranks = [int(round(2.0 * r)) for r in ranks]
        total_scaled = sum(scaled_ranks)
        limit = int(round(2.0 * w_obs))

        # dp[s] = number of subsets of scaled_ranks summing to s
        dp = [0] * (total_scaled + 1)
        dp[0] = 1
        curr_sum = 0
        for m in scaled_ranks:
            curr_sum += m
            for s in range(curr_sum, m - 1, -1):
                dp[s] += dp[s - m]

        count_le = sum(dp[:limit + 1])
        total_subsets = 2 ** n_nonzero
        # By symmetry of the null distribution around total_ranks / 2:
        # Pr(w <= w_obs) = 2 * Pr(w+ <= w_obs)
        p_val = min(1.0, (2.0 * count_le) / float(total_subsets))

    return WilcoxonResult(
        statistic=w_obs,
        p_value=float(p_val),
        n_total=n_total,
        n_nonzero=n_nonzero,
        has_ties=has_ties,
        method_used="exact",
        alternative="two-sided",
    )
