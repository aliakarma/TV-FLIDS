"""
evaluation/bootstrap_ci.py
Bias-Corrected and Accelerated (BCa) bootstrap confidence intervals for paired differences.
Reference: IEEE TIFS Manuscript §VI-E line 381, Supplementary §S8.

Guarantees:
  - Paired resampling: resamples paired differences directly, preserving pairing structure.
  - Statistic: sample median of paired differences.
  - Confidence level: 0.95 (95% two-sided interval).
  - Replicate count: 10,000 bootstrap resamples.
  - Seed handling: deterministic random generator with support for seed integer or string key.
  - Degenerate handling:
      * Zero spread (ptp < 1e-12) -> returns [d_0, d_0].
      * Degenerate acceleration / non-finite BCa bounds -> graceful fallback to percentile bootstrap.
  - Never independently bootstraps two methods and subtracts their intervals.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple, Union
import zlib

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class BootstrapCIResult:
    """Result of a bootstrap confidence interval calculation."""
    estimate: float                   # Sample median
    ci_low: float                     # Lower confidence bound
    ci_high: float                    # Upper confidence bound
    confidence_level: float           # 0.95
    n_resamples: int                  # 10,000
    method_used: str                  # "BCa", "percentile_fallback", or "degenerate_constant"
    seed: int


def derive_rng(seed_or_tag: Union[int, str, None] = None) -> np.random.Generator:
    """Derive deterministic numpy Generator from integer seed or string tag."""
    if seed_or_tag is None:
        return np.random.default_rng(20260914)
    if isinstance(seed_or_tag, int):
        return np.random.default_rng(seed_or_tag)
    # String tag -> hash to seed
    h = zlib.crc32(str(seed_or_tag).encode("utf-8"))
    return np.random.default_rng([20260914, h])


def compute_paired_bca_ci(
    differences: Union[np.ndarray, list],
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed_or_tag: Union[int, str, None] = None,
) -> BootstrapCIResult:
    """
    Compute 95% BCa bootstrap confidence interval for the median of paired differences.

    Args:
        differences: Array-like of paired differences (e.g. 100 * (TV-FLIDS - baseline)).
        confidence_level: Confidence level (default 0.95).
        n_resamples: Number of bootstrap resamples (default 10,000).
        seed_or_tag: Seed integer or string tag for reproducible randomness.

    Returns:
        BootstrapCIResult with median estimate and [ci_low, ci_high].
    """
    d = np.asarray(differences, dtype=float).flatten()
    n = len(d)
    if n == 0:
        return BootstrapCIResult(
            estimate=0.0,
            ci_low=0.0,
            ci_high=0.0,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method_used="degenerate_empty",
            seed=0,
        )

    median_val = float(np.median(d))

    if n == 1 or np.ptp(d) < 1e-12:
        return BootstrapCIResult(
            estimate=median_val,
            ci_low=median_val,
            ci_high=median_val,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method_used="degenerate_constant",
            seed=0,
        )

    rng = derive_rng(seed_or_tag)
    method_used = "BCa"
    lo = float("nan")
    hi = float("nan")

    try:
        res = stats.bootstrap(
            (d,),
            np.median,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method="BCa",
            random_state=rng,
        )
        lo = float(res.confidence_interval.low)
        hi = float(res.confidence_interval.high)
    except Exception:
        # BCa may fail on extreme ties or zero acceleration
        lo = float("nan")
        hi = float("nan")

    # Degenerate BCa acceleration / non-finite bounds -> percentile fallback
    if not (math.isfinite(lo) and math.isfinite(hi)):
        rng_fallback = derive_rng(seed_or_tag)
        try:
            res_p = stats.bootstrap(
                (d,),
                np.median,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                method="percentile",
                random_state=rng_fallback,
            )
            lo = float(res_p.confidence_interval.low)
            hi = float(res_p.confidence_interval.high)
            method_used = "percentile_fallback"
        except Exception:
            lo = median_val
            hi = median_val
            method_used = "fallback_median"

    seed_val = seed_or_tag if isinstance(seed_or_tag, int) else 0
    return BootstrapCIResult(
        estimate=median_val,
        ci_low=lo,
        ci_high=hi,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method_used=method_used,
        seed=seed_val,
    )
