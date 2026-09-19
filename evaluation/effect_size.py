"""
evaluation/effect_size.py
Paper-defined effect size statistics and favor counting.
Reference: IEEE TIFS Manuscript §VI-E, §VII, Supplementary §S8.

Guarantees:
  - Computes exact manuscript-defined effect size: paired median difference in percentage points:
      med(d) = median(100 * (TV-FLIDS - baseline))
  - Computes seed favor count:
      * Macro-F1: sum(d > 0)
      * ASR: sum(d < 0)
  - Retains comparison, endpoint, dataset, sample size, estimate, and favor ratio.
  - Does not invent unsupported qualitative labels (e.g. "large", "small").
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass(frozen=True)
class EffectSizeResult:
    """Effect size and directional favor statistics for a paired comparison."""
    dataset: str
    baseline: str
    endpoint: str                      # "f1" or "asr"
    sample_size: int                   # Number of paired seeds
    paired_median_diff: float          # Median of 100 * (TV-FLIDS - baseline)
    favor_count: int                   # Seeds favoring TV-FLIDS
    favor_ratio: float                 # favor_count / sample_size
    higher_is_better: bool

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def compute_paired_effect_size(
    differences: Union[np.ndarray, List[float]],
    endpoint: str,
    dataset: str,
    baseline: str,
) -> EffectSizeResult:
    """
    Compute paired median difference and favor count for a comparison.

    Args:
        differences: Array of paired percentage-point differences d = 100 * (TV-FLIDS - baseline).
        endpoint: "f1" (higher is better) or "asr" (lower is better).
        dataset: Dataset name (e.g. "nslkdd", "ciciot2023", "edgeiiotset").
        baseline: Baseline name (e.g. "fedavg", "fltrust").

    Returns:
        EffectSizeResult with paired median difference and favor metrics.
    """
    d = np.asarray(differences, dtype=float).flatten()
    n = len(d)
    if n == 0:
        return EffectSizeResult(
            dataset=dataset,
            baseline=baseline,
            endpoint=endpoint,
            sample_size=0,
            paired_median_diff=0.0,
            favor_count=0,
            favor_ratio=0.0,
            higher_is_better=(endpoint == "f1"),
        )

    med = float(np.median(d))
    higher_is_better = (endpoint == "f1")

    if higher_is_better:
        favor_count = int(np.sum(d > 0))
    else:
        favor_count = int(np.sum(d < 0))

    favor_ratio = float(favor_count / n) if n > 0 else 0.0

    return EffectSizeResult(
        dataset=dataset,
        baseline=baseline,
        endpoint=endpoint,
        sample_size=n,
        paired_median_diff=med,
        favor_count=favor_count,
        favor_ratio=favor_ratio,
        higher_is_better=higher_is_better,
    )
