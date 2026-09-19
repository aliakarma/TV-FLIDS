"""
evaluation/statistical_pipeline.py
Canonical statistical evaluation pipeline and reproducible analysis artifact generation.
Reference: IEEE TIFS Manuscript §VI-E, §VII, and Supplementary §S8.

Guarantees:
  - Canonical output schema preserving run IDs, raw values, differences, Wilcoxon, Holm, BCa CI, and effect sizes.
  - Corrects primary comparisons over the complete 84-hypothesis family (14 baselines x 3 datasets x 2 endpoints).
  - Explicit missing-data lifecycle: COMPLETE, INCOMPLETE, BLOCKED, INVALID, SYNTHETIC_TEST_ONLY.
  - Never presents incomplete comparisons as final significance claims.
  - Records full provenance: Git commit, dataset provenance, analysis seed, bootstrap replicates, environment.
  - Zero data fabrication: does not impute or substitute synthetic data in production runs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from campaign.artifacts import validate_result_schema
from evaluation.bootstrap_ci import compute_paired_bca_ci
from evaluation.delta_metrics import compute_delta_asr
from evaluation.effect_size import compute_paired_effect_size
from evaluation.holm import holm_bonferroni_adjust
from evaluation.pairing import PairedComparison, pair_primary_comparison, pair_delta_asr
from evaluation.result_loader import ResultLoader
from evaluation.wilcoxon import wilcoxon_signed_rank_test
from utils.provenance import git_state, package_versions, hardware


BASELINES_14 = [
    "fedavg", "krum", "multikrum", "trimmed_mean", "norm_clipping",
    "rfa", "bucketing", "foolsgold", "flame", "deepsight",
    "fldetector", "zeno", "fltrust", "baffle"
]

DATASETS_3 = ["nslkdd", "ciciot2023", "edgeiiotset"]
PRIMARY_ENDPOINTS = ["f1", "asr"]


@dataclass
class PrimaryComparisonResult:
    """Canonical schema for a single primary comparison."""
    dataset: str
    baseline: str
    endpoint: str                      # "f1" or "asr"
    block: str
    attack: str
    required_seeds: int
    num_paired_observations: int
    status: str                        # "COMPLETE", "INCOMPLETE", "BLOCKED", "INVALID", "SYNTHETIC_TEST_ONLY"
    tvflids_run_ids: List[str]
    baseline_run_ids: List[str]
    paired_seeds: List[int]
    tvflids_values: List[float]
    baseline_values: List[float]
    paired_differences: List[float]    # 100 * (TV-FLIDS - baseline)
    paired_median_diff: Optional[float]
    favor_count: Optional[int]
    favor_ratio: Optional[float]
    wilcoxon_statistic: Optional[float]
    raw_p_value: Optional[float]
    holm_p_value: Optional[float]
    bca_ci_low: Optional[float]
    bca_ci_high: Optional[float]
    confidence_level: float = 0.95
    claim_significant: Optional[bool] = None
    claim_favors_tvflids: Optional[bool] = None
    validation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class StatisticalPipeline:
    """
    Executes reproducible statistical analysis over campaign artifacts.
    """

    def __init__(
        self,
        loader: ResultLoader,
        output_dir: str = "results/analysis",
        bootstrap_replicates: int = 10_000,
        seed: int = 20260914,
        allow_synthetic: bool = False,
    ):
        self.loader = loader
        self.output_dir = os.path.abspath(output_dir)
        self.bootstrap_replicates = bootstrap_replicates
        self.seed = seed
        self.allow_synthetic = allow_synthetic

    def run_primary_evaluation(
        self,
        expected_seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Execute primary evaluation across all 84 comparisons (14 baselines x 3 datasets x 2 endpoints).
        Applies Holm-Bonferroni correction over the complete family of 84 hypotheses.

        Returns:
            Dict containing metadata and the list of 84 PrimaryComparisonResult objects.
        """
        results: List[PrimaryComparisonResult] = []

        # 1. Form pairs and compute per-comparison statistics
        for ds in DATASETS_3:
            for base in BASELINES_14:
                for ep in PRIMARY_ENDPOINTS:
                    comp = pair_primary_comparison(
                        loader=self.loader,
                        dataset=ds,
                        baseline=base,
                        endpoint=ep,
                        block="B2",
                        attack="label_flip_30",
                        expected_seeds=expected_seeds,
                    )

                    is_synthetic = any(
                        self.loader.get_by_run_id(p.tvflids_run_id).is_synthetic
                        for p in comp.pairs
                        if self.loader.get_by_run_id(p.tvflids_run_id)
                    )

                    status = comp.status
                    if is_synthetic:
                        status = "SYNTHETIC_TEST_ONLY"

                    tv_ids = [p.tvflids_run_id for p in comp.pairs]
                    base_ids = [p.baseline_run_id for p in comp.pairs]
                    p_seeds = [p.seed for p in comp.pairs]
                    tv_vals = [p.tvflids_val for p in comp.pairs]
                    base_vals = [p.baseline_val for p in comp.pairs]
                    diffs = [p.diff for p in comp.pairs]

                    if comp.is_complete or (self.allow_synthetic and len(comp.pairs) == comp.required_seeds):
                        # Compute Wilcoxon
                        w_res = wilcoxon_signed_rank_test(diffs)
                        stat = w_res.statistic
                        raw_p = w_res.p_value

                        # Compute Effect Size
                        eff = compute_paired_effect_size(diffs, ep, ds, base)
                        med_diff = eff.paired_median_diff
                        favor = eff.favor_count
                        ratio = eff.favor_ratio

                        # Compute BCa Bootstrap CI
                        seed_tag = f"prim_{ds}_{base}_{ep}"
                        ci_res = compute_paired_bca_ci(
                            diffs,
                            confidence_level=0.95,
                            n_resamples=self.bootstrap_replicates,
                            seed_or_tag=seed_tag,
                        )
                        ci_lo = ci_res.ci_low
                        ci_hi = ci_res.ci_high
                    else:
                        stat = None
                        raw_p = None
                        med_diff = None
                        favor = None
                        ratio = None
                        ci_lo = None
                        ci_hi = None

                    res_item = PrimaryComparisonResult(
                        dataset=ds,
                        baseline=base,
                        endpoint=ep,
                        block=comp.block,
                        attack=comp.attack,
                        required_seeds=comp.required_seeds,
                        num_paired_observations=len(comp.pairs),
                        status=status,
                        tvflids_run_ids=tv_ids,
                        baseline_run_ids=base_ids,
                        paired_seeds=p_seeds,
                        tvflids_values=tv_vals,
                        baseline_values=base_vals,
                        paired_differences=diffs,
                        paired_median_diff=med_diff,
                        favor_count=favor,
                        favor_ratio=ratio,
                        wilcoxon_statistic=stat,
                        raw_p_value=raw_p,
                        holm_p_value=None,    # To be adjusted across the family
                        bca_ci_low=ci_lo,
                        bca_ci_high=ci_hi,
                        confidence_level=0.95,
                        claim_significant=None,
                        claim_favors_tvflids=None,
                        validation_notes=comp.validation_errors,
                    )
                    results.append(res_item)

        # 2. Adjust across complete 84-hypothesis family
        # For incomplete comparisons, p-value is treated as 1.0 (no claim)
        raw_p_array = [r.raw_p_value if (r.raw_p_value is not None and r.status in ("COMPLETE", "SYNTHETIC_TEST_ONLY")) else 1.0 for r in results]
        holm_p_array = holm_bonferroni_adjust(raw_p_array)

        for r, adj_p in zip(results, holm_p_array):
            if r.status in ("COMPLETE", "SYNTHETIC_TEST_ONLY") and r.raw_p_value is not None:
                r.holm_p_value = float(adj_p)
                r.claim_significant = bool(adj_p < 0.05)

                # RQ1 Decision Rule: Holm p < 0.05 AND interval excludes 0 in favor direction
                # Macro-F1: lo > 0 favors TV-FLIDS
                # ASR: hi < 0 favors TV-FLIDS
                if r.bca_ci_low is not None and r.bca_ci_high is not None:
                    if r.endpoint == "f1":
                        r.claim_favors_tvflids = bool(r.claim_significant and r.bca_ci_low > 0)
                    else:
                        r.claim_favors_tvflids = bool(r.claim_significant and r.bca_ci_high < 0)
                else:
                    r.claim_favors_tvflids = False
            else:
                r.holm_p_value = None
                r.claim_significant = False
                r.claim_favors_tvflids = False

        # 3. Assemble canonical payload
        payload = {
            "metadata": {
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "total_comparisons": len(results),
                "family_size": 84,
                "endpoints": PRIMARY_ENDPOINTS,
                "datasets": DATASETS_3,
                "baselines": BASELINES_14,
                "bootstrap_replicates": self.bootstrap_replicates,
                "analysis_seed": self.seed,
                "git_provenance": git_state(),
                "environment": {
                    "python_version": sys.version.split()[0],
                    "packages": package_versions(),
                    "hardware": hardware(),
                },
            },
            "summary": {
                "complete_comparisons": sum(1 for r in results if r.status == "COMPLETE"),
                "incomplete_comparisons": sum(1 for r in results if r.status == "INCOMPLETE"),
                "blocked_comparisons": sum(1 for r in results if r.status == "BLOCKED"),
                "synthetic_test_comparisons": sum(1 for r in results if r.status == "SYNTHETIC_TEST_ONLY"),
                "claims_favoring_tvflids": sum(1 for r in results if r.claim_favors_tvflids),
            },
            "comparisons": [r.to_dict() for r in results],
        }

        # 4. Save analysis artifact
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "primary_evaluation.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return payload


def main() -> None:
    """CLI entry point for running statistical evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description="TV-FLIDS Statistical Evaluation Pipeline")
    parser.add_argument("--results-dir", default="results", help="Base directory containing campaign run artifacts")
    parser.add_argument("--output-dir", default="results/analysis", help="Directory to store analysis artifacts")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic test fixtures (TESTING ONLY)")
    args = parser.parse_args()

    loader = ResultLoader(base_dir=args.results_dir, allow_synthetic=args.allow_synthetic)
    n_loaded = loader.discover_and_load()
    print(f"[Stage 10] Discovered and loaded {n_loaded} valid completed runs.")

    pipeline = StatisticalPipeline(
        loader=loader,
        output_dir=args.output_dir,
        allow_synthetic=args.allow_synthetic,
    )
    res = pipeline.run_primary_evaluation()
    print(f"[Stage 10] Primary evaluation complete across {len(res['comparisons'])} comparisons.")
    print(f"[Stage 10] Complete: {res['summary']['complete_comparisons']}, "
          f"Incomplete: {res['summary']['incomplete_comparisons']}, "
          f"Blocked: {res['summary']['blocked_comparisons']}, "
          f"Synthetic: {res['summary']['synthetic_test_comparisons']}")


if __name__ == "__main__":
    main()
