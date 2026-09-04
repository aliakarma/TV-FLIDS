"""
experiments/run_extended_significance.py
Extended ten-seed significance test (audit ID E7, paper §VIII-B).

The paper's headline significance claim in §VIII-B is computed over an
extended 10-seed set: the original 5 paper seeds [42, 123, 456, 789, 1337]
(evaluation.statistical_testing.SEEDS) plus 5 additional seeds
[2024, 31415, 8080, 555, 999] (see EXTENDED_SEEDS in the same module).
Prior to this runner, none of the 5 additional seed values appeared
anywhere in the repo and there was no dedicated script that ran both
strategies across all 10 seeds and computed the paired one-sided Wilcoxon
signed-rank test used in the paper.

Default comparison: tvflids vs fltrust. FLTrust is TV-FLIDS's strongest
"root-of-trust" baseline in fl/baselines/ (it, like TV-FLIDS, relies on a
trusted server-side reference rather than purely geometric robustness like
Krum/Trimmed-Mean), which makes it the most natural default head-to-head
for a significance claim absent direct access to the audit's original
§VIII-B source text in this environment. Override with --strategies for
any other pair (e.g. tvflids vs krum).

Usage:
    python experiments/run_extended_significance.py
    python experiments/run_extended_significance.py --strategies tvflids krum --rounds 100
    python experiments/run_extended_significance.py --seeds 42 123 --rounds 2   # smoke-scale
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.run_experiment import run_experiment as _real_run_experiment
from evaluation.statistical_testing import (
    EXTENDED_SEEDS, compare_methods_wilcoxon, compute_cohens_d, compute_summary,
)

METRICS = ["final_accuracy", "final_f1_macro", "final_attack_success_rate"]


def _run_seeds(
    strategy_name: str,
    seeds: List[int],
    attack: str,
    num_rounds: int,
    partition_type: str,
    alpha: float,
    config_path: str,
    log_root: str,
    run_experiment_fn: Callable,
    verbose: bool,
) -> List[Dict]:
    """Run one strategy across `seeds`, returning the per-seed result dicts."""
    results = []
    for seed in seeds:
        result = run_experiment_fn(
            strategy_name=strategy_name,
            attack_config_name=attack,
            partition_type=partition_type,
            alpha=alpha,
            seed=seed,
            num_rounds=num_rounds,
            config_path=config_path,
            log_dir=f"{log_root}/{strategy_name}_{attack}_seed{seed}",
            verbose=False,
        )
        results.append(result)
        if verbose:
            print(f"  [{strategy_name}] seed={seed}: "
                  f"acc={result.get('final_accuracy', 0):.4f}")
    return results


def build_extended_significance_report(
    results_a: List[Dict],
    results_b: List[Dict],
    strategy_a: str,
    strategy_b: str,
    seeds: List[int],
    attack: str,
    num_rounds: int,
) -> Dict:
    """
    Pure aggregation/statistics step — no experiment execution here, so it
    can be unit-tested directly against fake-but-clearly-fake result dicts.

    Computes, per metric, mean±std for both strategies and a one-sided
    Wilcoxon signed-rank test (H0: no difference; alternative: A > B),
    plus Cohen's d, and wraps it all with provenance (seeds, strategies,
    attack, rounds, timestamp).
    """
    per_metric = {}
    for metric in METRICS:
        mean_a, std_a = compute_summary(results_a, metric)
        mean_b, std_b = compute_summary(results_b, metric)
        wtest = compare_methods_wilcoxon(results_a, results_b, metric)
        d = compute_cohens_d(results_a, results_b, metric)
        per_metric[metric] = {
            f"{strategy_a}_mean": mean_a, f"{strategy_a}_std": std_a,
            f"{strategy_b}_mean": mean_b, f"{strategy_b}_std": std_b,
            "wilcoxon_statistic": wtest.get("statistic"),
            "wilcoxon_p_value":   wtest.get("p_value"),
            "significant_p05":    wtest.get("significant"),
            "cohens_d":           d,
        }

    return {
        "provenance": {
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "seeds": list(seeds),
            "num_seeds": len(seeds),
            "attack": attack,
            "num_rounds": num_rounds,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "audit_id": "E7",
            "note": (
                "Wilcoxon test is one-sided (alternative='greater'), "
                "testing whether strategy_a > strategy_b per metric."
            ),
        },
        "per_metric": per_metric,
        "raw": {strategy_a: results_a, strategy_b: results_b},
    }


def run_extended_significance(
    strategy_a: str = "tvflids",
    strategy_b: str = "fltrust",
    seeds: Optional[List[int]] = None,
    attack: str = "label_flip_30",
    num_rounds: int = 100,
    partition_type: str = "noniid",
    alpha: float = 0.5,
    config_path: str = "config/fl_config.yaml",
    output_dir: str = "results/tables",
    log_root: str = "results/logs/extended_significance",
    run_experiment_fn: Callable = _real_run_experiment,
    verbose: bool = True,
) -> Dict:
    """
    Run strategy_a and strategy_b across `seeds` (default: the 10-seed
    EXTENDED_SEEDS set), compute the one-sided Wilcoxon test, and save
    results/tables/extended_significance_results.json.
    """
    if seeds is None:
        seeds = EXTENDED_SEEDS

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}\n{strategy_a.upper()} vs {strategy_b.upper()} "
              f"| {len(seeds)} seeds | attack={attack}\n{'='*60}")

    results_a = _run_seeds(
        strategy_a, seeds, attack, num_rounds, partition_type, alpha,
        config_path, log_root, run_experiment_fn, verbose,
    )
    results_b = _run_seeds(
        strategy_b, seeds, attack, num_rounds, partition_type, alpha,
        config_path, log_root, run_experiment_fn, verbose,
    )

    report = build_extended_significance_report(
        results_a, results_b, strategy_a, strategy_b, seeds, attack, num_rounds,
    )

    if verbose:
        print(f"\n[Extended Significance] {strategy_a} vs {strategy_b} "
              f"({len(seeds)} seeds)")
        for metric, row in report["per_metric"].items():
            p = row["wilcoxon_p_value"]
            sig = "*" if row["significant_p05"] else "ns"
            p_str = f"{p:.4f}" if p is not None else "N/A"
            print(f"  {metric:<28} p={p_str} {sig} d={row['cohens_d']:.3f}")

    out_path = os.path.join(output_dir, "extended_significance_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    if verbose:
        print(f"\n[Extended Significance] Saved to {out_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Extended 10-seed significance test (audit ID E7, paper §VIII-B)"
    )
    parser.add_argument("--strategies", nargs=2, default=["tvflids", "fltrust"],
                         metavar=("STRATEGY_A", "STRATEGY_B"),
                         help="Pair of strategies to compare (default: tvflids fltrust)")
    parser.add_argument("--seeds", nargs="+", type=int, default=EXTENDED_SEEDS,
                         help="Seeds to run (default: the paper's 10-seed extended set)")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--attack", default="label_flip_30")
    parser.add_argument("--partition", default="noniid", choices=["iid", "noniid"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--config", default="config/fl_config.yaml")
    parser.add_argument("--output", default="results/tables")
    args = parser.parse_args()

    run_extended_significance(
        strategy_a=args.strategies[0],
        strategy_b=args.strategies[1],
        seeds=args.seeds,
        attack=args.attack,
        num_rounds=args.rounds,
        partition_type=args.partition,
        alpha=args.alpha,
        config_path=args.config,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
