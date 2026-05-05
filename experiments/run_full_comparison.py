"""
experiments/run_full_comparison.py
Multi-seed full comparison: all 5 strategies vs all attack types.
Produces the main results Table 1 with mean ± std and Wilcoxon tests.

Reference: Guide §10.3
"""

import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.run_experiment import run_experiment
from evaluation.statistical_testing import (
    build_results_table, compare_methods_wilcoxon, SEEDS
)


STRATEGIES = ["fedavg", "krum", "trimmed_mean", "fltrust", "foolsgold", "tvflids"]
PRIMARY_ATTACK = "label_flip_30"
METRICS = ["final_accuracy", "final_f1_macro", "final_attack_success_rate"]


def run_full_comparison(
    strategies: list = None,
    attack: str = PRIMARY_ATTACK,
    seeds: list = None,
    num_rounds: int = 50,
    config_path: str = "config/fl_config.yaml",
    output_dir: str = "results/tables",
    verbose: bool = True,
) -> dict:
    """
    Run all strategies across all seeds for the primary attack.

    Returns:
        {strategy_name: [result_per_seed]}
    """
    if strategies is None:
        strategies = STRATEGIES
    if seeds is None:
        seeds = SEEDS[:3]   # Default to 3 seeds for speed; use all 5 for paper

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.upper()} | Attack: {attack}")
        print(f"{'='*60}")
        seed_results = []

        for seed in seeds:
            result = run_experiment(
                strategy_name=strategy,
                attack_config_name=attack,
                seed=seed,
                num_rounds=num_rounds,
                config_path=config_path,
                log_dir=f"results/logs/comparison/{strategy}_{attack}_seed{seed}",
                verbose=False,
            )
            seed_results.append(result)
            if verbose:
                print(f"  Seed {seed}: acc={result.get('final_accuracy', 0):.4f}")

        all_results[strategy] = seed_results

    # Print publication-ready table
    print("\n\n" + "="*80)
    print("RESULTS TABLE (mean ± std across seeds)")
    print("="*80)
    table = build_results_table(
        all_results,
        metrics=METRICS,
        reference_method="tvflids",
        print_output=True,
    )

    # Pairwise Wilcoxon tests: TV-FLIDS vs each baseline
    print("\n\nWilcoxon Tests (TV-FLIDS vs baselines):")
    print(f"{'Method':<20} {'Accuracy p-val':>18} {'Significant':>14}")
    print("-" * 55)
    tvflids_results = all_results.get("tvflids", [])
    for strategy, results in all_results.items():
        if strategy == "tvflids":
            continue
        wtest = compare_methods_wilcoxon(tvflids_results, results, "final_accuracy")
        sig = "YES *" if wtest["significant"] else "no"
        print(f"{strategy:<20} {wtest['p_value']:>18.4f} {sig:>14}")

    # Save
    out_path = os.path.join(output_dir, "full_comparison_results.json")
    with open(out_path, "w") as f:
        json.dump({"raw": all_results, "table": table}, f,
                   indent=2, default=str)
    print(f"\n[Comparison] Saved to {out_path}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full multi-strategy comparison")
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--attack",     default="label_flip_30")
    parser.add_argument("--seeds",      nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--rounds",     type=int, default=50)
    parser.add_argument("--config",     default="config/fl_config.yaml")
    parser.add_argument("--output",     default="results/tables")
    args = parser.parse_args()

    run_full_comparison(
        strategies=args.strategies,
        attack=args.attack,
        seeds=args.seeds,
        num_rounds=args.rounds,
        config_path=args.config,
        output_dir=args.output,
    )
