"""
experiments/run_full_comparison.py
Multi-seed full comparison across all strategies.
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
from evaluation.visualization import (
    figure1_convergence_curves, figure6_confusion_matrices
)
from data.preprocessing.nslkdd_pipeline import CLASS_NAMES


STRATEGIES = [
    "fedavg", "krum", "trimmed_mean", "fltrust",
    "foolsgold", "flame", "rfa", "tvflids",
]
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
    log_root = "results/logs/comparison"

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

    # ── Figure 1 and 6 generation ───────────────────────────────────
    try:
        seed_for_figs = seeds[0] if seeds else SEEDS[0]
        all_round_metrics = {}
        label_map = {
            "fedavg": "FedAvg",
            "krum": "Krum",
            "trimmed_mean": "TrimmedMean",
            "fltrust": "FLTrust",
            "foolsgold": "FoolsGold",
            "flame": "FLAME",
            "rfa": "RFA",
            "tvflids": "TV-FLIDS",
            "tvflids_fixed": "TV-FLIDS-Fixed",
        }
        for strategy in strategies:
            log_path = os.path.join(
                log_root, f"{strategy}_{attack}_seed{seed_for_figs}", "experiment_log.json"
            )
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    data = json.load(f)
                rounds = data.get("rounds", [])
                if rounds:
                    label = label_map.get(strategy, strategy)
                    all_round_metrics[label] = rounds
        if all_round_metrics:
            os.makedirs("results/figures", exist_ok=True)
            figure1_convergence_curves(
                all_round_metrics,
                save_path="results/figures/fig1_convergence.pdf",
            )

        fedavg_pred_path = os.path.join(
            log_root, f"fedavg_{attack}_seed{seed_for_figs}", "final_predictions.npz"
        )
        tvflids_pred_path = os.path.join(
            log_root, f"tvflids_{attack}_seed{seed_for_figs}", "final_predictions.npz"
        )
        if os.path.exists(fedavg_pred_path) and os.path.exists(tvflids_pred_path):
            fedavg_data = np.load(fedavg_pred_path)
            tvflids_data = np.load(tvflids_pred_path)
            figure6_confusion_matrices(
                y_true=fedavg_data["y_true"],
                y_pred_fedavg=fedavg_data["y_pred"],
                y_pred_tvflids=tvflids_data["y_pred"],
                class_names=CLASS_NAMES,
                save_path="results/figures/fig6_confusion.pdf",
            )
    except Exception as e:
        print(f"[Warning] Figure generation failed: {e}")
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
