"""
experiments/run_multi_attack_matrix.py
Full multi-attack x baseline matrix runner (audit ID E13, paper Table IX/X).

Runs every (strategy, attack) cell across the paper's 8 strategies x 3
attack types (gradient-scale GS, noise-injection NI, backdoor BD) x seeds,
and saves the full matrix. Prior to this runner, gradient_scale_30,
noise_30, and backdoor_20 (already-defined keys in
attacks.adversarial.ATTACK_CONFIGS) were never composed into a matrix
sweep — zero results files referenced them.

Usage:
    python experiments/run_multi_attack_matrix.py
    python experiments/run_multi_attack_matrix.py --strategies fedavg tvflids --attacks noise_30
    python experiments/run_multi_attack_matrix.py --seeds 42 123 --rounds 2   # smoke-scale
"""

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.run_experiment import run_experiment as _real_run_experiment
from evaluation.statistical_testing import SEEDS

# Same 8-strategy default comparison set as run_full_comparison.STRATEGIES
# (bucketing/deepsight intentionally excluded from the default matrix for
# the same reason documented there — pass --strategies explicitly to add).
STRATEGIES = [
    "fedavg", "krum", "trimmed_mean", "fltrust",
    "foolsgold", "flame", "rfa", "tvflids",
]
# GS = gradient-scale, NI = noise-injection, BD = backdoor
ATTACKS = ["gradient_scale_30", "noise_30", "backdoor_20"]
METRICS = ["final_accuracy", "final_f1_macro", "final_attack_success_rate"]


def run_multi_attack_matrix_jobs(
    strategies: List[str],
    attacks: List[str],
    seeds: List[int],
    num_rounds: int,
    config_path: str,
    log_root: str,
    run_experiment_fn: Callable,
    verbose: bool = True,
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Execute the (strategy x attack x seed) job grid and return the raw
    matrix: {strategy: {attack: [result_per_seed, ...]}}. Kept separate
    from aggregation so plumbing can be unit-tested without running jobs.
    """
    matrix: Dict[str, Dict[str, List[Dict]]] = {s: {} for s in strategies}

    for strategy in strategies:
        for attack in attacks:
            if verbose:
                print(f"\n[Matrix] strategy={strategy} attack={attack}")
            seed_results = []
            for seed in seeds:
                result = run_experiment_fn(
                    strategy_name=strategy,
                    attack_config_name=attack,
                    seed=seed,
                    num_rounds=num_rounds,
                    config_path=config_path,
                    log_dir=f"{log_root}/{strategy}_{attack}_seed{seed}",
                    verbose=False,
                )
                seed_results.append(result)
                if verbose:
                    print(f"    seed={seed}: acc={result.get('final_accuracy', 0):.4f}")
            matrix[strategy][attack] = seed_results

    return matrix


def aggregate_multi_attack_matrix(matrix: Dict[str, Dict[str, List[Dict]]]) -> Dict:
    """
    Pure aggregation step: mean+/-std per metric for each (strategy,
    attack) cell. Unit-testable on canned per-seed result dicts.
    """
    summary: Dict[str, Dict[str, Dict]] = {}
    for strategy, by_attack in matrix.items():
        summary[strategy] = {}
        for attack, seed_results in by_attack.items():
            cell = {}
            for metric in METRICS:
                vals = [r.get(metric, 0.0) for r in seed_results]
                cell[metric] = {
                    "mean": float(np.mean(vals)) if vals else 0.0,
                    "std":  float(np.std(vals)) if vals else 0.0,
                    "n_seeds": len(vals),
                }
            summary[strategy][attack] = cell
    return summary


def run_multi_attack_matrix(
    strategies: Optional[List[str]] = None,
    attacks: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    num_rounds: int = 100,
    config_path: str = "config/fl_config.yaml",
    output_dir: str = "results/tables",
    log_root: str = "results/logs/multi_attack_matrix",
    run_experiment_fn: Callable = _real_run_experiment,
    verbose: bool = True,
) -> Dict:
    """
    Run the full (strategy x attack x seed) matrix, aggregate, and save
    results/tables/multi_attack_matrix_results.json.
    """
    if strategies is None:
        strategies = STRATEGIES
    if attacks is None:
        attacks = ATTACKS
    if seeds is None:
        seeds = SEEDS

    os.makedirs(output_dir, exist_ok=True)

    matrix = run_multi_attack_matrix_jobs(
        strategies, attacks, seeds, num_rounds, config_path,
        log_root, run_experiment_fn, verbose,
    )
    summary = aggregate_multi_attack_matrix(matrix)

    if verbose:
        print("\n\n[Multi-Attack Matrix Summary]")
        print(f"{'Strategy':<14} {'Attack':<18} {'Accuracy':>16} "
              f"{'F1-Macro':>16} {'ASR':>16}")
        print("-" * 82)
        for strategy, by_attack in summary.items():
            for attack, cell in by_attack.items():
                acc = cell["final_accuracy"]
                f1 = cell["final_f1_macro"]
                asr = cell["final_attack_success_rate"]
                print(f"{strategy:<14} {attack:<18} "
                      f"{acc['mean']:.4f}±{acc['std']:.4f}  "
                      f"{f1['mean']:.4f}±{f1['std']:.4f}  "
                      f"{asr['mean']:.4f}±{asr['std']:.4f}")

    # Table IX/X (the default GS/NI/BD matrix) and Table XI (the ACK1/ACK2
    # adaptive matrix) are both produced by this runner. Writing both to one
    # filename would let whichever ran second silently overwrite the other's
    # artifact, so a non-default attack set gets its own namespaced file --
    # the same guard run_full_comparison.py applies to its protocol variants.
    default_set = (sorted(attacks) == sorted(ATTACKS)
                   and sorted(strategies) == sorted(STRATEGIES))
    suffix = "" if default_set else "_" + "_".join(sorted(attacks))
    out_path = os.path.join(
        output_dir, f"multi_attack_matrix_results{suffix}.json")
    payload = {
        "provenance": {
            "strategies": strategies,
            "attacks": attacks,
            "seeds": seeds,
            "num_rounds": num_rounds,
            "audit_id": "E13",
            "is_default_matrix": default_set,
        },
        "raw": matrix,
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    if verbose:
        print(f"\n[Multi-Attack Matrix] Saved to {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Full multi-attack x baseline matrix (audit ID E13, paper Table IX/X)"
    )
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--attacks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--config", default="config/fl_config.yaml")
    parser.add_argument("--output", default="results/tables")
    args = parser.parse_args()

    run_multi_attack_matrix(
        strategies=args.strategies,
        attacks=args.attacks,
        seeds=args.seeds,
        num_rounds=args.rounds,
        config_path=args.config,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
