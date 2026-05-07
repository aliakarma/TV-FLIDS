"""
experiments/run_dataset_comparison.py
Cross-dataset validation: does the TV-FLIDS advantage hold on UNSW-NB15?
"""

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.run_experiment import run_experiment


def run_dataset_comparison(
    datasets=None,
    strategies=None,
    seeds=None,
    num_rounds: int = 100,
    attack: str = "label_flip_30",
    config_path: str = "config/fl_config.yaml",
    output_path: str = "results/tables/dataset_comparison_results.json",
):
    if datasets is None:
        datasets = ["nslkdd", "unswnb15"]
    if strategies is None:
        strategies = ["fedavg", "fltrust", "tvflids"]
    if seeds is None:
        seeds = [42, 123, 456]

    results = {}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for dataset in datasets:
        results[dataset] = {}
        for strategy in strategies:
            seed_results = []
            for seed in seeds:
                res = run_experiment(
                    strategy_name=strategy,
                    attack_config_name=attack,
                    dataset=dataset,
                    seed=seed,
                    num_rounds=num_rounds,
                    config_path=config_path,
                    verbose=False,
                )
                seed_results.append(res)
            results[dataset][strategy] = seed_results

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Dataset Comparison] Saved to {output_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-dataset comparison")
    parser.add_argument("--datasets", nargs="+", default=["nslkdd", "unswnb15"])
    parser.add_argument("--strategies", nargs="+", default=["fedavg", "fltrust", "tvflids"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--attack", default="label_flip_30")
    parser.add_argument("--config", default="config/fl_config.yaml")
    parser.add_argument("--output", default="results/tables/dataset_comparison_results.json")
    args = parser.parse_args()

    run_dataset_comparison(
        datasets=args.datasets,
        strategies=args.strategies,
        seeds=args.seeds,
        num_rounds=args.rounds,
        attack=args.attack,
        config_path=args.config,
        output_path=args.output,
    )
