"""
experiments/run_noniid_sweep.py
Non-IID concentration sweep runner (audit ID E12, paper Table VIII).

Sweeps data heterogeneity level alpha_D in {0.1, 0.5, 1.0, IID} for a set
of strategies (default: Krum and TV-FLIDS, the paper's Table VIII pairing),
aggregating accuracy/F1/ASR mean+/-std per (strategy, condition) cell.

Previously this sweep was only reachable by hand-invoking
`run_experiment.py --alpha <value>` once per point; there was no dedicated
runner that swept all four heterogeneity conditions and aggregated the
result into one table.

Usage:
    python experiments/run_noniid_sweep.py
    python experiments/run_noniid_sweep.py --strategies fedavg tvflids --alphas 0.1 1.0
    python experiments/run_noniid_sweep.py --seeds 42 123 --rounds 2   # smoke-scale
"""

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.run_experiment import run_experiment as _real_run_experiment
from evaluation.statistical_testing import SEEDS

STRATEGIES = ["krum", "tvflids"]
ALPHAS = [0.1, 0.5, 1.0]
METRICS = ["final_accuracy", "final_f1_macro", "final_attack_success_rate"]


def _conditions(alphas: List[float]) -> List[Tuple[str, Optional[float], str]]:
    """
    Build the list of (partition_type, alpha, label) conditions: one
    noniid(alpha) condition per entry in `alphas`, plus a single trailing
    IID condition.
    """
    conds = [("noniid", a, f"alpha={a}") for a in alphas]
    conds.append(("iid", None, "IID"))
    return conds


def run_noniid_sweep_jobs(
    strategies: List[str],
    alphas: List[float],
    seeds: List[int],
    attack: str,
    num_rounds: int,
    config_path: str,
    log_root: str,
    run_experiment_fn: Callable,
    verbose: bool = True,
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Execute the (strategy x condition x seed) job grid and return the raw
    per-cell seed-result lists: {strategy: {condition_label: [result, ...]}}.
    Separated from aggregation so the aggregation step can be unit-tested
    on canned data without running any jobs.
    """
    conditions = _conditions(alphas)
    raw: Dict[str, Dict[str, List[Dict]]] = {s: {} for s in strategies}

    for strategy in strategies:
        for partition_type, alpha, label in conditions:
            if verbose:
                print(f"\n[NonIID Sweep] strategy={strategy} condition={label}")
            seed_results = []
            for seed in seeds:
                result = run_experiment_fn(
                    strategy_name=strategy,
                    attack_config_name=attack,
                    partition_type=partition_type,
                    alpha=alpha if alpha is not None else 0.5,
                    seed=seed,
                    num_rounds=num_rounds,
                    config_path=config_path,
                    log_dir=(f"{log_root}/{strategy}_{partition_type}_"
                              f"{label.replace('=', '')}_seed{seed}"),
                    verbose=False,
                )
                seed_results.append(result)
                if verbose:
                    print(f"    seed={seed}: acc={result.get('final_accuracy', 0):.4f}")
            raw[strategy][label] = seed_results

    return raw


def aggregate_noniid_sweep(raw: Dict[str, Dict[str, List[Dict]]]) -> Dict:
    """
    Pure aggregation step: mean+/-std per metric for each (strategy,
    condition) cell. No experiment execution — unit-testable on canned
    per-seed result dicts.
    """
    summary: Dict[str, Dict[str, Dict]] = {}
    for strategy, by_condition in raw.items():
        summary[strategy] = {}
        for label, seed_results in by_condition.items():
            cell = {}
            for metric in METRICS:
                vals = [r.get(metric, 0.0) for r in seed_results]
                cell[metric] = {
                    "mean": float(np.mean(vals)) if vals else 0.0,
                    "std":  float(np.std(vals)) if vals else 0.0,
                    "n_seeds": len(vals),
                }
            summary[strategy][label] = cell
    return summary


def run_noniid_sweep(
    strategies: Optional[List[str]] = None,
    alphas: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    attack: str = "label_flip_30",
    num_rounds: int = 100,
    config_path: str = "config/fl_config.yaml",
    output_dir: str = "results/tables",
    log_root: str = "results/logs/noniid_sweep",
    run_experiment_fn: Callable = _real_run_experiment,
    verbose: bool = True,
) -> Dict:
    """
    Sweep alpha_D in {alphas..., IID} for `strategies`, aggregate mean+/-std
    per (strategy, condition), and save results/tables/noniid_sweep_results.json.
    """
    if strategies is None:
        strategies = STRATEGIES
    if alphas is None:
        alphas = ALPHAS
    if seeds is None:
        seeds = SEEDS

    os.makedirs(output_dir, exist_ok=True)

    raw = run_noniid_sweep_jobs(
        strategies, alphas, seeds, attack, num_rounds, config_path,
        log_root, run_experiment_fn, verbose,
    )
    summary = aggregate_noniid_sweep(raw)

    if verbose:
        print("\n\n[NonIID Sweep Summary]")
        print(f"{'Strategy':<12} {'Condition':<12} {'Accuracy':>16} "
              f"{'F1-Macro':>16} {'ASR':>16}")
        print("-" * 76)
        for strategy, by_condition in summary.items():
            for label, cell in by_condition.items():
                acc = cell["final_accuracy"]
                f1 = cell["final_f1_macro"]
                asr = cell["final_attack_success_rate"]
                print(f"{strategy:<12} {label:<12} "
                      f"{acc['mean']:.4f}±{acc['std']:.4f}  "
                      f"{f1['mean']:.4f}±{f1['std']:.4f}  "
                      f"{asr['mean']:.4f}±{asr['std']:.4f}")

    out_path = os.path.join(output_dir, "noniid_sweep_results.json")
    payload = {
        "provenance": {
            "strategies": strategies,
            "alphas": alphas,
            "seeds": seeds,
            "attack": attack,
            "num_rounds": num_rounds,
            "audit_id": "E12",
        },
        "raw": raw,
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    if verbose:
        print(f"\n[NonIID Sweep] Saved to {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Non-IID concentration sweep (audit ID E12, paper Table VIII)"
    )
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--attack", default="label_flip_30")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--config", default="config/fl_config.yaml")
    parser.add_argument("--output", default="results/tables")
    args = parser.parse_args()

    run_noniid_sweep(
        strategies=args.strategies,
        alphas=args.alphas,
        seeds=args.seeds,
        attack=args.attack,
        num_rounds=args.rounds,
        config_path=args.config,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
