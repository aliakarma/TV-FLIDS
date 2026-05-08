"""
experiments/run_ablation.py
Ablation study runner — A1 through A5.

A1: TV-FLIDS without verification module
A2: TV-FLIDS without memory decay (decay=0)
A3: Similarity-only trust scoring (α=1, β=0, γ=0)
A4: Accuracy-only trust scoring (α=0, β=1, γ=0)
A5: IID vs Non-IID data partitioning

Reference: Guide §10.1 Stage 4
"""

import os
import sys
import json
import copy
import tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
from experiments.run_experiment import run_experiment, load_config
from evaluation.statistical_testing import compare_methods_wilcoxon, compute_cohens_d


ABLATION_CONFIGS = {
    "TV-FLIDS (Full)": {
        "strategy": "tvflids",
        "partition_type": "noniid", "alpha": 0.5,
        "trust_override": None, "no_verification": False,
    },
    "A1: No Verification": {
        "strategy": "tvflids",
        "partition_type": "noniid", "alpha": 0.5,
        "no_verification": True,
        "trust_override": None,
    },
    "A2: No Memory (decay=0)": {
        "strategy": "tvflids",
        "partition_type": "noniid", "alpha": 0.5,
        "no_verification": False,
        "trust_override": {"memory_decay": 0.0},
    },
    "A3: Similarity Only": {
        "strategy": "tvflids_fixed",
        "partition_type": "noniid", "alpha": 0.5,
        "no_verification": False,
        "trust_override": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    },
    "A4: Accuracy Only": {
        "strategy": "tvflids_fixed",
        "partition_type": "noniid", "alpha": 0.5,
        "no_verification": False,
        "trust_override": {"alpha": 0.0, "beta": 1.0, "gamma": 0.0},
    },
    "A5: IID Data": {
        "strategy": "tvflids",
        "partition_type": "iid", "alpha": 0.5,
        "no_verification": False,
        "trust_override": None,
    },
}


def run_ablation(
    attack_config_name: str = "label_flip_30",
    seeds: list = None,
    num_rounds: int = 50,
    config_path: str = "config/fl_config.yaml",
    output_dir: str = "results/tables",
):
    """Run all ablation studies and write results to JSON."""
    if seeds is None:
        seeds = [42, 123]   # Use fewer seeds for ablation by default

    os.makedirs(output_dir, exist_ok=True)

    config = load_config(config_path)
    all_results = {}

    for ablation_name, ablation_cfg in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_name}")
        print(f"{'='*60}")

        seed_results = []

        for seed in seeds:
            # Apply trust overrides to config
            exp_config = copy.deepcopy(config)
            if ablation_cfg.get("trust_override"):
                for k, v in ablation_cfg["trust_override"].items():
                    exp_config["trust"][k] = v

            # Disable verification by setting extreme thresholds
            if ablation_cfg.get("no_verification"):
                exp_config["verification"]["loss_threshold"] = -1e9
                exp_config["verification"]["cosine_threshold"] = -1e9
                exp_config["verification"]["zscore_threshold"] = 1e9

            # Write modified config to temp file
            tmp_config_path = os.path.join(
                tempfile.gettempdir(), f"ablation_config_{seed}.yaml"
            )
            with open(tmp_config_path, "w") as f:
                yaml.dump(exp_config, f)

            result = run_experiment(
                strategy_name=ablation_cfg["strategy"],
                attack_config_name=attack_config_name,
                partition_type=ablation_cfg["partition_type"],
                alpha=ablation_cfg["alpha"],
                seed=seed,
                num_rounds=num_rounds,
                config_path=tmp_config_path,
                log_dir=f"results/logs/ablation_{ablation_name.replace(' ', '_')}_{seed}",
                verbose=False,
            )
            seed_results.append(result)

        all_results[ablation_name] = seed_results

    # Summarize
    print("\n\n[Ablation Summary]")
    print(f"{'Ablation':<30} {'Accuracy':>12} {'F1-Macro':>12} {'ASR':>12}")
    print("-" * 70)

    summary_table = {}
    for name, results in all_results.items():
        accs = [r.get("final_accuracy", 0.0) for r in results]
        f1s  = [r.get("final_f1_macro", 0.0) for r in results]
        asrs = [r.get("final_attack_success_rate", 0.0) for r in results]

        row = {
            "accuracy":            (float(np.mean(accs)), float(np.std(accs))),
            "f1_macro":            (float(np.mean(f1s)),  float(np.std(f1s))),
            "attack_success_rate": (float(np.mean(asrs)), float(np.std(asrs))),
        }
        summary_table[name] = row

        print(f"{name:<30} "
              f"{np.mean(accs):.4f}±{np.std(accs):.4f}  "
              f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
              f"{np.mean(asrs):.4f}±{np.std(asrs):.4f}")

    full_results = all_results["TV-FLIDS (Full)"]
    print("\n[Ablation Significance vs TV-FLIDS Full]")
    print(f"{'Ablation':<30} {'Metric':<25} {'p-value':>10} {'Sig':>5} {'d':>8}")
    print("-" * 80)

    for name, results in all_results.items():
        if name == "TV-FLIDS (Full)":
            continue
        for metric in ["final_f1_macro", "final_attack_success_rate"]:
            wtest = compare_methods_wilcoxon(full_results, results, metric)
            d = compute_cohens_d(full_results, results, metric)
            sig = "*" if wtest.get("significant") else "ns"
            print(f"  {name:<28} {metric:<25} "
                  f"{wtest.get('p_value', 1.0):>10.4f} {sig:>5} {d:>8.3f}")

    # Save results
    out_path = os.path.join(output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump({"raw": all_results, "summary": summary_table}, f,
                   indent=2, default=str)
    print(f"\n[Ablation] Results saved to {out_path}")

    return summary_table


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run TV-FLIDS ablation studies")
    parser.add_argument("--attack",  default="label_flip_30")
    parser.add_argument("--rounds",  type=int, default=50)
    parser.add_argument("--seeds",   type=int, nargs="+", default=[42, 123])
    parser.add_argument("--output",  default="results/tables")
    args = parser.parse_args()

    run_ablation(
        attack_config_name=args.attack,
        seeds=args.seeds,
        num_rounds=args.rounds,
        output_dir=args.output,
    )
