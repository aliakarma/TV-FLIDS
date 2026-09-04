"""
experiments/run_ablation.py
Ablation study runner — A1 through A6.

A1: TV-FLIDS without verification module
A2: TV-FLIDS without memory decay (decay=0)
A3: Similarity-only trust scoring (α=1, β=0, γ=0)
A4: Accuracy-only trust scoring (α=0, β=1, γ=0)
A5: IID vs Non-IID data partitioning
A6: Fixed equal weights (α=β=γ=1/3, meta-gradient adaptation OFF; verification
    gate stays ON) — isolates the meta-gradient's marginal contribution over
    a static trust-signal blend. Paper: Table VII, §IX.

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
    "A6: Fixed Equal Weights": {
        # strategy="tvflids_fixed" disables the meta-gradient (adaptive=False,
        # see fl/strategy.py / experiments/run_experiment.py::make_strategy),
        # while the verification gate (signals) stays on. Setting alpha=beta=
        # gamma=1/3 fixes the trust blend at equal weights for the whole run,
        # isolating the meta-gradient's marginal contribution (paper §IX).
        "strategy": "tvflids_fixed",
        "partition_type": "noniid", "alpha": 0.5,
        "no_verification": False,
        "trust_override": {"alpha": 1.0 / 3.0, "beta": 1.0 / 3.0, "gamma": 1.0 / 3.0},
    },
}


def run_ablation(
    attack_config_name: str = "label_flip_30",
    seeds: list = None,
    num_rounds: int = 100,
    config_path: str = "config/fl_config.yaml",
    output_dir: str = "results/tables",
):
    """Run all ablation studies and write results to JSON."""
    if seeds is None:
        # Paper protocol (audit Priority 3 fix): 5 seeds, 100 rounds by
        # default. Pass --seeds/--rounds to reduce for a smoke test.
        seeds = [42, 123, 456, 789, 1337]

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

            # Disable verification by setting extreme thresholds. The warmup
            # annealing is switched off in the same breath: leaving it on would
            # anneal tau_L from -0.1 toward -1e9 rather than holding the arm at
            # a flat "accept everything", which is not what A1 is meant to
            # isolate. With annealing off, the extreme nominal values are used
            # verbatim from round 1.
            if ablation_cfg.get("no_verification"):
                exp_config["verification"]["loss_threshold"] = -1e9
                exp_config["verification"]["cosine_threshold"] = -1e9
                exp_config["verification"]["zscore_threshold"] = 1e9
                exp_config["verification"]["adaptive_thresholds"] = False

            # Write modified config to temp file. The arm name is part of the
            # filename so two arms sharing a seed cannot overwrite each other's
            # config.
            tmp_config_path = os.path.join(
                tempfile.gettempdir(),
                f"ablation_config_{ablation_name.replace(' ', '_').replace(':', '')}"
                f"_{seed}.yaml"
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
                # The colon is stripped as well as the space: ':' is a legal
                # POSIX filename character but illegal on Windows and unsafe in
                # Google Drive, so "A1: No Verification" would produce a log
                # directory that cannot be created, synced or archived off
                # Linux. This matches the sanitisation the temp config path
                # above already applies to the same arm name.
                log_dir=(f"results/logs/ablation_"
                         f"{ablation_name.replace(' ', '_').replace(':', '')}"
                         f"_{seed}"),
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

    # Figure 4 (results/figures/fig4_ablation.pdf). scripts/check_results.py
    # lists this file as required, but nothing produced it: figure4_ablation_bars
    # existed in evaluation/visualization.py with no call site anywhere, so the
    # completeness check could never pass even after a genuine full run.
    try:
        from evaluation.visualization import figure4_ablation_bars
        figure4_ablation_bars(
            summary_table,
            save_path="results/figures/fig4_ablation.pdf",
        )
        print("[Ablation] Figure 4 written to results/figures/fig4_ablation.pdf")
    except Exception as e:
        print(f"[Warning] Figure 4 generation failed: {e}")

    return summary_table


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run TV-FLIDS ablation studies")
    parser.add_argument("--attack",  default="label_flip_30")
    parser.add_argument("--rounds",  type=int, default=100)
    parser.add_argument("--seeds",   type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    parser.add_argument("--output",  default="results/tables")
    args = parser.parse_args()

    run_ablation(
        attack_config_name=args.attack,
        seeds=args.seeds,
        num_rounds=args.rounds,
        output_dir=args.output,
    )
