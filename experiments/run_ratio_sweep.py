"""
experiments/run_ratio_sweep.py
Adversarial ratio sweep: 0% to 60% Byzantine clients.

Generates the robustness curve (Figure 3) showing performance
degradation as adversarial participation increases.

Reference: Guide §17
"""

import os
import sys
import json
import tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.run_experiment import run_experiment

METHODS = ["fedavg", "krum", "fltrust", "tvflids"]
RATIOS  = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

RATIO_TO_ATTACK = {
    0.0:  "no_attack",
    0.10: "label_flip_10",
    0.20: "label_flip_20",
    0.30: "label_flip_30",
    0.40: "label_flip_30",   # reuse config, ratio overridden in experiment
    0.50: "label_flip_30",
    0.60: "label_flip_30",
}

# Map (method, ratio) → appropriate attack config key
# Note: for ratios > 0.30, we run label_flip_30 config which sets ratio=0.30
# The actual ratio is enforced via malicious_ids. To properly sweep,
# we directly manipulate via run_experiment's config override mechanism.


def run_ratio_sweep(
    methods: list = None,
    ratios: list = None,
    seeds: list = None,
    num_rounds: int = 50,
    config_path: str = "config/fl_config.yaml",
    output_dir: str = "results/tables",
):
    """
    Sweep adversarial ratio for all specified methods.

    Returns:
        ratio_results: {method: {ratio: {accuracy_mean, accuracy_std, ...}}}
    """
    if methods is None:
        methods = METHODS
    if ratios is None:
        ratios = RATIOS
    if seeds is None:
        seeds = [42, 123]

    os.makedirs(output_dir, exist_ok=True)
    ratio_results = {m: {} for m in methods}

    import yaml
    import copy
    from experiments.run_experiment import load_config
    from attacks.adversarial import ATTACK_CONFIGS

    base_config = load_config(config_path)

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")
        print(f"{'='*60}")

        for ratio in ratios:
            print(f"\n  Adversarial ratio: {ratio:.0%}")
            seed_results = []

            for seed in seeds:
                # Modify config to set correct adversarial ratio
                exp_config = copy.deepcopy(base_config)
                exp_config["adversarial"]["attack_ratio"] = ratio

                tmp_path = os.path.join(
                    tempfile.gettempdir(), f"ratio_sweep_{ratio}_{seed}.yaml"
                )
                with open(tmp_path, "w") as f:
                    yaml.dump(exp_config, f)

                # Determine attack name
                if ratio == 0.0:
                    attack_name = "no_attack"
                else:
                    attack_name = "label_flip_30"   # type; ratio controlled above

                result = run_experiment(
                    strategy_name=method,
                    attack_config_name=attack_name,
                    seed=seed,
                    num_rounds=num_rounds,
                    config_path=tmp_path,
                    log_dir=(f"results/logs/ratio_sweep/"
                              f"{method}_ratio{int(ratio*100)}_seed{seed}"),
                    verbose=False,
                )
                seed_results.append(result)

            accs = [r.get("final_accuracy", 0.0) for r in seed_results]
            asrs = [r.get("final_attack_success_rate", 0.0) for r in seed_results]

            ratio_results[method][ratio] = {
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std":  float(np.std(accs)),
                "asr_mean":      float(np.mean(asrs)),
                "asr_std":       float(np.std(asrs)),
            }
            print(f"    Acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
                  f"ASR={np.mean(asrs):.4f}±{np.std(asrs):.4f}")

    # Save results
    out_path = os.path.join(output_dir, "ratio_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(ratio_results, f, indent=2, default=str)
    print(f"\n[Ratio Sweep] Saved to {out_path}")

    # Generate Figure 3
    try:
        from evaluation.visualization import figure3_robustness_curve
        # Convert string keys to float for visualization
        vis_data = {
            m: {float(r): v for r, v in data.items()}
            for m, data in ratio_results.items()
        }
        figure3_robustness_curve(
            vis_data,
            save_path="results/figures/fig3_robustness_curve.pdf",
        )
    except Exception as e:
        print(f"[Warning] Figure 3 generation failed: {e}")

    return ratio_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adversarial ratio sweep")
    parser.add_argument("--methods", nargs="+", default=["fedavg", "tvflids"])
    parser.add_argument("--ratios",  nargs="+", type=float,
                        default=[0.0, 0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--seeds",   nargs="+", type=int, default=[42, 123])
    parser.add_argument("--rounds",  type=int, default=50)
    parser.add_argument("--output",  default="results/tables")
    args = parser.parse_args()

    run_ratio_sweep(
        methods=args.methods,
        ratios=args.ratios,
        seeds=args.seeds,
        num_rounds=args.rounds,
        output_dir=args.output,
    )
