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
    build_results_table_extended, compare_methods_wilcoxon, SEEDS
)
from evaluation.visualization import (
    figure1_convergence_curves, figure6_confusion_matrices
)
from data.preprocessing.nslkdd_pipeline import CLASS_NAMES


STRATEGIES = [
    "fedavg", "krum", "trimmed_mean", "fltrust",
    "foolsgold", "flame", "rfa", "tvflids",
]
# "bucketing" and "deepsight" (fl/baselines/bucketing_strategy.py,
# deepsight_strategy.py) are implemented and selectable via --strategy but
# intentionally left out of this default list: including them here would
# widen the default (paper's) 8-strategy x 5-seed comparison run, and this
# remediation pass is explicitly scoped to NOT trigger a full-scale
# experimental campaign. Pass them explicitly via --strategies to include.
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
    protocol: str = "main",
    dataset: str = "nslkdd",
) -> dict:
    """
    Run all strategies across all seeds for the primary attack.

    Args:
        protocol: "main" (paper Table V: global SMOTE, then D_val drawn from
            the balanced pool) or "leakage_free" (paper Section VIII-A /
            Table VI: D_val drawn pre-SMOTE, SMOTE applied per client after
            partitioning). Previously this runner had no protocol argument at
            all, so Table VI was only reachable by invoking run_experiment.py
            once per (strategy, seed) cell by hand.
        dataset: "nslkdd" (default), "unswnb15" or "ciciot2023".

    Returns:
        {strategy_name: [result_per_seed]}
    """
    if protocol not in ("main", "leakage_free"):
        raise ValueError(f"Unknown protocol '{protocol}'. "
                         "Choose 'main' or 'leakage_free'.")
    if strategies is None:
        strategies = STRATEGIES
    if seeds is None:
        # Paper protocol (audit Priority 3 fix): all 5 seeds by default.
        # Pass --seeds to reduce for a smoke test.
        seeds = SEEDS

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    # Keep protocol/dataset variants in separate log and output namespaces so a
    # leakage-free run can never overwrite the main-protocol Table V logs.
    suffix = "" if (protocol == "main" and dataset == "nslkdd") else f"_{dataset}_{protocol}"
    log_root = f"results/logs/comparison{suffix}"

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
                log_dir=f"{log_root}/{strategy}_{attack}_seed{seed}",
                dataset=dataset,
                protocol=protocol,
                verbose=False,
            )
            seed_results.append(result)
            if verbose:
                print(f"  Seed {seed}: acc={result.get('final_accuracy', 0):.4f}")

        all_results[strategy] = seed_results

    # Print publication-ready extended table
    print("\n\n" + "="*80)
    print("RESULTS TABLE (mean ± std, 95% CI, Cohen's d, Wilcoxon p)")
    print("="*80)
    table = build_results_table_extended(
        all_results,
        metrics=METRICS,
        reference_method="tvflids",
    )
    for method, row in table.items():
        print(f"\n{method}:")
        for metric in METRICS:
            m = row[metric]
            p_val = m["wilcoxon_p"]
            p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
            sig = "*" if m["significant"] else "ns"
            print(f"  {metric:<26} {m['formatted']:<45} p={p_str} {sig}")

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

    # Save. Non-default protocol/dataset variants get their own filename so a
    # leakage-free or cross-dataset run cannot silently overwrite the
    # main-protocol Table V artifact.
    out_path = os.path.join(output_dir, f"full_comparison_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({
            "raw": all_results,
            "table": table,
            "protocol": protocol,
            "dataset": dataset,
            "attack": attack,
            "seeds": seeds,
            "num_rounds": num_rounds,
        }, f, indent=2, default=str)
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

        # ── Figure 2 (canonical path) ─────────────────────────────
        # scripts/check_results.py requires results/figures/fig2_trust_evolution.pdf,
        # but every call site of figure2_trust_evolution passed a per-run name
        # (fig2_trust_<strategy>_seed<seed>.pdf), so the required file could
        # never appear and `make check-results` could not pass even after a
        # complete campaign. This is the same orphaned-output defect the
        # closure pass fixed for Figures 4 and 5. Written from the TV-FLIDS
        # run's own persisted trust history, or skipped if that run has none.
        tv_log_path = os.path.join(
            log_root, f"tvflids_{attack}_seed{seed_for_figs}", "experiment_log.json"
        )
        if os.path.exists(tv_log_path):
            with open(tv_log_path, "r") as f:
                tv_blob = json.load(f)
            extra = tv_blob.get("extra", {}) or {}
            trust_history = extra.get("trust_history") or {}
            malicious_ids = extra.get("malicious_ids") or []
            if trust_history:
                from evaluation.visualization import figure2_trust_evolution
                figure2_trust_evolution(
                    {int(cid): hist for cid, hist in trust_history.items()},
                    [int(c) for c in malicious_ids],
                    save_path="results/figures/fig2_trust_evolution.pdf",
                )
            else:
                print("[Comparison] Figure 2 skipped: the TV-FLIDS run carries "
                      "no trust_history (rerun against the current logger).")

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
    # Paper protocol (audit Priority 3 fix): all 5 seeds by default.
    parser.add_argument("--seeds",      nargs="+", type=int, default=SEEDS)
    parser.add_argument("--rounds",     type=int, default=50)
    parser.add_argument("--config",     default="config/fl_config.yaml")
    parser.add_argument("--output",     default="results/tables")
    parser.add_argument("--dataset",    default="nslkdd",
                        choices=["nslkdd", "unswnb15", "ciciot2023"])
    parser.add_argument("--protocol",   default="main",
                        choices=["main", "leakage_free"],
                        help="'main' = paper Table V; 'leakage_free' = paper "
                             "Section VIII-A / Table VI (D_val drawn pre-SMOTE, "
                             "SMOTE applied per client after partitioning)")
    args = parser.parse_args()

    run_full_comparison(
        strategies=args.strategies,
        attack=args.attack,
        seeds=args.seeds,
        num_rounds=args.rounds,
        config_path=args.config,
        output_dir=args.output,
        dataset=args.dataset,
        protocol=args.protocol,
    )
