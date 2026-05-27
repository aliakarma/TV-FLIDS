"""
scripts/run_theory_validation.py
Run all theoretical verification checks.

Usage:
    python scripts/run_theory_validation.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theory.proposition1_verification import run_verification_suite, verify_trust_convergence
from theory.convergence_analysis import compare_convergence_rates


def run_all():
    print("=" * 60)
    print("Theory Validation Suite")
    print("=" * 60)

    # 1. Proposition 1 on synthetic data
    print("\n[1/3] Proposition 1 — Synthetic verification (20 configs)")
    suite = run_verification_suite(n_configs=20)
    assert suite["verification_pass"], \
        f"FAIL: Prop 1 failed on {20 - suite['bound_holds']}/20 synthetic configs"
    print(f"  PASS: Bound holds on {suite['bound_holds']}/20 configs")

    # 2. Proposition 1 on real data
    prop1_path = "results/tables/proposition1_real.json"
    print(f"\n[2/3] Proposition 1 — Real experimental data ({prop1_path})")
    if not os.path.exists(prop1_path):
        print("  SKIP: File not found. Run with log_client_params=true first.")
    else:
        result = json.load(open(prop1_path))
        if result.get("error"):
            print(f"  FAIL: {result['error']}")
        else:
            holds = result["bound_holds"]
            ratio = result["bound_ratio"]
            print(f"  {'PASS' if holds else 'FAIL'}: bound_holds={holds}, ratio={ratio:.4f}")

    # 3. Convergence analysis
    cmp_path = "results/tables/full_comparison_results.json"
    print(f"\n[3/3] Convergence Rate Analysis")
    if not os.path.exists(cmp_path):
        print("  SKIP: Run full comparison first.")
    else:
        data = json.load(open(cmp_path))
        # Load round metrics from individual log files
        round_metrics = {}
        log_root = "results/logs/comparison"
        for strategy in ["fedavg", "fltrust", "tvflids"]:
            log_path = os.path.join(log_root, f"{strategy}_label_flip_30_seed42",
                                    "experiment_log.json")
            if os.path.exists(log_path):
                log_data = json.load(open(log_path))
                round_metrics[strategy] = log_data.get("rounds", [])

        if round_metrics:
            rates = compare_convergence_rates(round_metrics)
            print(f"\n  {'Method':<20} {'tau':>8} {'L_inf':>8} {'R2':>8}")
            print("  " + "-" * 45)
            for method, fit in rates.items():
                if "error" not in fit:
                    print(f"  {method:<20} {fit['tau']:>8.1f} "
                          f"{fit['L_inf']:>8.4f} {fit['r2']:>8.3f}")
        else:
            print("  SKIP: No round-level log files found.")

    print("\n[Theory Validation Complete]")


if __name__ == "__main__":
    run_all()
