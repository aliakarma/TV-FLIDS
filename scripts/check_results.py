"""
scripts/check_results.py
Pre-submission result completeness check.

Run: python scripts/check_results.py
"""
import json, os, sys

REQUIRED_FILES = [
    "results/tables/full_comparison_results.json",
    "results/tables/ablation_results.json",
    "results/tables/ratio_sweep_results.json",
    "results/figures/fig1_convergence.pdf",
    "results/figures/fig2_trust_evolution.pdf",
    "results/figures/fig3_robustness_curve.pdf",
    "results/figures/fig4_ablation.pdf",
    "results/figures/fig5_adaptive_weights.pdf",
    "results/figures/fig6_confusion.pdf",
]

REQUIRED_STRATEGIES = ["fedavg", "krum", "trimmed_mean", "fltrust",
                        "foolsgold", "flame", "rfa", "tvflids"]
REQUIRED_SEEDS = [42, 123, 456, 789, 1337]

def check():
    ok = True
    print("=== Result Completeness Check ===\n")

    for f in REQUIRED_FILES:
        exists = os.path.exists(f)
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {f}")
        if not exists:
            ok = False

    # Check full_comparison_results.json content
    cmp_path = "results/tables/full_comparison_results.json"
    if os.path.exists(cmp_path):
        data = json.load(open(cmp_path))
        raw = data.get("raw", {})
        print("\n=== Strategy × Seed Coverage ===\n")
        for strategy in REQUIRED_STRATEGIES:
            results = raw.get(strategy, [])
            seeds_found = [r.get("seed") for r in results]
            missing = [s for s in REQUIRED_SEEDS if s not in seeds_found]
            if missing:
                print(f"  [INCOMPLETE] {strategy}: missing seeds {missing}")
                ok = False
            else:
                accs = [r.get("final_accuracy", 0) for r in results]
                print(f"  [OK] {strategy}: {len(results)} seeds, "
                      f"acc={sum(accs)/len(accs):.4f}")

    print(f"\n{'[PASS] All checks passed.' if ok else '[FAIL] Fix missing items.'}")
    return ok

if __name__ == "__main__":
    sys.exit(0 if check() else 1)
