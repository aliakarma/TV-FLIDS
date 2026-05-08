"""
evaluation/statistical_testing.py
Statistical significance testing for TV-FLIDS results.

Required for IEEE/ACM publication:
  - 5-seed mean ± std reporting
  - Wilcoxon signed-rank test (non-parametric, paired)
  - McNemar's test (classifier comparison on same test set)

Reference: Guide §10.3
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from scipy.stats import wilcoxon

SEEDS = [42, 123, 456, 789, 1337]


def run_with_seeds(
    experiment_fn: Callable[[int], Dict],
    seeds: List[int] = SEEDS,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run an experiment across multiple seeds and collect results.

    Args:
        experiment_fn: callable(seed) → dict with metric keys.
        seeds:         List of integer seeds.
        verbose:       Print progress.

    Returns:
        List of result dicts, one per seed.
    """
    results = []
    for seed in seeds:
        if verbose:
            print(f"[Seeds] Running seed={seed} ...")
        np.random.seed(seed)
        r = experiment_fn(seed=seed)
        results.append(r)
        if verbose:
            acc = r.get("final_accuracy", r.get("accuracy", "N/A"))
            print(f"  → accuracy={acc}")
    return results


def compute_summary(results: List[Dict], metric: str) -> Tuple[float, float]:
    """Compute mean ± std for a metric across seeds."""
    vals = [r[metric] for r in results if metric in r]
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def format_result(mean: float, std: float, decimals: int = 4) -> str:
    """Format as 'mean ± std' for paper tables."""
    fmt = f".{decimals}f"
    return f"{mean:{fmt}} ± {std:{fmt}}"


def compare_methods_wilcoxon(
    results_a: List[Dict],
    results_b: List[Dict],
    metric: str = "final_accuracy",
    alpha: float = 0.05,
) -> Dict:
    """
    Wilcoxon signed-rank test (non-parametric, paired).

    Use to compare TV-FLIDS vs each baseline across 5 seeds.
    H0: no difference in median performance.
    If p < alpha: TV-FLIDS is significantly better.

    Returns dict with statistic, p_value, significant flag.
    """
    vals_a = [r[metric] for r in results_a if metric in r]
    vals_b = [r[metric] for r in results_b if metric in r]

    if len(vals_a) != len(vals_b) or len(vals_a) < 2:
        return {
            "statistic": 0.0, "p_value": 1.0,
            "significant": False, "error": "Insufficient paired data"
        }

    diffs = np.array(vals_a) - np.array(vals_b)
    if np.all(diffs == 0):
        return {
            "statistic": 0.0, "p_value": 1.0,
            "significant": False, "note": "All differences are zero"
        }

    try:
        stat, p_val = wilcoxon(vals_a, vals_b, alternative="greater")
        return {
            "statistic":        float(stat),
            "p_value":          float(p_val),
            "significant":      p_val < alpha,
            "mean_a":           float(np.mean(vals_a)),
            "mean_b":           float(np.mean(vals_b)),
            "effect_direction": "A > B" if np.mean(vals_a) > np.mean(vals_b) else "B > A",
        }
    except Exception as e:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False, "error": str(e)}


def compute_cohens_d(
    results_a: List[Dict],
    results_b: List[Dict],
    metric: str = "final_accuracy",
) -> float:
    """
    Cohen's d effect size between two method result sets.
    """
    vals_a = np.array([r[metric] for r in results_a if metric in r], dtype=float)
    vals_b = np.array([r[metric] for r in results_b if metric in r], dtype=float)
    if len(vals_a) == 0 or len(vals_b) == 0:
        return 0.0
    pooled_std = np.sqrt((np.std(vals_a) ** 2 + np.std(vals_b) ** 2) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(vals_a) - np.mean(vals_b)) / pooled_std)


def compute_bootstrap_ci(
    results: List[Dict],
    metric: str,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for a metric across seeds.
    Returns (lower, upper) bounds at the specified CI level.
    """
    vals = np.array([r[metric] for r in results if metric in r], dtype=float)
    if len(vals) == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        boot_means[i] = np.mean(rng.choice(vals, size=len(vals), replace=True))

    alpha = (1 - ci) / 2
    return (
        float(np.percentile(boot_means, 100 * alpha)),
        float(np.percentile(boot_means, 100 * (1 - alpha))),
    )


def build_results_table_extended(
    experiment_results_dict: Dict[str, List[Dict]],
    metrics: Optional[List[str]] = None,
    reference_method: str = "tvflids",
) -> Dict:
    """
    Full results table:
    mean ± std | 95% CI | Cohen's d | Wilcoxon p
    """
    if metrics is None:
        metrics = ["final_accuracy", "final_f1_macro", "final_attack_success_rate"]

    table = {}
    ref_results = experiment_results_dict.get(reference_method, [])

    for method, results in experiment_results_dict.items():
        row = {}
        for m in metrics:
            mean, std = compute_summary(results, m)
            ci_low, ci_hi = compute_bootstrap_ci(results, m)
            d = compute_cohens_d(ref_results, results, m) if ref_results else 0.0
            wtest = (
                compare_methods_wilcoxon(ref_results, results, m)
                if ref_results and method != reference_method
                else {}
            )
            row[m] = {
                "mean": mean,
                "std": std,
                "ci_95": (ci_low, ci_hi),
                "cohens_d": d,
                "wilcoxon_p": wtest.get("p_value", None),
                "significant": wtest.get("significant", None),
                "formatted": (
                    f"{mean:.4f}±{std:.4f} [{ci_low:.4f},{ci_hi:.4f}] d={d:.2f}"
                ),
            }
        table[method] = row

    return table


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict:
    """
    McNemar's test for comparing two classifiers on the same test set.

    Compares correctness patterns — more informative than accuracy alone.
    Use to compare TV-FLIDS vs FLTrust on NSL-KDD test set.

    Contingency table:
        Both correct    | A correct, B wrong
        A wrong, B correct | Both wrong

    Returns dict with chi2, p_value, significant flag, table.
    """
    both_correct = int(np.sum((y_pred_a == y_true) & (y_pred_b == y_true)))
    a_only       = int(np.sum((y_pred_a == y_true) & (y_pred_b != y_true)))
    b_only       = int(np.sum((y_pred_a != y_true) & (y_pred_b == y_true)))
    both_wrong   = int(np.sum((y_pred_a != y_true) & (y_pred_b != y_true)))

    # McNemar's test statistic (with continuity correction)
    n12 = a_only
    n21 = b_only
    if n12 + n21 == 0:
        return {
            "chi2": 0.0, "p_value": 1.0, "significant": False,
            "note": "No discordant pairs",
        }

    chi2 = (abs(n12 - n21) - 1) ** 2 / (n12 + n21)

    from scipy.stats import chi2 as chi2_dist
    p_val = float(1 - chi2_dist.cdf(chi2, df=1))

    return {
        "chi2":               float(chi2),
        "p_value":            p_val,
        "significant":        p_val < 0.05,
        "contingency_table":  [[both_correct, a_only], [b_only, both_wrong]],
        "n_discordant":       n12 + n21,
    }


def build_results_table(
    experiment_results_dict: Dict[str, List[Dict]],
    metrics: Optional[List[str]] = None,
    reference_method: Optional[str] = None,
    print_output: bool = True,
) -> Dict[str, Dict[str, str]]:
    """
    Build a publication-ready results table.

    Args:
        experiment_results_dict: {method_name: [results_per_seed]}
        metrics:                 Metric keys to include.
        reference_method:        If set, compute Wilcoxon p-values vs this method.
        print_output:            Print formatted table to stdout.

    Returns:
        Nested dict: {method: {metric: "mean ± std"}}
    """
    if metrics is None:
        metrics = ["final_accuracy", "f1_macro", "attack_success_rate"]

    table: Dict[str, Dict[str, str]] = {}
    p_values: Dict[str, Dict[str, float]] = {}

    ref_results = experiment_results_dict.get(reference_method, []) if reference_method else []

    for method, results in experiment_results_dict.items():
        table[method] = {}
        p_values[method] = {}
        for m in metrics:
            mean, std = compute_summary(results, m)
            table[method][m] = format_result(mean, std)
            if ref_results and method != reference_method:
                wtest = compare_methods_wilcoxon(ref_results, results, m)
                p_values[method][m] = wtest.get("p_value", 1.0)

    if print_output:
        col_w = 30
        header = f"{'Method':<20}" + "".join(f"  {m:<{col_w}}" for m in metrics)
        print(header)
        print("-" * len(header))
        for method, row in table.items():
            line = f"{method:<20}"
            for m in metrics:
                cell = row.get(m, "N/A")
                pv = p_values.get(method, {}).get(m)
                marker = " †" if pv is not None and pv < 0.05 else "  "
                line += f"  {cell + marker:<{col_w}}"
            print(line)
        if reference_method:
            print(f"\n† p < 0.05 (Wilcoxon signed-rank vs {reference_method})")

    return table
