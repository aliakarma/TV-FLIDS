"""
theory/convergence_analysis.py
Empirical convergence rate analysis for TV-FLIDS.

Fits model: acc(t) = L_inf - (L_inf - L0) * exp(-t / tau)
tau = convergence time constant (rounds to 95% of asymptote).
Smaller tau = faster convergence under attack.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, List


def fit_convergence_curve(accuracies: List[float]) -> Dict:
    """
    Fit exponential convergence model to an accuracy series.

    Returns:
        L_inf:              Asymptotic accuracy
        L0:                 Initial accuracy
        tau:                Convergence time constant (rounds)
        r2:                 Goodness of fit (R-squared)
        converges_by_round: Round at which 95% of L_inf is reached
    """
    t = np.arange(len(accuracies), dtype=float)
    y = np.array(accuracies, dtype=float)

    def model(t, L_inf, L0, tau):
        return L_inf - (L_inf - L0) * np.exp(-t / (tau + 1e-8))

    try:
        p0 = [max(y), y[0], len(y) / 3.0]
        popt, _ = curve_fit(model, t, y, p0=p0, maxfev=5000)
        y_pred = model(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-10))
        return {
            "L_inf": float(popt[0]),
            "L0": float(popt[1]),
            "tau": float(popt[2]),
            "r2": r2,
            "converges_by_round": int(popt[2] * 3),
        }
    except Exception as exc:
        return {"error": str(exc)}


def compare_convergence_rates(
    round_metrics_dict: Dict[str, List[Dict]],
) -> Dict[str, Dict]:
    """
    Compare tau across all methods from round_metrics dicts.

    Args:
        round_metrics_dict: {method: [{"accuracy": v, ...}, ...]}

    Returns:
        {method: convergence_fit_result}
    """
    results = {}
    for method, metrics in round_metrics_dict.items():
        accs = [m.get("accuracy", 0.0) for m in metrics]
        results[method] = fit_convergence_curve(accs)
    return results
