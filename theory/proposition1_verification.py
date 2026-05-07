"""
theory/proposition1_verification.py
Numerically verify Proposition 1: Bounded Byzantine Influence.

Proposition 1 states:
    ||w_TV - w*||_2 ≤ (f · τ_min) / (N_H · τ̄_H) · max_{i∈B} ||w_i - w*||_2

Reference: Guide §7.4
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import Dict, List


def verify_proposition1(
    trust_scores: np.ndarray,
    honest_ids: List[int],
    byzantine_ids: List[int],
    global_params: List[np.ndarray],
    honest_params: List[List[np.ndarray]],
    byzantine_params: List[List[np.ndarray]],
    tau_min: float = 0.01,
) -> Dict:
    """
    Numerically check the Proposition 1 bound.

    Args:
        trust_scores:     Array of T_i for all clients.
        honest_ids:       Indices of honest clients.
        byzantine_ids:    Indices of Byzantine clients.
        global_params:    Current global model parameters.
        honest_params:    Parameters from honest clients.
        byzantine_params: Parameters from Byzantine clients.
        tau_min:          Trust score floor.

    Returns:
        Dict with observed deviation, theoretical bound, and verification status.
    """
    def flatten_params(params_list: List[List[np.ndarray]]) -> np.ndarray:
        return np.array([np.concatenate([p.flatten() for p in ps]) for ps in params_list])

    def flatten_single(params: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([p.flatten() for p in params])

    n_h = len(honest_ids)
    n_b = len(byzantine_ids)
    f   = n_b

    if n_h == 0:
        return {'error': 'No honest clients'}

    # τ̄_H: mean trust score of honest clients
    tau_bar_H = float(np.mean(trust_scores[np.array(honest_ids)]))

    # w* = normalized honest aggregate
    honest_trust = trust_scores[np.array(honest_ids)]
    honest_weights = honest_trust / honest_trust.sum()
    honest_flat = flatten_params(honest_params)
    w_star_flat = np.sum(honest_flat * honest_weights[:, None], axis=0)

    # TV-FLIDS aggregate (all participating clients)
    all_ids  = honest_ids + byzantine_ids
    all_flat = np.vstack([flatten_params(honest_params), flatten_params(byzantine_params)])
    all_trust = trust_scores[np.array(all_ids)]
    agg_weights = all_trust / all_trust.sum()
    w_tv_flat = np.sum(all_flat * agg_weights[:, None], axis=0)

    # Observed deviation
    observed_deviation = float(np.linalg.norm(w_tv_flat - w_star_flat))

    # Theoretical bound
    byz_flat = flatten_params(byzantine_params)
    max_byz_deviation = float(np.max(np.linalg.norm(byz_flat - w_star_flat[None, :], axis=1))) \
        if n_b > 0 else 0.0
    theoretical_bound = (f * tau_min) / (n_h * tau_bar_H + 1e-8) * max_byz_deviation

    holds = observed_deviation <= theoretical_bound + 1e-8

    return {
        'observed_deviation':  observed_deviation,
        'theoretical_bound':   theoretical_bound,
        'bound_holds':         holds,
        'f':                   f,
        'N_H':                 n_h,
        'tau_bar_H':           tau_bar_H,
        'tau_min':             tau_min,
        'max_byz_deviation':   max_byz_deviation,
        'bound_ratio':         observed_deviation / (theoretical_bound + 1e-8),
    }


def verify_from_experiment_log(experiment_log_path: str, strategy_ref) -> Dict:
    """
    Verify Proposition 1 using actual trust scores and parameters from a live run.

    Args:
        experiment_log_path: Path to the experiment log JSON (for traceability).
        strategy_ref:         Live strategy instance with _last_round_data.
    """
    last_data = getattr(strategy_ref, "_last_round_data", None)
    if not last_data:
        return {
            "error": "No logged client params. Enable log_client_params in config.",
            "experiment_log": experiment_log_path,
        }

    result = verify_proposition1(
        trust_scores=last_data["trust_scores"],
        honest_ids=last_data["honest_ids"],
        byzantine_ids=last_data["byzantine_ids"],
        global_params=strategy_ref.model.get_parameters(),
        honest_params=[last_data["client_params"][i] for i in last_data["honest_ids"]],
        byzantine_params=[last_data["client_params"][i] for i in last_data["byzantine_ids"]],
        tau_min=strategy_ref.config["trust"]["min_trust"],
    )
    result["experiment_log"] = experiment_log_path
    return result


def run_verification_suite(n_configs: int = 10) -> Dict:
    """
    Run Proposition 1 verification across multiple random configurations.
    Reports how often the bound holds and by what margin.
    """
    from models.mlp import IDSMLP
    model = IDSMLP(41, 5)
    global_params = model.get_parameters()

    results = []
    np.random.seed(42)

    for _ in range(n_configs):
        n_total  = 20
        n_byz    = np.random.randint(1, 7)  # 1-6 Byzantine (5-30%)
        n_honest = n_total - n_byz
        tau_min  = 0.01

        honest_ids   = list(range(n_honest))
        byzantine_ids = list(range(n_honest, n_total))

        # Simulate trust scores: honest ≈ 0.7-1.0, Byzantine ≈ tau_min
        trust_scores = np.ones(n_total)
        trust_scores[np.array(honest_ids)] = np.random.uniform(0.6, 1.0, n_honest)
        trust_scores[np.array(byzantine_ids)] = tau_min

        # Simulate client parameters
        def perturb_params(scale: float) -> List[np.ndarray]:
            return [p + np.random.randn(*p.shape).astype(np.float32) * scale
                    for p in global_params]

        honest_params   = [perturb_params(0.05) for _ in range(n_honest)]
        byzantine_params = [perturb_params(2.0) for _ in range(n_byz)]  # Large perturbation

        result = verify_proposition1(
            trust_scores, honest_ids, byzantine_ids,
            global_params, honest_params, byzantine_params, tau_min=tau_min
        )
        results.append(result)

    n_holds = sum(1 for r in results if r.get('bound_holds', False))
    ratios  = [r['bound_ratio'] for r in results if 'bound_ratio' in r]

    summary = {
        'configs_tested':   n_configs,
        'bound_holds':      n_holds,
        'bound_holds_pct':  n_holds / n_configs * 100,
        'mean_ratio':       float(np.mean(ratios)),
        'max_ratio':        float(np.max(ratios)),
        'verification_pass': n_holds == n_configs,
    }

    print('\n[Proposition 1 Verification]')
    print(f"  Configs tested:     {n_configs}")
    print(f"  Bound holds:        {n_holds}/{n_configs} ({summary['bound_holds_pct']:.0f}%)")
    print(f"  Mean observed/bound:{summary['mean_ratio']:.4f}")
    print(f"  Max observed/bound: {summary['max_ratio']:.4f}")
    print(f"  PASS: {summary['verification_pass']}")
    return summary


if __name__ == '__main__':
    run_verification_suite(n_configs=20)
