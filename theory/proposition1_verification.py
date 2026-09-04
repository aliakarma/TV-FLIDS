"""
theory/proposition1_verification.py
Numerically verify Proposition 1 (Bounded Byzantine Influence), in the
steady-state specialization of Corollary 1.

ACCEPTED-SET NOTATION (paper Section V-A). Every set below is the Stage-1
*accepted* set, not the client population:

    A          the accepted set (verified + flagged), i.e. exactly the index
               set Eq. (15) aggregates over
    H_A        accepted honest clients,      N_{H,A} = |H_A|
    B_A        accepted Byzantine clients,   f_A     = |B_A|
    tau_bar_{H,A}   mean trust over H_A
    tau^max_{B,A}   max trust over B_A

Proposition 1 (general form, Eq. (18)) states, for N_{H,A} >= 1:

    ||w_TV - w*||_2 <= (f_A * tau^max_{B,A}) / (N_{H,A} * tau_bar_{H,A})
                       * max_{j in B_A} ||w_j - w*||_2

This module checks the Corollary 1 specialization tau^max_{B,A} = tau_min,
which is why `tau_min` (not a realized Byzantine maximum) appears in
`theoretical_bound` below: the synthetic harness in `run_verification_suite`
pins every Byzantine trust score at the floor.

DOMAIN. The proposition assumes N_{H,A} >= 1. With H_A empty, w* (Eq. (16))
and tau_bar_{H,A} are both undefined and the bound's denominator is zero, so
the statement has no content there. `verify_proposition1` reports that case as
outside the proposition's domain rather than returning a number.

The B_A = emptyset case IS inside the domain: the paper's stated convention
makes the right-hand side 0, w_TV = w*, and the bound holds with equality.

Reference: Guide §7.4
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
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
    Numerically check the Proposition 1 bound (Corollary 1 specialization).

    All client index arguments are ACCEPTED-set members (Section V-A): they
    are the clients Eq. (15) actually aggregates over, not the population.

    Args:
        trust_scores:     Array of T_i for all clients.
        honest_ids:       Indices of accepted honest clients (H_A).
        byzantine_ids:    Indices of accepted Byzantine clients (B_A).
        global_params:    Current global model parameters.
        honest_params:    Parameters from the accepted honest clients.
        byzantine_params: Parameters from the accepted Byzantine clients.
        tau_min:          Trust score floor, standing in for tau^max_{B,A}
                          under the Corollary 1 steady-state specialization.

    Returns:
        Dict with observed deviation, theoretical bound, and verification
        status; or, when H_A is empty, a dict with `outside_domain: True`
        recording that the proposition's N_{H,A} >= 1 hypothesis fails.
    """
    def flatten_params(params_list: List[List[np.ndarray]]) -> np.ndarray:
        return np.array([np.concatenate([p.flatten() for p in ps]) for ps in params_list])

    def flatten_single(params: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([p.flatten() for p in params])

    n_h = len(honest_ids)
    n_b = len(byzantine_ids)
    f   = n_b

    # Proposition 1 hypothesis: N_{H,A} >= 1. The accepted-set rescoping of
    # Section V-A makes H_A = emptyset reachable (Stage 1 can in principle
    # reject every honest client), and there the proposition simply does not
    # apply: w* is undefined, tau_bar_{H,A} is undefined, and the bound's
    # denominator N_{H,A} * tau_bar_{H,A} is 0. Report it as out-of-domain
    # rather than as a failed or passed verification.
    if n_h == 0:
        return {
            'error': 'No honest clients',
            'outside_domain': True,
            'domain_condition': 'N_{H,A} >= 1',
            'reason': ('Proposition 1 assumes at least one accepted honest '
                       'client; with H_A empty, w* and tau_bar_{H,A} are '
                       'undefined and the bound has no content.'),
            'N_H': 0,
            'f': f,
        }

    # τ̄_H: mean trust score of honest clients
    tau_bar_H = float(np.mean(trust_scores[np.array(honest_ids)]))

    # w* = normalized honest aggregate
    honest_trust = trust_scores[np.array(honest_ids)]
    honest_weights = honest_trust / honest_trust.sum()
    honest_flat = flatten_params(honest_params)
    w_star_flat = np.sum(honest_flat * honest_weights[:, None], axis=0)

    # TV-FLIDS aggregate over the accepted set A = H_A + B_A (Eq. (15)).
    # B_A = emptyset is a legitimate case, not an error: the proposition's
    # stated convention is that the right-hand side of Eq. (18) is 0 and
    # w^(t+1) = w*, so the bound holds with equality. np.vstack cannot be used
    # unconditionally here -- flatten_params([]) has shape (0,), not (0, D),
    # so vstack raised a dimension-mismatch ValueError on exactly the case the
    # proposition documents as well defined.
    all_ids  = honest_ids + byzantine_ids
    if n_b > 0:
        all_flat = np.vstack([flatten_params(honest_params),
                              flatten_params(byzantine_params)])
    else:
        all_flat = flatten_params(honest_params)
    all_trust = trust_scores[np.array(all_ids)]
    agg_weights = all_trust / all_trust.sum()
    w_tv_flat = np.sum(all_flat * agg_weights[:, None], axis=0)

    # Observed deviation
    observed_deviation = float(np.linalg.norm(w_tv_flat - w_star_flat))

    # Theoretical bound. With B_A empty there is no Byzantine perturbation to
    # maximize over, so max_byz_deviation is 0 by the same convention and the
    # bound collapses to 0 alongside the observed deviation.
    if n_b > 0:
        byz_flat = flatten_params(byzantine_params)
        max_byz_deviation = float(
            np.max(np.linalg.norm(byz_flat - w_star_flat[None, :], axis=1)))
    else:
        max_byz_deviation = 0.0
    theoretical_bound = (f * tau_min) / (n_h * tau_bar_H + 1e-8) * max_byz_deviation

    holds = observed_deviation <= theoretical_bound + 1e-8

    return {
        'observed_deviation':  observed_deviation,
        'theoretical_bound':   theoretical_bound,
        'bound_holds':         holds,
        'outside_domain':      False,
        # B_A = emptyset: max_byz_deviation = 0 so the bound is 0, and w_TV = w*
        # so observed_deviation is 0 too -- the documented equality convention.
        'byzantine_set_empty': n_b == 0,
        'f':                   f,
        'N_H':                 n_h,
        'tau_bar_H':           tau_bar_H,
        'tau_min':             tau_min,
        'max_byz_deviation':   max_byz_deviation,
        'bound_ratio':         observed_deviation / (theoretical_bound + 1e-8),
    }


def verify_trust_convergence(
    trust_history: dict,
    honest_ids: list,
    byzantine_ids: list,
) -> dict:
    """
    Lemma 1 (Trust Convergence): Verifies empirically that over rounds:
      - Mean honest client trust trends upward (honest_trust_trend > 0)
      - Mean Byzantine client trust trends downward (byzantine_trust_trend < 0)

    This makes the Proposition 1 bound non-trivial over time.
    """
    import numpy as np

    def mean_history(ids):
        histories = [trust_history[i] for i in ids if i in trust_history]
        if not histories:
            return np.array([])
        min_len = min(len(h) for h in histories)
        return np.mean([h[:min_len] for h in histories], axis=0)

    honest_mean  = mean_history(honest_ids)
    byz_mean     = mean_history(byzantine_ids)

    if len(honest_mean) < 3 or len(byz_mean) < 3:
        return {"error": "Insufficient rounds for trend analysis (need ≥ 3)"}

    t = np.arange(len(honest_mean))
    honest_trend = float(np.polyfit(t, honest_mean, 1)[0])
    byz_trend    = float(np.polyfit(t[:len(byz_mean)], byz_mean, 1)[0])

    return {
        "honest_final_mean_trust":    float(honest_mean[-1]),
        "byzantine_final_mean_trust": float(byz_mean[-1]),
        "honest_trust_trend":         honest_trend,
        "byzantine_trust_trend":      byz_trend,
        "trust_separation":           float(honest_mean[-1] - byz_mean[-1]),
        "lemma1_holds":               honest_trend >= 0 and byz_trend <= 0,
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


def run_verification_suite(n_configs: int = 10, seed: int = 42) -> Dict:
    """
    Run Proposition 1 verification across multiple random configurations.
    Reports how often the bound holds and by what margin.
    """
    from models.mlp import IDSMLP
    model = IDSMLP(41, 5)
    global_params = model.get_parameters()

    results = []
    np.random.seed(seed)

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

    # Per-config records (observed deviation, theoretical bound, ratio, f, N_H)
    # so downstream text can cite a specific configuration's real numbers
    # (e.g. "the tightest-bound configuration") instead of only the ratio.
    per_config = [
        {
            'config_index':      i,
            'f':                 r.get('f'),
            'N_H':                r.get('N_H'),
            'observed_deviation': r.get('observed_deviation'),
            'theoretical_bound':  r.get('theoretical_bound'),
            'bound_ratio':        r.get('bound_ratio'),
            'bound_holds':        r.get('bound_holds'),
        }
        for i, r in enumerate(results)
    ]
    tightest_by_ratio = max(per_config, key=lambda r: r['bound_ratio'])  # ratio closest to 1
    smallest_abs_dev = min(per_config, key=lambda r: r['observed_deviation'])

    summary = {
        'configs_tested':   n_configs,
        'seed':             seed,
        'bound_holds':      n_holds,
        'bound_holds_pct':  n_holds / n_configs * 100,
        'mean_ratio':       float(np.mean(ratios)),
        'std_ratio':        float(np.std(ratios)),
        'max_ratio':        float(np.max(ratios)),
        'min_ratio':        float(np.min(ratios)),
        'all_ratios':       ratios,
        'per_config':       per_config,
        'tightest_bound_config': tightest_by_ratio,       # ratio closest to 1
        'smallest_absolute_deviation_config': smallest_abs_dev,
        'verification_pass': n_holds == n_configs,
    }

    print('\n[Proposition 1 Verification]')
    print(f"  Configs tested:     {n_configs}")
    print(f"  Seed:               {seed}")
    print(f"  Bound holds:        {n_holds}/{n_configs} ({summary['bound_holds_pct']:.0f}%)")
    print(f"  Mean observed/bound:{summary['mean_ratio']:.4f} (+/- {summary['std_ratio']:.4f})")
    print(f"  Max observed/bound: {summary['max_ratio']:.4f}")
    print(f"  PASS: {summary['verification_pass']}")
    return summary


def _provenance() -> Dict:
    """Real run provenance: git commit, timestamp, script path, python/numpy version.
    Deliberately not a hardcoded/mock string (cf. the previously fabricated
    proposition1_real.json, which carried the string "mockhash" nowhere but
    whose sibling mock-generated files did — see results/_QUARANTINED_MOCK/).
    """
    import subprocess, sys, datetime, platform
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except Exception:
        commit = 'unknown (git unavailable)'
    return {
        'script': 'theory/proposition1_verification.py',
        'git_commit': commit,
        'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'python_version': sys.version.split()[0],
        'numpy_version': np.__version__,
        'platform': platform.platform(),
    }


def run_and_save(n_configs: int = 20, seed: int = 42,
                  out_path: str = 'results/tables/proposition1_real.json') -> Dict:
    """Run the verification suite for real and persist output with real
    provenance, to the exact path the paper's Prop. 1 claim should trace to.
    """
    summary = run_verification_suite(n_configs=n_configs, seed=seed)
    payload = {**summary, 'provenance': _provenance()}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Proposition 1] Saved real, provenanced result to {out_path}")
    return payload


if __name__ == '__main__':
    run_and_save(n_configs=20, seed=42)
