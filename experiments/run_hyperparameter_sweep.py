"""
experiments/run_hyperparameter_sweep.py
Baseline and TV-FLIDS hyperparameter sensitivity sweeps
(audit IDs E10 [baseline, paper Supp. Table S1] and
 E11 [TV-FLIDS, paper Supp. Table S2]).

This replaces the previous `audit_hyperparams.py`, which despite its name
only printed the currently-active config values from config/fl_config.yaml
and never swept anything. `audit_hyperparams.py` at the repo root is now a
thin backward-compatible CLI shim that delegates here (`--target baseline`).

--target baseline (audit E10, Supp. Table S1):
    Sweeps Krum's f' (assumed-Byzantine-count parameter, `num_byzantine` in
    KrumStrategy) and Trimmed-Mean's beta trim fraction, each over a
    4-point grid, via the new `strategy_kwargs_override` passthrough added
    to experiments/run_experiment.py::make_strategy() for exactly this
    purpose (None/omitted preserves all prior behavior).

    Krum f' grid: [2, 4, 6, 8]. At the default config (num_clients=20,
    adversarial.attack_ratio=0.3) the *true* Byzantine count under the
    default attack ratio is 20*0.3 = 6, so this grid spans clearly
    under-estimating (2, 4), correctly estimating (6), and over-estimating
    (8) the true adversarial count — the standard sensitivity axis for
    Krum's f' in the literature.

    Trimmed-Mean beta grid: [0.05, 0.10, 0.15, 0.20] (paper-suggested
    range; note the code's *default* beta = min(0.35, attack_ratio + 0.05)
    = 0.35 at the default 30% attack ratio is intentionally more
    conservative than this grid's upper bound — this sweep asks "how
    sensitive is Trimmed-Mean to trimming less than the code's own
    computed default").

--target tvflids (audit E11, Supp. Table S2):
    One-factor-at-a-time sweep of TV-FLIDS's own trust/verification
    hyperparameters, holding all others at their config/fl_config.yaml
    default:
      - trust.memory_decay == lambda  (grid: [0.7, 0.8, 0.9, 0.95], default 0.9)
      - trust.min_trust    == tau_min (grid: [0.001, 0.01, 0.05], default 0.01)
      - trust.meta_lr      == eta_meta (grid: [0.001, 0.01, 0.1], default 0.01)
      - verification.warmup_rounds == T_warm (grid: [5, 10, 20, 30], default 20)

    Symbol mapping note. Paper Supp. Table S2 lists lambda, tau_min,
    eta_meta and T_warm as four separate knobs, and all four exist in code:

      lambda   -> trust.memory_decay (TrustScorer.__init__'s memory_decay,
                  stored as self.decay). It is the EMA retention factor, not
                  a learning rate: the main paper's Section IX quotes its
                  horizon as 1/(1 - lambda) = 10 rounds at lambda = 0.9, and
                  Table S2 labels its block "Memory decay lambda" with the
                  grid {0.7, 0.8, 0.9, 0.95}.
      eta_meta -> trust.meta_lr, passed to
                  torch.optim.Adam([self.log_weights], lr=meta_lr).

    An earlier revision of this module asserted that lambda and eta_meta were
    two names for trust.meta_lr and swept only the latter, which left Table
    S2's memory-decay block unreachable from any runner. That reading cannot
    be right -- a retention factor in [0, 1) with an EMA horizon is not a
    learning rate -- and it is corrected here.

Usage:
    python experiments/run_hyperparameter_sweep.py --target baseline
    python experiments/run_hyperparameter_sweep.py --target tvflids
    python experiments/run_hyperparameter_sweep.py --target baseline --seeds 42 123 --rounds 2  # smoke-scale
"""

import argparse
import copy
import json
import os
import sys
import tempfile
from typing import Callable, Dict, List, Optional

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.run_experiment import run_experiment as _real_run_experiment, load_config
from evaluation.statistical_testing import SEEDS

METRICS = ["final_accuracy", "final_f1_macro", "final_attack_success_rate"]

# ── Baseline grids (audit E10, Supp. Table S1) ───────────────────────────
# Grids are the ones the manuscript states, not a nearby approximation of
# them. Supplementary Table S1 sweeps Krum's assumed tolerance over
# {4, 6, 8, 10} and Trimmed Mean's trim fraction over {0.10, 0.15, 0.20, 0.25},
# each bracketing its default on BOTH sides (f' = 6, beta_TM = 0.15). The grids
# previously coded here ({2,4,6,8} and {0.05,...,0.20}) put each default at the
# top of its range, so the sweep could not have shown a default to be
# conservative -- the direction the text explicitly asks about.
BASELINE_SWEEPS = {
    "krum_f_prime": {
        "strategy": "krum",
        "override_key": "num_byzantine",
        "grid": [4, 6, 8, 10],
        "default": 6,
        "paper_symbol": "Krum f'",
    },
    "trimmed_mean_beta": {
        "strategy": "trimmed_mean",
        "override_key": "beta",
        "grid": [0.10, 0.15, 0.20, 0.25],
        "default": 0.15,
        "paper_symbol": "Trimmed-Mean beta",
    },
}

# ── TV-FLIDS grids (audit E11, Supp. Table S2) ───────────────────────────
# Supplementary Table S2 tabulates four factors: memory decay lambda, trust
# floor tau_min, meta-learning rate eta_meta, and warmup length T_warm. The
# memory-decay factor had no entry here at all, so Table S2's first block was
# unreachable from the runner; and the eta_meta grid stopped at 0.05 where the
# table reports a decade in either direction (0.001, 0.01, 0.1).
#
# T_warm keeps the four-point grid {5, 10, 20, 30} rather than the table's
# three: the campaign specifically needs T_warm = 5 to test the corrected
# annealing schedule, which was disabled in the code path that produced the
# tabulated values. Table S2 gains the fourth row accordingly.
TVFLIDS_SWEEPS = {
    "memory_decay": {
        "section": "trust", "key": "memory_decay",
        "grid": [0.7, 0.8, 0.9, 0.95],
        "default": 0.9,
        "paper_symbol": "lambda (memory decay)",
    },
    "min_trust": {
        "section": "trust", "key": "min_trust",
        "grid": [0.001, 0.01, 0.05],
        "default": 0.01,
        "paper_symbol": "tau_min (trust floor)",
    },
    "meta_lr": {
        "section": "trust", "key": "meta_lr",
        "grid": [0.001, 0.01, 0.1],
        "default": 0.01,
        "paper_symbol": "eta_meta (meta-learning rate)",
    },
    "warmup_rounds": {
        "section": "verification", "key": "warmup_rounds",
        "grid": [5, 10, 20, 30],
        "default": 20,
        "paper_symbol": "T_warm (gate warmup length)",
    },
}


def _aggregate_cell(seed_results: List[Dict]) -> Dict:
    cell = {}
    for metric in METRICS:
        vals = [r.get(metric, 0.0) for r in seed_results]
        cell[metric] = {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "std":  float(np.std(vals)) if vals else 0.0,
            "n_seeds": len(vals),
        }
    return cell


# ── Baseline sweep (E10) ──────────────────────────────────────────────────

def run_baseline_sweep_jobs(
    sweeps: Dict[str, Dict],
    seeds: List[int],
    attack: str,
    num_rounds: int,
    config_path: str,
    log_root: str,
    run_experiment_fn: Callable,
    verbose: bool = True,
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Execute the baseline hyperparameter grid: for each sweep (krum_f_prime,
    trimmed_mean_beta), for each grid point, run all seeds via
    strategy_kwargs_override. Returns {sweep_name: {str(grid_value): [seed results]}}.
    """
    raw: Dict[str, Dict[str, List[Dict]]] = {}
    for sweep_name, spec in sweeps.items():
        raw[sweep_name] = {}
        strategy = spec["strategy"]
        override_key = spec["override_key"]
        for value in spec["grid"]:
            label = str(value)
            if verbose:
                print(f"\n[Baseline Sweep] {sweep_name}={value} (strategy={strategy})")
            seed_results = []
            for seed in seeds:
                result = run_experiment_fn(
                    strategy_name=strategy,
                    attack_config_name=attack,
                    seed=seed,
                    num_rounds=num_rounds,
                    config_path=config_path,
                    log_dir=(f"{log_root}/{sweep_name}_{label}_seed{seed}"),
                    verbose=False,
                    strategy_kwargs_override={override_key: value},
                )
                seed_results.append(result)
                if verbose:
                    print(f"    seed={seed}: acc={result.get('final_accuracy', 0):.4f}")
            raw[sweep_name][label] = seed_results
    return raw


def aggregate_baseline_sweep(raw: Dict[str, Dict[str, List[Dict]]]) -> Dict:
    """Pure aggregation step, unit-testable on canned per-seed dicts."""
    summary: Dict[str, Dict[str, Dict]] = {}
    for sweep_name, by_value in raw.items():
        summary[sweep_name] = {
            label: _aggregate_cell(seed_results)
            for label, seed_results in by_value.items()
        }
    return summary


def run_baseline_hyperparameter_sweep(
    seeds: Optional[List[int]] = None,
    attack: str = "label_flip_30",
    num_rounds: int = 100,
    config_path: str = "config/fl_config.yaml",
    output_dir: str = "results/tables",
    log_root: str = "results/logs/hyperparam_sweep_baseline",
    run_experiment_fn: Callable = _real_run_experiment,
    sweeps: Optional[Dict[str, Dict]] = None,
    verbose: bool = True,
) -> Dict:
    if seeds is None:
        seeds = SEEDS
    if sweeps is None:
        sweeps = BASELINE_SWEEPS

    os.makedirs(output_dir, exist_ok=True)

    raw = run_baseline_sweep_jobs(
        sweeps, seeds, attack, num_rounds, config_path, log_root,
        run_experiment_fn, verbose,
    )
    summary = aggregate_baseline_sweep(raw)

    if verbose:
        print("\n\n[Baseline Hyperparameter Sweep Summary] (audit E10)")
        for sweep_name, by_value in summary.items():
            print(f"\n{sweep_name} ({sweeps[sweep_name]['paper_symbol']}):")
            for label, cell in by_value.items():
                acc = cell["final_accuracy"]
                asr = cell["final_attack_success_rate"]
                print(f"  {label:<8} acc={acc['mean']:.4f}±{acc['std']:.4f}  "
                      f"asr={asr['mean']:.4f}±{asr['std']:.4f}")

    out_path = os.path.join(output_dir, "hyperparam_sweep_baseline_results.json")
    payload = {
        "provenance": {
            "audit_id": "E10",
            "sweeps": {k: dict(v) for k, v in sweeps.items()},
            "seeds": seeds,
            "attack": attack,
            "num_rounds": num_rounds,
        },
        "raw": raw,
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    if verbose:
        print(f"\n[Baseline Hyperparameter Sweep] Saved to {out_path}")

    return summary


# ── TV-FLIDS sweep (E11) ──────────────────────────────────────────────────

def run_tvflids_sweep_jobs(
    sweeps: Dict[str, Dict],
    seeds: List[int],
    attack: str,
    num_rounds: int,
    base_config_path: str,
    log_root: str,
    run_experiment_fn: Callable,
    verbose: bool = True,
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Execute the TV-FLIDS one-factor-at-a-time grid: for each factor
    (meta_lr, min_trust, warmup_rounds), for each grid point, write a temp
    config with just that one key overridden from the base config default,
    and run all seeds with strategy='tvflids'.
    """
    base_config = load_config(base_config_path)
    raw: Dict[str, Dict[str, List[Dict]]] = {}

    for factor_name, spec in sweeps.items():
        raw[factor_name] = {}
        section, key = spec["section"], spec["key"]
        for value in spec["grid"]:
            label = str(value)
            if verbose:
                print(f"\n[TV-FLIDS Sweep] {factor_name}={value}")

            exp_config = copy.deepcopy(base_config)
            exp_config[section][key] = value
            tmp_config_path = os.path.join(
                tempfile.gettempdir(), f"hp_sweep_tvflids_{factor_name}_{label}.yaml"
            )
            with open(tmp_config_path, "w") as f:
                yaml.dump(exp_config, f)

            seed_results = []
            for seed in seeds:
                result = run_experiment_fn(
                    strategy_name="tvflids",
                    attack_config_name=attack,
                    seed=seed,
                    num_rounds=num_rounds,
                    config_path=tmp_config_path,
                    log_dir=(f"{log_root}/{factor_name}_{label}_seed{seed}"),
                    verbose=False,
                )
                seed_results.append(result)
                if verbose:
                    print(f"    seed={seed}: acc={result.get('final_accuracy', 0):.4f}")
            raw[factor_name][label] = seed_results

    return raw


def aggregate_tvflids_sweep(raw: Dict[str, Dict[str, List[Dict]]]) -> Dict:
    """Pure aggregation step, unit-testable on canned per-seed dicts."""
    summary: Dict[str, Dict[str, Dict]] = {}
    for factor_name, by_value in raw.items():
        summary[factor_name] = {
            label: _aggregate_cell(seed_results)
            for label, seed_results in by_value.items()
        }
    return summary


def run_tvflids_hyperparameter_sweep(
    seeds: Optional[List[int]] = None,
    attack: str = "label_flip_30",
    num_rounds: int = 100,
    config_path: str = "config/fl_config.yaml",
    output_dir: str = "results/tables",
    log_root: str = "results/logs/hyperparam_sweep_tvflids",
    run_experiment_fn: Callable = _real_run_experiment,
    sweeps: Optional[Dict[str, Dict]] = None,
    verbose: bool = True,
) -> Dict:
    if seeds is None:
        seeds = SEEDS
    if sweeps is None:
        sweeps = TVFLIDS_SWEEPS

    os.makedirs(output_dir, exist_ok=True)

    raw = run_tvflids_sweep_jobs(
        sweeps, seeds, attack, num_rounds, config_path, log_root,
        run_experiment_fn, verbose,
    )
    summary = aggregate_tvflids_sweep(raw)

    if verbose:
        print("\n\n[TV-FLIDS Hyperparameter Sweep Summary] (audit E11)")
        for factor_name, by_value in summary.items():
            print(f"\n{factor_name} ({sweeps[factor_name]['paper_symbol']}):")
            for label, cell in by_value.items():
                acc = cell["final_accuracy"]
                asr = cell["final_attack_success_rate"]
                print(f"  {label:<8} acc={acc['mean']:.4f}±{acc['std']:.4f}  "
                      f"asr={asr['mean']:.4f}±{asr['std']:.4f}")

    out_path = os.path.join(output_dir, "hyperparam_sweep_tvflids_results.json")
    payload = {
        "provenance": {
            "audit_id": "E11",
            "sweeps": {k: dict(v) for k, v in sweeps.items()},
            "seeds": seeds,
            "attack": attack,
            "num_rounds": num_rounds,
            "note": (
                "Paper Supp. Table S2's four factors map to four distinct "
                "code parameters: lambda -> trust.memory_decay, "
                "tau_min -> trust.min_trust, eta_meta -> trust.meta_lr, "
                "T_warm -> verification.warmup_rounds. See module docstring "
                "for why the earlier lambda == eta_meta mapping was wrong."
            ),
        },
        "raw": raw,
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    if verbose:
        print(f"\n[TV-FLIDS Hyperparameter Sweep] Saved to {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sensitivity sweeps (audit IDs E10, E11)"
    )
    parser.add_argument("--target", choices=["baseline", "tvflids"], required=True,
                         help="Which sweep to run: 'baseline' (Krum f', Trimmed-Mean "
                              "beta) or 'tvflids' (meta_lr, min_trust, warmup_rounds)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--attack", default="label_flip_30")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--config", default="config/fl_config.yaml")
    parser.add_argument("--output", default="results/tables")
    args = parser.parse_args()

    if args.target == "baseline":
        run_baseline_hyperparameter_sweep(
            seeds=args.seeds, attack=args.attack, num_rounds=args.rounds,
            config_path=args.config, output_dir=args.output,
        )
    else:
        run_tvflids_hyperparameter_sweep(
            seeds=args.seeds, attack=args.attack, num_rounds=args.rounds,
            config_path=args.config, output_dir=args.output,
        )


if __name__ == "__main__":
    main()
