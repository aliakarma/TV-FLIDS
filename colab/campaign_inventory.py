"""
colab/campaign_inventory.py — the single authoritative enumeration of every
experiment cell the TV-FLIDS campaign must produce.

Everything else in colab/ derives from this module:

    generate_expected_results.py  ->  expected_results/   (the structure contract)
    parallel_runner.py            ->  the execution queue
    validate_campaign.py          ->  expected vs. actual comparison

Because all three read the same enumeration, `expected_results/<path>` and
`campaign_results/<path>` are isomorphic *by construction* rather than by
convention: there is no second place where a path could be spelled differently.

WHAT THIS MODULE DOES NOT CONTAIN
---------------------------------
No expected accuracy, F1, ASR, p-value, delta or improvement. Not one. A cell
record describes *identity, location, schema and completion criteria*; the
numbers come only from a genuine run. `metric_ranges()` gives mathematically
valid bounds (a probability lies in [0, 1]) — those are definitions, not
predictions.

WHY THE GRIDS ARE RESTATED HERE
-------------------------------
The grids below are duplicated from the repository's own runner constants so
that this module — and therefore the validator — imports without torch, flwr or
ray. Duplication that can drift silently would be worse than no duplication at
all, so `verify_against_repo()` imports the real runner modules and asserts
every grid matches, field by field. Run it (it is a notebook step, and
`--verify` here) in any environment where the scientific stack is installed. A
mismatch is a hard error, not a warning.

    python colab/campaign_inventory.py summary
    python colab/campaign_inventory.py list --phase A_main_comparison
    python colab/campaign_inventory.py json  --out /tmp/cells.json
    python colab/campaign_inventory.py verify        # needs torch/flwr/ray
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterator, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Seeds (evaluation/statistical_testing.py) ────────────────────────────────
SEEDS: List[int] = [42, 123, 456, 789, 1337]
EXTENDED_SEEDS: List[int] = SEEDS + [2024, 31415, 8080, 555, 999]

ROUNDS = 100                 # every campaign cell is a 100-round run
ROUND_LOG_ENTRIES = ROUNDS + 1   # round 0 (pre-training eval) + rounds 1..100

# ── Strategy / attack vocabularies ───────────────────────────────────────────
STRATS8 = ["fedavg", "krum", "trimmed_mean", "fltrust",
           "foolsgold", "flame", "rfa", "tvflids"]
MATRIX_ATTACKS = ["gradient_scale_30", "noise_30", "backdoor_20"]
ADAPTIVE_ATTACKS = ["ack1_evasion_30", "ack2_coalition_30"]
ADAPTIVE_STRATEGIES = ["fedavg", "fltrust", "tvflids"]
EXTRA_BASELINES = ["bucketing", "deepsight"]
MAIN_ATTACK = "label_flip_30"

# ── Ablation arms (experiments/run_ablation.py::ABLATION_CONFIGS) ────────────
# name -> (strategy, partition_type, alpha)
ABLATION_ARMS: Dict[str, Dict] = {
    "TV-FLIDS (Full)":         {"strategy": "tvflids",       "partition_type": "noniid", "alpha": 0.5},
    "A1: No Verification":     {"strategy": "tvflids",       "partition_type": "noniid", "alpha": 0.5},
    "A2: No Memory (decay=0)": {"strategy": "tvflids",       "partition_type": "noniid", "alpha": 0.5},
    "A3: Similarity Only":     {"strategy": "tvflids_fixed", "partition_type": "noniid", "alpha": 0.5},
    "A4: Accuracy Only":       {"strategy": "tvflids_fixed", "partition_type": "noniid", "alpha": 0.5},
    "A5: IID Data":            {"strategy": "tvflids",       "partition_type": "iid",    "alpha": 0.5},
    "A6: Fixed Equal Weights": {"strategy": "tvflids_fixed", "partition_type": "noniid", "alpha": 0.5},
}

# ── Non-IID sweep (experiments/run_noniid_sweep.py) ──────────────────────────
NONIID_STRATEGIES = ["krum", "tvflids"]
NONIID_ALPHAS = [0.1, 0.5, 1.0]

# ── Hyperparameter sweeps (experiments/run_hyperparameter_sweep.py) ──────────
BASELINE_SWEEP_GRIDS: Dict[str, Dict] = {
    "krum_f_prime":      {"strategy": "krum",         "grid": [4, 6, 8, 10],              "default": 6},
    "trimmed_mean_beta": {"strategy": "trimmed_mean", "grid": [0.10, 0.15, 0.20, 0.25],   "default": 0.15},
}
TVFLIDS_SWEEP_GRIDS: Dict[str, Dict] = {
    "memory_decay":  {"grid": [0.7, 0.8, 0.9, 0.95], "default": 0.9},
    "min_trust":     {"grid": [0.001, 0.01, 0.05],   "default": 0.01},
    "meta_lr":       {"grid": [0.001, 0.01, 0.1],    "default": 0.01},
    "warmup_rounds": {"grid": [5, 10, 20, 30],       "default": 20},
}

# ── Ratio sweep (experiments/run_ratio_sweep.py + campaign_lane.sh) ──────────
RATIO_METHODS = ["fedavg", "krum", "fltrust", "tvflids"]
RATIOS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

# ── Per-cell files written by utils/logger.py::ExperimentLogger ──────────────
# config.json is written at logger init (so it exists even for an interrupted
# cell); experiment_log.json and final_predictions.npz are written at the end.
CELL_FILES = ["config.json", "experiment_log.json", "final_predictions.npz"]
COMPLETION_FILE = "experiment_log.json"

# ── Round-log and summary schemas (read off a real experiment_log.json) ──────
ROUND_KEYS_ALWAYS = [
    "round", "timestamp", "accuracy", "f1_macro", "f1_weighted",
    "attack_success_rate", "false_negative_rate",
]
# Trust telemetry is emitted only by the trust-bearing strategies.
ROUND_KEYS_TRUST = [
    "trust_mean", "trust_min", "trust_max", "trust_std",
    "trust_adaptive_alpha", "trust_adaptive_beta", "trust_adaptive_gamma",
]
TRUST_STRATEGIES = ["tvflids", "tvflids_fixed"]

SUMMARY_KEYS = [
    "final_accuracy", "final_f1_macro", "final_attack_success_rate",
    "final_false_negative_rate", "peak_accuracy", "num_rounds",
    "num_evaluations", "strategy", "attack", "seed", "num_malicious",
    "malicious_ids", "comm_overhead_pct", "model_params",
    "compute_overhead_ms", "elapsed_seconds",
]
CONFIG_KEYS = [
    "strategy", "attack", "dataset", "partition_type", "alpha", "seed",
    "val_size", "protocol", "num_clients", "num_rounds", "fraction_fit",
    "fraction_evaluate", "local_epochs", "local_batch_size", "local_lr",
    "_config_hash",
]
PROVENANCE_KEYS = [
    "script", "argv", "timestamp_utc", "python_version", "python_executable",
    "packages", "hardware", "sim_client_cpus", "sim_client_gpus",
    "git_commit", "git_branch", "git_describe", "git_dirty",
    "git_dirty_paths", "started_utc", "config_hash", "log_dir",
]
# The subset a cell is *rejected* for missing (PHASE 20, "Provenance").
PROVENANCE_REQUIRED = [
    "git_commit", "config_hash", "seed", "dataset", "environment",
]


def metric_ranges() -> Dict[str, Dict]:
    """Mathematically valid bounds. Definitions, never predictions."""
    unit = {"min": 0.0, "max": 1.0, "why": "a probability / normalised rate"}
    return {
        "accuracy":            dict(unit),
        "f1_macro":            dict(unit),
        "f1_weighted":         dict(unit),
        "attack_success_rate": dict(unit),
        "false_negative_rate": dict(unit),
        "precision":           dict(unit),
        "recall":              dict(unit),
        "trust_mean":          dict(unit),
        "trust_min":           dict(unit),
        "trust_max":           dict(unit),
        "trust_std":           {"min": 0.0, "max": 1.0, "why": "std of values in [0,1]"},
        "trust_adaptive_alpha": dict(unit),
        "trust_adaptive_beta":  dict(unit),
        "trust_adaptive_gamma": dict(unit),
        "p_value":             {"min": 0.0, "max": 1.0, "why": "a probability"},
        "std":                 {"min": 0.0, "max": None, "why": "a standard deviation is non-negative"},
        "elapsed_seconds":     {"min": 0.0, "max": None, "why": "a duration is non-negative"},
        "compute_overhead_ms": {"min": 0.0, "max": None, "why": "a duration is non-negative"},
        "num_clients":         {"min": 1, "max": None, "why": "a positive integer count"},
        "round":               {"min": 0, "max": ROUNDS, "why": f"round 0 plus rounds 1..{ROUNDS}"},
    }


def invariants() -> Dict[str, object]:
    """Structural facts a completed campaign must satisfy.

    Every entry is either read from a configuration file, fixed by the
    campaign definition, or a tautology. None is an expected measurement.
    """
    return {
        "rounds_per_cell": ROUNDS,
        "round_log_entries_per_cell": ROUND_LOG_ENTRIES,
        "primary_seeds": SEEDS,
        "extension_seeds": EXTENDED_SEEDS,
        "n_primary_seeds": len(SEEDS),
        "n_extension_seeds": len(EXTENDED_SEEDS),
        "trust_adaptive_weights_sum_to_1": True,
        "a6_fixed_weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        "config_hash_must_not_equal": ["mockhash", "", None],
        "provenance_required": PROVENANCE_REQUIRED,
        "asr_direction": "lower_is_better",
        "accuracy_direction": "higher_is_better",
        "one_sided_test_direction_note": (
            "A one-sided test on ASR must be framed as TV-FLIDS < baseline; "
            "on accuracy/F1 as TV-FLIDS > baseline. Reusing one direction for "
            "both metrics inverts the ASR claim."
        ),
        # Read from config/fl_config.yaml + config/dataset_config.yaml at
        # campaign time by validate_campaign.py rather than asserted here, so
        # this module never contradicts the configuration actually used.
        "config_derived": ["num_clients", "val_size", "local_epochs",
                            "local_batch_size", "local_lr", "fraction_fit",
                            "fraction_evaluate", "adaptive_thresholds"],
    }


# ── Phase definitions ────────────────────────────────────────────────────────
# `priority` follows the requested execution order:
#   1 = main paper, 2 = main supporting analysis, 3 = supplementary.

def _cell(**kw) -> Dict:
    kw.setdefault("dataset", "nslkdd")
    kw.setdefault("protocol", "main")
    kw.setdefault("partition_type", "noniid")
    kw.setdefault("alpha", 0.5)
    kw.setdefault("rounds", ROUNDS)
    kw.setdefault("arm", None)
    kw["expected_files"] = list(CELL_FILES)
    kw["completion_file"] = COMPLETION_FILE
    kw["round_log_entries"] = ROUND_LOG_ENTRIES
    kw["round_keys"] = (ROUND_KEYS_ALWAYS + ROUND_KEYS_TRUST
                        if kw["strategy"] in TRUST_STRATEGIES
                        else list(ROUND_KEYS_ALWAYS))
    return kw


def _cells_A() -> Iterator[Dict]:
    for s in STRATS8:
        for seed in SEEDS:
            yield _cell(
                phase="A_main_comparison", strategy=s, attack=MAIN_ATTACK, seed=seed,
                log_dir=f"logs/comparison/{s}_{MAIN_ATTACK}_seed{seed}",
            )


def _cells_B() -> Iterator[Dict]:
    # run_full_comparison namespaces non-main protocols:
    #   suffix = f"_{dataset}_{protocol}"  ->  "_nslkdd_leakage_free"
    for s in STRATS8:
        for seed in SEEDS:
            yield _cell(
                phase="B_leakage_free", strategy=s, attack=MAIN_ATTACK, seed=seed,
                protocol="leakage_free",
                log_dir=f"logs/comparison_nslkdd_leakage_free/{s}_{MAIN_ATTACK}_seed{seed}",
            )


def _cells_C() -> Iterator[Dict]:
    for arm, cfg in ABLATION_ARMS.items():
        # Mirrors run_ablation.py exactly: space -> '_', colon dropped. The
        # colon must not survive into a path (illegal on Windows, unsafe in
        # Google Drive).
        slug = arm.replace(" ", "_").replace(":", "")
        for seed in SEEDS:
            yield _cell(
                phase="C_ablation", strategy=cfg["strategy"], attack=MAIN_ATTACK,
                seed=seed, arm=arm,
                partition_type=cfg["partition_type"], alpha=cfg["alpha"],
                log_dir=f"logs/ablation_{slug}_{seed}",
            )


def _cells_D() -> Iterator[Dict]:
    for s in NONIID_STRATEGIES:
        for a in NONIID_ALPHAS:
            label = f"alpha={a}".replace("=", "")   # -> "alpha0.1"
            for seed in SEEDS:
                yield _cell(
                    phase="D_noniid_sweep", strategy=s, attack=MAIN_ATTACK,
                    seed=seed, alpha=a, arm=f"alpha={a}",
                    log_dir=f"logs/noniid_sweep/{s}_noniid_{label}_seed{seed}",
                )


def _cells_E() -> Iterator[Dict]:
    for s in STRATS8:
        for atk in MATRIX_ATTACKS:
            for seed in SEEDS:
                yield _cell(
                    phase="E_multi_attack", strategy=s, attack=atk, seed=seed,
                    log_dir=f"logs/multi_attack_matrix/{s}_{atk}_seed{seed}",
                )


def _cells_F() -> Iterator[Dict]:
    for s in ADAPTIVE_STRATEGIES:
        for atk in ADAPTIVE_ATTACKS:
            for seed in SEEDS:
                yield _cell(
                    phase="F_adaptive_attacks", strategy=s, attack=atk, seed=seed,
                    log_dir=f"logs/multi_attack_matrix/{s}_{atk}_seed{seed}",
                )


def _cells_N() -> Iterator[Dict]:
    for seed in SEEDS:
        yield _cell(
            phase="N_clean_baseline", strategy="tvflids", attack="no_attack",
            seed=seed,
            log_dir=f"logs/multi_attack_matrix/tvflids_no_attack_seed{seed}",
        )


def _cells_G() -> Iterator[Dict]:
    for s in ["tvflids", "fltrust"]:
        for seed in EXTENDED_SEEDS:
            yield _cell(
                phase="G_extended_significance", strategy=s, attack=MAIN_ATTACK,
                seed=seed,
                log_dir=f"logs/extended_significance/{s}_{MAIN_ATTACK}_seed{seed}",
            )


def _cells_H() -> Iterator[Dict]:
    for name, spec in BASELINE_SWEEP_GRIDS.items():
        for value in spec["grid"]:
            for seed in SEEDS:
                yield _cell(
                    phase="H_hp_sweep_baseline", strategy=spec["strategy"],
                    attack=MAIN_ATTACK, seed=seed, arm=f"{name}={value}",
                    log_dir=f"logs/hyperparam_sweep_baseline/{name}_{value}_seed{seed}",
                )


def _cells_I() -> Iterator[Dict]:
    for name, spec in TVFLIDS_SWEEP_GRIDS.items():
        for value in spec["grid"]:
            for seed in SEEDS:
                yield _cell(
                    phase="I_hp_sweep_tvflids", strategy="tvflids",
                    attack=MAIN_ATTACK, seed=seed, arm=f"{name}={value}",
                    log_dir=f"logs/hyperparam_sweep_tvflids/{name}_{value}_seed{seed}",
                )


def _cells_J() -> Iterator[Dict]:
    for s in EXTRA_BASELINES:
        for seed in SEEDS:
            yield _cell(
                phase="J_bucketing_deepsight", strategy=s, attack=MAIN_ATTACK,
                seed=seed,
                log_dir=f"logs/comparison/{s}_{MAIN_ATTACK}_seed{seed}",
            )


def _cells_R() -> Iterator[Dict]:
    for m in RATIO_METHODS:
        for r in RATIOS:
            atk = "no_attack" if r == 0.0 else MAIN_ATTACK
            for seed in SEEDS:
                yield _cell(
                    phase="R_ratio_sweep", strategy=m, attack=atk, seed=seed,
                    arm=f"ratio={r}",
                    log_dir=f"logs/ratio_sweep/{m}_ratio{int(round(r * 100))}_seed{seed}",
                )


PHASES: List[Dict] = [
    {"phase": "A_main_comparison", "exec_mode": "cell", "shard_by": None, "priority": 1, "order": 1,
     "paper_artifact": "Table V; Figures 2-4",
     "runner": "experiments/run_full_comparison.py",
     "argv": ["--strategies", *STRATS8, "--attack", MAIN_ATTACK,
              "--seeds", *[str(s) for s in SEEDS], "--rounds", str(ROUNDS)],
     "aggregate_artifact": "tables/full_comparison_results.json",
     "cells": _cells_A},

    {"phase": "C_ablation", "exec_mode": "shard", "shard_by": ["--seeds"], "priority": 1, "order": 2,
     "paper_artifact": "Table VII",
     "runner": "experiments/run_ablation.py",
     "argv": ["--attack", MAIN_ATTACK, "--rounds", str(ROUNDS),
              "--seeds", *[str(s) for s in SEEDS]],
     "aggregate_artifact": "tables/ablation_results.json",
     "cells": _cells_C},

    {"phase": "D_noniid_sweep", "exec_mode": "cell", "shard_by": None, "priority": 1, "order": 3,
     "paper_artifact": "Table VIII",
     "runner": "experiments/run_noniid_sweep.py",
     "argv": ["--alphas", "0.1", "0.5", "1.0",
              "--seeds", *[str(s) for s in SEEDS], "--rounds", str(ROUNDS)],
     "aggregate_artifact": "tables/noniid_sweep_results.json",
     "cells": _cells_D},

    {"phase": "N_clean_baseline", "exec_mode": "cell", "shard_by": None, "priority": 1, "order": 4,
     "paper_artifact": "Table IX (no-attack row)",
     "runner": "experiments/run_multi_attack_matrix.py",
     "argv": ["--strategies", "tvflids", "--attacks", "no_attack",
              "--seeds", *[str(s) for s in SEEDS], "--rounds", str(ROUNDS)],
     "aggregate_artifact": "tables/multi_attack_matrix_results_no_attack.json",
     "cells": _cells_N},

    {"phase": "E_multi_attack", "exec_mode": "cell", "shard_by": None, "priority": 1, "order": 5,
     "paper_artifact": "Tables IX-X",
     "runner": "experiments/run_multi_attack_matrix.py",
     "argv": ["--seeds", *[str(s) for s in SEEDS], "--rounds", str(ROUNDS)],
     "aggregate_artifact": "tables/multi_attack_matrix_results.json",
     "cells": _cells_E},

    {"phase": "F_adaptive_attacks", "exec_mode": "cell", "shard_by": None, "priority": 1, "order": 6,
     "paper_artifact": "Table XI",
     "runner": "experiments/run_multi_attack_matrix.py",
     "argv": ["--strategies", *ADAPTIVE_STRATEGIES,
              "--attacks", *ADAPTIVE_ATTACKS,
              "--seeds", *[str(s) for s in SEEDS], "--rounds", str(ROUNDS)],
     "aggregate_artifact":
         "tables/multi_attack_matrix_results_ack1_evasion_30_ack2_coalition_30.json",
     "cells": _cells_F},

    {"phase": "R_ratio_sweep", "exec_mode": "shard", "shard_by": ["--methods", "--seeds"], "priority": 1, "order": 7,
     "paper_artifact": "Figure 5",
     "runner": "experiments/run_ratio_sweep.py",
     "argv": ["--methods", *RATIO_METHODS,
              "--ratios", *[str(r) for r in RATIOS],
              "--seeds", *[str(s) for s in SEEDS], "--rounds", str(ROUNDS)],
     "aggregate_artifact": "tables/ratio_sweep_results.json",
     "cells": _cells_R},

    {"phase": "B_leakage_free", "exec_mode": "cell", "shard_by": None, "priority": 2, "order": 8,
     "paper_artifact": "Table VI",
     "runner": "experiments/run_full_comparison.py",
     "argv": ["--strategies", *STRATS8, "--attack", MAIN_ATTACK,
              "--seeds", *[str(s) for s in SEEDS], "--rounds", str(ROUNDS),
              "--protocol", "leakage_free"],
     "aggregate_artifact": "tables/full_comparison_results_nslkdd_leakage_free.json",
     "cells": _cells_B},

    {"phase": "G_extended_significance", "exec_mode": "cell", "shard_by": None, "priority": 2, "order": 9,
     "paper_artifact": "Section VIII-B (ten-seed significance)",
     "runner": "experiments/run_extended_significance.py",
     "argv": ["--strategies", "tvflids", "fltrust", "--rounds", str(ROUNDS)],
     "aggregate_artifact": "tables/extended_significance_results.json",
     "cells": _cells_G},

    {"phase": "H_hp_sweep_baseline", "exec_mode": "shard", "shard_by": ["--seeds"], "priority": 3, "order": 10,
     "paper_artifact": "Supplementary Table S1",
     "runner": "experiments/run_hyperparameter_sweep.py",
     "argv": ["--target", "baseline",
              "--seeds", *[str(s) for s in SEEDS], "--rounds", str(ROUNDS)],
     "aggregate_artifact": "tables/hyperparam_sweep_baseline_results.json",
     "cells": _cells_H},

    {"phase": "I_hp_sweep_tvflids", "exec_mode": "shard", "shard_by": ["--seeds"], "priority": 3, "order": 11,
     "paper_artifact": "Supplementary Table S2",
     "runner": "experiments/run_hyperparameter_sweep.py",
     "argv": ["--target", "tvflids",
              "--seeds", *[str(s) for s in SEEDS], "--rounds", str(ROUNDS)],
     "aggregate_artifact": "tables/hyperparam_sweep_tvflids_results.json",
     "cells": _cells_I},

    {"phase": "J_bucketing_deepsight", "exec_mode": "cell", "shard_by": None, "priority": 3, "order": 12,
     "paper_artifact": "Supplementary Table S3",
     "runner": "experiments/run_full_comparison.py",
     "argv": ["--strategies", *EXTRA_BASELINES, "--attack", MAIN_ATTACK,
              "--seeds", *[str(s) for s in SEEDS], "--rounds", str(ROUNDS),
              "--output", "results/tables/_extra_baselines"],
     "aggregate_artifact": "tables/_extra_baselines/full_comparison_results.json",
     "cells": _cells_J},
]

# Phases with no cells of their own: they are derived from the cells above.
DERIVED_PHASES: List[Dict] = [
    {"phase": "L_overhead", "priority": 2,
     "paper_artifact": "Table XII (overhead)",
     "derived_from": "summary.compute_overhead_ms / comm_overhead_pct / "
                     "model_params of the A_main_comparison cells",
     "aggregate_artifact": "tables/overhead_results.json"},
    {"phase": "Stats_significance", "priority": 2,
     "paper_artifact": "Wilcoxon / Cohen's d columns of Tables V-XI",
     "derived_from": "the per-seed summaries inside each phase's aggregate artifact",
     "aggregate_artifact": "statistics/"},
]

# Phases that cannot run until an external dataset is supplied.
BLOCKED_PHASES: List[Dict] = [
    {"phase": "K_ciciot2023", "priority": 3,
     "paper_artifact": "Supplementary Table S4; Supplementary Figure S1",
     "runner": "experiments/run_dataset_comparison.py",
     "blocked_on": "data/raw/CICIoT2023_train.csv and CICIoT2023_test.csv are "
                    "absent from the repository. The phase is enumerated but "
                    "produces no queue entries until the dataset is present.",
     "would_be_cells": len(["fedavg", "fltrust", "tvflids"]) * 2 * len(SEEDS)},
]


# ── Public API ───────────────────────────────────────────────────────────────

def phase_map() -> Dict[str, Dict]:
    return {p["phase"]: p for p in PHASES}


def iter_cells(phases: Optional[List[str]] = None) -> Iterator[Dict]:
    """Yield every cell, in campaign priority order.

    `cell_id` is the cell's stable identity:
        <phase>/<strategy>/<attack>/<dataset>/<protocol>/seed<seed>[/<arm>]
    """
    wanted = set(phases) if phases else None
    for spec in sorted(PHASES, key=lambda p: p["order"]):
        if wanted and spec["phase"] not in wanted:
            continue
        for cell in spec["cells"]():
            cell["priority"] = spec["priority"]
            cell["order"] = spec["order"]
            cell["paper_artifact"] = spec["paper_artifact"]
            cell["runner"] = spec["runner"]
            cell["aggregate_artifact"] = spec["aggregate_artifact"]
            ident = (f"{cell['phase']}/{cell['strategy']}/{cell['attack']}/"
                     f"{cell['dataset']}/{cell['protocol']}/seed{cell['seed']}")
            if cell["arm"]:
                ident += "/" + str(cell["arm"]).replace(" ", "_")
            cell["cell_id"] = ident
            yield cell


def all_cells(phases: Optional[List[str]] = None) -> List[Dict]:
    return list(iter_cells(phases))


def cells_by_phase() -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for c in iter_cells():
        out.setdefault(c["phase"], []).append(c)
    return out


def unique_log_dirs() -> Dict[str, List[str]]:
    """log_dir -> [cell_id, ...].

    Phases can legitimately share a log_dir namespace (F and N both write under
    logs/multi_attack_matrix/). Two *different* cells resolving to the same
    log_dir would be a collision that silently overwrites a result, so the
    validator asserts every list here has length 1.
    """
    out: Dict[str, List[str]] = {}
    for c in iter_cells():
        out.setdefault(c["log_dir"], []).append(c["cell_id"])
    return out


# ── Drift guard ──────────────────────────────────────────────────────────────

def verify_against_repo() -> List[str]:
    """Import the real runner modules and assert every grid here matches.

    Returns a list of mismatch descriptions; empty means the duplication in
    this module is faithful. Requires the scientific stack (torch/flwr/ray).
    """
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    problems: List[str] = []

    def cmp(label, mine, theirs):
        if mine != theirs:
            problems.append(f"{label}: inventory={mine!r} repo={theirs!r}")

    from evaluation import statistical_testing as st
    cmp("SEEDS", SEEDS, list(st.SEEDS))
    cmp("EXTENDED_SEEDS", EXTENDED_SEEDS, list(st.EXTENDED_SEEDS))

    from experiments import run_multi_attack_matrix as mam
    cmp("STRATS8", STRATS8, list(mam.STRATEGIES))
    cmp("MATRIX_ATTACKS", MATRIX_ATTACKS, list(mam.ATTACKS))

    from experiments import run_noniid_sweep as nis
    cmp("NONIID_STRATEGIES", NONIID_STRATEGIES, list(nis.STRATEGIES))
    cmp("NONIID_ALPHAS", NONIID_ALPHAS, list(nis.ALPHAS))

    from experiments import run_ablation as abl
    cmp("ABLATION arm names", list(ABLATION_ARMS), list(abl.ABLATION_CONFIGS))
    for name, cfg in abl.ABLATION_CONFIGS.items():
        mine = ABLATION_ARMS.get(name)
        if mine is None:
            continue
        cmp(f"ablation[{name}].strategy", mine["strategy"], cfg["strategy"])
        cmp(f"ablation[{name}].partition_type", mine["partition_type"], cfg["partition_type"])
        cmp(f"ablation[{name}].alpha", mine["alpha"], cfg["alpha"])

    from experiments import run_hyperparameter_sweep as hps
    cmp("BASELINE sweep names", list(BASELINE_SWEEP_GRIDS), list(hps.BASELINE_SWEEPS))
    for name, spec in hps.BASELINE_SWEEPS.items():
        mine = BASELINE_SWEEP_GRIDS.get(name)
        if mine is None:
            continue
        cmp(f"baseline[{name}].grid", mine["grid"], list(spec["grid"]))
        cmp(f"baseline[{name}].strategy", mine["strategy"], spec["strategy"])
        cmp(f"baseline[{name}].default", mine["default"], spec["default"])
    cmp("TVFLIDS sweep names", list(TVFLIDS_SWEEP_GRIDS), list(hps.TVFLIDS_SWEEPS))
    for name, spec in hps.TVFLIDS_SWEEPS.items():
        mine = TVFLIDS_SWEEP_GRIDS.get(name)
        if mine is None:
            continue
        cmp(f"tvflids[{name}].grid", mine["grid"], list(spec["grid"]))
        cmp(f"tvflids[{name}].default", mine["default"], spec["default"])

    from experiments import run_ratio_sweep as rss
    cmp("RATIO_METHODS", RATIO_METHODS, list(rss.METHODS))
    cmp("RATIOS", RATIOS, list(rss.RATIOS))

    from attacks.adversarial import ATTACK_CONFIGS
    for atk in set([MAIN_ATTACK, "no_attack"] + MATRIX_ATTACKS + ADAPTIVE_ATTACKS):
        if atk not in ATTACK_CONFIGS:
            problems.append(f"attack {atk!r} is not in attacks.adversarial.ATTACK_CONFIGS")

    # A log_dir collision would let one cell overwrite another's result.
    for log_dir, ids in unique_log_dirs().items():
        if len(ids) > 1:
            problems.append(f"log_dir collision {log_dir}: {ids}")

    # Every log_dir must be creatable on Windows and syncable to Google Drive,
    # or a campaign cannot be archived off Linux. This is what caught the
    # colon in run_ablation.py's arm names.
    illegal = set('<>:"|?*\\')
    for log_dir in unique_log_dirs():
        bad = sorted(illegal & set(log_dir.replace("/", "")))
        if bad:
            problems.append(
                f"log_dir {log_dir!r} contains cross-platform-illegal "
                f"character(s) {bad}")

    # The inventory's ablation slug must equal the path run_ablation.py builds.
    # Read the runner's own expression rather than trusting this module.
    abl_src = os.path.join(ROOT, "experiments", "run_ablation.py")
    try:
        with open(abl_src, encoding="utf-8") as fh:
            src = fh.read()
        for arm in ABLATION_ARMS:
            slug = arm.replace(" ", "_").replace(":", "")
            if f"ablation_{slug}_" not in src.replace("\n", "") and (
                    ".replace(':', '')" not in src):
                problems.append(
                    "run_ablation.py no longer strips ':' from arm names; the "
                    "inventory's ablation log_dir slugs would be wrong")
                break
    except OSError as exc:
        problems.append(f"could not read run_ablation.py: {exc}")

    return problems


# ── CLI ──────────────────────────────────────────────────────────────────────

def _summary() -> int:
    by_phase = cells_by_phase()
    total = sum(len(v) for v in by_phase.values())
    print(f"TV-FLIDS campaign inventory — {total} cells, {ROUNDS} rounds each\n")
    print(f"{'phase':<26} {'prio':>4} {'cells':>6}  paper artifact")
    print("-" * 96)
    for spec in sorted(PHASES, key=lambda p: p["order"]):
        n = len(by_phase.get(spec["phase"], []))
        print(f"{spec['phase']:<26} {spec['priority']:>4} {n:>6}  {spec['paper_artifact']}")
    print("-" * 96)
    print(f"{'TOTAL EXECUTABLE CELLS':<26} {'':>4} {total:>6}")
    print()
    for spec in DERIVED_PHASES:
        print(f"  derived  {spec['phase']:<24} {spec['paper_artifact']}")
    for spec in BLOCKED_PHASES:
        print(f"  BLOCKED  {spec['phase']:<24} {spec['paper_artifact']}")
        print(f"           -> {spec['blocked_on']}")
    print()
    print(f"round-log entries per cell : {ROUND_LOG_ENTRIES} (round 0 + rounds 1..{ROUNDS})")
    print(f"per-cell files             : {', '.join(CELL_FILES)}")
    print(f"primary seeds              : {SEEDS}")
    print(f"extension seeds            : {EXTENDED_SEEDS}")
    print(f"distinct log_dirs          : {len(unique_log_dirs())}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="TV-FLIDS campaign cell inventory")
    ap.add_argument("mode", choices=["summary", "list", "json", "verify"])
    ap.add_argument("--phase", action="append", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "summary":
        return _summary()

    if args.mode == "list":
        for c in iter_cells(args.phase):
            print(f"{c['cell_id']:<78} {c['log_dir']}")
        return 0

    if args.mode == "json":
        cells = all_cells(args.phase)
        payload = {
            "rounds": ROUNDS,
            "round_log_entries": ROUND_LOG_ENTRIES,
            "seeds": SEEDS,
            "extended_seeds": EXTENDED_SEEDS,
            "n_cells": len(cells),
            "metric_ranges": metric_ranges(),
            "invariants": invariants(),
            "cells": cells,
        }
        text = json.dumps(payload, indent=2, default=str)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"wrote {args.out} ({len(cells)} cells)")
        else:
            print(text)
        return 0

    problems = verify_against_repo()
    if problems:
        print("[inventory] DRIFT DETECTED — this module disagrees with the repo:")
        for p in problems:
            print("  - " + p)
        return 1
    print("[inventory] OK: every grid matches the repository's own constants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
