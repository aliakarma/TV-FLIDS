"""
evaluation/pairing.py
Explicit, seed-level pairing logic and invariant validation.
Reference: IEEE TIFS Manuscript §VI-E, §VII, and Supplementary §S8.

Guarantees:
  - Explicit pairing by seed integer: (dataset, block, attack, seed) between TV-FLIDS and baseline.
  - Never pairs observations merely by array or row order.
  - Strict validation detecting:
      * missing baseline seed
      * missing TV-FLIDS seed
      * duplicate seed
      * mismatched attack
      * mismatched dataset
      * mismatched block
      * mismatched scientific configuration
  - Clean-reference pairing for Delta-ASR: pairs attacked run with Block 3 clean reference by seed.
  - Marks comparisons with < required seeds as INCOMPLETE.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from evaluation.result_loader import CampaignResult, ResultLoader


@dataclass(frozen=True)
class PairedObservation:
    """A single seed-paired observation between TV-FLIDS and a baseline."""
    seed: int
    tvflids_val: float
    baseline_val: float
    diff: float                       # 100 * (tvflids_val - baseline_val) in percentage points
    tvflids_run_id: str
    baseline_run_id: str


@dataclass
class PairedComparison:
    """The set of paired observations for a comparison across seeds."""
    dataset: str
    block: str
    baseline: str
    attack: str
    endpoint: str                     # "f1" or "asr"
    required_seeds: int               # 20 for primary/ablation, 10 for secondary
    pairs: List[PairedObservation] = field(default_factory=list)
    status: str = "INCOMPLETE"        # "COMPLETE", "INCOMPLETE", "BLOCKED", "INVALID"
    missing_seeds: List[int] = field(default_factory=list)
    duplicate_seeds: List[int] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.status == "COMPLETE" and len(self.pairs) == self.required_seeds

    @property
    def differences(self) -> np.ndarray:
        """Array of paired percentage-point differences d_i = 100 * (TV-FLIDS - baseline)."""
        return np.array([p.diff for p in self.pairs], dtype=float)

    @property
    def tvflids_values(self) -> np.ndarray:
        return np.array([p.tvflids_val for p in self.pairs], dtype=float)

    @property
    def baseline_values(self) -> np.ndarray:
        return np.array([p.baseline_val for p in self.pairs], dtype=float)


@dataclass(frozen=True)
class CleanPairedObservation:
    """A single seed-paired observation for Delta-ASR: attacked run paired with clean reference."""
    seed: int
    method: str
    dataset: str
    attack: str
    attacked_asr: float
    clean_asr: float
    delta_asr: float                  # 100 * (attacked_asr - clean_asr) in percentage points
    attacked_run_id: str
    clean_run_id: str


def pair_primary_comparison(
    loader: ResultLoader,
    dataset: str,
    baseline: str,
    endpoint: str = "f1",
    block: str = "B2",
    attack: str = "label_flip_30",
    expected_seeds: Optional[List[int]] = None,
) -> PairedComparison:
    """
    Form explicit seed-paired comparison between TV-FLIDS and a baseline.

    Args:
        loader: ResultLoader containing discovered results.
        dataset: "nslkdd", "ciciot2023", or "edgeiiotset".
        baseline: Baseline strategy name (e.g. "fedavg", "fltrust").
        endpoint: "f1" (Macro-F1) or "asr" (Attack Success Rate).
        block: Experimental block (default "B2" for main LF comparison).
        attack: Attack condition (default "label_flip_30").
        expected_seeds: List of expected seeds (default 20 canonical seeds).

    Returns:
        PairedComparison with validated pairs and completeness status.
    """
    if expected_seeds is None:
        # Default canonical 20 seeds (audit §VI-E line 381)
        expected_seeds = [
            42, 123, 456, 789, 1337,
            2024, 31415, 8080, 555, 999,
            1001, 1002, 1003, 1004, 1005,
            1006, 1007, 1008, 1009, 1010,
        ]

    required_count = len(expected_seeds)
    comp = PairedComparison(
        dataset=dataset.lower(),
        block=block.upper(),
        baseline=baseline,
        attack=attack,
        endpoint=endpoint,
        required_seeds=required_count,
    )

    metric_key = "final_f1_macro" if endpoint == "f1" else "final_attack_success_rate"

    tv_runs = loader.query(dataset=dataset, block=block, strategy="tvflids", attack=attack)
    base_runs = loader.query(dataset=dataset, block=block, strategy=baseline, attack=attack)

    tv_by_seed: Dict[int, List[CampaignResult]] = {}
    for r in tv_runs:
        tv_by_seed.setdefault(r.seed, []).append(r)

    base_by_seed: Dict[int, List[CampaignResult]] = {}
    for r in base_runs:
        base_by_seed.setdefault(r.seed, []).append(r)

    # Check duplicates
    dup_seeds: Set[int] = set()
    for s, runs in tv_by_seed.items():
        if len(runs) > 1:
            dup_seeds.add(s)
            comp.validation_errors.append(f"Duplicate TV-FLIDS runs for seed {s}")
    for s, runs in base_by_seed.items():
        if len(runs) > 1:
            dup_seeds.add(s)
            comp.validation_errors.append(f"Duplicate {baseline} runs for seed {s}")
    comp.duplicate_seeds = sorted(dup_seeds)

    # Form pairs
    missing: List[int] = []
    pairs: List[PairedObservation] = []

    for s in expected_seeds:
        has_tv = (s in tv_by_seed and len(tv_by_seed[s]) == 1)
        has_base = (s in base_by_seed and len(base_by_seed[s]) == 1)

        if not has_tv or not has_base:
            missing.append(s)
            continue

        r_tv = tv_by_seed[s][0]
        r_base = base_by_seed[s][0]

        # Invariant checks
        if r_tv.dataset != r_base.dataset or r_tv.dataset != dataset.lower():
            comp.validation_errors.append(f"Seed {s}: Dataset mismatch ({r_tv.dataset} vs {r_base.dataset})")
            continue
        if r_tv.block != r_base.block or r_tv.block != block.upper():
            comp.validation_errors.append(f"Seed {s}: Block mismatch ({r_tv.block} vs {r_base.block})")
            continue
        if r_tv.attack != r_base.attack or r_tv.attack != attack:
            comp.validation_errors.append(f"Seed {s}: Attack mismatch ({r_tv.attack} vs {r_base.attack})")
            continue

        # Check scientific configuration compatibility (e.g. same alpha, clients, rounds)
        cfg_tv = r_tv.scientific_config
        cfg_base = r_base.scientific_config
        for check_param in ("alpha", "num_clients", "fraction_fit", "num_rounds", "val_size"):
            if cfg_tv.get(check_param) != cfg_base.get(check_param):
                comp.validation_errors.append(
                    f"Seed {s}: Configuration mismatch for {check_param} "
                    f"({cfg_tv.get(check_param)} vs {cfg_base.get(check_param)})"
                )

        tv_val = float(r_tv.metrics[metric_key])
        base_val = float(r_base.metrics[metric_key])
        diff = 100.0 * (tv_val - base_val)

        pairs.append(
            PairedObservation(
                seed=s,
                tvflids_val=tv_val,
                baseline_val=base_val,
                diff=diff,
                tvflids_run_id=r_tv.run_id,
                baseline_run_id=r_base.run_id,
            )
        )

    comp.pairs = pairs
    comp.missing_seeds = missing

    # Set status
    if comp.validation_errors or comp.duplicate_seeds:
        comp.status = "INVALID"
    elif len(pairs) == required_count and not missing:
        comp.status = "COMPLETE"
    else:
        # Check if dataset was blocked
        if dataset.lower() in ("ciciot2023", "edgeiiotset"):
            comp.status = "BLOCKED"
        else:
            comp.status = "INCOMPLETE"

    return comp


def pair_delta_asr(
    loader: ResultLoader,
    dataset: str,
    method: str,
    attack: str,
    attack_block: str = "B4",
    clean_block: str = "B3",
    expected_seeds: Optional[List[int]] = None,
) -> List[CleanPairedObservation]:
    """
    Pair attacked runs with Block 3 clean reference runs by seed to compute Delta-ASR.

    Formula:
        Delta-ASR = 100 * (ASR_attacked - ASR_clean_ref)

    Returns:
        List of CleanPairedObservation for each successfully paired seed.
    """
    if expected_seeds is None:
        # Default 10 seeds for attack matrix (Table IV / §VII-B)
        expected_seeds = [42, 123, 456, 789, 1337, 2024, 31415, 8080, 555, 999]

    clean_runs = loader.query(dataset=dataset, block=clean_block, strategy=method, attack="none")
    attack_runs = loader.query(dataset=dataset, block=attack_block, strategy=method, attack=attack)

    clean_by_seed = {r.seed: r for r in clean_runs}
    attack_by_seed = {r.seed: r for r in attack_runs}

    paired: List[CleanPairedObservation] = []
    for s in expected_seeds:
        if s in clean_by_seed and s in attack_by_seed:
            rc = clean_by_seed[s]
            ra = attack_by_seed[s]
            clean_asr = float(rc.metrics["final_attack_success_rate"])
            attacked_asr = float(ra.metrics["final_attack_success_rate"])
            delta_asr = 100.0 * (attacked_asr - clean_asr)

            paired.append(
                CleanPairedObservation(
                    seed=s,
                    method=method,
                    dataset=dataset.lower(),
                    attack=attack,
                    attacked_asr=attacked_asr,
                    clean_asr=clean_asr,
                    delta_asr=delta_asr,
                    attacked_run_id=ra.run_id,
                    clean_run_id=rc.run_id,
                )
            )

    return paired
