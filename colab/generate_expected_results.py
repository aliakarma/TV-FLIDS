"""
colab/generate_expected_results.py — emit the expected_results/ contract tree.

    python colab/generate_expected_results.py                 # write expected_results/
    python colab/generate_expected_results.py --out-dir X     # elsewhere
    python colab/generate_expected_results.py --check         # report, write nothing

WHAT THIS PRODUCES
------------------
A tree that is path-for-path isomorphic with the campaign's real output:

    expected_results/logs/comparison/fedavg_label_flip_30_seed42/experiment_log.json
    campaign_results/logs/comparison/fedavg_label_flip_30_seed42/experiment_log.json

Same directory names, same filenames. The difference is the *content*: the file
under expected_results/ is a CONTRACT describing the required schema, required
provenance fields, valid metric ranges and completion criteria for the result
that belongs at the mirrored path. It carries no expected accuracy, F1, ASR,
p-value or delta — not one predicted number.

Every contract file is stamped with `"__expected_contract__": true` and a
`"__not_a_result__"` note so that neither a human nor a script can mistake it
for output. Nothing in the repository reads expected_results/ as data: the
runners write to results/, the table and figure generators read results/, and
validate_campaign.py reads these files only as a schema to check against.

The tree is generated, never hand-edited: it is regenerated from
colab/campaign_inventory.py, which is itself drift-checked against the
repository's own runner constants.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import campaign_inventory as inv  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "expected_results")

NOT_A_RESULT = (
    "STRUCTURE CONTRACT, NOT A RESULT. This file declares what must exist at "
    "the mirrored campaign_results/ path and what shape it must have. It "
    "contains no expected numerical outcome. Never read it as data; never "
    "copy it into results/ or campaign_results/."
)


def _contract(payload: Dict) -> Dict:
    out = {
        "__expected_contract__": True,
        "__not_a_result__": NOT_A_RESULT,
        "__generated_by__": "colab/generate_expected_results.py",
    }
    out.update(payload)
    return out


def _write(path: str, payload: Dict, check: bool, written: List[str]) -> None:
    written.append(path)
    if check:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def _write_text(path: str, text: str, check: bool, written: List[str]) -> None:
    written.append(path)
    if check:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def cell_contract(c: Dict) -> Dict:
    return _contract({
        "cell_id": c["cell_id"],
        "phase": c["phase"],
        "priority": c["priority"],
        "paper_artifact": c["paper_artifact"],
        "runner": c["runner"],
        "identity": {
            "strategy": c["strategy"],
            "attack": c["attack"],
            "dataset": c["dataset"],
            "protocol": c["protocol"],
            "partition_type": c["partition_type"],
            "alpha": c["alpha"],
            "seed": c["seed"],
            "arm": c["arm"],
            "rounds": c["rounds"],
        },
        "mirrors": {
            "expected_path": f"expected_results/{c['log_dir']}/experiment_log.json",
            "actual_path": f"campaign_results/{c['log_dir']}/experiment_log.json",
        },
        "expected": {
            "sibling_files": c["expected_files"],
            "round_log_entries": c["round_log_entries"],
            "round_index_first": 0,
            "round_index_last": c["rounds"],
            "round_keys": c["round_keys"],
            "summary_keys": inv.SUMMARY_KEYS,
            "config_keys": inv.CONFIG_KEYS,
            "provenance_keys": inv.PROVENANCE_KEYS,
        },
        "provenance_required": inv.PROVENANCE_REQUIRED,
        "metric_ranges": inv.metric_ranges(),
        "completion_criteria": [
            f"{c['log_dir']}/experiment_log.json exists, is non-empty and parses as JSON",
            f"rounds has exactly {c['round_log_entries']} entries",
            f"rounds[i].round covers 0..{c['rounds']} with no gap or duplicate",
            "every key in expected.round_keys is present on every round entry",
            "every key in expected.summary_keys is present in summary",
            "config matches identity above field for field",
            "config._config_hash is a real hash: not 'mockhash', not empty, not null",
            "provenance is present and carries every provenance_required field",
            "every metric lies inside metric_ranges",
            f"{c['log_dir']}/config.json and final_predictions.npz exist and are non-empty",
        ],
        "forbidden": {
            "config_hash_values": ["mockhash", "", None],
            "zero_byte_files": True,
            "quarantined_path_substring": "_QUARANTINED_MOCK",
        },
        "status": "pending",
    })


def aggregate_contract(spec: Dict, phase_cells: List[Dict]) -> Dict:
    seeds = sorted({c["seed"] for c in phase_cells})
    return _contract({
        "artifact": spec["aggregate_artifact"],
        "phase": spec["phase"],
        "priority": spec["priority"],
        "paper_artifact": spec["paper_artifact"],
        "produced_by": f"python {spec['runner']} {' '.join(spec['argv'])}",
        "mirrors": {
            "expected_path": f"expected_results/{spec['aggregate_artifact']}",
            "actual_path": f"campaign_results/{spec['aggregate_artifact']}",
        },
        "expected": {
            "backing_cells": len(phase_cells),
            "backing_cell_ids": [c["cell_id"] for c in phase_cells],
            "seeds_per_condition": seeds,
            "structure": {
                "raw": "per-condition list (or dict) of the per-seed summary dicts returned by run_experiment",
                "summary": "per-condition aggregate with mean / std / n_seeds per metric",
                "provenance": "the runner's own provenance block",
            },
            "per_seed_metrics": [
                "final_accuracy",
                "final_f1_macro",
                "final_attack_success_rate",
            ],
        },
        "consumed_by": "scripts/generate_manuscript_tables.py",
        "completion_criteria": [
            "file exists, is non-empty and parses as JSON",
            "every backing_cell_id above resolves to a complete cell log",
            "every condition carries exactly the seeds in seeds_per_condition",
            "no aggregate std is negative; no mean falls outside its metric range",
            "n_seeds equals the number of seeds actually present, not a constant",
        ],
        "status": "pending",
    })


def figures_contract() -> Dict:
    return _contract({
        "artifact_group": "manuscript figure bodies",
        "produced_by": "python scripts/generate_manuscript_figures.py --results-root campaign_results",
        "note": "Figure 1 is an analytical schematic with no experimental content and is intentionally never generated.",
        "figures": [
            {
                "artifact": "Paper/figures/fig_convergence.tex",
                "paper": "Figure 2",
                "source": "campaign_results/logs/comparison/<strategy>_label_flip_30_seed<S>/experiment_log.json -> rounds[].accuracy",
                "phase": "A_main_comparison",
            },
            {
                "artifact": "Paper/figures/fig_trust.tex",
                "paper": "Figure 3",
                "source": "the TV-FLIDS comparison logs -> extra.trust_history, extra.malicious_ids",
                "phase": "A_main_comparison",
            },
            {
                "artifact": "Paper/figures/fig_weights.tex",
                "paper": "Figure 4",
                "source": "the TV-FLIDS comparison logs -> extra.strategy_round_logs[].adaptive_{alpha,beta,gamma}",
                "phase": "A_main_comparison",
            },
            {
                "artifact": "Paper/figures/fig_robustness.tex",
                "paper": "Figure 5",
                "source": "campaign_results/tables/ratio_sweep_results.json",
                "phase": "R_ratio_sweep",
            },
            {
                "artifact": "Paper/figures/fig_ciciot.tex",
                "paper": "Supplementary Figure S1",
                "source": "campaign_results/tables/dataset_comparison_results.json",
                "phase": "K_ciciot2023",
            },
        ],
        "completion_criteria": [
            "each generated .tex is non-empty and carries the generator's 'GENERATED FILE' banner naming its source artifact(s)",
            "no figure body contains provisional or placeholder coordinates once its backing artifact exists",
            "a figure whose source artifact is absent must be SKIPPED by the generator, never invented",
            "any compiled PDF under campaign_results/figures/ starts with the bytes %PDF- and is larger than 0 bytes",
        ],
        "status": "pending",
    })


def statistics_contract() -> Dict:
    return _contract({
        "artifact_group": "statistical analysis",
        "produced_by": "evaluation/statistical_testing.py, called by each runner's aggregation step",
        "expected": {
            "tests": [
                "Wilcoxon signed-rank (paired, TV-FLIDS vs. each baseline)",
                "Cohen's d effect size",
                "95% confidence interval",
            ],
            "fields_per_comparison": [
                "p_value",
                "cohens_d",
                "ci_low",
                "ci_high",
                "n_seeds",
                "metric",
                "direction",
            ],
            "n_seeds_primary": len(inv.SEEDS),
            "n_seeds_extended": len(inv.EXTENDED_SEEDS),
        },
        "metric_ranges": {
            "p_value": {"min": 0.0, "max": 1.0, "why": "a probability"},
            "std": {"min": 0.0, "max": None, "why": "a standard deviation is non-negative"},
        },
        "completion_criteria": [
            "every p-value lies in [0, 1]",
            "a one-sided test on attack_success_rate is framed lower-is-better (TV-FLIDS < baseline); on accuracy and F1 higher-is-better",
            "n_seeds equals the number of seeds actually present in the backing artifact",
            "no p-value, effect size or confidence interval is carried over from expected_results/historical_reference/",
        ],
        "invariants": inv.invariants(),
        "status": "pending",
    })


def manifest_template(cells: List[Dict]) -> Dict:
    by_phase: Dict[str, int] = {}
    for c in cells:
        by_phase[c["phase"]] = by_phase.get(c["phase"], 0) + 1
    return _contract({
        "campaign": "TV-FLIDS full experimental campaign (Colab, T4)",
        "manifest_version": 1,
        "role": "TEMPLATE. Every cell below is listed with status 'pending'. The campaign writes campaign_results/final_manifest.json in the same shape with real statuses; colab/validate_campaign.py compares the two.",
        "rounds_per_cell": inv.ROUNDS,
        "round_log_entries_per_cell": inv.ROUND_LOG_ENTRIES,
        "seeds": inv.SEEDS,
        "extended_seeds": inv.EXTENDED_SEEDS,
        "n_cells": len(cells),
        "cells_by_phase": by_phase,
        "derived_phases": inv.DERIVED_PHASES,
        "blocked_phases": inv.BLOCKED_PHASES,
        "cells": [
            {
                "cell_id": c["cell_id"],
                "phase": c["phase"],
                "priority": c["priority"],
                "paper_artifact": c["paper_artifact"],
                "runner": c["runner"],
                "strategy": c["strategy"],
                "attack": c["attack"],
                "dataset": c["dataset"],
                "protocol": c["protocol"],
                "partition_type": c["partition_type"],
                "alpha": c["alpha"],
                "arm": c["arm"],
                "seed": c["seed"],
                "rounds": c["rounds"],
                "log_dir": c["log_dir"],
                "expected_files": c["expected_files"],
                "expected_round_log_entries": c["round_log_entries"],
                "expected_round_keys": c["round_keys"],
                "expected_summary_keys": inv.SUMMARY_KEYS,
                "aggregate_artifact": c["aggregate_artifact"],
                "provenance_required": inv.PROVENANCE_REQUIRED,
                "status": "pending",
                "config_hash": None,
                "started_utc": None,
                "finished_utc": None,
                "elapsed_seconds": None,
                "attempts": 0,
            }
            for c in cells
        ],
    })


def by_paper_artifact(cells: List[Dict]) -> Dict:
    idx: Dict[str, Dict] = {}
    for c in cells:
        art = c["paper_artifact"]
        e = idx.setdefault(art, {
            "paper_artifact": art,
            "phase": c["phase"],
            "priority": c["priority"],
            "aggregate_artifact": c["aggregate_artifact"],
            "n_cells": 0,
            "seeds": [],
            "cells": [],
        })
        e["n_cells"] += 1
        if c["seed"] not in e["seeds"]:
            e["seeds"].append(c["seed"])
        e["cells"].append({"cell_id": c["cell_id"], "seed": c["seed"],
                            "log_dir": c["log_dir"]})
    for e in idx.values():
        e["seeds"].sort()
    return _contract({
        "role": "Index from paper artifact to the cells that back it. Use it to "
                 "match a manuscript table or figure to every seed-level result "
                 "underneath it.",
        "artifacts": list(idx.values()),
    })


HISTORICAL_README = """# historical_reference/

**HISTORICAL - NON-TARGET - DO NOT USE FOR VALIDATION**

This directory exists so that previously published or previously computed
numbers have somewhere to live that is *outside* the validation path. It is
empty of targets by design.

Nothing here is an expected result. Specifically:

* No file in this directory may be read by `colab/validate_campaign.py`, by any
  runner under `experiments/`, or by `scripts/generate_manuscript_tables.py` /
  `scripts/generate_manuscript_figures.py`.
* No number here may be compared against a new result as if it were ground
  truth. A new result that disagrees with a historical value is not thereby
  wrong; the historical value is not evidence.
* Values previously identified as mock or stale live under
  `results/_QUARANTINED_MOCK/` as forensic evidence. They are **not** copied
  here, and they are **not** expected answers.

If you add anything to this directory, prefix the file with
`HISTORICAL_NON_TARGET_` and state inside it where the number came from and why
it is not a target.

The only legitimate source of a final numerical result is a genuine execution
of the corrected implementation, recorded under `campaign_results/` with full
provenance.
"""


def readme(cells: List[Dict], n_files: int) -> str:
    by_phase: Dict[str, int] = {}
    for c in cells:
        by_phase[c["phase"]] = by_phase.get(c["phase"], 0) + 1
    rows = "\n".join(
        f"| `{spec['phase']}` | {spec['priority']} | {by_phase.get(spec['phase'], 0)} | "
        f"{spec['paper_artifact']} | `{spec['aggregate_artifact']}` |"
        for spec in sorted(inv.PHASES, key=lambda p: p["order"])
    )
    return f"""# expected_results/ — the campaign's structure contract

**Generated file tree. Do not hand-edit.** Regenerate with:

```bash
python colab/generate_expected_results.py
```

## What this directory is

A path-for-path mirror of what a *successful* campaign must leave behind. For
every real output the campaign creates, there is a file here at the same
relative path:

```
expected_results/logs/comparison/fedavg_label_flip_30_seed42/experiment_log.json
campaign_results/logs/comparison/fedavg_label_flip_30_seed42/experiment_log.json
```

Same directory names. Same filenames. Only the content differs: the file under
`expected_results/` is a **contract** — required schema, required provenance
fields, mathematically valid metric ranges, completion criteria — while the
file under `campaign_results/` is the **result**.

## What this directory is NOT

It contains **no expected numerical outcome**: no target accuracy, F1, ASR,
p-value, improvement percentage, ablation delta or hyperparameter result. Those
would be fabricated evidence. The only numbers here are:

* mathematically valid bounds (a probability lies in `[0, 1]`; a standard
  deviation is `>= 0`);
* structural counts fixed by the campaign definition ({inv.ROUNDS} rounds,
  {inv.ROUND_LOG_ENTRIES} round-log entries, {len(inv.SEEDS)} primary seeds,
  {len(inv.EXTENDED_SEEDS)} extension seeds);
* configuration values read from `config/`.

Every contract file carries `"__expected_contract__": true` and a
`"__not_a_result__"` note. Nothing in the repository reads this tree as data.

## Layout

| Path | Holds |
| --- | --- |
| `README.md` | this file |
| `manifest_template.json` | every expected cell, status `pending` |
| `by_paper_artifact.json` | paper table/figure -> the cells backing it |
| `logs/**/experiment_log.json` | one per-cell contract per campaign cell |
| `tables/*.json` | one contract per aggregate artifact |
| `tables/_extra_baselines/*.json` | Supp. Table S3 aggregate contract |
| `figures/EXPECTED_FIGURES.json` | figure bodies, sources and skip rule |
| `statistics/EXPECTED_STATISTICS.json` | tests, fields, direction invariants |
| `historical_reference/` | **HISTORICAL - NON-TARGET**, empty of targets |

Per-round logs are not a separate tree: a cell's 101 round entries live inside
its own `experiment_log.json`, under `rounds[]`. The per-cell contract states
the required entry count, index range and per-round keys.

## Coverage

| Phase | Priority | Cells | Paper artifact | Aggregate artifact |
| --- | ---: | ---: | --- | --- |
{rows}

**{len(cells)} executable cells; {n_files} contract files.**

Derived, no cells of their own:

{chr(10).join(f"* `{d['phase']}` — {d['paper_artifact']} (from {d['derived_from']})" for d in inv.DERIVED_PHASES)}

Blocked:

{chr(10).join(f"* `{b['phase']}` — {b['paper_artifact']}. {b['blocked_on']}" for b in inv.BLOCKED_PHASES)}

## How it is used

```bash
python colab/validate_campaign.py --results-root campaign_results
```

The validator walks this tree, and for each contract checks the mirrored
`campaign_results/` path for existence, schema, provenance, metric validity and
the contract's own completion criteria. A missing cell, a truncated round log, a
`mockhash`, a zero-byte artifact or an out-of-range metric is a failure.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="report what would be written; write nothing")
    ap.add_argument("--clean", action="store_true",
                    help="remove the tree first (it is fully generated)")
    args = ap.parse_args()

    out = os.path.abspath(args.out_dir)
    cells = inv.all_cells()
    written: List[str] = []

    if args.clean and not args.check and os.path.isdir(out):
        shutil.rmtree(out)

    # Per-cell contracts, at the exact mirrored path.
    for c in cells:
        _write(os.path.join(out, *c["log_dir"].split("/"), "experiment_log.json"),
               cell_contract(c), args.check, written)

    # Aggregate artifact contracts.
    for spec in sorted(inv.PHASES, key=lambda p: p["order"]):
        phase_cells = [c for c in cells if c["phase"] == spec["phase"]]
        _write(os.path.join(out, *spec["aggregate_artifact"].split("/")),
               aggregate_contract(spec, phase_cells), args.check, written)

    # Groups that are not one-file-per-cell.
    _write(os.path.join(out, "figures", "EXPECTED_FIGURES.json"),
           figures_contract(), args.check, written)
    _write(os.path.join(out, "statistics", "EXPECTED_STATISTICS.json"),
           statistics_contract(), args.check, written)
    _write(os.path.join(out, "manifest_template.json"),
           manifest_template(cells), args.check, written)
    _write(os.path.join(out, "by_paper_artifact.json"),
           by_paper_artifact(cells), args.check, written)
    _write_text(os.path.join(out, "historical_reference", "README.md"),
                HISTORICAL_README, args.check, written)
    _write_text(os.path.join(out, "README.md"),
                readme(cells, len(written) + 1), args.check, written)

    verb = "would write" if args.check else "wrote"
    print(f"[expected] {verb} {len(written)} files under "
          f"{os.path.relpath(out, ROOT).replace(os.sep, '/')}/")
    print(f"[expected]   {len(cells)} per-cell contracts")
    print(f"[expected]   {len(inv.PHASES)} aggregate-artifact contracts")
    print("[expected]   + manifest_template.json, by_paper_artifact.json, "
          "figures/, statistics/, historical_reference/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
