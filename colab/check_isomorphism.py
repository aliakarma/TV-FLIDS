"""
colab/check_isomorphism.py — prove expected_results/ and campaign_results/ are
path-for-path isomorphic.

    python colab/check_isomorphism.py
    python colab/check_isomorphism.py --results-root campaign_results
    python colab/check_isomorphism.py --scaffold      # create the empty tree

Three assertions, each a way the mapping could silently break:

  1  Every path the inventory declares has a contract under expected_results/,
     at exactly that relative path.
  2  Every contract file under expected_results/ corresponds to a path the
     inventory declares (no orphan contracts left behind by an older grid).
  3  Every result present under the results root sits at a path that has a
     contract (no result written somewhere the contract does not describe).

Assertions 1 and 2 hold before a single experiment runs, so this is the check
to run right after regenerating the contract tree. Assertion 3 becomes
meaningful as the campaign fills in.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Set

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import campaign_inventory as inv  # noqa: E402

EXPECTED = "expected_results"

# Files that describe the tree rather than mirroring a result path.
META = {
    "README.md",
    "manifest_template.json",
    "by_paper_artifact.json",
    os.path.join("figures", "EXPECTED_FIGURES.json").replace(os.sep, "/"),
    os.path.join("statistics", "EXPECTED_STATISTICS.json").replace(os.sep, "/"),
    os.path.join("historical_reference", "README.md").replace(os.sep, "/"),
}


def declared_paths() -> Dict[str, str]:
    """relative path -> what declares it."""
    out: Dict[str, str] = {}
    for c in inv.all_cells():
        out[f"{c['log_dir']}/experiment_log.json"] = c["cell_id"]
    for spec in inv.PHASES:
        out[spec["aggregate_artifact"]] = spec["phase"]
    return out


def contract_paths(expected_dir: str) -> Set[str]:
    out: Set[str] = set()
    for p in glob.glob(os.path.join(expected_dir, "**", "*"), recursive=True):
        if not os.path.isfile(p):
            continue
        rel = os.path.relpath(p, expected_dir).replace(os.sep, "/")
        if rel in META or rel.startswith("historical_reference/"):
            continue
        out.add(rel)
    return out


# Keys whose *name* would make a number a prediction rather than a bound.
_BANNED_KEY = re.compile(
    r"(expected|target|predicted|reference)_?"
    r"(accuracy|f1|asr|attack_success|precision|recall|p_?value|"
    r"improvement|delta|gain)", re.I)
# Keys that name an outcome. A number here is a measurement, which a contract
# must never carry — except inside metric_ranges, where it is a valid bound.
_OUTCOME_KEYS = {
    "accuracy", "f1", "f1_macro", "f1_weighted", "asr", "attack_success_rate",
    "false_negative_rate", "precision", "recall", "p_value", "pvalue",
    "cohens_d", "improvement", "delta", "final_accuracy", "final_f1_macro",
    "final_attack_success_rate", "peak_accuracy",
}


def assert_no_targets(expected_dir: str) -> List[str]:
    """The whole point of the contract tree: it predicts no measurement.

    Walks every contract and reports any numeric value sitting under a key that
    names an experimental outcome. Bounds under `metric_ranges` are exempt --
    "accuracy lies in [0, 1]" is a definition, not a prediction.
    """
    problems: List[str] = []
    for p in sorted(glob.glob(os.path.join(expected_dir, "**", "*.json"),
                              recursive=True)):
        try:
            with open(p, encoding="utf-8") as fh:
                blob = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{p}: unreadable ({exc})")
            continue

        def walk(node, path: str = "") -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    kp = f"{path}.{k}" if path else k
                    numeric = (isinstance(v, (int, float))
                               and not isinstance(v, bool))
                    if numeric and _BANNED_KEY.search(k):
                        problems.append(f"{p}: {kp} = {v!r} reads as a "
                                         f"predicted outcome")
                    elif (numeric and k in _OUTCOME_KEYS
                          and "metric_ranges" not in kp):
                        problems.append(f"{p}: {kp} = {v!r} is an outcome "
                                         f"value outside metric_ranges")
                    walk(v, kp)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(blob)
    return problems


def scaffold(results_root: str, declared: Dict[str, str]) -> int:
    """Create the empty campaign tree so the mapping is visible before any run."""
    made = 0
    for rel in declared:
        d = os.path.join(results_root, *os.path.dirname(rel).split("/"))
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            made += 1
    for sub in ("figures", "statistics", "_campaign", "_campaign_logs"):
        os.makedirs(os.path.join(results_root, sub), exist_ok=True)
    readme = os.path.join(results_root, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as fh:
            fh.write(f"""# campaign_results/

The campaign's **actual** output. Path-for-path isomorphic with
`expected_results/`, which holds the structure contract for each of these
paths:

    expected_results/<path>   <- contract: schema, provenance, valid ranges
    campaign_results/<path>   <- result:   what the run actually produced

Written by the repository's own runners via `colab/parallel_runner.py`. On
Colab this directory normally lives on Google Drive, and `<repo>/results` is a
symlink to it, so the runners' hardcoded `results/...` prefix lands here.

Nothing in this tree is authored by hand. Every file is produced by an
execution and carries its own provenance.

Regenerate the mapping check with:

    python colab/check_isomorphism.py --results-root campaign_results
""")
        made += 1
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="campaign_results")
    ap.add_argument("--expected-dir", default=EXPECTED)
    ap.add_argument("--scaffold", action="store_true",
                    help="create the empty campaign_results/ tree")
    args = ap.parse_args()

    declared = declared_paths()
    contracts = contract_paths(args.expected_dir)

    print("=" * 74)
    print("EXPECTED / ACTUAL ISOMORPHISM")
    print("=" * 74)
    print(f"  inventory declares      {len(declared)} result path(s)")
    print(f"  expected_results holds  {len(contracts)} contract file(s)")

    errors: List[str] = []

    missing = sorted(set(declared) - contracts)
    for m in missing[:20]:
        errors.append(f"declared but has no contract: {m}")
    if len(missing) > 20:
        errors.append(f"(+{len(missing) - 20} more declared without contracts)")

    orphan = sorted(contracts - set(declared))
    for o in orphan[:20]:
        errors.append(f"contract with nothing declaring it (stale?): {o}")
    if len(orphan) > 20:
        errors.append(f"(+{len(orphan) - 20} more orphan contracts)")

    print(f"  declared without contract  {len(missing)}")
    print(f"  orphan contracts           {len(orphan)}")

    # The contract tree must predict no measurement. This is the scientific
    # integrity assertion, not a style check.
    targets = assert_no_targets(args.expected_dir)
    print(f"  predicted outcome values   {len(targets)}"
          + ("  (must be 0)" if targets else "  <- none, as required"))
    for t in targets[:20]:
        errors.append(t)
    if len(targets) > 20:
        errors.append(f"(+{len(targets) - 20} more predicted outcome values)")

    if args.scaffold:
        made = scaffold(args.results_root, declared)
        print(f"  scaffolded                 {made} director(y/ies) under "
              f"{args.results_root}/")

    # Assertion 3: nothing has been written outside the contract.
    if os.path.isdir(args.results_root):
        stray = []
        for p in glob.glob(os.path.join(args.results_root, "logs", "**",
                                         "experiment_log.json"), recursive=True):
            rel = os.path.relpath(p, args.results_root).replace(os.sep, "/")
            if rel not in declared:
                stray.append(rel)
        print(f"  results outside contract   {len(stray)}")
        for s in stray[:15]:
            errors.append(f"result at an undeclared path: {s}")
        if len(stray) > 15:
            errors.append(f"(+{len(stray) - 15} more results at undeclared paths)")
    else:
        print(f"  results root               {args.results_root} does not "
              f"exist yet (run with --scaffold)")

    print()
    if errors:
        for e in errors:
            print(f"  ERROR  {e}")
        print(f"\nNOT ISOMORPHIC: {len(errors)} problem(s).")
        return 1
    print("ISOMORPHIC: every declared path has a contract, every contract has "
          "a declaration,\nand no result sits outside the contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
