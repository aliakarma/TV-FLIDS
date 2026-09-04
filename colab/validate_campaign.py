"""
colab/validate_campaign.py — decide whether a campaign is genuinely finished.

    python colab/validate_campaign.py --results-root campaign_results
    python colab/validate_campaign.py --results-root campaign_results --priority 1
    python colab/validate_campaign.py --results-root campaign_results --json report.json
    python colab/validate_campaign.py --results-root campaign_results --strict

Exit code 0 only when every enabled check passes. `--strict` additionally fails
on warnings (e.g. a dirty git tree behind a result).

WHAT IT CHECKS
--------------
  completeness   every expected cell, seed, round and output file exists
  integrity      no mockhash, no quarantined output, no zero-byte artifact,
                 no malformed PDF, no contract file masquerading as a result
  provenance     git commit, config hash, seed, dataset, environment
  numerical      every metric inside its mathematically valid range
  statistical    test directions, seed counts, no reused historical p-values
  manuscript     every table and figure the manuscript inputs has a live source
  manifest       expected_results/manifest_template.json vs.
                 <results-root>/final_manifest.json

The expected side comes from colab/campaign_inventory.py, which is itself
drift-checked against the repository's own runner constants. This validator
never compares a result against a historical or published number: no target
value exists anywhere in expected_results/, and anything under
expected_results/historical_reference/ is refused as a comparison source.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import campaign_inventory as inv  # noqa: E402

MOCK_FINGERPRINTS = ["mockhash"]
FORBIDDEN_CONFIG_HASHES = {"mockhash", "", "none", "null", "placeholder", "todo"}
HISTORICAL_DIR = os.path.join("expected_results", "historical_reference")


def _now() -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            .isoformat().replace("+00:00", "Z"))


class Report:
    def __init__(self) -> None:
        self.sections: Dict[str, Dict] = {}

    def section(self, name: str) -> Dict:
        return self.sections.setdefault(
            name, {"errors": [], "warnings": [], "stats": {}})

    def err(self, section: str, msg: str) -> None:
        self.section(section)["errors"].append(msg)

    def warn(self, section: str, msg: str) -> None:
        self.section(section)["warnings"].append(msg)

    def stat(self, section: str, key: str, value) -> None:
        self.section(section)["stats"][key] = value

    @property
    def n_errors(self) -> int:
        return sum(len(s["errors"]) for s in self.sections.values())

    @property
    def n_warnings(self) -> int:
        return sum(len(s["warnings"]) for s in self.sections.values())


def _load_json(path: str) -> Tuple[Optional[dict], Optional[str]]:
    if not os.path.exists(path):
        return None, "absent"
    try:
        if os.path.getsize(path) == 0:
            return None, "zero bytes"
    except OSError as exc:
        return None, str(exc)
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _in_range(value, spec: Dict) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return False
    lo, hi = spec.get("min"), spec.get("max")
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


# ── 1. Completeness + numerical validity, cell by cell ───────────────────────

def check_cells(root: str, cells: List[Dict], rep: Report,
                sample_rounds: bool) -> Dict[str, Dict]:
    sec = "completeness"
    ranges = inv.metric_ranges()
    per_cell: Dict[str, Dict] = {}
    n_ok = n_missing = n_bad = 0

    for cell in cells:
        cdir = os.path.join(root, *cell["log_dir"].split("/"))
        lpath = os.path.join(cdir, "experiment_log.json")
        rec = {"cell_id": cell["cell_id"], "phase": cell["phase"],
               "seed": cell["seed"], "log_dir": cell["log_dir"],
               "status": "missing", "config_hash": None,
               "rounds_logged": None, "final_accuracy": None,
               "elapsed_seconds": None, "problems": []}

        blob, why = _load_json(lpath)
        if blob is None:
            rec["problems"].append(f"experiment_log.json {why}")
            # An interrupted cell leaves config.json but no log: report it as
            # incomplete rather than never-started, so it can be retried.
            if os.path.exists(os.path.join(cdir, "config.json")):
                rec["status"] = "incomplete"
                rec["problems"].append(
                    "config.json present without experiment_log.json: "
                    "interrupted mid-run")
            rep.err(sec, f"{cell['cell_id']}: experiment_log.json {why}")
            n_missing += 1
            per_cell[cell["cell_id"]] = rec
            continue

        # A contract file must never sit where a result belongs.
        if blob.get("__expected_contract__"):
            rec["status"] = "contract_in_results"
            msg = ("an expected_results contract file is sitting at a "
                   "campaign_results path")
            rec["problems"].append(msg)
            rep.err("integrity", f"{cell['cell_id']}: {msg} ({lpath})")
            n_bad += 1
            per_cell[cell["cell_id"]] = rec
            continue

        problems: List[str] = []
        cfg = blob.get("config") or {}
        summary = blob.get("summary") or {}
        prov = blob.get("provenance") or {}
        rounds = blob.get("rounds") or []

        rec["rounds_logged"] = len(rounds)
        rec["config_hash"] = cfg.get("_config_hash")
        rec["final_accuracy"] = summary.get("final_accuracy")
        rec["elapsed_seconds"] = blob.get("elapsed_seconds")

        # ── sibling files ──
        for fname in cell["expected_files"]:
            fpath = os.path.join(cdir, fname)
            if not os.path.exists(fpath):
                problems.append(f"{fname} absent")
            elif os.path.getsize(fpath) == 0:
                problems.append(f"{fname} is zero bytes")

        # ── round coverage ──
        want = cell["round_log_entries"]
        if len(rounds) != want:
            problems.append(f"{len(rounds)} round entries, expected {want}")
        idx = [r.get("round") for r in rounds]
        if idx != list(range(0, cell["rounds"] + 1)):
            missing = sorted(set(range(0, cell["rounds"] + 1)) - set(
                i for i in idx if isinstance(i, int)))
            dupes = len(idx) - len(set(idx))
            detail = []
            if missing:
                detail.append(f"missing rounds {missing[:6]}"
                               + ("..." if len(missing) > 6 else ""))
            if dupes:
                detail.append(f"{dupes} duplicate round index(es)")
            problems.append("round indices are not 0.."
                            f"{cell['rounds']} exactly"
                            + (": " + "; ".join(detail) if detail else ""))

        # ── per-round keys + metric ranges ──
        to_scan = rounds if not sample_rounds else (
            rounds[:2] + rounds[len(rounds) // 2: len(rounds) // 2 + 1]
            + rounds[-2:] if rounds else [])
        missing_keys = set()
        for r in to_scan:
            for k in cell["round_keys"]:
                if k not in r:
                    missing_keys.add(k)
            for k, v in r.items():
                if k in ranges and not _in_range(v, ranges[k]):
                    problems.append(
                        f"round {r.get('round')}: {k}={v!r} outside "
                        f"[{ranges[k]['min']}, {ranges[k]['max']}]")
        if missing_keys:
            problems.append(f"round entries missing key(s) {sorted(missing_keys)}")

        # trust weights must form a convex combination where they are logged
        if cell["strategy"] in inv.TRUST_STRATEGIES:
            for r in to_scan:
                a, b, g = (r.get("trust_adaptive_alpha"),
                           r.get("trust_adaptive_beta"),
                           r.get("trust_adaptive_gamma"))
                if all(isinstance(x, (int, float)) for x in (a, b, g)):
                    if abs((a + b + g) - 1.0) > 1e-6:
                        problems.append(
                            f"round {r.get('round')}: adaptive weights sum to "
                            f"{a + b + g:.6f}, not 1")
                        break

        # ── summary ──
        for k in inv.SUMMARY_KEYS:
            if k not in summary:
                problems.append(f"summary missing {k}")
        for k in ("final_accuracy", "final_f1_macro",
                  "final_attack_success_rate", "final_false_negative_rate",
                  "peak_accuracy"):
            if k in summary:
                base = k.replace("final_", "")
                spec = ranges.get(base if base in ranges else "accuracy")
                if not _in_range(summary[k], spec):
                    problems.append(f"summary.{k}={summary[k]!r} outside "
                                     f"[{spec['min']}, {spec['max']}]")
        if isinstance(summary.get("num_rounds"), int) and \
                summary["num_rounds"] != cell["rounds"]:
            problems.append(f"summary.num_rounds={summary['num_rounds']}, "
                             f"expected {cell['rounds']}")

        # ── identity must match what was asked for ──
        for field, want_val in (("strategy", cell["strategy"]),
                                 ("attack", cell["attack"]),
                                 ("dataset", cell["dataset"]),
                                 ("protocol", cell["protocol"]),
                                 ("seed", cell["seed"])):
            got = cfg.get(field)
            if got is not None and got != want_val:
                problems.append(f"config.{field}={got!r}, expected {want_val!r}")
        if cell["seed"] not in inv.EXTENDED_SEEDS:
            problems.append(f"seed {cell['seed']} is not in the declared seed set")

        # ── config hash ──
        ch = cfg.get("_config_hash")
        if ch is None or str(ch).strip().lower() in FORBIDDEN_CONFIG_HASHES:
            problems.append(f"config._config_hash is not a real hash: {ch!r}")
            rep.err("integrity", f"{cell['cell_id']}: config hash {ch!r}")

        # ── provenance ──
        prov_missing = []
        if not prov:
            prov_missing = list(inv.PROVENANCE_REQUIRED)
        else:
            if not prov.get("git_commit") or "unknown" in str(prov["git_commit"]):
                prov_missing.append("git_commit")
            if not prov.get("config_hash"):
                prov_missing.append("config_hash")
            if cfg.get("seed") is None:
                prov_missing.append("seed")
            if not cfg.get("dataset"):
                prov_missing.append("dataset")
            # "environment" is satisfied by the recorded stack + hardware.
            if not (prov.get("packages") and prov.get("python_version")):
                prov_missing.append("environment")
            if prov.get("git_dirty"):
                rep.warn("provenance",
                         f"{cell['cell_id']}: produced from a dirty tree "
                         f"({len(prov.get('git_dirty_paths') or [])} path(s))")
        if prov_missing:
            problems.append(f"provenance missing {prov_missing}")
            rep.err("provenance",
                    f"{cell['cell_id']}: provenance missing {prov_missing}")

        rec["problems"] = problems
        rec["status"] = "complete" if not problems else "invalid"
        if problems:
            n_bad += 1
            for p in problems[:6]:
                rep.err(sec, f"{cell['cell_id']}: {p}")
            if len(problems) > 6:
                rep.err(sec, f"{cell['cell_id']}: (+{len(problems)-6} more)")
        else:
            n_ok += 1
        per_cell[cell["cell_id"]] = rec

    rep.stat(sec, "cells_expected", len(cells))
    rep.stat(sec, "cells_complete", n_ok)
    rep.stat(sec, "cells_missing", n_missing)
    rep.stat(sec, "cells_invalid", n_bad)
    return per_cell


# ── 2. Seed coverage per phase ───────────────────────────────────────────────

def check_seed_coverage(cells: List[Dict], per_cell: Dict[str, Dict],
                        rep: Report) -> None:
    sec = "completeness"
    groups: Dict[Tuple, Dict[int, str]] = {}
    for c in cells:
        key = (c["phase"], c["strategy"], c["attack"], c["protocol"],
               str(c["arm"]), str(c["alpha"]))
        groups.setdefault(key, {})[c["seed"]] = \
            per_cell.get(c["cell_id"], {}).get("status", "missing")
    incomplete = 0
    for key, seeds in sorted(groups.items()):
        bad = sorted(s for s, st in seeds.items() if st != "complete")
        if bad:
            incomplete += 1
            rep.err(sec, "condition " + "/".join(str(k) for k in key)
                    + f": seed(s) {bad} not complete")
    rep.stat(sec, "conditions_total", len(groups))
    rep.stat(sec, "conditions_incomplete", incomplete)


# ── 3. Integrity sweep over the whole results tree ───────────────────────────

def check_integrity(root: str, rep: Report,
                    cells: List[Dict]) -> None:
    sec = "integrity"
    n_json = n_pdf = n_quarantined = 0

    # No expected cell or aggregate may live inside the quarantine.
    for cell in cells:
        if "_QUARANTINED_MOCK" in cell["log_dir"]:
            rep.err(sec, f"{cell['cell_id']}: expected log_dir is inside "
                          f"the quarantine ({cell['log_dir']})")
    for spec in inv.PHASES:
        if "_QUARANTINED_MOCK" in spec["aggregate_artifact"]:
            rep.err(sec, f"{spec['phase']}: aggregate artifact is inside "
                          f"the quarantine")
    for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not os.path.isfile(path):
            continue
        norm = path.replace(os.sep, "/")
        # results/_QUARANTINED_MOCK/ is the archival home of the fabricated
        # artifacts found during the forensic audit. It is *expected* to sit
        # under the results root and is *expected* to match every fabrication
        # fingerprint, so it is skipped here exactly as
        # scripts/check_results.py skips it. What matters is that no live cell
        # or aggregate path resolves inside it, which is asserted separately.
        if "_QUARANTINED_MOCK" in norm:
            n_quarantined += 1
            continue
        try:
            size = os.path.getsize(path)
        except OSError as exc:
            rep.err(sec, f"{norm}: {exc}")
            continue

        if norm.endswith(".json"):
            n_json += 1
            if size == 0:
                rep.err(sec, f"zero-byte JSON: {norm}")
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError as exc:
                rep.err(sec, f"{norm}: {exc}")
                continue
            hits = [fp for fp in MOCK_FINGERPRINTS if fp in text]
            if hits:
                rep.err(sec, f"fabrication fingerprint {hits} in {norm}")
            if '"__expected_contract__"' in text:
                rep.err(sec, f"expected_results contract file inside the "
                              f"results root: {norm}")
        elif norm.endswith(".pdf"):
            n_pdf += 1
            if size == 0:
                rep.err(sec, f"zero-byte PDF (placeholder, not a figure): {norm}")
                continue
            with open(path, "rb") as fh:
                if fh.read(5) != b"%PDF-":
                    rep.err(sec, f"malformed PDF (no %PDF- header): {norm}")
    rep.stat(sec, "json_files_scanned", n_json)
    rep.stat(sec, "pdf_files_scanned", n_pdf)
    rep.stat(sec, "quarantine_files_skipped", n_quarantined)

    # The historical reference must stay out of the validation path entirely.
    hist = os.path.join(ROOT, HISTORICAL_DIR)
    if os.path.isdir(hist):
        payloads = [p for p in glob.glob(os.path.join(hist, "**", "*"),
                                          recursive=True)
                    if os.path.isfile(p) and not p.endswith("README.md")]
        rep.stat(sec, "historical_reference_files", len(payloads))
        for p in payloads:
            base = os.path.basename(p)
            if not base.startswith("HISTORICAL_NON_TARGET_"):
                rep.warn(sec, f"{p}: file in historical_reference/ is not "
                               f"prefixed HISTORICAL_NON_TARGET_")


# ── 4. Aggregate artifacts + statistics ──────────────────────────────────────

def check_aggregates(root: str, cells: List[Dict], per_cell: Dict[str, Dict],
                     rep: Report, phases: Optional[List[str]]) -> None:
    sec = "aggregates"
    ranges = inv.metric_ranges()
    by_phase = inv.cells_by_phase()

    for spec in sorted(inv.PHASES, key=lambda p: p["order"]):
        if phases and spec["phase"] not in phases:
            continue
        pcells = by_phase.get(spec["phase"], [])
        if not pcells:
            continue
        all_cells_ok = all(
            per_cell.get(c["cell_id"], {}).get("status") == "complete"
            for c in pcells)
        apath = os.path.join(root, *spec["aggregate_artifact"].split("/"))
        blob, why = _load_json(apath)

        if blob is None:
            if all_cells_ok:
                rep.err(sec, f"{spec['phase']}: every cell is complete but the "
                              f"aggregate {spec['aggregate_artifact']} is {why}")
            else:
                rep.warn(sec, f"{spec['phase']}: aggregate absent ({why}); its "
                               f"cells are not all complete yet")
            continue
        if blob.get("__expected_contract__"):
            rep.err("integrity", f"{spec['phase']}: contract file at the "
                                  f"aggregate path {apath}")
            continue

        # Statistics carried inside the aggregate must be well-formed.
        _walk_stats(blob, f"{spec['phase']}:{spec['aggregate_artifact']}",
                    rep, ranges)


def _walk_stats(node, where: str, rep: Report, ranges: Dict,
                depth: int = 0) -> None:
    """Recursively validate p-values, stds and metric means in an artifact."""
    if depth > 8:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            kl = k.lower()
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if kl in ("p_value", "pvalue", "p") or kl.endswith("_p_value"):
                    if not _in_range(v, ranges["p_value"]):
                        rep.err("statistical",
                                f"{where}: {k}={v!r} is not a probability")
                elif kl.endswith("_std") or kl == "std":
                    if v < 0:
                        rep.err("statistical",
                                f"{where}: {k}={v!r} is a negative "
                                f"standard deviation")
                elif kl in ("n_seeds", "num_seeds"):
                    if not isinstance(v, int) or v < 1:
                        rep.err("statistical",
                                f"{where}: {k}={v!r} is not a positive count")
                    elif v not in (len(inv.SEEDS), len(inv.EXTENDED_SEEDS)):
                        rep.warn("statistical",
                                 f"{where}: {k}={v} is neither "
                                 f"{len(inv.SEEDS)} nor "
                                 f"{len(inv.EXTENDED_SEEDS)} seeds")
                elif kl in ("accuracy_mean", "asr_mean", "f1_mean",
                             "final_accuracy", "final_f1_macro",
                             "final_attack_success_rate"):
                    base = ("accuracy" if "accuracy" in kl else
                            "attack_success_rate" if "asr" in kl or "attack" in kl
                            else "f1_macro")
                    if not _in_range(v, ranges[base]):
                        rep.err("numerical",
                                f"{where}: {k}={v!r} outside [0, 1]")
            else:
                _walk_stats(v, f"{where}.{k}", rep, ranges, depth + 1)
    elif isinstance(node, list):
        for i, v in enumerate(node[:200]):
            _walk_stats(v, f"{where}[{i}]", rep, ranges, depth + 1)


def check_statistical_direction(rep: Report) -> None:
    """The one-sided direction must be metric-appropriate in the code itself."""
    sec = "statistical"
    path = os.path.join(ROOT, "evaluation", "statistical_testing.py")
    if not os.path.exists(path):
        rep.err(sec, "evaluation/statistical_testing.py is absent")
        return
    src = open(path, encoding="utf-8", errors="replace").read()
    rep.stat(sec, "asr_direction_expected", "lower_is_better")
    lowered = src.lower()
    if "alternative" not in lowered:
        rep.warn(sec, "statistical_testing.py names no `alternative`; confirm "
                       "every one-sided test states its direction explicitly")
    if "attack_success_rate" not in src and "asr" not in lowered:
        rep.warn(sec, "statistical_testing.py never mentions "
                       "attack_success_rate; confirm the ASR comparison uses "
                       "the lower-is-better direction rather than inheriting "
                       "the accuracy direction")


# ── 5. Manuscript mapping ────────────────────────────────────────────────────

def check_manuscript_mapping(root: str, rep: Report) -> None:
    sec = "manuscript"
    tables_dir = os.path.join(ROOT, "Paper", "tables")
    figs_dir = os.path.join(ROOT, "Paper", "figures")
    sources = {
        spec["aggregate_artifact"]: spec["phase"]
        for spec in inv.PHASES
    }
    present = {a for a in sources
               if os.path.exists(os.path.join(root, *a.split("/")))
               and os.path.getsize(os.path.join(root, *a.split("/"))) > 0}
    rep.stat(sec, "aggregate_artifacts_expected", len(sources))
    rep.stat(sec, "aggregate_artifacts_present", len(present))
    for a, phase in sorted(sources.items()):
        if a not in present:
            rep.warn(sec, f"{phase}: no live source at {a}; every manuscript "
                           f"table or figure fed by it stays unbacked")

    # The manuscript selects its table bodies with
    #   \IfFileExists{tables/tab_X.tex}{...}{tables/tab_X_provisional.tex}
    # so a *_provisional.tex file is the repository's deliberate, explicitly
    # labelled fallback, not a defect. What matters is (a) whether the
    # generated tab_X.tex exists yet, and (b) that a *generated* file never
    # contains provisional text.
    for d, kind in ((tables_dir, "table"), (figs_dir, "figure")):
        if not os.path.isdir(d):
            continue
        gen = sorted(glob.glob(os.path.join(d, "*.tex")))
        bodies = [p for p in gen
                  if not os.path.basename(p).endswith(
                      ("_provisional.tex", "_head.tex", "_foot.tex"))]
        placeholders = [p for p in gen
                        if os.path.basename(p).endswith("_provisional.tex")]
        rep.stat(sec, f"{kind}_tex_files", len(gen))
        rep.stat(sec, f"{kind}_generated_bodies", len(bodies))
        rep.stat(sec, f"{kind}_labelled_provisional", len(placeholders))

        for p in gen:
            base = os.path.basename(p)
            if os.path.getsize(p) == 0:
                rep.err(sec, f"zero-byte {kind} body: {base}")
                continue
            # *_head.tex / *_foot.tex are the hand-authored LaTeX scaffolding
            # around a generated body (column spec, caption, rules). They are
            # meant to be hand-maintained and carry no data.
            if base.endswith(("_head.tex", "_foot.tex")):
                continue
            try:
                text = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            is_placeholder = base.endswith("_provisional.tex")
            provisional = ("provisional" in text.lower()
                           or "placeholder" in text.lower())
            banner = "GENERATED FILE" in text

            if is_placeholder:
                # Expected to say provisional. Only report the consequence:
                # the corresponding generated body does not exist yet.
                stem = base[: -len("_provisional.tex")]
                if not os.path.exists(os.path.join(d, stem + ".tex")):
                    rep.warn(sec, f"{kind} '{stem}' is still served by its "
                                   f"labelled provisional body: no "
                                   f"{stem}.tex has been generated from a "
                                   f"real artifact yet")
                continue

            if provisional:
                # A generated body must never claim to be provisional.
                rep.err(sec, f"{base} is a generated {kind} body but contains "
                              f"provisional/placeholder text"
                              + (" (and carries the generated banner)"
                                 if banner else ""))
            elif not banner and base.startswith(("tab_", "fig_")):
                rep.warn(sec, f"{base} carries no 'GENERATED FILE' banner - "
                               f"confirm it was written by the generator and "
                               f"not by hand")


# ── 6. Manifest comparison ───────────────────────────────────────────────────

def check_manifest(root: str, per_cell: Dict[str, Dict], rep: Report) -> None:
    sec = "manifest"
    tpath = os.path.join(ROOT, "expected_results", "manifest_template.json")
    fpath = os.path.join(root, "final_manifest.json")
    template, why_t = _load_json(tpath)
    final, why_f = _load_json(fpath)

    if template is None:
        rep.err(sec, f"expected_results/manifest_template.json {why_t} "
                      f"(run colab/generate_expected_results.py)")
        return
    rep.stat(sec, "template_cells", template.get("n_cells"))

    if final is None:
        rep.warn(sec, f"{root}/final_manifest.json {why_f} - write it with "
                       f"colab/finalize_campaign.py once the run completes")
        return

    # Compare only within the validated scope, for the same reason.
    scope = set(per_cell)
    t_ids = {c["cell_id"] for c in template.get("cells", [])} & scope
    f_ids = {c["cell_id"] for c in final.get("cells", [])} & scope
    only_t = sorted(t_ids - f_ids)
    only_f = sorted(f_ids - t_ids)
    rep.stat(sec, "final_cells", len(f_ids))
    rep.stat(sec, "cells_missing_from_final", len(only_t))
    rep.stat(sec, "cells_not_in_template", len(only_f))
    for c in only_t[:20]:
        rep.err(sec, f"expected cell absent from final_manifest.json: {c}")
    if len(only_t) > 20:
        rep.err(sec, f"(+{len(only_t) - 20} more expected cells absent)")
    for c in only_f[:20]:
        rep.err(sec, f"final_manifest.json cell is not in the expected "
                      f"inventory: {c}")

    # A cell the manifest calls complete must actually be complete on disk.
    # Only cells validated in THIS invocation can be judged: a --phase or
    # --priority filter leaves the rest unexamined, and "unexamined" is not
    # evidence of anything.
    for c in final.get("cells", []):
        if c["cell_id"] not in per_cell:
            continue
        if c.get("status") == "complete":
            got = per_cell.get(c["cell_id"], {}).get("status")
            if got != "complete":
                rep.err(sec, f"{c['cell_id']}: final_manifest says complete "
                              f"but on-disk validation says {got!r}")


# ── Rendering ────────────────────────────────────────────────────────────────

def render(rep: Report, per_cell: Dict[str, Dict], strict: bool,
           show_table: bool) -> int:
    order = ["completeness", "integrity", "provenance", "numerical",
             "statistical", "aggregates", "manuscript", "manifest"]
    print("=" * 78)
    print("TV-FLIDS CAMPAIGN VALIDATION")
    print("=" * 78)
    for name in order:
        sec = rep.sections.get(name)
        if sec is None:
            continue
        status = "PASS" if not sec["errors"] else "FAIL"
        if not sec["errors"] and sec["warnings"] and strict:
            status = "FAIL"
        print(f"\n[{status}] {name}")
        for k, v in sec["stats"].items():
            print(f"    {k:<34} {v}")
        for e in sec["errors"][:25]:
            print(f"    ERROR   {e}")
        if len(sec["errors"]) > 25:
            print(f"    ERROR   (+{len(sec['errors']) - 25} more)")
        for w in sec["warnings"][:15]:
            print(f"    warn    {w}")
        if len(sec["warnings"]) > 15:
            print(f"    warn    (+{len(sec['warnings']) - 15} more)")

    if show_table:
        print("\n" + "=" * 78)
        print("PER-SEED RESULT MAP")
        print("=" * 78)
        print(f"{'Experiment':<26} {'Seed':>6} {'Status':<12} "
              f"{'Config Hash':<14} {'Rounds':>6} {'Output'}")
        print("-" * 118)
        for cid in sorted(per_cell):
            r = per_cell[cid]
            print(f"{r['phase']:<26} {r['seed']:>6} {r['status']:<12} "
                  f"{str(r['config_hash'] or '-'):<14} "
                  f"{str(r['rounds_logged'] or '-'):>6} {r['log_dir']}")

    print("\n" + "=" * 78)
    verdict = ("PASS" if rep.n_errors == 0 and
               (not strict or rep.n_warnings == 0) else "FAIL")
    print(f"{rep.n_errors} error(s), {rep.n_warnings} warning(s)  ->  {verdict}")
    print("=" * 78)
    return 0 if verdict == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="campaign_results")
    ap.add_argument("--priority", type=int, action="append", default=None)
    ap.add_argument("--phase", action="append", default=None)
    ap.add_argument("--json", default=None, help="write the full report here")
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as failures")
    ap.add_argument("--sample-rounds", action="store_true",
                    help="scan a sample of rounds per cell instead of all 101 "
                          "(much faster on a full 580-cell tree)")
    ap.add_argument("--table", action="store_true",
                    help="print the per-seed experiment/seed/status table")
    args = ap.parse_args()

    root = args.results_root
    if not os.path.isdir(root):
        print(f"[validate] results root {root!r} does not exist.")
        return 2

    cells = inv.all_cells(args.phase)
    if args.priority:
        cells = [c for c in cells if c["priority"] in args.priority]
    if not cells:
        print("[validate] no cells selected.")
        return 2

    rep = Report()
    rep.stat("completeness", "results_root", root)
    rep.stat("completeness", "validated_utc", _now())

    per_cell = check_cells(root, cells, rep, args.sample_rounds)
    check_seed_coverage(cells, per_cell, rep)
    check_integrity(root, rep, cells)
    check_aggregates(root, cells, per_cell, rep, args.phase)
    check_statistical_direction(rep)
    check_manuscript_mapping(root, rep)
    check_manifest(root, per_cell, rep)

    rc = render(rep, per_cell, args.strict, args.table)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "validated_utc": _now(),
                "results_root": root,
                "verdict": "PASS" if rc == 0 else "FAIL",
                "n_errors": rep.n_errors,
                "n_warnings": rep.n_warnings,
                "sections": rep.sections,
                "cells": per_cell,
            }, fh, indent=2, default=str)
        print(f"[validate] report -> {args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
