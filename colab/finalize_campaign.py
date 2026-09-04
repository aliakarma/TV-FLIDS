"""
colab/finalize_campaign.py — write <results-root>/final_manifest.json.

    python colab/finalize_campaign.py --results-root campaign_results

Walks the expected cell inventory, reads whatever is genuinely on disk for each
cell, and records it in the same shape as
expected_results/manifest_template.json so the two can be compared directly by
colab/validate_campaign.py.

Nothing is invented. A cell that never ran is recorded with status "pending" and
null metrics; a cell whose log is truncated is "failed" with the reason. The
per-cell metrics copied in are read verbatim from that cell's own
experiment_log.json summary — this file is an index of what exists, not a
recomputation of it.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import campaign_inventory as inv  # noqa: E402


def _now() -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            .isoformat().replace("+00:00", "Z"))


def _sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _cell_record(root: str, cell: Dict) -> Dict:
    cdir = os.path.join(root, *cell["log_dir"].split("/"))
    lpath = os.path.join(cdir, "experiment_log.json")
    rec = {
        "cell_id": cell["cell_id"],
        "phase": cell["phase"],
        "priority": cell["priority"],
        "paper_artifact": cell["paper_artifact"],
        "strategy": cell["strategy"],
        "attack": cell["attack"],
        "dataset": cell["dataset"],
        "protocol": cell["protocol"],
        "partition_type": cell["partition_type"],
        "alpha": cell["alpha"],
        "arm": cell["arm"],
        "seed": cell["seed"],
        "rounds": cell["rounds"],
        "log_dir": cell["log_dir"],
        "aggregate_artifact": cell["aggregate_artifact"],
        "status": "pending",
        "reason": None,
        "files": {},
        "rounds_logged": None,
        "config_hash": None,
        "git_commit": None,
        "started_utc": None,
        "finished_utc": None,
        "elapsed_seconds": None,
        "metrics": None,
        "log_sha256": None,
    }

    for fname in cell["expected_files"]:
        fpath = os.path.join(cdir, fname)
        rec["files"][fname] = (
            {"exists": True, "size_bytes": os.path.getsize(fpath)}
            if os.path.exists(fpath) else {"exists": False, "size_bytes": None})

    if not os.path.exists(lpath):
        if os.path.exists(os.path.join(cdir, "config.json")):
            rec["status"] = "failed"
            rec["reason"] = "interrupted: config.json written, no experiment_log.json"
        return rec

    try:
        with open(lpath, encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        rec["status"] = "failed"
        rec["reason"] = f"experiment_log.json unreadable: {exc}"
        return rec

    if blob.get("__expected_contract__"):
        rec["status"] = "failed"
        rec["reason"] = ("an expected_results contract file is sitting at this "
                          "result path")
        return rec

    cfg = blob.get("config") or {}
    summary = blob.get("summary") or {}
    prov = blob.get("provenance") or {}
    rounds = blob.get("rounds") or []

    rec["rounds_logged"] = len(rounds)
    rec["config_hash"] = cfg.get("_config_hash")
    rec["git_commit"] = prov.get("git_commit")
    rec["started_utc"] = prov.get("started_utc") or prov.get("timestamp_utc")
    rec["finished_utc"] = prov.get("timestamp_utc")
    rec["elapsed_seconds"] = blob.get("elapsed_seconds")
    rec["log_sha256"] = _sha256(lpath)
    rec["metrics"] = {
        k: summary.get(k) for k in
        ("final_accuracy", "final_f1_macro", "final_attack_success_rate",
         "final_false_negative_rate", "peak_accuracy")
        if k in summary
    } or None

    if len(rounds) != cell["round_log_entries"]:
        rec["status"] = "failed"
        rec["reason"] = (f"{len(rounds)} round entries, expected "
                          f"{cell['round_log_entries']}")
    elif "final_accuracy" not in summary:
        rec["status"] = "failed"
        rec["reason"] = "summary carries no final_accuracy"
    else:
        rec["status"] = "complete"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="campaign_results")
    ap.add_argument("--out", default=None,
                    help="default: <results-root>/final_manifest.json")
    args = ap.parse_args()

    root = args.results_root
    out = args.out or os.path.join(root, "final_manifest.json")

    cells = inv.all_cells()
    records = [_cell_record(root, c) for c in cells]

    counts: Dict[str, int] = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    by_phase: Dict[str, Dict[str, int]] = {}
    for r in records:
        by_phase.setdefault(r["phase"], {})
        by_phase[r["phase"]][r["status"]] = \
            by_phase[r["phase"]].get(r["status"], 0) + 1

    artifacts = []
    for spec in sorted(inv.PHASES, key=lambda p: p["order"]):
        apath = os.path.join(root, *spec["aggregate_artifact"].split("/"))
        artifacts.append({
            "phase": spec["phase"],
            "artifact": spec["aggregate_artifact"],
            "exists": os.path.exists(apath),
            "size_bytes": os.path.getsize(apath) if os.path.exists(apath) else None,
            "sha256": _sha256(apath) if os.path.exists(apath) else None,
        })

    try:
        from utils.provenance import git_state, hardware, package_versions
        env = {"git": git_state(), "hardware": hardware(),
               "packages": package_versions(),
               "python_version": sys.version.split()[0]}
    except Exception as exc:                                  # noqa: BLE001
        env = {"error": f"provenance unavailable: {type(exc).__name__}: {exc}",
               "python_version": sys.version.split()[0]}

    payload = {
        "campaign": "TV-FLIDS full experimental campaign (Colab, T4)",
        "manifest_version": 1,
        "role": "ACTUAL. Compare against expected_results/manifest_template.json "
                 "with colab/validate_campaign.py.",
        "generated_utc": _now(),
        "results_root": root,
        "rounds_per_cell": inv.ROUNDS,
        "round_log_entries_per_cell": inv.ROUND_LOG_ENTRIES,
        "seeds": inv.SEEDS,
        "extended_seeds": inv.EXTENDED_SEEDS,
        "n_cells": len(records),
        "counts": counts,
        "counts_by_phase": by_phase,
        "aggregate_artifacts": artifacts,
        "environment": env,
        "cells": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, out)

    print(f"[finalize] wrote {out}")
    print(f"[finalize]   {len(records)} cells: {counts}")
    present = sum(1 for a in artifacts if a["exists"])
    print(f"[finalize]   aggregate artifacts present: {present}/{len(artifacts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
