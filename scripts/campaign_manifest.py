"""
scripts/campaign_manifest.py - machine-readable manifest for an experimental
campaign.

Two modes:

    python scripts/campaign_manifest.py init     [--out results/_campaign/manifest.json]
    python scripts/campaign_manifest.py finalize [--out results/_campaign/manifest.json]

"init" records the *starting* state: git commit and working-tree cleanliness,
environment versions, hardware, the configuration files verbatim plus their
hashes, dataset files with sizes and SHA-256, the seed list, the planned
experiment list, and the campaign start time.

"finalize" walks every genuine run artifact actually on disk
(results/logs/**/experiment_log.json and results/tables/*.json), reads the
provenance block each one carries, hashes the artifact, and writes the whole
inventory back into the manifest alongside an end timestamp.

Nothing here invents a run. A cell that was never executed simply does not
appear; a run whose log lacks a provenance block is listed with
"provenance": null, which is a finding, not a default.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.provenance import (  # noqa: E402
    file_sha256, git_state, hardware, package_versions,
)

DEFAULT_OUT = os.path.join("results", "_campaign", "manifest.json")

CONFIG_FILES = [
    "config/fl_config.yaml",
    "config/dataset_config.yaml",
]

DATASET_FILES = [
    "data/raw/KDDTrain+.txt",
    "data/raw/KDDTest+.txt",
    "data/raw/UNSW_NB15_training-set.csv",
    "data/raw/UNSW_NB15_testing-set.csv",
    "data/raw/CICIoT2023_train.csv",
    "data/raw/CICIoT2023_test.csv",
]

# The campaign families, in execution order, each with the command that
# produces it and the paper artifact it backs.
PLANNED = [
    {"phase": "A", "name": "main_comparison", "make": "full-comparison",
     "paper": "Table V, Figures 2-4"},
    {"phase": "B", "name": "leakage_free", "make": "leakage-free",
     "paper": "Table VI"},
    {"phase": "C", "name": "ablation", "make": "ablation",
     "paper": "Table VII"},
    {"phase": "D", "name": "noniid_sweep", "make": "noniid-sweep",
     "paper": "Table VIII"},
    {"phase": "E", "name": "multi_attack_matrix", "make": "multi-attack",
     "paper": "Tables IX-X"},
    {"phase": "F", "name": "adaptive_attacks", "make": "multi-attack ACK1/ACK2",
     "paper": "Table XI"},
    {"phase": "G", "name": "extended_significance", "make": "extended-significance",
     "paper": "Section VIII-B"},
    {"phase": "H", "name": "hp_sweep_baseline", "make": "hp-sweep-baseline",
     "paper": "Supp. Table S1"},
    {"phase": "I", "name": "hp_sweep_tvflids", "make": "hp-sweep-tvflids",
     "paper": "Supp. Table S2"},
    {"phase": "J", "name": "bucketing_deepsight",
     "make": "full-comparison with bucketing deepsight",
     "paper": "Supp. Table S3"},
    {"phase": "K", "name": "ciciot2023", "make": "ciciot2023",
     "paper": "Supp. Table S4, Figure S1"},
    {"phase": "L", "name": "overhead", "make": "read from run summaries",
     "paper": "Table XII"},
    {"phase": "-", "name": "ratio_sweep", "make": "figures",
     "paper": "Figure 5"},
]

SEEDS = [42, 123, 456, 789, 1337]
EXTENDED_SEEDS = [42, 123, 456, 789, 1337, 2024, 31415, 8080, 555, 999]


def _now() -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            .isoformat().replace("+00:00", "Z"))


def _file_record(path: str) -> dict:
    exists = os.path.exists(path)
    return {
        "path": path,
        "exists": exists,
        "size_bytes": os.path.getsize(path) if exists else None,
        "sha256": file_sha256(path) if exists else None,
    }


def cmd_init(out: str) -> int:
    configs = []
    for p in CONFIG_FILES:
        rec = _file_record(p)
        rec["content"] = (open(p, encoding="utf-8").read()
                          if os.path.exists(p) else None)
        configs.append(rec)

    manifest = {
        "campaign": "TV-FLIDS full experimental campaign",
        "manifest_version": 1,
        "started_utc": _now(),
        "finished_utc": None,
        "git": git_state(),
        "environment": {
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": package_versions(),
        },
        "hardware": hardware(),
        "configs": configs,
        "datasets": [_file_record(p) for p in DATASET_FILES],
        "seeds": SEEDS,
        "extended_seeds": EXTENDED_SEEDS,
        "planned_experiments": PLANNED,
        "runs": [],
        "artifacts": [],
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print("[manifest] initialised " + out)
    print("  commit  : " + str(manifest["git"]["git_describe"]))
    print("  dirty   : " + str(manifest["git"]["git_dirty"]) +
          " (" + str(len(manifest["git"]["git_dirty_paths"])) + " paths)")
    print("  python  : " + manifest["environment"]["python_version"])
    present = sum(1 for d in manifest["datasets"] if d["exists"])
    print("  datasets: " + str(present) + "/" + str(len(manifest["datasets"])) +
          " present")
    return 0


def _collect_runs() -> list:
    runs = []
    pattern = os.path.join("results", "logs", "**", "experiment_log.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        norm = path.replace(os.sep, "/")
        if "_QUARANTINED_MOCK" in norm:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            runs.append({"log": norm, "error": str(exc)})
            continue
        cfg = blob.get("config", {}) or {}
        summary = blob.get("summary", {}) or {}
        runs.append({
            "log": norm,
            "sha256": file_sha256(path),
            "experiment_name": blob.get("experiment_name"),
            "strategy": cfg.get("strategy"),
            "attack": cfg.get("attack"),
            "dataset": cfg.get("dataset"),
            "protocol": cfg.get("protocol"),
            "partition_type": cfg.get("partition_type"),
            "alpha": cfg.get("alpha"),
            "seed": cfg.get("seed"),
            "num_rounds_configured": cfg.get("num_rounds"),
            "rounds_logged": len(blob.get("rounds", []) or []),
            "config_hash": cfg.get("_config_hash"),
            "final_accuracy": summary.get("final_accuracy"),
            "elapsed_seconds": blob.get("elapsed_seconds"),
            "provenance": blob.get("provenance"),
        })
    return runs


def _collect_artifacts() -> list:
    out = []
    patterns = [
        os.path.join("results", "tables", "*.json"),
        os.path.join("results", "tables", "*.tex"),
        os.path.join("results", "figures", "*.pdf"),
        os.path.join("Paper", "figures", "fig_*.tex"),
    ]
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            out.append(_file_record(path.replace(os.sep, "/")))
    return out


def cmd_finalize(out: str) -> int:
    if not os.path.exists(out):
        print("[manifest] " + out + " does not exist - run init first.")
        return 1
    with open(out, encoding="utf-8") as fh:
        manifest = json.load(fh)

    manifest["finished_utc"] = _now()
    manifest["git_at_finish"] = git_state()
    manifest["runs"] = _collect_runs()
    manifest["artifacts"] = _collect_artifacts()

    no_prov = [r["log"] for r in manifest["runs"] if not r.get("provenance")]
    manifest["runs_without_provenance"] = no_prov

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print("[manifest] finalised " + out)
    print("  runs recorded      : " + str(len(manifest["runs"])))
    print("  artifacts recorded : " + str(len(manifest["artifacts"])))
    print("  runs w/o provenance: " + str(len(no_prov)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="TV-FLIDS campaign manifest")
    ap.add_argument("mode", choices=["init", "finalize"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    return cmd_init(args.out) if args.mode == "init" else cmd_finalize(args.out)


if __name__ == "__main__":
    sys.exit(main())
