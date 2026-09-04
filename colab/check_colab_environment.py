"""
colab/check_colab_environment.py — report the Colab runtime, and fail early.

    python colab/check_colab_environment.py
    python colab/check_colab_environment.py --json env.json
    python colab/check_colab_environment.py --require-gpu     # exit 1 without CUDA

This reports facts and applies only hard compatibility gates. It never claims
the environment is equivalent to the paper's stated stack — that comparison is
`scripts/verify_environment.py`'s job, and it reports rather than asserts.

Hard gates (exit 1):
  * Python major.minor is not 3.10  ->  the pinned wheels target 3.10
  * a required package is missing entirely
  * `--require-gpu` and torch reports no CUDA device
  * the NSL-KDD dataset files are absent
  * the campaign inventory disagrees with the repository's runner constants
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The reproducible surface, as pinned in requirements.txt.
PINNED = [
    ("torch", "torch", "2.1.0"),
    ("flwr", "flwr", "1.6.0"),
    ("ray", "ray", "2.6.3"),
    ("numpy", "numpy", "1.26.2"),
    ("scipy", "scipy", "1.11.4"),
    ("pandas", "pandas", "2.1.3"),
    ("sklearn", "scikit-learn", "1.3.2"),
    ("imblearn", "imbalanced-learn", "0.11.0"),
    ("hdbscan", "hdbscan", "0.8.33"),
    ("matplotlib", "matplotlib", "3.8.2"),
    ("statsmodels", "statsmodels", "0.14.1"),
    ("yaml", "pyyaml", "6.0.1"),
]
REQUIRED_PYTHON = (3, 10)
DATASET_FILES = ["data/raw/KDDTrain+.txt", "data/raw/KDDTest+.txt"]
OPTIONAL_DATASET_FILES = [
    "data/raw/UNSW_NB15_training-set.csv",
    "data/raw/UNSW_NB15_testing-set.csv",
    "data/raw/CICIoT2023_train.csv",
    "data/raw/CICIoT2023_test.csv",
]


def _version(import_name: str, dist_name: str) -> Optional[str]:
    """Installed version, or None if the package is not importable at all."""
    try:
        importlib.import_module(import_name)
    except Exception:                                        # noqa: BLE001
        return None
    # Distribution metadata is authoritative and works for packages that
    # expose no __version__ attribute (hdbscan, for one).
    try:
        from importlib.metadata import version as _dist_version
        return _dist_version(dist_name)
    except Exception:                                        # noqa: BLE001
        pass
    mod = sys.modules.get(import_name)
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "installed (version unknown)"


def _matches_pin(installed: Optional[str], pinned: str) -> bool:
    """True when the installed build IS the pinned release.

    A PEP 440 local-version suffix identifies the build variant, not a
    different release: torch 2.1.0+cu121 and 2.1.0+cpu are both torch 2.1.0.
    Which variant is installed matters enormously for speed, so it is still
    reported verbatim — it is simply not a version mismatch.
    """
    if installed is None:
        return False
    return installed.split("+", 1)[0] == pinned


def collect() -> Dict:
    rep: Dict = {"errors": [], "warnings": []}

    rep["python"] = {
        "version": platform.python_version(),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "required_major_minor": f"{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}",
    }
    if sys.version_info[:2] != REQUIRED_PYTHON:
        rep["errors"].append(
            f"Python {platform.python_version()} but the pinned wheels target "
            f"{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}.x "
            f"(requirements.txt / .python-version)")

    rep["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "node": platform.node(),
    }

    # ── packages ──
    pkgs = {}
    for import_name, dist, pinned in PINNED:
        got = _version(import_name, dist)
        match = _matches_pin(got, pinned)
        variant = got.split("+", 1)[1] if got and "+" in got else None
        pkgs[dist] = {"import_name": import_name, "pinned": pinned,
                       "installed": got, "matches_pin": match,
                       "build_variant": variant}
        if got is None:
            rep["errors"].append(f"{dist} is not importable (pinned {pinned})")
        elif not match:
            rep["warnings"].append(
                f"{dist} {got} installed, {pinned} pinned - a genuine "
                f"difference from the pinned stack, recorded not corrected")
    rep["packages"] = pkgs

    # The torch build variant decides whether the GPU is used at all.
    tv = pkgs.get("torch", {}).get("installed") or ""
    if tv.endswith("+cpu"):
        rep["warnings"].append(
            "torch is the +cpu build: it cannot use the T4 no matter what "
            "nvidia-smi reports. Install the CUDA build "
            "(see colab/setup_colab.sh) before starting the campaign.")

    # ── CUDA / GPU ──
    gpu: Dict = {"cuda_available": False}
    try:
        import torch
        gpu["torch_version"] = torch.__version__
        gpu["torch_cuda_build"] = torch.version.cuda
        gpu["cuda_available"] = bool(torch.cuda.is_available())
        gpu["device_count"] = torch.cuda.device_count() if gpu["cuda_available"] else 0
        if gpu["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            gpu["device_name"] = props.name
            gpu["total_memory_gb"] = round(props.total_memory / 1024 ** 3, 2)
            gpu["capability"] = f"{props.major}.{props.minor}"
            gpu["cudnn"] = torch.backends.cudnn.version()
        else:
            rep["warnings"].append(
                "torch reports no CUDA device: the campaign will run on CPU")
    except Exception as exc:                                  # noqa: BLE001
        gpu["error"] = f"{type(exc).__name__}: {exc}"
        rep["errors"].append(f"could not query torch/CUDA: {exc}")
    rep["gpu"] = gpu

    if shutil.which("nvidia-smi"):
        try:
            gpu["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,"
                               "utilization.gpu,driver_version",
                 "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL).strip()
        except Exception as exc:                              # noqa: BLE001
            gpu["nvidia_smi"] = f"query failed: {exc}"

    # ── CPU / RAM / disk ──
    host: Dict = {"cpu_count": os.cpu_count()}
    try:
        host["cpu_count_affinity"] = len(os.sched_getaffinity(0))   # type: ignore[attr-defined]
    except (AttributeError, OSError):
        host["cpu_count_affinity"] = os.cpu_count()
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    host["ram_total_gb"] = round(
                        int(line.split()[1]) / 1024 ** 2, 2)
                elif line.startswith("MemAvailable:"):
                    host["ram_available_gb"] = round(
                        int(line.split()[1]) / 1024 ** 2, 2)
    except OSError:
        host["ram_total_gb"] = None
    for label, path in (("repo", ROOT), ("content", "/content"),
                        ("drive", "/content/drive/MyDrive")):
        if os.path.isdir(path):
            try:
                du = shutil.disk_usage(path)
                host[f"disk_{label}_free_gb"] = round(du.free / 1024 ** 3, 2)
                host[f"disk_{label}_total_gb"] = round(du.total / 1024 ** 3, 2)
            except OSError:
                pass
    rep["host"] = host
    if (host.get("ram_total_gb") or 0) and host["ram_total_gb"] < 10:
        rep["warnings"].append(
            f"only {host['ram_total_gb']} GB RAM: a 20-client Ray simulation "
            f"was memory-bound at ~9 GB on the previous host")

    # ── git ──
    def _git(*a) -> Optional[str]:
        try:
            return subprocess.check_output(["git", *a], cwd=ROOT, text=True,
                                            stderr=subprocess.DEVNULL).strip()
        except Exception:                                     # noqa: BLE001
            return None
    status = _git("status", "--porcelain")
    rep["git"] = {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--always", "--dirty"),
        "dirty": bool(status),
        "dirty_paths": [l[3:] for l in status.splitlines()] if status else [],
        "remote": _git("config", "--get", "remote.origin.url"),
    }
    if rep["git"]["dirty"]:
        rep["warnings"].append(
            f"working tree is dirty ({len(rep['git']['dirty_paths'])} path(s)): "
            f"every result will record a dirty provenance")

    # ── datasets ──
    ds = {}
    for rel in DATASET_FILES + OPTIONAL_DATASET_FILES:
        p = os.path.join(ROOT, rel)
        ds[rel] = {"exists": os.path.exists(p),
                    "size_bytes": os.path.getsize(p) if os.path.exists(p) else None}
    rep["datasets"] = ds
    for rel in DATASET_FILES:
        if not ds[rel]["exists"]:
            rep["errors"].append(f"required dataset {rel} is absent "
                                  f"(run: bash scripts/download_nslkdd.sh)")
    for rel in OPTIONAL_DATASET_FILES:
        if not ds[rel]["exists"]:
            rep["warnings"].append(
                f"optional dataset {rel} absent: the phases that need it "
                f"cannot run and will be reported as blocked, never faked")

    # ── inventory drift ──
    try:
        import campaign_inventory as inv
        problems = inv.verify_against_repo()
        rep["inventory"] = {
            "cells": len(inv.all_cells()),
            "phases": len(inv.PHASES),
            "drift": problems,
        }
        for p in problems:
            rep["errors"].append(f"campaign inventory drift: {p}")
    except Exception as exc:                                  # noqa: BLE001
        rep["inventory"] = {"error": f"{type(exc).__name__}: {exc}"}
        rep["errors"].append(f"could not verify the campaign inventory: {exc}")

    return rep


def render(rep: Dict) -> None:
    print("=" * 74)
    print("TV-FLIDS COLAB ENVIRONMENT")
    print("=" * 74)
    p = rep["python"]
    print(f"\nPython        {p['version']}  ({p['implementation']})  "
          f"required {p['required_major_minor']}.x")
    print(f"              {p['executable']}")
    pl = rep["platform"]
    print(f"Platform      {pl['system']} {pl['release']} {pl['machine']}")

    print("\nPackages")
    print(f"  {'package':<22} {'installed':<20} {'pinned':<10} "
          f"{'release':<8} build")
    for dist, info in rep["packages"].items():
        mark = "yes" if info["matches_pin"] else "NO"
        print(f"  {dist:<22} {str(info['installed']):<20} "
              f"{info['pinned']:<10} {mark:<8} "
              f"{info.get('build_variant') or '-'}")

    g = rep["gpu"]
    print("\nGPU / CUDA")
    print(f"  cuda_available        {g.get('cuda_available')}")
    print(f"  torch cuda build      {g.get('torch_cuda_build')}")
    if g.get("cuda_available"):
        print(f"  device                {g.get('device_name')}")
        print(f"  memory                {g.get('total_memory_gb')} GB")
        print(f"  capability            sm_{str(g.get('capability','')).replace('.','')}")
    if g.get("nvidia_smi"):
        print(f"  nvidia-smi            {g['nvidia_smi']}")

    h = rep["host"]
    print("\nHost")
    for k in sorted(h):
        print(f"  {k:<22} {h[k]}")

    gi = rep["git"]
    print("\nRepository")
    print(f"  commit                {gi['commit']}")
    print(f"  branch                {gi['branch']}")
    print(f"  describe              {gi['describe']}")
    print(f"  dirty                 {gi['dirty']} "
          f"({len(gi['dirty_paths'])} path(s))")
    print(f"  remote                {gi['remote']}")

    print("\nDatasets")
    for rel, info in rep["datasets"].items():
        size = "-" if info["size_bytes"] is None else f"{info['size_bytes']:,}"
        print(f"  {'OK ' if info['exists'] else 'ABS'} {rel:<44} {size:>14}")

    iv = rep.get("inventory", {})
    print("\nCampaign inventory")
    if "error" in iv:
        print(f"  ERROR {iv['error']}")
    else:
        print(f"  cells                 {iv['cells']}")
        print(f"  phases                {iv['phases']}")
        print(f"  drift vs repo         "
              f"{'none' if not iv['drift'] else iv['drift']}")

    print("\n" + "=" * 74)
    for w in rep["warnings"]:
        print(f"  warn   {w}")
    for e in rep["errors"]:
        print(f"  ERROR  {e}")
    ok = not rep["errors"]
    print(f"\n{'ENVIRONMENT OK' if ok else 'ENVIRONMENT NOT COMPATIBLE'}"
          f"  ({len(rep['errors'])} error(s), {len(rep['warnings'])} warning(s))")
    print("=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None)
    ap.add_argument("--require-gpu", action="store_true")
    args = ap.parse_args()

    rep = collect()
    if args.require_gpu and not rep["gpu"].get("cuda_available"):
        rep["errors"].append("--require-gpu was given but no CUDA device is "
                              "available")
    render(rep)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, default=str)
        print(f"[env] report -> {args.json}")
    return 1 if rep["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
