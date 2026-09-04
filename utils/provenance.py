"""
utils/provenance.py — genuine run provenance for every result-producing run.

Answers, for any artifact on disk: *exactly which code, configuration,
environment and machine produced this?* Every field is read from the live
process or from git at call time; nothing here is hardcoded, and there is no
fallback that invents a plausible-looking value. When a fact cannot be
determined (git missing, dirty tree, package absent) the field says so
explicitly rather than guessing.

Used by:
  * utils/logger.py::ExperimentLogger.save  — stamps every experiment_log.json
  * scripts/campaign_manifest.py            — the campaign-level manifest
"""

from __future__ import annotations

import datetime
import hashlib
import os
import platform
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Packages whose versions define the reproducible surface (paper Section VI-D).
_TRACKED = [
    "torch", "flwr", "ray", "numpy", "scipy", "pandas",
    "sklearn", "imblearn", "hdbscan", "matplotlib", "statsmodels",
]


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_state() -> Dict[str, Any]:
    """Commit, branch and working-tree cleanliness.

    A dirty tree is recorded as such — a result produced from uncommitted
    source is still reproducible only if the diff is known, so the list of
    modified paths is kept alongside the commit.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    dirty_paths: List[str] = []
    if status:
        dirty_paths = [ln[3:] for ln in status.splitlines() if ln.strip()]
    return {
        "git_commit": commit or "unknown (git unavailable)",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_describe": _git("describe", "--always", "--dirty") or "unknown",
        "git_dirty": bool(dirty_paths),
        "git_dirty_paths": dirty_paths,
    }


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in _TRACKED:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not installed"
    return versions


def hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count_logical": os.cpu_count(),
    }
    # /proc is the authoritative source on Linux/WSL2; absent elsewhere.
    try:
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    info["mem_total_kb"] = int(line.split()[1])
                    break
    except OSError:
        pass
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        info["torch_num_threads"] = torch.get_num_threads()
    except Exception:
        info["cuda_available"] = None
    return info


def file_sha256(path: str) -> Optional[str]:
    """SHA-256 of a file, or None if it does not exist / cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def run_provenance(script: Optional[str] = None,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The provenance block stamped onto a single result artifact."""
    prov: Dict[str, Any] = {
        "script": script or (sys.argv[0] if sys.argv else "unknown"),
        "argv": list(sys.argv),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
                                  .isoformat().replace("+00:00", "Z"),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": package_versions(),
        "hardware": hardware(),
        # Simulation-parallelism knob: changes wall-clock, and (because each
        # Ray worker owns its own RNG stream) the client->worker assignment.
        # Recorded so a timing or a trajectory can be traced to it.
        "sim_client_cpus": os.getenv("TVFLIDS_SIM_CLIENT_CPUS", "12"),
        "sim_client_gpus": os.getenv("TVFLIDS_SIM_CLIENT_GPUS", "0.0"),
    }
    prov.update(git_state())
    if extra:
        prov.update(extra)
    return prov
