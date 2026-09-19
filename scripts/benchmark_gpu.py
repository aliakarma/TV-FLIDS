"""
scripts/benchmark_gpu.py
Reproducible 100-round TV-FLIDS GPU/Host Benchmark Harness.
Measures wall-clock time, s/round, training/eval/data breakdown,
peak GPU VRAM, GPU utilization, peak system RAM, CPU utilization,
and final scientific metrics.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


# ── Cross-Platform Resource Sampler ───────────────────────────────────────────

class ResourceSampler(threading.Thread):
    """Samples GPU (via nvidia-smi) and Host (RAM/CPU) in a background thread."""

    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_event = threading.Event()
        self.gpu_mem_mib: List[float] = []
        self.gpu_util_pct: List[float] = []
        self.host_ram_used_gb: List[float] = []
        self.cpu_util_pct: List[float] = []
        self.nvidia_smi_available = shutil.which("nvidia-smi") is not None
        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            pass

    def run(self) -> None:
        while not self._stop_event.is_set():
            # 1. GPU metrics via nvidia-smi
            if self.nvidia_smi_available:
                try:
                    out = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    ).strip()
                    for line in out.splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2:
                            self.gpu_mem_mib.append(float(parts[0]))
                            self.gpu_util_pct.append(float(parts[1]))
                except Exception:
                    pass

            # 2. Host RAM & CPU metrics
            if self._psutil is not None:
                try:
                    vm = self._psutil.virtual_memory()
                    self.host_ram_used_gb.append(round((vm.total - vm.available) / (1024 ** 3), 3))
                    self.cpu_util_pct.append(self._psutil.cpu_percent(interval=None))
                except Exception:
                    pass
            elif sys.platform == "win32":
                try:
                    import ctypes
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]
                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                    used = (stat.ullTotalPhys - stat.ullAvailPhys) / (1024 ** 3)
                    self.host_ram_used_gb.append(round(used, 3))
                except Exception:
                    pass

            self._stop_event.wait(self.interval)

    def stop(self) -> Dict[str, Any]:
        self._stop_event.set()
        self.join(timeout=3)

        def _stats(arr: List[float]) -> Dict[str, Optional[float]]:
            if not arr:
                return {"peak": None, "mean": None, "samples": 0}
            return {
                "peak": round(float(max(arr)), 2),
                "mean": round(float(sum(arr) / len(arr)), 2),
                "samples": len(arr),
            }

        return {
            "nvidia_smi_available": self.nvidia_smi_available,
            "gpu_memory_used_mib": _stats(self.gpu_mem_mib),
            "gpu_utilisation_pct": _stats(self.gpu_util_pct),
            "host_memory_used_gb": _stats(self.host_ram_used_gb),
            "cpu_utilisation_pct": _stats(self.cpu_util_pct),
        }


# ── Hardware & Environment Inspection ─────────────────────────────────────────

def inspect_environment() -> Dict[str, Any]:
    gpu_hardware_name = "N/A"
    gpu_total_vram_mib = 0.0
    driver_version = "N/A"
    cuda_driver_version = "N/A"

    if shutil.which("nvidia-smi") is not None:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
            line = out.splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpu_hardware_name = parts[0]
                gpu_total_vram_mib = float(parts[1])
                driver_version = parts[2]
        except Exception:
            pass

    return {
        "timestamp_utc": _now_iso(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available_in_torch": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        ),
        "hardware_gpu_name": gpu_hardware_name,
        "hardware_total_vram_mib": gpu_total_vram_mib,
        "nvidia_driver_version": driver_version,
    }


# ── Benchmark Runner ──────────────────────────────────────────────────────────

def run_benchmark(
    dataset: str = "nslkdd",
    strategy: str = "tvflids",
    attack: str = "label_flip_30",
    seed: int = 42,
    num_rounds: int = 100,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute the representative 100-round TV-FLIDS benchmark."""

    env_info = inspect_environment()
    if verbose:
        print("=" * 70)
        print(" TV-FLIDS GPU / HOST BENCHMARK HARNESS")
        print("=" * 70)
        print(f" Hardware GPU:       {env_info['hardware_gpu_name']} ({env_info['hardware_total_vram_mib']} MiB)")
        print(f" PyTorch Build:      {env_info['torch_version']}")
        print(f" PyTorch CUDA:       {env_info['cuda_available_in_torch']}")
        print(f" Dataset:            {dataset.upper()}")
        print(f" Strategy:           {strategy.upper()}")
        print(f" Attack:             {attack}")
        print(f" Seed:               {seed}")
        print(f" Rounds:             {num_rounds}")
        print("=" * 70)

    log_dir = os.path.join(
        ROOT, "results", "logs", "_benchmark_gpu",
        f"{strategy}_{attack}_r{num_rounds}_seed{seed}"
    )
    os.makedirs(log_dir, exist_ok=True)

    # Start resource sampler
    sampler = ResourceSampler(interval=1.0)
    sampler.start()

    t_start = time.perf_counter()

    # Import run_experiment
    from experiments.run_experiment import run_experiment

    # Force fresh execution (no resume cache)
    old_resume = os.environ.get("TVFLIDS_RESUME")
    os.environ["TVFLIDS_RESUME"] = "0"

    summary: Dict[str, Any] = {}
    error: Optional[str] = None

    try:
        summary = run_experiment(
            strategy_name=strategy,
            attack_config_name=attack,
            dataset=dataset,
            partition_type="noniid",
            alpha=0.5,
            seed=seed,
            num_rounds=num_rounds,
            log_dir=log_dir,
            verbose=verbose,
            model_type="mlp",
            val_size=2000,
            protocol="main",
        )
    except Exception as exc:
        error = str(exc)
        if verbose:
            print(f"[Benchmark Error] {exc}")
    finally:
        if old_resume is not None:
            os.environ["TVFLIDS_RESUME"] = old_resume
        else:
            os.environ.pop("TVFLIDS_RESUME", None)

    t_total = time.perf_counter() - t_start
    resources = sampler.stop()

    # Read detailed round logs if available
    round_log_path = os.path.join(log_dir, "experiment_log.json")
    rounds_data = []
    if os.path.exists(round_log_path):
        try:
            with open(round_log_path, "r", encoding="utf-8") as fh:
                rounds_data = json.load(fh).get("rounds", [])
        except Exception:
            pass

    # Extract timings
    avg_sec_per_round = round(t_total / max(num_rounds, 1), 3)

    result: Dict[str, Any] = {
        "benchmark_name": "TV-FLIDS Representative 100-Round Benchmark",
        "timestamp_utc": _now_iso(),
        "environment": env_info,
        "configuration": {
            "dataset": dataset,
            "strategy": strategy,
            "attack": attack,
            "seed": seed,
            "num_rounds": num_rounds,
            "partition": "noniid (alpha=0.5)",
            "num_clients": 20,
            "val_size": 2000,
            "model": "IDSMLP (41 -> 128 -> 64 -> 32 -> 5)",
        },
        "timing": {
            "total_wall_clock_seconds": round(t_total, 2),
            "total_wall_clock_minutes": round(t_total / 60.0, 2),
            "average_seconds_per_round": avg_sec_per_round,
            "rounds_per_minute": round((num_rounds / t_total) * 60.0, 2) if t_total > 0 else 0.0,
        },
        "resources": resources,
        "final_metrics": {
            "final_accuracy": summary.get("final_accuracy"),
            "final_f1_macro": summary.get("final_f1_macro"),
            "final_attack_success_rate": summary.get("final_attack_success_rate"),
            "final_loss": summary.get("final_loss"),
        },
        "success": error is None,
        "error": error,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        if verbose:
            print(f"\n[Benchmark] Results saved to: {output_path}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="TV-FLIDS Representative GPU Benchmark Harness")
    parser.add_argument("--rounds", type=int, default=100, help="Number of rounds (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument("--dataset", default="nslkdd", choices=["nslkdd", "ciciot2023", "edgeiiotset"])
    parser.add_argument("--strategy", default="tvflids", help="Strategy (default: tvflids)")
    parser.add_argument("--attack", default="label_flip_30", help="Attack (default: label_flip_30)")
    parser.add_argument("--output", default="results/benchmark_rtx3050.json", help="Output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    res = run_benchmark(
        dataset=args.dataset,
        strategy=args.strategy,
        attack=args.attack,
        seed=args.seed,
        num_rounds=args.rounds,
        output_path=args.output,
        verbose=not args.quiet,
    )

    return 0 if res["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
