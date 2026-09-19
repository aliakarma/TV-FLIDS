"""
colab/benchmark_t4.py
Reproducible 100-Round TV-FLIDS Benchmark for NVIDIA T4 (Linux / Google Colab).
Executes the exact representative workload (NSL-KDD, TV-FLIDS, label_flip_30,
noniid alpha=0.5, 20 clients, seed 42, 100 rounds) and outputs a complete
telemetry and reproducibility report.
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
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


class LinuxGPUSampler(threading.Thread):
    """Polls nvidia-smi, /proc/meminfo, and /proc/loadavg while benchmark executes."""

    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_event = threading.Event()
        self.gpu_mem_mib: List[float] = []
        self.gpu_util_pct: List[float] = []
        self.host_mem_used_gb: List[float] = []
        self.cpu_load_1m: List[float] = []
        self.nvidia_smi_available = shutil.which("nvidia-smi") is not None

    def run(self) -> None:
        while not self._stop_event.is_set():
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

            # Read /proc/meminfo on Linux
            if os.path.exists("/proc/meminfo"):
                try:
                    total_kb = avail_kb = 0
                    with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                        for line in fh:
                            if line.startswith("MemTotal:"):
                                total_kb = int(line.split()[1])
                            elif line.startswith("MemAvailable:"):
                                avail_kb = int(line.split()[1])
                            if total_kb and avail_kb:
                                break
                    if total_kb and avail_kb:
                        used_gb = (total_kb - avail_kb) / (1024 ** 2)
                        self.host_mem_used_gb.append(round(used_gb, 3))
                except Exception:
                    pass

            # Read /proc/loadavg on Linux
            if os.path.exists("/proc/loadavg"):
                try:
                    with open("/proc/loadavg", "r", encoding="utf-8") as fh:
                        self.cpu_load_1m.append(float(fh.read().split()[0]))
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
            "host_memory_used_gb": _stats(self.host_mem_used_gb),
            "cpu_load_1m": _stats(self.cpu_load_1m),
        }


def inspect_t4_environment() -> Dict[str, Any]:
    gpu_name = "N/A"
    gpu_vram_mib = 0.0
    driver_version = "N/A"
    cuda_version = "N/A"

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
                gpu_name = parts[0]
                gpu_vram_mib = float(parts[1])
                driver_version = parts[2]
        except Exception:
            pass

    # Read CPU model from /proc/cpuinfo
    cpu_model = platform.processor()
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    # Read total RAM from /proc/meminfo
    total_ram_gb = 0.0
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        total_ram_gb = round(int(line.split()[1]) / (1024 ** 2), 2)
                        break
        except Exception:
            pass

    return {
        "timestamp_utc": _now_iso(),
        "platform": platform.platform(),
        "cpu_info": cpu_model,
        "system_ram_gb": total_ram_gb,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version_torch": torch.version.cuda if hasattr(torch.version, "cuda") else "N/A",
        "cuda_available_in_torch": torch.cuda.is_available(),
        "gpu_model": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else gpu_name),
        "gpu_vram_mib": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1)
            if torch.cuda.is_available() else gpu_vram_mib
        ),
        "driver_version": driver_version,
    }


def run_t4_benchmark(
    dataset: str = "nslkdd",
    strategy: str = "tvflids",
    attack: str = "label_flip_30",
    seed: int = 42,
    num_rounds: int = 100,
    output_path: str = "results/benchmark_t4.json",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute the exact 100-round representative TV-FLIDS benchmark on T4."""

    env_info = inspect_t4_environment()

    if verbose:
        print("=" * 72)
        print(" TV-FLIDS NVIDIA T4 BENCHMARK HARNESS")
        print("=" * 72)
        print(f" GPU Model:          {env_info['gpu_model']} ({env_info['gpu_vram_mib']} MiB)")
        print(f" CUDA Version:       {env_info['cuda_version_torch']}")
        print(f" PyTorch Version:    {env_info['torch_version']}")
        print(f" CPU Info:           {env_info['cpu_info']}")
        print(f" System RAM:         {env_info['system_ram_gb']} GB")
        print(f" Dataset:            {dataset.upper()}")
        print(f" Strategy:           {strategy.upper()}")
        print(f" Attack:             {attack}")
        print(f" Seed:               {seed}")
        print(f" Rounds:             {num_rounds}")
        print("=" * 72)

    log_dir = os.path.join(
        ROOT, "results", "logs", "_benchmark_t4",
        f"{strategy}_{attack}_r{num_rounds}_seed{seed}"
    )
    os.makedirs(log_dir, exist_ok=True)

    sampler = LinuxGPUSampler(interval=1.0)
    sampler.start()

    t_start = time.perf_counter()

    from experiments.run_experiment import run_experiment

    # Set Colab / GPU simulation environment variables
    os.environ["TVFLIDS_RESUME"] = "0"
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    os.environ.setdefault("TVFLIDS_SIM_CLIENT_CPUS", "1")
    os.environ.setdefault("TVFLIDS_SIM_CLIENT_GPUS", "0.05")

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
            print(f"[T4 Benchmark Error] {exc}")

    t_total = time.perf_counter() - t_start
    resources = sampler.stop()

    avg_sec_per_round = round(t_total / max(num_rounds, 1), 3)

    report: Dict[str, Any] = {
        "benchmark_name": "NVIDIA T4 100-Round TV-FLIDS Benchmark",
        "status": "MEASURED" if error is None and env_info["cuda_available_in_torch"] else "PENDING_EXTERNAL_EXECUTION",
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

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    if verbose:
        print(f"\n[T4 Benchmark] Report written to: {output_path}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="NVIDIA T4 TV-FLIDS Benchmark Harness")
    parser.add_argument("--rounds", type=int, default=100, help="Number of rounds (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument("--dataset", default="nslkdd", choices=["nslkdd", "ciciot2023", "edgeiiotset"])
    parser.add_argument("--strategy", default="tvflids", help="Strategy (default: tvflids)")
    parser.add_argument("--attack", default="label_flip_30", help="Attack (default: label_flip_30)")
    parser.add_argument("--output", default="results/benchmark_t4.json", help="Output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    res = run_t4_benchmark(
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
