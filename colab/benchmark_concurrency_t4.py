"""
colab/benchmark_concurrency_t4.py
Controlled Concurrency Benchmark for NVIDIA T4 (1, 2, and 4 workers).
Evaluates representative 100-round TV-FLIDS workload (or configurable rounds)
across concurrency levels to determine safe throughput, GPU/host saturation,
worker interference, and numerical determinism.

Usage:
    python colab/benchmark_concurrency_t4.py --rounds 100 --modes 1 2 4
    python colab/benchmark_concurrency_t4.py --rounds 5 --modes 1 2 4 --json bench_t4_concurrency.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
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


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


class ConcurrencyTelemetrySampler(threading.Thread):
    """Monitors GPU memory, GPU utilization, host RAM, and CPU load during concurrency tests."""

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
                        self.host_mem_used_gb.append(round((total_kb - avail_kb) / (1024 ** 2), 3))
                except Exception:
                    pass

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


def build_jobs(concurrency: int, rounds: int, root_dir: str) -> List[Dict[str, Any]]:
    # Fixed canonical seeds for benchmark
    base_seeds = [42, 123, 456, 789]
    seeds = base_seeds[:concurrency]

    jobs = []
    for i, seed in enumerate(seeds):
        log_dir = os.path.join(root_dir, f"c{concurrency}_w{i}_seed{seed}")
        cmd = [
            sys.executable, "experiments/run_experiment.py",
            "--strategy", "tvflids",
            "--attack", "label_flip_30",
            "--dataset", "nslkdd",
            "--seed", str(seed),
            "--rounds", str(rounds),
            "--log_dir", log_dir,
            "--quiet",
        ]
        jobs.append({
            "worker_idx": i,
            "seed": seed,
            "log_dir": log_dir,
            "cmd": cmd,
        })
    return jobs


def run_concurrency_mode(
    concurrency: int,
    rounds: int,
    base_root: str,
    client_gpus: Optional[str] = "0.05",
) -> Dict[str, Any]:
    mode_dir = os.path.join(base_root, f"mode_{concurrency}w")
    os.makedirs(mode_dir, exist_ok=True)
    jobs = build_jobs(concurrency, rounds, mode_dir)

    for j in jobs:
        shutil.rmtree(j["log_dir"], ignore_errors=True)
        os.makedirs(j["log_dir"], exist_ok=True)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["TVFLIDS_RESUME"] = "0"
    env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    if client_gpus is not None:
        env["TVFLIDS_SIM_CLIENT_GPUS"] = str(client_gpus)

    print(f"\n[Mode {concurrency} Workers] Launching {concurrency} jobs ({rounds} rounds each)...")
    sampler = ConcurrencyTelemetrySampler(interval=1.0)
    sampler.start()

    procs = []
    t0 = time.perf_counter()
    for j in jobs:
        lp = os.path.join(mode_dir, f"worker_{j['worker_idx']}_seed{j['seed']}.log")
        fh = open(lp, "w", encoding="utf-8", errors="replace")
        fh.write(f"# Worker {j['worker_idx']} command: {' '.join(j['cmd'])}\n\n")
        fh.flush()
        p = subprocess.Popen(j["cmd"], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
        procs.append((j, p, fh, lp))

    job_results = []
    for j, p, fh, lp in procs:
        rc = p.wait()
        fh.close()

        # Check completion
        metrics_file = os.path.join(j["log_dir"], "experiment_log.json")
        ok = (rc == 0 and os.path.exists(metrics_file))
        summary = {}
        if ok:
            try:
                with open(metrics_file, "r", encoding="utf-8") as mfh:
                    summary = json.load(mfh).get("summary", {})
            except Exception:
                ok = False

        job_results.append({
            "worker_idx": j["worker_idx"],
            "seed": j["seed"],
            "returncode": rc,
            "success": ok,
            "final_accuracy": summary.get("final_accuracy"),
            "final_f1_macro": summary.get("final_f1_macro"),
            "final_asr": summary.get("final_attack_success_rate"),
            "log": lp,
        })

    wall_sec = time.perf_counter() - t0
    telemetry = sampler.stop()

    n_ok = sum(1 for r in job_results if r["success"])
    n_fail = len(job_results) - n_ok
    runs_per_hour = round(n_ok / (wall_sec / 3600.0), 2) if wall_sec > 0 else 0.0

    print(f"  Wall Time:      {wall_sec:.1f}s ({wall_sec/60:.2f} min)")
    print(f"  Throughput:     {runs_per_hour} runs/hour ({n_ok}/{len(jobs)} passed, {n_fail} failed)")
    print(f"  GPU Peak VRAM:  {telemetry['gpu_memory_used_mib']['peak']} MiB | Util: {telemetry['gpu_utilisation_pct']['mean']}%")
    print(f"  Host Peak RAM:  {telemetry['host_memory_used_gb']['peak']} GB")

    return {
        "concurrency": concurrency,
        "rounds_per_job": rounds,
        "total_jobs": len(jobs),
        "jobs_passed": n_ok,
        "jobs_failed": n_fail,
        "stable": (n_fail == 0),
        "wall_clock_seconds": round(wall_sec, 2),
        "wall_clock_minutes": round(wall_sec / 60.0, 2),
        "runs_per_hour": runs_per_hour,
        "telemetry": telemetry,
        "jobs": job_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TV-FLIDS Controlled T4 Concurrency Benchmark (1, 2, 4 workers)")
    parser.add_argument("--rounds", type=int, default=100, help="Rounds per job (default: 100)")
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 2, 4], help="Concurrency levels (default: 1 2 4)")
    parser.add_argument("--root", default="results/_benchmark_t4_concurrency", help="Throwaway results directory")
    parser.add_argument("--client-gpus", default="0.05", help="GPU fraction per client actor")
    parser.add_argument("--json", default="results/benchmark_t4_concurrency.json", help="Summary JSON output path")

    args = parser.parse_args()

    print("=" * 75)
    print(" TV-FLIDS CONTROLLED T4 CONCURRENCY BENCHMARK (1, 2, 4 Workers)")
    print("=" * 75)
    print(f" Modes:         {args.modes}")
    print(f" Rounds / Job:  {args.rounds}")
    print(f" Root Dir:      {args.root}")
    print("=" * 75)

    mode_results = []
    for c in args.modes:
        mode_results.append(run_concurrency_mode(c, args.rounds, args.root, args.client_gpus))

    # Evaluate determinism: for seed 42 across all modes
    seed_42_accs = [
        next((j["final_accuracy"] for j in m["jobs"] if j["seed"] == 42 and j["success"]), None)
        for m in mode_results
    ]
    deterministic = (len(set(filter(None, seed_42_accs))) <= 1)

    summary_report = {
        "benchmark_name": "Controlled T4 Concurrency Benchmark",
        "timestamp_utc": _now_iso(),
        "rounds_per_job": args.rounds,
        "tested_modes": args.modes,
        "modes_summary": mode_results,
        "determinism_verified": deterministic,
        "recommendation": (
            max([m for m in mode_results if m["stable"]], key=lambda x: x["runs_per_hour"])["concurrency"]
            if any(m["stable"] for m in mode_results) else "None"
        ),
        "safety_guard": "8 or 16 workers must NOT be inferred safe without empirical measurement.",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(summary_report, fh, indent=2)

    print("\n" + "=" * 75)
    print(" CONCURRENCY BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'Workers':>8} {'Wall(min)':>10} {'Runs/Hour':>11} {'Peak VRAM(MiB)':>16} {'GPU Util%':>11} {'Peak RAM(GB)':>14} {'Stable':>8}")
    print("-" * 75)
    for m in mode_results:
        print(
            f"{m['concurrency']:>8} "
            f"{m['wall_clock_minutes']:>10.2f} "
            f"{m['runs_per_hour']:>11.2f} "
            f"{str(m['telemetry']['gpu_memory_used_mib']['peak']):>16} "
            f"{str(m['telemetry']['gpu_utilisation_pct']['mean']):>11} "
            f"{str(m['telemetry']['host_memory_used_gb']['peak']):>14} "
            f"{str(m['stable']):>8}"
        )
    print("=" * 75)
    print(f"Determinism Check (Seed 42): {'PASSED' if deterministic else 'DRIFT DETECTED'}")
    print(f"Safety Constraint: 8 or 16 workers must NOT be inferred safe without empirical measurement.")
    print(f"Report saved to: {args.json}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
