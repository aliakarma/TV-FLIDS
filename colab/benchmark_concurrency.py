"""
colab/benchmark_concurrency.py — measure, don't assume, the right concurrency.

    python colab/benchmark_concurrency.py --rounds 5
    python colab/benchmark_concurrency.py --rounds 5 --modes 1 2 --client-gpus 0.5
    python colab/benchmark_concurrency.py --rounds 3 --json bench.json

One T4 is one GPU. Running two campaign workers on it can help (it hides the
Ray/Flower start-up and the per-round Python overhead behind each other's
compute) or hurt (they contend for the same SMs, and two 20-client simulations
can exhaust either GPU memory or host RAM). Which one happens is a property of
this machine and this model size, so it is measured here rather than guessed.

For each concurrency mode the benchmark runs the SAME short jobs and records
wall time, throughput, peak GPU memory, GPU utilisation, CPU load and failures.
It then recommends the mode with the highest *stable* throughput — a mode with
any failure is never recommended, however fast it looked.

The benchmark writes into a throwaway results root and uses a deliberately
small round count, so it can never be mistaken for, or overwrite, a campaign
cell. Its numbers are timings, not scientific results.
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
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import campaign_inventory as inv  # noqa: E402


def _now() -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            .isoformat().replace("+00:00", "Z"))


# ── GPU / CPU sampling ───────────────────────────────────────────────────────

class Sampler(threading.Thread):
    """Polls nvidia-smi and /proc/loadavg while a mode runs."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_evt = threading.Event()
        self.gpu_mem_mib: List[float] = []
        self.gpu_util_pct: List[float] = []
        self.loadavg: List[float] = []
        self.host_mem_used_gb: List[float] = []
        self.available = shutil.which("nvidia-smi") is not None

    def run(self) -> None:
        while not self._stop_evt.is_set():
            if self.available:
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi",
                         "--query-gpu=memory.used,utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
                    for line in out.splitlines():
                        mem, util = [x.strip() for x in line.split(",")[:2]]
                        self.gpu_mem_mib.append(float(mem))
                        self.gpu_util_pct.append(float(util))
                except Exception:                             # noqa: BLE001
                    pass
            try:
                with open("/proc/loadavg", encoding="utf-8") as fh:
                    self.loadavg.append(float(fh.read().split()[0]))
            except OSError:
                pass
            try:
                with open("/proc/meminfo", encoding="utf-8") as fh:
                    total = avail = None
                    for line in fh:
                        if line.startswith("MemTotal:"):
                            total = int(line.split()[1])
                        elif line.startswith("MemAvailable:"):
                            avail = int(line.split()[1])
                        if total and avail:
                            break
                if total and avail:
                    self.host_mem_used_gb.append(
                        round((total - avail) / 1024 ** 2, 2))
            except OSError:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self) -> Dict:
        self._stop_evt.set()
        self.join(timeout=5)

        def agg(vals: List[float]) -> Dict:
            if not vals:
                return {"peak": None, "mean": None, "samples": 0}
            return {"peak": max(vals),
                    "mean": round(sum(vals) / len(vals), 2),
                    "samples": len(vals)}
        return {
            "nvidia_smi_available": self.available,
            "gpu_memory_used_mib": agg(self.gpu_mem_mib),
            "gpu_utilisation_pct": agg(self.gpu_util_pct),
            "host_loadavg_1m": agg(self.loadavg),
            "host_memory_used_gb": agg(self.host_mem_used_gb),
        }


# ── Job definition ───────────────────────────────────────────────────────────

def bench_jobs(root: str, rounds: int, n: int) -> List[Dict]:
    """`n` short, independent jobs that exercise the real code path.

    TV-FLIDS under label-flip is chosen deliberately: it is the heaviest
    strategy (verification gate + trust EMA + meta-gradient), so it is the
    honest thing to time. Distinct seeds keep the jobs independent.
    """
    seeds = inv.SEEDS[:n] if n <= len(inv.SEEDS) else (
        inv.EXTENDED_SEEDS * 4)[:n]
    jobs = []
    for i, seed in enumerate(seeds):
        log_dir = os.path.join(
            root, "logs", "_bench", f"tvflids_label_flip_30_r{rounds}_seed{seed}_w{i}"
        ).replace(os.sep, "/")
        jobs.append({
            "label": f"seed{seed}",
            "log_dir": log_dir,
            "cmd": [sys.executable, "experiments/run_experiment.py",
                    "--strategy", "tvflids", "--attack", "label_flip_30",
                    "--seed", str(seed), "--rounds", str(rounds),
                    "--log_dir", log_dir, "--quiet"],
        })
    return jobs


def run_mode(concurrency: int, rounds: int, root: str, out_dir: str,
             client_cpus: Optional[str], client_gpus: Optional[str]) -> Dict:
    """Run `concurrency` jobs simultaneously and measure the whole batch."""
    jobs = bench_jobs(root, rounds, concurrency)
    for j in jobs:
        shutil.rmtree(j["log_dir"], ignore_errors=True)
        os.makedirs(j["log_dir"], exist_ok=True)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["TVFLIDS_RESUME"] = "0"          # always execute; never reuse
    env["TVFLIDS_TENSORBOARD"] = "0"
    env.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    if client_cpus is not None:
        env["TVFLIDS_SIM_CLIENT_CPUS"] = str(client_cpus)
    if client_gpus is not None:
        env["TVFLIDS_SIM_CLIENT_GPUS"] = str(client_gpus)

    print(f"\n--- mode: {concurrency} concurrent worker(s), {rounds} rounds each ---")
    sampler = Sampler()
    sampler.start()

    procs = []
    t0 = time.time()
    for j in jobs:
        lp = os.path.join(out_dir, f"c{concurrency}_{j['label']}.log")
        fh = open(lp, "w", encoding="utf-8", errors="replace")
        fh.write(f"# cmd: {' '.join(j['cmd'])}\n\n")
        fh.flush()
        procs.append((j, subprocess.Popen(j["cmd"], cwd=ROOT, stdout=fh,
                                           stderr=subprocess.STDOUT, env=env), fh, lp))
    per_job = []
    for j, p, fh, lp in procs:
        rc = p.wait()
        fh.close()
        ok = rc == 0 and _cell_ok(j["log_dir"], rounds)
        per_job.append({"label": j["label"], "returncode": rc, "ok": ok,
                        "log": os.path.relpath(lp, ROOT).replace(os.sep, "/")})
    wall = time.time() - t0
    metrics = sampler.stop()

    n_ok = sum(1 for r in per_job if r["ok"])
    n_fail = len(per_job) - n_ok
    result = {
        "concurrency": concurrency,
        "rounds_per_job": rounds,
        "jobs": len(jobs),
        "wall_seconds": round(wall, 1),
        "wall_minutes": round(wall / 60, 2),
        "minutes_per_job": round((wall / 60) / max(len(jobs), 1), 2),
        "jobs_completed": n_ok,
        "failures": n_fail,
        # Throughput is the honest comparison metric: completed jobs per hour.
        "completed_jobs_per_hour": (round(n_ok / (wall / 3600), 2)
                                    if wall > 0 else None),
        "rounds_per_minute": (round(n_ok * rounds / (wall / 60), 2)
                              if wall > 0 else None),
        "stable": n_fail == 0,
        "resources": metrics,
        "per_job": per_job,
        "env": {k: env.get(k) for k in
                ("TVFLIDS_SIM_CLIENT_CPUS", "TVFLIDS_SIM_CLIENT_GPUS",
                 "OMP_NUM_THREADS")},
    }
    print(f"    wall {result['wall_minutes']:.2f} min | "
          f"{result['completed_jobs_per_hour']} jobs/h | "
          f"{n_ok} ok, {n_fail} failed")
    gm = metrics["gpu_memory_used_mib"]["peak"]
    gu = metrics["gpu_utilisation_pct"]["mean"]
    print(f"    GPU peak {gm} MiB | GPU util mean {gu} % | "
          f"host mem peak {metrics['host_memory_used_gb']['peak']} GB")
    return result


def _cell_ok(log_dir: str, rounds: int) -> bool:
    path = os.path.join(log_dir, "experiment_log.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    return (len(blob.get("rounds") or []) == rounds + 1
            and "final_accuracy" in (blob.get("summary") or {}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=5,
                    help="rounds per benchmark job (keep small)")
    ap.add_argument("--modes", type=int, nargs="+", default=[1, 2],
                    help="concurrency levels to compare")
    ap.add_argument("--root", default=None,
                    help="throwaway results root (default: "
                          "results/_campaign_logs/colab_bench)")
    ap.add_argument("--client-cpus", default=None)
    ap.add_argument("--client-gpus", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--keep", action="store_true",
                    help="keep the benchmark's throwaway cell logs")
    args = ap.parse_args()

    if args.rounds > 20:
        print(f"[bench] refusing --rounds {args.rounds}: this is a timing "
              f"probe, not an experiment. Use 20 or fewer.")
        return 2

    root = args.root or os.path.join("results", "_campaign_logs", "colab_bench")
    out_dir = os.path.join(root, "job_logs")
    os.makedirs(out_dir, exist_ok=True)

    try:
        import torch
        gpu = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
               else "none (CPU)")
        build = torch.__version__
    except Exception as exc:                                  # noqa: BLE001
        gpu, build = f"unknown ({exc})", "unknown"

    print("=" * 74)
    print("TV-FLIDS CONCURRENCY BENCHMARK")
    print("=" * 74)
    print(f"  torch        {build}")
    print(f"  GPU          {gpu}")
    print(f"  modes        {args.modes}")
    print(f"  rounds/job   {args.rounds}")
    print(f"  root         {root}")

    results = []
    for c in args.modes:
        results.append(run_mode(c, args.rounds, root, out_dir,
                                 args.client_cpus, args.client_gpus))

    stable = [r for r in results if r["stable"] and r["completed_jobs_per_hour"]]
    best = max(stable, key=lambda r: r["completed_jobs_per_hour"]) if stable else None

    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    print(f"{'Concurrency':>11} {'Wall(min)':>10} {'min/job':>8} "
          f"{'jobs/h':>8} {'GPUmem(MiB)':>12} {'GPUutil%':>9} "
          f"{'HostRAM(GB)':>12} {'Fail':>5} {'Rec':>4}")
    print("-" * 74)
    for r in results:
        res = r["resources"]
        rec = "yes" if best and r["concurrency"] == best["concurrency"] else ""
        print(f"{r['concurrency']:>11} {r['wall_minutes']:>10.2f} "
              f"{r['minutes_per_job']:>8.2f} "
              f"{str(r['completed_jobs_per_hour']):>8} "
              f"{str(res['gpu_memory_used_mib']['peak']):>12} "
              f"{str(res['gpu_utilisation_pct']['mean']):>9} "
              f"{str(res['host_memory_used_gb']['peak']):>12} "
              f"{r['failures']:>5} {rec:>4}")

    print()
    if best is None:
        print("  RECOMMENDATION: none of the tested modes was stable. Run the "
              "campaign at --workers 1 and investigate the job logs.")
        recommended = 1
    else:
        recommended = best["concurrency"]
        print(f"  RECOMMENDED MAX_PARALLEL_JOBS = {recommended}  "
              f"({best['completed_jobs_per_hour']} completed jobs/h, "
              f"0 failures)")
        others = [r for r in results if r["concurrency"] != recommended]
        for r in others:
            if not r["stable"]:
                print(f"    concurrency {r['concurrency']} rejected: "
                      f"{r['failures']} failure(s)")
            elif r["completed_jobs_per_hour"]:
                delta = (best["completed_jobs_per_hour"]
                         - r["completed_jobs_per_hour"])
                print(f"    concurrency {r['concurrency']}: "
                      f"{r['completed_jobs_per_hour']} jobs/h "
                      f"({delta:+.2f} vs recommended)")
        print("\n  These are timings on THIS runtime. They say nothing about "
              "any scientific quantity.")
        # Two honest caveats, printed every time so they cannot be forgotten.
        print(f"\n  CAVEAT 1 - short jobs favour parallelism. Each job here "
              f"ran {args.rounds} round(s), so Ray/Flower start-up (tens of "
              f"seconds, largely serial) is a large share of the measured "
              f"wall time. A 100-round campaign cell amortises that start-up "
              f"over ~50x more compute, so the real gain from concurrency is "
              f"SMALLER than this table suggests.")
        if not any(r["resources"]["gpu_memory_used_mib"]["peak"]
                   for r in results):
            print("  CAVEAT 2 - no GPU memory was used during this benchmark, "
                  "so it measured CPU contention only. A single T4 shared by "
                  "two workers contends very differently (one GPU, few "
                  "vCPUs). Re-run this on the T4 before trusting the "
                  "recommendation above.")

    payload = {
        "benchmarked_utc": _now(),
        "torch": build,
        "gpu": gpu,
        "rounds_per_job": args.rounds,
        "modes": args.modes,
        "recommended_max_parallel_jobs": recommended,
        "note": "Wall-clock timings only. Not a scientific result. The "
                 "benchmark's cell logs are throwaway and are written under a "
                 "_bench path that no campaign cell uses.",
        "results": results,
    }
    out_json = args.json or os.path.join(root, "benchmark.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\n[bench] report -> {out_json}")

    if not args.keep:
        shutil.rmtree(os.path.join(root, "logs", "_bench"), ignore_errors=True)
        print("[bench] throwaway cell logs removed (--keep to retain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
