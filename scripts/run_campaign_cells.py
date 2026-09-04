"""
scripts/run_campaign_cells.py - parallel pre-executor for campaign cells.

Why this exists
---------------
Every campaign family (run_full_comparison, run_ablation, run_multi_attack_matrix,
...) is a sequential loop over independent (strategy, attack, seed, ...) cells
inside a single process. On a 12-thread laptop one such process leaves most of
the machine idle, and a crash 30 hours in loses everything.

This script runs the *same* cells as separate processes, N at a time, writing
each cell's artifact to exactly the log_dir its own family runner would use.
The family runner is then invoked normally with TVFLIDS_RESUME=1: it finds each
cell's completed experiment_log.json, reuses that run's stored summary, and
performs the aggregation, statistics and table writing itself.

What this does NOT do
---------------------
It does not compute, transform, average or synthesise any metric. Each cell is
produced by an ordinary `python experiments/run_experiment.py ...` invocation
with the identical arguments the family runner would have passed. The only
difference is which process executes it and when.

Usage
-----
    python scripts/run_campaign_cells.py --jobs jobs.json --workers 2
    python scripts/run_campaign_cells.py --jobs jobs.json --workers 2 --dry-run

jobs.json is a list of objects; every key maps to a run_experiment.py flag:

    {"strategy": "tvflids", "attack": "label_flip_30", "seed": 42,
     "rounds": 100, "log_dir": "results/logs/comparison/tvflids_label_flip_30_seed42",
     "partition": "noniid", "alpha": 0.5, "protocol": "main",
     "config": "config/fl_config.yaml", "dataset": "nslkdd"}
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PRINT_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {msg}", flush=True)


def is_complete(log_dir: str, rounds: int) -> bool:
    """True only for a cell whose own log is present and covers every round."""
    path = os.path.join(log_dir, "experiment_log.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    # n FL rounds produce n+1 logged evaluation points (round 0 is the
    # pre-training baseline evaluation, not an FL round).
    logged = blob.get("rounds") or []
    return (len(logged) == rounds + 1
            and max((r.get("round", -1) for r in logged), default=-1) == rounds
            and "final_accuracy" in (blob.get("summary") or {}))


def build_cmd(job: Dict) -> List[str]:
    cmd = [sys.executable, "experiments/run_experiment.py",
           "--strategy", str(job["strategy"]),
           "--attack", str(job["attack"]),
           "--seed", str(job["seed"]),
           "--rounds", str(job["rounds"]),
           "--log_dir", str(job["log_dir"]),
           "--quiet"]
    for key, flag in (("partition", "--partition"), ("alpha", "--alpha"),
                      ("protocol", "--protocol"), ("config", "--config"),
                      ("dataset", "--dataset"), ("val_size", "--val-size"),
                      ("model", "--model")):
        if job.get(key) is not None:
            cmd += [flag, str(job[key])]
    return cmd


def run_job(job: Dict, out_root: str) -> Dict:
    log_dir = job["log_dir"]
    rounds = int(job["rounds"])
    if is_complete(log_dir, rounds):
        return {"job": job, "status": "cached", "seconds": 0.0}

    os.makedirs(log_dir, exist_ok=True)
    stdout_path = os.path.join(out_root,
                               log_dir.replace("/", "_").replace("\\", "_") + ".out")
    os.makedirs(os.path.dirname(stdout_path) or ".", exist_ok=True)

    started = time.time()
    with open(stdout_path, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(build_cmd(job), cwd=ROOT, stdout=fh,
                              stderr=subprocess.STDOUT)
    elapsed = time.time() - started

    ok = proc.returncode == 0 and is_complete(log_dir, rounds)
    return {
        "job": job,
        "status": "ok" if ok else "failed",
        "returncode": proc.returncode,
        "seconds": elapsed,
        "stdout": stdout_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Parallel campaign cell executor")
    ap.add_argument("--jobs", required=True, help="JSON file: list of job dicts")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-root", default="results/_campaign_logs/cells")
    ap.add_argument("--report", default=None,
                    help="Write a JSON execution report here")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.jobs, encoding="utf-8") as fh:
        jobs = json.load(fh)

    os.makedirs(args.out_root, exist_ok=True)

    todo = [j for j in jobs if not is_complete(j["log_dir"], int(j["rounds"]))]
    _log(f"{len(jobs)} cells total | {len(jobs) - len(todo)} already complete "
         f"| {len(todo)} to run | {args.workers} worker(s)")

    if args.dry_run:
        for j in todo:
            print("  " + " ".join(build_cmd(j)))
        return 0

    q: "queue.Queue[Dict]" = queue.Queue()
    for j in todo:
        q.put(j)

    results: List[Dict] = []
    results_lock = threading.Lock()
    done_count = [0]
    t0 = time.time()

    def worker(wid: int) -> None:
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            label = os.path.basename(job["log_dir"])
            _log(f"w{wid} start  {label}")
            res = run_job(job, args.out_root)
            with results_lock:
                results.append(res)
                done_count[0] += 1
                n, total = done_count[0], len(todo)
                rate = (time.time() - t0) / max(n, 1)
                eta = rate * (total - n) / 60.0
            _log(f"w{wid} {res['status']:6s} {label} "
                 f"({res['seconds']/60:.1f} min) [{n}/{total}] ETA {eta:.0f} min")
            q.task_done()

    threads = [threading.Thread(target=worker, args=(i + 1,), daemon=True)
               for i in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    _log(f"done: {n_ok} ok, {n_fail} failed, "
         f"{len(jobs) - len(todo)} already complete, "
         f"total wall {(time.time() - t0)/3600:.2f} h")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({"jobs_file": args.jobs, "results": results}, fh,
                      indent=2, default=str)
        _log(f"report -> {args.report}")

    if n_fail:
        for r in results:
            if r["status"] == "failed":
                _log(f"FAILED rc={r['returncode']} {r['job']['log_dir']} "
                     f"(see {r['stdout']})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
