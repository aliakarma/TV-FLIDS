"""
colab/parallel_runner.py — bounded, resumable, GPU-aware campaign execution.

    python colab/parallel_runner.py --workers 2                    # run everything
    python colab/parallel_runner.py --workers 2 --priority 1        # main paper only
    python colab/parallel_runner.py --workers 1 --phase A_main_comparison
    python colab/parallel_runner.py --status                        # queue snapshot
    python colab/parallel_runner.py --retry-failed --workers 2
    python colab/parallel_runner.py --dry-run --workers 2

DESIGN
------
Two stages, because the repository's runners come in two shapes.

  Stage 1 — cell production (parallel, bounded)
     "cell" phases:  every cell is a plain `experiments/run_experiment.py`
        invocation, so each cell is its own job. Maximum parallelism, and a
        crash loses at most one cell.
     "shard" phases:  the family runner writes a temp YAML per cell (ablation
        trust overrides, hyperparameter grids, ratio overrides). Those configs
        are scientific content, so this module refuses to reproduce them.
        Instead the family runner itself is sharded along the axes it already
        exposes on its CLI (`--seeds`, `--methods`), each shard writing its
        aggregate to a throwaway directory.

  Stage 2 — aggregation (sequential, cheap)
     Each phase's family runner is invoked once over the FULL grid with
     TVFLIDS_RESUME=1 and the real `--output`. Every cell is already on disk, so
     the runner reuses each stored summary and does only what it alone should
     do: aggregate, run the statistics, and write the phase artifact.

Nothing here computes, adjusts, averages or synthesises a metric. Stage 1 issues
the identical command the family runner would have issued — the per-cell command
is built by `scripts/run_campaign_cells.py::build_cmd`, imported rather than
copied, so the two can never drift. Stage 2 is the repository's own aggregation.

QUEUE
-----
`<state-dir>/queue.json` holds one record per unit of work with status
pending / running / complete / failed / retrying, attempt count, timings, the
resolved command, and the per-job log path. It is written atomically (temp file
+ os.replace) after every transition, so a Colab disconnect leaves a readable
queue. On restart, `complete` units are skipped, `running` units are re-queued
(the process is gone), and a unit whose cells are all already on disk is marked
complete without being run.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import queue
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

import campaign_inventory as inv                      # noqa: E402
from scripts.run_campaign_cells import build_cmd, is_complete   # noqa: E402

DEFAULT_STATE = os.path.join("results", "_campaign", "colab")
_LOCK = threading.Lock()
_QUEUE_LOCK = threading.Lock()


def _now() -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            .isoformat().replace("+00:00", "Z"))


def _log(msg: str) -> None:
    with _LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _atomic_write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


# ── Unit construction ────────────────────────────────────────────────────────
# A "unit" is one schedulable process. For a cell phase it is one cell; for a
# shard phase it is one shard of the family runner covering several cells.

def build_units(results_root: str,
                priorities: Optional[List[int]] = None,
                phases: Optional[List[str]] = None) -> List[Dict]:
    cells_by_phase = inv.cells_by_phase()
    units: List[Dict] = []

    for spec in sorted(inv.PHASES, key=lambda p: p["order"]):
        phase = spec["phase"]
        if phases and phase not in phases:
            continue
        if priorities and spec["priority"] not in priorities:
            continue
        cells = cells_by_phase.get(phase, [])
        if not cells:
            continue

        if spec["exec_mode"] == "cell":
            for c in cells:
                log_dir = os.path.join(results_root, *c["log_dir"].split("/"))
                job = {
                    "strategy": c["strategy"],
                    "attack": c["attack"],
                    "seed": c["seed"],
                    "rounds": c["rounds"],
                    "log_dir": log_dir.replace(os.sep, "/"),
                    "partition": c["partition_type"],
                    "alpha": c["alpha"],
                    "protocol": c["protocol"],
                    "dataset": c["dataset"],
                    "config": "config/fl_config.yaml",
                }
                units.append({
                    "unit_id": c["cell_id"],
                    "kind": "cell",
                    "phase": phase,
                    "priority": spec["priority"],
                    "order": spec["order"],
                    "paper_artifact": spec["paper_artifact"],
                    "cell_ids": [c["cell_id"]],
                    "log_dirs": [job["log_dir"]],
                    "rounds": c["rounds"],
                    "job": job,
                    "cmd": build_cmd(job),
                })
            continue

        # ── shard phase ──────────────────────────────────────────────────
        scratch = os.path.join(results_root, "_campaign", "shard_output",
                               phase).replace(os.sep, "/")
        axes = spec["shard_by"]
        if axes == ["--seeds"]:
            groups = [{"--seeds": [str(s)]} for s in sorted({c["seed"] for c in cells})]
        elif axes == ["--methods", "--seeds"]:
            groups = [
                {"--methods": [m], "--seeds": [str(s)]}
                for m in inv.RATIO_METHODS
                for s in sorted({c["seed"] for c in cells})
            ]
        else:
            raise ValueError(f"unhandled shard axes for {phase}: {axes}")

        for g in groups:
            argv = list(spec["argv"])
            # Replace the sharded axis values, keep every other flag verbatim.
            for flag, values in g.items():
                argv = _replace_flag(argv, flag, values)
            argv = _replace_flag(argv, "--output", [scratch])

            sel_seeds = {int(s) for s in g.get("--seeds", [])}
            sel_methods = set(g.get("--methods", []))
            shard_cells = [
                c for c in cells
                if (not sel_seeds or c["seed"] in sel_seeds)
                and (not sel_methods or c["strategy"] in sel_methods)
            ]
            label = "_".join(
                v for vals in g.values() for v in vals).replace("/", "_")
            units.append({
                "unit_id": f"{phase}/shard/{label}",
                "kind": "shard",
                "phase": phase,
                "priority": spec["priority"],
                "order": spec["order"],
                "paper_artifact": spec["paper_artifact"],
                "cell_ids": [c["cell_id"] for c in shard_cells],
                "log_dirs": [
                    os.path.join(results_root, *c["log_dir"].split("/")
                                  ).replace(os.sep, "/") for c in shard_cells
                ],
                "rounds": inv.ROUNDS,
                "job": None,
                "cmd": [sys.executable, spec["runner"], *argv],
            })

    units.sort(key=lambda u: (u["priority"], u["order"], u["unit_id"]))
    return units


def _replace_flag(argv: List[str], flag: str, values: List[str]) -> List[str]:
    """Set `flag` to `values`, replacing any existing occurrence."""
    out: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(argv[i])
        i += 1
    return out + [flag, *values]


def aggregation_commands(results_root: str,
                         priorities: Optional[List[int]] = None,
                         phases: Optional[List[str]] = None) -> List[Dict]:
    """Stage 2: one full-grid family-runner invocation per phase."""
    out = []
    tables = os.path.join(results_root, "tables").replace(os.sep, "/")
    for spec in sorted(inv.PHASES, key=lambda p: p["order"]):
        if phases and spec["phase"] not in phases:
            continue
        if priorities and spec["priority"] not in priorities:
            continue
        argv = list(spec["argv"])
        # J writes to its own sub-directory; every other phase to tables/.
        if "_extra_baselines" in spec["aggregate_artifact"]:
            argv = _replace_flag(argv, "--output",
                                 [os.path.join(tables, "_extra_baselines")
                                  .replace(os.sep, "/")])
        else:
            argv = _replace_flag(argv, "--output", [tables])
        out.append({
            "phase": spec["phase"],
            "priority": spec["priority"],
            "artifact": spec["aggregate_artifact"],
            "cmd": [sys.executable, spec["runner"], *argv],
        })
    return out


# ── Persistent queue ─────────────────────────────────────────────────────────

class Queue:
    def __init__(self, path: str, units: List[Dict], results_root: str):
        self.path = path
        self.results_root = results_root
        self.records: Dict[str, Dict] = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    blob = json.load(fh)
                self.records = {r["unit_id"]: r for r in blob.get("units", [])}
                _log(f"queue: resumed {len(self.records)} unit(s) from {path}")
            except (OSError, json.JSONDecodeError) as exc:
                _log(f"queue: {path} unreadable ({exc}); starting a new queue")
                self.records = {}

        for u in units:
            rec = self.records.get(u["unit_id"])
            if rec is None:
                rec = {
                    "unit_id": u["unit_id"], "kind": u["kind"],
                    "phase": u["phase"], "priority": u["priority"],
                    "order": u["order"], "paper_artifact": u["paper_artifact"],
                    "cell_ids": u["cell_ids"], "log_dirs": u["log_dirs"],
                    "rounds": u["rounds"], "cmd": u["cmd"],
                    "status": "pending", "attempts": 0,
                    "started_utc": None, "finished_utc": None,
                    "elapsed_seconds": None, "returncode": None,
                    "stdout_log": None, "config_snapshot": None,
                    "note": None,
                }
                self.records[u["unit_id"]] = rec
            else:
                # The command can legitimately change (different results root);
                # always adopt the freshly resolved one.
                rec["cmd"] = u["cmd"]
                rec["log_dirs"] = u["log_dirs"]
                rec["cell_ids"] = u["cell_ids"]
                # A unit that was 'running' when the session died is not running.
                if rec["status"] == "running":
                    rec["status"] = "pending"
                    rec["note"] = "re-queued: session ended while running"
        self.save()

    def save(self) -> None:
        with _QUEUE_LOCK:
            payload = {
                "queue_version": 1,
                "updated_utc": _now(),
                "results_root": self.results_root,
                "counts": self.counts(),
                "units": [self.records[k] for k in sorted(self.records)],
            }
            _atomic_write_json(self.path, payload)

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for r in self.records.values():
            c[r["status"]] = c.get(r["status"], 0) + 1
        return c

    def set(self, unit_id: str, **kw) -> None:
        self.records[unit_id].update(kw)
        self.save()

    def refresh_completion(self) -> int:
        """Mark as complete any unit whose every cell log is already valid."""
        n = 0
        for r in self.records.values():
            if r["status"] == "complete":
                continue
            dirs = r["log_dirs"]
            if dirs and all(is_complete(d, int(r["rounds"])) for d in dirs):
                r["status"] = "complete"
                r["note"] = "cells already on disk (resume)"
                n += 1
        if n:
            self.save()
        return n


# ── Execution ────────────────────────────────────────────────────────────────

def snapshot_config(state_dir: str, unit_id: str) -> Optional[str]:
    """Copy the configuration files verbatim, next to the unit's log.

    Real content, real hash — the snapshot is the bytes the run saw.
    """
    import hashlib
    out_dir = os.path.join(state_dir, "config_snapshots",
                           unit_id.replace("/", "__"))
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for rel in ("config/fl_config.yaml", "config/dataset_config.yaml"):
        src = os.path.join(ROOT, rel)
        if not os.path.exists(src):
            manifest[rel] = None
            continue
        raw = open(src, "rb").read()
        dst = os.path.join(out_dir, os.path.basename(rel))
        with open(dst, "wb") as fh:
            fh.write(raw)
        manifest[rel] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "snapshot": os.path.relpath(dst, ROOT).replace(os.sep, "/"),
        }
    mpath = os.path.join(out_dir, "config_snapshot.json")
    _atomic_write_json(mpath, {"unit_id": unit_id, "captured_utc": _now(),
                               "files": manifest})
    return os.path.relpath(mpath, ROOT).replace(os.sep, "/")


def run_unit(rec: Dict, state_dir: str, env: Dict[str, str]) -> Dict:
    log_path = os.path.join(state_dir, "job_logs",
                            rec["unit_id"].replace("/", "__") + ".log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    snap = snapshot_config(state_dir, rec["unit_id"])

    for d in rec["log_dirs"]:
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"# unit    : {rec['unit_id']}\n")
        fh.write(f"# phase   : {rec['phase']}\n")
        fh.write(f"# cmd     : {' '.join(rec['cmd'])}\n")
        fh.write(f"# started : {_now()}\n\n")
        fh.flush()
        proc = subprocess.run(rec["cmd"], cwd=ROOT, stdout=fh,
                              stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0

    done = all(is_complete(d, int(rec["rounds"])) for d in rec["log_dirs"])
    ok = proc.returncode == 0 and done
    note = None
    if proc.returncode == 0 and not done:
        missing = [d for d in rec["log_dirs"]
                   if not is_complete(d, int(rec["rounds"]))]
        note = (f"runner exited 0 but {len(missing)} cell log(s) are absent or "
                f"incomplete, e.g. {missing[0]}")
    return {
        "status": "complete" if ok else "failed",
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "stdout_log": os.path.relpath(log_path, ROOT).replace(os.sep, "/"),
        "config_snapshot": snap,
        "finished_utc": _now(),
        "note": note,
    }


def campaign_env(workers: int, client_cpus: Optional[str],
                 client_gpus: Optional[str]) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["TVFLIDS_RESUME"] = "1"          # never recompute a completed cell
    env["TVFLIDS_TENSORBOARD"] = "0"     # event files are not the record
    env.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    if client_cpus is not None:
        env["TVFLIDS_SIM_CLIENT_CPUS"] = str(client_cpus)
    if client_gpus is not None:
        env["TVFLIDS_SIM_CLIENT_GPUS"] = str(client_gpus)
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="results",
                    help="campaign output root (usually a symlink to "
                          "campaign_results/ on Drive)")
    ap.add_argument("--state-dir", default=None,
                    help="queue + job logs (default: <results-root>/_campaign/colab)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--priority", type=int, action="append", default=None,
                    help="restrict to priority 1/2/3 (repeatable)")
    ap.add_argument("--phase", action="append", default=None)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--client-cpus", default=None,
                    help="TVFLIDS_SIM_CLIENT_CPUS for each Ray virtual client")
    ap.add_argument("--client-gpus", default=None,
                    help="TVFLIDS_SIM_CLIENT_GPUS (e.g. 0.5 for 2 clients/GPU)")
    ap.add_argument("--status", action="store_true", help="print the queue and exit")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-aggregation", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="run at most N units this session (for a timed probe)")
    args = ap.parse_args()

    results_root = args.results_root
    state_dir = args.state_dir or os.path.join(results_root, "_campaign", "colab")
    os.makedirs(state_dir, exist_ok=True)
    qpath = os.path.join(state_dir, "queue.json")

    units = build_units(results_root, args.priority, args.phase)
    q = Queue(qpath, units, results_root)
    cached = q.refresh_completion()

    if args.retry_failed:
        # Respect this invocation's --phase/--priority selection, as execution does.
        selectable = {u["unit_id"] for u in units}
        n = 0
        for uid, r in q.records.items():
            if uid in selectable and r["status"] == "failed":
                r["status"] = "retrying"
                r["attempts"] = 0
                n += 1
        q.save()
        _log(f"queue: {n} failed unit(s) marked for retry")

    if args.status:
        print(json.dumps({"queue": qpath, "counts": q.counts()}, indent=2))
        print(f"\n{'unit':<74} {'status':<10} {'att':>3} {'min':>7}")
        print("-" * 100)
        for k in sorted(q.records):
            r = q.records[k]
            mins = "" if r["elapsed_seconds"] is None else f"{r['elapsed_seconds']/60:.1f}"
            print(f"{k[:73]:<74} {r['status']:<10} {r['attempts']:>3} {mins:>7}")
        return 0

    _log(f"results root : {results_root}")
    _log(f"state dir    : {state_dir}")
    _log(f"units        : {len(q.records)} ({cached} already complete on disk)")
    _log(f"counts       : {q.counts()}")

    env = campaign_env(args.workers, args.client_cpus, args.client_gpus)

    # The queue is campaign-wide and persists across invocations, so it can
    # hold units outside this invocation's --phase/--priority selection. Only
    # the units selected *now* may execute, or a scoped run would quietly
    # widen itself to everything the queue has ever seen.
    selected = {u["unit_id"] for u in units}
    todo = [q.records[k] for k in sorted(
        q.records, key=lambda k: (q.records[k]["priority"],
                                   q.records[k]["order"], k))
            if k in selected
            and q.records[k]["status"] in ("pending", "retrying")]
    if args.limit:
        todo = todo[:args.limit]

    if args.dry_run:
        _log(f"dry run: {len(todo)} unit(s) would execute, {args.workers} at a time")
        for r in todo[:40]:
            print("  " + " ".join(r["cmd"]))
        if len(todo) > 40:
            print(f"  ... and {len(todo) - 40} more")
        if not args.skip_aggregation:
            print("\n  stage 2 (aggregation, sequential):")
            for a in aggregation_commands(results_root, args.priority, args.phase):
                print("    " + " ".join(a["cmd"]))
        return 0

    if not args.aggregate_only and todo:
        work: "queue.Queue[Dict]" = queue.Queue()
        for r in todo:
            work.put(r)
        done = [0]
        t0 = time.time()
        total = len(todo)

        def worker(wid: int) -> None:
            while True:
                try:
                    rec = work.get_nowait()
                except queue.Empty:
                    return
                if all(is_complete(d, int(rec["rounds"])) for d in rec["log_dirs"]):
                    q.set(rec["unit_id"], status="complete",
                          note="cells already on disk (resume)")
                    _log(f"w{wid} cached {rec['unit_id']}")
                    work.task_done()
                    continue
                q.set(rec["unit_id"], status="running",
                      started_utc=_now(), attempts=rec["attempts"] + 1)
                _log(f"w{wid} start  {rec['unit_id']}")
                try:
                    res = run_unit(rec, state_dir, env)
                except Exception as exc:                     # noqa: BLE001
                    res = {"status": "failed", "returncode": None,
                           "elapsed_seconds": None, "finished_utc": _now(),
                           "note": f"{type(exc).__name__}: {exc}"}
                if res["status"] == "failed" and rec["attempts"] < args.max_attempts:
                    res["status"] = "retrying"
                    work.put(rec)
                q.set(rec["unit_id"], **res)
                with _LOCK:
                    done[0] += 1
                    n = done[0]
                    rate = (time.time() - t0) / max(n, 1)
                    eta = rate * max(total - n, 0) / 60.0
                mins = res["elapsed_seconds"] or 0
                _log(f"w{wid} {res['status']:<9} {rec['unit_id']} "
                     f"({mins/60:.1f} min) [{n}/{total}] ETA {eta:.0f} min")
                if res.get("note"):
                    _log(f"     note: {res['note']}")
                work.task_done()

        threads = [threading.Thread(target=worker, args=(i + 1,), daemon=True)
                   for i in range(max(1, args.workers))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        _log(f"stage 1 done in {(time.time()-t0)/3600:.2f} h | counts {q.counts()}")

    # ── Stage 2: aggregation ────────────────────────────────────────────
    if args.skip_aggregation:
        _log("stage 2 skipped (--skip-aggregation)")
        return 0 if not q.counts().get("failed") else 1

    agg_dir = os.path.join(state_dir, "aggregation")
    os.makedirs(agg_dir, exist_ok=True)
    agg_report = []
    for a in aggregation_commands(results_root, args.priority, args.phase):
        phase_units = [r for r in q.records.values() if r["phase"] == a["phase"]]
        if phase_units and not all(r["status"] == "complete" for r in phase_units):
            incomplete = sum(1 for r in phase_units if r["status"] != "complete")
            _log(f"stage 2 skip {a['phase']}: {incomplete} unit(s) not complete "
                 f"- aggregating now would write a partial artifact")
            agg_report.append({"phase": a["phase"], "status": "skipped_incomplete",
                                "incomplete_units": incomplete})
            continue
        lp = os.path.join(agg_dir, a["phase"] + ".log")
        _log(f"stage 2 aggregate {a['phase']} -> {a['artifact']}")
        with open(lp, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(f"# cmd: {' '.join(a['cmd'])}\n\n")
            fh.flush()
            proc = subprocess.run(a["cmd"], cwd=ROOT, stdout=fh,
                                  stderr=subprocess.STDOUT, env=env)
        art = os.path.join(results_root, *a["artifact"].split("/"))
        ok = proc.returncode == 0 and os.path.exists(art) and os.path.getsize(art) > 0
        agg_report.append({
            "phase": a["phase"], "artifact": a["artifact"],
            "returncode": proc.returncode,
            "artifact_written": os.path.exists(art),
            "status": "ok" if ok else "failed",
            "log": os.path.relpath(lp, ROOT).replace(os.sep, "/"),
        })
        _log(f"stage 2 {a['phase']}: {'ok' if ok else 'FAILED'} (rc={proc.returncode})")

    _atomic_write_json(os.path.join(state_dir, "aggregation_report.json"),
                       {"generated_utc": _now(), "phases": agg_report})

    counts = q.counts()
    _log(f"final queue counts: {counts}")
    bad = counts.get("failed", 0) + sum(
        1 for a in agg_report if a["status"] == "failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
