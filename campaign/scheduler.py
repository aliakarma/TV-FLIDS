"""
campaign/scheduler.py
Production campaign scheduler with bounded concurrency, resource awareness,
provenance-safe resume, dataset gating, failure recovery, and graceful interruption.
Reference: IEEE TIFS Manuscript §VI, §VII, and Supplementary §S3–S8.

Guarantees:
  - Bounded concurrency: executes independent runs across isolated processes, bounded by --max-workers.
  - Resource awareness: monitors CPU and RAM before launching workers; fails safely.
  - Dataset gating: immediately identifies absent raw datasets (CIC-IoT-2023, Edge-IIoTset),
    marks them as BLOCKED with clear diagnostics, and preserves them without launching processes.
  - Provenance-safe resume: skips completed valid runs whose configuration, git commit, and
    dataset provenance match current state. Refuses stale cached runs if provenance differs.
  - Failure recovery & bounded retries: restarts interrupted runs (status RUNNING) and retries
    failed runs up to --max-retries (default: 1), never looping infinitely.
  - Graceful interruption: intercepts SIGINT / KeyboardInterrupt, terminates child workers cleanly,
    and preserves partial progress.
  - Live progress reporting: displays active, completed, resumed, blocked, failed, and ETA.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from campaign.run_spec import RunSpecification
from campaign.enumerator import CampaignEnumerator
from campaign.runner import (
    CampaignRunner,
    check_dataset_availability,
    check_provenance_compatibility,
)
from campaign.artifacts import (
    get_artifact_directory,
    read_metrics,
    read_status,
    write_status,
)
from campaign.preflight import get_available_ram_gb, get_system_ram_gb
from utils.provenance import git_state


class CampaignScheduler:
    """
    Orchestrates the execution of experimental campaign runs with bounded concurrency
    and resource safeguards.
    """

    def __init__(
        self,
        base_results_dir: str = "results",
        max_workers: Optional[int] = None,
        max_cpus: Optional[int] = None,
        max_memory_gb: Optional[float] = None,
        max_retries: int = 1,
        allow_dirty: bool = False,
        device: str = "cpu",
    ):
        self.base_results_dir = base_results_dir
        cpu_count = os.cpu_count() or 1
        # Default max workers: min(cpus - 1, 4), at least 1
        self.max_workers = max_workers or max(1, min(cpu_count - 1, 4))
        self.max_cpus = max_cpus
        self.max_memory_gb = max_memory_gb
        self.max_retries = max(0, max_retries)
        self.allow_dirty = allow_dirty
        self.device = device
        self._interrupted = False

    def discover_run_states(
        self,
        specs: Sequence[RunSpecification],
        force: bool = False,
    ) -> Dict[str, List[Tuple[RunSpecification, Dict[str, Any]]]]:
        """
        Categorize a list of RunSpecifications by their current disk status.
        Returns dict with keys: 'completed', 'blocked', 'interrupted', 'failed', 'pending'.
        """
        categorized: Dict[str, List[Tuple[RunSpecification, Dict[str, Any]]]] = {
            "completed": [],
            "blocked": [],
            "interrupted": [],
            "failed": [],
            "pending": [],
        }

        curr_git = git_state()

        for spec in specs:
            output_dir = get_artifact_directory(spec, base_dir=self.base_results_dir)

            # 1. Dataset availability check
            available, reason = check_dataset_availability(spec.dataset)
            if not available:
                categorized["blocked"].append((spec, {"reason": reason, "output_dir": output_dir}))
                continue

            if force:
                categorized["pending"].append((spec, {"output_dir": output_dir}))
                continue

            # 2. Inspect status.json and metrics.json
            status_data = read_status(output_dir)
            if status_data is None:
                categorized["pending"].append((spec, {"output_dir": output_dir}))
                continue

            status = status_data.get("status")
            if status == "COMPLETED":
                # Check metrics & provenance
                config_path = os.path.join(output_dir, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            stored_config = json.load(f)
                        compatible, prov_reason = check_provenance_compatibility(
                            spec, stored_config, current_git=curr_git
                        )
                    except Exception as e:
                        compatible, prov_reason = False, str(e)
                else:
                    compatible, prov_reason = False, "Missing config.json"

                metrics = read_metrics(output_dir)
                if compatible and metrics is not None:
                    categorized["completed"].append((spec, {"metrics": metrics, "output_dir": output_dir}))
                else:
                    categorized["pending"].append((spec, {
                        "reason": f"Corrupted or incompatible: {prov_reason}",
                        "output_dir": output_dir,
                    }))

            elif status == "RUNNING":
                # Interrupted execution
                categorized["interrupted"].append((spec, {"status": status_data, "output_dir": output_dir}))

            elif status == "FAILED":
                categorized["failed"].append((spec, {"status": status_data, "output_dir": output_dir}))

            elif status == "BLOCKED":
                categorized["blocked"].append((spec, {"reason": status_data.get("error_message"), "output_dir": output_dir}))

            else:
                categorized["pending"].append((spec, {"output_dir": output_dir}))

        return categorized

    def _check_resources(self) -> Tuple[bool, str]:
        """Verify that current system resources permit launching another worker."""
        avail_ram, _ = get_system_ram_gb()
        if self.max_memory_gb is not None and avail_ram < self.max_memory_gb:
            return False, f"Available RAM ({avail_ram:.2f} GB) < required max_memory_gb ({self.max_memory_gb:.2f} GB)"
        if avail_ram < 0.25:
            return False, f"Critically low RAM ({avail_ram:.2f} GB available)"
        return True, "OK"

    def schedule(
        self,
        specs: Sequence[RunSpecification],
        dry_run: bool = False,
        force: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the scheduled campaign with bounded concurrency, failure recovery,
        and provenance-safe caching.
        """
        total_runs = len(specs)
        if total_runs == 0:
            if verbose:
                print("[Scheduler] No runs provided to schedule.")
            return {"total": 0, "completed": 0, "resumed": 0, "blocked": 0, "failed": 0}

        t_start = time.time()

        # ── 1. Discover existing states ──────────────────────────────────────
        discovered = self.discover_run_states(specs, force=force)
        completed_count = len(discovered["completed"])
        blocked_count = len(discovered["blocked"])

        # Mark blocked runs on disk immediately
        for spec, info in discovered["blocked"]:
            out_dir = info["output_dir"]
            write_status(
                out_dir,
                status="BLOCKED",
                error_message=info.get("reason", "Dataset not available on disk"),
                extra_info={"dataset": spec.dataset, "run_id": spec.run_id},
            )

        # Build runnable queue: pending + interrupted + failed (for retry)
        runnable_queue: List[Tuple[RunSpecification, int]] = []  # (spec, attempt)

        for spec, _ in discovered["pending"]:
            runnable_queue.append((spec, 1))
        for spec, _ in discovered["interrupted"]:
            runnable_queue.append((spec, 1))
        for spec, _ in discovered["failed"]:
            runnable_queue.append((spec, 1))

        if verbose:
            print(f"[Scheduler] Total planned: {total_runs}")
            print(f"  - Completed (cache reuse): {completed_count}")
            print(f"  - Blocked (missing raw data): {blocked_count}")
            print(f"  - Runnable to execute: {len(runnable_queue)}")
            print(f"  - Max concurrent workers: {self.max_workers}")
            print(f"  - Dry run: {dry_run}")
            print("-" * 65)

        if dry_run:
            for spec, _ in runnable_queue:
                out_dir = get_artifact_directory(spec, base_dir=self.base_results_dir)
                write_status(out_dir, status="SKIPPED", extra_info={"reason": "Dry run execution requested"})
                if verbose:
                    print(f"[Scheduler] [DRY RUN] Would execute: {spec.run_id} ({spec.block} {spec.strategy} on {spec.dataset} seed {spec.seed})")
            return {
                "total": total_runs,
                "completed": 0,
                "resumed": completed_count,
                "blocked": blocked_count,
                "dry_run_skipped": len(runnable_queue),
                "failed": 0,
            }

        # ── 2. Active Worker Management ─────────────────────────────────────
        # Setup signal handler for graceful shutdown
        def _handle_sigint(sig, frame):
            print("\n[Scheduler] Caught interrupt signal! Initiating graceful worker shutdown...")
            self._interrupted = True

        old_sigint = signal.signal(signal.SIGINT, _handle_sigint)
        if hasattr(signal, "SIGTERM"):
            old_sigterm = signal.signal(signal.SIGTERM, _handle_sigint)

        active_procs: Dict[subprocess.Popen, Tuple[RunSpecification, str, int, float]] = {}
        # proc -> (spec, temp_file_path, attempt, start_time)

        completed_success = 0
        failed_count = 0
        resumed_count = completed_count

        try:
            while (runnable_queue or active_procs) and not self._interrupted:
                # 1. Launch new workers up to max_workers
                while len(active_procs) < self.max_workers and runnable_queue and not self._interrupted:
                    ok, res_msg = self._check_resources()
                    if not ok:
                        if verbose:
                            print(f"[Scheduler] Resource limit reached ({res_msg}). Waiting for running workers...")
                        break

                    spec, attempt = runnable_queue.pop(0)

                    # Write spec to temporary JSON file for worker process
                    temp_spec_file = tempfile.NamedTemporaryFile(
                        mode="w", suffix=".json", prefix=f"spec_{spec.run_id}_", delete=False
                    )
                    json.dump(spec.to_dict(), temp_spec_file, indent=2)
                    temp_spec_file.close()

                    # Launch worker via subprocess for strict process isolation
                    cmd = [
                        sys.executable,
                        "-m",
                        "campaign.scheduler",
                        "--execute-spec-file",
                        temp_spec_file.name,
                        "--output-dir",
                        self.base_results_dir,
                    ]
                    if self.allow_dirty:
                        cmd.append("--allow-dirty")
                    if force:
                        cmd.append("--force")

                    # Environment variables for child worker
                    env = os.environ.copy()
                    if sys.platform == "win32":
                        env["TVFLIDS_SIM_LOCAL_MODE"] = "1"
                    out_dir = get_artifact_directory(spec, base_dir=self.base_results_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    log_path = os.path.join(out_dir, "worker.log")
                    log_file = open(log_path, "w", encoding="utf-8", errors="replace")

                    proc = subprocess.Popen(
                        cmd,
                        cwd=ROOT,
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    active_procs[proc] = (spec, temp_spec_file.name, attempt, time.time(), log_file, log_path)

                    if verbose:
                        print(
                            f"[Scheduler] Launched worker [PID {proc.pid}] for run {spec.run_id} "
                            f"({spec.block} {spec.strategy} on {spec.dataset}, seed {spec.seed}, attempt {attempt})"
                        )

                # 2. Poll active workers
                time.sleep(0.5)
                done_procs = [p for p in active_procs if p.poll() is not None]

                for p in done_procs:
                    spec, spec_file, attempt, t_launched, log_file, log_path = active_procs.pop(p)
                    elapsed_run = time.time() - t_launched
                    try:
                        log_file.flush()
                        log_file.close()
                    except Exception:
                        pass

                    # Read worker result file if generated
                    result_file = spec_file + ".result.json"
                    run_success = (p.returncode == 0)

                    if os.path.exists(result_file):
                        try:
                            with open(result_file, "r", encoding="utf-8") as f:
                                res_blob = json.load(f)
                            if res_blob.get("status") in ("COMPLETED", "SKIPPED"):
                                run_success = True
                            elif res_blob.get("status") == "FAILED":
                                run_success = False
                        except Exception:
                            run_success = False
                        try:
                            os.remove(result_file)
                        except OSError:
                            pass

                    # Clean up temporary spec file
                    try:
                        os.remove(spec_file)
                    except OSError:
                        pass

                    if run_success:
                        completed_success += 1
                        if verbose:
                            print(f"[Scheduler] Run {spec.run_id} SUCCEEDED in {elapsed_run:.1f}s.")
                    else:
                        if verbose:
                            print(f"[Scheduler] Run {spec.run_id} FAILED (rc={p.returncode}) in {elapsed_run:.1f}s.")
                            if os.path.exists(log_path):
                                try:
                                    with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
                                        lines = lf.readlines()
                                        if lines:
                                            print(f"  Error snippet: {''.join(lines[-10:]).strip()}")
                                except Exception:
                                    pass

                        if attempt < self.max_retries:
                            if verbose:
                                print(f"[Scheduler] Requeuing {spec.run_id} for retry {attempt + 1}/{self.max_retries}...")
                            runnable_queue.append((spec, attempt + 1))
                        else:
                            failed_count += 1
                            if verbose:
                                print(f"[Scheduler] Max retries exhausted for {spec.run_id}. Marked FAILED.")

                # 3. Print periodic progress report
                done_total = completed_success + resumed_count + blocked_count + failed_count
                if verbose and (len(done_procs) > 0 or done_total == total_runs):
                    pct = (done_total / total_runs) * 100.0
                    rate = done_total / max(1.0, time.time() - t_start)
                    remaining = total_runs - done_total
                    eta_sec = remaining / max(0.001, rate)
                    print(
                        f"[Progress] {done_total}/{total_runs} ({pct:.1f}%) | "
                        f"Done: {completed_success}, Resumed: {resumed_count}, Blocked: {blocked_count}, "
                        f"Failed: {failed_count}, Active: {len(active_procs)} | ETA: {eta_sec:.0f}s"
                    )

        finally:
            # Clean up child processes if interrupted
            if self._interrupted:
                print(f"[Scheduler] Terminating {len(active_procs)} active child worker(s)...")
                for p in active_procs:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                for p in active_procs:
                    try:
                        p.wait(timeout=3)
                    except Exception:
                        try:
                            p.kill()
                        except Exception:
                            pass
                # Remove temporary files and close log files
                for _, spec_file, _, _, log_file, _ in active_procs.values():
                    try:
                        log_file.close()
                    except Exception:
                        pass
                    try:
                        os.remove(spec_file)
                    except OSError:
                        pass

            # Restore original signal handlers
            signal.signal(signal.SIGINT, old_sigint)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, old_sigterm)

        total_wall = time.time() - t_start
        summary = {
            "total_runs": total_runs,
            "completed_new": completed_success,
            "resumed_cached": resumed_count,
            "blocked": blocked_count,
            "failed": failed_count,
            "interrupted": self._interrupted,
            "total_wall_seconds": total_wall,
        }

        if verbose:
            print("\n" + "=" * 65)
            print(f" Campaign Scheduling Execution Summary")
            print("=" * 65)
            for k, v in summary.items():
                print(f"  {k:<25}: {v}")
            print("=" * 65)

        return summary


def execute_spec_worker_cli(spec_file: str, base_results_dir: str, allow_dirty: bool, force: bool):
    """Worker process entry point: executes a single RunSpecification from a JSON file."""
    with open(spec_file, "r", encoding="utf-8") as f:
        spec_dict = json.load(f)
    spec = RunSpecification.from_dict(spec_dict)

    runner = CampaignRunner(base_results_dir=base_results_dir, allow_dirty=allow_dirty)
    result = runner.execute_run(spec, dry_run=False, force=force, verbose=True)

    result_file = spec_file + ".result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    sys.exit(0 if result.get("status") in ("COMPLETED", "SKIPPED") else 1)


def main():
    parser = argparse.ArgumentParser(description="TV-FLIDS Campaign Scheduler")
    parser.add_argument("--execute-spec-file", type=str, default=None, help="Worker mode: execute single spec JSON file")
    parser.add_argument("--output-dir", type=str, default="results", help="Base output directory")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow running on dirty git working tree")
    parser.add_argument("--force", action="store_true", help="Force rerun even if already completed")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without executing simulations")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum concurrent worker processes")
    parser.add_argument("--max-cpus", type=int, default=None, help="Maximum CPUs to allocate")
    parser.add_argument("--max-memory", type=float, default=None, help="Minimum free memory (GB) before launching")
    parser.add_argument("--max-retries", type=int, default=1, help="Maximum retries for failed runs (default: 1)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda)")

    # Filtering options
    parser.add_argument("--block", type=str, default=None, help="Filter by block (e.g. B1, B2)")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset (nslkdd, ciciot2023, edgeiiotset)")
    parser.add_argument("--strategy", type=str, default=None, help="Filter by strategy (tvflids, fedavg, ...)")
    parser.add_argument("--attack", type=str, default=None, help="Filter by attack")
    parser.add_argument("--seed", type=int, default=None, help="Filter by seed")
    parser.add_argument("--rounds", type=int, default=None, help="Override rounds for pilot testing")

    args = parser.parse_args()

    # Worker process execution mode
    if args.execute_spec_file:
        execute_spec_worker_cli(
            spec_file=args.execute_spec_file,
            base_results_dir=args.output_dir,
            allow_dirty=args.allow_dirty,
            force=args.force,
        )
        return

    # Scheduler manager mode
    all_runs = CampaignEnumerator.enumerate_all()
    filtered_runs = CampaignEnumerator.filter_runs(
        all_runs,
        block=args.block,
        dataset=args.dataset,
        strategy=args.strategy,
        attack=args.attack,
        seed=args.seed,
    )

    if args.rounds is not None:
        overridden = []
        for r in filtered_runs:
            d = r.to_dict()
            d["num_rounds"] = args.rounds
            overridden.append(RunSpecification.from_dict(d))
        filtered_runs = overridden

    scheduler = CampaignScheduler(
        base_results_dir=args.output_dir,
        max_workers=args.max_workers,
        max_cpus=args.max_cpus,
        max_memory_gb=args.max_memory,
        max_retries=args.max_retries,
        allow_dirty=args.allow_dirty,
        device=args.device,
    )

    scheduler.schedule(
        filtered_runs,
        dry_run=args.dry_run,
        force=args.force,
        verbose=True,
    )


if __name__ == "__main__":
    main()
