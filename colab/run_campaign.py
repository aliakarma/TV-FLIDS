"""
colab/run_campaign.py — one entry point for the whole Colab campaign.

    python colab/run_campaign.py link                 # results/ -> Drive campaign root
    python colab/run_campaign.py smoke                # tiny pre-campaign test
    python colab/run_campaign.py benchmark            # pick the concurrency
    python colab/run_campaign.py run                  # execute the queue
    python colab/run_campaign.py run --priority 1     # main paper first
    python colab/run_campaign.py status               # queue snapshot
    python colab/run_campaign.py finalize             # write final_manifest.json
    python colab/run_campaign.py validate             # expected vs. actual
    python colab/run_campaign.py artifacts            # tables + figures from results
    python colab/run_campaign.py report               # the per-seed result map

Every setting comes from colab/config_colab.yaml unless overridden on the
command line. This module only orchestrates: it computes no metric, and each
step is a documented invocation of the repository's own code.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONFIG_PATH = os.path.join(HERE, "config_colab.yaml")


def load_config(path: str = CONFIG_PATH) -> Dict:
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _run(cmd: List[str], **kw) -> int:
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, **kw).returncode


def _py() -> str:
    return sys.executable


# ── Persistent storage ───────────────────────────────────────────────────────

def cmd_link(cfg: Dict, args) -> int:
    """Point <repo>/results at the durable campaign root.

    The repository's runners and its table generator both hardcode a
    "results/..." prefix, so rather than fork that path handling, `results`
    becomes a symlink to the campaign root. The runners write
    "results/logs/comparison/..." unchanged; the bytes land in
    "<campaign root>/logs/comparison/...", which is exactly the relative path
    expected_results/ mirrors.
    """
    p = cfg["persistence"]
    drive_root = args.drive_root or p["drive_root"]
    campaign_dir = p.get("campaign_dir", "campaign_results")
    target = os.path.join(drive_root, campaign_dir)
    mounted = os.path.isdir(os.path.dirname(drive_root.rstrip("/")))

    if args.local or not mounted:
        target = args.target or p["fallback_root"]
        print(f"[link] Drive not mounted (or --local given).")
        print(f"[link] Using SESSION-LOCAL storage: {target}")
        print(f"[link] *** Everything here is LOST when the session ends. ***")
        print(f"[link] Mount Drive and re-run `link` to make the campaign "
              f"survive a disconnect.")
    else:
        print(f"[link] Drive campaign root: {target}")

    os.makedirs(target, exist_ok=True)
    for sub in ("logs", "tables", "figures", "statistics",
                 "_campaign", "_campaign_logs"):
        os.makedirs(os.path.join(target, sub), exist_ok=True)

    results = os.path.join(ROOT, "results")
    if os.path.islink(results):
        current = os.readlink(results)
        if os.path.abspath(current) == os.path.abspath(target):
            print(f"[link] results -> {current} (already correct)")
            return 0
        print(f"[link] results currently -> {current}; repointing")
        os.unlink(results)
    elif os.path.isdir(results):
        # A real directory with content must never be silently replaced.
        entries = [e for e in os.listdir(results) if not e.startswith(".")]
        if entries:
            print(f"[link] REFUSING to replace the real directory "
                  f"{results} - it holds {len(entries)} entr(y/ies):")
            for e in sorted(entries)[:12]:
                print(f"         {e}")
            print("[link] Move or merge it into the campaign root first, e.g.")
            print(f"[link]   cp -a {results}/. {target}/")
            print(f"[link]   rm -rf {results}")
            print(f"[link] then re-run: python colab/run_campaign.py link")
            return 1
        os.rmdir(results)

    os.symlink(target, results, target_is_directory=True)
    print(f"[link] results -> {os.readlink(results)}")

    probe = os.path.join(results, ".link_probe")
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        assert os.path.exists(os.path.join(target, ".link_probe"))
        os.unlink(probe)
        print("[link] write-through verified")
    except Exception as exc:                                  # noqa: BLE001
        print(f"[link] WRITE TEST FAILED: {exc}")
        return 1
    return 0


# ── Steps ────────────────────────────────────────────────────────────────────

def cmd_smoke(cfg: Dict, args) -> int:
    s = cfg["smoke_test"]
    return _run([_py(), "colab/smoke_test.py",
                 "--rounds", str(args.rounds or s["rounds"]),
                 "--clients", str(s["clients"]),
                 "--seed", str(s["seed"]),
                 "--attack", s["attack"]])


def cmd_benchmark(cfg: Dict, args) -> int:
    b = cfg["benchmark"]
    e = cfg["execution"]
    cmd = [_py(), "colab/benchmark_concurrency.py",
           "--rounds", str(args.rounds or b["rounds_per_job"]),
           "--modes", *[str(m) for m in b["modes"]],
           "--client-cpus", str(e["sim_client_cpus"]),
           "--client-gpus", str(e["sim_client_gpus"])]
    rc = _run(cmd)
    bench = os.path.join(ROOT, "results", "_campaign_logs",
                         "colab_bench", "benchmark.json")
    if os.path.exists(bench):
        try:
            with open(bench, encoding="utf-8") as fh:
                rec = json.load(fh).get("recommended_max_parallel_jobs")
            print(f"\n[benchmark] recommended max_parallel_jobs = {rec}")
            print(f"[benchmark] set execution.max_parallel_jobs in "
                  f"colab/config_colab.yaml to that value, or pass "
                  f"--workers {rec} to `run`.")
        except (OSError, json.JSONDecodeError):
            pass
    return rc


def cmd_run(cfg: Dict, args) -> int:
    e = cfg["execution"]
    workers = args.workers or e["max_parallel_jobs"]
    prios = args.priority or cfg["priority"]["run"]
    cmd = [_py(), "colab/parallel_runner.py",
           "--results-root", "results",
           "--workers", str(workers),
           "--max-attempts", str(e["max_attempts"]),
           "--client-cpus", str(e["sim_client_cpus"]),
           "--client-gpus", str(e["sim_client_gpus"])]
    for p in prios:
        cmd += ["--priority", str(p)]
    for ph in (args.phase or []):
        cmd += ["--phase", ph]
    if args.retry_failed:
        cmd.append("--retry-failed")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.skip_aggregation:
        cmd.append("--skip-aggregation")
    return _run(cmd)


def cmd_status(cfg: Dict, args) -> int:
    return _run([_py(), "colab/parallel_runner.py",
                 "--results-root", "results", "--status"])


def cmd_finalize(cfg: Dict, args) -> int:
    return _run([_py(), "colab/finalize_campaign.py",
                 "--results-root", "results"])


def cmd_validate(cfg: Dict, args) -> int:
    v = cfg["validation"]
    cmd = [_py(), "colab/validate_campaign.py", "--results-root", "results",
           "--json", "results/_campaign/validation_report.json"]
    if args.strict or v.get("strict"):
        cmd.append("--strict")
    if v.get("sample_rounds") and not args.full_rounds:
        cmd.append("--sample-rounds")
    for p in (args.priority or []):
        cmd += ["--priority", str(p)]
    for ph in (args.phase or []):
        cmd += ["--phase", ph]
    return _run(cmd)


def cmd_report(cfg: Dict, args) -> int:
    cmd = [_py(), "colab/validate_campaign.py", "--results-root", "results",
           "--table", "--sample-rounds"]
    for ph in (args.phase or []):
        cmd += ["--phase", ph]
    return _run(cmd)


def cmd_artifacts(cfg: Dict, args) -> int:
    """Regenerate manuscript tables and figure bodies from real artifacts.

    Both generators SKIP anything whose backing artifact is absent rather than
    inventing it, so running this early is safe: it simply produces less.
    """
    rc = 0
    rc |= _run([_py(), "scripts/generate_manuscript_tables.py"])
    rc |= _run([_py(), "scripts/generate_manuscript_figures.py",
                "--results-root", "results"])
    print("\n[artifacts] coverage check (non-zero exit = something unbacked):")
    _run([_py(), "scripts/generate_manuscript_tables.py", "--check"])
    _run([_py(), "scripts/generate_manuscript_figures.py",
          "--results-root", "results", "--check"])
    return rc


def cmd_all(cfg: Dict, args) -> int:
    """link -> smoke -> run -> finalize -> validate, stopping on failure."""
    for name, fn in (("link", cmd_link), ("smoke", cmd_smoke),
                     ("run", cmd_run), ("finalize", cmd_finalize),
                     ("validate", cmd_validate)):
        print("\n" + "#" * 74)
        print(f"# {name}")
        print("#" * 74)
        rc = fn(cfg, args)
        if rc != 0 and name in ("link", "smoke"):
            print(f"\n[all] {name} failed (rc={rc}); stopping. "
                  f"The campaign is not started.")
            return rc
        if rc != 0:
            print(f"\n[all] {name} returned {rc}; continuing so that the "
                  f"report reflects reality.")
    return 0


COMMANDS = {
    "link": cmd_link, "smoke": cmd_smoke, "benchmark": cmd_benchmark,
    "run": cmd_run, "status": cmd_status, "finalize": cmd_finalize,
    "validate": cmd_validate, "report": cmd_report,
    "artifacts": cmd_artifacts, "all": cmd_all,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--priority", type=int, action="append", default=None)
    ap.add_argument("--phase", action="append", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-aggregation", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--full-rounds", action="store_true",
                    help="validate every round of every cell, not a sample")
    ap.add_argument("--local", action="store_true",
                    help="link: use session-local storage even if Drive is up")
    ap.add_argument("--drive-root", default=None)
    ap.add_argument("--target", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    return COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
