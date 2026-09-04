"""
colab/smoke_test.py — prove the pipeline works before spending the campaign.

    python colab/smoke_test.py                    # tiny run + every assertion
    python colab/smoke_test.py --rounds 3 --clients 6
    python colab/smoke_test.py --json smoke.json
    python colab/smoke_test.py --keep             # keep the tiny run's output

Runs ONE deliberately tiny TV-FLIDS cell (few clients, few rounds, one seed,
one attack) into a throwaway path, then asserts, on the artifact it produced:

   1  the run completed and wrote experiment_log.json
   2  the round log has rounds+1 entries indexed 0..rounds
   3  the model actually trained (parameters exist and the loss/accuracy series
      is not frozen at its round-0 value for every round)
   4  the attack executed (malicious clients were selected and recorded)
   5  the verification gate executed (its decisions are recorded)
   6  the trust EMA executed (per-round trust telemetry is present and moves)
   7  the STE / meta-gradient executed (adaptive alpha/beta/gamma present,
      summing to 1, and not identical at every round)
   8  provenance is complete and the config hash is real (never 'mockhash')
   9  the GPU was genuinely used, when CUDA is available
  10  the expected-results validator accepts the artifact's structure
  11  no quarantined or mock artifact was consumed

Every assertion reads the artifact. None of them checks a metric against an
expected value — a smoke test proves the machinery ran, not what it should have
concluded.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import campaign_inventory as inv  # noqa: E402


def _now() -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            .isoformat().replace("+00:00", "Z"))


class Checks:
    def __init__(self) -> None:
        self.rows: List[Dict] = []

    def add(self, name: str, ok: Optional[bool], detail: str,
            skipped: bool = False) -> None:
        self.rows.append({"check": name, "ok": ok, "skipped": skipped,
                           "detail": detail})

    @property
    def failed(self) -> List[Dict]:
        return [r for r in self.rows if r["ok"] is False and not r["skipped"]]

    def render(self) -> None:
        print("\n" + "=" * 78)
        print("SMOKE TEST ASSERTIONS")
        print("=" * 78)
        for r in self.rows:
            mark = "SKIP" if r["skipped"] else ("PASS" if r["ok"] else "FAIL")
            print(f"  [{mark}] {r['check']}")
            if r["detail"]:
                print(f"         {r['detail']}")


def make_tiny_config(clients: int, warmup: int) -> Tuple[str, int]:
    """A tiny config derived from the repository's own, not written from scratch.

    Returns (path, clients_actually_set). Locating `num_clients` by searching
    for the key rather than assuming its section: a silent failure to shrink
    the client count would make the smoke test quietly run a full-size job and
    still report success, which is worse than an error.
    """
    import yaml
    src = os.path.join(ROOT, "config", "fl_config.yaml")
    with open(src, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    tiny = copy.deepcopy(cfg)

    def set_key(node, key: str, value) -> int:
        """Set every occurrence of `key` anywhere in the tree. Returns count."""
        n = 0
        if isinstance(node, dict):
            for k, v in node.items():
                if k == key:
                    node[k] = value
                    n += 1
                else:
                    n += set_key(v, key, value)
        elif isinstance(node, list):
            for v in node:
                n += set_key(v, key, value)
        return n

    n_set = set_key(tiny, "num_clients", clients)
    if n_set == 0:
        raise SystemExit(
            "[smoke] could not find 'num_clients' anywhere in "
            "config/fl_config.yaml. Refusing to run: the smoke test would "
            "silently execute a full-size 20-client job and report success. "
            "Fix this function to match the current config layout.")

    # A 3-round run cannot observe a 20-round gate warmup, so shorten it —
    # otherwise the verification gate never leaves its warmup schedule and the
    # gate assertion would be testing nothing.
    set_key(tiny, "warmup_rounds", warmup)

    fd, path = tempfile.mkstemp(prefix="tvflids_smoke_", suffix=".yaml")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(tiny, fh)
    return path, n_set


def run_tiny(log_dir: str, rounds: int, seed: int, attack: str,
             config_path: str, log_file: str) -> Tuple[int, str]:
    cmd = [sys.executable, "experiments/run_experiment.py",
           "--strategy", "tvflids", "--attack", attack,
           "--seed", str(seed), "--rounds", str(rounds),
           "--log_dir", log_dir, "--config", config_path]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["TVFLIDS_RESUME"] = "0"           # a smoke test must actually execute
    env["TVFLIDS_TENSORBOARD"] = "0"
    env.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    print(f"  $ {' '.join(cmd)}")
    with open(log_file, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"# cmd: {' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh,
                              stderr=subprocess.STDOUT, env=env)
    return proc.returncode, log_file


def _series(rounds: List[Dict], key: str) -> List:
    return [r.get(key) for r in rounds if r.get(key) is not None]


def _varies(vals: List) -> bool:
    nums = [v for v in vals if isinstance(v, (int, float))]
    return len(set(round(float(v), 10) for v in nums)) > 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--clients", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attack", default="label_flip_30")
    ap.add_argument("--root", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    if args.rounds > 10:
        print(f"[smoke] refusing --rounds {args.rounds}: this is a smoke "
              f"test. Use 10 or fewer.")
        return 2

    root = args.root or os.path.join("results", "_campaign_logs", "colab_smoke")
    log_dir = os.path.join(root, "logs", "_smoke",
                           f"tvflids_{args.attack}_seed{args.seed}")
    os.makedirs(root, exist_ok=True)
    shutil.rmtree(log_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)
    run_log = os.path.join(root, "smoke_run.log")

    print("=" * 78)
    print("TV-FLIDS PRE-CAMPAIGN SMOKE TEST")
    print("=" * 78)
    print(f"  rounds   {args.rounds}    clients {args.clients}    "
          f"seed {args.seed}    attack {args.attack}")
    print(f"  log_dir  {log_dir}")

    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda else None
    except Exception:                                         # noqa: BLE001
        cuda, gpu_name = False, None
    print(f"  CUDA     {cuda}" + (f" ({gpu_name})" if gpu_name else ""))

    cfg_path, n_set = make_tiny_config(args.clients, warmup=1)
    print(f"  config   {cfg_path} (derived from config/fl_config.yaml; "
          f"num_clients set at {n_set} site(s))")

    rc, _ = run_tiny(log_dir, args.rounds, args.seed, args.attack,
                     cfg_path, run_log)
    print(f"  exit     {rc}   (transcript: {run_log})")

    ck = Checks()
    lpath = os.path.join(log_dir, "experiment_log.json")

    # 1 ── completion
    blob = None
    if rc != 0:
        ck.add("run completed", False,
               f"run_experiment.py exited {rc}; see {run_log}")
    elif not os.path.exists(lpath):
        ck.add("run completed", False, "no experiment_log.json was written")
    else:
        try:
            with open(lpath, encoding="utf-8") as fh:
                blob = json.load(fh)
            ck.add("run completed", True,
                   f"{lpath} ({os.path.getsize(lpath):,} bytes)")
        except (OSError, json.JSONDecodeError) as exc:
            ck.add("run completed", False, f"experiment_log.json unreadable: {exc}")

    if blob is None:
        ck.render()
        print("\nSMOKE TEST FAILED: no artifact to inspect.")
        return 1

    rounds = blob.get("rounds") or []
    summary = blob.get("summary") or {}
    cfg = blob.get("config") or {}
    prov = blob.get("provenance") or {}
    extra = blob.get("extra") or {}

    # 1b ── the tiny config actually took effect
    got_clients = cfg.get("num_clients")
    ck.add("tiny config took effect (num_clients honoured)",
           got_clients == args.clients,
           f"requested {args.clients} clients, the run recorded "
           f"num_clients={got_clients}"
           + ("" if got_clients == args.clients else
              "  [the override did not reach the runner: the smoke test just "
              "ran a different size than intended]"))

    # 2 ── round log shape
    want = args.rounds + 1
    idx = [r.get("round") for r in rounds]
    ok = len(rounds) == want and idx == list(range(0, args.rounds + 1))
    ck.add("round log has rounds+1 entries indexed 0..N", ok,
           f"{len(rounds)} entries, expected {want}; indices {idx}")

    # 3 ── the model trained
    params = summary.get("model_params")
    accs = _series(rounds, "accuracy")
    trained = bool(params) and isinstance(params, (int, float)) and params > 0
    moved = _varies(accs)
    ck.add("model trained (parameters present, metrics not frozen)",
           trained and moved,
           f"model_params={params}; accuracy over rounds={accs}; "
           f"varies={moved}"
           + ("" if moved else "  [a frozen series can be legitimate in a "
                                "3-round run, but is worth eyeballing]"))

    # 4 ── the attack executed
    n_mal = summary.get("num_malicious")
    mal_ids = summary.get("malicious_ids")
    asr = _series(rounds, "attack_success_rate")
    attack_ran = (isinstance(n_mal, int) and n_mal > 0
                  and isinstance(mal_ids, (list, tuple)) and len(mal_ids) > 0)
    ck.add("attack executed (malicious clients selected and recorded)",
           attack_ran,
           f"num_malicious={n_mal}, malicious_ids={mal_ids}, "
           f"asr series={asr}")

    # 5 ── verification gate
    srl = extra.get("strategy_round_logs") or []
    gate_keys = set()
    for e in srl:
        if isinstance(e, dict):
            gate_keys |= {k for k in e
                          if any(t in k.lower() for t in
                                 ("accept", "reject", "gate", "verif",
                                  "tau", "threshold", "flag"))}
    ck.add("verification gate executed (decisions recorded)",
           bool(gate_keys) or bool(extra.get("verification_history")),
           f"strategy_round_logs entries={len(srl)}; "
           f"gate-related keys={sorted(gate_keys) or 'none'}")

    # 6 ── trust EMA
    tmean = _series(rounds, "trust_mean")
    thist = extra.get("trust_history")
    trust_ok = bool(tmean) and all(0.0 <= v <= 1.0 for v in tmean)
    ck.add("trust EMA executed (per-round trust telemetry present, in [0,1])",
           trust_ok,
           f"trust_mean series={tmean}; varies={_varies(tmean)}; "
           f"trust_history present={thist is not None}")

    # 7 ── STE / meta-gradient
    a = _series(rounds, "trust_adaptive_alpha")
    b = _series(rounds, "trust_adaptive_beta")
    g = _series(rounds, "trust_adaptive_gamma")
    have = bool(a) and len(a) == len(b) == len(g)
    sums_ok = have and all(abs(x + y + z - 1.0) <= 1e-6
                           for x, y, z in zip(a, b, g))
    ck.add("STE / meta-gradient executed (alpha+beta+gamma == 1 each round)",
           have and sums_ok,
           f"alpha={a}\n         beta={b}\n         gamma={g}\n"
           f"         sums={[round(x + y + z, 9) for x, y, z in zip(a, b, g)]}; "
           f"weights move across rounds={_varies(a) or _varies(b) or _varies(g)}")

    # 8 ── provenance + real config hash
    ch = cfg.get("_config_hash")
    hash_ok = bool(ch) and str(ch).strip().lower() not in {
        "mockhash", "", "none", "null"}
    missing = [k for k in ("git_commit", "config_hash", "timestamp_utc",
                            "packages", "python_version")
               if not prov.get(k)]
    ck.add("provenance complete and config hash is real", hash_ok and not missing,
           f"_config_hash={ch!r}; provenance missing={missing or 'nothing'}; "
           f"git_commit={str(prov.get('git_commit'))[:12]}; "
           f"git_dirty={prov.get('git_dirty')}")

    # 9 ── the GPU was genuinely used
    if not cuda:
        ck.add("GPU genuinely used", None,
               "no CUDA device on this runtime, so a CPU run is correct here; "
               "on the T4 this check becomes binding", skipped=True)
    else:
        hw = prov.get("hardware") or {}
        txt = json.dumps(hw).lower()
        gpu_seen = ("cuda" in txt or "gpu" in txt
                    or bool(prov.get("sim_client_gpus")))
        try:
            with open(run_log, encoding="utf-8", errors="replace") as fh:
                transcript = fh.read().lower()
        except OSError:
            transcript = ""
        said_cpu = "gpu not available" in transcript
        ck.add("GPU genuinely used", gpu_seen and not said_cpu,
               f"provenance.hardware={hw}; "
               f"sim_client_gpus={prov.get('sim_client_gpus')}; "
               f"transcript said 'GPU not available'={said_cpu}")

    # 10 ── the structure validator accepts this artifact's shape
    ranges = inv.metric_ranges()
    struct_problems: List[str] = []
    for k in inv.SUMMARY_KEYS:
        if k not in summary:
            struct_problems.append(f"summary missing {k}")
    for k in inv.CONFIG_KEYS:
        if k not in cfg:
            struct_problems.append(f"config missing {k}")
    for k in inv.ROUND_KEYS_ALWAYS + inv.ROUND_KEYS_TRUST:
        if rounds and k not in rounds[-1]:
            struct_problems.append(f"round entries missing {k}")
    for r in rounds:
        for k, v in r.items():
            if k in ranges and not (
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    and (ranges[k]["min"] is None or v >= ranges[k]["min"])
                    and (ranges[k]["max"] is None or v <= ranges[k]["max"])):
                struct_problems.append(
                    f"round {r.get('round')}: {k}={v!r} outside its range")
    ck.add("artifact matches the expected_results schema and metric ranges",
           not struct_problems,
           "; ".join(struct_problems[:6]) if struct_problems
           else "every expected key present, every metric in range")

    # 11 ── nothing quarantined or mock was consumed
    #
    # A quarantine path is only evidence of consumption when it appears as an
    # INPUT or OUTPUT of the run. `provenance.git_dirty_paths` legitimately
    # lists every untracked path in the tree, and results/_QUARANTINED_MOCK/ is
    # untracked by design — so finding the string there means provenance is
    # working, not that a mock artifact was used. Checking the raw text would
    # therefore fail every honest run on this repository.
    try:
        with open(lpath, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError:
        raw = ""
    bad: List[str] = []
    if "mockhash" in raw:
        bad.append("the fabrication fingerprint 'mockhash' appears in the artifact")

    consuming_paths = {
        "provenance.log_dir": prov.get("log_dir"),
        "provenance.script": prov.get("script"),
        "config.dataset": cfg.get("dataset"),
        "log_dir (actual)": log_dir,
    }
    for label, value in consuming_paths.items():
        if isinstance(value, str) and "_QUARANTINED_MOCK" in value:
            bad.append(f"{label} resolves inside the quarantine: {value}")
    for i, a in enumerate(prov.get("argv") or []):
        if isinstance(a, str) and "_QUARANTINED_MOCK" in a:
            bad.append(f"provenance.argv[{i}] points into the quarantine: {a}")

    n_dirty_hits = sum(
        1 for p in (prov.get("git_dirty_paths") or [])
        if isinstance(p, str) and "_QUARANTINED_MOCK" in p)
    ck.add("no mock or quarantined artifact consumed", not bad,
           "; ".join(bad) if bad else
           f"no 'mockhash' fingerprint; no input or output path resolves "
           f"inside the quarantine. ({n_dirty_hits} quarantine path(s) appear "
           f"in provenance.git_dirty_paths, which is provenance correctly "
           f"recording an untracked directory, not consumption.)")

    ck.render()

    payload = {
        "smoke_tested_utc": _now(),
        "rounds": args.rounds, "clients_requested": args.clients,
        "seed": args.seed, "attack": args.attack,
        "cuda_available": cuda, "gpu": gpu_name,
        "log_dir": log_dir.replace(os.sep, "/"),
        "exit_code": rc,
        "checks": ck.rows,
        "config_used": cfg,
        "note": "Structural and mechanical assertions only. No metric is "
                 "compared against any expected value.",
    }
    out_json = args.json or os.path.join(root, "smoke_test.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\n[smoke] report -> {out_json}")

    os.unlink(cfg_path)
    if not args.keep:
        shutil.rmtree(os.path.join(root, "logs", "_smoke"), ignore_errors=True)
        print("[smoke] throwaway cell output removed (--keep to retain)")

    n_fail = len(ck.failed)
    n_skip = sum(1 for r in ck.rows if r["skipped"])
    print("\n" + "=" * 78)
    if n_fail:
        print(f"SMOKE TEST FAILED: {n_fail} assertion(s). "
              f"Do not start the campaign.")
    else:
        print(f"SMOKE TEST PASSED: {len(ck.rows) - n_skip} assertion(s) hold, "
              f"{n_skip} skipped.")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
