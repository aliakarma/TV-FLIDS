"""
scripts/generate_manuscript_tables.py
Results artifacts -> the manuscript's table bodies (Paper/tables/tab_*.tex).

This is the table-side counterpart of scripts/generate_manuscript_figures.py and
follows the same two rules:

  * A table whose backing artifact is missing is SKIPPED with a message. No file
    is written, so the manuscript keeps its provisional, explicitly-labelled
    rows and an invented number cannot reach the paper by accident.
  * Nothing is computed that the runners did not already produce. Means and
    standard deviations come from the per-seed values stored in each result
    JSON; this module only formats them and marks the best cell per column.

Each emitted file contains the table's data rows only (everything that would
sit between \\midrule and \\bottomrule), so the manuscript wraps them as:

    \\IfFileExists{tables/tab_main.tex}{\\input{tables/tab_main.tex}}{<provisional>}

Usage:
    python scripts/generate_manuscript_tables.py
    python scripts/generate_manuscript_tables.py --check
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TABLES_IN = os.path.join(ROOT, "results", "tables")
LOGS_IN = os.path.join(ROOT, "results", "logs")
OUT_DIR = os.path.join(ROOT, "Paper", "tables")

ACC, F1, ASR = "final_accuracy", "final_f1_macro", "final_attack_success_rate"
METRICS = (ACC, F1, ASR)
# Higher is better for accuracy and macro-F1; lower is better for ASR.
HIGHER_IS_BETTER = {ACC: True, F1: True, ASR: False}

STRATEGY_LABEL = {
    "fedavg": "FedAvg (no defense)",
    "krum": "Krum",
    "trimmed_mean": "Trimmed Mean",
    "fltrust": "FLTrust",
    "foolsgold": "FoolsGold",
    "flame": "FLAME",
    "rfa": "RFA",
    "bucketing": "Bucketing",
    "deepsight": "DeepSight (reduced fidelity)",
    "tvflids": "\\textbf{TV-FLIDS (ours)}",
    "tvflids_fixed": "TV-FLIDS (fixed weights)",
}
MAIN_ORDER = ["fedavg", "krum", "trimmed_mean", "fltrust",
              "foolsgold", "flame", "rfa"]

ATTACK_LABEL = {
    "no_attack": "No attack (clean)",
    "label_flip_30": "Label Flip (LF$_{30}$)",
    "gradient_scale_30": "Gradient Scale (GS$_{10}$)",
    "noise_30": "Noise Injection (NI$_{0.5}$)",
    "backdoor_20": "Backdoor (BD$_{0.1}$)",
    "ack1_evasion_30": "ACK1",
    "ack2_coalition_30": "ACK2",
}

_written: List[str] = []
_skipped: List[Tuple[str, str]] = []


# ── small helpers ────────────────────────────────────────────────────────

def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _mean_std(seed_results: Sequence[dict], metric: str) -> Optional[Tuple[float, float]]:
    vals = [r.get(metric) for r in seed_results if r.get(metric) is not None]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


def _cell(ms: Optional[Tuple[float, float]], best: bool = False) -> str:
    if ms is None:
        return "--"
    mean, std = ms
    m = f"{mean:.4f}"
    if best:
        m = f"\\best{{{m}}}"
    return f"{m} $\\pm$ {std:.4f}"


def _best_index(values: List[Optional[Tuple[float, float]]], metric: str) -> Optional[int]:
    idxs = [i for i, v in enumerate(values) if v is not None]
    if not idxs:
        return None
    key = (lambda i: values[i][0]) if HIGHER_IS_BETTER[metric] else (lambda i: -values[i][0])
    return max(idxs, key=key)


def _emit(name: str, lines: List[str]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"tab_{name}.tex")
    header = [
        "% GENERATED FILE - do not edit by hand.",
        "% Produced by scripts/generate_manuscript_tables.py from the result",
        "% artifacts named in its log output. Regenerate rather than editing.",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(header + lines) + "\n")
    _written.append(f"tab_{name}.tex")
    print(f"  -> wrote Paper/tables/tab_{name}.tex ({len(lines)} row(s))")


def _skip(name: str, why: str) -> None:
    _skipped.append((name, why))
    print(f"  [skip {name}] {why}")
    print("  -> NOT generated; manuscript keeps its provisional rows")


# ── comparison-style tables (Strategy | Acc | F1 | ASR) ──────────────────

def _comparison_rows(raw: Dict[str, List[dict]], order: List[str],
                     tail: List[str]) -> List[str]:
    """Rows for a strategy-comparison table, with per-column best in bold.

    `order` are the rows above the final rule, `tail` the rows below it.
    """
    present = [s for s in order + tail if raw.get(s)]
    stats = {s: {m: _mean_std(raw[s], m) for m in METRICS} for s in present}
    best = {m: _best_index([stats[s][m] for s in present], m) for m in METRICS}
    best_strategy = {m: (present[i] if i is not None else None)
                     for m, i in best.items()}

    lines: List[str] = []
    for group, strategies in (("head", [s for s in order if s in present]),
                              ("tail", [s for s in tail if s in present])):
        if group == "tail" and strategies and lines:
            lines.append("\\midrule")
        for s in strategies:
            cells = " & ".join(
                _cell(stats[s][m], best_strategy[m] == s) for m in METRICS)
            lines.append(f"{STRATEGY_LABEL.get(s, s)} & {cells} \\\\")
    return lines


def table_main() -> None:
    print("\nTable V (main comparison)")
    data = _load(os.path.join(TABLES_IN, "full_comparison_results.json"))
    if data is None:
        _skip("main", "missing results/tables/full_comparison_results.json")
        return
    raw = data.get("raw", {})
    if not raw.get("tvflids"):
        _skip("main", "artifact has no tvflids rows")
        return
    _emit("main", _comparison_rows(raw, MAIN_ORDER, ["tvflids"]))


def table_leakage_free() -> None:
    print("\nTable VI (leakage-free protocol)")
    path = os.path.join(TABLES_IN,
                        "full_comparison_results_nslkdd_leakage_free.json")
    data = _load(path)
    if data is None:
        _skip("leakage_free", f"missing {os.path.relpath(path, ROOT)}")
        return
    raw = data.get("raw", {})
    if not raw.get("tvflids"):
        _skip("leakage_free", "artifact has no tvflids rows")
        return
    _emit("leakage_free", _comparison_rows(raw, MAIN_ORDER, ["tvflids"]))


def table_recent_baselines() -> None:
    """Supp. Table S3: the seven main baselines plus Bucketing and DeepSight."""
    print("\nSupp. Table S3 (Bucketing / DeepSight)")
    main = _load(os.path.join(TABLES_IN, "full_comparison_results.json"))
    if main is None:
        _skip("recent_baselines", "missing full_comparison_results.json")
        return
    raw = dict(main.get("raw", {}))
    extra = _load(os.path.join(TABLES_IN, "_extra_baselines",
                               "full_comparison_results.json"))
    if extra is not None:
        raw.update(extra.get("raw", {}))
    if not (raw.get("bucketing") and raw.get("deepsight")):
        _skip("recent_baselines",
              "no bucketing/deepsight rows (run the extra-baselines comparison)")
        return
    _emit("recent_baselines",
          _comparison_rows(raw, MAIN_ORDER, ["bucketing", "deepsight", "tvflids"]))


# ── ablation ────────────────────────────────────────────────────────────

def table_ablation() -> None:
    print("\nTable VII (ablation A1-A6)")
    data = _load(os.path.join(TABLES_IN, "ablation_results.json"))
    if data is None:
        _skip("ablation", "missing results/tables/ablation_results.json")
        return
    raw = data.get("raw", data)
    if not isinstance(raw, dict) or not raw:
        _skip("ablation", "artifact carries no per-arm results")
        return

    full_key = next((k for k in raw if k.lower().startswith("tv-flids")), None)
    if full_key is None:
        _skip("ablation", "no full-system arm in artifact")
        return
    full_acc = _mean_std(raw[full_key], ACC)

    order = [full_key] + sorted(k for k in raw if k != full_key)
    lines: List[str] = []
    for arm in order:
        seed_results = raw[arm]
        acc, f1, asr = (_mean_std(seed_results, m) for m in METRICS)
        delta = "--"
        if acc is not None and full_acc is not None and arm != full_key:
            delta = f"{(acc[0] - full_acc[0]) * 100:+.2f}"
        label = arm.replace("_", " ")
        if arm == full_key:
            label = "\\textbf{TV-FLIDS (full)}"
            delta = "--"
        lines.append(f"{label} & {_cell(acc)} & {_cell(f1)} & {_cell(asr)} "
                     f"& {delta} \\\\")
        if arm == full_key:
            lines.append("\\midrule")
    _emit("ablation", lines)


# ── non-IID sweep ───────────────────────────────────────────────────────

def table_noniid() -> None:
    print("\nTable VIII (non-IID concentration sweep)")
    data = _load(os.path.join(TABLES_IN, "noniid_sweep_results.json"))
    if data is None:
        _skip("noniid_sweep", "missing results/tables/noniid_sweep_results.json")
        return
    summary = data.get("summary", {})
    if not summary:
        _skip("noniid_sweep", "artifact carries no summary")
        return

    conditions: List[str] = []
    for by_cond in summary.values():
        for label in by_cond:
            if label not in conditions:
                conditions.append(label)

    def _sort_key(label: str) -> float:
        try:
            return float(label.split("=")[-1])
        except ValueError:
            return float("inf")     # "iid" sorts last

    conditions.sort(key=_sort_key)

    rows: List[Tuple[str, str, str]] = []
    if "krum" in summary:
        rows.append(("Krum (Acc)", "krum", ACC))
    for label, metric in (("TV-FLIDS (Acc)", ACC), ("TV-FLIDS (F1)", F1),
                          ("TV-FLIDS (ASR)", ASR)):
        if "tvflids" in summary:
            rows.append((label, "tvflids", metric))

    lines = []
    for label, strategy, metric in rows:
        cells = []
        for cond in conditions:
            cell = summary.get(strategy, {}).get(cond, {}).get(metric)
            cells.append("--" if cell is None
                         else f"{cell['mean']:.4f} $\\pm$ {cell['std']:.4f}")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    if not lines:
        _skip("noniid_sweep", "neither krum nor tvflids present in summary")
        return
    print(f"  conditions: {conditions}")
    _emit("noniid_sweep", lines)


# ── attack matrix ───────────────────────────────────────────────────────

def _matrix_artifacts() -> Dict[str, Dict[str, List[dict]]]:
    """Merge every multi-attack matrix artifact into {strategy: {attack: [..]}}."""
    merged: Dict[str, Dict[str, List[dict]]] = {}
    for path in sorted(glob.glob(os.path.join(
            TABLES_IN, "multi_attack_matrix_results*.json"))):
        blob = _load(path)
        if blob is None:
            continue
        for strategy, by_attack in (blob.get("raw") or {}).items():
            merged.setdefault(strategy, {}).update(by_attack)
    # The label-flip column lives in the main comparison artifact.
    main = _load(os.path.join(TABLES_IN, "full_comparison_results.json"))
    if main is not None:
        for strategy, seed_results in (main.get("raw") or {}).items():
            merged.setdefault(strategy, {}).setdefault(
                "label_flip_30", seed_results)
    return merged


def table_attacks() -> None:
    """Table IX: TV-FLIDS across attack types."""
    print("\nTable IX (TV-FLIDS under four attack types)")
    merged = _matrix_artifacts()
    tv = merged.get("tvflids", {})
    wanted = ["no_attack", "label_flip_30", "gradient_scale_30",
              "noise_30", "backdoor_20"]
    have = [a for a in wanted if tv.get(a)]
    if len(have) < 2:
        _skip("attacks", f"only {have} available for tvflids")
        return
    missing = [a for a in wanted if a not in have]
    if missing:
        print(f"  note: {missing} not yet available; rows omitted, not invented")
    lines = []
    for attack in have:
        cells = " & ".join(_cell(_mean_std(tv[attack], m)) for m in METRICS)
        lines.append(f"{ATTACK_LABEL.get(attack, attack)} & {cells} \\\\")
    _emit("attacks", lines)


def table_multiattack_baselines() -> None:
    """Table X: every strategy under GS / NI / BD."""
    print("\nTable X (baselines under GS / NI / BD)")
    merged = _matrix_artifacts()
    blocks = [("gradient_scale_30", "Gradient Scale (GS$_{10}$)"),
              ("noise_30", "Noise Injection (NI$_{0.5}$)"),
              ("backdoor_20", "Backdoor (BD$_{0.1}$)")]
    order = MAIN_ORDER + ["tvflids"]
    available = [(a, t) for a, t in blocks
                 if any(merged.get(s, {}).get(a) for s in order)]
    if not available:
        _skip("multiattack_baselines",
              "no gradient_scale/noise/backdoor cells on disk")
        return

    lines: List[str] = []
    for i, (attack, title) in enumerate(available):
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        present = [s for s in order if merged.get(s, {}).get(attack)]
        stats = {s: {m: _mean_std(merged[s][attack], m) for m in METRICS}
                 for s in present}
        best = {m: _best_index([stats[s][m] for s in present], m)
                for m in METRICS}
        best_strategy = {m: (present[j] if j is not None else None)
                         for m, j in best.items()}
        for s in present:
            cells = " & ".join(
                _cell(stats[s][m], best_strategy[m] == s) for m in METRICS)
            lines.append(f"{STRATEGY_LABEL.get(s, s)} & {cells} \\\\")
    _emit("multiattack_baselines", lines)


def table_adaptive() -> None:
    """Table XI: ACK1 / ACK2, with the non-adaptive LF30 rows for reference."""
    print("\nTable XI (adaptive attacks ACK1 / ACK2)")
    merged = _matrix_artifacts()
    adaptive = ["ack1_evasion_30", "ack2_coalition_30"]
    if not any(merged.get(s, {}).get(a)
               for s in ("fltrust", "tvflids") for a in adaptive):
        _skip("adaptive", "no ACK1/ACK2 cells on disk")
        return

    lines: List[str] = []
    ref = [s for s in ("fltrust", "tvflids")
           if merged.get(s, {}).get("label_flip_30")]
    for s in ref:
        cells = " & ".join(
            _cell(_mean_std(merged[s]["label_flip_30"], m)) for m in METRICS)
        lines.append(f"LF$_{{30}}$ (non-adaptive) & "
                     f"{STRATEGY_LABEL.get(s, s)} & {cells} \\\\")

    for attack in adaptive:
        present = [s for s in ("fedavg", "fltrust", "tvflids")
                   if merged.get(s, {}).get(attack)]
        if not present:
            continue
        lines.append("\\midrule")
        for s in present:
            cells = " & ".join(
                _cell(_mean_std(merged[s][attack], m)) for m in METRICS)
            lines.append(f"{ATTACK_LABEL[attack]} & "
                         f"{STRATEGY_LABEL.get(s, s)} & {cells} \\\\")
    _emit("adaptive", lines)


# ── hyperparameter sweeps ───────────────────────────────────────────────

def table_baseline_hp() -> None:
    print("\nSupp. Table S1 (baseline hyperparameter sweep)")
    data = _load(os.path.join(TABLES_IN,
                              "hyperparam_sweep_baseline_results.json"))
    if data is None:
        _skip("baseline_hp", "missing hyperparam_sweep_baseline_results.json")
        return
    summary = data.get("summary", {})
    sweeps = (data.get("provenance") or {}).get("sweeps", {})
    if not summary:
        _skip("baseline_hp", "artifact carries no summary")
        return

    titles = {
        "krum_f_prime": ("\\textbf{Krum} (assumed tolerance $f'$)",
                         lambda v: f"$f'={v}$"),
        "trimmed_mean_beta": ("\\textbf{Trimmed Mean} (trim fraction "
                              "$\\beta_{\\mathrm{TM}}$)",
                              lambda v: f"$\\beta_{{\\mathrm{{TM}}}}={v}$"),
    }
    lines: List[str] = []
    for i, (name, by_value) in enumerate(summary.items()):
        title, fmt = titles.get(name, (name.replace("_", " "), lambda v: str(v)))
        default = str((sweeps.get(name) or {}).get("default", ""))
        if i:
            lines.append("\\midrule")
        lines.append(f"{title} & \\textbf{{Accuracy}} \\\\")
        lines.append("\\midrule")
        for label, cell in by_value.items():
            mark = " $^{\\ast}$" if label == default else ""
            acc = cell.get(ACC, {})
            lines.append(f"{fmt(label)}{mark} & "
                         f"{acc.get('mean', float('nan')):.4f} $\\pm$ "
                         f"{acc.get('std', float('nan')):.4f} \\\\")
    _emit("baseline_hp", lines)


def table_tvflids_hp() -> None:
    print("\nSupp. Table S2 (TV-FLIDS hyperparameter sweep)")
    data = _load(os.path.join(TABLES_IN,
                              "hyperparam_sweep_tvflids_results.json"))
    if data is None:
        _skip("tvflids_hp", "missing hyperparam_sweep_tvflids_results.json")
        return
    summary = data.get("summary", {})
    sweeps = (data.get("provenance") or {}).get("sweeps", {})
    if not summary:
        _skip("tvflids_hp", "artifact carries no summary")
        return

    titles = {
        "memory_decay": "Memory decay $\\lambda$",
        "min_trust": "Trust floor $\\tau_{\\min}$",
        "meta_lr": "Meta-LR $\\eta_{\\mathrm{meta}}$",
        "warmup_rounds": "Warmup length $T_{\\mathrm{warm}}$",
    }
    lines: List[str] = []
    for i, (name, by_value) in enumerate(summary.items()):
        if i:
            lines.append("\\midrule")
        title = titles.get(name, name.replace("_", " "))
        default = str((sweeps.get(name) or {}).get("default", ""))
        values = list(by_value.items())
        lines.append(f"\\multirow{{{len(values)}}}{{*}}{{{title}}}")
        for label, cell in values:
            mark = " $^{\\ast}$" if label == default else ""
            acc, asr = cell.get(ACC, {}), cell.get(ASR, {})
            lines.append(
                f"& ${label}${mark} & "
                f"{acc.get('mean', float('nan')):.4f} $\\pm$ "
                f"{acc.get('std', float('nan')):.4f} & "
                f"{asr.get('mean', float('nan')):.4f} $\\pm$ "
                f"{asr.get('std', float('nan')):.4f} \\\\")
    _emit("tvflids_hp", lines)


# ── overhead ────────────────────────────────────────────────────────────

STAGE_LABEL = [
    ("client_processing", "\quad Client update processing"),
    ("verification", "\quad Verification gate (Checks 1--3)"),
    ("trust_scoring", "\quad Trust scoring"),
    ("meta_gradient", "\quad Meta-gradient update"),
    ("aggregation", "\quad Trust-weighted aggregation"),
]


def _overhead_samples(strategy: str, key: str) -> List[dict]:
    """Every compute_overhead_ms block on disk for a strategy, main protocol."""
    out = []
    pattern = os.path.join(LOGS_IN, "comparison",
                           f"{strategy}_label_flip_30_seed*",
                           "experiment_log.json")
    for path in sorted(glob.glob(pattern)):
        blob = _load(path)
        if blob is None:
            continue
        ov = (blob.get("summary") or {}).get("compute_overhead_ms")
        if ov and any(k.startswith(key) for k in ov):
            out.append(ov)
    return out


def _avg(samples: List[dict], field: str) -> Optional[float]:
    vals = [s[field] for s in samples if field in s]
    return float(np.mean(vals)) if vals else None


def table_overhead() -> None:
    """Table XII, decomposed exactly as the instrumentation measures it.

    The manuscript's historical layout folded the server's own aggregation
    cost into a single "FedAvg aggregation (baseline)" row and listed only
    three TV-FLIDS additions. What the released instrumentation actually
    measures is: one per-client local-training time (reported by the clients),
    and five server stages inside TVFLIDSStrategy.aggregate_fit, against
    FedAvgStrategy's own aggregate_fit as the baseline. The rows below are
    those measurements and nothing else, so every cell traces to a timing the
    code took.
    """
    print("\nTable XII (per-round overhead, measured on this hardware)")
    tv = _overhead_samples("tvflids", "total_mean_ms")
    fa = _overhead_samples("fedavg", "fedavg_total_mean_ms")
    if not tv:
        _skip("overhead", "no TV-FLIDS runs with compute_overhead_ms on disk")
        return

    server_total = _avg(tv, "total_mean_ms")
    if not server_total:
        _skip("overhead", "TV-FLIDS runs carry no total_mean_ms")
        return

    client_train = _avg(tv, "client_training_mean_ms")
    if client_train is None:
        _skip("overhead",
              "TV-FLIDS runs carry no client_training_mean_ms - rerun the "
              "comparison against the instrumented client")
        return

    tv_total = client_train + server_total
    lines: List[str] = [
        f"Client local training (mean per client) & {client_train:.1f} & "
        f"{100.0 * client_train / tv_total:.1f} \\\\",
        "\midrule",
        "\textit{TV-FLIDS server stages:} & & \\\\",
    ]
    for key, label in STAGE_LABEL:
        ms = _avg(tv, f"{key}_mean_ms")
        if ms is None:
            continue
        lines.append(f"{label} & {ms:.1f} & {100.0 * ms / tv_total:.1f} \\\\")
    lines.append("\midrule")
    lines.append(f"\textbf{{TV-FLIDS total}} & \textbf{{{tv_total:.1f}}} & "
                 f"\textbf{{100.0}} \\\\")

    fa_server = _avg(fa, "fedavg_total_mean_ms")
    fa_client = _avg(fa, "client_training_mean_ms")
    if fa_server is not None and fa_client is not None:
        fa_total = fa_client + fa_server
        delta = tv_total - fa_total
        lines.append(f"\quad FedAvg server aggregation & {fa_server:.1f} & n/a \\\\")
        lines.append(f"\textbf{{FedAvg total}} & {fa_total:.1f} & n/a \\\\")
        lines.append(f"\textbf{{Overhead vs.\ FedAvg}} & "
                     f"{delta:+.1f} ({100.0 * delta / fa_total:+.1f}\%) & n/a \\\\")
        print(f"  TV-FLIDS {tv_total:.1f} ms vs FedAvg {fa_total:.1f} ms "
              f"-> {100.0 * delta / fa_total:+.1f}%  "
              f"(over {len(tv)} / {len(fa)} runs)")
    else:
        print("  note: no instrumented FedAvg runs; the relative overhead row "
              "is omitted rather than assumed")
    _emit("overhead", lines)


# ── CIC-IoT-2023 ────────────────────────────────────────────────────────

def table_ciciot() -> None:
    print("\nSupp. Table S4 (CIC-IoT-2023 cross-dataset)")
    data = _load(os.path.join(TABLES_IN, "dataset_comparison_results.json"))
    if data is None:
        _skip("ciciot", "missing results/tables/dataset_comparison_results.json")
        return
    block = (data.get("ciciot2023") or (data.get("raw") or {}).get("ciciot2023"))
    if not block:
        _skip("ciciot", "artifact has no ciciot2023 block")
        return
    order = [s for s in ["fedavg", "krum", "fltrust", "tvflids"] if block.get(s)]
    if not order:
        _skip("ciciot", "ciciot2023 block has no recognised strategies")
        return
    _emit("ciciot", _comparison_rows(
        {s: block[s] for s in order},
        [s for s in order if s != "tvflids"],
        ["tvflids"] if "tvflids" in order else []))


TABLES = [
    ("main", table_main),
    ("leakage_free", table_leakage_free),
    ("ablation", table_ablation),
    ("noniid_sweep", table_noniid),
    ("attacks", table_attacks),
    ("multiattack_baselines", table_multiattack_baselines),
    ("adaptive", table_adaptive),
    ("overhead", table_overhead),
    ("baseline_hp", table_baseline_hp),
    ("tvflids_hp", table_tvflids_hp),
    ("recent_baselines", table_recent_baselines),
    ("ciciot", table_ciciot),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report coverage without writing files; "
                         "exit 1 if any table is unbacked")
    args = ap.parse_args()

    print("=== Manuscript table regeneration ===")
    print(f"  results root : {os.path.join(ROOT, 'results')}")
    print(f"  output dir   : {OUT_DIR}")
    print(f"  mode         : {'CHECK ONLY (no files written)' if args.check else 'WRITE'}")

    global _emit
    if args.check:
        real_emit = _emit

        def _dry(name: str, lines: List[str]) -> None:  # noqa: ANN001
            _written.append(f"tab_{name}.tex")
            print(f"  -> would write Paper/tables/tab_{name}.tex "
                  f"({len(lines)} row(s))")
        _emit = _dry  # type: ignore[assignment]

    for _, fn in TABLES:
        fn()

    print("\n=== Summary ===")
    print(f"  generated : {', '.join(_written) if _written else '(none)'}")
    print(f"  pending   : "
          f"{', '.join(n for n, _ in _skipped) if _skipped else '(none)'}")
    if _skipped:
        print("\n  Pending tables need their campaign phase to be run. Nothing "
              "was invented for them; the manuscript continues to compile "
              "against its provisional, explicitly-labelled rows.")
    return 1 if (args.check and _skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
