"""
scripts/generate_manuscript_figures.py
Regenerate the manuscript's data-driven figures from real experiment output.

WHY THIS EXISTS
---------------
The manuscript renders Figures 2-5 (and Supplementary Figure S1) as pgfplots
coordinate lists rather than \\includegraphics of a rendered image. That choice
is deliberate -- the figures then inherit the document's fonts, line widths and
column geometry -- but until now the coordinate lists were hand-authored, so
regenerating experimental results did not update the manuscript. This script
supplies the missing link: it reads the result artifacts a campaign produces
and writes the pgfplots bodies the manuscript inputs.

CONTRACT WITH THE MANUSCRIPT
----------------------------
For each figure the manuscript contains, inside its own ``axis`` environment:

    \\IfFileExists{figures/fig_<name>.tex}{%
        \\input{figures/fig_<name>.tex}%
    }{%
        <provisional hand-authored coordinates>
    }

So:
  * if this script has written the file, the real curve is used;
  * if it has not, the manuscript still compiles against the provisional
    coordinates, which the caption and Section IX/VII text mark as
    pending regeneration.

This script NEVER fabricates data. A figure whose backing artifact is missing
or unreadable is skipped with a message, and no .tex file is written for it,
so a stale or invented curve can never silently reach the manuscript.

USAGE
-----
    make manuscript-figures            # after a campaign has produced results
    python scripts/generate_manuscript_figures.py [--results-root results]
                                                  [--out-dir Paper/figures]
                                                  [--check]

``--check`` reports what could and could not be generated and exits non-zero if
anything is missing, without writing files. Useful in CI and before a
submission build.

INPUT ARTIFACTS
---------------
  Figure 2 (convergence)  results/logs/comparison/<strategy>_<attack>_seed<S>/
                          experiment_log.json  ->  rounds[].accuracy
                          Produced by: make full-comparison
  Figure 3 (trust)        the TV-FLIDS log above -> extra.trust_history,
                          extra.malicious_ids
                          Produced by: make full-comparison
  Figure 4 (weights)      the TV-FLIDS log above -> extra.strategy_round_logs[]
                          .adaptive_{alpha,beta,gamma}
                          Produced by: make full-comparison
  Figure 5 (robustness)   results/tables/ratio_sweep_results.json
                          Produced by: make figures
  Figure S1 (CIC-IoT)     results/tables/dataset_comparison_results.json
                          Produced by: make ciciot2023

Figure 1 (the architecture diagram) is intentionally NOT generated: it is an
analytical schematic of the four-stage pipeline with no experimental content,
so it stays hand-authored in the manuscript.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_RESULTS_ROOT = os.path.join(ROOT, "results")
DEFAULT_OUT_DIR = os.path.join(ROOT, "Paper", "figures")
DEFAULT_ATTACK = "label_flip_30"

# Plot every PLOT_STRIDE-th round, matching the downsampling the manuscript's
# figure-provenance note describes ("downsampled to 5-round intervals for
# plotting clarity at single-column width").
PLOT_STRIDE = 5

BANNER = (
    "%% GENERATED FILE -- DO NOT EDIT BY HAND.\n"
    "%% Written by scripts/generate_manuscript_figures.py\n"
    "%% Source artifact(s): {sources}\n"
    "%% Regenerate with: make manuscript-figures\n"
)


# ── artifact loading ─────────────────────────────────────────────────────────

def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [unreadable] {path}: {exc}")
        return None


def _comparison_logs(results_root: str, strategy: str,
                     attack: str) -> List[Tuple[str, dict]]:
    """Every seed's experiment log for one strategy, sorted by path."""
    pattern = os.path.join(results_root, "logs", "comparison",
                           f"{strategy}_{attack}_seed*", "experiment_log.json")
    out = []
    for path in sorted(glob.glob(pattern)):
        data = _load_json(path)
        if data is not None:
            out.append((path, data))
    return out


def _mean_over_seeds(series: Sequence[Sequence[float]]) -> List[float]:
    """Element-wise mean, truncated to the shortest run present."""
    usable = [s for s in series if s]
    if not usable:
        return []
    n = min(len(s) for s in usable)
    return [sum(s[i] for s in usable) / len(usable) for i in range(n)]


def _downsample(values: Sequence[float], stride: int = PLOT_STRIDE
                ) -> List[Tuple[int, float]]:
    """(round, value) pairs at `stride` intervals, always keeping the last."""
    if not values:
        return []
    idx = list(range(0, len(values), stride))
    if idx[-1] != len(values) - 1:
        idx.append(len(values) - 1)
    # Rounds are 1-indexed on the server (Flower's first round is 1).
    return [(i + 1, float(values[i])) for i in idx]


def _coords(points: Sequence[Tuple[int, float]], places: int = 4) -> str:
    return "".join(f"({r},{v:.{places}f})" for r, v in points)


def _write(out_dir: str, name: str, body: str, sources: Sequence[str],
           dry_run: bool) -> str:
    path = os.path.join(out_dir, f"fig_{name}.tex")
    if dry_run:
        return path
    os.makedirs(out_dir, exist_ok=True)
    rel = ", ".join(os.path.relpath(s, ROOT).replace("\\", "/") for s in sources)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(BANNER.format(sources=rel))
        fh.write(body)
    return path


# ── figure builders ──────────────────────────────────────────────────────────
# Each returns (body, sources) or None when its artifact is unavailable.

CONVERGENCE_SERIES = [
    ("tvflids", "TV-FLIDS", "blue, thick, smooth"),
    ("fltrust", "FLTrust", "orange, thick, dashed, smooth"),
    ("krum", "Krum", "teal, thick, dotted, smooth"),
    ("fedavg", "FedAvg", "red!70!black, thin, densely dotted, smooth"),
]


def build_convergence(results_root: str, attack: str):
    """Figure 2: per-round test accuracy, seed-averaged."""
    lines, sources = [], []
    for strategy, label, style in CONVERGENCE_SERIES:
        logs = _comparison_logs(results_root, strategy, attack)
        if not logs:
            print(f"  [skip convergence] no logs for strategy '{strategy}'")
            return None
        series = [[float(r["accuracy"]) for r in data.get("rounds", [])
                   if "accuracy" in r] for _, data in logs]
        mean = _mean_over_seeds(series)
        if not mean:
            print(f"  [skip convergence] '{strategy}' logs carry no accuracy")
            return None
        sources.extend(p for p, _ in logs)
        lines.append(f"\t\t\t\t\\addplot[{style}] coordinates {{"
                     f"{_coords(_downsample(mean))}}};")
        lines.append(f"\t\t\t\t\\addlegendentry{{{label}}}")
    return "\n".join(lines) + "\n", sources


def build_trust(results_root: str, attack: str):
    """Figure 3: mean honest vs. Byzantine trust per round, seed-averaged."""
    logs = _comparison_logs(results_root, "tvflids", attack)
    if not logs:
        print("  [skip trust] no TV-FLIDS comparison logs")
        return None

    honest_runs, byz_runs, n_honest, n_byz = [], [], None, None
    for path, data in logs:
        extra = data.get("extra", {})
        history = extra.get("trust_history")
        malicious = extra.get("malicious_ids")
        if not history or malicious is None:
            print(f"  [skip trust] {path} lacks extra.trust_history / "
                  "extra.malicious_ids (run produced before the logger "
                  "persisted them)")
            return None
        mal = {str(c) for c in malicious}
        honest = [v for k, v in history.items() if k not in mal]
        byz = [v for k, v in history.items() if k in mal]
        if not honest or not byz:
            print(f"  [skip trust] {path} has an empty honest or Byzantine set")
            return None
        n_honest, n_byz = len(honest), len(byz)
        honest_runs.append(_mean_over_seeds(honest))
        byz_runs.append(_mean_over_seeds(byz))

    honest_mean = _mean_over_seeds(honest_runs)
    byz_mean = _mean_over_seeds(byz_runs)
    if not honest_mean or not byz_mean:
        print("  [skip trust] trust histories are empty")
        return None

    tau_min = 0.01
    last_round = max(len(honest_mean), len(byz_mean))
    body = (
        f"\t\t\t\t\\addplot[blue!70!black, thick] coordinates "
        f"{{{_coords(_downsample(honest_mean))}}};\n"
        f"\t\t\t\t\\addlegendentry{{Honest clients ($N_{{\\mathcal{{H}}}}"
        f"{{=}}{n_honest}$)}}\n"
        f"\t\t\t\t\\addplot[red!70!black, thick, dashed] coordinates "
        f"{{{_coords(_downsample(byz_mean))}}};\n"
        f"\t\t\t\t\\addlegendentry{{Byzantine clients ($f{{=}}{n_byz}$)}}\n"
        f"\t\t\t\t\\addplot[gray!60, thin, densely dashed] coordinates "
        f"{{(1,{tau_min})({last_round},{tau_min})}};\n"
        f"\t\t\t\t\\addlegendentry{{Trust floor $\\tau_{{\\min}}$}}\n"
    )
    return body, [p for p, _ in logs]


WEIGHT_SERIES = [
    ("adaptive_alpha", r"$\alpha$ (similarity)", "blue, thick"),
    ("adaptive_beta", r"$\beta$ (accuracy)", "orange, thick, dashed"),
    ("adaptive_gamma", r"$\gamma$ (anomaly)", "teal, thick, dotted"),
]


def build_weights(results_root: str, attack: str):
    """Figure 4: meta-gradient (alpha, beta, gamma) trajectory."""
    logs = _comparison_logs(results_root, "tvflids", attack)
    if not logs:
        print("  [skip weights] no TV-FLIDS comparison logs")
        return None

    per_key: Dict[str, List[List[float]]] = {k: [] for k, _, _ in WEIGHT_SERIES}
    for path, data in logs:
        rounds = data.get("extra", {}).get("strategy_round_logs")
        if not rounds:
            print(f"  [skip weights] {path} lacks extra.strategy_round_logs")
            return None
        for key in per_key:
            series = [float(r[key]) for r in rounds if key in r]
            if not series:
                print(f"  [skip weights] {path} carries no '{key}' "
                      "(was the run non-adaptive?)")
                return None
            per_key[key].append(series)

    lines = []
    for key, label, style in WEIGHT_SERIES:
        mean = _mean_over_seeds(per_key[key])
        lines.append(f"\t\t\t\t\\addplot[{style}] coordinates "
                     f"{{{_coords(_downsample(mean))}}};")
        lines.append(f"\t\t\t\t\\addlegendentry{{{label}}}")
    return "\n".join(lines) + "\n", [p for p, _ in logs]


ROBUST_SERIES = [
    ("tvflids", "TV-FLIDS", "blue, thick, mark=*, mark size=1.6"),
    ("fltrust", "FLTrust", "orange, thick, dashed, mark=square*, mark size=1.6"),
    ("krum", "Krum", "teal, thick, dotted, mark=triangle*, mark size=2"),
    ("fedavg", "FedAvg", "red!70!black, thin, densely dotted, mark=o, mark size=1.6"),
]


def build_robustness(results_root: str, attack: str):
    """Figure 5: accuracy vs. adversarial client fraction."""
    path = os.path.join(results_root, "tables", "ratio_sweep_results.json")
    data = _load_json(path)
    if not data:
        print(f"  [skip robustness] missing or empty {path}")
        return None

    lines = []
    for strategy, label, style in ROBUST_SERIES:
        series = data.get(strategy)
        if not series:
            print(f"  [skip robustness] no '{strategy}' entry in {path}")
            return None
        points = []
        for ratio, stats in sorted(series.items(), key=lambda kv: float(kv[0])):
            if "accuracy_mean" not in stats:
                print(f"  [skip robustness] '{strategy}' ratio {ratio} has no "
                      "accuracy_mean")
                return None
            points.append((float(ratio), float(stats["accuracy_mean"])))
        coords = "".join(f"({r:.2f},{v:.4f})" for r, v in points)
        lines.append(f"\t\t\t\t\\addplot[{style}] coordinates {{{coords}}};")
        lines.append(f"\t\t\t\t\\addlegendentry{{{label}}}")
    return "\n".join(lines) + "\n", [path]


CICIOT_SERIES = [
    ("accuracy", "Accuracy", "fill=blue!55"),
    ("f1_macro", "Macro-F1", "fill=orange!70"),
]
CICIOT_STRATEGIES = [("fedavg", "FedAvg"), ("krum", "Krum"),
                     ("fltrust", "FLTrust"), ("tvflids", "TV-FLIDS")]


def build_ciciot(results_root: str, attack: str):
    """Supplementary Figure S1: CIC-IoT-2023 accuracy and macro-F1 bars."""
    path = os.path.join(results_root, "tables",
                        "dataset_comparison_results.json")
    data = _load_json(path)
    if not data:
        print(f"  [skip ciciot] missing or empty {path}")
        return None
    block = data.get("ciciot2023")
    if not block:
        print(f"  [skip ciciot] {path} has no 'ciciot2023' block")
        return None

    lines = []
    for metric, _label, style in CICIOT_SERIES:
        bars = []
        for strategy, name in CICIOT_STRATEGIES:
            runs = block.get(strategy)
            if not runs:
                print(f"  [skip ciciot] no '{strategy}' runs in {path}")
                return None
            key = f"final_{metric}"
            vals = [float(r[key]) for r in runs if key in r]
            if not vals:
                print(f"  [skip ciciot] '{strategy}' runs carry no {key}")
                return None
            bars.append(f"({name},{sum(vals) / len(vals):.4f})")
        lines.append(f"\t\t\t\t\\addplot[{style}] coordinates "
                     f"{{{' '.join(bars)}}};")
    lines.append("\t\t\t\t\\legend{Accuracy, Macro-F1}")
    return "\n".join(lines) + "\n", [path]


BUILDERS = [
    ("convergence", "Figure 2  (per-round accuracy)", build_convergence),
    ("trust", "Figure 3  (trust evolution)", build_trust),
    ("weights", "Figure 4  (meta-gradient weights)", build_weights),
    ("robustness", "Figure 5  (robustness curve)", build_robustness),
    ("ciciot", "Figure S1 (CIC-IoT-2023)", build_ciciot),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate manuscript pgfplots data from experiment output")
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--attack", default=DEFAULT_ATTACK)
    ap.add_argument("--check", action="store_true",
                    help="report only; write nothing; exit 1 if incomplete")
    args = ap.parse_args(argv)

    print("=== Manuscript figure regeneration ===")
    print(f"  results root : {args.results_root}")
    print(f"  output dir   : {args.out_dir}")
    print(f"  attack       : {args.attack}")
    if args.check:
        print("  mode         : CHECK ONLY (no files written)")
    print()

    written, missing = [], []
    for name, title, builder in BUILDERS:
        print(f"{title}")
        built = builder(args.results_root, args.attack)
        if built is None:
            missing.append(name)
            print("  -> NOT generated; manuscript keeps its provisional "
                  "coordinates\n")
            continue
        body, sources = built
        path = _write(args.out_dir, name, body, sources, dry_run=args.check)
        written.append(name)
        verb = "would write" if args.check else "wrote"
        print(f"  -> {verb} {os.path.relpath(path, ROOT)}\n")

    print("=== Summary ===")
    print(f"  generated : {', '.join(written) if written else '(none)'}")
    print(f"  pending   : {', '.join(missing) if missing else '(none)'}")
    if missing:
        print("\n  Pending figures need the corresponding campaign to be run:")
        print("    convergence / trust / weights -> make full-comparison")
        print("    robustness                    -> make figures")
        print("    ciciot                        -> make ciciot2023")
        print("  Nothing was invented for them; the manuscript continues to "
              "compile against its provisional, explicitly-labelled "
              "coordinates.")
        return 1 if args.check else 0
    print("\n  All data-driven manuscript figures are backed by real output.")
    print("  Rebuild the PDFs to pick them up:  latexmk -pdf Paper/TV-FLIDS.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
