"""
scripts/generate_tables.py
Convert results JSON to LaTeX table format for paper insertion.

Usage:
    python scripts/generate_tables.py \
        --input results/tables/full_comparison_results.json \
        --output results/tables/table1.tex
"""
import argparse
import json
import numpy as np
from pathlib import Path


STRATEGY_LABELS = {
    "fedavg":        "FedAvg (no defense)",
    "krum":          "Krum \\cite{blanchard2017nips}",
    "trimmed_mean":  "Trimmed Mean \\cite{yin2018icml}",
    "fltrust":       "FLTrust \\cite{cao2021fltrust}",
    "foolsgold":     "FoolsGold \\cite{fung2018foolsgold}",
    "flame":         "FLAME \\cite{nguyen2022flame}",
    "rfa":           "RFA \\cite{pillutla2022rfa}",
    "tvflids":       "\\textbf{TV-FLIDS (Ours)}",
}

METRIC_COLS = [
    ("final_accuracy",            "Accuracy"),
    ("final_f1_macro",            "F1-Macro"),
    ("final_attack_success_rate", "ASR $\\downarrow$"),
]


def fmt(mean, std):
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def generate_latex_table(data: dict) -> str:
    raw = data.get("raw", {})
    table_data = data.get("table", {})

    header_cols = " & ".join(c[1] for c in METRIC_COLS)
    col_spec = "l" + "c" * len(METRIC_COLS)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Strategy comparison under 30\\% label flip attack on NSL-KDD (Non-IID $\\alpha=0.5$). "
        "Mean $\\pm$ std over 5 seeds. $\\dagger$ $p < 0.05$ (Wilcoxon signed-rank vs.~TV-FLIDS).}",
        "\\label{tab:main_results}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"Strategy & {header_cols} \\\\",
        "\\midrule",
    ]

    for strategy, results in raw.items():
        if not results:
            continue
        label = STRATEGY_LABELS.get(strategy, strategy)
        row_cells = []
        for metric_key, _ in METRIC_COLS:
            vals = [r.get(metric_key, 0.0) for r in results if metric_key in r]
            if vals:
                row_cells.append(fmt(float(np.mean(vals)), float(np.std(vals))))
            else:
                row_cells.append("—")
        lines.append(f"{label} & {' & '.join(row_cells)} \\\\")
        if strategy == "rfa":
            lines.append("\\midrule")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/tables/full_comparison_results.json")
    parser.add_argument("--output", default="results/tables/table1.tex")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    tex = generate_latex_table(data)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(tex)
    print(f"[Tables] LaTeX table saved to {args.output}")
    print(tex)


if __name__ == "__main__":
    main()
