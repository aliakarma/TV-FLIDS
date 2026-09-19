"""
evaluation/table_generator.py
LaTeX table generation from canonical statistical analysis artifacts.
Reference: IEEE TIFS Manuscript Table III, Table IV, and Supplementary Tables S3–S4.

Guarantees:
  - Generates publication-ready LaTeX table snippets from analysis artifacts.
  - Explicitly tags incomplete or missing data with '--' or '[INCOMPLETE]'.
  - Never overwrites manuscript files directly with test fixture values.
  - Clear provenance header on all generated snippets.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def format_p_value(p: Optional[float]) -> str:
    """Format p-value for LaTeX table."""
    if p is None:
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def format_diff_pp(diff: Optional[float]) -> str:
    """Format percentage point difference with sign for LaTeX table."""
    if diff is None:
        return "--"
    s = f"{diff:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s.replace("-", "$-$")


def format_interval(lo: Optional[float], hi: Optional[float]) -> str:
    """Format confidence interval [lo, hi] for LaTeX table."""
    if lo is None or hi is None:
        return "[--, --]"
    s_lo = format_diff_pp(lo)
    s_hi = format_diff_pp(hi)
    return f"[{s_lo}, {s_hi}]"


def generate_table_s3_pairwise(
    primary_eval_payload: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate Supplementary Table S3: Primary Tests Under LF (20 Seeds).
    Paired median difference (TV-FLIDS minus baseline) with Holm-adjusted p-value in parentheses.
    """
    comparisons = primary_eval_payload.get("comparisons", [])
    comp_map = {(c["dataset"].lower(), c["baseline"].lower(), c["endpoint"].lower()): c for c in comparisons}

    baselines_order = [
        ("fedavg", "FedAvg"),
        ("krum", "Krum"),
        ("multikrum", "Multi-Krum"),
        ("trimmed_mean", "Trimmed Mean"),
        ("norm_clipping", "Norm Clipping"),
        ("rfa", "RFA"),
        ("bucketing", "Bucketing"),
        ("foolsgold", "FoolsGold"),
        ("flame", "FLAME"),
        ("deepsight", "DeepSight"),
        ("fldetector", "FLDetector"),
        ("zeno", "Zeno"),
        ("fltrust", "FLTrust"),
        ("baffle", "BaFFLe"),
    ]
    datasets = [("nslkdd", "NSL-KDD"), ("ciciot2023", "CIC-IoT-2023"), ("edgeiiotset", "Edge-IIoTset")]

    lines = [
        "% TV-FLIDS Generated Supplementary Table S3",
        "% Note: Real campaign execution required for verified experimental values.",
        "\\begin{tabular}{@{}l*{6}{c}@{}}",
        "\\toprule",
        "& \\multicolumn{2}{c}{\\textbf{NSL-KDD}} & \\multicolumn{2}{c}{\\textbf{CIC-IoT-2023}} & \\multicolumn{2}{c}{\\textbf{Edge-IIoTset}} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}",
        "\\textbf{Baseline} & Macro-F1 & ASR & Macro-F1 & ASR & Macro-F1 & ASR \\\\",
        "\\midrule",
    ]

    for base_key, base_name in baselines_order:
        row_cells = [base_name]
        for ds_key, ds_name in datasets:
            for ep in ("f1", "asr"):
                c = comp_map.get((ds_key, base_key, ep))
                if c and c.get("status") == "COMPLETE" and c.get("paired_median_diff") is not None:
                    diff_str = format_diff_pp(c["paired_median_diff"])
                    p_str = format_p_value(c.get("holm_p_value"))
                    cell = f"{diff_str} ({p_str})"
                else:
                    cell = "--"
                row_cells.append(cell)
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
    ])

    content = "\n".join(lines) + "\n"
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    return content


def generate_table_s4_intervals(
    primary_eval_payload: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate Supplementary Table S4: Paired Differences Against Closest Baselines with BCa Intervals.
    """
    comparisons = primary_eval_payload.get("comparisons", [])
    comp_map = {(c["dataset"].lower(), c["baseline"].lower(), c["endpoint"].lower()): c for c in comparisons}

    focus_baselines = [
        ("fltrust", "FLTrust"),
        ("zeno", "Zeno"),
        ("fldetector", "FLDetector"),
    ]
    datasets = [("nslkdd", "NSL-KDD"), ("ciciot2023", "CIC-IoT-2023"), ("edgeiiotset", "Edge-IIoTset")]

    lines = [
        "% TV-FLIDS Generated Supplementary Table S4",
        "% Note: Real campaign execution required for verified experimental values.",
        "\\begin{tabular}{@{}lllcccc@{}}",
        "\\toprule",
        "\\textbf{Baseline} & \\textbf{Dataset} & \\textbf{Endpoint} & \\textbf{Median} & \\textbf{95\\% Interval} & \\textbf{Seeds} & \\textbf{Holm $p$} \\\\",
        "\\midrule",
    ]

    for base_key, base_name in focus_baselines:
        for ds_key, ds_name in datasets:
            for ep in ("f1", "asr"):
                ep_label = "Macro-F1" if ep == "f1" else "ASR"
                c = comp_map.get((ds_key, base_key, ep))
                if c and c.get("status") == "COMPLETE" and c.get("paired_median_diff") is not None:
                    med_str = format_diff_pp(c["paired_median_diff"])
                    ci_str = format_interval(c.get("bca_ci_low"), c.get("bca_ci_high"))
                    seeds_str = f"{c.get('favor_count', 0)}/{c.get('required_seeds', 20)}"
                    p_str = format_p_value(c.get("holm_p_value"))
                else:
                    med_str = "--"
                    ci_str = "[--, --]"
                    seeds_str = "--"
                    p_str = "--"

                row = f"{base_name} & {ds_name} & {ep_label} & {med_str} & {ci_str} & {seeds_str} & {p_str} \\\\"
                lines.append(row)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
    ])

    content = "\n".join(lines) + "\n"
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    return content
