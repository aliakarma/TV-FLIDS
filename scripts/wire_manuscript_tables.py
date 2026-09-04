"""
scripts/wire_manuscript_tables.py
One-shot rewrite: make each data table in the manuscript read its rows from
Paper/tables/tab_<name>.tex when that file exists, and keep its current
provisional rows when it does not.

This is the table-side equivalent of the treatment Paper/figures/ already gets:

    \\midrule
    \\IfFileExists{tables/tab_main.tex}{%
        \\input{tables/tab_main.tex}%
    }{%
        <the rows currently in the manuscript, untouched>
    }
    \\bottomrule

so the document compiles identically before the campaign and picks up genuine
values afterwards, with no hand-typed number ever entering the .tex.

The script is idempotent: a table already wrapped is left alone. It edits only
the region between a table's header rule and its \\bottomrule, and it refuses to
touch a table whose structure it cannot identify unambiguously.

Usage:
    python scripts/wire_manuscript_tables.py [--dry-run]
"""

from __future__ import annotations

import argparse
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# label in the .tex  ->  basename of the generated rows file
MAIN_TABLES = {
    "tab:main": "main",
    "tab:leakage_free": "leakage_free",
    "tab:ablation": "ablation",
    "tab:noniid_sweep": "noniid_sweep",
    "tab:attacks": "attacks",
    "tab:multiattack_baselines": "multiattack_baselines",
    "tab:adaptive": "adaptive",
    "tab:overhead": "overhead",
}
SUPP_TABLES = {
    "tab:baseline-hp-sweep": "baseline_hp",
    "tab:own-hp-sweep": "tvflids_hp",
    "tab:recent-baselines": "recent_baselines",
    "tab:ciciot": "ciciot",
}

MAIN_TEX = os.path.join(ROOT, "Paper", "TV-FLIDS.tex")
SUPP_TEX = os.path.join(ROOT, "Paper", "TV-FLIDS_supplementary.tex")


def wire_one(src: str, label: str, name: str) -> tuple[str, str]:
    """Return (new_source, status) for one table."""
    lab = "\\label{" + label + "}"
    at = src.find(lab)
    if at < 0:
        return src, "label not found"

    tab_start = src.find("\\begin{tabular}", at)
    tab_end = src.find("\\end{tabular}", tab_start)
    if tab_start < 0 or tab_end < 0:
        return src, "no tabular after label"

    block = src[tab_start:tab_end]
    if "\\IfFileExists" in block:
        return src, "already wired"

    # The body is everything after the LAST header rule that precedes the
    # first data row, up to \bottomrule. Header rows sit between \toprule and
    # the first \midrule; data rows follow it.
    first_mid = block.find("\\midrule")
    bottom = block.rfind("\\bottomrule")
    if first_mid < 0 or bottom < 0 or bottom < first_mid:
        return src, "no \\midrule/\\bottomrule pair"

    body_start = first_mid + len("\\midrule")
    body = block[body_start:bottom]
    if "\\\\" not in body:
        return src, "no data rows between rules"

    indent = "\t\t\t"
    wrapped = (
        "\n" + indent + "%% Rows below are the provisional, explicitly-labelled\n"
        + indent + "%% values. scripts/generate_manuscript_tables.py writes\n"
        + indent + "%% figures/../tables/tab_" + name + ".tex from genuine result\n"
        + indent + "%% artifacts; when that file exists it supersedes them.\n"
        + indent + "%% Regenerate with: make manuscript-tables\n"
        + indent + "\\IfFileExists{tables/tab_" + name + ".tex}{%\n"
        + indent + "\t\\input{tables/tab_" + name + ".tex}%\n"
        + indent + "}{%"
        + body.rstrip() + "\n"
        + indent + "}\n" + indent
    )

    new_block = block[:body_start] + wrapped + block[bottom:]
    return src[:tab_start] + new_block + src[tab_end:], "wired"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rc = 0
    for tex_path, mapping in ((MAIN_TEX, MAIN_TABLES), (SUPP_TEX, SUPP_TABLES)):
        src = io.open(tex_path, encoding="utf-8", newline="").read()
        original = src
        print(f"\n{os.path.relpath(tex_path, ROOT)}")
        for label, name in mapping.items():
            src, status = wire_one(src, label, name)
            flag = "  " if status in ("wired", "already wired") else "!!"
            print(f"  {flag} {label:32s} -> tab_{name}.tex   [{status}]")
            if flag == "!!":
                rc = 1
        if not args.dry_run and src != original:
            io.open(tex_path, "w", encoding="utf-8", newline="").write(src)
            print(f"  updated {os.path.relpath(tex_path, ROOT)}")
    if args.dry_run:
        print("\n(dry run: nothing written)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
