r"""
scripts/wire_manuscript_tables.py
One-shot rewrite: make each data table in the manuscript swappable, so that a
generated table supersedes the provisional one without any hand editing.

The whole tabular is the swappable unit:

    \IfFileExists{tables/tab_main.tex}%
        {\def\tvfTable{tables/tab_main.tex}}%
        {\def\tvfTable{tables/tab_main_provisional.tex}}%
    \input{\tvfTable}

Two earlier, more obvious designs do not work and are recorded here so they are
not retried:

  * Wrapping the rows in \IfFileExists's brace group opens a group around them,
    which breaks every table containing a \multicolumn row ("! Misplaced
    \omit") -- most of them.
  * Keeping the tabular in the manuscript and \input-ing only the rows fails as
    soon as an inserted file contains \midrule ("! Misplaced \noalign"), since
    \noalign material must follow \cr directly.

At top level \input is unconditionally safe, so the generator emits complete
tabulars, assembled from the manuscript's own head and foot (extracted here
into tab_<name>_head.tex / _foot.tex) plus generated rows.

The provisional tabular is moved verbatim into tab_<name>_provisional.tex. It is
neither deleted nor edited: until a campaign produces the real artifact the
document typesets exactly what it typeset before, still marked as provisional by
its caption.

Idempotent: a table already wired is left alone.

Usage:
    python scripts/wire_manuscript_tables.py [--dry-run]
"""

from __future__ import annotations

import argparse
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "Paper", "tables")

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

PROVISIONAL_HEADER = """% Provisional table, moved verbatim out of the manuscript by
% scripts/wire_manuscript_tables.py. This is what the document typesets until
% scripts/generate_manuscript_tables.py writes tab_NAME.tex from a genuine
% result artifact, at which point that file supersedes this one. The
% surrounding caption marks these values as pending regeneration.
"""

HEAD_FOOT_HEADER = """% Extracted verbatim from the manuscript by
% scripts/wire_manuscript_tables.py, so that a generated table reuses the
% manuscript's own column specification and header row rather than
% re-deriving them. Edit the manuscript, then re-run that script.
"""


def wire_one(src: str, label: str, name: str, write: bool):
    """Replace one table's whole tabular with a swappable \\input."""
    lab = "\\label{" + label + "}"
    at = src.find(lab)
    if at < 0:
        return src, "label not found"

    tab_start = src.find("\\begin{tabular}", at)
    end_token = "\\end{tabular}"
    tab_end = src.find(end_token, tab_start)
    if tab_start < 0 or tab_end < 0:
        return src, "no tabular after label"
    tab_end += len(end_token)

    if "\\tvfTable" in src[at:tab_end]:
        return src, "already wired"

    block = src[tab_start:tab_end]
    first_mid = block.find("\\midrule")
    bottom = block.rfind("\\bottomrule")
    if first_mid < 0 or bottom < 0 or bottom < first_mid:
        return src, "no midrule/bottomrule pair"

    cut = first_mid + len("\\midrule")
    head, body, foot = block[:cut], block[cut:bottom], block[bottom:]
    if "\\\\" not in body:
        return src, "no data rows between rules"

    if write:
        os.makedirs(OUT_DIR, exist_ok=True)
        pieces = (
            ("_head", head, HEAD_FOOT_HEADER),
            ("_foot", foot, HEAD_FOOT_HEADER),
            ("_provisional", head + body.rstrip("\n") + "\n" + foot,
             PROVISIONAL_HEADER.replace("NAME", name)),
        )
        for suffix, content, hdr in pieces:
            path = os.path.join(OUT_DIR, "tab_" + name + suffix + ".tex")
            with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(hdr + content.strip("\n") + "\n")

    ind = "\t\t"
    replacement = "\n".join([
        ind + "%% The whole tabular is swappable. When",
        ind + "%% scripts/generate_manuscript_tables.py has written",
        ind + "%% tables/tab_" + name + ".tex from a genuine result artifact it is",
        ind + "%% used; otherwise the provisional table is, and the caption",
        ind + "%% marks it as pending. Regenerate with: make manuscript-tables",
        ind + "\\IfFileExists{tables/tab_" + name + ".tex}%",
        ind + "\t{\\def\\tvfTable{tables/tab_" + name + ".tex}}%",
        ind + "\t{\\def\\tvfTable{tables/tab_" + name + "_provisional.tex}}%",
        ind + "\\input{\\tvfTable}",
    ])
    return src[:tab_start] + replacement + src[tab_end:], "wired"


def main() -> int:
    ap = argparse.ArgumentParser(description="Wire manuscript tables")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rc = 0
    for tex_path, mapping in ((MAIN_TEX, MAIN_TABLES), (SUPP_TEX, SUPP_TABLES)):
        src = io.open(tex_path, encoding="utf-8", newline="").read()
        original = src
        print("\n" + os.path.relpath(tex_path, ROOT))
        for label, name in mapping.items():
            src, status = wire_one(src, label, name, write=not args.dry_run)
            ok = status in ("wired", "already wired")
            print(f"  {'  ' if ok else '!!'} {label:32s} "
                  f"-> tab_{name}.tex   [{status}]")
            if not ok:
                rc = 1
        if not args.dry_run and src != original:
            io.open(tex_path, "w", encoding="utf-8", newline="").write(src)
            print("  updated " + os.path.relpath(tex_path, ROOT))
    if args.dry_run:
        print("\n(dry run: nothing written)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
