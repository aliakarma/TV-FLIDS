"""
scripts/check_manuscript.py
Integrity checks for the manuscript sources and the compiled PDFs.

WHY THIS EXISTS
---------------
A malformed cross-reference survived a clean LaTeX compile. Section VII-A read

    ... to Section~<CR>ef{sec:adaptive}, where ...

where <CR> is a raw carriage-return byte that had replaced the backslash of
``\\ref``. LaTeX treats a bare CR as whitespace, so:

  * there was no undefined-reference warning -- no reference command survived
    to be undefined;
  * there was no compile error;
  * the PDF silently typeset "Section efsec:adaptive", with the braces eaten as
    grouping characters, so even grepping the extracted PDF text for "ef{"
    found nothing.

Three independent detectors were needed to catch it, and all three live here:
a byte-level scan of the source, a token-level scan for reference commands
missing their backslash, and a text-level scan of the compiled PDF.

USAGE
-----
    python scripts/check_manuscript.py              # source + PDF + log
    python scripts/check_manuscript.py --source-only

Exit status is 0 when every check passes, 1 otherwise. PDF and log checks are
skipped (not failed) when the artifacts are absent, so the source checks can
run in an environment with no TeX installation.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_paper = os.path.join(ROOT, "Paper", "IEEE")
PAPER_DIR = _default_paper if os.path.isdir(_default_paper) else os.path.join(ROOT, "Paper")
DOCUMENTS = ["TV-FLIDS", "TV-FLIDS_supplementary"]

# Commands whose leading backslash going missing is silent rather than fatal:
# the bare name typesets as ordinary text and the braces vanish as grouping.
REFERENCE_COMMANDS = ["ref", "eqref", "label", "cite", "pageref", "autoref",
                      "includegraphics", "input", "caption",
                      # "ef" catches the residue left when BOTH the backslash
                      # and the 'r' are lost, which is how the original defect
                      # rendered once the CR was normalized away. The negative
                      # look-behind on letters keeps legitimate "\eqref{" and
                      # "\ref{" from matching, since there "ef{" is preceded
                      # by a letter.
                      "ef"]

# Text that should never survive into a compiled PDF.
PDF_FORBIDDEN = [
    ("??", "an undefined reference or citation"),
    ("ef{", "a reference command that lost its backslash"),
    ("efsec:", "a \\ref whose backslash was replaced by whitespace"),
    ("eqrefeq:", "an \\eqref whose backslash was replaced by whitespace"),
    ("[?]", "an unresolved citation key"),
    ("TODO", "leftover draft marker"),
    ("FIXME", "leftover draft marker"),
    ("XXX", "leftover draft marker"),
    ("\\ref{", "a reference command printed verbatim"),
]

LOG_FORBIDDEN = [
    ("There were undefined references", "undefined reference"),
    ("Citation ", "undefined citation"),          # matched with 'undefined'
]


def _fail(problems: List[str], msg: str) -> None:
    problems.append(msg)
    print(f"  [FAIL] {msg}")


# ── source-level checks ──────────────────────────────────────────────────────

def check_control_bytes(path: str, problems: List[str]) -> None:
    """No stray CR, no form feed, no NUL, no other control characters.

    A lone CR inside a line is the specific corruption that produced the
    Section~ef{sec:adaptive} defect. Tab and newline are legitimate.
    """
    with open(path, "rb") as fh:
        raw = fh.read()

    for m in re.finditer(rb"\r(?!\n)", raw):
        i = m.start()
        line = raw[:i].count(b"\n") + 1
        ctx = raw[max(0, i - 60):i].decode("utf-8", "replace")
        _fail(problems,
              f"{os.path.basename(path)}:{line} stray carriage return "
              f"(not part of CRLF) after: ...{ctx!r}")

    for m in re.finditer(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]", raw):
        i = m.start()
        line = raw[:i].count(b"\n") + 1
        _fail(problems,
              f"{os.path.basename(path)}:{line} control byte "
              f"0x{raw[i]:02x} in source")

    # A macro whose leading backslash was replaced by the control character its
    # first letter spells as an escape: \textbf -> TAB + "extbf", \nu -> LF +
    # "u", \, and so on. These typeset as ordinary text, so LaTeX reports
    # nothing. Anchored on the macro name so ordinary indentation is not hit.
    mangled = rb"[\t\n\r\x08\x0b\x0c\x07](ext(?:bf|tt|it|sc|superscript)\{|" \
              rb"abular|ableofcontents|ewcommand|ewline|oindent|ef\{|" \
              rb"egin\{|ibliography)"
    for m in re.finditer(mangled, raw):
        i = m.start()
        line = raw[:i].count(b"\n") + 1
        _fail(problems,
              f"{os.path.basename(path)}:{line} macro is missing its leading "
              f"backslash (replaced by control byte 0x{raw[i]:02x}): "
              f"{m.group(1).decode('utf-8', 'replace')}")


def check_reference_commands(path: str, problems: List[str]) -> None:
    """Every reference-like token must be preceded by exactly one backslash."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    names = "|".join(REFERENCE_COMMANDS)
    # A command name immediately followed by '{' whose preceding character is
    # not a backslash and not a letter (so "Xref{" inside a longer word, or a
    # legitimate "\eqref{", is not reported).
    pattern = re.compile(rf"(?<![A-Za-z\\])({names})\{{")

    for n, line in enumerate(lines, start=1):
        code = line.split("%", 1)[0] if not line.lstrip().startswith("%") else ""
        for m in pattern.finditer(code):
            _fail(problems,
                  f"{os.path.basename(path)}:{n} '{m.group(1)}{{' is missing "
                  f"its leading backslash: ...{code[max(0, m.start()-40):m.end()+20].strip()}")


def check_balanced_generated_inputs(problems: List[str]) -> None:
    """Any generated figure body must be brace-balanced before \\input."""
    fig_dir = os.path.join(PAPER_DIR, "figures")
    if not os.path.isdir(fig_dir):
        return
    for name in sorted(os.listdir(fig_dir)):
        if not name.endswith(".tex"):
            continue
        path = os.path.join(fig_dir, name)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        stripped = "\n".join(
            ln.split("%", 1)[0] for ln in text.splitlines())
        if stripped.count("{") != stripped.count("}"):
            _fail(problems, f"figures/{name} has unbalanced braces; "
                            "\\input would break the surrounding axis")


# ── artifact-level checks ────────────────────────────────────────────────────

def _pdf_text(pdf_path: str) -> str:
    if not shutil.which("pdftotext"):
        return ""
    try:
        out = subprocess.run(["pdftotext", "-layout", pdf_path, "-"],
                             capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.decode("utf-8", "replace")


def check_pdf(doc: str, problems: List[str]) -> bool:
    pdf = os.path.join(PAPER_DIR, f"{doc}.pdf")
    if not os.path.exists(pdf):
        print(f"  [skip] {doc}.pdf not built")
        return False
    text = _pdf_text(pdf)
    if not text:
        print(f"  [skip] {doc}.pdf: pdftotext unavailable or produced nothing")
        return False
    for needle, why in PDF_FORBIDDEN:
        idx = text.find(needle)
        if idx != -1:
            ctx = " ".join(text[max(0, idx - 70):idx + 70].split())
            _fail(problems, f"{doc}.pdf contains {needle!r} ({why}): ...{ctx}...")
    print(f"  [ok] {doc}.pdf text scanned ({len(text):,} chars)")
    return True


def check_log(doc: str, problems: List[str]) -> bool:
    log = os.path.join(PAPER_DIR, f"{doc}.log")
    if not os.path.exists(log):
        print(f"  [skip] {doc}.log not present")
        return False
    with open(log, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    for line in text.splitlines():
        low = line.lower()
        if "undefined" in low and ("reference" in low or "citation" in low):
            _fail(problems, f"{doc}.log: {line.strip()}")
        if "undefined control sequence" in low:
            _fail(problems, f"{doc}.log: {line.strip()}")
    print(f"  [ok] {doc}.log scanned")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Manuscript integrity checks")
    ap.add_argument("--source-only", action="store_true",
                    help="skip PDF and log checks")
    args = ap.parse_args(argv)

    problems: List[str] = []

    print("=== Source checks ===")
    for doc in DOCUMENTS:
        path = os.path.join(PAPER_DIR, f"{doc}.tex")
        if not os.path.exists(path):
            _fail(problems, f"missing source {doc}.tex")
            continue
        check_control_bytes(path, problems)
        check_reference_commands(path, problems)
        print(f"  [ok] {doc}.tex scanned")
    check_balanced_generated_inputs(problems)

    if not args.source_only:
        print("\n=== Compiled PDF checks ===")
        for doc in DOCUMENTS:
            check_pdf(doc, problems)
        print("\n=== LaTeX log checks ===")
        for doc in DOCUMENTS:
            check_log(doc, problems)

    print("\n=== Summary ===")
    if problems:
        print(f"  {len(problems)} problem(s) found.")
        return 1
    print("  All manuscript integrity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
