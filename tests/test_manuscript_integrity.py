"""
tests/test_manuscript_integrity.py
Exercises scripts/check_manuscript.py against the real sources and against
fixtures reproducing the defect it was written for.

BACKGROUND. A cross-reference in Section VII-A read

    ... to Section~<CR>ef{sec:adaptive}, where ...

with a raw carriage-return byte where the backslash of \\ref belonged. LaTeX
treats a bare CR as whitespace, so the document compiled with no error, no
undefined-reference warning, and no reference command left to be undefined. It
typeset as "Section efsec:adaptive" -- and because TeX ate the braces as
grouping characters, even grepping the extracted PDF text for "ef{" missed it.

The checker therefore looks at three levels, and this file tests each:
byte-level (stray control characters), token-level (macros missing a leading
backslash, including ones whose backslash became the control character its
first letter spells as a Python escape, e.g. \\textbf -> TAB + "extbf"), and
text-level (the compiled PDF).
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.check_manuscript as cm  # noqa: E402

BS = bytes([92])
CR = bytes([13])
TAB = bytes([9])
CRLF = CR + bytes([10])


@pytest.fixture
def paper_dir(tmp_path, monkeypatch):
    """Redirect the checker at a throwaway Paper/ directory."""
    d = tmp_path / "Paper"
    d.mkdir()
    monkeypatch.setattr(cm, "PAPER_DIR", str(d))
    return d


def _write(paper_dir, main=b"", supp=b"clean text\n"):
    (paper_dir / "TV-FLIDS.tex").write_bytes(main)
    (paper_dir / "TV-FLIDS_supplementary.tex").write_bytes(supp)


class TestRealManuscript:
    """The shipped sources must pass, and must actually be present."""

    def test_sources_exist(self):
        for name in cm.DOCUMENTS:
            assert os.path.exists(os.path.join(cm.PAPER_DIR, f"{name}.tex")), (
                f"{name}.tex is missing from Paper/")

    def test_source_checks_pass(self):
        assert cm.main(["--source-only"]) == 0


class TestCatchesTheOriginalDefect:

    def test_stray_carriage_return(self, paper_dir):
        _write(paper_dir,
               b"See Section~" + CR + b"ef{sec:adaptive} for details." + CRLF)
        assert cm.main(["--source-only"]) == 1

    def test_ef_residue_without_a_control_byte(self, paper_dir):
        _write(paper_dir, b"See Section~ef{sec:adaptive} for details." + CRLF)
        assert cm.main(["--source-only"]) == 1

    def test_ref_missing_only_its_backslash(self, paper_dir):
        _write(paper_dir, b"See Section~ref{sec:adaptive} for details." + CRLF)
        assert cm.main(["--source-only"]) == 1

    def test_textbf_whose_backslash_became_a_tab(self, paper_dir):
        _write(paper_dir, b"Caption: " + TAB + b"extbf{Pending} here." + CRLF)
        assert cm.main(["--source-only"]) == 1

    @pytest.mark.parametrize("name", [b"eqref", b"label", b"cite", b"input",
                                      b"includegraphics", b"caption"])
    def test_other_reference_commands(self, paper_dir, name):
        _write(paper_dir, b"Broken " + name + b"{x} here." + CRLF)
        assert cm.main(["--source-only"]) == 1


class TestNoFalsePositives:

    def test_well_formed_commands_pass(self, paper_dir):
        body = (b"Fine: " + BS + b"ref{a}, " + BS + b"eqref{b}, " + BS
                + b"label{c}, " + BS + b"cite{d}, " + BS + b"textbf{e}, " + BS
                + b"caption{f}." + CRLF)
        _write(paper_dir, body)
        assert cm.main(["--source-only"]) == 0

    def test_command_name_inside_a_longer_word_passes(self, paper_dir):
        # "ef{" preceded by a letter, as inside \eqref{, must not fire.
        _write(paper_dir, b"The chef{s} and " + BS + b"eqref{eq:x}." + CRLF)
        assert cm.main(["--source-only"]) == 0

    def test_ordinary_indentation_tabs_pass(self, paper_dir):
        _write(paper_dir, TAB + TAB + b"indented body text" + CRLF)
        assert cm.main(["--source-only"]) == 0

    def test_percent_comments_are_ignored(self, paper_dir):
        _write(paper_dir, b"%% a comment mentioning ref{x} and label{y}" + CRLF)
        assert cm.main(["--source-only"]) == 0

    def test_crlf_line_endings_are_not_stray_carriage_returns(self, paper_dir):
        _write(paper_dir, b"line one" + CRLF + b"line two" + CRLF)
        assert cm.main(["--source-only"]) == 0


class TestMissingSources:

    def test_absent_source_is_reported(self, paper_dir):
        (paper_dir / "TV-FLIDS.tex").write_bytes(b"ok" + CRLF)
        # supplementary deliberately not written
        assert cm.main(["--source-only"]) == 1
