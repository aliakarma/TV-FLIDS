"""
tests/test_equation_crossrefs.py
Keeps the equation numbers quoted in code comments in step with the manuscript.

Several modules cite the paper by equation number ("Eq. (6)", "Eq. (18)") so a
reader of the code can find the specification it implements. Those numbers move
whenever an equation is inserted anywhere earlier in the document, and nothing
otherwise notices: a stale "Eq. (17)" points at a real but wrong equation, which
is worse than no citation at all. This test resolves each label against the
compiled Paper/TV-FLIDS.aux and checks the quoted number.

It SKIPS when the .aux is absent (no TeX toolchain, or the paper has not been
built), so it never blocks a run in an environment without LaTeX. Build with
`make paper` or `latexmk -pdf -cd Paper/TV-FLIDS.tex` to activate it.
"""

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUX = os.path.join(ROOT, "Paper", "TV-FLIDS.aux")

# (file, label, every literal "Eq. (N)" spelling that must resolve to it)
CITATIONS = [
    ("utils/ste.py", "eq:hatw", ["Eq. (13)"]),
    ("utils/ste.py", "eq:signal_range", ["range argument of (10)"]),
    ("theory/proposition1_verification.py", "eq:agg", ["Eq. (15)"]),
    ("theory/proposition1_verification.py", "eq:wstar", ["Eq. (16)"]),
    ("theory/proposition1_verification.py", "eq:prop1", ["Eq. (18)"]),
    ("tests/test_proposition1_domain.py", "eq:wstar", ["Eq. (16)"]),
    ("tests/test_proposition1_domain.py", "eq:prop1", ["Eq. (18)"]),
    ("tests/test_forward_pass_count.py", "eq:acc", ["Eq. (8)"]),
    ("tests/test_a1_ablation_semantics.py", "eq:anom", ["Eq. (9)"]),
    ("tests/test_a1_ablation_semantics.py", "eq:trust", ["Eq. (6)"]),
    ("tests/test_config_defaults.py", "eq:softmax", ["Eq. (11)"]),
    ("tests/test_config_defaults.py", "eq:tau_anneal", ["Eq. (5)"]),
    ("config/fl_config.yaml", "eq:trust", ["Eq. (6)"]),
    ("config/fl_config.yaml", "eq:softmax", ["Eq. (11)"]),
    ("fl/strategy.py", "eq:tau_anneal", ["Eq. (5)"]),
    ("trust/verification.py", "eq:tau_anneal", ["Eq. (5)"]),
]


def _labels():
    """label -> printed number, from the compiled .aux."""
    with open(AUX, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", text):
        out[m.group(1)] = m.group(2)
    return out


pytestmark = pytest.mark.skipif(
    not os.path.exists(AUX),
    reason="Paper/TV-FLIDS.aux not built; run `make paper` to enable")


@pytest.fixture(scope="module")
def labels():
    found = _labels()
    assert found, "the .aux carries no \\newlabel entries"
    return found


@pytest.mark.parametrize("path,label,spellings", CITATIONS,
                         ids=[f"{p}:{l}" for p, l, _ in CITATIONS])
def test_quoted_equation_number_matches_the_manuscript(path, label, spellings,
                                                       labels):
    assert label in labels, f"label {label} is not defined in the manuscript"
    number = labels[label]

    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as fh:
        source = fh.read()

    for spelling in spellings:
        # The spelling encodes the expected number; check it is the live one.
        quoted = re.search(r"\((\d+)\)", spelling).group(1)
        assert quoted == number, (
            f"{path} cites {label} as ({quoted}), but the manuscript now "
            f"numbers it ({number}). Update the comment, or the citation "
            f"points a reader at the wrong equation.")
        assert spelling in source, (
            f"{path} no longer contains {spelling!r}; this cross-reference "
            f"check has gone stale and should be updated or removed.")


def test_every_cited_label_exists(labels):
    missing = sorted({l for _, l, _ in CITATIONS if l not in labels})
    assert not missing, f"labels cited from code but absent from the paper: {missing}"
