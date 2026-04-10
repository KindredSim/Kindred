"""Tests for MechanismHighlighter keyword/pattern alignment with DSL parser."""

from __future__ import annotations

import pytest
from PySide6 import QtGui

from kindred.gui.widgets.mechanism_highlighter import MechanismHighlighter

pytestmark = [pytest.mark.gui]

# Expected RGB values from MechanismHighlighter format definitions
KEYWORD = (147, 112, 219)
SPECIES = (70, 130, 180)
OPERATOR = (220, 20, 60)
RATE = (34, 139, 34)
NUMBER = (255, 140, 0)
COMMENT = (128, 128, 128)
ENERGY = (0, 206, 209)
INITIAL = (199, 21, 133)


def _color_at(doc: QtGui.QTextDocument, hl: MechanismHighlighter, pos: int):
    """Return (r, g, b) of the format applied at document position *pos*, or None."""
    hl.rehighlight()
    block = doc.findBlock(pos)
    if not block.isValid():
        return None
    local = pos - block.position()
    for fr in block.layout().formats():
        if fr.start <= local < fr.start + fr.length:
            c = fr.format.foreground().color()
            return (c.red(), c.green(), c.blue())
    return None


@pytest.fixture
def doc_and_hl(qt_app):
    doc = QtGui.QTextDocument()
    hl = MechanismHighlighter(doc)
    return doc, hl


# ---------------------------------------------------------------------------
# Valid keywords — should be highlighted as KEYWORD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word", [
    "reaction", "equilibrium", "init", "initial",
    "state", "edge", "temp_const", "temp_step", "temp_response",
])
def test_valid_keyword_highlighted(doc_and_hl, word):
    doc, hl = doc_and_hl
    doc.setPlainText(word)
    assert _color_at(doc, hl, 0) == KEYWORD


def test_time_keyword_highlighted(doc_and_hl):
    doc, hl = doc_and_hl
    doc.setPlainText("time")
    assert _color_at(doc, hl, 0) == KEYWORD


# ---------------------------------------------------------------------------
# Removed keywords — must NOT be highlighted as KEYWORD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word", [
    "reversible", "irreversible", "conditions", "units", "temperature",
])
def test_removed_keyword_not_highlighted(doc_and_hl, word):
    doc, hl = doc_and_hl
    doc.setPlainText(word)
    assert _color_at(doc, hl, 0) != KEYWORD


# ---------------------------------------------------------------------------
# Valid energy terms — should be highlighted as ENERGY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("term", [
    "Ea=", "dG_act=", "dG_eq=", "A=", "T=", "energy=",
])
def test_valid_energy_term_highlighted(doc_and_hl, term):
    doc, hl = doc_and_hl
    doc.setPlainText(term)
    assert _color_at(doc, hl, 0) == ENERGY


@pytest.mark.parametrize("term", ["\u0394G\u2021=", "\u0394G\u00b0="])
def test_unicode_energy_term_highlighted(doc_and_hl, term):
    doc, hl = doc_and_hl
    doc.setPlainText(term)
    assert _color_at(doc, hl, 0) == ENERGY


@pytest.mark.parametrize("term", ["C0=", "C\u00b0=", "degeneracy="])
def test_new_energy_term_highlighted(doc_and_hl, term):
    doc, hl = doc_and_hl
    doc.setPlainText(term)
    assert _color_at(doc, hl, 0) == ENERGY


# ---------------------------------------------------------------------------
# Removed energy terms — must NOT be highlighted as ENERGY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("term", [
    "activation_energy=", "enthalpy=", "entropy=",
])
def test_removed_energy_term_not_highlighted(doc_and_hl, term):
    doc, hl = doc_and_hl
    doc.setPlainText(term)
    assert _color_at(doc, hl, 0) != ENERGY


# ---------------------------------------------------------------------------
# Valid rate patterns — should be highlighted as RATE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("term", ["k=", "kf=", "kr=", "K="])
def test_valid_rate_pattern_highlighted(doc_and_hl, term):
    doc, hl = doc_and_hl
    doc.setPlainText(term)
    assert _color_at(doc, hl, 0) == RATE


@pytest.mark.parametrize("term", ["Keq=", "K_eq=", "kEQ=", "KF=", "Kr="])
def test_equilibrium_alias_and_case_insensitive_rate_patterns_highlighted(doc_and_hl, term):
    doc, hl = doc_and_hl
    doc.setPlainText(term)
    assert _color_at(doc, hl, 0) == RATE


@pytest.mark.parametrize("term", ["kappa=", "\u03ba="])
def test_new_rate_pattern_highlighted(doc_and_hl, term):
    doc, hl = doc_and_hl
    doc.setPlainText(term)
    assert _color_at(doc, hl, 0) == RATE


# ---------------------------------------------------------------------------
# Removed / tightened rate patterns — must NOT be highlighted as RATE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("term", ["rate=", "k1=", "kf2=", "K1="])
def test_removed_rate_pattern_not_highlighted(doc_and_hl, term):
    doc, hl = doc_and_hl
    doc.setPlainText(term)
    assert _color_at(doc, hl, 0) != RATE


# ---------------------------------------------------------------------------
# Comment priority — comment format must override everything else
# ---------------------------------------------------------------------------

def test_comment_overrides_keywords(doc_and_hl):
    doc, hl = doc_and_hl
    text = "# reaction A -> B"
    doc.setPlainText(text)
    # Every character should be COMMENT
    for i in range(len(text)):
        assert _color_at(doc, hl, i) == COMMENT, (
            f"pos {i} (char {text[i]!r}) should be COMMENT"
        )


def test_inline_comment_overrides_rest(doc_and_hl):
    doc, hl = doc_and_hl
    text = "reaction: A -> B # k=1.0"
    doc.setPlainText(text)
    hash_pos = text.index("#")
    # Before #: "reaction" should be KEYWORD
    assert _color_at(doc, hl, 0) == KEYWORD
    # After #: everything should be COMMENT
    for i in range(hash_pos, len(text)):
        assert _color_at(doc, hl, i) == COMMENT, (
            f"pos {i} (char {text[i]!r}) should be COMMENT"
        )


# ---------------------------------------------------------------------------
# Existing patterns still work — species, numbers, operators, initials
# ---------------------------------------------------------------------------

def test_species_highlighted(doc_and_hl):
    doc, hl = doc_and_hl
    doc.setPlainText("ATP")
    assert _color_at(doc, hl, 0) == SPECIES


def test_number_highlighted(doc_and_hl):
    doc, hl = doc_and_hl
    doc.setPlainText("1.5e-3")
    assert _color_at(doc, hl, 0) == NUMBER


def test_operator_highlighted(doc_and_hl):
    doc, hl = doc_and_hl
    text = "A -> B"
    doc.setPlainText(text)
    arrow_pos = text.index("->")
    assert _color_at(doc, hl, arrow_pos) == OPERATOR


def test_fat_arrow_operator_highlighted(doc_and_hl):
    doc, hl = doc_and_hl
    text = "A => B"
    doc.setPlainText(text)
    arrow_pos = text.index("=>")
    assert _color_at(doc, hl, arrow_pos) == OPERATOR


def test_initial_condition_highlighted(doc_and_hl):
    doc, hl = doc_and_hl
    doc.setPlainText("[A]=1.0")
    assert _color_at(doc, hl, 0) == INITIAL


# ---------------------------------------------------------------------------
# Multi-token priority: specific patterns override species catch-all
# ---------------------------------------------------------------------------

def test_multi_token_priority(doc_and_hl):
    doc, hl = doc_and_hl
    text = "reaction: A -> B ; Ea=50"
    doc.setPlainText(text)
    # "reaction" → KEYWORD (not default)
    assert _color_at(doc, hl, 0) == KEYWORD
    # "A" → SPECIES
    a_pos = text.index("A")
    assert _color_at(doc, hl, a_pos) == SPECIES
    # "->" → OPERATOR
    arrow_pos = text.index("->")
    assert _color_at(doc, hl, arrow_pos) == OPERATOR
    # "Ea=" → ENERGY (overrides species catch-all on "E")
    ea_pos = text.index("Ea=")
    assert _color_at(doc, hl, ea_pos) == ENERGY
    # "50" → NUMBER
    num_pos = text.index("50")
    assert _color_at(doc, hl, num_pos) == NUMBER


def test_algebra_section_does_not_apply_mechanism_rate_highlighting(doc_and_hl):
    doc, hl = doc_and_hl
    text = "\n".join(
        [
            "equilibrium: A <-> B ; K=1 ; Keq=2 ; K_eq=3 ; KF=4 ; Kr=5 ; kappa=6",
            "# Algebra",
            "param K = 1",
            "param KF = 2",
            "param Kr = 3",
        ]
    )
    doc.setPlainText(text)

    for token in ["K=", "Keq=", "K_eq=", "KF=", "Kr=", "kappa="]:
        assert _color_at(doc, hl, text.index(token)) == RATE

    for token in ["K", "KF", "Kr"]:
        pos = text.rindex(f"param {token}") + len("param ")
        assert _color_at(doc, hl, pos) != RATE


def test_mechanism_rate_highlighting_resumes_after_algebra_section_boundary(doc_and_hl):
    doc, hl = doc_and_hl
    text = "\n".join(
        [
            "equilibrium: A <-> B ; K=1",
            "# Algebra",
            "param K = 1",
            "# Notes",
            "equilibrium: B <-> C ; Keq=2",
        ]
    )
    doc.setPlainText(text)

    algebra_pos = text.index("param K") + len("param ")
    assert _color_at(doc, hl, algebra_pos) != RATE

    resumed_pos = text.rindex("Keq=")
    assert _color_at(doc, hl, resumed_pos) == RATE
