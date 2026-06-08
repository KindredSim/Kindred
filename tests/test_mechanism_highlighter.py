from __future__ import annotations

import pytest
from PySide6 import QtGui

from kindred.gui.widgets.mechanism_highlighter import MechanismHighlighter

pytestmark = pytest.mark.gui


def _has_rate_highlight(token: str) -> bool:
    text = f"equilibrium: A <-> B ; kf=1.0 ; {token}2.0"
    document = QtGui.QTextDocument()
    highlighter = MechanismHighlighter(document)
    document.setPlainText(text)
    highlighter.rehighlight()

    token_start = text.index(token)
    token_end = token_start + len(token)
    expected = highlighter.formats["rate"]
    for span in document.firstBlock().layout().formats():
        span_start = int(span.start)
        span_end = span_start + int(span.length)
        if span_start >= token_end or span_end <= token_start:
            continue
        if (
            span.format.foreground().color() == expected.foreground().color()
            and span.format.fontWeight() == expected.fontWeight()
        ):
            return True
    return False


@pytest.mark.parametrize("token", ["Keq=", "keq=", "KEQ="])
def test_highlighter_marks_exact_keq_key_as_valid_rate_syntax(token):
    assert _has_rate_highlight(token)


@pytest.mark.parametrize("token", ["K=", "K_eq=", "k_eq="])
def test_highlighter_does_not_mark_legacy_keq_aliases_as_valid_rate_syntax(token):
    assert not _has_rate_highlight(token)
