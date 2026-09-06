# kindred/gui/widgets/mechanism_highlighter.py
"""Syntax highlighter for mechanism DSL."""

from __future__ import annotations

import re
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.theme_manager import semantic_color

from kindred.core.simulator.algebra_section import is_algebra_line

__all__ = ["MechanismHighlighter"]


class MechanismHighlighter(QtGui.QSyntaxHighlighter):
    """
    Syntax highlighter for Kindred mechanism DSL.

    Highlights:
    - Keywords (reaction, equilibrium, time, etc.) - bold purple
    - Species names (A, B, ATP, etc.) - blue
    - Operators (->, =>, <->, <=>, +) - muted in light mode, red in dark mode
    - Rate constants (k=, kf=, kr=, Keq=, local kappa=) - green
    - Numbers (1.0, 1e-5, etc.) - orange
    - Comments (#...) - gray
    - Energy terms (Ea=, dG_act=, etc.) - cyan
    - Initial conditions ([A]=) - magenta

    Example:
        highlighter = MechanismHighlighter(text_edit.document())
    """

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        """
        Initialize syntax highlighter.

        Parameters
        ----------
        parent : QObject, optional
            Parent object (usually the document to highlight)
        """
        super().__init__(parent)

        self.formats = {}
        for role in ("keyword", "species", "operator", "rate", "number", "comment", "energy", "initial"):
            text_format = QtGui.QTextCharFormat()
            if role in {"keyword", "operator", "rate", "energy", "initial"}:
                text_format.setFontWeight(QtGui.QFont.Bold)
            self.formats[role] = text_format

        # Define highlighting rules (order matters — setFormat is last-wins,
        # so broad catch-alls go first and specific patterns override them).
        self.rules = []

        # 1. Species names (broad catch-all, lowest effective priority)
        species_pattern = r'\b[A-Z][A-Za-z0-9_]*\b'
        self.rules.append((re.compile(species_pattern), 'species', True))

        # 2. Numbers (1.0, 1e-5, .5, etc.)
        number_pattern = r'\b\d+\.?\d*(?:[eE][+-]?\d+)?\b|\.\d+(?:[eE][+-]?\d+)?\b'
        self.rules.append((re.compile(number_pattern), 'number', True))

        # 3. Rate constants (k=, kf=, kr=, Keq=, local kappa=)
        rate_patterns = [
            (r'\bk\s*=', 0),                           # k=
            (r'\bk[fr]\s*=', re.IGNORECASE),           # kf=, kr=
            (r'\bKeq\s*=', re.IGNORECASE),             # Keq=
            (r'\bkappa\s*=', re.IGNORECASE),           # kappa=
        ]
        for pattern, flags in rate_patterns:
            self.rules.append((re.compile(pattern, flags), 'rate', False))

        # 4. Energy/thermodynamic terms (Ea=, dG_act=, dG_eq=, A=, T=, etc.)
        energy_terms = [
            r'\bEa\s*=', r'\bdG_act\s*=', r'\bdG_eq\s*=', r'\bA\s*=',
            r'\bT\s*=', r'\benergy\s*=',
            r'\bC0\s*=', r'\bdegeneracy\s*=',
        ]
        for term in energy_terms:
            self.rules.append((re.compile(term, re.IGNORECASE), 'energy', True))

        # 5. Keywords (reaction, equilibrium, init, etc.)
        keywords = [
            'reaction', 'equilibrium',
            'init', 'initial', 'time',
            'temp_step', 'temp_response', 'state', 'edge', 'intervention',
        ]
        keyword_pattern = r'\b(' + '|'.join(keywords) + r')\b'
        self.rules.append((re.compile(keyword_pattern, re.IGNORECASE), 'keyword', False))

        # 6. Algebra keywords (param, let)
        algebra_keywords = ['param', 'let']
        algebra_keyword_pattern = r'\b(' + '|'.join(algebra_keywords) + r')\b'
        self.rules.append((re.compile(algebra_keyword_pattern, re.IGNORECASE), 'keyword', True))

        # 7. Operators (->, =>, <->, <=>, +)
        # Applied after rate/energy so arrow `=` is not consumed by `A=` patterns.
        operators = [
            r'<=>',  # Reversible (alternate)
            r'<->',  # Reversible
            r'=>',   # Forward (alternate)
            r'->',   # Forward
            r'\+',   # Addition (in reactions)
        ]
        for op in operators:
            self.rules.append((re.compile(op), 'operator', True))

        # 8. Initial conditions ([Species] = value)
        self.rules.append((re.compile(r'\[[A-Za-z_][A-Za-z0-9_]*\]\s*='), 'initial', True))

        # 9. Comments (applied last — highest effective priority)
        self.rules.append((re.compile(r'#[^\n]*'), 'comment', True))
        self.set_theme(dark_mode=QtWidgets.QApplication.palette().color(QtGui.QPalette.Text).lightness() > 128)

    def set_theme(self, *, dark_mode: bool) -> None:
        colors = {
            "keyword": ("#7040a0", "#c5a0ef"),
            "species": ("#246398", "#8bc5f2"),
            "number": ("#925000", "#ffc078"),
            "energy": ("#006a70", "#7ddde2"),
            "initial": ("#983172", "#ef9ed2"),
        }
        semantic_roles = {"comment": "muted", "rate": "success", "operator": "error" if dark_mode else "muted"}
        self.formats["operator"].setFontWeight(QtGui.QFont.Bold if dark_mode else QtGui.QFont.Normal)
        for role, text_format in self.formats.items():
            color = (
                semantic_color(semantic_roles[role], dark_mode)
                if role in semantic_roles
                else QtGui.QColor(colors[role][int(dark_mode)])
            )
            text_format.setForeground(color)
        self.rehighlight()

    def highlightBlock(self, text: str):
        """
        Highlight a block of text.

        This is called automatically by Qt for each line of text.

        Parameters
        ----------
        text : str
            Text to highlight
        """
        algebra_line = is_algebra_line(text)
        self.setCurrentBlockState(0)

        # Apply all rules in order
        for pattern, format_name, applies_in_algebra in self.rules:
            if algebra_line and not applies_in_algebra:
                continue
            format_obj = self.formats[format_name]

            # Find all matches
            for match in pattern.finditer(text):
                start = match.start()
                length = match.end() - match.start()
                self.setFormat(start, length, format_obj)
