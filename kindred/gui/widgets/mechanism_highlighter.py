# kindred/gui/widgets/mechanism_highlighter.py
"""Syntax highlighter for mechanism DSL."""

from __future__ import annotations

import re
from typing import Optional

from PySide6 import QtCore, QtGui

from kindred.core.simulator.algebra_section import is_algebra_line

__all__ = ["MechanismHighlighter"]


class MechanismHighlighter(QtGui.QSyntaxHighlighter):
    """
    Syntax highlighter for Kindred mechanism DSL.

    Highlights:
    - Keywords (reaction, equilibrium, time, etc.) - bold purple
    - Species names (A, B, ATP, etc.) - blue
    - Operators (->, =>, <->, <=>, <-, +) - red bold
    - Rate constants (k=, kf=, kr=, K=, kappa=) - green
    - Numbers (1.0, 1e-5, etc.) - orange
    - Comments (#...) - gray italic
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

        # Define text formats
        self.formats = {}

        # Keywords (reaction, equilibrium, etc.)
        keyword_format = QtGui.QTextCharFormat()
        keyword_format.setForeground(QtGui.QColor(147, 112, 219))  # Medium purple
        keyword_format.setFontWeight(QtGui.QFont.Bold)
        self.formats['keyword'] = keyword_format

        # Species names (A, B, ATP, etc.)
        species_format = QtGui.QTextCharFormat()
        species_format.setForeground(QtGui.QColor(70, 130, 180))  # Steel blue
        self.formats['species'] = species_format

        # Operators (->, <->, <-, =)
        operator_format = QtGui.QTextCharFormat()
        operator_format.setForeground(QtGui.QColor(220, 20, 60))  # Crimson red
        operator_format.setFontWeight(QtGui.QFont.Bold)
        self.formats['operator'] = operator_format

        # Rate constants (k=, kf=, kr=, K=)
        rate_format = QtGui.QTextCharFormat()
        rate_format.setForeground(QtGui.QColor(34, 139, 34))  # Forest green
        rate_format.setFontWeight(QtGui.QFont.Bold)
        self.formats['rate'] = rate_format

        # Numbers (1.0, 1e-5, etc.)
        number_format = QtGui.QTextCharFormat()
        number_format.setForeground(QtGui.QColor(255, 140, 0))  # Dark orange
        self.formats['number'] = number_format

        # Comments (#...)
        comment_format = QtGui.QTextCharFormat()
        comment_format.setForeground(QtGui.QColor(128, 128, 128))  # Gray
        comment_format.setFontItalic(True)
        self.formats['comment'] = comment_format

        # Energy terms (Ea=, dG_act=, dG_eq=, A=)
        energy_format = QtGui.QTextCharFormat()
        energy_format.setForeground(QtGui.QColor(0, 206, 209))  # Dark turquoise
        energy_format.setFontWeight(QtGui.QFont.Bold)
        self.formats['energy'] = energy_format

        # Initial conditions ([A]=)
        initial_format = QtGui.QTextCharFormat()
        initial_format.setForeground(QtGui.QColor(199, 21, 133))  # Medium violet red
        initial_format.setFontWeight(QtGui.QFont.Bold)
        self.formats['initial'] = initial_format

        # Define highlighting rules (order matters — setFormat is last-wins,
        # so broad catch-alls go first and specific patterns override them).
        self.rules = []

        # 1. Species names (broad catch-all, lowest effective priority)
        species_pattern = r'\b[A-Z][A-Za-z0-9_]*\b'
        self.rules.append((re.compile(species_pattern), 'species', True))

        # 2. Numbers (1.0, 1e-5, .5, etc.)
        number_pattern = r'\b\d+\.?\d*(?:[eE][+-]?\d+)?\b|\.\d+(?:[eE][+-]?\d+)?\b'
        self.rules.append((re.compile(number_pattern), 'number', True))

        # 3. Rate constants (k=, kf=, kr=, K=, kappa=, κ=)
        rate_patterns = [
            (r'\bk[fr]?\s*=', re.IGNORECASE),          # k=, kf=, kr=
            (r'\bK(?:eq|_eq)?\s*=', re.IGNORECASE),    # K=, Keq=, K_eq=
            (r'\bkappa\s*=', re.IGNORECASE),           # kappa=
            (r'\bκ\s*=', 0),                           # κ= (Unicode kappa)
        ]
        for pattern, flags in rate_patterns:
            self.rules.append((re.compile(pattern, flags), 'rate', False))

        # 4. Energy/thermodynamic terms (Ea=, dG_act=, dG_eq=, A=, T=, etc.)
        energy_terms = [
            r'\bEa\s*=', r'\bdG_act\s*=', r'\bdG_eq\s*=', r'\bA\s*=',
            r'\bT\s*=', r'\benergy\s*=', r'\bΔG‡\s*=', r'\bΔG°\s*=',
            r'\bC0\s*=', r'\bC°\s*=', r'\bdegeneracy\s*=',
        ]
        for term in energy_terms:
            self.rules.append((re.compile(term, re.IGNORECASE), 'energy', True))

        # 5. Keywords (reaction, equilibrium, init, etc.)
        keywords = [
            'reaction', 'equilibrium',
            'init', 'initial', 'time',
            'temp_const', 'temp_step', 'temp_response', 'state', 'edge',
        ]
        keyword_pattern = r'\b(' + '|'.join(keywords) + r')\b'
        self.rules.append((re.compile(keyword_pattern, re.IGNORECASE), 'keyword', False))

        # 6. Operators (->, =>, <->, <=>, <-, +)
        # Applied after rate/energy so arrow `=` is not consumed by `A=` patterns.
        operators = [
            r'<=>',  # Reversible (alternate)
            r'<->',  # Reversible
            r'=>',   # Forward (alternate)
            r'->',   # Forward
            r'<-',   # Backward (rare)
            r'\+',   # Addition (in reactions)
        ]
        for op in operators:
            self.rules.append((re.compile(op), 'operator', True))

        # 7. Initial conditions ([Species] = value)
        self.rules.append((re.compile(r'\[[A-Za-z_][A-Za-z0-9_]*\]\s*='), 'initial', True))

        # 8. Comments (applied last — highest effective priority)
        self.rules.append((re.compile(r'#[^\n]*'), 'comment', True))

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
