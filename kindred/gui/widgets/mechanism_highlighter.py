# kindred/gui/widgets/mechanism_highlighter.py
"""Syntax highlighter for mechanism DSL."""

from __future__ import annotations

import re
from typing import Optional

from PySide6 import QtCore, QtGui

__all__ = ["MechanismHighlighter"]


class MechanismHighlighter(QtGui.QSyntaxHighlighter):
    """
    Syntax highlighter for Kindred mechanism DSL.

    Highlights:
    - Keywords (reaction, equilibrium, reversible, etc.) - bold purple
    - Species names (A, B, ATP, etc.) - blue
    - Operators (->, <->, <=>, <-, =) - red bold
    - Rate constants (k=, kf=, kr=, K=) - green
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

        # Define highlighting rules (order matters!)
        self.rules = []

        # 1. Comments (highest priority - once in comment, nothing else matters)
        self.rules.append((re.compile(r'#[^\n]*'), 'comment'))

        # 2. Initial conditions ([Species] = value)
        self.rules.append((re.compile(r'\[[A-Za-z_][A-Za-z0-9_]*\]\s*='), 'initial'))

        # 3. Keywords (reaction, equilibrium, reversible, init, etc.)
        keywords = [
            'reaction', 'equilibrium', 'reversible', 'irreversible',
            'init', 'initial', 'conditions', 'units', 'temperature',
            'temp_const', 'temp_step', 'temp_response', 'state', 'edge'
        ]
        keyword_pattern = r'\b(' + '|'.join(keywords) + r')\b'
        self.rules.append((re.compile(keyword_pattern, re.IGNORECASE), 'keyword'))

        # 4. Energy/thermodynamic terms (Ea=, dG_act=, dG_eq=, A=, T=)
        energy_terms = [
            r'\bEa\s*=', r'\bdG_act\s*=', r'\bdG_eq\s*=', r'\bA\s*=',
            r'\bactivation_energy\s*=', r'\benthalpy\s*=', r'\bentropy\s*=',
            r'\bT\s*=', r'\benergy\s*=', r'\bΔG‡\s*=', r'\bΔG°\s*='
        ]
        for term in energy_terms:
            self.rules.append((re.compile(term, re.IGNORECASE), 'energy'))

        # 5. Rate constants (k=, kf=, kr=, K=, k1=, etc.)
        rate_patterns = [
            r'\bk[fr]?\d*\s*=',  # k=, kf=, kr=, k1=, kf2=, etc.
            r'\bK\d*\s*=',        # K=, K1=, K2=, etc.
            r'\brate\s*=',
        ]
        for pattern in rate_patterns:
            self.rules.append((re.compile(pattern), 'rate'))

        # 6. Operators (->, <->, <=>, <-, =, +)
        operators = [
            r'<=>',  # Reversible (alternate)
            r'<->',  # Reversible
            r'->',   # Forward
            r'<-',   # Backward (rare)
            r'\+',   # Addition (in reactions)
        ]
        for op in operators:
            self.rules.append((re.compile(op), 'operator'))

        # 7. Numbers (1.0, 1e-5, .5, etc.)
        # Match scientific notation and decimals
        number_pattern = r'\b\d+\.?\d*(?:[eE][+-]?\d+)?\b|\.\d+(?:[eE][+-]?\d+)?\b'
        self.rules.append((re.compile(number_pattern), 'number'))

        # 8. Species names (A, B, ATP, H2O, etc.)
        # Match capitalized words and chemical formulas
        species_pattern = r'\b[A-Z][A-Za-z0-9_]*\b'
        self.rules.append((re.compile(species_pattern), 'species'))

    def highlightBlock(self, text: str):
        """
        Highlight a block of text.

        This is called automatically by Qt for each line of text.

        Parameters
        ----------
        text : str
            Text to highlight
        """
        # Apply all rules in order
        for pattern, format_name in self.rules:
            format_obj = self.formats[format_name]

            # Find all matches
            for match in pattern.finditer(text):
                start = match.start()
                length = match.end() - match.start()
                self.setFormat(start, length, format_obj)
