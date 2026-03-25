# kindred/gui/widgets/collapsible_section.py
"""Collapsible section widget for GUI panels."""

from __future__ import annotations

from typing import Optional

from PySide6 import QtWidgets

__all__ = ["CollapsibleSection"]


class CollapsibleSection(QtWidgets.QWidget):
    """
    Collapsible section with header and content area.

    Features:
    - Click header to expand/collapse
    - Visual arrow indicator (▼/▶)
    - Styled header button
    - Flexible content area

    Example:
        section = CollapsibleSection("Solver Settings")
        content = QtWidgets.QWidget()
        # ... populate content ...
        section.set_content_widget(content)
    """

    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None):
        """
        Initialize collapsible section.

        Parameters
        ----------
        title : str
            Header title text
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self._title = str(title)
        self._is_collapsed = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header button
        self._header = QtWidgets.QPushButton()
        self._header.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 6px;
                font-weight: bold;
            }
        """)
        self._header.clicked.connect(self.toggle)
        layout.addWidget(self._header)

        # Content container
        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._content)
        self._update_header_text()

    def set_content_widget(self, widget: QtWidgets.QWidget):
        """
        Set the content widget for this section.

        Clears any existing content first.

        Parameters
        ----------
        widget : QWidget
            Widget to display in content area
        """
        # Clear existing
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._content_layout.addWidget(widget)

    def toggle(self):
        """Toggle collapsed state."""
        self._is_collapsed = not self._is_collapsed
        self._content.setVisible(not self._is_collapsed)
        self._update_header_text()

    @property
    def is_collapsed(self) -> bool:
        """Check if section is currently collapsed."""
        return self._is_collapsed

    def set_collapsed(self, collapsed: bool):
        """
        Set collapsed state programmatically.

        Parameters
        ----------
        collapsed : bool
            True to collapse, False to expand
        """
        if self._is_collapsed != collapsed:
            self.toggle()

    def _update_header_text(self) -> None:
        arrow = "▶" if self._is_collapsed else "▼"
        self._header.setText(f"{arrow} {self._title}")
