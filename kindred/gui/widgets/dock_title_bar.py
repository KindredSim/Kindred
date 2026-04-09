# kindred/gui/widgets/dock_title_bar.py
"""Custom dock widget title bar with minimize/restore and close buttons."""

from __future__ import annotations

from PySide6 import QtWidgets

__all__ = ["DockTitleBar"]

_MINIMIZE_GLYPH = "\u2014"  # em dash
_RESTORE_GLYPH = "\u25a1"  # white square
_CLOSE_GLYPH = "\u00d7"  # multiplication sign


class DockTitleBar(QtWidgets.QWidget):
    """Compact title bar for QDockWidget with minimize/restore and close."""

    def __init__(
        self,
        title: str,
        dock: QtWidgets.QDockWidget,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dock = dock
        self._minimized = False

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 4, 2)
        layout.setSpacing(4)

        self._title_label = QtWidgets.QLabel(str(title))
        self._title_label.setObjectName("dockTitleLabel")
        font = self._title_label.font()
        font.setBold(True)
        self._title_label.setFont(font)
        layout.addWidget(self._title_label, stretch=1)

        self._minimize_btn = QtWidgets.QToolButton()
        self._minimize_btn.setObjectName("dockMinimizeButton")
        self._minimize_btn.setText(_MINIMIZE_GLYPH)
        self._minimize_btn.setAutoRaise(True)
        self._minimize_btn.setFixedSize(20, 20)
        self._minimize_btn.clicked.connect(self._toggle_minimize)
        layout.addWidget(self._minimize_btn)

        self._close_btn = QtWidgets.QToolButton()
        self._close_btn.setObjectName("dockCloseButton")
        self._close_btn.setText(_CLOSE_GLYPH)
        self._close_btn.setAutoRaise(True)
        self._close_btn.setFixedSize(20, 20)
        self._close_btn.clicked.connect(self._close_dock)
        layout.addWidget(self._close_btn)

        self._dock.visibilityChanged.connect(self._on_visibility_changed)

    def set_title(self, title: str) -> None:
        self._title_label.setText(str(title))

    def is_minimized(self) -> bool:
        return self._minimized

    def restore(self) -> None:
        if not self._minimized:
            return
        self._minimized = False
        self._minimize_btn.setText(_MINIMIZE_GLYPH)
        content = self._dock.widget()
        if content is not None:
            content.setVisible(True)

    def _toggle_minimize(self) -> None:
        if self._minimized:
            self.restore()
        else:
            self._minimized = True
            self._minimize_btn.setText(_RESTORE_GLYPH)
            content = self._dock.widget()
            if content is not None:
                content.setVisible(False)

    def _close_dock(self) -> None:
        self._dock.close()

    def _on_visibility_changed(self, visible: bool) -> None:
        if visible and self._minimized:
            self.restore()
