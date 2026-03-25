from __future__ import annotations

from typing import Callable

from PySide6 import QtWidgets

__all__ = ["FloatingDockContainer"]


class FloatingDockContainer(QtWidgets.QWidget):
    """Wrap dock content with a floating-only re-dock affordance."""

    def __init__(
        self,
        *,
        content: QtWidgets.QWidget,
        dock: QtWidgets.QDockWidget,
        on_dock_back: Callable[[], None],
        on_reset_layout: Callable[[], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._dock = dock
        self._on_dock_back = on_dock_back
        self._on_reset_layout = on_reset_layout
        self._content = content

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._floating_banner = QtWidgets.QFrame(self)
        self._floating_banner.setObjectName("floatingDockBanner")
        self._floating_banner.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self._floating_banner.setVisible(False)

        banner_layout = QtWidgets.QHBoxLayout(self._floating_banner)
        banner_layout.setContentsMargins(8, 6, 8, 6)
        banner_layout.setSpacing(8)

        self._message_label = QtWidgets.QLabel(self._floating_banner)
        self._message_label.setWordWrap(True)
        banner_layout.addWidget(self._message_label, stretch=1)

        self._dock_back_button = QtWidgets.QPushButton("Dock Back", self._floating_banner)
        self._dock_back_button.setObjectName("dockBackButton")
        self._dock_back_button.clicked.connect(self._dock_back)
        banner_layout.addWidget(self._dock_back_button)

        self._reset_layout_button = QtWidgets.QPushButton("Restore Default Layout", self._floating_banner)
        self._reset_layout_button.setObjectName("resetLayoutButton")
        self._reset_layout_button.setToolTip("Restore all panels to the default workspace layout.")
        self._reset_layout_button.clicked.connect(self._on_reset_layout)
        banner_layout.addWidget(self._reset_layout_button)

        layout.addWidget(self._floating_banner)
        layout.addWidget(self._content, stretch=1)

        self._dock.topLevelChanged.connect(self._sync_floating_banner)
        self._dock.windowTitleChanged.connect(self._update_message)
        self._update_message(self._dock.windowTitle())
        self._sync_floating_banner(self._dock.isFloating())

    def _dock_back(self) -> None:
        self._on_dock_back()

    def _sync_floating_banner(self, floating: bool) -> None:
        self._floating_banner.setVisible(bool(floating))

    def _update_message(self, title: str) -> None:
        panel_title = str(title).strip() or "This panel"
        self._message_label.setText(
            f"{panel_title} is floating. Drag the title bar to snap it back to a side, or use Dock Back. "
            "Restore Default Layout returns all panels to the default workspace layout."
        )
