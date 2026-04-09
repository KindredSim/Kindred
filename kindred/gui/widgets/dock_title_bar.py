# kindred/gui/widgets/dock_title_bar.py
"""Custom dock widget title bar with minimize/restore and close buttons."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

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
        self._drag_press_local: QtCore.QPoint | None = None
        self._drag_press_global: QtCore.QPoint | None = None
        self._drag_active = False
        self._dock_minimum_width_before_minimize: int | None = None
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

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
        if self._dock_minimum_width_before_minimize is not None:
            self._dock.setMinimumWidth(int(self._dock_minimum_width_before_minimize))
        self._dock_minimum_width_before_minimize = None
        self.updateGeometry()

    def _toggle_minimize(self) -> None:
        if self._minimized:
            self.restore()
        else:
            self._minimized = True
            self._minimize_btn.setText(_RESTORE_GLYPH)
            self._dock_minimum_width_before_minimize = int(self._dock.minimumWidth())
            self._dock.setMinimumWidth(
                max(
                    int(self._dock_minimum_width_before_minimize),
                    int(self.minimumSizeHint().width()),
                )
            )
            content = self._dock.widget()
            if content is not None:
                content.setVisible(False)
            self.updateGeometry()

    def _close_dock(self) -> None:
        self._dock.close()

    def _on_visibility_changed(self, visible: bool) -> None:
        if visible and self._minimized:
            self.restore()

    def minimumSizeHint(self) -> QtCore.QSize:
        layout = self.layout()
        margins = layout.contentsMargins() if isinstance(layout, QtWidgets.QLayout) else QtCore.QMargins()
        spacing = int(layout.spacing()) if isinstance(layout, QtWidgets.QLayout) else 0
        label_hint = self._title_label.minimumSizeHint()
        button_width = self._minimize_btn.sizeHint().width() + self._close_btn.sizeHint().width()
        width = margins.left() + label_hint.width() + button_width + margins.right() + (spacing * 2)
        height = margins.top() + max(
            label_hint.height(),
            self._minimize_btn.sizeHint().height(),
            self._close_btn.sizeHint().height(),
        ) + margins.bottom()
        return QtCore.QSize(max(1, width), max(1, height))

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSizeHint()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self._drag_target_allowed(event.position().toPoint())
        ):
            self._drag_press_local = event.position().toPoint()
            self._drag_press_global = event.globalPosition().toPoint()
            self._drag_active = False
            self._forward_mouse_event_to_dock(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            self._drag_press_local is None
            or self._drag_press_global is None
            or not (event.buttons() & QtCore.Qt.MouseButton.LeftButton)
        ):
            super().mouseMoveEvent(event)
            return
        if not self._drag_active:
            delta = event.globalPosition().toPoint() - self._drag_press_global
            if delta.manhattanLength() < QtWidgets.QApplication.startDragDistance():
                self._forward_mouse_event_to_dock(event)
                event.accept()
                return
            self._drag_active = True
        if self._forward_mouse_event_to_dock(event):
            event.accept()
            return
        if self._dock.isFloating():
            self._dock.move(event.globalPosition().toPoint() - self._drag_press_local)
            event.accept()
            return
        self._dock.setFloating(True)
        self._dock.move(event.globalPosition().toPoint() - self._drag_press_local)
        event.accept()
        return

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_press_local is not None and self._drag_target_allowed(event.position().toPoint()):
            self._forward_mouse_event_to_dock(event)
        self._drag_press_local = None
        self._drag_press_global = None
        self._drag_active = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self._drag_target_allowed(event.position().toPoint())
        ):
            self._dock.setFloating(not self._dock.isFloating())
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _drag_target_allowed(self, local_pos: QtCore.QPoint) -> bool:
        target = self.childAt(local_pos)
        while target is not None and target is not self:
            if target in {self._minimize_btn, self._close_btn}:
                return False
            target = target.parentWidget()
        return True

    def _forward_mouse_event_to_dock(self, event: QtGui.QMouseEvent) -> bool:
        dock_local = QtCore.QPointF(self._dock.mapFromGlobal(event.globalPosition().toPoint()))
        forwarded = QtGui.QMouseEvent(
            event.type(),
            dock_local,
            event.globalPosition(),
            event.button(),
            event.buttons(),
            event.modifiers(),
        )
        QtWidgets.QApplication.sendEvent(self._dock, forwarded)
        return bool(forwarded.isAccepted())
