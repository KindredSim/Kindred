"""Shared slider dragging, press state, and numeric-editor interaction."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets


@dataclass(slots=True)
class SliderPress:
    """Single physical press record owned by its Slider widget."""

    start_position: int
    moved: bool = False


class NumericValueEdit(QtWidgets.QLineEdit):
    """A raw numeric draft editor with local Escape/focus notifications."""

    editStarted = QtCore.Signal()
    editEscape = QtCore.Signal()

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        self.editStarted.emit()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.editEscape.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class FineDragSlider(QtWidgets.QSlider):
    """Keep native slider signals while attenuating Fine pointer movement."""

    def __init__(
        self,
        orientation: QtCore.Qt.Orientation,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(orientation, parent)
        # Observe a missing release when the pointer returns without a button.
        self.setMouseTracking(True)
        self._fine_mode = False
        self._pointer_origin: QtCore.QPoint | None = None
        self._drag_position = 0.0

    def set_fine_mode(self, enabled: bool) -> None:
        """Change sensitivity quietly, rebasing any ongoing handle gesture."""
        if self._fine_mode == enabled:
            return
        self._fine_mode = enabled
        if self.isSliderDown():
            self._drag_position = float(self.sliderPosition())
            self._pointer_origin = QtGui.QCursor.pos()
            if enabled:
                self._attach_cursor()

    def _slider_geometry(
        self,
    ) -> tuple[QtWidgets.QStyleOptionSlider, QtCore.QRect, QtCore.QRect, int]:
        option = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            option,
            QtWidgets.QStyle.SubControl.SC_SliderGroove,
            self,
        )
        handle = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            option,
            QtWidgets.QStyle.SubControl.SC_SliderHandle,
            self,
        )
        return option, groove, handle, max(1, groove.width() - handle.width())

    def _attach_cursor(self) -> None:
        _, _, handle, _ = self._slider_geometry()
        self._pointer_origin = self.mapToGlobal(handle.center())
        if QtGui.QCursor.pos() != self._pointer_origin:
            QtGui.QCursor.setPos(self._pointer_origin)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        # The wheel scrolls the Sliders pane; it never changes a value.
        event.ignore()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self.orientation() == QtCore.Qt.Orientation.Horizontal
        ):
            option, groove, handle, span = self._slider_geometry()
            if not handle.contains(event.position().toPoint()):
                position = int(
                    round(event.position().x() - groove.x() - handle.width() / 2)
                )
                self.setValue(
                    QtWidgets.QStyle.sliderValueFromPosition(
                        self.minimum(), self.maximum(),
                        max(0, min(position, span)), span, option.upsideDown,
                    )
                )
                event.accept()
                return
        super().mousePressEvent(event)
        if self._fine_mode and self.isSliderDown():
            self._drag_position = float(self.sliderPosition())
            self._attach_cursor()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self.isSliderDown():
            event.ignore()
            return
        if not event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.end_drag()
            event.accept()
            return
        if self._pointer_origin is None:
            super().mouseMoveEvent(event)
            return

        # Read the OS pointer, not queued coordinates from before a cursor warp.
        pointer = QtGui.QCursor.pos()
        delta = pointer.x() - self._pointer_origin.x()
        option, _, _, span = self._slider_geometry()
        movement = delta * (self.maximum() - self.minimum()) / span
        if self._fine_mode:
            movement /= 10.0
        self._drag_position = max(
            self.minimum(), min(
                self.maximum(),
                self._drag_position + (-movement if option.upsideDown else movement),
            ),
        )
        position = round(self._drag_position)
        self._pointer_origin = pointer
        self.setSliderPosition(position)
        # Value publication can synchronously reset, hide, or rebuild the row.
        if self._pointer_origin is not None and self.isSliderDown():
            if self.sliderPosition() != position:
                self._drag_position = float(self.sliderPosition())
            if self._fine_mode:
                self._attach_cursor()
        event.accept()

    def end_drag(self) -> None:
        """Settle native handle state; callers may block signals for retirement."""
        self._pointer_origin = None
        self._drag_position = 0.0
        # QSlider keeps pressedControl separately from QAbstractSlider's down
        # flag. Its release handler clears both, including on interruption.
        position = QtCore.QPointF(self.rect().center())
        release = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonRelease,
            position,
            QtCore.QPointF(self.mapToGlobal(position.toPoint())),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        super().mouseReleaseEvent(release)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.end_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def event(self, event: QtCore.QEvent) -> bool:
        if self.isSliderDown() and (
            event.type() in {
                QtCore.QEvent.Type.UngrabMouse,
                QtCore.QEvent.Type.WindowDeactivate,
                QtCore.QEvent.Type.Hide,
            }
            or (
                event.type() == QtCore.QEvent.Type.EnabledChange
                and not self.isEnabled()
            )
        ):
            self.end_drag()
        return super().event(event)
