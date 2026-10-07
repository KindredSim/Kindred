"""Widget-relative tutorial presentation; no lesson or application policy."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Callable
from weakref import WeakKeyDictionary

from PySide6 import QtCore, QtGui, QtWidgets
from pyqtgraph.Qt import isQObjectAlive


def scroll_into_view(widget: QtWidgets.QWidget) -> None:
    """Reveal a known, constructed target after its dock has settled layout."""

    def reveal():
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QtWidgets.QScrollArea):
                parent.ensureWidgetVisible(widget)
            if parent.isWindow():
                break
            parent = parent.parentWidget()

    # The context cancels the callback if the target is destroyed in the meantime.
    QtCore.QTimer.singleShot(0, widget, reveal)


@dataclass(frozen=True)
class Highlight:
    widget: QtWidgets.QWidget
    rect: Callable[[], QtCore.QRect] | None = None
    reveal: Callable[[], None] | None = None

    def visible_rect(self) -> QtCore.QRect:
        """Clip to visible ancestors, including scroll-area viewports."""
        if not isQObjectAlive(self.widget) or not self.widget.isVisible():
            return QtCore.QRect()
        local = self.widget.rect() if self.rect is None else self.rect()
        result = QtCore.QRect(self.widget.mapToGlobal(local.topLeft()), local.size())
        current = self.widget
        while current is not None:
            clip = QtCore.QRect(current.mapToGlobal(QtCore.QPoint()), current.size())
            result = result.intersected(clip)
            if current.isWindow():
                break
            current = current.parentWidget()
        return result


def menu_target(
    menu: QtWidgets.QMenu, action_matches: Callable | None = None
) -> Highlight:
    """Follow a submenu action through its owning menus, resolving on each refresh."""
    # In PySide6 6.7, dropping a temporary QAction.menu() owner can invalidate
    # the Python menu wrapper while its Qt object remains alive. Resolve child
    # menus through their actual Qt parent instead of borrowing that wrapper.
    if action_matches is not None:
        action = next((item for item in menu.actions() if action_matches(item)), None)
        if action is not None and menu.isVisible():
            return Highlight(menu, lambda: menu.actionGeometry(action))
        for child in menu.findChildren(
            QtWidgets.QMenu, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly
        ):
            action = next(
                (item for item in child.actions() if action_matches(item)), None
            )
            if action is not None:
                if child.isVisible():
                    return Highlight(child, lambda: child.actionGeometry(action))
                if menu.isVisible():
                    return Highlight(
                        menu, lambda: menu.actionGeometry(child.menuAction())
                    )
                break
    bar = menu.parentWidget()
    return Highlight(bar, lambda: bar.actionGeometry(menu.menuAction()))


class _Spotlight(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setObjectName("tutorialSpotlight")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.holes: tuple[QtCore.QRect, ...] = ()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        shade = QtGui.QPainterPath()
        shade.addRect(QtCore.QRectF(self.rect()))
        cutout = QtGui.QPainterPath()
        cutout.setFillRule(QtCore.Qt.FillRule.WindingFill)
        for hole in self.holes:
            cutout.addRoundedRect(QtCore.QRectF(hole), 8, 8)
        painter.fillPath(shade.subtracted(cutout), QtGui.QColor(0, 0, 0, 165))
        painter.setPen(
            QtGui.QPen(self.palette().color(QtGui.QPalette.ColorRole.Highlight), 3)
        )
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        for hole in self.holes:
            painter.drawRoundedRect(hole, 8, 8)


class _DragHandle(QtWidgets.QLabel):
    moveRequested = QtCore.Signal(QtCore.QPoint)

    def __init__(self, parent):
        super().__init__(parent)
        self._offset = None
        self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
        self.setToolTip("Drag to move the tutorial.")

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._offset = event.globalPosition().toPoint() - self.window().pos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._offset is not None:
            self.moveRequested.emit(event.globalPosition().toPoint() - self._offset)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._offset = None
        super().mouseReleaseEvent(event)


class _ResizeGrip(QtWidgets.QSizeGrip):
    sizeEdited = QtCore.Signal()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.ColorRole.WindowText))
        corner = self.rect().bottomRight() - QtCore.QPoint(2, 2)
        for offset in (4, 8, 12):
            painter.drawLine(
                corner - QtCore.QPoint(offset, 0), corner - QtCore.QPoint(0, offset)
            )

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.sizeEdited.emit()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.sizeEdited.emit()


class TutorialCard(QtWidgets.QFrame):
    previousRequested = QtCore.Signal()
    nextRequested = QtCore.Signal()
    exitRequested = QtCore.Signal()
    recoveryRequested = QtCore.Signal()
    skipRequested = QtCore.Signal()
    sizeEdited = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(
            parent, QtCore.Qt.WindowType.Tool | QtCore.Qt.WindowType.FramelessWindowHint
        )
        self.setObjectName("tutorialCard")
        self.setMouseTracking(True)
        self._resize_start = None
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAutoFillBackground(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setAccessibleName("Interactive tutorial")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetNoConstraint)
        outer.setContentsMargins(18, 12, 18, 12)
        header = QtWidgets.QHBoxLayout()
        self.progress = _DragHandle(self)
        header.addWidget(self.progress, 1)
        self.collapse_button = QtWidgets.QToolButton(self)
        self.collapse_button.setText("−")
        self.collapse_button.setToolTip("Minimize tutorial.")
        self.collapse_button.setAccessibleName("Minimize tutorial")
        header.addWidget(self.collapse_button)
        self.close_button = QtWidgets.QToolButton(self)
        self.close_button.setText("×")
        self.close_button.setToolTip("Close tutorial and keep your work.")
        self.close_button.setAccessibleName("Close tutorial")
        self.close_button.clicked.connect(self.exitRequested)
        header.addWidget(self.close_button)
        outer.addLayout(header)
        self.contents = QtWidgets.QWidget(self)
        outer.addWidget(self.contents, 1)
        layout = QtWidgets.QVBoxLayout(self.contents)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.title = _DragHandle(self)
        self.title.setWordWrap(True)
        body_font = self.font()
        body_font.setPointSizeF(body_font.pointSizeF() + 1.0)
        font = QtGui.QFont(body_font)
        font.setPointSizeF(body_font.pointSizeF() + 4.0)
        font.setBold(True)
        self.title.setFont(font)
        layout.addWidget(self.title)
        self.body = QtWidgets.QLabel(self)
        self.body.setFont(body_font)
        self.body.setWordWrap(True)
        self.body.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.body.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.body.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.scroll = scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.Box)
        scroll.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.body.setMargin(12)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setWidget(self.body)
        layout.addWidget(scroll, 1)
        self.link = QtWidgets.QLabel(self)
        self.link.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.link.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self.link.setOpenExternalLinks(True)
        self.link.setWordWrap(True)
        layout.addWidget(self.link)
        self.status = QtWidgets.QLabel(self)
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.recovery = QtWidgets.QPushButton(self)
        self.recovery.clicked.connect(self.recoveryRequested)
        layout.addWidget(self.recovery)
        row = QtWidgets.QHBoxLayout()
        self.skip_button = QtWidgets.QPushButton("Skip Step", self)
        self.skip_button.setToolTip("Move on without verifying this step.")
        self.previous = QtWidgets.QPushButton("Previous", self)
        self.next_button = QtWidgets.QPushButton("Next", self)
        self.skip_button.clicked.connect(self.skipRequested)
        self.previous.clicked.connect(self.previousRequested)
        self.next_button.clicked.connect(self.nextRequested)
        row.addWidget(self.skip_button)
        row.addStretch()
        row.addWidget(self.previous)
        row.addWidget(self.next_button)
        layout.addLayout(row)
        options = QtWidgets.QHBoxLayout()
        self.auto_advance = QtWidgets.QCheckBox("Advance Automatically", self)
        self.auto_advance.setChecked(True)
        self.auto_advance.setToolTip(
            "Advance after completing an action in this step. Reading, manual checks, and already-complete steps wait for Next."
        )
        layout.addWidget(self.auto_advance)
        self.highlight_checkbox = QtWidgets.QCheckBox("Highlight Current Step", self)
        self.highlight_checkbox.setChecked(True)
        self.reset_position = QtWidgets.QPushButton("Reset Position", self)
        self.resize_grip = _ResizeGrip(self)
        self.resize_grip.setFixedSize(22, 22)
        self.resize_grip.setToolTip("Drag to resize the tutorial.")
        options.addWidget(self.highlight_checkbox)
        options.addStretch()
        options.addWidget(self.reset_position)
        options.addWidget(self.resize_grip)
        layout.addLayout(options)
        for label in (self.progress, self.title):
            label.setTextFormat(QtCore.Qt.TextFormat.PlainText)

    def preferred_size(self) -> QtCore.QSize:
        """Fit the instruction document and controls, bounded later to the screen."""
        width = min(
            max(620, self.body.fontMetrics().averageCharWidth() * 62),
            self.screen().availableGeometry().width() - 24,
        )
        self.contents.layout().activate()
        self.layout().activate()
        chrome = (
            self.layout().totalHeightForWidth(width) - self.scroll.sizeHint().height()
        )
        margins = self.layout().contentsMargins()
        inner = self.contents.layout().contentsMargins()
        # QLabel.heightForWidth already includes the instruction text's padding.
        text_width = (
            width
            - margins.left()
            - margins.right()
            - inner.left()
            - inner.right()
            - 2 * self.scroll.frameWidth()
        )
        height = (
            chrome + self.body.heightForWidth(text_width) + 2 * self.scroll.frameWidth()
        )
        return QtCore.QSize(width, max(self.minimumHeight(), height))

    def set_collapsed(self, collapsed: bool) -> None:
        self.contents.setVisible(not collapsed)
        self.collapse_button.setText("□" if collapsed else "−")
        action = "Restore tutorial" if collapsed else "Minimize tutorial"
        self.collapse_button.setToolTip(action)
        self.collapse_button.setAccessibleName(action)
        self.setMinimumSize(0, 0) if collapsed else self.setMinimumSize(460, 340)
        self.layout().activate()

    def _edges(self, point):
        if self.contents.isHidden():
            return (False,) * 4
        return (
            point.x() < 8,
            point.x() >= self.width() - 8,
            point.y() < 8,
            point.y() >= self.height() - 8,
        )

    def mousePressEvent(self, event):
        edges = self._edges(event.position().toPoint())
        if event.button() == QtCore.Qt.MouseButton.LeftButton and any(edges):
            self._resize_start = (
                event.globalPosition().toPoint(),
                self.geometry(),
                edges,
            )
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resize_start is not None:
            start, original, (left, right, top, bottom) = self._resize_start
            delta = event.globalPosition().toPoint() - start
            rect = QtCore.QRect(original)
            if left:
                rect.setLeft(
                    min(
                        original.left() + delta.x(),
                        original.right() - self.minimumWidth() + 1,
                    )
                )
            if right:
                rect.setRight(
                    max(
                        original.right() + delta.x(),
                        original.left() + self.minimumWidth() - 1,
                    )
                )
            if top:
                rect.setTop(
                    min(
                        original.top() + delta.y(),
                        original.bottom() - self.minimumHeight() + 1,
                    )
                )
            if bottom:
                rect.setBottom(
                    max(
                        original.bottom() + delta.y(),
                        original.top() + self.minimumHeight() - 1,
                    )
                )
            self.setGeometry(rect)
            self.sizeEdited.emit()
            event.accept()
            return
        left, right, top, bottom = self._edges(event.position().toPoint())
        cursor = (
            QtCore.Qt.CursorShape.SizeFDiagCursor
            if (left and top) or (right and bottom)
            else QtCore.Qt.CursorShape.SizeBDiagCursor
            if (right and top) or (left and bottom)
            else QtCore.Qt.CursorShape.SizeHorCursor
            if left or right
            else QtCore.Qt.CursorShape.SizeVerCursor
            if top or bottom
            else QtCore.Qt.CursorShape.ArrowCursor
        )
        self.setCursor(cursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resize_start is not None:
            self._resize_start = None
            self.sizeEdited.emit()
        super().mouseReleaseEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.exitRequested.emit()
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # Only events delivered to the card navigate the tutorial. Never
        # intercept the application's editor, slider, menu or dialog keys.
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.exitRequested.emit()
        elif not self.contents.isVisible():
            if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Space):
                self.collapse_button.click()
            else:
                super().keyPressEvent(event)
        elif event.key() == QtCore.Qt.Key.Key_Left:
            self.previous.click()
        elif event.key() in (QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_Return):
            self.next_button.click()
        else:
            super().keyPressEvent(event)


class TutorialOverlay(QtCore.QObject):
    """One card with widget-relative spotlights in each target window."""

    changed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self._window = parent
        self.card = TutorialCard(parent)
        self._veils: dict[QtWidgets.QWidget, _Spotlight] = {}
        self._dialog: QtWidgets.QDialog | None = None
        self._positions = WeakKeyDictionary()
        self._expanded_size = QtCore.QSize()
        self._size_edited = False
        self._collapsed = False
        self._last_progress = None
        self._needs_placement = True
        self._needs_resize = True
        self.card.progress.moveRequested.connect(self._move_card)
        self.card.title.moveRequested.connect(self._move_card)
        self.card.resize_grip.sizeEdited.connect(self._remember_size)
        self.card.sizeEdited.connect(self._remember_size)
        self.card.collapse_button.clicked.connect(self._toggle_collapsed)
        self.card.reset_position.clicked.connect(self._reset_position)
        self.card.highlight_checkbox.toggled.connect(self.changed)
        self.card.auto_advance.toggled.connect(self.changed)
        self.card.set_collapsed(False)

    def _move_card(self, point: QtCore.QPoint) -> None:
        rect = self._bounded(QtCore.QRect(point, self.card.size()))
        self.card.move(rect.topLeft())
        parent = self.card.parentWidget()
        self._positions[parent] = parent.mapFromGlobal(rect.topLeft())

    def _remember_size(self) -> None:
        self._size_edited = True
        self.card.setGeometry(self._bounded(self.card.geometry()))
        self._expanded_size = self.card.size()
        parent = self.card.parentWidget()
        self._positions[parent] = parent.mapFromGlobal(self.card.pos())

    def _toggle_collapsed(self) -> None:
        self._collapsed = not self._collapsed
        self.card.set_collapsed(self._collapsed)
        self._needs_resize = True
        self.changed.emit()

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def expand(self) -> None:
        if self._collapsed:
            self._toggle_collapsed()

    def _reset_position(self) -> None:
        self._positions.pop(self.card.parentWidget(), None)
        self._needs_placement = True
        self.changed.emit()

    def _bounded(self, rect: QtCore.QRect) -> QtCore.QRect:
        screen = (
            QtWidgets.QApplication.screenAt(rect.center())
            or self.card.parentWidget().screen()
        )
        area = screen.availableGeometry().adjusted(12, 12, -12, -12)
        size = rect.size().boundedTo(area.size())
        return QtCore.QRect(
            max(area.left(), min(rect.x(), area.right() - size.width() + 1)),
            max(area.top(), min(rect.y(), area.bottom() - size.height() + 1)),
            size.width(),
            size.height(),
        )

    def _release_dialog(self, *_args) -> None:
        if self._dialog is not None:
            if isQObjectAlive(self._dialog):
                self._dialog.finished.disconnect(self._release_dialog)
                self._dialog.destroyed.disconnect(self._release_dialog)
            self._dialog = None
        self.card.hide()
        self.card.setParent(self._window, self.card.windowFlags())
        self._clear_veils()
        self._needs_placement = True

    def _clear_veils(self) -> None:
        for veil in self._veils.values():
            if isQObjectAlive(veil):
                veil.hide()
                veil.deleteLater()
        self._veils.clear()

    def hide(self) -> None:
        self.card.hide()
        for veil in self._veils.values():
            if isQObjectAlive(veil):
                veil.hide()

    def close(self) -> None:
        self._release_dialog()
        self._positions.clear()
        self._expanded_size = QtCore.QSize()
        self._size_edited = False
        self._collapsed = False
        self._last_progress = None
        self._needs_placement = self._needs_resize = True
        self.card.set_collapsed(False)
        self.card.highlight_checkbox.setChecked(True)
        self.card.auto_advance.setChecked(True)
        # A hidden Windows tool window can retain its former modal owner after
        # Qt reparenting. Release its native surface through Qt before that owner
        # dies; showing the retained card in a new lesson creates a fresh surface.
        self.card.destroy()

    def present(
        self,
        highlights: tuple[Highlight, ...],
        *,
        title: str,
        body: str,
        progress: str,
        status: str,
        complete: bool,
        previous: bool,
        final: bool,
        recovery: str = "",
        manual: bool = False,
        skipped: bool = False,
        link: tuple[str, str] | None = None,
    ) -> None:
        # Qt can destroy a target window between refreshes. Drop its presentation
        # references here; the lesson resolves a fresh target through its owner.
        highlights = tuple(item for item in highlights if isQObjectAlive(item.widget))
        app = QtWidgets.QApplication.instance()
        modal = app.activeModalWidget()
        if isinstance(modal, QtWidgets.QFileDialog):
            self.hide()
            return
        # Untaught confirmations remain fully operable. Their ordinary owner
        # decides the result; closing them returns guidance to the lesson.
        window = highlights[0].widget.window() if highlights else self._window
        if modal is not None and (
            not highlights
            or any(item.widget.window() is not modal for item in highlights)
        ):
            highlights = (Highlight(modal),)
            window = modal
            status = "Complete or cancel the open dialog to continue."
            complete = False
            recovery = ""
        popup = app.activePopupWidget()
        card_parent = (
            modal or self._window if isinstance(window, QtWidgets.QMenu) else window
        )
        parent_changed = self.card.parentWidget() is not card_parent
        if parent_changed:
            self._release_dialog()
            self.card.setParent(card_parent, self.card.windowFlags())
            if isinstance(card_parent, QtWidgets.QDialog):
                self._dialog = card_parent
                card_parent.finished.connect(self._release_dialog)
                card_parent.destroyed.connect(self._release_dialog)
        targets: dict[QtWidgets.QWidget, list[QtCore.QRect]] = {}
        target_rect = QtCore.QRect()
        for item in highlights:
            rect = item.visible_rect()
            if not rect.isEmpty():
                targets.setdefault(item.widget.window(), []).append(rect)
                target_rect = target_rect.united(rect)
        for host in tuple(self._veils):
            if host not in targets or not isQObjectAlive(host):
                veil = self._veils.pop(host)
                if isQObjectAlive(veil):
                    veil.hide()
                    veil.deleteLater()
        for host, rects in targets.items():
            veil = self._veils.get(host)
            if veil is None or not isQObjectAlive(veil):
                veil = self._veils[host] = _Spotlight(host)
            veil.setGeometry(host.rect())
            veil.holes = tuple(
                QtCore.QRect(host.mapFromGlobal(rect.topLeft()), rect.size()).adjusted(
                    -6, -6, 6, 6
                )
                for rect in rects
            )
            veil.setVisible(
                not self._collapsed and self.card.highlight_checkbox.isChecked()
            )
            if veil.isVisible():
                veil.raise_()
                veil.update()
        card = self.card
        card.link.setVisible(link is not None)
        link_text = (
            ""
            if link is None
            else f'<a href="{escape(link[1], quote=True)}">{escape(link[0])}</a>'
        )
        if card.link.text() != link_text:
            card.link.setText(link_text)
        compact_status = (
            "Skipped"
            if skipped
            else "Manual check"
            if manual
            else "Ready"
            if complete
            else ""
        )
        for label, text in (
            (card.title, title),
            (
                card.body,
                (
                    "<style>"
                    "p { margin-top: 0px; margin-bottom: 12px; }"
                    "h3 { font-size: "
                    f"{card.body.font().pointSizeF() + 2.0:g}pt; "
                    "margin-top: 16px; margin-bottom: 8px; }"
                    "ul, ol { margin-top: 0px; margin-bottom: 12px; margin-left: 22px; }"
                    "li { margin-top: 0px; margin-bottom: 6px; }"
                    "</style>" + body
                ),
            ),
            (
                card.progress,
                progress
                + (
                    f" · {compact_status}" if self._collapsed and compact_status else ""
                ),
            ),
            (card.status, status),
        ):
            if label.text() != text:
                label.setText(text)
        card.next_button.setEnabled(complete)
        card.next_button.setText(
            "Finish" if final else "Continue" if manual else "Next"
        )
        card.previous.setEnabled(previous)
        card.skip_button.setVisible(not final)
        card.recovery.setText(recovery)
        card.recovery.setVisible(bool(recovery))
        card.setPalette(card_parent.palette())
        step_changed = self._last_progress != progress
        self._last_progress = progress
        show = not card.isVisible()
        if not self._collapsed and (
            self._expanded_size.isEmpty() or (step_changed and not self._size_edited)
        ):
            self._expanded_size = card.preferred_size()
        size = card.sizeHint() if self._collapsed else self._expanded_size
        if card_parent in self._positions:
            point = card_parent.mapToGlobal(self._positions[card_parent])
            rect = self._bounded(QtCore.QRect(point, size))
            if card.geometry() != rect:
                card.setGeometry(rect)
        elif (
            self._needs_placement
            or parent_changed
            or (step_changed and not self._collapsed)
        ):
            self._place(target_rect, card_parent, popup, size)
        elif (
            self._needs_resize
            or (self._collapsed and card.size() != size)
            or self._bounded(card.geometry()) != card.geometry()
        ):
            card.setGeometry(self._bounded(QtCore.QRect(card.pos(), size)))
        self._needs_placement = self._needs_resize = False
        card.show()
        if step_changed:
            card.scroll.verticalScrollBar().setValue(0)
        if popup is None and (
            show or parent_changed or (step_changed and not self._collapsed)
        ):
            card.raise_()

    def _place(
        self,
        target: QtCore.QRect,
        parent: QtWidgets.QWidget,
        popup: QtWidgets.QWidget | None,
        size: QtCore.QSize,
    ) -> None:
        screen = parent.screen().availableGeometry().adjusted(12, 12, -12, -12)
        bounds = QtCore.QRect(
            parent.mapToGlobal(QtCore.QPoint()), parent.size()
        ).intersected(screen)
        avoid = target.adjusted(-12, -12, 12, 12)
        if popup is not None:
            avoid = avoid.united(
                QtCore.QRect(popup.mapToGlobal(QtCore.QPoint()), popup.size())
            )
        width = min(size.width(), screen.width())
        height = min(size.height(), screen.height())
        candidates = []
        for area in (bounds, screen):
            for x, y in (
                (avoid.right() + 1, avoid.top()),
                (avoid.left() - width, avoid.top()),
                (avoid.left(), avoid.bottom() + 1),
                (avoid.left(), avoid.top() - height),
                (area.left(), area.top()),
                (area.right() - width + 1, area.bottom() - height + 1),
            ):
                x = max(screen.left(), min(x, screen.right() - width + 1))
                y = max(screen.top(), min(y, screen.bottom() - height + 1))
                candidate = QtCore.QRect(x, y, width, height)
                overlap = candidate.intersected(avoid)
                candidates.append((overlap.width() * overlap.height(), candidate))
                if area.contains(candidate) and not candidate.intersects(avoid):
                    self.card.setGeometry(candidate)
                    return
        self.card.setGeometry(min(candidates, key=lambda item: item[0])[1])
