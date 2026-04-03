"""
Interactive tutorial overlay system.

Provides step-by-step tutorials with UI highlighting:
- Semi-transparent overlay covering entire window
- Spotlight highlighting of specific UI elements
- Instruction boxes with arrows pointing to elements
- Navigation controls (Next, Previous, Skip)
- Progress indicator
"""

from __future__ import annotations

import logging
from typing import Optional, List

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

logger = logging.getLogger(__name__)

__all__ = ["TutorialOverlay", "TutorialStep"]


class TutorialStep:
    """
    A single step in a tutorial.

    Attributes
    ----------
    title : str
        Step title
    instruction : str
        Instruction text (supports HTML)
    target_widget : str, optional
        Object name of widget to highlight (e.g., "runSimulationAction")
    target_rect : QtCore.QRect, optional
        Explicit rectangle to highlight (overrides target_widget)
    arrow_direction : str
        Arrow direction: "top", "bottom", "left", "right", or "none"
    """

    def __init__(
        self,
        title: str,
        instruction: str,
        target_widget: Optional[str] = None,
        target_rect: Optional[QtCore.QRect] = None,
        arrow_direction: str = "top",
    ):
        self.title = title
        self.instruction = instruction
        self.target_widget = target_widget
        self.target_rect = target_rect
        self.arrow_direction = arrow_direction


class TutorialOverlay(QtWidgets.QWidget):
    """
    Semi-transparent overlay for interactive tutorials.

    Features:
    - Darkens entire window except highlighted area
    - Shows instruction box with arrow
    - Navigation buttons (Next, Previous, Skip)
    - Progress indicator
    - Automatically positions instruction box away from highlighted area
    """

    # Signals
    tutorialCompleted = QtCore.Signal()
    tutorialSkipped = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget, steps: List[TutorialStep]):
        """
        Initialize tutorial overlay.

        Parameters
        ----------
        parent : QWidget
            Parent widget (usually main window)
        steps : list of TutorialStep
            Tutorial steps
        """
        super().__init__(parent)

        self.steps = steps
        self.current_step = 0

        # Make overlay fill parent
        self.setGeometry(parent.rect())

        # Semi-transparent background (dark overlay)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        # Create instruction box
        self._instruction_box = QtWidgets.QFrame(self)
        self._instruction_box.setAttribute(Qt.WA_StyledBackground, True)
        self._instruction_box.setAutoFillBackground(True)
        self._instruction_box.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self._instruction_box.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)

        box_layout = QtWidgets.QVBoxLayout(self._instruction_box)
        box_layout.setContentsMargins(16, 16, 16, 16)
        box_layout.setSpacing(12)

        # Title label
        self._title_label = QtWidgets.QLabel()
        title_font = self._title_label.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        self._title_label.setWordWrap(True)
        box_layout.addWidget(self._title_label)

        # Instruction label
        self._instruction_label = QtWidgets.QLabel()
        instruction_font = self._instruction_label.font()
        instruction_font.setPointSize(10)
        self._instruction_label.setFont(instruction_font)
        self._instruction_label.setWordWrap(True)
        self._instruction_label.setTextFormat(Qt.TextFormat.RichText)
        box_layout.addWidget(self._instruction_label)

        # Progress label
        self._progress_label = QtWidgets.QLabel()
        progress_font = self._progress_label.font()
        progress_font.setPointSize(9)
        self._progress_label.setFont(progress_font)
        self._progress_label.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(self._progress_label)

        # Button row
        button_layout = QtWidgets.QHBoxLayout()

        self._skip_btn = QtWidgets.QPushButton("Skip Tutorial")
        self._skip_btn.clicked.connect(self._on_skip)
        button_layout.addWidget(self._skip_btn)

        button_layout.addStretch()

        self._prev_btn = QtWidgets.QPushButton("← Previous")
        self._prev_btn.clicked.connect(self._on_previous)
        button_layout.addWidget(self._prev_btn)

        self._next_btn = QtWidgets.QPushButton("Next →")
        self._next_btn.clicked.connect(self._on_next)
        self._next_btn.setDefault(True)
        button_layout.addWidget(self._next_btn)

        box_layout.addLayout(button_layout)

        # Show first step
        self._update_step()

    def _update_step(self):
        """Update UI for current step."""
        if self.current_step < 0 or self.current_step >= len(self.steps):
            return

        step = self.steps[self.current_step]

        # Update labels
        self._title_label.setText(step.title)
        self._instruction_label.setText(step.instruction)
        self._progress_label.setText(f"Step {self.current_step + 1} of {len(self.steps)}")

        # Update buttons
        self._prev_btn.setEnabled(self.current_step > 0)

        if self.current_step == len(self.steps) - 1:
            self._next_btn.setText("Finish")
        else:
            self._next_btn.setText("Next →")

        # Position instruction box
        self._position_instruction_box()

        # Trigger repaint
        self.update()

    def _position_instruction_box(self):
        """Position instruction box away from highlighted area."""
        _ = self.steps[self.current_step]

        # Get highlighted rect
        highlight_rect = self._get_highlight_rect()

        # Instruction box size
        box_width = 400
        box_height = self._instruction_box.sizeHint().height()

        # Try positions in order: bottom, right, top, left, center
        parent_rect = self.rect()

        # Try bottom
        if highlight_rect:
            x = max(20, min(parent_rect.width() - box_width - 20, highlight_rect.center().x() - box_width // 2))
            y = highlight_rect.bottom() + 20

            if y + box_height + 20 <= parent_rect.height():
                self._instruction_box.setGeometry(x, y, box_width, box_height)
                return

            # Try top
            y = highlight_rect.top() - box_height - 20
            if y >= 20:
                self._instruction_box.setGeometry(x, y, box_width, box_height)
                return

            # Try right
            x = highlight_rect.right() + 20
            y = max(20, min(parent_rect.height() - box_height - 20, highlight_rect.center().y() - box_height // 2))

            if x + box_width + 20 <= parent_rect.width():
                self._instruction_box.setGeometry(x, y, box_width, box_height)
                return

            # Try left
            x = highlight_rect.left() - box_width - 20
            if x >= 20:
                self._instruction_box.setGeometry(x, y, box_width, box_height)
                return

        # Default: center
        x = (parent_rect.width() - box_width) // 2
        y = (parent_rect.height() - box_height) // 2
        self._instruction_box.setGeometry(x, y, box_width, box_height)

    def _get_highlight_rect(self) -> Optional[QtCore.QRect]:
        """Get rectangle to highlight for current step."""
        step = self.steps[self.current_step]

        if step.target_rect:
            return step.target_rect

        if step.target_widget:
            # Find widget by object name
            target = self.parent().findChild(QtWidgets.QWidget, step.target_widget)
            if not target:
                # Try finding QAction
                target = self.parent().findChild(QtGui.QAction, step.target_widget)

            if target:
                if isinstance(target, QtWidgets.QWidget):
                    # Map widget position to overlay coordinates
                    global_pos = target.mapToGlobal(QtCore.QPoint(0, 0))
                    local_pos = self.mapFromGlobal(global_pos)
                    return QtCore.QRect(local_pos, target.size())
                elif isinstance(target, QtGui.QAction):
                    # Try to find an associated toolbar widget (PyQt5 compat;
                    # PySide6 does not expose associatedWidgets on QAction).
                    for widget in getattr(target, "associatedWidgets", list)():
                        if isinstance(widget, (QtWidgets.QToolButton, QtWidgets.QPushButton)):
                            global_pos = widget.mapToGlobal(QtCore.QPoint(0, 0))
                            local_pos = self.mapFromGlobal(global_pos)
                            return QtCore.QRect(local_pos, widget.size())

                    # Fallback: highlight the parent menu in the menu bar
                    rect = self._get_menu_bar_rect_for_action(target)
                    if rect is not None:
                        return rect

        return None

    def _get_menu_bar_rect_for_action(
        self, action: QtGui.QAction
    ) -> Optional[QtCore.QRect]:
        """Return the menu-bar geometry for the top-level menu that contains *action*.

        Walks the QMenuBar's top-level menus (one level only — no submenu
        recursion) and returns the menu-bar item rectangle mapped into
        overlay coordinates.  Returns ``None`` if the action is not found
        or the resulting rectangle is empty.
        """
        parent = self.parent()
        menu_bar = getattr(parent, "menuBar", None)
        if menu_bar is None:
            return None
        menu_bar = menu_bar()
        if not isinstance(menu_bar, QtWidgets.QMenuBar):
            return None

        for top_action in menu_bar.actions():
            menu = top_action.menu()
            if menu is None:
                continue
            if action in menu.actions():
                geom = menu_bar.actionGeometry(top_action)
                if geom.isEmpty():
                    return None
                global_pos = menu_bar.mapToGlobal(geom.topLeft())
                local_pos = self.mapFromGlobal(global_pos)
                return QtCore.QRect(local_pos, geom.size())

        return None

    def paintEvent(self, event: QtGui.QPaintEvent):
        """Paint semi-transparent overlay with spotlight."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Draw dark overlay
        overlay_color = QtGui.QColor(0, 0, 0, 180)  # Semi-transparent black
        painter.fillRect(self.rect(), overlay_color)

        # Cut out spotlight area
        highlight_rect = self._get_highlight_rect()
        if highlight_rect:
            # Expand highlight rect for padding
            padding = 8
            spotlight_rect = highlight_rect.adjusted(-padding, -padding, padding, padding)

            # Draw rounded spotlight
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            painter.setBrush(Qt.transparent)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(spotlight_rect, 8, 8)

            # Draw highlight border
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            painter.setPen(QtGui.QPen(QtGui.QColor("#4A90E2"), 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(spotlight_rect, 8, 8)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        """Handle parent resize."""
        super().resizeEvent(event)
        self._position_instruction_box()

    def _on_next(self):
        """Handle Next button click."""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self._update_step()
        else:
            # Finished tutorial
            self.tutorialCompleted.emit()
            self.close()

    def _on_previous(self):
        """Handle Previous button click."""
        if self.current_step > 0:
            self.current_step -= 1
            self._update_step()

    def _on_skip(self):
        """Handle Skip button click."""
        self.tutorialSkipped.emit()
        self.close()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Handle keyboard navigation."""
        if event.key() == Qt.Key_Right or event.key() == Qt.Key_Return:
            self._on_next()
        elif event.key() == Qt.Key_Left:
            self._on_previous()
        elif event.key() == Qt.Key_Escape:
            self._on_skip()
        else:
            super().keyPressEvent(event)
