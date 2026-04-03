"""Regression tests for the tutorial overlay spotlight logic."""

import pytest
from PySide6 import QtCore, QtGui

from kindred.gui.widgets.tutorial_overlay import TutorialOverlay, TutorialStep


@pytest.mark.gui
def test_qaction_without_associated_widgets_does_not_crash(main_window, qtbot):
    """QAction.associatedWidgets() does not exist in PySide6.

    _get_highlight_rect() must not raise AttributeError when the target is a
    QAction that lacks the associatedWidgets method.  It should fall through to
    the menu-bar fallback (or return None) without error.

    Regression: 09b97e8 introduced a call to associatedWidgets() that crashes
    on PySide6==6.7.2.
    """
    qtbot.addWidget(main_window)

    action = QtGui.QAction("Test Action", main_window)
    action.setObjectName("_test_tutorial_action")

    # Ensure associatedWidgets is truly absent (PySide6 6.7.2)
    assert not hasattr(action, "associatedWidgets")

    step = TutorialStep(
        title="Test",
        instruction="Test instruction",
        target_widget="_test_tutorial_action",
        arrow_direction="top",
    )

    overlay = TutorialOverlay(main_window, [step])
    qtbot.addWidget(overlay)

    # Must not raise AttributeError
    rect = overlay._get_highlight_rect()

    # rect may be None (no menu-bar match for a synthetic action) — that's fine;
    # the point is that it doesn't crash.
    assert rect is None or isinstance(rect, QtCore.QRect)
