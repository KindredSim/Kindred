"""Regression tests for the tutorial overlay spotlight logic."""

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

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


@pytest.mark.gui
def test_mask_excludes_spotlight_when_target_has_rect(qtbot):
    """Overlay mask must exclude the spotlight area so clicks reach the target widget."""
    parent = QtWidgets.QWidget()
    parent.setGeometry(0, 0, 800, 600)
    qtbot.addWidget(parent)

    target_rect = QtCore.QRect(200, 200, 100, 50)
    step = TutorialStep(
        title="Test",
        instruction="Click the button",
        target_rect=target_rect,
        arrow_direction="top",
    )

    overlay = TutorialOverlay(parent, [step])
    overlay.setGeometry(parent.rect())
    qtbot.addWidget(overlay)

    mask = overlay.mask()
    assert not mask.isEmpty(), "Mask should be set when a target rect exists"

    # Spotlight center must be excluded from the mask (clicks pass through)
    spotlight_center = target_rect.center()
    assert not mask.contains(spotlight_center), "Spotlight center should be excluded from mask"

    # A point in the dark overlay area must be inside the mask (overlay captures clicks)
    assert mask.contains(QtCore.QPoint(5, 5)), "Dark overlay area should be inside mask"


@pytest.mark.gui
def test_mask_cleared_for_informational_step(qtbot):
    """Overlay mask must be cleared for steps with no target widget."""
    parent = QtWidgets.QWidget()
    parent.setGeometry(0, 0, 800, 600)
    qtbot.addWidget(parent)

    step_with_target = TutorialStep(
        title="Step 1",
        instruction="Click the button",
        target_rect=QtCore.QRect(200, 200, 100, 50),
    )
    step_no_target = TutorialStep(
        title="Step 2",
        instruction="Read this information",
    )

    overlay = TutorialOverlay(parent, [step_with_target, step_no_target])
    overlay.setGeometry(parent.rect())
    qtbot.addWidget(overlay)

    # Step 0 has a target — mask should be set
    assert not overlay.mask().isEmpty()

    # Advance to step 1 (informational, no target)
    overlay.current_step = 1
    overlay._update_step()

    assert overlay.mask().isEmpty(), "Mask should be cleared for informational steps"


@pytest.mark.gui
def test_instruction_box_visible_when_overlapping_spotlight(qtbot):
    """Instruction box must remain inside the mask even if it overlaps the spotlight."""
    parent = QtWidgets.QWidget()
    parent.setGeometry(0, 0, 800, 600)
    qtbot.addWidget(parent)

    # Large target rect that forces the instruction box to overlap with the spotlight
    # (covers most of the parent, so fallback center position overlaps)
    step = TutorialStep(
        title="Test",
        instruction="This box must be visible",
        target_rect=QtCore.QRect(50, 50, 700, 500),
        arrow_direction="top",
    )

    overlay = TutorialOverlay(parent, [step])
    overlay.setGeometry(parent.rect())
    qtbot.addWidget(overlay)

    mask = overlay.mask()
    box_center = overlay._instruction_box.geometry().center()
    assert mask.contains(box_center), (
        "Instruction box center must be inside the mask so the box remains visible"
    )
