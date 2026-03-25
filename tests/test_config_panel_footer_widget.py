from __future__ import annotations

import pytest
from PySide6.QtTest import QSignalSpy


pytestmark = [pytest.mark.gui]


def test_config_panel_footer_emits_signals_and_updates_enabled_state(qt_app, qtbot):
    from kindred.gui.widgets.config_panel_footer import ConfigPanelFooter

    widget = ConfigPanelFooter(
        show_dirty=True,
        show_secondary_error=True,
        show_divider=True,
        show_reset=True,
        messages_position="after_body",
        apply_requires_no_error=True,
        button_order=("reset", "revert", "apply"),
    )
    qtbot.addWidget(widget)
    widget.show()
    qt_app.processEvents()

    apply_spy = QSignalSpy(widget.applyRequested)
    revert_spy = QSignalSpy(widget.revertRequested)
    reset_spy = QSignalSpy(widget.resetRequested)

    assert not widget.apply_button.isEnabled()
    assert not widget.revert_button.isEnabled()
    assert widget.reset_button is not None
    assert not widget.reset_button.isEnabled()
    assert not widget.dirty_label.isVisible()
    assert not widget.error_label.isVisible()
    assert widget.secondary_error_label is not None
    assert not widget.secondary_error_label.isVisible()

    widget.set_dirty(True)
    qt_app.processEvents()
    assert widget.apply_button.isEnabled()
    assert widget.revert_button.isEnabled()
    assert widget.dirty_label.isVisible()
    assert not widget.reset_button.isEnabled()

    widget.apply_button.click()
    widget.revert_button.click()
    widget.reset_button.setEnabled(True)
    widget.reset_button.click()
    qt_app.processEvents()
    assert int(apply_spy.count()) == 1
    assert int(revert_spy.count()) == 1
    assert int(reset_spy.count()) == 1

    widget.set_error("Invalid")
    qt_app.processEvents()
    assert widget.error_label.isVisible()
    assert not widget.apply_button.isEnabled()
    assert widget.revert_button.isEnabled()

    widget.set_error(None)
    qt_app.processEvents()
    assert not widget.error_label.isVisible()
    assert widget.apply_button.isEnabled()

    widget.set_secondary_error("Blocked")
    qt_app.processEvents()
    assert widget.secondary_error_label.isVisible()

    widget.set_secondary_error(None)
    qt_app.processEvents()
    assert not widget.secondary_error_label.isVisible()

    widget.set_dirty(False)
    qt_app.processEvents()
    assert not widget.apply_button.isEnabled()
    assert not widget.revert_button.isEnabled()
    assert not widget.dirty_label.isVisible()


def test_config_panel_footer_normalizes_button_order_tokens_for_layout(qt_app, qtbot):
    from kindred.gui.widgets.config_panel_footer import ConfigPanelFooter

    widget = ConfigPanelFooter(
        show_reset=True,
        button_order=(" Reset ", " ReVert ", " Apply "),
    )
    qtbot.addWidget(widget)
    widget.show()
    qt_app.processEvents()

    assert widget.reset_button is not None
    buttons_layout = widget.layout().itemAt(widget.layout().count() - 1).layout()
    assert buttons_layout is not None

    assert buttons_layout.indexOf(widget.reset_button) >= 0
    assert buttons_layout.indexOf(widget.revert_button) >= 0
    assert buttons_layout.indexOf(widget.apply_button) >= 0

    order = []
    for i in range(buttons_layout.count()):
        item = buttons_layout.itemAt(i)
        w = item.widget() if item is not None else None
        if w is not None:
            order.append(w.text())
    assert order == ["Reset", "Revert", "Apply"]


def test_config_panel_footer_enabled_overrides_persist_until_cleared(qt_app, qtbot):
    from kindred.gui.widgets.config_panel_footer import ConfigPanelFooter

    widget = ConfigPanelFooter(show_dirty=True, apply_requires_no_error=True)
    qtbot.addWidget(widget)
    widget.show()
    qt_app.processEvents()

    widget.set_dirty(True)
    qt_app.processEvents()
    assert widget.apply_button.isEnabled()
    assert widget.revert_button.isEnabled()

    widget.set_apply_enabled_override(False)
    widget.set_revert_enabled_override(False)
    qt_app.processEvents()
    assert not widget.apply_button.isEnabled()
    assert not widget.revert_button.isEnabled()

    widget.set_error("Invalid")
    qt_app.processEvents()
    assert not widget.apply_button.isEnabled()
    assert not widget.revert_button.isEnabled()

    widget.set_error(None)
    qt_app.processEvents()
    assert not widget.apply_button.isEnabled()
    assert not widget.revert_button.isEnabled()

    widget.set_dirty(False)
    qt_app.processEvents()
    widget.set_dirty(True)
    qt_app.processEvents()
    assert not widget.apply_button.isEnabled()
    assert not widget.revert_button.isEnabled()

    widget.set_apply_enabled_override(None)
    widget.set_revert_enabled_override(None)
    qt_app.processEvents()
    assert widget.apply_button.isEnabled()
    assert widget.revert_button.isEnabled()
