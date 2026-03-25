import pytest
from PySide6 import QtWidgets

from kindred.gui.widgets.state_network_editor import StateNetworkEditor

pytestmark = [pytest.mark.gui]


def _process_events_bounded(iterations: int = 20) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    for _ in range(int(iterations)):
        app.processEvents()


def test_state_network_editor_debounces_validate_and_change_signal(qt_app):
    """
    Regression test for UI freeze/memory blow-up.

    If the editor receives a burst of change notifications (e.g., due to signal
    re-entrancy or platform-specific widget churn), it must not synchronously
    validate/emit for every event in the burst.
    """
    editor = StateNetworkEditor()

    validate_calls = {"n": 0}
    emitted = {"n": 0}

    original_validate = editor._validate

    def _wrapped_validate():
        validate_calls["n"] += 1
        return original_validate()

    editor._validate = _wrapped_validate  # type: ignore[method-assign]
    editor.stateNetworkChanged.connect(lambda: emitted.__setitem__("n", emitted["n"] + 1))

    for _ in range(50):
        editor._on_states_changed()

    # The debounced handler runs via the Qt event loop.
    _process_events_bounded()

    assert validate_calls["n"] == 1
    assert emitted["n"] == 1

