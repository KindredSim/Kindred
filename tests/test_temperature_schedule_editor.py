from PySide6 import QtWidgets

from kindred.gui.widgets.temperature_schedule_editor import TemperatureScheduleDialog
import pytest

pytestmark = pytest.mark.gui



def test_temperature_schedule_editor_emits_temp_response_dsl_and_preview_contract(qtbot):
    dialog = TemperatureScheduleDialog()
    qtbot.addWidget(dialog)

    dialog._mode_combo.setCurrentText("First-Order Response")
    dialog._template_combo.setCurrentText("Step Change (25°C → 50°C at t=50)")
    dialog._tau_spin.setValue(10.0)

    dsl = dialog.get_dsl()
    assert dsl == "temp_response: t=[0.0,50.0,100.0], T=[298.15,323.15], tau=10.0"

    times, actual, setpoint = dialog._build_preview_series()
    assert len(times) == len(actual) == len(setpoint)
    assert actual[0] == 298.15
    assert setpoint[-1] == 323.15
    assert max(actual) < max(setpoint)

    received = []
    dialog.scheduleCreated.connect(received.append)
    ok_button = dialog.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    ok_button.click()

    assert received == [dsl]
