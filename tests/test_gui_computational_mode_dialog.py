from contextlib import suppress
import re

import pytest
import shiboken6
from PySide6 import QtCore, QtWidgets

from kindred.gui.widgets.computational_mode_dialog import ComputationalModeDialog


pytestmark = [pytest.mark.gui]


def _find_dialog(title: str) -> QtWidgets.QDialog | None:
    for widget in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(widget, QtWidgets.QDialog) and widget.windowTitle() == title and widget.isVisible():
            return widget
    return None


def _dialog_button(
    dialog: QtWidgets.QDialog, which: QtWidgets.QDialogButtonBox.StandardButton
) -> QtWidgets.QAbstractButton:
    box = dialog.findChild(QtWidgets.QDialogButtonBox)
    assert box is not None
    btn = box.button(which)
    assert btn is not None
    return btn


def _process_deferred_deletes(iterations: int = 5) -> None:
    for _ in range(max(1, int(iterations))):
        QtCore.QCoreApplication.processEvents()
    with suppress(RuntimeError, TypeError):
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    for _ in range(max(1, int(iterations))):
        QtCore.QCoreApplication.processEvents()


def _wait_for_dialog_deletion(dialog: QtWidgets.QDialog, qtbot, *, close_via=None) -> None:
    destroyed = {"fired": False}
    dialog.destroyed.connect(lambda *_args: destroyed.__setitem__("fired", True))
    if close_via is None:
        QtCore.QTimer.singleShot(0, dialog.reject)
    else:
        close_via()
    _process_deferred_deletes()
    qtbot.waitUntil(lambda: not shiboken6.isValid(dialog), timeout=2000)
    assert destroyed["fired"] is True


class _ContractMainWindow(QtWidgets.QWidget):
    def __init__(self, *, mechanism_text: str = "", temperature_k: float = 298.15) -> None:
        super().__init__()
        self._mechanism_text = str(mechanism_text)
        self._temperature_k = float(temperature_k)
        self.set_text_calls: list[tuple[str, str, bool]] = []
        self.populate_calls: list[tuple[object, bool, bool]] = []
        self.get_mechanism_text_calls = 0
        self.mechanism_reactions_text_raw_calls = 0
        self.temperature_spinbox_value_calls = 0

    def __getattr__(self, name: str):
        if str(name).startswith("_"):
            raise AssertionError(f"dialog touched private MainWindow attribute: {name}")
        raise AttributeError(name)

    def mechanism_reactions_text_raw(self) -> str:
        self.mechanism_reactions_text_raw_calls += 1
        return str(self._mechanism_text)

    def set_mechanism_reactions_text_with_optional_undo(
        self,
        new_text: str,
        description: str,
        *,
        record_undo: bool,
    ) -> None:
        self.set_text_calls.append((str(new_text), str(description), bool(record_undo)))
        self._mechanism_text = str(new_text)

    def get_mechanism_text(self) -> str:
        self.get_mechanism_text_calls += 1
        return str(self._mechanism_text)

    def temperature_spinbox_value(self) -> float:
        self.temperature_spinbox_value_calls += 1
        return float(self._temperature_k)

    def populate_energy_mode_variables_from_mechanism(
        self,
        mechanism: object,
        *,
        refresh_sliders: bool,
        preserve_visibility: bool = False,
    ) -> None:
        self.populate_calls.append((mechanism, bool(refresh_sliders), bool(preserve_visibility)))


def _set_table_cell(table: QtWidgets.QTableWidget, row: int, col: int, text: str) -> None:
    item = table.item(row, col)
    if item is None:
        item = QtWidgets.QTableWidgetItem()
        table.setItem(row, col, item)
    item.setText(str(text))


def test_computational_mode_dialog_loads_existing_block_via_public_mainwindow_methods(qt_app):
    _ = qt_app
    mechanism_text = "\n".join(
        [
            "# === Computational Mode ===",
            "comp: T = 310 K",
            "comp: pressure = 1 atm",
            "comp: energy_unit = hartree",
            "comp: std_default = 1 M",
            "comp: kfast_default = 1e9",
            "comp: species A type=GS G=0",
            "comp: species B type=GS G=-0.01",
            "comp: rxn A <-> B",
            "# === End Computational Mode ===",
            "# === Generated from Computational Mode ===",
            "energy=kJ/mol",
            "equilibrium: A <=> B ; kf=1e9 ; kr=1e9",
            "# === End Generated from Computational Mode ===",
        ]
    )
    host = _ContractMainWindow(mechanism_text=mechanism_text)

    dialog = ComputationalModeDialog(host)
    ui = getattr(dialog, "_cm_ui", None)
    assert ui is not None

    assert host.mechanism_reactions_text_raw_calls >= 1
    assert ui.species_table.rowCount() >= 2
    assert ui.reactions_table.rowCount() >= 1
    assert float(ui.temperature_spin.value()) == pytest.approx(310.0, rel=0, abs=0.0)
    assert ui.output_energy_unit.currentText() == "kJ/mol"


def test_computational_mode_dialog_apply_uses_public_mainwindow_methods(qt_app):
    _ = qt_app
    host = _ContractMainWindow()
    dialog = ComputationalModeDialog(host)

    dialog._temperature_spin.setValue(300.0)
    dialog._pressure_value.setValue(1.0)
    dialog._pressure_unit.setCurrentText("atm")
    dialog._energy_unit.setCurrentText("hartree")
    dialog._std_default_spin.setValue(1.0)

    dialog._add_species_row()
    dialog._add_species_row()
    dialog._add_species_row()
    _set_table_cell(dialog._species_table, 0, 0, "A")
    _set_table_cell(dialog._species_table, 0, 1, "GS")
    _set_table_cell(dialog._species_table, 0, 2, "0.0")
    _set_table_cell(dialog._species_table, 0, 3, "1.0")
    _set_table_cell(dialog._species_table, 0, 4, "1.0")
    _set_table_cell(dialog._species_table, 0, 5, "1")
    _set_table_cell(dialog._species_table, 1, 0, "B")
    _set_table_cell(dialog._species_table, 1, 1, "GS")
    _set_table_cell(dialog._species_table, 1, 2, "-0.01")
    _set_table_cell(dialog._species_table, 1, 3, "1.0")
    _set_table_cell(dialog._species_table, 1, 4, "1.0")
    _set_table_cell(dialog._species_table, 1, 5, "1")
    _set_table_cell(dialog._species_table, 2, 0, "TS1")
    _set_table_cell(dialog._species_table, 2, 1, "TS")
    _set_table_cell(dialog._species_table, 2, 2, "0.02")
    _set_table_cell(dialog._species_table, 2, 3, "1.0")
    _set_table_cell(dialog._species_table, 2, 4, "1.0")
    _set_table_cell(dialog._species_table, 2, 5, "1")

    dialog._add_reaction_row()
    _set_table_cell(dialog._reactions_table, 0, 0, "A <-> B")
    _set_table_cell(dialog._reactions_table, 0, 1, "TS1")
    _set_table_cell(dialog._reactions_table, 0, 2, "")

    assert dialog.apply() is True
    assert host.set_text_calls, "dialog should write updated mechanism text through MainWindow public API"
    updated_text, description, record_undo = host.set_text_calls[-1]
    assert description == "Apply Computational Mode"
    assert record_undo is True
    assert "# === Computational Mode ===" in updated_text
    assert "# === Generated from Computational Mode ===" in updated_text
    assert host.get_mechanism_text_calls >= 1
    assert host.temperature_spinbox_value_calls >= 1
    assert host.populate_calls and host.populate_calls[-1][1] is True


def test_gui_computational_mode_dialog_writes_blocks_and_populates_energy_sliders(main_window, qtbot):
    # Open dialog non-blocking.
    main_window._open_computational_mode()
    QtWidgets.QApplication.processEvents()

    dialog = _find_dialog("Computational Mode")
    assert dialog is not None

    # Populate a minimal unimolecular TS channel.
    ui = getattr(dialog, "_cm_ui", None)
    assert ui is not None

    ui.temperature_spin.setValue(300.0)
    ui.pressure_value.setValue(1.0)
    ui.pressure_unit.setCurrentText("atm")
    ui.energy_unit.setCurrentText("hartree")
    ui.std_default_spin.setValue(1.0)

    # Species rows: A (GS), B (GS), TS1 (TS)
    ui.add_species_btn.click()
    ui.add_species_btn.click()
    ui.add_species_btn.click()
    QtWidgets.QApplication.processEvents()
    assert ui.species_table.rowCount() >= 3

    def _set_cell(row: int, col: int, text: str) -> None:
        item = ui.species_table.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            ui.species_table.setItem(row, col, item)
        item.setText(str(text))

    # Columns: Name, Type, G, Std(M), Cref(M), Degeneracy
    _set_cell(0, 0, "A")
    _set_cell(0, 1, "GS")
    _set_cell(0, 2, "0.0")
    _set_cell(0, 3, "1.0")
    _set_cell(0, 4, "1.0")
    _set_cell(0, 5, "1")

    _set_cell(1, 0, "B")
    _set_cell(1, 1, "GS")
    _set_cell(1, 2, "-0.01")
    _set_cell(1, 3, "1.0")
    _set_cell(1, 4, "1.0")
    _set_cell(1, 5, "1")

    _set_cell(2, 0, "TS1")
    _set_cell(2, 1, "TS")
    _set_cell(2, 2, "0.02")
    _set_cell(2, 3, "1.0")
    _set_cell(2, 4, "1.0")
    _set_cell(2, 5, "1")

    # Channel row.
    ui.add_reaction_btn.click()
    QtWidgets.QApplication.processEvents()
    assert ui.reactions_table.rowCount() >= 1
    def _set_cell(r, c, t):
        if ui.reactions_table.item(r, c) is None:
            ui.reactions_table.setItem(r, c, QtWidgets.QTableWidgetItem(str(t)))
        else:
            ui.reactions_table.item(r, c).setText(str(t))

    # Columns: Equation, Via(TS), Fast k
    _set_cell(0, 0, "A <-> B")
    _set_cell(0, 1, "TS1")
    _set_cell(0, 2, "")

    # Apply twice; output must be deterministic (second Apply is a no-op).
    ui.apply_btn.click()
    QtWidgets.QApplication.processEvents()
    first = main_window.mechanism_reactions_text_raw()
    ui.apply_btn.click()
    QtWidgets.QApplication.processEvents()
    second = main_window.mechanism_reactions_text_raw()
    assert second == first

    assert "# === Computational Mode ===" in second
    assert "# === End Computational Mode ===" in second
    assert "# === Generated from Computational Mode ===" in second
    assert "# === End Generated from Computational Mode ===" in second
    assert re.search(r"^comp:\s*species\s+A\b", second, flags=re.MULTILINE)
    assert re.search(r"^state:\s*A\b", second, flags=re.MULTILINE)

    # The Apply action re-parses and should populate energy-mode sliders.
    variables = main_window._mechanism_editor._variable_sliders.get_variables()
    assert any(name.startswith("dGact_fwd__") for name in variables), "Expected ΔG‡_fwd slider variable"
    assert any(name.startswith("dG_eq__") for name in variables), "Expected ΔG° slider variable"

    # Derived kf/kr/K are display-only (not present as sliders in energy mode).
    assert not any(name.startswith(("k", "kf", "kr", "K")) for name in variables)

    # Close dialog.
    _wait_for_dialog_deletion(dialog, qtbot)


def test_gui_computational_mode_fast_equilibrium_exposes_dG_slider_and_updates_kr(main_window, qtbot):
    import math

    from kindred.core.constants import R
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism

    # Open dialog non-blocking.
    main_window._open_computational_mode()
    QtWidgets.QApplication.processEvents()

    dialog = _find_dialog("Computational Mode")
    assert dialog is not None
    ui = getattr(dialog, "_cm_ui", None)
    assert ui is not None

    ui.temperature_spin.setValue(300.0)
    ui.pressure_value.setValue(1.0)
    ui.pressure_unit.setCurrentText("atm")
    ui.energy_unit.setCurrentText("hartree")
    ui.std_default_spin.setValue(1.0)

    # Species rows: A (GS), B (GS)
    ui.add_species_btn.click()
    ui.add_species_btn.click()
    QtWidgets.QApplication.processEvents()
    assert ui.species_table.rowCount() >= 2

    def _set_sp_cell(row: int, col: int, text: str) -> None:
        item = ui.species_table.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            ui.species_table.setItem(row, col, item)
        item.setText(str(text))

    # Columns: Name, Type, G, Std(M), Cref(M), Degeneracy
    _set_sp_cell(0, 0, "A")
    _set_sp_cell(0, 1, "GS")
    _set_sp_cell(0, 2, "0.0")
    _set_sp_cell(0, 3, "1.0")
    _set_sp_cell(0, 4, "1.0")
    _set_sp_cell(0, 5, "1")

    _set_sp_cell(1, 0, "B")
    _set_sp_cell(1, 1, "GS")
    _set_sp_cell(1, 2, "-0.01")
    _set_sp_cell(1, 3, "1.0")
    _set_sp_cell(1, 4, "1.0")
    _set_sp_cell(1, 5, "1")

    # Fast equilibrium reaction row (no TS via).
    ui.add_reaction_btn.click()
    QtWidgets.QApplication.processEvents()
    assert ui.reactions_table.rowCount() >= 1
    ui.reactions_table.setItem(0, 0, QtWidgets.QTableWidgetItem("A <-> B"))
    ui.reactions_table.setItem(0, 1, QtWidgets.QTableWidgetItem(""))
    ui.reactions_table.setItem(0, 2, QtWidgets.QTableWidgetItem(""))

    ui.apply_btn.click()
    QtWidgets.QApplication.processEvents()

    dsl0 = main_window.mechanism_reactions_text_raw()
    assert "equilibrium:" in dsl0

    variables0 = main_window._mechanism_editor._variable_sliders.get_variables()
    fast_vars = [k for k in variables0.keys() if str(k).startswith("dG_eq_fast__")]
    assert fast_vars, "Expected ΔG° slider variable for fast equilibrium"
    assert not any(str(k).startswith("dGact_fwd__") for k in variables0.keys())

    var_name = str(sorted(fast_vars)[0])
    base_dG = float(variables0[var_name])

    # Commit a slider change and ensure the generated equilibrium line updates kr (kf fixed).
    new_dG = base_dG + 5.0
    main_window._commit_slider_value(var_name, new_dG)
    QtWidgets.QApplication.processEvents()

    dsl1 = main_window.mechanism_reactions_text_raw()
    eq_line = next((ln.strip() for ln in dsl1.splitlines() if ln.strip().lower().startswith("equilibrium:")), "")
    assert eq_line

    tokens = {}
    for segment in [p.strip() for p in eq_line.split(";")[1:] if p.strip()]:
        for part in [piece.strip() for piece in segment.split(",") if piece.strip()]:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            tokens[k.strip()] = v.strip()
    assert float(tokens["kf"]) == pytest.approx(1e9, rel=0, abs=0.0)

    K_new = math.exp(-(float(new_dG) * 1000.0) / (R * 300.0))
    expected_kr = float(1e9 / K_new)
    assert float(tokens["kr"]) == pytest.approx(expected_kr, abs=1.0)

    mech = parse_dsl_to_mechanism(dsl1, initials={"A": 1.0, "B": 0.0})
    assert len(mech.reactions) == 0
    assert len(mech.equilibria) == 1
    eq = mech.equilibria[0]
    assert float(eq.kf) == pytest.approx(1e9, rel=0, abs=0.0)
    assert float(eq.kr) == pytest.approx(expected_kr, abs=1.0)

    # Close dialog.
    _wait_for_dialog_deletion(dialog, qtbot)


def test_gui_computational_mode_fast_equilibrium_rewrite_blocks_constrained_kr(main_window, monkeypatch):
    source_text = "\n".join(
        [
            "energy=kJ/mol",
            "T=300",
            "# === Generated from Computational Mode ===",
            "equilibrium: A <=> B ; kf=1e9 ; kr=2e8 ; dG_eq=4 ; cm_id=feq__A__B",
            "# === End Generated from Computational Mode ===",
            "",
            "# Algebra",
            "param kr1 = 2",
        ]
    )

    monkeypatch.setattr(
        main_window,
        "_collect_energy_overrides",
        lambda *args, **kwargs: [
            (
                "dG_eq_fast__feq__A__B",
                9.0,
                {
                    "type": "energy",
                    "role": "dG_eq_fast",
                    "cm_id": "feq__A__B",
                    "unit": "kJ/mol",
                    "kf_fixed": 1e9,
                    "std_ratio": 1.0,
                },
            )
        ],
    )

    updated_text = main_window._apply_energy_overrides_to_computational_mode_fast_equilibria(source_text)

    assert updated_text == source_text


def test_gui_computational_mode_ok_does_not_close_on_error_and_does_not_modify_dsl(main_window, qtbot):
    # Baseline DSL
    before = main_window.mechanism_reactions_text_raw()

    main_window._open_computational_mode()
    QtWidgets.QApplication.processEvents()
    dialog = _find_dialog("Computational Mode")
    assert dialog is not None
    ui = getattr(dialog, "_cm_ui", None)
    assert ui is not None

    ui.temperature_spin.setValue(300.0)
    ui.pressure_value.setValue(1.0)
    ui.pressure_unit.setCurrentText("atm")
    ui.energy_unit.setCurrentText("hartree")
    ui.std_default_spin.setValue(1.0)

    # Define only A, but reference B in reaction to force an apply error.
    ui.add_species_btn.click()
    QtWidgets.QApplication.processEvents()
    assert ui.species_table.rowCount() >= 1
    ui.species_table.setItem(0, 0, QtWidgets.QTableWidgetItem("A"))
    ui.species_table.setItem(0, 1, QtWidgets.QTableWidgetItem("GS"))
    ui.species_table.setItem(0, 2, QtWidgets.QTableWidgetItem("0.0"))
    ui.species_table.setItem(0, 3, QtWidgets.QTableWidgetItem("1.0"))
    ui.species_table.setItem(0, 4, QtWidgets.QTableWidgetItem("1.0"))
    ui.species_table.setItem(0, 5, QtWidgets.QTableWidgetItem("1"))

    ui.add_reaction_btn.click()
    QtWidgets.QApplication.processEvents()
    assert ui.reactions_table.rowCount() >= 1
    ui.reactions_table.setItem(0, 0, QtWidgets.QTableWidgetItem("A <-> B"))
    ui.reactions_table.setItem(0, 1, QtWidgets.QTableWidgetItem(""))
    ui.reactions_table.setItem(0, 2, QtWidgets.QTableWidgetItem(""))

    ok_btn = _dialog_button(dialog, QtWidgets.QDialogButtonBox.StandardButton.Ok)
    ok_btn.click()
    QtWidgets.QApplication.processEvents()

    # OK must not close when Apply fails; dialog should stay open with an error message.
    assert dialog.isVisible()
    err = getattr(dialog, "_error_label", None)
    assert err is not None and err.isVisible()

    after = main_window.mechanism_reactions_text_raw()
    assert after == before

    cancel_btn = _dialog_button(dialog, QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    _wait_for_dialog_deletion(dialog, qtbot, close_via=cancel_btn.click)


def test_gui_computational_mode_cancel_destroys_dialog_before_fixture_teardown(main_window, qtbot):
    main_window._open_computational_mode()
    QtWidgets.QApplication.processEvents()
    dialog = _find_dialog("Computational Mode")
    assert dialog is not None

    cancel_btn = _dialog_button(dialog, QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    _wait_for_dialog_deletion(dialog, qtbot, close_via=cancel_btn.click)

    assert getattr(main_window, "_computational_mode_dialog", None) is None
    assert _find_dialog("Computational Mode") is None


def test_gui_computational_mode_ok_applies_and_closes_and_reopen_loads_saved(main_window, qtbot):
    main_window._open_computational_mode()
    QtWidgets.QApplication.processEvents()
    dialog = _find_dialog("Computational Mode")
    assert dialog is not None
    ui = getattr(dialog, "_cm_ui", None)
    assert ui is not None

    ui.temperature_spin.setValue(310.0)
    ui.pressure_value.setValue(1.0)
    ui.pressure_unit.setCurrentText("atm")
    ui.energy_unit.setCurrentText("hartree")
    ui.std_default_spin.setValue(1.0)

    # Species rows: A (GS), B (GS)
    ui.add_species_btn.click()
    ui.add_species_btn.click()
    QtWidgets.QApplication.processEvents()
    assert ui.species_table.rowCount() >= 2

    def _set_sp(row: int, col: int, text: str) -> None:
        item = ui.species_table.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            ui.species_table.setItem(row, col, item)
        item.setText(str(text))

    _set_sp(0, 0, "A")
    _set_sp(0, 1, "GS")
    _set_sp(0, 2, "0.0")
    _set_sp(0, 3, "1.0")
    _set_sp(0, 4, "1.0")
    _set_sp(0, 5, "1")

    _set_sp(1, 0, "B")
    _set_sp(1, 1, "GS")
    _set_sp(1, 2, "-0.01")
    _set_sp(1, 3, "1.0")
    _set_sp(1, 4, "1.0")
    _set_sp(1, 5, "1")

    ui.add_reaction_btn.click()
    QtWidgets.QApplication.processEvents()
    assert ui.reactions_table.rowCount() >= 1
    ui.reactions_table.setItem(0, 0, QtWidgets.QTableWidgetItem("A <-> B"))
    ui.reactions_table.setItem(0, 1, QtWidgets.QTableWidgetItem(""))
    ui.reactions_table.setItem(0, 2, QtWidgets.QTableWidgetItem(""))

    ok_btn = _dialog_button(dialog, QtWidgets.QDialogButtonBox.StandardButton.Ok)
    _wait_for_dialog_deletion(dialog, qtbot, close_via=ok_btn.click)

    # OK applies and closes.
    assert _find_dialog("Computational Mode") is None
    dsl = main_window.mechanism_reactions_text_raw()
    assert "# === Computational Mode ===" in dsl
    assert "# === End Computational Mode ===" in dsl
    assert "# === Generated from Computational Mode ===" in dsl
    assert "# === End Generated from Computational Mode ===" in dsl

    # Re-open and ensure it loads existing comp block rather than showing blank tables.
    main_window._open_computational_mode()
    QtWidgets.QApplication.processEvents()
    dialog2 = _find_dialog("Computational Mode")
    assert dialog2 is not None
    ui2 = getattr(dialog2, "_cm_ui", None)
    assert ui2 is not None
    assert ui2.species_table.rowCount() >= 2
    names = set()
    for r in range(ui2.species_table.rowCount()):
        item = ui2.species_table.item(r, 0)
        if item is not None and item.text().strip():
            names.add(item.text().strip())
    assert {"A", "B"} <= names
    assert float(ui2.temperature_spin.value()) == pytest.approx(310.0, rel=0, abs=0.0)

    cancel_btn = _dialog_button(dialog2, QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    _wait_for_dialog_deletion(dialog2, qtbot, close_via=cancel_btn.click)


def test_gui_computational_mode_cancel_closes_without_applying(main_window, qtbot):
    before = main_window.mechanism_reactions_text_raw()

    main_window._open_computational_mode()
    QtWidgets.QApplication.processEvents()
    dialog = _find_dialog("Computational Mode")
    assert dialog is not None
    ui = getattr(dialog, "_cm_ui", None)
    assert ui is not None

    ui.add_species_btn.click()
    QtWidgets.QApplication.processEvents()
    ui.species_table.setItem(0, 0, QtWidgets.QTableWidgetItem("A"))
    ui.species_table.setItem(0, 2, QtWidgets.QTableWidgetItem("0.0"))

    cancel_btn = _dialog_button(dialog, QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    _wait_for_dialog_deletion(dialog, qtbot, close_via=cancel_btn.click)

    after = main_window.mechanism_reactions_text_raw()
    assert after == before
