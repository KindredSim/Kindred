from __future__ import annotations

import logging

import pytest
from PySide6 import QtWidgets

from kindred.gui.mixins.fitting_mixin import FittingMixin
from kindred.gui.mixins.ports import FittingMixinPorts


pytestmark = [pytest.mark.gui]


class _MechanismEditor:
    def __init__(self, text: str) -> None:
        self._text = text

    def reactions_text(self) -> str:
        return self._text

    def set_reactions_text(self, text: str) -> None:
        self._text = str(text)


class _Host(QtWidgets.QWidget, FittingMixin):
    def __init__(self, mechanism_text: str) -> None:
        super().__init__()
        self.editor = _MechanismEditor(mechanism_text)
        self._fitting_ports = FittingMixinPorts(
            mechanism_editor=self.editor,
            dataset_manager=None,
            data_manager_getter=lambda: None,
            status_setter=lambda _text: None,
            temperature_getter=lambda: 298.15,
            num_points_getter=lambda: 100,
        )

    def _update_variable_in_mechanism(
        self,
        _name: str,
        _value: float,
        *,
        source_text: str | None = None,
        commit: bool = True,
    ) -> str:
        raise RuntimeError("metadata drift")


def test_write_fit_results_logs_updater_errors_distinctly(qt_app, caplog):
    _ = qt_app
    host = _Host("reaction: A -> B; k=1.0")
    caplog.set_level(logging.WARNING, logger="kindred.gui.mixins.fitting_mixin")

    try:
        host._write_fit_results_to_mechanism({"k1": 2.0})
    finally:
        host.close()

    assert host.editor.reactions_text() == "reaction: A -> B ; k=2"
    assert not any("missing/invalid" in record.getMessage() for record in caplog.records)
    assert not any("failed while applying" in record.getMessage() for record in caplog.records)
