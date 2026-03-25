from __future__ import annotations

import logging

import pytest
from PySide6 import QtWidgets

from kindred.gui.main_window import MainWindow
from kindred.gui.mixins.fitting_mixin import FittingMixin
from kindred.gui.mixins.ports import FittingMixinPorts


class _MainWindowRecorderHost:
    pass


class _MechanismEditor:
    def __init__(self, text: str = "reaction: A -> B; k=1.0") -> None:
        self._text = str(text)
        self._reactions_text = self

    def toPlainText(self) -> str:
        return self._text

    def setPlainText(self, text: str) -> None:
        self._text = str(text)

    def reactions_text(self) -> str:
        return self._text

    def set_reactions_text(self, text: str) -> None:
        self._text = str(text)


class _FittingRecorderHost(QtWidgets.QWidget, FittingMixin):
    def __init__(self) -> None:
        super().__init__()
        self.editor = _MechanismEditor()
        self._fitting_ports = FittingMixinPorts(
            mechanism_editor=self.editor,
            dataset_manager=object(),
            data_manager_getter=lambda: object(),
            status_setter=lambda _text: None,
            temperature_getter=lambda: 298.15,
            num_points_getter=lambda: 100,
        )

    def _get_mechanism_text(self) -> str:
        return self.editor.reactions_text()

    def _extract_mechanism_initials(self, mechanism_text: str) -> dict[str, float]:
        _ = mechanism_text
        return {}

    def _sync_batch_species_columns(self, species: list[str]) -> None:
        _ = species

    def _batch_initials_for_row(self, row: int) -> dict[str, float]:
        _ = row
        return {}

    def _get_solver_settings(self) -> dict[str, object]:
        return {}

    def _register_fit_window(self, window: QtWidgets.QWidget) -> None:
        _ = window

    def _write_fit_results_to_mechanism(self, parameters: dict[str, float]) -> None:
        _ = parameters

    def _apply_dataset_initial_updates(self, dataset_id: str, updates: dict[str, float]) -> None:
        _ = (dataset_id, updates)


@pytest.mark.unit
def test_main_window_best_effort_wrapper_matches_shared_helper_contract(caplog) -> None:
    host = _MainWindowRecorderHost()
    caplog.set_level(logging.DEBUG, logger="kindred.gui.main_window")

    for _ in range(4):
        MainWindow._record_best_effort_failure(
            host,
            "main.key",
            message="MainWindow best effort",
        )

    assert host._best_effort_failures == {"main.key"}
    assert host._best_effort_failure_counts == {"main.key": 4}
    messages = [record.getMessage() for record in caplog.records if record.name == "kindred.gui.main_window"]
    assert messages == [
        "MainWindow best effort (key=main.key count=1)",
        "MainWindow best effort (key=main.key count=2)",
        "MainWindow best effort (key=main.key count=3)",
    ]


@pytest.mark.gui
def test_fitting_mixin_best_effort_wrapper_preserves_fitting_attrs(qt_app, caplog) -> None:
    _ = qt_app
    host = _FittingRecorderHost()
    caplog.set_level(logging.DEBUG, logger="kindred.gui.mixins.fitting_mixin")

    try:
        for _ in range(4):
            host._record_fitting_best_effort_failure(
                "fit.key",
                message="FittingMixin best effort",
            )
    finally:
        host.close()

    assert host._fitting_best_effort_failures == {"fit.key"}
    assert host._fitting_best_effort_failure_counts == {"fit.key": 4}
    assert not hasattr(host, "_best_effort_failures")
    assert not hasattr(host, "_best_effort_failure_counts")
    messages = [record.getMessage() for record in caplog.records if record.name == "kindred.gui.mixins.fitting_mixin"]
    assert messages == [
        "FittingMixin best effort (key=fit.key count=1)",
        "FittingMixin best effort (key=fit.key count=2)",
        "FittingMixin best effort (key=fit.key count=3)",
    ]


@pytest.mark.gui
def test_fitting_launch_context_preserves_owner_best_effort_recorder_callback(qt_app) -> None:
    _ = qt_app
    host = _FittingRecorderHost()

    try:
        context = host._build_global_fit_launch_context()
        context.record_best_effort_failure(
            "launch.key",
            message="Launch best effort",
        )
    finally:
        host.close()

    assert host._fitting_best_effort_failures == {"launch.key"}
    assert host._fitting_best_effort_failure_counts == {"launch.key": 1}
