"""
Test GUI error handling and edge cases.

This module tests that GUI components gracefully handle error conditions:
- Missing data scenarios
- Invalid user inputs
- Import errors for optional dependencies
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from PySide6 import QtWidgets, QtCore

pytestmark = [pytest.mark.gui, pytest.mark.slow]


@pytest.fixture(autouse=True)
def suppress_modal_dialogs(monkeypatch):
    """Prevent modal dialogs from blocking tests."""
    # Use lightweight stubs so per-test patches can still assert calls/messages.
    for name in ("critical", "information", "warning", "question"):
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            name,
            lambda *args, **kwargs: QtWidgets.QMessageBox.Ok,
        )


class TestExportErrorHandling:
    """Test export functions handle missing data gracefully."""

    def test_export_csv_with_no_data(self, main_window):
        """Test exporting CSV when no simulation has been run."""
        main_window._plot_tabs = MagicMock()
        main_window._plot_tabs.get_current_plot.return_value = None

        with patch.object(QtWidgets.QMessageBox, 'warning') as mock_warning:
            main_window.project_controller.export_data()
            assert mock_warning.called


class TestFittingErrorHandling:
    """Test fitting workflows handle errors gracefully."""

    def test_global_fit_with_no_datasets(self, main_window):
        """Test global fit when no datasets are loaded."""
        # Mock the data panel to return empty datasets
        mock_panel = MagicMock()
        mock_panel.get_datasets.return_value = {}
        main_window._right_panel = MagicMock()
        main_window._right_panel._data_manager = mock_panel

        with patch.object(QtWidgets.QMessageBox, 'warning') as mock_warning:
            main_window._run_global_fit()
            # Should warn about no datasets
            assert mock_warning.called
            call_args = mock_warning.call_args[0]
            assert 'dataset' in call_args[2].lower()

    def test_global_fit_with_single_dataset_autoselect(self, main_window, monkeypatch):
        """Test global fit opens and proceeds with a single dataset."""
        # Mock single dataset
        mock_panel = MagicMock()
        mock_panel.get_datasets.return_value = {'test.csv': {'t': [0, 1], 'species': {'A': [1, 0.5]}}}
        main_window._right_panel._data_manager = mock_panel

        monkeypatch.setattr(
            main_window._dataset_manager,
            "scan_mechanism_parameters",
            lambda mech: [{"name": "k", "value": 0.1, "min": 0.01, "max": 1.0}],
        )
        monkeypatch.setattr(
            main_window,
            "_extract_mechanism_initials",
            lambda mechanism: {"A": 1.0},
        )

        created = {}

        class _FakeWindow(QtWidgets.QDialog):
            def __init__(self, *args, **kwargs):
                super().__init__()
                created["kwargs"] = kwargs

            def setWindowTitle(self, title):
                created["title"] = title

            def show(self):
                created["show"] = True

            def raise_(self):
                pass

            def activateWindow(self):
                pass

        monkeypatch.setattr("kindred.gui.fitting.window.FittingWindow", _FakeWindow)

        class _DialogMustNotBeConstructed:
            def __init__(self, *args, **kwargs):
                raise AssertionError("GlobalFitConfigDialog must not be used in the launch flow")

        monkeypatch.setattr("kindred.gui.fitting.global_fit_config.GlobalFitConfigDialog", _DialogMustNotBeConstructed)

        main_window._run_global_fit()
        assert created, "Global fitting window should be opened"
        assert "window" in main_window._status_label.text().lower()

    def test_global_fit_with_multiple_datasets_opens_window_directly(self, main_window, monkeypatch):
        """Global Fit should open the main window immediately without an initial modal config dialog."""
        # Mock multiple datasets
        mock_panel = MagicMock()
        datasets = {
            'dataset1.csv': {'t': [0, 1], 'species': {'A': [1, 0.5]}},
            'dataset2.csv': {'t': [0, 1], 'species': {'A': [1, 0.6]}}
        }
        mock_panel.get_datasets.return_value = datasets
        main_window._right_panel._data_manager = mock_panel

        main_window._mechanism_editor._reactions_text.setPlainText(
            "\n".join(
                [
                    "reaction: A -> B; k=0.2",
                    "initial: A=1.0",
                    "initial: B=0.0",
                ]
            )
        )
        monkeypatch.setattr(
            main_window._dataset_manager,
            "scan_mechanism_parameters",
            lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        )

        created = {}

        class _FakeWindow(QtWidgets.QDialog):
            def __init__(self, *args, **kwargs):
                super().__init__()
                created["kwargs"] = kwargs

            def setWindowTitle(self, *_):
                pass

            def show(self):
                created["show"] = True

            def raise_(self):
                pass

            def activateWindow(self):
                pass

        monkeypatch.setattr("kindred.gui.fitting.window.FittingWindow", _FakeWindow)

        class _DialogMustNotBeConstructed:
            def __init__(self, *args, **kwargs):
                raise AssertionError("GlobalFitConfigDialog must not be used in the launch flow")

        monkeypatch.setattr("kindred.gui.fitting.global_fit_config.GlobalFitConfigDialog", _DialogMustNotBeConstructed)

        main_window._run_global_fit()
        assert created, "Global fitting window should be opened"
        assert len(created["kwargs"].get("dataset_entries") or []) == 2


class TestDSLParsingEdgeCases:
    """Test DSL parsing handles edge cases."""

    def test_simulation_with_empty_mechanism(self, main_window):
        """Test running simulation with empty mechanism."""
        main_window._mechanism_editor._reactions_text.clear()

        with patch.object(QtWidgets.QMessageBox, 'warning') as mock_warning:
            main_window.simulation_controller.run_simulation()
            # Should warn about no mechanism
            assert mock_warning.called

    def test_simulation_with_malformed_dsl(self, main_window):
        """Test simulation with syntactically invalid DSL."""
        # Set malformed DSL
        main_window._mechanism_editor._reactions_text.setPlainText(
            "this is not valid DSL at all!!!"
        )

        # Short-circuit the worker to emit an error immediately without threading.
        with patch(
            "kindred.gui.simulation_worker.SimulationWorker.start",
            lambda self: self.error.emit("Failed to parse mechanism"),
        ):
            with patch.object(QtWidgets.QMessageBox, "critical") as mock_critical:
                main_window.simulation_controller.run_simulation()
                QtCore.QCoreApplication.processEvents()
                assert mock_critical.called


class TestMenuActionsSafetyNet:
    """Test that all menu actions are safe to call without setup."""
    @pytest.mark.parametrize(
        "opener",
        [
            "_configure_fitting",
            "_open_solver_settings",
            "_open_temperature_schedule_editor",
            "_open_template_manager",
        ],
    )
    def test_dialog_opens_without_setup(self, main_window, opener):
        """Each menu dialog should open safely even when exec rejects immediately."""
        with patch.object(QtWidgets.QDialog, "exec", return_value=QtWidgets.QDialog.Rejected):
            getattr(main_window, opener)()
