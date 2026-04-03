"""
Regression tests for fitted parameter display in completion dialog
and ResultsSummaryDialog.

Tests must fail before the feature is implemented and pass after.
"""
from __future__ import annotations

import numpy as np
import pytest

from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult


pytestmark = [pytest.mark.gui]


def _make_result(
    shared_params=None,
    dataset_params=None,
    success=True,
    chi_squared=1.234,
    dataset_errors=None,
):
    """Build a minimal GlobalFitResult with the given parameters."""
    result = GlobalFitResult(
        success=success,
        shared_params=shared_params or {},
        dataset_params=dataset_params or {},
        uncertainties=None,
        global_chi_squared=chi_squared,
        global_r_squared=0.95,
        dataset_info=[
            DatasetFitInfo(
                dataset_id="ds1",
                r_squared=0.95,
                chi_squared=chi_squared,
                rmse=0.1,
                mae=0.1,
                residuals=np.array([0.1, 0.1], dtype=float),
                n_points=10,
                weight=1.0,
            )
        ],
        nfev=42,
        message="Optimization terminated successfully.",
    )
    if dataset_errors is not None:
        result.dataset_errors = dict(dataset_errors)
    return result


def _make_window():
    """Build a minimal FittingWindow for spec tests."""
    from kindred.gui.fitting.window import FittingWindow

    t = np.arange(0, 10, dtype=float)
    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": t.copy(),
            "species_data": {"A": t.copy()},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]

    def simulation_func(_params):
        return {"t": t.copy(), "species": {"A": t.copy()}}

    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=simulation_func,
        mechanism_species=["A"],
    )


# ------------------------------------------------------------------
# 1. Completion dialog spec includes parameter values
# ------------------------------------------------------------------


class TestCompletionDialogParamValues:
    """Completion QMessageBox text must contain fitted parameter values."""

    def test_success_includes_param_values(self, qt_app):
        window = _make_window()
        try:
            qt_app.processEvents()
            result = _make_result(shared_params={"k1": 0.12345, "k2": 6.789})
            severity, title, text = window._global_fit_completion_dialog_spec(result)
            assert severity == "ok"
            assert "k1" in text
            assert "k2" in text
            assert "0.12345" in text
            assert "6.789" in text
        finally:
            window.close()
            qt_app.processEvents()

    def test_warning_includes_param_values(self, qt_app):
        window = _make_window()
        try:
            qt_app.processEvents()
            result = _make_result(shared_params={"k1": 0.5}, success=False)
            severity, title, text = window._global_fit_completion_dialog_spec(result)
            assert severity == "warn"
            assert "k1" in text
        finally:
            window.close()
            qt_app.processEvents()

    def test_many_params_truncated(self, qt_app):
        window = _make_window()
        try:
            qt_app.processEvents()
            # 15 parameters should trigger truncation (>10)
            params = {f"p{i}": float(i) * 0.1 for i in range(15)}
            result = _make_result(shared_params=params)
            severity, title, text = window._global_fit_completion_dialog_spec(result)
            assert severity == "ok"
            assert "more" in text.lower()
        finally:
            window.close()
            qt_app.processEvents()

    def test_no_params_no_crash(self, qt_app):
        window = _make_window()
        try:
            qt_app.processEvents()
            result = _make_result(shared_params={})
            severity, title, text = window._global_fit_completion_dialog_spec(result)
            assert severity == "ok"
        finally:
            window.close()
            qt_app.processEvents()

    def test_dataset_params_included(self, qt_app):
        """Dataset-specific fitted values must appear in completion text."""
        window = _make_window()
        try:
            qt_app.processEvents()
            result = _make_result(
                shared_params={"k1": 0.5},
                dataset_params={"ds1": {"init_A": 3.14}},
            )
            severity, title, text = window._global_fit_completion_dialog_spec(result)
            assert severity == "ok"
            assert "init_A" in text
            assert "3.14" in text
        finally:
            window.close()
            qt_app.processEvents()

    def test_dataset_only_params_shown(self, qt_app):
        """Runs with only dataset-specific params must still show values."""
        window = _make_window()
        try:
            qt_app.processEvents()
            result = _make_result(
                shared_params={},
                dataset_params={"ds1": {"init_A": 2.71}},
            )
            severity, title, text = window._global_fit_completion_dialog_spec(result)
            assert severity == "ok"
            assert "init_A" in text
            assert "2.71" in text
        finally:
            window.close()
            qt_app.processEvents()

    def test_failure_does_not_show_params(self, qt_app):
        """Failed fits (dataset_errors) must not show param values."""
        window = _make_window()
        try:
            qt_app.processEvents()
            result = _make_result(
                shared_params={"k1": 0.5},
                dataset_errors={"ds1": "X alignment failed"},
            )
            severity, title, text = window._global_fit_completion_dialog_spec(result)
            assert severity == "fail"
            assert "k1 =" not in text
        finally:
            window.close()
            qt_app.processEvents()


# ------------------------------------------------------------------
# 2. ResultsSummaryDialog shows parameter values
# ------------------------------------------------------------------


class TestResultsSummaryDialogParamValues:
    """ResultsSummaryDialog must display fitted parameters when provided."""

    def test_shows_params_when_provided(self, qt_app):
        from kindred.gui.fitting.run_results_tab import ResultsSummaryDialog

        qt_app.processEvents()
        fitted_params = {"k1": 0.12345, "k2": 6.789}
        dialog = ResultsSummaryDialog(
            stamp={"solver": "test"},
            stamp_hash="abc123",
            stamp_short="abc123",
            stats={"Datasets": 1},
            fitted_params=fitted_params,
        )
        try:
            all_text = _collect_label_text(dialog)
            assert "k1" in all_text
            assert "k2" in all_text
            assert "0.12345" in all_text
            assert "6.789" in all_text
        finally:
            dialog.close()
            qt_app.processEvents()

    def test_no_params_shows_placeholder(self, qt_app):
        from kindred.gui.fitting.run_results_tab import ResultsSummaryDialog

        qt_app.processEvents()
        dialog = ResultsSummaryDialog(
            stamp={"solver": "test"},
            stamp_hash="abc123",
            stamp_short="abc123",
            stats={"Datasets": 1},
            fitted_params={},
        )
        try:
            all_text = _collect_label_text(dialog)
            assert "no fitted parameters" in all_text.lower()
        finally:
            dialog.close()
            qt_app.processEvents()

    def test_no_params_kwarg_no_crash(self, qt_app):
        from kindred.gui.fitting.run_results_tab import ResultsSummaryDialog

        qt_app.processEvents()
        dialog = ResultsSummaryDialog(
            stamp={"solver": "test"},
            stamp_hash="abc123",
            stamp_short="abc123",
        )
        try:
            all_text = _collect_label_text(dialog)
            assert isinstance(all_text, str)
        finally:
            dialog.close()
            qt_app.processEvents()

    def test_refresh_updates_params(self, qt_app):
        from kindred.gui.fitting.run_results_tab import ResultsSummaryDialog

        qt_app.processEvents()
        dialog = ResultsSummaryDialog(
            stamp={"solver": "test"},
            stamp_hash="abc123",
            stamp_short="abc123",
            fitted_params={"k1": 1.0},
        )
        try:
            dialog.refresh(
                stamp={"solver": "test"},
                stamp_hash="def456",
                stamp_short="def456",
                stats=None,
                fitted_params={"k1": 2.0, "k_new": 9.99},
            )
            all_text = _collect_label_text(dialog)
            assert "k_new" in all_text
            assert "9.99" in all_text
        finally:
            dialog.close()
            qt_app.processEvents()

    def test_dataset_params_shown(self, qt_app):
        """Dataset-specific fitted values must appear in the dialog."""
        from kindred.gui.fitting.run_results_tab import ResultsSummaryDialog

        qt_app.processEvents()
        dialog = ResultsSummaryDialog(
            stamp={"solver": "test"},
            stamp_hash="abc123",
            stamp_short="abc123",
            fitted_params={"k1": 0.5},
            dataset_fitted_params=[("Dataset 1", {"init_A": 3.14})],
        )
        try:
            all_text = _collect_label_text(dialog)
            assert "k1" in all_text
            assert "init_A" in all_text
            assert "3.14" in all_text
            assert "Dataset 1" in all_text
        finally:
            dialog.close()
            qt_app.processEvents()

    def test_dataset_only_params_no_placeholder(self, qt_app):
        """Runs with only per-dataset params must not show the placeholder."""
        from kindred.gui.fitting.run_results_tab import ResultsSummaryDialog

        qt_app.processEvents()
        dialog = ResultsSummaryDialog(
            stamp={"solver": "test"},
            stamp_hash="abc123",
            stamp_short="abc123",
            fitted_params={},
            dataset_fitted_params=[("Dataset 1", {"init_A": 2.71})],
        )
        try:
            all_text = _collect_label_text(dialog)
            assert "no fitted parameters" not in all_text.lower()
            assert "init_A" in all_text
        finally:
            dialog.close()
            qt_app.processEvents()

    def test_duplicate_labels_disambiguated(self, qt_app):
        """Datasets sharing the same label must be disambiguated with ID."""
        from kindred.gui.fitting.run_results_tab import RunResultsTab

        qt_app.processEvents()
        tab = RunResultsTab()
        tab._dataset_entries = [
            {"id": "ds1", "label": "Experiment", "include": True},
            {"id": "ds2", "label": "Experiment", "include": True},
        ]
        tab._last_dataset_fitted_params = {
            "ds1": {"init_A": 1.0},
            "ds2": {"init_B": 2.0},
        }
        try:
            formatted = tab._format_dataset_params_for_dialog()
            labels = [lbl for lbl, _vals in formatted]
            assert "Experiment (ds1)" in labels
            assert "Experiment (ds2)" in labels
        finally:
            tab.close()
            qt_app.processEvents()

    def test_unique_labels_not_disambiguated(self, qt_app):
        """Datasets with unique labels should not get ID appended."""
        from kindred.gui.fitting.run_results_tab import RunResultsTab

        qt_app.processEvents()
        tab = RunResultsTab()
        tab._dataset_entries = [
            {"id": "ds1", "label": "Alpha", "include": True},
            {"id": "ds2", "label": "Beta", "include": True},
        ]
        tab._last_dataset_fitted_params = {
            "ds1": {"init_A": 1.0},
            "ds2": {"init_B": 2.0},
        }
        try:
            formatted = tab._format_dataset_params_for_dialog()
            labels = [lbl for lbl, _vals in formatted]
            assert "Alpha" in labels
            assert "Beta" in labels
            assert not any("(" in lbl for lbl in labels)
        finally:
            tab.close()
            qt_app.processEvents()


def _collect_label_text(widget):
    """Collect all QLabel text from a widget tree for assertion."""
    from PySide6 import QtWidgets

    texts = []
    for child in widget.findChildren(QtWidgets.QLabel):
        texts.append(child.text())
    return "\n".join(texts)
