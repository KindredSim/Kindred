"""
Results tab widget for the fitting window.

Extracted from FittingWindow — owns the run-stamp data and statistics
(displayed via a non-modal popup dialog triggered from the footer).
Read-only display with one-way data flow from fit-completion results.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

from kindred.gui.widgets.grid_plot_view import GridPlotView

if TYPE_CHECKING:
    from kindred.core.analysis.global_fitting import GlobalFitResult


def _get_clipboard():
    """
    Clipboard accessor seam (monkeypatchable in tests).

    Returns a Qt clipboard-like object with `setText(str)` / `text()` when available,
    otherwise returns None.
    """
    try:
        app = QtWidgets.QApplication.instance()
        return app.clipboard() if app is not None else None
    except Exception:
        return None


class ResultsSummaryDialog(QtWidgets.QDialog):
    """Non-modal popup displaying fit run stamp and statistics."""

    statusMessage = Signal(str)

    def __init__(
        self,
        stamp: dict,
        stamp_hash: str,
        stamp_short: str,
        stats: Optional[Dict[str, Any]] = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("Results Summary")
        self.resize(480, 360)

        self._stamp = dict(stamp)
        self._stamp_hash = str(stamp_hash)
        self._stamp_short = str(stamp_short)

        layout = QtWidgets.QVBoxLayout(self)

        info_label = QtWidgets.QLabel("Review and copy the most recent fit run stamp.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(info_label)

        row = QtWidgets.QHBoxLayout()
        self._run_stamp_label = QtWidgets.QLabel(f"Stamp: {stamp_short}")
        self._run_stamp_label.setObjectName("global_fit_run_stamp_label")
        self._run_stamp_label.setStyleSheet("font-size: 11px;")

        self._copy_stamp_button = QtWidgets.QPushButton("Copy")
        self._copy_stamp_button.setObjectName("global_fit_copy_stamp_button")
        self._copy_stamp_button.clicked.connect(self._on_copy_run_stamp_short)

        self._copy_stamp_json_button = QtWidgets.QPushButton("Copy JSON")
        self._copy_stamp_json_button.setObjectName("global_fit_copy_stamp_json_button")
        self._copy_stamp_json_button.clicked.connect(self._on_copy_run_stamp_json)

        row.addWidget(self._run_stamp_label, stretch=1)
        row.addWidget(self._copy_stamp_button)
        row.addWidget(self._copy_stamp_json_button)
        layout.addLayout(row)

        # Statistics form
        stats_group = QtWidgets.QGroupBox("Fit Statistics")
        form = QtWidgets.QFormLayout(stats_group)
        self._stats_labels: Dict[str, QtWidgets.QLabel] = {}
        for label_text in ["Datasets", "Series", "Points", "Parameters", "DF", "SSQ", "Weighted SSQ", "-logL"]:
            value_label = QtWidgets.QLabel("\u2014")
            form.addRow(f"{label_text}:", value_label)
            self._stats_labels[label_text] = value_label
        layout.addWidget(stats_group)

        if stats:
            self._update_stats(stats)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _update_stats(self, stats: Dict[str, Any]) -> None:
        for key, label in self._stats_labels.items():
            value = stats.get(key)
            if value is None:
                label.setText("\u2014")
            elif isinstance(value, float):
                label.setText(f"{value:.6g}")
            else:
                label.setText(str(value))

    # ------------------------------------------------------------------
    # Copy handlers
    # ------------------------------------------------------------------

    def _on_copy_run_stamp_short(self) -> None:
        stamp_short = self._stamp_short.strip()
        if not stamp_short:
            return
        clipboard = _get_clipboard()
        if clipboard is None:
            self.statusMessage.emit("Clipboard unavailable")
            return
        try:
            clipboard.setText(stamp_short)
        except Exception:
            self.statusMessage.emit("Failed to copy stamp")
            return
        self.statusMessage.emit("Copied stamp hash")

    def refresh(
        self,
        stamp: dict,
        stamp_hash: str,
        stamp_short: str,
        stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update all stamp and stats content from the owning tab."""
        self._stamp = dict(stamp)
        self._stamp_hash = str(stamp_hash)
        self._stamp_short = str(stamp_short)
        self._run_stamp_label.setText(f"Stamp: {stamp_short}")
        self._update_stats(stats or {})

    def _on_copy_run_stamp_json(self) -> None:
        stamp = self._stamp
        if not isinstance(stamp, dict) or not stamp:
            return
        clipboard = _get_clipboard()
        if clipboard is None:
            self.statusMessage.emit("Clipboard unavailable")
            return
        try:
            text = json.dumps(stamp, sort_keys=True, indent=2, ensure_ascii=True)
        except Exception:
            self.statusMessage.emit("Failed to serialize stamp")
            return
        try:
            clipboard.setText(text)
        except Exception:
            self.statusMessage.emit("Failed to copy stamp")
            return
        self.statusMessage.emit("Copied stamp JSON")


# Keep old name importable for tests that reference it directly.
RunStampDialog = ResultsSummaryDialog


def _as_float_array(values: object) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def _normalize_species_names(names: Sequence[object]) -> List[str]:
    return [str(name).strip() for name in names if str(name).strip()]


def _format_metric(value: object) -> str:
    try:
        return f"{float(value):.2e}"
    except Exception:
        return "nan"


def _objective_cost_from_result(result: "GlobalFitResult") -> float:
    objective = _as_float_array(getattr(result, "objective_residuals", None))
    if objective.size:
        return float(np.sum(objective**2))
    return float(
        sum(
            np.sum(_as_float_array(getattr(info, "residuals", None)) ** 2)
            for info in (getattr(result, "dataset_info", None) or [])
        )
    )


class FitTrackerPanel(QtWidgets.QWidget):
    """Compact text-only fit tracker surface for the Results tab."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("global_fit_tracker_panel")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setMinimumHeight(88)
        self.setMaximumHeight(124)
        self._previous_cost: Optional[float] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        self._summary_label = QtWidgets.QLabel(self)
        self._summary_label.setObjectName("global_fit_tracker_summary_label")
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._summary_label)

        self._params_label = QtWidgets.QLabel(self)
        self._params_label.setObjectName("global_fit_tracker_params_label")
        self._params_label.setWordWrap(True)
        self._params_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._params_label, stretch=1)

        self.clear()

    def clear(self) -> None:
        self._previous_cost = None
        self._summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._summary_label.setStyleSheet("color: #888; font-size: 12px;")
        self._summary_label.setText("Run a fit to see results here.")
        self._params_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._params_label.setStyleSheet("color: #888; font-size: 11px;")
        self._params_label.setText("")

    def update_from_best(self, payload_dict: Dict[str, Any]) -> None:
        iteration = int(payload_dict.get("iteration", 0))
        try:
            cost = float(payload_dict.get("cost"))
        except Exception:
            cost = float("nan")
        delta_text = ""
        if self._previous_cost is not None and np.isfinite(cost):
            delta_text = f" (delta {cost - self._previous_cost:.2e})"
        self._previous_cost = cost if np.isfinite(cost) else self._previous_cost
        self._set_live_text(
            f"Iter {iteration} | Cost {_format_metric(cost)}{delta_text}",
            dict(payload_dict.get("shared_params") or {}),
        )

    def update_final(self, iteration: int, cost: float, shared_params: Dict[str, float]) -> None:
        self._previous_cost = float(cost) if np.isfinite(float(cost)) else self._previous_cost
        self._set_live_text(
            f"Iter {int(iteration)} | Cost {_format_metric(cost)}",
            dict(shared_params or {}),
        )

    def _set_live_text(self, summary_text: str, shared_params: Dict[str, float]) -> None:
        self._summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._summary_label.setStyleSheet("font-size: 12px; font-weight: 600;")
        self._summary_label.setText(summary_text)
        self._params_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._params_label.setStyleSheet("font-size: 11px;")
        if shared_params:
            parts = [f"{name}={value:.6g}" for name, value in sorted(shared_params.items(), key=lambda item: str(item[0]))]
            self._params_label.setText(" | ".join(parts))
        else:
            self._params_label.setText("")


class RunResultsTab(QtWidgets.QWidget):
    """Drop-in replacement for FittingWindow._create_run_results_tab()."""

    statusMessage = Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_run_stamp: dict = {}
        self._last_run_stamp_hash: str = ""
        self._last_run_stamp_short: str = ""
        self._last_stats: Dict[str, Any] = {}
        self._stamp_dialog: Optional[ResultsSummaryDialog] = None
        self._dark_mode = False
        self._dataset_entries: List[Dict[str, Any]] = []
        self._fit_targets_by_dataset: Dict[str, List[str]] = {}
        self._dataset_plot_views: Dict[str, GridPlotView] = {}
        self._all_datasets_plot_view: Optional[GridPlotView] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self._tracker_panel = FitTrackerPanel(self)
        layout.addWidget(self._tracker_panel, stretch=0)

        self._subtabs = QtWidgets.QTabBar(self)
        self._subtabs.setObjectName("global_fit_results_subtabs")
        self._subtabs.setDocumentMode(True)
        self._subtabs.setDrawBase(False)
        self._subtabs.hide()
        layout.addWidget(self._subtabs, stretch=0)

        self._subtab_stack = QtWidgets.QStackedWidget(self)
        self._subtab_stack.setObjectName("global_fit_results_subtab_stack")
        layout.addWidget(self._subtab_stack, stretch=1)

        self._subtabs.currentChanged.connect(self._subtab_stack.setCurrentIndex)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_run_stamp(self, stamp: dict, stamp_hash: str, stamp_short: str) -> None:
        """Store stamp data for later display via the popup dialog."""
        self._last_run_stamp = dict(stamp)
        self._last_run_stamp_hash = str(stamp_hash)
        self._last_run_stamp_short = str(stamp_short)
        self._last_stats = {}
        self._tracker_panel.clear()
        self._refresh_plot_views()
        if self._stamp_dialog is not None and self._stamp_dialog.isVisible():
            self._stamp_dialog.refresh(
                self._last_run_stamp,
                self._last_run_stamp_hash,
                self._last_run_stamp_short,
                None,
            )

    def open_results_summary_dialog(self) -> None:
        """Show the results summary popup dialog."""
        if not self._last_run_stamp:
            return
        if self._stamp_dialog is not None and self._stamp_dialog.isVisible():
            self._stamp_dialog.raise_()
            self._stamp_dialog.activateWindow()
            return
        dialog = ResultsSummaryDialog(
            self._last_run_stamp,
            self._last_run_stamp_hash,
            self._last_run_stamp_short,
            stats=self._last_stats or None,
            parent=self,
        )
        dialog.statusMessage.connect(self.statusMessage)
        dialog.destroyed.connect(lambda: setattr(self, '_stamp_dialog', None))
        self._stamp_dialog = dialog
        dialog.show()

    def update_statistics(self, stats: Dict[str, Any]) -> None:
        """Store stat values; update dialog if open."""
        self._last_stats = dict(stats)
        if self._stamp_dialog is not None and self._stamp_dialog.isVisible():
            self._stamp_dialog.refresh(
                self._last_run_stamp,
                self._last_run_stamp_hash,
                self._last_run_stamp_short,
                self._last_stats,
            )

    def rebuild_subtabs(
        self,
        dataset_entries: Sequence[Dict[str, Any]],
        fit_targets_by_dataset: Dict[str, Sequence[str]],
    ) -> None:
        self._dataset_entries = [
            entry for entry in list(dataset_entries or [])
            if bool(entry.get("include", True))
        ]
        self._fit_targets_by_dataset = {
            str(ds_id): _normalize_species_names(list(species or []))
            for ds_id, species in (fit_targets_by_dataset or {}).items()
        }
        self._tracker_panel.clear()
        self._clear_subtabs()
        self._dataset_plot_views = {}

        for entry in self._dataset_entries:
            ds_id = str(entry.get("id") or "").strip()
            if not ds_id:
                continue
            title = str(entry.get("label") or ds_id)
            plot_view = self._create_plot_view(f"global_fit_results_plot_{ds_id}")
            self._dataset_plot_views[ds_id] = plot_view
            self._subtab_stack.addWidget(plot_view)
            self._subtabs.addTab(title)

        self._all_datasets_plot_view = None
        if self._dataset_entries:
            self._all_datasets_plot_view = self._create_plot_view("global_fit_results_plot_all_datasets")
            self._subtab_stack.addWidget(self._all_datasets_plot_view)
            self._subtabs.addTab("All Datasets")

        self._subtabs.setVisible(self._subtabs.count() > 0)
        if self._subtabs.count() > 0:
            self._subtabs.setCurrentIndex(0)
            self._subtab_stack.setCurrentIndex(0)
        self._refresh_plot_views()

    def push_live_update(self, payload: Dict[str, Any]) -> None:
        if any(key in payload for key in ("iteration", "cost", "shared_params")):
            self._tracker_panel.update_from_best(payload)
        plot_model_series = payload.get("plot_model_series")
        model_series = plot_model_series if isinstance(plot_model_series, dict) and plot_model_series else (payload.get("model_series") or {})
        plot_model_x = payload.get("plot_model_x") if isinstance(plot_model_series, dict) and plot_model_series else {}
        dataset_stats = payload.get("dataset_stats") or {}
        self._refresh_plot_views(
            model_series_by_dataset=model_series if isinstance(model_series, dict) else {},
            model_x_by_dataset=plot_model_x if isinstance(plot_model_x, dict) else {},
            dataset_stats_by_dataset=dataset_stats if isinstance(dataset_stats, dict) else {},
        )

    def push_final_result(self, result: "GlobalFitResult", dataset_entries: Sequence[Dict[str, Any]]) -> None:
        current_ids = [str(entry.get("id") or "").strip() for entry in self._dataset_entries]
        incoming_ids = [
            str(entry.get("id") or "").strip()
            for entry in list(dataset_entries or [])
            if bool(entry.get("include", True))
        ]
        if current_ids != incoming_ids:
            self.rebuild_subtabs(dataset_entries, self._fit_targets_by_dataset)

        dataset_stats = {
            str(info.dataset_id): {
                "chi_squared": float(info.chi_squared),
                "r_squared": float(info.r_squared),
            }
            for info in (getattr(result, "dataset_info", None) or [])
        }
        plot_series = getattr(result, "plot_model_series", None) or {}
        use_plot_x = bool(isinstance(plot_series, dict) and plot_series)
        model_series = plot_series if use_plot_x else (getattr(result, "model_series", None) or {})
        model_x_by_dataset = (getattr(result, "plot_model_x", None) or {}) if use_plot_x else {}
        self._tracker_panel.update_final(
            iteration=int(getattr(result, "nfev", 0)),
            cost=_objective_cost_from_result(result),
            shared_params=dict(getattr(result, "shared_params", {}) or {}),
        )
        self._refresh_plot_views(
            model_series_by_dataset=model_series if isinstance(model_series, dict) else {},
            model_x_by_dataset=model_x_by_dataset if isinstance(model_x_by_dataset, dict) else {},
            dataset_stats_by_dataset=dataset_stats,
        )

    def clear(self) -> None:
        self._dataset_entries = []
        self._fit_targets_by_dataset = {}
        self._dataset_plot_views = {}
        self._all_datasets_plot_view = None
        self._tracker_panel.clear()
        self._clear_subtabs()

    def set_view_autorange_locked(self, running: bool) -> None:
        for plot_view in list(self._dataset_plot_views.values()):
            plot_view.set_autorange_locked(bool(running))
        if self._all_datasets_plot_view is not None:
            self._all_datasets_plot_view.set_autorange_locked(bool(running))

    def set_dark_mode(self, enabled: bool) -> None:
        self._dark_mode = bool(enabled)
        for plot_view in list(self._dataset_plot_views.values()):
            plot_view.set_dark_mode(self._dark_mode)
        if self._all_datasets_plot_view is not None:
            self._all_datasets_plot_view.set_dark_mode(self._dark_mode)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear_subtabs(self) -> None:
        while self._subtabs.count():
            self._subtabs.removeTab(self._subtabs.count() - 1)
        while self._subtab_stack.count():
            widget = self._subtab_stack.widget(0)
            self._subtab_stack.removeWidget(widget)
            widget.deleteLater()
        self._subtabs.hide()

    def _create_plot_view(self, object_name: str) -> GridPlotView:
        plot_view = GridPlotView(self)
        plot_view.setObjectName(object_name)
        plot_view.set_controls_visible(False)
        plot_view.set_dark_mode(self._dark_mode)
        return plot_view

    def _refresh_plot_views(
        self,
        *,
        model_series_by_dataset: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        model_x_by_dataset: Optional[Dict[str, np.ndarray]] = None,
        dataset_stats_by_dataset: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        model_series_by_dataset = model_series_by_dataset or {}
        model_x_by_dataset = model_x_by_dataset or {}
        dataset_stats_by_dataset = dataset_stats_by_dataset or {}
        overlay_payloads: List[Dict[str, Any]] = []
        overlay_species: set[str] = set()
        for entry in self._dataset_entries:
            ds_id = str(entry.get("id") or "").strip()
            if not ds_id:
                continue
            plot_view = self._dataset_plot_views.get(ds_id)
            if plot_view is None:
                continue
            payload = self._build_grid_dataset_payload(
                entry,
                self._fit_targets_by_dataset.get(ds_id, []),
                model_series=model_series_by_dataset.get(ds_id),
                model_x=model_x_by_dataset.get(ds_id),
                dataset_stats=dataset_stats_by_dataset.get(ds_id),
            )
            species_names = sorted((payload or {}).get("all_species", {}).keys())
            plot_view.set_datasets([payload] if payload is not None else [])
            plot_view.set_species_selection(species_names)
            plot_view.set_dark_mode(self._dark_mode)
            if payload is not None:
                overlay_payloads.append(payload)
                overlay_species.update(species_names)

        if self._all_datasets_plot_view is not None:
            self._all_datasets_plot_view.set_datasets(overlay_payloads)
            self._all_datasets_plot_view.set_species_selection(sorted(overlay_species))
            self._all_datasets_plot_view.set_dark_mode(self._dark_mode)

    def _build_grid_dataset_payload(
        self,
        entry: Dict[str, Any],
        fitted_species_list: Sequence[str],
        *,
        model_series: Optional[Dict[str, np.ndarray]],
        model_x: Optional[np.ndarray],
        dataset_stats: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, Any]]:
        species_data = entry.get("species_data") or entry.get("species") or {}
        if not isinstance(species_data, dict):
            return None
        fitted_species = sorted(
            [
                str(name).strip()
                for name in fitted_species_list
                if str(name).strip() and str(name).strip() in species_data
            ]
        )
        if not fitted_species:
            return None

        all_species = {
            name: _as_float_array(species_data.get(name))
            for name in fitted_species
        }
        current_species = fitted_species[0]
        data_x = _as_float_array(entry.get("x_obs", entry.get("t")))
        data_y = all_species.get(current_species, np.asarray([], dtype=float))
        filtered_model_series = None
        if isinstance(model_series, dict):
            filtered_model_series = {
                name: _as_float_array(values)
                for name, values in model_series.items()
                if str(name).strip() in all_species
            }
            if not filtered_model_series:
                filtered_model_series = None
        chi_squared = None
        r_squared = None
        if isinstance(dataset_stats, dict):
            try:
                chi_squared = float(dataset_stats.get("chi_squared")) if dataset_stats.get("chi_squared") is not None else None
            except Exception:
                chi_squared = None
            try:
                r_squared = float(dataset_stats.get("r_squared")) if dataset_stats.get("r_squared") is not None else None
            except Exception:
                r_squared = None
        x_name = str(entry.get("x_name") or "t").strip() or "t"
        return {
            "name": str(entry.get("label") or entry.get("id") or ""),
            "data_x": data_x,
            "data_y": data_y,
            "model_x": _as_float_array(model_x) if model_x is not None else None,
            "model_y": None if filtered_model_series is None else filtered_model_series.get(current_species),
            "model_series": filtered_model_series,
            "chi_squared": chi_squared,
            "r_squared": r_squared,
            "all_species": all_species,
            "current_species": current_species,
            "x_label": "Time" if x_name == "t" else x_name,
            "x_units": "s" if x_name == "t" else None,
        }
