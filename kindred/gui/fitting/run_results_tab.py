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

from kindred.core.analysis.global_fit_projection import (
    FitRenderDatasetProjection,
    FitRenderProjection,
)
from kindred.gui.widgets.grid_plot_view import GridPlotView
from kindred.gui.display_name_policy import TAB_LABEL_MAX_CHARS, compact_dataset_label, dataset_alias

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
        fitted_params: Optional[Dict[str, float]] = None,
        dataset_fitted_params: Optional[List[tuple]] = None,
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

        # Fitted parameters section
        params_group = QtWidgets.QGroupBox("Fitted Parameters")
        params_group.setObjectName("fitted_params_group")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        params_layout.setContentsMargins(0, 0, 0, 0)
        self._params_content_label = QtWidgets.QLabel()
        self._params_content_label.setObjectName("fitted_params_content_label")
        self._params_content_label.setWordWrap(True)
        self._params_content_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._params_content_label.setStyleSheet("font-size: 11px;")
        self._params_content_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        params_scroll = QtWidgets.QScrollArea()
        params_scroll.setObjectName("fitted_params_scroll")
        params_scroll.setWidgetResizable(True)
        params_scroll.setMaximumHeight(200)
        params_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        params_scroll.setWidget(self._params_content_label)
        params_layout.addWidget(params_scroll)
        self._params_group = params_group
        layout.addWidget(params_group)

        if stats:
            self._update_stats(stats)
        self._update_fitted_params(fitted_params, dataset_fitted_params)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _update_fitted_params(
        self,
        fitted_params: Optional[Dict[str, float]],
        dataset_fitted_params: Optional[List[tuple]] = None,
    ) -> None:
        shared = dict(fitted_params) if fitted_params else {}
        ds_entries = list(dataset_fitted_params) if dataset_fitted_params else []
        if not shared and not ds_entries:
            self._params_content_label.setText("No fitted parameters available.")
            self._params_content_label.setStyleSheet("font-size: 11px; color: #888;")
            return
        self._params_content_label.setStyleSheet("font-size: 11px;")
        lines: List[str] = []
        if shared:
            for name, value in sorted(shared.items(), key=lambda kv: str(kv[0])):
                lines.append(f"{name} = {value:.6g}")
        for ds_label, vals in ds_entries:
            if not isinstance(vals, dict) or not vals:
                continue
            if lines:
                lines.append("")
            lines.append(f"{ds_label}:")
            for name, value in sorted(vals.items(), key=lambda kv: str(kv[0])):
                lines.append(f"  {name} = {value:.6g}")
        self._params_content_label.setText("\n".join(lines))

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
        fitted_params: Optional[Dict[str, float]] = None,
        dataset_fitted_params: Optional[List[tuple]] = None,
    ) -> None:
        """Update all stamp and stats content from the owning tab."""
        self._stamp = dict(stamp)
        self._stamp_hash = str(stamp_hash)
        self._stamp_short = str(stamp_short)
        self._run_stamp_label.setText(f"Stamp: {stamp_short}")
        self._update_stats(stats or {})
        self._update_fitted_params(fitted_params, dataset_fitted_params)

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
    _ALL_DATASETS_TAB_KEY = "__all_datasets__"

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_run_stamp: dict = {}
        self._last_run_stamp_hash: str = ""
        self._last_run_stamp_short: str = ""
        self._last_stats: Dict[str, Any] = {}
        self._last_fitted_params: Dict[str, float] = {}
        self._last_dataset_fitted_params: Dict[str, Dict[str, float]] = {}
        self._stamp_dialog: Optional[ResultsSummaryDialog] = None
        self._dark_mode = False
        self._dataset_entries: List[Dict[str, Any]] = []
        self._dataset_tab_ids: List[str] = []
        self._fit_targets_by_dataset: Dict[str, List[str]] = {}
        self._dataset_plot_views: Dict[str, GridPlotView] = {}
        self._all_datasets_plot_view: Optional[GridPlotView] = None
        self._render_projection: Optional[FitRenderProjection] = None
        self._stale_plot_view_keys: set[str] = set()
        self._view_autorange_locked = False

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
        self._subtabs.currentChanged.connect(self._on_subtab_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_run_stamp(self, stamp: dict, stamp_hash: str, stamp_short: str) -> None:
        """Store stamp data for later display via the popup dialog."""
        self._last_run_stamp = dict(stamp)
        self._last_run_stamp_hash = str(stamp_hash)
        self._last_run_stamp_short = str(stamp_short)
        self._last_stats = {}
        self._last_fitted_params = {}
        self._last_dataset_fitted_params = {}
        self._render_projection = None
        self._stale_plot_view_keys.clear()
        self._tracker_panel.clear()
        self._refresh_plot_views(refresh_all=True)
        if self._stamp_dialog is not None and self._stamp_dialog.isVisible():
            self._stamp_dialog.refresh(
                self._last_run_stamp,
                self._last_run_stamp_hash,
                self._last_run_stamp_short,
                None,
                fitted_params=None,
                dataset_fitted_params=None,
            )

    def clear_fitted_params(self) -> None:
        """Clear stored fitted params (e.g. after an invalid/failed fit)."""
        self._last_fitted_params = {}
        self._last_dataset_fitted_params = {}

    def _clear_failed_run_state(
        self,
        dataset_entries: Sequence[Dict[str, Any]],
        fit_targets_by_dataset: Dict[str, Sequence[str]],
    ) -> None:
        # Clearing the run stamp is the authoritative reset for summary, tracker,
        # cached model payloads, and any open summary dialog state.
        self.set_run_stamp({}, "", "")
        self.rebuild_subtabs(dataset_entries, fit_targets_by_dataset)

    def _dataset_label_for_id(self, dataset_id: str) -> str:
        ds_id = str(dataset_id or "").strip()
        for entry in self._dataset_entries or []:
            if str(entry.get("id") or "").strip() == ds_id:
                return str(entry.get("label", "") or "").strip() or str(dataset_id)
        return str(dataset_id)

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
            fitted_params=self._last_fitted_params or None,
            dataset_fitted_params=self._format_dataset_params_for_dialog(),
            parent=self,
        )
        dialog.statusMessage.connect(self.statusMessage)
        dialog.destroyed.connect(lambda: setattr(self, '_stamp_dialog', None))
        self._stamp_dialog = dialog
        dialog.show()

    def _format_dataset_params_for_dialog(self) -> Optional[List[tuple]]:
        """Return [(display_label, {name: value}), ...] keyed by dataset ID order.

        Uses display labels with disambiguation when two datasets share a label.
        Returns None when there are no dataset-specific fitted parameters.
        """
        if not self._last_dataset_fitted_params:
            return None
        ids = sorted(
            (ds_id for ds_id, vals in self._last_dataset_fitted_params.items()
             if isinstance(vals, dict) and vals),
            key=str,
        )
        if not ids:
            return None
        raw_labels = {ds_id: self._dataset_label_for_id(ds_id) for ds_id in ids}
        label_counts: Dict[str, int] = {}
        for lbl in raw_labels.values():
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        result: List[tuple] = []
        for ds_id in ids:
            lbl = raw_labels[ds_id]
            display_full = f"{lbl} ({ds_id})" if label_counts[lbl] > 1 else lbl
            result.append((display_full, dict(self._last_dataset_fitted_params[ds_id])))
        return result

    def update_statistics(self, stats: Dict[str, Any]) -> None:
        """Store stat values; update dialog if open."""
        self._last_stats = dict(stats)
        if self._stamp_dialog is not None and self._stamp_dialog.isVisible():
            self._stamp_dialog.refresh(
                self._last_run_stamp,
                self._last_run_stamp_hash,
                self._last_run_stamp_short,
                self._last_stats,
                fitted_params=self._last_fitted_params or None,
                dataset_fitted_params=self._format_dataset_params_for_dialog(),
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
        self._dataset_tab_ids = []

        for index, entry in enumerate(self._dataset_entries):
            ds_id = str(entry.get("id") or "").strip()
            if not ds_id:
                continue
            full_title = str(entry.get("label") or ds_id)
            compact_title = compact_dataset_label(full_title, max_chars=TAB_LABEL_MAX_CHARS)
            plot_view = self._create_plot_view(f"global_fit_results_plot_{ds_id}")
            self._dataset_tab_ids.append(ds_id)
            self._dataset_plot_views[ds_id] = plot_view
            self._subtab_stack.addWidget(plot_view)
            tab_index = self._subtabs.addTab(compact_title.display)
            self._subtabs.setTabToolTip(tab_index, compact_title.full)
            self._subtabs.setTabData(tab_index, {"dataset_id": ds_id, "alias": dataset_alias(index), "full_label": compact_title.full})

        self._all_datasets_plot_view = None
        if self._dataset_entries:
            self._all_datasets_plot_view = self._create_plot_view("global_fit_results_plot_all_datasets")
            self._subtab_stack.addWidget(self._all_datasets_plot_view)
            self._subtabs.addTab("All Datasets")

        self._subtabs.setVisible(self._subtabs.count() > 0)
        if self._subtabs.count() > 0:
            self._subtabs.setCurrentIndex(0)
            self._subtab_stack.setCurrentIndex(0)
        self._refresh_plot_views(refresh_all=True)

    def push_live_update(
        self,
        payload: Dict[str, Any],
        *,
        refresh_all: bool = False,
        update_tracker: bool = True,
    ) -> None:
        if update_tracker and any(key in payload for key in ("iteration", "cost", "shared_params")):
            self._tracker_panel.update_from_best(payload)

    def push_render_projection(
        self,
        projection: FitRenderProjection,
        *,
        refresh_all: bool = False,
    ) -> None:
        self._render_projection = projection
        self._refresh_plot_views(refresh_all=refresh_all)

    def clear_render_projection(
        self,
        dataset_ids: Optional[Sequence[str]] = None,
        *,
        refresh_all: bool = True,
    ) -> None:
        _ = dataset_ids
        self._render_projection = None
        self._refresh_plot_views(refresh_all=refresh_all)

    def push_final_result(self, result: "GlobalFitResult", dataset_entries: Sequence[Dict[str, Any]]) -> None:
        current_ids = [str(entry.get("id") or "").strip() for entry in self._dataset_entries]
        incoming_ids = [
            str(entry.get("id") or "").strip()
            for entry in list(dataset_entries or [])
            if bool(entry.get("include", True))
        ]
        if current_ids != incoming_ids:
            self.rebuild_subtabs(dataset_entries, self._fit_targets_by_dataset)

        self._last_fitted_params = dict(getattr(result, "shared_params", {}) or {})
        raw_ds_params = getattr(result, "dataset_params", None) or {}
        self._last_dataset_fitted_params = {
            str(ds_id): dict(vals)
            for ds_id, vals in raw_ds_params.items()
            if isinstance(vals, dict) and vals
        }
        self._tracker_panel.update_final(
            iteration=int(getattr(result, "nfev", 0)),
            cost=_objective_cost_from_result(result),
            shared_params=dict(getattr(result, "shared_params", {}) or {}),
        )

    def clear(self) -> None:
        self._dataset_entries = []
        self._dataset_tab_ids = []
        self._fit_targets_by_dataset = {}
        self._dataset_plot_views = {}
        self._all_datasets_plot_view = None
        self._render_projection = None
        self._last_fitted_params = {}
        self._last_dataset_fitted_params = {}
        self._stale_plot_view_keys.clear()
        self._tracker_panel.clear()
        self._clear_subtabs()

    def set_view_autorange_locked(self, running: bool) -> None:
        self._view_autorange_locked = bool(running)
        for plot_view in list(self._dataset_plot_views.values()):
            plot_view.set_autorange_locked(self._view_autorange_locked)
        if self._all_datasets_plot_view is not None:
            self._all_datasets_plot_view.set_autorange_locked(self._view_autorange_locked)

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
        self._stale_plot_view_keys.clear()
        self._subtabs.hide()

    def _create_plot_view(self, object_name: str) -> GridPlotView:
        plot_view = GridPlotView(self)
        plot_view.setObjectName(object_name)
        plot_view.set_controls_visible(False)
        plot_view.set_dark_mode(self._dark_mode)
        plot_view.set_autorange_locked(self._view_autorange_locked)
        return plot_view

    def _refresh_plot_views(
        self,
        *,
        refresh_all: bool = True,
    ) -> None:
        plot_keys = self._all_plot_view_keys()
        if refresh_all:
            for key in plot_keys:
                self._refresh_plot_view_for_key(key)
            self._stale_plot_view_keys.clear()
            return

        visible_key = self._current_plot_view_key()
        if visible_key is None:
            self._stale_plot_view_keys.update(plot_keys)
            return

        self._refresh_plot_view_for_key(visible_key)
        self._stale_plot_view_keys.update(plot_keys)
        self._stale_plot_view_keys.discard(visible_key)

    def _apply_plot_species_selection(self, plot_view: GridPlotView, species_names: Sequence[str]) -> None:
        species = [str(name).strip() for name in species_names if str(name).strip()]
        if not species:
            return

        def _safe_set() -> None:
            try:
                plot_view.set_species_selection(species)
            except RuntimeError:
                return
            except Exception:
                return

        redraw_timer = getattr(plot_view, "_redraw_timer", None)
        if redraw_timer is not None and getattr(redraw_timer, "isActive", lambda: False)():
            def _apply_after_redraw() -> None:
                try:
                    redraw_timer.timeout.disconnect(_apply_after_redraw)
                except (RuntimeError, TypeError):
                    pass
                _safe_set()

            redraw_timer.timeout.connect(_apply_after_redraw)
            return

        _safe_set()

    def _all_plot_view_keys(self) -> List[str]:
        keys = list(self._dataset_tab_ids)
        if self._all_datasets_plot_view is not None:
            keys.append(self._ALL_DATASETS_TAB_KEY)
        return keys

    def _current_plot_view_key(self) -> Optional[str]:
        return self._plot_view_key_for_index(self._subtabs.currentIndex())

    def _plot_view_key_for_index(self, index: int) -> Optional[str]:
        if index < 0:
            return None
        if 0 <= index < len(self._dataset_tab_ids):
            return self._dataset_tab_ids[index]
        if (
            index == len(self._dataset_tab_ids)
            and self._all_datasets_plot_view is not None
        ):
            return self._ALL_DATASETS_TAB_KEY
        return None

    def _dataset_entry_for_id(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        for entry in self._dataset_entries:
            if str(entry.get("id") or "").strip() == str(dataset_id or "").strip():
                return entry
        return None

    def _render_projection_for_dataset(self, dataset_id: str) -> Optional[FitRenderDatasetProjection]:
        projection = self._render_projection
        if not isinstance(projection, FitRenderProjection):
            return None
        dataset_projection = projection.datasets.get(str(dataset_id or "").strip())
        if not isinstance(dataset_projection, FitRenderDatasetProjection):
            return None
        if dataset_projection.status != "ok":
            return None
        return dataset_projection

    def _refresh_plot_view_for_key(self, key: str) -> None:
        if key == self._ALL_DATASETS_TAB_KEY:
            plot_view = self._all_datasets_plot_view
            if plot_view is None:
                return
            overlay_payloads: List[Dict[str, Any]] = []
            overlay_species: set[str] = set()
            for ds_id in self._dataset_tab_ids:
                entry = self._dataset_entry_for_id(ds_id)
                if entry is None:
                    continue
                payload = self._build_grid_dataset_payload(
                    entry,
                    self._fit_targets_by_dataset.get(ds_id, []),
                    dataset_projection=self._render_projection_for_dataset(ds_id),
                )
                if payload is None:
                    continue
                overlay_payloads.append(payload)
                overlay_species.update((payload.get("all_species") or {}).keys())
            plot_view.set_datasets(overlay_payloads)
            self._apply_plot_species_selection(plot_view, sorted(overlay_species))
            return

        plot_view = self._dataset_plot_views.get(key)
        entry = self._dataset_entry_for_id(key)
        if plot_view is None or entry is None:
            return
        payload = self._build_grid_dataset_payload(
            entry,
            self._fit_targets_by_dataset.get(key, []),
            dataset_projection=self._render_projection_for_dataset(key),
        )
        species_names = sorted((payload or {}).get("all_species", {}).keys())
        plot_view.set_datasets([payload] if payload is not None else [])
        self._apply_plot_species_selection(plot_view, species_names)

    def _on_subtab_changed(self, index: int) -> None:
        key = self._plot_view_key_for_index(int(index))
        if key is None or key not in self._stale_plot_view_keys:
            return
        self._refresh_plot_view_for_key(key)
        self._stale_plot_view_keys.discard(key)

    def _build_grid_dataset_payload(
        self,
        entry: Dict[str, Any],
        fitted_species_list: Sequence[str],
        *,
        dataset_projection: Optional[FitRenderDatasetProjection] = None,
    ) -> Optional[Dict[str, Any]]:
        species_data = entry.get("species_data") or entry.get("species") or {}
        if not isinstance(species_data, dict):
            return None
        requested_species = sorted(
            [
                str(name).strip()
                for name in fitted_species_list
                if str(name).strip() and str(name).strip() in species_data
            ]
        )
        if dataset_projection is not None:
            source_species_data = dict(dataset_projection.observed_series)
            fitted_species = [
                name
                for name in requested_species
                if name in source_species_data
            ]
            if not fitted_species:
                fitted_species = sorted(str(name) for name in source_species_data if str(name))
        else:
            source_species_data = species_data
            fitted_species = list(requested_species)
        if not fitted_species:
            return None
        all_species = {
            name: _as_float_array(source_species_data.get(name))
            for name in fitted_species
        }
        current_species = fitted_species[0]
        data_x = (
            dataset_projection.observed_x_for_species(current_species)
            if dataset_projection is not None
            else _as_float_array(entry.get("x_obs", entry.get("t")))
        )
        data_y = all_species.get(current_species, np.asarray([], dtype=float))
        chi_squared = None
        r_squared = None
        if dataset_projection is not None:
            dataset_stats = dict(dataset_projection.dataset_stats)
            try:
                chi_squared = float(dataset_stats.get("chi_squared")) if dataset_stats.get("chi_squared") is not None else None
            except Exception:
                chi_squared = None
            try:
                r_squared = float(dataset_stats.get("r_squared")) if dataset_stats.get("r_squared") is not None else None
            except Exception:
                r_squared = None
        x_name = str(entry.get("x_name") or "t").strip() or "t"
        x_label = (
            dataset_projection.observed_x_label
            if dataset_projection is not None
            else ("Time" if x_name == "t" else x_name)
        )
        entry_id = str(entry.get("id") or "").strip()
        entry_index = 0
        try:
            entry_index = next(
                i for i, candidate in enumerate(self._dataset_entries)
                if str(candidate.get("id") or "").strip() == entry_id
            )
        except Exception:
            entry_index = 0
        full_name = str(entry.get("label") or entry.get("id") or "")
        return {
            "name": dataset_alias(entry_index),
            "full_name": full_name,
            "data_x": data_x,
            "data_y": data_y,
            "fit_render_projection": dataset_projection,
            "chi_squared": chi_squared,
            "r_squared": r_squared,
            "all_species": all_species,
            "current_species": current_species,
            "x_label": x_label,
            "x_units": "s" if x_name == "t" else None,
        }
