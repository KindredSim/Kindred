"""Physical Fit Results and Summary views.

The session owner retains the admitted Fit candidate.  This module only
renders call-local values and emits ordinary action requests.  In particular,
no candidate, result, report, projection, identity, or Apply policy is kept
by either view.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

from kindred.core.analysis.fit_statistics import FitMetric
from kindred.gui.display_name_policy import (
    STATUS_ITEM_LABEL_MAX_CHARS,
    TAB_LABEL_MAX_CHARS,
    compact_dataset_label,
    compact_text,
    dataset_alias,
)
from kindred.gui.fitting.reporting import fit_candidate_tracker
from kindred.gui.fitting.worker import FitCandidate, FitCurveDataset, FitLiveUpdate
from kindred.gui.widgets.grid_plot_view import DatasetPlotPresentation, GridPlotView

__all__ = ["FitTrackerPanel", "ResultsSummaryDialog", "RunResultsTab"]

ColorResolver = Callable[[str], object]

def _format_metric(value: object) -> str:
    if isinstance(value, FitMetric):
        return (
            f"{value.value:.2e}"
            if value.available
            else f"unavailable ({value.unavailable_reason})"
        )
    try:
        number = float(value)
        return f"{number:.2e}" if math.isfinite(number) else "unavailable"
    except Exception:
        return "unavailable"


def _display(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_display(item) for item in value) + "]"
    return str(value)


class ResultsSummaryDialog(QtWidgets.QDialog):
    """A physical summary popup rendered from one synchronous summary value."""

    def __init__(
        self,
        summary: Mapping[str, object] | None = None,
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("Results Summary")
        self.resize(560, 480)
        layout = QtWidgets.QVBoxLayout(self)
        info_label = QtWidgets.QLabel("Review the most recent fit run.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        self._body = QtWidgets.QTextEdit(self)
        self._body.setObjectName("results_summary_body")
        self._body.setReadOnly(True)
        self._body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._body, stretch=1)
        self.refresh(summary or {})

    def refresh(self, summary: Mapping[str, object]) -> None:
        """Immediately render one call-local plain Summary mapping."""

        report_text = summary.get("report_text") if isinstance(summary, Mapping) else None
        self._body.setPlainText(
            report_text if isinstance(report_text, str) and report_text else "No summary available."
        )

class FitTrackerPanel(QtWidgets.QWidget):
    """Compact physical live/terminal tracker."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("global_fit_tracker_panel")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setMinimumHeight(108)
        self.setMaximumHeight(240)
        self._previous_cost: float | None = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        self._summary_label = QtWidgets.QLabel(self)
        self._summary_label.setObjectName("global_fit_tracker_summary_label")
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._summary_label)
        self._params_label = QtWidgets.QLabel(self)
        self._params_label.setObjectName("global_fit_tracker_params_label")
        self._params_label.setWordWrap(True)
        self._params_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
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

    def update_from_best(
        self,
        update: FitLiveUpdate,
        *,
        dataset_labels: Mapping[str, str] | None = None,
    ) -> None:
        cost = float(update.cost)
        delta_text = ""
        if self._previous_cost is not None and math.isfinite(cost):
            delta_text = f" (delta {cost - self._previous_cost:.2e})"
        if math.isfinite(cost):
            self._previous_cost = cost
        labels = dict(dataset_labels or {})
        lines = [
            f"{compact_text(name, max_chars=STATUS_ITEM_LABEL_MAX_CHARS).display}="
            f"{float(value):.6g}"
            for name, value in update.parameters
        ]
        tooltips = [f"Parameter ID: {name}" for name, _value in update.parameters]
        for index, (dataset_id, values) in enumerate(update.scalar_values_by_dataset):
            if not values:
                continue
            label = compact_dataset_label(
                labels.get(dataset_id, dataset_alias(index)),
                max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
            )
            lines.append(
                f"[{label.display}] "
                + " | ".join(
                    f"{compact_text(name, max_chars=STATUS_ITEM_LABEL_MAX_CHARS).display}="
                    f"{float(value):.6g} (Fit-local; not applied to Run)"
                    for name, value in values
                )
            )
            tooltips.append(f"{label.full}: Dataset ID {dataset_id}")
        for index, (set_id, values) in enumerate(update.initial_values_by_set):
            if not values:
                continue
            label = compact_text(
                f"Initial set {index + 1}",
                max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
            )
            lines.append(
                f"[{label.display}] "
                + " | ".join(
                    f"{compact_text(name, max_chars=STATUS_ITEM_LABEL_MAX_CHARS).display}="
                    f"{float(value):.6g}"
                    for name, value in values
                )
            )
            tooltips.append(f"{label.full}: Set ID {set_id}")
        self._set_live_text(
            f"Objective calls {int(update.iteration)} | Weighted objective SSQ "
            f"{_format_metric(cost)}{delta_text}",
            lines,
            tooltips,
        )

    def update_final(self, candidate: FitCandidate) -> None:
        """Render terminal values synchronously from the retained candidate."""

        summary, lines, tooltips = fit_candidate_tracker(candidate)
        self._set_live_text(summary, lines, tooltips)

    def _set_live_text(
        self,
        summary_text: str,
        parameter_lines: Sequence[str],
        tooltip_lines: Sequence[str],
    ) -> None:
        self._summary_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._summary_label.setStyleSheet("font-size: 12px; font-weight: 600;")
        self._summary_label.setText(summary_text)
        self._params_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._params_label.setStyleSheet("font-size: 11px;")
        self._params_label.setText("\n".join(parameter_lines))
        self._params_label.setToolTip("\n".join(tooltip_lines))


class RunResultsTab(QtWidgets.QWidget):
    """Physical Results action surface with no scientific custody."""

    statusMessage = Signal(str)
    copyAvailableChanged = Signal(bool)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        color_resolver: ColorResolver | None = None,
    ) -> None:
        super().__init__(parent)
        self._color_resolver = color_resolver
        self._summary_dialog: ResultsSummaryDialog | None = None
        self._dark_mode = False
        self._view_autorange_locked = False
        self._curves: tuple[FitCurveDataset, ...] = ()
        self._dataset_presentation: Mapping[str, DatasetPlotPresentation] = (
            MappingProxyType({})
        )
        self._dataset_tab_ids: list[tuple[str, str]] = []
        self._dataset_labels: dict[str, str] = {}
        self._dataset_plot_views: dict[tuple[str, str], GridPlotView] = {}
        self._all_datasets_plot_view: GridPlotView | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        self._tracker_panel = FitTrackerPanel(self)
        layout.addWidget(self._tracker_panel)

        self._render_diagnostic_label = QtWidgets.QLabel(self)
        self._render_diagnostic_label.setWordWrap(True)
        self._render_diagnostic_label.hide()
        layout.addWidget(self._render_diagnostic_label)
        self._terminal_failure_label = QtWidgets.QLabel(self)
        self._terminal_failure_label.setObjectName("global_fit_terminal_failure_label")
        self._terminal_failure_label.setWordWrap(True)
        self._terminal_failure_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._terminal_failure_label.hide()
        layout.addWidget(self._terminal_failure_label)

        self._subtabs = QtWidgets.QTabBar(self)
        self._subtabs.setObjectName("global_fit_results_subtabs")
        self._subtabs.setDocumentMode(True)
        self._subtabs.setDrawBase(False)
        self._subtabs.hide()
        layout.addWidget(self._subtabs)
        self._subtab_stack = QtWidgets.QStackedWidget(self)
        self._subtab_stack.setObjectName("global_fit_results_subtab_stack")
        layout.addWidget(self._subtab_stack, stretch=1)
        self._subtabs.currentChanged.connect(self._on_subtab_changed)

    def set_live_best(self, update: FitLiveUpdate) -> None:
        self._tracker_panel.update_from_best(
            update,
            dataset_labels=self._dataset_labels,
        )

    def set_live_curves(self, curves: Sequence[FitCurveDataset]) -> None:
        self._install_plot_projection(curves, {})

    def set_terminal_presentation(
        self,
        candidate: FitCandidate | None,
        *,
        curves: Sequence[FitCurveDataset] = (),
        curve_failure: str | None = None,
        plot_presentation: Mapping[str, DatasetPlotPresentation] | None = None,
    ) -> bool:
        """Consume one call-local typed terminal presentation."""
        self._terminal_failure_label.hide()
        if candidate is None:
            self._tracker_panel.clear()
            self._render_diagnostic_label.clear()
            self._render_diagnostic_label.hide()
        else:
            self._tracker_panel.update_final(candidate)
        self._install_plot_projection(curves, plot_presentation or {})
        if curve_failure is not None:
            self._render_diagnostic_label.setText(
                "Fit curves unavailable: " + str(curve_failure)
            )
            self._render_diagnostic_label.show()
        return self.copy_eligible()

    def render_results_summary(
        self,
        summary: Mapping[str, object],
    ) -> None:
        """Render a synchronous owner-provided human-readable Summary."""

        if self._summary_dialog is not None and self._summary_dialog.isVisible():
            self._summary_dialog.refresh(summary)
            self._summary_dialog.raise_()
            self._summary_dialog.activateWindow()
            return
        dialog = ResultsSummaryDialog(summary, parent=self)
        dialog.destroyed.connect(lambda: setattr(self, "_summary_dialog", None))
        self._summary_dialog = dialog
        dialog.show()

    def clear_candidate(self) -> None:
        """Clear only physical result widgets; the owner controls custody."""

        self._tracker_panel.clear()
        self._terminal_failure_label.clear()
        self._terminal_failure_label.hide()
        self._render_diagnostic_label.clear()
        self._render_diagnostic_label.hide()
        self._install_plot_projection((), {})
        if self._summary_dialog is not None and self._summary_dialog.isVisible():
            self._summary_dialog.close()

    def set_terminal_failure(self, failure: object) -> None:
        chain: list[str] = []
        current = failure
        while current is not None:
            code = getattr(current, "code", None)
            code_text = f" [{code}]" if code else ""
            message = getattr(current, "message", str(current))
            chain.append(f"{type(current).__name__}{code_text}: {message}")
            current = getattr(current, "cause", None)
        self._terminal_failure_label.setText(
            "Fit failed — " + " Caused by: ".join(chain)
        )
        payload = getattr(failure, "to_payload", None)
        self._terminal_failure_label.setToolTip(
            _display(payload() if callable(payload) else str(failure))
        )
        self._terminal_failure_label.show()

    def set_terminal_unavailable(self, cause: object) -> None:
        chain: list[str] = []
        current = cause
        while current is not None:
            code = getattr(current, "code", None)
            code_text = f" [{code}]" if code else ""
            chain.append(
                f"{type(current).__name__}{code_text}: "
                f"{getattr(current, 'message', current)}"
            )
            current = getattr(current, "cause", None)
        text = "Fit result scientifically unavailable — " + " Caused by: ".join(chain)
        self._terminal_failure_label.setText(text)
        self._terminal_failure_label.show()

    def copy_eligible(self) -> bool:
        return bool(self._curves)

    def copy_all_to_clipboard(self) -> None:
        view = self._all_datasets_plot_view
        if view is None or not self._curves:
            raise ValueError("No Fit curves are available to copy.")
        view.copy_projection_to_clipboard(self._curves)

    def set_view_autorange_locked(self, running: bool) -> None:
        self._view_autorange_locked = bool(running)
        for plot_view in self._dataset_plot_views.values():
            plot_view.set_autorange_locked(self._view_autorange_locked)
        if self._all_datasets_plot_view is not None:
            self._all_datasets_plot_view.set_autorange_locked(self._view_autorange_locked)

    def set_dark_mode(self, enabled: bool) -> None:
        self._dark_mode = bool(enabled)
        for plot_view in self._dataset_plot_views.values():
            plot_view.set_dark_mode(self._dark_mode)
        if self._all_datasets_plot_view is not None:
            self._all_datasets_plot_view.set_dark_mode(self._dark_mode)

    def _ensure_subtabs(self, curves: Sequence[FitCurveDataset]) -> None:
        ids = [("dataset", curve.dataset_id) for curve in curves]
        labels = {curve.dataset_id: curve.display_name for curve in curves}
        if ids == self._dataset_tab_ids and labels == self._dataset_labels:
            return
        self._dataset_tab_ids = ids
        self._dataset_labels = labels
        self._dataset_plot_views = {}
        self._all_datasets_plot_view = None
        self._clear_subtabs()
        label_counts: dict[str, int] = {}
        for label in labels.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        for index, (curve, view_identity) in enumerate(zip(curves, ids, strict=True)):
            full_label = curve.display_name
            if label_counts[full_label] > 1:
                full_label = f"{full_label} ({curve.dataset_id})"
            compact_label = compact_dataset_label(full_label, max_chars=TAB_LABEL_MAX_CHARS)
            plot_view = self._create_plot_view(
                f"global_fit_results_plot_{curve.dataset_id}", view_identity
            )
            self._dataset_plot_views[view_identity] = plot_view
            self._subtab_stack.addWidget(plot_view)
            tab_index = self._subtabs.addTab(compact_label.display)
            self._subtabs.setTabToolTip(
                tab_index, f"{compact_label.full}\nDataset ID: {curve.dataset_id}"
            )
            self._subtabs.setTabData(
                tab_index,
                {
                    "view_identity": view_identity,
                    "alias": dataset_alias(index),
                    "full_label": compact_label.full,
                },
            )
        self._all_datasets_plot_view = self._create_plot_view(
            "global_fit_results_plot_all_datasets", ("aggregate", "all")
        )
        self._subtab_stack.addWidget(self._all_datasets_plot_view)
        self._subtabs.addTab("All Datasets")
        self._subtabs.setVisible(self._subtabs.count() > 0)
        self._subtabs.setCurrentIndex(0)
        self._subtab_stack.setCurrentIndex(0)

    def _create_plot_view(
        self, object_name: str, view_id: tuple[str, str]
    ) -> GridPlotView:
        plot_view = GridPlotView(
            color_resolver=self._color_resolver,
            show_generic_controls=False,
            parent=self,
        )
        plot_view.setObjectName(object_name)
        plot_view.set_dark_mode(self._dark_mode)
        plot_view.set_autorange_locked(self._view_autorange_locked)
        return plot_view

    def _install_plot_projection(
        self,
        curves: Sequence[FitCurveDataset],
        presentation: Mapping[str, DatasetPlotPresentation],
    ) -> bool:
        values = tuple(curves)
        ids = [item.dataset_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("Fit curve datasets must have unique dataset IDs.")
        self._curves = values
        self._dataset_presentation = MappingProxyType(dict(presentation))
        if values:
            self._ensure_subtabs(values)
            rendered = self._refresh_active_plot_view()
        else:
            # Terminal curve failure can leave the existing dataset tabs in
            # place while Summary and Apply remain valid. Clear every
            # physical view so an inactive tab cannot reveal its old curves.
            for plot_view in self._dataset_plot_views.values():
                plot_view.set_projection((), self._dataset_presentation)
            if self._all_datasets_plot_view is not None:
                self._all_datasets_plot_view.set_projection(
                    (), self._dataset_presentation
                )
            rendered = False
        self.copyAvailableChanged.emit(self.copy_eligible())
        return rendered

    def _active_plot_view_id(self) -> tuple[str, str] | None:
        index = self._subtabs.currentIndex()
        if index < 0:
            return None
        if index == len(self._dataset_tab_ids):
            return ("aggregate", "all")
        if 0 <= index < len(self._dataset_tab_ids):
            return self._dataset_tab_ids[index]
        return None

    def _refresh_active_plot_view(self) -> bool:
        view_id = self._active_plot_view_id()
        return bool(view_id is not None and self._refresh_plot_view(view_id))

    def _refresh_plot_view(self, view_id: tuple[str, str]) -> bool:
        if view_id[0] == "aggregate":
            plot_view = self._all_datasets_plot_view
            curves = self._curves
        else:
            plot_view = self._dataset_plot_views.get(view_id)
            curves = tuple(
                curve for curve in self._curves if curve.dataset_id == view_id[1]
            )
        if plot_view is None:
            return False
        rendered = plot_view.set_projection(curves, self._dataset_presentation)
        self.copyAvailableChanged.emit(self.copy_eligible())
        return bool(rendered)

    def _on_subtab_changed(self, index: int) -> None:
        if 0 <= int(index) < self._subtab_stack.count():
            self._subtab_stack.setCurrentIndex(int(index))
        self._refresh_active_plot_view()
        self.copyAvailableChanged.emit(self.copy_eligible())

    def _clear_subtabs(self) -> None:
        while self._subtabs.count():
            self._subtabs.removeTab(self._subtabs.count() - 1)
        while self._subtab_stack.count():
            widget = self._subtab_stack.widget(0)
            self._subtab_stack.removeWidget(widget)
            widget.deleteLater()
        self._subtabs.hide()
