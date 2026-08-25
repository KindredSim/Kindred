"""Physical Fit Results and Summary views.

The session owner retains the admitted Fit candidate.  This module only
renders call-local values and emits ordinary action requests.  In particular,
no candidate, result, report, projection, identity, or Apply policy is kept
by either view.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

from kindred.core.analysis.fit_statistics import FitMetric
from kindred.gui.display_name_policy import (
    DIAGNOSTIC_LABEL_MAX_CHARS,
    STATUS_ITEM_LABEL_MAX_CHARS,
    TAB_LABEL_MAX_CHARS,
    compact_dataset_label,
    compact_diagnostic_text,
    compact_text,
    dataset_alias,
)
from kindred.gui.fitting.launch import FitInitialApplyProposal
from kindred.gui.fitting.reporting import (
    fit_action_projection_bootstrap_lines,
    fit_action_projection_hash,
    fit_action_projection_profile_lines,
    fit_action_projection_report_text,
)
from kindred.gui.fitting.worker import FitCurveDataset, FitLiveUpdate
from kindred.gui.widgets.grid_plot_view import DatasetPlotPresentation, GridPlotView

__all__ = ["FitTrackerPanel", "ResultsSummaryDialog", "RunResultsTab"]

logger = logging.getLogger(__name__)

ColorResolver = Callable[[str], object]
ActionCallback = Callable[[], object]
SharedRefitCallback = Callable[[tuple[str, ...], str], object]
ApplyCallback = Callable[[tuple[str, ...], tuple[tuple[str, str], ...]], object]

_APPLY_SCOPE_LABELS = {
    "parameters": "Parameters",
    "initial_concentrations": "Initial Conditions",
    "observable_definitions": "Observable Definitions",
}


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


def _json_text(value: object, *, indent: int | None = None) -> str:
    try:
        return json.dumps(value, sort_keys=True, indent=indent, ensure_ascii=True)
    except (TypeError, ValueError):
        return str(value)


def _row_text(row: Mapping[str, object]) -> str:
    location = ""
    if row.get("dataset_id") is not None:
        location = f" [{row['dataset_id']}]"
    elif row.get("set_id") is not None:
        location = f" [{row['set_id']}]"
    change_percent = (
        f"{_display(row.get('percentage_change'))}%"
        if row.get("percentage_change") is not None
        else str(row.get("percentage_unavailable_reason") or "unavailable")
    )
    return (
        f"{row.get('display_name', row.get('name'))}{location}: "
        f"fitted={_display(row.get('fitted_value'))}; "
        f"nominal={_display(row.get('nominal_value'))}; "
        f"change={_display(row.get('absolute_change'))}; "
        f"change%={change_percent}; "
        f"bounds={_display(row.get('minimum'))}..{_display(row.get('maximum'))} "
        f"({row.get('bound_state')})"
    )


class ResultsSummaryDialog(QtWidgets.QDialog):
    """A physical summary popup rendered from one synchronous summary value."""

    statusMessage = Signal(str)
    copyHashRequested = Signal()
    copyJsonRequested = Signal()

    def __init__(
        self,
        identity: str,
        summary: Mapping[str, object] | None = None,
        *,
        copy_hash_requested: ActionCallback | None = None,
        copy_json_requested: ActionCallback | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("Results Summary")
        self.resize(560, 480)
        self._hash_action = copy_hash_requested
        self._copy_json_action = copy_json_requested

        layout = QtWidgets.QVBoxLayout(self)
        info_label = QtWidgets.QLabel(
            "Review the most recent fit run. Copy actions request fresh values "
            "from the Fit session owner."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        row = QtWidgets.QHBoxLayout()
        self._run_identity_label = QtWidgets.QLabel("Run: —")
        self._run_identity_label.setObjectName("global_fit_copy_hash_label")
        self._run_identity_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._copy_identity_button = QtWidgets.QPushButton("Copy")
        self._copy_identity_button.setObjectName("global_fit_copy_identity_button")
        self._copy_identity_button.clicked.connect(self._on_copy_identity_short)
        self._copy_identity_json_button = QtWidgets.QPushButton("Copy JSON")
        self._copy_identity_json_button.setObjectName(
            "global_fit_copy_identity_json_button"
        )
        self._copy_identity_json_button.clicked.connect(self._on_copy_identity_json)
        row.addWidget(self._run_identity_label, stretch=1)
        row.addWidget(self._copy_identity_button)
        row.addWidget(self._copy_identity_json_button)
        layout.addLayout(row)

        self._body = QtWidgets.QTextEdit(self)
        self._body.setObjectName("results_summary_body")
        self._body.setReadOnly(True)
        self._body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._body, stretch=1)
        self.refresh(identity, summary or {})

    def refresh(self, identity: str, summary: Mapping[str, object]) -> None:
        """Immediately render one call-local plain Summary mapping."""

        identity = str(identity or "").strip().lower()
        self._run_identity_label.setText(f"Run: {identity or '—'}")
        self._run_identity_label.setToolTip(identity)
        self._copy_identity_button.setEnabled(bool(identity))
        self._copy_identity_json_button.setEnabled(bool(identity))
        lines: list[str] = []
        if isinstance(summary, Mapping):
            result = summary.get("result", summary)
            if not isinstance(result, Mapping):
                result = summary
            for key, label in (
                ("status", "Completion"),
                ("optimizer_converged", "Optimizer convergence"),
                ("diagnostic_only", "Diagnostic only"),
                ("weighted_sse", "Weighted SSE"),
                ("weighted_mse", "Weighted MSE"),
            ):
                if key in result:
                    lines.append(f"{label}: {_display(result[key])}")
            for title, key in (
                ("Shared estimates", "shared_estimates"),
                ("Per-dataset estimates", "per_dataset_estimates"),
                ("Initial estimates", "initial_estimates"),
                ("Per-dataset metrics", "metrics_by_dataset"),
                ("Aggregates", "aggregates"),
            ):
                rows = summary.get(key, ())
                if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                    continue
                if not rows:
                    continue
                lines.append("")
                lines.append(f"{title}:")
                for row in rows:
                    if isinstance(row, Mapping):
                        lines.append(f"  {_row_text(row)}")
            for title, formatter in (
                ("Profile likelihood", fit_action_projection_profile_lines),
                ("Residual Bootstrap", fit_action_projection_bootstrap_lines),
            ):
                try:
                    formatted = formatter(summary)
                except (TypeError, ValueError, KeyError):
                    formatted = ()
                lines.append("")
                lines.append(f"{title}:")
                lines.extend(f"  {line}" for line in formatted)
            weighting = summary.get("weighting")
            if weighting:
                lines.append("")
                lines.append("Weighting and provenance:")
                lines.append(_json_text(weighting, indent=2))
            report_text = summary.get("report_text")
            if report_text:
                lines.append("")
                lines.append(str(report_text))
        self._body.setPlainText("\n".join(lines) or "No summary available.")

    def _on_copy_identity_short(self) -> None:
        if self._hash_action is not None:
            self._hash_action()
            return
        self.copyHashRequested.emit()

    def _on_copy_identity_json(self) -> None:
        if self._copy_json_action is not None:
            try:
                self._copy_json_action()
            except Exception:
                logger.exception("Results Summary Copy JSON request failed")
                self.statusMessage.emit("Failed to copy run identity JSON")
            return
        self.copyJsonRequested.emit()


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
        set_labels: Mapping[str, str] | None = None,
    ) -> None:
        cost = float(update.cost)
        delta_text = ""
        if self._previous_cost is not None and math.isfinite(cost):
            delta_text = f" (delta {cost - self._previous_cost:.2e})"
        if math.isfinite(cost):
            self._previous_cost = cost
        labels = dict(dataset_labels or {})
        initial_labels = dict(set_labels or {})
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
                initial_labels.get(set_id, f"Initial set {index + 1}"),
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

    def update_final(self, projection: Mapping[str, object]) -> None:
        """Render terminal values while retaining no projection."""

        result = projection.get("result", {})
        estimates = projection.get("estimates", {})
        if not isinstance(result, Mapping):
            result = {}
        if not isinstance(estimates, Mapping):
            estimates = {}
        lines: list[str] = []
        tooltips: list[str] = []
        for key in ("shared", "per_dataset", "initial"):
            rows = estimates.get(key, ())
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                for row in rows:
                    if isinstance(row, Mapping):
                        lines.append(_row_text(row))
                        if row.get("name"):
                            tooltips.append(f"Parameter ID: {row['name']}")
        try:
            lines.extend(f"{line}" for line in fit_action_projection_profile_lines(projection))
            lines.extend(f"{line}" for line in fit_action_projection_bootstrap_lines(projection))
        except (TypeError, ValueError, KeyError):
            pass
        self._set_live_text(
            f"Fit status {result.get('status', 'unavailable')} | Weighted objective SSQ "
            f"{_display(result.get('weighted_sse'))}",
            lines,
            tooltips,
        )

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
    initialSelectionChanged = Signal()
    copyHashRequested = Signal()
    sharedRefitRequested = Signal(tuple, str)
    applyRequested = Signal(tuple, tuple)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        color_resolver: ColorResolver | None = None,
        copy_hash_requested: ActionCallback | None = None,
        shared_refit_handler: SharedRefitCallback | None = None,
        apply_handler: ApplyCallback | None = None,
    ) -> None:
        super().__init__(parent)
        self._color_resolver = color_resolver
        self._hash_action = copy_hash_requested
        self._shared_refit_handler = shared_refit_handler
        self._apply_handler = apply_handler
        self._initial_apply_checkboxes: dict[tuple[str, str], QtWidgets.QCheckBox] = {}
        self._shared_refit_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        self._summary_dialog: ResultsSummaryDialog | None = None
        self._dark_mode = False
        self._view_autorange_locked = False
        self._curves: tuple[FitCurveDataset, ...] = ()
        self._dataset_presentation: Mapping[str, DatasetPlotPresentation] = (
            MappingProxyType({})
        )
        self._dataset_tab_ids: list[tuple[str, str]] = []
        self._dataset_labels: dict[str, str] = {}
        self._set_labels: dict[str, str] = {}
        self._dataset_plot_views: dict[tuple[str, str], GridPlotView] = {}
        self._all_datasets_plot_view: GridPlotView | None = None
        self._stale_plot_view_ids: set[tuple[str, str]] = set()
        self._render_failures: dict[tuple[str, str], str] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        self._tracker_panel = FitTrackerPanel(self)
        layout.addWidget(self._tracker_panel)

        identity_row = QtWidgets.QHBoxLayout()
        self._run_identity_label = QtWidgets.QLabel("Run: —", self)
        self._run_identity_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._copy_run_hash_button = QtWidgets.QPushButton("Copy hash", self)
        self._copy_run_hash_button.setEnabled(False)
        self._copy_run_hash_button.clicked.connect(self._request_copy_hash)
        identity_row.addWidget(self._run_identity_label, stretch=1)
        identity_row.addWidget(self._copy_run_hash_button)
        layout.addLayout(identity_row)

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
        self._terminal_report_label = QtWidgets.QLabel(self)
        self._terminal_report_label.setObjectName("global_fit_terminal_report_label")
        self._terminal_report_label.setWordWrap(True)
        self._terminal_report_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._terminal_report_label.hide()
        layout.addWidget(self._terminal_report_label)

        self._apply_group = QtWidgets.QGroupBox("Apply to Project", self)
        apply_layout = QtWidgets.QVBoxLayout(self._apply_group)
        self._apply_scope_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        self._apply_scope_labels: dict[str, QtWidgets.QLabel] = {}
        for scope, display_name in _APPLY_SCOPE_LABELS.items():
            row = QtWidgets.QHBoxLayout()
            checkbox = QtWidgets.QCheckBox(display_name, self._apply_group)
            checkbox.setObjectName(f"fit_apply_scope_{scope}")
            checkbox.setEnabled(False)
            self._apply_scope_checkboxes[scope] = checkbox
            label = QtWidgets.QLabel("—", self._apply_group)
            label.setWordWrap(True)
            self._apply_scope_labels[scope] = label
            row.addWidget(checkbox)
            row.addWidget(label, stretch=1)
            apply_layout.addLayout(row)
        self._apply_button = QtWidgets.QPushButton("Apply selected", self._apply_group)
        self._apply_button.setEnabled(False)
        self._apply_button.clicked.connect(self.request_apply)
        apply_layout.addWidget(self._apply_button)
        self._apply_group.hide()
        layout.addWidget(self._apply_group)

        self._initial_apply_group = QtWidgets.QGroupBox(
            "Apply Initial Conditions — selected cells", self
        )
        self._initial_apply_layout = QtWidgets.QVBoxLayout(self._initial_apply_group)
        self._initial_apply_group.hide()
        layout.addWidget(self._initial_apply_group)

        self._shared_refit_group = QtWidgets.QGroupBox(
            "Create shared values from this per-dataset fit", self
        )
        shared_layout = QtWidgets.QVBoxLayout(self._shared_refit_group)
        self._shared_refit_warning_label = QtWidgets.QLabel(self._shared_refit_group)
        self._shared_refit_warning_label.setWordWrap(True)
        shared_layout.addWidget(self._shared_refit_warning_label)
        explanation = QtWidgets.QLabel(
            "Select parameter names and a starting-estimate statistic. The Fit "
            "session owner adjudicates the request and owns the source Fit.",
            self._shared_refit_group,
        )
        explanation.setWordWrap(True)
        shared_layout.addWidget(explanation)
        self._shared_refit_parameter_layout = QtWidgets.QVBoxLayout()
        shared_layout.addLayout(self._shared_refit_parameter_layout)
        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(QtWidgets.QLabel("Starting estimate", self._shared_refit_group))
        self._shared_refit_seed_combo = QtWidgets.QComboBox(self._shared_refit_group)
        self._shared_refit_seed_combo.addItem("Arithmetic mean", "arithmetic_mean")
        self._shared_refit_seed_combo.addItem("Geometric mean", "geometric_mean")
        seed_row.addWidget(self._shared_refit_seed_combo, stretch=1)
        shared_layout.addLayout(seed_row)
        self._shared_refit_button = QtWidgets.QPushButton(
            "Configure and Run New Fit", self._shared_refit_group
        )
        self._shared_refit_button.clicked.connect(self._request_shared_refit)
        shared_layout.addWidget(self._shared_refit_button)
        self._shared_refit_group.hide()
        layout.addWidget(self._shared_refit_group)

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

    def set_identity_labels(
        self, *, dataset_labels: Mapping[str, str], set_labels: Mapping[str, str]
    ) -> None:
        self._dataset_labels = {str(k): str(v) for k, v in dataset_labels.items()}
        self._set_labels = {str(k): str(v) for k, v in set_labels.items()}

    def set_live_best(self, update: FitLiveUpdate) -> None:
        self._tracker_panel.update_from_best(
            update,
            dataset_labels=self._dataset_labels,
            set_labels=self._set_labels,
        )

    def set_live_curves(self, curves: Sequence[FitCurveDataset]) -> None:
        self._install_plot_projection(curves, {})

    def set_copy_hash(self, copy_hash: str) -> None:
        """Render a hash as physical label text; actions still go to the owner."""

        value = str(copy_hash or "").strip().lower()
        self._run_identity_label.setText(f"Run: {value or '—'}")
        self._run_identity_label.setToolTip(value)
        self._copy_run_hash_button.setEnabled(bool(value))

    def _request_copy_hash(self) -> None:
        if self._hash_action is not None:
            self._hash_action()
        else:
            self.copyHashRequested.emit()

    def set_apply_display(
        self,
        scope_labels: Mapping[str, str],
        enabled_scopes: Sequence[str] = (),
    ) -> None:
        """Render owner-provided Apply text and enabled state in widgets."""

        enabled = {str(scope) for scope in enabled_scopes}
        visible = bool(scope_labels) or bool(enabled)
        self._apply_group.setVisible(visible)
        for scope, checkbox in self._apply_scope_checkboxes.items():
            checkbox.setEnabled(scope in enabled)
            checkbox.setChecked(scope in enabled)
            label = self._apply_scope_labels[scope]
            text = str(scope_labels.get(scope, "Unavailable"))
            label.setText(text)
            label.setToolTip(text)
        self._apply_button.setEnabled(bool(enabled))

    def request_apply(self) -> None:
        scopes = tuple(
            scope
            for scope, checkbox in self._apply_scope_checkboxes.items()
            if checkbox.isChecked() and checkbox.isEnabled()
        )
        keys = self.selected_initial_keys()
        if self._apply_handler is not None:
            self._apply_handler(scopes, keys)
        else:
            self.applyRequested.emit(scopes, keys)

    def set_initial_apply_proposal(self, proposal: FitInitialApplyProposal | None) -> None:
        while self._initial_apply_layout.count():
            item = self._initial_apply_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._initial_apply_checkboxes = {}
        if proposal is None or not proposal.cells:
            self._initial_apply_group.hide()
            self.initialSelectionChanged.emit()
            return
        for cell in proposal.cells:
            change = cell.fitted_value - cell.nominal_value
            change_percent = (
                f"{(change / abs(cell.nominal_value)) * 100.0:.6g}%"
                if cell.nominal_value != 0.0
                else "unavailable (nominal value is zero)"
            )
            set_label = self._set_labels.get(cell.set_id, cell.set_id)
            datasets = ", ".join(
                self._dataset_labels.get(dataset_id, dataset_id)
                for dataset_id in cell.dataset_ids
            ) or "no dataset label"
            checkbox = QtWidgets.QCheckBox(
                f"{set_label} / {cell.species} [{cell.unit}] — datasets {datasets}; "
                f"captured={cell.captured_value:.6g}; nominal={cell.nominal_value:.6g}; "
                f"fitted={cell.fitted_value:.6g}; change={change:.6g}; "
                f"change%={change_percent}; bounds=[{cell.minimum:.6g}, "
                f"{cell.maximum:.6g}]",
                self._initial_apply_group,
            )
            checkbox.setToolTip(
                f"Set ID: {cell.set_id}\n"
                f"Dataset IDs: {', '.join(cell.dataset_ids) or 'none'}"
            )
            checkbox.setChecked(True)
            checkbox.toggled.connect(lambda _checked: self.initialSelectionChanged.emit())
            self._initial_apply_checkboxes[cell.key] = checkbox
            self._initial_apply_layout.addWidget(checkbox)
        self._initial_apply_group.show()
        self.initialSelectionChanged.emit()

    def selected_initial_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            key for key, checkbox in self._initial_apply_checkboxes.items()
            if checkbox.isChecked()
        )

    def set_shared_refit_projection(
        self,
        values: Sequence[Mapping[str, object]],
        *,
        warning: str = "",
    ) -> None:
        """Render transient per-dataset rows as physical selectable widgets."""

        while self._shared_refit_parameter_layout.count():
            item = self._shared_refit_parameter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._shared_refit_checkboxes = {}
        for row in values:
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            display_name = str(row.get("display_name") or name)
            checkbox = QtWidgets.QCheckBox(display_name, self._shared_refit_group)
            checkbox.setObjectName(f"fit_shared_refit_parameter_{name}")
            checkbox.setChecked(True)
            checkbox.setToolTip(
                f"Parameter ID: {name}\n"
                f"Per-dataset values: {_display(row.get('values', ())) }"
            )
            self._shared_refit_checkboxes[name] = checkbox
            self._shared_refit_parameter_layout.addWidget(checkbox)
        self._shared_refit_warning_label.setText(str(warning))
        self._shared_refit_group.setVisible(bool(self._shared_refit_checkboxes))
        self._shared_refit_button.setEnabled(bool(self._shared_refit_checkboxes))

    def clear_shared_refit_projection(self) -> None:
        self.set_shared_refit_projection(())

    def _request_shared_refit(self) -> None:
        selected = tuple(
            name for name, checkbox in self._shared_refit_checkboxes.items()
            if checkbox.isChecked() and checkbox.isEnabled()
        )
        statistic = str(self._shared_refit_seed_combo.currentData() or "arithmetic_mean")
        if not selected:
            self.statusMessage.emit("Select an eligible parameter for Shared refit.")
            return
        if self._shared_refit_handler is not None:
            self._shared_refit_handler(selected, statistic)
        else:
            self.sharedRefitRequested.emit(selected, statistic)

    def set_terminal_presentation(
        self,
        projection: Mapping[str, object] | None,
        *,
        copy_hash: str | None = None,
        report_text: str | None = None,
        report_failure: str | None = None,
        curves: Sequence[FitCurveDataset] = (),
        curve_failure: str | None = None,
        plot_presentation: Mapping[str, DatasetPlotPresentation] | None = None,
        shared_refit_rows: Sequence[Mapping[str, object]] = (),
        shared_refit_warning: str = "",
    ) -> bool:
        """Consume one call-local terminal presentation and render physical state."""

        if copy_hash is None and isinstance(projection, Mapping):
            copy_hash = fit_action_projection_hash(projection)
        if report_text is None and isinstance(projection, Mapping):
            try:
                report_text = fit_action_projection_report_text(projection)
            except (TypeError, ValueError, KeyError):
                report_text = None
        if copy_hash is not None:
            self.set_copy_hash(copy_hash)
        self._terminal_failure_label.hide()
        self._terminal_report_label.hide()
        self._clear_render_failures()
        if projection is None:
            self._tracker_panel.clear()
            self.clear_shared_refit_projection()
        else:
            self._tracker_panel.update_final(projection)
            rows = shared_refit_rows
            if not rows:
                estimates = projection.get("estimates", {})
                if isinstance(estimates, Mapping):
                    rows = tuple(
                        row for row in estimates.get("per_dataset", ())
                        if isinstance(row, Mapping)
                    )
            self.set_shared_refit_projection(rows, warning=shared_refit_warning)
        self.set_terminal_report(report_text, failure=report_failure)
        self._install_plot_projection(curves, plot_presentation or {})
        if curve_failure is not None:
            self._record_render_failure(("aggregate", "all"), curve_failure)
        return self.copy_eligible()

    def set_terminal_report(
        self, report_text: str | None, *, failure: str | None = None
    ) -> None:
        if failure:
            self._terminal_report_label.setText(
                f"No terminal report available: {failure}"
            )
            self._terminal_report_label.setToolTip(str(failure))
        elif report_text:
            self._terminal_report_label.setText(str(report_text))
            self._terminal_report_label.setToolTip("")
        else:
            self._terminal_report_label.clear()
            self._terminal_report_label.setToolTip("")
            self._terminal_report_label.hide()
            return
        self._terminal_report_label.show()

    def render_results_summary(
        self,
        summary: Mapping[str, object],
        *,
        identity: str | None = None,
        copy_hash_requested: ActionCallback | None = None,
        copy_json_requested: ActionCallback | None = None,
    ) -> None:
        """Render a synchronous owner-provided Summary without retaining it."""

        if identity is None:
            identity = str(summary.get("copy_hash") or "")
        if self._summary_dialog is not None and self._summary_dialog.isVisible():
            self._summary_dialog.refresh(str(identity), summary)
            self._summary_dialog.raise_()
            self._summary_dialog.activateWindow()
            return
        dialog = ResultsSummaryDialog(
            str(identity),
            summary,
            copy_hash_requested=copy_hash_requested or self._hash_action,
            copy_json_requested=copy_json_requested,
            parent=self,
        )
        dialog.statusMessage.connect(self.statusMessage)
        dialog.destroyed.connect(lambda: setattr(self, "_summary_dialog", None))
        self._summary_dialog = dialog
        dialog.show()

    def clear_candidate(self) -> None:
        """Clear only physical result widgets; the owner controls custody."""

        self.set_copy_hash("")
        self._tracker_panel.clear()
        self.set_terminal_report(None)
        self._terminal_failure_label.clear()
        self._terminal_failure_label.hide()
        self.set_initial_apply_proposal(None)
        self.clear_shared_refit_projection()
        self._clear_render_failures()
        self._install_plot_projection((), {})
        self.set_apply_display({}, ())
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
            _json_text(payload() if callable(payload) else str(failure), indent=2)
        )
        self._terminal_failure_label.show()

    def set_terminal_unavailable(
        self, cause: object, *, older_retained_last_fit: bool = False
    ) -> None:
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
        if older_retained_last_fit:
            text += "\nOlder retained Last Fit state is shown separately."
        self._terminal_failure_label.setText(text)
        self._terminal_failure_label.show()

    def copy_eligible(self) -> bool:
        view = self._all_datasets_plot_view
        return bool(
            self._curves
            and view is not None
            and (
                ("aggregate", "all") in self._stale_plot_view_ids
                or view.copy_eligible()
            )
        )

    def copy_all_to_clipboard(self) -> None:
        view = self._all_datasets_plot_view
        if view is None or not self._curves:
            raise ValueError("No rendered Fit curves are available to copy.")
        if ("aggregate", "all") in self._stale_plot_view_ids:
            if not self._refresh_plot_view(("aggregate", "all")):
                return
        if not view.copy_eligible():
            self.copyAvailableChanged.emit(False)
            return
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
        plot_view.renderFailed.connect(
            lambda message, identity=view_id: self._record_render_failure(identity, message)
        )
        plot_view.renderStateChanged.connect(
            lambda _rendered: self.copyAvailableChanged.emit(self.copy_eligible())
        )
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
        self._stale_plot_view_ids.update(self._dataset_plot_views)
        if self._all_datasets_plot_view is not None:
            self._stale_plot_view_ids.add(("aggregate", "all"))
        rendered = self._refresh_active_plot_view()
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
        if view_id not in self._stale_plot_view_ids:
            plot_view = (
                self._all_datasets_plot_view
                if view_id[0] == "aggregate"
                else self._dataset_plot_views.get(view_id)
            )
            return bool(plot_view is not None and plot_view.copy_eligible())
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
        try:
            rendered = plot_view.set_projection(curves, self._dataset_presentation)
        except Exception as exc:
            plot_view.reject_projection(exc)
            rendered = False
        self._stale_plot_view_ids.discard(view_id)
        if not curves or rendered:
            self._clear_render_failure(view_id)
        elif view_id not in self._render_failures:
            self._record_render_failure(view_id, "Fit plot rendering failed.")
        self.copyAvailableChanged.emit(self.copy_eligible())
        return bool(rendered)

    def _on_subtab_changed(self, index: int) -> None:
        if 0 <= int(index) < self._subtab_stack.count():
            self._subtab_stack.setCurrentIndex(int(index))
        self._refresh_active_plot_view()
        self.copyAvailableChanged.emit(self.copy_eligible())

    def _record_render_failure(self, view_id: tuple[str, str], message: object) -> None:
        diagnostic = str(message or "Fit plot rendering failed.")
        self._render_failures[view_id] = diagnostic
        compact = compact_diagnostic_text(diagnostic, max_chars=DIAGNOSTIC_LABEL_MAX_CHARS)
        self._render_diagnostic_label.setText(
            "Fitted values remain available; plot unavailable: "
            f"{compact.display or 'Fit plot rendering failed.'}"
        )
        self._render_diagnostic_label.setToolTip(compact.full)
        self._render_diagnostic_label.show()
        self.statusMessage.emit(diagnostic)

    def _clear_render_failure(self, view_id: tuple[str, str]) -> None:
        self._render_failures.pop(view_id, None)
        self._refresh_render_diagnostic()

    def _clear_render_failures(self) -> None:
        self._render_failures.clear()
        self._refresh_render_diagnostic()

    def _refresh_render_diagnostic(self) -> None:
        if not self._render_failures:
            self._render_diagnostic_label.clear()
            self._render_diagnostic_label.setToolTip("")
            self._render_diagnostic_label.hide()
            return
        diagnostic = next(reversed(self._render_failures.values()))
        compact = compact_diagnostic_text(diagnostic, max_chars=DIAGNOSTIC_LABEL_MAX_CHARS)
        self._render_diagnostic_label.setText(
            "Fitted values remain available; plot unavailable: "
            f"{compact.display or 'Fit plot rendering failed.'}"
        )
        self._render_diagnostic_label.setToolTip(compact.full)
        self._render_diagnostic_label.show()

    def _clear_subtabs(self) -> None:
        self._stale_plot_view_ids.clear()
        while self._subtabs.count():
            self._subtabs.removeTab(self._subtabs.count() - 1)
        while self._subtab_stack.count():
            widget = self._subtab_stack.widget(0)
            self._subtab_stack.removeWidget(widget)
            widget.deleteLater()
        self._subtabs.hide()
