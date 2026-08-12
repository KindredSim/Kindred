"""Historical Results-tab presentation bound to the current typed Fit values."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

from kindred.core.analysis.fit_statistics import FitMetric
from kindred.core.analysis.global_fitting import SharedRefitPromotion
from kindred.gui.display_name_policy import (
    DIAGNOSTIC_LABEL_MAX_CHARS,
    PLOT_TITLE_LABEL_MAX_CHARS,
    STATUS_ITEM_LABEL_MAX_CHARS,
    TAB_LABEL_MAX_CHARS,
    compact_dataset_label,
    compact_diagnostic_text,
    compact_plot_title_label,
    compact_text,
    dataset_alias,
)
from kindred.gui.fitting.contracts import FitRunIdentity, FitWorkerFailure
from kindred.gui.fitting.launch import (
    FitInitialApplyProposal,
    FitSharedRefitIntent,
)
from kindred.gui.fitting.reporting import (
    FitEstimateReportRow,
    FitTerminalReportError,
    FitTerminalReport,
    fit_bound_state,
    fit_metric_text,
    fit_terminal_extension_lines,
    fit_terminal_report_payload,
)
from kindred.gui.fitting.worker import (
    FitCandidate,
    FitCurveDataset,
    FitLiveUpdate,
    project_fit_candidate_curves,
)
from kindred.gui.widgets.grid_plot_view import (
    DatasetPlotPresentation,
    GridPlotView,
)

__all__ = ["FitTrackerPanel", "ResultsSummaryDialog", "RunResultsTab"]

logger = logging.getLogger(__name__)

ColorResolver = Callable[[str], object]
_APPLY_SCOPE_LABELS = {
    "parameters": "Parameters",
    "initial_concentrations": "Initial Conditions",
    "observable_definitions": "Observable Definitions",
}


@dataclass(frozen=True, slots=True)
class _FitResultViewIdentity:
    """Tagged Fit view identity kept outside opaque dataset-ID strings."""

    kind: Literal["dataset", "aggregate"]
    dataset_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"dataset", "aggregate"}:
            raise ValueError("Fit result view kind must be dataset or aggregate.")
        if self.kind == "aggregate":
            if self.dataset_id is not None:
                raise ValueError("Aggregate Fit view cannot carry a dataset ID.")
            return
        dataset_id = str(self.dataset_id or "").strip()
        if not dataset_id:
            raise ValueError("Dataset Fit view requires an opaque dataset ID.")
        object.__setattr__(self, "dataset_id", dataset_id)


_FIT_AGGREGATE_VIEW = _FitResultViewIdentity("aggregate")


def _get_clipboard():
    """Return the application clipboard when a Qt application is available."""

    try:
        application = QtWidgets.QApplication.instance()
        return application.clipboard() if application is not None else None
    except Exception:
        return None


def _format_metric(value: object) -> str:
    if isinstance(value, FitMetric):
        return fit_metric_text(value, precision=2)
    try:
        number = float(value)
        return f"{number:.2e}" if math.isfinite(number) else "unavailable"
    except Exception:
        return "unavailable"


def _plain_json_data(value: object) -> object:
    """Normalize one export payload into strict JSON-compatible plain data."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Fit run export contains a non-finite number.")
        return value
    if isinstance(value, np.generic):
        return _plain_json_data(value.item())
    if isinstance(value, np.ndarray):
        return _plain_json_data(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _plain_json_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_plain_json_data(item) for item in value]
    raise TypeError(
        "Fit run export contains unsupported value "
        f"{type(value).__name__}."
    )


def _export_array(value: object) -> list[object]:
    """Materialize one result array only for an explicit verbose export."""

    return np.asarray(value).reshape(-1).tolist()


def _fit_run_identity_export_payload(
    candidate: FitCandidate,
    report: FitTerminalReport,
) -> dict[str, object]:
    """Build the complete Copy Run JSON view without retaining it."""

    if not isinstance(candidate, FitCandidate):
        raise TypeError("candidate must be FitCandidate.")
    if not isinstance(report, FitTerminalReport):
        raise TypeError("Fit export requires FitTerminalReport.")
    identity_payload = candidate.run_identity.payload
    if not isinstance(identity_payload, Mapping):
        raise TypeError("Fit run identity payload must be a JSON object.")
    curves = project_fit_candidate_curves(candidate)
    export = dict(identity_payload)
    export["schema_version"] = 2
    export["export_kind"] = "verbose_fit_run"
    export["fit_report"] = fit_terminal_report_payload(report)
    export["curves"] = [
        {
            "dataset_id": curve.dataset_id,
            "display_name": curve.display_name,
            "x_name": curve.x_name,
            "y_semantics": [
                (name, semantic.value) for name, semantic in curve.y_semantics
            ],
            "model": {
                name: {
                    "x": _export_array(dict(curve.model_x)[name]),
                    "y": _export_array(dict(curve.model_y)[name]),
                    "segment_ids": _export_array(dict(curve.model_segment_ids)[name]),
                    "segment_keys": _export_array(dict(curve.model_segment_keys)[name]),
                    "discontinuity_ids": _export_array(
                        dict(curve.model_discontinuity_ids)[name]
                    ),
                }
                for name, _values in curve.model_y
            },
            "observed": {
                name: {
                    "x": _export_array(dict(curve.observed_x)[name]),
                    "y": _export_array(dict(curve.observed_y)[name]),
                    "source_rows": _export_array(dict(curve.observed_source_rows)[name]),
                    "raw_observation_id": dict(curve.observed_raw_ids)[name],
                    "raw_column": dict(curve.observed_raw_columns)[name],
                }
                for name, _values in curve.observed_y
            },
            "objective": {
                name: {
                    "x": _export_array(dict(curve.objective_model_x)[name]),
                    "y": _export_array(dict(curve.objective_model_y)[name]),
                    "segment_ids": _export_array(
                        dict(curve.objective_model_segment_ids)[name]
                    ),
                    "segment_keys": _export_array(
                        dict(curve.objective_model_segment_keys)[name]
                    ),
                    "discontinuity_ids": _export_array(
                        dict(curve.objective_model_discontinuity_ids)[name]
                    ),
                    "selection_provenance": _export_array(
                        dict(curve.objective_selection_provenance)[name]
                    ),
                    "observation_ids": _export_array(
                        dict(curve.objective_observation_ids)[name]
                    ),
                    "raw_columns": _export_array(dict(curve.objective_raw_columns)[name]),
                    "source_rows": _export_array(dict(curve.objective_source_rows)[name]),
                }
                for name, _values in curve.objective_model_y
            },
        }
        for curve in sorted(curves, key=lambda item: item.dataset_id)
    ]
    return export


def _fit_run_identity_export_json(
    candidate: FitCandidate,
    report: FitTerminalReport,
) -> str:
    """Serialize the complete Fit export at its single final plain-data boundary."""

    return json.dumps(
        _plain_json_data(_fit_run_identity_export_payload(candidate, report)),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )


class ResultsSummaryDialog(QtWidgets.QDialog):
    """Non-modal popup displaying the current Fit run identity and statistics."""

    statusMessage = Signal(str)

    def __init__(
        self,
        identity: FitRunIdentity,
        candidate: FitCandidate | None = None,
        report: FitTerminalReport | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if not isinstance(identity, FitRunIdentity):
            raise TypeError("Results summary requires a FitRunIdentity.")
        if candidate is not None and not isinstance(candidate, FitCandidate):
            raise TypeError("candidate must be FitCandidate.")
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("Results Summary")
        self.resize(480, 360)

        self._identity = identity
        self._candidate = candidate
        self._report = report

        layout = QtWidgets.QVBoxLayout(self)

        info_label = QtWidgets.QLabel(
            "Review and copy the most recent fit run identity."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(info_label)

        row = QtWidgets.QHBoxLayout()
        self._run_identity_label = QtWidgets.QLabel(f"Run: {identity.short_hash}")
        self._run_identity_label.setObjectName("global_fit_run_identity_label")
        self._run_identity_label.setStyleSheet("font-size: 11px;")

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

        self._body_scroll = QtWidgets.QScrollArea(self)
        self._body_scroll.setObjectName("results_summary_body_scroll")
        self._body_scroll.setWidgetResizable(True)
        self._body_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._body_scroll.setMinimumHeight(0)
        self._body_content = QtWidgets.QWidget(self._body_scroll)
        body_layout = QtWidgets.QVBoxLayout(self._body_content)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)
        self._body_scroll.setWidget(self._body_content)
        layout.addWidget(self._body_scroll, stretch=1)

        stats_group = QtWidgets.QGroupBox("Fit Statistics")
        form = QtWidgets.QFormLayout(stats_group)
        self._stats_labels: dict[str, QtWidgets.QLabel] = {}
        for label_text in (
            "Datasets",
            "Series",
            "Points",
            "Parameters",
            "DF",
            "Completion",
            "Joint optimizer convergence",
            "Joint optimizer diagnostic",
            "Weighted SSE",
            "Weighted MSE",
            "SciPy nfev",
            "Terminal replay count",
            "Objective calls",
            "Message",
            "Weighted objective SSQ",
        ):
            field_label = QtWidgets.QLabel(f"{label_text}:")
            field_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            value_label = QtWidgets.QLabel("\u2014")
            value_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            form.addRow(field_label, value_label)
            self._stats_labels[label_text] = value_label
        body_layout.addWidget(stats_group)

        params_group = QtWidgets.QGroupBox("Fitted Parameters")
        params_group.setObjectName("fitted_params_group")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        params_layout.setContentsMargins(0, 0, 0, 0)
        self._params_content_label = QtWidgets.QLabel()
        self._params_content_label.setObjectName("fitted_params_content_label")
        self._params_content_label.setWordWrap(True)
        self._params_content_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._params_content_label.setStyleSheet("font-size: 11px;")
        self._params_content_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        params_layout.addWidget(self._params_content_label)
        body_layout.addWidget(params_group)

        metrics_group = QtWidgets.QGroupBox("Per-dataset Metrics")
        metrics_layout = QtWidgets.QVBoxLayout(metrics_group)
        self._dataset_metrics_label = QtWidgets.QLabel(metrics_group)
        self._dataset_metrics_label.setWordWrap(True)
        self._dataset_metrics_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._dataset_metrics_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        metrics_layout.addWidget(self._dataset_metrics_label)
        body_layout.addWidget(metrics_group)

        estimates_group = QtWidgets.QGroupBox("SI-ready Estimate Tables")
        estimates_layout = QtWidgets.QVBoxLayout(estimates_group)
        self._estimate_tables_label = QtWidgets.QLabel(estimates_group)
        self._estimate_tables_label.setWordWrap(True)
        self._estimate_tables_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._estimate_tables_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        estimates_layout.addWidget(self._estimate_tables_label)
        body_layout.addWidget(estimates_group)

        aggregates_group = QtWidgets.QGroupBox("Per-dataset Aggregates")
        aggregates_layout = QtWidgets.QVBoxLayout(aggregates_group)
        self._aggregates_label = QtWidgets.QLabel(aggregates_group)
        self._aggregates_label.setWordWrap(True)
        self._aggregates_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._aggregates_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        aggregates_layout.addWidget(self._aggregates_label)
        body_layout.addWidget(aggregates_group)

        validity_group = QtWidgets.QGroupBox("Validity and Terminal Methods")
        validity_layout = QtWidgets.QVBoxLayout(validity_group)
        self._validity_label = QtWidgets.QLabel(validity_group)
        self._validity_label.setWordWrap(True)
        self._validity_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._validity_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        validity_layout.addWidget(self._validity_label)
        body_layout.addWidget(validity_group)
        body_layout.addStretch(1)

        self.refresh(
            identity,
            report=report,
        )

    def _update_fitted_params(self, report: FitTerminalReport | None) -> None:
        if report is None:
            self._params_content_label.setText("No fitted parameters available.")
            self._params_content_label.setToolTip("")
            self._params_content_label.setStyleSheet("font-size: 11px; color: #888;")
            return
        shared = tuple(
            (row.display_name, row.fitted_value)
            for row in report.shared_estimates
        )
        dataset_values: dict[str, list[FitEstimateReportRow]] = {}
        for row in report.per_dataset_estimates:
            if row.dataset_id is not None:
                dataset_values.setdefault(row.dataset_id, []).append(row)
        initial_values: dict[str, list[FitEstimateReportRow]] = {}
        for row in report.initial_estimates:
            if row.set_id is not None:
                initial_values.setdefault(row.set_id, []).append(row)
        if not shared and not dataset_values and not initial_values:
            self._params_content_label.setText("No fitted parameters available.")
            self._params_content_label.setToolTip("")
            self._params_content_label.setStyleSheet("font-size: 11px; color: #888;")
            return

        self._params_content_label.setStyleSheet("font-size: 11px;")
        lines = [
            f"{name} = {value:.6g}"
            for name, value in sorted(shared, key=lambda item: str(item[0]))
        ]
        tooltip_lines: list[str] = []
        for dataset_id, rows in sorted(dataset_values.items()):
            if not rows:
                continue
            if lines:
                lines.append("")
            lines.append(f"{rows[0].dataset_display_name or dataset_id}:")
            lines.extend(
                f"  {row.display_name} = {row.fitted_value:.6g} — "
                "Fit-local; not applied to Run"
                for row in sorted(rows, key=lambda item: item.display_name)
            )
            tooltip_lines.append(
                f"{rows[0].dataset_display_name or dataset_id}: Dataset ID {dataset_id}"
            )
        for set_id, rows in sorted(initial_values.items()):
            if not rows:
                continue
            if lines:
                lines.append("")
            lines.append(
                f"{rows[0].set_display_name or set_id} initial concentrations:"
            )
            lines.extend(
                f"  {row.display_name} = {row.fitted_value:.6g}"
                for row in sorted(rows, key=lambda item: item.display_name)
            )
            tooltip_lines.append(f"{rows[0].set_display_name or set_id}: Set ID {set_id}")
        self._params_content_label.setText("\n".join(lines))
        self._params_content_label.setToolTip("\n".join(tooltip_lines))

    def _update_dataset_metrics(self, report: FitTerminalReport | None) -> None:
        if report is None or not report.metrics_by_dataset:
            self._dataset_metrics_label.setText("No per-dataset metrics available.")
            self._dataset_metrics_label.setToolTip("")
            return
        lines: list[str] = []
        tooltip_lines: list[str] = []
        for row in report.metrics_by_dataset:
            lines.append(
                f"{row.dataset_display_name}: weighted SSE = "
                f"{fit_metric_text(row.weighted_sse)}; weighted MSE = "
                f"{fit_metric_text(row.weighted_mse)}; weighted RMSE = "
                f"{fit_metric_text(row.weighted_rmse)}; weighted MAE = "
                f"{fit_metric_text(row.weighted_mae)}; effective weight = "
                f"{row.effective_weight:.6g}; points = {row.n_points}"
            )
            lines.extend(
                f"  {compact_text(series, max_chars=STATUS_ITEM_LABEL_MAX_CHARS).display}: "
                f"R² = {fit_metric_text(value)}"
                for series, value in row.r_squared_by_series
            )
            tooltip_lines.append(
                f"{row.dataset_display_name}: Dataset ID {row.dataset_id}"
            )
        self._dataset_metrics_label.setText("\n".join(lines))
        self._dataset_metrics_label.setToolTip("\n".join(tooltip_lines))

    @staticmethod
    def _estimate_line(row: FitEstimateReportRow) -> str:
        identity = row.dataset_id or row.set_id or "joint"
        percentage = (
            f"{row.percentage_change:.6g}%"
            if row.percentage_change is not None
            else f"unavailable ({row.percentage_unavailable_reason})"
        )
        minimum = "−∞" if not math.isfinite(row.minimum) else f"{row.minimum:.6g}"
        maximum = "∞" if not math.isfinite(row.maximum) else f"{row.maximum:.6g}"
        return (
            f"{row.scope} | {identity} | {row.display_name} [{row.unit}] | "
            f"captured={row.captured_value:.6g}; nominal={row.nominal_value:.6g}; "
            f"fitted={row.fitted_value:.6g}; change={row.absolute_change:.6g}; "
            f"change%={percentage}; bounds=[{minimum}, {maximum}]; "
            f"bound state={row.bound_state}; valid={str(row.valid).lower()} "
            f"({row.validity_diagnostic}); captured source={row.captured_value_source}; "
            f"nominal source={row.nominal_value_source}"
        )

    def _update_terminal_report(self, report: FitTerminalReport | None) -> None:
        if report is None:
            self._estimate_tables_label.setText("No terminal estimate report available.")
            self._aggregates_label.setText("No per-dataset aggregates available.")
            self._validity_label.setText("No terminal validity report available.")
            self._validity_label.setToolTip("")
            return
        estimate_rows = (
            *report.shared_estimates,
            *report.per_dataset_estimates,
            *report.initial_estimates,
        )
        self._estimate_tables_label.setText(
            "\n".join(self._estimate_line(row) for row in estimate_rows)
            if estimate_rows
            else "No fitted estimates available."
        )
        aggregate_lines: list[str] = []
        for row in report.aggregates:
            sample_sd = "—" if row.sample_sd is None else f"{row.sample_sd:.6g}"
            standard_error = (
                "—" if row.standard_error is None else f"{row.standard_error:.6g}"
            )
            geometric_mean = (
                "—" if row.geometric_mean is None else f"{row.geometric_mean:.6g}"
            )
            geometric_dispersion = (
                "—"
                if row.geometric_dispersion is None
                else f"{row.geometric_dispersion:.6g}"
            )
            geometric_note = (
                f" ({row.geometric_unavailable_reason})"
                if row.geometric_unavailable_reason
                else ""
            )
            aggregate_lines.append(
                f"{row.display_name} [{row.unit}]: n={row.n}; mean={row.arithmetic_mean:.6g}; "
                f"sample SD={sample_sd}; SE={standard_error}; median={row.median:.6g}; "
                f"IQR={row.iqr:.6g}; min={row.minimum:.6g}; max={row.maximum:.6g}; "
                f"geometric mean={geometric_mean}; geometric dispersion={geometric_dispersion}"
                f"{geometric_note}"
            )
        self._aggregates_label.setText(
            "\n".join(aggregate_lines)
            if aggregate_lines
            else "No Per-dataset fitted scalar aggregates available."
        )
        method_lines = [
            f"Overall Fit status: {report.status}",
            "Joint optimizer convergence: "
            + ("converged" if report.joint_optimizer_converged else "did not converge"),
            f"Joint optimizer diagnostic: {report.joint_optimizer_diagnostic}",
            report.publication.message,
        ]
        method_lines.extend(report.validity_diagnostics or ("No validity warnings.",))
        if report.shared_refit_provenance:
            method_lines.append(
                "Shared refit provenance: "
                + json.dumps(
                    _plain_json_data(report.shared_refit_provenance),
                    sort_keys=True,
                )
            )
        for item in report.terminal_extensions:
            method_lines.extend(fit_terminal_extension_lines(item))
        self._validity_label.setText("\n".join(method_lines))
        self._validity_label.setToolTip(
            json.dumps(
                _plain_json_data(report.structured_diagnostics),
                sort_keys=True,
                indent=2,
            )
            if report.structured_diagnostics
            else ""
        )

    def _update_stats(self, report: FitTerminalReport | None) -> None:
        values: Mapping[str, object] = {}
        if report is not None:
            values = {
                "Datasets": report.dataset_count,
                "Series": report.series_count,
                "Points": report.point_count,
                "Parameters": report.parameter_count,
                "DF": report.degrees_of_freedom,
                "Completion": report.status,
                "Joint optimizer convergence": (
                    "converged"
                    if report.joint_optimizer_converged
                    else "did not converge"
                ),
                "Joint optimizer diagnostic": report.joint_optimizer_diagnostic,
                "Weighted SSE": fit_metric_text(report.weighted_sse),
                "Weighted MSE": fit_metric_text(report.weighted_mse),
                "SciPy nfev": report.scipy_nfev,
                "Terminal replay count": report.terminal_replay_count,
                "Objective calls": report.objective_call_count,
                "Message": report.message,
                "Weighted objective SSQ": fit_metric_text(report.weighted_sse),
            }
        for key, label in self._stats_labels.items():
            value = values.get(key)
            if value is None:
                label.setText("\u2014")
                label.setToolTip("")
            elif isinstance(value, float):
                label.setText(f"{value:.6g}")
                label.setToolTip("")
            else:
                compact = compact_diagnostic_text(
                    value,
                    max_chars=DIAGNOSTIC_LABEL_MAX_CHARS,
                )
                label.setText(compact.display)
                label.setToolTip(compact.tooltip)

    def refresh(
        self,
        identity: FitRunIdentity,
        *,
        candidate: FitCandidate | None = None,
        report: FitTerminalReport | None = None,
    ) -> None:
        if not isinstance(identity, FitRunIdentity):
            raise TypeError("Results summary requires a FitRunIdentity.")
        self._identity = identity
        if candidate is not None:
            if not isinstance(candidate, FitCandidate):
                raise TypeError("candidate must be FitCandidate.")
            self._candidate = candidate
        self._report = report
        self._run_identity_label.setText(f"Run: {identity.short_hash}")
        if report is not None and not isinstance(report, FitTerminalReport):
            raise TypeError("Results summary report must be FitTerminalReport.")
        self._update_stats(report)
        self._update_fitted_params(report)
        self._update_dataset_metrics(report)
        self._update_terminal_report(report)

    def _on_copy_identity_short(self) -> None:
        clipboard = _get_clipboard()
        if clipboard is None:
            self.statusMessage.emit("Clipboard unavailable")
            return
        try:
            clipboard.setText(self._identity.short_hash)
        except Exception:
            self.statusMessage.emit("Failed to copy run identity")
            return
        self.statusMessage.emit("Copied run identity hash")

    def _on_copy_identity_json(self) -> None:
        clipboard = _get_clipboard()
        if clipboard is None:
            self.statusMessage.emit("Clipboard unavailable")
            return
        try:
            if self._candidate is None:
                clipboard.setText(self._identity.readable_json())
            else:
                if self._report is None:
                    raise ValueError("Fit terminal report is unavailable.")
                clipboard.setText(
                    _fit_run_identity_export_json(self._candidate, self._report)
                )
        except Exception as exc:
            logger.exception(
                "Failed to copy run identity JSON (%s): %s",
                type(exc).__name__,
                exc,
            )
            self.statusMessage.emit("Failed to copy run identity")
            return
        self.statusMessage.emit("Copied run identity JSON")


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
        if not isinstance(update, FitLiveUpdate):
            raise TypeError("Fit tracker requires a FitLiveUpdate.")
        cost = float(update.cost)
        delta_text = ""
        if self._previous_cost is not None and math.isfinite(cost):
            delta_text = f" (delta {cost - self._previous_cost:.2e})"
        if math.isfinite(cost):
            self._previous_cost = cost
        lines, tooltips = self._update_parameter_lines(
            update,
            dataset_labels=dataset_labels,
            set_labels=set_labels,
        )
        self._set_live_text(
            f"Objective calls {int(update.iteration)} | Weighted objective SSQ "
            f"{_format_metric(cost)}{delta_text}",
            lines,
            tooltips,
        )

    def update_final(
        self,
        report: FitTerminalReport,
    ) -> None:
        if not isinstance(report, FitTerminalReport):
            raise TypeError("Fit tracker requires a FitTerminalReport.")
        cost = report.weighted_sse
        if cost.available:
            self._previous_cost = cost.value
        lines, tooltips = self._report_parameter_lines(report)
        objective_calls = (
            "unavailable"
            if report.objective_call_count is None
            else str(report.objective_call_count)
        )
        self._set_live_text(
            f"SciPy nfev {report.scipy_nfev} | "
            f"Terminal replay count {report.terminal_replay_count} | "
            f"Objective calls {objective_calls} | "
            f"Weighted objective SSQ {_format_metric(cost)}",
            lines,
            tooltips,
        )

    @staticmethod
    def _update_parameter_lines(
        update: FitLiveUpdate,
        *,
        dataset_labels: Mapping[str, str] | None = None,
        set_labels: Mapping[str, str] | None = None,
    ) -> tuple[list[str], list[str]]:
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
            alias = compact_text(
                initial_labels.get(set_id, f"Initial set {index + 1}"),
                max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
            )
            lines.append(
                f"[{alias.display}] "
                + " | ".join(
                    f"{compact_text(name, max_chars=STATUS_ITEM_LABEL_MAX_CHARS).display}="
                    f"{float(value):.6g}"
                    for name, value in values
                )
            )
            tooltips.append(f"{alias.full}: Set ID {set_id}")
        return lines, tooltips

    @staticmethod
    def _report_parameter_lines(
        report: FitTerminalReport,
    ) -> tuple[list[str], list[str]]:
        if not isinstance(report, FitTerminalReport):
            raise TypeError("Fit tracker requires a FitTerminalReport.")
        lines = [
            f"{compact_text(row.display_name, max_chars=STATUS_ITEM_LABEL_MAX_CHARS).display}="
            f"{row.fitted_value:.6g}"
            for row in report.shared_estimates
        ]
        tooltips = [
            f"{row.display_name}: Parameter ID {row.name}"
            for row in report.shared_estimates
        ]
        dataset_rows: dict[str, list[FitEstimateReportRow]] = {}
        for row in report.per_dataset_estimates:
            if row.dataset_id is not None:
                dataset_rows.setdefault(row.dataset_id, []).append(row)
        for index, (dataset_id, rows) in enumerate(sorted(dataset_rows.items())):
            label = compact_dataset_label(
                rows[0].dataset_display_name or dataset_alias(index),
                max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
            )
            lines.append(
                f"[{label.display}] "
                + " | ".join(
                    f"{compact_text(row.display_name, max_chars=STATUS_ITEM_LABEL_MAX_CHARS).display}="
                    f"{row.fitted_value:.6g} (Fit-local; not applied to Run)"
                    for row in sorted(rows, key=lambda item: item.display_name)
                )
            )
            tooltips.append(f"{label.full}: Dataset ID {dataset_id}")
        initial_rows: dict[str, list[FitEstimateReportRow]] = {}
        for row in report.initial_estimates:
            if row.set_id is not None:
                initial_rows.setdefault(row.set_id, []).append(row)
        for index, (set_id, rows) in enumerate(sorted(initial_rows.items())):
            alias = compact_text(
                rows[0].set_display_name or f"Initial set {index + 1}",
                max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
            )
            lines.append(
                f"[{alias.display}] "
                + " | ".join(
                    f"{compact_text(row.display_name, max_chars=STATUS_ITEM_LABEL_MAX_CHARS).display}="
                    f"{row.fitted_value:.6g}"
                    for row in sorted(rows, key=lambda item: item.display_name)
                )
            )
            tooltips.append(f"{alias.full}: Set ID {set_id}")
        return lines, tooltips

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
    """Historical Results page with typed live and terminal Fit inputs."""

    statusMessage = Signal(str)
    copyAvailableChanged = Signal(bool)
    initialSelectionChanged = Signal()
    sharedRefitRequested = Signal(object)
    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        color_resolver: ColorResolver | None = None,
    ) -> None:
        super().__init__(parent)
        if color_resolver is not None and not callable(color_resolver):
            raise TypeError("color_resolver must be callable.")
        self._color_resolver = color_resolver
        self._run_identity: FitRunIdentity | None = None
        self._candidate: FitCandidate | None = None
        self._last_report: FitTerminalReport | None = None
        self._initial_apply_checkboxes: dict[
            tuple[str, str], QtWidgets.QCheckBox
        ] = {}
        self._shared_refit_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        self._shared_refit_rows: dict[str, tuple[FitEstimateReportRow, ...]] = {}
        self._shared_refit_source_run_hash: str | None = None
        self._shared_refit_source_origin: tuple[str, str] | None = None
        self._shared_refit_seed_data = "arithmetic_mean"
        self._available_apply_scopes: frozenset[str] = frozenset()
        self._unavailable_apply_reasons: Mapping[str, str] = MappingProxyType({})
        self._summary_dialog: ResultsSummaryDialog | None = None
        self._dark_mode = False
        self._view_autorange_locked = False
        self._curves: tuple[FitCurveDataset, ...] = ()
        self._dataset_presentation: Mapping[str, DatasetPlotPresentation] = (
            MappingProxyType({})
        )
        self._dataset_tab_ids: list[_FitResultViewIdentity] = []
        self._dataset_labels: dict[str, str] = {}
        self._set_labels: dict[str, str] = {}
        self._dataset_plot_views: dict[_FitResultViewIdentity, GridPlotView] = {}
        self._all_datasets_plot_view: GridPlotView | None = None
        self._stale_plot_view_ids: set[_FitResultViewIdentity] = set()
        self._render_failures: dict[_FitResultViewIdentity, str] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self._tracker_panel = FitTrackerPanel(self)
        layout.addWidget(self._tracker_panel, stretch=0)

        identity_row = QtWidgets.QHBoxLayout()
        self._run_identity_label = QtWidgets.QLabel("Run: —", self)
        self._run_identity_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._copy_run_hash_button = QtWidgets.QPushButton("Copy hash", self)
        self._copy_run_hash_button.setEnabled(False)
        self._copy_run_hash_button.clicked.connect(self._copy_short_hash)
        identity_row.addWidget(self._run_identity_label, stretch=1)
        identity_row.addWidget(self._copy_run_hash_button)
        layout.addLayout(identity_row)

        self._render_diagnostic_label = QtWidgets.QLabel(self)
        self._render_diagnostic_label.setWordWrap(True)
        self._render_diagnostic_label.hide()
        layout.addWidget(self._render_diagnostic_label, stretch=0)

        self._terminal_failure_label = QtWidgets.QLabel(self)
        self._terminal_failure_label.setObjectName("global_fit_terminal_failure_label")
        self._terminal_failure_label.setWordWrap(True)
        self._terminal_failure_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._terminal_failure_label.hide()
        layout.addWidget(self._terminal_failure_label, stretch=0)

        self._apply_group = QtWidgets.QGroupBox("Apply to Project", self)
        apply_form = QtWidgets.QFormLayout(self._apply_group)
        self._apply_scope_labels: dict[str, QtWidgets.QLabel] = {}
        for scope, display_name in _APPLY_SCOPE_LABELS.items():
            value_label = QtWidgets.QLabel("—", self._apply_group)
            value_label.setWordWrap(True)
            apply_form.addRow(f"{display_name}:", value_label)
            self._apply_scope_labels[scope] = value_label
        self._apply_group.hide()
        layout.addWidget(self._apply_group, stretch=0)

        self._initial_apply_group = QtWidgets.QGroupBox(
            "Apply Initial Conditions — selected cells",
            self,
        )
        self._initial_apply_layout = QtWidgets.QVBoxLayout(
            self._initial_apply_group
        )
        self._initial_apply_group.hide()
        layout.addWidget(self._initial_apply_group, stretch=0)

        self._shared_refit_group = QtWidgets.QGroupBox(
            "Create shared values from this per-dataset fit",
            self,
        )
        shared_refit_layout = QtWidgets.QVBoxLayout(self._shared_refit_group)
        self._shared_refit_warning_label = QtWidgets.QLabel(
            self._shared_refit_group
        )
        self._shared_refit_warning_label.setWordWrap(True)
        self._shared_refit_warning_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        shared_refit_layout.addWidget(self._shared_refit_warning_label)
        self._shared_refit_explanation_label = QtWidgets.QLabel(
            "The starting estimate only initializes a new fit. Nothing is "
            "applied to the project until that Shared fit completes successfully.",
            self._shared_refit_group,
        )
        self._shared_refit_explanation_label.setWordWrap(True)
        self._shared_refit_explanation_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        shared_refit_layout.addWidget(self._shared_refit_explanation_label)
        self._shared_refit_parameter_layout = QtWidgets.QVBoxLayout()
        shared_refit_layout.addLayout(self._shared_refit_parameter_layout)
        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(QtWidgets.QLabel("Starting estimate", self._shared_refit_group))
        self._shared_refit_seed_combo = QtWidgets.QComboBox(
            self._shared_refit_group
        )
        self._shared_refit_seed_combo.setObjectName("fit_shared_refit_seed_combo")
        self._shared_refit_seed_combo.currentIndexChanged.connect(
            self._on_shared_refit_seed_changed
        )
        seed_row.addWidget(self._shared_refit_seed_combo, stretch=1)
        shared_refit_layout.addLayout(seed_row)
        self._shared_refit_button = QtWidgets.QPushButton(
            "Configure and Run New Fit",
            self._shared_refit_group,
        )
        self._shared_refit_button.setObjectName("fit_configure_shared_refit_button")
        self._shared_refit_button.clicked.connect(self._request_shared_refit)
        shared_refit_layout.addWidget(self._shared_refit_button)
        self._shared_refit_group.hide()
        layout.addWidget(self._shared_refit_group, stretch=0)

        self._subtabs = QtWidgets.QTabBar(self)
        self._subtabs.setObjectName("global_fit_results_subtabs")
        self._subtabs.setDocumentMode(True)
        self._subtabs.setDrawBase(False)
        self._subtabs.hide()
        layout.addWidget(self._subtabs, stretch=0)

        self._subtab_stack = QtWidgets.QStackedWidget(self)
        self._subtab_stack.setObjectName("global_fit_results_subtab_stack")
        layout.addWidget(self._subtab_stack, stretch=1)

        self._subtabs.currentChanged.connect(self._on_subtab_changed)

    def set_identity_labels(
        self,
        *,
        dataset_labels: Mapping[str, str],
        set_labels: Mapping[str, str],
    ) -> None:
        """Install bounded presentation labels while IDs remain typed identity."""

        self._dataset_labels = {
            str(key): str(value) for key, value in dataset_labels.items()
        }
        self._set_labels = {str(key): str(value) for key, value in set_labels.items()}

    def set_live_best(self, update: FitLiveUpdate) -> None:
        """Present one immutable live worker update in the historical tracker."""

        self._tracker_panel.update_from_best(
            update,
            dataset_labels=self._dataset_labels,
            set_labels=self._set_labels,
        )

    def set_live_curves(self, curves: Sequence[FitCurveDataset]) -> None:
        """Render immutable live curves in per-dataset and combined subtabs."""

        self._install_plot_projection(curves, {})

    def set_run_identity(self, identity: FitRunIdentity) -> None:
        """Present the immutable terminal identity used by the copy dialog."""

        if not isinstance(identity, FitRunIdentity):
            raise TypeError("identity must be a FitRunIdentity.")
        self._run_identity = identity
        self._run_identity_label.setText(f"Run: {identity.short_hash}")
        self._run_identity_label.setToolTip(identity.short_hash)
        self._copy_run_hash_button.setEnabled(True)
        self._refresh_summary_dialog()

    def _clear_shared_refit_controls(self) -> None:
        """Discard only the Results draft projection for Shared refit."""

        while self._shared_refit_parameter_layout.count():
            item = self._shared_refit_parameter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._shared_refit_checkboxes = {}
        self._shared_refit_rows = {}
        self._shared_refit_source_run_hash = None
        self._shared_refit_source_origin = None
        self._shared_refit_seed_combo.blockSignals(True)
        self._shared_refit_seed_combo.clear()
        self._shared_refit_seed_combo.blockSignals(False)
        self._shared_refit_seed_data = "arithmetic_mean"
        self._shared_refit_group.hide()
        self._shared_refit_button.setEnabled(False)

    def _shared_refit_source_rows(
        self,
        candidate: FitCandidate,
        report: FitTerminalReport,
    ) -> dict[str, tuple[FitEstimateReportRow, ...]]:
        """Return complete fitted Per-dataset coordinates for the source Fit."""

        per_dataset_names = {
            parameter.name
            for parameter in candidate.setup_snapshot.parameters
            if parameter.scope == "per_dataset" and parameter.selected_config.fit
        }
        local_scalar_names = {
            scalar_name
            for observable in getattr(candidate, "local_observables", ())
            for scalar_name in observable.scalar_names
        }
        grouped: dict[str, list[FitEstimateReportRow]] = {}
        for row in report.per_dataset_estimates:
            if row.name in per_dataset_names and row.name not in local_scalar_names:
                grouped.setdefault(row.name, []).append(row)
        return {
            name: tuple(sorted(rows, key=lambda row: row.dataset_id or ""))
            for name, rows in grouped.items()
            if rows
        }

    def _shared_refit_statistic_is_valid(self, statistic: str) -> bool:
        if statistic not in {"arithmetic_mean", "geometric_mean"}:
            return False
        selected = tuple(
            name
            for name, checkbox in self._shared_refit_checkboxes.items()
            if checkbox.isChecked()
        )
        if not selected:
            return False
        for name in selected:
            rows = self._shared_refit_rows.get(name, ())
            values = tuple(row.fitted_value for row in rows)
            if not values:
                return False
            if statistic == "geometric_mean" and any(value <= 0.0 for value in values):
                return False
            try:
                seed = (
                    math.exp(sum(math.log(value) for value in values) / len(values))
                    if statistic == "geometric_mean"
                    else sum(values) / len(values)
                )
            except (OverflowError, ValueError):
                return False
            minimum = max(row.minimum for row in rows)
            maximum = min(row.maximum for row in rows)
            if not math.isfinite(seed) or minimum >= maximum or not minimum <= seed <= maximum:
                return False
        return True

    def _refresh_shared_refit_seed_options(self) -> None:
        selected_names = tuple(
            name
            for name, checkbox in self._shared_refit_checkboxes.items()
            if checkbox.isChecked()
        )
        geometric_valid = bool(selected_names) and self._shared_refit_statistic_is_valid(
            "geometric_mean"
        )
        desired = self._shared_refit_seed_data
        if desired == "geometric_mean" and not geometric_valid:
            desired = "arithmetic_mean"
        self._shared_refit_seed_combo.blockSignals(True)
        self._shared_refit_seed_combo.clear()
        self._shared_refit_seed_combo.addItem(
            "Arithmetic mean",
            "arithmetic_mean",
        )
        if geometric_valid:
            self._shared_refit_seed_combo.addItem(
                "Geometric mean",
                "geometric_mean",
            )
        index = self._shared_refit_seed_combo.findData(desired)
        self._shared_refit_seed_combo.setCurrentIndex(max(0, index))
        self._shared_refit_seed_data = str(
            self._shared_refit_seed_combo.currentData() or "arithmetic_mean"
        )
        self._shared_refit_seed_combo.blockSignals(False)
        self._refresh_shared_refit_button()

    def _refresh_shared_refit_button(self) -> None:
        selected = tuple(
            name
            for name, checkbox in self._shared_refit_checkboxes.items()
            if checkbox.isChecked() and checkbox.isEnabled()
        )
        self._shared_refit_button.setEnabled(
            bool(
                self._shared_refit_source_run_hash
                and self._shared_refit_source_origin
                and self._last_report
                and selected
            )
            and self._shared_refit_statistic_is_valid(self._shared_refit_seed_data)
        )

    def _on_shared_refit_seed_changed(self, _index: int) -> None:
        data = self._shared_refit_seed_combo.currentData()
        self._shared_refit_seed_data = str(data or "arithmetic_mean")
        self._refresh_shared_refit_button()

    def _project_shared_refit_context(
        self,
        candidate: FitCandidate,
        report: FitTerminalReport,
    ) -> None:
        """Project one Results-owned Shared-refit draft from terminal truth."""

        self._clear_shared_refit_controls()
        self._shared_refit_source_run_hash = candidate.run_identity.short_hash
        self._shared_refit_source_origin = (
            candidate.origin.tab_id,
            candidate.origin.capture_id,
        )
        rows_by_name = self._shared_refit_source_rows(candidate, report)
        if not rows_by_name:
            return
        self._shared_refit_rows = rows_by_name
        for name, rows in rows_by_name.items():
            display_name = rows[0].display_name
            checkbox = QtWidgets.QCheckBox(
                display_name,
                self._shared_refit_group,
            )
            checkbox.setObjectName(f"fit_shared_refit_parameter_{name}")
            checkbox.setChecked(True)
            checkbox.setToolTip(
                f"Use the Per-dataset estimates for {display_name} "
                "as diagnostic seed material for a new Shared Fit."
            )
            checkbox.toggled.connect(self._refresh_shared_refit_seed_options)
            self._shared_refit_checkboxes[name] = checkbox
            self._shared_refit_parameter_layout.addWidget(checkbox)

        if report.joint_optimizer_converged:
            warning = (
                "This completed Per-dataset Fit supplies starting values only. "
                "Only a newly completed, converged Shared Fit is the publication "
                "authority."
            )
            self._shared_refit_warning_label.setStyleSheet("font-weight: 600;")
        else:
            warning = (
                "Diagnostic seed material: this Per-dataset Fit did not converge. "
                "Its finite estimates may initialize a new Fit, but they are not "
                "evidence and cannot be applied or published. Only a newly "
                "completed, converged Shared Fit is the publication authority."
            )
            self._shared_refit_warning_label.setStyleSheet(
                "font-weight: 700; color: #a33;"
            )
        self._shared_refit_warning_label.setText(warning)
        self._shared_refit_group.show()
        self._refresh_shared_refit_seed_options()

    def _request_shared_refit(self) -> None:
        report = self._last_report
        source_run_hash = self._shared_refit_source_run_hash
        source_origin = self._shared_refit_source_origin
        if source_run_hash is None or source_origin is None or report is None:
            self.statusMessage.emit("Shared refit requires a terminal Per-dataset Fit.")
            return
        selected_names = tuple(
            name
            for name, checkbox in self._shared_refit_checkboxes.items()
            if checkbox.isChecked() and checkbox.isEnabled()
        )
        statistic = str(
            self._shared_refit_seed_combo.currentData() or "arithmetic_mean"
        )
        if not selected_names or not self._shared_refit_statistic_is_valid(statistic):
            self.statusMessage.emit(
                "Select an eligible parameter and a scientifically valid starting estimate."
            )
            return
        promotions: list[SharedRefitPromotion] = []
        try:
            for name in selected_names:
                rows = self._shared_refit_rows[name]
                minimum = max(row.minimum for row in rows)
                maximum = min(row.maximum for row in rows)
                if minimum >= maximum:
                    raise ValueError(
                        f"{rows[0].display_name} has no common finite bounds."
                    )
                log_scale = all(row.log_scale for row in rows)
                if log_scale and minimum <= 0.0:
                    raise ValueError(
                        f"{rows[0].display_name} requires positive logarithmic bounds."
                    )
                promotions.append(
                    SharedRefitPromotion(
                        parameter_name=name,
                        minimum=minimum,
                        maximum=maximum,
                        log10=log_scale,
                        seed_statistic=statistic,
                    )
                )
        except (TypeError, ValueError) as exc:
            self.statusMessage.emit(f"Shared refit request rejected: {exc}")
            return
        intent = FitSharedRefitIntent(
            origin_tab_id=source_origin[0],
            origin_capture_id=source_origin[1],
            source_run_hash=source_run_hash,
            promotions=tuple(promotions),
        )
        self.sharedRefitRequested.emit(intent)

    def set_candidate(
        self,
        candidate: FitCandidate,
        *,
        report: FitTerminalReport,
        available_scopes: frozenset[str] | None = None,
        unavailable_reasons: Mapping[str, str] | None = None,
    ) -> bool:
        """Present a typed candidate; Apply feasibility remains owner-owned."""

        if not isinstance(candidate, FitCandidate):
            raise TypeError("candidate must be a FitCandidate.")
        if not isinstance(report, FitTerminalReport):
            raise TypeError("report must be a FitTerminalReport.")
        self.restore_terminal_semantics(
            candidate,
            report=report,
            available_scopes=available_scopes,
            unavailable_reasons=unavailable_reasons,
        )
        self._clear_render_failures()
        try:
            curves = project_fit_candidate_curves(candidate)
            presentation = self._candidate_plot_presentation(candidate, curves)
            rendered = self._install_plot_projection(curves, presentation)
        except Exception as exc:
            self._reject_terminal_projection(exc)
            return False
        return bool(curves and rendered)

    def restore_terminal_semantics(
        self,
        candidate: FitCandidate,
        *,
        report: FitTerminalReport,
        available_scopes: frozenset[str] | None = None,
        unavailable_reasons: Mapping[str, str] | None = None,
    ) -> None:
        """Restore Summary/Run/Apply truth without touching physical plots."""

        if not isinstance(candidate, FitCandidate):
            raise TypeError("candidate must be FitCandidate.")
        if not isinstance(report, FitTerminalReport):
            raise TypeError("report must be a FitTerminalReport.")
        self.set_apply_availability(available_scopes, unavailable_reasons)
        self.set_run_identity(candidate.run_identity)
        self._candidate = candidate
        self._terminal_failure_label.hide()
        self._last_report = report
        self._project_apply_availability()
        self._tracker_panel.update_final(report)
        self._project_shared_refit_context(candidate, report)
        self._refresh_summary_dialog()

    def set_shared_refit_source_candidate(
        self,
        candidate: FitCandidate,
        *,
        report: FitTerminalReport,
    ) -> None:
        """Project retry controls from the retained Per-dataset source.

        The visible terminal candidate/report remains the Last Fit truth; this
        separate projection only restores the Results draft used to author a
        retry after an unsuccessful Shared terminal.
        """

        if not isinstance(candidate, FitCandidate):
            raise TypeError("Shared-refit source must be a FitCandidate.")
        if not isinstance(report, FitTerminalReport):
            raise TypeError("Shared-refit source report must be a FitTerminalReport.")
        self._project_shared_refit_context(candidate, report)

    def mark_terminal_projection_unavailable(self, error: BaseException) -> None:
        """Disable physical plot/Copy authority while retaining terminal data."""

        self._reject_terminal_projection(error)

    def set_terminal_report_failure(self, error: BaseException) -> None:
        """Present semantic-report failure distinctly from physical plotting."""

        if not isinstance(error, FitTerminalReportError):
            raise TypeError("Semantic terminal failure must be FitTerminalReportError.")
        self._last_report = None
        self._tracker_panel.clear()
        payload = {
            "failure_type": type(error).__name__,
            "message": str(error),
        }
        self._terminal_failure_label.setText(
            "Fit terminal report failed — " + str(error)
        )
        self._terminal_failure_label.setToolTip(
            json.dumps(payload, sort_keys=True, indent=2)
        )
        self._terminal_failure_label.show()
        self.statusMessage.emit(str(error))
        self.copyAvailableChanged.emit(False)

    def set_terminal_failure(self, failure: FitWorkerFailure) -> None:
        """Present a typed worker failure without flattening its provenance."""

        if not isinstance(failure, FitWorkerFailure):
            raise TypeError("Terminal Fit failure must be FitWorkerFailure.")
        chain: list[str] = []
        current: FitWorkerFailure | None = failure
        while current is not None:
            code = f" [{current.code}]" if current.code else ""
            chain.append(
                f"{current.exception_type}{code}: {current.message}"
            )
            current = current.typed_cause
        text = "Fit failed — " + " Caused by: ".join(chain)
        self._terminal_failure_label.setText(text)
        self._terminal_failure_label.setToolTip(
            json.dumps(failure.to_payload(), sort_keys=True, indent=2)
        )
        self._terminal_failure_label.show()

    def set_initial_apply_proposal(
        self,
        proposal: FitInitialApplyProposal | None,
    ) -> None:
        """Display every owner-approved IC mutation as an independent choice."""

        if proposal is not None and not isinstance(
            proposal,
            FitInitialApplyProposal,
        ):
            raise TypeError("IC Apply proposal must be typed or None.")
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
            state = fit_bound_state(
                cell.fitted_value,
                cell.minimum,
                cell.maximum,
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
                f"{cell.maximum:.6g}]; bound state={state}",
                self._initial_apply_group,
            )
            checkbox.setToolTip(
                f"Set ID: {cell.set_id}\n"
                f"Dataset IDs: {', '.join(cell.dataset_ids) or 'none'}"
            )
            checkbox.setChecked(True)
            checkbox.toggled.connect(
                lambda _checked: self.initialSelectionChanged.emit()
            )
            self._initial_apply_checkboxes[cell.key] = checkbox
            self._initial_apply_layout.addWidget(checkbox)
        self._initial_apply_group.show()
        self.initialSelectionChanged.emit()

    def selected_initial_keys(self) -> tuple[tuple[str, str], ...]:
        """Return only independently checked owner-proposed IC cells."""

        return tuple(
            key
            for key, checkbox in self._initial_apply_checkboxes.items()
            if checkbox.isChecked()
        )

    def clear_candidate(self) -> None:
        """Clear all terminal/live result presentation without touching ownership."""

        self._run_identity = None
        self._candidate = None
        self._last_report = None
        self._available_apply_scopes = frozenset()
        self._unavailable_apply_reasons = MappingProxyType({})
        self._apply_group.hide()
        self._run_identity_label.setText("Run: —")
        self._run_identity_label.setToolTip("")
        self._copy_run_hash_button.setEnabled(False)
        self._tracker_panel.clear()
        self._terminal_failure_label.clear()
        self._terminal_failure_label.hide()
        self.set_initial_apply_proposal(None)
        self._clear_shared_refit_controls()
        self._clear_render_failures()
        self._install_plot_projection((), {})
        self.copyAvailableChanged.emit(False)
        if self._summary_dialog is not None and self._summary_dialog.isVisible():
            self._summary_dialog.close()

    def clear(self) -> None:
        """Clear the result presentation and remove its dataset subtabs."""

        self.clear_candidate()
        self._dataset_tab_ids = []
        self._dataset_labels = {}
        self._dataset_plot_views = {}
        self._all_datasets_plot_view = None
        self._clear_subtabs()

    def open_results_summary_dialog(self) -> None:
        """Show the historical non-modal Results Summary popup."""

        identity = self._run_identity
        if identity is None:
            return
        if self._summary_dialog is not None and self._summary_dialog.isVisible():
            self._summary_dialog.raise_()
            self._summary_dialog.activateWindow()
            return
        dialog = ResultsSummaryDialog(
            identity,
            candidate=self._candidate,
            report=self._last_report,
            parent=self,
        )
        dialog.statusMessage.connect(self.statusMessage)
        dialog.destroyed.connect(lambda: setattr(self, "_summary_dialog", None))
        self._summary_dialog = dialog
        dialog.show()

    def copy_run_identity_json(self) -> None:
        """Generate and copy verbose Run JSON only for an explicit request."""

        identity = self._run_identity
        clipboard = _get_clipboard()
        if identity is None or clipboard is None:
            raise ValueError("Fit run JSON is unavailable.")
        if self._candidate is None:
            clipboard.setText(identity.readable_json())
        else:
            if self._last_report is None:
                raise ValueError("Fit terminal report is unavailable.")
            clipboard.setText(
                _fit_run_identity_export_json(self._candidate, self._last_report)
            )

    def _copy_short_hash(self) -> None:
        identity = self._run_identity
        clipboard = _get_clipboard()
        if identity is None or clipboard is None:
            self.statusMessage.emit("Clipboard unavailable")
            return
        try:
            clipboard.setText(identity.short_hash)
        except Exception:
            self.statusMessage.emit("Failed to copy run identity")
            return
        self.statusMessage.emit("Copied run identity hash")

    @staticmethod
    def _normalize_apply_availability(
        available_scopes: frozenset[str] | None,
        unavailable_reasons: Mapping[str, str] | None,
    ) -> tuple[frozenset[str], dict[str, str]]:
        if available_scopes is None and unavailable_reasons is None:
            return frozenset(), {}
        available = frozenset(str(scope) for scope in (available_scopes or ()))
        reasons = {
            str(scope): str(reason or "").strip()
            for scope, reason in (unavailable_reasons or {}).items()
        }
        return available, reasons

    def set_apply_availability(
        self,
        available_scopes: frozenset[str] | None,
        unavailable_reasons: Mapping[str, str] | None,
    ) -> None:
        """Project owner-retained Apply feasibility independently of the view."""

        available, reasons = self._normalize_apply_availability(
            available_scopes,
            unavailable_reasons,
        )
        self._available_apply_scopes = available
        self._unavailable_apply_reasons = MappingProxyType(reasons)
        self._project_apply_availability()

    def _project_apply_availability(self) -> None:
        has_owner_projection = bool(
            self._available_apply_scopes or self._unavailable_apply_reasons
        )
        self._apply_group.setVisible(has_owner_projection)
        for scope, label in self._apply_scope_labels.items():
            if scope in self._available_apply_scopes:
                if scope == "parameters" and self._last_report is not None:
                    disposition = self._last_report.publication.message
                    label.setText(disposition)
                    label.setToolTip(disposition)
                else:
                    label.setText("Available")
                    label.setToolTip("")
                continue
            reason = self._unavailable_apply_reasons.get(scope, "")
            if scope == "parameters" and self._last_report is not None:
                reason = " ".join(
                    item
                    for item in (
                        reason,
                        self._last_report.publication.message,
                    )
                    if item
                )
            compact = compact_diagnostic_text(
                reason,
                max_chars=DIAGNOSTIC_LABEL_MAX_CHARS,
            )
            label.setText(compact.display or "Unavailable")
            label.setToolTip(compact.full if reason else "")

    def copy_eligible(self) -> bool:
        return bool(
            self._curves
            and self._all_datasets_plot_view is not None
            and (
                _FIT_AGGREGATE_VIEW in self._stale_plot_view_ids
                or self._all_datasets_plot_view.copy_eligible()
            )
        )

    def copy_all_to_clipboard(self) -> None:
        view = self._all_datasets_plot_view
        candidate = self._candidate
        if view is None or candidate is None:
            raise ValueError("No rendered Fit curves are available to copy.")
        if _FIT_AGGREGATE_VIEW in self._stale_plot_view_ids:
            if not self._refresh_plot_view(_FIT_AGGREGATE_VIEW):
                return
        if not view.copy_eligible():
            self.copyAvailableChanged.emit(False)
            return
        try:
            curves = project_fit_candidate_curves(candidate)
        except Exception as exc:
            view.reject_projection(exc)
            self._stale_plot_view_ids.discard(_FIT_AGGREGATE_VIEW)
            self.copyAvailableChanged.emit(False)
            return
        view.copy_projection_to_clipboard(curves)

    def set_view_autorange_locked(self, running: bool) -> None:
        """Retain the historical presentation state at this composition boundary."""

        self._view_autorange_locked = bool(running)
        for plot_view in list(self._dataset_plot_views.values()):
            plot_view.set_autorange_locked(self._view_autorange_locked)
        if self._all_datasets_plot_view is not None:
            self._all_datasets_plot_view.set_autorange_locked(
                self._view_autorange_locked
            )

    def set_dark_mode(self, enabled: bool) -> None:
        """Retain the historical presentation state at this composition boundary."""

        self._dark_mode = bool(enabled)
        for plot_view in list(self._dataset_plot_views.values()):
            plot_view.set_dark_mode(self._dark_mode)
        if self._all_datasets_plot_view is not None:
            self._all_datasets_plot_view.set_dark_mode(self._dark_mode)

    def _ensure_subtabs(self, curves: Sequence[FitCurveDataset]) -> None:
        ids = [
            _FitResultViewIdentity("dataset", curve.dataset_id)
            for curve in curves
        ]
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
            compact_label = compact_dataset_label(
                full_label,
                max_chars=TAB_LABEL_MAX_CHARS,
            )
            plot_view = self._create_plot_view(
                f"global_fit_results_plot_{curve.dataset_id}",
                view_identity,
            )
            self._dataset_plot_views[view_identity] = plot_view
            self._subtab_stack.addWidget(plot_view)
            tab_index = self._subtabs.addTab(compact_label.display)
            self._subtabs.setTabToolTip(
                tab_index,
                f"{compact_label.full}\nDataset ID: {curve.dataset_id}",
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
            "global_fit_results_plot_all_datasets",
            _FIT_AGGREGATE_VIEW,
        )
        self._subtab_stack.addWidget(self._all_datasets_plot_view)
        self._subtabs.addTab("All Datasets")
        self._subtabs.setVisible(self._subtabs.count() > 0)
        self._subtabs.setCurrentIndex(0)
        self._subtab_stack.setCurrentIndex(0)

    def _create_plot_view(
        self,
        object_name: str,
        view_id: _FitResultViewIdentity,
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
            lambda message, identity=view_id: self._record_render_failure(
                identity,
                message,
            )
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
        """Install one Fit payload and project only the active Results view."""

        values = tuple(curves)
        if any(not isinstance(item, FitCurveDataset) for item in values):
            raise TypeError("Fit curves must contain FitCurveDataset values.")
        ids = [item.dataset_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("Fit curve datasets must have unique dataset IDs.")
        if not isinstance(presentation, Mapping):
            raise TypeError("Fit plot presentation must be a mapping.")
        self._curves = values
        self._dataset_presentation = MappingProxyType(dict(presentation))
        if values:
            self._ensure_subtabs(values)
        self._stale_plot_view_ids.update(self._dataset_plot_views)
        if self._all_datasets_plot_view is not None:
            self._stale_plot_view_ids.add(_FIT_AGGREGATE_VIEW)
        rendered = self._refresh_active_plot_view()
        self.copyAvailableChanged.emit(self.copy_eligible())
        return rendered

    def _active_plot_view_id(self) -> _FitResultViewIdentity | None:
        index = self._subtabs.currentIndex()
        if index < 0:
            return None
        if index == len(self._dataset_tab_ids):
            return _FIT_AGGREGATE_VIEW
        if 0 <= index < len(self._dataset_tab_ids):
            return self._dataset_tab_ids[index]
        return None

    def _refresh_active_plot_view(self) -> bool:
        view_id = self._active_plot_view_id()
        return bool(view_id is not None and self._refresh_plot_view(view_id))

    def _refresh_plot_view(self, view_id: _FitResultViewIdentity) -> bool:
        if view_id not in self._stale_plot_view_ids:
            plot_view = (
                self._all_datasets_plot_view
                if view_id.kind == "aggregate"
                else self._dataset_plot_views.get(view_id)
            )
            return bool(plot_view is not None and plot_view.copy_eligible())
        if view_id.kind == "aggregate":
            plot_view = self._all_datasets_plot_view
            curves = self._curves
        else:
            plot_view = self._dataset_plot_views.get(view_id)
            curves = tuple(
                curve
                for curve in self._curves
                if curve.dataset_id == view_id.dataset_id
            )
        if plot_view is None:
            return False
        try:
            rendered = plot_view.set_projection(
                curves,
                self._dataset_presentation,
            )
        except Exception as exc:
            plot_view.reject_projection(exc)
            rendered = False
        self._stale_plot_view_ids.discard(view_id)
        if not curves:
            self._clear_render_failure(view_id)
        elif rendered:
            self._clear_render_failure(view_id)
        elif view_id not in self._render_failures:
            self._record_render_failure(view_id, "Fit plot rendering failed.")
        # A failed Grid emits its render-state signal before this lazy view is
        # removed from the stale set.  Re-emit after settlement so aggregate
        # Copy cannot retain that transient pre-render authority.
        self.copyAvailableChanged.emit(self.copy_eligible())
        return bool(rendered)

    def _on_subtab_changed(self, index: int) -> None:
        if 0 <= int(index) < self._subtab_stack.count():
            self._subtab_stack.setCurrentIndex(int(index))
        self._refresh_active_plot_view()
        self.copyAvailableChanged.emit(self.copy_eligible())

    def _reject_terminal_projection(self, error: BaseException) -> None:
        """Clear physical/Copy authority while retaining terminal semantics."""

        self._curves = ()
        self._dataset_presentation = MappingProxyType({})
        self._stale_plot_view_ids.clear()
        for plot_view in (
            *self._dataset_plot_views.values(),
            *((self._all_datasets_plot_view,) if self._all_datasets_plot_view else ()),
        ):
            plot_view.reject_projection(error)
        self._record_render_failure(_FIT_AGGREGATE_VIEW, str(error))
        self.copyAvailableChanged.emit(False)

    def _record_render_failure(
        self,
        view_id: _FitResultViewIdentity,
        message: object,
    ) -> None:
        diagnostic = str(message or "Fit plot rendering failed.")
        self._render_failures[view_id] = diagnostic
        compact = compact_diagnostic_text(
            diagnostic,
            max_chars=DIAGNOSTIC_LABEL_MAX_CHARS,
        )
        self._render_diagnostic_label.setText(
            "Fitted values remain available; plot unavailable: "
            f"{compact.display or 'Fit plot rendering failed.'}"
        )
        self._render_diagnostic_label.setToolTip(compact.full)
        self._render_diagnostic_label.show()
        self.statusMessage.emit(diagnostic)

    def _clear_render_failure(self, view_id: _FitResultViewIdentity) -> None:
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
        compact = compact_diagnostic_text(
            diagnostic,
            max_chars=DIAGNOSTIC_LABEL_MAX_CHARS,
        )
        self._render_diagnostic_label.setText(
            "Fitted values remain available; plot unavailable: "
            f"{compact.display or 'Fit plot rendering failed.'}"
        )
        self._render_diagnostic_label.setToolTip(compact.full)
        self._render_diagnostic_label.show()

    def _candidate_plot_presentation(
        self,
        candidate: FitCandidate,
        curves: Sequence[FitCurveDataset],
    ) -> Mapping[str, DatasetPlotPresentation]:
        """Freeze terminal plot text directly from the result's dataset info."""

        infos_by_id = {
            str(getattr(info, "dataset_id", "")): info
            for info in tuple(getattr(candidate.result, "dataset_info", ()) or ())
            if str(getattr(info, "dataset_id", ""))
        }
        presentation: dict[str, DatasetPlotPresentation] = {}
        for curve in curves:
            info = infos_by_id.get(curve.dataset_id)
            if info is None:
                continue
            try:
                mse_text = (
                    "Weighted MSE = "
                    + fit_metric_text(info.weighted_residual_mse, precision=3)
                )
            except (AttributeError, TypeError, ValueError):
                continue
            compact_name = compact_plot_title_label(
                curve.display_name,
                max_chars=PLOT_TITLE_LABEL_MAX_CHARS,
            )
            tooltip_parts = [
                compact_name.full,
                f"Dataset ID: {curve.dataset_id}",
                mse_text,
            ]
            r_squared_by_series = getattr(info, "r_squared_by_series", {})
            if isinstance(r_squared_by_series, Mapping):
                for series_name, value in r_squared_by_series.items():
                    try:
                        r_squared_text = fit_metric_text(value)
                    except (TypeError, ValueError):
                        continue
                    tooltip_parts.append(f"{str(series_name)}: R² = {r_squared_text}")
            presentation[curve.dataset_id] = (
                f"{compact_name.display} ({mse_text})",
                "\n".join(tooltip_parts),
            )
        return MappingProxyType(presentation)

    def _clear_subtabs(self) -> None:
        self._stale_plot_view_ids.clear()
        while self._subtabs.count():
            self._subtabs.removeTab(self._subtabs.count() - 1)
        while self._subtab_stack.count():
            widget = self._subtab_stack.widget(0)
            self._subtab_stack.removeWidget(widget)
            widget.deleteLater()
        self._subtabs.hide()

    def _refresh_summary_dialog(self) -> None:
        if self._summary_dialog is None or not self._summary_dialog.isVisible():
            return
        identity = self._run_identity
        if identity is None:
            self._summary_dialog.close()
            return
        self._summary_dialog.refresh(
            identity,
            candidate=self._candidate,
            report=self._last_report,
        )
