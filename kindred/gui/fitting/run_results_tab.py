"""Historical Results-tab presentation bound to the current typed Fit values."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

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
from kindred.gui.fitting.contracts import FitRunIdentity
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
DatasetParameterPresentation: TypeAlias = tuple[str, str, Mapping[str, float]]
SetInitialPresentation: TypeAlias = tuple[str, str, Mapping[str, float]]
DatasetMetricPresentation: TypeAlias = tuple[
    str,
    str,
    float,
    float,
    float,
    float,
    int,
    tuple[tuple[str, float], ...],
]
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
    try:
        return f"{float(value):.2e}"
    except Exception:
        return "nan"


def _objective_cost(result: object) -> float:
    residuals = getattr(result, "objective_residuals", None)
    try:
        values = np.asarray(residuals, dtype=float).reshape(-1)
    except Exception:
        values = np.asarray([], dtype=float)
    if values.size:
        return float(np.sum(values**2))
    try:
        return float(getattr(result, "weighted_residual_sse"))
    except Exception:
        return float("nan")


def _parameter_pairs(values: object) -> tuple[tuple[str, float], ...]:
    if isinstance(values, Mapping):
        items = values.items()
    else:
        items = values or ()
    normalized: list[tuple[str, float]] = []
    for name, value in items:
        normalized.append((str(name), float(value)))
    return tuple(normalized)


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


def _fit_run_identity_export_payload(candidate: FitCandidate) -> dict[str, object]:
    """Build the complete Copy Run JSON view without retaining it."""

    if not isinstance(candidate, FitCandidate):
        raise TypeError("candidate must be FitCandidate.")
    identity_payload = candidate.run_identity.payload
    if not isinstance(identity_payload, Mapping):
        raise TypeError("Fit run identity payload must be a JSON object.")
    result = candidate.result
    curves = project_fit_candidate_curves(candidate)
    terminal_replay: dict[str, object] = {
        "fit_solver": identity_payload.get("fit_solver"),
        "status": result.completion.status,
        "optimizer_converged": bool(result.completion.optimizer_converged),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "weighted_residual_sse": float(result.weighted_residual_sse),
        "weighted_residual_mse": float(result.weighted_residual_mse),
        "shared_parameters": {
            str(name): float(value)
            for name, value in sorted(result.shared_params.items())
        },
        "dataset_parameters": {
            str(dataset_id): {
                str(name): float(value)
                for name, value in sorted(values.items())
            }
            for dataset_id, values in sorted(result.dataset_params.items())
        },
        "objective_rows": [],
    }
    dataset_info = {str(info.dataset_id): info for info in result.dataset_info}
    terminal_replay["effective_dataset_weights"] = {
        dataset_id: float(info.weight)
        for dataset_id, info in dataset_info.items()
    }
    terminal_replay["solver_provenance_by_dataset"] = {
        str(dataset_id): simulation.provenance.solver
        for dataset_id, simulation in result.simulation_results_by_dataset.items()
    }
    for curve in sorted(curves, key=lambda item: item.dataset_id):
        objective_model_x = dict(curve.objective_model_x)
        objective_model_y = dict(curve.objective_model_y)
        objective_segment_ids = dict(curve.objective_model_segment_ids)
        objective_segment_keys = dict(curve.objective_model_segment_keys)
        objective_discontinuities = dict(curve.objective_model_discontinuity_ids)
        objective_rows = []
        for series_name, _values in curve.objective_model_y:
            objective_rows.append(
                {
                    "model_target": series_name,
                    "model_x": _export_array(objective_model_x[series_name]),
                    "model_y": _export_array(objective_model_y[series_name]),
                    "segment_ids": _export_array(
                        objective_segment_ids[series_name]
                    ),
                    "segment_keys": _export_array(
                        objective_segment_keys[series_name]
                    ),
                    "discontinuity_ids": _export_array(
                        objective_discontinuities[series_name]
                    ),
                    "selection_provenance": _export_array(
                        dict(curve.objective_selection_provenance)[series_name]
                    ),
                    "raw_observation_ids": _export_array(
                        dict(curve.objective_observation_ids)[series_name]
                    ),
                    "raw_columns": _export_array(
                        dict(curve.objective_raw_columns)[series_name]
                    ),
                    "source_rows": _export_array(
                        dict(curve.objective_source_rows)[series_name]
                    ),
                }
            )
        terminal_replay["objective_rows"].append(
            {"dataset_id": curve.dataset_id, "series": objective_rows}
        )
    export = dict(identity_payload)
    export["schema_version"] = 2
    export["export_kind"] = "verbose_fit_run"
    export["candidate"] = {
        "origin_tab_id": candidate.origin.tab_id,
        "origin_capture_id": candidate.origin.capture_id,
        "parameters": dict(candidate.parameters),
        "scalar_values_by_dataset": {
            dataset_id: dict(values)
            for dataset_id, values in candidate.scalar_values_by_dataset
        },
        "initial_values_by_set": {
            set_id: dict(values)
            for set_id, values in candidate.initial_values_by_set
        },
        "mechanism_text": candidate.setup_snapshot.mechanism_source.reactions_text,
        "prepared_analysis_identity": candidate.setup_snapshot.prepared_dsl.analysis.identity,
    }
    export["terminal_replay"] = terminal_replay
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


def _fit_run_identity_export_json(candidate: FitCandidate) -> str:
    """Serialize the complete Fit export at its single final plain-data boundary."""

    return json.dumps(
        _plain_json_data(_fit_run_identity_export_payload(candidate)),
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
        stats: Mapping[str, object] | None = None,
        fitted_params: Sequence[tuple[str, float]] | Mapping[str, float] | None = None,
        dataset_fitted_params: Sequence[DatasetParameterPresentation] | None = None,
        set_initial_values: Sequence[SetInitialPresentation] | None = None,
        dataset_metrics: Sequence[DatasetMetricPresentation] | None = None,
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
            "Optimizer converged",
            "Optimizer diagnostic",
            "Weighted SSE",
            "Weighted MSE",
            "Optimizer evaluations",
            "Message",
            "Objective SSQ",
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
        body_layout.addStretch(1)

        self.refresh(
            identity,
            stats=stats,
            fitted_params=fitted_params,
            dataset_fitted_params=dataset_fitted_params,
            set_initial_values=set_initial_values,
            dataset_metrics=dataset_metrics,
        )

    def _update_fitted_params(
        self,
        fitted_params: Sequence[tuple[str, float]] | Mapping[str, float] | None,
        dataset_fitted_params: Sequence[DatasetParameterPresentation] | None,
        set_initial_values: Sequence[SetInitialPresentation] | None,
    ) -> None:
        shared = _parameter_pairs(fitted_params)
        dataset_values = list(dataset_fitted_params or ())
        initial_values = list(set_initial_values or ())
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
        for dataset_label, dataset_id, values in dataset_values:
            pairs = _parameter_pairs(values)
            if not pairs:
                continue
            if lines:
                lines.append("")
            lines.append(f"{dataset_label}:")
            lines.extend(
                f"  {name} = {value:.6g}"
                for name, value in sorted(pairs, key=lambda item: str(item[0]))
            )
            tooltip_lines.append(f"{dataset_label}: Dataset ID {dataset_id}")
        for set_label, set_id, values in initial_values:
            pairs = _parameter_pairs(values)
            if not pairs:
                continue
            if lines:
                lines.append("")
            lines.append(f"{set_label} initial concentrations:")
            lines.extend(
                f"  {name} = {value:.6g}"
                for name, value in sorted(pairs, key=lambda item: str(item[0]))
            )
            tooltip_lines.append(f"{set_label}: Set ID {set_id}")
        self._params_content_label.setText("\n".join(lines))
        self._params_content_label.setToolTip("\n".join(tooltip_lines))

    def _update_dataset_metrics(
        self,
        dataset_metrics: Sequence[DatasetMetricPresentation] | None,
    ) -> None:
        if not dataset_metrics:
            self._dataset_metrics_label.setText("No per-dataset metrics available.")
            self._dataset_metrics_label.setToolTip("")
            return
        lines: list[str] = []
        tooltip_lines: list[str] = []
        for (
            display_name,
            dataset_id,
            weighted_mse,
            weighted_rmse,
            weighted_mae,
            effective_weight,
            n_points,
            r_squared,
        ) in dataset_metrics:
            lines.append(
                f"{display_name}: weighted MSE = {weighted_mse:.6g}; "
                f"RMSE = {weighted_rmse:.6g}; MAE = {weighted_mae:.6g}; "
                f"effective weight = {effective_weight:.6g}; points = {n_points}"
            )
            lines.extend(
                f"  {compact_text(series, max_chars=STATUS_ITEM_LABEL_MAX_CHARS).display}: "
                f"R² = {value:.6g}"
                for series, value in r_squared
            )
            tooltip_lines.append(f"{display_name}: Dataset ID {dataset_id}")
        self._dataset_metrics_label.setText("\n".join(lines))
        self._dataset_metrics_label.setToolTip("\n".join(tooltip_lines))

    def _update_stats(self, stats: Mapping[str, object] | None) -> None:
        values = stats or {}
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
        stats: Mapping[str, object] | None = None,
        fitted_params: Sequence[tuple[str, float]] | Mapping[str, float] | None = None,
        dataset_fitted_params: Sequence[DatasetParameterPresentation] | None = None,
        set_initial_values: Sequence[SetInitialPresentation] | None = None,
        dataset_metrics: Sequence[DatasetMetricPresentation] | None = None,
    ) -> None:
        if not isinstance(identity, FitRunIdentity):
            raise TypeError("Results summary requires a FitRunIdentity.")
        self._identity = identity
        if candidate is not None:
            if not isinstance(candidate, FitCandidate):
                raise TypeError("candidate must be FitCandidate.")
            self._candidate = candidate
        self._run_identity_label.setText(f"Run: {identity.short_hash}")
        self._update_stats(stats)
        self._update_fitted_params(
            fitted_params,
            dataset_fitted_params,
            set_initial_values,
        )
        self._update_dataset_metrics(dataset_metrics)

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
                clipboard.setText(_fit_run_identity_export_json(self._candidate))
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
            f"Objective calls {int(update.iteration)} | Objective SSQ "
            f"{_format_metric(cost)}{delta_text}",
            lines,
            tooltips,
        )

    def update_final(
        self,
        candidate: FitCandidate,
        *,
        set_labels: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(candidate, FitCandidate):
            raise TypeError("Fit tracker requires a FitCandidate.")
        cost = _objective_cost(candidate.result)
        if math.isfinite(cost):
            self._previous_cost = cost
        lines, tooltips = self._candidate_parameter_lines(
            candidate,
            set_labels=set_labels,
        )
        self._set_live_text(
            "Optimizer evaluations "
            f"{int(candidate.result.nfev)} | Objective SSQ {_format_metric(cost)}",
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
                    f"{float(value):.6g}"
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
    def _candidate_parameter_lines(
        candidate: FitCandidate,
        *,
        set_labels: Mapping[str, str] | None = None,
    ) -> tuple[list[str], list[str]]:
        parameter_labels = {
            parameter.name: parameter.display_name
            for parameter in candidate.setup_snapshot.parameters
        }
        dataset_labels = {
            mapped.dataset_id: mapped.display_name
            for mapped in candidate.origin.mapped_datasets
        }
        initial_labels = dict(set_labels or {})
        lines = [
            f"{compact_text(parameter_labels.get(name, name), max_chars=STATUS_ITEM_LABEL_MAX_CHARS).display}="
            f"{float(value):.6g}"
            for name, value in candidate.parameters
        ]
        tooltips = [
            f"{parameter_labels.get(name, name)}: Parameter ID {name}"
            for name, _value in candidate.parameters
        ]
        for index, (dataset_id, values) in enumerate(
            candidate.scalar_values_by_dataset
        ):
            if not values:
                continue
            label = compact_dataset_label(
                dataset_labels.get(dataset_id, dataset_alias(index)),
                max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
            )
            lines.append(
                f"[{label.display}] "
                + " | ".join(
                    f"{compact_text(parameter_labels.get(name, name), max_chars=STATUS_ITEM_LABEL_MAX_CHARS).display}="
                    f"{float(value):.6g}"
                    for name, value in values
                )
            )
            tooltips.append(f"{label.full}: Dataset ID {dataset_id}")
        for index, (set_id, values) in enumerate(candidate.initial_values_by_set):
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
        self._last_stats: dict[str, object] = {}
        self._last_fitted_params: tuple[tuple[str, float], ...] = ()
        self._last_dataset_fitted_params: tuple[DatasetParameterPresentation, ...] = ()
        self._last_set_initial_values: tuple[SetInitialPresentation, ...] = ()
        self._last_dataset_metrics: tuple[DatasetMetricPresentation, ...] = ()
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

    def set_candidate(
        self,
        candidate: FitCandidate,
        *,
        available_scopes: frozenset[str] | None = None,
        unavailable_reasons: Mapping[str, str] | None = None,
    ) -> bool:
        """Present a typed candidate; Apply feasibility remains owner-owned."""

        if not isinstance(candidate, FitCandidate):
            raise TypeError("candidate must be a FitCandidate.")
        self.restore_terminal_semantics(
            candidate,
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
        available_scopes: frozenset[str] | None = None,
        unavailable_reasons: Mapping[str, str] | None = None,
    ) -> None:
        """Restore Summary/Run/Apply truth without touching physical plots."""

        if not isinstance(candidate, FitCandidate):
            raise TypeError("candidate must be FitCandidate.")
        self.set_apply_availability(available_scopes, unavailable_reasons)
        self.set_run_identity(candidate.run_identity)
        self._candidate = candidate
        self._last_fitted_params = tuple(candidate.parameters)
        self._last_dataset_fitted_params = self._dataset_parameter_entries(candidate)
        self._last_set_initial_values = self._set_initial_entries(candidate)
        self._last_dataset_metrics = self._dataset_metric_entries(candidate)
        self._last_stats = self._candidate_stats(candidate)
        self._tracker_panel.update_final(candidate, set_labels=self._set_labels)
        self._refresh_summary_dialog()

    def mark_terminal_projection_unavailable(self, error: BaseException) -> None:
        """Disable physical plot/Copy authority while retaining terminal data."""

        self._reject_terminal_projection(error)

    def clear_candidate(self) -> None:
        """Clear all terminal/live result presentation without touching ownership."""

        self._run_identity = None
        self._candidate = None
        self._last_stats = {}
        self._last_fitted_params = ()
        self._last_dataset_fitted_params = ()
        self._last_set_initial_values = ()
        self._last_dataset_metrics = ()
        self._available_apply_scopes = frozenset()
        self._unavailable_apply_reasons = MappingProxyType({})
        self._apply_group.hide()
        self._run_identity_label.setText("Run: —")
        self._run_identity_label.setToolTip("")
        self._copy_run_hash_button.setEnabled(False)
        self._tracker_panel.clear()
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
            stats=self._last_stats or None,
            fitted_params=self._last_fitted_params or None,
            dataset_fitted_params=self._last_dataset_fitted_params or None,
            set_initial_values=self._last_set_initial_values or None,
            dataset_metrics=self._last_dataset_metrics or None,
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
            clipboard.setText(_fit_run_identity_export_json(self._candidate))

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
                label.setText("Available")
                label.setToolTip("")
                continue
            reason = self._unavailable_apply_reasons.get(scope, "")
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
                mse_text = f"MSE = {float(info.weighted_residual_mse):.3e}"
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
                        r_squared_text = f"{float(value):.6g}"
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

    def _dataset_parameter_entries(
        self,
        candidate: FitCandidate,
    ) -> tuple[DatasetParameterPresentation, ...]:
        raw_values = {
            dataset_id: dict(values)
            for dataset_id, values in candidate.scalar_values_by_dataset
        }
        labels = {
            mapped.dataset_id: mapped.display_name
            for mapped in candidate.origin.mapped_datasets
        }
        counts: dict[str, int] = {}
        for label in labels.values():
            counts[label] = counts.get(label, 0) + 1
        entries: list[DatasetParameterPresentation] = []
        for dataset_id in sorted(raw_values, key=str):
            values = raw_values[dataset_id]
            if not isinstance(values, Mapping) or not values:
                continue
            normalized_id = str(dataset_id)
            label = labels.get(normalized_id, normalized_id)
            if counts.get(label, 0) > 1:
                label = f"{label} ({normalized_id})"
            compact_label = compact_dataset_label(
                label,
                max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
            )
            entries.append((compact_label.display, normalized_id, values))
        return tuple(entries)

    def _dataset_metric_entries(
        self,
        candidate: FitCandidate,
    ) -> tuple[DatasetMetricPresentation, ...]:
        labels = {
            mapped.dataset_id: mapped.display_name
            for mapped in candidate.origin.mapped_datasets
        }
        counts: dict[str, int] = {}
        for label in labels.values():
            counts[label] = counts.get(label, 0) + 1
        entries: list[DatasetMetricPresentation] = []
        for info in candidate.result.dataset_info:
            dataset_id = str(info.dataset_id)
            label = labels.get(dataset_id, dataset_id)
            if counts.get(label, 0) > 1:
                label = f"{label} ({dataset_id})"
            compact_label = compact_dataset_label(
                label,
                max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
            )
            entries.append(
                (
                    compact_label.display,
                    dataset_id,
                    float(info.weighted_residual_mse),
                    float(info.weighted_rmse),
                    float(info.weighted_mae),
                    float(info.weight),
                    int(info.n_points),
                    tuple(
                        sorted(
                            (
                                str(series_name),
                                float(value),
                            )
                            for series_name, value in info.r_squared_by_series.items()
                        )
                    ),
                )
            )
        return tuple(entries)

    def _set_initial_entries(
        self,
        candidate: FitCandidate,
    ) -> tuple[SetInitialPresentation, ...]:
        entries: list[SetInitialPresentation] = []
        for index, (set_id, values) in enumerate(candidate.initial_values_by_set):
            if not values:
                continue
            compact = compact_dataset_label(
                self._set_labels.get(set_id, f"Initial set {index + 1}"),
                max_chars=STATUS_ITEM_LABEL_MAX_CHARS,
            )
            entries.append((compact.display, set_id, dict(values)))
        return tuple(entries)

    @staticmethod
    def _optimizer_coordinate_count(candidate: FitCandidate) -> int:
        setup = candidate.setup_snapshot
        included_dataset_count = sum(
            1 for dataset in setup.datasets if dataset.included
        )
        parameter_coordinates = sum(
            1 if parameter.scope == "shared" else included_dataset_count
            for parameter in setup.parameters
            if parameter.selected_config.fit
        )
        fitted_initial_keys = {
            (set_id, species)
            for set_id, values in candidate.initial_values_by_set
            for species, _value in values
        }
        initial_scopes = dict(setup.initial_scopes)
        global_initial_species = {
            initial.species
            for initial in setup.global_initials
            if (
                initial.fit
                and initial_scopes.get(initial.species) == "global"
                and any(
                    (set_id, initial.species) in fitted_initial_keys
                    for set_id, _value in candidate.initial_values_by_set
                )
            )
        }
        local_initial_coordinates = sum(
            1
            for initial in setup.initials
            if (
                initial.fit
                and initial_scopes.get(initial.species) == "local"
                and (initial.set_id, initial.species) in fitted_initial_keys
            )
        )
        return (
            parameter_coordinates
            + len(global_initial_species)
            + local_initial_coordinates
        )

    @staticmethod
    def _optimizer_diagnostic_text(candidate: FitCandidate) -> str:
        diagnostic = candidate.result.completion.optimizer_diagnostic
        if diagnostic is None:
            return "None"
        failure = diagnostic.failure
        message = str(failure.get("message") or "Optimizer diagnostic")
        dataset_text = f" [{diagnostic.dataset_id}]" if diagnostic.dataset_id else ""
        return f"{diagnostic.phase}{dataset_text}: {message}"

    def _candidate_stats(self, candidate: FitCandidate) -> dict[str, object]:
        result = candidate.result
        infos = tuple(result.dataset_info)
        points = sum(int(info.n_points) for info in infos)
        series = sum(len(values) for values in result.model_series.values())
        coordinate_count = self._optimizer_coordinate_count(candidate)
        completion = result.completion
        return {
            "Datasets": len(result.simulation_results_by_dataset),
            "Series": series,
            "Points": points,
            "Parameters": coordinate_count,
            "DF": points - coordinate_count,
            "Completion": completion.status,
            "Optimizer converged": completion.optimizer_converged,
            "Optimizer diagnostic": self._optimizer_diagnostic_text(candidate),
            "Weighted SSE": float(result.weighted_residual_sse),
            "Weighted MSE": float(result.weighted_residual_mse),
            "Optimizer evaluations": int(result.nfev),
            "Message": str(result.message),
            "Objective SSQ": _objective_cost(result),
        }

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
            stats=self._last_stats or None,
            fitted_params=self._last_fitted_params or None,
            dataset_fitted_params=self._last_dataset_fitted_params or None,
            set_initial_values=self._last_set_initial_values or None,
            dataset_metrics=self._last_dataset_metrics or None,
        )
