"""Typed Fit reporting derived only from immutable terminal science."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from kindred.core.analysis.fit_statistics import (
    FitMetric,
    finite_positive_mean_normalize,
)
from kindred.core.analysis.global_fit_execution import (
    CentredFixedEffectsSettlement,
    FitCoordinate,
)
from kindred.core.fitting_completion import FitTerminalExtension
from kindred.core.simulation_failure import FactualCause
from kindred.gui.fitting.launch import FitPublicationClassification
from kindred.gui.fitting.worker import FitCandidate

__all__ = [
    "FitAggregateReportRow",
    "FitEstimateReportRow",
    "FitMetricReportRow",
    "FitProfileIntervalReportRow",
    "FitPublicationReport",
    "FitTerminalReportError",
    "FitTerminalReport",
    "build_fit_terminal_report",
    "fit_metric_text",
    "fit_bound_state",
    "fit_primary_model_lines",
    "fit_terminal_extension_lines",
    "fit_terminal_report_payload",
]

FitEstimateKind = Literal[
    "shared_parameter",
    "per_dataset_parameter",
    "centred_overall",
    "centred_deviation",
    "initial",
]
FitBoundState = Literal["lower", "upper", "interior"]


class FitTerminalReportError(RuntimeError):
    """Semantic terminal report construction failed.

    This is deliberately separate from physical plot failures.  The Fit owner
    can retain the numerical candidate for diagnosis while disabling Apply
    until a report can be built, without mislabelling the failure as plotting.
    """




def _required(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank.")
    return text


def _finite(value: object, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite.")
    return number


def _fit_coordinate_display(value: object) -> str:
    """Cross the report's visible-name boundary without parsing identities.

    Centred overall and deviation coordinates intentionally share the
    underlying parameter name. Scope-qualified labels keep those estimands
    visibly distinct while all policy decisions continue to use the typed
    value itself.
    """

    if isinstance(value, FitCoordinate):
        if value.scope == "centred_overall":
            return f"Overall · {value.name}"
        if value.scope == "centred_deviation":
            return f"Deviation · {value.dataset_id}::{value.name}"
        if value.scope in {"shared", "coupled"}:
            return value.display_name
        return value.display_name
    return str(value)


def _fit_coordinate_machine_identity(value: object) -> object:
    """Serialize typed coordinates explicitly in machine-readable reports."""

    if isinstance(value, FactualCause):
        return _fit_coordinate_machine_identity(value.to_payload())
    if isinstance(value, FitCoordinate):
        return value.to_machine_payload()
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, raw_item in value.items():
            if isinstance(raw_key, FitCoordinate):
                key = raw_key.to_machine_key()
            else:
                key = str(raw_key)
            normalized[key] = _fit_coordinate_machine_identity(raw_item)
        return normalized
    if isinstance(value, tuple):
        return tuple(_fit_coordinate_machine_identity(item) for item in value)
    if isinstance(value, list):
        return [_fit_coordinate_machine_identity(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [
            _fit_coordinate_machine_identity(item)
            for item in sorted(value, key=repr)
        ]
    if isinstance(value, np.ndarray):
        return _fit_coordinate_machine_identity(value.tolist())
    if isinstance(value, np.generic):
        return _fit_coordinate_machine_identity(value.item())
    return value


def _bound(value: object, field: str) -> float:
    """Accept finite or one-sided bounds without fabricating infinity data.

    Fit setup bounds are finite, but typed uncertainty intervals may be
    one-sided.  Preserve those endpoints for semantic reporting and mark
    them explicitly in machine payloads; reject NaN, which has no ordering
    or truthful bound-state meaning.
    """

    number = float(value)
    if math.isnan(number):
        raise ValueError(f"{field} must not be NaN.")
    return number


def _metric_payload(metric: FitMetric) -> dict[str, object]:
    if not isinstance(metric, FitMetric):
        raise TypeError("Fit report metrics must be typed FitMetric values.")
    return metric.to_payload()


def fit_metric_text(metric: FitMetric, *, precision: int = 6) -> str:
    """Format one typed metric without inventing a numeric fallback."""

    if not isinstance(metric, FitMetric):
        raise TypeError("Fit metric presentation requires FitMetric.")
    if metric.available:
        return f"{metric.value:.{int(precision)}g}"
    return f"unavailable ({metric.unavailable_reason})"


def fit_bound_state(value: float, minimum: float, maximum: float) -> FitBoundState:
    """Classify a value using finite endpoint/scale tolerance only.

    Infinite bounds describe an unbounded side, not an endpoint at infinity.
    In particular, ``inf - finite`` must never become the tolerance used to
    classify every finite estimate as bound-limited.
    """

    value = float(value)
    minimum = float(minimum)
    maximum = float(maximum)
    finite_scales = [1.0, abs(value)]
    if math.isfinite(minimum):
        finite_scales.append(abs(minimum))
    if math.isfinite(maximum):
        finite_scales.append(abs(maximum))
    tolerance = max(finite_scales) * 1.0e-9
    if math.isfinite(minimum) and math.isclose(
        value, minimum, rel_tol=1.0e-9, abs_tol=tolerance
    ):
        return "lower"
    if math.isfinite(maximum) and math.isclose(
        value, maximum, rel_tol=1.0e-9, abs_tol=tolerance
    ):
        return "upper"
    return "interior"


_bound_state = fit_bound_state


def _change(
    fitted: float,
    nominal: float,
) -> tuple[float, float | None, str | None]:
    absolute = fitted - nominal
    if nominal == 0.0:
        return absolute, None, "nominal value is zero"
    return absolute, (absolute / abs(nominal)) * 100.0, None


@dataclass(frozen=True, slots=True)
class FitMetricReportRow:
    dataset_id: str
    weighted_sse: FitMetric
    weighted_mse: FitMetric
    weighted_rmse: FitMetric
    weighted_mae: FitMetric
    r_squared_by_series: tuple[tuple[str, FitMetric], ...]
    effective_weight: float
    n_points: int
    dataset_display_name: str = ""
    x_point_count: int = 0
    x_weighted_residual_sse: FitMetric | None = None
    x_weighted_residual_mse: FitMetric | None = None
    x_weighted_rmse: FitMetric | None = None
    x_weighted_mae: FitMetric | None = None
    x_delta_x: tuple[float, ...] = ()
    x_match_kinds: tuple[str, ...] = ()
    x_sample_kinds: tuple[str, ...] = ()
    x_source_rows: tuple[int, ...] = ()
    x_segment_ids: tuple[int, ...] = ()
    x_segment_keys: tuple[str, ...] = ()
    x_discontinuity_ids: tuple[int, ...] = ()
    x_identity_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _required(self.dataset_id, "dataset ID"))
        metrics = (
            self.weighted_sse,
            self.weighted_mse,
            self.weighted_rmse,
            self.weighted_mae,
        )
        if any(not isinstance(item, FitMetric) for item in metrics):
            raise TypeError("Dataset report metrics must be typed FitMetric values.")
        r_squared = tuple(self.r_squared_by_series)
        if any(
            not str(name or "").strip() or not isinstance(metric, FitMetric)
            for name, metric in r_squared
        ):
            raise TypeError("R-squared report entries must be named FitMetric values.")
        if len({name for name, _metric in r_squared}) != len(r_squared):
            raise ValueError("R-squared report entries repeat a series.")
        object.__setattr__(self, "r_squared_by_series", r_squared)
        object.__setattr__(
            self,
            "effective_weight",
            _finite(self.effective_weight, "effective dataset weight"),
        )
        object.__setattr__(self, "n_points", int(self.n_points))
        if self.effective_weight <= 0.0 or self.n_points <= 0:
            raise ValueError("Dataset report weights and point counts must be positive.")
        x_metrics = (
            self.x_weighted_residual_sse,
            self.x_weighted_residual_mse,
            self.x_weighted_rmse,
            self.x_weighted_mae,
        )
        if any(item is not None and not isinstance(item, FitMetric) for item in x_metrics):
            raise TypeError("Dataset X report metrics must be typed FitMetric values.")
        x_count = int(self.x_point_count)
        if x_count < 0:
            raise ValueError("Dataset X report point count cannot be negative.")
        object.__setattr__(self, "x_point_count", x_count)
        object.__setattr__(self, "x_delta_x", tuple(float(item) for item in self.x_delta_x))
        object.__setattr__(self, "x_match_kinds", tuple(str(item) for item in self.x_match_kinds))
        object.__setattr__(self, "x_sample_kinds", tuple(str(item) for item in self.x_sample_kinds))
        object.__setattr__(self, "x_source_rows", tuple(int(item) for item in self.x_source_rows))
        object.__setattr__(self, "x_segment_ids", tuple(int(item) for item in self.x_segment_ids))
        object.__setattr__(self, "x_segment_keys", tuple(str(item) for item in self.x_segment_keys))
        object.__setattr__(self, "x_discontinuity_ids", tuple(int(item) for item in self.x_discontinuity_ids))
        if self.x_identity_hash is not None:
            identity = str(self.x_identity_hash).strip()
            if not identity:
                raise ValueError("Dataset X report identity hash cannot be blank.")
            object.__setattr__(self, "x_identity_hash", identity)
        object.__setattr__(
            self,
            "dataset_display_name",
            str(self.dataset_display_name or self.dataset_id),
        )


@dataclass(frozen=True, slots=True)
class FitEstimateReportRow:
    kind: FitEstimateKind
    name: str
    display_name: str
    scope: str
    unit: str
    captured_value: float
    nominal_value: float
    fitted_value: float
    minimum: float
    maximum: float
    bound_state: FitBoundState
    absolute_change: float
    percentage_change: float | None
    percentage_unavailable_reason: str | None
    valid: bool
    validity_diagnostic: str
    dataset_id: str | None = None
    set_id: str | None = None
    dataset_ids: tuple[str, ...] = ()
    log_scale: bool = False
    dataset_display_name: str | None = None
    set_display_name: str | None = None
    captured_value_source: str = "project origin default"
    nominal_value_source: str = "Fit setup"

    def __post_init__(self) -> None:
        if self.kind not in {
            "shared_parameter",
            "per_dataset_parameter",
            "centred_overall",
            "centred_deviation",
            "initial",
        }:
            raise ValueError("Unknown Fit estimate report kind.")
        object.__setattr__(self, "name", _required(self.name, "estimate name"))
        object.__setattr__(self, "display_name", _required(self.display_name, "estimate display name"))
        object.__setattr__(self, "scope", _required(self.scope, "estimate scope"))
        object.__setattr__(self, "unit", _required(self.unit, "estimate unit"))
        for field in (
            "captured_value",
            "nominal_value",
            "fitted_value",
            "absolute_change",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        minimum = _bound(self.minimum, "minimum")
        maximum = _bound(self.maximum, "maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        if minimum >= maximum or not minimum <= self.fitted_value <= maximum:
            raise ValueError("Fit estimate value must remain inside its terminal bounds.")
        if self.bound_state not in {"lower", "upper", "interior"}:
            raise ValueError("Unknown Fit estimate bound state.")
        if self.percentage_change is not None:
            object.__setattr__(self, "percentage_change", _finite(self.percentage_change, "percentage change"))
            if self.percentage_unavailable_reason is not None:
                raise ValueError("Available percentage change cannot have a reason.")
        elif not str(self.percentage_unavailable_reason or "").strip():
            raise ValueError("Unavailable percentage change requires a reason.")
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "validity_diagnostic", _required(self.validity_diagnostic, "validity diagnostic"))
        dataset_ids = tuple(_required(item, "dataset ID") for item in self.dataset_ids)
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("Fit estimate repeats a dataset identity.")
        object.__setattr__(self, "dataset_ids", dataset_ids)
        object.__setattr__(self, "log_scale", bool(self.log_scale))
        if self.dataset_display_name is not None:
            object.__setattr__(
                self,
                "dataset_display_name",
                _required(self.dataset_display_name, "dataset display name"),
            )
        if self.set_display_name is not None:
            object.__setattr__(
                self,
                "set_display_name",
                _required(self.set_display_name, "set display name"),
            )
        object.__setattr__(
            self,
            "captured_value_source",
            _required(self.captured_value_source, "captured value source"),
        )
        object.__setattr__(
            self,
            "nominal_value_source",
            _required(self.nominal_value_source, "nominal value source"),
        )


@dataclass(frozen=True, slots=True)
class FitAggregateReportRow:
    name: str
    display_name: str
    unit: str
    n: int
    arithmetic_mean: float
    sample_sd: float | None
    standard_error: float | None
    median: float
    iqr: float
    minimum: float
    maximum: float
    geometric_mean: float | None
    geometric_spread: float | None
    geometric_unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class FitPublicationReport:
    shared_parameter_names: tuple[str, ...]
    fit_local_parameter_names: tuple[str, ...]
    classifications: tuple[FitPublicationClassification, ...] = ()
    unavailable_reasons: tuple[tuple[str, str], ...] = ()
    display_names: Mapping[str, str] = MappingProxyType({})
    terminal_permission: bool = True

    def __post_init__(self) -> None:
        classifications = tuple(self.classifications)
        if any(not isinstance(item, FitPublicationClassification) for item in classifications):
            raise TypeError("Fit publication report requires typed classifications.")
        names = tuple(item.parameter_name for item in classifications)
        if len(names) != len(set(names)):
            raise ValueError("Fit publication classifications repeat a scalar.")
        object.__setattr__(self, "shared_parameter_names", tuple(str(item) for item in self.shared_parameter_names))
        object.__setattr__(self, "fit_local_parameter_names", tuple(str(item) for item in self.fit_local_parameter_names))
        object.__setattr__(self, "classifications", classifications)
        object.__setattr__(self, "unavailable_reasons", tuple((str(name), str(reason)) for name, reason in self.unavailable_reasons))
        object.__setattr__(
            self,
            "display_names",
            MappingProxyType({str(name): str(value) for name, value in self.display_names.items()}),
        )
        object.__setattr__(self, "terminal_permission", bool(self.terminal_permission))

    @property
    def message(self) -> str:
        parts: list[str] = []
        if not self.terminal_permission:
            parts.append(
                "Terminal publication permission is unavailable because the "
                "Fit did not converge; estimates are diagnostic only."
            )
        if self.shared_parameter_names:
            parts.append(
                "Overall/Shared values eligible for publication: "
                + ", ".join(self.shared_parameter_names)
                + "."
            )
        else:
            parts.append("No fitted Shared project values are eligible for publication.")
        if self.fit_local_parameter_names:
            parts.append(
                "Non-publishable Fit values remain Fit-only: "
                + ", ".join(self.fit_local_parameter_names)
                + "."
            )
        observable_names = tuple(
            item.parameter_name
            for item in self.classifications
            if item.eligible and item.route == "observable_definitions"
        )
        if observable_names:
            parts.append(
                "Fit-local scalar values eligible via Observable Definitions: "
                + ", ".join(observable_names)
                + "."
            )
        if self.unavailable_reasons:
            parts.append(
                "Publication unavailable: "
                + "; ".join(
                    f"{name}: {reason}" for name, reason in self.unavailable_reasons
                )
                + "."
            )
        routed = tuple(
            f"{self.display_names.get(item.parameter_name, item.parameter_name)} → "
            f"{item.route.replace('_', ' ')}"
            for item in self.classifications
            if item.eligible
        )
        if routed:
            parts.append("Publication routes: " + ", ".join(routed) + ".")
        return " ".join(parts)


_PROFILE_INTERVAL_STATUSES = frozenset(
    {"available", "partial", "unavailable", "stopped", "failed"}
)
_PROFILE_RESOLUTION_SCALARS = frozenset(
    {
        "status",
        "reason",
        "evaluation_index",
        "last_evaluation_index",
        "optimizer_value",
        "fixed_optimizer_value",
        "fixed_value",
        "residual",
        "residual_sse",
        "weighted_residual_sse",
        "likelihood_ratio",
        "likelihood_ratio_raw",
        "likelihood_ratio_log",
        "likelihood_ratio_outcome",
        "threshold",
        "coordinate_tolerance",
        "likelihood_ratio_tolerance",
        "bracket_width",
        "width",
    }
)
_PROFILE_RESOLUTION_NESTED = frozenset(
    {
        "bracket",
        "bounds",
        "cause",
        "lower",
        "upper",
        "residual",
        "tolerances",
        "threshold",
    }
)


def _freeze_profile_resolution(value: object) -> object:
    """Freeze compact Profile evidence without retaining mutable trace data."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_profile_resolution(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_profile_resolution(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(
            _freeze_profile_resolution(item)
            for item in sorted(value, key=repr)
        )
    if isinstance(value, np.generic):
        return _freeze_profile_resolution(value.item())
    return value


def _profile_resolution_value(value: object) -> object:
    """Retain scalar/bracket evidence while excluding traces and nuisance maps."""

    if isinstance(value, Mapping):
        compact: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if key in _PROFILE_RESOLUTION_SCALARS or key in _PROFILE_RESOLUTION_NESTED:
                if isinstance(raw_value, Mapping):
                    if key == "cause":
                        nested = {
                            str(cause_key): str(cause_value)
                            for cause_key, cause_value in raw_value.items()
                            if str(cause_key)
                            in {"domain", "origin", "code", "type", "message"}
                            and cause_value is not None
                        }
                    else:
                        nested = _profile_resolution_value(raw_value)
                    if isinstance(nested, Mapping) and nested:
                        compact[key] = nested
                elif isinstance(raw_value, (tuple, list)):
                    compact[key] = tuple(
                        _profile_resolution_value(item) for item in raw_value
                    )
                elif raw_value is not None:
                    compact[key] = _fit_coordinate_machine_identity(raw_value)
            elif key == "messages" and isinstance(raw_value, Mapping):
                messages = {
                    str(message_key): str(message_value)
                    for message_key, message_value in sorted(
                        raw_value.items(), key=lambda entry: str(entry[0])
                    )
                    if message_value is not None
                }
                if messages:
                    compact[key] = messages
        return compact
    if isinstance(value, (tuple, list)):
        return tuple(_profile_resolution_value(item) for item in value)
    return _fit_coordinate_machine_identity(value)


def _profile_compact_resolution(
    interval: Mapping[str, object] | None,
    *,
    extension: FitTerminalExtension,
) -> Mapping[str, object]:
    """Project only report-sized Profile resolution evidence."""

    raw_resolution = (
        interval.get("resolution")
        if isinstance(interval, Mapping)
        else None
    )
    compact = (
        _profile_resolution_value(raw_resolution)
        if isinstance(raw_resolution, Mapping)
        else {}
    )
    if not isinstance(compact, Mapping):
        compact = {}
    result = dict(compact)
    if isinstance(interval, Mapping):
        reason = interval.get("reason")
        if reason is not None and "reason" not in result:
            result["reason"] = str(reason)
    if not result:
        result["status"] = extension.status
        if extension.unavailable_reason:
            result["reason"] = extension.unavailable_reason
        cause = extension.cause
        if cause is not None:
            cause_payload = cause.to_payload()
            result["cause"] = {
                key: str(cause_payload[key])
                for key in ("domain", "origin", "code", "type", "message")
                if cause_payload.get(key) is not None
            }
    return MappingProxyType(
        {
            str(key): _freeze_profile_resolution(value)
            for key, value in sorted(result.items(), key=lambda entry: str(entry[0]))
        }
    )


def _profile_row_sort_key(row: "FitProfileIntervalReportRow") -> str:
    return row.coordinate.to_machine_key()


@dataclass(frozen=True, slots=True)
class FitProfileIntervalReportRow:
    """Immutable report-owned projection of one Profile interval."""

    coordinate: FitCoordinate
    display_name: str
    estimate: float | None
    confidence: float | None
    lower: float | None
    upper: float | None
    status: str
    interval_kind: str
    lower_outcome: str | None
    upper_outcome: str | None
    resolution: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate, FitCoordinate):
            raise TypeError("Profile report coordinates require FitCoordinate values.")
        display_name = _required(self.display_name, "Profile coordinate display name")
        if display_name != _fit_coordinate_display(self.coordinate):
            raise ValueError(
                "Profile report coordinate display name must match its typed identity."
            )
        status = _required(self.status, "Profile interval status")
        if status not in _PROFILE_INTERVAL_STATUSES:
            raise ValueError(f"Unsupported Profile interval status: {status!r}")
        interval_kind = _required(self.interval_kind, "Profile interval kind")
        for field in ("estimate", "lower", "upper"):
            value = getattr(self, field)
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Profile interval {field} must be finite or None.")
            object.__setattr__(self, field, number)
        if self.confidence is not None:
            confidence = float(self.confidence)
            if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
                raise ValueError("Profile interval confidence must be in (0, 1).")
            object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "interval_kind", interval_kind)
        object.__setattr__(
            self,
            "lower_outcome",
            None if self.lower_outcome is None else str(self.lower_outcome),
        )
        object.__setattr__(
            self,
            "upper_outcome",
            None if self.upper_outcome is None else str(self.upper_outcome),
        )
        if not isinstance(self.resolution, Mapping):
            raise TypeError("Profile interval resolution must be a mapping.")
        compact_resolution = _profile_resolution_value(self.resolution)
        if not isinstance(compact_resolution, Mapping):
            raise TypeError("Profile interval resolution must be a mapping.")
        frozen_resolution = _freeze_profile_resolution(compact_resolution)
        if not isinstance(frozen_resolution, Mapping):
            raise TypeError("Profile interval resolution must be a mapping.")
        object.__setattr__(
            self,
            "resolution",
            MappingProxyType(dict(frozen_resolution)),
        )


@dataclass(frozen=True, slots=True)
class FitTerminalReport:
    status: str
    joint_optimizer_converged: bool
    joint_optimizer_diagnostic: str
    weighted_sse: FitMetric
    weighted_mse: FitMetric
    metrics_by_dataset: tuple[FitMetricReportRow, ...]
    shared_estimates: tuple[FitEstimateReportRow, ...]
    per_dataset_estimates: tuple[FitEstimateReportRow, ...]
    initial_estimates: tuple[FitEstimateReportRow, ...]
    aggregates: tuple[FitAggregateReportRow, ...]
    publication: FitPublicationReport
    validity_diagnostics: tuple[str, ...]
    terminal_extensions: tuple[FitTerminalExtension, ...]
    shared_refit_provenance: Mapping[str, object]
    structured_diagnostics: tuple[Mapping[str, object], ...] = ()
    message: str = ""
    scipy_nfev: int = 0
    terminal_replay_count: int = 0
    objective_call_count: int | None = None
    dataset_count: int = 0
    series_count: int = 0
    point_count: int = 0
    parameter_count: int = 0
    degrees_of_freedom: int = 0
    primary_model: str = "ordinary"
    primary_model_details: Mapping[str, object] = MappingProxyType({})
    centred_fixed_effects_settlements: tuple[CentredFixedEffectsSettlement, ...] = ()
    known_input_context: Mapping[str, object] = MappingProxyType({})
    weighting_provenance: Mapping[str, object] = MappingProxyType({})
    profile_intervals: tuple[FitProfileIntervalReportRow, ...] = ()

    @property
    def terminal_permission(self) -> bool:
        """Return the one terminal publication permission for this report."""

        return self.publication.terminal_permission

    def __post_init__(self) -> None:
        object.__setattr__(self, "message", str(self.message))
        for field in (
            "scipy_nfev",
            "terminal_replay_count",
            "dataset_count",
            "series_count",
            "point_count",
            "parameter_count",
        ):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError(f"Fit terminal report {field} must be nonnegative.")
            object.__setattr__(self, field, value)
        if self.objective_call_count is not None:
            objective_calls = int(self.objective_call_count)
            if objective_calls < 0:
                raise ValueError(
                    "Fit terminal report objective_call_count must be nonnegative."
                )
            object.__setattr__(self, "objective_call_count", objective_calls)
        object.__setattr__(self, "degrees_of_freedom", int(self.degrees_of_freedom))
        primary_model = str(self.primary_model or "").strip()
        if primary_model not in {"ordinary", "centred_fixed_effects"}:
            raise ValueError(f"Unsupported primary Fit model: {primary_model!r}")
        details = MappingProxyType(dict(self.primary_model_details or {}))
        object.__setattr__(self, "primary_model", primary_model)
        object.__setattr__(self, "primary_model_details", details)
        settlements = tuple(self.centred_fixed_effects_settlements or ())
        if any(
            not isinstance(item, CentredFixedEffectsSettlement)
            for item in settlements
        ):
            raise TypeError("Centred report settlements require typed DTOs.")
        object.__setattr__(self, "centred_fixed_effects_settlements", settlements)
        object.__setattr__(
            self,
            "known_input_context",
            MappingProxyType(dict(self.known_input_context or {})),
        )
        weighting_provenance = self.weighting_provenance or {}
        if not isinstance(weighting_provenance, Mapping):
            raise TypeError("Fit weighting provenance must be a mapping.")
        object.__setattr__(
            self,
            "weighting_provenance",
            MappingProxyType(dict(weighting_provenance)),
        )
        diagnostics = tuple(self.structured_diagnostics)
        if any(not isinstance(item, Mapping) for item in diagnostics):
            raise TypeError("Fit terminal diagnostics must be mappings.")
        object.__setattr__(
            self,
            "structured_diagnostics",
            tuple(MappingProxyType(dict(item)) for item in diagnostics),
        )
        profile_intervals = tuple(self.profile_intervals or ())
        if any(
            not isinstance(item, FitProfileIntervalReportRow)
            for item in profile_intervals
        ):
            raise TypeError("Fit Profile report rows require typed projections.")
        ordered_profile_intervals = tuple(
            sorted(profile_intervals, key=_profile_row_sort_key)
        )
        identities = tuple(
            item.coordinate.to_machine_key() for item in ordered_profile_intervals
        )
        if len(identities) != len(set(identities)):
            raise ValueError("Fit Profile report rows repeat a coordinate identity.")
        object.__setattr__(self, "profile_intervals", ordered_profile_intervals)


def _display_value(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, FitCoordinate):
        # Typed coordinates retain their scope-qualified identity at the
        # visible-name boundary; machine payloads retain the full identity.
        return _fit_coordinate_display(value)
    if isinstance(value, np.ndarray):
        return _display_value(value.tolist())
    if isinstance(value, np.generic):
        return _display_value(value.item())
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_display_value(item) for item in value) + "]"
    return str(value)


def _extension_failure_lines(extension: FitTerminalExtension) -> list[str]:
    """Expose owner-supplied typed failure causes without flattening them."""

    lines: list[str] = []
    cause = extension.cause
    if cause is not None:
        # ``FitTerminalExtension.cause`` is the canonical optional-analysis
        # fact.  Render its complete chain here so two optional analyses with
        # different causes remain visibly distinct even when their details use
        # the same generic failure keys.
        current: object | None = cause
        label = "cause"
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if not isinstance(current, FactualCause):
                break
            payload = current.to_payload()
            lines.append(
                f"  {label}: domain={_display_value(payload.get('domain'))}; "
                f"origin={_display_value(payload.get('origin'))}; "
                f"code={_display_value(payload.get('code'))}; "
                f"type={_display_value(payload.get('type'))}; "
                f"message={_display_value(payload.get('message'))}"
            )
            for field in ("context", "details", "suggestion", "examples"):
                value = payload.get(field)
                if value:
                    lines.append(f"  {label}.{field}: {_display_value(value)}")
            current = current.cause
            label += ".cause"
    return lines


def fit_terminal_extension_lines(
    extension: FitTerminalExtension,
) -> tuple[str, ...]:
    """Render named terminal DTO fields without calculating missing science."""

    if not isinstance(extension, FitTerminalExtension):
        raise TypeError("Terminal method presentation requires FitTerminalExtension.")
    lines = [f"{extension.kind}: {extension.method}; status={extension.status}"]
    if extension.unavailable_reason:
        lines.append(f"  unavailable reason: {extension.unavailable_reason}")
    lines.extend(_extension_failure_lines(extension))
    for field, label in (
        ("confidence_level", "confidence level"),
        ("likelihood_model", "likelihood model"),
        ("likelihood_ratio_threshold", "likelihood-ratio threshold"),
        ("fixed_coordinate_policy", "profile policy"),
        ("typed_coordinate_roster", "typed coordinate roster"),
        ("reoptimization", "reoptimized roster"),
        ("optimizer_method", "optimizer"),
        ("max_bracketing_steps_per_side", "maximum bracketing steps per side"),
        ("max_nfev_per_point", "evaluation budget per point"),
        ("observation_count", "observations"),
        ("terminal_weighted_residual_sse", "terminal weighted SSE"),
        ("seed", "deterministic seed"),
        ("replicates", "configured full refits"),
        ("draw_law", "actual draw law"),
        ("observation_error", "observation error law"),
        ("weight_multiplier", "normalized weight formula"),
        ("minimum_successful_replicates", "minimum successful refits"),
        ("minimum_success_fraction", "minimum success fraction"),
        ("successful_replicates", "successful refits"),
        ("failed_replicates", "failed refits"),
        ("required_successes", "required successes"),
        ("interval_construction", "interval construction"),
    ):
        if field in extension.provenance:
            lines.append(f"  {label}: {_display_value(extension.provenance[field])}")
    if extension.kind == "profile_likelihood":
        intervals = extension.details.get("intervals", {})
        if isinstance(intervals, Mapping):
            rows = []
            for coordinate, raw_interval in intervals.items():
                if not isinstance(coordinate, FitCoordinate):
                    continue
                interval = raw_interval if isinstance(raw_interval, Mapping) else {}
                rows.append(
                    f"{_fit_coordinate_display(coordinate)}="
                    f"{_display_value(interval.get('status'))}"
                )
            if rows:
                lines.append("  interval statuses: " + "; ".join(sorted(rows)))
    else:
        intervals = extension.details.get("intervals", {})
        if isinstance(intervals, Mapping):
            for coordinate, raw_interval in intervals.items():
                interval = raw_interval if isinstance(raw_interval, Mapping) else {}
                lines.append(
                    f"  {_fit_coordinate_display(coordinate)}: "
                    f"status={_display_value(interval.get('status'))}; "
                    f"lower={_display_value(interval.get('lower'))}; "
                    f"upper={_display_value(interval.get('upper'))}; "
                    f"reason={_display_value(interval.get('reason'))}"
                )
    if "requested_parameters" in extension.details:
        lines.append(
            "  requested coordinates: "
            + _display_value(extension.details["requested_parameters"])
        )
    return tuple(lines)


def fit_primary_model_lines(
    primary_model: str,
    primary_model_details: Mapping[str, object],
) -> tuple[str, ...]:
    """Render the selected ordinary or centred primary settlement."""

    model = str(primary_model or "").strip()
    if model not in {"ordinary", "centred_fixed_effects"}:
        raise ValueError(f"Unsupported primary Fit model: {model!r}")
    if not isinstance(primary_model_details, Mapping):
        raise TypeError("Primary Fit model details must be a mapping.")
    return (f"Primary Fit model: {model}",)


def _percentile(sorted_values: tuple[float, ...], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("A percentile requires values.")
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _aggregate(
    *,
    name: str,
    display_name: str,
    unit: str,
    values: tuple[float, ...],
    log_scale: bool,
) -> FitAggregateReportRow:
    ordered = tuple(sorted(values))
    count = len(ordered)
    mean = statistics.fmean(ordered)
    sample_sd = statistics.stdev(ordered) if count > 1 else None
    standard_error = sample_sd / math.sqrt(count) if sample_sd is not None else None
    positive_log = log_scale and all(value > 0.0 for value in ordered)
    geometric_mean = None
    geometric_spread = None
    reason: str | None = "requires positive log-scale estimates"
    if positive_log:
        log_values = tuple(math.log(value) for value in ordered)
        geometric_mean = math.exp(statistics.fmean(log_values))
        geometric_spread = (
            math.exp(statistics.stdev(log_values)) if count > 1 else None
        )
        reason = None if count > 1 else "geometric spread requires at least two estimates"
    return FitAggregateReportRow(
        name=name,
        display_name=display_name,
        unit=unit,
        n=count,
        arithmetic_mean=mean,
        sample_sd=sample_sd,
        standard_error=standard_error,
        median=statistics.median(ordered),
        iqr=_percentile(ordered, 0.75) - _percentile(ordered, 0.25),
        minimum=ordered[0],
        maximum=ordered[-1],
        geometric_mean=geometric_mean,
        geometric_spread=geometric_spread,
        geometric_unavailable_reason=reason,
    )


def _publication_classifications(
    candidate: FitCandidate,
) -> tuple[FitPublicationClassification, ...]:
    """Normalize publication-owner decisions without deriving policy."""

    parameter_by_name = {item.name: item for item in candidate.setup_snapshot.parameters}
    values = tuple(candidate.setup_snapshot.publication_classifications)
    if any(not isinstance(item, FitPublicationClassification) for item in values):
        raise FitTerminalReportError(
            "publication owner must supply FitPublicationClassification values"
        )
    by_name = {item.parameter_name: item for item in values}
    expected = set(parameter_by_name)
    if set(by_name) != expected:
        missing = sorted(expected - set(by_name))
        unexpected = sorted(set(by_name) - expected)
        raise FitTerminalReportError(
            "publication classification roster does not match terminal Fit "
            f"scalars (missing={missing!r}, unexpected={unexpected!r})"
        )
    # Keep the complete terminal roster, including fixed scalars. Estimate
    # tables remain fitted-only projections; classifications are the owner's
    # complete route truth for Summary, Apply, and machine output.
    return tuple(by_name[item.name] for item in candidate.setup_snapshot.parameters)


def _parameter_rows(
    candidate: FitCandidate,
) -> tuple[tuple[FitEstimateReportRow, ...], tuple[FitEstimateReportRow, ...]]:
    nominal_parameters = {
        item.name: item for item in candidate.nominal_setup_snapshot.parameters
    }
    captured_parameters = {
        item.name: item for item in candidate.origin.parameter_defaults
    }
    fitted_shared = dict(candidate.parameters)
    fitted_by_dataset = {
        dataset_id: dict(values)
        for dataset_id, values in candidate.scalar_values_by_dataset
    }
    included_ids = {
        item.dataset_id
        for item in candidate.setup_snapshot.datasets
        if item.included
    }
    dataset_display_names = {
        mapped.dataset_id: mapped.display_name
        for mapped in candidate.origin.mapped_datasets
    }
    shared: list[FitEstimateReportRow] = []
    individual: list[FitEstimateReportRow] = []
    centred_settlements = {
        item.parameter_name: item
        for item in candidate.result.centred_fixed_effects_settlements
    }
    if len(centred_settlements) != len(
        candidate.result.centred_fixed_effects_settlements
    ):
        raise FitTerminalReportError("Centred terminal settlement repeats a parameter.")
    refit_provenance = candidate.result.shared_refit_provenance
    promotion_provenance: Mapping[str, object] = {}
    shared_refit_terminal_route: str | None = None
    if refit_provenance:
        raw_promotions = refit_provenance.get("promotions")
        if (
            refit_provenance.get("execution")
            != "Promote to Shared and Refit"
            or refit_provenance.get("terminal_route")
            not in {
                "optimizer_terminal",
                "user_stop_best_valid_bounded_replay",
            }
            or refit_provenance.get("starting_value_authority")
            != "shared_refit_execution"
            or refit_provenance.get("terminal_value_authority")
            != "result.shared_params"
            or not isinstance(raw_promotions, Mapping)
            or not raw_promotions
        ):
            raise FitTerminalReportError(
                "Shared-refit terminal provenance is incomplete."
            )
        promotion_provenance = raw_promotions
        shared_refit_terminal_route = str(refit_provenance["terminal_route"])
    local_scalar_names = {
        scalar_name
        for observable in candidate.local_observables
        for scalar_name in observable.scalar_names
    }
    terminal_converged = bool(candidate.result.completion.optimizer_converged)
    terminal_validity_suffix = (
        ""
        if terminal_converged
        else "; optimizer did not converge; estimates are diagnostic only"
    )
    for terminal in candidate.setup_snapshot.parameters:
        nominal = nominal_parameters[terminal.name]
        captured = captured_parameters.get(terminal.name)
        fit_local_provenance = terminal.name in local_scalar_names
        if captured is None:
            if not fit_local_provenance:
                raise FitTerminalReportError(
                    "terminal Fit scalar is absent from captured project defaults: "
                    + terminal.name
                )
            captured_value = nominal.selected_config.value
            captured_value_source = "Fit-local starting value (not a project default)"
        else:
            captured_value = captured.shared_config.value
            captured_value_source = "captured project Shared default"
        nominal_configuration = nominal.selected_config
        # Fixed values are terminal configuration truth too.  They may be
        # absent from the result's fitted coordinate maps, so the request's
        # executed configuration is the source for those rows.  This keeps
        # Shared, Per-dataset, and Fit-local fixed values visible uniformly.
        value_is_configuration_only = not nominal_configuration.fit
        if terminal.scope == "centred_fixed_effects":
            settlement = centred_settlements.get(terminal.name)
            if settlement is None:
                raise FitTerminalReportError(
                    f"canonical result omitted centred settlement {terminal.name!r}"
                )
            configuration = terminal.shared_config
            overall = _finite(
                settlement.overall_decoded_value,
                f"{terminal.name} overall",
            )
            nominal_value = configuration.value
            absolute, percentage, unavailable = _change(overall, nominal_value)
            shared.append(
                FitEstimateReportRow(
                    kind="centred_overall",
                    name=terminal.name,
                    display_name=f"Overall · {terminal.display_name}",
                    scope="Overall",
                    unit=terminal.unit,
                    captured_value=(
                        captured.shared_config.value
                        if captured is not None
                        else nominal_value
                    ),
                    nominal_value=nominal_value,
                    fitted_value=overall,
                    minimum=configuration.minimum,
                    maximum=configuration.maximum,
                    bound_state=_bound_state(
                        overall,
                        configuration.minimum,
                        configuration.maximum,
                    ),
                    absolute_change=absolute,
                    percentage_change=percentage,
                    percentage_unavailable_reason=unavailable,
                    valid=terminal_converged,
                    validity_diagnostic=(
                        "decoded centred overall value is finite and within "
                        "configured bounds"
                        + terminal_validity_suffix
                    ),
                    log_scale=configuration.log10,
                    captured_value_source="captured project Shared default",
                    nominal_value_source="Fit setup",
                )
            )
            deviation_bounds = (
                math.log10(configuration.maximum)
                - math.log10(configuration.minimum)
                if configuration.log10
                else configuration.maximum - configuration.minimum
            )
            for deviation in settlement.deviations:
                value = _finite(
                    deviation.decoded_value,
                    f"{terminal.name} deviation",
                )
                dataset_id = str(deviation.coordinate.dataset_id)
                absolute, percentage, unavailable = _change(value, 0.0)
                individual.append(
                    FitEstimateReportRow(
                        kind="centred_deviation",
                        name=terminal.name,
                        display_name=(
                            f"Deviation · {dataset_id}::{terminal.display_name}"
                        ),
                        scope="Centred dataset deviation (Fit-only)",
                        unit=("decades" if configuration.log10 else terminal.unit),
                        captured_value=0.0,
                        nominal_value=0.0,
                        fitted_value=value,
                        minimum=-deviation_bounds,
                        maximum=deviation_bounds,
                        bound_state=_bound_state(
                            value,
                            -deviation_bounds,
                            deviation_bounds,
                        ),
                        absolute_change=absolute,
                        percentage_change=percentage,
                        percentage_unavailable_reason=unavailable,
                        valid=terminal_converged,
                        validity_diagnostic=(
                            "optimizer-scale centred deviation; derived reference "
                            "included and no deviation is publishable"
                            + terminal_validity_suffix
                        ),
                        dataset_id=dataset_id,
                        dataset_ids=(dataset_id,),
                        log_scale=configuration.log10,
                        dataset_display_name=dataset_display_names.get(dataset_id),
                        captured_value_source="centred terminal settlement",
                        nominal_value_source="centred zero-deviation seed",
                    )
                )
            continue
        if terminal.scope == "shared":
            configuration = terminal.selected_config
            if nominal_configuration.fit:
                if terminal.name not in fitted_shared:
                    raise FitTerminalReportError(
                        f"canonical result omitted fitted Shared coordinate {terminal.name!r}"
                    )
                raw_fitted = fitted_shared[terminal.name]
            else:
                raw_fitted = nominal_configuration.value
            fitted = _finite(
                raw_fitted,
                terminal.name,
            )
            promotion = promotion_provenance.get(terminal.name)
            if promotion is not None:
                if (
                    not isinstance(promotion, Mapping)
                    or "seed_value" not in promotion
                ):
                    raise FitTerminalReportError(
                        f"Shared-refit seed provenance is invalid for {terminal.name!r}."
                    )
                nominal_value = _finite(
                    promotion["seed_value"],
                    f"{terminal.name} Shared-refit seed",
                )
                validity_diagnostic = (
                    "finite and within configured bounds; terminal Shared-refit "
                    "result is numerical authority and the admitted "
                    "central-tendency seed is provenance only"
                )
                if (
                    shared_refit_terminal_route
                    == "user_stop_best_valid_bounded_replay"
                ):
                    validity_diagnostic += (
                        "; published by user Stop from one best-valid bounded replay"
                    )
            else:
                nominal_value = nominal_configuration.value
                validity_diagnostic = (
                    "fixed configuration value; finite and within configured bounds"
                    if value_is_configuration_only
                    else "finite and within configured bounds"
                )
            if fit_local_provenance:
                captured_value_source = "Fit-local starting value (not a project default)"
                validity_diagnostic += "; Fit-local supplemental scalar provenance"
            if not value_is_configuration_only:
                validity_diagnostic += terminal_validity_suffix
            absolute, percentage, unavailable = _change(fitted, nominal_value)
            shared.append(
                FitEstimateReportRow(
                    kind="shared_parameter",
                    name=terminal.name,
                    display_name=terminal.display_name,
                    scope=("Shared Fit-local scalar" if fit_local_provenance else "Shared"),
                    unit=terminal.unit,
                    captured_value=captured_value,
                    nominal_value=nominal_value,
                    fitted_value=fitted,
                    minimum=configuration.minimum,
                    maximum=configuration.maximum,
                    bound_state=_bound_state(fitted, configuration.minimum, configuration.maximum),
                    absolute_change=absolute,
                    percentage_change=percentage,
                    percentage_unavailable_reason=unavailable,
                    valid=terminal_converged or value_is_configuration_only,
                    validity_diagnostic=validity_diagnostic,
                    log_scale=configuration.log10,
                    captured_value_source=captured_value_source,
                    nominal_value_source=(
                        "admitted Shared-refit central-tendency seed"
                        if promotion is not None
                        else "Fit-local setup admission"
                        if fit_local_provenance
                        else "Fit setup"
                    ),
                )
            )
            continue
        configuration = nominal_configuration
        nominal_values = dict(configuration.dataset_values)
        captured_values = (
            dict(captured.per_dataset_config.dataset_values)
            if captured is not None
            else dict(nominal.per_dataset_config.dataset_values)
        )
        for dataset_id in sorted(included_ids):
            dataset_result = fitted_by_dataset.get(dataset_id, {})
            if nominal_configuration.fit and terminal.name not in dataset_result:
                raise FitTerminalReportError(
                    f"canonical result omitted {terminal.name!r} for dataset {dataset_id!r}"
                )
            fitted = _finite(
                dataset_result[terminal.name]
                if nominal_configuration.fit
                else nominal_values[dataset_id],
                terminal.name,
            )
            nominal_value = nominal_values[dataset_id]
            captured_value = captured_values.get(
                dataset_id,
                nominal.per_dataset_config.value,
            )
            captured_value_source = (
                "Fit-local starting value (not a project default)"
                if fit_local_provenance
                else "captured project Per-dataset default"
            )
            absolute, percentage, unavailable = _change(fitted, nominal_value)
            individual.append(
                FitEstimateReportRow(
                    kind="per_dataset_parameter",
                    name=terminal.name,
                    display_name=terminal.display_name,
                    scope=(
                        "Per-dataset Fit-local scalar"
                        if fit_local_provenance
                        else "Per-dataset"
                    ),
                    unit=terminal.unit,
                    captured_value=captured_value,
                    nominal_value=nominal_value,
                    fitted_value=fitted,
                    minimum=configuration.minimum,
                    maximum=configuration.maximum,
                    bound_state=_bound_state(fitted, configuration.minimum, configuration.maximum),
                    absolute_change=absolute,
                    percentage_change=percentage,
                    percentage_unavailable_reason=unavailable,
                    valid=terminal_converged or value_is_configuration_only,
                    validity_diagnostic=(
                        (
                            "fixed configuration value; finite and within configured bounds"
                            if value_is_configuration_only
                            else "finite and within configured bounds"
                        )
                        + (
                            "; Fit-local supplemental scalar provenance"
                            if fit_local_provenance
                            else ""
                        )
                    ) + (
                        terminal_validity_suffix
                        if not value_is_configuration_only
                        else ""
                    ),
                    dataset_id=dataset_id,
                    dataset_ids=(dataset_id,),
                    log_scale=configuration.log10,
                    dataset_display_name=dataset_display_names.get(dataset_id),
                    captured_value_source=captured_value_source,
                    nominal_value_source=(
                        "Fit-local setup admission"
                        if fit_local_provenance
                        else "Fit setup"
                    ),
                )
            )
    return tuple(shared), tuple(individual)


def _initial_rows(candidate: FitCandidate) -> tuple[FitEstimateReportRow, ...]:
    fitted = {
        (set_id, species): value
        for set_id, values in candidate.initial_values_by_set
        for species, value in values
    }
    captured = {
        (mapped.set_id, species): value
        for mapped in candidate.origin.mapped_datasets
        for species, value in mapped.initial_values
    }
    datasets_by_set: dict[str, tuple[str, ...]] = {}
    set_display_names: dict[str, str] = {}
    for mapped in candidate.origin.mapped_datasets:
        datasets_by_set[mapped.set_id] = (
            *datasets_by_set.get(mapped.set_id, ()),
            mapped.dataset_id,
        )
        set_display_names[mapped.set_id] = mapped.set_name
    scopes = dict(candidate.nominal_setup_snapshot.initial_scopes)
    rows: list[FitEstimateReportRow] = []
    terminal_converged = bool(candidate.result.completion.optimizer_converged)
    terminal_validity_suffix = (
        ""
        if terminal_converged
        else "; optimizer did not converge; estimates are diagnostic only"
    )
    for item in candidate.nominal_setup_snapshot.initials:
        key = (item.set_id, item.species)
        if not item.fit or scopes.get(item.species) != "local" or key not in fitted:
            continue
        value = fitted[key]
        absolute, percentage, unavailable = _change(value, item.initial)
        rows.append(
            FitEstimateReportRow(
                kind="initial",
                name=item.species,
                display_name=item.species,
                scope="Per-dataset IC",
                unit="model concentration",
                captured_value=captured[key],
                nominal_value=item.initial,
                fitted_value=value,
                minimum=item.minimum,
                maximum=item.maximum,
                bound_state=_bound_state(value, item.minimum, item.maximum),
                absolute_change=absolute,
                percentage_change=percentage,
                percentage_unavailable_reason=unavailable,
                valid=terminal_converged,
                validity_diagnostic=(
                    "finite, nonnegative, and within configured bounds"
                    + terminal_validity_suffix
                ),
                set_id=item.set_id,
                dataset_ids=datasets_by_set.get(item.set_id, ()),
                log_scale=item.log10,
                    set_display_name=set_display_names.get(item.set_id, item.set_id),
            )
        )
    global_by_species = {
        item.species: item
        for item in candidate.nominal_setup_snapshot.global_initials
        if item.fit and scopes.get(item.species) == "global"
    }
    for species, item in sorted(global_by_species.items()):
        for key, value in sorted(fitted.items()):
            set_id, fitted_species = key
            if fitted_species != species:
                continue
            absolute, percentage, unavailable = _change(value, item.initial)
            rows.append(
                FitEstimateReportRow(
                    kind="initial",
                    name=species,
                    display_name=species,
                    scope="Shared Fit IC coordinate; independently applied per set",
                    unit="model concentration",
                    captured_value=captured[key],
                    nominal_value=item.initial,
                    fitted_value=value,
                    minimum=item.minimum,
                    maximum=item.maximum,
                    bound_state=_bound_state(value, item.minimum, item.maximum),
                    absolute_change=absolute,
                    percentage_change=percentage,
                    percentage_unavailable_reason=unavailable,
                    valid=terminal_converged,
                    validity_diagnostic=(
                        "finite, nonnegative, and within configured bounds"
                        + terminal_validity_suffix
                    ),
                    set_id=set_id,
                    dataset_ids=datasets_by_set.get(set_id, ()),
                    log_scale=item.log10,
                    set_display_name=set_display_names.get(set_id, set_id),
                )
            )
    return tuple(rows)


def _weighting_provenance(candidate: FitCandidate) -> Mapping[str, object]:
    """Project the terminal residual weighting contract without recalculation.

    Dataset effective weights come from the canonical terminal ``DatasetFitInfo``
    rows. Raw dataset weights come only from the captured setup in user mode;
    implicit mode intentionally does not retain or expose those setup values.
    Target normalization and square-root multipliers mirror the execution
    contract, while the projection itself remains report-owned and immutable.
    """

    if not isinstance(candidate, FitCandidate):
        raise TypeError("Fit weighting provenance requires FitCandidate.")
    mode = str(candidate.request.dataset_weight_mode)
    if mode not in {"implicit", "user"}:
        raise FitTerminalReportError(
            f"Unsupported Fit dataset weight mode: {mode!r}."
        )

    setup_by_id = {
        item.dataset_id: item
        for item in candidate.nominal_setup_snapshot.datasets
        if item.included
    }
    terminal_by_id = {
        str(info.dataset_id): info
        for info in candidate.result.dataset_info
    }
    if set(setup_by_id) != set(terminal_by_id):
        raise FitTerminalReportError(
            "Weighting provenance dataset roster does not match the terminal result."
        )

    datasets: list[Mapping[str, object]] = []
    for dataset_id in sorted(terminal_by_id):
        info = terminal_by_id[dataset_id]
        local = setup_by_id[dataset_id]
        effective_dataset_weight = _finite(
            info.weight,
            f"{dataset_id} effective dataset weight",
        )
        if effective_dataset_weight <= 0.0:
            raise FitTerminalReportError(
                f"{dataset_id} effective dataset weight must be positive."
            )
        dataset_residual_multiplier = math.sqrt(effective_dataset_weight)
        configured_target_weights = {
            str(name): _finite(value, f"{dataset_id}:{name} target weight")
            for name, value in local.target_weights
        }
        normalized_target_weights = finite_positive_mean_normalize(
            configured_target_weights
        )
        targets: list[Mapping[str, object]] = []
        for target in sorted(local.target_species):
            configured_target_weight = configured_target_weights[target]
            normalized_target_weight = normalized_target_weights[target]
            target_residual_multiplier = math.sqrt(normalized_target_weight)
            targets.append(
                MappingProxyType(
                    {
                        "target_key": f"{dataset_id}::{target}",
                        "target": target,
                        "configured_target_weight": configured_target_weight,
                        "normalized_target_weight": normalized_target_weight,
                        "target_residual_multiplier": target_residual_multiplier,
                        "combined_residual_multiplier": (
                            dataset_residual_multiplier
                            * target_residual_multiplier
                        ),
                    }
                )
            )
        dataset_payload: dict[str, object] = {
            "dataset_id": dataset_id,
            "point_count": int(info.n_points),
            "normalized_effective_dataset_weight": effective_dataset_weight,
            "dataset_residual_multiplier": dataset_residual_multiplier,
            "targets": tuple(targets),
        }
        if local.parametric is not None:
            x_name = local.parametric.x_name
            x_weight = configured_target_weights.get(x_name)
            if x_weight is None:
                raise FitTerminalReportError(
                    f"{dataset_id} Parametric-X target weight is missing."
                )
            x_source_id = next(
                (
                    raw_id
                    for raw_id, target in local.observation_targets
                    if target == x_name
                ),
                None,
            )
            mapped = next(
                (
                    item
                    for item in candidate.origin.mapped_datasets
                    if item.dataset_id == dataset_id
                ),
                None,
            )
            x_source = (
                None
                if mapped is None or x_source_id is None
                else next(
                    (
                        observation
                        for observation in mapped.observations
                        if observation.raw_observation_id == x_source_id
                    ),
                    None,
                )
            )
            if x_source is None:
                raise FitTerminalReportError(
                    f"{dataset_id} Parametric-X source capture is missing."
                )
            x_rows = tuple(int(row) for row in info.x_source_rows)
            x_by_row = {
                int(row): (float(time), float(value))
                for row, time, value in zip(
                    x_source.source_rows,
                    x_source.t,
                    x_source.y,
                    strict=True,
                )
            }
            selected_x = tuple(x_by_row[row][1] for row in x_rows)
            selected_t = tuple(x_by_row[row][0] for row in x_rows)
            projection = candidate.result.model_x_projection_by_dataset.get(dataset_id)
            if projection is None:
                raise FitTerminalReportError(
                    f"{dataset_id} Parametric-X terminal projection is missing."
                )
            dataset_payload["x"] = MappingProxyType(
                {
                    "name": x_name,
                    "raw_observation_id": x_source.raw_observation_id,
                    "raw_column": x_source.raw_column,
                    "configured_target_weight": x_weight,
                    "normalized_target_weight": normalized_target_weights[x_name],
                    "target_residual_multiplier": math.sqrt(
                        normalized_target_weights[x_name]
                    ),
                    "combined_residual_multiplier": (
                        dataset_residual_multiplier
                        * math.sqrt(normalized_target_weights[x_name])
                    ),
                    "point_count": int(info.x_point_count),
                    "weighted_sse": (
                        None
                        if info.x_weighted_residual_sse is None
                        else info.x_weighted_residual_sse.to_payload()
                    ),
                    "weighted_mse": (
                        None
                        if info.x_weighted_residual_mse is None
                        else info.x_weighted_residual_mse.to_payload()
                    ),
                    "weighted_rmse": (
                        None
                        if info.x_weighted_rmse is None
                        else info.x_weighted_rmse.to_payload()
                    ),
                    "weighted_mae": (
                        None
                        if info.x_weighted_mae is None
                        else info.x_weighted_mae.to_payload()
                    ),
                    "selected_time": selected_t,
                    "selected_values": selected_x,
                    "selected_model_values": tuple(
                        float(value) for value in projection.selected_model_x
                    ),
                    "source_rows": x_rows,
                    "delta_x": tuple(info.x_delta_x),
                    "match_kinds": tuple(info.x_match_kinds),
                    "sample_kinds": tuple(info.x_sample_kinds),
                    "segment_ids": tuple(
                        int(value) for value in projection.segment_ids
                    ),
                    "segment_keys": tuple(projection.segment_keys),
                    "discontinuity_ids": tuple(
                        int(value) for value in projection.discontinuity_ids
                    ),
                    "identity_hash": info.x_identity_hash,
                }
            )
        if mode == "user":
            dataset_payload["original_user_dataset_weight"] = float(local.weight)
        datasets.append(MappingProxyType(dataset_payload))

    return MappingProxyType(
        {
            "dataset_weight_mode": mode,
            "combined_residual_weighting_rule": (
                "weighted_residual = raw_residual * "
                "sqrt(normalized_effective_dataset_weight) * "
                "sqrt(normalized_target_weight)"
            ),
            "datasets": tuple(datasets),
        }
    )


def _profile_optional_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FitTerminalReportError(
            f"Profile interval {field} is not numeric."
        ) from exc
    if not math.isfinite(number):
        raise FitTerminalReportError(
            f"Profile interval {field} must be finite or unavailable."
        )
    return number


def _profile_coordinate_roster(
    candidate: FitCandidate,
    extension: FitTerminalExtension,
    intervals: Mapping[object, object],
) -> tuple[FitCoordinate, ...]:
    """Resolve only the typed core roster; never reconstruct coordinates from text."""

    coordinates: list[FitCoordinate] = []
    for raw_coordinate, raw_interval in intervals.items():
        if not isinstance(raw_coordinate, FitCoordinate):
            raise FitTerminalReportError(
                "Profile interval roster requires typed FitCoordinate keys."
            )
        if not isinstance(raw_interval, Mapping):
            raise FitTerminalReportError(
                "Profile interval evidence must be a mapping."
            )
        embedded = raw_interval.get("coordinate")
        if embedded is not None and embedded != raw_coordinate:
            raise FitTerminalReportError(
                "Profile interval evidence repeats a conflicting coordinate identity."
            )
        coordinates.append(raw_coordinate)
    if not coordinates:
        config = candidate.request.profile_likelihood
        if config is None:
            raise FitTerminalReportError(
                "Profile extension omitted its typed coordinate roster."
            )
        coordinates.extend(
            coordinate
            for coordinate in config.parameter_names
            if isinstance(coordinate, FitCoordinate)
        )
        if len(coordinates) != len(config.parameter_names):
            raise FitTerminalReportError(
                "Profile request roster contains a non-typed coordinate."
            )
    ordered = tuple(sorted(coordinates, key=lambda item: item.to_machine_key()))
    identities = tuple(item.to_machine_key() for item in ordered)
    if len(identities) != len(set(identities)):
        raise FitTerminalReportError("Profile coordinate roster repeats an identity.")
    return ordered


def _profile_interval_rows(
    candidate: FitCandidate,
    extensions: tuple[FitTerminalExtension, ...],
) -> tuple[FitProfileIntervalReportRow, ...]:
    """Build one compact row per core-owned Profile coordinate."""

    profile_extensions = tuple(
        item for item in extensions if item.kind == "profile_likelihood"
    )
    if not profile_extensions:
        return ()
    if len(profile_extensions) != 1:
        raise FitTerminalReportError("Terminal Fit has repeated Profile extensions.")
    extension = profile_extensions[0]
    raw_intervals = extension.details.get("intervals", {})
    if raw_intervals is None:
        raw_intervals = {}
    if not isinstance(raw_intervals, Mapping):
        raise FitTerminalReportError("Profile intervals must be a mapping.")
    intervals_by_identity = {
        coordinate.to_machine_key(): raw_interval
        for coordinate, raw_interval in raw_intervals.items()
        if isinstance(coordinate, FitCoordinate)
    }
    roster = _profile_coordinate_roster(candidate, extension, raw_intervals)
    rows: list[FitProfileIntervalReportRow] = []
    for coordinate in roster:
        raw_interval = intervals_by_identity.get(coordinate.to_machine_key())
        interval = raw_interval if isinstance(raw_interval, Mapping) else None
        if interval is None:
            status = extension.status
            interval_kind = "unavailable"
        else:
            status = str(interval.get("status") or "").strip()
            interval_kind = str(interval.get("interval_kind") or "").strip()
            if not status or not interval_kind:
                raise FitTerminalReportError(
                    "Profile interval evidence omitted status or interval kind."
                )
        if status not in _PROFILE_INTERVAL_STATUSES:
            raise FitTerminalReportError(
                f"Unsupported Profile interval status: {status!r}"
            )
        confidence = _profile_optional_float(
            (
                interval.get(
                    "confidence",
                    extension.provenance.get("confidence_level"),
                )
                if interval is not None
                else extension.provenance.get("confidence_level")
            ),
            "confidence",
        )
        rows.append(
            FitProfileIntervalReportRow(
                coordinate=coordinate,
                display_name=_fit_coordinate_display(coordinate),
                estimate=_profile_optional_float(
                    interval.get("estimate") if interval is not None else None,
                    "estimate",
                ),
                confidence=confidence,
                lower=_profile_optional_float(
                    interval.get("lower") if interval is not None else None,
                    "lower",
                ),
                upper=_profile_optional_float(
                    interval.get("upper") if interval is not None else None,
                    "upper",
                ),
                status=status,
                interval_kind=interval_kind,
                lower_outcome=(
                    None
                    if interval is None or interval.get("lower_outcome") is None
                    else str(interval.get("lower_outcome"))
                ),
                upper_outcome=(
                    None
                    if interval is None or interval.get("upper_outcome") is None
                    else str(interval.get("upper_outcome"))
                ),
                resolution=_profile_compact_resolution(
                    interval,
                    extension=extension,
                ),
            )
        )
    return tuple(rows)


def _build_fit_terminal_report(
    candidate: FitCandidate,
) -> FitTerminalReport:
    """Build the sole typed reporting projection from one terminal candidate."""

    if not isinstance(candidate, FitCandidate):
        raise TypeError("Fit reporting requires FitCandidate.")
    result = candidate.result
    completion = result.completion
    profile_intervals = _profile_interval_rows(
        candidate,
        tuple(completion.terminal_extensions),
    )
    primary_model_details = completion.primary_model_details
    try:
        classifications = _publication_classifications(candidate)
        shared, individual = _parameter_rows(candidate)
    except FitTerminalReportError:
        raise
    except Exception as exc:
        raise FitTerminalReportError(
            f"terminal Fit report construction failed: {exc}"
        ) from exc
    initials = _initial_rows(candidate)
    by_name: dict[str, list[FitEstimateReportRow]] = {}
    for row in individual:
        if row.kind == "per_dataset_parameter":
            by_name.setdefault(row.name, []).append(row)
    aggregates = tuple(
        _aggregate(
            name=name,
            display_name=rows[0].display_name,
            unit=rows[0].unit,
            values=tuple(row.fitted_value for row in rows),
            log_scale=rows[0].log_scale,
        )
        for name, rows in sorted(by_name.items())
    )
    metrics = tuple(
        FitMetricReportRow(
            dataset_id=str(info.dataset_id),
            weighted_sse=info.weighted_residual_sse,
            weighted_mse=info.weighted_residual_mse,
            weighted_rmse=info.weighted_rmse,
            weighted_mae=info.weighted_mae,
            r_squared_by_series=tuple(sorted(info.r_squared_by_series.items())),
            effective_weight=info.weight,
            n_points=info.n_points,
            x_point_count=info.x_point_count,
            x_weighted_residual_sse=info.x_weighted_residual_sse,
            x_weighted_residual_mse=info.x_weighted_residual_mse,
            x_weighted_rmse=info.x_weighted_rmse,
            x_weighted_mae=info.x_weighted_mae,
            x_delta_x=info.x_delta_x,
            x_match_kinds=info.x_match_kinds,
            x_sample_kinds=info.x_sample_kinds,
            x_source_rows=info.x_source_rows,
            x_segment_ids=(
                ()
                if result.model_x_projection_by_dataset.get(str(info.dataset_id)) is None
                else tuple(
                    int(value)
                    for value in result.model_x_projection_by_dataset[
                        str(info.dataset_id)
                    ].segment_ids
                )
            ),
            x_segment_keys=(
                ()
                if result.model_x_projection_by_dataset.get(str(info.dataset_id)) is None
                else tuple(
                    result.model_x_projection_by_dataset[
                        str(info.dataset_id)
                    ].segment_keys
                )
            ),
            x_discontinuity_ids=(
                ()
                if result.model_x_projection_by_dataset.get(str(info.dataset_id)) is None
                else tuple(
                    int(value)
                    for value in result.model_x_projection_by_dataset[
                        str(info.dataset_id)
                    ].discontinuity_ids
                )
            ),
            x_identity_hash=info.x_identity_hash,
            dataset_display_name=next(
                (
                    mapped.display_name
                    for mapped in candidate.origin.mapped_datasets
                    if mapped.dataset_id == str(info.dataset_id)
                ),
                str(info.dataset_id),
            ),
        )
        for info in result.dataset_info
    )
    publication_report = FitPublicationReport(
        shared_parameter_names=tuple(
            sorted(
                next(
                    parameter.display_name
                    for parameter in candidate.setup_snapshot.parameters
                    if parameter.name == item.parameter_name
                )
                for item in classifications
                if item.eligible and item.route == "parameters"
            )
        ),
        fit_local_parameter_names=tuple(
            sorted(
                next(
                    parameter.display_name
                    for parameter in candidate.setup_snapshot.parameters
                    if parameter.name == item.parameter_name
                )
                for item in classifications
                if not item.eligible or item.route == "fit_local_only"
            )
        ),
        classifications=classifications,
        unavailable_reasons=tuple(
            (
                next(
                    parameter.display_name
                    for parameter in candidate.setup_snapshot.parameters
                    if parameter.name == item.parameter_name
                ),
                (
                    "Fit-local only"
                    + ("; Shared refit required" if item.requires_shared_refit else "")
                    if item.route == "fit_local_only"
                    else "Publication route is unavailable"
                ),
            )
            for item in classifications
            if not item.eligible
        ),
        display_names={
            item.name: item.display_name
            for item in candidate.setup_snapshot.parameters
        },
        terminal_permission=bool(
            completion.primary_outcome is not None
            and completion.primary_outcome.publication_allowed
        ),
    )
    diagnostic = completion.optimizer_diagnostic
    if diagnostic is None:
        diagnostic_text = "none"
    else:
        failure = diagnostic.failure
        diagnostic_text = (
            f"{diagnostic.phase}"
            + (f" [{diagnostic.dataset_id}]" if diagnostic.dataset_id else "")
            + f": {failure.message}"
        )
    structured_diagnostics = (
        (
            MappingProxyType(
                {
                    "phase": diagnostic.phase,
                    "dataset_id": diagnostic.dataset_id,
                    "failure": _fit_coordinate_machine_identity(
                        diagnostic.failure
                    ),
                }
            ),
        )
        if diagnostic is not None
        else ()
    )
    validity = [row.validity_diagnostic for row in (*shared, *individual, *initials)]
    validity.extend(
        f"{row.dataset_id} {name} unavailable: {metric.unavailable_reason}"
        for row in metrics
        for name, metric in (
            ("weighted SSE", row.weighted_sse),
            ("weighted MSE", row.weighted_mse),
            ("weighted RMSE", row.weighted_rmse),
            ("weighted MAE", row.weighted_mae),
        )
        if not metric.available
    )
    validity.extend(
        f"{row.dataset_id} {series} R-squared unavailable: {metric.unavailable_reason}"
        for row in metrics
        for series, metric in row.r_squared_by_series
        if not metric.available
    )
    included_infos = tuple(result.dataset_info)
    point_count = sum(int(info.n_points) for info in included_infos)
    series_count = sum(len(values) for values in result.model_series.values())
    included_dataset_count = len(included_infos)
    fitted_parameter_count = sum(
        1
        if parameter.scope in {"shared", "centred_fixed_effects"}
        else included_dataset_count
        for parameter in candidate.setup_snapshot.parameters
        if parameter.selected_config.fit
    )
    fitted_parameter_count += sum(
        max(0, len(item.dataset_ids) - 1)
        for item in result.centred_fixed_effects_settlements
    )
    fitted_initial_keys = {
        (set_id, species)
        for set_id, values in candidate.initial_values_by_set
        for species, _value in values
    }
    initial_scopes = dict(candidate.setup_snapshot.initial_scopes)
    initial_coordinate_count = sum(
        1
        for initial in candidate.setup_snapshot.initials
        if (
            initial.fit
            and initial_scopes.get(initial.species) == "local"
            and (initial.set_id, initial.species) in fitted_initial_keys
        )
    )
    global_initial_coordinate_count = len(
        {
            initial.species
            for initial in candidate.setup_snapshot.global_initials
            if (
                initial.fit
                and initial_scopes.get(initial.species) == "global"
                and any(
                    (set_id, initial.species) in fitted_initial_keys
                    for set_id, _values in candidate.initial_values_by_set
                )
            )
        }
    )
    parameter_count = (
        fitted_parameter_count
        + initial_coordinate_count
        + global_initial_coordinate_count
    )
    completion = result.completion
    return FitTerminalReport(
        status=completion.status,
        joint_optimizer_converged=completion.optimizer_converged,
        joint_optimizer_diagnostic=diagnostic_text,
        weighted_sse=result.weighted_residual_sse,
        weighted_mse=result.weighted_residual_mse,
        metrics_by_dataset=metrics,
        shared_estimates=shared,
        per_dataset_estimates=individual,
        initial_estimates=initials,
        aggregates=aggregates,
        publication=publication_report,
        validity_diagnostics=tuple(dict.fromkeys(validity)),
        terminal_extensions=tuple(completion.terminal_extensions),
        shared_refit_provenance=MappingProxyType(
            dict(result.shared_refit_provenance)
        ),
        structured_diagnostics=structured_diagnostics,
        message=str(result.message),
        scipy_nfev=int(result.scipy_nfev),
        terminal_replay_count=int(result.terminal_replay_count),
        objective_call_count=(
            None
            if result.objective_call_count is None
            else int(result.objective_call_count)
        ),
        dataset_count=included_dataset_count,
        series_count=series_count,
        point_count=point_count,
        parameter_count=parameter_count,
        degrees_of_freedom=point_count - parameter_count,
        primary_model=str(getattr(completion, "primary_model", "ordinary")),
        primary_model_details=MappingProxyType(dict(primary_model_details)),
        centred_fixed_effects_settlements=(
            tuple(result.centred_fixed_effects_settlements)
        ),
        known_input_context=MappingProxyType(
            {
                "temperature_K": float(
                    candidate.origin.scientific_settings.temperature
                ),
                "relative_tolerance": float(
                    candidate.origin.scientific_settings.rtol
                ),
                "absolute_tolerance": float(
                    candidate.origin.scientific_settings.atol
                ),
                "analytic_jacobian_policy": bool(
                    candidate.origin.scientific_settings.analytic_jacobian_policy
                ),
                "wegscheider_policy": bool(
                    candidate.origin.scientific_settings.wegscheider_policy
                ),
                "solver": str(candidate.request.solver),
                "included_dataset_ids": tuple(
                    sorted(
                        item.dataset_id
                        for item in candidate.setup_snapshot.datasets
                        if item.included
                    )
                ),
                "dataset_conditions": tuple(
                    {
                        "dataset_id": item.dataset_id,
                        "included": bool(item.included),
                        "resolved_t_start": float(item.t_start),
                        "data_range_min": float(item.data_range_min),
                        "data_range_max": float(item.data_range_max),
                    }
                    for item in sorted(
                        candidate.setup_snapshot.datasets,
                        key=lambda value: value.dataset_id,
                    )
                ),
                "known_condition_source": (
                    "captured Fit origin scientific settings"
                ),
            }
        ),
        weighting_provenance=_weighting_provenance(candidate),
        profile_intervals=profile_intervals,
    )


def build_fit_terminal_report(
    candidate: FitCandidate,
) -> FitTerminalReport:
    """Build one semantic terminal report with an exact failure boundary."""

    try:
        return _build_fit_terminal_report(candidate)
    except FitTerminalReportError:
        raise
    except Exception as exc:
        raise FitTerminalReportError(
            f"terminal Fit report construction failed: {exc}"
        ) from exc


def _estimate_payload(row: FitEstimateReportRow) -> dict[str, object]:
    payload = {
        field: getattr(row, field)
        for field in row.__dataclass_fields__
    }
    # JSON has no infinity.  Preserve the distinction explicitly instead of
    # emitting invalid JSON or silently replacing an unbounded terminal bound.
    for field in ("minimum", "maximum"):
        value = float(payload[field])
        if not math.isfinite(value):
            payload[field] = None
            payload[f"{field}_unbounded"] = True
        else:
            payload[f"{field}_unbounded"] = False
    return payload


def _aggregate_payload(row: FitAggregateReportRow) -> dict[str, object]:
    return {
        field: getattr(row, field)
        for field in row.__dataclass_fields__
    }


def _profile_interval_payload(
    row: FitProfileIntervalReportRow,
) -> dict[str, object]:
    """Serialize only the compact report projection, never Profile traces."""

    return {
        "coordinate": row.coordinate.to_machine_payload(),
        "display_name": row.display_name,
        "estimate": row.estimate,
        "confidence": row.confidence,
        "lower": row.lower,
        "upper": row.upper,
        "status": row.status,
        "interval_kind": row.interval_kind,
        "lower_outcome": row.lower_outcome,
        "upper_outcome": row.upper_outcome,
        "resolution": _fit_coordinate_machine_identity(row.resolution),
    }


def fit_terminal_report_payload(report: FitTerminalReport) -> Mapping[str, object]:
    """Return strict machine-readable reporting data without UI inference."""

    if not isinstance(report, FitTerminalReport):
        raise TypeError("Fit export requires FitTerminalReport.")
    metrics = []
    for row in report.metrics_by_dataset:
        metrics.append(
            {
                "dataset_id": row.dataset_id,
                "dataset_display_name": row.dataset_display_name,
                "weighted_sse": _metric_payload(row.weighted_sse),
                "weighted_mse": _metric_payload(row.weighted_mse),
                "weighted_rmse": _metric_payload(row.weighted_rmse),
                "weighted_mae": _metric_payload(row.weighted_mae),
                "r_squared_by_series": {
                    name: _metric_payload(metric)
                    for name, metric in row.r_squared_by_series
                },
                "effective_weight": row.effective_weight,
                "n_points": row.n_points,
                "x": {
                    "point_count": row.x_point_count,
                    "weighted_sse": (
                        None
                        if row.x_weighted_residual_sse is None
                        else _metric_payload(row.x_weighted_residual_sse)
                    ),
                    "weighted_mse": (
                        None
                        if row.x_weighted_residual_mse is None
                        else _metric_payload(row.x_weighted_residual_mse)
                    ),
                    "weighted_rmse": (
                        None
                        if row.x_weighted_rmse is None
                        else _metric_payload(row.x_weighted_rmse)
                    ),
                    "weighted_mae": (
                        None
                        if row.x_weighted_mae is None
                        else _metric_payload(row.x_weighted_mae)
                    ),
                    "delta_x": row.x_delta_x,
                    "match_kinds": row.x_match_kinds,
                    "sample_kinds": row.x_sample_kinds,
                    "source_rows": row.x_source_rows,
                    "segment_ids": row.x_segment_ids,
                    "segment_keys": row.x_segment_keys,
                    "discontinuity_ids": row.x_discontinuity_ids,
                    "identity_hash": row.x_identity_hash,
                },
            }
        )
    extensions: list[dict[str, object]] = []
    for item in report.terminal_extensions:
        extension_payload: dict[str, object] = {
            "kind": item.kind,
            "status": item.status,
            "method": item.method,
            "provenance": _fit_coordinate_machine_identity(item.provenance),
            "details": _fit_coordinate_machine_identity(item.details),
            "unavailable_reason": item.unavailable_reason,
        }
        if item.cause is not None:
            extension_payload["cause"] = _fit_coordinate_machine_identity(
                item.cause
            )
        extensions.append(extension_payload)
    payload: dict[str, object] = {
        "status": report.status,
        "message": report.message,
        "scipy_nfev": report.scipy_nfev,
        "terminal_replay_count": report.terminal_replay_count,
        "objective_call_count": report.objective_call_count,
        "dataset_count": report.dataset_count,
        "series_count": report.series_count,
        "point_count": report.point_count,
        "parameter_count": report.parameter_count,
        "degrees_of_freedom": report.degrees_of_freedom,
        "primary_model": report.primary_model,
        "primary_model_details": _fit_coordinate_machine_identity(
            report.primary_model_details
        ),
        "joint_optimizer_converged": report.joint_optimizer_converged,
        "terminal_permission": report.terminal_permission,
        "joint_optimizer_diagnostic": report.joint_optimizer_diagnostic,
        "weighted_sse": _metric_payload(report.weighted_sse),
        "weighted_mse": _metric_payload(report.weighted_mse),
        "metrics_by_dataset": metrics,
        "shared_estimates": [_estimate_payload(row) for row in report.shared_estimates],
        "per_dataset_estimates": [_estimate_payload(row) for row in report.per_dataset_estimates],
        "initial_estimates": [_estimate_payload(row) for row in report.initial_estimates],
        "per_dataset_aggregates": [_aggregate_payload(row) for row in report.aggregates],
        "publication": {
            "shared_parameter_names": report.publication.shared_parameter_names,
            "fit_local_parameter_names": report.publication.fit_local_parameter_names,
            "display_names": dict(report.publication.display_names),
            "terminal_permission": report.publication.terminal_permission,
            "classifications": [
                {
                    "parameter_name": item.parameter_name,
                    "scope": item.scope,
                    "route": item.route,
                    "eligible": item.eligible,
                    "requires_shared_refit": item.requires_shared_refit,
                    "fit_local": item.fit_local,
                    "provenance": (
                        {
                            "origin_tab_id": item.provenance.origin_tab_id,
                            "origin_capture_id": item.provenance.origin_capture_id,
                            "observable_name": item.provenance.observable_name,
                            "admission_scope": item.provenance.admission_scope,
                            "admitted_value": item.provenance.admitted_value,
                        }
                        if item.provenance is not None
                        else None
                    ),
                }
                for item in report.publication.classifications
            ],
            "message": report.publication.message,
        },
        "validity_diagnostics": report.validity_diagnostics,
        "structured_diagnostics": [
            _fit_coordinate_machine_identity(item)
            for item in report.structured_diagnostics
        ],
        "terminal_extensions": extensions,
        "profile_intervals": [
            _profile_interval_payload(row) for row in report.profile_intervals
        ],
        "shared_refit_provenance": _fit_coordinate_machine_identity(
            report.shared_refit_provenance
        ),
        "weighting_provenance": _fit_coordinate_machine_identity(
            report.weighting_provenance
        ),
    }
    payload["centred_fixed_effects_settlements"] = [
        {
            "parameter_name": settlement.parameter_name,
            "dataset_ids": settlement.dataset_ids,
            "reference_dataset_id": settlement.reference_dataset_id,
            "overall_coordinate": _fit_coordinate_machine_identity(
                settlement.overall_coordinate
            ),
            "overall_optimizer_value": settlement.overall_optimizer_value,
            "overall_decoded_value": settlement.overall_decoded_value,
            "deviations": [
                {
                    "coordinate": _fit_coordinate_machine_identity(
                        deviation.coordinate
                    ),
                    "optimizer_value": deviation.optimizer_value,
                    "decoded_value": deviation.decoded_value,
                }
                for deviation in settlement.deviations
            ],
            "dataset_values": dict(settlement.dataset_values),
        }
        for settlement in report.centred_fixed_effects_settlements
    ]
    payload["known_input_context"] = _fit_coordinate_machine_identity(
        report.known_input_context
    )
    return MappingProxyType(payload)
