"""Typed Fit reporting derived only from immutable terminal science."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

from kindred.core.analysis.fit_statistics import FitMetric
from kindred.core.fitting_completion import FitTerminalExtension
from kindred.gui.fitting.launch import FitPublicationClassification
from kindred.gui.fitting.worker import FitCandidate

__all__ = [
    "FitAggregateReportRow",
    "FitEstimateReportRow",
    "FitMetricReportRow",
    "FitPublicationReport",
    "FitTerminalReportError",
    "FitTerminalReport",
    "build_fit_terminal_report",
    "fit_metric_text",
    "fit_bound_state",
    "fit_terminal_extension_lines",
    "fit_terminal_report_payload",
]

FitEstimateKind = Literal[
    "shared_parameter",
    "per_dataset_parameter",
    "hierarchical_coordinate",
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


def _bound(value: object, field: str) -> float:
    """Accept only the finite bounds admitted by the Fit setup owner."""

    return _finite(value, field)


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
            "hierarchical_coordinate",
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
    geometric_dispersion: float | None
    geometric_unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class FitPublicationReport:
    shared_parameter_names: tuple[str, ...]
    fit_local_parameter_names: tuple[str, ...]
    classifications: tuple[FitPublicationClassification, ...] = ()
    unavailable_reasons: tuple[tuple[str, str], ...] = ()
    display_names: Mapping[str, str] = MappingProxyType({})

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

    @property
    def message(self) -> str:
        parts: list[str] = []
        if self.shared_parameter_names:
            parts.append(
                "Shared values eligible for publication: "
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
        diagnostics = tuple(self.structured_diagnostics)
        if any(not isinstance(item, Mapping) for item in diagnostics):
            raise TypeError("Fit terminal diagnostics must be mappings.")
        object.__setattr__(
            self,
            "structured_diagnostics",
            tuple(MappingProxyType(dict(item)) for item in diagnostics),
        )


def _display_value(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_display_value(item) for item in value) + "]"
    return str(value)


def _extension_failure_lines(extension: FitTerminalExtension) -> list[str]:
    """Expose owner-supplied typed failure causes without flattening them."""

    lines: list[str] = []
    for source_name, source in (
        ("provenance", extension.provenance),
        ("details", extension.details),
    ):
        if not isinstance(source, Mapping):
            continue
        for field in (
            "failure_kind",
            "failure",
            "failure_details",
            "fatal_cause",
            "fatal",
        ):
            if field in source:
                lines.append(
                    f"  {field.replace('_', ' ')} ({source_name}): "
                    f"{_display_value(source[field])}"
                )
    return lines


def fit_terminal_extension_lines(
    extension: FitTerminalExtension,
) -> tuple[str, ...]:
    """Render named terminal DTO fields without calculating missing science."""

    if not isinstance(extension, FitTerminalExtension):
        raise TypeError("Terminal method presentation requires FitTerminalExtension.")
    lines = [
        f"{extension.kind}: {extension.method}; status={extension.status}"
    ]
    if extension.unavailable_reason:
        lines.append(f"  unavailable reason: {extension.unavailable_reason}")
    lines.extend(_extension_failure_lines(extension))
    provenance = extension.provenance
    if extension.kind == "profile_likelihood":
        for field, label in (
            ("confidence_level", "confidence level"),
            ("likelihood_model", "likelihood model"),
            ("likelihood_ratio_threshold", "likelihood-ratio threshold"),
            ("fixed_coordinate_policy", "profile policy"),
            ("optimizer_method", "optimizer"),
            ("points_per_side", "points per side"),
            ("max_nfev_per_point", "evaluation budget per point"),
            ("observation_count", "observations"),
            ("terminal_weighted_residual_sse", "terminal weighted SSE"),
        ):
            if field in provenance:
                lines.append(f"  {label}: {_display_value(provenance[field])}")
        intervals = extension.details.get("intervals", {})
        if isinstance(intervals, Mapping):
            for coordinate, raw_interval in intervals.items():
                interval = raw_interval if isinstance(raw_interval, Mapping) else {}
                lines.append(
                    f"  {coordinate}: status={_display_value(interval.get('status'))}; "
                    f"kind={_display_value(interval.get('interval_kind'))}; "
                    f"estimate={_display_value(interval.get('estimate'))}; "
                    f"lower={_display_value(interval.get('lower'))} "
                    f"({_display_value(interval.get('lower_outcome'))}); "
                    f"upper={_display_value(interval.get('upper'))} "
                    f"({_display_value(interval.get('upper_outcome'))}); "
                    f"optimizer bounds={_display_value(interval.get('optimizer_bounds'))}; "
                    f"physical bounds={_display_value(interval.get('physical_bounds'))}; "
                    f"evaluations={_display_value(interval.get('evaluations'))}; "
                    f"reason={_display_value(interval.get('reason'))}"
                )
        if "requested_parameters" in extension.details:
            lines.append(
                "  requested coordinates: "
                + _display_value(extension.details["requested_parameters"])
            )
    elif extension.kind == "bootstrap":
        for field, label in (
            ("seed", "deterministic seed"),
            ("replicates", "configured full refits"),
            ("confidence_level", "confidence level"),
            ("resampling", "resampling"),
            ("preserved_fields", "preserved structure"),
            ("replicate_execution", "replicate execution"),
            ("minimum_successful_replicates", "minimum successful refits"),
            ("minimum_success_fraction", "minimum success fraction"),
            ("successful_replicates", "successful refits"),
            ("failed_replicates", "failed refits"),
            ("required_successes", "required successes"),
            ("interval_construction", "interval construction"),
        ):
            if field in provenance:
                lines.append(f"  {label}: {_display_value(provenance[field])}")
        intervals = extension.details.get("intervals", {})
        if isinstance(intervals, Mapping):
            for coordinate, raw_interval in intervals.items():
                interval = raw_interval if isinstance(raw_interval, Mapping) else {}
                lines.append(
                    f"  {coordinate}: status={_display_value(interval.get('status'))}; "
                    f"lower={_display_value(interval.get('lower'))}; "
                    f"upper={_display_value(interval.get('upper'))}; "
                    f"construction={_display_value(interval.get('construction'))}; "
                    f"reason={_display_value(interval.get('reason'))}"
                )
        roster = extension.details.get("replicate_roster", ())
        if isinstance(roster, (tuple, list)):
            status_counts: dict[str, int] = {}
            failures: list[str] = []
            for raw_record in roster:
                if not isinstance(raw_record, Mapping):
                    continue
                status = str(raw_record.get("status", "unknown"))
                status_counts[status] = status_counts.get(status, 0) + 1
                if raw_record.get("failure") is not None and len(failures) < 3:
                    failures.append(_display_value(raw_record["failure"]))
            lines.append(
                "  replicate provenance: "
                f"{len(roster)} records; statuses={_display_value(status_counts)}"
            )
            if failures:
                lines.append("  replicate failure samples: " + _display_value(failures))
        if "replay_metadata" in extension.details:
            lines.append(
                "  replay metadata: "
                + _display_value(extension.details["replay_metadata"])
            )
        if "requested_parameters" in extension.details:
            lines.append(
                "  requested coordinates: "
                + _display_value(extension.details["requested_parameters"])
            )
    elif extension.kind == "hierarchical":
        details = extension.details
        for field, label in (
            ("parameter_name", "parameter"),
            ("dataset_ids", "datasets"),
            ("missing_optimizer_coordinates", "missing optimizer coordinates"),
            ("population_location", "population location"),
            ("population_location_optimizer_scale", "population location on optimizer scale"),
            ("between_dataset_dispersion", "between-dataset dispersion"),
            ("estimated_positive_dispersion", "estimated positive dispersion"),
            ("dispersion_boundary", "dispersion lower-bound state"),
            ("overall_converged", "overall convergence"),
            ("optimizer_message", "optimizer message"),
            ("physical_bounds", "physical bounds"),
            ("optimizer_bounds", "optimizer bounds"),
            ("dispersion_bounds", "dispersion bounds"),
            ("identifiability", "identifiability"),
            ("identifiability_diagnostics", "identifiability diagnostics"),
            ("uncertainty", "uncertainty provenance"),
            ("fit_local_only", "Fit-local only"),
            ("publication_requires_shared_refit", "publication requires Shared refit"),
        ):
            if field in details:
                lines.append(f"  {label}: {_display_value(details[field])}")
        effects = details.get("individual_dataset_effects", {})
        if isinstance(effects, Mapping):
            for dataset_id, effect in effects.items():
                lines.append(
                    f"  dataset {dataset_id}: {_display_value(effect)}"
                )
        uncertainty = details.get("uncertainty")
        if isinstance(uncertainty, Mapping):
            uncertainty_status = uncertainty.get("status")
            if uncertainty_status not in (None, "available"):
                lines.append(
                    "  uncertainty unavailable: "
                    + _display_value(uncertainty.get("reason", uncertainty_status))
                )
        elif details.get("overall_converged") is False:
            lines.append("  uncertainty unavailable: nonconverged")
        for field, label in (
            ("likelihood_model", "likelihood model"),
            ("random_effects", "random effects"),
            ("marginalization", "marginalization"),
            ("optimizer_scale", "optimizer scale"),
            ("optimizer_method", "optimizer"),
            ("max_nfev", "outer evaluation budget"),
            ("inner_max_nfev", "inner evaluation budget"),
            ("outer_nfev", "outer evaluations"),
            ("uncertainty_method", "uncertainty method"),
            ("fit_local_only", "Fit-local only"),
            ("publication_requires_shared_refit", "publication requires Shared refit"),
        ):
            if field in provenance:
                lines.append(f"  {label}: {_display_value(provenance[field])}")
    return tuple(lines)


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
    geometric_dispersion = None
    reason: str | None = "requires positive log-scale estimates"
    if positive_log:
        log_values = tuple(math.log(value) for value in ordered)
        geometric_mean = math.exp(statistics.fmean(log_values))
        geometric_dispersion = (
            math.exp(statistics.stdev(log_values)) if count > 1 else None
        )
        reason = None if count > 1 else "geometric dispersion requires at least two estimates"
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
        geometric_dispersion=geometric_dispersion,
        geometric_unavailable_reason=reason,
    )


def _publication_classifications(
    candidate: FitCandidate,
    supplied: Mapping[str, FitPublicationClassification]
    | Sequence[FitPublicationClassification]
    | None,
) -> tuple[FitPublicationClassification, ...]:
    """Normalize publication-owner decisions without deriving policy."""

    parameter_by_name = {item.name: item for item in candidate.setup_snapshot.parameters}
    if supplied is None:
        values = tuple(candidate.setup_snapshot.publication_classifications)
    else:
        values = tuple(supplied.values()) if isinstance(supplied, Mapping) else tuple(supplied)
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
    refit_provenance = candidate.result.shared_refit_provenance
    promotion_provenance = (
        refit_provenance.get("promotions", {})
        if isinstance(refit_provenance, Mapping)
        else {}
    )
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
        if not nominal_configuration.fit:
            continue
        if terminal.scope == "shared":
            configuration = terminal.selected_config
            fitted = _finite(fitted_shared[terminal.name], terminal.name)
            promotion = (
                promotion_provenance.get(terminal.name)
                if isinstance(promotion_provenance, Mapping)
                else None
            )
            if isinstance(promotion, Mapping):
                nominal_value = _finite(
                    promotion["seed_value"],
                    f"{terminal.name} Shared-refit seed",
                )
                captured_value = captured_value
                validity_diagnostic = (
                    "finite and within configured bounds; completed Shared "
                    "refit is authority and the central-tendency seed is "
                    "provenance only"
                )
            else:
                nominal_value = nominal_configuration.value
                captured_value = captured_value
                validity_diagnostic = "finite and within configured bounds"
            if fit_local_provenance:
                captured_value_source = "Fit-local starting value (not a project default)"
                validity_diagnostic += "; Fit-local supplemental scalar provenance"
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
                    valid=terminal_converged,
                    validity_diagnostic=validity_diagnostic,
                    log_scale=configuration.log10,
                    captured_value_source=captured_value_source,
                    nominal_value_source=(
                        "Fit-local setup admission"
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
            fitted = _finite(fitted_by_dataset[dataset_id][terminal.name], terminal.name)
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
                    kind=(
                        "hierarchical_coordinate"
                        if terminal.scope == "hierarchical"
                        else "per_dataset_parameter"
                    ),
                    name=terminal.name,
                    display_name=terminal.display_name,
                    scope=(
                        "Hierarchical input coordinate (Fit-only)"
                        if terminal.scope == "hierarchical"
                        else (
                            "Per-dataset Fit-local scalar"
                            if fit_local_provenance
                            else "Per-dataset"
                        )
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
                    valid=terminal_converged,
                    validity_diagnostic=(
                        "finite and within configured bounds; Fit-local supplemental "
                        "scalar provenance"
                        if fit_local_provenance
                        else "finite and within configured bounds"
                    ) + terminal_validity_suffix,
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


def _diagnostic_payload(diagnostic: object) -> Mapping[str, object] | None:
    if diagnostic is None:
        return None
    failure = getattr(diagnostic, "failure", None)
    if not isinstance(failure, Mapping):
        raise FitTerminalReportError("Fit diagnostic has no structured failure payload.")
    return MappingProxyType(
        {
            "phase": str(getattr(diagnostic, "phase", "optimizer")),
            "dataset_id": getattr(diagnostic, "dataset_id", None),
            "failure": dict(failure),
        }
    )


def _build_fit_terminal_report(
    candidate: FitCandidate,
    *,
    publication: Mapping[str, FitPublicationClassification]
    | Sequence[FitPublicationClassification]
    | None = None,
) -> FitTerminalReport:
    """Build the sole typed reporting projection from one terminal candidate."""

    if not isinstance(candidate, FitCandidate):
        raise TypeError("Fit reporting requires FitCandidate.")
    result = candidate.result
    try:
        classifications = _publication_classifications(candidate, publication)
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
    )
    completion = result.completion
    diagnostic = completion.optimizer_diagnostic
    if diagnostic is None:
        diagnostic_text = "none"
    else:
        failure = diagnostic.failure
        diagnostic_text = (
            f"{diagnostic.phase}"
            + (f" [{diagnostic.dataset_id}]" if diagnostic.dataset_id else "")
            + f": {failure.get('message') or 'optimizer diagnostic'}"
        )
    structured_diagnostics = tuple(
        payload
        for payload in (_diagnostic_payload(diagnostic),)
        if payload is not None
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
        1 if parameter.scope == "shared" else included_dataset_count
        for parameter in candidate.setup_snapshot.parameters
        if parameter.selected_config.fit
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
    )


def build_fit_terminal_report(
    candidate: FitCandidate,
    *,
    publication: Mapping[str, FitPublicationClassification]
    | Sequence[FitPublicationClassification]
    | None = None,
) -> FitTerminalReport:
    """Build one semantic terminal report with an exact failure boundary."""

    try:
        return _build_fit_terminal_report(candidate, publication=publication)
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
            }
        )
    extensions = [
        {
            "kind": item.kind,
            "status": item.status,
            "method": item.method,
            "provenance": dict(item.provenance),
            "details": dict(item.details),
            "unavailable_reason": item.unavailable_reason,
        }
        for item in report.terminal_extensions
    ]
    return MappingProxyType(
        {
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
            "joint_optimizer_converged": report.joint_optimizer_converged,
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
                                "dataset_id": item.provenance.dataset_id,
                                "raw_observation_id": item.provenance.raw_observation_id,
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
                dict(item) for item in report.structured_diagnostics
            ],
            "terminal_extensions": extensions,
            "shared_refit_provenance": dict(report.shared_refit_provenance),
        }
    )
