"""Synchronous human-readable Fit presentation from retained typed facts.

``FitCandidate`` remains the sole terminal science owner. This module builds
small call-local row/text values for the physical Results views; it does not
define a versioned action graph, identity, or serialization format.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Mapping
from typing import Literal

import numpy as np

from kindred.core.analysis.fit_statistics import FitMetric
from kindred.core.analysis.global_fit_execution import FitCoordinate
from kindred.core.simulation_failure import FactualCause
from kindred.gui.fitting.constants import INITIAL_PREFIX
from kindred.gui.fitting.worker import FitCandidate

__all__ = [
    "fit_bound_state",
    "fit_candidate_report_text",
    "fit_candidate_tracker",
]

FitBoundState = Literal["lower", "upper", "interior"]


def _plain(value: object) -> object:
    """Make one presentation leaf from an admitted typed value."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, FitMetric):
        return _plain(value.to_payload())
    if isinstance(value, FitCoordinate):
        return _plain(value.to_machine_payload())
    if isinstance(value, FactualCause):
        return _plain(value.to_payload())
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            normalized_key = (
                key.to_machine_key() if isinstance(key, FitCoordinate) else str(key)
            )
            result[normalized_key] = _plain(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_plain(item) for item in sorted(value, key=repr)]
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return _plain(to_payload())
    if dataclasses.is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    return str(value)


def fit_bound_state(
    value: float,
    minimum: float,
    maximum: float,
    log_scale: bool = False,
) -> FitBoundState:
    """Classify an estimate against bounds in its optimizer coordinate.

    Fit results retain decoded physical values, while a log10 fit optimizes
    their base-10 logarithms.  Bound classification therefore has to happen
    after applying the same encoding used by the optimizer.  A side's
    tolerance is local to that encoded value and bound; the opposite bound
    must not make a small-scale estimate appear to be on a boundary.
    """

    value = float(value)
    minimum = float(minimum)
    maximum = float(maximum)

    def encode(raw_value: float) -> float | None:
        if not math.isfinite(raw_value):
            return None
        if log_scale:
            if raw_value <= 0.0:
                return None
            return math.log10(raw_value)
        return raw_value

    encoded_value = encode(value)
    encoded_minimum = encode(minimum)
    encoded_maximum = encode(maximum)
    if encoded_value is None:
        return "interior"

    def is_at_bound(encoded_bound: float | None) -> bool:
        if encoded_bound is None:
            return False
        # Exact equality is checked separately so a value equal to both
        # malformed/degenerate sides has deterministic lower precedence.
        if encoded_value == encoded_bound:
            return True
        absolute_tolerance = 8.0 * max(
            math.ulp(encoded_value), math.ulp(encoded_bound)
        )
        return math.isclose(
            encoded_value,
            encoded_bound,
            rel_tol=1.0e-9,
            abs_tol=absolute_tolerance,
        )

    if encoded_minimum is not None and encoded_value == encoded_minimum:
        return "lower"
    if encoded_maximum is not None and encoded_value == encoded_maximum:
        return "upper"
    if is_at_bound(encoded_minimum):
        return "lower"
    if is_at_bound(encoded_maximum):
        return "upper"
    return "interior"


def _metric(value: FitMetric | None) -> object:
    if value is None:
        return None
    if not isinstance(value, FitMetric):
        raise TypeError("Fit metrics require FitMetric values.")
    return _plain(value.to_payload())


def _change(fitted: float, nominal: float) -> tuple[float, float | None, str | None]:
    absolute = float(fitted) - float(nominal)
    if nominal == 0.0:
        return absolute, None, "nominal value is zero"
    return absolute, absolute / abs(float(nominal)) * 100.0, None


def _terminal_dataset_value(
    candidate: FitCandidate,
    dataset_id: str,
    parameter_name: str,
) -> float:
    """Return one fitted dataset coordinate from the terminal result.

    Per-dataset presentation is allowed to publish only coordinates retained
    by the terminal result.  In particular, an omitted coordinate is not a
    signal to display its setup value.
    """

    try:
        value = candidate.result.dataset_params[dataset_id][parameter_name]
    except KeyError as exc:
        raise ValueError(
            "Fit presentation omitted terminal fitted coordinate "
            f"{dataset_id!r}::{parameter_name!r}."
        ) from exc
    return float(value)


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


def _display_coordinate(value: object) -> str:
    if isinstance(value, FitCoordinate):
        if value.scope == "centred_overall":
            return f"Overall · {value.name}"
        if value.scope == "centred_deviation":
            return f"Deviation · {value.dataset_id}::{value.name}"
        return value.display_name
    return str(value)


def _profile_endpoint(endpoint: object) -> dict[str, object]:
    return {
        "status": endpoint.status,
        "evaluation_index": endpoint.evaluation_index,
        "optimizer_value": endpoint.optimizer_value,
        "likelihood_ratio": endpoint.likelihood_ratio,
        "coordinate_tolerance": endpoint.coordinate_tolerance,
        "likelihood_ratio_tolerance": endpoint.likelihood_ratio_tolerance,
        "bracket_width": endpoint.bracket_width,
        "reason": endpoint.reason,
        "last_evaluation_index": endpoint.last_evaluation_index,
        "evaluation_count": endpoint.evaluation_count,
        "failure": _plain(endpoint.failure),
    }


def _profile_interval(coordinate: object, interval: object) -> dict[str, object]:
    return {
        "coordinate": _plain(coordinate),
        "display_name": _display_coordinate(coordinate),
        "status": interval.status,
        "interval_kind": interval.interval_kind,
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "lower_outcome": interval.lower_outcome,
        "upper_outcome": interval.upper_outcome,
        "optimizer_bounds": list(interval.optimizer_bounds),
        "physical_bounds": list(interval.physical_bounds),
        "lower_resolution": _profile_endpoint(interval.lower_resolution),
        "upper_resolution": _profile_endpoint(interval.upper_resolution),
        "endpoint_messages": _plain(interval.endpoint_messages),
    }


def _profile_data(candidate: FitCandidate) -> dict[str, object]:
    result = candidate.result.profile_likelihood_result
    if result is None:
        requested = candidate.request.profile_likelihood is not None
        return {
            "requested": requested,
            "status": "not_requested" if not requested else "unavailable",
            "unavailable_reason": None if not requested else "Profile likelihood result is unavailable.",
            "diagnostic": None,
            "intervals": [],
        }
    return {
        "requested": True,
        "status": result.status,
        "method": result.method,
        "requested_parameters": list(result.requested_parameters),
        "confidence_level": result.confidence_level,
        "likelihood_ratio_threshold": result.likelihood_ratio_threshold,
        "likelihood_model": result.likelihood_model,
        "likelihood_ratio_transport": result.likelihood_ratio_transport,
        "fixed_coordinate_policy": result.fixed_coordinate_policy,
        "warm_start_policy": result.warm_start_policy,
        "optimizer_method": result.optimizer_method,
        "max_bracketing_steps_per_side": result.max_bracketing_steps_per_side,
        "max_nfev_per_point": result.max_nfev_per_point,
        "observation_count": result.observation_count,
        "terminal_weighted_residual_sse": result.terminal_weighted_residual_sse,
        "unavailable_reason": result.unavailable_reason,
        "diagnostic": _plain(result.diagnostic),
        "provenance": _plain(result.provenance),
        "intervals": [
            _profile_interval(coordinate, interval)
            for coordinate, interval in sorted(result.intervals.items(), key=lambda item: str(item[0]))
        ],
    }


def _bootstrap_interval(coordinate: object, interval: object) -> dict[str, object]:
    return {
        "coordinate": _plain(coordinate),
        "display_name": _display_coordinate(coordinate),
        "status": interval.status,
        "lower": interval.lower,
        "upper": interval.upper,
        "construction": interval.construction,
        "reason": interval.reason,
    }


def _bootstrap_data(candidate: FitCandidate) -> dict[str, object]:
    result = candidate.result.residual_bootstrap_result
    if result is None:
        requested = candidate.request.residual_bootstrap is not None
        return {
            "requested": requested,
            "status": "not_requested" if not requested else "unavailable",
            "unavailable_reason": None if not requested else "Residual Bootstrap result is unavailable.",
            "diagnostic": None,
            "intervals": [],
        }
    return {
        "requested": True,
        "status": result.status,
        "method": result.method,
        "requested_parameters": list(result.requested_parameters),
        "seed": result.seed,
        "replicates": result.replicates,
        "confidence_level": result.confidence_level,
        "resampling": result.resampling,
        "stratum_manifest_version": result.stratum_manifest_version,
        "draw_recipe_version": result.draw_recipe_version,
        "draw_algorithm": result.draw_algorithm,
        "residual_centering": result.residual_centering,
        "replay_authority": result.replay_authority,
        "preserved_fields": list(result.preserved_fields),
        "replicate_execution": result.replicate_execution,
        "refit_validity": result.refit_validity,
        "minimum_successful_replicates": result.minimum_successful_replicates,
        "minimum_success_fraction": result.minimum_success_fraction,
        "required_successes": result.required_successes,
        "successful_replicates": result.successful_replicates,
        "failed_replicates": result.failed_replicates,
        "interval_construction": result.interval_construction,
        "intervals": [
            _bootstrap_interval(coordinate, interval)
            for coordinate, interval in sorted(result.intervals.items(), key=lambda item: str(item[0]))
        ],
        "primary_fit_diagnostic": _plain(result.primary_fit_diagnostic),
        "primary_fit_unavailable": _plain(result.primary_fit_unavailable),
        "first_failed_replicate_diagnostic": _plain(result.first_failed_replicate_diagnostic),
        "first_failed_replicate_unavailable": _plain(result.first_failed_replicate_unavailable),
        "unavailable_reason": result.unavailable_reason,
        "diagnostic": _plain(result.diagnostic),
        "provenance": _plain(result.provenance),
    }


def _dataset_metrics(candidate: FitCandidate) -> list[dict[str, object]]:
    result = candidate.result
    display_names = {mapped.dataset_id: mapped.display_name for mapped in candidate.origin.mapped_datasets}
    metrics: list[dict[str, object]] = []
    for info in sorted(result.dataset_info, key=lambda item: str(item.dataset_id)):
        dataset_id = str(info.dataset_id)
        metric: dict[str, object] = {
            "dataset_id": dataset_id,
            "dataset_display_name": display_names.get(dataset_id, dataset_id),
            "weighted_sse": _metric(info.weighted_residual_sse),
            "weighted_mse": _metric(info.weighted_residual_mse),
            "weighted_rmse": _metric(info.weighted_rmse),
            "weighted_mae": _metric(info.weighted_mae),
            "r_squared_by_series": {str(name): _metric(value) for name, value in sorted(info.r_squared_by_series.items())},
            "effective_weight": float(info.weight),
        }
        metrics.append(metric)
    return metrics


def _estimate_rows(candidate: FitCandidate) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    request = candidate.request
    result = candidate.result
    terminal_setup = candidate.setup_snapshot
    included_ids = tuple(
        item.dataset_id for item in terminal_setup.datasets if item.included
    )
    display_names = {item.dataset_id: item.display_name for item in candidate.origin.mapped_datasets}
    request_parameters = {item.name: item for item in request.parameters}
    settlements = {item.parameter_name: item for item in result.centred_fixed_effects_settlements}
    local_names = set(getattr(request.fit_local_admission, "scalar_names", ()) or ())
    converged = bool(result.optimizer_converged)
    suffix = "" if converged else "; optimizer did not converge; estimates are diagnostic only"
    shared: list[dict[str, object]] = []
    individual: list[dict[str, object]] = []

    def append_row(target: list[dict[str, object]], **row: object) -> None:
        target.append({str(key): _plain(value) for key, value in row.items()})

    for parameter in terminal_setup.parameters:
        configuration = parameter.selected_config
        launch_parameter = request_parameters.get(parameter.name)
        launch_configuration = (
            configuration
            if launch_parameter is None
            else launch_parameter.selected_config
        )
        origin_parameter = next(
            (
                item
                for item in candidate.origin.parameter_defaults
                if item.name == parameter.name
            ),
            None,
        )
        if origin_parameter is None:
            captured_value = launch_configuration.value
            captured_source = "Fit setup value"
        elif parameter.scope in {"shared", "centred_fixed_effects"}:
            captured_value = origin_parameter.shared_config.value
            captured_source = "captured project Shared default"
        else:
            captured_values = dict(origin_parameter.per_dataset_config.dataset_values)
            captured_value = captured_values.get(
                included_ids[0], origin_parameter.per_dataset_config.value
            )
            captured_source = "captured project Per-dataset default"
        fit_local = parameter.name in local_names or parameter.fit_local_provenance is not None
        if parameter.scope == "centred_fixed_effects":
            settlement = settlements.get(parameter.name)
            if settlement is None:
                raise ValueError(f"canonical result omitted centred settlement {parameter.name!r}")
            value = float(settlement.overall_decoded_value)
            nominal = float(launch_configuration.value)
            absolute, percentage, unavailable = _change(value, nominal)
            append_row(shared, kind="centred_overall", name=parameter.name,
                display_name=f"Overall · {parameter.display_name}", scope="Overall", unit=parameter.unit,
                captured_value=captured_value, nominal_value=nominal, fitted_value=value,
                minimum=configuration.minimum, maximum=configuration.maximum,
                bound_state=fit_bound_state(
                    value,
                    configuration.minimum,
                    configuration.maximum,
                    log_scale=configuration.log10,
                ),
                absolute_change=absolute, percentage_change=percentage,
                percentage_unavailable_reason=unavailable, valid=converged,
                validity_diagnostic="decoded centred overall value is finite and within configured bounds" + suffix,
                dataset_ids=list(settlement.dataset_ids),
                captured_value_source=captured_source, nominal_value_source="Fit setup")
            deviation_bounds = math.log10(configuration.maximum) - math.log10(configuration.minimum) if configuration.log10 else configuration.maximum - configuration.minimum
            for deviation in settlement.deviations:
                value = float(deviation.decoded_value)
                absolute, percentage, unavailable = _change(value, 0.0)
                dataset_id = str(deviation.coordinate.dataset_id)
                append_row(individual, kind="centred_deviation", name=parameter.name,
                    display_name=f"Deviation · {dataset_id}::{parameter.display_name}",
                    scope="Centred dataset deviation (Fit-only)", unit="decades" if configuration.log10 else parameter.unit,
                    captured_value=0.0, nominal_value=0.0, fitted_value=value,
                    minimum=-deviation_bounds, maximum=deviation_bounds,
                    bound_state=fit_bound_state(
                        value,
                        -deviation_bounds,
                        deviation_bounds,
                    ),
                    absolute_change=absolute, percentage_change=percentage,
                    percentage_unavailable_reason=unavailable, valid=converged,
                    validity_diagnostic="centred optimizer-scale deviation; derived reference included and no deviation is publishable" + suffix,
                    dataset_id=dataset_id, dataset_ids=[dataset_id], dataset_display_name=display_names.get(dataset_id),
                    captured_value_source="centred terminal settlement", nominal_value_source="centred zero-deviation seed")
            continue
        if parameter.scope == "shared":
            if parameter.selected_config.fit:
                try:
                    value = float(result.shared_params[parameter.name])
                except KeyError as exc:
                    raise ValueError(
                        f"canonical result omitted shared fitted coordinate {parameter.name!r}"
                    ) from exc
            else:
                # Fixed Shared rows are setup-only and deliberately do not
                # become fitted/presentation rows.
                continue
            nominal = float(launch_configuration.value)
            nominal_source = "Fit-local setup admission" if fit_local else "Fit setup"
            absolute, percentage, unavailable = _change(value, nominal)
            append_row(shared, kind="shared_parameter", name=parameter.name, display_name=parameter.display_name,
                scope="Shared Fit-local scalar" if fit_local else "Shared", unit=parameter.unit,
                captured_value=captured_value, nominal_value=nominal, fitted_value=value,
                minimum=configuration.minimum, maximum=configuration.maximum,
                bound_state=fit_bound_state(
                    value,
                    configuration.minimum,
                    configuration.maximum,
                    log_scale=configuration.log10,
                ),
                absolute_change=absolute, percentage_change=percentage,
                percentage_unavailable_reason=unavailable, valid=converged or not configuration.fit,
                validity_diagnostic="finite and within configured bounds" + ("; Fit-local supplemental scalar provenance" if fit_local else "") + (suffix if configuration.fit else ""),
                captured_value_source="Fit-local starting value (not a project default)" if fit_local else captured_source,
                nominal_value_source=nominal_source)
            continue
        # Fixed Per-dataset coordinates are not terminal fitted values and
        # therefore have no presentation row.  Fitted coordinates below must
        # come from the terminal dataset-first result without nominal fallback.
        if not configuration.fit:
            continue
        values = dict(launch_configuration.dataset_values)
        captured_values = (
            dict(origin_parameter.per_dataset_config.dataset_values)
            if origin_parameter is not None
            else values
        )
        for dataset_id in included_ids:
            value = _terminal_dataset_value(candidate, dataset_id, parameter.name)
            if dataset_id not in values:
                raise ValueError(
                    f"Fit setup omitted nominal dataset coordinate {dataset_id!r} "
                    f"for {parameter.name!r}."
                )
            nominal = float(values[dataset_id])
            captured = float(captured_values.get(dataset_id, nominal))
            absolute, percentage, unavailable = _change(value, nominal)
            append_row(individual, kind="per_dataset_parameter", name=parameter.name, display_name=parameter.display_name,
                scope="Per-dataset Fit-local scalar" if fit_local else "Per-dataset", unit=parameter.unit,
                captured_value=captured, nominal_value=nominal, fitted_value=value,
                minimum=configuration.minimum, maximum=configuration.maximum,
                bound_state=fit_bound_state(
                    value,
                    configuration.minimum,
                    configuration.maximum,
                    log_scale=configuration.log10,
                ),
                absolute_change=absolute, percentage_change=percentage,
                percentage_unavailable_reason=unavailable, valid=converged or not configuration.fit,
                validity_diagnostic="finite and within configured bounds" + ("; Fit-local supplemental scalar provenance" if fit_local else "") + (suffix if configuration.fit else ""),
                dataset_id=dataset_id, dataset_ids=[dataset_id], dataset_display_name=display_names.get(dataset_id),
                captured_value_source="Fit-local starting value (not a project default)" if fit_local else captured_source,
                nominal_value_source="Fit-local setup admission" if fit_local else "Fit setup")
    return shared, individual


def _initial_rows(candidate: FitCandidate) -> list[dict[str, object]]:
    request = candidate.request
    result = candidate.result
    origin = candidate.origin
    terminal_setup = candidate.setup_snapshot
    included_ids = {item.dataset_id for item in terminal_setup.datasets if item.included}
    launch_initials = {
        (item.set_id, item.species): item for item in request.initials
    }
    launch_global_initials = {
        item.species: item for item in request.global_initials
    }
    mapped_by_set: dict[str, list[object]] = {}
    for mapped in origin.mapped_datasets:
        if mapped.dataset_id not in included_ids:
            continue
        mapped_by_set.setdefault(mapped.set_id, []).append(mapped)
    scopes = dict(terminal_setup.initial_scopes)
    rows: list[dict[str, object]] = []
    converged = bool(result.optimizer_converged)
    suffix = "" if converged else "; optimizer did not converge; estimates are diagnostic only"
    for item in terminal_setup.initials:
        if not item.fit or scopes.get(item.species) != "local":
            continue
        launch_item = launch_initials.get((item.set_id, item.species), item)
        for mapped in mapped_by_set.get(item.set_id, ()):
            value = _terminal_dataset_value(
                candidate,
                mapped.dataset_id,
                f"{INITIAL_PREFIX}{item.species}",
            )
            absolute, percentage, unavailable = _change(value, launch_item.initial)
            captured = dict(mapped.initial_values).get(item.species, launch_item.initial)
            rows.append({"kind": "initial", "name": item.species, "display_name": item.species,
                "scope": "Per-dataset IC", "unit": "model concentration", "captured_value": captured,
                "nominal_value": launch_item.initial, "fitted_value": value, "minimum": launch_item.minimum, "maximum": launch_item.maximum,
                "bound_state": fit_bound_state(
                    value,
                    launch_item.minimum,
                    launch_item.maximum,
                    log_scale=bool(launch_item.log10),
                ), "absolute_change": absolute,
                "percentage_change": percentage, "percentage_unavailable_reason": unavailable, "valid": converged,
                "validity_diagnostic": "finite, nonnegative, and within configured bounds" + suffix,
                "set_id": item.set_id, "dataset_ids": [mapped.dataset_id], "dataset_display_name": mapped.display_name,
                "set_display_name": mapped.set_name,
                "captured_value_source": "captured mapped dataset initial value", "nominal_value_source": "Fit setup"})
    for item in terminal_setup.global_initials:
        if not item.fit or scopes.get(item.species) != "global":
            continue
        launch_item = launch_global_initials.get(item.species, item)
        for mapped_items in mapped_by_set.values():
            for mapped in mapped_items:
                value = _terminal_dataset_value(
                    candidate,
                    mapped.dataset_id,
                    f"{INITIAL_PREFIX}{item.species}",
                )
                absolute, percentage, unavailable = _change(value, launch_item.initial)
                captured = dict(mapped.initial_values).get(item.species, launch_item.initial)
                rows.append({"kind": "initial", "name": item.species, "display_name": item.species,
                    "scope": "Shared Fit IC coordinate; independently applied per set", "unit": "model concentration",
                    "captured_value": captured, "nominal_value": launch_item.initial, "fitted_value": value,
                    "minimum": launch_item.minimum, "maximum": launch_item.maximum,
                    "bound_state": fit_bound_state(
                        value,
                        launch_item.minimum,
                        launch_item.maximum,
                        log_scale=bool(launch_item.log10),
                    ), "absolute_change": absolute,
                    "percentage_change": percentage, "percentage_unavailable_reason": unavailable, "valid": converged,
                    "validity_diagnostic": "finite, nonnegative, and within configured bounds" + suffix,
                    "set_id": mapped.set_id, "dataset_ids": [mapped.dataset_id], "dataset_display_name": mapped.display_name,
                    "set_display_name": mapped.set_name,
                    "captured_value_source": "captured mapped dataset initial value", "nominal_value_source": "Fit setup"})
    return rows


def _aggregate_rows(individual: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in individual:
        if row.get("kind") == "per_dataset_parameter":
            grouped.setdefault(str(row["name"]), []).append(row)
    aggregates: list[dict[str, object]] = []
    for name, rows in sorted(grouped.items()):
        values = tuple(sorted(float(row["fitted_value"]) for row in rows))
        count = len(values)
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if count > 1 else None

        def percentile(percentile: float) -> float:
            position = (count - 1) * percentile
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return values[lower]
            fraction = position - lower
            return values[lower] * (1.0 - fraction) + values[upper] * fraction

        aggregates.append({"name": name, "display_name": rows[0]["display_name"], "unit": rows[0]["unit"], "n": count,
            "arithmetic_mean": mean, "sample_sd": sd, "standard_error": None if sd is None else sd / math.sqrt(count),
            "median": statistics.median(values), "iqr": percentile(0.75) - percentile(0.25),
            "minimum": values[0], "maximum": values[-1]})
    return aggregates


def _known_input_context(candidate: FitCandidate) -> dict[str, object]:
    settings = candidate.origin.scientific_settings
    request = candidate.request
    return {"temperature_K": float(settings.temperature), "relative_tolerance": float(settings.rtol),
        "absolute_tolerance": float(settings.atol), "analytic_jacobian_policy": bool(settings.analytic_jacobian_policy),
        "solver": str(request.solver), "included_dataset_ids": list(request.dispatch.included_dataset_ids),
        "dataset_conditions": [{"dataset_id": item.dataset_id, "included": bool(item.included),
            "resolved_t_start": float(item.t_start), "data_range_min": float(item.data_range_min),
            "data_range_max": float(item.data_range_max)} for item in request.datasets],
        "known_condition_source": "captured Fit origin scientific settings"}


def _estimate_line(row: Mapping[str, object]) -> str:
    percentage = _display(row.get("percentage_change")) + "%" if row.get("percentage_change") is not None else str(row.get("percentage_unavailable_reason") or "unavailable")
    location = f" [{row['dataset_id']}]" if row.get("dataset_id") is not None else ""
    if row.get("set_id") is not None:
        location = f" [{row['set_id']}]"
    return (f"{row.get('display_name', row.get('name'))}{location}: fitted={_display(row.get('fitted_value'))}; "
        f"nominal={_display(row.get('nominal_value'))}; Δ={_display(row.get('absolute_change'))}; Δ%={percentage}; "
        f"bounds={_display(row.get('minimum'))}..{_display(row.get('maximum'))} ({row.get('bound_state')})")


def _profile_lines(profile: Mapping[str, object]) -> tuple[str, ...]:
    lines = [f"Profile likelihood: status={profile.get('status')}"]
    if not profile.get("requested"):
        return (*lines, "  not requested")
    if profile.get("method"):
        lines.append(f"  method: {profile['method']}")
    if profile.get("unavailable_reason"):
        lines.append(f"  unavailable reason: {profile['unavailable_reason']}")
    if profile.get("diagnostic"):
        lines.append(f"  diagnostic: {_display(profile['diagnostic'])}")
    for interval in profile.get("intervals", ()):
        if isinstance(interval, Mapping):
            lines.append(f"  {interval.get('display_name', interval.get('coordinate'))}: status={interval.get('status')}; lower={_display(interval.get('lower'))}; upper={_display(interval.get('upper'))}; lower outcome={interval.get('lower_outcome')}; upper outcome={interval.get('upper_outcome')}")
            for side in ("lower_resolution", "upper_resolution"):
                endpoint = interval.get(side)
                if isinstance(endpoint, Mapping) and endpoint.get("failure"):
                    lines.append(f"    {side.removesuffix('_resolution')} failure: {_display(endpoint['failure'])}")
    return tuple(lines)


def _bootstrap_lines(bootstrap: Mapping[str, object]) -> tuple[str, ...]:
    lines = [f"Residual Bootstrap: status={bootstrap.get('status')}"]
    if not bootstrap.get("requested"):
        return (*lines, "  not requested")
    if bootstrap.get("method"):
        lines.append(f"  method: {bootstrap['method']}")
    lines.append(f"  replicates: {bootstrap.get('successful_replicates')} successful, {bootstrap.get('failed_replicates')} failed of {bootstrap.get('replicates')}")
    if bootstrap.get("unavailable_reason"):
        lines.append(f"  unavailable reason: {bootstrap['unavailable_reason']}")
    if bootstrap.get("diagnostic"):
        lines.append(f"  diagnostic: {_display(bootstrap['diagnostic'])}")
    if bootstrap.get("first_failed_replicate_diagnostic"):
        lines.append(
            "  first failed replicate diagnostic: "
            f"{_display(bootstrap['first_failed_replicate_diagnostic'])}"
        )
    if bootstrap.get("first_failed_replicate_unavailable"):
        lines.append(
            "  first failed replicate unavailable: "
            f"{_display(bootstrap['first_failed_replicate_unavailable'])}"
        )
    for interval in bootstrap.get("intervals", ()):
        if isinstance(interval, Mapping):
            lines.append(f"  {interval.get('display_name', interval.get('coordinate'))}: status={interval.get('status')}; lower={_display(interval.get('lower'))}; upper={_display(interval.get('upper'))}; reason={_display(interval.get('reason'))}")
    return tuple(lines)


def _render_report(data: Mapping[str, object]) -> str:
    lines = [f"Fit status: {data.get('status')}", f"Message: {data.get('message') or 'none'}",
        f"Primary Fit model: {data.get('primary_model') or 'ordinary'}",
        f"Optimizer converged: {str(data.get('optimizer_converged')).lower()}",
        f"Diagnostic only: {str(data.get('diagnostic_only')).lower()}",
        f"Weighted SSE: {_metric_text(data.get('weighted_sse'))}", f"Weighted MSE: {_metric_text(data.get('weighted_mse'))}"]
    if data.get("optimizer_diagnostic"):
        lines.append(f"Optimizer diagnostic: {_display(data['optimizer_diagnostic'])}")
    for title, key in (("Shared estimates", "shared_estimates"), ("Per-dataset estimates", "per_dataset_estimates"), ("Initial estimates", "initial_estimates")):
        rows = data.get(key, ())
        if rows:
            lines.append(f"{title}:")
            lines.extend(f"  {_estimate_line(row)}" for row in rows if isinstance(row, Mapping))
    aggregates = data.get("aggregates")
    if aggregates:
        lines.append("Aggregate metrics:")
        for row in aggregates:
            if isinstance(row, Mapping):
                lines.append(f"  {row.get('display_name', row.get('name'))}: n={row.get('n')}; mean={_display(row.get('arithmetic_mean'))}; median={_display(row.get('median'))}; IQR={_display(row.get('iqr'))}; SD={_display(row.get('sample_sd'))}")
    lines.append("Dataset metrics:")
    for row in data.get("metrics_by_dataset", ()):
        if not isinstance(row, Mapping):
            continue
        lines.append(f"  {row.get('dataset_display_name', row.get('dataset_id'))}: weighted SSE={_metric_text(row.get('weighted_sse'))}; weighted MSE={_metric_text(row.get('weighted_mse'))}; weighted RMSE={_metric_text(row.get('weighted_rmse'))}; weighted MAE={_metric_text(row.get('weighted_mae'))}; effective weight={_display(row.get('effective_weight'))}")
        r_squared = row.get("r_squared_by_series")
        if isinstance(r_squared, Mapping):
            lines.extend(f"    R² {name}={_metric_text(metric)}" for name, metric in sorted(r_squared.items()))
    weighting = data.get("weighting")
    if isinstance(weighting, Mapping):
        lines.append(f"Weighting mode: {weighting.get('dataset_weight_mode')}")
        lines.append(f"Weighting rule: {weighting.get('combined_residual_weighting_rule')}")
    lines.append("Profile likelihood:")
    lines.extend(f"  {line}" for line in _profile_lines(data["profile_likelihood"]))
    lines.append("Residual Bootstrap:")
    lines.extend(f"  {line}" for line in _bootstrap_lines(data["residual_bootstrap"]))
    known = data.get("known_input_context")
    if isinstance(known, Mapping):
        lines.append("Known input context:")
        for key in ("temperature_K", "relative_tolerance", "absolute_tolerance", "analytic_jacobian_policy", "solver", "included_dataset_ids"):
            if key in known:
                lines.append(f"  {key}: {_display(known[key])}")
        for condition in known.get("dataset_conditions", ()):
            if isinstance(condition, Mapping):
                lines.append(f"  dataset condition: {condition.get('dataset_id')}; included={_display(condition.get('included'))}; t_start={_display(condition.get('resolved_t_start'))}; range={_display((condition.get('data_range_min'), condition.get('data_range_max')))}")
    return "\n".join(lines)


def _metric_text(value: object) -> str:
    if not isinstance(value, Mapping):
        return _display(value)
    if bool(value.get("available")):
        return _display(value.get("value"))
    return f"unavailable ({value.get('unavailable_reason')})"


def fit_candidate_tracker(candidate: FitCandidate) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return the small terminal tracker projection synchronously."""

    result = candidate.result
    shared, individual = _estimate_rows(candidate)
    initial = _initial_rows(candidate)
    metrics = _dataset_metrics(candidate)
    lines: list[str] = []
    tooltips: list[str] = []
    for row in (*shared, *individual, *initial):
        lines.append(_estimate_line(row))
        if row.get("name"):
            tooltips.append(f"Parameter ID: {row['name']}")
    for row in metrics:
        if isinstance(row, Mapping):
            label = row.get("dataset_display_name", row.get("dataset_id"))
            lines.append(
                f"Metrics [{label}]: SSE={_metric_text(row.get('weighted_sse'))}; "
                f"MSE={_metric_text(row.get('weighted_mse'))}; "
                f"RMSE={_metric_text(row.get('weighted_rmse'))}"
            )
    return (
        f"Fit status {result.status} | Weighted objective SSQ "
        f"{_metric_text(_metric(result.weighted_residual_sse))}",
        tuple(lines),
        tuple(tooltips),
    )


def fit_candidate_report_text(candidate: FitCandidate) -> str:
    """Return one synchronous human-readable Summary string."""

    result = candidate.result
    shared, individual = _estimate_rows(candidate)
    metrics = _dataset_metrics(candidate)
    data = {
        "status": result.status,
        "message": str(result.message),
        "optimizer_converged": bool(result.optimizer_converged),
        "diagnostic_only": bool(result.diagnostic_only),
        "optimizer_diagnostic": _plain(result.optimizer_diagnostic),
        "primary_model": str(result.primary_model),
        "weighted_sse": _metric(result.weighted_residual_sse),
        "weighted_mse": _metric(result.weighted_residual_mse),
        "shared_estimates": shared,
        "per_dataset_estimates": individual,
        "initial_estimates": _initial_rows(candidate),
        "aggregates": _aggregate_rows(individual),
        "metrics_by_dataset": metrics,
        "weighting": {
            "dataset_weight_mode": str(candidate.request.dataset_weight_mode),
            "combined_residual_weighting_rule": "weighted_residual = raw_residual * sqrt(normalized_effective_dataset_weight) * sqrt(normalized_target_weight)",
        },
        "profile_likelihood": _profile_data(candidate),
        "residual_bootstrap": _bootstrap_data(candidate),
        "known_input_context": _known_input_context(candidate),
    }
    return _render_report(data)
