"""Call-local Fit action projections and deterministic presentation formatters.

``FitCandidate`` is retained by :mod:`fit_session_owner`; this module never
retains a report, a row model, or a second copy of terminal science.  The
single projection built here is ordinary Python data and is consumed
synchronously by Copy Run, the report text, and Results Summary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from typing import Literal

import numpy as np

from kindred.core.analysis.fit_statistics import (
    FitMetric,
    finite_positive_mean_normalize,
)
from kindred.core.analysis.global_fit_execution import FitCoordinate
from kindred.core.simulation_failure import FactualCause
from kindred.gui.fitting.constants import INITIAL_PREFIX
from kindred.gui.fitting.worker import FitCandidate

__all__ = [
    "build_fit_action_projection",
    "fit_action_projection_hash",
    "fit_action_projection_json",
    "fit_action_projection_payload",
    "fit_action_projection_report_text",
    "fit_action_projection_summary",
    "fit_action_projection_profile_lines",
    "fit_action_projection_bootstrap_lines",
    "fit_action_projection_primary_model_lines",
    "fit_bound_state",
]

FitBoundState = Literal["lower", "upper", "interior"]


def _plain(value: object) -> object:
    """Convert an admitted value to a plain JSON-compatible value graph."""

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
            if isinstance(key, FitCoordinate):
                normalized_key = key.to_machine_key()
            else:
                normalized_key = str(key)
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


def fit_bound_state(value: float, minimum: float, maximum: float) -> FitBoundState:
    """Classify an estimate against finite sides of configured bounds."""

    value = float(value)
    minimum = float(minimum)
    maximum = float(maximum)
    scales = [1.0, abs(value)]
    if math.isfinite(minimum):
        scales.append(abs(minimum))
    if math.isfinite(maximum):
        scales.append(abs(maximum))
    tolerance = max(scales) * 1.0e-9
    if math.isfinite(minimum) and math.isclose(
        value, minimum, rel_tol=1.0e-9, abs_tol=tolerance
    ):
        return "lower"
    if math.isfinite(maximum) and math.isclose(
        value, maximum, rel_tol=1.0e-9, abs_tol=tolerance
    ):
        return "upper"
    return "interior"


def _metric(value: FitMetric | None) -> object:
    if value is None:
        return None
    if not isinstance(value, FitMetric):
        raise TypeError("Fit projection metrics require FitMetric values.")
    return _plain(value.to_payload())


def _change(fitted: float, nominal: float) -> tuple[float, float | None, str | None]:
    absolute = float(fitted) - float(nominal)
    if nominal == 0.0:
        return absolute, None, "nominal value is zero"
    return absolute, absolute / abs(float(nominal)) * 100.0, None


def _configuration_payload(configuration: object) -> dict[str, object]:
    return {
        "value": float(configuration.value),
        "minimum": float(configuration.minimum),
        "maximum": float(configuration.maximum),
        "fit": bool(configuration.fit),
        "log10": bool(configuration.log10),
        "dataset_values": [
            {"dataset_id": str(dataset_id), "value": float(value)}
            for dataset_id, value in sorted(configuration.dataset_values)
        ],
    }


def _request_projection(candidate: FitCandidate) -> dict[str, object]:
    request = candidate.request
    origin = candidate.origin
    datasets = []
    for dataset in sorted(request.datasets, key=lambda item: item.dataset_id):
        datasets.append(
            {
                "dataset_id": dataset.dataset_id,
                "included": bool(dataset.included),
                "t_start": float(dataset.t_start),
                "data_range_min": float(dataset.data_range_min),
                "data_range_max": float(dataset.data_range_max),
                "maximum_observations_per_target": dataset.maximum_observations_per_target,
                "weight": float(dataset.weight),
                "target_species": list(dataset.target_species),
                "target_weights": [
                    {"target": name, "weight": float(value)}
                    for name, value in dataset.target_weights
                ],
                "observation_targets": [
                    {"raw_observation_id": raw_id, "target": target}
                    for raw_id, target in dataset.observation_targets
                ],
                "parametric": _plain(dataset.parametric),
            }
        )
    parameters = []
    for parameter in sorted(request.parameters, key=lambda item: item.name):
        parameters.append(
            {
                "name": parameter.name,
                "display_name": parameter.display_name,
                "scope": parameter.scope,
                "unit": parameter.unit,
                "family": parameter.family,
                "role": parameter.role,
                "scientific_category": parameter.scientific_category,
                "allow_exact_zero": bool(parameter.allow_exact_zero),
                "fit_local_provenance": _plain(parameter.fit_local_provenance),
                "shared_config": _configuration_payload(parameter.shared_config),
                "per_dataset_config": _configuration_payload(parameter.per_dataset_config),
            }
        )
    origin_datasets = []
    for mapped in sorted(origin.mapped_datasets, key=lambda item: item.dataset_id):
        origin_datasets.append(
            {
                "dataset_id": mapped.dataset_id,
                "display_name": mapped.display_name,
                "set_id": mapped.set_id,
                "set_name": mapped.set_name,
                "included_at_launch": bool(mapped.included_at_launch),
                "resolved_t_start": float(mapped.resolved_t_start),
                "initial_values": [
                    {"species": name, "value": float(value)}
                    for name, value in mapped.initial_values
                ],
                "observations": [
                    {
                        "raw_observation_id": observation.raw_observation_id,
                        "raw_column": observation.raw_column,
                        "model_target": observation.model_target,
                        "t": list(observation.t),
                        "y": list(observation.y),
                        "source_rows": list(observation.source_rows),
                    }
                    for observation in mapped.observations
                ],
            }
        )
    settings = origin.scientific_settings
    return {
        "solver": str(request.solver),
        "dataset_weight_mode": str(request.dataset_weight_mode),
        "datasets": datasets,
        "parameters": parameters,
        "initials": [_plain(item) for item in request.initials],
        "global_initials": [_plain(item) for item in request.global_initials],
        "initial_scopes": [
            {"species": species, "scope": scope}
            for species, scope in request.initial_scopes
        ],
        "defaults": _plain(request.defaults),
        "profile_likelihood": _plain(request.profile_likelihood),
        "residual_bootstrap": _plain(request.residual_bootstrap),
        "fit_local_admission": _plain(request.fit_local_admission),
        "origin": {
            "tab_id": origin.tab_id,
            "tab_name": origin.tab_name,
            "mechanism_source": _plain(origin.mechanism_source),
            "scientific_settings": _plain(settings),
            "fit_defaults": _plain(origin.fit_defaults),
            "mapped_datasets": origin_datasets,
        },
        "resolved_auto_modes": [
            {"dataset_id": dataset_id, "mode": mode}
            for dataset_id, mode in sorted(candidate.resolved_auto_modes)
        ],
    }


def _model_identity_projection(result: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_id in sorted(result.model_series):
        for series_name in sorted(result.model_series[dataset_id]):
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "series": series_name,
                    "model_values": _plain(result.model_series[dataset_id][series_name]),
                    "model_x": _plain(result.model_x_by_species[dataset_id][series_name]),
                    "segment_ids": _plain(result.model_segment_ids[dataset_id][series_name]),
                    "segment_keys": _plain(result.model_segment_keys[dataset_id][series_name]),
                    "discontinuity_ids": _plain(
                        result.model_discontinuity_ids[dataset_id][series_name]
                    ),
                    "selection": _plain(
                        result.model_selection_provenance[dataset_id][series_name]
                    ),
                    "observation_ids": _plain(
                        result.model_observation_ids[dataset_id][series_name]
                    ),
                    "raw_columns": _plain(result.model_raw_columns[dataset_id][series_name]),
                    "source_rows": _plain(result.model_source_rows[dataset_id][series_name]),
                }
            )
    return rows


def _simulation_projection(result: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_id in sorted(result.simulation_results_by_dataset):
        simulation = result.simulation_results_by_dataset[dataset_id]
        rows.append(
            {
                "dataset_id": dataset_id,
                "t": _plain(simulation.t),
                "series": _plain(simulation.series),
                "segment_ids": _plain(simulation.segment_ids),
                "segment_keys": _plain(simulation.segment_keys),
                "discontinuity_ids": _plain(simulation.discontinuity_ids),
                "solver_provenance": _plain(simulation.solver_provenance),
                "preparation_warnings": _plain(simulation.preparation_warnings),
                "requested_algebra_outputs": _plain(simulation.requested_algebra_outputs),
                "succeeded_algebra_outputs": _plain(simulation.succeeded_algebra_outputs),
                "failed_algebra_outputs": _plain(simulation.failed_algebra_outputs),
                "omitted_algebra_outputs": _plain(simulation.omitted_algebra_outputs),
                "algebra_errors": _plain(simulation.algebra_errors),
                "scalars": _plain(simulation.scalars),
                "parameters": _plain(simulation.parameters),
                "intervention_execution": _plain(simulation.intervention_execution),
                "presentation_boundaries": _plain(simulation.presentation_boundaries),
            }
        )
    return rows


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
    }


def _profile_evaluation(evaluation: object) -> dict[str, object]:
    return {
        "evaluation_index": evaluation.evaluation_index,
        "stage": evaluation.stage,
        "discovery_step": evaluation.discovery_step,
        "refinement_step": evaluation.refinement_step,
        "fixed_optimizer_value": evaluation.fixed_optimizer_value,
        "fixed_value": evaluation.fixed_value,
        "status": evaluation.status,
        "warm_start_policy": evaluation.warm_start_policy,
        "optimizer_converged": evaluation.optimizer_converged,
        "optimized_parameters": _plain(evaluation.optimized_parameters),
        "weighted_residual_sse": evaluation.weighted_residual_sse,
        "likelihood_ratio": evaluation.likelihood_ratio,
        "likelihood_ratio_raw": evaluation.likelihood_ratio_raw,
        "likelihood_ratio_log": evaluation.likelihood_ratio_log,
        "likelihood_ratio_outcome": evaluation.likelihood_ratio_outcome,
        "failure": _plain(evaluation.failure),
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
        "lower_evaluations": [_profile_evaluation(item) for item in interval.lower_evaluations],
        "upper_evaluations": [_profile_evaluation(item) for item in interval.upper_evaluations],
        "lower_resolution": _profile_endpoint(interval.lower_resolution),
        "upper_resolution": _profile_endpoint(interval.upper_resolution),
        "endpoint_messages": _plain(interval.endpoint_messages),
    }


def _profile_projection(candidate: FitCandidate) -> dict[str, object]:
    result = candidate.result.profile_likelihood_result
    if result is None:
        return {
            "requested": candidate.request.profile_likelihood is not None,
            "status": "not_requested" if candidate.request.profile_likelihood is None else "unavailable",
            "unavailable_reason": (
                None
                if candidate.request.profile_likelihood is None
                else "Profile likelihood result is unavailable."
            ),
            "diagnostic": None,
            "intervals": [],
        }
    intervals = [
        _profile_interval(coordinate, interval)
        for coordinate, interval in sorted(result.intervals.items(), key=lambda item: str(item[0]))
    ]
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
        "intervals": intervals,
    }


def _bootstrap_interval(coordinate: object, interval: object) -> dict[str, object]:
    return {
        "coordinate": _plain(coordinate),
        "display_name": _display_coordinate(coordinate),
        "status": interval.status,
        "lower": interval.lower,
        "upper": interval.upper,
        "construction": interval.construction,
        "successful_values": list(interval.successful_values),
        "reason": interval.reason,
    }


def _bootstrap_replicate(replicate: object) -> dict[str, object]:
    return {
        "replicate": replicate.replicate,
        "status": replicate.status,
        "parameters": _plain(replicate.parameters),
        "failure": _plain(replicate.failure),
        "unavailable": _plain(replicate.unavailable),
    }


def _bootstrap_projection(candidate: FitCandidate) -> dict[str, object]:
    result = candidate.result.residual_bootstrap_result
    if result is None:
        return {
            "requested": candidate.request.residual_bootstrap is not None,
            "status": "not_requested" if candidate.request.residual_bootstrap is None else "unavailable",
            "unavailable_reason": (
                None
                if candidate.request.residual_bootstrap is None
                else "Residual Bootstrap result is unavailable."
            ),
            "diagnostic": None,
            "intervals": [],
            "replicate_roster": [],
        }
    intervals = [
        _bootstrap_interval(coordinate, interval)
        for coordinate, interval in sorted(result.intervals.items(), key=lambda item: str(item[0]))
    ]
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
        "stratum_manifest": _plain(result.stratum_manifest),
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
        "intervals": intervals,
        "replicate_roster": [_bootstrap_replicate(item) for item in result.replicate_roster],
        "primary_fit_diagnostic": _plain(result.primary_fit_diagnostic),
        "primary_fit_unavailable": _plain(result.primary_fit_unavailable),
        "unavailable_reason": result.unavailable_reason,
        "diagnostic": _plain(result.diagnostic),
        "provenance": _plain(result.provenance),
    }


def _result_projection(candidate: FitCandidate) -> dict[str, object]:
    result = candidate.result
    request_datasets = {item.dataset_id: item for item in candidate.request.datasets}
    mapped_datasets = {item.dataset_id: item for item in candidate.origin.mapped_datasets}
    metrics = []
    display_names = {
        mapped.dataset_id: mapped.display_name
        for mapped in candidate.origin.mapped_datasets
    }
    for info in sorted(result.dataset_info, key=lambda item: str(item.dataset_id)):
        dataset_id = str(info.dataset_id)
        x_projection = result.model_x_projection_by_dataset.get(dataset_id)
        x_payload: dict[str, object] = {
            "point_count": int(info.x_point_count),
            "weighted_sse": _metric(info.x_weighted_residual_sse),
            "weighted_mse": _metric(info.x_weighted_residual_mse),
            "weighted_rmse": _metric(info.x_weighted_rmse),
            "weighted_mae": _metric(info.x_weighted_mae),
            "delta_x": _plain(info.x_delta_x),
            "match_kinds": _plain(info.x_match_kinds),
            "sample_kinds": _plain(info.x_sample_kinds),
            "source_rows": _plain(info.x_source_rows),
        }
        if x_projection is not None:
            x_payload.update(
                {
                    "selected_time": _plain(x_projection.selected_time),
                    "selected_model_x": _plain(x_projection.selected_model_x),
                    "segment_ids": _plain(x_projection.segment_ids),
                    "segment_keys": _plain(x_projection.segment_keys),
                    "discontinuity_ids": _plain(x_projection.discontinuity_ids),
                }
            )
        setup_dataset = request_datasets.get(dataset_id)
        mapped_dataset = mapped_datasets.get(dataset_id)
        if setup_dataset is not None and mapped_dataset is not None and setup_dataset.parametric is not None:
            x_name = setup_dataset.parametric.x_name
            raw_id = next(
                (
                    raw_observation_id
                    for raw_observation_id, target in setup_dataset.observation_targets
                    if target == x_name
                ),
                None,
            )
            source = next(
                (
                    observation
                    for observation in mapped_dataset.observations
                    if observation.raw_observation_id == raw_id
                ),
                None,
            )
            if source is not None:
                by_row = {
                    int(row): (float(time), float(value))
                    for row, time, value in zip(source.source_rows, source.t, source.y, strict=True)
                }
                selected_rows = tuple(int(row) for row in info.x_source_rows)
                x_payload.update(
                    {
                        "name": x_name,
                        "raw_observation_id": source.raw_observation_id,
                        "raw_column": source.raw_column,
                        "selected_values": [by_row[row][1] for row in selected_rows],
                    }
                )
        metrics.append(
            {
                "dataset_id": dataset_id,
                "dataset_display_name": display_names.get(dataset_id, dataset_id),
                "weighted_sse": _metric(info.weighted_residual_sse),
                "weighted_mse": _metric(info.weighted_residual_mse),
                "weighted_rmse": _metric(info.weighted_rmse),
                "weighted_mae": _metric(info.weighted_mae),
                "r_squared_by_series": {
                    str(name): _metric(value)
                    for name, value in sorted(info.r_squared_by_series.items())
                },
                "effective_weight": float(info.weight),
                "n_points": int(info.n_points),
                "x": x_payload,
            }
        )
    return {
        "status": result.status,
        "message": str(result.message),
        "optimizer_converged": bool(result.optimizer_converged),
        "diagnostic_only": bool(result.diagnostic_only),
        "optimizer_diagnostic": None if result.optimizer_diagnostic is None else _plain(result.optimizer_diagnostic),
        "optimizer_vector": _plain(result.optimizer_vector),
        "scipy_nfev": int(result.scipy_nfev),
        "terminal_replay_count": int(result.terminal_replay_count),
        "objective_call_count": result.objective_call_count,
        "primary_model": str(result.primary_model),
        "primary_model_details": _plain(result.primary_model_details),
        "weighted_sse": _metric(result.weighted_residual_sse),
        "weighted_mse": _metric(result.weighted_residual_mse),
        "shared_parameters": _plain(result.shared_params),
        "dataset_parameters": _plain(result.dataset_params),
        "metrics_by_dataset": metrics,
        "centred_fixed_effects_settlements": _plain(result.centred_fixed_effects_settlements),
        "shared_refit_provenance": _plain(result.shared_refit_provenance),
        "profile_likelihood": _profile_projection(candidate),
        "residual_bootstrap": _bootstrap_projection(candidate),
        "model_identity": _model_identity_projection(result),
        "simulation_results": _simulation_projection(result),
    }


def _estimate_rows(candidate: FitCandidate) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    request = candidate.request
    result = candidate.result
    included_ids = tuple(sorted(item.dataset_id for item in request.datasets if item.included))
    display_names = {item.dataset_id: item.display_name for item in candidate.origin.mapped_datasets}
    origin_parameters = {item.name: item for item in candidate.origin.parameter_defaults}
    result_shared = dict(result.shared_params)
    result_dataset = {dataset_id: dict(values) for dataset_id, values in result.dataset_params.items()}
    settlements = {item.parameter_name: item for item in result.centred_fixed_effects_settlements}
    promotion_map: Mapping[str, object] = {}
    raw_promotions = result.shared_refit_provenance.get("promotions") if isinstance(result.shared_refit_provenance, Mapping) else None
    if isinstance(raw_promotions, Mapping):
        promotion_map = raw_promotions
    local_names = set(getattr(request.fit_local_admission, "scalar_names", ()) or ())
    converged = bool(result.optimizer_converged)
    suffix = "" if converged else "; optimizer did not converge; estimates are diagnostic only"
    shared: list[dict[str, object]] = []
    individual: list[dict[str, object]] = []

    def append_row(target: list[dict[str, object]], **row: object) -> None:
        target.append({str(key): _plain(value) for key, value in row.items()})

    for parameter in request.parameters:
        configuration = parameter.selected_config
        origin_parameter = origin_parameters.get(parameter.name)
        if origin_parameter is None:
            captured_value = configuration.value
            captured_source = "Fit setup value"
        elif parameter.scope in {"shared", "centred_fixed_effects"}:
            captured_value = origin_parameter.shared_config.value
            captured_source = "captured project Shared default"
        else:
            captured_values = dict(origin_parameter.per_dataset_config.dataset_values)
            captured_value = captured_values.get(included_ids[0], origin_parameter.per_dataset_config.value)
            captured_source = "captured project Per-dataset default"
        fit_local = parameter.name in local_names or parameter.fit_local_provenance is not None
        if parameter.scope == "centred_fixed_effects":
            settlement = settlements.get(parameter.name)
            if settlement is None:
                raise ValueError(f"canonical result omitted centred settlement {parameter.name!r}")
            value = float(settlement.overall_decoded_value)
            nominal = float(configuration.value)
            absolute, percentage, unavailable = _change(value, nominal)
            append_row(
                shared,
                kind="centred_overall", name=parameter.name,
                display_name=f"Overall · {parameter.display_name}", scope="Overall", unit=parameter.unit,
                captured_value=captured_value, nominal_value=nominal, fitted_value=value,
                minimum=configuration.minimum, maximum=configuration.maximum,
                bound_state=fit_bound_state(value, configuration.minimum, configuration.maximum),
                absolute_change=absolute, percentage_change=percentage,
                percentage_unavailable_reason=unavailable, valid=converged,
                validity_diagnostic="decoded centred overall value is finite and within configured bounds" + suffix,
                dataset_ids=list(settlement.dataset_ids), log_scale=configuration.log10,
                captured_value_source=captured_source, nominal_value_source="Fit setup",
            )
            deviation_bounds = (math.log10(configuration.maximum) - math.log10(configuration.minimum) if configuration.log10 else configuration.maximum - configuration.minimum)
            for deviation in settlement.deviations:
                value = float(deviation.decoded_value)
                absolute, percentage, unavailable = _change(value, 0.0)
                dataset_id = str(deviation.coordinate.dataset_id)
                append_row(
                    individual,
                    kind="centred_deviation", name=parameter.name,
                    display_name=f"Deviation · {dataset_id}::{parameter.display_name}",
                    scope="Centred dataset deviation (Fit-only)",
                    unit="decades" if configuration.log10 else parameter.unit,
                    captured_value=0.0, nominal_value=0.0, fitted_value=value,
                    minimum=-deviation_bounds, maximum=deviation_bounds,
                    bound_state=fit_bound_state(value, -deviation_bounds, deviation_bounds),
                    absolute_change=absolute, percentage_change=percentage,
                    percentage_unavailable_reason=unavailable, valid=converged,
                    validity_diagnostic="centred optimizer-scale deviation; derived reference included and no deviation is publishable" + suffix,
                    dataset_id=dataset_id, dataset_ids=[dataset_id],
                    dataset_display_name=display_names.get(dataset_id), log_scale=configuration.log10,
                    captured_value_source="centred terminal settlement", nominal_value_source="centred zero-deviation seed",
                )
            continue
        if parameter.scope == "shared":
            value = float(result_shared.get(parameter.name, configuration.value))
            nominal = float(configuration.value)
            promotion = promotion_map.get(parameter.name)
            if isinstance(promotion, Mapping) and "seed_value" in promotion:
                nominal = float(promotion["seed_value"])
                nominal_source = "admitted Shared-refit central-tendency seed"
            else:
                nominal_source = "Fit-local setup admission" if fit_local else "Fit setup"
            absolute, percentage, unavailable = _change(value, nominal)
            append_row(
                shared,
                kind="shared_parameter", name=parameter.name, display_name=parameter.display_name,
                scope="Shared Fit-local scalar" if fit_local else "Shared", unit=parameter.unit,
                captured_value=captured_value, nominal_value=nominal, fitted_value=value,
                minimum=configuration.minimum, maximum=configuration.maximum,
                bound_state=fit_bound_state(value, configuration.minimum, configuration.maximum),
                absolute_change=absolute, percentage_change=percentage,
                percentage_unavailable_reason=unavailable, valid=converged or not configuration.fit,
                validity_diagnostic="finite and within configured bounds" + ("; Fit-local supplemental scalar provenance" if fit_local else "") + (suffix if configuration.fit else ""),
                log_scale=configuration.log10,
                captured_value_source="Fit-local starting value (not a project default)" if fit_local else captured_source,
                nominal_value_source=nominal_source,
            )
            continue
        values = dict(configuration.dataset_values)
        captured_values = dict(origin_parameter.per_dataset_config.dataset_values) if origin_parameter is not None else values
        result_values = result_dataset.get(parameter.name, {})
        for dataset_id in included_ids:
            value = float(result_values.get(dataset_id, values[dataset_id]))
            nominal = float(values[dataset_id])
            captured = float(captured_values.get(dataset_id, nominal))
            absolute, percentage, unavailable = _change(value, nominal)
            append_row(
                individual,
                kind="per_dataset_parameter", name=parameter.name, display_name=parameter.display_name,
                scope="Per-dataset Fit-local scalar" if fit_local else "Per-dataset", unit=parameter.unit,
                captured_value=captured, nominal_value=nominal, fitted_value=value,
                minimum=configuration.minimum, maximum=configuration.maximum,
                bound_state=fit_bound_state(value, configuration.minimum, configuration.maximum),
                absolute_change=absolute, percentage_change=percentage,
                percentage_unavailable_reason=unavailable, valid=converged or not configuration.fit,
                validity_diagnostic="finite and within configured bounds" + ("; Fit-local supplemental scalar provenance" if fit_local else "") + (suffix if configuration.fit else ""),
                dataset_id=dataset_id, dataset_ids=[dataset_id], dataset_display_name=display_names.get(dataset_id),
                log_scale=configuration.log10,
                captured_value_source="Fit-local starting value (not a project default)" if fit_local else captured_source,
                nominal_value_source="Fit-local setup admission" if fit_local else "Fit setup",
            )
    return shared, individual


def _initial_rows(candidate: FitCandidate) -> list[dict[str, object]]:
    request = candidate.request
    result = candidate.result
    origin = candidate.origin
    by_dataset = {
        dataset_id: {
            name[len(INITIAL_PREFIX):]: float(value)
            for name, value in values.items()
            if str(name).startswith(INITIAL_PREFIX)
        }
        for dataset_id, values in result.dataset_params.items()
    }
    mapped_by_set: dict[str, list[object]] = {}
    for mapped in origin.mapped_datasets:
        mapped_by_set.setdefault(mapped.set_id, []).append(mapped)
    scopes = dict(request.initial_scopes)
    rows: list[dict[str, object]] = []
    converged = bool(result.optimizer_converged)
    suffix = "" if converged else "; optimizer did not converge; estimates are diagnostic only"
    for item in request.initials:
        if not item.fit or scopes.get(item.species) != "local":
            continue
        for mapped in mapped_by_set.get(item.set_id, ()):
            if item.species not in by_dataset.get(mapped.dataset_id, {}):
                continue
            value = by_dataset[mapped.dataset_id][item.species]
            absolute, percentage, unavailable = _change(value, item.initial)
            captured = dict(mapped.initial_values).get(item.species, item.initial)
            rows.append(
                {
                    "kind": "initial", "name": item.species, "display_name": item.species,
                    "scope": "Per-dataset IC", "unit": "model concentration",
                    "captured_value": captured, "nominal_value": item.initial, "fitted_value": value,
                    "minimum": item.minimum, "maximum": item.maximum,
                    "bound_state": fit_bound_state(value, item.minimum, item.maximum),
                    "absolute_change": absolute, "percentage_change": percentage,
                    "percentage_unavailable_reason": unavailable, "valid": converged,
                    "validity_diagnostic": "finite, nonnegative, and within configured bounds" + suffix,
                    "set_id": item.set_id, "dataset_ids": [mapped.dataset_id],
                    "dataset_display_name": mapped.display_name, "log_scale": bool(item.log10),
                    "set_display_name": mapped.set_name,
                    "captured_value_source": "captured mapped dataset initial value", "nominal_value_source": "Fit setup",
                }
            )
    for item in request.global_initials:
        if not item.fit or scopes.get(item.species) != "global":
            continue
        for mapped_items in mapped_by_set.values():
            for mapped in mapped_items:
                if item.species not in by_dataset.get(mapped.dataset_id, {}):
                    continue
                value = by_dataset[mapped.dataset_id][item.species]
                absolute, percentage, unavailable = _change(value, item.initial)
                captured = dict(mapped.initial_values).get(item.species, item.initial)
                rows.append(
                    {
                        "kind": "initial", "name": item.species, "display_name": item.species,
                        "scope": "Shared Fit IC coordinate; independently applied per set", "unit": "model concentration",
                        "captured_value": captured, "nominal_value": item.initial, "fitted_value": value,
                        "minimum": item.minimum, "maximum": item.maximum,
                        "bound_state": fit_bound_state(value, item.minimum, item.maximum),
                        "absolute_change": absolute, "percentage_change": percentage,
                        "percentage_unavailable_reason": unavailable, "valid": converged,
                        "validity_diagnostic": "finite, nonnegative, and within configured bounds" + suffix,
                        "set_id": mapped.set_id, "dataset_ids": [mapped.dataset_id],
                        "dataset_display_name": mapped.display_name, "log_scale": bool(item.log10),
                        "set_display_name": mapped.set_name,
                        "captured_value_source": "captured mapped dataset initial value", "nominal_value_source": "Fit setup",
                    }
                )
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
        log_scale = bool(rows[0].get("log_scale"))
        geometric_mean = None
        geometric_spread = None
        unavailable_reason: str | None = "requires positive log-scale estimates"
        if log_scale and all(value > 0.0 for value in values):
            logs = tuple(math.log(value) for value in values)
            geometric_mean = math.exp(statistics.fmean(logs))
            geometric_spread = math.exp(statistics.stdev(logs)) if count > 1 else None
            unavailable_reason = None if count > 1 else "geometric spread requires at least two estimates"
        def percentile(percentile: float) -> float:
            position = (count - 1) * percentile
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return values[lower]
            fraction = position - lower
            return values[lower] * (1.0 - fraction) + values[upper] * fraction
        aggregates.append(
            {
                "name": name, "display_name": rows[0]["display_name"], "unit": rows[0]["unit"], "n": count,
                "arithmetic_mean": mean, "sample_sd": sd,
                "standard_error": None if sd is None else sd / math.sqrt(count),
                "median": statistics.median(values), "iqr": percentile(0.75) - percentile(0.25),
                "minimum": values[0], "maximum": values[-1],
                "geometric_mean": geometric_mean, "geometric_spread": geometric_spread,
                "geometric_unavailable_reason": unavailable_reason,
            }
        )
    return aggregates


def _weighting_projection(candidate: FitCandidate, metrics: list[dict[str, object]]) -> dict[str, object]:
    request = candidate.request
    setup_by_id = {item.dataset_id: item for item in request.datasets if item.included}
    metric_by_id = {str(item["dataset_id"]): item for item in metrics}
    datasets: list[dict[str, object]] = []
    for dataset_id in sorted(metric_by_id):
        setup = setup_by_id[dataset_id]
        metric = metric_by_id[dataset_id]
        effective = float(metric["effective_weight"])
        configured = {str(name): float(value) for name, value in setup.target_weights}
        normalized = finite_positive_mean_normalize(configured)
        targets = [
            {
                "target_key": f"{dataset_id}::{target}", "target": target,
                "configured_target_weight": configured[target], "normalized_target_weight": normalized[target],
                "target_residual_multiplier": math.sqrt(normalized[target]),
                "combined_residual_multiplier": math.sqrt(effective * normalized[target]),
            }
            for target in setup.target_species
        ]
        x = metric["x"]
        x_payload = None
        if setup.parametric is not None:
            x_name = setup.parametric.x_name
            x_payload = {
                "name": x_name, "configured_target_weight": configured[x_name],
                "normalized_target_weight": normalized[x_name],
                "target_residual_multiplier": math.sqrt(normalized[x_name]),
                "combined_residual_multiplier": math.sqrt(effective * normalized[x_name]),
                "point_count": x["point_count"], "weighted_sse": x["weighted_sse"],
                "weighted_mse": x["weighted_mse"], "weighted_rmse": x["weighted_rmse"], "weighted_mae": x["weighted_mae"],
                "selected_time": x.get("selected_time"), "selected_model_x": x.get("selected_model_x"),
                "source_rows": x["source_rows"], "delta_x": x["delta_x"],
                "match_kinds": x["match_kinds"], "sample_kinds": x["sample_kinds"],
                "segment_ids": x.get("segment_ids", []), "segment_keys": x.get("segment_keys", []),
                "discontinuity_ids": x.get("discontinuity_ids", []),
            }
        dataset_payload: dict[str, object] = {
            "dataset_id": dataset_id, "point_count": int(metric["n_points"]),
            "normalized_effective_dataset_weight": effective, "dataset_residual_multiplier": math.sqrt(effective),
            "targets": targets, "x": x_payload,
        }
        if str(request.dataset_weight_mode) == "user":
            dataset_payload["original_user_dataset_weight"] = float(setup.weight)
        datasets.append(dataset_payload)
    return {
        "dataset_weight_mode": str(request.dataset_weight_mode),
        "combined_residual_weighting_rule": "weighted_residual = raw_residual * sqrt(normalized_effective_dataset_weight) * sqrt(normalized_target_weight)",
        "datasets": datasets,
    }


def _known_input_context(candidate: FitCandidate) -> dict[str, object]:
    origin = candidate.origin
    request = candidate.request
    settings = origin.scientific_settings
    return {
        "temperature_K": float(settings.temperature), "relative_tolerance": float(settings.rtol),
        "absolute_tolerance": float(settings.atol), "analytic_jacobian_policy": bool(settings.analytic_jacobian_policy),
        "solver": str(request.solver),
        "included_dataset_ids": list(request.dispatch.included_dataset_ids),
        "dataset_conditions": [
            {"dataset_id": item.dataset_id, "included": bool(item.included), "resolved_t_start": float(item.t_start), "data_range_min": float(item.data_range_min), "data_range_max": float(item.data_range_max)}
            for item in request.datasets
        ],
        "known_condition_source": "captured Fit origin scientific settings",
    }


def build_fit_action_projection(candidate: FitCandidate) -> dict[str, object]:
    """Build one complete, plain, call-local graph for a Fit terminal action.

    Stable top-level keys are ``schema_version``, ``request``, ``result``,
    ``resolved_auto_modes``, ``estimates``, ``metrics``, ``weighting``,
    ``known_input_context``, and ``apply_display_names``.  ``result`` owns
    explicit ``profile_likelihood`` and ``residual_bootstrap`` mappings;
    neither is represented by a generic extension envelope.  ``copy_hash``
    is added only for immediate action use.
    """

    if not isinstance(candidate, FitCandidate):
        raise TypeError("Fit action projection requires FitCandidate.")
    request = _request_projection(candidate)
    result = _result_projection(candidate)
    shared, individual = _estimate_rows(candidate)
    initials = _initial_rows(candidate)
    metrics = list(result["metrics_by_dataset"])
    projection: dict[str, object] = {
        "schema_version": 1, "request": request, "result": result,
        "resolved_auto_modes": request["resolved_auto_modes"],
        "estimates": {"shared": shared, "per_dataset": individual, "initial": initials, "aggregates": _aggregate_rows(individual)},
        "metrics": metrics, "weighting": _weighting_projection(candidate, metrics),
        "known_input_context": _known_input_context(candidate),
        "apply_display_names": {"parameters": [{"name": parameter.name, "display_name": parameter.display_name, "scope": parameter.scope} for parameter in sorted(candidate.request.parameters, key=lambda item: item.name)], "initials": [{"set_id": item.set_id, "species": item.species} for item in candidate.request.initials], "observable_names": sorted(getattr(candidate.request.fit_local_admission, "observable_names", ()) or ())},
    }
    projection["copy_hash"] = fit_action_projection_hash(projection)
    return projection


def _check_projection(projection: Mapping[str, object]) -> None:
    if not isinstance(projection, Mapping):
        raise TypeError("Fit action formatters require a projection mapping.")


_HASH_BRANCH_KEYS = (
    "request",
    "result",
    "resolved_auto_modes",
    "estimates",
    "metrics",
    "weighting",
    "known_input_context",
)
_HASH_EXCLUDED_KEYS = frozenset(
    {
        "schema_version",
        "copy_hash",
        "apply_display_names",
        "display_name",
        "dataset_display_name",
        "set_display_name",
        "set_name",
        "tab_id",
        "tab_name",
        "presentation_boundaries",
    }
)


def _scientific_hash_value(value: object) -> object:
    """Select scientific values from one existing projection graph.

    The omitted names are the complete layout/display/internal surface.  All
    other admitted inputs, source and model arrays, result metrics, uncertainty,
    and provenance remain hash inputs.  This is a transient serialization view;
    it is never retained beside the canonical projection.
    """

    plain = _plain(value)
    if isinstance(plain, Mapping):
        return {
            str(key): _scientific_hash_value(item)
            for key, item in plain.items()
            if str(key) not in _HASH_EXCLUDED_KEYS
        }
    if isinstance(plain, list):
        return [_scientific_hash_value(item) for item in plain]
    return plain


def _scientific_hash_branch(projection: Mapping[str, object]) -> dict[str, object]:
    """Select the canonical projection branches that define a Fit run."""

    return {
        key: _scientific_hash_value(projection[key])
        for key in _HASH_BRANCH_KEYS
        if key in projection
    }


def _result_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """Select a projection result or an already-selected Summary mapping."""

    result = value.get("result")
    if isinstance(result, Mapping):
        return result
    return value


def fit_action_projection_hash(projection: Mapping[str, object]) -> str:
    """Return the deterministic twelve-character SHA-256 Copy hash."""

    _check_projection(projection)
    encoded = json.dumps(
        _scientific_hash_branch(projection),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def fit_action_projection_payload(projection: Mapping[str, object]) -> dict[str, object]:
    """Return the plain Copy Run payload, excluding internal hash material."""

    _check_projection(projection)
    payload = {str(key): _plain(value) for key, value in projection.items()}
    payload["copy_hash"] = fit_action_projection_hash(projection)
    return payload


def fit_action_projection_json(projection: Mapping[str, object], *, indent: int | None = 2) -> str:
    """Serialize Copy Run JSON from the same graph used by report and Summary."""

    return json.dumps(fit_action_projection_payload(projection), sort_keys=True, indent=indent, ensure_ascii=True)


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


def _estimate_line(row: Mapping[str, object]) -> str:
    percentage = _display(row.get("percentage_change")) + "%" if row.get("percentage_change") is not None else str(row.get("percentage_unavailable_reason") or "unavailable")
    location = f" [{row['dataset_id']}]" if row.get("dataset_id") is not None else ""
    if row.get("set_id") is not None:
        location = f" [{row['set_id']}]"
    return f"{row.get('display_name', row.get('name'))}{location}: fitted={_display(row.get('fitted_value'))}; nominal={_display(row.get('nominal_value'))}; Δ={_display(row.get('absolute_change'))}; Δ%={percentage}; bounds={_display(row.get('minimum'))}..{_display(row.get('maximum'))} ({row.get('bound_state')})"


def fit_action_projection_profile_lines(projection: Mapping[str, object]) -> tuple[str, ...]:
    """Render direct Profile facts from a projection or Summary mapping."""

    _check_projection(projection)
    profile = _result_mapping(projection).get("profile_likelihood")
    if not isinstance(profile, Mapping):
        raise ValueError("Profile projection is malformed.")
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
            lines.append(
                f"  {interval.get('display_name', interval.get('coordinate'))}: "
                f"status={interval.get('status')}; lower={_display(interval.get('lower'))}; "
                f"upper={_display(interval.get('upper'))}; "
                f"lower outcome={interval.get('lower_outcome')}; upper outcome={interval.get('upper_outcome')}"
            )
    return tuple(lines)


def fit_action_projection_bootstrap_lines(projection: Mapping[str, object]) -> tuple[str, ...]:
    """Render direct Bootstrap facts from a projection or Summary mapping."""

    _check_projection(projection)
    bootstrap = _result_mapping(projection).get("residual_bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("Bootstrap projection is malformed.")
    lines = [f"Residual Bootstrap: status={bootstrap.get('status')}"]
    if not bootstrap.get("requested"):
        return (*lines, "  not requested")
    if bootstrap.get("method"):
        lines.append(f"  method: {bootstrap['method']}")
    lines.append(
        f"  replicates: {bootstrap.get('successful_replicates')} successful, "
        f"{bootstrap.get('failed_replicates')} failed of {bootstrap.get('replicates')}"
    )
    if bootstrap.get("unavailable_reason"):
        lines.append(f"  unavailable reason: {bootstrap['unavailable_reason']}")
    if bootstrap.get("diagnostic"):
        lines.append(f"  diagnostic: {_display(bootstrap['diagnostic'])}")
    for interval in bootstrap.get("intervals", ()):
        if isinstance(interval, Mapping):
            lines.append(
                f"  {interval.get('display_name', interval.get('coordinate'))}: "
                f"status={interval.get('status')}; lower={_display(interval.get('lower'))}; "
                f"upper={_display(interval.get('upper'))}; reason={_display(interval.get('reason'))}"
            )
    return tuple(lines)


def fit_action_projection_primary_model_lines(projection: Mapping[str, object]) -> tuple[str, ...]:
    _check_projection(projection)
    result = _result_mapping(projection)
    return (f"Primary Fit model: {result.get('primary_model') or 'ordinary'}",)


def _metric_text_payload(value: object) -> str:
    if not isinstance(value, Mapping):
        return _display(value)
    if bool(value.get("available")):
        return _display(value.get("value"))
    return f"unavailable ({value.get('unavailable_reason')})"


def fit_action_projection_report_text(projection: Mapping[str, object]) -> str:
    """Render terminal report text from the canonical action graph."""

    _check_projection(projection)
    result = _result_mapping(projection)
    estimates = projection["estimates"]
    if not isinstance(result, Mapping) or not isinstance(estimates, Mapping):
        raise ValueError("Fit action projection is malformed.")
    lines = [f"Fit status: {result.get('status')}", f"Message: {result.get('message') or 'none'}", *fit_action_projection_primary_model_lines(projection), f"Optimizer converged: {str(result.get('optimizer_converged')).lower()}", f"Diagnostic only: {str(result.get('diagnostic_only')).lower()}", f"Weighted SSE: {_metric_text_payload(result.get('weighted_sse'))}", f"Weighted MSE: {_metric_text_payload(result.get('weighted_mse'))}"]
    if result.get("optimizer_diagnostic"):
        lines.append(f"Optimizer diagnostic: {_display(result['optimizer_diagnostic'])}")
    for title, key in (("Shared estimates", "shared"), ("Per-dataset estimates", "per_dataset"), ("Initial estimates", "initial")):
        rows = estimates.get(key, ())
        if rows:
            lines.append(f"{title}:")
            lines.extend(f"  {_estimate_line(row)}" for row in rows if isinstance(row, Mapping))
    if estimates.get("aggregates"):
        lines.append("Aggregate metrics:")
        for row in estimates["aggregates"]:
            lines.append(f"  {row.get('display_name', row.get('name'))}: n={row.get('n')}; mean={_display(row.get('arithmetic_mean'))}; median={_display(row.get('median'))}; IQR={_display(row.get('iqr'))}; SD={_display(row.get('sample_sd'))}")
    lines.append("Dataset metrics:")
    for row in projection.get("metrics", ()):
        if not isinstance(row, Mapping):
            continue
        lines.append(f"  {row.get('dataset_display_name', row.get('dataset_id'))}: weighted SSE={_metric_text_payload(row.get('weighted_sse'))}; weighted MSE={_metric_text_payload(row.get('weighted_mse'))}; weighted RMSE={_metric_text_payload(row.get('weighted_rmse'))}; weighted MAE={_metric_text_payload(row.get('weighted_mae'))}; effective weight={_display(row.get('effective_weight'))}")
        r_squared = row.get("r_squared_by_series")
        if isinstance(r_squared, Mapping):
            lines.extend(f"    R² {name}={_metric_text_payload(metric)}" for name, metric in sorted(r_squared.items()))
    weighting = projection.get("weighting")
    if isinstance(weighting, Mapping):
        lines.append(f"Weighting mode: {weighting.get('dataset_weight_mode')}")
        lines.append(f"Weighting rule: {weighting.get('combined_residual_weighting_rule')}")
    lines.append("Profile likelihood:")
    lines.extend(f"  {line}" for line in fit_action_projection_profile_lines(projection))
    lines.append("Residual Bootstrap:")
    lines.extend(f"  {line}" for line in fit_action_projection_bootstrap_lines(projection))
    known = projection.get("known_input_context")
    if isinstance(known, Mapping):
        lines.append("Known input context:")
        for key in (
            "temperature_K",
            "relative_tolerance",
            "absolute_tolerance",
            "analytic_jacobian_policy",
            "solver",
            "included_dataset_ids",
        ):
            if key in known:
                lines.append(f"  {key}: {_display(known[key])}")
        dataset_conditions = known.get("dataset_conditions")
        if isinstance(dataset_conditions, (tuple, list)):
            for condition in dataset_conditions:
                if isinstance(condition, Mapping):
                    lines.append(
                        "  dataset condition: "
                        f"{condition.get('dataset_id')}; "
                        f"included={_display(condition.get('included'))}; "
                        f"t_start={_display(condition.get('resolved_t_start'))}; "
                        f"range={_display((condition.get('data_range_min'), condition.get('data_range_max')))}"
                    )
    shared_refit = result.get("shared_refit_provenance")
    if isinstance(shared_refit, Mapping) and shared_refit:
        lines.append("Shared-refit provenance:")
        for key in (
            "execution",
            "terminal_route",
            "starting_value_authority",
            "terminal_value_authority",
        ):
            if key in shared_refit:
                lines.append(f"  {key}: {_display(shared_refit[key])}")
        promotions = shared_refit.get("promotions")
        if isinstance(promotions, Mapping):
            lines.append(f"  promoted parameters: {_display(sorted(str(name) for name in promotions))}")
    return "\n".join(lines)


def fit_action_projection_summary(projection: Mapping[str, object]) -> dict[str, object]:
    """Return Results Summary tables/text from the same action graph."""

    _check_projection(projection)
    estimates = projection.get("estimates")
    result = projection.get("result")
    if not isinstance(estimates, Mapping) or not isinstance(result, Mapping):
        raise ValueError("Fit action projection is malformed.")
    return {
        "copy_hash": fit_action_projection_hash(projection),
        "status": result.get("status"), "optimizer_converged": result.get("optimizer_converged"), "diagnostic_only": result.get("diagnostic_only"),
        "weighted_sse": result.get("weighted_sse"), "weighted_mse": result.get("weighted_mse"),
        "shared_estimates": estimates.get("shared", []), "per_dataset_estimates": estimates.get("per_dataset", []), "initial_estimates": estimates.get("initial", []), "aggregates": estimates.get("aggregates", []),
        "metrics_by_dataset": projection.get("metrics", []), "weighting": projection.get("weighting", {}),
        "profile_likelihood": result.get("profile_likelihood", {}), "residual_bootstrap": result.get("residual_bootstrap", {}),
        "uncertainty": {
            "profile_likelihood": result.get("profile_likelihood", {}),
            "residual_bootstrap": result.get("residual_bootstrap", {}),
        },
        "known_input_context": projection.get("known_input_context", {}),
        "shared_refit_provenance": result.get("shared_refit_provenance", {}),
        "report_text": fit_action_projection_report_text(projection),
    }
