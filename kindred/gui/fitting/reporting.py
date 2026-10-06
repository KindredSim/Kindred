"""Synchronous human-readable Fit presentation from retained typed facts.

``FitCandidate`` remains the sole terminal science owner. This module builds
small call-local row/text values for the physical Results views; it does not
define a versioned action graph, identity, or serialization format.
"""

from __future__ import annotations

import dataclasses
import html
import math
import statistics
from collections.abc import Mapping

import numpy as np

from kindred.core.analysis.fit_statistics import FitMetric
from kindred.core.analysis.global_fit_execution import FitCoordinate
from kindred.core.quantity_units import format_unit_for_display
from kindred.core.simulation_failure import FactualCause
from kindred.gui.display_name_policy import scientific_symbol_html
from kindred.gui.fitting.constants import FIT_DATASET_WEIGHT_MODE_LABELS, INITIAL_PREFIX
from kindred.gui.fitting.launch import FitOriginSnapshot, FitParameterVariable, fit_uncertainty_coordinates
from kindred.gui.fitting.execution import (
    FitCandidate, FitLiveUpdate, _fit_coordinate_display_label, fit_bound_state,
)

__all__ = [
    "fit_candidate_presentation",
    "fit_identity_text",
    "fit_overview",
    "fit_completion_presentation",
    "fit_live_presentation",
]


def fit_identity_text(text: str, origin: FitOriginSnapshot) -> str:
    """Replace retained dataset and set identities with their readable names."""

    for mapped in origin.mapped_datasets:
        text = text.replace(mapped.dataset_id, mapped.display_name)
    for mapped in origin.mapped_datasets:
        text = text.replace(mapped.set_id, mapped.set_name)
    return text


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
    if isinstance(value, FactualCause):
        return _plain(value.to_payload())
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
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


def _metric(value: FitMetric | None) -> object:
    if value is None:
        return None
    if not isinstance(value, FitMetric):
        raise TypeError("Fit metrics require FitMetric values.")
    return _plain(value.to_payload())


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
        return "—"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_display(item) for item in value) + "]"
    return str(value)


def _profile_endpoint(endpoint: object) -> dict[str, object]:
    return {"reason": endpoint.reason}


def _profile_interval(coordinate: FitCoordinate, interval: object) -> dict[str, object]:
    return {
        "coordinate": coordinate,
        "status": interval.status,
        "lower": interval.lower,
        "upper": interval.upper,
        "lower_outcome": interval.lower_outcome,
        "upper_outcome": interval.upper_outcome,
        "lower_resolution": _profile_endpoint(interval.lower_resolution),
        "upper_resolution": _profile_endpoint(interval.upper_resolution),
        "endpoint_messages": _plain(interval.endpoint_messages),
    }


def _roster_positions(setup) -> dict[FitCoordinate, int]:
    """Uncertainty coordinates in roster order: parameters, then datasets in Fit order."""
    return {
        coordinate: index
        for index, coordinate in enumerate(fit_uncertainty_coordinates(setup.parameters, setup.datasets))
    }


def _profile_data(candidate: FitCandidate) -> dict[str, object]:
    result = candidate.result.profile_likelihood_result
    request_setup = candidate.request.setup
    if result is None:
        requested = request_setup.profile_likelihood is not None
        return {
            "requested": requested,
            "status": "not_requested" if not requested else "unavailable",
            "unavailable_reason": None if not requested else "Profile likelihood result is unavailable.",
            "diagnostic": None,
            "intervals": [],
        }
    roster = _roster_positions(request_setup)
    return {
        "requested": True,
        "status": result.status,
        "method": result.method,
        "confidence_level": result.confidence_level,
        "facts": (
            ("Likelihood-ratio threshold", result.likelihood_ratio_threshold),
            ("Likelihood model", result.likelihood_model),
            ("Observations", result.observation_count),
        ),
        "unavailable_reason": result.unavailable_reason,
        "diagnostic": _plain(result.diagnostic),
        "intervals": [
            _profile_interval(coordinate, interval)
            for coordinate, interval in sorted(result.intervals.items(), key=lambda item: roster[item[0]])
        ],
    }


def _bootstrap_interval(coordinate: FitCoordinate, interval: object) -> dict[str, object]:
    return {
        "coordinate": coordinate,
        "status": interval.status,
        "lower": interval.lower,
        "upper": interval.upper,
        "reason": interval.reason,
    }


def _bootstrap_data(candidate: FitCandidate) -> dict[str, object]:
    result = candidate.result.residual_bootstrap_result
    request_setup = candidate.request.setup
    if result is None:
        requested = request_setup.residual_bootstrap is not None
        return {
            "requested": requested,
            "status": "not_requested" if not requested else "unavailable",
            "unavailable_reason": None if not requested else "Residual Bootstrap result is unavailable.",
            "diagnostic": None,
            "intervals": [],
        }
    roster = _roster_positions(request_setup)
    return {
        "requested": True,
        "status": result.status,
        "method": result.method,
        "replicates": result.replicates,
        "confidence_level": result.confidence_level,
        "facts": (
            ("Seed", result.seed),
            ("Resampling", result.resampling),
            ("Residual centering", result.residual_centering),
            ("Required successful refits", result.required_successes),
        ),
        "successful_replicates": result.successful_replicates,
        "failed_replicates": result.failed_replicates,
        "intervals": [
            _bootstrap_interval(coordinate, interval)
            for coordinate, interval in sorted(result.intervals.items(), key=lambda item: roster[item[0]])
        ],
        "first_failed_replicate_diagnostic": _plain(result.first_failed_replicate_diagnostic),
        "first_failed_replicate_unavailable": _plain(result.first_failed_replicate_unavailable),
        "unavailable_reason": result.unavailable_reason,
        "diagnostic": _plain(result.diagnostic),
    }


def _dataset_metrics(candidate: FitCandidate) -> list[dict[str, object]]:
    result = candidate.result
    display_names = {mapped.dataset_id: mapped.display_name for mapped in candidate.origin.mapped_datasets}
    positions = {dataset_id: index for index, dataset_id in enumerate(display_names)}
    metrics: list[dict[str, object]] = []
    for info in sorted(result.dataset_info, key=lambda item: positions[str(item.dataset_id)]):
        dataset_id = str(info.dataset_id)
        metric: dict[str, object] = {
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
    request_setup = request.setup
    result = candidate.result
    terminal_setup = candidate.setup_snapshot
    included_ids = tuple(
        item.dataset_id for item in terminal_setup.datasets if item.included
    )
    display_names = {item.dataset_id: item.display_name for item in candidate.origin.mapped_datasets}
    request_parameters = {item.name: item for item in request_setup.parameters}
    settlements = {item.parameter_name: item for item in result.centred_fixed_effects_settlements}
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
        if parameter.scope == "centred_fixed_effects":
            # Fixed centered parameters are dispatched as fixed dataset values
            # and have no settlement or presentation row.
            if not configuration.fit:
                continue
            starts = dict(launch_configuration.dataset_values)
            raw_starts = {
                dataset_id: math.log10(starts[dataset_id]) if configuration.log10 else starts[dataset_id]
                for dataset_id in included_ids
            }
            # The overall start is the one the optimizer was dispatched with.
            layout = request.dispatch.ordinary_optimizer_layout
            overall_index = layout.centred_overall_index[parameter.name]
            starting_centre = float(layout.x0[overall_index])
            nominal = float(layout.physical_x0[overall_index])
            settlement = settlements.get(parameter.name)
            if settlement is None:
                raise ValueError(f"canonical result omitted centered settlement {parameter.name!r}")
            value = float(settlement.overall_decoded_value)
            append_row(shared, kind="centred_overall", name=parameter.name,
                display_name=f"Overall · {parameter.display_name}", scope="Overall", unit=parameter.unit,
                nominal_value=nominal, fitted_value=value,
                minimum=configuration.minimum, maximum=configuration.maximum,
                bound_state=fit_bound_state(
                    value,
                    configuration.minimum,
                    configuration.maximum,
                    log_scale=configuration.log10,
                ),
                dataset_ids=list(settlement.dataset_ids))
            for dataset_id in settlement.dataset_ids:
                value = float(settlement.dataset_values[dataset_id])
                append_row(individual, kind="centred_effective", name=parameter.name,
                    display_name=parameter.display_name, scope="Effective value",
                    unit=parameter.unit, fitted_value=value, nominal_value=starts[dataset_id],
                    minimum=configuration.minimum, maximum=configuration.maximum,
                    bound_state=fit_bound_state(value, configuration.minimum, configuration.maximum, configuration.log10),
                    dataset_id=dataset_id, dataset_display_name=display_names[dataset_id])
            deviation_bounds = math.log10(configuration.maximum) - math.log10(configuration.minimum) if configuration.log10 else configuration.maximum - configuration.minimum
            for deviation in settlement.deviations:
                value = float(deviation.decoded_value)
                dataset_id = str(deviation.coordinate.dataset_id)
                nominal_deviation = raw_starts[dataset_id] - starting_centre
                append_row(individual, kind="centred_deviation", name=parameter.name,
                    display_name=parameter.display_name,
                    scope="Log10-coordinate deviation" if configuration.log10 else "Linear-coordinate deviation", unit="decades" if configuration.log10 else parameter.unit,
                    nominal_value=nominal_deviation, fitted_value=value,
                    minimum=-deviation_bounds, maximum=deviation_bounds,
                    bound_state=fit_bound_state(
                        value,
                        -deviation_bounds,
                        deviation_bounds,
                    ),
                    dataset_id=dataset_id, dataset_ids=[dataset_id], dataset_display_name=display_names.get(dataset_id))
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
            append_row(shared, kind="shared_parameter", name=parameter.name, display_name=parameter.display_name,
                scope="Shared", unit=parameter.unit,
                nominal_value=nominal, fitted_value=value,
                minimum=configuration.minimum, maximum=configuration.maximum,
                bound_state=fit_bound_state(
                    value,
                    configuration.minimum,
                    configuration.maximum,
                    log_scale=configuration.log10,
                ))
            continue
        # Fixed Per-dataset coordinates are not terminal fitted values and
        # therefore have no presentation row.  Fitted coordinates below must
        # come from the terminal dataset-first result without nominal fallback.
        if not configuration.fit:
            continue
        values = dict(launch_configuration.dataset_values)
        for dataset_id in included_ids:
            value = _terminal_dataset_value(candidate, dataset_id, parameter.name)
            if dataset_id not in values:
                raise ValueError(
                    f"Fit setup omitted nominal dataset coordinate {dataset_id!r} "
                    f"for {parameter.name!r}."
                )
            nominal = float(values[dataset_id])
            append_row(individual, kind="per_dataset_parameter", name=parameter.name, display_name=parameter.display_name,
                scope="Per-dataset", unit=parameter.unit,
                nominal_value=nominal, fitted_value=value,
                minimum=configuration.minimum, maximum=configuration.maximum,
                bound_state=fit_bound_state(
                    value,
                    configuration.minimum,
                    configuration.maximum,
                    log_scale=configuration.log10,
                ),
                dataset_id=dataset_id, dataset_ids=[dataset_id], dataset_display_name=display_names.get(dataset_id))
    return shared, individual


def _initial_rows(candidate: FitCandidate) -> list[dict[str, object]]:
    request = candidate.request
    request_setup = request.setup
    origin = candidate.origin
    terminal_setup = candidate.setup_snapshot
    included_ids = {item.dataset_id for item in terminal_setup.datasets if item.included}
    launch_initials = {
        (item.set_id, item.species): item for item in request_setup.initials
    }
    mapped_by_set: dict[str, list[object]] = {}
    for mapped in origin.mapped_datasets:
        if mapped.dataset_id not in included_ids:
            continue
        mapped_by_set.setdefault(mapped.set_id, []).append(mapped)
    grouped_pairs = {
        (dataset_id, group.species)
        for group in terminal_setup.shared_initial_groups
        for dataset_id in group.member_dataset_ids
    }
    rows: list[dict[str, object]] = []
    for item in terminal_setup.initials:
        if not item.fit:
            continue
        launch_item = launch_initials.get((item.set_id, item.species), item)
        for mapped in mapped_by_set.get(item.set_id, ()):
            if (mapped.dataset_id, item.species) in grouped_pairs:
                continue
            value = _terminal_dataset_value(
                candidate,
                mapped.dataset_id,
                f"{INITIAL_PREFIX}{item.species}",
            )
            rows.append({"kind": "initial", "name": item.species, "display_name": item.species,
                "scope": "Per-dataset IC", "unit": "M",
                "nominal_value": launch_item.initial, "fitted_value": value, "minimum": launch_item.minimum, "maximum": launch_item.maximum,
                "bound_state": fit_bound_state(
                    value,
                    launch_item.minimum,
                    launch_item.maximum,
                    log_scale=bool(launch_item.log10),
                ),
                "dataset_ids": [mapped.dataset_id], "dataset_display_name": mapped.display_name})
    launch_groups = {
        (item.species, item.origin_dataset_id): item
        for item in request_setup.shared_initial_groups
    }
    for item in terminal_setup.shared_initial_groups:
        if not item.fit:
            continue
        launch_item = launch_groups.get(
            (item.species, item.origin_dataset_id), item
        )
        # Members are listed in Fit dataset order.
        for mapped in origin.mapped_datasets:
            dataset_id = mapped.dataset_id
            if dataset_id not in item.member_dataset_ids or dataset_id not in included_ids:
                continue
            value = _terminal_dataset_value(
                candidate,
                dataset_id,
                f"{INITIAL_PREFIX}{item.species}",
            )
            rows.append({"kind": "initial", "name": item.species, "display_name": item.species,
                "scope": "Shared Fit IC group", "unit": "M",
                "nominal_value": launch_item.initial, "fitted_value": value,
                "minimum": launch_item.minimum, "maximum": launch_item.maximum,
                "bound_state": fit_bound_state(
                    value,
                    launch_item.minimum,
                    launch_item.maximum,
                    log_scale=bool(launch_item.log10),
                ),
                "dataset_ids": [mapped.dataset_id], "dataset_display_name": mapped.display_name})
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
            "arithmetic_mean": mean, "sample_sd": sd,
            "median": statistics.median(values), "iqr": percentile(0.75) - percentile(0.25)})
    return aggregates


def _metric_text(value: object) -> str:
    if not isinstance(value, Mapping):
        return _display(value)
    if bool(value.get("available")):
        return _display(value.get("value"))
    return f"Unavailable: {value.get('unavailable_reason')}"


def _text(value: object) -> str:
    return html.escape(_display(value)).replace("\n", "<br>")


def _unit_html(unit: object) -> str:
    text = str(unit or "")
    return format_unit_for_display(text)


def _table(headers: tuple[str, ...], rows: object, *, rich_columns: tuple[int, ...] = ()) -> str:
    return ('<table cellspacing="0" cellpadding="5"><tr>'
            + "".join(f'<th align="left">{_text(header)}</th>' for header in headers)
            + "</tr>" + "".join("<tr>" + "".join(
                f'<td valign="top">{value if column in rich_columns else _text(value)}</td>'
                for column, value in enumerate(row)
            ) + "</tr>" for row in rows) + "</table>")


def _diagnostic(value: object, origin: FitOriginSnapshot) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        # Human failure chains retain every message, without a nested dict dump.
        parts = [str(value[key]) for key in ("type", "message", "reason") if value.get(key)]
        if value.get("cause"):
            parts.append(_diagnostic(value["cause"], origin))
        if not parts:
            parts = [f"{key.replace('_', ' ')}: {_diagnostic(item, origin)}"
                     for key, item in value.items() if item is not None]
        text = " — ".join(parts)
    elif isinstance(value, (list, tuple)):
        text = "; ".join(_diagnostic(item, origin) for item in value)
    else:
        text = str(value)
    return fit_identity_text(text, origin)


def _analysis_section(title: str, data: Mapping[str, object], origin: FitOriginSnapshot,
                      symbols: Mapping[str, str], coordinate_units: Mapping[tuple[str, str, str | None], str]) -> str:
    if not data["requested"]:
        return ""
    section = f"<h3>{_text(title)} — {_text(str(data['status']).replace('_', ' '))}</h3>"
    for key in ("unavailable_reason", "diagnostic", "first_failed_replicate_diagnostic",
                "first_failed_replicate_unavailable"):
        if data.get(key):
            section += f"<p><b>{_text(key.replace('_', ' ').capitalize())}:</b> {_text(_diagnostic(data[key], origin))}</p>"
    if "replicates" in data:
        section += (f"<p>{_text(data['successful_replicates'])} successful; "
                    f"{_text(data['failed_replicates'])} failed of {_text(data['replicates'])} replicates.</p>")
    if data.get("method"):
        section += f"<p>Method: {_text(data['method'])}; confidence level: {_text(data['confidence_level'])}.</p>"
    labels = {mapped.dataset_id: mapped.display_name for mapped in origin.mapped_datasets}
    rows = []
    for interval in data["intervals"]:
        coordinate = interval["coordinate"]
        name = symbols.get(coordinate.name, _text(coordinate.name))
        if coordinate.dataset_id:
            name = f"{_text(labels[coordinate.dataset_id])} · {name}"
        if coordinate.scope == "centred_overall":
            name = "Overall · " + name
        elif coordinate.scope == "centred_deviation":
            name = "Deviation · " + name
        unit = coordinate_units.get((coordinate.scope, coordinate.name, coordinate.dataset_id), "")
        outcomes = [str(interval[key]) for key in ("reason", "lower_outcome", "upper_outcome") if interval.get(key)]
        for side in ("lower", "upper"):
            endpoint = interval.get(side + "_resolution")
            if endpoint:
                if endpoint.get("reason"):
                    outcomes.append(f"{side}: {endpoint['reason']}")
        if interval.get("endpoint_messages"):
            outcomes.append(_diagnostic(interval["endpoint_messages"], origin))
        rows.append((name, interval["lower"], interval["upper"], _unit_html(unit), interval["status"], "; ".join(outcomes)))
    if rows:
        section += _table(("Parameter", "Lower", "Upper", "Unit", "Outcome", "Details"), rows, rich_columns=(0, 3))
    return section


def fit_overview(phase: str, model: str, objective: str, datasets: int, parameters: int) -> tuple[str, ...]:
    """One compact count convention for live and terminal Fit presentation."""
    return (phase, f"SSQ {objective}", f"{datasets} datasets", f"{parameters} parameters", model)


# Identity, rich/plain symbol, complete context, value and unit. These are
# call-local GUI entries, never retained scientific results.
FitValueEntry = tuple[tuple[str, ...], str, str, str, float, str]


def fit_completion_presentation(candidate: FitCandidate) -> tuple[str, bool]:
    """Describe typed settlement facts and whether execution reached completion."""
    result = candidate.result
    stopped = result.optimizer_diagnostic is not None and result.optimizer_diagnostic.type == "FittingStopped"
    text = ("Fit stopped · Best complete candidate retained" if stopped else
            "Fit converged" if result.optimizer_converged else
            "Fit finished · Optimizer did not converge")
    complete = not stopped
    for label, requested, analysis in (
        ("Profile", candidate.request.setup.profile_likelihood, result.profile_likelihood_result),
        ("Bootstrap", candidate.request.setup.residual_bootstrap, result.residual_bootstrap_result),
    ):
        if requested is None:
            continue
        status = "unavailable" if analysis is None else analysis.status
        if analysis is not None and analysis.provenance.get("settlement") == "not_started_after_stop":
            status = "not started after Stop"
        complete = complete and status == "available"
        text += f" · {label} {'complete' if status == 'available' else status}"
    return text, complete


def fit_live_presentation(
    update: FitLiveUpdate, parameters: tuple[FitParameterVariable, ...],
    dataset_labels: Mapping[str, str], initial_labels: Mapping[str, str],
) -> tuple[tuple[FitValueEntry, ...], str]:
    """Use admitted parameter roles and the worker's coordinate-label producer."""
    entries: list[FitValueEntry] = []
    overall = dict(update.parameters)
    scalars = {identity: dict(values) for identity, values in update.scalar_values_by_dataset}
    initials = {identity: dict(values) for identity, values in update.initial_values_by_set}
    def scalar(parameter: FitParameterVariable, context: str, identity: tuple[str, ...], value: float) -> None:
        entries.append((identity, scientific_symbol_html(parameter.name, family=parameter.family, role=parameter.role),
                        parameter.name, context, value, parameter.unit))
    for parameter in parameters:
        if parameter.scope not in {"shared", "centred_fixed_effects"}:
            continue
        scope = "centred_overall" if parameter.scope == "centred_fixed_effects" else "shared"
        key = _fit_coordinate_display_label(FitCoordinate(scope, parameter.name))
        if key in overall:
            scalar(parameter, "Overall" if scope == "centred_overall" else "Shared", (scope, parameter.name), overall[key])
    for dataset_id, label in dataset_labels.items():
        values = scalars.get(dataset_id, {})
        for parameter in parameters:
            if parameter.name in values:
                context = label + (" · Effective value" if parameter.scope == "centred_fixed_effects" else " · Per-dataset")
                scalar(parameter, context, ("dataset", dataset_id, parameter.name), values[parameter.name])
    for set_id, label in initial_labels.items():
        for name, value in initials.get(set_id, {}).items():
            entries.append((("initial", set_id, name), html.escape(name), name,
                            label + " · Initial", value, "M"))
    body = '<h3>Best complete candidate</h3>' + _table(
        ("Dataset / scope", "Parameter", "Value", "Unit"),
        ((context, symbol, value, _unit_html(unit)) for _identity, symbol, _plain_symbol, context, value, unit in entries), rich_columns=(1, 3))
    return tuple(entries), body


def fit_candidate_presentation(candidate: FitCandidate) -> tuple[tuple[str, ...], str, str, str, tuple[FitValueEntry, ...]]:
    """Project one terminal into its compact header, document and last provenance."""
    result = candidate.result
    shared, individual = _estimate_rows(candidate)
    initial = _initial_rows(candidate)
    phase, _complete = fit_completion_presentation(candidate)
    model = candidate.origin.tab_name
    objective = _metric_text(_metric(result.weighted_residual_sse))
    overview = fit_overview(phase, model, objective, len(result.dataset_info), len(candidate.request.setup.parameters))
    termination = _diagnostic(result.message, candidate.origin)
    body = f"<h2>{_text(phase)}</h2><p>{_text(model)}</p><p>{_text(termination)}</p>"
    body += f"<p><b>Weighted residual SSE:</b> {_text(objective)} &nbsp; <b>Weighted MSE:</b> {_text(_metric_text(_metric(result.weighted_residual_mse)))}</p>"
    model_kind = "Overall + dataset deviations (centered fixed effects)" if result.primary_model == "centred_fixed_effects" else "Ordinary global fit"
    body += f"<p>Model: {_text(model_kind)}.</p>"
    body += f"<p>Optimizer {'converged' if result.optimizer_converged else 'did not converge'}.</p>"
    parameters = {parameter.name: parameter for parameter in candidate.setup_snapshot.parameters}
    symbols = {name: scientific_symbol_html(name, family=parameter.family, role=parameter.role)
               for name, parameter in parameters.items()}
    def parameter_name(row: Mapping[str, object]) -> str:
        name = str(row["name"])
        if row.get("kind") == "initial" or name not in parameters:
            return _text(row["display_name"])
        label = parameters[name].display_name
        symbol = symbols[name]
        if symbol == html.escape(name):
            return _text(label)
        return symbol + (_text(label[len(name):]) if label.startswith(name)
                         else " · " + _text(label))
    def estimates(title: str, rows: list[dict[str, object]]) -> str:
        if not rows:
            return ""
        return f"<h3>{_text(title)}</h3>" + _table(
            ("Parameter", "Scope", "Estimate", "Unit", "Initial Guess", "Bounds", "Bound Position"),
            ((parameter_name(row), row["scope"], row["fitted_value"], _unit_html(row["unit"]), row["nominal_value"],
              f"{_display(row['minimum'])} to {_display(row['maximum'])}", row["bound_state"])
             for row in rows), rich_columns=(0, 3))
    body += estimates("Overall and shared estimates", shared)
    effective = [row for row in individual if row["kind"] != "centred_deviation"] + initial
    for mapped in candidate.origin.mapped_datasets:
        rows = [row for row in effective if row.get("dataset_display_name") == mapped.display_name
                and (row.get("dataset_id") == mapped.dataset_id or mapped.dataset_id in row.get("dataset_ids", ()))]
        body += estimates(f"Effective values · {mapped.display_name}", rows)
    deviations = [row for row in individual if row["kind"] == "centred_deviation"]
    if deviations:
        body += "<h3>Dataset deviations (Fit only)</h3><p>Linear deviations use physical units; logarithmic deviations are in log10 coordinates (decades). Effective physical values above come directly from the retained settlement.</p>"
        body += _table(("Dataset", "Parameter", "Coordinate", "Deviation", "Unit", "Initial Guess", "Bounds"),
                       ((row["dataset_display_name"], parameter_name(row), row["scope"], row["fitted_value"], _unit_html(row["unit"]),
                         row["nominal_value"], f"{_display(row['minimum'])} to {_display(row['maximum'])}") for row in deviations), rich_columns=(1, 4))
    aggregates = _aggregate_rows(individual)
    if aggregates:
        body += "<h3>Per-dataset estimate statistics</h3>" + _table(
            ("Parameter", "Datasets", "Mean", "Median", "Sample SD", "IQR", "Unit"),
            ((parameter_name(row), row["n"], row["arithmetic_mean"], row["median"], row["sample_sd"], row["iqr"], _unit_html(row["unit"])) for row in aggregates), rich_columns=(0, 6))
    metrics = _dataset_metrics(candidate)
    body += "<h3>Dataset metrics</h3>" + _table(
        ("Dataset", "Weighted SSE", "Weighted MSE", "Weighted RMSE", "Weighted MAE", "Effective Weight"),
        ((row["dataset_display_name"], *(_metric_text(row[key]) for key in ("weighted_sse", "weighted_mse", "weighted_rmse", "weighted_mae")), row["effective_weight"]) for row in metrics))
    r_squared = [(row["dataset_display_name"], name, _metric_text(value)) for row in metrics for name, value in row["r_squared_by_series"].items()]
    if r_squared:
        body += _table(("Dataset", "Series", "R²"), r_squared)
    profile, bootstrap = _profile_data(candidate), _bootstrap_data(candidate)
    coordinate_units: dict[tuple[str, str, str | None], str] = {}
    for parameter in candidate.setup_snapshot.parameters:
        if parameter.scope == "shared":
            coordinate_units[("shared", parameter.name, None)] = parameter.unit
        elif parameter.scope == "centred_fixed_effects":
            coordinate_units[("centred_overall", parameter.name, None)] = parameter.unit
            deviation_unit = "decades" if parameter.selected_config.log10 else parameter.unit
            for dataset_id, _value in parameter.selected_config.dataset_values:
                coordinate_units[("centred_deviation", parameter.name, dataset_id)] = deviation_unit
        else:
            for dataset_id, _value in parameter.selected_config.dataset_values:
                coordinate_units[("dataset", parameter.name, dataset_id)] = parameter.unit
    body += _analysis_section("Profile likelihood", profile, candidate.origin, symbols, coordinate_units)
    body += _analysis_section("Residual bootstrap", bootstrap, candidate.origin, symbols, coordinate_units)
    body += "<h3>Apply effect</h3><p>Apply publishes compatible shared estimates and fitted initial values to the project."
    if result.centred_fixed_effects_settlements:
        body += " For Overall + deviations parameters, only the overall value is published; dataset-specific effective values and deviations remain Fit results."
    if any(parameter.scope == "per_dataset" for parameter in candidate.setup_snapshot.parameters):
        body += " Per-dataset parameter values must first become Shared through Configure Shared Fit and an ordinary Shared Fit."
    body += "</p>"
    settings = candidate.origin.scientific_settings
    request_setup = candidate.request.setup
    provenance = "<h3>Technical provenance</h3>" + _table(("Setting", "Value"), (
        ("Temperature", f"{_display(float(candidate.origin.prepared_dsl.analysis.temperature_K))} K"),
        ("Relative tolerance", float(settings.rtol)),
        ("Absolute tolerance", float(settings.atol)),
        ("Analytic Jacobian", "on" if settings.analytic_jacobian_policy else "off"),
        ("Solver", request_setup.solver),
    ))
    mapped_by_id = {mapped.dataset_id: mapped for mapped in candidate.origin.mapped_datasets}
    provenance += _table(("Dataset", "Initial-Condition Set", "X Quantity", "Simulation Interval", "Experimental Time Range"),
                         ((mapped_by_id[item.dataset_id].display_name, mapped_by_id[item.dataset_id].set_name,
                           item.parametric.x_name if item.parametric else "t",
                           f"{_display(item.t_start)} to {_display(item.simulation_end)}",
                           "All paired observations" if item.parametric else
                           f"{_display(float(item.data_range_min))} to {_display(float(item.data_range_max))}")
                          for item in request_setup.datasets if item.included))
    if any(item.parametric is not None for item in request_setup.datasets if item.included):
        provenance += "<p>Measured-X fitting jointly matches weighted X and Y discrepancies on uninterrupted simulation segments. Recorded time is unused. Matched locations do not identify experimental times or establish absolute rate identifiability.</p>"
    provenance += f"<p>Weighting: {_text(FIT_DATASET_WEIGHT_MODE_LABELS[request_setup.dataset_weight_mode])}. Weighted residual = raw residual × √normalized dataset weight × √normalized target weight.</p>"
    for title, analysis in (("Profile likelihood", profile), ("Residual bootstrap", bootstrap)):
        if analysis.get("facts"):
            provenance += _table((title, "Value"), (
                (label, fit_identity_text(value, candidate.origin) if isinstance(value, str) else value)
                for label, value in analysis["facts"]
            ))
    entries: list[FitValueEntry] = []
    ordered = shared + [row for mapped in candidate.origin.mapped_datasets
                        for row in effective if mapped.dataset_id == row.get("dataset_id")
                        or mapped.dataset_id in row.get("dataset_ids", ())]
    for row in ordered:
        name = str(row["name"])
        initial_value = row["kind"] == "initial"
        symbol = html.escape(name) if initial_value else symbols[name]
        context = str(row["scope"])
        dataset_id = (str(row.get("dataset_id") or next(iter(row.get("dataset_ids", ())), ""))
                      if row.get("dataset_display_name") else "")
        if row.get("dataset_display_name"):
            context = str(row["dataset_display_name"]) + " · " + context
        entries.append(((str(row["kind"]), dataset_id, name), symbol, name, context,
                        float(row["fitted_value"]), str(row["unit"])))
    return overview, phase, body, provenance, tuple(entries)
