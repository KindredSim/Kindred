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
from collections.abc import Mapping, Sequence

import numpy as np

from kindred.core.analysis.fit_statistics import FitMetric
from kindred.core.analysis.global_fit_execution import FitCoordinate, parameter_group_variable_id
from kindred.core.fitting_optimization import fit_bound_position
from kindred.core.quantity_units import SECOND, format_unit_for_display
from kindred.core.simulation_failure import FactualCause
from kindred.core.validation import six_digit_text
from kindred.gui.display_name_policy import scientific_symbol_html
from kindred.gui.fitting.constants import (
    DE_CONVERGENCE_TOLERANCE_LABEL,
    DIFFERENTIAL_EVOLUTION_METHOD,
    FIT_DATASET_WEIGHT_MODE_LABELS,
    FUNCTION_TOLERANCE_LABEL,
    INITIAL_PREFIX,
    STEP_TOLERANCE_LABEL,
    fit_method_label,
)
from kindred.gui.fitting.curve_projection import prepared_fit_units, scale_values
from kindred.gui.fitting.launch import (
    FitOriginSnapshot,
    FitParameterVariable,
    FitSetupSnapshot,
    fit_coordinate_datasets_text,
    fit_group_labels,
    fit_parameter_overall,
    fit_uncertainty_coordinates,
)
from kindred.gui.fitting.execution import (
    FitCandidate, FitLiveUpdate, published_fitted_concentration,
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
        return six_digit_text(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_display(item) for item in value) + "]"
    return str(value)


def _profile_endpoint(endpoint: object) -> dict[str, object]:
    return {"reason": endpoint.reason, "limiting_bounds": endpoint.limiting_bounds}


def _profile_interval(coordinate: FitCoordinate, interval: object) -> dict[str, object]:
    return {
        "coordinate": coordinate,
        "status": interval.status,
        "lower": interval.lower,
        "upper": interval.upper,
        "lower_resolution": _profile_endpoint(interval.lower_resolution),
        "upper_resolution": _profile_endpoint(interval.upper_resolution),
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
            ("Likelihood-ratio threshold", _threshold_text(result)),
            ("Likelihood model", result.likelihood_model),
            ("Confidence level", f"{_display(100.0 * result.confidence_level)}%"),
        ),
        "unavailable_reason": result.unavailable_reason,
        "diagnostic": _plain(result.diagnostic),
        "intervals": [
            _profile_interval(coordinate, interval)
            for coordinate, interval in sorted(result.intervals.items(), key=lambda item: roster[item[0]])
        ],
    }


def _threshold_text(result: object) -> str:
    """The threshold used with its F(1, n−p) degrees of freedom."""

    if result.likelihood_ratio_threshold is None:
        return "Unavailable"
    freedom = result.observation_count - result.fitted_value_count
    return f"{six_digit_text(result.likelihood_ratio_threshold)} from F(1, {freedom})"


def _bootstrap_interval(coordinate: FitCoordinate, interval: object) -> dict[str, object]:
    return {
        "coordinate": coordinate,
        "status": interval.status,
        "lower": interval.lower,
        "upper": interval.upper,
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

    def by_dataset(texts: Mapping[str, str]) -> str:
        """The procedure once when every dataset shares it, else per dataset in table order."""

        if len(set(texts.values())) == 1:
            return next(iter(texts.values()))
        return "\n".join(
            f"{mapped.display_name}: {texts[mapped.dataset_id]}"
            for mapped in candidate.origin.mapped_datasets if mapped.dataset_id in texts
        )

    return {
        "requested": True,
        "status": result.status,
        "method": result.method,
        "replicates": result.replicates,
        "confidence_level": result.confidence_level,
        "facts": (
            ("Seed", result.seed),
            ("Resampling", by_dataset(result.resampling)),
            ("Residual centering", by_dataset(result.residual_centering)),
            ("Required successful refits", result.required_successes),
            ("Confidence level", f"{_display(100.0 * result.confidence_level)}%"),
        ),
        "successful_replicates": result.successful_replicates,
        "failed_replicates": result.failed_replicates,
        "intervals": [
            _bootstrap_interval(coordinate, interval)
            for coordinate, interval in sorted(result.intervals.items(), key=lambda item: roster[item[0]])
        ],
        "first_unavailable_replicate": _plain(
            None if result.first_failed_replicate_unavailable is None
            else result.first_failed_replicate_unavailable.factual_cause
        ),
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
            "effective_weight": float(info.weight),
        }
        metrics.append(metric)
    return metrics


def _series_metric_rows(candidate: FitCandidate) -> list[tuple[str, str, str, str, str]]:
    """R² and raw RMS residual per dataset and observable, X block included.

    Residuals are retained in the scientific unit; each RMS is shown in the
    unit the Fit plot shows that observable in.
    """

    names = {item.dataset_id: item.display_name for item in candidate.origin.mapped_datasets}
    infos = {info.dataset_id: info for info in candidate.result.dataset_info}
    prepared_dsl = candidate.origin.prepared_dsl
    rows: list[tuple[str, str, str, str, str]] = []
    for prepared in candidate.request.dispatch.prepared_datasets:
        info = infos[prepared.dataset_id]
        x = prepared.parametric_x_observation
        x_name = "t" if x is None else x.model_target
        y_units = {item.model_target: item.unit for item in prepared.selected_observations}
        source_units = dict(y_units)
        if x is not None:
            source_units[x_name] = x.unit
        _display_units, measured = prepared_fit_units(
            prepared_dsl, x_name, tuple(y_units), SECOND if x is None else x.unit,
            tuple(y_units.items()),
        )
        for series, rms in info.raw_rms_by_series.items():
            r_squared = info.r_squared_by_series.get(series)
            shown = (
                _display(scale_values((rms.value,), source_units[series], measured[series])[0])
                if rms.available else _metric_text(_metric(rms))
            )
            rows.append((
                names[prepared.dataset_id], series,
                "—" if r_squared is None else _metric_text(_metric(r_squared)),
                shown, format_unit_for_display(measured[series]) or "",
            ))
    return rows


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
                bound_state=fit_bound_position(
                    value, configuration.minimum, configuration.maximum, log10=configuration.log10,
                ))
            continue
        # Fixed Per-dataset values are not terminal fitted values and
        # therefore have no presentation row.  Fitted values below must come
        # from the terminal dataset-first result without nominal fallback.
        values = dict(launch_configuration.dataset_values)
        labels = fit_group_labels(parameter, terminal_setup.datasets, candidate.origin)
        group_of = {member: group for group in configuration.groups for member in group}
        # A group is one fitted value: one row, under its group label.
        shown_groups: set[tuple[str, ...]] = set()
        for dataset_id in included_ids:
            if dataset_id not in configuration.varied_dataset_ids:
                continue
            group = group_of.get(dataset_id)
            if group in shown_groups:
                continue
            if group is not None:
                shown_groups.add(group)
            value = _terminal_dataset_value(candidate, dataset_id, parameter.name)
            if dataset_id not in values:
                raise ValueError(
                    f"Fit setup omitted nominal dataset coordinate {dataset_id!r} "
                    f"for {parameter.name!r}."
                )
            nominal = float(values[dataset_id])
            append_row(individual, kind="per_dataset_parameter", name=parameter.name, display_name=parameter.display_name,
                scope=display_names[dataset_id] if group is None else labels[group[0]][1], unit=parameter.unit,
                nominal_value=nominal, fitted_value=value,
                minimum=configuration.minimum, maximum=configuration.maximum,
                bound_state=fit_bound_position(
                    value, configuration.minimum, configuration.maximum, log10=configuration.log10,
                ),
                identity=("dataset", dataset_id, parameter.name) if group is None
                else ("group", group[0], parameter.name))
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
            rows.append({"kind": "initial", "name": item.species, "display_name": f"initial {item.species}",
                "scope": mapped.display_name, "unit": "M",
                "nominal_value": launch_item.initial,
                "fitted_value": published_fitted_concentration(value, field=f"fitted initial {item.species!r}"),
                "minimum": launch_item.minimum, "maximum": launch_item.maximum,
                "bound_state": fit_bound_position(
                    value, launch_item.minimum, launch_item.maximum,
                    log10=launch_item.log10,
                ),
                "identity": ("initial", mapped.dataset_id, item.species)})
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
        # A shared initial group is one fitted value: one row naming its
        # included members in Fit dataset order.
        members = [mapped for mapped in origin.mapped_datasets
                   if mapped.dataset_id in item.member_dataset_ids and mapped.dataset_id in included_ids]
        if not members:
            continue
        value = _terminal_dataset_value(
            candidate,
            members[0].dataset_id,
            f"{INITIAL_PREFIX}{item.species}",
        )
        rows.append({"kind": "initial", "name": item.species, "display_name": f"initial {item.species}",
            "scope": ", ".join(mapped.display_name for mapped in members), "unit": "M",
            "nominal_value": launch_item.initial,
            "fitted_value": published_fitted_concentration(value, field=f"fitted initial {item.species!r}"),
            "minimum": launch_item.minimum, "maximum": launch_item.maximum,
            "bound_state": fit_bound_position(
                value, launch_item.minimum, launch_item.maximum,
                log10=launch_item.log10,
            ),
            "identity": ("initial group", item.origin_dataset_id, item.species)})
    return rows


def _overall_statistics(candidate: FitCandidate) -> str:
    """One table per fitted Per-dataset parameter, from its one overall.

    Groups count once and every statistic is arithmetic, in the parameter's
    unit.
    """

    setup = candidate.setup_snapshot
    body = ""
    for parameter in setup.parameters:
        if parameter.scope != "per_dataset":
            continue
        overall = fit_parameter_overall(parameter, setup.datasets, candidate.origin)
        if overall is None:
            continue
        entries, mean = overall
        values = sorted(value for _label, value in entries)
        count = len(values)

        def percentile(fraction: float) -> float:
            position = (count - 1) * fraction
            lower, upper = math.floor(position), math.ceil(position)
            return values[lower] + (values[upper] - values[lower]) * (position - lower)

        spread = statistics.stdev(values) if count > 1 else None
        rows = [(label, value, value - mean) for label, value in entries]
        summary = (
            ("Count", count),
            ("Mean", mean),
            ("Median", statistics.median(values)),
            ("Sample SD", spread),
            ("Standard error", None if spread is None else spread / math.sqrt(count)),
            ("IQR", percentile(0.75) - percentile(0.25)),
        )
        unit = format_unit_for_display(parameter.unit)
        body += (
            f"<h3>{scientific_symbol_html(parameter.name, family=parameter.family, role=parameter.role)}"
            f" Per-Dataset Values{' (' + unit + ')' if parameter.unit != '1' else ''}</h3>"
            + _table(("Dataset or Group", "Estimate", "Difference from Mean"), rows)
            + _table(("Statistic", "Value"), summary)
        )
    return body


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
    """A plain message, or a cause chain's messages, each once and as sentences."""

    if isinstance(value, Mapping):
        sentences: list[str] = []
        while value is not None:
            message = str(value["message"])
            message = message if message.endswith((".", "!", "?")) else message + "."
            if message not in sentences:
                sentences.append(message)
            value = value["cause"]
        text = " ".join(sentences)
    else:
        text = str(value)
    return fit_identity_text(text, origin)


_SEARCH_LIMIT_REACHED = "not limited within the search range"
_NO_CROSSING = "threshold crossing could not be located"
# Plain words for each reason the core records for a Profile side.
_ENDPOINT_REASONS = {
    "search_limit_reached": _SEARCH_LIMIT_REACHED,
    "scientific_unavailable": "a refit on this side failed",
    "binary64_representability_exhausted": _NO_CROSSING,
    "midpoint_refit_budget_exhausted": _NO_CROSSING,
}


def _and_join(items: Sequence[str]) -> str:
    """English list with the serial comma."""

    if len(items) < 3:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _interval_text(interval: Mapping[str, object], confidence: float) -> tuple[str, str]:
    """The interval as stated, and its confidence label.

    A confidence level is stated only beside numbers.  A Profile interval with
    one number is one-sided at (1 + level)/2, whether its other side is a
    limit or did not resolve.
    """

    lower, upper = interval["lower"], interval["upper"]
    one_sided = f"one-sided at {_display(100.0 * (1.0 + confidence) / 2.0)}%"
    if interval["status"] == "one-sided":
        if lower is not None:
            return f"≥ {_display(lower)}", one_sided
        if upper is not None:
            return f"≤ {_display(upper)}", one_sided
        return _SEARCH_LIMIT_REACHED, ""
    if lower is None and upper is None:
        return "not determined", ""
    text = f"{_display(lower)} to {_display(upper)}"
    if lower is None or upper is None:
        return text, one_sided
    return text, f"{_display(100.0 * confidence)}%"


def _analysis_section(title: str, data: Mapping[str, object], request: object,
                      names: Mapping[str, str], coordinate_units: Mapping[FitCoordinate, str]) -> str:
    """One analysis's rows; ``names`` holds each parameter's rich Summary name."""
    if not data["requested"]:
        return ""
    setup = request.setup
    origin = setup.origin
    section = f"<h3>{_text(title)} — {_text(str(data['status']).capitalize())}</h3>"
    if data.get("unavailable_reason"):
        section += f"<p>{_text(_diagnostic(data['unavailable_reason'], origin))}</p>"
    if data.get("diagnostic"):
        section += f"<p>Cause: {_text(_diagnostic(data['diagnostic'], origin))}</p>"
    if data.get("first_unavailable_replicate"):
        section += (f"<p>First unavailable replicate: "
                    f"{_text(_diagnostic(data['first_unavailable_replicate'], origin))}</p>")
    if "replicates" in data:
        section += (f"<p>{_text(data['successful_replicates'])} of {_text(data['replicates'])} "
                    f"replicates succeeded, {_text(data['failed_replicates'])} failed.</p>")
    if data.get("method"):
        section += f"<p>Method: {_text(data['method'])}.</p>"
    dataset_names = {item.dataset_id: item.display_name for item in origin.mapped_datasets}
    coupled = {item.variable_id: item for item in request.dispatch.ordinary_coupled_variables}
    group_targets = {
        parameter_group_variable_id(parameter.name, group[0]): FitCoordinate("group", parameter.name, group[0])
        for parameter in setup.parameters
        for group in parameter.selected_config.groups
    }

    def label(coordinate: FitCoordinate) -> str:
        """Rich name of any fitted coordinate, including an initial concentration or group."""

        if coordinate.scope == "coupled" and coordinate.name in group_targets:
            coordinate = group_targets[coordinate.name]
        elif coordinate.scope == "coupled":
            spec = coupled[coordinate.name]
            members = ", ".join(dataset_names[item] for item in spec.dataset_ids)
            return _text(f"initial {spec.parameter_name[len(INITIAL_PREFIX):]} — {members}")
        datasets = fit_coordinate_datasets_text(coordinate, setup)
        return names[coordinate.name] + (f" — {_text(datasets)}" if datasets else "")

    rows = []
    for interval in data["intervals"]:
        coordinate = interval["coordinate"]
        text, level = _interval_text(interval, float(data["confidence_level"]))
        details = [_text(level)] if level else []
        for side in ("lower", "upper"):
            endpoint = interval.get(side + "_resolution") or {}
            reason = endpoint.get("reason")
            # A reason the interval itself states is not repeated.
            if reason and _ENDPOINT_REASONS[reason] != text:
                details.append(_text(f"{side.capitalize()} end: {_ENDPOINT_REASONS[reason]}"))
            limiting = [f"the {_text(bound)} bound of {label(other)}"
                        for other, bound in endpoint.get("limiting_bounds", ())]
            if limiting:
                details.append(f"{_text(side.capitalize())} end is set by {_and_join(limiting)}")
        rows.append((label(coordinate), text, _unit_html(coordinate_units[coordinate]), "<br>".join(details)))
    if rows:
        section += _table(("Parameter", "Interval", "Unit", "Details"), rows, rich_columns=(0, 2, 3))
    return section


def fit_overview(phase: str, model: str, objective: str, datasets: int, fitted_values: int) -> tuple[str, ...]:
    """One compact count convention for live and terminal Fit presentation."""

    def count(number: int, noun: str) -> str:
        return f"{number} {noun}" if number == 1 else f"{number} {noun}s"

    return (phase, f"Weighted SSE {objective}", count(datasets, "dataset"), count(fitted_values, "fitted value"), model)


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
    update: FitLiveUpdate, setup: FitSetupSnapshot,
    dataset_labels: Mapping[str, str], initial_labels: Mapping[str, str],
) -> tuple[tuple[FitValueEntry, ...], str]:
    """Present live values with the running setup's parameter roles; a group shows once."""
    parameters = setup.parameters
    entries: list[FitValueEntry] = []
    overall = dict(update.parameters)
    scalars = {identity: dict(values) for identity, values in update.scalar_values_by_dataset}
    initials = {identity: dict(values) for identity, values in update.initial_values_by_set}
    def scalar(parameter: FitParameterVariable, context: str, identity: tuple[str, ...], value: float) -> None:
        entries.append((identity, scientific_symbol_html(parameter.name, family=parameter.family, role=parameter.role),
                        parameter.name, context, value, parameter.unit))
    for parameter in parameters:
        if parameter.scope == "shared" and parameter.name in overall:
            scalar(parameter, "Shared", ("shared", parameter.name), overall[parameter.name])
    for parameter in parameters:
        if parameter.scope == "shared":
            continue
        labels = fit_group_labels(parameter, setup.datasets, setup.origin)
        group_of = {member: group for group in parameter.selected_config.groups for member in group}
        shown_groups: set[tuple[str, ...]] = set()
        for dataset_id, label in dataset_labels.items():
            value = scalars.get(dataset_id, {}).get(parameter.name)
            group = group_of.get(dataset_id)
            if value is None or group in shown_groups:
                continue
            if group is None:
                scalar(parameter, label, ("dataset", dataset_id, parameter.name), value)
            else:
                shown_groups.add(group)
                scalar(parameter, labels[group[0]][1], ("group", group[0], parameter.name), value)
    # A shared initial group shows once, under its first included member,
    # naming every included member; later members are skipped.
    group_context: dict[tuple[str, str], str | None] = {}
    for group in setup.shared_initial_groups:
        members = [mapped.set_id for mapped in setup.origin.mapped_datasets
                   if mapped.dataset_id in group.member_dataset_ids and mapped.set_id in initial_labels]
        for index, set_id in enumerate(members):
            group_context[(set_id, group.species)] = (
                ", ".join(initial_labels[member] for member in members) if index == 0 else None
            )
    for set_id, label in initial_labels.items():
        for name, value in initials.get(set_id, {}).items():
            context = group_context.get((set_id, name), label)
            if context is not None:
                entries.append((("initial", set_id, name), html.escape(f"initial {name}"), f"initial {name}",
                                context, value, "M"))
    body = '<h3>Best Complete Candidate</h3>' + _table(
        ("Scope", "Parameter", "Value", "Unit"),
        ((context, symbol, value, _unit_html(unit)) for _identity, symbol, _plain_symbol, context, value, unit in entries), rich_columns=(1, 3))
    return tuple(entries), body


_BOUND_POSITION_WORDS = {"lower": "at minimum", "upper": "at maximum", "interior": "within bounds"}


def fit_candidate_presentation(candidate: FitCandidate) -> tuple[tuple[str, ...], str, str, tuple[FitValueEntry, ...]]:
    """Project one terminal into its compact header, Results document and Provenance document."""
    result = candidate.result
    shared, individual = _estimate_rows(candidate)
    rows = shared + individual + _initial_rows(candidate)
    phase, _complete = fit_completion_presentation(candidate)
    model = candidate.origin.tab_name
    objective = _metric_text(_metric(result.weighted_residual_sse))
    fitted_values = len(candidate.request.dispatch.ordinary_optimizer_layout.opt_param_keys())
    overview = fit_overview(phase, model, objective, len(result.dataset_info), fitted_values)
    body = f"<h2>{_text(phase)}</h2><p>Mechanism: {_text(model)}</p>"
    diagnostic = result.optimizer_diagnostic
    if diagnostic is not None and diagnostic.type == "FittingIdentifiability":
        body += f"<p><b>Warning:</b> {_text(diagnostic.message)}</p>"
    body += (f"<p><b>Weighted SSE:</b> {_text(objective)} &nbsp; "
             f"<b>Weighted MSE:</b> {_text(_metric_text(_metric(result.weighted_residual_mse)))}</p>")
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
    if rows:
        body += "<h3>Estimates</h3>" + _table(
            ("Parameter", "Scope", "Estimate", "Unit", "Initial Guess", "Bounds", "Bound Position"),
            ((parameter_name(row), row["scope"], row["fitted_value"], _unit_html(row["unit"]), row["nominal_value"],
              f"{_display(row['minimum'])} to {_display(row['maximum'])}", _BOUND_POSITION_WORDS[row["bound_state"]])
             for row in rows), rich_columns=(0, 3))
    body += _overall_statistics(candidate)
    metrics = _dataset_metrics(candidate)
    body += "<h3>Dataset Metrics</h3>" + _table(
        ("Dataset", "Weighted SSE", "Weighted MSE", "Weighted RMSE", "Weighted MAE", "Effective Weight"),
        ((row["dataset_display_name"], *(_metric_text(row[key]) for key in ("weighted_sse", "weighted_mse", "weighted_rmse", "weighted_mae")), row["effective_weight"]) for row in metrics))
    series_rows = _series_metric_rows(candidate)
    if series_rows:
        body += _table(("Dataset", "Series", "R²", "RMS Residual", "Unit"), series_rows, rich_columns=(4,))
    profile, bootstrap = _profile_data(candidate), _bootstrap_data(candidate)
    units = {item.name: item.unit for item in candidate.setup_snapshot.parameters}
    coordinate_units = {
        coordinate: units[coordinate.name]
        for coordinate in fit_uncertainty_coordinates(
            candidate.request.setup.parameters, candidate.request.setup.datasets,
        )
    }
    names = {
        parameter.name: parameter_name({"name": parameter.name, "display_name": parameter.display_name})
        for parameter in candidate.setup_snapshot.parameters
    }
    body += _analysis_section("Profile Likelihood", profile, candidate.request, names, coordinate_units)
    body += _analysis_section("Residual Bootstrap", bootstrap, candidate.request, names, coordinate_units)

    settings = candidate.origin.scientific_settings
    request_setup = candidate.request.setup
    defaults = request_setup.defaults
    differential_evolution = defaults.method == DIFFERENTIAL_EVOLUTION_METHOD
    optimizer_rows = [
        ("Optimization method", fit_method_label(defaults.method)),
        ("Maximum evaluations", defaults.max_nfev),
        (DE_CONVERGENCE_TOLERANCE_LABEL if differential_evolution else FUNCTION_TOLERANCE_LABEL,
         float(defaults.ftol)),
    ]
    if not differential_evolution:
        optimizer_rows.append((STEP_TOLERANCE_LABEL, float(defaults.xtol)))
    if differential_evolution and defaults.use_seed:
        optimizer_rows.append(("Seed", defaults.seed))
    provenance = (f"<p>Optimizer message: {_text(_diagnostic(result.message, candidate.origin))}</p>"
                  "<h3>Mechanism</h3><pre>"
                  + html.escape(candidate.origin.mechanism_source.reactions_text) + "</pre>")
    provenance += "<h3>Settings</h3>" + _table(("Setting", "Value"), (
        ("Temperature", f"{_display(float(candidate.origin.prepared_dsl.analysis.temperature_K))} K"),
        ("Relative tolerance", float(settings.rtol)),
        ("Absolute tolerance", float(settings.atol)),
        ("Analytic Jacobian", "On" if settings.analytic_jacobian_policy else "Off"),
        ("Solver", request_setup.solver),
        *optimizer_rows,
        ("Observations", sum(info.residuals.size for info in result.dataset_info)),
        ("Fitted values", fitted_values),
    ))
    dataset_names = {item.dataset_id: item.display_name for item in candidate.origin.mapped_datasets}
    included_ids = {item.dataset_id for item in request_setup.datasets if item.included}
    fixed_rows = []
    for parameter in request_setup.parameters:
        config = parameter.selected_config
        if parameter.scope == "shared":
            if not config.fit:
                fixed_rows.append((parameter, "Shared", _display(config.value)))
            continue
        # Each included dataset whose value was used as entered.
        fixed_rows.extend(
            (parameter, dataset_names[dataset_id], _display(value))
            for dataset_id, value in config.dataset_values
            if dataset_id in included_ids and dataset_id not in config.varied_dataset_ids
        )
    if fixed_rows:
        provenance += "<h3>Fixed Values</h3>" + _table(
            ("Parameter", "Scope", "Value", "Unit"),
            ((parameter_name({"name": parameter.name, "display_name": parameter.display_name}),
              scope, value, _unit_html(parameter.unit)) for parameter, scope, value in fixed_rows),
            rich_columns=(0, 3),
        )
    mapped_by_id = {mapped.dataset_id: mapped for mapped in candidate.origin.mapped_datasets}
    included = [item for item in request_setup.datasets if item.included]
    provenance += "<h3>Datasets</h3>" + _table(
        ("Dataset", "Initial-Condition Set", "X Quantity", "Simulation Interval", "Experimental Time Range"),
        ((mapped_by_id[item.dataset_id].display_name, mapped_by_id[item.dataset_id].set_name,
          item.parametric.x_name if item.parametric else "t",
          f"{_display(item.t_start)} to {_display(item.simulation_end)}",
          "All paired observations" if item.parametric else
          f"{_display(float(item.data_range_min))} to {_display(float(item.data_range_max))}")
         for item in included))
    if any(item.parametric is not None for item in included):
        provenance += "<p>Measured-X fitting jointly matches weighted X and Y discrepancies on uninterrupted simulation segments. Recorded time is unused. Matched locations do not identify experimental times or establish absolute rate identifiability.</p>"
    # The starting concentration each included dataset used, in M.
    starts = {(item.set_id, item.species): item.initial for item in request_setup.initials}
    species = [name for name, _value in mapped_by_id[included[0].dataset_id].initial_values] if included else []
    if species:
        provenance += "<h3>Initial Concentrations (M)</h3>" + _table(
            ("Dataset", *species),
            ((mapped_by_id[item.dataset_id].display_name,
              *(starts[(mapped_by_id[item.dataset_id].set_id, name)] for name in species))
             for item in included))
    provenance += f"<p>Weighting: {_text(FIT_DATASET_WEIGHT_MODE_LABELS[request_setup.dataset_weight_mode])}. Weighted residual = raw residual × √normalized dataset weight × √normalized target weight.</p>"
    for title, analysis in (("Profile Likelihood", profile), ("Residual Bootstrap", bootstrap)):
        if analysis.get("facts"):
            provenance += f"<h3>{title}</h3>" + _table(("Setting", "Value"), (
                (label, fit_identity_text(value, candidate.origin) if isinstance(value, str) else value)
                for label, value in analysis["facts"]
            ))
    entries = tuple(
        (tuple(row["identity"]) if row["kind"] != "shared_parameter" else ("shared", str(row["name"])),
         html.escape(str(row["display_name"])) if row["kind"] == "initial" else symbols[str(row["name"])],
         str(row["display_name"]) if row["kind"] == "initial" else str(row["name"]),
         str(row["scope"]), float(row["fitted_value"]), str(row["unit"]))
        for row in rows
    )
    return overview, body, provenance, entries
